#ifndef __GSALGO_INDICATOR_KRIGING_COMMAND_H__56F6BFCF_ACC9_4F63_A197_57316908E436__
#define __GSALGO_INDICATOR_KRIGING_COMMAND_H__56F6BFCF_ACC9_4F63_A197_57316908E436__

#include "ik_params.h"
#include "sugarbox_grid.h"
#include "progress_reporter.h"
#include "cdf_utils.h"
#include "pretty_printer.h"
#include "my_kriging_weights.h"
#include "precalculated_covariance.h"
#include "kriging_interpolation.h"
#include "neighbourhood_lookup.h"
#include "is_informed_predicate.h"
#include "cov_model.h"
#include "kriging_stats.h"
#include "api.h"
#include <cmath>
#include <exception>

// OpenBLAS thread-control API — file-scope declarations so the in-function
// branch compiles on Apple Clang too (extern "C" inside function bodies is
// rejected there).  HPGL_USE_OPENBLAS is defined by CMake ONLY when the
// build actually links OpenBLAS (BLA_VENDOR=OpenBLAS or BLAS library-name
// detection), so the openblas_* symbols are never referenced on Linux
// builds that link a different BLAS vendor (M-32: undefined-symbol dlopen
// failure with netlib/BLIS/FlexiBLAS/ATLAS).
#ifdef HPGL_USE_OPENBLAS
extern "C" {
	void openblas_set_num_threads(int);
	int openblas_get_num_threads(void);
}
#endif

namespace hpgl
{
	
	typedef std::vector<indicator_probability_t> marginal_probs_t;

	

	/*template<
		typename grid_t,
		typename data_t,
		typename means_t,
		typename weight_calculator_t>*/

	namespace detail
	{
		// Correct indicator kriging probabilities for order relations.
		// Ensures: 1) Monotonicity P(k) <= P(k+1), 2) Bounds [0,1].
		//
		// Algorithm: GSLIB 2nd ed. (1998) ORDREL.FOR applies DIFFERENT
		// corrections for continuous (ivtype=1) and categorical (ivtype=0)
		// variables:
		//   - continuous: clip all CDF estimates to [0,1], upward
		//     fill-forward pass, downward fill-backward pass, then average
		//     the two corrected sequences (this replaces the earlier
		//     iterative pairwise averaging — Deutsch & Journel, 1992,
		//     1st ed. — which diverges from GSLIB on inputs with multiple
		//     cascading violations).
		//   - categorical: clip each per-category probability to [0,1] and
		//     RENORMALIZE by the total: `sumcdf = sum(ccdf1);
		//     if(sumcdf.le.0.0) sumcdf=1.0; ccdfo(i) = ccdf1(i)/sumcdf`.
		//     No up/down/average passes.
		//
		// E-M56 (categorical S>1 truncation FLIP): the old "scale top to
		// 1.0" Step-5 diverged when the per-category PMF summed to S > 1
		// (reachable with per-category covariance models — the standard
		// multi-category IK/SIS configuration): the [0,1] clip pinned the
		// cumulative top at 1.0, making the scale a no-op and TRUNCATING the
		// excess mass onto the earlier categories (executed proof:
		// p=[0.6,0.7,0.5] → old HPGL masses [0.6,0.4,0.0] argmax 0 vs GSLIB
		// [0.333,0.389,0.278] argmax 1 — a silent category-selection FLIP).
		//
		// The two branches are selected by input shape: the production
		// callers (IK indicator_kriging.h, SIS sequential_indicator_
		// simulation.cpp) always pass the CUMULATIVE CDF of already-sanitized
		// per-category probabilities — non-decreasing by construction — while
		// the continuous envelope exists precisely to repair NON-monotone CDF
		// estimates. Monotonicity dispatch (NaN-safe: NaN comparisons are
		// false, so NaN never marks a sequence decreasing) therefore routes
		// every production call into the categorical renormalization and any
		// raw CDF-estimate sequence into the continuous envelope.
		inline void correct_order_relations(std::vector<indicator_probability_t> & probs)
		{
			if (probs.empty())
				return;

			const size_t n = probs.size();

			bool monotone = true;
			for (size_t i = 1; i < n && monotone; ++i)
			{
				if (probs[i] < probs[i - 1])
					monotone = false;
			}

			if (monotone)
			{
				// GSLIB categorical (ivtype=0) renormalization. The input is
				// the cumulative CDF of per-category probabilities; recover
				// the per-category masses by backward difference, clip them
				// to [0,1] (GSLIB's per-probability clip), sum, and rebuild
				// the cumulative of the renormalized distribution — identical
				// to GSLIB for S <= 1 (mass/sum == cdf/top) and GSLIB-correct
				// for S > 1.
				std::vector<indicator_probability_t> masses(n);
				indicator_probability_t prev = 0.0f;
				double sumcdf = 0.0;
				for (size_t i = 0; i < n; ++i)
				{
					// NaN-safe (II-12): a NaN cumulative entry would poison
					// the backward difference; hold the previous level. The
					// callers sanitize per-category probabilities to [0,1]
					// BEFORE accumulating, so NaN cannot occur on the real
					// paths — this is defense-in-depth for direct misuse.
					indicator_probability_t v = probs[i];
					if (!std::isfinite(v))
						v = prev;
					// Defensive monotonicity (sanitized input is already
					// non-decreasing); guarantees non-negative masses.
					if (v < prev)
						v = prev;
					indicator_probability_t mass = v - prev;
					// GSLIB [0,1] clip applied per per-category probability.
					if (mass < 0.0f)
						mass = 0.0f;
					else if (mass > 1.0f)
						mass = 1.0f;
					masses[i] = mass;
					sumcdf += mass;
					prev = v;
				}
				// GSLIB sumcdf <= 0 guard: divisor forced to 1.0 (all-zero
				// input stays all-zero; most_probable_category/sample then
				// fall back to category 0 — matching GSLIB's degenerate
				// outcome).
				if (sumcdf <= 0.0)
					sumcdf = 1.0;
				indicator_probability_t cum = 0.0f;
				for (size_t i = 0; i < n; ++i)
				{
					cum += static_cast<indicator_probability_t>(masses[i] / sumcdf);
					probs[i] = cum;
				}
				return;
			}

			std::vector<indicator_probability_t> ccdf1(n);
			std::vector<indicator_probability_t> ccdf2(n);

			// Step 1: Clip probabilities to [0, 1] (both working copies).
			// NaN-safe (II-12): NaN bypasses relational comparisons, so treat
			// it as 0.0 — a NaN probability would otherwise propagate through
			// both passes and the average, then make most_probable_category
			// (cdf_utils.cpp) silently return category 0.
			for (size_t i = 0; i < n; ++i)
			{
				indicator_probability_t v = probs[i];
				if (!std::isfinite(v))
					v = 0.0f;
				if (v < 0.0)
					v = 0.0;
				else if (v > 1.0)
					v = 1.0;
				ccdf1[i] = v;
				ccdf2[i] = v;
			}

			// Step 2: Upward pass — fill-forward (GSLIB ordrel).
			//   do i=2,ncut: if ccdf1(i)<ccdf1(i-1): ccdf1(i)=ccdf1(i-1)
			for (size_t i = 1; i < n; ++i)
			{
				if (ccdf1[i] < ccdf1[i - 1])
					ccdf1[i] = ccdf1[i - 1];
			}

			// Step 3: Downward pass — fill-backward (GSLIB ordrel).
			//   do i=ncut-1,1,-1: if ccdf2(i)>ccdf2(i+1): ccdf2(i)=ccdf2(i+1)
			for (size_t i = n - 1; i > 0; --i)
			{
				if (ccdf2[i - 1] > ccdf2[i])
					ccdf2[i - 1] = ccdf2[i];
			}

			// Step 4: Average the two monotone corrections.
			// GSLIB continuous (ivtype=1) branch ends here — NO top-scaling:
			// the [0,1] clamp + monotone envelope is the complete correction
			// for continuous variables (E-M56; the old Step-5 "scale top to
			// 1.0" belonged to the categorical normalization, which now has
			// its own branch above).
			for (size_t i = 0; i < n; ++i)
				probs[i] = static_cast<indicator_probability_t>(0.5 * (ccdf1[i] + ccdf2[i]));
		}

		template<typename cov_t>
		void create_precalucated_cov_models(const ik_params_t & params, std::vector<cov_t> & result)
		{
			result.resize(params.m_category_count);
			for (size_t i = 0; i < params.m_category_count; ++i)
			{
				result[i].init(cov_model_t(params.m_cov_params[i]), params.m_radiuses[i]);
			}
		}

		template<typename nl_t, typename prop_t>
		void add_defined_cells(std::vector<nl_t> & neighbour_lookups, const prop_t & prop)
		{
			for (int node = 0, end_node = prop.size(); node < end_node; ++node)
			{
				if (prop.is_informed(node))
				{
					for (int i = 0, end_i = neighbour_lookups.size(); i < end_i; ++i)
					{
						neighbour_lookups[i].add_node(node);
					}
				}
			}
		}
	}

	template<typename grid_t, typename marginal_probs_t>
	void do_indicator_kriging(
		const indicator_property_array_t & input_property, 
		const grid_t & grid, 
		const ik_params_t & params, 
		const marginal_probs_t & mps,
		indicator_property_array_t & output_property)
	{
		using namespace hpgl::detail;

		// E2-167: size-consistency validation at the kernel entry (sibling
		// pattern cont_kriging.h:72, median_ik.cpp:51-55). The kernel had
		// zero HPGL_CHECK — direct-C++ misuse (output smaller than input →
		// set_at OOB; grid smaller than input → add_defined_cells / lookup
		// indices beyond the grid) silently corrupted instead of failing.
		HPGL_CHECK(input_property.size() == output_property.size() && grid.size() == input_property.size(),
			"do_indicator_kriging: size mismatch between grid, input, and output");

		print_algo_name("Indicator Kriging");
		print_params(params);
		if (params.m_category_count == 0)
			return;

		progress_reporter_t report(grid.size());

		typedef precalculated_covariances_t cov_t;
		std::vector<cov_t> covariances;
		create_precalucated_cov_models(params, covariances);		

		typedef indexed_neighbour_lookup_t<grid_t, cov_t> nl_t;
		std::vector<nl_t> nblookups;

		for (size_t i = 0; i < params.m_category_count; ++i)
		{			
			nblookups.push_back(nl_t(&grid, &covariances[i], params.m_nb_params[i]));
		}	

		add_defined_cells(nblookups, input_property);	

		typedef typename nl_t::coord_t coord_t;

		std::vector<indicator_array_adapter_t> ind_props;
		for (size_t i = 0; i < params.m_category_count; ++i)
		{
			ind_props.push_back(indicator_array_adapter_t(&input_property, i));
		}

		report.start();

		size_t size = input_property.size();
		static const int LP_BATCH_SIZE = 1000;

		// BLAS thread control: prevent oversubscription when BLAS internal
		// threading combines with OpenMP parallel region threads.
		// Reference-counted: concurrent kriging calls safely share the
		// process-wide BLAS thread count. RAII guard (F-M8): restored even
		// when an exception unwinds through the parallel region.
#if defined(HPGL_USE_MKL) || defined(USE_INTEL_MKL)
		extern "C" void mkl_set_num_threads(int);
		extern "C" int mkl_get_max_threads(void);
		detail::blas_thread_guard_t blas_guard(mkl_get_max_threads, mkl_set_num_threads);
#elif defined(HPGL_USE_OPENBLAS)
		// OpenBLAS (any platform): limit internal threads to 1 so they don't
		// multiply with OpenMP threads.  HPGL_USE_OPENBLAS is defined by CMake
		// ONLY when the build actually links OpenBLAS — on Linux with
		// netlib/BLIS/FlexiBLAS/ATLAS the macro is undefined, so the openblas_*
		// symbols are never referenced (M-32).  Declarations at file scope above.
		detail::blas_thread_guard_t blas_guard(openblas_get_num_threads, openblas_set_num_threads);
#else
		// Other BLAS (Linux netlib/BLIS/FlexiBLAS/ATLAS, macOS Accelerate, …):
		// no thread-count API available.  Accelerate manages its own thread
		// pool via GCD and does not oversubscribe with OpenMP in the same way.
#endif
		// F-M9: catch allocation failures (correct_order_relations ccdf1/ccdf2,
		// ws.A.resize) inside the region and rethrow after it, so the C ABI
		// catch converts them to a clean error instead of std::terminate.
		// F-M5: track per-category kriging outcomes for stats population.
		unsigned long points_calculated = 0;
		unsigned long points_without_neighbours = 0;
		unsigned long points_singularity = 0;
		unsigned long nodes_processed = 0;
		double sum_categories = 0;
		std::exception_ptr parallel_error;
		// M-5: cooperative cancellation flag shared by all threads.
		// #pragma omp cancel for is a silent no-op in every default build
		// (the cancel-var ICV is false; OMP_CANCELLATION is never set), so
		// the documented user-cancel feature was never effective. Each
		// thread checks this atomic flag at the top of every iteration and
		// skips the expensive kriging body once set.
		std::atomic<bool> loop_stop{false};
		#pragma omp parallel
		{
			int local_lap_count = 0;
			std::vector<indicator_probability_t> probs;
			// Per-thread workspace: eliminates 5 heap allocations per
			// (node × indicator) pair. Vectors are re-filled on each
			// kriging_interpolation_ws call — same memory, zero heap churn.
			kriging_ws_t<indicator_value_t, coord_t> ws;
			#pragma omp for schedule(dynamic) reduction(+: points_calculated) reduction(+: points_without_neighbours) reduction(+: points_singularity) reduction(+: nodes_processed) reduction(+: sum_categories)
			for (size_t node_idx = 0; node_idx < size; ++node_idx)
			{
				// M-5: cooperative cancellation — see loop_stop above.
				if (loop_stop.load(std::memory_order_relaxed)) {
#ifndef _OPENMP
					break;
#else
					continue;
#endif
				}
				try
				{
				probs.clear();
				for (size_t idx = 0; idx < params.m_category_count; ++idx)
				{
					indicator_probability_t prob;

					ki_result_t ki_result = kriging_interpolation_ws(ind_props[idx], is_informed_predicate_t<indicator_property_array_t>(input_property), node_idx, covariances[idx], mps[idx], nblookups[idx], sk_weight_calculator_t(), prob, ws);

					if (ki_result != ki_result_t::KI_SUCCESS)
					{
						prob = mps[idx][node_idx];
					}
					switch (ki_result)
					{
					case ki_result_t::KI_SUCCESS: ++points_calculated; break;
					case ki_result_t::KI_NO_NEIGHBOURS: ++points_without_neighbours; break;
					case ki_result_t::KI_SINGULARITY: ++points_singularity; break;
					}
					// II-12: sanitize each kriged probability (finite + [0,1],
					// marginal fallback) BEFORE the cumulative-CDF conversion and
					// order-relations correction. A NaN probability survives
					// correct_order_relations (NaN comparisons are false) and
					// makes most_probable_category silently return category 0.
					prob = static_cast<indicator_probability_t>(
						detail::sanitize_probability(prob, mps[idx][node_idx]));
					probs.push_back(prob);
				}

				// indicator_array_adapter_t uses exclusive encoding
				// (== m_value ? 1 : 0) so kriged probabilities are per-category
				// PMF values. Convert to cumulative CDF before order relations
				// correction, which assumes monotonic [0,1]-bounded values.
				for (size_t i = 1; i < probs.size(); ++i)
					probs[i] += probs[i - 1];

				// Apply order relations correction to ensure monotonicity and [0,1] bounds
				correct_order_relations(probs);

				indicator_value_t category = most_probable_category(probs);
				output_property.set_at(node_idx, category);
				sum_categories += static_cast<double>(category);
				++nodes_processed;

				// Batch progress updates to reduce critical section contention.
				// Each thread accumulates laps locally and flushes in batches,
				// reducing lock acquisitions from ~10M to ~10M/LP_BATCH_SIZE.
			local_lap_count++;
			if (local_lap_count >= LP_BATCH_SIZE)
			{
				#pragma omp critical
				{
					report.next_lap(local_lap_count);
				}
				local_lap_count = 0;
				if (report.cancelled()) {
					// M-5: set the cooperative stop flag instead of
					// `#pragma omp cancel for` (a silent no-op in default
					// builds).
					loop_stop.store(true, std::memory_order_relaxed);
				}
			}
				} // try
				catch (const std::exception &)
				{
					// F-M9: allocation failure (correct_order_relations
					// ccdf1/ccdf2, ws.A.resize) inside the region. Record and
					// cancel; rethrow after the region so the C ABI catch
					// converts it to a clean error instead of std::terminate.
					#pragma omp critical
					{
						if (!parallel_error)
							parallel_error = std::current_exception();
					}
					// M-5: cooperative stop — sets the shared flag so all
					// threads skip remaining work (see loop_stop above).
					loop_stop.store(true, std::memory_order_relaxed);
				}
		}
		// Flush remaining laps for this thread
		if (local_lap_count > 0)
		{
			#pragma omp critical
			{
				report.next_lap(local_lap_count);
			}
		}
		// NOTE: No #pragma omp cancel for here — the for-loop worksharing
		// construct has already ended, and OMP §2.17.1 requires the cancel
		// directive to appear within the worksharing construct it targets.
		// Cancellation is handled inside the loop body at lines 216-222.
	}

		// F-M9: rethrow the recorded allocation failure on the calling
		// thread (outside the parallel region). The RAII blas_guard above is
		// destroyed during unwind, restoring the BLAS thread count.
		if (parallel_error)
			std::rethrow_exception(parallel_error);

		report.stop();
		// F-M5: populate kriging stats (previously indicator kriging never
		// called set_kriging_stats, leaving stale stats from a prior
		// kriging observable via hpgl_get_kriging_stats — api.h:188-193
		// zero-init promise violated). Points are per-category kriging
		// evaluations; mean is the average output category.
		{
			kriging_stats_t stats;
			stats.m_points_calculated = points_calculated;
			stats.m_points_without_neighbours = points_without_neighbours;
			stats.m_points_singularity = points_singularity;
			stats.m_mean = nodes_processed > 0 ? sum_categories / static_cast<double>(nodes_processed) : 0;
			stats.m_speed_nps = report.iterations_per_second();
			set_kriging_stats(stats);

			// M-6: emit the stderr failure signal on the indicator-kriging
			// failure paths (mirrors cont_kriging.h, simple_cokriging_markI.cpp,
			// and SIS). Previously singular/no-neighbour systems silently
			// substituted the marginal probability with no observability.
			if (stats.m_points_singularity > 0 || stats.m_points_without_neighbours > 0)
			{
				fprintf(stderr,
					"HPGL: kriging failures: %lu singularity, %lu no-neighbours (of %lu total)\n",
					stats.m_points_singularity,
					stats.m_points_without_neighbours,
					static_cast<unsigned long>(size));
			}
		}
		{
			std::ostringstream oss;
			oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
			write(oss.str());
		}
	}
}



#endif //__GSALGO_INDICATOR_KRIGING_COMMAND_H__56F6BFCF_ACC9_4F63_A197_57316908E436__
