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
		// Correct indicator kriging probabilities for order relations
		// Ensures: 1) Monotonicity P(k) <= P(k+1), 2) Bounds [0,1]
		// Reference: Deutsch & Journel (1998), Section V.6.3
		// Iterative averaging: repeat until monotonic to handle cascading violations
		inline void correct_order_relations(std::vector<indicator_probability_t> & probs)
		{
			if (probs.empty())
				return;

			// Step 1: Clamp probabilities to [0, 1]
			for (size_t i = 0; i < probs.size(); ++i)
			{
				if (probs[i] < 0.0)
					probs[i] = 0.0;
				else if (probs[i] > 1.0)
					probs[i] = 1.0;
			}

			// Step 2: Iterative averaging until monotonic
			// A single pass may not fix cascading violations, so repeat until
			// no inversions remain. Using 2× probs.size() ensures convergence
			// even for small category counts where cascading violations require
			// multiple passes (worst case: alternating violations propagate
			// one position per pass).
			bool changed = true;
			for (size_t iter = 0; iter < probs.size() * 2 && changed; ++iter)
			{
				changed = false;
				for (size_t i = 0; i + 1 < probs.size(); ++i)
				{
					if (probs[i] > probs[i + 1])
					{
						double avg = (probs[i] + probs[i + 1]) / 2.0;
						probs[i] = avg;
						probs[i + 1] = avg;
						changed = true;
					}
				}
			}
		}

		template<typename cov_t>
		void create_precalucated_cov_models(const ik_params_t & params, std::vector<cov_t> & result)
		{
			result.resize(params.m_category_count);
			for (int i = 0; i < params.m_category_count; ++i)
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

		for (int i = 0; i < params.m_category_count; ++i)
		{			
			nblookups.push_back(nl_t(&grid, &covariances[i], params.m_nb_params[i]));
		}	

		add_defined_cells(nblookups, input_property);	

		typedef typename nl_t::coord_t coord_t;

		std::vector<indicator_array_adapter_t> ind_props;
		for (int i = 0; i < params.m_category_count; ++i)
		{
			ind_props.push_back(indicator_array_adapter_t(&input_property, i));
		}

		report.start();

		size_t size = input_property.size();
		static const int LP_BATCH_SIZE = 1000;

		// BLAS thread control: prevent oversubscription when BLAS internal
		// threading combines with OpenMP parallel region threads.
		// Each OpenMP thread may call LAPACK (dsysv) which can spawn
		// additional threads if BLAS is configured for multi-threading.
#if defined(HPGL_USE_MKL) || defined(USE_INTEL_MKL)
		extern "C" void mkl_set_num_threads(int);
		extern "C" int mkl_get_max_threads(void);
		int _saved_blas_threads = mkl_get_max_threads();
		mkl_set_num_threads(1);
#elif defined(__linux__)
		// OpenBLAS: limit internal threads to 1 so they don't multiply
		// with OpenMP threads. Declared extern to avoid header dependency.
		extern "C" void openblas_set_num_threads(int);
		extern "C" int openblas_get_num_threads(void);
		int _saved_blas_threads = openblas_get_num_threads();
		openblas_set_num_threads(1);
#else
		// macOS Accelerate / other BLAS: no thread-count API available.
		// Accelerate manages its own thread pool via GCD and does not
		// oversubscribe with OpenMP in the same way.
#endif
		#pragma omp parallel
		{
			int local_lap_count = 0;
			std::vector<indicator_probability_t> probs;
			// Per-thread workspace: eliminates 5 heap allocations per
			// (node × indicator) pair. Vectors are re-filled on each
			// kriging_interpolation_ws call — same memory, zero heap churn.
			kriging_ws_t<indicator_value_t, coord_t> ws;
			#pragma omp for schedule(dynamic)
			for (node_index_t node_idx = 0;	node_idx < size; ++node_idx)
			{
				probs.clear();
				for (int idx = 0; idx < params.m_category_count; ++idx)
				{
					indicator_probability_t prob;

					ki_result_t ki_result = kriging_interpolation_ws(ind_props[idx], is_informed_predicate_t<indicator_property_array_t>(input_property), node_idx, covariances[idx], mps[idx], nblookups[idx], sk_weight_calculator_t(), prob, ws);

					if (ki_result != ki_result_t::KI_SUCCESS)
					{
						prob = mps[idx][node_idx];
					}
					probs.push_back(prob);
				}

				// Apply order relations correction to ensure monotonicity and [0,1] bounds
				correct_order_relations(probs);

				output_property.set_at(node_idx, most_probable_category(probs));			

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
		}

		// Restore BLAS thread count after parallel region completes
#if defined(HPGL_USE_MKL) || defined(USE_INTEL_MKL)
		mkl_set_num_threads(_saved_blas_threads);
#elif defined(__linux__)
		openblas_set_num_threads(_saved_blas_threads);
#endif

		report.stop();
		{
			std::ostringstream oss;
			oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
			write(oss.str());
		}
	}
}



#endif //__GSALGO_INDICATOR_KRIGING_COMMAND_H__56F6BFCF_ACC9_4F63_A197_57316908E436__
