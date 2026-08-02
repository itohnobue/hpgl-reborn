#include "stdafx.h"
#include "median_ik.h"

#include "covariance_field.h"
#include "progress_reporter.h"
#include "my_kriging_weights.h"
#include "kriging_interpolation.h"
#include "sugarbox_indexed_neighbour_lookup.h"
#include "is_informed_predicate.h"
#include "mean_provider.h"
#include "output.h"
#include "kriging_stats.h"
#include "api.h"
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
	inline indicator_index_t choose_indicator(indicator_probability_t prob)
	{
		if (prob <= 0.5)
			return 0;
		else
			return 1;
	}

	
void median_ik_for_two_indicators(
		const median_ik_params & params, 
		const sugarbox_grid_t & grid,
		const indicator_property_array_t & input_property,
		indicator_property_array_t & output_property
)
{
	typedef sugarbox_grid_t grid_t;

	progress_reporter_t report(grid.size());
	HPGL_CHECK(input_property.size() == output_property.size(),
		"median_ik_for_two_indicators: input property size does not match output property size");

	HPGL_CHECK(input_property.size() == grid.size(),
		"median_ik_for_two_indicators: property size does not match grid size");


	size_t prop_size = input_property.size();
	
	covariance_field_t cov_field(params.m_radiuses, cov_model_t(params));

	std::vector<node_index_t> indices;

	indicator_array_adapter_t prop_adapter(&input_property, 1);

	typedef indexed_neighbour_lookup_t<grid_t, covariance_field_t> nl_t;

	nl_t nl(&grid, &cov_field, params);

	for (node_index_t node = 0; node < input_property.size(); ++node)
	{
		if (input_property.is_informed(node))
			nl.add_node(node);
	}	

	report.start();
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
	// F-M9: catch allocation failures inside the region and rethrow after
	// it so the C ABI catch converts them to a clean error instead of
	// std::terminate. F-M5: track kriging outcomes for stats population.
	unsigned long points_calculated = 0;
	unsigned long points_without_neighbours = 0;
	unsigned long points_singularity = 0;
	// 2-M-33: count every node that received an output category (including
	// failure fallbacks) so m_mean uses a numerator==denominator pair like
	// the four sibling consumers (cont_kriging, indicator_kriging, cokriging,
	// SIS). Previously the denominator was points_calculated (KI_SUCCESS only)
	// while sum_categories accumulated over ALL nodes — a biased mean on
	// partial failure.
	unsigned long nodes_processed = 0;
	double sum_categories = 0;
	std::exception_ptr parallel_error;
	// M-5: cooperative cancellation flag shared by all threads.
	// #pragma omp cancel for is a silent no-op in every default build
	// (the cancel-var ICV is false; OMP_CANCELLATION is never set), so
	// the documented user-cancel feature was never effective. Each thread
	// checks this atomic flag at the top of every iteration and skips the
	// expensive kriging body once set.
	std::atomic<bool> loop_stop{false};
	#pragma omp parallel
	{
		int local_lap_count = 0;
		#pragma omp for schedule(dynamic) reduction(+: points_calculated) reduction(+: points_without_neighbours) reduction(+: points_singularity) reduction(+: nodes_processed) reduction(+: sum_categories)
		for (size_t idx = 0; idx < prop_size; ++idx)
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
			double prob;
			indicator_index_t ind_index;

			ki_result_t ki_result = kriging_interpolation(
				prop_adapter, 
				is_informed_predicate_t<indicator_property_array_t>(input_property), 
				idx, 				 
				cov_field, 
				single_mean_t(params.m_marginal_probs[1]), 
				nl, sk_weight_calculator_t(), prob);
		
			if (ki_result != ki_result_t::KI_SUCCESS)
			{
				prob = params.m_marginal_probs[1];
			}
			switch (ki_result)
			{
			case ki_result_t::KI_SUCCESS: ++points_calculated; break;
			case ki_result_t::KI_NO_NEIGHBOURS: ++points_without_neighbours; break;
			case ki_result_t::KI_SINGULARITY: ++points_singularity; break;
			}

			ind_index = choose_indicator(prob);
			output_property.set_at(idx, ind_index);
			sum_categories += static_cast<double>(ind_index);
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
				// F-M9: allocation failure inside the region. Record and
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
		// Cancellation is handled inside the loop body at lines 117-123.
	}

	// F-M9: rethrow the recorded allocation failure on the calling thread
	// (outside the parallel region). The RAII blas_guard above is destroyed
	// during unwind, restoring the BLAS thread count.
	if (parallel_error)
		std::rethrow_exception(parallel_error);

	report.stop();
	// F-M5: populate kriging stats (previously median IK never called
	// set_kriging_stats, leaving stale stats observable — api.h:188-193
	// zero-init promise violated). Mean is the average output category.
	// 2-M-33: denominator is nodes_processed (all nodes that received an
	// output category), not points_calculated (KI_SUCCESS only) — the sum
	// includes failure-fallback categories.
	{
		kriging_stats_t stats;
		stats.m_points_calculated = points_calculated;
		stats.m_points_without_neighbours = points_without_neighbours;
		stats.m_points_singularity = points_singularity;
		stats.m_mean = nodes_processed > 0 ? sum_categories / static_cast<double>(nodes_processed) : 0;
		stats.m_speed_nps = report.iterations_per_second();
		set_kriging_stats(stats);

		// M-6: emit the stderr failure signal on the median-IK failure paths
		// (mirrors cont_kriging.h, simple_cokriging_markI.cpp, indicator_kriging.h,
		// and SIS). Previously singular/no-neighbour systems silently
		// substituted the marginal probability with no observability.
		if (stats.m_points_singularity > 0 || stats.m_points_without_neighbours > 0)
		{
			fprintf(stderr,
				"HPGL: kriging failures: %lu singularity, %lu no-neighbours (of %lu total)\n",
				stats.m_points_singularity,
				stats.m_points_without_neighbours,
				static_cast<unsigned long>(prop_size));
		}
	}
	std::ostringstream oss;
	oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
	write(oss.str());
}
}

