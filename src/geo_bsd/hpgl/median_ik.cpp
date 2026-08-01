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
	// process-wide BLAS thread count.
#if defined(HPGL_USE_MKL) || defined(USE_INTEL_MKL)
	extern "C" void mkl_set_num_threads(int);
	extern "C" int mkl_get_max_threads(void);
	detail::blas_thread_acquire(mkl_get_max_threads, mkl_set_num_threads);
#elif defined(__linux__)
	// OpenBLAS: limit internal threads to 1 so they don't multiply
	// with OpenMP threads. Declared extern to avoid header dependency.
	extern "C" void openblas_set_num_threads(int);
	extern "C" int openblas_get_num_threads(void);
	detail::blas_thread_acquire(openblas_get_num_threads, openblas_set_num_threads);
#else
	// macOS Accelerate / other BLAS: no thread-count API available.
	// Accelerate manages its own thread pool via GCD and does not
	// oversubscribe with OpenMP in the same way.
#endif
	#pragma omp parallel
	{
		int local_lap_count = 0;
		#pragma omp for schedule(dynamic)
		for (size_t idx = 0; idx < prop_size; ++idx)
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

			ind_index = choose_indicator(prob);
			output_property.set_at(idx, ind_index);
			
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
#ifdef _OPENMP
					// OpenMP §2.11.2: break is forbidden inside a
					// worksharing-loop construct. Use cancel-for to
					// prevent new iterations; the current iteration
					// finishes naturally (no more real work after
					// this check). In single-threaded builds (no
					// OpenMP), the plain 'break' is standard-conformant.
					#pragma omp cancel for
#else
					break;
#endif
				}
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

	// Restore BLAS thread count after parallel region completes
#if defined(HPGL_USE_MKL) || defined(USE_INTEL_MKL)
	detail::blas_thread_restore(mkl_set_num_threads);
#elif defined(__linux__)
	detail::blas_thread_restore(openblas_set_num_threads);
#endif

	report.stop();
	std::ostringstream oss;
	oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
	write(oss.str());
}
}

