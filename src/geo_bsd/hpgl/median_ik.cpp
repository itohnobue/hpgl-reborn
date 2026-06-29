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
		#pragma omp for schedule(dynamic)
		for (node_index_t idx = 0; idx < prop_size; ++idx)
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
	std::ostringstream oss;
	oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
	write(oss.str());
}
}

