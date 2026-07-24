#ifndef CONT_KRIGING_H_INCLUDED_LJLDFJVW450934VDV9ONV09NOASU92N34FOKLSDFGP3Q98SXNP
#define CONT_KRIGING_H_INCLUDED_LJLDFJVW450934VDV9ONV09NOASU92N34FOKLSDFGP3Q98SXNP

#include "combiner.h"
#include "covariance_field.h"
#include "progress_reporter.h"
#include "kriging_stats.h"
#include "output.h"
#include <limits>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "typedefs.h"
#include "select.h"
#include "precalculated_covariance.h"
#include "kriging_interpolation.h"
#include "neighbourhood_lookup.h"
#include "is_informed_predicate.h"
#include "cov_model.h"

// OpenBLAS thread-control API — file-scope declarations for macOS builds
// where extern "C" is not permitted inside function bodies (Apple Clang).
// Define HPGL_USE_OPENBLAS via CMake when building with -DBLA_VENDOR=OpenBLAS
// on macOS.  Linux builds use the __linux__ code-path inside the function
// (GCC accepts extern "C" at block scope as an extension).
#ifdef HPGL_USE_OPENBLAS
extern "C" {
	void openblas_set_num_threads(int);
	int openblas_get_num_threads(void);
}
#endif

namespace hpgl
{
	/*!
 	* Enumerationg specifing ways of handling kriging errors (absence of neighbours or singularity of matrix)
 	*/
	enum kriging_failure_handling
	{
		mean_on_failure, //!< Put the mean value
		undefined_on_failure //!< Leave node undefined
	};

	/*! 
 	*  Generic kriging alghorithm for continuous data. Uses OpenMP.
 	*  
 	*/
	template<
			typename grid_t, //!< Grid-With-Neighbour-Lookup concept
			typename data_t, //!< Property concept
			typename means_t, //!< Mean provider concept
			typename covariances_t, //!< Covariance Model Concept
			typename weight_calculator_t //!< Weight-Calculator concept. 
		>
	void cont_kriging(
			const data_t & input_property, //!< input data
			const grid_t & grid, 
			const neighbourhood_param_t /*ok_params_t*/ & params, //!< parameters of neighbourhood search
			const means_t & means, //!< mean values of data
			const covariances_t & cov, //!< covariance model
			const weight_calculator_t & wc, //!< Weight calculator specifies methods of calculating weights (OK, SK or LVM Kriging)		
			data_t & output_property, //!< resulting data
			progress_reporter_t & report, //!< object for tracking progress
			kriging_stats_t & stats, //!< returns some statistics of calculation
			kriging_failure_handling fh = mean_on_failure //!< Way of handling kriging errors (absence of neighbours, singularity).
		)
	{
		HPGL_CHECK(input_property.size() == output_property.size() && grid.size() == input_property.size(),
			"cont_kriging: size mismatch between grid, input, and output");
		
		double sum = 0;
		stats.m_points_calculated = 0;
		stats.m_points_without_neighbours = 0;
		stats.m_points_singularity = 0;
		stats.m_mean = 0;				
				
		typedef indexed_neighbour_lookup_t<grid_t, covariances_t> nl_t;
		typedef typename nl_t::coord_t coord_t;
		typedef typename data_t::value_type value_t;
		nl_t neighbour_lookup(&grid, &cov, params);

		for (node_index_t node = 0; node < input_property.size(); ++node)
		{
			if (input_property.is_informed(node))
			{
				neighbour_lookup.add_node(node);
			}
		}

		report.start();
		node_index_t idx_end = input_property.size();

		// Pre-initialise uninformed cells to NaN so that unprocessed cells
		// (e.g. after cancellation) are distinguishable from computed cells.
		// The OpenMP loop below overwrites these with computed values or
		// mean-on-failure as each cell is processed.
		for (node_index_t idx = 0; idx < idx_end; ++idx)
		{
			if (!input_property.is_informed(idx))
			{
				output_property.set_at(idx, std::numeric_limits<value_t>::quiet_NaN());
			}
		}

		unsigned long points_calculated = 0;
		unsigned long points_without_neighbours = 0;
		unsigned long points_singularity = 0;
		unsigned long points_processed = 0;
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
#elif defined(HPGL_USE_OPENBLAS)
		// macOS OpenBLAS: limit internal threads to 1.  Declarations are
		// at file scope (above) because Apple Clang rejects extern "C"
		// inside function bodies.  Define HPGL_USE_OPENBLAS via CMake
		// when building with -DBLA_VENDOR=OpenBLAS on macOS.
		detail::blas_thread_acquire(openblas_get_num_threads, openblas_set_num_threads);
#else
		// macOS Accelerate / other BLAS: no thread-count API available.
		// Accelerate manages its own thread pool via GCD and does not
		// oversubscribe with OpenMP in the same way.
#endif
#pragma omp parallel
{
		// Per-thread workspace: pre-allocates vectors once, reused across
		// all node iterations via resize() (allocation-free after first use).
		kriging_ws_t<value_t, coord_t> ws;
		int local_lap_count = 0;
		#pragma omp for schedule(dynamic) reduction(+: points_calculated) reduction(+: points_without_neighbours) reduction(+: points_singularity) reduction(+: points_processed) reduction(+: sum) 
		for(node_index_t idx = 0; idx < idx_end; ++idx)	
		{	
			if (!input_property.is_informed(idx))
			{				
				cont_value_t value;
				switch(kriging_interpolation_ws(input_property, is_informed_predicate_t<data_t>(input_property), idx, cov, means, neighbour_lookup, wc, value, ws))
				{
				case ki_result_t::KI_SUCCESS:
					output_property.set_at(idx, value);
					points_calculated++;				
					points_processed++;
					sum += value;
					break;
				case ki_result_t::KI_NO_NEIGHBOURS:
					points_without_neighbours++;
					if (fh == kriging_failure_handling::mean_on_failure)
					{
						output_property.set_at(idx, means[idx]);
						points_processed++;
						sum += means[idx];
					}
					break;
				case ki_result_t::KI_SINGULARITY:
					++points_singularity;
					if (fh == kriging_failure_handling::mean_on_failure)
					{
						output_property.set_at(idx, means[idx]);
						points_processed++;
						sum += means[idx];
					}
					break;
				}				
			}
			else
			{
				output_property.set_at(idx, input_property.get_at(idx));
			}

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
					#pragma omp cancel for
#endif
					break;
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
		// Cancellation is handled inside the loop body at lines 176-182.
}	

		// Restore BLAS thread count after parallel region completes
#if defined(HPGL_USE_MKL) || defined(USE_INTEL_MKL)
		detail::blas_thread_restore(mkl_set_num_threads);
#elif defined(__linux__)
		detail::blas_thread_restore(openblas_set_num_threads);
#elif defined(HPGL_USE_OPENBLAS)
		detail::blas_thread_restore(openblas_set_num_threads);
#endif

		report.stop();
		stats.m_points_calculated = points_calculated;
		stats.m_points_without_neighbours = points_without_neighbours;
		stats.m_points_singularity = points_singularity;
		stats.m_mean = points_processed > 0 ? sum / points_processed : 0;
		stats.m_speed_nps = report.iterations_per_second();

		// Report kriging failures to stderr so they are visible even when
		// HPGL_LOG_ON is not defined. The Python wrapper reads stats for
		// error propagation; this provides a human-readable warning as well.
		if (stats.m_points_singularity > 0 || stats.m_points_without_neighbours > 0)
		{
			fprintf(stderr,
				"HPGL: kriging failures: %lu singularity, %lu no-neighbours (of %lu total)\n",
				stats.m_points_singularity,
				stats.m_points_without_neighbours,
				static_cast<unsigned long>(idx_end));
		}

		{
			std::ostringstream oss;
			oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
			write(oss.str());
		}
	}
 	
}
#endif //CONT_KRIGING_H_INCLUDED_LJLDFJVW450934VDV9ONV09NOASU92N34FOKLSDFGP3Q98SXNP
