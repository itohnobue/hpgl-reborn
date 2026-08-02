#ifndef CONT_KRIGING_H_INCLUDED_LJLDFJVW450934VDV9ONV09NOASU92N34FOKLSDFGP3Q98SXNP
#define CONT_KRIGING_H_INCLUDED_LJLDFJVW450934VDV9ONV09NOASU92N34FOKLSDFGP3Q98SXNP

#include "combiner.h"
#include "covariance_field.h"
#include "progress_reporter.h"
#include "kriging_stats.h"
#include "output.h"
#include <exception>
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
// HPGL_USE_OPENBLAS is defined by CMake ONLY when the build actually links
// OpenBLAS (BLA_VENDOR=OpenBLAS or BLAS library-name detection), so the
// openblas_* symbols are never referenced on Linux builds that link a
// different BLAS vendor (M-32: undefined-symbol dlopen failure with
// netlib/BLIS/FlexiBLAS/ATLAS).
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

		// No NaN pre-initialisation: output cells that cannot be kriged
		// (no neighbours / singular system under undefined_on_failure) keep
		// their initial value (0.0) and mask=0, so they remain uninformed.
		// The mask is the authoritative "informed" signal — NaN data is not
		// written because the Python wrapper rejects NaN output with
		// RuntimeError, and the tests expect graceful completion for
		// sparse/empty data (cells left uninformed, not NaN).

		unsigned long points_calculated = 0;
		unsigned long points_without_neighbours = 0;
		unsigned long points_singularity = 0;
		unsigned long points_processed = 0;
		static const int LP_BATCH_SIZE = 1000;

		// BLAS thread control: prevent oversubscription when BLAS internal
		// threading combines with OpenMP parallel region threads.
		// Reference-counted: concurrent kriging calls safely share the
		// process-wide BLAS thread count. RAII guard (F-M8): the process-wide
		// BLAS thread count is restored even when an exception unwinds
		// through the parallel region (previously a plain acquire/restore
		// pair left BLAS pinned to 1 thread if bad_alloc escaped).
#if defined(HPGL_USE_MKL) || defined(USE_INTEL_MKL)
		extern "C" void mkl_set_num_threads(int);
		extern "C" int mkl_get_max_threads(void);
		detail::blas_thread_guard_t blas_guard(mkl_get_max_threads, mkl_set_num_threads);
#elif defined(HPGL_USE_OPENBLAS)
		// OpenBLAS (any platform): limit internal threads to 1 so they don't
		// multiply with OpenMP threads.  HPGL_USE_OPENBLAS is defined by
		// CMake ONLY when the build actually links OpenBLAS (BLA_VENDOR=OpenBLAS
		// or BLAS library-name detection).  On Linux with netlib/BLIS/
		// FlexiBLAS/ATLAS the macro is undefined, so the openblas_* symbols
		// are never referenced — no undefined-symbol dlopen failure of
		// libhpgl.so (M-32).  Declarations are at file scope (above) because
		// Apple Clang rejects extern "C" inside function bodies.
		detail::blas_thread_guard_t blas_guard(openblas_get_num_threads, openblas_set_num_threads);
#else
		// Other BLAS (Linux netlib/BLIS/FlexiBLAS/ATLAS, macOS Accelerate, …):
		// no thread-count API available.  Accelerate manages its own thread
		// pool via GCD and does not oversubscribe with OpenMP in the same way.
#endif
		// F-M9: an allocation failure (e.g. ws.A.resize at
		// my_kriging_weights.h:256 with a huge coord_size) inside the OpenMP
		// worksharing loop must not escape the region (that is UB → the C ABI
		// catch never sees it; the process aborts). Catch inside the region,
		// record the exception, cancel the loop, and rethrow after the region
		// so the C API converts it to a clean hpgl_exception error.
		std::exception_ptr parallel_error;
		// M-5: cooperative cancellation flag shared by all threads.
		// #pragma omp cancel for is a silent no-op in every default build
		// (the cancel-var ICV is false; OMP_CANCELLATION is never set), so the
		// documented user-cancel feature was never effective on the parallel
		// path. Instead, each thread checks this atomic flag at the top of
		// every iteration and skips the expensive kriging body once set. In
		// non-OpenMP builds a plain break is legal (the loop is serial).
		std::atomic<bool> loop_stop{false};
#pragma omp parallel
{
		// Per-thread workspace: pre-allocates vectors once, reused across
		// all node iterations via resize() (allocation-free after first use).
		kriging_ws_t<value_t, coord_t> ws;
		int local_lap_count = 0;
		#pragma omp for schedule(dynamic) reduction(+: points_calculated) reduction(+: points_without_neighbours) reduction(+: points_singularity) reduction(+: points_processed) reduction(+: sum) 
		for(node_index_t idx = 0; idx < idx_end; ++idx)	
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
					// M-5: set the cooperative stop flag instead of
					// `#pragma omp cancel for` (a silent no-op in default
					// builds). Other threads observe the flag at their next
					// iteration and skip the kriging body.
					loop_stop.store(true, std::memory_order_relaxed);
				}
			}
			} // try
			catch (const std::exception &)
			{
				// F-M9: allocation failure inside the region. Record the
				// exception and cancel the worksharing loop; rethrow after
				// the region so the C API catch converts it to a clean
				// error. Without this the exception escapes the region
				// (UB per OpenMP §2.13.6) and the process aborts before
				// the C ABI catch can run.
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
		// Cancellation is handled inside the loop body at lines 176-182.
}

		// F-M9: rethrow the recorded allocation failure on the calling
		// thread (outside the parallel region). The RAII blas_guard above is
		// destroyed during unwind, restoring the BLAS thread count.
		if (parallel_error)
			std::rethrow_exception(parallel_error);

		report.stop();
		// points_calculated semantics (F-N6): count ONLY successfully kriged
		// cells in BOTH failure-handling modes.
		//  - mean_on_failure (SK/LVM): KI_SUCCESS cells only — the Python
		//    F-33 warning contract ("calculated < expected" detects
		//    no-neighbour mean-fill) fires when failures occur.
		//  - undefined_on_failure (OK): previously counted every uninformed
		//    cell the loop processed (successes + no-neighbour + singular),
		//    which made m_points_calculated == expected always and the
		//    _check_kriging_failure_stats warning branch (geo.py:1272) a
		//    no-op for OK. Counting successes only lets that branch fire.
		//    The "ran but everything failed" vs "never ran" distinction is
		//    preserved via points_without_neighbours / points_singularity,
		//    which remain nonzero when cells were left undefined.
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
