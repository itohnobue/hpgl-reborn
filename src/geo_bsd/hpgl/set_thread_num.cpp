#include "stdafx.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mutex>

namespace hpgl
{
	// F-193: hard upper bound for the OpenMP thread count. The Python
	// wrapper only WARNS above 4x the CPU count (geo.py set_thread_num) and
	// the C API validated nothing above zero, so a caller could ask libomp
	// for an arbitrary thread team — each parallel region then reserves that
	// many stacks/TLS blocks and the run fails deep inside the runtime
	// (thread-creation failure) instead of at the call. 1024 exceeds the
	// core count of any current single-image machine (the largest shipping
	// x86/ARM parts stay below 512) while leaving a numeric margin for
	// deeply nested partitions; a system that genuinely needs more than
	// this must raise the constant deliberately.
	static const int MAX_OMP_THREADS = 1024;

	// Thread-safe wrapper for OpenMP settings
	// Note: OpenMP runtime functions are generally thread-safe, but we add
	// documentation and validation for clarity.

	/// Sets the number of OpenMP threads to use for parallel regions.
	///
	/// Thread Safety: This function affects the global OpenMP thread pool.
	/// All subsequent parallel regions will use the specified thread count.
	///
	/// WARNING: If called from within a parallel region, behavior is undefined.
	/// WARNING: Not thread-safe with concurrent calls to set_thread_num().
	///          Ensure external synchronization if multiple threads may call this.
	///
	/// @param n_threads Number of threads (must be > 0 and <= 1024; 0 is
	///        implementation-defined and rejected here).
	/// @return true if successful, false if n_threads is invalid
	bool set_thread_num(int n_threads)
	{
		// Input validation: reject negative and zero, as OpenMP leaves 0
		// implementation-defined.  The Python wrapper also rejects zero.
		// Without this guard n_threads == 0 may silently map to
		// implementation-specific threading on some OpenMP runtimes.
		if (n_threads <= 0) {
			return false;
		}

		// F-193: bound the request above. omp_set_num_threads() accepts any
		// positive value, so without this gate an unbounded count from the
		// caller is only rejected much later by the runtime (thread-creation
		// failure inside the first parallel region) — or, worse, succeeds and
		// oversubscribes the machine. Reject it here where the caller can see
		// it and the C API can turn it into a clean error.
		if (n_threads > MAX_OMP_THREADS) {
			fprintf(stderr,
				"[HPGL WARNING] set_thread_num(%d) exceeds the supported "
				"maximum of %d threads — rejected.\n",
				n_threads, MAX_OMP_THREADS);
			fflush(stderr);
			return false;
		}

#ifdef _OPENMP
		// OpenMP §2.5: calling omp_set_num_threads from within an active
		// parallel region is undefined behaviour.  Guard against this.
		// The caller should set the thread count before entering the
		// parallel region; if we're already inside one, no-op with a
		// diagnostic so that callers can detect the misuse.
		if (omp_in_parallel()) {
			fprintf(stderr,
				"[HPGL WARNING] set_thread_num(%d) called from inside "
				"a parallel region — ignored. Call before entering "
				"the parallel region.\n", n_threads);
			fflush(stderr);
			return false;
		}
		// OpenMP handles invalid values (e.g., negative values already filtered)
		// omp_set_num_threads is thread-safe with respect to OpenMP runtime
		omp_set_num_threads(n_threads);
#else
		(void)n_threads; // suppress unused parameter warning
#endif
		return true;
	}

	/// Gets the maximum number of OpenMP threads that could be used.
	///
	/// Thread Safety: Thread-safe. Returns the value set by set_thread_num()
	/// or the default OpenMP value if never set.
	///
	/// @return Maximum number of threads available for parallel regions
	int get_thread_num()
	{
#ifdef _OPENMP
		// omp_get_max_threads() is thread-safe
		return omp_get_max_threads();
#else
		return 1;
#endif
	}

	/// Gets the number of threads currently in the team executing the parallel region.
	/// If called outside a parallel region, returns 1.
	///
	/// Thread Safety: Thread-safe.
	///
	/// @return Number of threads in the current parallel team
	int get_current_thread_num()
	{
#ifdef _OPENMP
		// omp_get_num_threads() is thread-safe
		return omp_get_num_threads();
#else
		return 1;
#endif
	}

	/// Checks if currently executing within a parallel region.
	///
	/// Thread Safety: Thread-safe.
	///
	/// @return true if inside a parallel region, false otherwise
	bool in_parallel_region()
	{
#ifdef _OPENMP
		// omp_in_parallel() is thread-safe
		return omp_in_parallel() != 0;
#else
		return false;
#endif
	}
}
