#include "stdafx.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mutex>

namespace hpgl
{
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
	/// @param n_threads Number of threads (must be > 0). If 0, implementation-defined.
	/// @return true if successful, false if n_threads is invalid
	bool set_thread_num(int n_threads)
	{
		// Input validation
		if (n_threads < 0) {
			return false;
		}

#ifdef _OPENMP
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
