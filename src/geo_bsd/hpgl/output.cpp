#include "stdafx.h"
#include "api.h"
#include <string>
#include <mutex>
#ifdef _OPENMP
#include <omp.h>
#endif

// Handler function pointers — set once at startup (before any concurrent calls).
// Protected by s_output_pair_mutex / s_progress_pair_mutex to ensure handler+param
// pair is always read/written atomically.
struct output_handler_pair_t {
	int (*handler)(char * data, void * param);
	void * param;
};
static output_handler_pair_t s_output_pair{nullptr, nullptr};
// Recursive: the pair mutex is held across the handler invocation (F-16) so a
// concurrent setter cannot free the trampoline mid-invoke; a handler that itself
// calls hpgl_set_output_handler re-enters on the same thread and must not deadlock.
static std::recursive_mutex s_output_pair_mutex;

struct progress_handler_pair_t {
	int (*handler)(char * stage, int percentage, void * param);
	void * param;
};
static progress_handler_pair_t s_progress_pair{nullptr, nullptr};
static std::recursive_mutex s_progress_pair_mutex;

// Mutex serializes handler invocations from concurrent threads.
// The write() and update_progress() handlers may be called from
// multiple OpenMP threads simultaneously. Without serialization,
// concurrent calls to a Python ctypes callback can corrupt GIL state
// or interleave output.
static std::mutex s_handler_mutex;

// Thread-local reentrancy guard (F-53). A Python output/progress handler that
// calls back into HPGL (e.g. a kriging call that itself emits write()/
// update_progress()) would re-enter this module on the same thread. Re-invoking
// the handler would recurse forever; re-locking s_handler_mutex would deadlock.
// While set, reentrant calls fall back to the default std::cout path.
static thread_local bool t_in_handler = false;

// F-M22: progress/output handlers must never be invoked from an OpenMP worker
// thread. A Python handler that re-enters a _hpgl_call_lock-guarded geo API
// (the documented pattern this codebase supports same-thread via RLock) blocks
// forever on the main thread's lock while the main thread waits at the OpenMP
// barrier — a cross-thread reentrant-handler deadlock (empirically reproduced:
// 7/11 callbacks fired on worker threads). Only the master thread (the thread
// that entered the parallel region — i.e. the thread holding the API lock) may
// invoke the handler; worker-thread calls fall back to the default stream.
static bool handler_invocation_allowed()
{
#ifdef _OPENMP
	return !omp_in_parallel() || omp_get_thread_num() == 0;
#else
	return true;
#endif
}


namespace hpgl
{
	void write(const char * str)
	{
		if (t_in_handler)
		{
			// Reentrant call from within a handler invocation on this thread.
			// Fall back to the default stream instead of re-invoking the
			// handler (which would recurse forever) or re-locking the
			// invocation mutex (which would self-deadlock).
			std::cout << "[LOG2]";
			std::cout << str;
			std::cout.flush();
			return;
		}
		// Lock order: s_handler_mutex first, then the pair mutex (consistent
		// with update_progress, so a handler calling a setter on another pair
		// cannot deadlock against a concurrent invocation on that pair).
		std::lock_guard<std::mutex> hlock(s_handler_mutex);
		// Hold the pair mutex across the invocation (F-16): hpgl_set_output_handler
		// blocks until the handler call completes, so a concurrent clear cannot
		// free the trampoline/param we are about to call.
		std::lock_guard<std::recursive_mutex> lock(s_output_pair_mutex);
		// F-M22: never invoke the handler from an OpenMP worker thread (see
		// handler_invocation_allowed) — a worker-thread callback re-entering
		// the API deadlocks against the main thread's lock at the barrier.
		if (s_output_pair.handler && handler_invocation_allowed())
		{
			t_in_handler = true;
			s_output_pair.handler(const_cast<char*>(str), s_output_pair.param);
			t_in_handler = false;
		}
		else
		{
			std::cout << "[LOG2]";
			std::cout << str;
			std::cout.flush();
		}
	}

	void write(const std::string & str)
	{
		write(str.c_str());
	}

	int update_progress(const char * stage, int percentage)
	{
		if (t_in_handler)
		{
			// Reentrant call from within a handler invocation on this thread
			// (F-53) — fall back to the default stream (same semantics as the
			// no-handler path below).
			if (percentage == 0)
			{
				std::cout << stage << ": ";
			}
			else if (percentage == -1)
			{
				std::cout << "Done.\n";
			}
			else
			{
				std::cout << percentage << "%... ";
			}
			std::cout.flush();
			return 0;
		}
		// Lock order: s_handler_mutex first, then the pair mutex (see write()).
		std::lock_guard<std::mutex> hlock(s_handler_mutex);
		// Hold the pair mutex across the invocation (F-16), same rationale as write().
		std::lock_guard<std::recursive_mutex> lock(s_progress_pair_mutex);
		// F-M22: never invoke the handler from an OpenMP worker thread (see
		// handler_invocation_allowed). A progress handler re-entering a
		// _hpgl_call_lock-guarded geo API from a worker blocks forever while
		// the main thread waits at the OpenMP barrier (empirically reproduced).
		if (s_progress_pair.handler && handler_invocation_allowed())
		{
			t_in_handler = true;
			int rc = s_progress_pair.handler(const_cast<char*>(stage), percentage, s_progress_pair.param);
			t_in_handler = false;
			return rc;
		}
		else
		{
			if (percentage == 0)
			{
				std::cout << stage << ": ";
			}
			else if (percentage == -1)
			{
				std::cout << "Done.\n";
			}
			else
			{
				std::cout << percentage << "%... ";
			}
			std::cout.flush();
			return 0;
		}
	}
}

HPGL_API void hpgl_set_output_handler(int (*handler)(char * data, void * param), void * param)
{
	std::lock_guard<std::recursive_mutex> lock(s_output_pair_mutex);
	s_output_pair.handler = handler;
	s_output_pair.param = param;
}

HPGL_API void hpgl_set_progress_handler(int (*handler)(char * stage, int percentage, void * param), void * param)
{
	std::lock_guard<std::recursive_mutex> lock(s_progress_pair_mutex);
	s_progress_pair.handler = handler;
	s_progress_pair.param = param;
}
