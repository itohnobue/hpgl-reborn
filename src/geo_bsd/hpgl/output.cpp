#include "stdafx.h"
#include "api.h"
#include <string>
#include <atomic>
#include <mutex>

// Handler function pointers — set once at startup (before any concurrent calls).
// std::atomic provides defense-in-depth memory ordering even though the contract
// is single-threaded startup configuration. This is the standard ctypes callback
// pattern; synchronization deferred to caller.
static std::atomic<int (*)(char * data, void * param)> s_handler{nullptr};
// s_param stores an opaque handle to a caller-owned Python object (e.g. a
// Python file-like or StringIO passed via ctypes). The caller MUST ensure the
// object remains alive for the lifetime of all C++ operations that invoke the
// output handler. If Python GC collects the object, this pointer dangles.
// The Python side (hpgl_wrap.py) is responsible for holding a reference.
static std::atomic<void *> s_param{nullptr};

static std::atomic<int (*)(char * stage, int percentage, void * param)> s_progress_handler{nullptr};
static std::atomic<void *> s_progress_handler_param{nullptr};

// Mutex serializes handler invocations from concurrent threads.
// The write() and update_progress() handlers may be called from
// multiple OpenMP threads simultaneously. Without serialization,
// concurrent calls to a Python ctypes callback can corrupt GIL state
// or interleave output.
static std::mutex s_handler_mutex;


namespace hpgl
{
	void write(const char * str)
	{
		auto h = s_handler.load(std::memory_order_acquire);
		if (h)
		{
			std::lock_guard<std::mutex> lock(s_handler_mutex);
			h(const_cast<char*>(str), s_param.load(std::memory_order_relaxed));
		}
		else
		{
			std::lock_guard<std::mutex> lock(s_handler_mutex);
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
		auto ph = s_progress_handler.load();
		if (ph)
		{
			std::lock_guard<std::mutex> lock(s_handler_mutex);
			return ph(const_cast<char*>(stage), percentage, s_progress_handler_param.load());
		}
		else
		{
			std::lock_guard<std::mutex> lock(s_handler_mutex);
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
	s_handler.store(handler);
	s_param.store(param);
}

HPGL_API void hpgl_set_progress_handler(int (*handler)(char * stage, int percentage, void * param), void * param)
{
	s_progress_handler.store(handler);
	s_progress_handler_param.store(param);
}
