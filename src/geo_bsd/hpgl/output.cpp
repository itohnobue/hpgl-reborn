#include "stdafx.h"
#include "api.h"
#include <string>
#include <mutex>

// Handler function pointers — set once at startup (before any concurrent calls).
// Protected by s_output_pair_mutex / s_progress_pair_mutex to ensure handler+param
// pair is always read/written atomically.
struct output_handler_pair_t {
	int (*handler)(char * data, void * param);
	void * param;
};
static output_handler_pair_t s_output_pair{nullptr, nullptr};
static std::mutex s_output_pair_mutex;

struct progress_handler_pair_t {
	int (*handler)(char * stage, int percentage, void * param);
	void * param;
};
static progress_handler_pair_t s_progress_pair{nullptr, nullptr};
static std::mutex s_progress_pair_mutex;

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
		int (*h)(char*, void*) = nullptr;
		void *p = nullptr;
		{
			std::lock_guard<std::mutex> lock(s_output_pair_mutex);
			h = s_output_pair.handler;
			p = s_output_pair.param;
		}
		if (h)
		{
			std::lock_guard<std::mutex> lock(s_handler_mutex);
			h(const_cast<char*>(str), p);
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
		int (*ph)(char*, int, void*) = nullptr;
		void *pp = nullptr;
		{
			std::lock_guard<std::mutex> lock(s_progress_pair_mutex);
			ph = s_progress_pair.handler;
			pp = s_progress_pair.param;
		}
		if (ph)
		{
			std::lock_guard<std::mutex> lock(s_handler_mutex);
			return ph(const_cast<char*>(stage), percentage, pp);
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
	std::lock_guard<std::mutex> lock(s_output_pair_mutex);
	s_output_pair.handler = handler;
	s_output_pair.param = param;
}

HPGL_API void hpgl_set_progress_handler(int (*handler)(char * stage, int percentage, void * param), void * param)
{
	std::lock_guard<std::mutex> lock(s_progress_pair_mutex);
	s_progress_pair.handler = handler;
	s_progress_pair.param = param;
}
