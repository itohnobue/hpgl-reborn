#include "stdafx.h"
#include "api.h"
#include <string>
#include <atomic>

// Handler function pointers — set once at startup (before any concurrent calls).
// std::atomic provides defense-in-depth memory ordering even though the contract
// is single-threaded startup configuration. This is the standard ctypes callback
// pattern; synchronization deferred to caller.
static std::atomic<int (*)(char * data, void * param)> s_handler{nullptr};
static std::atomic<void *> s_param{nullptr};

static std::atomic<int (*)(char * stage, int percentage, void * param)> s_progress_handler{nullptr};
static std::atomic<void *> s_progress_handler_param{nullptr};


namespace hpgl
{
	void write(const char * str)
	{
		auto h = s_handler.load(std::memory_order_acquire);
		if (h)
		{
			h(const_cast<char*>(str), s_param.load(std::memory_order_relaxed));
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

	void update_progress(const char * stage, int percentage)
	{
		auto ph = s_progress_handler.load();
		if (ph)
		{
			ph(const_cast<char*>(stage), percentage, s_progress_handler_param.load());
		}
		else
		{
			if (percentage == 0)
			{
				write(std::string(stage) + ": ");
			}
			else if (percentage == -1)
			{
				write("Done.\n");
			}
			else
			{
				write(std::to_string(percentage) + "%... ");
			}
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
