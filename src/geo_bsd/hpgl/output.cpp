#include "stdafx.h"
#include "api.h"
#include <string>

static int (*s_handler)(char * data, void * param) = nullptr;
static void * s_param = nullptr;


static int (*s_progress_handler)(char * stage, int percentage, void * param) = nullptr;
static void * s_progress_handler_param = nullptr;


namespace hpgl
{
	void write(const char * str)
	{
		if (s_handler)
		{
			s_handler((char*)str, s_param);
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
		if (s_progress_handler)
		{
			s_progress_handler((char*)stage, percentage, s_progress_handler_param);
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
	s_handler = handler;
	s_param = param;
}

HPGL_API void hpgl_set_progress_handler(int (*handler)(char * stage, int percentage, void * param), void * param)
{
	s_progress_handler  = handler;
	s_progress_handler_param = param;
}
