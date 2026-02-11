#ifndef __OUTPUT_H__AA0B5121_31E3_4DFA_9EEB_C1EACF2AC47B
#define __OUTPUT_H__AA0B5121_31E3_4DFA_9EEB_C1EACF2AC47B

#include <string>

namespace hpgl
{
	void write(const char * str);
	void write(const std::string & str);

	void update_progress(const char * stage, int percentage);
}

// Simplified logging macro - uses string concatenation instead of boost::format
#define LOGWARNING(msg) hpgl::write(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + __FUNCTION__ + " Warning: " + msg)

#endif //__OUTPUT_H__AA0B5121_31E3_4DFA_9EEB_C1EACF2AC47B
