#include "stdafx.h"
/**
 * \file bs_assert.cpp
 * \brief bs_assert impl
 * \author Sergey Miryanov
 * \date 14.05.2008
 */
#include "bs_assert.h"
#include <stdlib.h>
#include <cassert>
#include <iostream>

 //#include "bos_report.h"

namespace blue_sky {
	namespace bs_assert {

		//BS_API_PLUGIN assert_factory *asserter::factory = 0;

		asserter::assert_state
			asserter::ask_user() const
		{
#ifdef _DEBUG
#if defined (_WIN32)
			int return_code = _CrtDbgReport(_CRT_ERROR, file, line, "", "Expression: %s\nValues: %s", cond_s, var_list.empty() ? "" : var_list.c_str());

			if (return_code == -1)
				return assert_state::STATE_KILL;
			else if (return_code == 0)
				return assert_state::STATE_IGNORE;
			else if (return_code == 1)
				return assert_state::STATE_BREAK;
#else // defined UNIX
			std::cout << "Assertion! '" << cond_s << "' file:'" << file << "',line:" << line << ". What do you want to do: [k]ill,[d]ebug,[i]gnore,ignore [a]ll? ";
			char return_code;
			std::cin >> return_code;

			if (return_code == 'k')
				return STATE_KILL;
			else if (return_code == 'i')
				return STATE_IGNORE;
			else if (return_code == 'd')
				return STATE_BREAK;
			else if (return_code == 'a')
				return STATE_IGNORE_ALL;
#endif	// #if defined (_WIN32)
#endif	// #ifdef _DEBUG

			return assert_state::STATE_IGNORE;
		}

		bool
			asserter::handle() const
		{
			if (cond || ignore_all())
				return true;

			assert_state state = ask_user();
			switch (state)
			{
			case assert_state::STATE_KILL:
				exit(1);
				break;
			case assert_state::STATE_BREAK:
				break;
			case assert_state::STATE_IGNORE:
				return true;
			case assert_state::STATE_IGNORE_ALL:
				ignore_all() = true;
				return true;
			}

			return false;
		}

		asserter*
			asserter::make(bool b)
		{
			if (factory())
				return factory()->make(b, file, line, cond_s);

			return new asserter(b, file, line, cond_s);
		}

		asserter*
			assert_factory::make(bool b, const char* file, int line, const char* cond_str)
		{
			return new asserter(b, file, line, cond_str);
		}

	} // namespace bs_assert
} // namespace blue_sky

