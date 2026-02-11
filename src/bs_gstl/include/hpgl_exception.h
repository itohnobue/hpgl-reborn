/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


#ifndef HPGL_EXCEPTION_H_INCLUDED_3904FHNEWC94323VM0349MNP0ENAPW934RNVPSEVRVQW5434CV54VG34
#define HPGL_EXCEPTION_H_INCLUDED_3904FHNEWC94323VM0349MNP0ENAPW934RNVPSEVRVQW5434CV54VG34

#include <stdexcept>
#include <string>

namespace hpgl
{
	class hpgl_exception : public std::exception
	{
		std::string m_where;
		std::string m_what;
		std::string m_message;
	public:
		hpgl_exception(const std::string & a_where, const std::string & what);

		virtual const char* what()const noexcept
		{
			return m_message.c_str();
		}

		~hpgl_exception() override = default;
	};
}

#endif //HPGL_EXCEPTION_H_INCLUDED_3904FHNEWC94323VM0349MNP0ENAPW934RNVPSEVRVQW5434CV54VG34
