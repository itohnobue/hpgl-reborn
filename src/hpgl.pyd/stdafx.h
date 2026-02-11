/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


#ifndef __STDAFX_H__C0C62E4A_23E8_4FD8_9C6C_9361610D7977
#define __STDAFX_H__C0C62E4A_23E8_4FD8_9C6C_9361610D7977

//#define _SECURE_SCL 0


#include <vector>
#include <deque>
#include <list>
#include <map>
#include <string>
#include <iostream>
#include <time.h>
#include <fstream>
#include <ostream>
#include <sstream>
#include <exception>
#include <memory>

// Modern C++ standard library replaces boost::smart_ptr
// Using std::shared_ptr, std::unique_ptr instead

// String operations use standard library
// Using std::stringstream, std::string instead of boost::format

// Pybind11 for Python bindings (replaces boost::python)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#endif //__STDAFX_H__C0C62E4A_23E8_4FD8_9C6C_9361610D7977
