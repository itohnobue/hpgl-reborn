/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


#ifndef __NUMPY_UTILS_H__5A160939_4007_4FAE_94CD_277C1F492D31____
#define __NUMPY_UTILS_H__5A160939_4007_4FAE_94CD_277C1F492D31____

#include "hpgl_exception.h"

namespace hpgl
{
	inline bool check_axis_order(py::object obj)
	{
		bool is_f_contiguous = py::cast<bool>(obj.attr("flags")["F_CONTIGUOUS"]);
		return is_f_contiguous;
	}


	template<typename T, char kind>
	T * get_buffer_from_ndarray(py::object obj, int size, const std::string & context)
	{
		using namespace std;

		if (!check_axis_order(obj))
			throw hpgl_exception(context, "Array is not F_CONTIGUOUS");

		std::string strkind = py::cast<std::string>(obj.attr("dtype").attr("kind"));

		if (strkind[0] != kind)
		{
			std::ostringstream oss;
			oss << "Invalid dtype.kind: " << strkind << ". Expected " << kind << ".";
			throw hpgl_exception(context, oss.str());
		}

		int item_size_2 = py::cast<int>(obj.attr("itemsize"));

		if (item_size_2 != sizeof(T))
		{
			std::ostringstream oss;
			oss << "Invalid itemsize: " << item_size_2 << ". Expected " << sizeof(T) << ".";
			throw hpgl_exception(context, oss.str());
		}

		// Use Python 3 buffer protocol (Py_buffer) instead of removed bf_getwritebuffer
		Py_buffer view;
		PyObject * array_obj = obj.ptr();
		if (PyObject_GetBuffer(array_obj, &view, PyBUF_WRITABLE | PyBUF_SIMPLE) != 0)
		{
			throw hpgl_exception(context, "Failed to get writable buffer from array.");
		}

		Py_ssize_t buf_size = view.len;
		void * result = view.buf;
		PyBuffer_Release(&view);

		if (buf_size != static_cast<Py_ssize_t>(size) * static_cast<Py_ssize_t>(sizeof(T)))
		{
			std::ostringstream oss;
			oss << "Invalid buffer size: " << buf_size << ". Expected " << (static_cast<Py_ssize_t>(size) * sizeof(T)) << ".";
			throw hpgl_exception(context, oss.str());
		}
		return (T*)result;
	}

	inline int get_ndarray_len(py::object obj)
	{
		// Use numpy array's size attribute (total number of elements)
		return py::cast<int>(obj.attr("size"));
	}

	inline std::shared_ptr<indicator_property_array_t> ind_prop_from_tuple(
		py::tuple arr)
	{
		int size = get_ndarray_len(arr[py::int_(0)]);
		std::shared_ptr<indicator_property_array_t> result
			(new indicator_property_array_t(
			get_buffer_from_ndarray<indicator_value_t, 'u'>(arr[py::int_(0)], size, "prop_from_tuple"),
			get_buffer_from_ndarray<unsigned char, 'u'>(arr[py::int_(1)], size, "prop_from_tuple"),
			size,
			py::cast<int>(arr[py::int_(2)])));
		return result;
	}

	inline std::shared_ptr<cont_property_array_t> cont_prop_from_tuple(
		py::tuple arr)
	{
		int size = get_ndarray_len(arr[py::int_(0)]);
		std::shared_ptr<cont_property_array_t> result
			(new cont_property_array_t(
			get_buffer_from_ndarray<cont_value_t, 'f'>(arr[py::int_(0)], size, "prop_from_tuple"),
			get_buffer_from_ndarray<unsigned char, 'u'>(arr[py::int_(1)], size, "prop_from_tuple"),
			size));
		return result;
	}
}

#endif //__NUMPY_UTILS_H__5A160939_4007_4FAE_94CD_277C1F492D31____
