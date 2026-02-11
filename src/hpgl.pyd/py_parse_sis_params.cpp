/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


#include "stdafx.h"

#include "ik_params.h"
#include "hpgl_exception.h"

namespace hpgl
{
	namespace
	{
		const char * c_param_cov_model = "cov_model";
		const char * c_param_cov_type = "type";
		const char * c_param_ranges = "ranges";
		const char * c_param_angles = "angles";
		const char * c_param_sill = "sill";
		const char * c_param_nugget = "nugget";
		const char * c_param_radiuses = "radiuses";
		const char * c_param_max_neighbours = "max_neighbours";
		const char * c_param_marg_prob = "marginal_prob";

		bool is_valid_parameter(const std::string param_name)
		{
			return
				param_name == c_param_cov_model ||
				param_name == c_param_radiuses ||
				param_name == c_param_max_neighbours ||
				param_name == c_param_marg_prob;
		}

		void check_parameters(PyObject * params_dictionary)
		{
			PyObject * keys = PyDict_Keys(params_dictionary);
			Py_ssize_t idx, size;
			size = PyList_Size(keys);
			for (idx = 0; idx < size; ++idx)
			{
				char * key = PyString_AsString(PyList_GetItem(keys, idx));
				if (!is_valid_parameter(key)) {
					std::ostringstream oss;
					oss << "Invalid sis parameter '" << key << "'";
					throw hpgl_exception("py_sis", oss.str());
				}
			}
		}
	}

	void parse_sis_params(py::object params, ik_params_t & ik_params)
	{
		using namespace hpgl;
		Py_ssize_t idx, end_idx;
		end_idx = py::len(params);
		for (idx = 0; idx < end_idx; ++idx)
		{

			indicator_param_t indicator_param;
			py::object ik_dict = params[idx];
			check_parameters(ik_dict.ptr());
			py::object cov_model            = ik_dict[c_param_cov_model];
			py::object radiuses             = ik_dict[c_param_radiuses];
			py::object max_neighbours       = ik_dict[c_param_max_neighbours];
			py::object marginal_probability = ik_dict[c_param_marg_prob];

			indicator_param.m_covariance_type = (covariance_type_t)(int) py::cast<int>(cov_model.attr(c_param_cov_type));

			{
				py::object ranges = cov_model.attr(c_param_ranges);
				double r1 = py::cast<double>(ranges[py::int_(0)]);
				double r2 = py::cast<double>(ranges[py::int_(1)]);
				double r3 = py::cast<double>(ranges[py::int_(2)]);
				indicator_param.set_ranges(r1, r2, r3);
			}

			{
				py::object angles = cov_model.attr(c_param_angles);
				double a1 = py::cast<double>(angles[py::int_(0)]);
				double a2 = py::cast<double>(angles[py::int_(1)]);
				double a3 = py::cast<double>(angles[py::int_(2)]);
				indicator_param.set_angles(a1, a2, a3);
			}

			indicator_param.m_sill = py::cast<double>(cov_model.attr(c_param_sill));
			indicator_param.m_nugget = py::cast<double>(cov_model.attr(c_param_nugget));
			{
				int r1 = py::cast<int>(radiuses[py::int_(0)]);
				int r2 = py::cast<int>(radiuses[py::int_(1)]);
				int r3 = py::cast<int>(radiuses[py::int_(2)]);
				indicator_param.set_radiuses(r1, r2, r3);
			}
			indicator_param.m_max_neighbours = py::cast<int>(max_neighbours);
			indicator_param.m_marginal_prob = py::cast<double>(marginal_probability);
			ik_params.add_indicator(indicator_param);
		}
	}
}
