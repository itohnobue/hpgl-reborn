/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


#include "stdafx.h"

#include <lvm_data.h>

#include "py_lvm_data.h"
#include "py_mean_data.h"
#include "generate_complement_array.h"

namespace hpgl
{
	std::shared_ptr<indicator_lvm_data_t> extract_lvm_data(py::object & mean_data, int indicator_count)
	{
		std::shared_ptr<indicator_lvm_data_t> lvm_data;
		bool e1_check = py::isinstance<py_indicator_lvm_data_t>(mean_data);
		bool e2_check = py::isinstance<py::list>(mean_data);
		bool e3_check = py::isinstance<py_mean_data_t>(mean_data);

		if (e1_check)
		{
			py_indicator_lvm_data_t d = py::cast<py_indicator_lvm_data_t>(mean_data);
			lvm_data = d.m_lvm_data;
		}
		if (e3_check)
		{
			std::vector<std::shared_ptr<std::vector<mean_t> > > lvms;
			py_mean_data_t plvm = py::cast<py_mean_data_t>(mean_data);
			if (indicator_count == 1)
			{
				lvms.push_back(plvm.m_data);
			} else if (indicator_count == 2)
			{
				lvm_data.reset(new indicator_lvm_data_t());
				std::shared_ptr<std::vector<mean_t> > lvm0  = generate_complement_array(plvm.m_data);
				lvms.push_back(lvm0);
				lvms.push_back(plvm.m_data);
			} else {
				std::ostringstream oss;
				oss << "Property has " << indicator_count << " indicators, but only one mean array given.";
				throw hpgl_exception("extract_lvm_data", oss.str());
			}
			lvm_data->assign(lvms);
		}
		else if (e2_check)
		{
			lvm_data.reset(new indicator_lvm_data_t());
			std::vector<std::shared_ptr<std::vector<mean_t> > > lvms;

			py::list l = py::cast<py::list>(mean_data);
			int lvm_count = (int) py::len(l);

			if (lvm_count == 1 && indicator_count == 2)
			{
				py_mean_data_t plvm1 = py::cast<py_mean_data_t>(l[py::int_(0)]);
				std::shared_ptr<std::vector<mean_t> > lvm0  = generate_complement_array(plvm1.m_data);
				lvms.push_back(lvm0);
				lvms.push_back(plvm1.m_data);
			}
			else if (indicator_count == lvm_count)
			{
				for (int i = 0; i < lvm_count; ++i)
				{
					py_mean_data_t plvm = py::cast<py_mean_data_t>(l[py::int_(i)]);
					lvms.push_back(plvm.m_data);
				}
			}
			else
			{
				std::ostringstream oss;
				oss << "Property has " << indicator_count << " indicators, but " << lvm_count << " mean arrays given.";
				throw hpgl_exception("hpgl::py_sis_lvm", oss.str());
			}
			lvm_data->assign(lvms);
		}
		return lvm_data;
	}
}
