/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


#include "stdafx.h"
#include "py_grid.h"
#include "neighbourhood_param.h"
#include "covariance_param.h"
#include <hpgl_core.h>

#include "numpy_utils.h"


namespace hpgl
{
	namespace python
	{
		void py_simple_cokriging_markI(
			py::tuple input_array,
			const py_grid_t & grid,
			py::tuple secondary_data,
			mean_t primary_mean,
			mean_t secondary_mean,
			double secondary_variance,
			double correlation_coef,
			const py::tuple & radiuses,
			int max_neighbours,
			int covariance_type,
			const py::tuple & ranges,
			double sill,
			double nugget,
			const py::tuple & angles,
			py::tuple out_array)
		{
			if (py::len(radiuses) != 3) {
				std::ostringstream oss;
				oss << "len(radiuses) = " << py::len(radiuses) << ". Should be 3";
				throw hpgl_exception("py_simple_cokriging_markI", oss.str());
			}
			if (py::len(ranges) != 3) {
				std::ostringstream oss;
				oss << "len(ranges) = " << py::len(ranges) << ". Should be 3";
				throw hpgl_exception("py_simple_cokriging_markI", oss.str());
			}
			if (py::len(angles) != 3) {
				std::ostringstream oss;
				oss << "len(angles) = " << py::len(angles) << ". Should be 3";
				throw hpgl_exception("py_simple_cokriging_markI", oss.str());
			}

			neighbourhood_param_t np;
			np.m_max_neighbours = max_neighbours;

			covariance_param_t pcp;
			pcp.m_nugget = nugget;
			pcp.m_sill = sill;
			pcp.m_covariance_type = (covariance_type_t) covariance_type;

			for (int i = 0; i < 3; ++i)
			{
				np.m_radiuses[i] = py::cast<int>(radiuses[py::int_(i)]);
			       	pcp.m_ranges[i] = py::cast<double>(ranges[py::int_(i)]);
		       		pcp.m_angles[i] = py::cast<double>(angles[py::int_(i)]);
			}

			sp_double_property_array_t primary_prop = cont_prop_from_tuple(input_array);
			sp_double_property_array_t secondary_prop = cont_prop_from_tuple(secondary_data);
			sp_double_property_array_t out_prop = cont_prop_from_tuple(out_array);


			simple_cokriging_markI(*grid.m_sugarbox_geometry, *primary_prop,
				*secondary_prop, primary_mean, secondary_mean,
				secondary_variance, correlation_coef, np, pcp, *out_prop);
		}

		void py_simple_cokriging_markII(
			py_grid_t grid,
			py::dict primary_data,
			py::dict secondary_data,
			double correlation_coef,
			py::tuple radiuses,
			int max_neighbours,
			py::tuple out_array)
		{
			sp_double_property_array_t input_prop =
				cont_prop_from_tuple(py::cast<py::tuple>(primary_data["data"]));
			sp_double_property_array_t secondary_prop =
				cont_prop_from_tuple(py::cast<py::tuple>(secondary_data["data"]));
			sp_double_property_array_t out_prop =
				cont_prop_from_tuple(out_array);

			mean_t primary_mean = py::cast<mean_t>(primary_data["mean"]);
			mean_t secondary_mean = py::cast<mean_t>(secondary_data["mean"]);

			py::object primary_cov_model = primary_data["cov_model"];
			py::object secondary_cov_model = secondary_data["cov_model"];

			py::object primary_ranges = primary_cov_model.attr("ranges");
			py::object secondary_ranges = secondary_cov_model.attr("ranges");

			py::object primary_angles = primary_cov_model.attr("angles");
			py::object secondary_angles = secondary_cov_model.attr("angles");

			double primary_sill = py::cast<double>(primary_cov_model.attr("sill"));
			double secondary_sill = py::cast<double>(secondary_cov_model.attr("sill"));

			double primary_nugget = py::cast<double>(primary_cov_model.attr("nugget"));
			double secondary_nugget = py::cast<double>(secondary_cov_model.attr("nugget"));

			covariance_type_t primary_cov_type =  (covariance_type_t)((int)py::cast<int>(primary_cov_model.attr("type")));
			covariance_type_t secondary_cov_type =   (covariance_type_t)((int)py::cast<int>(secondary_cov_model.attr("type")));

			covariance_param_t primary_cov_params;
			covariance_param_t secondary_cov_params;
			neighbourhood_param_t np;
			np.m_max_neighbours = max_neighbours;

			for (int i = 0; i < 3; ++i)
			{
				np.m_radiuses[i] = py::cast<int>(radiuses[py::int_(i)]);
			    primary_cov_params.m_ranges[i] = py::cast<double>(primary_ranges[py::int_(i)]);
		       	primary_cov_params.m_angles[i] = py::cast<double>(primary_angles[py::int_(i)]);
				secondary_cov_params.m_ranges[i] = py::cast<double>(secondary_ranges[py::int_(i)]);
		       	secondary_cov_params.m_angles[i] = py::cast<double>(secondary_angles[py::int_(i)]);
			}
			primary_cov_params.m_sill = primary_sill;
			primary_cov_params.m_nugget = primary_nugget;
			primary_cov_params.m_covariance_type = primary_cov_type;

			secondary_cov_params.m_sill = secondary_sill;
			secondary_cov_params.m_nugget = secondary_nugget;
			secondary_cov_params.m_covariance_type = secondary_cov_type;

			simple_cokriging_markII(
				*grid.m_sugarbox_geometry,
				*input_prop,
				*secondary_prop,
				primary_mean, secondary_mean, correlation_coef, np, primary_cov_params, secondary_cov_params, *out_prop);
		}

	}
}
