/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


#include "stdafx.h"

#include <property_array.h>
#include <property_writer.h>
#include "py_ok_param.h"
#include "py_sk_params.h"
#include "py_grid.h"
#include "py_sgs_params.h"
#include <load_property_from_file.h>
#include "py_median_ik.h"
#include <hpgl_core.h>
#include "progress_reporter.h"
#include "extract_indicator_values.h"
#include "py_kriging_weights.h"

using namespace hpgl;

namespace hpgl
{
	void py_indicator_kriging(
		py::tuple input_array,
		py::tuple output_array,
		const py_grid_t & grid,
		py::object params);

	void py_sis(
		py::tuple array,
		const py_grid_t & grid,
		py::object params, int seed, bool use_vpc, bool use_corellogram, py::object mask_data);

	void py_sis_lvm(
			const py::tuple & array,
			const py_grid_t & grid,
			py::object params,
			int seed,
			py::object mean_data,
			bool use_corellogram,
			py::object mask_data);

	namespace python
	{
		void py_read_inc_file_float(
			const std::string & filename,
			double undefined_value,
			int size,
			py::object odata,
			py::object omask);

		void py_read_inc_file_byte(
			const std::string & filename,
			int undefined_value,
			int size,
			py::object data,
			py::object mask,
			py::list ind_values);

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
			py::tuple out_array);

		void py_simple_cokriging_markII(
			py_grid_t grid,
			py::dict primary_data,
			py::dict secondary_data,
			double correlation_coef,
			py::tuple radiuses,
			int max_neighbours,
			py::tuple out_array);


	}
}

void write_byte_property(
		py::tuple property,
		const char * filename,
		const char * name,
		double undefined_value,
		py::list indicator_values)
{
	std::vector<unsigned char> remap_table;
	sp_byte_property_array_t prop = ind_prop_from_tuple(property);
	extract_indicator_values(
			indicator_values,
			indicator_count(*prop),
			remap_table);

	hpgl::property_writer_t writer;
	writer.init(filename, name);
	writer.write_byte(prop, (unsigned char)undefined_value, remap_table);
}

void write_cont_property(
		py::tuple property,
		const char * filename,
		const char * name,
		double undefined_value)
{
	sp_double_property_array_t prop = cont_prop_from_tuple(property);
	hpgl::property_writer_t writer;
	writer.init(filename, name);
	writer.write_double(prop, undefined_value);
}

// Hmm... can't we just give constructor to py_grid_t?
py_grid_t create_sugarbox_grid(int x, int y, int z)
{
	using namespace hpgl;

	py_grid_t result;

	sp_sugarbox_grid_t grid(new sugarbox_grid_t());
	grid->init(x, y, z);

	result.m_sugarbox_geometry = grid;
	return result;
}

py_ok_params_t create_ok_params()
{
	return py_ok_params_t();
}

py_sk_params_t create_sk_params()
{
	return py_sk_params_t();
}

py_sgs_params_t create_sgs_params()
{
	return py_sgs_params_t();
}



void ordinary_kriging(
		py::tuple input_array,
		py::tuple output_array,
		py_grid_t grid,
		const py_ok_params_t & param,
		bool use_new_cov)
{
	using namespace hpgl;

	sp_double_property_array_t in_prop = cont_prop_from_tuple(input_array);
	sp_double_property_array_t out_prop = cont_prop_from_tuple(output_array);

	hpgl::ordinary_kriging(*in_prop, *grid.m_sugarbox_geometry, param.m_ok_param, *out_prop, use_new_cov);
}

void simple_kriging(
		py::tuple input_array,
		py::tuple output_array,
		const py_grid_t & grid,
		const py_sk_params_t & param)
{
	using namespace hpgl;
	sp_double_property_array_t in_prop = cont_prop_from_tuple(input_array);
	sp_double_property_array_t out_prop = cont_prop_from_tuple(output_array);
	hpgl::simple_kriging(*in_prop, *grid.m_sugarbox_geometry, param.m_sk_params, *out_prop);
}

void lvm_kriging(
		py::tuple input_array,
		py::tuple output_array,
		const py_grid_t & grid,
		const py_ok_params_t & param,
		py::object mean_data)
{
	using namespace hpgl;
	sp_double_property_array_t in_prop = cont_prop_from_tuple(input_array);
	sp_double_property_array_t out_prop = cont_prop_from_tuple(output_array);
	mean_t * means = get_buffer_from_ndarray<mean_t, 'f'>(mean_data, out_prop->size(), "lvm_kriging");
	hpgl::lvm_kriging(*in_prop, means, *grid.m_sugarbox_geometry, param.m_ok_param, *out_prop);
}

void sgs_simulation(
		py::tuple output_array,
		const py_grid_t & grid,
		const py_sgs_params_t & param,
		py::object cdf_property,
		py::object mask_data)
{
	sp_double_property_array_t out_prop = cont_prop_from_tuple(output_array);

	unsigned char * mask =
			mask_data.ptr() == Py_None ? nullptr
			: get_buffer_from_ndarray<unsigned char,'u'>(mask_data, out_prop->size(), "sgs_simulation");

	hpgl_non_parametric_cdf_t cdf_data;
	const hpgl_non_parametric_cdf_t * cdf_ptr = nullptr;
	try
	{
		py::tuple cdf_tuple = py::cast<py::tuple>(cdf_property);
		int cdf_size = get_ndarray_len(cdf_tuple[py::int_(0)]);
		cdf_data.m_values = get_buffer_from_ndarray<float, 'f'>(cdf_tuple[py::int_(0)], cdf_size, "sgs_simulation cdf values");
		cdf_data.m_probs = get_buffer_from_ndarray<float, 'f'>(cdf_tuple[py::int_(1)], cdf_size, "sgs_simulation cdf probs");
		cdf_data.m_size = cdf_size;
		cdf_ptr = &cdf_data;
	}
	catch (const py::cast_error &)
	{
		cdf_ptr = nullptr;
	}

	hpgl::sequential_gaussian_simulation(*grid.m_sugarbox_geometry, param.m_sgs_params, *out_prop, cdf_ptr, mask);
}

void sgs_lvm_simulation(
		py::tuple output_array,
		const py_grid_t & grid,
		const py_sgs_params_t & param,
		py::object mean_data,
		py::object cdf_property,
		py::object mask_data)
{
	sp_double_property_array_t out_prop = cont_prop_from_tuple(output_array);

	unsigned char * mask =
			mask_data.ptr() == Py_None ? nullptr
			: get_buffer_from_ndarray<unsigned char,'u'>(mask_data, out_prop->size(), "sgs_lvm_simulation");

	hpgl_non_parametric_cdf_t cdf_data;
	const hpgl_non_parametric_cdf_t * cdf_ptr = nullptr;
	try
	{
		py::tuple cdf_tuple = py::cast<py::tuple>(cdf_property);
		int cdf_size = get_ndarray_len(cdf_tuple[py::int_(0)]);
		cdf_data.m_values = get_buffer_from_ndarray<float, 'f'>(cdf_tuple[py::int_(0)], cdf_size, "sgs_lvm_simulation cdf values");
		cdf_data.m_probs = get_buffer_from_ndarray<float, 'f'>(cdf_tuple[py::int_(1)], cdf_size, "sgs_lvm_simulation cdf probs");
		cdf_data.m_size = cdf_size;
		cdf_ptr = &cdf_data;
	}
	catch (const py::cast_error &)
	{
		cdf_ptr = nullptr;
	}

	mean_t * lvm_data = get_buffer_from_ndarray<mean_t, 'f'>(mean_data, out_prop->size(), "sgs_lvm_simulation");
	hpgl::sequential_gaussian_simulation_lvm(*grid.m_sugarbox_geometry, param.m_sgs_params, lvm_data, *out_prop, cdf_ptr, mask);
}

// Pybind11 module definition
PYBIND11_MODULE(hpgl, m)
{
	using namespace hpgl;
	using namespace hpgl::python;

	// Thread management functions
	m.def("set_thread_num", &set_thread_num,
		"Set the number of threads for parallel operations");
	m.def("get_thread_num", &get_thread_num,
		"Get the current number of threads");

	// Grid class
	py::class_<py_grid_t>(m, "grid")
		.def(py::init<sugarbox_grid_size_t,
				sugarbox_grid_size_t,
				sugarbox_grid_size_t>())
		.def_readwrite("m_sugarbox_geometry", &py_grid_t::m_sugarbox_geometry);

	// Ordinary Kriging Parameters
	py::class_<py_ok_params_t>(m, "ok_params")
		.def(py::init<>())
		.def("set_covariance_type", &py_ok_params_t::set_covariance_type,
			"Set covariance type (0=Spherical, 1=Exponential, 2=Gaussian)")
		.def("set_ranges", &py_ok_params_t::set_ranges,
			"Set ranges for anisotropic variogram")
		.def("set_angles", &py_ok_params_t::set_angles,
			"Set angles for anisotropic variogram")
		.def("set_sill", &py_ok_params_t::set_sill,
			"Set variogram sill")
		.def("set_nugget", &py_ok_params_t::set_nugget,
			"Set variogram nugget effect")
		.def("set_radiuses", &py_ok_params_t::set_radiuses,
			"Set search radiuses")
		.def("set_max_neighbours", &py_ok_params_t::set_max_neighbours,
			"Set maximum number of neighbours");

	// Simple Kriging Parameters
	py::class_<py_sk_params_t>(m, "sk_params")
		.def(py::init<>())
		.def("set_covariance_type", &py_sk_params_t::set_covariance_type,
			"Set covariance type")
		.def("set_ranges", &py_sk_params_t::set_ranges,
			"Set ranges for anisotropic variogram")
		.def("set_angles", &py_sk_params_t::set_angles,
			"Set angles for anisotropic variogram")
		.def("set_sill", &py_sk_params_t::set_sill,
			"Set variogram sill")
		.def("set_nugget", &py_sk_params_t::set_nugget,
			"Set variogram nugget effect")
		.def("set_radiuses", &py_sk_params_t::set_radiuses,
			"Set search radiuses")
		.def("set_max_neighbours", &py_sk_params_t::set_max_neighbours,
			"Set maximum number of neighbours")
		.def("set_mean", &py_sk_params_t::set_mean,
			"Set mean value for simple kriging");

	// Median Indicator Kriging Parameters
	py::class_<py_median_ik_params_t>(m, "median_ik_params")
		.def(py::init<>())
		.def("set_covariance_type", &py_median_ik_params_t::set_covariance_type,
			"Set covariance type")
		.def("set_ranges", &py_median_ik_params_t::set_ranges,
			"Set ranges")
		.def("set_angles", &py_median_ik_params_t::set_angles,
			"Set angles")
		.def("set_sill", &py_median_ik_params_t::set_sill,
			"Set sill")
		.def("set_nugget", &py_median_ik_params_t::set_nugget,
			"Set nugget")
		.def("set_radiuses", &py_median_ik_params_t::set_radiuses,
			"Set search radiuses")
		.def("set_max_neighbours", &py_median_ik_params_t::set_max_neighbours,
			"Set maximum neighbours")
		.def("set_marginal_probs", &py_median_ik_params_t::set_marginal_probs,
			"Set marginal probabilities");

	// Sequential Gaussian Simulation Parameters
	py::class_<py_sgs_params_t>(m, "sgs_params")
		.def(py::init<>())
		.def("set_covariance_type", &py_sgs_params_t::set_covariance_type,
			"Set covariance type")
		.def("set_ranges", &py_sgs_params_t::set_ranges,
			"Set ranges")
		.def("set_angles", &py_sgs_params_t::set_angles,
			"Set angles")
		.def("set_sill", &py_sgs_params_t::set_sill,
			"Set sill")
		.def("set_nugget", &py_sgs_params_t::set_nugget,
			"Set nugget")
		.def("set_radiuses", &py_sgs_params_t::set_radiuses,
			"Set search radiuses")
		.def("set_max_neighbours", &py_sgs_params_t::set_max_neighbours,
			"Set maximum neighbours")
		.def("set_mean", &py_sgs_params_t::set_mean,
			"Set mean value")
		.def("set_kriging_kind", &py_sgs_params_t::set_kriging_kind,
			"Set kriging kind")
		.def("set_seed", &py_sgs_params_t::set_seed,
			"Set random seed")
		.def("set_mean_kind", &py_sgs_params_t::set_mean_kind,
			"Set mean kind")
		.def("set_min_neighbours", &py_sgs_params_t::set_min_neighbours,
			"Set minimum neighbours");


	// I/O functions
	m.def("read_inc_file_float", &py_read_inc_file_float,
		"Read INC file with float data");
	m.def("read_inc_file_byte", &py_read_inc_file_byte,
		"Read INC file with byte data");

	m.def("write_byte_property", &write_byte_property,
		"Write byte property to file");
	m.def("write_cont_property", &write_cont_property,
		"Write continuous property to file");

	// Factory functions
	m.def("create_sugarbox_grid", &create_sugarbox_grid,
		"Create a sugarbox grid");
	m.def("create_ok_params", &create_ok_params,
		"Create ordinary kriging parameters");
	m.def("create_sk_params", &create_sk_params,
		"Create simple kriging parameters");

	m.def("create_sgs_params", &create_sgs_params,
		"Create SGS parameters");
	m.def("create_median_ik_params", &py_create_median_ik_params,
		"Create median indicator kriging parameters");

	m.def("simple_kriging_weights", &py_calculate_kriging_weight,
		"Calculate simple kriging weights");

	// Kriging algorithms
	m.def("ordinary_kriging", &::ordinary_kriging,
		"Ordinary kriging estimation");
	m.def("simple_kriging", &::simple_kriging,
		"Simple kriging estimation");
	m.def("indicator_kriging", &py_indicator_kriging,
		"Indicator kriging estimation");
	m.def("lvm_kriging", &::lvm_kriging,
		"Local varying mean kriging");
	m.def("median_ik", &py_median_ik,
		"Median indicator kriging");

	// Simulation algorithms
	m.def("sgs_simulation", &sgs_simulation,
		"Sequential Gaussian Simulation");
	m.def("sgs_lvm_simulation", &sgs_lvm_simulation,
		"Sequential Gaussian Simulation with Local Varying Mean");
	m.def("sis_simulation", &py_sis,
		"Sequential Indicator Simulation");
	m.def("sis_simulation_lvm", &py_sis_lvm,
		"Sequential Indicator Simulation with Local Varying Mean");

	// Cokriging
	m.def("simple_cokriging_markI", &hpgl::python::py_simple_cokriging_markI,
		"Simple cokriging Mark I");
	m.def("simple_cokriging_markII", &hpgl::python::py_simple_cokriging_markII,
		"Simple cokriging Mark II");

	// Module metadata
	m.attr("__version__") = "1.0.0";
	m.attr("__author__") = "HPGL Team";
}
