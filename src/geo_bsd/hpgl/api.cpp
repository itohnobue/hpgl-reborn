#include "stdafx.h"
#include <iostream>
#include <memory>
#include <sstream>

#include "api.h"
#include "api_helpers.hpp"
#include "hpgl_core.h"
#include "sugarbox_grid.h"
#include "ok_params.h"
#include "sk_params.h"
#include "sgs_params.h"
#include "ik_params.h"
#include "median_ik.h"
#include "property_writer.h"
#include "progress_reporter.h"
#include "hpgl_exception.h"
#include "output.h"



/// Initializes the common covariance fields shared by ok_params_t, sk_params_t,
/// median_ik_params, etc. from a C API params struct pointer.
/// Eliminates copy-pasted 7-line init blocks across 4 C API wrappers.
template<typename CpT, typename ParamsT>
static void init_cov_params_base(ParamsT & p, const CpT * params)
{
	p.m_covariance_type = (hpgl::covariance_type_t) params->m_covariance_type;
	p.set_ranges(
			params->m_ranges[0],
			params->m_ranges[1],
			params->m_ranges[2]);
	p.set_angles(
			params->m_angles[0],
			params->m_angles[1],
			params->m_angles[2]);
	p.set_sill(params->m_sill);
	p.set_nugget(params->m_nugget);
	p.validate();
}

static void
handle_exception(const std::exception & ex)
{
	hpgl::set_last_exception_message(ex.what());
}

/// Validates that a C API struct pointer parameter is non-null.
/// Returns 0 on success, -1 on null (stores error message).
static int validate_pointer(const void * ptr, const char * param_name)
{
	if (ptr == nullptr)
	{
		std::ostringstream oss;
		oss << "Null pointer argument: " << param_name;
		hpgl::set_last_exception_message(oss.str().c_str());
		return -1;
	}
	return 0;
}

/// Validates a struct pointer parameter for void-returning C API functions.
/// Throws hpgl_exception if null (caught by existing try/catch handlers).
static void validate_pointer_or_throw(const void * ptr, const char * param_name)
{
	if (ptr == nullptr)
	{
		std::ostringstream oss;
		oss << "Null pointer argument: " << param_name;
		throw hpgl::hpgl_exception("C API", oss.str());
	}
}

/// Validates get_shape_volume result, returns negative size.
/// Returns original value if >= 0, -1 otherwise (stores error message).
static int validate_shape_volume(int volume, const char * context)
{
	if (volume < 0)
	{
		std::ostringstream oss;
		oss << context << ": grid volume overflow or invalid shape";
		hpgl::set_last_exception_message(oss.str().c_str());
		return -1;
	}
	return volume;
}

/// Validates get_shape_volume result for void-returning functions.
/// Throws hpgl_exception if volume is invalid.
static void validate_shape_volume_or_throw(int volume, const char * context)
{
	if (volume < 0)
	{
		std::ostringstream oss;
		oss << context << ": grid volume overflow or invalid shape";
		throw hpgl::hpgl_exception("C API", oss.str());
	}
}

extern "C" {

HPGL_API char * hpgl_get_last_exception_message()
{
	try
	{
		thread_local std::string cached_message;
		cached_message = hpgl::get_last_exception_message();
		// Returns pointer to thread_local buffer. Valid until next call on same thread.
		// Python ctypes (c_char_p) copies the data immediately, so no lifetime issue.
		return const_cast<char *>(cached_message.c_str());
	}
	catch (const std::exception & ex)
	{
		handle_exception(ex);
		return const_cast<char *>("");
	}
}

HPGL_API int hpgl_set_thread_num(int n_threads)
{
	try
	{
		if (!hpgl::set_thread_num(n_threads))
		{
			std::ostringstream oss;
			oss << "hpgl_set_thread_num: invalid thread count " << n_threads;
			hpgl::set_last_exception_message(oss.str().c_str());
			return -1;
		}
		return 0;
	}
	catch (const std::exception & ex)
	{
		handle_exception(ex);
		return -1;
	}
}

HPGL_API int hpgl_get_thread_num()
{
	return hpgl::get_thread_num();
}

HPGL_API int hpgl_read_inc_file_float(
		char * filename,
		float undefined_value,
		int size,
		float * data,
		unsigned char * mask)
{
	if (validate_pointer(filename, "filename (read_inc_file_float)") != 0) return -1;
	if (validate_pointer(data, "data (read_inc_file_float)") != 0) return -1;
	if (size <= 0)
	{
		hpgl::set_last_exception_message("read_inc_file_float: size must be positive");
		return -1;
	}
	// mask is optional: may be nullptr (all cells active)
	try
	{
		hpgl::read_inc_file_float(
				filename,
				undefined_value,
				size,
				data,
				mask);
		return 0;
	}
	catch (const std::exception & ex)
	{
		handle_exception(ex);
		return -1;
	}
}

HPGL_API int hpgl_read_inc_file_byte(
		char * filename,
		int undefined_value,
		int size,
		unsigned char * data,
		unsigned char * mask,
		unsigned char * values,
		int values_count)
{
	if (validate_pointer(filename, "filename (read_inc_file_byte)") != 0) return -1;
	if (validate_pointer(data, "data (read_inc_file_byte)") != 0) return -1;
	if (validate_pointer(mask, "mask (read_inc_file_byte)") != 0) return -1;
	if (validate_pointer(values, "values (read_inc_file_byte)") != 0) return -1;
	if (values_count <= 0)
	{
		hpgl::set_last_exception_message("read_inc_file_byte: values_count must be positive");
		return -1;
	}
	if (size <= 0)
	{
		hpgl::set_last_exception_message("read_inc_file_byte: size must be positive");
		return -1;
	}
	try
	{
		hpgl::read_inc_file_byte(
				filename,
				undefined_value,
				size,
				data,
				mask);

		for (int i = 0; i < size; ++i)
		{
			if (mask[i] != 0)
			{
				unsigned char original_value = data[i];
				bool mapped = false;
				for (int j = 0; j < values_count; ++j)
				{
					if (original_value == values[j])
					{
						data[i] = j;
						mapped = true;
						break;
					}
				}
				if (!mapped)
				{
					std::ostringstream oss;
					oss << "Unmapped byte value " << static_cast<int>(original_value)
					    << " at index " << i;
					throw hpgl::hpgl_exception("hpgl_read_inc_file_byte", oss.str());
				}
			}
		}
		return 0;
	}
	catch (const std::exception & ex)
	{
		handle_exception(ex);
		return -1;
	}
}

HPGL_API int hpgl_write_inc_file_float(
		char * filename,
		hpgl_cont_masked_array_t * arr,
		float undefined_value,
		char * name)
{
	if (validate_pointer(filename, "filename (write_inc_file_float)") != 0) return -1;
	if (validate_pointer(arr, "arr (write_inc_file_float)") != 0) return -1;
	if (validate_pointer(name, "name (write_inc_file_float)") != 0) return -1;
	try
	{
		using namespace hpgl;
		int vol = validate_shape_volume(get_shape_volume(&(arr->m_shape)), "write_inc_file_float");
		if (vol < 0) return -1;
		property_writer_t writer;
		writer.init(filename, name);
		cont_property_array_t prop(
				arr->m_data,
				arr->m_mask,
				vol);
		writer.write_double(prop, undefined_value);
		return 0;
	}
	catch (const std::exception & ex)
	{
		handle_exception(ex);
		return -1;
	}
}

void init_remap_table(unsigned char * values, int values_count, int indicator_count, std::vector<unsigned char> & remap_table)
{
	// Validate against negative counts which cause UB in pointer arithmetic
	// (values + negative_count is undefined behavior).
	if (values_count < 0)
		throw hpgl::hpgl_exception("init_remap_table", "Negative values_count");
	if (indicator_count < 0)
		throw hpgl::hpgl_exception("init_remap_table", "Negative indicator_count");
	if (indicator_count > 255)
		throw hpgl::hpgl_exception("init_remap_table",
			"indicator_count exceeds unsigned char max (255)");

	if (values == nullptr)
	{
		// No remap table provided: use identity mapping [0, 1, 2, ...]
		for (int i = 0; i < indicator_count; ++i)
			remap_table.push_back(i);
		return;
	}
	if (values_count == indicator_count)
	{
		remap_table.assign(values, values+values_count);
	}
	else if (values_count > indicator_count)
	{
		std::ostringstream oss;
		oss << "Given " << values_count << " values for " << indicator_count << " indicators. Ignoring extra values.\n";
		LOGWARNING(oss.str());
		remap_table.assign(values, values + indicator_count);
	}
	else
	{
		// Throw when caller provides remap values but fewer than
		// indicator_count — discarding caller data silently would
		// produce wrong output. The two callers (write_inc_file_byte,
		// write_gslib_byte_property) catch std::exception and return
		// error codes to the caller.
		std::ostringstream oss;
		oss << "Mismatch: " << values_count << " remap values provided for "
		    << indicator_count << " indicators. Expected " << indicator_count
		    << " values (one per indicator).";
		throw hpgl::hpgl_exception("init_remap_table", oss.str());
	}
}

HPGL_API int hpgl_write_inc_file_byte(
		char * filename,
		hpgl_ind_masked_array_t * arr,
		int undefined_value,
		char * name,
		unsigned char * values,
		int values_count)
{
	if (validate_pointer(filename, "filename (write_inc_file_byte)") != 0) return -1;
	if (validate_pointer(arr, "arr (write_inc_file_byte)") != 0) return -1;
	if (validate_pointer(name, "name (write_inc_file_byte)") != 0) return -1;
	if (values_count < 0)
	{
		hpgl::set_last_exception_message("write_inc_file_byte: negative values_count");
		return -1;
	}
	try
	{
		using namespace hpgl;
		int vol = validate_shape_volume(get_shape_volume(&(arr->m_shape)), "write_inc_file_byte");
		if (vol < 0) return -1;
		std::vector<unsigned char> remap_table;
		init_remap_table(values, values_count, arr->m_indicator_count, remap_table);

		// Validate undefined_value fits in unsigned char before write.
		// Out-of-range int (e.g. negative sentinels like -999) silently
		// wraps via modulo-256 arithmetic, producing corrupted output
		// with no warning. Zero range check was present before this fix.
		if (!(undefined_value >= 0 && undefined_value <= 255))
		{
			std::ostringstream oss;
			oss << "write_inc_file_byte: undefined_value " << undefined_value
			    << " out of range for unsigned char [0, 255]";
			throw hpgl::hpgl_exception("hpgl_write_inc_file_byte", oss.str());
		}

		property_writer_t writer;
		writer.init(filename, name);
		indicator_property_array_t prop(
				arr->m_data,
				arr->m_mask,
				vol,
				arr->m_indicator_count);
		writer.write_byte(prop, undefined_value, remap_table);
		return 0;
	}
	catch (const std::exception & ex)
	{
		handle_exception(ex);
		return -1;
	}
}

HPGL_API int
hpgl_write_gslib_cont_property(
		hpgl_cont_masked_array_t * data,
		const char * filename,
		const char * name,
		double undefined_value)
{
	if (validate_pointer(data, "data (write_gslib_cont_property)") != 0) return -1;
	if (validate_pointer(filename, "filename (write_gslib_cont_property)") != 0) return -1;
	if (validate_pointer(name, "name (write_gslib_cont_property)") != 0) return -1;
	try
	{
		using namespace hpgl;
		int size = validate_shape_volume(get_shape_volume(&data->m_shape), "write_gslib_cont_property");
		if (size < 0) return -1;
		sp_double_property_array_t prop = std::make_shared<cont_property_array_t>(data->m_data, data->m_mask, size);
		hpgl::property_writer_t writer;
		writer.init(filename, name);
		writer.write_gslib_double(prop, undefined_value);
		return 0;
	}
	catch (const std::exception & ex)
	{
		handle_exception(ex);
		return -1;
	}
}

HPGL_API int
hpgl_write_gslib_byte_property(
		hpgl_ind_masked_array_t * data,
		const char * filename,
		const char * name,
		double undefined_value,
		unsigned char * values,
		int values_count)
{
	if (validate_pointer(data, "data (write_gslib_byte_property)") != 0) return -1;
	if (validate_pointer(filename, "filename (write_gslib_byte_property)") != 0) return -1;
	if (validate_pointer(name, "name (write_gslib_byte_property)") != 0) return -1;
	if (validate_pointer(values, "values (write_gslib_byte_property)") != 0) return -1;
	if (values_count < 0)
	{
		hpgl::set_last_exception_message("write_gslib_byte_property: negative values_count");
		return -1;
	}
	try
	{
		using namespace hpgl;
		int size = validate_shape_volume(get_shape_volume(&data->m_shape), "write_gslib_byte_property");
		if (size < 0) return -1;
		sp_byte_property_array_t prop = std::make_shared<indicator_property_array_t>(data->m_data, data->m_mask, size, data->m_indicator_count);
		std::vector<unsigned char> remap_table;
		init_remap_table(values, values_count, data->m_indicator_count, remap_table);

		// Validate undefined_value fits in unsigned char before cast.
		// Cast of out-of-range double (e.g. negative sentinels like -999.0)
		// to unsigned char is undefined behavior.
		// Uses IEEE 754 property: NaN comparisons always return false,
		// so !(NaN >= 0.0 && NaN <= 255.0) is true, catching NaN as well.
		if (!(undefined_value >= 0.0 && undefined_value <= 255.0))
		{
			std::ostringstream oss;
			oss << "write_gslib_byte_property: undefined_value " << undefined_value
			    << " out of range for unsigned char [0, 255]";
			throw hpgl::hpgl_exception("hpgl_write_gslib_byte_property", oss.str());
		}

		property_writer_t writer;
		writer.init(filename, name);
		writer.write_gslib_byte(prop, static_cast<unsigned char>(undefined_value), remap_table);
		return 0;
	}
	catch (const std::exception & ex)
	{
		handle_exception(ex);
		return -1;
	}
}

HPGL_API void hpgl_ordinary_kriging(
    hpgl_cont_masked_array_t * input_data,
    hpgl_ok_params_t * params,
    hpgl_cont_masked_array_t * output_data)
{
	try
	{
	using namespace hpgl;
	validate_pointer_or_throw(input_data, "input_data (ordinary_kriging)");
	validate_pointer_or_throw(params, "params (ordinary_kriging)");
	validate_pointer_or_throw(output_data, "output_data (ordinary_kriging)");

	int in_size = get_shape_volume(&input_data->m_shape);
	validate_shape_volume_or_throw(in_size, "ordinary_kriging input");
	int out_size = get_shape_volume(&output_data->m_shape);
	validate_shape_volume_or_throw(out_size, "ordinary_kriging output");

	if (in_size != out_size)
		throw hpgl_exception("hpgl_ordinary_kriging", "input and output shape volume mismatch");

	cont_property_array_t in_prop(input_data->m_data, input_data->m_mask, in_size);
	cont_property_array_t out_prop(output_data->m_data, output_data->m_mask, out_size);

	sugarbox_grid_t grid;
	init_grid(grid, &input_data->m_shape);

	ok_params_t ok_p;
	init_cov_params_base(ok_p, params);

	ok_p.set_radiuses(
			params->m_radiuses[0],
			params->m_radiuses[1],
			params->m_radiuses[2]);

	ok_p.m_max_neighbours = params->m_max_neighbours;

	hpgl::ordinary_kriging(in_prop, grid, ok_p, out_prop, true);
	}
	catch (const std::exception & ex) { handle_exception(ex); }
}

static void init_sk_params(hpgl_sk_params_t * params, hpgl::sk_params_t & sk_p)
{
	using namespace hpgl;
	init_cov_params_base(sk_p, params);
	
	sk_p.set_radiuses(
			params->m_radiuses[0],
			params->m_radiuses[1],
			params->m_radiuses[2]);

	sk_p.m_max_neighbours = params->m_max_neighbours;

	if (!params->m_automatic_mean)
	{
		sk_p.set_mean(params->m_mean);
	}
}

HPGL_API void hpgl_simple_kriging(
    float * input_data,
    unsigned char * input_mask,
    hpgl_shape_t * input_data_shape,
    hpgl_sk_params_t * params,
    float * output_data,
    unsigned char * output_mask,
    hpgl_shape_t * output_data_shape)
{
	try
	{
	using namespace hpgl;
	validate_pointer_or_throw(input_data, "input_data (simple_kriging)");
	validate_pointer_or_throw(output_data, "output_data (simple_kriging)");
	validate_pointer_or_throw(input_data_shape, "input_data_shape (simple_kriging)");
	validate_pointer_or_throw(params, "params (simple_kriging)");
	validate_pointer_or_throw(output_data_shape, "output_data_shape (simple_kriging)");

	int in_size = get_shape_volume(input_data_shape);
	validate_shape_volume_or_throw(in_size, "simple_kriging input");
	int out_size = get_shape_volume(output_data_shape);
	validate_shape_volume_or_throw(out_size, "simple_kriging output");

	if (in_size != out_size)
		throw hpgl_exception("hpgl_simple_kriging", "input and output shape volume mismatch");

	cont_property_array_t in_prop(input_data, input_mask, in_size);
	cont_property_array_t out_prop(output_data, output_mask, out_size);

	sugarbox_grid_t grid;
	grid.init(
			input_data_shape->m_data[0],
			input_data_shape->m_data[1],
			input_data_shape->m_data[2]
			  );

	sk_params_t sk_p;
	init_sk_params(params, sk_p);

	hpgl::simple_kriging(in_prop, grid, sk_p, out_prop);
	}
	catch (const std::exception & ex) { handle_exception(ex); }
}

HPGL_API int
hpgl_simple_kriging_weights(
		float * center_coords,
		float * neighbours_x,
		float * neighbours_y,
		float * neighbours_z,
		int neighbours_count,
		hpgl_cov_params_t * params,
		float * weights)
{
	if (validate_pointer(params, "params (simple_kriging_weights)") != 0) return -1;
	if (validate_pointer(weights, "weights (simple_kriging_weights)") != 0) return -1;
	if (validate_pointer(center_coords, "center_coords (simple_kriging_weights)") != 0) return -1;
	if (validate_pointer(neighbours_x, "neighbours_x (simple_kriging_weights)") != 0) return -1;
	if (validate_pointer(neighbours_y, "neighbours_y (simple_kriging_weights)") != 0) return -1;
	if (validate_pointer(neighbours_z, "neighbours_z (simple_kriging_weights)") != 0) return -1;
	if (neighbours_count < 0)
	{
		hpgl::set_last_exception_message("simple_kriging_weights: negative neighbours_count");
		return -1;
	}
	try
	{
	using namespace hpgl;
	real_location_t center(center_coords[0], center_coords[1], center_coords[2]);

	std::vector<real_location_t> neighbour_coords(neighbours_count);
	for (int i = 0; i < neighbours_count; ++i)
	{
		neighbour_coords[i][0] = neighbours_x[i];
		neighbour_coords[i][1] = neighbours_y[i];
		neighbour_coords[i][2] = neighbours_z[i];
	}

	std::vector<kriging_weight_t> weights2;
	double variance;

	sk_params_t sk_p;
	init_cov_params_base(sk_p, params);

	simple_kriging_weights(
			&sk_p,
			center,
			neighbour_coords,
			weights2,
			variance);

	for (int i = 0; i < neighbours_count; ++i)
	{
		weights[i] = weights2.at(i);
	}
	return 0;
	}
	catch (const std::exception & ex) { handle_exception(ex); return -1; }
}

HPGL_API void hpgl_lvm_kriging(
    float * input_data,
    unsigned char * input_mask,
    hpgl_shape_t * input_data_shape,
    float * mean_data,
    hpgl_shape_t * mean_data_shape,
    hpgl_ok_params_t * params,
    float * output_data,
    unsigned char * output_mask,
    hpgl_shape_t * output_data_shape)
{
	try
	{
	using namespace hpgl;
	validate_pointer_or_throw(input_data, "input_data (lvm_kriging)");
	validate_pointer_or_throw(mean_data, "mean_data (lvm_kriging)");
	validate_pointer_or_throw(output_data, "output_data (lvm_kriging)");
	validate_pointer_or_throw(input_data_shape, "input_data_shape (lvm_kriging)");
	validate_pointer_or_throw(mean_data_shape, "mean_data_shape (lvm_kriging)");
	validate_pointer_or_throw(params, "params (lvm_kriging)");
	validate_pointer_or_throw(output_data_shape, "output_data_shape (lvm_kriging)");

	int size = get_shape_volume(input_data_shape);
	validate_shape_volume_or_throw(size, "lvm_kriging input");
	int out_size = get_shape_volume(output_data_shape);
	validate_shape_volume_or_throw(out_size, "lvm_kriging output");

	if (size != out_size)
		throw hpgl_exception("hpgl_lvm_kriging", "input and output shape volume mismatch");
	
	// Validate means array size matches grid volume to prevent OOB read
	// in subtract_means / add_means (lvm_utils.h)
	int mean_size = get_shape_volume(mean_data_shape);
	validate_shape_volume_or_throw(mean_size, "lvm_kriging means");
	if (mean_size != size)
	{
		std::ostringstream oss;
		oss << "lvm_kriging: means volume (" << mean_size
		    << ") != input volume (" << size << ")";
		throw hpgl_exception("hpgl_lvm_kriging", oss.str());
	}

	cont_property_array_t input_prop(input_data, input_mask, size);
	sugarbox_grid_t grid;
	init_grid(grid, input_data_shape);

	ok_params_t ok_p;
	init_cov_params_base(ok_p, params);

	ok_p.set_radiuses(
			params->m_radiuses[0],
			params->m_radiuses[1],
			params->m_radiuses[2]);

	ok_p.m_max_neighbours = params->m_max_neighbours;

	cont_property_array_t out_prop(output_data, output_mask, out_size);
	lvm_kriging(input_prop, mean_data, grid, ok_p, out_prop);
	}
	catch (const std::exception & ex) { handle_exception(ex); }
}

HPGL_API void
hpgl_indicator_kriging(
		hpgl_ind_masked_array_t * in_data,
		hpgl_ind_masked_array_t * out_data,
		hpgl_ik_params_t * params,
		int indicator_count)
{
	try
	{
	using namespace hpgl;
	validate_pointer_or_throw(in_data, "in_data (indicator_kriging)");
	validate_pointer_or_throw(out_data, "out_data (indicator_kriging)");
	validate_pointer_or_throw(params, "params (indicator_kriging)");

	// Validate indicator_count matches the data's actual indicator_count.
	// Direct C callers may pass mismatched values, leading to out-of-bounds reads
	// in init_sis_params or indicator_kriging.
	if (indicator_count != in_data->m_indicator_count)
		throw hpgl_exception("hpgl_indicator_kriging",
			"indicator_count mismatch with in_data->m_indicator_count");

	// Defense-in-depth: indicator_index_t is unsigned char (max 255).
	// indicator_count > 255 causes wrap-around in cdf_utils.cpp most_probable_category loop.
	if (indicator_count > 255)
		throw hpgl_exception("hpgl_indicator_kriging",
			"indicator_count exceeds unsigned char max (255)");

	int size = get_shape_volume(&in_data->m_shape);
	validate_shape_volume_or_throw(size, "indicator_kriging input");
	int size2 = get_shape_volume(&out_data->m_shape);
		validate_shape_volume_or_throw(size2, "indicator_kriging output");
		if (size != size2)
			throw hpgl_exception("hpgl_indicator_kriging", "input and output size mismatch");
	// Validate out_data indicator_count matches the validated indicator_count
	// parameter to prevent downstream OOB reads from consumers that trust
	// out_data->m_indicator_count (e.g. write_gslib_byte_property at api.cpp:408).
	if (out_data->m_indicator_count != indicator_count)
		throw hpgl_exception("hpgl_indicator_kriging",
			"out_data indicator_count mismatch with validated indicator_count");
	indicator_property_array_t in_prop(in_data->m_data, in_data->m_mask, size, in_data->m_indicator_count);
	indicator_property_array_t out_prop(out_data->m_data, out_data->m_mask, size2, out_data->m_indicator_count);

	ik_params_t ikp;
	init_sis_params(params, indicator_count, &ikp);

	sugarbox_grid_t grid;
	init_grid(grid, &(in_data->m_shape));

	indicator_kriging(in_prop, grid, ikp, out_prop);
	}
	catch (const std::exception & ex) { handle_exception(ex); }
}

HPGL_API void
hpgl_sgs_simulation(
		hpgl_cont_masked_array_t * data,
		hpgl_sgs_params_t * params,
		hpgl_non_parametric_cdf_t * cdf,
		double * mean,
		hpgl_ubyte_array_t * simulation_mask)
{
	try
	{
	using namespace hpgl;
	validate_pointer_or_throw(data, "data (sgs_simulation)");
	validate_pointer_or_throw(params, "params (sgs_simulation)");

	int size = get_shape_volume(&(data->m_shape));
	validate_shape_volume_or_throw(size, "sgs_simulation");
	cont_property_array_t prop(data->m_data, data->m_mask, size);

	sugarbox_grid_t grid;
	init_grid(grid, &(data->m_shape));

	sgs_params_t sgs_p;
	init_sgs_params(params, &sgs_p);

	if (mean != 0)
		sgs_p.set_mean(*mean);
	sgs_p.m_lvm = 0;
	sgs_p.m_mean_kind = mean != 0 ? mean_kind_t::e_mean_stationary : mean_kind_t::e_mean_stationary_auto;
	hpgl::sequential_gaussian_simulation(
			grid,
		    sgs_p,
			prop,
			cdf,
			(simulation_mask != 0 && simulation_mask->m_data != 0) ? simulation_mask->m_data : 0);
	}
	catch (const std::exception & ex) { handle_exception(ex); }
}


HPGL_API void hpgl_sgs_lvm_simulation(
		hpgl_cont_masked_array_t * data,
		hpgl_sgs_params_t * params,
		hpgl_non_parametric_cdf_t * cdf,
		hpgl_float_array_t * means,
		hpgl_ubyte_array_t * simulation_mask)
{
	try
	{
	using namespace hpgl;
	validate_pointer_or_throw(data, "data (sgs_lvm_simulation)");
	validate_pointer_or_throw(params, "params (sgs_lvm_simulation)");
	validate_pointer_or_throw(means, "means (sgs_lvm_simulation)");

	int size = get_shape_volume(&(data->m_shape));
	validate_shape_volume_or_throw(size, "sgs_lvm_simulation");
	cont_property_array_t prop(data->m_data, data->m_mask, size);

	sugarbox_grid_t grid;
	init_grid(grid, &(data->m_shape));

	sgs_params_t sgs_p;
	init_sgs_params(params, &sgs_p);

	// Defensive: ensure means data pointer is valid before use
	if (means->m_data == nullptr)
	{
		throw hpgl_exception("hpgl_sgs_lvm_simulation", "Null means data pointer");
	}
	
	// Validate means shape volume matches grid volume to prevent OOB read
	// in sequential_gaussian_simulation_lvm (mean_data_vec.assign at line 109)
	int mean_vol = get_shape_volume(&(means->m_shape));
	validate_shape_volume_or_throw(mean_vol, "sgs_lvm_simulation means");
	if (mean_vol != size)
	{
		std::ostringstream oss;
		oss << "sgs_lvm_simulation: means volume (" << mean_vol
		    << ") != grid volume (" << size << ")";
		throw hpgl_exception("hpgl_sgs_lvm_simulation", oss.str());
	}

	sgs_p.m_lvm = means->m_data;
	sgs_p.m_mean_kind = mean_kind_t::e_mean_varying;

	hpgl::sequential_gaussian_simulation_lvm(
			grid,
			sgs_p,
			means->m_data,
			prop,
			cdf,
			(simulation_mask != 0 && simulation_mask->m_data != 0) ? simulation_mask->m_data : 0);
	}
	catch (const std::exception & ex) { handle_exception(ex); }
}

HPGL_API void hpgl_median_ik(
		hpgl_ind_masked_array_t * in_data,
		hpgl_median_ik_params_t * params,
		hpgl_ind_masked_array_t * out_data)
{
	try
	{
	using namespace hpgl;
	validate_pointer_or_throw(in_data, "in_data (median_ik)");
	validate_pointer_or_throw(params, "params (median_ik)");
	validate_pointer_or_throw(out_data, "out_data (median_ik)");

	int size = get_shape_volume(&(in_data->m_shape));
	validate_shape_volume_or_throw(size, "median_ik input");

	int out_size = get_shape_volume(&(out_data->m_shape));
	validate_shape_volume_or_throw(out_size, "median_ik output");

	if (size != out_size)
		throw hpgl_exception("hpgl_median_ik",
			"input and output shape volume mismatch");

	sugarbox_grid_t grid;
	init_grid(grid, &(in_data->m_shape));

	median_ik_params mik_p;
	init_cov_params_base(mik_p, params);

	mik_p.set_radiuses(
			params->m_radiuses[0],
			params->m_radiuses[1],
			params->m_radiuses[2]);

	mik_p.m_max_neighbours = params->m_max_neighbours;
	mik_p.m_marginal_probs[0] = params->m_marginal_probs[0];
	mik_p.m_marginal_probs[1] = params->m_marginal_probs[1];

	// median_ik requires exactly 2 indicators — the data layout is
	// interleaved and median_ik_for_two_indicators reads every other byte.
	// A mismatched indicator_count would read wrong cells.
	if (in_data->m_indicator_count != 2)
		throw hpgl_exception("hpgl_median_ik",
			"indicator_count must be 2 (median IK is defined for binary indicators)");

	indicator_property_array_t in_prop(
			in_data->m_data,
			in_data->m_mask,
			size,
			2);
	indicator_property_array_t out_prop(
			out_data->m_data,
			out_data->m_mask,
			size,
			2);
	median_ik_for_two_indicators(mik_p, grid, in_prop, out_prop);
	}
	catch (const std::exception & ex) { handle_exception(ex); }
}

HPGL_API void
hpgl_sis_simulation(
		hpgl_ind_masked_array_t * data,
		hpgl_ik_params_t * params,
		int indicator_count,
		int64_t seed,
		hpgl_ubyte_array_t * simulation_mask)
{
	try
	{
	using namespace hpgl;
	validate_pointer_or_throw(data, "data (sis_simulation)");
	validate_pointer_or_throw(params, "params (sis_simulation)");

	// Validate indicator_count matches the data's actual indicator_count.
	// Direct C callers may pass mismatched values, leading to out-of-bounds reads.
	if (indicator_count != data->m_indicator_count)
		throw hpgl_exception("hpgl_sis_simulation",
			"indicator_count mismatch with data->m_indicator_count");

	// Defense-in-depth: indicator_index_t is unsigned char (max 255).
	// indicator_count > 255 causes infinite loop in do_sis().
	if (indicator_count > 255)
		throw hpgl_exception("hpgl_sis_simulation",
			"indicator_count exceeds unsigned char max (255)");

	// Guard against m_category_count == 0 which causes do_sis() to
	// silently corrupt data (sample() returns SIZE_MAX, wraps to 255).
	if (indicator_count <= 0)
		throw hpgl_exception("hpgl_sis_simulation",
			"indicator_count must be positive");

	int size = get_shape_volume(&data->m_shape);
	validate_shape_volume_or_throw(size, "sis_simulation");
	indicator_property_array_t prop(data->m_data, data->m_mask, size, indicator_count);

	sugarbox_grid_t grid;
	init_grid(grid, &data->m_shape);

	ik_params_t ikp;
	init_sis_params(params, indicator_count, &ikp);

	progress_reporter_t rep(size);

	sequential_indicator_simulation(
			prop,
			grid,
			ikp,
			seed,
			rep,
			(simulation_mask != 0 && simulation_mask->m_data != 0) ? simulation_mask->m_data : 0);
	}
	catch (const std::exception & ex) { handle_exception(ex); }
}

HPGL_API void
hpgl_sis_simulation_lvm(
		hpgl_ind_masked_array_t * data,
		hpgl_ik_params_t * params,
		hpgl_float_array_t * mean_data,
		int indicator_count,
		int64_t seed,
		hpgl_ubyte_array_t * simulation_mask,
		int use_correlograms)
{
	try
	{
	using namespace hpgl;
	validate_pointer_or_throw(data, "data (sis_simulation_lvm)");
	validate_pointer_or_throw(params, "params (sis_simulation_lvm)");
	validate_pointer_or_throw(mean_data, "mean_data (sis_simulation_lvm)");

	// Validate indicator_count matches the data's actual indicator_count.
	// Direct C callers may pass mismatched values, leading to out-of-bounds reads.
	if (indicator_count != data->m_indicator_count)
		throw hpgl_exception("hpgl_sis_simulation_lvm",
			"indicator_count mismatch with data->m_indicator_count");

	// Defense-in-depth: indicator_index_t is unsigned char (max 255).
	// indicator_count > 255 causes infinite loop in do_sis().
	if (indicator_count > 255)
		throw hpgl_exception("hpgl_sis_simulation_lvm",
			"indicator_count exceeds unsigned char max (255)");

	int size = get_shape_volume(&data->m_shape);
	validate_shape_volume_or_throw(size, "sis_simulation_lvm");
	indicator_property_array_t prop(data->m_data, data->m_mask, size, indicator_count);

	// NOTE: indicator_count is validated below against the reasonable range
	// [0, 1000000] in all builds to prevent out-of-bounds access from direct
	// C callers. The Python caller (hpgl_wrap.py) also enforces this contract.
	sugarbox_grid_t grid;
	init_grid(grid, &data->m_shape);

	ik_params_t ikp;
	init_sis_params(params, indicator_count, &ikp);

	progress_reporter_t rep(size);

	// Build means array from mean_data structs.
	// NOTE: indicator_count is validated against the reasonable range [0, 1000000]
	// below in all builds to prevent out-of-bounds access from direct C callers.
	std::vector<const mean_t *> means;

	if (indicator_count <= 0 || indicator_count > 1000000)
		throw hpgl_exception("hpgl_sis_simulation_lvm",
			"indicator_count out of reasonable range [1, 1000000]");
	for (int i = 0; i < indicator_count; ++i)
	{
		if (mean_data[i].m_data == nullptr)
		{
			std::ostringstream oss;
			oss << "Null mean_data[" << i << "].m_data pointer";
			throw hpgl_exception("hpgl_sis_simulation_lvm", oss.str());
		}
		// Validate each indicator's mean array shape matches grid volume
		// to prevent out-of-bounds reads in the LVM simulation kernel.
		// (cf. hpgl_sgs_lvm_simulation which validates means shape identically.)
		int mean_vol = get_shape_volume(&(mean_data[i].m_shape));
		validate_shape_volume_or_throw(mean_vol, "sis_simulation_lvm means");
		if (mean_vol != size)
		{
			std::ostringstream oss;
			oss << "sis_simulation_lvm: means[" << i << "] volume (" << mean_vol
			    << ") != grid volume (" << size << ")";
			throw hpgl_exception("hpgl_sis_simulation_lvm", oss.str());
		}
		means.push_back(mean_data[i].m_data);
	}

	sequential_indicator_simulation_lvm(
			prop,
			grid,
			ikp,
			seed,
			&means[0],
			rep,
			use_correlograms != 0,
			(simulation_mask != 0 && simulation_mask->m_data != 0) ? simulation_mask->m_data : 0);
	}
	catch (const std::exception & ex) { handle_exception(ex); }
}

HPGL_API void
hpgl_simple_cokriging_mark1(
		hpgl_cont_masked_array_t * input_data,
		hpgl_cont_masked_array_t * secondary_data,
		hpgl_cokriging_m1_params_t * params,
		hpgl_cont_masked_array_t * output_data)
{
	try
	{
		using namespace hpgl;
		validate_pointer_or_throw(input_data, "input_data (cokriging_m1)");
		validate_pointer_or_throw(secondary_data, "secondary_data (cokriging_m1)");
		validate_pointer_or_throw(params, "params (cokriging_m1)");
		validate_pointer_or_throw(output_data, "output_data (cokriging_m1)");

		int size = get_shape_volume(&input_data->m_shape);
		validate_shape_volume_or_throw(size, "cokriging_m1 primary");
		int size2 = get_shape_volume(&secondary_data->m_shape);
		validate_shape_volume_or_throw(size2, "cokriging_m1 secondary");
		int size3 = get_shape_volume(&output_data->m_shape);
		validate_shape_volume_or_throw(size3, "cokriging_m1 output");

		if (size != size2)
		{
			std::ostringstream oss;
			oss << "Size of secondary data (" << size2 << ") is different from size of primary data (" << size << ")";
			throw hpgl_exception("hpgl_simple_cokriging_mark1", oss.str());
		}

		if (size != size3)
		{
			std::ostringstream oss;
			oss << "Size of output data (" << size3 << ") is different from size of primary data (" << size << ")";
			throw hpgl_exception("hpgl_simple_cokriging_mark1", oss.str());
		}

		cont_property_array_t primary_prop(
				input_data->m_data, input_data->m_mask, size);
		cont_property_array_t secondary_prop(
				secondary_data->m_data, secondary_data->m_mask, size2);
		cont_property_array_t output_prop(
				output_data->m_data, output_data->m_mask, size3);

		neighbourhood_param_t np;
		np.m_max_neighbours = params->m_max_neighbours;

		covariance_param_t cp;
		cp.m_covariance_type = (covariance_type_t) params->m_covariance_type;
		cp.set_sill(params->m_sill);
		cp.set_nugget(params->m_nugget);
		cp.validate();

		cp.set_ranges(
				params->m_ranges[0],
				params->m_ranges[1],
				params->m_ranges[2]);
		cp.set_angles(
				params->m_angles[0],
				params->m_angles[1],
				params->m_angles[2]);
		for (int i = 0; i < 3; ++i)
		{
			np.m_radiuses[i] = params->m_radiuses[i];
		}

		sugarbox_grid_t grid;
		init_grid(grid, &input_data->m_shape);

		simple_cokriging_markI(
				grid,
				primary_prop,
				secondary_prop,
				params->m_primary_mean,
				params->m_secondary_mean,
				params->m_secondary_variance,
				params->m_correlation_coef,
				np,
				cp,
				output_prop);
	}
	catch (const std::exception& ex)
	{
		handle_exception(ex);
	}
}

HPGL_API void
hpgl_simple_cokriging_mark2(
		hpgl_cont_masked_array_t * primary_data,
		hpgl_cont_masked_array_t * secondary_data,
		hpgl_cokriging_m2_params_t * params,
		hpgl_cont_masked_array_t * output_data)
{
	try
	{
		using namespace hpgl;
		validate_pointer_or_throw(primary_data, "primary_data (cokriging_m2)");
		validate_pointer_or_throw(secondary_data, "secondary_data (cokriging_m2)");
		validate_pointer_or_throw(params, "params (cokriging_m2)");
		validate_pointer_or_throw(output_data, "output_data (cokriging_m2)");

		int size = get_shape_volume(&primary_data->m_shape);
		validate_shape_volume_or_throw(size, "cokriging_m2 primary");
		int size2 = get_shape_volume(&secondary_data->m_shape);
		validate_shape_volume_or_throw(size2, "cokriging_m2 secondary");
		int size3 = get_shape_volume(&output_data->m_shape);
		validate_shape_volume_or_throw(size3, "cokriging_m2 output");

		if (size != size2)
		{
			std::ostringstream oss;
			oss << "Size of secondary data (" << size2 << ") is different from size of primary data (" << size << ")";
			throw hpgl_exception("hpgl_simple_cokriging_mark2", oss.str());
		}

		if (size != size3)
		{
			std::ostringstream oss;
			oss << "Size of output data (" << size3 << ") is different from size of primary data (" << size << ")";
			throw hpgl_exception("hpgl_simple_cokriging_mark2", oss.str());
		}

		cont_property_array_t primary_prop(
				primary_data->m_data, primary_data->m_mask, size);
		cont_property_array_t secondary_prop(
				secondary_data->m_data, secondary_data->m_mask, size2);
		cont_property_array_t out_prop(
				output_data->m_data, output_data->m_mask, size3);

		sugarbox_grid_t grid;
		init_grid(grid, &primary_data->m_shape);

		covariance_param_t primary_cp, secondary_cp;

		init_cov_params_base(primary_cp, &params->m_primary_cov_params);
		init_cov_params_base(secondary_cp, &params->m_secondary_cov_params);

		neighbourhood_param_t np;
		np.m_max_neighbours = params->m_max_neighbours;
		for (int i = 0; i < 3; ++i)
		{
			np.m_radiuses[i] = params->m_radiuses[i];
		}

		simple_cokriging_markII(
				grid, primary_prop,
				secondary_prop,
				params->m_primary_mean,
				params->m_secondary_mean,
				params->m_correlation_coef,
				np,
				primary_cp,
				secondary_cp,
				out_prop);
	}
	catch (const std::exception& ex)
	{
		handle_exception(ex);
	}
}

} // extern "C"
