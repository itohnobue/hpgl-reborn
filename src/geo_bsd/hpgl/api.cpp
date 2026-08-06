#include "stdafx.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>

#include "api.h"
#include "api_helpers.hpp"
#include "hpgl_core.h"
#include "kriging_stats.h"
#include "sugarbox_grid.h"
#include "ok_params.h"
#include "sk_params.h"
#include "sgs_params.h"
#include "ik_params.h"
#include "median_ik.h"
#include "property_writer.h"
#include "progress_reporter.h"
#include "hpgl_exception.h"
#include "covariance_field.h"
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

/// Validates that all three grid dimensions (sx, sy, sz) are positive.
/// get_shape_volume checks only the product — even-negative dimensions
/// produce a positive product that passes volume validation but causes
/// undefined behavior in grid initialization.
static void validate_shape_dims_or_throw(hpgl_shape_t * shape, const char * context)
{
	for (int i = 0; i < 3; ++i)
	{
		if (shape->m_data[i] <= 0)
		{
			std::ostringstream oss;
			oss << context << ": dimension " << i << " is " << shape->m_data[i]
			    << " — all dimensions must be positive";
			throw hpgl::hpgl_exception("C API", oss.str());
		}
	}
}

/// Hard upper bound on m_max_neighbours accepted by the C API.
/// The neighbour lookups reserve O(max_neighbours) per grid node
/// (sugarbox_neighbour_lookup.h:40-42), so an unbounded value (e.g.
/// 2e9, which passes the `< 0` check) causes ~32GB of heap reserve.
/// E-M85: the previous bound of 100000 admitted per-node kriging
/// matrices of 80-240 GB + ~3.3e14 flop solves (OOM/DoS); the bound is
/// lowered to 10000 — the NEIGHBOUR_WORK_CAP the lookup agents chose as
/// the effective scan/result cap on both lookup paths
/// (sugarbox_neighbour_lookup.h, sugarbox_indexed_neighbour_lookup.h),
/// so no legal config is truncated by the new bound. The bound is also
/// aligned with the internal cokriging system-size limit
/// (simple_cokriging_markI.cpp build_system: ms > 10001 = cap + 1 for
/// the secondary equation). PR-07: values above the solver's safe limit
/// must be rejected with a clear error, not silently degraded.
static const int MAX_NEIGHBOURS_UPPER_BOUND = 10000;

/// Hard upper bound on hpgl_non_parametric_cdf_t::m_size accepted by the C
/// API. The kernel trusts m_size as the length of the m_values/m_probs
/// arrays in std::lower_bound (non_parametric_cdf.h:233,271) — raw C
/// pointers carry no length metadata, so an unbounded m_size makes the range
/// walk perform heap OOB reads (III-34). The bound is far above any
/// legitimate non-parametric CDF (built from conditioning data values; the
/// Python layer passes the CdfData length) while keeping the worst-case
/// array footprint (values + probs, 8 bytes/pair) bounded.
static const long long MAX_NON_PARAMETRIC_CDF_SIZE = 100000000;

/// Validates m_max_neighbours for a C API kriging/simulation entry point.
/// Throws hpgl_exception (caught by the caller's try/catch) on negative or
/// pathologically large values.
static void validate_max_neighbours_or_throw(int max_neighbours, const char * context)
{
	if (max_neighbours < 0)
	{
		std::ostringstream oss;
		oss << context << ": m_max_neighbours cannot be negative";
		throw hpgl::hpgl_exception("C API", oss.str());
	}
	if (max_neighbours > MAX_NEIGHBOURS_UPPER_BOUND)
	{
		std::ostringstream oss;
		oss << context << ": m_max_neighbours " << max_neighbours
		    << " exceeds maximum allowed (" << MAX_NEIGHBOURS_UPPER_BOUND << ")";
		throw hpgl::hpgl_exception("C API", oss.str());
	}
}

/// Validates m_max_neighbours on KRIGING entry points (M-31).
/// Kriging requires at least one conditioning neighbour — a zero
/// max_neighbours produces an empty neighbourhood → SK all-mean-fill /
/// OK all-undefined for direct C callers, with no error. The < 0 and
/// upper-bound checks mirror validate_max_neighbours_or_throw. Simulation
/// entry points (SGS/SIS, where max_neighbours=0 means unconditional
/// simulation) must keep using validate_max_neighbours_or_throw.
static void validate_kriging_max_neighbours_or_throw(int max_neighbours, const char * context)
{
	if (max_neighbours < 1)
	{
		std::ostringstream oss;
		oss << context << ": m_max_neighbours must be at least 1 for kriging (got "
		    << max_neighbours << ")";
		throw hpgl::hpgl_exception("C API", oss.str());
	}
	if (max_neighbours > MAX_NEIGHBOURS_UPPER_BOUND)
	{
		std::ostringstream oss;
		oss << context << ": m_max_neighbours " << max_neighbours
		    << " exceeds maximum allowed (" << MAX_NEIGHBOURS_UPPER_BOUND << ")";
		throw hpgl::hpgl_exception("C API", oss.str());
	}
}

// Defined in neighbourhood_param.cpp — rejects a zero-radius search
// neighbourhood on kriging paths (F-34). Simulation paths (SGS zero-radius
// CDF draw) are intentionally exempt.
namespace hpgl {
	void validate_kriging_radiuses_or_throw(
			const sugarbox_search_ellipsoid_t & radiuses,
			const char * context);
}

/// Validates simulation_mask shape matches grid shape when mask is provided.
/// Mismatched shapes cause out-of-bounds array access in simulation kernels.
static void validate_simulation_mask_shape_or_throw(
	hpgl_ubyte_array_t * simulation_mask,
	hpgl_shape_t * grid_shape,
	const char * context)
{
	if (simulation_mask != nullptr && simulation_mask->m_data != nullptr)
	{
		for (int i = 0; i < 3; ++i)
		{
			if (simulation_mask->m_shape.m_data[i] != grid_shape->m_data[i])
			{
				std::ostringstream oss;
				oss << context << ": simulation_mask shape[" << i << "] ("
				    << simulation_mask->m_shape.m_data[i]
				    << ") != grid shape[" << i << "] ("
				    << grid_shape->m_data[i] << ")";
				throw hpgl::hpgl_exception(context, oss.str());
			}
		}
	}
}

/// E-M77: content scan of the non-parametric CDF arrays (m_values/m_probs)
/// at the C boundary. non_parametric_cdf_2_t trusts the raw arrays for
/// std::lower_bound (non_parametric_cdf.h:233,271) — a NaN value/prob
/// silently produces NaN simulated values for direct-C callers (the CDF is
/// the one directly-consumed array class lacking the codebase's II-02..II-06
/// content scan), and an unsorted array makes lower_bound undefined. The
/// Python wrapper validates (cdf.py:50-53); the C boundary is the last
/// line of defense. m_size > 0 and non-null m_values/m_probs must already
/// be validated by the caller.
static void validate_cdf_content_or_throw(
	const hpgl_non_parametric_cdf_t * cdf,
	const char * context)
{
	const long long n = cdf->m_size;
	for (long long i = 0; i < n; ++i)
	{
		if (!std::isfinite(cdf->m_values[i]) || !std::isfinite(cdf->m_probs[i]))
		{
			std::ostringstream oss;
			oss << context << ": cdf values/probs[" << i << "] is not finite";
			throw hpgl::hpgl_exception("C API", oss.str());
		}
		if (i > 0)
		{
			if (cdf->m_values[i] < cdf->m_values[i - 1])
			{
				std::ostringstream oss;
				oss << context << ": cdf m_values must be sorted non-decreasing "
				    << "(values[" << (i - 1) << "]=" << cdf->m_values[i - 1]
				    << " > values[" << i << "]=" << cdf->m_values[i] << ")";
				throw hpgl::hpgl_exception("C API", oss.str());
			}
			if (cdf->m_probs[i] < cdf->m_probs[i - 1])
			{
				std::ostringstream oss;
				oss << context << ": cdf m_probs must be sorted non-decreasing "
				    << "(probs[" << (i - 1) << "]=" << cdf->m_probs[i - 1]
				    << " > probs[" << i << "]=" << cdf->m_probs[i] << ")";
				throw hpgl::hpgl_exception("C API", oss.str());
			}
		}
	}
}

// Thread-local storage for kriging statistics — populated by the
// kriging implementation functions (ordinary_kriging, simple_kriging,
// lvm_kriging) and retrieved via hpgl_get_kriging_stats().
namespace {
	thread_local hpgl::kriging_stats_t g_last_kriging_stats;
}

namespace hpgl {
	void set_kriging_stats(const kriging_stats_t & stats)
	{
		g_last_kriging_stats = stats;
	}

	void reset_kriging_stats()
	{
		g_last_kriging_stats.m_points_calculated = 0;
		g_last_kriging_stats.m_points_without_neighbours = 0;
		g_last_kriging_stats.m_points_singularity = 0;
		g_last_kriging_stats.m_mean = 0;
		g_last_kriging_stats.m_speed_nps = 0;
	}
}

extern "C" {

// ============================================================================
// C-API validation registry (pat-20260802223236 — recurring class: entry
// points that don't validate what the Python wrapper validates). EVERY
// exported hpgl_* entry point MUST have a row here. The generated
// completeness test (test_api_validation_registry.py) walks the api.h
// declarations and the library's exported symbols, and fails when any entry
// point is missing a registry row. When adding a NEW entry point, add its
// row here at the same time — the test enforces the pairing.
// ============================================================================
struct hpgl_validation_registry_entry_t {
	const char * m_name;
	const char * m_validation;
};

static const hpgl_validation_registry_entry_t HPGL_VALIDATION_REGISTRY[] = {
	{ "hpgl_get_api_validation_registry_count", "accessor — no user input" },
	{ "hpgl_get_api_validation_registry_name", "index bounds (see registry accessors)" },
	{ "hpgl_get_api_validation_registry_validation", "index bounds (see registry accessors)" },
	{ "hpgl_get_last_exception_message", "accessor — no user input" },
	{ "hpgl_set_output_handler", "handler pointer (null clears)" },
	{ "hpgl_set_progress_handler", "handler pointer (null clears)" },
	{ "hpgl_set_thread_num", "thread count (set_thread_num.cpp validates > 0)" },
	{ "hpgl_get_thread_num", "accessor — no user input" },
	{ "hpgl_get_kriging_stats", "accessor — no user input" },
	{ "hpgl_read_inc_file_float", "filename/data/mask pointers; size > 0" },
	{ "hpgl_read_inc_file_byte", "filename/data/mask/values pointers; size > 0; values_count in [1, 256]; remap coverage" },
	{ "hpgl_write_inc_file_float", "filename/arr/name pointers; shape dims > 0; volume; data non-null" },
	{ "hpgl_write_inc_file_byte", "filename/arr/name pointers; shape dims > 0; volume; values_count >= 0; remap table; undefined_value in [0, 255]" },
	{ "hpgl_write_gslib_cont_property", "data/filename/name pointers; shape dims > 0; volume; data non-null" },
	{ "hpgl_write_gslib_byte_property", "data/filename/name/values pointers; shape dims > 0; volume; values_count >= 0; remap table; undefined_value in [0, 255]" },
	{ "hpgl_ordinary_kriging", "input/params/output pointers; shape dims > 0; equal volumes; data non-null; max_neighbours in [1, 10000]; non-zero radius; cov params" },
	{ "hpgl_simple_kriging", "data/mask/shape/params/output pointers; shape dims > 0; equal volumes; max_neighbours in [1, 10000]; non-zero radius; mean finite; cov params" },
	{ "hpgl_simple_kriging_weights", "params/weights/center/neighbour pointers; neighbours_count in [0, 10000]; weights2 size match" },
	{ "hpgl_lvm_kriging", "data/mean/params/output pointers; shape dims > 0; equal volumes; mean volume == input volume; mean finite; max_neighbours in [1, 10000]; non-zero radius" },
	{ "hpgl_sgs_simulation", "data/params pointers; shape dims > 0; data non-null; max_neighbours in [0, 10000]; kriging_kind valid; cdf size in [0, 1e8] + non-null arrays; mean finite; mask shape match" },
	{ "hpgl_sgs_lvm_simulation", "data/params/means pointers; shape dims > 0; data/means non-null; means volume == grid volume; means finite; max_neighbours in [0, 10000]; cdf size in [0, 1e8]; mask shape match" },
	{ "hpgl_indicator_kriging", "in/out/params pointers; shape dims > 0; indicator_count in [1, 255] matches in/out; data non-null; per-category max_neighbours in [1, 10000]; non-zero radius" },
	{ "hpgl_median_ik", "in/params/out pointers; shape dims > 0; equal volumes; data non-null; indicator_count == 2; max_neighbours in [1, 10000]; non-zero radius; marginal_probs in [0, 1] sum ~1" },
	{ "hpgl_sis_simulation", "data/params pointers; shape dims > 0; indicator_count in [1, 255] matches data; data non-null; per-category max_neighbours in [0, 10000]; mask shape match" },
	{ "hpgl_sis_simulation_lvm", "data/params/mean pointers; shape dims > 0; indicator_count in [1, 255] matches data; data non-null; per-category max_neighbours in [0, 10000]; per-mean volume == grid volume; means finite; mask shape match" },
	{ "hpgl_simple_cokriging_mark1", "input/secondary/output/params pointers; shape dims > 0; equal volumes + per-dim equality; data non-null; max_neighbours in [1, 10000]; non-zero radius; means finite; secondary_variance finite >= 0" },
	{ "hpgl_simple_cokriging_mark2", "primary/secondary/output/params pointers; shape dims > 0; equal volumes + per-dim equality; data non-null; max_neighbours in [1, 10000]; non-zero radius; means finite" },
};

static const int HPGL_VALIDATION_REGISTRY_COUNT =
	static_cast<int>(sizeof(HPGL_VALIDATION_REGISTRY) / sizeof(HPGL_VALIDATION_REGISTRY[0]));

HPGL_API int hpgl_get_api_validation_registry_count()
{
	return HPGL_VALIDATION_REGISTRY_COUNT;
}

HPGL_API const char * hpgl_get_api_validation_registry_name(int index)
{
	if (index < 0 || index >= HPGL_VALIDATION_REGISTRY_COUNT)
		return "";
	return HPGL_VALIDATION_REGISTRY[index].m_name;
}

HPGL_API const char * hpgl_get_api_validation_registry_validation(int index)
{
	if (index < 0 || index >= HPGL_VALIDATION_REGISTRY_COUNT)
		return "";
	return HPGL_VALIDATION_REGISTRY[index].m_validation;
}

HPGL_API hpgl_kriging_stats_t hpgl_get_kriging_stats()
{
	hpgl_kriging_stats_t result;
	result.m_points_calculated = g_last_kriging_stats.m_points_calculated;
	result.m_points_without_neighbours = g_last_kriging_stats.m_points_without_neighbours;
	result.m_points_singularity = g_last_kriging_stats.m_points_singularity;
	result.m_mean = g_last_kriging_stats.m_mean;
	result.m_speed_nps = g_last_kriging_stats.m_speed_nps;
	return result;
}

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
	// E2-55: undefined_value is the exact-equality mask sentinel in the
	// reader (read_inc_file.cpp: v == undefined_value → masked). A NaN
	// undefined_value matches NO token (IEEE-754: NaN == NaN is false), so
	// every cell would read as informed — silently discarding the file's
	// missing-value convention. The byte sibling rejects out-of-range
	// sentinels (read_inc_file.cpp:370-380); the float entry needs the
	// finiteness gate for sibling consistency (a non-finite sentinel is
	// never a legitimate exact-match marker).
	if (!std::isfinite(undefined_value))
	{
		hpgl::set_last_exception_message("read_inc_file_float: undefined_value must be finite");
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
	// II-50(a) / III-03: values_count maps a byte value to the index j
	// written into data[i]. values_count > 256 makes `data[i] = j` overflow
	// the unsigned char cell for j >= 256 (mod-256 wrap) and is nonsensical
	// for a byte domain. Sibling writers cap via init_remap_table (<= 255
	// indicators); the reader admits the full 256 distinct byte values.
	if (values_count > 256)
	{
		hpgl::set_last_exception_message("read_inc_file_byte: values_count must be <= 256");
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

		// II-50(b): build the remap in a temp buffer and copy to the caller
		// ONLY on success. Pre-fix the remap mutated the caller's buffer in
		// place, so an unmapped-byte throw midway left a torn buffer (some
		// cells remapped, others not) observable by the caller even though
		// the call returned an error.
		std::vector<unsigned char> mapped_data(data, data + size);
		for (int i = 0; i < size; ++i)
		{
			if (mask[i] != 0)
			{
				unsigned char original_value = mapped_data[i];
				bool mapped = false;
				for (int j = 0; j < values_count; ++j)
				{
					if (original_value == values[j])
					{
						mapped_data[i] = j;
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
		std::copy(mapped_data.begin(), mapped_data.end(), data);
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
		validate_shape_dims_or_throw(&(arr->m_shape), "write_inc_file_float");
		int vol = validate_shape_volume(get_shape_volume(&(arr->m_shape)), "write_inc_file_float");
		if (vol < 0) return -1;
		property_writer_t writer;
		writer.init(filename, name);
		if (arr->m_data == nullptr)
		{
			hpgl::set_last_exception_message("hpgl_write_inc_file_float: Null data pointer in arr");
			return -1;
		}
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

	if (values == nullptr || values_count == 0)
	{
		// No remap table provided: use identity mapping [0, 1, 2, ...]
		// values_count == 0 also means "no remap values" — the Python layer
		// passes a non-null pointer even for an empty numpy array, so the
		// null-check alone cannot detect the documented identity default
		// (F-05).
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
		validate_shape_dims_or_throw(&(arr->m_shape), "write_inc_file_byte");
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
		if (arr->m_data == nullptr)
		{
			hpgl::set_last_exception_message("hpgl_write_inc_file_byte: Null data pointer in arr");
			return -1;
		}
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
		validate_shape_dims_or_throw(&data->m_shape, "write_gslib_cont_property");
		int size = validate_shape_volume(get_shape_volume(&data->m_shape), "write_gslib_cont_property");
		if (size < 0) return -1;
		if (data->m_data == nullptr)
		{
			hpgl::set_last_exception_message("hpgl_write_gslib_cont_property: Null data pointer");
			return -1;
		}
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
		validate_shape_dims_or_throw(&data->m_shape, "write_gslib_byte_property");
		int size = validate_shape_volume(get_shape_volume(&data->m_shape), "write_gslib_byte_property");
		if (size < 0) return -1;
		if (data->m_data == nullptr)
		{
			hpgl::set_last_exception_message("hpgl_write_gslib_byte_property: Null data pointer");
			return -1;
		}
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
	// F-N2: zero thread-local stats before the call so a validation
	// failure or stats-less path cannot leave stale stats observable.
	reset_kriging_stats();
	validate_pointer_or_throw(input_data, "input_data (ordinary_kriging)");
	validate_pointer_or_throw(params, "params (ordinary_kriging)");
	validate_pointer_or_throw(output_data, "output_data (ordinary_kriging)");

	validate_shape_dims_or_throw(&input_data->m_shape, "ordinary_kriging input shape");
	validate_shape_dims_or_throw(&output_data->m_shape, "ordinary_kriging output shape");

	int in_size = get_shape_volume(&input_data->m_shape);
	validate_shape_volume_or_throw(in_size, "ordinary_kriging input");
	int out_size = get_shape_volume(&output_data->m_shape);
	validate_shape_volume_or_throw(out_size, "ordinary_kriging output");

	if (in_size != out_size)
		throw hpgl_exception("hpgl_ordinary_kriging", "input and output shape volume mismatch");

	// Validate m_data pointer before constructing property array.
	// A null m_data pointer causes HPGL_CHECK→abort() (SIGABRT) which
	// Python cannot catch (F-28).
	if (input_data->m_data == nullptr)
		throw hpgl_exception("hpgl_ordinary_kriging", "Null data pointer in input_data");
	if (output_data->m_data == nullptr)
		throw hpgl_exception("hpgl_ordinary_kriging", "Null data pointer in output_data");

	// E2-53: reject aliased input/output buffers. The kernel reads input
	// live while writing output (cont_kriging.h:163-166,169,180) — an
	// aliased buffer (in == out) produces progressive overwrite plus an
	// OpenMP data race on the mask. Python always passes fresh clones;
	// this guard protects direct-C callers. Mask aliasing is rejected
	// only when both masks are non-null (two null masks are the
	// all-active convention, not an alias).
	if (input_data->m_data == output_data->m_data)
		throw hpgl_exception("hpgl_ordinary_kriging",
			"input_data and output_data must not be the same buffer (aliased input/output is unsupported)");
	if (input_data->m_mask != nullptr && input_data->m_mask == output_data->m_mask)
		throw hpgl_exception("hpgl_ordinary_kriging",
			"input_data and output_data masks must not be the same buffer (aliased masks are unsupported)");

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
	validate_kriging_radiuses_or_throw(ok_p.m_radiuses, "hpgl_ordinary_kriging");

	// M-31: kriging requires >= 1 neighbour — reject max_neighbours=0.
	validate_kriging_max_neighbours_or_throw(params->m_max_neighbours, "hpgl_ordinary_kriging");
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
	validate_kriging_radiuses_or_throw(sk_p.m_radiuses, "init_sk_params");

	// M-31: kriging requires >= 1 neighbour — reject max_neighbours=0.
	validate_kriging_max_neighbours_or_throw(params->m_max_neighbours, "init_sk_params");
	sk_p.m_max_neighbours = params->m_max_neighbours;

	if (!params->m_automatic_mean)
	{
		// II-02: m_mean is consumed directly as the SK mean — a NaN/Inf
		// mean silently produces all-NaN output for direct-C callers (the
		// Python wrapper validates finiteness itself). No sibling isfinite
		// gate existed at the C boundary.
		if (!std::isfinite(params->m_mean))
			throw hpgl_exception("init_sk_params", "m_mean must be finite");
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
	// F-N2: zero thread-local stats before the call (stale-stat promise).
	reset_kriging_stats();
	validate_pointer_or_throw(input_data, "input_data (simple_kriging)");
	validate_pointer_or_throw(output_data, "output_data (simple_kriging)");
	validate_pointer_or_throw(input_data_shape, "input_data_shape (simple_kriging)");
	validate_pointer_or_throw(params, "params (simple_kriging)");
	validate_pointer_or_throw(output_data_shape, "output_data_shape (simple_kriging)");

	validate_shape_dims_or_throw(input_data_shape, "simple_kriging input shape");
	validate_shape_dims_or_throw(output_data_shape, "simple_kriging output shape");

	int in_size = get_shape_volume(input_data_shape);
	validate_shape_volume_or_throw(in_size, "simple_kriging input");
	int out_size = get_shape_volume(output_data_shape);
	validate_shape_volume_or_throw(out_size, "simple_kriging output");

	if (in_size != out_size)
		throw hpgl_exception("hpgl_simple_kriging", "input and output shape volume mismatch");

	// E2-53: reject aliased input/output buffers — the kernel reads input
	// live while writing output (cont_kriging.h:163-166,169,180); an
	// aliased buffer produces progressive overwrite + an OpenMP data race.
	// Python always passes fresh clones; this guard protects direct-C
	// callers. Mask aliasing is rejected only when both masks are
	// non-null (two null masks are the all-active convention).
	if (input_data == output_data)
		throw hpgl_exception("hpgl_simple_kriging",
			"input_data and output_data must not be the same buffer (aliased input/output is unsupported)");
	if (input_mask != nullptr && input_mask == output_mask)
		throw hpgl_exception("hpgl_simple_kriging",
			"input_mask and output_mask must not be the same buffer (aliased masks are unsupported)");

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
	// F-N2 / II-49: zero thread-local stats BEFORE the validation gates.
	// The 6 pointer checks and the neighbours_count gate all return -1
	// without entering the try block — pre-fix, reset_kriging_stats() sat
	// inside the try, so a validation failure left stale stats from a prior
	// kriging call observable via hpgl_get_kriging_stats().
	hpgl::reset_kriging_stats();
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
	// 2-M-2: this was the ONLY kriging entry point without the
	// MAX_NEIGHBOURS_UPPER_BOUND gate (11 siblings enforce it via
	// validate_max_neighbours_or_throw / validate_kriging_max_neighbours_or_throw).
	// An unbounded neighbours_count (a) dereferences the neighbour arrays up to
	// count — a heap OOB read when count exceeds the arrays' actual length — and
	// (b) allocates A = count² (100k pts → 160 GB) + O(count³) solve → OOM/DoS.
	// The raw C pointers carry no length metadata, so this gate bounds the
	// dereference loop and the system size; the Python wrapper additionally
	// validates count against the actual ndarray lengths (geo.py shape check).
	if (neighbours_count > MAX_NEIGHBOURS_UPPER_BOUND)
	{
		hpgl::set_last_exception_message(
			("simple_kriging_weights: neighbours_count " + std::to_string(neighbours_count)
			 + " exceeds maximum allowed (" + std::to_string(MAX_NEIGHBOURS_UPPER_BOUND) + ")").c_str());
		return -1;
	}
	try
	{
	using namespace hpgl;
	// got-20260724074703: the neighbour coordinates are consumed directly
	// as covariance-model distances (cov_model.h operator() → transfrom_and_norm
	// → spherical/gaussian/exponential). A NaN/Inf coordinate silently
	// produces NaN weights (the Python wrapper validates coordinate
	// finiteness; a direct C caller bypasses it). Scan before the kernel
	// runs — the cov_model_t isfinite-first guards (cov_model.h) would catch
	// it downstream, but a loud error here is far cheaper than a NaN
	// weights array.
	for (int i = 0; i < 3; ++i)
	{
		if (!std::isfinite(center_coords[i]))
		{
			hpgl::set_last_exception_message(
				("simple_kriging_weights: center_coords[" + std::to_string(i)
				 + "] is not finite").c_str());
			return -1;
		}
	}
	for (int i = 0; i < neighbours_count; ++i)
	{
		if (!std::isfinite(neighbours_x[i]) || !std::isfinite(neighbours_y[i]) || !std::isfinite(neighbours_z[i]))
		{
			hpgl::set_last_exception_message(
				("simple_kriging_weights: neighbour_coords[" + std::to_string(i)
				 + "] is not finite").c_str());
			return -1;
		}
	}
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

	// F-38: on kriging solve failure (e.g. a singular matrix from
	// all-identical neighbour points) simple_kriging_weights raises a clean
	// hpgl_exception, caught below. Guard the copy anyway so a short
	// weights2 can never surface the standard library's internal
	// out-of-range text ("vector" on libc++, "vector::_M_range_check..." on
	// libstdc++) — the caller must get a meaningful kriging-failure message
	// instead.
	if (weights2.size() != static_cast<size_t>(neighbours_count))
	{
		hpgl::set_last_exception_message(
			"simple_kriging_weights: kriging solve failed (check for identical or degenerate neighbour points)");
		return -1;
	}
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
	// F-N2: zero thread-local stats before the call (stale-stat promise).
	reset_kriging_stats();
	validate_pointer_or_throw(input_data, "input_data (lvm_kriging)");
	validate_pointer_or_throw(mean_data, "mean_data (lvm_kriging)");
	validate_pointer_or_throw(output_data, "output_data (lvm_kriging)");
	validate_pointer_or_throw(input_data_shape, "input_data_shape (lvm_kriging)");
	validate_pointer_or_throw(mean_data_shape, "mean_data_shape (lvm_kriging)");
	validate_pointer_or_throw(params, "params (lvm_kriging)");
	validate_pointer_or_throw(output_data_shape, "output_data_shape (lvm_kriging)");

	validate_shape_dims_or_throw(input_data_shape, "lvm_kriging input shape");
	validate_shape_dims_or_throw(output_data_shape, "lvm_kriging output shape");
	validate_shape_dims_or_throw(mean_data_shape, "lvm_kriging means shape");

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

	// II-03: mean_data contents are consumed directly as the per-node local
	// mean by subtract_means/add_means (lvm_utils.h) — a NaN/Inf mean
	// silently poisons every kriged estimate. Pointer/shape/volume are
	// already validated; scan the contents for finiteness.
	for (int i = 0; i < mean_size; ++i)
	{
		if (!std::isfinite(mean_data[i]))
		{
			std::ostringstream oss;
			oss << "lvm_kriging: mean_data[" << i << "] is not finite";
			throw hpgl_exception("hpgl_lvm_kriging", oss.str());
		}
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
	validate_kriging_radiuses_or_throw(ok_p.m_radiuses, "hpgl_lvm_kriging");

	// M-31: kriging requires >= 1 neighbour — reject max_neighbours=0.
	validate_kriging_max_neighbours_or_throw(params->m_max_neighbours, "hpgl_lvm_kriging");
	ok_p.m_max_neighbours = params->m_max_neighbours;

	// E2-53: reject aliased input/output buffers — the kernel reads input
	// live while writing output (cont_kriging.h); an aliased buffer
	// produces progressive overwrite + an OpenMP data race. Python always
	// passes fresh clones; this guard protects direct-C callers. Mask
	// aliasing is rejected only when both masks are non-null.
	if (input_data == output_data)
		throw hpgl_exception("hpgl_lvm_kriging",
			"input_data and output_data must not be the same buffer (aliased input/output is unsupported)");
	if (input_mask != nullptr && input_mask == output_mask)
		throw hpgl_exception("hpgl_lvm_kriging",
			"input_mask and output_mask must not be the same buffer (aliased masks are unsupported)");

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
	// F-N2: zero thread-local stats before the call (stale-stat promise).
	reset_kriging_stats();
	validate_pointer_or_throw(in_data, "in_data (indicator_kriging)");
	validate_pointer_or_throw(out_data, "out_data (indicator_kriging)");
	validate_pointer_or_throw(params, "params (indicator_kriging)");

	validate_shape_dims_or_throw(&in_data->m_shape, "indicator_kriging input shape");
	validate_shape_dims_or_throw(&out_data->m_shape, "indicator_kriging output shape");

	// Validate indicator_count matches the data's actual indicator_count.
	// Direct C callers may pass mismatched values, leading to out-of-bounds reads
	// in init_sis_params or indicator_kriging.
	if (indicator_count != in_data->m_indicator_count)
		throw hpgl_exception("hpgl_indicator_kriging",
			"indicator_count mismatch with in_data->m_indicator_count");

	// III-35: indicator_count <= 0 silently no-ops — the kernel loops zero
	// categories and every node falls through to the fallback
	// (indicator_kriging.h:150-151), so a 0-category call looks successful
	// but writes nothing meaningful. SIS siblings throw on <= 0
	// (api.cpp:1249-1251,1327-1329) — mirror that guard.
	if (indicator_count <= 0)
		throw hpgl_exception("hpgl_indicator_kriging",
			"indicator_count must be positive");

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

	// Validate m_data pointers before constructing property arrays.
	// A null m_data pointer causes HPGL_CHECK→abort() (SIGABRT) which
	// Python cannot catch (F-28).
	if (in_data->m_data == nullptr)
		throw hpgl_exception("hpgl_indicator_kriging", "Null data pointer in in_data");
	if (out_data->m_data == nullptr)
		throw hpgl_exception("hpgl_indicator_kriging", "Null data pointer in out_data");

	// E2-53: reject aliased input/output buffers — the IK kernel reads
	// input live while writing output (indicator_kriging.h:253,286); an
	// aliased buffer produces progressive overwrite + an OpenMP data race.
	// Python always passes fresh clones; this guard protects direct-C
	// callers. Mask aliasing is rejected only when both masks are
	// non-null.
	if (in_data->m_data == out_data->m_data)
		throw hpgl_exception("hpgl_indicator_kriging",
			"in_data and out_data must not be the same buffer (aliased input/output is unsupported)");
	if (in_data->m_mask != nullptr && in_data->m_mask == out_data->m_mask)
		throw hpgl_exception("hpgl_indicator_kriging",
			"in_data and out_data masks must not be the same buffer (aliased masks are unsupported)");

	// PR-05 (F-34): reject a zero search radius on the indicator kriging
	// path. An all-zero ellipsoid yields an empty neighbourhood and every
	// node silently degrades to mean/noise fill. Each indicator category
	// carries its own search radius (init_sis_params pushes one ellipsoid
	// per category). Simulation paths (SIS zero-radius CDF draw) are
	// intentionally exempt — the guard lives on kriging entry points only.
	for (int i = 0; i < indicator_count; ++i)
	{
		sugarbox_search_ellipsoid_t radiuses(
				params[i].m_radiuses[0],
				params[i].m_radiuses[1],
				params[i].m_radiuses[2]);
		validate_kriging_radiuses_or_throw(radiuses, "hpgl_indicator_kriging");
	}

	// PR-07 (F-H1): enforce the max-neighbours cap on the ONLY kriging entry
	// point that lacked it. hpgl_indicator_kriging was the sole sibling with
	// no validate_max_neighbours_or_throw call — an unbounded value (e.g.
	// 2e9, which passes init_sis_params' `< 0` check) flows into the
	// neighbour lookup's per-node reserve() (sugarbox_neighbour_lookup.h:40-42)
	// → ~32GB/thread heap reserve inside the OpenMP region → uncatchable
	// std::terminate. All 10 other entry points (api.cpp:631,650,821,940,994,
	// 1070,1158,1233,1341,1451) enforce the same bound.
	// M-31: kriging requires >= 1 neighbour — reject max_neighbours=0.
	for (int i = 0; i < indicator_count; ++i)
		validate_kriging_max_neighbours_or_throw(params[i].m_max_neighbours, "hpgl_indicator_kriging");

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
	// F-N2: zero thread-local stats before the call (stale-stat promise).
	reset_kriging_stats();
	validate_pointer_or_throw(data, "data (sgs_simulation)");
	validate_pointer_or_throw(params, "params (sgs_simulation)");

	validate_shape_dims_or_throw(&data->m_shape, "sgs_simulation");

	// Validate m_data pointer before constructing property array.
	// The constructor stores nullptr silently; later operator[] access
	// triggers HPGL_CHECK→abort() which Python cannot catch.
	if (data->m_data == nullptr)
		throw hpgl_exception("hpgl_sgs_simulation", "Null data pointer in cont_masked_array");

	int size = get_shape_volume(&(data->m_shape));
	validate_shape_volume_or_throw(size, "sgs_simulation");
	cont_property_array_t prop(data->m_data, data->m_mask, size);

	// E2-113: finite-scan the conditioning DATA contents — NaN/Inf data
	// silently produces NaN simulated values with kriging_failures==0
	// (the kernel consumes informed values only; sibling means-scan
	// pattern in hpgl_sgs_lvm_simulation). Uninformed cells may hold
	// arbitrary sentinel values (the Python wrapper keeps NaN at masked
	// cells), so only informed cells are scanned.
	{
		const unsigned char * d_mask = data->m_mask;
		for (int i = 0; i < size; ++i)
		{
			if (d_mask == nullptr || d_mask[i] != 0)
			{
				if (!std::isfinite(data->m_data[i]))
				{
					std::ostringstream oss;
					oss << "hpgl_sgs_simulation: conditioning data[" << i << "] is not finite";
					throw hpgl_exception("hpgl_sgs_simulation", oss.str());
				}
			}
		}
	}

	sugarbox_grid_t grid;
	init_grid(grid, &(data->m_shape));

	sgs_params_t sgs_p;
	validate_max_neighbours_or_throw(params->m_max_neighbours, "hpgl_sgs_simulation");
	init_sgs_params(params, &sgs_p);

	// E2-112: validate the search radiuses at the C boundary BEFORE the
	// kernel call. The SGS kernel's in-place forward CDF transform
	// (sequential_gaussian_simulation.cpp:44) destroys the caller's
	// conditioning buffer BEFORE the kernel's throwing radius guard —
	// the covariance-box volume guard (precalculated_covariances_t::init →
	// validate_covariance_radiuses_or_throw) — so a failed run silently
	// corrupted the direct-C caller's data.
	// Sibling parity: the 7 kriging entries gate radiuses at the C
	// boundary; the simulation family had no gate. Zero radius stays
	// legal (documented unconditional-simulation mode). The check below
	// mirrors the kernel's throwing check exactly.
	// R-21 (CONFIRMED MEDIUM): the former "radius > 10x grid extent"
	// heuristic is NOT part of this gate anymore — it rejected shipped
	// workflows (book 7.3/2_var.py radiuses (160,40,1) on a 10x10x1 grid;
	// sample-scripts/sk_test.py radiuses (20,20,20) on a 286x10x1 grid)
	// and the (2r+1)³ volume cap alone bounds the memory (see the
	// neighbour_lookup_t ctor in sugarbox_neighbour_lookup.h).
	{
		sugarbox_search_ellipsoid_t sgs_radiuses(
			params->m_radiuses[0], params->m_radiuses[1], params->m_radiuses[2]);
		validate_covariance_radiuses_or_throw(sgs_radiuses, "hpgl_sgs_simulation");
	}

	// F-18 / III-34: validate the non-parametric CDF struct at the C
	// boundary before handing it to the kernel. null m_values/m_probs with
	// m_size > 0 cause SIGSEGV inside non_parametric_cdf_2_t::operator()
	// (std::lower_bound over a null range); m_size is trusted verbatim as
	// the lower_bound range (non_parametric_cdf.h:233,271), so a negative or
	// absurdly large size performs heap OOB reads. A null cdf pointer is
	// valid — the kernel treats it as "no transform".
	if (cdf != nullptr)
	{
		if (cdf->m_size < 0 || cdf->m_size > MAX_NON_PARAMETRIC_CDF_SIZE)
		{
			std::ostringstream oss;
			oss << "sgs_simulation: cdf m_size " << cdf->m_size
			    << " out of range [0, " << MAX_NON_PARAMETRIC_CDF_SIZE << "]";
			throw hpgl_exception("hpgl_sgs_simulation", oss.str());
		}
		if (cdf->m_size > 0)
		{
			if (cdf->m_values == nullptr)
				throw hpgl_exception("hpgl_sgs_simulation",
					"cdf m_values is null (m_size > 0)");
			if (cdf->m_probs == nullptr)
				throw hpgl_exception("hpgl_sgs_simulation",
					"cdf m_probs is null (m_size > 0)");
			// E-M77: content scan — finiteness of values/probs plus
			// non-decreasing sortedness (std::lower_bound preconditions at
			// non_parametric_cdf.h:233,271). NaN CDF entries silently
			// produce NaN simulated values for direct-C callers; unsorted
			// arrays make lower_bound undefined.
			validate_cdf_content_or_throw(cdf, "hpgl_sgs_simulation");
		}
	}

	if (mean != 0)
	{
		// F-18: a non-finite stationary mean silently produces NaN
		// simulated values (the kernel consumes it directly).
		if (!std::isfinite(*mean))
			throw hpgl_exception("hpgl_sgs_simulation", "mean must be finite");
		sgs_p.set_mean(*mean);
	}
	// 2-M-1(c): m_mean_kind is consumed by sequential_gaussian_simulation
	// (auto vs user stationary mean); e_mean_varying is set on the LVM path.
	sgs_p.m_mean_kind = mean != 0 ? mean_kind_t::e_mean_stationary : mean_kind_t::e_mean_stationary_auto;

	// Validate simulation mask shape matches grid shape to prevent
	// out-of-bounds memory access in the simulation kernel.
	validate_simulation_mask_shape_or_throw(simulation_mask, &data->m_shape, "hpgl_sgs_simulation");

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
	// F-N2: zero thread-local stats before the call (stale-stat promise).
	reset_kriging_stats();
	validate_pointer_or_throw(data, "data (sgs_lvm_simulation)");
	validate_pointer_or_throw(params, "params (sgs_lvm_simulation)");
	validate_pointer_or_throw(means, "means (sgs_lvm_simulation)");

	validate_shape_dims_or_throw(&data->m_shape, "sgs_lvm_simulation");
	validate_shape_dims_or_throw(&means->m_shape, "sgs_lvm_simulation means");

	// Validate m_data pointer before constructing property array.
	// The constructor stores nullptr silently; later operator[] access
	// triggers HPGL_CHECK→abort() which Python cannot catch.
	if (data->m_data == nullptr)
		throw hpgl_exception("hpgl_sgs_lvm_simulation", "Null data pointer in cont_masked_array");

	int size = get_shape_volume(&(data->m_shape));
	validate_shape_volume_or_throw(size, "sgs_lvm_simulation");
	cont_property_array_t prop(data->m_data, data->m_mask, size);

	// E2-113: finite-scan the conditioning DATA contents (sibling of the
	// hpgl_sgs_simulation scan) — NaN/Inf informed data silently produces
	// NaN simulated values with kriging_failures==0.
	{
		const unsigned char * d_mask = data->m_mask;
		for (int i = 0; i < size; ++i)
		{
			if (d_mask == nullptr || d_mask[i] != 0)
			{
				if (!std::isfinite(data->m_data[i]))
				{
					std::ostringstream oss;
					oss << "hpgl_sgs_lvm_simulation: conditioning data[" << i << "] is not finite";
					throw hpgl_exception("hpgl_sgs_lvm_simulation", oss.str());
				}
			}
		}
	}

	sugarbox_grid_t grid;
	init_grid(grid, &(data->m_shape));

	sgs_params_t sgs_p;
	validate_max_neighbours_or_throw(params->m_max_neighbours, "hpgl_sgs_lvm_simulation");
	init_sgs_params(params, &sgs_p);

	// E2-112: search-radius gate at the C boundary BEFORE the kernel call
	// (sibling of the hpgl_sgs_simulation gate) — the in-place forward CDF
	// transform destroys the conditioning buffer before the kernel's
	// throwing radius guard (covariance volume cap) can run. Zero radius
	// stays legal. The check below mirrors the kernel's throwing check
	// exactly. R-21: the former "radius > 10x grid extent" heuristic is
	// removed (see the hpgl_sgs_simulation gate comment) — the (2r+1)³
	// volume cap alone bounds the memory.
	{
		sugarbox_search_ellipsoid_t sgs_radiuses(
			params->m_radiuses[0], params->m_radiuses[1], params->m_radiuses[2]);
		validate_covariance_radiuses_or_throw(sgs_radiuses, "hpgl_sgs_lvm_simulation");
	}

	// F-18 / III-34: validate the non-parametric CDF struct at the C
	// boundary (sibling of the hpgl_sgs_simulation gate). null
	// m_values/m_probs with m_size > 0 SIGSEGV in the kernel's
	// std::lower_bound; m_size is trusted verbatim as the range length.
	if (cdf != nullptr)
	{
		if (cdf->m_size < 0 || cdf->m_size > MAX_NON_PARAMETRIC_CDF_SIZE)
		{
			std::ostringstream oss;
			oss << "sgs_lvm_simulation: cdf m_size " << cdf->m_size
			    << " out of range [0, " << MAX_NON_PARAMETRIC_CDF_SIZE << "]";
			throw hpgl_exception("hpgl_sgs_lvm_simulation", oss.str());
		}
		if (cdf->m_size > 0)
		{
			if (cdf->m_values == nullptr)
				throw hpgl_exception("hpgl_sgs_lvm_simulation",
					"cdf m_values is null (m_size > 0)");
			if (cdf->m_probs == nullptr)
				throw hpgl_exception("hpgl_sgs_lvm_simulation",
					"cdf m_probs is null (m_size > 0)");
			// E-M77: content scan (sibling of the hpgl_sgs_simulation
			// gate) — finiteness + non-decreasing sortedness of the CDF
			// arrays (lower_bound preconditions).
			validate_cdf_content_or_throw(cdf, "hpgl_sgs_lvm_simulation");
		}
	}

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

	// II-04: means contents are consumed as the per-node local mean — NaN
	// means produce NaN simulated values silently. Volume is already
	// validated; scan the contents for finiteness.
	for (int i = 0; i < mean_vol; ++i)
	{
		if (!std::isfinite(means->m_data[i]))
		{
			std::ostringstream oss;
			oss << "sgs_lvm_simulation: means data[" << i << "] is not finite";
			throw hpgl_exception("hpgl_sgs_lvm_simulation", oss.str());
		}
	}

	// 2-M-1(c): m_mean_kind is consumed by sequential_gaussian_simulation
	// (the LVM entry point routes the varying mean via the mean_data
	// parameter; e_mean_varying records the mode for print/contract parity).
	sgs_p.m_mean_kind = mean_kind_t::e_mean_varying;

	// Validate simulation mask shape matches grid shape to prevent
	// out-of-bounds memory access in the simulation kernel.
	validate_simulation_mask_shape_or_throw(simulation_mask, &data->m_shape, "hpgl_sgs_lvm_simulation");

	// E2-109: pass the validated means volume as the explicit mean_data
	// length contract (the C++ entry now validates mean_data_size ==
	// output size before walking the raw pointer).
	hpgl::sequential_gaussian_simulation_lvm(
			grid,
			sgs_p,
			means->m_data,
			static_cast<size_t>(mean_vol),
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
	// F-N2: zero thread-local stats before the call (stale-stat promise).
	reset_kriging_stats();
	validate_pointer_or_throw(in_data, "in_data (median_ik)");
	validate_pointer_or_throw(params, "params (median_ik)");
	validate_pointer_or_throw(out_data, "out_data (median_ik)");

	validate_shape_dims_or_throw(&in_data->m_shape, "median_ik input shape");
	validate_shape_dims_or_throw(&out_data->m_shape, "median_ik output shape");

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
	validate_kriging_radiuses_or_throw(mik_p.m_radiuses, "hpgl_median_ik");

	// M-31: kriging requires >= 1 neighbour — reject max_neighbours=0.
	validate_kriging_max_neighbours_or_throw(params->m_max_neighbours, "hpgl_median_ik");
	mik_p.m_max_neighbours = params->m_max_neighbours;
	// 2-M-35 (R-11): marginal_probs are consumed directly as the SK mean and
	// the failure-fallback probability (median_ik.cpp:146,151) — a value
	// outside [0,1] silently produces all-1/all-0 output for direct-C callers
	// (e.g. marginal_probs[1]=5.0 → choose_indicator always returns 1) or a
	// mean-leaked estimate. Mirror the Python wrapper's validation
	// (geo.py:1879-1884 → validate_probability [0,1] + validate_probability_sum
	// with 0.01 tolerance at validation.py:834-881). NaN fails the range
	// checks too (NaN comparisons are false).
	{
		const double p0 = params->m_marginal_probs[0];
		const double p1 = params->m_marginal_probs[1];
		if (!(p0 >= 0.0 && p0 <= 1.0) || !(p1 >= 0.0 && p1 <= 1.0))
			throw hpgl_exception("hpgl_median_ik",
				"marginal_probs must be in [0, 1]");
		const double prob_sum = p0 + p1;
		if (prob_sum < 0.99 || prob_sum > 1.01)
			throw hpgl_exception("hpgl_median_ik",
				"marginal_probs must sum to 1.0 (within 0.01)");
	}
	mik_p.m_marginal_probs[0] = params->m_marginal_probs[0];
	mik_p.m_marginal_probs[1] = params->m_marginal_probs[1];

	// median_ik requires exactly 2 indicators — the data layout is
	// interleaved and median_ik_for_two_indicators reads every other byte.
	// A mismatched indicator_count would read wrong cells.
	if (in_data->m_indicator_count != 2)
		throw hpgl_exception("hpgl_median_ik",
			"indicator_count must be 2 (median IK is defined for binary indicators)");

	// Update out_data->m_indicator_count to match the actual indicator count
	// used (2).  Downstream consumers (e.g. write_gslib_byte_property) read
	// m_indicator_count from the struct, and a stale value would produce
	// wrong output.
	out_data->m_indicator_count = 2;

	// Validate m_data pointers before constructing property arrays.
	// A null m_data pointer causes HPGL_CHECK→abort() (SIGABRT) which
	// Python cannot catch (F-28).
	if (in_data->m_data == nullptr)
		throw hpgl_exception("hpgl_median_ik", "Null data pointer in in_data");
	if (out_data->m_data == nullptr)
		throw hpgl_exception("hpgl_median_ik", "Null data pointer in out_data");

	// E2-53: reject aliased input/output buffers — the median-IK kernel
	// reads input live while writing output; an aliased buffer produces
	// progressive overwrite + an OpenMP data race. Python always passes
	// fresh clones; this guard protects direct-C callers. Mask aliasing
	// is rejected only when both masks are non-null.
	if (in_data->m_data == out_data->m_data)
		throw hpgl_exception("hpgl_median_ik",
			"in_data and out_data must not be the same buffer (aliased input/output is unsupported)");
	if (in_data->m_mask != nullptr && in_data->m_mask == out_data->m_mask)
		throw hpgl_exception("hpgl_median_ik",
			"in_data and out_data masks must not be the same buffer (aliased masks are unsupported)");

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
	// F-N2: zero thread-local stats before the call (stale-stat promise).
	reset_kriging_stats();
	validate_pointer_or_throw(data, "data (sis_simulation)");
	validate_pointer_or_throw(params, "params (sis_simulation)");

	validate_shape_dims_or_throw(&data->m_shape, "sis_simulation");

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

	// Validate m_data pointer before constructing property array.
	// The constructor stores nullptr silently; later operator[] access
	// triggers HPGL_CHECK→abort() which Python cannot catch.
	if (data->m_data == nullptr)
		throw hpgl_exception("hpgl_sis_simulation", "Null data pointer in ind_masked_array");

	int size = get_shape_volume(&data->m_shape);
	validate_shape_volume_or_throw(size, "sis_simulation");
	indicator_property_array_t prop(data->m_data, data->m_mask, size, indicator_count);

	sugarbox_grid_t grid;
	init_grid(grid, &data->m_shape);

	for (int i = 0; i < indicator_count; ++i)
		validate_max_neighbours_or_throw(params[i].m_max_neighbours, "hpgl_sis_simulation");

	ik_params_t ikp;
	init_sis_params(params, indicator_count, &ikp);

	progress_reporter_t rep(size);

	// Validate simulation mask shape matches grid shape to prevent
	// out-of-bounds memory access in the simulation kernel.
	validate_simulation_mask_shape_or_throw(simulation_mask, &data->m_shape, "hpgl_sis_simulation");

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
	// F-N2: zero thread-local stats before the call (stale-stat promise).
	reset_kriging_stats();
	validate_pointer_or_throw(data, "data (sis_simulation_lvm)");
	validate_pointer_or_throw(params, "params (sis_simulation_lvm)");
	validate_pointer_or_throw(mean_data, "mean_data (sis_simulation_lvm)");

	validate_shape_dims_or_throw(&data->m_shape, "sis_simulation_lvm");

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

	// Validate indicator_count against reasonable range BEFORE expensive
	// grid/params initialization to fail fast on invalid input.
	if (indicator_count <= 0 || indicator_count > 1000000)
		throw hpgl_exception("hpgl_sis_simulation_lvm",
			"indicator_count out of reasonable range [1, 1000000]");

	// Validate m_data pointer before constructing property array.
	// The constructor stores nullptr silently; later operator[] access
	// triggers HPGL_CHECK→abort() which Python cannot catch.
	if (data->m_data == nullptr)
		throw hpgl_exception("hpgl_sis_simulation_lvm", "Null data pointer in ind_masked_array");

	indicator_property_array_t prop(data->m_data, data->m_mask, size, indicator_count);

	sugarbox_grid_t grid;
	init_grid(grid, &data->m_shape);

	for (int i = 0; i < indicator_count; ++i)
		validate_max_neighbours_or_throw(params[i].m_max_neighbours, "hpgl_sis_simulation_lvm");

	ik_params_t ikp;
	init_sis_params(params, indicator_count, &ikp);

	progress_reporter_t rep(size);

	// Build means array from mean_data structs.
	std::vector<const mean_t *> means;

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
		// II-05: per-mean contents are consumed directly as the local mean
		// in the LVM indicator kernel — a NaN mean produces NaN category
		// probabilities and sample() silently returns category 0. Scan the
		// contents for finiteness.
		{
			const float * mean_vals = mean_data[i].m_data;
			for (int j = 0; j < mean_vol; ++j)
			{
				if (!std::isfinite(mean_vals[j]))
				{
					std::ostringstream oss;
					oss << "sis_simulation_lvm: means[" << i << "] data[" << j
					    << "] is not finite";
					throw hpgl_exception("hpgl_sis_simulation_lvm", oss.str());
				}
			}
		}
		means.push_back(mean_data[i].m_data);
	}

	// Validate simulation mask shape matches grid shape to prevent
	// out-of-bounds memory access in the simulation kernel.
	validate_simulation_mask_shape_or_throw(simulation_mask, &data->m_shape, "hpgl_sis_simulation_lvm");

	// E2-118: pass the validated mean-array count and row size as the
	// explicit contract (the C++ entry now validates count ==
	// indicator_count and row size == grid volume before dereferencing
	// the raw pointer-to-pointer).
	sequential_indicator_simulation_lvm(
			prop,
			grid,
			ikp,
			seed,
			&means[0],
			static_cast<size_t>(indicator_count),
			static_cast<size_t>(size),
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
		// F-N2: zero thread-local stats before the call (stale-stat promise).
		reset_kriging_stats();
		validate_pointer_or_throw(input_data, "input_data (cokriging_m1)");
		validate_pointer_or_throw(secondary_data, "secondary_data (cokriging_m1)");
		validate_pointer_or_throw(params, "params (cokriging_m1)");
		validate_pointer_or_throw(output_data, "output_data (cokriging_m1)");

		validate_shape_dims_or_throw(&input_data->m_shape, "cokriging_m1 primary");
		validate_shape_dims_or_throw(&secondary_data->m_shape, "cokriging_m1 secondary");
		validate_shape_dims_or_throw(&output_data->m_shape, "cokriging_m1 output");

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

		// III-36: volume-only validation admits equal-volume per-dimension
		// mismatches (e.g. primary (2,2,2) with secondary (1,2,4) — both
		// volume 8). The markI kernel indexes secondary_data by the primary
		// flat index (simple_cokriging_markI.cpp:365-367), so a per-dim
		// mismatch silently permutes the secondary field. Validate per-dim
		// equality (mirror of the lvm_kriging R-13 per-dim pattern).
		for (int i = 0; i < 3; ++i)
		{
			if (input_data->m_shape.m_data[i] != secondary_data->m_shape.m_data[i])
			{
				std::ostringstream oss;
				oss << "cokriging_m1: secondary shape[" << i << "] ("
				    << secondary_data->m_shape.m_data[i]
				    << ") != primary shape[" << i << "] ("
				    << input_data->m_shape.m_data[i] << ")";
				throw hpgl_exception("hpgl_simple_cokriging_mark1", oss.str());
			}
		}

		// Validate m_data pointers before constructing property arrays.
		// A null m_data pointer causes HPGL_CHECK→abort() (SIGABRT) which
		// Python cannot catch (F-28).
		if (input_data->m_data == nullptr)
			throw hpgl_exception("hpgl_simple_cokriging_mark1", "Null data pointer in input_data");
		if (secondary_data->m_data == nullptr)
			throw hpgl_exception("hpgl_simple_cokriging_mark1", "Null data pointer in secondary_data");
		if (output_data->m_data == nullptr)
			throw hpgl_exception("hpgl_simple_cokriging_mark1", "Null data pointer in output_data");

		// E2-53: reject aliased input/output buffers — the cokriging kernel
		// reads primary AND secondary live while writing output
		// (process_node_loop: input_prop.is_informed(i), secondary_data[i]);
		// an aliased buffer produces progressive overwrite. Python always
		// passes fresh clones; this guard protects direct-C callers. Mask
		// aliasing is rejected only when both masks are non-null.
		if (input_data->m_data == output_data->m_data)
			throw hpgl_exception("hpgl_simple_cokriging_mark1",
				"input_data and output_data must not be the same buffer (aliased input/output is unsupported)");
		if (secondary_data->m_data == output_data->m_data)
			throw hpgl_exception("hpgl_simple_cokriging_mark1",
				"secondary_data and output_data must not be the same buffer (aliased secondary/output is unsupported)");
		if (input_data->m_mask != nullptr && input_data->m_mask == output_data->m_mask)
			throw hpgl_exception("hpgl_simple_cokriging_mark1",
				"input_data and output_data masks must not be the same buffer (aliased masks are unsupported)");
		if (secondary_data->m_mask != nullptr && secondary_data->m_mask == output_data->m_mask)
			throw hpgl_exception("hpgl_simple_cokriging_mark1",
				"secondary_data and output_data masks must not be the same buffer (aliased masks are unsupported)");

		cont_property_array_t primary_prop(
				input_data->m_data, input_data->m_mask, size);
		cont_property_array_t secondary_prop(
				secondary_data->m_data, secondary_data->m_mask, size2);
		cont_property_array_t output_prop(
				output_data->m_data, output_data->m_mask, size3);

		neighbourhood_param_t np;
		// M-31: kriging requires >= 1 neighbour — reject max_neighbours=0.
		validate_kriging_max_neighbours_or_throw(params->m_max_neighbours, "hpgl_simple_cokriging_mark1");
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
		np.set_radiuses(
			params->m_radiuses[0],
			params->m_radiuses[1],
			params->m_radiuses[2]);
		validate_kriging_radiuses_or_throw(np.m_radiuses, "hpgl_simple_cokriging_mark1");

		sugarbox_grid_t grid;
		init_grid(grid, &input_data->m_shape);

		// II-06: means/variance are consumed directly by the markI kernel
		// (simple_cokriging_markI.cpp) — a NaN primary/secondary mean or
		// secondary_variance silently produces NaN output (mean-fill) for
		// direct-C callers. Validate finiteness at the C boundary.
		if (!std::isfinite(params->m_primary_mean) || !std::isfinite(params->m_secondary_mean))
		{
			std::ostringstream oss;
			oss << "cokriging_m1: primary_mean/secondary_mean must be finite (got "
			    << params->m_primary_mean << " / " << params->m_secondary_mean << ")";
			throw hpgl_exception("hpgl_simple_cokriging_mark1", oss.str());
		}
		if (!std::isfinite(params->m_secondary_variance) || params->m_secondary_variance < 0.0)
		{
			std::ostringstream oss;
			oss << "cokriging_m1: secondary_variance must be finite and >= 0 (got "
			    << params->m_secondary_variance << ")";
			throw hpgl_exception("hpgl_simple_cokriging_mark1", oss.str());
		}

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
		// F-N2: zero thread-local stats before the call (stale-stat promise).
		reset_kriging_stats();
		validate_pointer_or_throw(primary_data, "primary_data (cokriging_m2)");
		validate_pointer_or_throw(secondary_data, "secondary_data (cokriging_m2)");
		validate_pointer_or_throw(params, "params (cokriging_m2)");
		validate_pointer_or_throw(output_data, "output_data (cokriging_m2)");

		validate_shape_dims_or_throw(&primary_data->m_shape, "cokriging_m2 primary");
		validate_shape_dims_or_throw(&secondary_data->m_shape, "cokriging_m2 secondary");
		validate_shape_dims_or_throw(&output_data->m_shape, "cokriging_m2 output");

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

		// III-36: volume-only validation admits equal-volume per-dimension
		// mismatches (e.g. primary (2,2,2) with secondary (1,2,4) — both
		// volume 8). The markII kernel indexes secondary_data by the primary
		// flat index, so a per-dim mismatch silently permutes the secondary
		// field. Validate per-dim equality (mirror of the lvm_kriging R-13
		// per-dim pattern; sibling of the mark1 gate above).
		for (int i = 0; i < 3; ++i)
		{
			if (primary_data->m_shape.m_data[i] != secondary_data->m_shape.m_data[i])
			{
				std::ostringstream oss;
				oss << "cokriging_m2: secondary shape[" << i << "] ("
				    << secondary_data->m_shape.m_data[i]
				    << ") != primary shape[" << i << "] ("
				    << primary_data->m_shape.m_data[i] << ")";
				throw hpgl_exception("hpgl_simple_cokriging_mark2", oss.str());
			}
		}

		// Validate m_data pointers before constructing property arrays.
		// A null m_data pointer causes HPGL_CHECK→abort() (SIGABRT) which
		// Python cannot catch (F-28).
		if (primary_data->m_data == nullptr)
			throw hpgl_exception("hpgl_simple_cokriging_mark2", "Null data pointer in primary_data");
		if (secondary_data->m_data == nullptr)
			throw hpgl_exception("hpgl_simple_cokriging_mark2", "Null data pointer in secondary_data");
		if (output_data->m_data == nullptr)
			throw hpgl_exception("hpgl_simple_cokriging_mark2", "Null data pointer in output_data");

		// E2-53: reject aliased input/output buffers — the cokriging kernel
		// reads primary AND secondary live while writing output
		// (process_node_loop: input_prop.is_informed(i), secondary_data[i]);
		// an aliased buffer produces progressive overwrite. Python always
		// passes fresh clones; this guard protects direct-C callers. Mask
		// aliasing is rejected only when both masks are non-null.
		if (primary_data->m_data == output_data->m_data)
			throw hpgl_exception("hpgl_simple_cokriging_mark2",
				"primary_data and output_data must not be the same buffer (aliased input/output is unsupported)");
		if (secondary_data->m_data == output_data->m_data)
			throw hpgl_exception("hpgl_simple_cokriging_mark2",
				"secondary_data and output_data must not be the same buffer (aliased secondary/output is unsupported)");
		if (primary_data->m_mask != nullptr && primary_data->m_mask == output_data->m_mask)
			throw hpgl_exception("hpgl_simple_cokriging_mark2",
				"primary_data and output_data masks must not be the same buffer (aliased masks are unsupported)");
		if (secondary_data->m_mask != nullptr && secondary_data->m_mask == output_data->m_mask)
			throw hpgl_exception("hpgl_simple_cokriging_mark2",
				"secondary_data and output_data masks must not be the same buffer (aliased masks are unsupported)");

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
		// M-31: kriging requires >= 1 neighbour — reject max_neighbours=0.
		validate_kriging_max_neighbours_or_throw(params->m_max_neighbours, "hpgl_simple_cokriging_mark2");
		np.m_max_neighbours = params->m_max_neighbours;
		np.set_radiuses(
			params->m_radiuses[0],
			params->m_radiuses[1],
			params->m_radiuses[2]);
		validate_kriging_radiuses_or_throw(np.m_radiuses, "hpgl_simple_cokriging_mark2");

		// II-06: means are consumed directly by the markII kernel — a NaN
		// primary/secondary mean silently produces NaN output (mean-fill)
		// for direct-C callers. correlation_coef range/NaN validation lives
		// in the simple_cokriging_markI.cpp entry guards; the means are not
		// checked anywhere before this gate.
		if (!std::isfinite(params->m_primary_mean) || !std::isfinite(params->m_secondary_mean))
		{
			std::ostringstream oss;
			oss << "cokriging_m2: primary_mean/secondary_mean must be finite (got "
			    << params->m_primary_mean << " / " << params->m_secondary_mean << ")";
			throw hpgl_exception("hpgl_simple_cokriging_mark2", oss.str());
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
