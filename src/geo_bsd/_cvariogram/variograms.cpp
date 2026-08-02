#include <math.h>
#include <cmath>
#include <memory.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <limits>
#include <mutex>
#include <random>
#include <string>

#include "api.h"

namespace {
    constexpr int MAX_POINT_SET_SIZE = 1000000;

    // F-38: magnitude caps on template parameters. These mirror the Python
    // wrapper caps (cvariogram.py / validation.py MAX_RADIUS) so a
    // pathological template (e.g. lag_separation=1e6 with num_lags=10000)
    // fails fast instead of looping ~1e11 times in the search window.
    constexpr double MAX_LAG_SEPARATION = 1e6;
    constexpr double MAX_FIRST_LAG_DISTANCE = 1e6;
    constexpr double MAX_TEMPLATE_RADIUS = 1e6;   // R1/R2/R3
    constexpr int MAX_NUM_LAGS = 10000;
    // Mirrors variogram.py MAX_WINDOW_VOLUME: bounds the total number of
    // integer offsets in the search window, so even per-parameter values
    // under the individual caps cannot combine into an effectively-infinite
    // window loop (lag_sep * num_lags * R2 * R3 product guard).
    constexpr double MAX_WINDOW_VOLUME = 100000000.0;

    // F-H2: total-work cap for the point-set path
    // (calc_variograms_from_point_set). The pair loop is O(size^2) and each
    // in-tunnel pair bins into up to lag_count lags, so the worst-case work
    // is size^2 * lag_count. MAX_POINT_SET_SIZE (1e6) combined with
    // MAX_NUM_LAGS (1e4) would permit up to 1e16 pair-lag operations — an
    // effectively-infinite loop. This cap rejects the product before the
    // loop starts. Mirrored in cvariogram.py (CalcVariogramsFromPointSet).
    constexpr double MAX_TOTAL_PAIR_LAG_WORK = 1e12;
    // F-M12: total-work cap for the grid path (calc_variograms). The F-38
    // MAX_WINDOW_VOLUME cap bounds only the number of window OFFSETS; the
    // inner loop then iterates ALL grid cells for every in-tunnel offset, so
    // the real work is window_volume * grid_volume. A maximal window (1e8)
    // over a maximal grid (1e9 cells) would be 1e17 cell-offset iterations.
    // This cap rejects the product before the loop starts. Mirrored in
    // cvariogram.py (CalcVariograms).
    constexpr double MAX_TOTAL_GRID_WORK = 1e12;

    /// Thread-safe error storage for the cvariogram module.
    std::string last_cvariogram_error;
    std::mutex last_cvariogram_error_mutex;
}

/// Exported: retrieves the last error message (thread-safe, C ABI).
///
/// Lifetime: The returned pointer is valid until the next call to
/// cvar_get_last_error() from the same thread. Callers that need
/// the error string beyond the next call should copy it.
extern "C" const char * cvar_get_last_error(void)
{
    try
    {
        std::lock_guard<std::mutex> lock(last_cvariogram_error_mutex);
        thread_local char cached[1024];
        snprintf(cached, sizeof(cached), "%s", last_cvariogram_error.c_str());
        return cached;
    }
    catch (const std::exception & ex)
    {
        cvar_set_last_error(ex.what());
        return "";
    }
}

/// Internal: stores an error message (thread-safe).
void cvar_set_last_error(const char * message)
{
    std::lock_guard<std::mutex> lock(last_cvariogram_error_mutex);
    last_cvariogram_error = message;
}

/// Exported: clears the last error (thread-safe, C ABI).
///
/// The Python wrapper snapshots the C-side error before each computation and
/// suppresses a post-call error identical to the snapshot (stale suppression).
/// Because the C-side error was never cleared, two consecutive identical
/// failures were both suppressed after the first raise. Python calls this
/// after a successful computation so the snapshot is empty on the next call
/// and consecutive identical failures are no longer suppressed (F-37).
/// M-17: declared in api.h with CVAR_API so it is dllexport'ed on Windows
/// (previously defined extern "C" without the macro and not declared — the
/// Python hasattr() check silently disabled stale-error clearing).
CVAR_API void cvar_clear_last_error(void)
{
    std::lock_guard<std::mutex> lock(last_cvariogram_error_mutex);
    last_cvariogram_error.clear();
}

namespace {

/// Per-thread RNG engine, seeded from std::random_device (mixed with time as
/// entropy backup). Thread-safe by construction — each thread gets its own
/// independent engine with no shared mutable state.
thread_local std::mt19937 g_tls_rng(
    []() -> std::mt19937::result_type {
        std::random_device rd;
        // Mix random_device entropy with time: on platforms where
        // random_device is deterministic (e.g. older MinGW), time
        // adds uniquification; on true-entropy platforms, the XOR
        // is harmless.
        auto seed = static_cast<std::mt19937::result_type>(
            static_cast<std::mt19937::result_type>(rd()) ^
            static_cast<std::mt19937::result_type>(time(nullptr)));
        return seed;
    }()
);

thread_local std::uniform_int_distribution<int> g_tls_percent_dist(0, 99);

/// Validates template geometry before the expensive search loop (F-38/F-40).
///
/// Returns true when the template is usable. On a degenerate template
/// (zero/NaN range, zero/NaN direction vector, non-finite lag parameters) or
/// a parameter that exceeds the magnitude caps, sets the module error and
/// returns false so callers fail loudly instead of silently producing an
/// all-zero variogram or looping ~1e11 times.
bool validate_template(variogram_search_template_t * templ)
{
    if (templ == nullptr)
    {
        cvar_set_last_error("validate_template: templ is null");
        return false;
    }

    const double R1 = templ->m_ellipsoid.m_R1;
    const double R2 = templ->m_ellipsoid.m_R2;
    const double R3 = templ->m_ellipsoid.m_R3;

    // F-40: degenerate ranges (zero or NaN) silently return false from
    // is_in_tunnel and produce an all-zero variogram with no error.
    if (!std::isfinite(R1) || !std::isfinite(R2) || !std::isfinite(R3) ||
        R1 <= 0.0 || R2 <= 0.0 || R3 <= 0.0)
    {
        cvar_set_last_error("variogram template: ellipsoid ranges must be finite and positive");
        return false;
    }

    // F-40: zero or NaN direction vectors make every tunnel test false
    // (silent all-zero). Check all three vectors for finiteness and norm.
    const vector_t * dirs[3] = {
        &templ->m_ellipsoid.m_direction1,
        &templ->m_ellipsoid.m_direction2,
        &templ->m_ellipsoid.m_direction3
    };
    for (int d = 0; d < 3; ++d)
    {
        double norm2 = 0.0;
        for (int c = 0; c < 3; ++c)
        {
            const double v = dirs[d]->m_data[c];
            if (!std::isfinite(v))
            {
                cvar_set_last_error("variogram template: direction vectors must be finite");
                return false;
            }
            norm2 += v * v;
        }
        if (norm2 == 0.0)
        {
            cvar_set_last_error("variogram template: direction vectors must be non-zero");
            return false;
        }
    }

    // F-38: magnitude caps mirror the Python wrapper so a pathological
    // template cannot drive an effectively-infinite search-window loop.
    if (!std::isfinite(templ->m_lag_separation) ||
        templ->m_lag_separation <= 0.0 ||
        templ->m_lag_separation > MAX_LAG_SEPARATION)
    {
        cvar_set_last_error("variogram template: lag_separation must be in (0, 1e6]");
        return false;
    }
    if (!std::isfinite(templ->m_first_lag_distance) ||
        templ->m_first_lag_distance < 0.0 ||
        templ->m_first_lag_distance > MAX_FIRST_LAG_DISTANCE)
    {
        cvar_set_last_error("variogram template: first_lag_distance must be in [0, 1e6]");
        return false;
    }
    if (!std::isfinite(templ->m_lag_width) || templ->m_lag_width <= 0.0)
    {
        cvar_set_last_error("variogram template: lag_width must be finite and positive");
        return false;
    }
    if (templ->m_num_lags <= 0 || templ->m_num_lags > MAX_NUM_LAGS)
    {
        cvar_set_last_error("variogram template: num_lags must be in [1, 10000]");
        return false;
    }
    if (R1 > MAX_TEMPLATE_RADIUS || R2 > MAX_TEMPLATE_RADIUS || R3 > MAX_TEMPLATE_RADIUS)
    {
        cvar_set_last_error("variogram template: ellipsoid ranges must not exceed 1e6");
        return false;
    }

    return true;
}

/// Validates a non-null pointer. Sets error and returns false on null.
template<typename T>
bool validate_ptr(T * ptr, const char * param_name)
{
    if (ptr == nullptr)
    {
        std::string msg = std::string("Null pointer argument: ") + param_name;
        cvar_set_last_error(msg.c_str());
        fprintf(stderr, "[HPGL ERROR] %s\n", msg.c_str());
        fflush(stderr);
        return false;
    }
    return true;
}

}

double dot_product(vector_t * vec1, vector_t * vec2)
{
	double result = 0.0;
	for (int i = 0; i < 3; ++i)
	{
		result += vec1->m_data[i] * vec2->m_data[i];
	}
	return result;
}
void dot_product_v(vector_t * vec1, vector_t * vec2, double * results, int count)
{
	for (int i = 0; i < count; ++i)
	{
		results[i] = dot_product(&(vec1[i]), &(vec2[i]));
	}
}

bool is_in_tunnel(
		variogram_search_template_t * templ,
		vector_t * vec)
{
    try
    {
	if (!validate_ptr(templ, "templ (is_in_tunnel)")) return false;
	if (!validate_ptr(vec, "vec (is_in_tunnel)")) return false;

	// M22: Guard against zero direction vectors — if any ellipsoid
	// direction is the zero vector, is_in_tunnel returns true for ALL
	// input vectors (degenerate case). Bail early and signal (F-40).
	if ((templ->m_ellipsoid.m_direction1.m_data[0] == 0.0
	  && templ->m_ellipsoid.m_direction1.m_data[1] == 0.0
	  && templ->m_ellipsoid.m_direction1.m_data[2] == 0.0)
	 || (templ->m_ellipsoid.m_direction2.m_data[0] == 0.0
	  && templ->m_ellipsoid.m_direction2.m_data[1] == 0.0
	  && templ->m_ellipsoid.m_direction2.m_data[2] == 0.0)
	 || (templ->m_ellipsoid.m_direction3.m_data[0] == 0.0
	  && templ->m_ellipsoid.m_direction3.m_data[1] == 0.0
	  && templ->m_ellipsoid.m_direction3.m_data[2] == 0.0))
	{
		cvar_set_last_error("is_in_tunnel: ellipsoid direction vector is zero");
		return false;
	}

	double ss1 = fabs(dot_product(vec, &(templ->m_ellipsoid.m_direction1)));
	double ss2 = fabs(dot_product(vec, &(templ->m_ellipsoid.m_direction2)));
	double ss3 = fabs(dot_product(vec, &(templ->m_ellipsoid.m_direction3)));

	// F-40: NaN range (NaN angles flow into direction vectors / ranges)
	// silently returned false before; signal instead.
	if (!std::isfinite(ss1) || !std::isfinite(ss2) || !std::isfinite(ss3))
	{
		cvar_set_last_error("is_in_tunnel: direction vectors or ranges contain NaN");
		return false;
	}

	if (templ->m_ellipsoid.m_R1 == 0 ||
	    templ->m_ellipsoid.m_R2 == 0 ||
	    templ->m_ellipsoid.m_R3 == 0)
	{
		cvar_set_last_error("is_in_tunnel: ellipsoid range must be non-zero");
		return false;
	}

	double s1 = ss1 / templ->m_ellipsoid.m_R1;
	double s2 = ss2 / templ->m_ellipsoid.m_R2;
	double s3 = ss3 / templ->m_ellipsoid.m_R3;

	double dist = sqrt(s2*s2 + s3*s3);
	bool result = (dist <= 1.0) && (templ->m_tol_distance * dist <= s1);
	return result;
    }
    catch (const std::exception & ex)
    {
        cvar_set_last_error(ex.what());
        return false;
    }
}

bool is_in_tunnel_v(
		variogram_search_template_t * templ,
		vector_t * vec, bool * results, int count)
{
    try
    {
	if (!validate_ptr(templ, "templ (is_in_tunnel_v)")) return false;
	if (!validate_ptr(vec, "vec (is_in_tunnel_v)")) return false;
	if (!validate_ptr(results, "results (is_in_tunnel_v)")) return false;

	for (int i = 0; i < count; ++i)
	{
		results[i] = is_in_tunnel(templ, &(vec[i]));
	}
	return true;
    }
    catch (const std::exception & ex)
    {
        cvar_set_last_error(ex.what());
        return false;
    }
}



void vec_by_scalar(vector_t * vec, double scal, vector_t * result)
{
	for (int i = 0; i < 3; ++i)
	{
		result->m_data[i] = vec->m_data[i] * scal;
	}
}

void sum_vec(vector_t * vec1, vector_t * vec2, vector_t * result)
{
	for (int i = 0; i < 3; ++i)
	{
		result->m_data[i] = vec1->m_data[i] + vec2->m_data[i];
	}
}

void set_max(double * current_max, double candidate)
{
	if (candidate > *current_max)
		*current_max = candidate;
}

void set_min(double * current_min, double candidate)
{
	if (candidate < *current_min)
		*current_min = candidate;
}


void calc_search_template_window(
		variogram_search_template_t * templ,
		search_template_window_t * window)
{
    try
    {
	if (!validate_ptr(templ, "templ (calc_search_template_window)")) return;
	if (!validate_ptr(window, "window (calc_search_template_window)")) return;

	double max = 1e10;
	double mini, maxi, minj, maxj, mink, maxk;
	mini = minj = mink = max;
	maxi = maxj = maxk = -max;
	
	for (int i = 0; i < 2; ++i)
		for (int j = -1; j < 3; j += 2)
			for (int k = -1; k < 3; k += 2)
			{
				vector_t DI = {0};
				vector_t DJ = {0};
				vector_t DK = {0};
				vector_t V = {0};

				// F-61: include half the lag width in the window extent so the
				// window covers the full lag band, matching the Python
				// reference _CalcSearchTemplateWindow (variogram.py).
				vec_by_scalar(&templ->m_ellipsoid.m_direction1,
					templ->m_lag_separation * templ->m_num_lags
						+ templ->m_first_lag_distance + templ->m_lag_width / 2.0, &DI);
				vec_by_scalar(&DI, i, &DI);

				vec_by_scalar(&templ->m_ellipsoid.m_direction2, templ->m_ellipsoid.m_R2, &DJ);
				vec_by_scalar(&DJ, j, &DJ);
				
				vec_by_scalar(&templ->m_ellipsoid.m_direction3, templ->m_ellipsoid.m_R3, &DK);
				vec_by_scalar(&DK, k, &DK);

				sum_vec(&DI, &DJ, &V);
				sum_vec(&V, &DK, &V);

				set_min(&mini, V.m_data[0]);
				set_max(&maxi, V.m_data[0]);
				set_min(&minj, V.m_data[1]);
				set_max(&maxj, V.m_data[1]);
				set_min(&mink, V.m_data[2]);
				set_max(&maxk, V.m_data[2]);
			}
	window->m_min_i = mini;
	window->m_max_i = maxi;
	window->m_min_j = minj;
	window->m_max_j = maxj;
	window->m_min_k = mink;
	window->m_max_k = maxk;
    }
    catch (const std::exception & ex)
    {
        cvar_set_last_error(ex.what());
    }
}

struct lag_t
{
	int m_index;
	double m_distance;
	double m_start;
	double m_end;
};

void init_lag_list(variogram_search_template_t * templ, lag_t * lags, int count)
{
	int lag_count = count;
	for (int i = 0; i < lag_count; ++i)
	{
		lags[i].m_index = i;
		lags[i].m_distance = i * templ->m_lag_separation + templ->m_first_lag_distance;
		double width = templ->m_lag_width;
		lags[i].m_start = lags[i].m_distance - width / 2;
		lags[i].m_end = lags[i].m_distance + width / 2;
	}
}

struct lag_statistics_t
{
	double m_cov_sum;
	int64_t m_cov_count;
};

bool is_inside(hard_data_t * data, int x, int y, int z)
{
	if (x < 0 || y < 0 || z < 0)
		return false;
	if (x >= data->m_mask_shape[0] ||
		y >= data->m_mask_shape[1] ||
		z >= data->m_mask_shape[2])
		return false;
	return true;
}

double get_value(hard_data_t * data, int x, int y, int z)
{
	int idx = 
		data->m_data_strides[0] * x +
		data->m_data_strides[1] * y +
		data->m_data_strides[2] * z;

	return data->m_data[idx];
}

double calc_dist(vector_t * vec)
{
	double result = 0;
	for (int i = 0; i < 3; ++i)
	{
		result += vec->m_data[i] * vec->m_data[i]; //pow(vec->m_data[i], 2.0);
	}
	return sqrt(result);
}

int64_t get_offset(vector_t * vec, int * strides)
{
	int64_t result = 0;
	for (int i = 0; i < 3; ++i)
		result += static_cast<int64_t>(vec->m_data[i]) * strides[i];
	return result;
}

void 
update_lags(
		variogram_search_template_t * templ,
		lag_statistics_t * lag_stats,
		int lag_count,
		double dist,
		double var)
{
	if (templ->m_lag_separation == 0) {
		cvar_set_last_error("update_lags: lag_separation is zero, cannot bin lags");
		return;
	}
	int lag_min = (int) ceil( (dist - (templ->m_lag_width / 2) - templ->m_first_lag_distance) / templ->m_lag_separation);
	if (lag_min >= lag_count)
		return;
	int lag_max = (int) floor( (dist + (templ->m_lag_width / 2) - templ->m_first_lag_distance) / templ->m_lag_separation);
	if (lag_max < 0)
		return;
	if (lag_max >= lag_count)
		lag_max = lag_count - 1;
	if (lag_min < 0)
		lag_min = 0;
	
	for (int lag_idx = lag_min; lag_idx <= lag_max; ++lag_idx)
		//for (int lag_idx = 0; lag_idx < lag_count; ++lag_idx)
	{
		//						if (lags[lag_idx].m_start <= dist
		//	&& dist < lags[lag_idx].m_end)
		{
			lag_stats[lag_idx].m_cov_sum += var;
			lag_stats[lag_idx].m_cov_count += 1;
		}
	}
}

/// Internal kernel shared by the exported calc_variograms (time-seeded
/// default) and calc_variograms_seeded (caller-provided seed, 2-M-3).
static void calc_variograms_impl(
		variogram_search_template_t * templ,
		hard_data_t * data,
		float * result_covariations,
		int result_length,
		int percentToUse)		
{
    lag_statistics_t * lag_stats = nullptr;
    try
    {
	if (!validate_ptr(templ, "templ (calc_variograms)")) return;
	if (!validate_ptr(data, "data (calc_variograms)")) return;
	if (!validate_ptr(result_covariations, "result_covariations (calc_variograms)")) return;

	// F-38/F-40: reject degenerate or oversized templates before the
	// search loop so we fail loudly instead of looping ~1e11 times or
	// silently returning an all-zero variogram.
	if (!validate_template(templ)) return;

	int lag_count = templ->m_num_lags <= result_length 
		? templ->m_num_lags
		: result_length;

	lag_stats = (lag_statistics_t *) calloc(lag_count, sizeof(lag_statistics_t));
	if (!lag_stats) {
		fprintf(stderr, "[HPGL ERROR] calc_variograms: calloc(lag_stats) failed\n");
		fflush(stderr);
		cvar_set_last_error("calc_variograms: calloc(lag_stats) failed — out of memory");
		return;
	}
	for (int i = 0; i < lag_count; ++i)
	{
		lag_stats[i].m_cov_count = 0;
		lag_stats[i].m_cov_sum = 0.0;
	}

	//lag_t * lags = (lag_t*) calloc(lag_count, sizeof(lag_t));
	//init_lag_list(templ, lags, lag_count);

	search_template_window_t window;
	calc_search_template_window(templ, &window);	

	// F-38: bound the total number of window offsets. Even with each
	// individual parameter under its cap, lag_sep * num_lags * R2 * R3 can
	// produce ~1e11 loop iterations (pre-fix: effectively infinite hang).
	double window_volume =
		(std::ceil(window.m_max_i) - std::floor(window.m_min_i) + 1.0) *
		(std::ceil(window.m_max_j) - std::floor(window.m_min_j) + 1.0) *
		(std::ceil(window.m_max_k) - std::floor(window.m_min_k) + 1.0);
	if (window_volume > MAX_WINDOW_VOLUME)
	{
		cvar_set_last_error("calc_variograms: search window volume exceeds maximum (1e8)");
		free(lag_stats);
		lag_stats = nullptr;
		return;
	}

	// F-M12: the F-38 cap above bounds only the number of window OFFSETS.
	// For every in-tunnel offset the inner loop iterates ALL grid cells, so
	// the effective total work is window_volume * grid_volume. A maximal
	// window (1e8 offsets) over a maximal grid (1e9 cells) would run 1e17
	// cell-offset iterations — an effectively-infinite loop even though each
	// quantity is under its individual cap. Compute the grid volume in
	// double (int dims can overflow: 1e7^3 = 1e21 > INT_MAX) and reject the
	// product before the loop.
	double grid_volume =
		static_cast<double>(data->m_data_shape[0]) *
		static_cast<double>(data->m_data_shape[1]) *
		static_cast<double>(data->m_data_shape[2]);
	if (window_volume * grid_volume > MAX_TOTAL_GRID_WORK)
	{
		cvar_set_last_error("calc_variograms: total work (window offsets x grid cells) exceeds maximum (1e12)");
		free(lag_stats);
		lag_stats = nullptr;
		return;
	}

	for (int i2 = (int)floor(window.m_min_i); i2 <= (int)ceil(window.m_max_i); ++i2)
		for (int j2 = (int)floor(window.m_min_j); j2 <= (int)ceil(window.m_max_j); ++j2)
			for (int k2 = (int)floor(window.m_min_k); k2 <= (int)ceil(window.m_max_k); ++k2)
			{
				vector_t vec;
				vec.m_data[0] = i2;
				vec.m_data[1] = j2;
				vec.m_data[2] = k2;

				int64_t doffset = get_offset(&vec, data->m_data_strides);
				int64_t moffset = get_offset(&vec, data->m_mask_strides);

				if (is_in_tunnel(templ, &vec))
				{
					int x1, y1, z1;
					float * dpx, * dpy, * dpz;
					unsigned char * mpx, * mpy, *mpz;
					for (z1 = 0, dpz = data->m_data, mpz = data->m_mask; 
						 z1 < data->m_data_shape[2];
						 ++z1, dpz += data->m_data_strides[2], mpz += data->m_mask_strides[2])
						for (y1 = 0, dpy = dpz, mpy = mpz; 
							 y1 < data->m_data_shape[1]; 
							 ++y1, dpy += data->m_data_strides[1], mpy += data->m_mask_strides[1])
							for (x1 = 0, dpx = dpy, mpx = mpy; 
								 x1 < data->m_data_shape[0]; 
								 ++x1, dpx += data->m_data_strides[0], mpx += data->m_mask_strides[0])
							{
								if (*mpx != 0)
								{
									double v1 = *dpx;
									if (!std::isfinite(v1)) continue;
									//double v1 = get_value(data, x1, y1, z1);
									int x = x1 + vec.m_data[0];
									int y = y1 + vec.m_data[1];
									int z = z1 + vec.m_data[2];
									if (is_inside(data, x,y,z) && mpx[moffset])
									{
										if (g_tls_percent_dist(g_tls_rng) >= percentToUse)
											continue;
										// Directional projection onto the
										// principal anisotropy axis for lag
										// binning (see calc_lag_areas).
										double dist = fabs(dot_product(&vec, &(templ->m_ellipsoid.m_direction1)));
										double v2 = dpx[doffset];
										if (!std::isfinite(v2)) continue;
										//double v2 = get_value(data, x, y, z);
										double var = v1 - v2;
										var = var*var;
										update_lags(templ, lag_stats, lag_count, dist, var);									
									}
								}
							}
				}
			}

	// F-40 convergence: an empty lag set (zero pairs found in the search
	// template) is a legitimate outcome for a VALID template on sparse data
	// (e.g. first_lag_distance beyond the maximum data separation). Degenerate
	// templates are already rejected by validate_template() above; do not
	// treat zero pairs as an error here.
	for (int i = 0; i < lag_count; ++i)
	{
		if (lag_stats[i].m_cov_count > 0)
			result_covariations[i] = lag_stats[i].m_cov_sum / lag_stats[i].m_cov_count / 2;
		else
			result_covariations[i] = 0;
	}

	free(lag_stats);
	lag_stats = nullptr;
	//free(lags);
    }
    catch (const std::exception & ex)
    {
        free(lag_stats);
        cvar_set_last_error(ex.what());
    }
}

/// Exported: grid-path variogram with the legacy time-seeded RNG behavior.
/// Declared in api.h with CVAR_API (C linkage / Windows export).
void calc_variograms(
		variogram_search_template_t * templ,
		hard_data_t * data,
		float * result_covariations,
		int result_length,
		int percentToUse)
{
    calc_variograms_impl(templ, data, result_covariations, result_length, percentToUse);
}

/// Exported: grid-path variogram with a caller-provided RNG seed (2-M-3).
/// Re-seeds the per-thread mt19937 so identical inputs + identical seed
/// produce an identical variogram (reproducible published experiments).
/// The seed is honored literally — 0 is a valid seed.  Declared in api.h
/// with CVAR_API (C linkage / Windows export).
void calc_variograms_seeded(
		variogram_search_template_t * templ,
		hard_data_t * data,
		float * result_covariations,
		int result_length,
		int percentToUse,
		uint64_t seed)
{
    // 2-M-3: deterministic re-seed of the thread_local engine on the calling
    // thread. The unseeded entry point leaves the engine at its
    // random_device^time default (legacy non-deterministic behavior).
    g_tls_rng.seed(static_cast<std::mt19937::result_type>(seed));
    calc_variograms_impl(templ, data, result_covariations, result_length, percentToUse);
}

void calc_variograms_from_point_set(
		variogram_search_template_t * templ,
		cont_point_set_t * point_set,
		float * result_covariations,
		int result_length)
{
    lag_statistics_t * lag_stats = nullptr;
    lag_t * lags = nullptr;
    try
    {
	if (!validate_ptr(templ, "templ (calc_variograms_from_point_set)")) return;
	if (!validate_ptr(result_covariations, "result_covariations (calc_variograms_from_point_set)")) return;

	if (point_set == nullptr || point_set->size > MAX_POINT_SET_SIZE)
	{
		fprintf(stderr,
			"[HPGL ERROR] calc_variograms_from_point_set: point_set size %d exceeds maximum %d\n",
			point_set ? point_set->size : 0, MAX_POINT_SET_SIZE);
		fflush(stderr);
		cvar_set_last_error("calc_variograms_from_point_set: invalid point_set");
		return;
	}

	// Validate member pointers to prevent null dereference at lines 668-670
	if (point_set->xs == nullptr || point_set->ys == nullptr ||
	    point_set->zs == nullptr || point_set->values == nullptr)
	{
		fprintf(stderr,
			"[HPGL ERROR] calc_variograms_from_point_set: point_set member pointer is null (xs=%p ys=%p zs=%p values=%p)\n",
			(void*)point_set->xs, (void*)point_set->ys,
			(void*)point_set->zs, (void*)point_set->values);
		fflush(stderr);
		cvar_set_last_error("calc_variograms_from_point_set: null member pointer in point_set");
		return;
	}

	// F-38/F-40: reject degenerate or oversized templates before the
	// O(n^2) pair loop.
	if (!validate_template(templ)) return;

	int lag_count = templ->m_num_lags <= result_length 
		? templ->m_num_lags
		: result_length;

	// F-H2: work-based cap on the O(size^2 * lag_count) pair loop below.
	// The count cap above (MAX_POINT_SET_SIZE) bounds the input SIZE only;
	// the loop cost is size^2 * num_lags, so a size of 1e6 with 1e4 lags
	// would run 1e16 pair-lag operations — an effectively-infinite loop.
	// Compute the work estimate in double (size^2 for 1e6 points is 1e12,
	// safely inside double but the product with lag_count must not overflow
	// int) and reject it before any allocation or loop work.
	const double pair_lag_work =
		static_cast<double>(point_set->size) *
		static_cast<double>(point_set->size) *
		static_cast<double>(lag_count);
	if (pair_lag_work > MAX_TOTAL_PAIR_LAG_WORK)
	{
		fprintf(stderr,
			"[HPGL ERROR] calc_variograms_from_point_set: estimated pair-lag work %.3g exceeds maximum %.3g\n",
			pair_lag_work, MAX_TOTAL_PAIR_LAG_WORK);
		fflush(stderr);
		cvar_set_last_error("calc_variograms_from_point_set: point-set pair-lag work exceeds maximum (1e12)");
		return;
	}

	lag_stats = (lag_statistics_t *) calloc(lag_count, sizeof(lag_statistics_t));
	if (!lag_stats) {
		fprintf(stderr, "[HPGL ERROR] calc_variograms_from_point_set: calloc(lag_stats) failed\n");
		fflush(stderr);
		cvar_set_last_error("calc_variograms_from_point_set: calloc(lag_stats) failed — out of memory");
		return;
	}
	for (int i = 0; i < lag_count; ++i)
	{
		lag_stats[i].m_cov_count = 0;
		lag_stats[i].m_cov_sum = 0.0;
	}

	lags = (lag_t*) calloc(lag_count, sizeof(lag_t));
	if (!lags) {
		free(lag_stats);
		fprintf(stderr, "[HPGL ERROR] calc_variograms_from_point_set: calloc(lags) failed\n");
		fflush(stderr);
		cvar_set_last_error("calc_variograms_from_point_set: calloc(lags) failed — out of memory");
		return;
	}
	init_lag_list(templ, lags, lag_count);

	// F-M13-cpp: contiguity contract. The point-set arrays (xs/ys/zs/values)
	// are read LINEARLY via pointer arithmetic (point_set->xs[idx1],
	// point_set->values[idx2], ...) with no stride math, exactly like the
	// grid path reads its contiguous flat buffer. This is safe ONLY when the
	// caller passes C-contiguous 1-D float32 arrays of `size` elements.
	//
	// The Python wrapper (cvariogram.py CalcVariogramsFromPointSet) MUST
	// enforce ndim == 1 and C-contiguity on the "X"/"Y"/"Z"/"Property"
	// arrays before constructing cont_point_set_t (sibling precedent:
	// CStackLayers validates layer contiguity at the Python boundary,
	// cvariogram.py CStackLayers). A non-contiguous view (e.g. a strided
	// slice of a 2-D array) would be read here as if it were contiguous,
	// producing garbage coordinate/value pairs or OOB reads.
	//
	// C++ side is stride-agnostic-safe ONLY under that contract; the C++
	// cont_point_set_t struct carries no shape/strides metadata to detect
	// violations, so the validation must happen at the FFI boundary.
	for (int idx1 = 0; idx1 < point_set->size; ++idx1)
	{
		for (int idx2 = 0; idx2 < point_set->size; ++idx2)
		{
			if (idx1 == idx2) continue;
			vector_t vec;
			vec.m_data[0] = point_set->xs[idx1] - point_set->xs[idx2];
			vec.m_data[1] = point_set->ys[idx1] - point_set->ys[idx2];
			vec.m_data[2] = point_set->zs[idx1] - point_set->zs[idx2];
			if (is_in_tunnel(templ, &vec))
			{
				double v1 = point_set->values[idx1];
				double v2 = point_set->values[idx2];
				if (!std::isfinite(v1) || !std::isfinite(v2)) continue;
				// Directional projection onto the principal
				// anisotropy axis for lag binning (see calc_lag_areas).
				double dist = fabs(dot_product(&vec, &(templ->m_ellipsoid.m_direction1)));
				double var = pow(v1 - v2, 2);
				for (int lag_idx = 0; lag_idx < lag_count; ++lag_idx)
				{
					if (lags[lag_idx].m_start <= dist
						&& dist < lags[lag_idx].m_end)
					{
						lag_stats[lag_idx].m_cov_sum += var;
						lag_stats[lag_idx].m_cov_count += 1;
					}
				}
			}
		}
	}

	// F-40 convergence: an empty lag set (no valid pairs) is a legitimate
	// outcome for a VALID template on sparse data; degenerate templates are
	// already rejected by validate_template() above. Do not signal an error.
	for (int i = 0; i < lag_count; ++i)
	{
		if (lag_stats[i].m_cov_count == 0)
			result_covariations[i] = 0;
		else
			result_covariations[i] = lag_stats[i].m_cov_sum / lag_stats[i].m_cov_count / 2;
	}

	free(lag_stats);
	free(lags);
	lag_stats = nullptr;
	lags = nullptr;
    }
    catch (const std::exception & ex)
    {
        free(lag_stats);
        free(lags);
        cvar_set_last_error(ex.what());
    }
}
		
