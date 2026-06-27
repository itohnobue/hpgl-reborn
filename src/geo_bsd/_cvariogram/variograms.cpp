#include <math.h>
#include <memory.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits>
#include <mutex>
#include <string>

#include "api.h"

namespace {
    constexpr int MAX_POINT_SET_SIZE = 1000000;

    /// Thread-safe error storage for the cvariogram module.
    std::string last_cvariogram_error;
    std::mutex last_cvariogram_error_mutex;
}

/// Exported: retrieves the last error message (thread-safe, C ABI).
extern "C" const char * cvar_get_last_error(void)
{
    std::lock_guard<std::mutex> lock(last_cvariogram_error_mutex);
    thread_local std::string cached;
    cached = last_cvariogram_error;
    return cached.c_str();
}

/// Internal: stores an error message (thread-safe).
void cvar_set_last_error(const char * message)
{
    std::lock_guard<std::mutex> lock(last_cvariogram_error_mutex);
    last_cvariogram_error = message;
}

namespace {

/// Seeds rand() once at first use with the current time.
void seed_rand_once()
{
    static bool seeded = false;
    if (!seeded)
    {
        srand(static_cast<unsigned int>(time(nullptr)));
        seeded = true;
    }
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
	if (!validate_ptr(templ, "templ (is_in_tunnel)")) return false;
	if (!validate_ptr(vec, "vec (is_in_tunnel)")) return false;

	// M22: Guard against zero direction vectors — if any ellipsoid
	// direction is the zero vector, is_in_tunnel returns true for ALL
	// input vectors (degenerate case). Bail early.
	if ((templ->m_ellipsoid.m_direction1.m_data[0] == 0.0
	  && templ->m_ellipsoid.m_direction1.m_data[1] == 0.0
	  && templ->m_ellipsoid.m_direction1.m_data[2] == 0.0)
	 || (templ->m_ellipsoid.m_direction2.m_data[0] == 0.0
	  && templ->m_ellipsoid.m_direction2.m_data[1] == 0.0
	  && templ->m_ellipsoid.m_direction2.m_data[2] == 0.0)
	 || (templ->m_ellipsoid.m_direction3.m_data[0] == 0.0
	  && templ->m_ellipsoid.m_direction3.m_data[1] == 0.0
	  && templ->m_ellipsoid.m_direction3.m_data[2] == 0.0))
		return false;

	double ss1 = dot_product(vec, &(templ->m_ellipsoid.m_direction1));
	double ss2 = dot_product(vec, &(templ->m_ellipsoid.m_direction2));
	double ss3 = dot_product(vec, &(templ->m_ellipsoid.m_direction3));

	if (templ->m_ellipsoid.m_R2 == 0 || templ->m_ellipsoid.m_R3 == 0)
		return false;

	double s2 = ss2 / templ->m_ellipsoid.m_R2;
	double s3 = ss3 / templ->m_ellipsoid.m_R3;

	double dist = sqrt(s2*s2 + s3*s3);
	bool result = (dist <= 1.0) && (templ->m_tol_distance * dist <= ss1);
	return result;
}

bool is_in_tunnel_v(
		variogram_search_template_t * templ,
		vector_t * vec, bool * results, int count)
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
	if (!validate_ptr(templ, "templ (calc_search_template_window)")) return;
	if (!validate_ptr(window, "window (calc_search_template_window)")) return;

	double max = 1e10;
	double mini, maxi, minj, maxj, mink, maxk;
	mini = minj = mink = max;
	maxi = maxj = maxk = -max;
	
	for (int i = 0; i < 2; ++i)
		for (int j = -1; j < 3; j += 2)
			for (int k = -1; k < 3; k+=2)
			{
				vector_t DI = {0};
				vector_t DJ = {0};
				vector_t DK = {0};
				vector_t V = {0};

				vec_by_scalar(&templ->m_ellipsoid.m_direction1, templ->m_lag_separation, &DI);
				vec_by_scalar(&DI, templ->m_num_lags, &DI);
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

lag_point_t * calc_lag_areas(variogram_search_template_t * templ, int * points_count)
{
    if (!validate_ptr(templ, "templ (calc_lag_areas)")) return nullptr;
    if (!validate_ptr(points_count, "points_count (calc_lag_areas)")) return nullptr;

    if (templ->m_num_lags <= 0) {
        fprintf(stderr, "[HPGL ERROR] calc_lag_areas: num_lags must be positive\n");
        fflush(stderr);
        cvar_set_last_error("calc_lag_areas: num_lags must be positive");
        return nullptr;
    }

    search_template_window_t window;
    calc_search_template_window(templ, &window);
    int mini = (int) floor(window.m_min_i);
    int minj = (int) floor(window.m_min_j);
    int mink = (int) floor(window.m_min_k);

    int maxi = (int) ceil(window.m_max_i);
    int maxj = (int) ceil(window.m_max_j);
    int maxk = (int) ceil(window.m_max_k);

    lag_t * lags = (lag_t *) calloc(templ->m_num_lags, sizeof(lag_t));
    if (!lags) {
        fprintf(stderr, "[HPGL ERROR] calc_lag_areas: calloc(lags) failed\n");
        fflush(stderr);
        cvar_set_last_error("calc_lag_areas: calloc(lags) failed — out of memory");
        return nullptr;
    }
    //lag_t * first_lag = lags;
    //lag_t * last_lag = &lags[templ->m_num_lags - 1];

    init_lag_list(templ, lags, templ->m_num_lags);

    // M26: Pre-allocate with conservative upper bound (total iteration space)
    // and fill in a single pass instead of counting first then re-iterating.
    {
        size_t span_i = static_cast<size_t>(maxi - mini + 1);
        size_t span_j = static_cast<size_t>(maxj - minj + 1);
        size_t span_k = static_cast<size_t>(maxk - mink + 1);
        size_t max_points = static_cast<size_t>(templ->m_num_lags);
        bool overflow = false;

        // Overflow-safe multiplication
        if (span_i > 0 && max_points > SIZE_MAX / span_i) overflow = true;
        else max_points *= span_i;
        if (!overflow && span_j > 0 && max_points > SIZE_MAX / span_j) overflow = true;
        else if (!overflow) max_points *= span_j;
        if (!overflow && span_k > 0 && max_points > SIZE_MAX / span_k) overflow = true;
        else if (!overflow) max_points *= span_k;
        // Ensure max_points fits in int for points_count output
        if (!overflow && max_points > static_cast<size_t>(std::numeric_limits<int>::max()))
            overflow = true;

        if (overflow) {
            free(lags);
            fprintf(stderr, "[HPGL ERROR] calc_lag_areas: search template window too large\n");
            fflush(stderr);
            cvar_set_last_error("calc_lag_areas: search template window too large");
            return nullptr;
        }

        lag_point_t * result = (lag_point_t *) calloc(max_points, sizeof(lag_point_t));
        if (!result) {
            free(lags);
            fprintf(stderr, "[HPGL ERROR] calc_lag_areas: calloc(result) failed\n");
            fflush(stderr);
            cvar_set_last_error("calc_lag_areas: calloc(result) failed — out of memory");
            return nullptr;
        }

        int current_point = 0;
        vector_t vec;

        for (int lag_idx = 0; lag_idx < templ->m_num_lags; ++lag_idx)
        {
            lag_t * lag = &lags[lag_idx];
            for (int i = mini; i <= maxi; ++i)
            {
                for (int j = minj; j <= maxj; ++j)
                {
                    for (int k = mink; k <= maxk; ++k)
                    {
                        vec.m_data[0] = i;
                        vec.m_data[1] = j;
                        vec.m_data[2] = k;
                        double dist = sqrt( (double)i*i + (double)j*j + (double)k*k );
                        if (is_in_tunnel(templ, &vec)
                                && lag->m_start <= dist
                                && dist < lag->m_end)
                        {
                            // Upper-bound guard: should never trigger since
                            // max_points >= total iteration count
                            if (static_cast<size_t>(current_point) >= max_points) {
                                free(lags);
                                free(result);
                                fprintf(stderr, "[HPGL ERROR] calc_lag_areas: buffer overflow\n");
                                fflush(stderr);
                                cvar_set_last_error("calc_lag_areas: internal buffer overflow");
                                return nullptr;
                            }
                            lag_point_t * lp = &result[current_point++];
                            lp->m_coords[0] = i;
                            lp->m_coords[1] = j;
                            lp->m_coords[2] = k;
                            lp->m_lag_index = lag_idx;
                        }
                    }
                }
            }
        }

        *points_count = current_point;
        free(lags);
        return result;
    }
}


struct lag_statistics_t
{
	double m_cov_sum;
	int m_cov_count;
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

int get_offset(vector_t * vec, int * strides)
{
	int result = 0;
	for (int i = 0; i < 3; ++i)
		result += vec->m_data[i] * strides[i];
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
	if (templ->m_lag_separation == 0)
		return;
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

void calc_variograms(
		variogram_search_template_t * templ,
		hard_data_t * data,
		float * result_covariations,
		int result_length,
		int percentToUse)		
{
	if (!validate_ptr(templ, "templ (calc_variograms)")) return;
	if (!validate_ptr(data, "data (calc_variograms)")) return;
	if (!validate_ptr(result_covariations, "result_covariations (calc_variograms)")) return;

	seed_rand_once();

	int lag_count = templ->m_num_lags <= result_length 
		? templ->m_num_lags
		: result_length;

	lag_statistics_t * lag_stats = (lag_statistics_t *) calloc(lag_count, sizeof(lag_statistics_t));
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

	for (int i2 = (int)floor(window.m_min_i); i2 <= (int)ceil(window.m_max_i); ++i2)
		for (int j2 = (int)floor(window.m_min_j); j2 <= (int)ceil(window.m_max_j); ++j2)
			for (int k2 = (int)floor(window.m_min_k); k2 <= (int)ceil(window.m_max_k); ++k2)
			{
				vector_t vec;
				vec.m_data[0] = i2;
				vec.m_data[1] = j2;
				vec.m_data[2] = k2;

				int doffset = get_offset(&vec, data->m_data_strides);
				int moffset = get_offset(&vec, data->m_mask_strides);

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
									//double v1 = get_value(data, x1, y1, z1);
									int x = x1 + vec.m_data[0];
									int y = y1 + vec.m_data[1];
									int z = z1 + vec.m_data[2];
									if (is_inside(data, x,y,z) && mpx[moffset])
									{
										if ((rand() % 100) >= percentToUse)
											continue;
										double dist = calc_dist(&vec);
										double v2 = dpx[doffset];
										//double v2 = get_value(data, x, y, z);
										double var = v1 - v2;
										var = var*var;
										update_lags(templ, lag_stats, lag_count, dist, var);									
									}
								}
							}
				}
			}

	for (int i = 0; i < lag_count; ++i)
	{
		if (lag_stats[i].m_cov_count > 0)
			result_covariations[i] = lag_stats[i].m_cov_sum / lag_stats[i].m_cov_count / 2;
		else
			result_covariations[i] = 0;
	}

	free(lag_stats);
	//free(lags);
}

void calc_variograms_from_point_set(
		variogram_search_template_t * templ,
		cont_point_set_t * point_set,
		float * result_covariations,
		int result_length)
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

	int lag_count = templ->m_num_lags <= result_length 
		? templ->m_num_lags
		: result_length;

	lag_statistics_t * lag_stats = (lag_statistics_t *) calloc(lag_count, sizeof(lag_statistics_t));
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

	lag_t * lags = (lag_t*) calloc(lag_count, sizeof(lag_t));
	if (!lags) {
		free(lag_stats);
		fprintf(stderr, "[HPGL ERROR] calc_variograms_from_point_set: calloc(lags) failed\n");
		fflush(stderr);
		cvar_set_last_error("calc_variograms_from_point_set: calloc(lags) failed — out of memory");
		return;
	}
	init_lag_list(templ, lags, lag_count);

	for (int idx1 = 0; idx1 < point_set->size; ++idx1)
	{
		for (int idx2 = 0; idx2 < point_set->size; ++idx2)
		{
			vector_t vec;
			vec.m_data[0] = point_set->xs[idx1] - point_set->xs[idx2];
			vec.m_data[1] = point_set->ys[idx1] - point_set->ys[idx2];
			vec.m_data[2] = point_set->zs[idx1] - point_set->zs[idx2];
			if (is_in_tunnel(templ, &vec))
			{
				double dist = calc_dist(&vec);
				double var = pow(point_set->values[idx1] - point_set->values[idx2], 2);
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

	for (int i = 0; i < lag_count; ++i)
	{
		if (lag_stats[i].m_cov_count == 0)
			result_covariations[i] = 0;
		else
			result_covariations[i] = lag_stats[i].m_cov_sum / lag_stats[i].m_cov_count / 2;
	}

	free(lag_stats);
	free(lags);
}
		
