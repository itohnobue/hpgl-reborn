#include <exception>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include "api.h"

double dot_row_col(double * m1, double * m2, int row, int col)
{
	double result = 0.0;
	for (int i = 0; i < 3; ++i)
	{
		result += m1[row * 3 + i] * m2[i * 3 + col];
	}
	return result;
}

void multiple_matrices(double * m1, double * m2, double * m3)
{
	double result[9];
	for (int row = 0; row < 3; ++row)
	{
		for (int col = 0; col < 3; ++col)
		{
			result[row * 3 + col] = dot_row_col(m1, m2, row, col);
		}
	}
	memcpy(m3, result, sizeof(result));
}

void cpycol(double * m, vector_t * dest, int col)
{
	if (col < 0 || col >= 3)
	{
		cvar_set_last_error("cpycol: column index out of range [0, 2]");
		fprintf(stderr, "[HPGL ERROR] cpycol: column index %d out of range [0, 2]\n", col);
		fflush(stderr);
		return;
	}
	for (int i = 0; i < 3; ++i)
	{
		dest->m_data[i] = m[i*3 + col];
	}
}

void fill_ellipsoid_directions(ellipsoid_t * ell, double azimuth, double dip, double rotation)
{
    try
    {
	if (ell == nullptr)
	{
		cvar_set_last_error("fill_ellipsoid_directions: ell is null");
		fprintf(stderr, "[HPGL ERROR] fill_ellipsoid_directions: ell is null\n");
		fflush(stderr);
		return;
	}

	// F-054: full-precision pi, matching this repo's own convention
	// (hpgl/cov_model.h:29). The previous 3.14159265358 carried a relative
	// error of 3.12e-12, which at a 45-degree azimuth made
	// cos(raz) - sin(raz) = +3.46e-12: the is_in_tunnel cone test then
	// resolved exact-bisector ties by axis, so the C++ grid and point-set
	// kernels binned a symmetric template's axis-0 offsets and dropped the
	// mirrored axis-1 offsets, diverging from the Python twin
	// (variogram.py, numpy.radians).
	double coef = 3.141592653589793238462643383279502884 / 180.0;
	double razimuth = azimuth * coef;
	double rdip = dip * coef;
	double rrot = rotation * coef;

	double A[9] = {
		cos(razimuth), -sin(razimuth), 0,
		sin(razimuth), cos(razimuth), 0,
		0, 0, 1
	};

	double B[9] = {
		cos(rdip), 0, -sin(rdip),
		0, 1, 0,
		sin(rdip), 0, cos(rdip)
	};

	double C[9] = {
		1, 0, 0,
		0, cos(rrot), -sin(rrot),
		0, sin(rrot), cos(rrot)
	};

	double M[9] = {0};
	multiple_matrices(A, B, M);
	multiple_matrices(M, C, M);

	cpycol(M, &(ell->m_direction1), 0);
	cpycol(M, &(ell->m_direction2), 1);
	cpycol(M, &(ell->m_direction3), 2);
    }
    catch (const std::exception & ex)
    {
        cvar_set_last_error(ex.what());
    }
}

