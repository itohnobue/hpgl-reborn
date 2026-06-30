#include "stdafx.h"
#include <cmath>
#include <limits>
#include "property_array.h"

namespace hpgl
{
	bool gauss_solve(double * A, double * B, double * X, int size)
	{
		HPGL_CHECK(A != nullptr && B != nullptr && X != nullptr, "gauss_solve: null pointer argument");
		HPGL_CHECK(size > 0, "gauss_solve: invalid size");

		// Save original A and B for residual quality check after solve.
		// A and B are modified in-place during elimination; the originals
		// are needed to compute ||Ax - b|| after back-substitution.
		std::vector<double> A_orig(A, A + size * size);
		std::vector<double> B_orig(B, B + size);

		std::vector<int> flags(size, 0);
		std::vector<int> order(size, 0);
		for (int i = 0; i < size; ++i)
		{
			//searching for non zero row;
			bool found = false;
			int row = -1;
			for (int j = 0; j < size; ++j)
			{
				if (flags[j] == 0 && A[j * size + i] != 0)
				{
					found = true;
					row = j;
					flags[j] = 1;
					order[i] = j;
					break;
				}
			}

			if (!found)
			{
				return false; //matrix is singular
			}

			//normalize row
			double coef = A[row * size + i];

			if (std::abs(coef) < std::numeric_limits<double>::epsilon())
			{
				return false; // Coefficient is too close to zero
			}

			for (int j = i; j < size; ++j)
			{
				A[row * size + j] /= coef;
			}
			B[row] /= coef;

			//subtract row
			for (int j = 0; j < size; ++j)
			{
				if (flags[j] == 0)
				{
					double coef =  A[j * size + i];
					for (int k = i; k < size; ++k)
					{
						A[j * size + k] -= coef * A[row * size + k];
					}
					B[j] -= coef * B[row];
				}
			}
		}

		for (int i = size-1; i >=0 ; --i)
		{
			int row = order[i];
			X[i] = B[row];
			for (int j = size-1; j >i; --j)
			{
				X[i] -= A[row * size + j] * X[j];
			}
		}

		// Residual quality check: compute ||Ax - b||_inf using the original
		// (unmodified) matrix and RHS.  On the Cholesky-failure fallback path
		// the matrix may be ill-conditioned; Gaussian elimination without
		// pivoting can silently lose significant digits.  The residual norm
		// catches degraded solutions before they propagate as kriging weights.
		{
			double max_residual = 0.0;
			double data_scale = 1.0;
			for (int i = 0; i < size; ++i) {
				double ax = 0.0;
				for (int j = 0; j < size; ++j)
					ax += A_orig[i * size + j] * X[j];
				double res = std::abs(ax - B_orig[i]);
				if (res > max_residual) max_residual = res;
				// Track magnitude of original data for relative tolerance
				for (int j = 0; j < size; ++j)
					data_scale = std::max(data_scale, std::abs(A_orig[i * size + j]));
				data_scale = std::max(data_scale, std::abs(B_orig[i]));
			}
			// sqrt(eps) * size ≈ 1.5e-8 * n: tolerance for acceptable round-off.
			// Residuals exceeding this indicate the solution lost significant
			// digits and should not be trusted.
			double tol = std::sqrt(std::numeric_limits<double>::epsilon()) * data_scale * size;
			if (max_residual > tol)
				return false;
		}

		return true;
	}

	bool cholesky_decomposition(double * A, double * A_U, double * A_L, int size)
	{
		HPGL_CHECK(A != nullptr && A_U != nullptr && A_L != nullptr, "cholesky_decomposition: null pointer argument");
		HPGL_CHECK(size > 0, "cholesky_decomposition: invalid size");

		double V = 0.0;

		// inside matrix [L(i,j)]
		for (int j = 0; j < size; j++)
		{
			for(int i = j; i < size; i++)
			{
				if(i==j)
				{
					// main diagonals [L(i,i)]
						V = A[i*size + i];
						for (int k = 0; k <= i-1; k++)
						{
							V -= (A_U[k*size + i] * A_U[k * size + i]);
						}

					if (V < std::numeric_limits<double>::epsilon())
					{
						return false;
					}

						A_L[i*size + i] = sqrt(V);
						A_U[i*size + i] = sqrt(V);
				}
				else
				{
						V = 0.0;
						for (int k = 0; k <= j-1; k++)
						{
							V += A_U[k*size + i] * A_U[k*size + j];
						}

						if( std::abs(A_U[j*size + j]) < std::numeric_limits<double>::epsilon() )
						{
							return false;
						}

						A_U[j*size + i] = (1 / A_U[j*size + j]) * (A[j*size + i] - V);
						A_L[i*size + j] = A_U[j*size + i];
				}
			}
		}
		return true;
	}

	void cholesky_solve(double * A_L, double * A_U, double * B, double * X, int size)
	{
		HPGL_CHECK(A_L != nullptr && A_U != nullptr && B != nullptr && X != nullptr, "cholesky_solve: null pointer argument");
		HPGL_CHECK(size > 0, "cholesky_solve: invalid size");

		std::vector<double> X_R(size,0.0);

		for (int i = 0; i <size ; i++)
		{
			X_R[i] = B[i];
			for (int j = 0; j <i; j++)
			{
				X_R[i] -= A_L[i * size + j] * X_R[j];
			}

			if (std::abs(A_L[i * size + i]) < std::numeric_limits<double>::epsilon())
			{
				X_R[i] = 0.0;
			}
			else
			{
				X_R[i] /= A_L[i * size + i];
			}
		}

		for (int i = size-1; i >=0 ; --i)
		{
			X[i] = X_R[i];
			for (int j = size-1; j >i; --j)
			{
				X[i] -= A_U[i * size + j] * X[j];
			}

			if (std::abs(A_U[i * size + i]) < std::numeric_limits<double>::epsilon())
			{
				X[i] = 0.0;
			}
			else
			{
				X[i] /= A_U[i * size + i];
			}
		}
    }

	bool cholesky_old(double * A, double * B, double * X, int size)
	{
		HPGL_CHECK(A != nullptr && B != nullptr && X != nullptr, "cholesky_old: null pointer argument");
		HPGL_CHECK(size > 0, "cholesky_old: invalid size");

		double V = 0.0;

		std::vector<double> A_U(size*size,0.0);
		std::vector<double> A_L(size*size,0.0);

		// inside matrix [L(i,j)]
		for (int j = 0; j < size; j++)
		{
			for(int i = j; i < size; i++)
			{
				if(i==j)
				{
					// main diagonals [L(i,i)]
					V = A[i*size + i];
						for (int k = 0; k <= i-1; k++)
						{
							V -= (A_U[k*size + i] * A_U[k * size + i]);
						}

					// isfinite guard: NaN bypasses `V < epsilon` (NaN < X is always false).
					if (!std::isfinite(V) || V < std::numeric_limits<double>::epsilon())
					{
						return false;
					}

					A_L[i*size + i] = sqrt(V);
					A_U[i*size + i] = sqrt(V);
				}
				else
				{
						V = 0.0;
						for (int k = 0; k <= j-1; k++)
						{
							V += A_U[k*size + i] * A_U[k*size + j];
						}

						if( std::abs(A_U[j*size + j]) < std::numeric_limits<double>::epsilon() )
						{
							return false;
						}

						A_U[j*size + i] = (1 / A_U[j*size + j]) * (A[j*size + i] - V);
						A_L[i*size + j] = A_U[j*size + i];
				}
			}
		}

		std::vector<double> X_R(size,0.0);

		for (int i = 0; i <size ; i++)
		{
			X_R[i] = B[i];
			for (int j = 0; j <i; j++)
			{
				X_R[i] -= A_L[i * size + j] * X_R[j];
			}

			// isfinite guard: NaN bypasses epsilon check
			double al = A_L[i * size + i];
			if (!std::isfinite(al) || std::abs(al) < std::numeric_limits<double>::epsilon())
			{
				X_R[i] = 0.0;
			}
			else
			{
				X_R[i] /= al;
			}
		}

		for (int i = size-1; i >=0 ; --i)
		{
			X[i] = X_R[i];
			for (int j = size-1; j >i; --j)
			{
				X[i] -= A_U[i * size + j] * X[j];
			}

			// isfinite guard: NaN bypasses epsilon check
			double au = A_U[i * size + i];
			if (!std::isfinite(au) || std::abs(au) < std::numeric_limits<double>::epsilon())
			{
				X[i] = 0.0;
			}
			else
			{
				X[i] /= au;
			}
		}

		return true;
    }


	}
