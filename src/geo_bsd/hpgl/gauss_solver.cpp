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
						V = 0.0;
						V += A[i*size + i];
						for (int k = 0; k <= i-1; k++)
						{
							V -= (A_U[k*size + i] * A_U[k * size + i]);
						}

						if( V <= 0)
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

		if (size > 0 && static_cast<size_t>(size) > SIZE_MAX / static_cast<size_t>(size))
		{
			return false;
		}

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
						V = 0.0;
						V += A[i*size + i];
						for (int k = 0; k <= i-1; k++)
						{
							V -= (A_U[k*size + i] * A_U[k * size + i]);
						}

						if( V <= 0)
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

		return true;
    }


	}
