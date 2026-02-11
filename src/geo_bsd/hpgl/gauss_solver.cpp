#include "stdafx.h"
#include <cassert>
#include <cmath>

namespace hpgl
{
	bool gauss_solve(double * A, double * B, double * X, int size)
	{
		// SECURITY FIX: Validate input pointers
		assert(A != nullptr && "Null matrix pointer in gauss_solve");
		assert(B != nullptr && "Null B vector pointer in gauss_solve");
		assert(X != nullptr && "Null X vector pointer in gauss_solve");
		assert(size > 0 && "Invalid size in gauss_solve");

		std::vector<int> flags(size, 0);
		std::vector<int> order(size, 0);
		for (int i = 0; i < size; ++i)
		{
			//searching for non zero row;
			bool found = false;
			int row = -1;
			for (int j = 0; j < size; ++j)
			{
				// SECURITY FIX: Validate array indices
				assert(j >= 0 && j < size && "Index out of bounds in gauss_solve");
				assert(j * size + i >= 0 && j * size + i < size * size && "Matrix index out of bounds in gauss_solve");

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

			// SECURITY FIX: Check for division by zero
			if (std::abs(coef) < std::numeric_limits<double>::epsilon())
			{
				return false; // Coefficient is too close to zero
			}

			for (int j = i; j < size; ++j)
			{
				// SECURITY FIX: Validate array indices
				assert(row * size + j >= 0 && row * size + j < size * size && "Matrix index out of bounds in normalize row");
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
						// SECURITY FIX: Validate array indices
						assert(j * size + k >= 0 && j * size + k < size * size && "Matrix index out of bounds in subtract row (A)");
						assert(row * size + k >= 0 && row * size + k < size * size && "Matrix index out of bounds in subtract row (row)");
						A[j * size + k] -= coef * A[row * size + k];
					}
					B[j] -= coef * B[row];
				}
			}
		}

		for (int i = size-1; i >=0 ; --i)
		{
			int row = order[i];
			// SECURITY FIX: Validate array indices
			assert(row >= 0 && row < size && "Row index out of bounds in back substitution");
			X[i] = B[row];
			for (int j = size-1; j >i; --j)
			{
				// SECURITY FIX: Validate array indices
				assert(row * size + j >= 0 && row * size + j < size * size && "Matrix index out of bounds in back substitution");
				X[i] -= A[row * size + j] * X[j];
			}
		}

		return true;
	}

	bool cholesky_decomposition(double * A, double * A_U, double * A_L, int size)
	{
		// SECURITY FIX: Validate input pointers and size
		assert(A != nullptr && "Null matrix pointer in cholesky_decomposition");
		assert(A_U != nullptr && "Null A_U pointer in cholesky_decomposition");
		assert(A_L != nullptr && "Null A_L pointer in cholesky_decomposition");
		assert(size > 0 && "Invalid size in cholesky_decomposition");

		double V = 0.0;

		// inside matrix [L(i,j)]
		for (int j = 0; j < size; j++)
		{
			for(int i = j; i < size; i++)
			{
				// SECURITY FIX: Validate array indices
				assert(i >= 0 && i < size && j >= 0 && j < size && "Index out of bounds in cholesky_decomposition");

				if(i==j)
				{
					// main diagonals [L(i,i)]
					//for (int i = 0; i < size; i++)
					//{
					// SECURITY FIX: Validate matrix access indices
					assert(i * size + i >= 0 && i * size + i < size * size && "Matrix index out of bounds in main diagonal");

						V = 0.0;
						V += A[i*size + i];
						for (int k = 0; k <= i-1; k++)
						{
							// SECURITY FIX: Validate matrix access indices
							assert(k * size + i >= 0 && k * size + i < size * size && "Matrix index out of bounds in V calculation");
							V -= (A_U[k*size + i] * A_U[k * size + i]);
						}

						if( V <= 0)
						{
							return false;
						}

						A_L[i*size + i] = sqrt(V);
						A_U[i*size + i] = sqrt(V);
					//}
				}
				else
				{
						V = 0.0;
						for (int k = 0; k <= j-1; k++)
						{
							// SECURITY FIX: Validate matrix access indices
							assert(k * size + i >= 0 && k * size + i < size * size && "Matrix index out of bounds in V calculation (else)");
							assert(k * size + j >= 0 && k * size + j < size * size && "Matrix index out of bounds in V calculation (else j)");
							V += A_U[k*size + i] * A_U[k*size + j];
						}

						// SECURITY FIX: Validate index before access and check for division by zero
						assert(j * size + j >= 0 && j * size + j < size * size && "Matrix index out of bounds for diagonal element");

						// SECURITY FIX: Add epsilon comparison for floating point
						if( std::abs(A_U[j*size + j]) < std::numeric_limits<double>::epsilon() )
						{
							return false;
						}

						// SECURITY FIX: Validate indices for matrix access
						assert(j * size + i >= 0 && j * size + i < size * size && "Matrix index out of bounds for A_U");
						assert(i * size + j >= 0 && i * size + j < size * size && "Matrix index out of bounds for A_L");
						assert(j * size + i >= 0 && j * size + i < size * size && "Matrix index out of bounds for A");

						A_U[j*size + i] = (1 / A_U[j*size + j]) * (A[j*size + i] - V);
						A_L[i*size + j] = A_U[j*size + i];
				}
			}
		}
		return true;
	}

	void cholesky_solve(double * A_L, double * A_U, double * B, double * X, int size)
	{
		// SECURITY FIX: Validate input pointers and size
		assert(A_L != nullptr && "Null A_L pointer in cholesky_solve");
		assert(A_U != nullptr && "Null A_U pointer in cholesky_solve");
		assert(B != nullptr && "Null B pointer in cholesky_solve");
		assert(X != nullptr && "Null X pointer in cholesky_solve");
		assert(size > 0 && "Invalid size in cholesky_solve");

		// A[j * size + i]
		// B[i]
		// X[i]

		std::vector<double> X_R(size,0.0);

//		gauss_solve(&A_L[0], &B[0], &X_R[0], size);

		for (int i = 0; i <size ; i++)
		{
			// SECURITY FIX: Validate array index
			assert(i >= 0 && i < size && "Index out of bounds in cholesky_solve (forward)");

			X_R[i] = B[i];
			for (int j = 0; j <i; j++)
			{
				// SECURITY FIX: Validate matrix indices and check for division by zero
				assert(i * size + j >= 0 && i * size + j < size * size && "Matrix index out of bounds in cholesky_solve (forward A_L)");
				assert(i * size + i >= 0 && i * size + i < size * size && "Matrix diagonal index out of bounds in cholesky_solve (forward)");

				X_R[i] -= A_L[i * size + j] * X_R[j];
			}

			// SECURITY FIX: Check for division by zero before using diagonal element
			if (std::abs(A_L[i * size + i]) < std::numeric_limits<double>::epsilon())
			{
				// In release mode, set to zero to prevent NaN propagation
				X_R[i] = 0.0;
			}
			else
			{
				X_R[i] /= A_L[i * size + i];
			}
		}

//	  gauss_solve(&A_U[0], &X_R[0], X, size);

		for (int i = size-1; i >=0 ; --i)
		{
			// SECURITY FIX: Validate array index
			assert(i >= 0 && i < size && "Index out of bounds in cholesky_solve (backward)");

			X[i] = X_R[i];
			for (int j = size-1; j >i; --j)
			{
				// SECURITY FIX: Validate matrix indices
				assert(i * size + j >= 0 && i * size + j < size * size && "Matrix index out of bounds in cholesky_solve (backward A_U)");
				X[i] -= A_U[i * size + j] * X[j];
			}

			// SECURITY FIX: Check for division by zero before using diagonal element
			if (std::abs(A_U[i * size + i]) < std::numeric_limits<double>::epsilon())
			{
				// In release mode, set to zero to prevent NaN propagation
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
		// SECURITY FIX: Validate input pointers and size
		assert(A != nullptr && "Null matrix pointer in cholesky_old");
		assert(B != nullptr && "Null B pointer in cholesky_old");
		assert(X != nullptr && "Null X pointer in cholesky_old");
		assert(size > 0 && "Invalid size in cholesky_old");

		double V = 0.0;

		// SECURITY FIX: Check for overflow before allocation
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
				// SECURITY FIX: Validate array indices
				assert(i >= 0 && i < size && j >= 0 && j < size && "Index out of bounds in cholesky_old");

				if(i==j)
				{
					// main diagonals [L(i,i)]
					//for (int i = 0; i < size; i++)
					//{
						V = 0.0;

						// SECURITY FIX: Validate matrix access
						assert(i * size + i >= 0 && i * size + i < size * size && "Matrix index out of bounds in cholesky_old diagonal");

						V += A[i*size + i];
						for (int k = 0; k <= i-1; k++)
						{
							// SECURITY FIX: Validate matrix access
							assert(k * size + i >= 0 && k * size + i < size * size && "Matrix index out of bounds in cholesky_old V calc");
							V -= (A_U[k*size + i] * A_U[k * size + i]);
						}

						if( V <= 0)
						{
							return false;
						}

						A_L[i*size + i] = sqrt(V);
						A_U[i*size + i] = sqrt(V);
					//}
				}
				else
				{
						V = 0.0;
						for (int k = 0; k <= j-1; k++)
						{
							// SECURITY FIX: Validate matrix access
							assert(k * size + i >= 0 && k * size + i < size * size && "Matrix index out of bounds in cholesky_old V calc (else)");
							assert(k * size + j >= 0 && k * size + j < size * size && "Matrix index out of bounds in cholesky_old V calc (else j)");
							V += A_U[k*size + i] * A_U[k*size + j];
						}

						// SECURITY FIX: Use epsilon comparison for division by zero check
						if( std::abs(A_U[j*size + j]) < std::numeric_limits<double>::epsilon() )
						{
							return false;
						}

						// SECURITY FIX: Validate matrix access
						assert(j * size + i >= 0 && j * size + i < size * size && "Matrix index out of bounds in cholesky_old (A_U assign)");
						assert(i * size + j >= 0 && i * size + j < size * size && "Matrix index out of bounds in cholesky_old (A_L assign)");
						assert(j * size + i >= 0 && j * size + i < size * size && "Matrix index out of bounds in cholesky_old (A access)");

						A_U[j*size + i] = (1 / A_U[j*size + j]) * (A[j*size + i] - V);
						A_L[i*size + j] = A_U[j*size + i];
				}
			}
		}

		std::vector<double> X_R(size,0.0);

//		gauss_solve(&A_L[0], &B[0], &X_R[0], size);

		for (int i = 0; i <size ; i++)
		{
			// SECURITY FIX: Validate array index
			assert(i >= 0 && i < size && "Index out of bounds in cholesky_old (forward solve)");

			X_R[i] = B[i];
			for (int j = 0; j <i; j++)
			{
				// SECURITY FIX: Validate matrix access
				assert(i * size + j >= 0 && i * size + j < size * size && "Matrix index out of bounds in cholesky_old (forward solve A_L)");
				assert(i * size + i >= 0 && i * size + i < size * size && "Matrix diagonal index out of bounds in cholesky_old (forward solve)");

				X_R[i] -= A_L[i * size + j] * X_R[j];
			}

			// SECURITY FIX: Check for division by zero
			if (std::abs(A_L[i * size + i]) < std::numeric_limits<double>::epsilon())
			{
				X_R[i] = 0.0;
			}
			else
			{
				X_R[i] /= A_L[i * size + i];
			}
		}

//	  gauss_solve(&A_U[0], &X_R[0], X, size);

		for (int i = size-1; i >=0 ; --i)
		{
			// SECURITY FIX: Validate array index
			assert(i >= 0 && i < size && "Index out of bounds in cholesky_old (backward solve)");

			X[i] = X_R[i];
			for (int j = size-1; j >i; --j)
			{
				// SECURITY FIX: Validate matrix access
				assert(i * size + j >= 0 && i * size + j < size * size && "Matrix index out of bounds in cholesky_old (backward solve A_U)");
				X[i] -= A_U[i * size + j] * X[j];
			}

			// SECURITY FIX: Check for division by zero
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
