#ifndef __MY_KRIGING_WEIGHTS_H__B6211BC7_74C1_4D96_AB05_286A62D0F003
#define __MY_KRIGING_WEIGHTS_H__B6211BC7_74C1_4D96_AB05_286A62D0F003

//#define HPGL_SOLVER
#define LAPACK_SOLVER

#include <cassert>
#include <cstdio>
#include <limits>

// Use LAPACK compatibility header
// Supports Intel MKL, OpenBLAS, and CLAPACK
#include "lapack_compat.h"


#include "sugarbox_grid.h"
#include "property_array.h"
#include "typedefs.h"
#include "gauss_solver.h"
#include "logging.h"


namespace hpgl
{
	// SECURITY FIX: Safe allocation helper to prevent integer overflow
	namespace detail {
		// LAPACK error handler with proper error codes
		inline void handle_lapack_error(int info, const char* operation, int matrix_size = -1) {
			if (info == 0) return; // No error

			char error_msg[256];
			if (info < 0) {
				// Invalid argument at position -info
				snprintf(error_msg, sizeof(error_msg),
					"LAPACK Error in %s: Invalid argument at position %d. Matrix size: %d",
					operation, -info, matrix_size);
			} else {
				// Matrix is not positive definite or singular
				snprintf(error_msg, sizeof(error_msg),
					"LAPACK Error in %s: Matrix not positive definite or singular. Failed at diagonal %d. Matrix size: %d",
					operation, info, matrix_size);
			}
			HPGL_LOG_STRING(error_msg);

			// For now, log the error but don't throw exception (to maintain API compatibility)
			// Consider throwing std::runtime_error(error_msg) in future versions
		}
		inline bool safe_multiply_size_t(size_t a, size_t b, size_t& result) {
			if (a == 0 || b == 0) {
				result = 0;
				return true;
			}
			if (a > SIZE_MAX / b) {
				return false; // Overflow would occur
			}
			result = a * b;
			return true;
		}
	}

	template<typename covariances_t, bool calc_variance, typename coord_t>
	bool sk_kriging_weights_3(
			coord_t center_coord,
			const std::vector<coord_t> & coords,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights,
			double & variance)
	{
		HPGL_LOG_STRING("Sk weights");
		HPGL_LOG_NEIGHBOURS(center_coord, coords);

		// SECURITY FIX: Validate input size before processing
		if (coords.size() <= 0)
		{
			HPGL_LOG_STRING("No neighbours.");
			return false;
		}

		// SECURITY FIX: Check for integer overflow in size calculation
		const size_t coord_size = coords.size();
		size_t matrix_size = 0;
		if (!detail::safe_multiply_size_t(coord_size, coord_size, matrix_size))
		{
			HPGL_LOG_STRING("Security: Matrix size overflow detected.");
			return false;
		}

		// SECURITY FIX: Validate size fits in int for LAPACK compatibility
		if (coord_size > static_cast<size_t>(std::numeric_limits<int>::max()))
		{
			HPGL_LOG_STRING("Security: Coordinate count exceeds int max.");
			return false;
		}

		const int size = static_cast<int>(coord_size);

		// SECURITY FIX: Use pre-validated size for allocation
		std::vector<double> A(matrix_size);
		std::vector<double> b(coord_size);
		std::vector<double> b2(coord_size);
		weights.resize(coord_size);

		// SECURITY FIX: Assert vector sizes match expected allocation in debug builds
		assert(A.size() == coord_size * coord_size && "Matrix A size mismatch");
		assert(b.size() == coord_size && "Vector b size mismatch");
		assert(b2.size() == coord_size && "Vector b2 size mismatch");

		//build invariant
		for (int i = 0, end_i = size; i < end_i; ++i)
		{
			for (int j = i, end_j = end_i; j < end_j; ++j)
			{
				// SECURITY FIX: Validate indices are within bounds
				assert(i >= 0 && i < size && j >= 0 && j < size && "Index out of bounds");
				assert(i * size + j >= 0 && static_cast<size_t>(i * size + j) < matrix_size && "Matrix A index out of bounds");

				A[i*size + j] = covariances(coords[i], coords[j]);
				A[j*size + i] = A[i*size + j];
			}
			// SECURITY FIX: Validate vector index
			assert(i >= 0 && i < size && "Vector b index out of bounds");
			b[i] = covariances(coords[i], center_coord);
			b2[i] = b[i];
		}

		HPGL_LOG_SYSTEM(&A[0], &b[0], size);

#ifdef HPGL_SOLVER

		// std::cout << "HPGL SOLVER MATRIX SIZE: " << size << std::endl;

		// INTERNAL

		std::vector<double> A_U(size*size,0.0);
		std::vector<double> A_L(size*size,0.0);

		//bool system_solved = gauss_solve(&A[0], &b[0], &weights[0], size);	
		bool system_solved = cholesky_decomposition(&A[0], &A_U[0], &A_L[0], size);
		cholesky_solve(&A_L[0], &A_U[0], &b[0], &weights[0], size);

		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);

#endif

#ifdef LAPACK_SOLVER

		// std::cout << "LAPACK SOLVER MATRIX SIZE: " << size << std::endl;

		// CLAPACK
		bool system_solved = false;

		integer info_dec = 100;
		integer info_solve = 100;
		integer size_lap = size;
		integer b_size = 1;
		char matrix_type = 'U';

		// Cholesky decomposition
		dpotrf_(&matrix_type, &size_lap, &A[0], &size_lap, &info_dec);

		// Handle decomposition errors
		detail::handle_lapack_error(info_dec, "dpotrf_ (Cholesky decomposition)", size);

		if (info_dec != 0) {
			system_solved = false;
			HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
			return system_solved;
		}

		// Solve
		for (size_t i = 0; i < size; i ++)
			weights[i] = b[i];

		dpotrs_(&matrix_type, &size_lap, &b_size, &A[0],  &size_lap, &weights[0], &size_lap, &info_solve );

		// Handle solve errors
		detail::handle_lapack_error(info_solve, "dpotrs_ (Cholesky solver)", size);

		if (info_solve == 0) system_solved = true;

		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);

#endif

		//bool system_solved = cholesky_old(&A[0], &b[0], &weights[0], size);	

		if (calc_variance)
		{
			if (system_solved)
			{		
			
				sugarbox_location_t center_loc;
				double cr0 = covariances(center_coord, center_coord);				
				variance = cr0;
				for (int i = 0, end_i = (int) coords.size(); i < end_i; ++i)
				{
					variance -= weights[i] * b2[i];
				}
			}
			else
			{
				variance = -1;
			}
		}	
		return system_solved;
	}	
	
	template<typename covariances_t, bool calc_variance, typename coord_t>
	bool ok_kriging_weights_3(
			coord_t center,
			const std::vector<coord_t> & coords,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights,
			double & variance)
	{
		HPGL_LOG_STRING("Ok weights.");

		// SECURITY FIX: Validate input size before processing
		if (coords.size() <= 0)
		{
			HPGL_LOG_STRING("No neighbours.");
			return false;
		}

		// SECURITY FIX: Check for integer overflow in size calculation
		const size_t coord_size = coords.size();
		size_t matrix_size = 0;
		if (!detail::safe_multiply_size_t(coord_size, coord_size, matrix_size))
		{
			HPGL_LOG_STRING("Security: Matrix size overflow detected.");
			return false;
		}

		// SECURITY FIX: Validate size fits in int for LAPACK compatibility
		if (coord_size > static_cast<size_t>(std::numeric_limits<int>::max()))
		{
			HPGL_LOG_STRING("Security: Coordinate count exceeds int max.");
			return false;
		}

		const int size = static_cast<int>(coord_size);

		// SECURITY FIX: Use pre-validated size for allocation
		std::vector<double> A(matrix_size);
		std::vector<double> b(coord_size);
		std::vector<double> b2(coord_size);
		weights.resize(coord_size);

		// SECURITY FIX: Assert vector sizes match expected allocation in debug builds
		assert(A.size() == coord_size * coord_size && "Matrix A size mismatch");
		assert(b.size() == coord_size && "Vector b size mismatch");
		assert(b2.size() == coord_size && "Vector b2 size mismatch");

		//build invariant
		for (int i = 0, end_i = size; i < end_i; ++i)
		{
			for (int j = i, end_j = end_i; j < end_j; ++j)
			{
				// SECURITY FIX: Validate indices are within bounds
				assert(i >= 0 && i < size && j >= 0 && j < size && "Index out of bounds");
				assert(i * size + j >= 0 && static_cast<size_t>(i * size + j) < matrix_size && "Matrix A index out of bounds");

				A[i*size + j] = covariances(coords[i], coords[j]);
				A[j*size + i] = A[i*size + j];
			}
			// SECURITY FIX: Validate vector index
			assert(i >= 0 && i < size && "Vector b index out of bounds");
			b[i] = covariances(coords[i], center);
			b2[i] = b[i];
		}

		HPGL_LOG_SYSTEM(&A[0], &b[0], size);

		//bool system_solved = gauss_solve(&A[0], &b[0], &weights[0], size);		

		std::vector<double> ones(size, 1);
		
		std::vector<double> ones_result(size, 1);
		std::vector<double> sk_weights(size);

#ifdef HPGL_SOLVER

		// INTERNAL
		std::vector<double> A_U(size*size,0.0);
		std::vector<double> A_L(size*size,0.0);
		
		bool system_solved = cholesky_decomposition(&A[0], &A_U[0], &A_L[0], size);

		cholesky_solve(&A_L[0], &A_U[0], &b[0], &sk_weights[0], size);	
		cholesky_solve(&A_L[0], &A_U[0], &ones[0], &ones_result[0], size);	
#endif

#ifdef LAPACK_SOLVER

		// CLAPACK
		bool system_solved = false;

		integer info_dec = 100;
		integer info_solve = 100;
		integer size_lap = size;
		integer b_size = 1;
		char matrix_type = 'U';

		// Cholesky decomposition
		dpotrf_(&matrix_type, &size_lap, &A[0], &size_lap, &info_dec);

		// Handle decomposition errors
		detail::handle_lapack_error(info_dec, "dpotrf_ (OK Cholesky decomposition)", size);

		if (info_dec != 0) {
			system_solved = false;
			HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
			return system_solved;
		}

		// Solve
		for (size_t i = 0; i < size; i ++)
		{
			sk_weights[i] = b[i];
			ones_result[i] = ones[i];
		}

		dpotrs_(&matrix_type, &size_lap, &b_size, &A[0],  &size_lap, &sk_weights[0], &size_lap, &info_solve );
		dpotrs_(&matrix_type, &size_lap, &b_size, &A[0],  &size_lap, &ones_result[0], &size_lap, &info_solve );

		// Handle solve errors
		detail::handle_lapack_error(info_solve, "dpotrs_ (OK Cholesky solver)", size);

		if (info_solve == 0) system_solved = true;

#endif

		double SumSK = 0;
		double SumOnes = 0;

		for(int k = 0; k < size; k++)
		{
			SumSK += sk_weights[k];
			SumOnes += ones_result[k];
		}

		double mu = (SumSK - 1) / SumOnes;

		for (int k = 0; k < size; k++)
		{
			weights[k] = sk_weights[k] - mu * ones_result[k];
		}

		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);

		if (calc_variance)
		{
			if (system_solved)
			{
				sugarbox_location_t center_loc;
				double cr0 = covariances(center, center);
				variance = cr0;
				for (int i = 0, end_i = (int) coords.size(); i < end_i; ++i)
				{
					variance -= weights[i] * b2[i];
				}
				// OK kriging variance: add the Lagrange multiplier (mu)
				variance += mu;
			}
			else
			{
				variance = -1;
			}
		}
		weights.resize(coords.size());
		return system_solved;
	}
	
	template<typename covariances_t, typename coord_t>
	bool corellogramed_weights_3(
			coord_t center,
			mean_t center_mean,
			const std::vector<coord_t> & coords,
			const covariances_t & cov,
			const std::vector<mean_t> & means,
			std::vector<kriging_weight_t> & weights
			)
	{
		// SECURITY FIX: Validate input size before processing
		if (coords.size() <= 0)
			return false;

		const size_t coord_size = coords.size();

		// SECURITY FIX: Check for integer overflow in size calculation
		size_t matrix_size = 0;
		if (!detail::safe_multiply_size_t(coord_size, coord_size, matrix_size))
		{
			HPGL_LOG_STRING("Security: Matrix size overflow detected.");
			return false;
		}

		// SECURITY FIX: Validate size fits in int for LAPACK compatibility
		if (coord_size > static_cast<size_t>(std::numeric_limits<int>::max()))
		{
			HPGL_LOG_STRING("Security: Coordinate count exceeds int max.");
			return false;
		}

		const int size = static_cast<int>(coord_size);

		std::vector<double> A(matrix_size);
		std::vector<double> b(coord_size);
		weights.resize(coord_size);

		// SECURITY FIX: Validate means vector size matches coords size
		if (means.size() != coord_size)
		{
			HPGL_LOG_STRING("Security: Means vector size mismatch.");
			return false;
		}

		// SECURITY FIX: Assert vector sizes in debug builds
		assert(A.size() == coord_size * coord_size && "Matrix A size mismatch");
		assert(b.size() == coord_size && "Vector b size mismatch");

		double meanc = center_mean;
		double delta = 0.00001;

		if(meanc == 0)
		{
			meanc += delta;
		}
		if(meanc == 1)
		{
			meanc -= delta;
		}

		double sigmac = sqrt(meanc * (1 - meanc));

		std::vector<double> sigmas(coord_size);

		for (int i = 0; i < size; ++i)
		{
			// SECURITY FIX: Validate array access with bounds check
			assert(i >= 0 && i < size && "sigmas index out of bounds");

			double meani = means[i];

			if(meani == 0)
			{
				meani += delta;
			}
			if(meani == 1)
			{
				meani -= delta;
			}

			sigmas[i] = sqrt(meani * (1-meani));
		}


		//build invariant
		for (int i = 0; i < size; ++i)
		{
			for (int j = 0; j < size; ++j)
			{
				// SECURITY FIX: Validate matrix indices are within bounds
				assert(i >= 0 && i < size && j >= 0 && j < size && "Matrix index out of bounds");
				assert(i * size + j >= 0 && static_cast<size_t>(i * size + j) < matrix_size && "Matrix A index out of bounds");

				A[i * size + j] =
					cov(coords[i], coords[j]) * (sigmas[i] * sigmas[j]);
			}
			b[i] = cov(coords[i], center) * (sigmas[i] * sigmac);
		}

#ifdef HPGL_SOLVER

		// INTERNAL
		std::vector<double> A_U(size*size,0.0);
		std::vector<double> A_L(size*size,0.0);
		
		//bool system_solved = gauss_solve(&A[0], &b[0], &weights[0], size);	
		bool system_solved = cholesky_decomposition(&A[0], &A_U[0], &A_L[0], size);
		cholesky_solve(&A_L[0], &A_U[0], &b[0], &weights[0], size);

#endif

#ifdef LAPACK_SOLVER

		// CLAPACK
		bool system_solved = false;

		integer info_dec = 100;
		integer info_solve = 100;
		integer size_lap = size;
		integer b_size = 1;
		char matrix_type = 'U';

		// Cholesky decomposition
		dpotrf_(&matrix_type, &size_lap, &A[0], &size_lap, &info_dec);

		// Handle decomposition errors
		detail::handle_lapack_error(info_dec, "dpotrf_ (Corellogram Cholesky decomposition)", size);

		if (info_dec != 0) {
			system_solved = false;
			return system_solved;
		}

		// Solve
		for (size_t i = 0; i < size; i ++)
			weights[i] = b[i];

		dpotrs_(&matrix_type, &size_lap, &b_size, &A[0],  &size_lap, &weights[0], &size_lap, &info_solve );

		// Handle solve errors
		detail::handle_lapack_error(info_solve, "dpotrs_ (Corellogram Cholesky solver)", size);

		if (info_solve == 0) system_solved = true;

#endif

/*
		for (int i = 0; i < size; ++i)
		{
			weights[i] *= sigmac / sigmas[i];
		}
*/
		return system_solved;
	}

}

#endif //__MY_KRIGING_WEIGHTS_H__B6211BC7_74C1_4D96_AB05_286A62D0F003
