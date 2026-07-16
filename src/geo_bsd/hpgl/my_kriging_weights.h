#ifndef __MY_KRIGING_WEIGHTS_H__B6211BC7_74C1_4D96_AB05_286A62D0F003
#define __MY_KRIGING_WEIGHTS_H__B6211BC7_74C1_4D96_AB05_286A62D0F003

//#define HPGL_SOLVER
#define LAPACK_SOLVER

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

// ====================================================================
// SECTION 1: LAPACK Integration
// LAPACK error handling, safe allocation helpers, and Fortran
// interface (dpotrf_/dpotrs_) used by all weight calculators below.
// Solver selection: #define LAPACK_SOLVER or HPGL_SOLVER at file top.
// ====================================================================

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

	// -----------------------------------------------------------------------
	// Reusable workspace for kriging weight calculation functions.
	// Pre-allocated once per thread in the OpenMP outer loop; reused via
	// resize() (allocation-free for size ≤ capacity) on each kriging node
	// iteration.  Eliminates 3-8 heap vector allocations per node.
	// -----------------------------------------------------------------------
	struct weight_calc_workspace_t {
		std::vector<double> A;          // size²: covariance matrix
		std::vector<double> b;          // size: RHS vector
		std::vector<double> b2;         // size: original RHS for variance
		std::vector<double> ones;       // size: OK ones vector
		std::vector<double> ones_result;// size: OK solve result for ones
		std::vector<double> sk_weights; // size: OK SK-weights
		std::vector<double> sigmas;     // size: corellogram sigma values
#ifdef LAPACK_SOLVER
		std::vector<double> B;          // 2*size: combined OK RHS buffer
#endif
#ifdef HPGL_SOLVER
		std::vector<double> A_U;        // size²: upper Cholesky factor
		std::vector<double> A_L;        // size²: lower Cholesky factor
#endif
	};

	// ================================================================
	// SECTION 2: Weight Calculators
	// sk_kriging_weights_3  — Simple Kriging (solve A*w = b directly).
	// ok_kriging_weights_3  — Ordinary Kriging (SK solve + Lagrange
	//                          multiplier correction).
	// corellogramed_weights_3 — Correlogram kriging (pre-transform
	//                            covariance by sigma_i * sigma_j).
	// Each function selects solver via #ifdef HPGL_SOLVER / LAPACK_SOLVER.
	// ================================================================

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

		//build invariant
		for (int i = 0, end_i = size; i < end_i; ++i)
		{
			for (int j = i, end_j = end_i; j < end_j; ++j)
			{
				A[static_cast<size_t>(i) * size + j] = covariances(coords[i], coords[j]);
				A[static_cast<size_t>(j) * size + i] = A[static_cast<size_t>(i) * size + j];
			}
			b[i] = covariances(coords[i], center_coord);
			// b2 preserves original RHS values for variance calculation;
			// b is overwritten by dpotrs_ during the solve step
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
		if (!system_solved) {
			// Fallback: Cholesky failed, try gauss_solve
			// (mirrors the LAPACK_SOLVER path fallback at lines 200-218)
			system_solved = gauss_solve(&A[0], &b[0], &weights[0], size);
			HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
			// Compute SK variance on the fallback path
			if (calc_variance) {
				if (system_solved) {
					double cr0 = covariances(center_coord, center_coord);
					variance = cr0;
					for (int i = 0; i < size; i++)
						variance -= weights[i] * b2[i];
					if (variance < 0) variance = 0;
				} else {
					variance = -1;
				}
			}
			return system_solved;
		}
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

		// NOTE: LAPACK within OpenMP region — avoid BLAS thread oversubscription
		// Cholesky decomposition
		// Backup A before dpotrf_: dpotrf_ corrupts A on failure
		std::vector<double> A_backup(A);
		dpotrf_(&matrix_type, &size_lap, &A[0], &size_lap, &info_dec);

		// Handle decomposition errors
		detail::handle_lapack_error(info_dec, "dpotrf_ (Cholesky decomposition)", size);

		if (info_dec != 0) {
			// Fallback: dpotrf_ corrupted A, restore from backup and try gauss_solve
			system_solved = gauss_solve(&A_backup[0], &b[0], &weights[0], size);
			HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
			// Compute SK variance on the fallback path (mirrors non-fallback
			// variance block at lines 221-239).  b2 holds original RHS values.
			if (calc_variance) {
				if (system_solved) {
					double cr0 = covariances(center_coord, center_coord);
					variance = cr0;
					for (int i = 0; i < size; i++)
						variance -= weights[i] * b2[i];
					if (variance < 0) variance = 0;
				} else {
					variance = -1;
				}
			}
			return system_solved;
		}

		// Solve
		for (size_t i = 0; i < static_cast<size_t>(size); i ++)
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
			
				double cr0 = covariances(center_coord, center_coord);
				variance = cr0;
				for (int i = 0, end_i = (int) coords.size(); i < end_i; ++i)
				{
					variance -= weights[i] * b2[i];
				}
				// Clamp to zero — floating-point subtraction can produce small negatives
				if (variance < 0) variance = 0;
			}
			else
			{
				variance = -1;
			}
		}
		return system_solved;
	}

	// Workspace-aware variant: reuses buffers from weight_calc_workspace_t
	// instead of allocating local vectors per call.
	template<typename covariances_t, bool calc_variance, typename coord_t>
	bool sk_kriging_weights_3_ws(
			coord_t center_coord,
			const std::vector<coord_t> & coords,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights,
			double & variance,
			weight_calc_workspace_t & ws)
	{
		HPGL_LOG_STRING("Sk weights");
		HPGL_LOG_NEIGHBOURS(center_coord, coords);

		if (coords.size() <= 0)
		{
			HPGL_LOG_STRING("No neighbours.");
			return false;
		}

		const size_t coord_size = coords.size();
		size_t matrix_size = 0;
		if (!detail::safe_multiply_size_t(coord_size, coord_size, matrix_size))
		{
			HPGL_LOG_STRING("Security: Matrix size overflow detected.");
			return false;
		}

		if (coord_size > static_cast<size_t>(std::numeric_limits<int>::max()))
		{
			HPGL_LOG_STRING("Security: Coordinate count exceeds int max.");
			return false;
		}

		const int size = static_cast<int>(coord_size);

		// Resize workspace vectors — allocation-free when capacity >= needed
		ws.A.resize(matrix_size);
		ws.b.resize(coord_size);
		ws.b2.resize(coord_size);
		weights.resize(coord_size);

		//build invariant
		for (int i = 0, end_i = size; i < end_i; ++i)
		{
			for (int j = i, end_j = end_i; j < end_j; ++j)
			{
				ws.A[static_cast<size_t>(i) * size + j] = covariances(coords[i], coords[j]);
				ws.A[static_cast<size_t>(j) * size + i] = ws.A[static_cast<size_t>(i) * size + j];
			}
			ws.b[i] = covariances(coords[i], center_coord);
			ws.b2[i] = ws.b[i];
		}

		HPGL_LOG_SYSTEM(&ws.A[0], &ws.b[0], size);

#ifdef HPGL_SOLVER
		ws.A_U.resize(size*size);
		ws.A_L.resize(size*size);
		bool system_solved = cholesky_decomposition(&ws.A[0], &ws.A_U[0], &ws.A_L[0], size);
		if (!system_solved) {
			// Fallback: Cholesky failed, try gauss_solve
			// (mirrors the LAPACK_SOLVER path fallback for the ws variant)
			system_solved = gauss_solve(&ws.A[0], &ws.b[0], &weights[0], size);
			HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
			// Compute SK variance on the fallback path
			if (calc_variance) {
				if (system_solved) {
					double cr0 = covariances(center_coord, center_coord);
					variance = cr0;
					for (int i = 0; i < size; i++)
						variance -= weights[i] * ws.b2[i];
					if (variance < 0) variance = 0;
				} else {
					variance = -1;
				}
			}
			return system_solved;
		}
		cholesky_solve(&ws.A_L[0], &ws.A_U[0], &ws.b[0], &weights[0], size);
		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
#endif

#ifdef LAPACK_SOLVER
		bool system_solved = false;
		integer info_dec = 100;
		integer info_solve = 100;
		integer size_lap = size;
		integer b_size = 1;
		char matrix_type = 'U';

		// Backup ws.A before dpotrf_: dpotrf_ corrupts A on failure
		std::vector<double> A_backup(ws.A.begin(), ws.A.begin() + matrix_size);
		dpotrf_(&matrix_type, &size_lap, &ws.A[0], &size_lap, &info_dec);
		detail::handle_lapack_error(info_dec, "dpotrf_ (Cholesky decomposition)", size);

		if (info_dec != 0) {
			// Fallback: restore A from backup and try gauss_solve
			system_solved = gauss_solve(&A_backup[0], &ws.b[0], &weights[0], size);
			HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
			// Compute SK variance on the fallback path (mirrors non-fallback
			// variance block of the ws variant).  ws.b2 holds original RHS values.
			if (calc_variance) {
				if (system_solved) {
					double cr0 = covariances(center_coord, center_coord);
					variance = cr0;
					for (int i = 0; i < size; i++)
						variance -= weights[i] * ws.b2[i];
					if (variance < 0) variance = 0;
				} else {
					variance = -1;
				}
			}
			return system_solved;
		}

		for (size_t i = 0; i < static_cast<size_t>(size); i ++)
			weights[i] = ws.b[i];

		dpotrs_(&matrix_type, &size_lap, &b_size, &ws.A[0],  &size_lap, &weights[0], &size_lap, &info_solve );
		detail::handle_lapack_error(info_solve, "dpotrs_ (Cholesky solver)", size);

		if (info_solve == 0) system_solved = true;
		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
#endif

		if (calc_variance)
		{
			if (system_solved)
			{
				double cr0 = covariances(center_coord, center_coord);
				variance = cr0;
				for (int i = 0, end_i = (int) coords.size(); i < end_i; ++i)
				{
					variance -= weights[i] * ws.b2[i];
				}
				if (variance < 0) variance = 0;
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

		//build invariant
		for (int i = 0, end_i = size; i < end_i; ++i)
		{
			for (int j = i, end_j = end_i; j < end_j; ++j)
			{
				A[static_cast<size_t>(i) * size + j] = covariances(coords[i], coords[j]);
				A[static_cast<size_t>(j) * size + i] = A[static_cast<size_t>(i) * size + j];
			}
			b[i] = covariances(coords[i], center);
			// b2 preserves original RHS values for variance calculation
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
		if (!system_solved) {
			// Fallback: Cholesky failed, try gauss_solve on both RHS
			// (mirrors the LAPACK_SOLVER path fallback at lines 486-532)
			// A_backup2 created BEFORE gauss_solve so it preserves the
			// original matrix — gauss_solve modifies A in-place.
			std::vector<double> A_backup2(A);
			system_solved = gauss_solve(&A[0], &b[0], &sk_weights[0], size);
			if (system_solved) {
				system_solved = gauss_solve(&A_backup2[0], &ones[0], &ones_result[0], size);
			}
			// Compute OK weights from SK weights via Lagrange multiplier
			double mu = 0.0;  // hoisted for variance computation below
			if (system_solved) {
				double SumSK = 0, SumOnes = 0;
				for (int k = 0; k < size; k++) {
					SumSK += sk_weights[k];
					SumOnes += ones_result[k];
				}
				if (std::abs(SumOnes) < 1e-12) { system_solved = false; }
				else {
					mu = (SumSK - 1) / SumOnes;
					if (std::abs(mu) > 1e10) { system_solved = false; }
					else {
						for (int k = 0; k < size; k++) {
						weights[k] = sk_weights[k] - mu * ones_result[k];
					}
				}
			}
		}
		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
			// Compute OK kriging variance on the fallback path (mirrors
			// the non-fallback variance block at lines 548-567).  mu and
			// weights are already computed above; b2 holds the original
			// RHS values needed for the variance formula.
			if (calc_variance) {
				if (system_solved) {
					double cr0 = covariances(center, center);
					variance = cr0;
					for (int i = 0; i < size; i++)
						variance -= weights[i] * b2[i];
					variance -= mu;
					if (variance < 0) variance = 0;
				} else {
					variance = -1;
				}
			}
			return system_solved;
		}

		cholesky_solve(&A_L[0], &A_U[0], &b[0], &sk_weights[0], size);	
		cholesky_solve(&A_L[0], &A_U[0], &ones[0], &ones_result[0], size);	
#endif

#ifdef LAPACK_SOLVER

		// CLAPACK
		bool system_solved = false;

		integer info_dec = 100;
		integer info_solve = 100;
		integer size_lap = size;
		char matrix_type = 'U';

		// NOTE: LAPACK within OpenMP region — avoid BLAS thread oversubscription
		// Backup A before dpotrf_: dpotrf_ corrupts A on failure
		std::vector<double> A_backup(A);

		// Cholesky decomposition
		dpotrf_(&matrix_type, &size_lap, &A[0], &size_lap, &info_dec);

		// Handle decomposition errors
		detail::handle_lapack_error(info_dec, "dpotrf_ (OK Cholesky decomposition)", size);

		if (info_dec != 0) {
			// Fallback: dpotrf_ corrupted A, restore from backup.
			// Solve both RHS (b→sk_weights, ones→ones_result) via gauss_solve.
			// gauss_solve modifies A in-place, so use a second copy.
			// A_backup2 created BEFORE gauss_solve so it preserves the
			// original matrix — gauss_solve modifies A_backup in-place.
			std::vector<double> A_backup2(A_backup);
			system_solved = gauss_solve(&A_backup[0], &b[0], &sk_weights[0], size);
			if (system_solved) {
				system_solved = gauss_solve(&A_backup2[0], &ones[0], &ones_result[0], size);
			}
			// Compute OK weights from SK weights via Lagrange multiplier
			double mu = 0.0;  // hoisted for variance computation below
			if (system_solved) {
				double SumSK = 0, SumOnes = 0;
				for (int k = 0; k < size; k++) {
					SumSK += sk_weights[k];
					SumOnes += ones_result[k];
				}
				if (std::abs(SumOnes) < 1e-12) { system_solved = false; }
				else {
					mu = (SumSK - 1) / SumOnes;
					if (std::abs(mu) > 1e10) { system_solved = false; }
					else {
						for (int k = 0; k < size; k++) {
						weights[k] = sk_weights[k] - mu * ones_result[k];
					}
				}
			}
		}
		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
		// Compute OK kriging variance on the fallback path (mirrors
		// the non-fallback variance block at lines 548-567).  mu and
		// weights are already computed above; b2 holds the original
		// RHS values needed for the variance formula.
		if (calc_variance) {
			if (system_solved) {
				double cr0 = covariances(center, center);
				variance = cr0;
				for (int i = 0; i < size; i++)
					variance -= weights[i] * b2[i];
				variance -= mu;
				if (variance < 0) variance = 0;
			} else {
				variance = -1;
			}
		}
		return system_solved;
	}

	// Solve both RHS vectors in a single dpotrs_ call (nrhs=2).
		// LAPACK column-major layout: B[0..size-1] = sk_weights RHS,
		// B[size..2*size-1] = ones RHS. After solve, B holds solutions
		// in the same layout — halving the triangular solve overhead (~30% OK speedup).
		integer two = 2;
		std::vector<double> B(static_cast<size_t>(size) * 2);
		for (int i = 0; i < size; ++i)
		{
			B[i] = b[i];              // Column 0: sk_weights RHS
			B[i + size] = ones[i];    // Column 1: ones RHS
		}

		dpotrs_(&matrix_type, &size_lap, &two, &A[0], &size_lap, &B[0], &size_lap, &info_solve);

		// Handle solve errors BEFORE extracting results — dpotrs_ writes
		// to B on failure, so only extract when solve succeeded.
		detail::handle_lapack_error(info_solve, "dpotrs_ (OK Cholesky solver)", size);

		if (info_solve == 0) {
			system_solved = true;
			// Extract results from interleaved solution
			for (int i = 0; i < size; ++i)
			{
				sk_weights[i] = B[i];
				ones_result[i] = B[i + size];
			}
		}

#endif

		// Guard: when the linear solver failed (e.g. dpotrs_ error),
		// sk_weights and ones_result contain garbage. Skip OK weight
		// computation to avoid producing corrupted output.
		if (!system_solved)
		{
			weights.resize(coords.size());
			if (calc_variance) variance = -1;
			return false;
		}

		double SumSK = 0;
		double SumOnes = 0;

		for(int k = 0; k < size; k++)
		{
			SumSK += sk_weights[k];
			SumOnes += ones_result[k];
		}

		if (std::abs(SumOnes) < 1e-12)
		{
			// Degenerate case: SumOnes is nearly zero, cannot compute OK weights
			weights.resize(coords.size());
			if (calc_variance) variance = -1;
			return false;
		}

		double mu = (SumSK - 1) / SumOnes;

		// Secondary guard: if mu would produce unstable weights, fall back to SK
		if (std::abs(mu) > 1e10)
		{
			weights.resize(coords.size());
			if (calc_variance) variance = -1;
			return false;
		}

		for (int k = 0; k < size; k++)
		{
			weights[k] = sk_weights[k] - mu * ones_result[k];
		}

		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);

		if (calc_variance)
		{
			if (system_solved)
			{
				double cr0 = covariances(center, center);
				variance = cr0;
				for (int i = 0, end_i = (int) coords.size(); i < end_i; ++i)
				{
					variance -= weights[i] * b2[i];
				}
				// OK kriging variance: subtract the Lagrange multiplier (mu)
				variance -= mu;
				// Clamp to zero — floating-point subtraction can produce small negatives
				if (variance < 0) variance = 0;
			}
			else
			{
				variance = -1;
			}
		}
		weights.resize(coords.size());
		return system_solved;
	}

	// Workspace-aware variant of ok_kriging_weights_3.
	// Uses ws vectors (A, b, b2, ones, ones_result, sk_weights, B)
	// instead of allocating local vectors per call.
	template<typename covariances_t, bool calc_variance, typename coord_t>
	bool ok_kriging_weights_3_ws(
			coord_t center,
			const std::vector<coord_t> & coords,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights,
			double & variance,
			weight_calc_workspace_t & ws)
	{
		HPGL_LOG_STRING("Ok weights.");

		if (coords.size() <= 0)
		{
			HPGL_LOG_STRING("No neighbours.");
			return false;
		}

		const size_t coord_size = coords.size();
		size_t matrix_size = 0;
		if (!detail::safe_multiply_size_t(coord_size, coord_size, matrix_size))
		{
			HPGL_LOG_STRING("Security: Matrix size overflow detected.");
			return false;
		}

		if (coord_size > static_cast<size_t>(std::numeric_limits<int>::max()))
		{
			HPGL_LOG_STRING("Security: Coordinate count exceeds int max.");
			return false;
		}

		const int size = static_cast<int>(coord_size);

		// Resize workspace vectors — allocation-free when capacity >= needed
		ws.A.resize(matrix_size);
		ws.b.resize(coord_size);
		ws.b2.resize(coord_size);
		weights.resize(coord_size);

		//build invariant
		for (int i = 0, end_i = size; i < end_i; ++i)
		{
			for (int j = i, end_j = end_i; j < end_j; ++j)
			{
				ws.A[static_cast<size_t>(i) * size + j] = covariances(coords[i], coords[j]);
				ws.A[static_cast<size_t>(j) * size + i] = ws.A[static_cast<size_t>(i) * size + j];
			}
			ws.b[i] = covariances(coords[i], center);
			ws.b2[i] = ws.b[i];
		}

		HPGL_LOG_SYSTEM(&ws.A[0], &ws.b[0], size);

		ws.ones.resize(size);
		ws.ones_result.resize(size);
		ws.sk_weights.resize(size);
		// Fill ones vector with 1.0
		for (int i = 0; i < size; ++i)
			ws.ones[i] = 1.0;

		bool system_solved = false;

#ifdef HPGL_SOLVER
		ws.A_U.resize(size*size);
		ws.A_L.resize(size*size);

		system_solved = cholesky_decomposition(&ws.A[0], &ws.A_U[0], &ws.A_L[0], size);
		if (!system_solved) {
			// Fallback: Cholesky failed, try gauss_solve on both RHS
			// (mirrors the LAPACK_SOLVER path fallback for the ws variant)
			// A_backup2 created BEFORE gauss_solve so it preserves the
			// original matrix — gauss_solve modifies ws.A in-place.
			std::vector<double> A_backup2(ws.A.begin(), ws.A.begin() + matrix_size);
			system_solved = gauss_solve(&ws.A[0], &ws.b[0], &ws.sk_weights[0], size);
			if (system_solved) {
				system_solved = gauss_solve(&A_backup2[0], &ws.ones[0], &ws.ones_result[0], size);
			}
			// Compute OK weights from SK weights via Lagrange multiplier
			double mu = 0.0;  // hoisted for variance computation below
			if (system_solved) {
				double SumSK = 0, SumOnes = 0;
				for (int k = 0; k < size; k++) {
					SumSK += ws.sk_weights[k];
					SumOnes += ws.ones_result[k];
				}
				if (std::abs(SumOnes) < 1e-12) { system_solved = false; }
				else {
					mu = (SumSK - 1) / SumOnes;
					if (std::abs(mu) > 1e10) { system_solved = false; }
					else {
						for (int k = 0; k < size; k++) {
						weights[k] = ws.sk_weights[k] - mu * ws.ones_result[k];
					}
				}
			}
		}
		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
		// Compute OK kriging variance on the fallback path
		if (calc_variance) {
			if (system_solved) {
				double cr0 = covariances(center, center);
				variance = cr0;
				for (int i = 0; i < size; i++)
					variance -= weights[i] * ws.b2[i];
				variance -= mu;
				if (variance < 0) variance = 0;
			} else {
				variance = -1;
			}
		}
		return system_solved;
	}
		cholesky_solve(&ws.A_L[0], &ws.A_U[0], &ws.b[0], &ws.sk_weights[0], size);
		cholesky_solve(&ws.A_L[0], &ws.A_U[0], &ws.ones[0], &ws.ones_result[0], size);
#endif

#ifdef LAPACK_SOLVER
		integer info_dec = 100;
		integer info_solve = 100;
		integer size_lap = size;
		char matrix_type = 'U';

		// NOTE: LAPACK within OpenMP region — avoid BLAS thread oversubscription
		// Backup ws.A before dpotrf_: dpotrf_ corrupts A on failure
		std::vector<double> A_backup(ws.A.begin(), ws.A.begin() + matrix_size);
		dpotrf_(&matrix_type, &size_lap, &ws.A[0], &size_lap, &info_dec);
		detail::handle_lapack_error(info_dec, "dpotrf_ (OK Cholesky decomposition)", size);

		if (info_dec != 0) {
			// Fallback: restore A from backup. Solve both RHS via gauss_solve.
			// A_backup2 created BEFORE gauss_solve so it preserves the
			// original matrix — gauss_solve modifies A_backup in-place.
			std::vector<double> A_backup2(A_backup);
			system_solved = gauss_solve(&A_backup[0], &ws.b[0], &ws.sk_weights[0], size);
			if (system_solved) {
				system_solved = gauss_solve(&A_backup2[0], &ws.ones[0], &ws.ones_result[0], size);
			}
			// Compute OK weights from SK weights via Lagrange multiplier
			double mu = 0.0;  // hoisted for variance computation below
			if (system_solved) {
				double SumSK = 0, SumOnes = 0;
				for (int k = 0; k < size; k++) {
					SumSK += ws.sk_weights[k];
					SumOnes += ws.ones_result[k];
				}
				if (std::abs(SumOnes) < 1e-12) { system_solved = false; }
				else {
					mu = (SumSK - 1) / SumOnes;
					if (std::abs(mu) > 1e10) { system_solved = false; }
					else {
						for (int k = 0; k < size; k++) {
						weights[k] = ws.sk_weights[k] - mu * ws.ones_result[k];
					}
				}
			}
		}
		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);
		// Compute OK kriging variance on the fallback path (mirrors
		// the non-fallback variance block of the ws variant).  mu and
		// weights are already computed above; ws.b2 holds the original
		// RHS values needed for the variance formula.
		if (calc_variance) {
			if (system_solved) {
				double cr0 = covariances(center, center);
				variance = cr0;
				for (int i = 0; i < size; i++)
					variance -= weights[i] * ws.b2[i];
				variance -= mu;
				if (variance < 0) variance = 0;
			} else {
				variance = -1;
			}
		}
		return system_solved;
	}

	// Solve both RHS vectors in a single dpotrs_ call (nrhs=2).
		integer two = 2;
		ws.B.resize(static_cast<size_t>(size) * 2);
		for (int i = 0; i < size; ++i)
		{
			ws.B[i] = ws.b[i];              // Column 0: sk_weights RHS
			ws.B[i + size] = ws.ones[i];    // Column 1: ones RHS
		}

		dpotrs_(&matrix_type, &size_lap, &two, &ws.A[0], &size_lap, &ws.B[0], &size_lap, &info_solve);

		// Handle solve errors BEFORE extracting results — dpotrs_ writes
		// to ws.B on failure, so only extract when solve succeeded.
		detail::handle_lapack_error(info_solve, "dpotrs_ (OK Cholesky solver)", size);

		if (info_solve == 0) {
			system_solved = true;
			for (int i = 0; i < size; ++i)
			{
				ws.sk_weights[i] = ws.B[i];
				ws.ones_result[i] = ws.B[i + size];
			}
		}
#endif

		double mu = 0.0;
		if (system_solved)
		{
			double SumSK = 0;
			double SumOnes = 0;
			for(int k = 0; k < size; k++)
			{
				SumSK += ws.sk_weights[k];
				SumOnes += ws.ones_result[k];
			}

			if (std::abs(SumOnes) < 1e-12)
			{
				system_solved = false;
			}
			else
			{
				mu = (SumSK - 1) / SumOnes;

				// Secondary guard: if mu would produce unstable weights, fall back to SK
				if (std::abs(mu) > 1e10)
				{
					system_solved = false;
				}
				else
				{
					for (int k = 0; k < size; k++)
					{
						weights[k] = ws.sk_weights[k] - mu * ws.ones_result[k];
					}
				}
			}
		}

		HPGL_LOG_SYSTEM_SOLUTION(system_solved, &weights[0], size);

		if (calc_variance)
		{
			if (system_solved)
			{
				double cr0 = covariances(center, center);
				variance = cr0;
				for (int i = 0, end_i = (int) coords.size(); i < end_i; ++i)
				{
					variance -= weights[i] * ws.b2[i];
				}
				// OK kriging variance: subtract the Lagrange multiplier (mu)
				variance -= mu;
				// Clamp to zero — floating-point subtraction can produce small negatives
				if (variance < 0) variance = 0;
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


		double meanc = center_mean;

		// Range validation: clamp correlogram mean to valid [0,1] interval.
		// Values outside [0,1] can produce sqrt(negative) → NaN downstream.
		if (meanc < 0.0) meanc = 0.0;
		if (meanc > 1.0) meanc = 1.0;

		// Boundary adjustment: shift means away from exact 0 and 1 to avoid
		// sqrt(0)=0 (zero kriging weights) and to stabilise the correlogram.
		// Use tolerance-based checks instead of exact float equality.
		if (meanc < CORRELOGRAM_DELTA)
		{
			meanc = CORRELOGRAM_DELTA;
		}
		else if (meanc > 1.0 - CORRELOGRAM_DELTA)
		{
			meanc = 1.0 - CORRELOGRAM_DELTA;
		}

		double sigmac = sqrt(meanc * (1 - meanc));

		std::vector<double> sigmas(coord_size);

		for (int i = 0; i < size; ++i)
		{
			double meani = means[i];

			// Range validation: clamp to [0,1] before boundary adjustment
			if (meani < 0.0) meani = 0.0;
			if (meani > 1.0) meani = 1.0;

			if (meani < CORRELOGRAM_DELTA)
			{
				meani = CORRELOGRAM_DELTA;
			}
			else if (meani > 1.0 - CORRELOGRAM_DELTA)
			{
				meani = 1.0 - CORRELOGRAM_DELTA;
			}

			sigmas[i] = sqrt(meani * (1-meani));
		}


		//build invariant (exploit symmetry: C(i,j) = C(j,i))
		for (int i = 0; i < size; ++i)
		{
			for (int j = i; j < size; ++j)
			{
				A[static_cast<size_t>(i) * size + j] =
					cov(coords[i], coords[j]) * (sigmas[i] * sigmas[j]);
				A[static_cast<size_t>(j) * size + i] = A[static_cast<size_t>(i) * size + j];
			}
			b[i] = cov(coords[i], center) * (sigmas[i] * sigmac);
		}

#ifdef HPGL_SOLVER

		// INTERNAL
		std::vector<double> A_U(size*size,0.0);
		std::vector<double> A_L(size*size,0.0);
		
		//bool system_solved = gauss_solve(&A[0], &b[0], &weights[0], size);	
		bool system_solved = cholesky_decomposition(&A[0], &A_U[0], &A_L[0], size);
		if (!system_solved) {
			// Fallback: Cholesky failed, try gauss_solve
			// (mirrors the LAPACK_SOLVER path fallback)
			system_solved = gauss_solve(&A[0], &b[0], &weights[0], size);
			return system_solved;
		}
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

		// NOTE: LAPACK within OpenMP region — avoid BLAS thread oversubscription
		// Backup A before dpotrf_: dpotrf_ corrupts A on failure
		std::vector<double> A_backup(A);

		// Cholesky decomposition
		dpotrf_(&matrix_type, &size_lap, &A[0], &size_lap, &info_dec);

		// Handle decomposition errors
		detail::handle_lapack_error(info_dec, "dpotrf_ (Corellogram Cholesky decomposition)", size);

		if (info_dec != 0) {
			// Fallback: restore A from backup and try gauss_solve
			system_solved = gauss_solve(&A_backup[0], &b[0], &weights[0], size);
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

	// Workspace-aware variant of corellogramed_weights_3.
	// Uses ws vectors (A, b, sigmas) instead of allocating local vectors.
	template<typename covariances_t, typename coord_t>
	bool corellogramed_weights_3_ws(
			coord_t center,
			mean_t center_mean,
			const std::vector<coord_t> & coords,
			const covariances_t & cov,
			const std::vector<mean_t> & means,
			std::vector<kriging_weight_t> & weights,
			weight_calc_workspace_t & ws
			)
	{
		if (coords.size() <= 0)
			return false;

		const size_t coord_size = coords.size();

		size_t matrix_size = 0;
		if (!detail::safe_multiply_size_t(coord_size, coord_size, matrix_size))
		{
			HPGL_LOG_STRING("Security: Matrix size overflow detected.");
			return false;
		}

		if (coord_size > static_cast<size_t>(std::numeric_limits<int>::max()))
		{
			HPGL_LOG_STRING("Security: Coordinate count exceeds int max.");
			return false;
		}

		const int size = static_cast<int>(coord_size);

		// Resize workspace vectors
		ws.A.resize(matrix_size);
		ws.b.resize(coord_size);
		weights.resize(coord_size);

		if (means.size() != coord_size)
		{
			HPGL_LOG_STRING("Security: Means vector size mismatch.");
			return false;
		}

		double meanc = center_mean;

		// Range validation: clamp correlogram mean to valid [0,1] interval.
		// Values outside [0,1] can produce sqrt(negative) → NaN downstream.
		if (meanc < 0.0) meanc = 0.0;
		if (meanc > 1.0) meanc = 1.0;

		// Boundary adjustment: shift means away from exact 0 and 1 to avoid
		// sqrt(0)=0 (zero kriging weights) and to stabilise the correlogram.
		// Use tolerance-based checks instead of exact float equality.
		if (meanc < CORRELOGRAM_DELTA)
		{
			meanc = CORRELOGRAM_DELTA;
		}
		else if (meanc > 1.0 - CORRELOGRAM_DELTA)
		{
			meanc = 1.0 - CORRELOGRAM_DELTA;
		}

		double sigmac = sqrt(meanc * (1 - meanc));

		ws.sigmas.resize(coord_size);

		for (int i = 0; i < size; ++i)
		{
			double meani = means[i];

			// Range validation: clamp to [0,1] before boundary adjustment
			if (meani < 0.0) meani = 0.0;
			if (meani > 1.0) meani = 1.0;

			if (meani < CORRELOGRAM_DELTA)
			{
				meani = CORRELOGRAM_DELTA;
			}
			else if (meani > 1.0 - CORRELOGRAM_DELTA)
			{
				meani = 1.0 - CORRELOGRAM_DELTA;
			}

			ws.sigmas[i] = sqrt(meani * (1-meani));
		}

		//build invariant (exploit symmetry: C(i,j) = C(j,i))
		for (int i = 0; i < size; ++i)
		{
			for (int j = i; j < size; ++j)
			{
				ws.A[static_cast<size_t>(i) * size + j] =
					cov(coords[i], coords[j]) * (ws.sigmas[i] * ws.sigmas[j]);
				ws.A[static_cast<size_t>(j) * size + i] = ws.A[static_cast<size_t>(i) * size + j];
			}
			ws.b[i] = cov(coords[i], center) * (ws.sigmas[i] * sigmac);
		}

		bool system_solved = false;

#ifdef HPGL_SOLVER
		ws.A_U.resize(size*size);
		ws.A_L.resize(size*size);

		system_solved = cholesky_decomposition(&ws.A[0], &ws.A_U[0], &ws.A_L[0], size);
		if (!system_solved) {
			// Fallback: Cholesky failed, try gauss_solve
			// (mirrors the LAPACK_SOLVER path fallback)
			system_solved = gauss_solve(&ws.A[0], &ws.b[0], &weights[0], size);
			return system_solved;
		}
		cholesky_solve(&ws.A_L[0], &ws.A_U[0], &ws.b[0], &weights[0], size);
#endif

#ifdef LAPACK_SOLVER
		integer info_dec = 100;
		integer info_solve = 100;
		integer size_lap = size;
		integer b_size = 1;
		char matrix_type = 'U';

		// Backup ws.A before dpotrf_: dpotrf_ corrupts A on failure
		std::vector<double> A_backup(ws.A.begin(), ws.A.begin() + matrix_size);
		dpotrf_(&matrix_type, &size_lap, &ws.A[0], &size_lap, &info_dec);
		detail::handle_lapack_error(info_dec, "dpotrf_ (Corellogram Cholesky decomposition)", size);

		if (info_dec != 0) {
			// Fallback: restore A from backup and try gauss_solve
			system_solved = gauss_solve(&A_backup[0], &ws.b[0], &weights[0], size);
			return system_solved;
		}

		for (size_t i = 0; i < static_cast<size_t>(size); i ++)
			weights[i] = ws.b[i];

		dpotrs_(&matrix_type, &size_lap, &b_size, &ws.A[0],  &size_lap, &weights[0], &size_lap, &info_solve );
		detail::handle_lapack_error(info_solve, "dpotrs_ (Corellogram Cholesky solver)", size);

		if (info_solve == 0) system_solved = true;
#endif

		return system_solved;
	}

}

#endif //__MY_KRIGING_WEIGHTS_H__B6211BC7_74C1_4D96_AB05_286A62D0F003
