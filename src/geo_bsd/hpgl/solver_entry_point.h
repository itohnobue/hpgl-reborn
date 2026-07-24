#ifndef __SOLVER_ENTRY_POINT_H__E3A9B7F1_8C2D_4D56_AF90_126D84E5C339__
#define __SOLVER_ENTRY_POINT_H__E3A9B7F1_8C2D_4D56_AF90_126D84E5C339__

// ====================================================================
// Unified SPD Solver Entry Point
//
// Encapsulates the backup → decompose → solve → fallback pattern that
// was duplicated across 7 call sites (6 in my_kriging_weights.h + 1 in
// simple_cokriging_markI.cpp).
//
// Functions:
//   lapack_spd_solve_1rhs  — single RHS (SK, correlogram, cokriging)
//   lapack_spd_solve_2rhs  — dual RHS combined solve (OK path)
//
// Both functions:
//   - Call dpotrf_ (Cholesky decomposition) on the input matrix A.
//   - On dpotrf_ failure: fall back to gauss_solve using a backup of
//     the original matrix (A_orig).  gauss_solve performs Gaussian
//     elimination with partial pivoting and residual validation.
//   - On dpotrf_ success: call dpotrs_ (triangular solve) using the
//     Cholesky factor.  The 2rhs variant uses nrhs=2 for combined solve.
//   - Report LAPACK errors via handle_lapack_error.
//   - Return true iff the solve succeeded (Cholesky or fallback).
//
// Preserves all existing guards:
//   - gauss_solve: std::isfinite(coef) before division, residual check
//   - cholesky_decomposition (HPGL path): !std::isfinite(V) || V < epsilon
//   - dpotrf_: INFO return handled by handle_lapack_error
//
// Thread safety: no static/global mutable state.  Stack-local temporaries
// only (std::vector in 2rhs fallback path).  Callable from OpenMP regions.
// ====================================================================

#include <algorithm>
#include <cstdio>
#include <vector>

#include "lapack_compat.h"
#include "gauss_solver.h"
#include "logging.h"

namespace hpgl {
namespace detail {

// -----------------------------------------------------------------------
// LAPACK error handler with proper error codes.
//
// Moved here from my_kriging_weights.h (lines 77–96) so both
// my_kriging_weights.h and simple_cokriging_markI.cpp can share it
// without circular includes.
// -----------------------------------------------------------------------
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

// -----------------------------------------------------------------------
// Single-RHS LAPACK SPD solver.
//
// Handles: A_backup → dpotrf_(A) → (on fail) gauss_solve(A_orig, B, X)
//                                   → (on success) copy B→X + dpotrs_(A, X)
//
// Parameters:
//   A       [in/out]  Covariance matrix (size × size, row-major).
//                     On input:  original symmetric-positive-definite matrix.
//                     On output: Cholesky factor (upper triangle modified
//                                by dpotrf_ in-place).
//   size     Matrix dimension (n).
//   X       [out]     Solution vector (size doubles).  Always receives
//                     the result when the function returns true.
//   B       [in/out]  RHS vector (size doubles).
//                     On the dpotrs_ path: B is copied to X before
//                     dpotrs_ overwrites X; B itself is unmodified.
//                     On the fallback path: gauss_solve modifies
//                     B in-place during elimination.
//   A_orig  [in/out]  Backup of the original matrix A (size × size,
//                     row-major).  On the dpotrs_ path A_orig is
//                     unmodified.  On the fallback path gauss_solve
//                     modifies A_orig in-place (Gaussian elimination).
//   label    Context string for error messages (e.g. "SK Cholesky").
//
// Returns true if solve succeeded (Cholesky or gauss_solve fallback).
// When returning false, X may contain garbage — the caller must treat
// the result as invalid.
// -----------------------------------------------------------------------
inline bool lapack_spd_solve_1rhs(
	double* A, int size,
	double* X, double* B,
	double* A_orig,
	const char* label)
{
	// Pre-check: scan A for NaN/Inf values before calling dpotrf_.
	// dpotrf_ does not validate inputs and may silently produce NaN
	// results or invoke undefined behavior on non-finite inputs.
	for (int i = 0; i < size * size; ++i) {
		if (!std::isfinite(A[i])) {
			char error_msg[256];
			snprintf(error_msg, sizeof(error_msg),
				"LAPACK Error in %s: Non-finite value (NaN/Inf) in covariance matrix A[%d]. Matrix size: %d",
				label, i, size);
			HPGL_LOG_STRING(error_msg);
			return false;
		}
	}

	// Pre-check: scan RHS vector B for NaN/Inf values before solver dispatch.
	// dpotrs_ and gauss_solve do not validate RHS inputs and may silently
	// produce NaN solutions (F-24).
	for (int i = 0; i < size; ++i) {
		if (!std::isfinite(B[i])) {
			char error_msg[256];
			snprintf(error_msg, sizeof(error_msg),
				"LAPACK Error in %s: Non-finite value (NaN/Inf) in RHS vector B[%d]. Matrix size: %d",
				label, i, size);
			HPGL_LOG_STRING(error_msg);
			return false;
		}
	}

	integer info_dec = 100;
	integer info_solve = 100;
	integer size_lap = size;
	integer nrhs = 1;
	char uplo = 'U';

	// Step 1: Cholesky decomposition (modifies A in-place)
	dpotrf_(&uplo, &size_lap, A, &size_lap, &info_dec);

	// Report any LAPACK errors (info < 0 = invalid arg, info > 0 = not SPD)
	handle_lapack_error(info_dec, label, size);

	if (info_dec != 0) {
		// Step 2a (fallback): dpotrf_ failed — use Gaussian elimination
		// on the original (backed-up) matrix.  gauss_solve performs
		// partial pivoting + residual quality check.
		// Preserves: gauss_solve's std::isfinite(coef) guard,
		// residual check ||Ax - b||_inf vs sqrt(eps) * data_scale * size.
		return gauss_solve(A_orig, B, X, size);
	}

	// Step 2b (success): dpotrf_ succeeded — copy RHS to output buffer
	// then call dpotrs_ (triangular solve).  dpotrs_ works in-place;
	// X is both the input RHS and the output solution.
	for (int i = 0; i < size; ++i)
		X[i] = B[i];

	dpotrs_(&uplo, &size_lap, &nrhs, A, &size_lap, X, &size_lap, &info_solve);

	// Report any LAPACK solve errors
	handle_lapack_error(info_solve, label, size);

	return (info_solve == 0);
}

// -----------------------------------------------------------------------
// Dual-RHS LAPACK SPD solver (OK path — combined dpotrs_ with nrhs=2).
//
// Handles: A_backup → dpotrf_(A) → (on fail) gauss_solve(A_orig, B0, X0)
//                                            gauss_solve(A_orig_copy, B1, X1)
//                                   → (on success) build combined B →
//                                            dpotrs_(A, B, nrhs=2) →
//                                            extract X0, X1 from B
//
// Parameters:
//   A       [in/out]  Covariance matrix (size × size, row-major).
//                     On output: Cholesky factor (upper triangle).
//   size     Matrix dimension (n).
//   X0      [out]     Solution for first RHS (size doubles).
//   B0      [in/out]  First RHS vector (size doubles).
//                     Modified by gauss_solve on the fallback path.
//   X1      [out]     Solution for second RHS (size doubles).
//   B1      [in/out]  Second RHS vector (size doubles).
//                     Modified by gauss_solve on the fallback path.
//   A_orig  [in/out]  Backup of original matrix A (size × size).
//                     Modified by gauss_solve on the fallback path.
//   label    Context string for error messages.
//
// Returns true if both RHS solves succeeded.
//
// On the dpotrs_ path (common case): builds a combined column-major
// buffer [B0 | B1], calls dpotrs_ with nrhs=2, extracts X0, X1.
// On the fallback path: calls gauss_solve twice with separate buffers
// (preserves gauss_solve's X ≠ B contract).
//
// Note: allocates a temporary std::vector<double>(2*size) on the
// dpotrs_ path.  This allocation is O(n) and negligible compared to
// the O(n³) Cholesky decomposition.
// -----------------------------------------------------------------------
inline bool lapack_spd_solve_2rhs(
	double* A, int size,
	double* X0, double* B0,
	double* X1, double* B1,
	double* A_orig,
	const char* label)
{
	// Pre-check: scan A for NaN/Inf values before calling dpotrf_.
	// dpotrf_ does not validate inputs and may silently produce NaN
	// results or invoke undefined behavior on non-finite inputs.
	for (int i = 0; i < size * size; ++i) {
		if (!std::isfinite(A[i])) {
			char error_msg[256];
			snprintf(error_msg, sizeof(error_msg),
				"LAPACK Error in %s: Non-finite value (NaN/Inf) in covariance matrix A[%d]. Matrix size: %d",
				label, i, size);
			HPGL_LOG_STRING(error_msg);
			return false;
		}
	}

	// Pre-check: scan RHS vectors B0, B1 for NaN/Inf values before solver dispatch.
	// dpotrs_ and gauss_solve do not validate RHS inputs and may silently
	// produce NaN solutions (F-24).
	for (int i = 0; i < size; ++i) {
		if (!std::isfinite(B0[i])) {
			char error_msg[256];
			snprintf(error_msg, sizeof(error_msg),
				"LAPACK Error in %s: Non-finite value (NaN/Inf) in RHS vector B0[%d]. Matrix size: %d",
				label, i, size);
			HPGL_LOG_STRING(error_msg);
			return false;
		}
	}
	for (int i = 0; i < size; ++i) {
		if (!std::isfinite(B1[i])) {
			char error_msg[256];
			snprintf(error_msg, sizeof(error_msg),
				"LAPACK Error in %s: Non-finite value (NaN/Inf) in RHS vector B1[%d]. Matrix size: %d",
				label, i, size);
			HPGL_LOG_STRING(error_msg);
			return false;
		}
	}

	integer info_dec = 100;
	integer info_solve = 100;
	integer size_lap = size;
	integer two = 2;
	char uplo = 'U';

	// Step 1: Cholesky decomposition
	dpotrf_(&uplo, &size_lap, A, &size_lap, &info_dec);
	handle_lapack_error(info_dec, label, size);

	if (info_dec != 0) {
		// Step 2a (fallback): dpotrf_ failed → Gaussian elimination
		// Copy A_orig BEFORE first gauss_solve — the first call
		// modifies it in-place during Gaussian elimination.
		std::vector<double> A_copy(A_orig, A_orig + static_cast<size_t>(size) * size);
		bool ok = gauss_solve(A_orig, B0, X0, size);
		if (ok) {
			// Second RHS: use the preserved copy.
			ok = gauss_solve(A_copy.data(), B1, X1, size);
		}
		return ok;
	}

	// Step 2b (success): build combined column-major RHS buffer
	// [B0 | B1] and call dpotrs_ with nrhs=2.
	// dpotrs_ works in-place: the combined buffer contains
	// solutions [X0 | X1] on return.
	std::vector<double> B(static_cast<size_t>(size) * 2, 0.0);
	for (int i = 0; i < size; ++i) {
		B[i] = B0[i];              // Column 0: first RHS
		B[i + size] = B1[i];       // Column 1: second RHS
	}

	dpotrs_(&uplo, &size_lap, &two, A, &size_lap,
	        B.data(), &size_lap, &info_solve);

	// Report errors BEFORE reading results — dpotrs_ may write
	// garbage to B on failure.
	handle_lapack_error(info_solve, label, size);

	if (info_solve == 0) {
		// Extract solutions from column-major buffer
		for (int i = 0; i < size; ++i) {
			X0[i] = B[i];
			X1[i] = B[i + size];
		}
		return true;
	}

	return false;
}

} // namespace detail
} // namespace hpgl

#endif // __SOLVER_ENTRY_POINT_H__E3A9B7F1_8C2D_4D56_AF90_126D84E5C339__
