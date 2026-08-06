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
#include <cmath>
#include <cstdio>
#include <limits>
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
//   target_variance  (default 0.0) Kriging TARGET variance for the
//            Schur-consistency reference.  When > 0.0 the consistency
//            check uses it instead of A_orig[0]; when 0.0 (default)
//            the A_orig[0] behavior is preserved (SK/cokriging).
//            R4-01: the σ-scaled correlogram path builds
//            A[i][j] = C(h_ij)·σ_i·σ_j, so A[0][0] = C(0)·σ₀² is the
//            FIRST datum's variance, not the target variance C(0)·σc²;
//            the correlogram callers pass C(0)·σc² to keep valid
//            estimators (σc > σ0) from being spuriously rejected.
//            Forwarded to the gauss_solve fallback (gate ON for 1rhs)
//            so both 1rhs sites share the same reference.
//
// Returns true if solve succeeded (Cholesky or gauss_solve fallback).
// When returning false, X may contain garbage — the caller must treat
// the result as invalid.
// -----------------------------------------------------------------------
inline bool lapack_spd_solve_1rhs(
	double* A, int size,
	double* X, double* B,
	double* A_orig,
	const char* label,
	double target_variance = 0.0)
{
	// Pre-check: scan A for NaN/Inf values before calling dpotrf_.
	// dpotrf_ does not validate inputs and may silently produce NaN
	// results or invoke undefined behavior on non-finite inputs.
	// I2-23: size*size must be computed in size_t — the signed-int
	// product overflows for size > 46340, wraps negative, and silently
	// skips the NaN scan.
	const size_t n_elements = static_cast<size_t>(size) * static_cast<size_t>(size);
	for (size_t i = 0; i < n_elements; ++i) {
		if (!std::isfinite(A[i])) {
			char error_msg[256];
			snprintf(error_msg, sizeof(error_msg),
				"LAPACK Error in %s: Non-finite value (NaN/Inf) in covariance matrix A[%zu]. Matrix size: %d",
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
		// R4-01: forward the caller's target variance (default 0.0 → the
		// gauss fallback keeps its A_orig[0] reference) so both 1rhs sites
		// use the SAME consistency reference — the σ-scaled correlogram
		// path's C(0)·σc² reaches the fallback gate too.  The magnitude
		// gate stays ON for the 1rhs fallback.
		return gauss_solve(A_orig, B, X, size, true, target_variance);
	}

	// Step 2b (success): dpotrf_ succeeded — copy RHS to output buffer
	// then call dpotrs_ (triangular solve).  dpotrs_ works in-place;
	// X is both the input RHS and the output solution.
	for (int i = 0; i < size; ++i)
		X[i] = B[i];

	dpotrs_(&uplo, &size_lap, &nrhs, A, &size_lap, X, &size_lap, &info_solve);

	// Report any LAPACK solve errors
	handle_lapack_error(info_solve, label, size);

	if (info_solve != 0)
		return false;

	// II-09: solution-quality validation on the dpotrs_ success path.
	// dpotrs_ returns INFO=0 even for near-singular-but-SPD systems, which
	// can yield huge/garbage weights (compiled LAPACK probe: near-singular
	// SPD A → INFO=0, weights [1.0e12, -1.0e12] reported as success) →
	// wild estimates reported as KI_SUCCESS by every 1rhs caller
	// (SK, correlogram, cokriging). Two guards, mirroring existing
	// conventions:
	//   - magnitude: |X|_inf against a dynamic-range-scaled 1e3-family
	//     bound shared with the gauss fallback (gauss_solver.cpp) and the
	//     2rhs OK gate.
	//     R-19: the previous 1e10 bound (introduced by fix 828443e)
	//     accepted the real wild class ~1e4–1e5 and split path-dependently
	//     from the fallback's 1e3 — the same 1rhs caller got different
	//     verdicts depending on whether dpotrf_ succeeded.
	//     R2-01: the pass-2 scale-normalized AND-form
	//     (|X|_inf > 1e3 && |X|_inf·(max|A_orig|/max|B|) > 1e3) then
	//     over-rejected legitimate cross-scale cokriging: for the 2×2
	//     A=[[1e4,0.5],[0.5,1e-4]] with scale_a/scale_b ≈ 2e4 the
	//     normalized bound 1e3·scale_b/scale_a ≈ 0.05 is FAR below the
	//     legal secondary weight 6666.7 = O(ρ·σp/σs) — the normalization
	//     made the bound TIGHTER for cross-scale systems (wrong
	//     direction).  The bound is now 1e3 · dynamic_range with
	//     dynamic_range = sqrt(max|A_orig| / min|A_orig|_nz) ∈ [1, 1e12]:
	//     a wider internal scale range legitimately tolerates larger
	//     weights.  Accepted cokriging range: σp/σs up to ~1e4-1e6 →
	//     bound 1e7-1e9 ≫ O(ρ·σp/σs); the 2×2 example passes (dr = 1e4
	//     → bound 1e7 → 6666.7 ≤ 1e7).  Still rejected: unit-scale
	//     near-null wild weights (E-H1 bypass, |X| ~ 1e4-1e5, dr ~ 1 →
	//     bound 1e3) and the wild 1e12 probe (dr ≤ 1e6 → bound ≤ 1e9).
	//     All-zero A_orig → degenerate → wild.
	//     R3-02: the pre-filter is now AND-ed with a kriging-variance
	//     (Schur) consistency check (B'X > A_orig[0]·(1 + 1e-8·size),
	//     computed below) — same formula as gauss_solve's internal gate
	//     (gauss_solver.cpp).  Rationale: the (1−ρ²)⁻¹ correlation
	//     amplification makes the legal high-ρ cokriging corner
	//     (ρ ≳ 0.9995) arithmetically indistinguishable from the pinned
	//     wild class by any |X|-magnitude bound, so the separation must
	//     come from the consistency of B with A (kriging variance
	//     σK² = c − B'X ≥ 0).  Every ρ > 1/√2 ≈ 0.7071 corner has
	//     B'X = ρ²σp²/(1−ρ²) > c → negative variance → invalid estimator
	//     → rejected; the pinned wild class (B = [1,−1] → B'X = 2e12
	//     > c = 1) is rejected too.  Fires only when the pre-filter
	//     already fired (artificial-RHS pinned tests unaffected).
	//   - residual: ||A_orig·X − B||_inf vs sqrt(eps)·data_scale·size,
	//     identical to gauss_solve's residual check (gauss_solver.cpp:96-121).
	// A_orig holds the original matrix (unmodified on the dpotrs_ path),
	// so the residual is computable after the in-place Cholesky solve.
	{
		double max_abs_weight = 0.0;
		double scale_a_max = 0.0;
		double scale_a_min = 0.0; // min over NON-ZERO |A_orig| entries
		for (int i = 0; i < size; ++i)
		{
			max_abs_weight = std::max(max_abs_weight, std::abs(X[i]));
			for (int j = 0; j < size; ++j)
			{
				const double abs_a = std::abs(A_orig[static_cast<size_t>(i) * size + j]);
				scale_a_max = std::max(scale_a_max, abs_a);
				if (abs_a > 0.0)
					scale_a_min = (scale_a_min == 0.0) ? abs_a : std::min(scale_a_min, abs_a);
			}
		}
		// R2-01: dynamic-range-scaled bound.  The previous AND-form
		// (|X|_inf > 1e3 && |X|_inf·(scale_a/scale_b) > 1e3) made the
		// bound TIGHTER for legitimate cross-scale systems: for the
		// cokriging 2×2 (A=[[1e4,0.5],[0.5,1e-4]], X=[−0.333, 6666.7])
		// scale_a/scale_b ≈ 2e4 → relative 1.33e8 > 1e3 → rejected,
		// while v2.0.3's 1e10 bound accepted it.  The bound is now
		// 1e3 · dynamic_range with dynamic_range = sqrt(scale_a_max /
		// scale_a_min) ∈ [1.0, 1e12] (min over NON-ZERO entries: an
		// exact-zero covariance entry is legitimate and must not drive
		// the range to infinity): a system with a wider internal scale
		// range legitimately tolerates larger secondary weights
		// O(ρ·σp/σs).  Accepted cokriging range: σp/σs up to ~1e4-1e6
		// → bound 1e7-1e9 ≫ O(ρ·σp/σs).  Still rejected: unit-scale
		// near-null wild weights (E-H1 bypass, |X| ~ 1e4-1e5,
		// dynamic_range ~ 1 → bound 1e3) and the wild 1e12 probe
		// (dynamic_range ≤ 1e6 → bound ≤ 1e9 < 1e12).  All entries of
		// A_orig zero (scale_a_min == 0) → degenerate system → wild.
		// R3-02: the pre-filter is now AND-ed with a kriging-variance
		// (Schur) consistency check — the pre-filter alone cannot
		// separate the legal high-ρ cokriging corner from the pinned
		// wild class (both are A ≈ [[1,1−ε],[1−ε,1]] with |X| ~
		// 5e11-1e12; the (1−ρ²)⁻¹ amplification is invisible to
		// A-internal dynamic range).  The separation is in the
		// CONSISTENCY of B with A: the kriging variance
		// σK² = c − B'X (c = A_orig[0] = target variance = sill for
		// SK/correlogram, σp² for cokriging) must be non-negative for a
		// realizable estimator.  Flag wild only when the pre-filter
		// fires AND B'X > c·(1 + 1e-8·size).  For the mark-II cokriging
		// corner B'X = ρ²σp²/(1−ρ²) > c = σp² ⇔ ρ > 1/√2 ≈ 0.7071 —
		// every ρ ≳ 0.9995 corner has NEGATIVE kriging variance
		// (invalid cross-covariance model) and is rejected
		// ratio-independently; the pinned wild class (B = [1,−1] on
		// A = [[1,1−ε],[1−ε,1]] → B'X = 2e12 > c = 1) is rejected too.
		// The second condition fires ONLY when the pre-filter already
		// fired, so the artificial-RHS pinned tests (|X| ≤ 2 → pre-filter
		// no-fire) are unaffected.  B is the ORIGINAL RHS here (unmodified
		// on the dpotrs_ path — copied to X before the solve) and A_orig
		// is the unmodified original matrix; same formula as gauss_solve's
		// internal gate (gauss_solver.cpp).
		bool wild = false;
		if (!std::isfinite(max_abs_weight))
			wild = true;
		else if (scale_a_min == 0.0)
			wild = true;
		else
		{
			const double dynamic_range =
				std::min(std::max(std::sqrt(scale_a_max / scale_a_min), 1.0), 1e12);
			if (max_abs_weight > 1e3 * dynamic_range)
			{
				double bx = 0.0;
				for (int i = 0; i < size; ++i)
					bx += B[i] * X[i];
				// R4-01: the reference is the kriging TARGET variance.
				// A_orig[0] is correct for SK (C(0)·sill) and cokriging
				// (σp²) — but the σ-scaled correlogram path builds
				// A[i][j] = C(h_ij)·σ_i·σ_j, so A[0][0] = C(0)·σ₀² is the
				// FIRST datum's variance, and valid estimators with
				// σc > σ0 were spuriously rejected.  The correlogram
				// callers pass target_variance = C(0)·σc² (the true target
				// variance, sill-scaled); default 0.0 keeps the A_orig[0]
				// behavior for SK/cokriging.  Identical formula to
				// gauss_solve's internal gate (gauss_solver.cpp).
				const double reference_variance =
					(target_variance > 0.0) ? target_variance : A_orig[0];
				if (bx > reference_variance * (1.0 + 1e-8 * size))
					wild = true;
			}
		}
		if (wild)
		{
			char error_msg[256];
			snprintf(error_msg, sizeof(error_msg),
				"LAPACK Error in %s: solution magnitude |X|_inf = %.3g exceeds dynamic stability bound 1e3·sqrt(max|A|/min|A|) (near-singular system). Matrix size: %d",
				label, max_abs_weight, size);
			HPGL_LOG_STRING(error_msg);
			return false;
		}

		double max_residual = 0.0;
		double data_scale = 1.0;
		for (int i = 0; i < size; ++i)
		{
			double ax = 0.0;
			for (int j = 0; j < size; ++j)
				ax += A_orig[static_cast<size_t>(i) * size + j] * X[j];
			double res = std::abs(ax - B[i]);
			if (res > max_residual) max_residual = res;
			for (int j = 0; j < size; ++j)
				data_scale = std::max(data_scale, std::abs(A_orig[static_cast<size_t>(i) * size + j]));
			data_scale = std::max(data_scale, std::abs(B[i]));
		}
		// sqrt(eps) * size ≈ 1.5e-8 * n: tolerance for acceptable round-off,
		// matching gauss_solve's residual gate (gauss_solver.cpp:115-118).
		double tol = std::sqrt(std::numeric_limits<double>::epsilon()) * data_scale * size;
		if (max_residual > tol)
		{
			char error_msg[256];
			snprintf(error_msg, sizeof(error_msg),
				"LAPACK Error in %s: solution residual ||Ax-b||_inf = %.3g exceeds tolerance %.3g (unreliable solution). Matrix size: %d",
				label, max_residual, tol, size);
			HPGL_LOG_STRING(error_msg);
			return false;
		}
	}

	return true;
}

// -----------------------------------------------------------------------
// OK final-weight magnitude check (R-01 correction of E-H1/E-M87).
//
// The OK path solves the dual-RHS system A·[X0|X1] = [B0|1] and combines
// the solutions into the final weights w = X0 − mu·X1 with
// mu = (ΣX0 − 1)/ΣX1 (my_kriging_weights.h ok_kriging_weights_3/_3_ws).
// The Stage-6 gate measured max(|X0|_inf, |X1|_inf) against an absolute
// 1e3 bound — but X1 = A⁻¹·1 ∝ 1/sill for the OK system (A = sill·M), so
// legal small-sill models (sill ≲ 3e-4, sanctioned down to MIN_SILL =
// 1e-6) were rejected at every node even though the FINAL weights are
// scale-invariant (proven bit-identical across sills: max|Δw| = 3.2e-16).
// Measuring the final combination is scale-invariant by construction and
// preserves the E-H1 bypass catch: in the sign-alternating near-null
// bypass X0 is wild and mu = O(1), so w = X0 − mu·X1 stays wild.
//
// Mirrors the OK callers' own guards: |ΣX1| < 1e-12 and |mu| > 1e10 make
// the callers fail their weight computation (my_kriging_weights.h), so
// this check fails identically here — the downstream outcome
// (KI_SINGULARITY) is unchanged, but the solver reports the invalid
// solution instead of letting garbage flow to the caller's guards.
//
// Returns false when the combination is undefined (guards above) or the
// final weight magnitude is non-finite; on success sets max_abs_weight
// to max_i |X0[i] − mu·X1[i]| (the caller then compares it to the bound).
// -----------------------------------------------------------------------
inline bool ok_final_weight_magnitude(
	const double* X0, const double* X1, int size, double& max_abs_weight)
{
	double SumSK = 0.0;
	double SumOnes = 0.0;
	for (int i = 0; i < size; ++i)
	{
		SumSK += X0[i];
		SumOnes += X1[i];
	}
	if (std::abs(SumOnes) < 1e-12)
		return false;
	const double mu = (SumSK - 1.0) / SumOnes;
	if (!std::isfinite(mu) || std::abs(mu) > 1e10)
		return false;
	max_abs_weight = 0.0;
	for (int i = 0; i < size; ++i)
		max_abs_weight = std::max(max_abs_weight, std::abs(X0[i] - mu * X1[i]));
	return std::isfinite(max_abs_weight);
}

// -----------------------------------------------------------------------
// Dual-RHS LAPACK SPD solver (OK path — combined dpotrs_ with nrhs=2).
//
// Handles: A_backup → dpotrf_(A) → (on fail) gauss_solve(A_orig, B0, X0, false)
//                                            gauss_solve(A_orig_copy, B1, X1, false)
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
// Note: the dpotrs_ path needs a combined column-major RHS buffer of
// 2*size doubles.  E-M82: the caller may pass a reusable workspace
// buffer (weight_calc_workspace_t::B) via the `work` parameter so the
// buffer is not heap-allocated on every call (per-node allocation
// churn inside OpenMP kriging loops); the supplied buffer is resized
// (allocation-free when capacity suffices) and fully overwritten on
// each call.  When `work` is nullptr a function-local buffer is used.
// -----------------------------------------------------------------------
inline bool lapack_spd_solve_2rhs(
	double* A, int size,
	double* X0, double* B0,
	double* X1, double* B1,
	double* A_orig,
	const char* label,
	std::vector<double>* work = nullptr)
{
	// Pre-check: scan A for NaN/Inf values before calling dpotrf_.
	// dpotrf_ does not validate inputs and may silently produce NaN
	// results or invoke undefined behavior on non-finite inputs.
	// I2-23: size*size must be computed in size_t — the signed-int
	// product overflows for size > 46340, wraps negative, and silently
	// skips the NaN scan.
	const size_t n_elements = static_cast<size_t>(size) * static_cast<size_t>(size);
	for (size_t i = 0; i < n_elements; ++i) {
		if (!std::isfinite(A[i])) {
			char error_msg[256];
			snprintf(error_msg, sizeof(error_msg),
				"LAPACK Error in %s: Non-finite value (NaN/Inf) in covariance matrix A[%zu]. Matrix size: %d",
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
		// R3-01: both solves run with apply_magnitude_gate = false — the
		// internal raw-solve magnitude gate is BYPASSED on this path
		// (X1 = A⁻¹·1 ∝ 1/sill is legitimately large for legal small-sill
		// OK models, sill ≲ 3e-4, so any raw |X1| bound would reject them);
		// the final-weight gate below is the authoritative 2rhs magnitude
		// check.  gauss_solve's isfinite / singular-pivot / residual checks
		// still run unconditionally.
		std::vector<double> A_copy(A_orig, A_orig + static_cast<size_t>(size) * size);
		bool ok = gauss_solve(A_orig, B0, X0, size, false);
		if (ok) {
			// Second RHS: use the preserved copy.
			ok = gauss_solve(A_copy.data(), B1, X1, size, false);
		}
		if (!ok)
			return false;
		// R-01: same scale-invariant final-weight gate as the dpotrs_
		// success path below.  gauss_solve's internal magnitude gate
		// cannot judge the OK combination — X1 = A⁻¹·1 ∝ 1/sill is
		// legitimately large on small-sill models (the internal gate is
		// scale-aware for exactly this reason), and the E-H1 bypass
		// algebra (X0 wild, mu = O(1)) is only visible in the final
		// combination w = X0 − mu·X1.  B0/B1 were modified in place by
		// gauss_solve, but the gate needs only the solutions X0/X1.
		// R3-01: the two gauss_solve calls above run with
		// apply_magnitude_gate = false (see the Step 2a comment) — THIS
		// final-weight gate is the authoritative 2rhs magnitude check and
		// is verified to catch every wild variant (E-H1 bypass |w| = 1e4
		// > 1e3, E-M88 non-cancelling |mu| > 1e10 → undefined) while
		// accepting the small-sill OK class (sill=1e-4 → w = [0.5,0.5],
		// |w| = 0.5 ≤ 1e3).  X0/X1 availability is guaranteed here: both
		// gauss_solve calls completed successfully before this block.
		{
			double max_abs_weight = 0.0;
			bool combination_defined = ok_final_weight_magnitude(X0, X1, size, max_abs_weight);
			if (!combination_defined)
			{
				char error_msg[256];
				snprintf(error_msg, sizeof(error_msg),
					"LAPACK Error in %s: OK weight combination undefined (|SumX1| < 1e-12 or |mu| > 1e10). Matrix size: %d",
					label, size);
				HPGL_LOG_STRING(error_msg);
				return false;
			}
			if (max_abs_weight > 1e3)
			{
				char error_msg[256];
				snprintf(error_msg, sizeof(error_msg),
					"LAPACK Error in %s: solution magnitude |w|_inf = %.3g exceeds stability bound 1e3 (near-singular system). Matrix size: %d",
					label, max_abs_weight, size);
				HPGL_LOG_STRING(error_msg);
				return false;
			}
		}
		return true;
	}

	// Step 2b (success): build combined column-major RHS buffer
	// [B0 | B1] and call dpotrs_ with nrhs=2.
	// dpotrs_ works in-place: the combined buffer contains
	// solutions [X0 | X1] on return.
	// E-M82: reuse the caller-provided workspace buffer when supplied
	// (allocation-free resize when capacity suffices); otherwise fall
	// back to a function-local buffer.  Every element of B is written
	// below before dpotrs_, so no zero-initialization is required.
	std::vector<double> local_B;
	double* B = nullptr;
	if (work != nullptr) {
		work->resize(static_cast<size_t>(size) * 2);
		B = work->data();
	} else {
		local_B.resize(static_cast<size_t>(size) * 2);
		B = local_B.data();
	}
	for (int i = 0; i < size; ++i) {
		B[i] = B0[i];              // Column 0: first RHS
		B[i + size] = B1[i];       // Column 1: second RHS
	}

	dpotrs_(&uplo, &size_lap, &two, A, &size_lap,
	        B, &size_lap, &info_solve);

	// Report errors BEFORE reading results — dpotrs_ may write
	// garbage to B on failure.
	handle_lapack_error(info_solve, label, size);

	if (info_solve == 0) {
		// Extract solutions from column-major buffer
		for (int i = 0; i < size; ++i) {
			X0[i] = B[i];
			X1[i] = B[i + size];
		}

		// E-H1/R-01: II-09 solution-quality validation on the 2rhs dpotrs_
		// success path — mirrors lapack_spd_solve_1rhs, which the prior
		// fix 828443e added to the 1rhs sibling only.
		// dpotrs_ returns INFO=0 even for near-singular-but-SPD systems,
		// and the OK-level SumOnes/mu guards in my_kriging_weights.h
		// (|SumOnes| < 1e-12, |mu| > 1e10) are mathematically bypassable:
		// with a near-null direction v of A satisfying v'1 ≈ 0 (e.g. the
		// pairwise-duplicate direction [1,-1,0,...]), X0 = A⁻¹·B0 carries
		// the wild component (B0'v/λ_min)·v while X1 = A⁻¹·1 stays modest
		// (1 has no component along v) — SumSK ≈ 0 by cancellation,
		// SumOnes = O(1), mu = O(1): every guard passes and
		// w = X0 − mu·X1 stays wild (ADV-X1 bypass algebra).  Two guards:
		//   - magnitude: final-weight bound max|w|_inf > 1e3 → fail.
		//     R-01: measured on the SCALE-INVARIANT final combination
		//     w = X0 − mu·X1 (ok_final_weight_magnitude), NOT the raw
		//     intermediates — |X1|_inf ∝ 1/sill made the previous
		//     max(|X0|_inf, |X1|_inf) gate reject legal small-sill OK
		//     models (sill ≲ 3e-4) at every node while the final weights
		//     are bit-identical across sills (max|Δw| = 3.2e-16) and
		//     satisfy |w| ≲ 0.5.  The bypass catch is preserved: X0 wild
		//     ⇒ w stays wild (mu = O(1) there).  Deliberately tighter
		//     than the old 1rhs 1e10 bound: real Gaussian-model
		//     near-duplicate geometry produces wild weights ~1e4–1e5
		//     (ADV-X1 §4), which 1e10 would pass.
		//   - residual: ||A_orig·X − B||_inf vs sqrt(eps)·data_scale·size
		//     for BOTH RHS, identical tolerance to gauss_solve
		//     (gauss_solver.cpp) and the 1rhs gate.
		// A_orig holds the original matrix (unmodified on the dpotrs_
		// path) and B0/B1 still hold the original RHS (dpotrs_ overwrote
		// only the combined buffer), so both are computable here.
		{
			double max_abs_weight = 0.0;
			bool combination_defined = ok_final_weight_magnitude(X0, X1, size, max_abs_weight);
			if (!combination_defined)
			{
				char error_msg[256];
				snprintf(error_msg, sizeof(error_msg),
					"LAPACK Error in %s: OK weight combination undefined (|SumX1| < 1e-12 or |mu| > 1e10). Matrix size: %d",
					label, size);
				HPGL_LOG_STRING(error_msg);
				return false;
			}
			if (max_abs_weight > 1e3)
			{
				char error_msg[256];
				snprintf(error_msg, sizeof(error_msg),
					"LAPACK Error in %s: solution magnitude |w|_inf = %.3g exceeds stability bound 1e3 (near-singular system). Matrix size: %d",
					label, max_abs_weight, size);
				HPGL_LOG_STRING(error_msg);
				return false;
			}

			double max_residual = 0.0;
			double data_scale = 1.0;
			for (int i = 0; i < size; ++i)
			{
				double ax0 = 0.0;
				double ax1 = 0.0;
				for (int j = 0; j < size; ++j)
				{
					double aij = A_orig[static_cast<size_t>(i) * size + j];
					ax0 += aij * X0[j];
					ax1 += aij * X1[j];
				}
				max_residual = std::max(max_residual, std::abs(ax0 - B0[i]));
				max_residual = std::max(max_residual, std::abs(ax1 - B1[i]));
				for (int j = 0; j < size; ++j)
					data_scale = std::max(data_scale, std::abs(A_orig[static_cast<size_t>(i) * size + j]));
				data_scale = std::max(data_scale, std::abs(B0[i]));
				data_scale = std::max(data_scale, std::abs(B1[i]));
			}
			// sqrt(eps) * size ≈ 1.5e-8 * n: tolerance for acceptable
			// round-off, matching gauss_solve's residual gate.
			double tol = std::sqrt(std::numeric_limits<double>::epsilon()) * data_scale * size;
			if (max_residual > tol)
			{
				char error_msg[256];
				snprintf(error_msg, sizeof(error_msg),
					"LAPACK Error in %s: solution residual ||Ax-b||_inf = %.3g exceeds tolerance %.3g (unreliable solution). Matrix size: %d",
					label, max_residual, tol, size);
				HPGL_LOG_STRING(error_msg);
				return false;
			}
		}

		return true;
	}

	return false;
}

} // namespace detail
} // namespace hpgl

#endif // __SOLVER_ENTRY_POINT_H__E3A9B7F1_8C2D_4D56_AF90_126D84E5C339__
