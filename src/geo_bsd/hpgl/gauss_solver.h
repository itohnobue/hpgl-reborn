#ifndef __GAUSS_SOLVER_H__2513A17A_DAF0_43DF_ADF0_BD80D8F29F38____
#define __GAUSS_SOLVER_H__2513A17A_DAF0_43DF_ADF0_BD80D8F29F38____

namespace hpgl
{
	// R3-01: apply_magnitude_gate (default true) lets the 2rhs gauss-fallback
	// caller (solver_entry_point.h lapack_spd_solve_2rhs) bypass the internal
	// raw-solve magnitude gate: X1 = A^-1 * 1 scales as 1/sill on legal
	// small-sill OK models, so the caller's own scale-invariant final-weight
	// gate (ok_final_weight_magnitude) is the authoritative 2rhs measure.
	// The isfinite/singular-pivot/residual checks run unconditionally.
	// R4-01: target_variance (default 0.0) lets a caller whose system matrix
	// A[0][0] is NOT the kriging target variance (the σ-scaled correlogram
	// path: A[0][0] = C(0)·σ₀², first datum's variance) pass the TRUE target
	// variance C(0)·σc² so the Schur-consistency term references the target
	// location's variance instead of spuriously rejecting valid estimators
	// with σc > σ0.  Default 0.0 preserves the A_orig[0] reference for
	// SK/cokriging callers.  Only consulted when the magnitude gate is on.
	bool gauss_solve(double * A, double * B, double * X, int size, bool apply_magnitude_gate = true, double target_variance = 0.0);

	void cholesky_solve(double * A_L, double * A_U, double * B, double * X, int size);
	bool cholesky_decomposition(double * A, double * A_U, double * A_L, int size);

	bool cholesky_old(double * A, double * B, double * X, int size);
}

#endif //__GAUSS_SOLVER_H__2513A17A_DAF0_43DF_ADF0_BD80D8F29F38____