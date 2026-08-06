#include "stdafx.h"
#include <cmath>
#include <limits>
#include "property_array.h"

namespace hpgl
{
	bool gauss_solve(double * A, double * B, double * X, int size, bool apply_magnitude_gate, double target_variance)
	{
		HPGL_CHECK(A != nullptr && B != nullptr && X != nullptr, "gauss_solve: null pointer argument");
		HPGL_CHECK(size > 0, "gauss_solve: invalid size");

		// F-M10 (I2-23 sibling): all matrix index arithmetic and the
		// size*size element count must be computed in size_t — the
		// signed-int product overflows for size > 46340, wraps negative,
		// and silently corrupts allocations/indexing.
		const size_t n = static_cast<size_t>(size);

		// Save original A and B for residual quality check after solve.
		// A and B are modified in-place during elimination; the originals
		// are needed to compute ||Ax - b|| after back-substitution.
		std::vector<double> A_orig(A, A + n * n);
		std::vector<double> B_orig(B, B + n);

		std::vector<int> flags(size, 0);
		std::vector<int> order(size, 0);
		for (int i = 0; i < size; ++i)
		{
			// Partial pivoting: select row with maximum absolute value in column i.
			// Previously used first-nonzero coefficient, which can silently lose
			// significant digits for ill-conditioned matrices on the Cholesky-failure
			// fallback path.  Max-abs pivoting reduces round-off error magnification.
			bool found = false;
			int row = -1;
			double max_abs = 0.0;
			for (int j = 0; j < size; ++j)
			{
				if (flags[j] == 0)
				{
					double abs_val = std::abs(A[static_cast<size_t>(j) * n + i]);
					if (abs_val > max_abs)
					{
						max_abs = abs_val;
						found = true;
						row = j;
					}
				}
			}

			if (!found)
			{
				return false; //matrix is singular
			}
			flags[row] = 1;
			order[i] = row;

			//normalize row
			double coef = A[static_cast<size_t>(row) * n + i];

			if (!std::isfinite(coef) || std::abs(coef) < std::numeric_limits<double>::epsilon())
			{
				return false; // Coefficient is NaN, Inf, or too close to zero
			}

			for (int j = i; j < size; ++j)
			{
				A[static_cast<size_t>(row) * n + j] /= coef;
			}
			B[row] /= coef;

			//subtract row
			for (int j = 0; j < size; ++j)
			{
				if (flags[j] == 0)
				{
					double coef =  A[static_cast<size_t>(j) * n + i];
					for (int k = i; k < size; ++k)
					{
						A[static_cast<size_t>(j) * n + k] -= coef * A[static_cast<size_t>(row) * n + k];
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
				X[i] -= A[static_cast<size_t>(row) * n + j] * X[j];
			}
		}

		// E-M87 + R-01: II-09 solution-magnitude guard on the fallback path.
		// The residual check below does NOT scale with ||x̂||: backward-
		// stable elimination yields a small residual even for wild
		// solutions (e.g. diag(1,1e-12), b=(1,1) → X=[1,1e12] with exact
		// residual 0), so without this gate a near-singular system would
		// report garbage weights as success to every 1rhs/2rhs caller
		// (no caller-side |weights| check exists).  The dpotrs_ success
		// path carries an explicit magnitude gate (solver_entry_point.h);
		// the fallback must too.  The gate must be scale-aware:
		//
		// R-01: the original ABSOLUTE bound (|X|_inf > 1e3) was not
		// scale-invariant — for the OK dual-RHS direction X1 = A⁻¹·1 the
		// solution scales as 1/sill (A = sill·M ⇒ X1 = (1/sill)·M⁻¹·1),
		// so legal small-sill models (sill ≲ 3e-4, sanctioned down to
		// MIN_SILL = 1e-6) were rejected on this path even though the
		// final OK weights are scale-invariant.
		//
		// R2-01: the pass-2 AND-form (|X|_inf > 1e3 AND
		// |X|_inf·(max|A_orig|/max|B_orig|) > 1e3) then over-rejected
		// legitimate cross-scale cokriging: for the 2×2
		// A=[[1e4,0.5],[0.5,1e-4]] with X=[−0.333, 6666.7] the relative
		// measure is 6666.7·(1e4/0.5) = 1.33e8 > 1e3 → rejected, while
		// v2.0.3's 1e10 bound accepted it.  The scale normalization by
		// max|B| made the effective bound 1e3·max|B|/max|A| ≈ 0.05 —
		// FAR below the legal secondary weight 6666.7 = O(ρ·σp/σs):
		// the bound got TIGHTER for cross-scale systems (wrong
		// direction).  The bound is now 1e3 · dynamic_range with
		// dynamic_range = sqrt(max|A_orig| / min|A_orig|_nz) ∈ [1, 1e12]
		// (min over NON-ZERO entries — an exact-zero covariance entry is
		// legitimate and must not drive the range to infinity):
		// a wider internal scale range legitimately tolerates larger
		// secondary weights.  Accepted cokriging range: σp/σs up to
		// ~1e4-1e6 → bound 1e7-1e9 ≫ O(ρ·σp/σs); the 2×2 example passes
		// (dr = sqrt(1e4/1e-4) = 1e4 → bound 1e7 → 6666.7 ≤ 1e7).
		// Still rejected: unit-scale near-null wild weights (E-H1
		// bypass, |X| ~ 1e4-1e5, dr ~ 1 → bound 1e3), the wild 1e12
		// probe (dr ≤ 1e6 → bound ≤ 1e9 < 1e12), and all-zero A_orig
		// (degenerate).  The formula is IDENTICAL to the 1rhs dpotrs_
		// gate (solver_entry_point.h) so both 1rhs branches give the
		// same verdict per caller.  The 2rhs OK caller additionally
		// applies ok_final_weight_magnitude (solver_entry_point.h) after
		// both gauss_solve calls, which judges the scale-invariant final
		// combination w = X0 − mu·X1 — the raw-solve gate above only
		// screens each solve's own magnitude.
		//
		// R3-01: the raw-solve gate is NOT the correct measure for the
		// 2rhs gauss-fallback path — X1 = A⁻¹·1 ∝ 1/sill is legitimately
		// large for legal small-sill OK models (sill ≲ 3e-4), so the 2rhs
		// caller (solver_entry_point.h lapack_spd_solve_2rhs) invokes
		// both gauss_solve calls with apply_magnitude_gate = false and
		// relies on its own scale-invariant final-weight gate
		// (ok_final_weight_magnitude, which judges w = X0 − mu·X1 and
		// catches every wild variant: E-H1 bypass |w| = 1e4 > 1e3,
		// E-M88 non-cancelling |mu| > 1e10 → undefined).  The 1rhs and
		// legacy callers keep the default apply_magnitude_gate = true.
		// The isfinite checks, the singular-pivot guard, and the residual
		// check below run UNCONDITIONALLY regardless of the flag.
		//
		// R3-02: the pre-filter alone is provably incapable of separating
		// the legal high-ρ cokriging corner from the pinned wild class:
		// at ρ → 1 both are A ≈ [[1,1−ε],[1−ε,1]] with |X| ~ 5e11-1e12
		// (the (1−ρ²)⁻¹ amplification is invisible to A-internal dynamic
		// range, dr ≈ 1) — no threshold on |X| or any function of
		// (A, B, X) magnitudes separates them.  The separation is in the
		// CONSISTENCY of B with A: the kriging variance
		// σK² = c − B'X (c = A_orig[0][0] = target variance = sill for
		// SK/correlogram, σp² for cokriging; B_orig = original RHS saved
		// above) must be non-negative for a realizable estimator.  The
		// gate is now a conjunction: the pre-filter fires AND
		// B'X > c·(1 + 1e-8·size).  For the mark-II cokriging corner
		// B'X = ρ²σp²/(1−ρ²) > c = σp² ⇔ ρ > 1/√2 ≈ 0.7071 — every
		// ρ ≳ 0.9995 corner has NEGATIVE kriging variance (invalid
		// cross-covariance model: the augmented field covariance is
		// indefinite) and is rejected ratio-independently; the pinned
		// wild class (B = [1,−1] on A = [[1,1−ε],[1−ε,1]] → B'X = 2e12
		// > c = 1) is rejected too.  The second condition fires ONLY when
		// the pre-filter already fired, so the artificial-RHS pinned
		// tests (|X| ≤ 2 → pre-filter no-fire) are unaffected.
		if (apply_magnitude_gate)
		{
			double max_abs_weight = 0.0;
			double scale_a_max = 0.0;
			double scale_a_min = 0.0; // min over NON-ZERO |A_orig| entries
			for (int i = 0; i < size; ++i)
			{
				max_abs_weight = std::max(max_abs_weight, std::abs(X[i]));
				for (int j = 0; j < size; ++j)
				{
					const double abs_a = std::abs(A_orig[static_cast<size_t>(i) * n + j]);
					scale_a_max = std::max(scale_a_max, abs_a);
					if (abs_a > 0.0)
						scale_a_min = (scale_a_min == 0.0) ? abs_a : std::min(scale_a_min, abs_a);
				}
			}
			// R2-01: dynamic-range-scaled bound — IDENTICAL formula to the
			// 1rhs dpotrs_ gate (solver_entry_point.h): the pass-2 AND-form
			// normalization (|X|_inf·(max|A|/max|B|) > 1e3) made the bound
			// TIGHTER for cross-scale cokriging (2×2 A=[[1e4,0.5],[0.5,1e-4]],
			// X=[−0.333, 6666.7]: normalized bound ≈ 0.05 < legal secondary
			// weight 6666.7 = O(ρ·σp/σs) → wrongly rejected).  Now
			// bound = 1e3·dynamic_range with dynamic_range =
			// sqrt(max|A_orig|/min|A_orig|_nz) ∈ [1.0, 1e12]: accepted
			// cokriging range σp/σs ≤ ~1e4-1e6 (2×2: dr = 1e4 → bound 1e7 →
			// 6666.7 passes), unit-scale near-null wild weights (E-H1
			// bypass, |X| ~ 1e4-1e5, dr ~ 1 → bound 1e3) and the wild 1e12
			// class (dr ≤ 1e6 → bound ≤ 1e9) still rejected.  All-zero
			// A_orig (scale_a_min == 0) → degenerate → wild.
			// R3-02: the pre-filter is now AND-ed with a kriging-variance
			// (Schur) consistency check — same formula as the 1rhs dpotrs_
			// gate in solver_entry_point.h (see the block comment above for
			// the ratio-independence proof: B'X = ρ²σp²/(1−ρ²) > c ⇔ ρ > 1/√2).
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
					// Variance-consistency: B_orig is the ORIGINAL RHS
					// (saved before elimination modified B in place).
					// R4-01: the reference is the kriging TARGET variance.
					// A_orig[0] is correct for SK (C(0)·sill) and cokriging
					// (σp²), but on the σ-scaled correlogram path A[0][0] =
					// C(0)·σ₀² is the FIRST datum's variance — the correlogram
					// caller passes target_variance = C(0)·σc² (sill-scaled);
					// default 0.0 keeps the A_orig[0] behavior.
					double bx = 0.0;
					for (int i = 0; i < size; ++i)
						bx += B_orig[static_cast<size_t>(i)] * X[i];
					const double reference_variance =
						(target_variance > 0.0) ? target_variance : A_orig[0];
					if (bx > reference_variance * (1.0 + 1e-8 * size))
						wild = true;
				}
			}
			if (wild)
				return false;
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
					ax += A_orig[static_cast<size_t>(i) * n + j] * X[j];
				double res = std::abs(ax - B_orig[i]);
				if (res > max_residual) max_residual = res;
				// Track magnitude of original data for relative tolerance
				for (int j = 0; j < size; ++j)
					data_scale = std::max(data_scale, std::abs(A_orig[static_cast<size_t>(i) * n + j]));
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

		// F-M10 (I2-23 sibling): matrix index arithmetic in size_t so a
		// size > 46340 cannot overflow the signed-int products.
		const size_t n = static_cast<size_t>(size);

		double V = 0.0;

		// inside matrix [L(i,j)]
		for (int j = 0; j < size; j++)
		{
			for(int i = j; i < size; i++)
			{
				if(i==j)
				{
					// main diagonals [L(i,i)]
						V = A[static_cast<size_t>(i) * n + i];
						for (int k = 0; k <= i-1; k++)
						{
							V -= (A_U[static_cast<size_t>(k) * n + i] * A_U[static_cast<size_t>(k) * n + i]);
						}

				// isfinite guard: NaN bypasses `V < epsilon` (NaN < X is always false).
				if (!std::isfinite(V) || V < std::numeric_limits<double>::epsilon())
				{
					return false;
				}

					A_L[static_cast<size_t>(i) * n + i] = sqrt(V);
					A_U[static_cast<size_t>(i) * n + i] = sqrt(V);
				}
				else
				{
						V = 0.0;
						for (int k = 0; k <= j-1; k++)
						{
							V += A_U[static_cast<size_t>(k) * n + i] * A_U[static_cast<size_t>(k) * n + j];
						}

						if( std::abs(A_U[static_cast<size_t>(j) * n + j]) < std::numeric_limits<double>::epsilon() )
						{
							return false;
						}

						A_U[static_cast<size_t>(j) * n + i] = (1 / A_U[static_cast<size_t>(j) * n + j]) * (A[static_cast<size_t>(j) * n + i] - V);
						A_L[static_cast<size_t>(i) * n + j] = A_U[static_cast<size_t>(j) * n + i];
				}
			}
		}
		return true;
	}

	void cholesky_solve(double * A_L, double * A_U, double * B, double * X, int size)
	{
		HPGL_CHECK(A_L != nullptr && A_U != nullptr && B != nullptr && X != nullptr, "cholesky_solve: null pointer argument");
		HPGL_CHECK(size > 0, "cholesky_solve: invalid size");

		// F-M10 (I2-23 sibling): matrix index arithmetic in size_t.
		const size_t n = static_cast<size_t>(size);

		std::vector<double> X_R(size,0.0);

		for (int i = 0; i <size ; i++)
		{
			X_R[i] = B[i];
			for (int j = 0; j <i; j++)
			{
				X_R[i] -= A_L[static_cast<size_t>(i) * n + j] * X_R[j];
			}

			// isfinite guard: NaN bypasses epsilon check
			double al = A_L[static_cast<size_t>(i) * n + i];
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
				X[i] -= A_U[static_cast<size_t>(i) * n + j] * X[j];
			}

			// isfinite guard: NaN bypasses epsilon check
			double au = A_U[static_cast<size_t>(i) * n + i];
			if (!std::isfinite(au) || std::abs(au) < std::numeric_limits<double>::epsilon())
			{
				X[i] = 0.0;
			}
			else
			{
				X[i] /= au;
			}
		}
    }

	bool cholesky_old(double * A, double * B, double * X, int size)
	{
		HPGL_CHECK(A != nullptr && B != nullptr && X != nullptr, "cholesky_old: null pointer argument");
		HPGL_CHECK(size > 0, "cholesky_old: invalid size");

		// F-M10 (I2-23 sibling): the size*size element counts and all
		// matrix index arithmetic must be size_t — signed-int products
		// overflow for size > 46340, wrap negative, and silently corrupt
		// the A_U/A_L allocations below.
		const size_t n = static_cast<size_t>(size);

		double V = 0.0;

		std::vector<double> A_U(n*n,0.0);
		std::vector<double> A_L(n*n,0.0);

		// inside matrix [L(i,j)]
		for (int j = 0; j < size; j++)
		{
			for(int i = j; i < size; i++)
			{
				if(i==j)
				{
					// main diagonals [L(i,i)]
					V = A[static_cast<size_t>(i) * n + i];
						for (int k = 0; k <= i-1; k++)
						{
							V -= (A_U[static_cast<size_t>(k) * n + i] * A_U[static_cast<size_t>(k) * n + i]);
						}

					// isfinite guard: NaN bypasses `V < epsilon` (NaN < X is always false).
					if (!std::isfinite(V) || V < std::numeric_limits<double>::epsilon())
					{
						return false;
					}

					A_L[static_cast<size_t>(i) * n + i] = sqrt(V);
					A_U[static_cast<size_t>(i) * n + i] = sqrt(V);
				}
				else
				{
						V = 0.0;
						for (int k = 0; k <= j-1; k++)
						{
							V += A_U[static_cast<size_t>(k) * n + i] * A_U[static_cast<size_t>(k) * n + j];
						}

						if( std::abs(A_U[static_cast<size_t>(j) * n + j]) < std::numeric_limits<double>::epsilon() )
						{
							return false;
						}

						A_U[static_cast<size_t>(j) * n + i] = (1 / A_U[static_cast<size_t>(j) * n + j]) * (A[static_cast<size_t>(j) * n + i] - V);
						A_L[static_cast<size_t>(i) * n + j] = A_U[static_cast<size_t>(j) * n + i];
				}
			}
		}

		std::vector<double> X_R(size,0.0);

		for (int i = 0; i <size ; i++)
		{
			X_R[i] = B[i];
			for (int j = 0; j <i; j++)
			{
				X_R[i] -= A_L[static_cast<size_t>(i) * n + j] * X_R[j];
			}

			// isfinite guard: NaN bypasses epsilon check
			double al = A_L[static_cast<size_t>(i) * n + i];
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
				X[i] -= A_U[static_cast<size_t>(i) * n + j] * X[j];
			}

			// isfinite guard: NaN bypasses epsilon check
			double au = A_U[static_cast<size_t>(i) * n + i];
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
