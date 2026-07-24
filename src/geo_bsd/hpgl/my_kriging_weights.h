#ifndef __MY_KRIGING_WEIGHTS_H__B6211BC7_74C1_4D96_AB05_286A62D0F003
#define __MY_KRIGING_WEIGHTS_H__B6211BC7_74C1_4D96_AB05_286A62D0F003

#ifndef LAPACK_SOLVER
#define LAPACK_SOLVER
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cmath>
#include <mutex>

// Unified SPD solver entry point — provides handle_lapack_error,
// lapack_spd_solve_1rhs, lapack_spd_solve_2rhs.  Also includes
// lapack_compat.h, gauss_solver.h, and logging.h transitively.
#include "solver_entry_point.h"

#include "sugarbox_grid.h"
#include "property_array.h"
#include "typedefs.h"

// ====================================================================
// SECTION 1: LAPACK Integration
// LAPACK error handling, safe allocation helpers, and Fortran
// interface (dpotrf_/dpotrs_) used by all weight calculators below.
// Solver selection: #define LAPACK_SOLVER at file top.
// ====================================================================

namespace hpgl
{
	// ================================================================
	// BLAS thread-count guard for concurrent kriging function calls.
	// Process-wide BLAS thread count is reference-counted so that
	// concurrent calls to different kriging functions do not lose the
	// original thread count through non-atomic save/restore races.
	// ================================================================
	namespace detail {
		inline std::mutex& blas_thread_mutex() {
			static std::mutex m;
			return m;
		}
		inline int& blas_ref_count() {
			static int c = 0;
			return c;
		}
		inline int& blas_saved_threads() {
			static int s = 0;
			return s;
		}
		/// Acquire BLAS thread-count guard: sets BLAS threads to 1.
		/// Only the first caller saves the original count.
		/// Must be paired with blas_thread_restore().
		inline void blas_thread_acquire(int (*getter)(), void (*setter)(int)) {
			std::lock_guard<std::mutex> lock(blas_thread_mutex());
			if (blas_ref_count() == 0) {
				blas_saved_threads() = getter();
			}
			blas_ref_count()++;
			setter(1);
		}
		/// Release BLAS thread-count guard: restores original count
		/// when all callers have released. Must be paired with
		/// blas_thread_acquire().
		inline void blas_thread_restore(void (*setter)(int)) {
			std::lock_guard<std::mutex> lock(blas_thread_mutex());
			blas_ref_count()--;
			if (blas_ref_count() == 0) {
				setter(blas_saved_threads());
			}
		}
	}

	// SECURITY FIX: Safe allocation helper to prevent integer overflow
	namespace detail {
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
		std::vector<double> A_backup;   // size²: backup of A for fallback path
#ifdef LAPACK_SOLVER
		std::vector<double> B;          // 2*size: combined OK RHS buffer
#endif
	};

	// ================================================================
	// SECTION 2: Weight Calculators
	// sk_kriging_weights_3  — Simple Kriging (solve A*w = b directly).
	// ok_kriging_weights_3  — Ordinary Kriging (SK solve + Lagrange
	//                          multiplier correction).
	// corellogramed_weights_3 — Correlogram kriging (pre-transform
	//                            covariance by sigma_i * sigma_j).
	// Each function uses LAPACK_SOLVER (dpotrf_ / dpotrs_ with gauss_solve fallback).
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

#ifdef LAPACK_SOLVER

		// NOTE: LAPACK within OpenMP region — avoid BLAS thread oversubscription.
		// Unified SPD solver: backup → dpotrf_ → (on fail) gauss_solve →
		// (on success) dpotrs_.  Handles all LAPACK calls, error reporting,
		// and fallback logic.  weights[] receives the solution on success.
		std::vector<double> A_backup(A);

		bool system_solved = detail::lapack_spd_solve_1rhs(
			&A[0], size, &weights[0], &b[0],
			&A_backup[0], "SK Cholesky");

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
				// Clamp to zero — floating-point subtraction can produce small negatives.
				// IEEE 754: NaN < 0 evaluates to false, so also guard against NaN/Inf.
				if (!std::isfinite(variance) || variance < 0) variance = 0;
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

#ifdef LAPACK_SOLVER
		// Unified SPD solver: backup → dpotrf_ → (on fail) gauss_solve →
		// (on success) dpotrs_.  weights[] receives the solution on success.
		ws.A_backup.resize(matrix_size);
		std::copy(ws.A.begin(), ws.A.begin() + matrix_size, ws.A_backup.begin());

		bool system_solved = detail::lapack_spd_solve_1rhs(
			&ws.A[0], size, &weights[0], &ws.b[0],
			&ws.A_backup[0], "SK Cholesky (ws)");

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
				// Clamp to zero — floating-point subtraction can produce small negatives.
				// IEEE 754: NaN < 0 evaluates to false, so also guard against NaN/Inf.
				if (!std::isfinite(variance) || variance < 0) variance = 0;
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


#ifdef LAPACK_SOLVER
		// NOTE: LAPACK within OpenMP region — avoid BLAS thread oversubscription.
		// Unified SPD solver for OK (nrhs=2): backup → dpotrf_ →
		// (on fail) two gauss_solve calls for both RHS →
		// (on success) combined dpotrs_ with nrhs=2.
		// sk_weights[] and ones_result[] receive solutions from either path.
		std::vector<double> A_backup(A);

		bool system_solved = detail::lapack_spd_solve_2rhs(
			&A[0], size,
			&sk_weights[0], &b[0],
			&ones_result[0], &ones[0],
			&A_backup[0],
			"OK Cholesky");

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
				// Clamp to zero — floating-point subtraction can produce small negatives.
				// IEEE 754: NaN < 0 evaluates to false, so also guard against NaN/Inf.
				if (!std::isfinite(variance) || variance < 0) variance = 0;
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

#ifdef LAPACK_SOLVER
		// Unified SPD solver for OK (nrhs=2): backup → dpotrf_ →
		// (on fail) two gauss_solve calls →
		// (on success) combined dpotrs_ with nrhs=2.
		// ws.sk_weights[] and ws.ones_result[] receive solutions.
		ws.A_backup.resize(matrix_size);
		std::copy(ws.A.begin(), ws.A.begin() + matrix_size, ws.A_backup.begin());

		system_solved = detail::lapack_spd_solve_2rhs(
			&ws.A[0], size,
			&ws.sk_weights[0], &ws.b[0],
			&ws.ones_result[0], &ws.ones[0],
			&ws.A_backup[0],
			"OK Cholesky (ws)");
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
				// Clamp to zero — floating-point subtraction can produce small negatives.
				// IEEE 754: NaN < 0 evaluates to false, so also guard against NaN/Inf.
				if (!std::isfinite(variance) || variance < 0) variance = 0;
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
		// IEEE 754 NaN passes both < 0.0 and > 1.0 checks unmodified,
		// so detect and clamp NaN explicitly.
		if (std::isnan(meanc)) meanc = 0.0;
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
			if (std::isnan(meani)) meani = 0.0;
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

#ifdef LAPACK_SOLVER

		// NOTE: LAPACK within OpenMP region — avoid BLAS thread oversubscription.
		// Unified SPD solver: backup → dpotrf_ → (on fail) gauss_solve →
		// (on success) dpotrs_.  weights[] receives the solution on success.
		std::vector<double> A_backup(A);

		bool system_solved = detail::lapack_spd_solve_1rhs(
			&A[0], size, &weights[0], &b[0],
			&A_backup[0], "Corellogram Cholesky");

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
		// IEEE 754 NaN passes both < 0.0 and > 1.0 checks unmodified,
		// so detect and clamp NaN explicitly.
		if (std::isnan(meanc)) meanc = 0.0;
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
			if (std::isnan(meani)) meani = 0.0;
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

#ifdef LAPACK_SOLVER
		// Unified SPD solver: backup → dpotrf_ → (on fail) gauss_solve →
		// (on success) dpotrs_.  weights[] receives the solution on success.
		ws.A_backup.resize(matrix_size);
		std::copy(ws.A.begin(), ws.A.begin() + matrix_size, ws.A_backup.begin());

		system_solved = detail::lapack_spd_solve_1rhs(
			&ws.A[0], size, &weights[0], &ws.b[0],
			&ws.A_backup[0], "Corellogram Cholesky (ws)");
#endif

		return system_solved;
	}

}

#endif //__MY_KRIGING_WEIGHTS_H__B6211BC7_74C1_4D96_AB05_286A62D0F003
