/**
 * Unit tests for lapack_spd_solve_1rhs and lapack_spd_solve_2rhs.
 *
 * Uses the same minimal assertion-based framework as test_hpgl_core.cpp.
 * Tests cover:
 *   - lapack_spd_solve_1rhs: known SPD matrix (success path)
 *   - lapack_spd_solve_1rhs: non-SPD matrix (fallback to gauss_solve)
 *   - lapack_spd_solve_1rhs: NaN in A matrix (early rejection)
 *   - lapack_spd_solve_2rhs: known SPD matrix with dual RHS (success path)
 *   - lapack_spd_solve_2rhs: non-SPD matrix (fallback to gauss_solve)
 *   - lapack_spd_solve_2rhs: NaN in A matrix (early rejection)
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// TNT matrix library (required by HPGL covariance headers)
#include <tnt.h>

// Include the solver entry point header
#include "src/geo_bsd/hpgl/solver_entry_point.h"

// ---- Minimal test framework ----

static int g_tests_run = 0;
static int g_tests_failed = 0;
static const char *g_current_test = "";

#define TEST(name)                                          \
    g_current_test = name;                                   \
    g_tests_run++;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL: %s:%d in %s: (%s) is false\n",         \
                         __FILE__, __LINE__, g_current_test, #cond);            \
            g_tests_failed++;                                                   \
            return;                                                             \
        }                                                                      \
    } while (0)

#define CHECK_CLOSE(a, b, eps)                                                 \
    do {                                                                       \
        double _a = (a);                                                       \
        double _b = (b);                                                       \
        if (std::abs(_a - _b) > (eps)) {                                       \
            std::fprintf(stderr,                                                \
                         "FAIL: %s:%d in %s: |%.15g - %.15g| > %.15g\n",       \
                         __FILE__, __LINE__, g_current_test, _a, _b, (double)(eps)); \
            g_tests_failed++;                                                   \
            return;                                                             \
        }                                                                      \
    } while (0)

// ---- Tests ----

/// Solve a known 3x3 SPD system with lapack_spd_solve_1rhs.
/// A = [[4, 2, 2],
///      [2, 5, 1],
///      [2, 1, 6]]
/// Solution for B = [1, 1, 1]^T → X ≈ [0.125, 0.125, 0.125]
void test_1rhs_known_spd_success() {
    TEST("lapack_spd_solve_1rhs 3x3 SPD success");
    int size = 3;
    // Positive-definite matrix from test_hpgl_core.cpp cholesky tests
    std::vector<double> A = {
        4.0, 2.0, 2.0,
        2.0, 5.0, 1.0,
        2.0, 1.0, 6.0
    };
    std::vector<double> B = { 1.0, 1.0, 1.0 };
    std::vector<double> X(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_1rhs(
        A.data(), size, X.data(), B.data(),
        A_backup.data(), "test 1rhs SPD");

    CHECK(ok);
    // Verify against known solution computed offline:
    // A * x = [1, 1, 1] => x ≈ [0.139535, 0.069767, 0.081395]
    // (Let's use a simpler B that gives integer solution)
    // Actually, just verify ||Ax - B|| is small
    // This is a residual check — A*X should ≈ B
    double Ax_minus_B[3] = {
        (4.0 * X[0] + 2.0 * X[1] + 2.0 * X[2]) - 1.0,
        (2.0 * X[0] + 5.0 * X[1] + 1.0 * X[2]) - 1.0,
        (2.0 * X[0] + 1.0 * X[1] + 6.0 * X[2]) - 1.0
    };
    for (int i = 0; i < 3; ++i) {
        CHECK_CLOSE(Ax_minus_B[i], 0.0, 1e-12);
    }
}

/// Same known 3x3 SPD, but with B = A * [1, 1, 1]^T = [8, 8, 9].
/// Expected solution: X = [1, 1, 1].
void test_1rhs_known_spd_integer_solution() {
    TEST("lapack_spd_solve_1rhs 3x3 SPD integer solution");
    int size = 3;
    std::vector<double> A = {
        4.0, 2.0, 2.0,
        2.0, 5.0, 1.0,
        2.0, 1.0, 6.0
    };
    // B = A * [1, 1, 1]^T
    std::vector<double> B = { 8.0, 8.0, 9.0 };
    std::vector<double> X(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_1rhs(
        A.data(), size, X.data(), B.data(),
        A_backup.data(), "test 1rhs SPD int");

    CHECK(ok);
    CHECK_CLOSE(X[0], 1.0, 1e-12);
    CHECK_CLOSE(X[1], 1.0, 1e-12);
    CHECK_CLOSE(X[2], 1.0, 1e-12);
}

/// Non-SPD matrix: dpotrf_ should fail, forcing fallback to gauss_solve.
/// The matrix is stored row-major but read column-major by LAPACK
/// (uplo='U'), so the effective symmetric matrix is built from the upper
/// triangle read in column-major order: A[0]=(0,0), A[2]=(0,1), A[3]=(1,1).
/// For the storage [1,2,2,4]: (0,0)=1, (0,1)=2, (1,1)=4 → [[1,2],[2,4]],
/// det = 1*4 - 2*2 = 0 → singular → NOT SPD → dpotrf_ fails → gauss_solve
/// also fails (linearly dependent rows). PR-04: the previous storage
/// [2,4,1,2] read column-major as [[2,1],[1,2]] (det 3, SPD) so dpotrf_
/// succeeded and the fallback was never exercised.
void test_1rhs_non_spd_fallback() {
    TEST("lapack_spd_solve_1rhs 2x2 non-SPD (fallback to gauss_solve)");
    int size = 2;
    // This matrix is NOT positive-definite — rows are linearly dependent.
    // dpotrf_ will fail; gauss_solve must handle it.
    std::vector<double> A = {
        1.0, 2.0,
        2.0, 4.0
    };
    std::vector<double> B = { 6.0, 3.0 };
    std::vector<double> X(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_1rhs(
        A.data(), size, X.data(), B.data(),
        A_backup.data(), "test 1rhs non-SPD");

    // gauss_solve should detect singularity and return false
    CHECK(!ok);
}

/// Non-SPD that is actually solvable by gauss_solve (not singular,
/// just not SPD). Use an asymmetric matrix.
/// Storage [1,3,3,1] read column-major with uplo='U' gives the effective
/// symmetric matrix [[1,3],[3,1]] (det = 1*1 - 3*3 = -8) → NOT SPD →
/// dpotrf_ fails → gauss_solve must succeed on the row-major A_orig
/// [[1,3],[3,1]]. B = A*[2,1]^T = [5,7] so X = [2,1] verifies the
/// fallback solution. PR-04: the previous storage [1,5,0,2] read
/// column-major as [[1,0],[0,2]] (SPD diagonal) so dpotrf_ succeeded.
void test_1rhs_non_spd_solvable() {
    TEST("lapack_spd_solve_1rhs 2x2 non-SPD but non-singular (fallback)");
    int size = 2;
    // Asymmetric, non-SPD but non-singular: dpotrf_ will fail,
    // gauss_solve should succeed.
    std::vector<double> A = {
        1.0, 3.0,
        3.0, 1.0
    };
    // B = A * [2, 1]^T = [1*2+3*1, 3*2+1*1] = [5, 7]
    std::vector<double> B = { 5.0, 7.0 };
    std::vector<double> X(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_1rhs(
        A.data(), size, X.data(), B.data(),
        A_backup.data(), "test 1rhs non-SPD solvable");

    // PR-04 regression assertion: the gauss fallback must actually trigger.
    // dpotrf_ reads the upper triangle column-major: for storage [1,3,3,1]
    // that is [[1,3],[3,1]] with det = -8 < 0, which is NOT positive
    // definite — dpotrf_ MUST fail (info > 0). The matrix is symmetric, so
    // a successful dpotrf_ would otherwise produce the same solution and
    // mask a regression where the fallback never runs. Probing dpotrf_
    // directly proves the Cholesky path is genuinely unavailable, so the
    // ok==true result below can only come from gauss_solve.
    {
        char uplo = 'U';
        integer n = size;
        integer info = 0;
        std::vector<double> probe(A);
        dpotrf_(&uplo, &n, probe.data(), &n, &info);
        CHECK(info > 0);   // genuinely non-SPD → fallback required
    }

    CHECK(ok);
    CHECK_CLOSE(X[0], 2.0, 1e-12);
    CHECK_CLOSE(X[1], 1.0, 1e-12);
}

/// NaN in A matrix: the NaN pre-check should reject early and return false.
void test_1rhs_nan_in_a_rejected() {
    TEST("lapack_spd_solve_1rhs NaN in A (early rejection)");
    int size = 2;
    std::vector<double> A = {
        4.0, std::numeric_limits<double>::quiet_NaN(),
        2.0, 5.0
    };
    std::vector<double> B = { 6.0, 8.0 };
    std::vector<double> X(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_1rhs(
        A.data(), size, X.data(), B.data(),
        A_backup.data(), "test 1rhs NaN");

    // Should be rejected before dpotrf_ is called
    CHECK(!ok);
}

/// Inf in A matrix: the Inf pre-check should reject early.
void test_1rhs_inf_in_a_rejected() {
    TEST("lapack_spd_solve_1rhs Inf in A (early rejection)");
    int size = 2;
    std::vector<double> A = {
        4.0, 2.0,
        2.0, std::numeric_limits<double>::infinity()
    };
    std::vector<double> B = { 6.0, 8.0 };
    std::vector<double> X(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_1rhs(
        A.data(), size, X.data(), B.data(),
        A_backup.data(), "test 1rhs Inf");

    CHECK(!ok);
}

/// lapack_spd_solve_2rhs with known SPD matrix.
/// Solve A * [X0 | X1] = [B0 | B1] via combined dpotrs_ with nrhs=2.
void test_2rhs_known_spd_success() {
    TEST("lapack_spd_solve_2rhs 3x3 SPD dual RHS success");
    int size = 3;
    std::vector<double> A = {
        4.0, 2.0, 2.0,
        2.0, 5.0, 1.0,
        2.0, 1.0, 6.0
    };
    // B0 = A * [1, 0, 0]^T = [4, 2, 2]
    // B1 = A * [0, 1, 0]^T = [2, 5, 1]
    std::vector<double> B0 = { 4.0, 2.0, 2.0 };
    std::vector<double> B1 = { 2.0, 5.0, 1.0 };
    std::vector<double> X0(size, 0.0);
    std::vector<double> X1(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_2rhs(
        A.data(), size,
        X0.data(), B0.data(),
        X1.data(), B1.data(),
        A_backup.data(), "test 2rhs SPD");

    CHECK(ok);
    CHECK_CLOSE(X0[0], 1.0, 1e-12);
    CHECK_CLOSE(X0[1], 0.0, 1e-12);
    CHECK_CLOSE(X0[2], 0.0, 1e-12);
    CHECK_CLOSE(X1[0], 0.0, 1e-12);
    CHECK_CLOSE(X1[1], 1.0, 1e-12);
    CHECK_CLOSE(X1[2], 0.0, 1e-12);
}

/// Non-SPD matrix: dpotrf_ should fail, fallback to two gauss_solve calls.
void test_2rhs_non_spd_fallback() {
    TEST("lapack_spd_solve_2rhs 2x2 non-SPD (fallback to gauss_solve)");
    int size = 2;
    // Asymmetric non-SPD matrix
    std::vector<double> A = {
        2.0, 1.0,
        4.0, 3.0
    };
    // B0 = A * [1, 0]^T = [2, 4]
    // B1 = A * [0, 1]^T = [1, 3]
    std::vector<double> B0 = { 2.0, 4.0 };
    std::vector<double> B1 = { 1.0, 3.0 };
    std::vector<double> X0(size, 0.0);
    std::vector<double> X1(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_2rhs(
        A.data(), size,
        X0.data(), B0.data(),
        X1.data(), B1.data(),
        A_backup.data(), "test 2rhs non-SPD");

    CHECK(ok);
    CHECK_CLOSE(X0[0], 1.0, 1e-12);
    CHECK_CLOSE(X0[1], 0.0, 1e-12);
    CHECK_CLOSE(X1[0], 0.0, 1e-12);
    CHECK_CLOSE(X1[1], 1.0, 1e-12);
}

/// Singular non-SPD: both dpotrf_ and gauss_solve should fail.
/// Storage [1,2,2,4] read column-major with uplo='U' gives the effective
/// symmetric matrix [[1,2],[2,4]] (det = 1*4 - 2*2 = 0) → singular → NOT
/// SPD → dpotrf_ fails → gauss_solve also fails (linearly dependent rows).
/// PR-04: the previous storage [2,4,1,2] read column-major as [[2,1],[1,2]]
/// (det 3, SPD) so dpotrf_ succeeded and the assertion never held.
void test_2rhs_singular_returns_false() {
    TEST("lapack_spd_solve_2rhs 2x2 singular (both fail)");
    int size = 2;
    std::vector<double> A = {
        1.0, 2.0,
        2.0, 4.0  // linearly dependent rows
    };
    std::vector<double> B0 = { 6.0, 3.0 };
    std::vector<double> B1 = { 4.0, 2.0 };
    std::vector<double> X0(size, 0.0);
    std::vector<double> X1(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_2rhs(
        A.data(), size,
        X0.data(), B0.data(),
        X1.data(), B1.data(),
        A_backup.data(), "test 2rhs singular");

    // Singular matrix — gauss_solve should fail too
    CHECK(!ok);
}

/// NaN in A matrix: early rejection before dpotrf_.
void test_2rhs_nan_in_a_rejected() {
    TEST("lapack_spd_solve_2rhs NaN in A (early rejection)");
    int size = 2;
    std::vector<double> A = {
        4.0, 2.0,
        std::numeric_limits<double>::quiet_NaN(), 5.0
    };
    std::vector<double> B0 = { 6.0, 8.0 };
    std::vector<double> B1 = { 4.0, 3.0 };
    std::vector<double> X0(size, 0.0);
    std::vector<double> X1(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_2rhs(
        A.data(), size,
        X0.data(), B0.data(),
        X1.data(), B1.data(),
        A_backup.data(), "test 2rhs NaN");

    CHECK(!ok);
}

// II-09: near-singular-but-SPD systems must be rejected on the dpotrs_
// success path. dpotrs_ returns INFO=0 for these matrices (they ARE SPD),
// but the solution is garbage-huge (probe: [1.0e12, -1.0e12]) and would be
// reported as KI_SUCCESS by SK/correlogram/cokriging without the
// solution-quality guard. Pre-fix: ok == true. Post-fix: the magnitude
// guard (|X|_inf > 1e3 · dynamic_range) rejects the wild solution.
void test_1rhs_near_singular_spd_rejected() {
    TEST("lapack_spd_solve_1rhs near-singular SPD rejected (II-09)");
    int size = 2;
    // A = [[1, 1-eps], [1-eps, 1]] with eps=1e-12: det ~ 2e-12 > 0 → SPD
    // (dpotrf INFO=0) but nearly singular; the solution for B=[1,-1] is
    // ~[1e12, -1e12] — huge garbage weights.
    const double eps = 1e-12;
    std::vector<double> A = {
        1.0, 1.0 - eps,
        1.0 - eps, 1.0
    };
    std::vector<double> B = { 1.0, -1.0 };
    std::vector<double> X(size, 0.0);
    std::vector<double> A_backup(A);

    // Confirm the matrix genuinely IS SPD — dpotrf_ must succeed, otherwise
    // this test would be exercising the gauss fallback, not the new guard.
    {
        char uplo = 'U';
        integer n = size;
        integer info = 0;
        std::vector<double> probe(A);
        dpotrf_(&uplo, &n, probe.data(), &n, &info);
        CHECK(info == 0);   // SPD — dpotrs_ success path is reached
    }

    bool ok = hpgl::detail::lapack_spd_solve_1rhs(
        A.data(), size, X.data(), B.data(),
        A_backup.data(), "test near-singular SPD");

    // Pre-fix: INFO=0 → returns true with |X| ~ 1e12. Post-fix: rejected.
    CHECK(!ok);
    // The guard must have caught the wild magnitude (not some other path).
    CHECK(std::max(std::abs(X[0]), std::abs(X[1])) > 1e10 || !std::isfinite(X[0]) || !std::isfinite(X[1]));
}

// E-H1 (Stage-8 TEST-ADD T-10): the 2rhs dpotrs_ success path must reject
// near-singular-but-SPD systems. The OK path solves A·[X0|X1] = [B0|B1]; the
// II-09 quality guard was added to the 1rhs sibling only (fix 828443e), so
// the 2rhs path reported INFO=0 → KI_SUCCESS with wild weights (~1e12) for
// direct-C callers. Mirror of the 1rhs near-singular test with TWO RHS both
// aligned with the near-null direction [1,-1]: the OK combination is
// undefined (|SumX1| ~ 0) → must be rejected.
void test_2rhs_near_singular_spd_rejected() {
    TEST("lapack_spd_solve_2rhs near-singular SPD rejected (E-H1)");
    int size = 2;
    const double eps = 1e-12;
    std::vector<double> A = {
        1.0, 1.0 - eps,
        1.0 - eps, 1.0
    };
    std::vector<double> B0 = { 1.0, -1.0 };
    std::vector<double> B1 = { 1.0, -1.0 };
    std::vector<double> X0(size, 0.0);
    std::vector<double> X1(size, 0.0);
    std::vector<double> A_backup(A);

    // Confirm the matrix genuinely IS SPD — dpotrf_ must succeed, otherwise
    // this test would be exercising the gauss fallback, not the guard on the
    // dpotrs_ success path.
    {
        char uplo = 'U';
        integer n = size;
        integer info = 0;
        std::vector<double> probe(A);
        dpotrf_(&uplo, &n, probe.data(), &n, &info);
        CHECK(info == 0);   // SPD — dpotrs_ success path is reached
    }

    bool ok = hpgl::detail::lapack_spd_solve_2rhs(
        A.data(), size,
        X0.data(), B0.data(),
        X1.data(), B1.data(),
        A_backup.data(), "test 2rhs near-singular SPD");

    // Pre-fix (E-H1): no quality guard on the 2rhs path → ok == true with
    // |X| ~ 1e12 garbage weights. Post-fix: the OK weight combination is
    // undefined (X1 = A⁻¹·[1,-1] = 1e12·[1,-1] → SumOnes ~ 0) → rejected.
    CHECK(!ok);
}

// R-01 (Stage-8 TEST-ADD T-22): the OK dual-RHS solve must ACCEPT legal
// small-sill models. X1 = A⁻¹·1 ∝ 1/sill is legitimately large for sill ≲
// 3e-4, but the FINAL weights w = X0 − mu·X1 are scale-invariant. Pre-fix the
// absolute bound max(|X0|,|X1|) > 1e3 rejected these systems at every node
// (bit-identical to the accepted ones, max|Δw| = 3.2e-16). Post-fix the gate
// measures the scale-invariant final combination.
void test_2rhs_small_sill_ok_accepted() {
    TEST("lapack_spd_solve_2rhs small-sill OK accepted (R-01)");
    int size = 2;
    // A = sill·M with sill = 1e-4, M = [[2,0.5],[0.5,2]] — well-conditioned.
    // X1 = A⁻¹·[1,1] = [4000, 4000] (∝ 1/sill) — huge intermediate, small
    // final weights.
    const double sill = 1e-4;
    std::vector<double> A = {
        2.0 * sill, 0.5 * sill,
        0.5 * sill, 2.0 * sill
    };
    // B0 = A·[0.25, 0.25] so the first RHS has a modest exact solution.
    std::vector<double> B0 = { 0.625 * sill, 0.625 * sill };
    std::vector<double> B1 = { 1.0, 1.0 };
    std::vector<double> X0(size, 0.0);
    std::vector<double> X1(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_2rhs(
        A.data(), size,
        X0.data(), B0.data(),
        X1.data(), B1.data(),
        A_backup.data(), "test small-sill OK");

    // Post-fix: the final combination w = X0 − mu·X1 is small (≈ [0.5, 0.5])
    // → accepted. Pre-fix (R-01): |X1|_inf = 4000 > 1e3 → rejected.
    CHECK(ok);
    // X0 = A⁻¹B0 = [0.25, 0.25] — the first RHS solve is exact.
    CHECK_CLOSE(X0[0], 0.25, 1e-9);
    CHECK_CLOSE(X0[1], 0.25, 1e-9);
    // The final OK weights stay within the stability bound (w ≈ [0.5, 0.5]).
    double mu = (X0[0] + X0[1] - 1.0) / (X1[0] + X1[1]);
    double w0 = X0[0] - mu * X1[0];
    double w1 = X0[1] - mu * X1[1];
    CHECK(std::max(std::abs(w0), std::abs(w1)) <= 1e3);
    // Residual: A·X0 == B0 (the solved system is exact; A_backup holds the
    // original matrix — the in-place dpotrf_ modified A).
    CHECK_CLOSE(A_backup[0] * X0[0] + A_backup[1] * X0[1], B0[0], 1e-9);
    CHECK_CLOSE(A_backup[2] * X0[0] + A_backup[3] * X0[1], B0[1], 1e-9);
}

// R4-01 (Stage-8 TEST-ADD T-17): the Schur-consistency reference on the
// σ-scaled correlogram (SIS-LVM) path must be the kriging TARGET variance
// C(0)·σc², not A[0][0] = C(0)·σ0² (the first datum's variance). With
// heterogeneous means σc > σ0 the old reference spuriously rejected VALID
// estimators (kriging variance ≥ 0) whenever the pre-filter fired. Uses the
// adversarial worked example: A = σ0²·[[1, 1−3e-5],[1−3e-5, 1]] with
// σ0 = sqrt(CORRELOGRAM_DELTA·(1−CORRELOGRAM_DELTA)) ≈ 3.162e-3 (means clamped
// to 1e-5), σc = 0.5, b scaled by σ0·σc.
void test_correlogram_sigma_c_gt_sigma_0_accepted() {
    TEST("lapack_spd_solve_1rhs correlogram σc > σ0 near-dup accepted (R4-01)");
    int size = 2;
    const double sigma0_sq = 1e-5;          // σ0² for means clamped to 1e-5
    const double sigma_c_sq = 0.25;         // σc² for center mean 0.5
    // A = σ0²·[[1, 1−3e-5],[1−3e-5, 1]] — near-dup conditioning data.
    std::vector<double> A = {
        sigma0_sq, sigma0_sq * (1.0 - 3e-5),
        sigma0_sq * (1.0 - 3e-5), sigma0_sq
    };
    // b = C(h_ic)·σ0·σc — cov-to-center ≈ 0.40/0.407 (asymmetric, so the
    // near-null direction [1,-1] is excited and the pre-filter fires).
    std::vector<double> B = { 6.3249e-4, 6.4309e-4 };
    std::vector<double> X(size, 0.0);
    std::vector<double> A_backup(A);

    // Post-fix behavior (target_variance = C(0)·σc² = 0.25): the estimator is
    // VALID (kriging variance σc² − B'X ≥ 0) and must be ACCEPTED.
    bool ok_target = hpgl::detail::lapack_spd_solve_1rhs(
        A.data(), size, X.data(), B.data(),
        A_backup.data(), "test correlogram σc>σ0", sigma_c_sq);
    CHECK(ok_target);

    // The solved system is exact: A·X == B.
    CHECK_CLOSE(A_backup[0] * X[0] + A_backup[1] * X[1], B[0], 1e-9);
    CHECK_CLOSE(A_backup[2] * X[0] + A_backup[3] * X[1], B[1], 1e-9);

    // Mechanism proof: B'X lies between the two references — above σ0²
    // (the pre-fix reference, which made the estimator look "wild") and
    // below σc² (the true target variance — the estimator is valid).
    double bx = B[0] * X[0] + B[1] * X[1];
    CHECK(bx > sigma0_sq);
    CHECK(bx <= sigma_c_sq);
    // Kriging variance non-negative: σc² − B'X ≥ 0.
    CHECK(sigma_c_sq - bx >= -1e-6);

    // Pre-fix behavior (target_variance = 0 → reference A_orig[0] = σ0²):
    // B'X ≫ σ0²·(1+tol) → spuriously rejected as "wild".
    std::vector<double> A2 = A_backup;      // fresh original (A was Cholesky-factored)
    std::vector<double> A2_backup(A_backup);
    std::vector<double> X2(size, 0.0);
    bool ok_default = hpgl::detail::lapack_spd_solve_1rhs(
        A2.data(), size, X2.data(), B.data(),
        A2_backup.data(), "test correlogram σc>σ0 default", 0.0);
    CHECK(!ok_default);
}

// ---- Main ----

int main() {
    // 1rhs tests
    test_1rhs_known_spd_success();
    test_1rhs_known_spd_integer_solution();
    test_1rhs_non_spd_fallback();
    test_1rhs_non_spd_solvable();
    test_1rhs_nan_in_a_rejected();
    test_1rhs_inf_in_a_rejected();

    // 2rhs tests
    test_2rhs_known_spd_success();
    test_2rhs_non_spd_fallback();
    test_2rhs_singular_returns_false();
    test_2rhs_nan_in_a_rejected();

    // II-09: near-singular SPD rejection on the dpotrs_ success path.
    test_1rhs_near_singular_spd_rejected();

    // Stage-8 TEST-ADD (E-H1 2rhs gate, R-01 small-sill OK acceptance,
    // R4-01 correlogram σc > σ0 acceptance).
    test_2rhs_near_singular_spd_rejected();
    test_2rhs_small_sill_ok_accepted();
    test_correlogram_sigma_c_gt_sigma_0_accepted();

    std::printf("C++ solver entry point tests: %d run, %d failed\n",
                g_tests_run, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
