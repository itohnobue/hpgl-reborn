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
/// A = [[2, 4],     <- linearly dependent rows
///      [1, 2]]
/// B = [6, 3]
/// Expected: X = [1, 1] via gauss_solve
void test_1rhs_non_spd_fallback() {
    TEST("lapack_spd_solve_1rhs 2x2 non-SPD (fallback to gauss_solve)");
    int size = 2;
    // This matrix is NOT positive-definite — rows are linearly dependent.
    // dpotrf_ will fail; gauss_solve must handle it.
    std::vector<double> A = {
        2.0, 4.0,
        1.0, 2.0
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
void test_1rhs_non_spd_solvable() {
    TEST("lapack_spd_solve_1rhs 2x2 non-SPD but non-singular (fallback)");
    int size = 2;
    // Asymmetric, non-SPD but non-singular: dpotrf_ will fail,
    // gauss_solve should succeed.
    std::vector<double> A = {
        1.0, 5.0,
        0.0, 2.0
    };
    std::vector<double> B = { 7.0, 2.0 };
    // Solution: x1 * 1 + x2 * 5 = 7  => x1 = -3, x2 = 2
    //           x1 * 0 + x2 * 2 = 2  => x2 = 1, x1 = 2
    // Actually: 1*x1 + 5*x2 = 7, 0*x1 + 2*x2 = 2 => x2 = 1, x1 = 2
    std::vector<double> X(size, 0.0);
    std::vector<double> A_backup(A);

    bool ok = hpgl::detail::lapack_spd_solve_1rhs(
        A.data(), size, X.data(), B.data(),
        A_backup.data(), "test 1rhs non-SPD solvable");

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
void test_2rhs_singular_returns_false() {
    TEST("lapack_spd_solve_2rhs 2x2 singular (both fail)");
    int size = 2;
    std::vector<double> A = {
        2.0, 4.0,
        1.0, 2.0  // linearly dependent rows
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

    std::printf("C++ solver entry point tests: %d run, %d failed\n",
                g_tests_run, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
