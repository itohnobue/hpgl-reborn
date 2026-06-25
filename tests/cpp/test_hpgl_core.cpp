/**
 * Minimal C++ unit tests for HPGL core numerical algorithms.
 *
 * Uses a simple assertion-based framework (no external dependencies) to keep
 * the project minimal per CRITICAL CONSTRAINTS.
 *
 * Tests cover:
 *   - gauss_solve: solve a known 3x3 linear system
 *   - cholesky_decomposition: factor a known positive-definite matrix
 *   - cov_model_t: covariance value at zero lag includes nugget (C(0) = sill)
 *   - cov_model_t: covariance decays with distance
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// TNT matrix library (required by HPGL covariance headers)
#include <tnt.h>

// Include HPGL headers (from the source tree)
#include "src/geo_bsd/hpgl/gauss_solver.h"
#include "src/geo_bsd/hpgl/cov_model.h"

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

void test_gauss_solve_3x3_known_system() {
    TEST("gauss_solve 3x3 known system");
    // Solve:
    //   2x +  y - z = 8
    //  -3x -  y + 2z = -11
    //  -2x +  y + 2z = -3
    // Expected solution: x=2, y=3, z=-1
    int size = 3;
    std::vector<double> A = {
        2.0,  1.0, -1.0,
       -3.0, -1.0,  2.0,
       -2.0,  1.0,  2.0
    };
    std::vector<double> B = { 8.0, -11.0, -3.0 };
    std::vector<double> X(size, 0.0);

    // Make a mutable copy since gauss_solve modifies A in-place
    std::vector<double> A_work = A;
    std::vector<double> B_work = B;

    bool ok = hpgl::gauss_solve(A_work.data(), B_work.data(), X.data(), size);
    CHECK(ok);
    CHECK_CLOSE(X[0],  2.0, 1e-14);
    CHECK_CLOSE(X[1],  3.0, 1e-14);
    CHECK_CLOSE(X[2], -1.0, 1e-14);
}

void test_gauss_solve_2x2_diagonal() {
    TEST("gauss_solve 2x2 diagonal");
    int size = 2;
    std::vector<double> A = {
        3.0, 0.0,
        0.0, 4.0
    };
    std::vector<double> B = { 15.0, 8.0 };
    std::vector<double> X(size, 0.0);

    std::vector<double> A_work = A;
    std::vector<double> B_work = B;

    bool ok = hpgl::gauss_solve(A_work.data(), B_work.data(), X.data(), size);
    CHECK(ok);
    CHECK_CLOSE(X[0], 5.0, 1e-14);
    CHECK_CLOSE(X[1], 2.0, 1e-14);
}

void test_gauss_solve_singular_returns_false() {
    TEST("gauss_solve singular matrix returns false");
    int size = 2;
    // Linearly dependent rows
    std::vector<double> A = {
        2.0, 4.0,
        1.0, 2.0
    };
    std::vector<double> B = { 6.0, 3.0 };
    std::vector<double> X(size, 0.0);

    std::vector<double> A_work = A;
    std::vector<double> B_work = B;

    bool ok = hpgl::gauss_solve(A_work.data(), B_work.data(), X.data(), size);
    CHECK(!ok);  // Should return false for singular matrix
}

void test_cholesky_decomposition_3x3() {
    TEST("cholesky_decomposition 3x3 PD matrix");
    int size = 3;
    // Positive-definite matrix:
    //   4  2  2
    //   2  5  1
    //   2  1  6
    std::vector<double> A = {
        4.0, 2.0, 2.0,
        2.0, 5.0, 1.0,
        2.0, 1.0, 6.0
    };
    std::vector<double> A_U(size * size, 0.0);
    std::vector<double> A_L(size * size, 0.0);
    std::vector<double> A_work = A;

    bool ok = hpgl::cholesky_decomposition(A_work.data(), A_U.data(), A_L.data(), size);
    CHECK(ok);

    // Verify L * U = original A (with L = U^T since it's symmetric)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A_L[i * size + k] * A_U[k * size + j];
            }
            CHECK_CLOSE(sum, A[i * size + j], 1e-12);
        }
    }
}

void test_covariance_zero_lag_equals_sill() {
    TEST("cov_model_t C(0) equals sill");
    // Covariance model: exponential, sill=2.0, nugget=0.5
    // C(0) should be sill = 2.0 (includes nugget effect)
    double ranges_a[3] = { 10.0, 10.0, 10.0 };
    double angles_a[3] = { 0.0, 0.0, 0.0 };
    hpgl::cov_model_t cov(
        hpgl::covariance_type_t::COV_EXPONENTIAL,
        ranges_a, angles_a, 2.0, 0.5);

    double c0 = cov(0.0);
    CHECK_CLOSE(c0, 2.0, 1e-14);  // C(0) = sill

    // Also test coordinate-based call at same point
    double pt1[3] = { 5.0, 3.0, 2.0 };
    double pt2[3] = { 5.0, 3.0, 2.0 };
    double c0_coord = cov(pt1, pt2);
    CHECK_CLOSE(c0_coord, 2.0, 1e-14);
}

void test_covariance_decays_with_distance() {
    TEST("cov_model_t decays with distance");
    double ranges_a[3] = { 5.0, 5.0, 5.0 };
    double angles_a[3] = { 0.0, 0.0, 0.0 };
    hpgl::cov_model_t cov(
        hpgl::covariance_type_t::COV_SPHERICAL,
        ranges_a, angles_a, 1.0, 0.0);

    double c0 = cov(0.0);
    double c2 = cov(2.0);
    double c5 = cov(5.0);
    double c8 = cov(8.0);

    // C(0) = sill = 1.0
    CHECK_CLOSE(c0, 1.0, 1e-14);
    // C(2) < C(0) — should decay
    CHECK(c2 < c0);
    CHECK(c2 > 0.0);
    // C(5) = 0 beyond range for spherical
    CHECK_CLOSE(c5, 0.0, 1e-14);
    // C(8) = 0 beyond range
    CHECK_CLOSE(c8, 0.0, 1e-14);
}

void test_covariance_nugget_contributes_at_lag_zero() {
    TEST("cov_model_t nugget contributes to C(0)");
    double ranges_a[3] = { 10.0, 10.0, 10.0 };
    double angles_a[3] = { 0.0, 0.0, 0.0 };

    // Model with zero nugget
    hpgl::cov_model_t cov_no_nug(
        hpgl::covariance_type_t::COV_GAUSSIAN,
        ranges_a, angles_a, 1.0, 0.0);
    // Model with nugget = 0.3
    hpgl::cov_model_t cov_with_nug(
        hpgl::covariance_type_t::COV_GAUSSIAN,
        ranges_a, angles_a, 1.0, 0.3);

    double c0_no_nug = cov_no_nug(0.0);
    double c0_with_nug = cov_with_nug(0.0);

    // C(0) = sill in both cases (= 1.0, nugget is part of sill)
    CHECK_CLOSE(c0_no_nug, 1.0, 1e-14);
    CHECK_CLOSE(c0_with_nug, 1.0, 1e-14);

    // The key test: at a non-zero distance, the nugget reduces covariance
    // C(h) = (sill - nugget) * rho(h) for h > 0
    // With nugget=0.3: C(h) = 0.7 * rho(h)
    // With nugget=0.0: C(h) = 1.0 * rho(h)
    double h = 2.0;
    double ch_no_nug = cov_no_nug(h);
    double ch_with_nug = cov_with_nug(h);

    // With nugget, the non-zero-lag covariance is reduced
    CHECK(ch_with_nug < ch_no_nug);
    CHECK(ch_with_nug > 0.0);
}

void test_covariance_spherical_at_range_boundary() {
    TEST("cov_model_t spherical at range boundary");
    double ranges_a[3] = { 10.0, 10.0, 10.0 };
    double angles_a[3] = { 0.0, 0.0, 0.0 };
    hpgl::cov_model_t cov(
        hpgl::covariance_type_t::COV_SPHERICAL,
        ranges_a, angles_a, 1.0, 0.0);

    // At range, covariance should be zero
    double c_at_range = cov(10.0);
    CHECK_CLOSE(c_at_range, 0.0, 1e-14);

    // Slightly below range should be positive
    double c_near_range = cov(9.9);
    CHECK(c_near_range > 0.0);
}

void test_cholesky_solve_3x3() {
    TEST("cholesky_solve 3x3 after decomposition");
    int size = 3;
    std::vector<double> A = {
        4.0, 2.0, 2.0,
        2.0, 5.0, 1.0,
        2.0, 1.0, 6.0
    };
    std::vector<double> B = { 10.0, 13.0, 15.0 };
    // Expected solution for Ax = B: x = [1.0, 2.0, 2.0]
    // Check: 4*1 + 2*2 + 2*2 = 4+4+4 = 12? Hmm let me compute properly
    // 4x + 2y + 2z = 10  => 4 + 4 + 4 = 12, not 10.
    // Let me use a simpler RHS: B = A*x for known x.
    // Use x = [1.0, 1.0, 1.0]
    // B = [4+2+2=8, 2+5+1=8, 2+1+6=9]
    // So let me redo...

    std::vector<double> A_U(size * size, 0.0);
    std::vector<double> A_L(size * size, 0.0);
    std::vector<double> A_work = A;

    bool ok = hpgl::cholesky_decomposition(A_work.data(), A_U.data(), A_L.data(), size);
    CHECK(ok);

    // Use a known RHS: x_true = [1, 1, 1], B = A * x_true
    std::vector<double> B_vec = { 8.0, 8.0, 9.0 };
    std::vector<double> X(size, 0.0);

    hpgl::cholesky_solve(A_L.data(), A_U.data(), B_vec.data(), X.data(), size);

    CHECK_CLOSE(X[0], 1.0, 1e-12);
    CHECK_CLOSE(X[1], 1.0, 1e-12);
    CHECK_CLOSE(X[2], 1.0, 1e-12);
}

// ---- Main ----

int main() {
    test_gauss_solve_3x3_known_system();
    test_gauss_solve_2x2_diagonal();
    test_gauss_solve_singular_returns_false();
    test_cholesky_decomposition_3x3();
    test_covariance_zero_lag_equals_sill();
    test_covariance_decays_with_distance();
    test_covariance_nugget_contributes_at_lag_zero();
    test_covariance_spherical_at_range_boundary();
    test_cholesky_solve_3x3();

    std::printf("C++ unit tests: %d run, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
