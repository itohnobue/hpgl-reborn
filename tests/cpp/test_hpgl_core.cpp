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
 *   - C API / writer / handler regression tests (F-05, F-16, F-25, F-34,
 *     F-43, F-52, F-53, F-54, I2-56)
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <thread>
#include <vector>

// TNT matrix library (required by HPGL covariance headers)
#include <tnt.h>

// Include HPGL headers (from the source tree)
#include "src/geo_bsd/hpgl/gauss_solver.h"
#include "src/geo_bsd/hpgl/cov_model.h"
#include "src/geo_bsd/hpgl/gaussian_distribution.h"
#include "src/geo_bsd/hpgl/precalculated_covariance.h"
#include "src/geo_bsd/hpgl/covariance_field.h"
#include "src/geo_bsd/hpgl/indicator_kriging.h"
#include "src/geo_bsd/hpgl/sugarbox_indexed_neighbour_lookup.h"
#include "src/geo_bsd/hpgl/neighbourhood_param.h"
#include "src/geo_bsd/hpgl/hpgl_core.h"
#include "src/geo_bsd/hpgl/kriging_stats.h"
#include "src/geo_bsd/hpgl/sk_params.h"
#include "src/geo_bsd/hpgl/sgs_params.h"
#include "src/geo_bsd/hpgl/sequential_simulation.h"
#include "src/geo_bsd/hpgl/mean_provider.h"
#include "src/geo_bsd/hpgl/my_kriging_weights.h"
#include "src/geo_bsd/hpgl/api.h"
#include "src/geo_bsd/hpgl/api.h"
#include "src/geo_bsd/hpgl/output.h"
#include "src/geo_bsd/hpgl/property_writer.h"
#include "src/geo_bsd/hpgl/property_array.h"

#ifndef _WIN32
#include <unistd.h>
#endif

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

// ---- C API / writer / handler regression tests ----

#ifndef _WIN32
// Creates a unique temporary directory for file I/O tests.
static std::string make_temp_dir()
{
    char tmpl[] = "/tmp/hpgl_cpp_test_XXXXXX";
    char * dir = mkdtemp(tmpl);
    return std::string(dir == nullptr ? "." : dir);
}
#else
static std::string make_temp_dir() { return std::string("."); }
#endif

static void write_inc_text_file(const std::string & path, const char * content)
{
    FILE * f = fopen(path.c_str(), "w");
    if (f != nullptr)
    {
        fputs(content, f);
        fclose(f);
    }
}

static void init_shape(hpgl_shape_t & shape, int nx, int ny, int nz)
{
    shape.m_data[0] = nx;
    shape.m_data[1] = ny;
    shape.m_data[2] = nz;
    shape.m_strides[0] = 0;
    shape.m_strides[1] = 0;
    shape.m_strides[2] = 0;
}

// F-05: write_inc_file_byte with values_count==0 (empty remap values) must use
// the documented identity mapping, not throw "Mismatch: 0 remap values".
void test_write_inc_file_byte_identity_remap_zero_values()
{
    TEST("F-05: write_inc_file_byte values_count=0 uses identity remap");
    std::string dir = make_temp_dir();
    std::string filename = dir + "/identity.inc";
    unsigned char data[4] = {0, 1, 0, 1};
    unsigned char mask[4] = {1, 1, 1, 1};
    hpgl_ind_masked_array_t arr;
    arr.m_data = data;
    arr.m_mask = mask;
    init_shape(arr.m_shape, 2, 2, 1);
    arr.m_indicator_count = 2;
    // Python always passes a non-null pointer, even for an empty array.
    unsigned char empty_values[1] = {0};
    char name[] = "identity";
    char * fname = const_cast<char *>(filename.c_str());
    int rc = hpgl_write_inc_file_byte(fname, &arr, 255, name, empty_values, 0);
    CHECK(rc == 0);  // pre-fix: -1 with "Mismatch: 0 remap values"
    std::remove(filename.c_str());
}

// F-43: write-path functions must reject non-positive shape dims even when the
// volume product is non-negative (e.g. {0,5,5} has volume 0).
void test_write_inc_file_float_rejects_nonpositive_dims()
{
    TEST("F-43: write_inc_file_float rejects non-positive shape dims");
    std::string dir = make_temp_dir();
    std::string filename = dir + "/zerodim.inc";
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    unsigned char mask[4] = {1, 1, 1, 1};
    hpgl_cont_masked_array_t arr;
    arr.m_data = data;
    arr.m_mask = mask;
    init_shape(arr.m_shape, 0, 5, 5);  // volume 0 passes product validation
    char name[] = "zerodim";
    char * fname = const_cast<char *>(filename.c_str());
    int rc = hpgl_write_inc_file_float(fname, &arr, -99.0f, name);
    CHECK(rc != 0);  // pre-fix: volume-0 shape silently succeeds
    std::remove(filename.c_str());
}

// F-25: kriging must reject a pathologically large max_neighbours (only a
// `< 0` check existed; O(max_neighbours) reserve per node makes 2e9 -> ~32GB).
void test_ordinary_kriging_rejects_huge_max_neighbours()
{
    TEST("F-25: ordinary_kriging rejects huge max_neighbours");
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    unsigned char in_mask[4] = {1, 1, 1, 1};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char out_mask[4] = {0, 0, 0, 0};
    hpgl_cont_masked_array_t in, out;
    in.m_data = in_data;
    in.m_mask = in_mask;
    init_shape(in.m_shape, 2, 2, 1);
    out.m_data = out_data;
    out.m_mask = out_mask;
    init_shape(out.m_shape, 2, 2, 1);

    hpgl_ok_params_t params;
    params.m_covariance_type = 0;  // COV_SPHERICAL
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 50000000;  // above the 1e7 upper bound

    hpgl_ordinary_kriging(&in, &params, &out);
    const char * msg = hpgl_get_last_exception_message();
    CHECK(msg != nullptr && strstr(msg, "maximum") != nullptr);
}

// F-34: kriging must reject a zero search radius (which silently mean-fills
// every node); SGS zero-radius CDF draw is exempt by design.
void test_ordinary_kriging_rejects_zero_radius()
{
    TEST("F-34: ordinary_kriging rejects zero search radius");
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    unsigned char in_mask[4] = {1, 1, 1, 1};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char out_mask[4] = {0, 0, 0, 0};
    hpgl_cont_masked_array_t in, out;
    in.m_data = in_data;
    in.m_mask = in_mask;
    init_shape(in.m_shape, 2, 2, 1);
    out.m_data = out_data;
    out.m_mask = out_mask;
    init_shape(out.m_shape, 2, 2, 1);

    hpgl_ok_params_t params;
    params.m_covariance_type = 0;  // COV_SPHERICAL
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 0; params.m_radiuses[1] = 0; params.m_radiuses[2] = 0;
    params.m_max_neighbours = 8;

    hpgl_ordinary_kriging(&in, &params, &out);
    const char * msg = hpgl_get_last_exception_message();
    CHECK(msg != nullptr && strstr(msg, "radius") != nullptr);
}

// F-52: a write that fails mid-stream (NaN value) must not leave <file>.tmp
// behind on the failure path.
void test_property_writer_removes_tmp_on_failure()
{
    TEST("F-52: failed write leaves no .tmp file behind");
    std::string dir = make_temp_dir();
    std::string filename = dir + "/leak_test.inc";
    float data[3] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    unsigned char mask[3] = {1, 1, 1};
    hpgl::cont_property_array_t prop(data, mask, 3);
    hpgl::property_writer_t writer;
    writer.init(filename, "leak");
    try
    {
        writer.write_double(prop, -99.0);
    }
    catch (const hpgl::hpgl_exception &)
    {
        // expected — non-finite values are rejected
    }
    FILE * tmp = fopen((filename + ".tmp").c_str(), "r");
    CHECK(tmp == nullptr);  // pre-fix: .tmp remains on disk
    if (tmp != nullptr)
        fclose(tmp);
    std::remove(filename.c_str());
}

// F-16: a concurrent hpgl_set_output_handler must block while a handler is
// being invoked, so it cannot free the trampoline/param mid-invoke.
static std::atomic<bool> g_slow_handler_started{false};
static int slow_handler(char * data, void * param)
{
    (void)data;
    (void)param;
    g_slow_handler_started = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 0;
}

void test_handler_setter_blocks_during_invocation()
{
    TEST("F-16: set_output_handler blocks during in-flight invocation");
    g_slow_handler_started = false;
    hpgl_set_output_handler(slow_handler, nullptr);
    std::thread writer([] { hpgl::write("blocking-write"); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!g_slow_handler_started.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(g_slow_handler_started.load());

    auto start = std::chrono::steady_clock::now();
    hpgl_set_output_handler(nullptr, nullptr);  // must block ~100ms
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    writer.join();
    CHECK(elapsed_ms >= 50);  // pre-fix: setter returns immediately
}

// F-53: a handler that calls back into HPGL (re-entrant hpgl::write) must not
// self-deadlock.
static std::atomic<bool> g_reentrant_ran{false};
static int reentrant_handler(char * data, void * param)
{
    (void)data;
    (void)param;
    hpgl::write("inner-from-handler");  // re-entrant call
    g_reentrant_ran = true;
    return 0;
}

void test_reentrant_handler_no_deadlock()
{
    TEST("F-53: re-entrant handler does not deadlock");
    g_reentrant_ran = false;
    hpgl_set_output_handler(reentrant_handler, nullptr);
    std::future<void> fut = std::async(std::launch::async, [] { hpgl::write("outer"); });
    bool completed = fut.wait_for(std::chrono::seconds(3)) == std::future_status::ready;
    hpgl_set_output_handler(nullptr, nullptr);
    CHECK(completed);  // pre-fix: deadlocks (times out)
    CHECK(g_reentrant_ran.load());
}

// F-54: a mid-line '/' token must be skipped (Python slow parser semantics),
// not terminate the data read.
void test_read_inc_file_skips_midline_slash()
{
    TEST("F-54: mid-line '/' token is skipped");
    std::string dir = make_temp_dir();
    std::string filename = dir + "/midline.inc";
    write_inc_text_file(filename, "mid\n1 2 / 3 4\n/\n");
    float data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char mask[4] = {0, 0, 0, 0};
    char * fname = const_cast<char *>(filename.c_str());
    int rc = hpgl_read_inc_file_float(fname, -99.0f, 4, data, mask);
    CHECK(rc == 0);  // pre-fix: -1 "Unexpected end of data."
    CHECK_CLOSE(data[0], 1.0f, 1e-6);
    CHECK_CLOSE(data[1], 2.0f, 1e-6);
    CHECK_CLOSE(data[2], 3.0f, 1e-6);
    CHECK_CLOSE(data[3], 4.0f, 1e-6);
    std::remove(filename.c_str());
}

// I2-56: the fast reader must not silently truncate trailing tokens beyond
// `size`; it must raise on a count mismatch like the slow parser path.
void test_read_inc_file_rejects_extra_tokens()
{
    TEST("I2-56: fast reader rejects extra tokens beyond size");
    std::string dir = make_temp_dir();
    std::string filename = dir + "/extra.inc";
    write_inc_text_file(filename, "extra\n1\n2\n3\n4\n5\n6\n/\n");
    float data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char mask[4] = {0, 0, 0, 0};
    char * fname = const_cast<char *>(filename.c_str());
    int rc = hpgl_read_inc_file_float(fname, -99.0f, 4, data, mask);
    CHECK(rc != 0);  // pre-fix: rc == 0 (silently truncated to [1,2,3,4])
    std::remove(filename.c_str());
}

// Sanity: a legitimate file with exactly `size` values plus a '/' terminator
// still loads (guards against over-tightening the count check).
void test_read_inc_file_exact_size_with_terminator()
{
    TEST("read_inc_file_float exact size with '/' terminator");
    std::string dir = make_temp_dir();
    std::string filename = dir + "/exact.inc";
    write_inc_text_file(filename, "exact\n1\n2\n3\n4\n/\n");
    float data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char mask[4] = {0, 0, 0, 0};
    char * fname = const_cast<char *>(filename.c_str());
    int rc = hpgl_read_inc_file_float(fname, -99.0f, 4, data, mask);
    CHECK(rc == 0);
    CHECK_CLOSE(data[0], 1.0f, 1e-6);
    CHECK_CLOSE(data[3], 4.0f, 1e-6);
    std::remove(filename.c_str());
}

// F-14: m_min_neighbours must be wired into sequential Gaussian simulation
// (GSLIB ndmin semantics): nodes with fewer than m_min_neighbours
// conditioning data are left unsimulated rather than simulated from the
// marginal distribution. Pre-fix code ignored m_min_neighbours entirely
// (stored/printed but never used) — every uninformed node was simulated.
void test_sgs_min_neighbours_wired() {
    TEST("SGS m_min_neighbours leaves sparse nodes unsimulated (F-14)");
    // 4×4×1 grid, two informed cells far apart so most nodes have zero
    // neighbours at radius 1.
    float data[16] = {1.0f, 0, 0, 0,
                      0, 0, 0, 0,
                      0, 0, 0, 0,
                      0, 0, 0, 5.0f};
    unsigned char mask[16] = {1, 0, 0, 0,
                              0, 0, 0, 0,
                              0, 0, 0, 0,
                              0, 0, 0, 1};
    float out_data[16] = {0};
    unsigned char out_mask[16] = {0};
    hpgl_cont_masked_array_t in, out;
    in.m_data = data;
    in.m_mask = mask;
    init_shape(in.m_shape, 4, 4, 1);
    out.m_data = out_data;
    out.m_mask = out_mask;
    init_shape(out.m_shape, 4, 4, 1);

    hpgl_sgs_params_t params;
    params.m_covariance_type = 0;  // COV_SPHERICAL
    params.m_ranges[0] = 3; params.m_ranges[1] = 3; params.m_ranges[2] = 3;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 8;
    params.m_kriging_kind = 1;  // KRIG_SIMPLE
    params.m_seed = 42;
    params.m_min_neighbours = 2;  // ndmin: 2 conditioning data required

    double mean = 0.0;
    // NOTE: hpgl_sgs_simulation writes the result in-place into `in`
    // (the cont_masked_array_t passed as `data`), not into `out`.
    hpgl_sgs_simulation(&in, &params, nullptr, &mean, nullptr);

    // Radius 1: only nodes adjacent to an informed cell have >= 1
    // neighbour; most have 0 < min_neighbours(2). With the F-14 fix,
    // those nodes must remain unsimulated (in.m_mask == 0). Pre-fix, all
    // uninformed nodes were simulated from the marginal (in.m_mask == 1).
    // Node index: idx = z*16 + y*4 + x.
    // (1,1,0) → idx 5: 0 neighbours → unsimulated.
    CHECK(in.m_mask[5] == 0);
    // (2,0,0) → idx 8: 0 neighbours → unsimulated.
    CHECK(in.m_mask[8] == 0);
    // (2,1,0) → idx 9: 0 neighbours → unsimulated.
    CHECK(in.m_mask[9] == 0);
    // (1,0,0) → idx 1: 1 neighbour (node 0) < 2 → unsimulated.
    CHECK(in.m_mask[1] == 0);
    // Hard data are preserved.
    CHECK(in.m_mask[0] == 1);
    CHECK(in.m_mask[15] == 1);
}

// F-04 / I2-27: gaussian_cdf_t::inverse must SATURATE the tails instead of
// returning the mean for p >= 1.0 (max datum → p=1.0f → normal score 0.0
// → median collapse on round-trip) or p <= 0.0 (LVM mean below data-min →
// p=0.0 → same collapse). Pre-fix code returned m_mean (0.0) for both.
void test_gaussian_inverse_tail_saturation() {
    TEST("gaussian_cdf_t::inverse tail saturation (F-04/I2-27)");
    hpgl::gaussian_cdf_t cdf;  // N(0,1)
    // High side: p = 1.0 (or float32-rounded > 1.0) must map to a large
    // positive finite quantile, NOT to the mean (0.0).
    double q_high = cdf.inverse(1.0);
    CHECK(std::isfinite(q_high));
    CHECK(q_high > 3.0);                 // saturated tail, not the median
    CHECK(q_high > cdf.inverse(0.999));  // monotone even at the boundary
    // Low side: p = 0.0 must map to a large negative finite quantile.
    double q_low = cdf.inverse(0.0);
    CHECK(std::isfinite(q_low));
    CHECK(q_low < -3.0);
    CHECK(q_low < cdf.inverse(0.001));
    // Round-trip sanity: the saturated quantile inverts back to ~1.0.
    CHECK_CLOSE(cdf.prob(q_high), 1.0, 1e-9);
    CHECK_CLOSE(cdf.prob(q_low), 0.0, 1e-9);
}

// F-20: correct_order_relations must implement the GSLIB ORDREL two-pass
// envelope (clip → upward fill-forward → downward fill-backward → average)
// and normalize the resulting CDF so the recovered PDF sums to 1.0.
// Pre-fix code used iterative pairwise averaging (D&J 1st ed.) which
// diverges on cascading violations.
void test_order_relations_gslib_envelope() {
    TEST("correct_order_relations GSLIB two-pass envelope (F-20)");
    std::vector<hpgl::indicator_probability_t> probs = {0.9f, 0.2f, 0.6f, 0.4f, 1.0f};
    // Manual GSLIB ordrel on this input:
    // clip: [0.9, 0.2, 0.6, 0.4, 1.0]
    // up pass (ccdf1): [0.9, 0.9, 0.9, 0.9, 1.0]
    // down pass (ccdf2): [0.2, 0.2, 0.4, 0.4, 1.0]
    //   (i=4: c2[3]=0.4>c2[4]=1.0? no; i=3: c2[2]=0.6>c2[3]=0.4 → 0.4;
    //    i=2: c2[1]=0.2>c2[2]=0.4? no; i=1: c2[0]=0.9>c2[1]=0.2 → 0.2)
    // average: [0.55, 0.55, 0.65, 0.65, 1.0]
    hpgl::detail::correct_order_relations(probs);
    // Monotonic non-decreasing.
    for (size_t i = 1; i < probs.size(); ++i) {
        CHECK(probs[i] >= probs[i - 1]);
    }
    // Bounded [0,1].
    for (size_t i = 0; i < probs.size(); ++i) {
        CHECK(probs[i] >= 0.0f && probs[i] <= 1.0f);
    }
    // Normalized: the CDF ends at 1.0 (recovered PDF sums to 1.0).
    CHECK_CLOSE(probs.back(), 1.0f, 1e-5f);
    // Exact two-pass envelope values (up/down average, then normalize to 1).
    CHECK_CLOSE(probs[0], 0.55f / 1.0f, 1e-4f);
    CHECK_CLOSE(probs[1], 0.55f / 1.0f, 1e-4f);
    CHECK_CLOSE(probs[2], 0.65f / 1.0f, 1e-4f);
    CHECK_CLOSE(probs[3], 0.65f / 1.0f, 1e-4f);
}

// F-21: precalculated_covariances_t must return the EXACT covariance for
// data-to-data pairs beyond the search box instead of truncating to 0.
// Pre-fix operator() returned 0 for out-of-box pairs, making the kriging
// LHS inconsistent with the exact RHS.
void test_precalculated_covariance_exact_beyond_box() {
    TEST("precalculated_covariances_t exact covariance beyond box (F-21)");
    double ranges[3] = {10.0, 10.0, 10.0};
    double angles[3] = {0.0, 0.0, 0.0};
    hpgl::cov_model_t model(hpgl::covariance_type_t::COV_EXPONENTIAL,
                            ranges, angles, 1.0, 0.0);
    hpgl::xyz_tuple_t<double> radiuses(2.0, 2.0, 2.0);
    hpgl::precalculated_covariances_t pcov(model, radiuses);
    // Two points 5 apart — outside the ±2 box, inside exponential support.
    double p1[3] = {0.0, 0.0, 0.0};
    double p2[3] = {5.0, 0.0, 0.0};
    double exact = model(p1, p2);        // reference: exp(-3*5/10) = e^-1.5
    double from_functor = pcov(p1, p2);
    CHECK_CLOSE(exact, 0.2231301601, 1e-9);  // e^-1.5
    CHECK_CLOSE(from_functor, exact, 1e-12); // functor now exact, not 0
    // In-box pair still uses the precomputed table.
    double p3[3] = {1.0, 0.0, 0.0};
    CHECK_CLOSE(pcov(p1, p3), model(p1, p3), 1e-12);
}

// I2-24: precalculated_covariances_t must throw a catchable hpgl_exception
// instead of abort() when the covariance volume exceeds INT_MAX.
// Pre-fix code called abort() → uncatchable SIGABRT for radius >= 645.
void test_precalculated_covariance_throws_on_huge_volume() {
    TEST("precalculated_covariances_t throws on huge volume (I2-24)");
    double ranges[3] = {10.0, 10.0, 10.0};
    double angles[3] = {0.0, 0.0, 0.0};
    hpgl::cov_model_t model(hpgl::covariance_type_t::COV_EXPONENTIAL,
                            ranges, angles, 1.0, 0.0);
    hpgl::xyz_tuple_t<double> huge_radiuses(700.0, 700.0, 700.0);
    bool threw = false;
    try {
        hpgl::precalculated_covariances_t pcov(model, huge_radiuses);
    } catch (const hpgl::hpgl_exception &) {
        threw = true;
    }
    CHECK(threw);
}

// F-09: the clusterizer fast path must only return nodes within the
// configured search radius. Pre-fix find() collected the whole 3×3×3
// cluster box (~2×radius) and filtered only by cov > sill/100, so
// beyond-radius nodes entered the ordinary-kriging neighbourhood for
// infinite-support covariance models.
void test_indexed_lookup_radius_bound() {
    TEST("indexed_neighbour_lookup_t radius bound on fast path (F-09)");
    // Geometry: radius 5, center at (12,12,12). The center sits in cluster
    // cell (2,2,2) ([10,15) per axis); the 3×3×3 cluster box spans cells
    // (1..3) = [5,20) per axis. A hard datum at x=19 (cell 3) is INSIDE the
    // cluster box (distance 7 > radius 5) and its exponential covariance
    // (range 8, sill 1) at h=7 is exp(-3*7/8) ≈ 0.072 > sill/100 = 0.01,
    // so pre-fix find() admits it. The radius check must exclude it.
    hpgl::sugarbox_grid_t grid;
    grid.init(25, 25, 25);
    double ranges[3] = {8.0, 8.0, 8.0};   // exponential: infinite support
    double angles[3] = {0.0, 0.0, 0.0};
    hpgl::cov_model_t cov(hpgl::covariance_type_t::COV_EXPONENTIAL,
                          ranges, angles, 1.0, 0.0);
    hpgl::neighbourhood_param_t nb;
    nb.set_radiuses(5, 5, 5);   // search radius 5
    nb.m_max_neighbours = 64;

    hpgl::indexed_neighbour_lookup_t<hpgl::sugarbox_grid_t, hpgl::cov_model_t>
        lookup(&grid, &cov, nb);

    struct all_defined_t { bool operator()(hpgl::node_index_t) const { return true; } };
    all_defined_t all_defined;

    auto add = [&](int x, int y, int z) {
        hpgl::sugarbox_location_t loc(x, y, z);
        hpgl::node_index_t idx = grid.get_index(loc);
        lookup.add_node(idx);
        return idx;
    };
    hpgl::node_index_t center_idx = add(12, 12, 12);
    hpgl::node_index_t d1 = add(13, 12, 12);  // distance 1 ≤ 5
    hpgl::node_index_t d2 = add(14, 12, 12);  // distance 2 ≤ 5
    hpgl::node_index_t d3 = add(19, 12, 12);  // distance 7 > 5, in cluster box

    hpgl::sugarbox_location_t node_coord;
    std::vector<hpgl::node_index_t> indices;
    std::vector<hpgl::sugarbox_location_t> coords;
    lookup.find(center_idx, all_defined, node_coord, indices, coords);

    bool found_d3 = false;
    bool found_d1 = false;
    bool found_d2 = false;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] == d3) found_d3 = true;
        if (indices[i] == d1) found_d1 = true;
        if (indices[i] == d2) found_d2 = true;
    }
    CHECK(found_d1);   // within radius → present
    CHECK(found_d2);   // within radius → present
    CHECK(!found_d3);  // beyond radius → MUST be excluded (F-09)
}

// F-19/F-22: simple cokriging must (a) populate kriging_stats_t (F-19) and
// (b) drop the secondary equation when the secondary is missing at the
// target (F-22). We run cokriging on a tiny grid with a fully-missing
// secondary: the result must equal plain primary kriging (no phantom
// secondary equation), and stats must be observable.
void test_cokriging_stats_and_missing_secondary() {
    TEST("simple_cokriging stats + missing-secondary handling (F-19/F-22)");
    hpgl::sugarbox_grid_t grid;
    grid.init(4, 4, 1);

    const int n = 16;
    std::vector<float> primary_data(n, 0.0f);
    std::vector<unsigned char> primary_mask(n, 0);
    // Hard data: two informed cells.
    primary_data[0] = 1.0f;  primary_mask[0] = 1;
    primary_data[1] = 3.0f;  primary_mask[1] = 1;
    // Secondary: fully missing (all mask 0) → secondary_present=false
    // everywhere → the secondary equation must be dropped (F-22).
    std::vector<float> secondary_data(n, 0.0f);
    std::vector<unsigned char> secondary_mask(n, 0);
    std::vector<float> output_data(n, 0.0f);
    std::vector<unsigned char> output_mask(n, 0);

    hpgl::cont_property_array_t primary(primary_data.data(), primary_mask.data(), n);
    hpgl::cont_property_array_t secondary(secondary_data.data(), secondary_mask.data(), n);
    hpgl::cont_property_array_t output(output_data.data(), output_mask.data(), n);

    hpgl::neighbourhood_param_t nb;
    nb.set_radiuses(1, 1, 0);
    nb.m_max_neighbours = 8;

    hpgl::covariance_param_t cp;
    cp.m_covariance_type = hpgl::covariance_type_t::COV_EXPONENTIAL;
    cp.set_ranges(3.0, 3.0, 3.0);
    cp.set_angles(0.0, 0.0, 0.0);
    cp.set_sill(1.0);
    cp.set_nugget(0.0);

    hpgl::simple_cokriging_markI(grid, primary, secondary,
        0.0f, 0.0f, 1.0, 0.5, nb, cp, output);

    // F-19: failure counters observable via the public C API.
    hpgl_kriging_stats_t stats = hpgl_get_kriging_stats();
    // 14 uninformed nodes; all have >= 1 neighbour within radius 1 on a
    // 4×4 grid, so points_calculated should be > 0.
    CHECK(stats.m_points_calculated > 0);

    // F-22: with a fully-missing secondary, the estimate is plain primary
    // kriging. Compare against simple_kriging (primary-only) on the same
    // data — outputs must be close.
    std::vector<float> sk_output_data(n, 0.0f);
    std::vector<unsigned char> sk_output_mask(n, 0);
    hpgl::cont_property_array_t sk_output(sk_output_data.data(), sk_output_mask.data(), n);
    hpgl::sk_params_t skp;
    skp.m_covariance_type = hpgl::covariance_type_t::COV_EXPONENTIAL;
    skp.set_ranges(3.0, 3.0, 3.0);
    skp.set_angles(0.0, 0.0, 0.0);
    skp.set_sill(1.0);
    skp.set_nugget(0.0);
    skp.set_radiuses(1, 1, 0);
    skp.m_max_neighbours = 8;
    skp.set_mean(0.0);
    hpgl::simple_kriging(primary, grid, skp, sk_output);

    for (int i = 0; i < n; ++i) {
        if (output_mask[i] && sk_output_mask[i]) {
            CHECK_CLOSE(output_data[i], sk_output_data[i], 1e-3);
        }
    }
}

// PR-05 (F-34 partial): hpgl_indicator_kriging must reject a zero search
// radius on the >=3-category path (the 2-category path redirects to
// median_ik, which is already guarded). Pre-fix: the radius guard was not
// wired into hpgl_indicator_kriging, so radius-0 silently mean-filled.
// This is the only kriging entry point (of 7) missing the guard.
void test_indicator_kriging_rejects_zero_radius() {
    TEST("PR-05: indicator_kriging rejects zero search radius");
    const int cells = 4;          // 2x2x1 grid
    const int cats = 3;           // >= 3 categories → real indicator path
    unsigned char in_data[cells * cats] = {0,1,2, 0,1,2, 0,1,2, 0,1,2};
    unsigned char in_mask[cells * cats] = {1,1,1, 1,1,1, 1,1,1, 1,1,1};
    unsigned char out_data[cells * cats] = {0};
    unsigned char out_mask[cells * cats] = {0};
    hpgl_ind_masked_array_t in, out;
    in.m_data = in_data;
    in.m_mask = in_mask;
    init_shape(in.m_shape, 2, 2, 1);
    in.m_indicator_count = cats;
    out.m_data = out_data;
    out.m_mask = out_mask;
    init_shape(out.m_shape, 2, 2, 1);
    out.m_indicator_count = cats;

    hpgl_ik_params_t params[cats];
    for (int i = 0; i < cats; ++i) {
        params[i].m_covariance_type = 0;  // COV_SPHERICAL
        params[i].m_ranges[0] = 1; params[i].m_ranges[1] = 1; params[i].m_ranges[2] = 1;
        params[i].m_angles[0] = 0; params[i].m_angles[1] = 0; params[i].m_angles[2] = 0;
        params[i].m_sill = 1.0;
        params[i].m_nugget = 0.0;
        params[i].m_radiuses[0] = 0; params[i].m_radiuses[1] = 0; params[i].m_radiuses[2] = 0;
        params[i].m_max_neighbours = 8;
        params[i].m_marginal_prob = 1.0 / cats;
    }

    hpgl_indicator_kriging(&in, &out, params, cats);
    // Distinguish from any stale message left by an earlier test: the
    // guard's exception context is "hpgl_indicator_kriging".
    const char * msg = hpgl_get_last_exception_message();
    CHECK(msg != nullptr && strstr(msg, "hpgl_indicator_kriging") != nullptr);
}

// PR-06 (F-21 sibling): covariance_field_t::operator() must return the EXACT
// covariance for data-to-data pairs beyond the search box instead of
// truncating to 0. Pre-fix operator() returned 0 for out-of-box pairs,
// making the median_ik kriging LHS inconsistent with the exact RHS.
void test_covariance_field_exact_beyond_box() {
    TEST("covariance_field_t exact covariance beyond box (PR-06)");
    double ranges[3] = {10.0, 10.0, 10.0};
    double angles[3] = {0.0, 0.0, 0.0};
    hpgl::cov_model_t model(hpgl::covariance_type_t::COV_EXPONENTIAL,
                            ranges, angles, 1.0, 0.0);
    hpgl::covariance_field_t field(2, 2, 2, model);
    // Two points 5 apart — outside the ±2 box, inside exponential support.
    hpgl::sugarbox_location_t p1(0, 0, 0);
    hpgl::sugarbox_location_t p2(5, 0, 0);
    double exact = model(p1, p2);           // reference: exp(-3*5/10) = e^-1.5
    double from_functor = field(p1, p2);
    CHECK_CLOSE(exact, 0.2231301601, 1e-9); // e^-1.5
    CHECK_CLOSE(from_functor, exact, 1e-12); // functor now exact, not 0
    // In-box pair still uses the precomputed table.
    hpgl::sugarbox_location_t p3(1, 0, 0);
    CHECK_CLOSE(field(p1, p3), model(p1, p3), 1e-12);
}

// PR-07 (F-25/I2-23 bound mismatch): the C-API cokriging entry point must
// reject max_neighbours in (1e5, 1e7] with a clear error. Pre-fix the API
// bound was 1e7 while the solver silently degraded (KI_SINGULARITY →
// mean-fill) above 1e5, so 200000 was accepted then silently mean-filled.
void test_cokriging_rejects_huge_max_neighbours() {
    TEST("PR-07: cokriging rejects max_neighbours above internal limit");
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    unsigned char in_mask[4] = {1, 1, 1, 1};
    float sec_data[4] = {4.0f, 3.0f, 2.0f, 1.0f};
    unsigned char sec_mask[4] = {1, 1, 1, 1};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char out_mask[4] = {0, 0, 0, 0};
    hpgl_cont_masked_array_t in, sec, out;
    in.m_data = in_data;  in.m_mask = in_mask;  init_shape(in.m_shape, 2, 2, 1);
    sec.m_data = sec_data; sec.m_mask = sec_mask; init_shape(sec.m_shape, 2, 2, 1);
    out.m_data = out_data; out.m_mask = out_mask; init_shape(out.m_shape, 2, 2, 1);

    hpgl_cokriging_m1_params_t params;
    params.m_covariance_type = 0;  // COV_SPHERICAL
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 200000;  // within (1e5, 1e7] — pre-fix accepted
    params.m_primary_mean = 0.0;
    params.m_secondary_mean = 0.0;
    params.m_secondary_variance = 1.0;
    params.m_correlation_coef = 0.5;

    hpgl_simple_cokriging_mark1(&in, &sec, &params, &out);
    // Distinguish from any stale message: the entry point's exception
    // context is "hpgl_simple_cokriging_mark1".
    const char * msg = hpgl_get_last_exception_message();
    CHECK(msg != nullptr && strstr(msg, "hpgl_simple_cokriging_mark1") != nullptr);
}

// ---- FIX-stage regression tests (s6-fix-cpp-engines) ----

// M-3: GSLIB sgsim downgrades OK→SK when fewer than 4 conditioning data are
// available (`if(ktype.eq.1.and.(nclose+ncnode).lt.4)lktype=0`). The OK solve's
// Lagrange-multiplier variance correction inflates variance for tiny
// neighbourhoods (a nugget-only model gives 2×sill for n=1); the fix delegates
// R-5 / R-4 (M-3 reconciliation): the GSLIB sgsim OK→SK <4-data downgrade is
// SGS-ONLY. The shared ok_kriging_weights_3(_ws) calculators keep the public
// hpgl_ordinary_kriging (the kt3d analog) OK contract for n<4 — weights sum
// to 1 — while ok_sgs_weight_calculator_t delegates to the SK solve below 4
// data (GSLIB: `if(ktype.eq.1.and.(nclose+ncnode).lt.4)lktype=0`) so the
// SGS-OK variance is not inflated by the Lagrange multiplier.
void test_ok_sgs_only_downgrades_under_4_neighbours() {
    TEST("R-5/M-3: OK→SK downgrade is SGS-only — public OK sums to 1, sgs calculator downgrades");
    double ranges[3] = {10.0, 10.0, 10.0};
    double angles[3] = {0.0, 0.0, 0.0};
    hpgl::cov_model_t cov(hpgl::covariance_type_t::COV_EXPONENTIAL,
                          ranges, angles, 1.0, 0.0);

    // n=2: public OK weights must sum to 1 (contract preserved).
    {
        std::vector<hpgl::sugarbox_location_t> coords(2, hpgl::sugarbox_location_t(0, 0, 0));
        coords[0] = hpgl::sugarbox_location_t(1, 0, 0);
        coords[1] = hpgl::sugarbox_location_t(2, 0, 0);
        std::vector<hpgl::kriging_weight_t> ok_weights;
        double ok_var = 0.0;
        bool ok = hpgl::ok_kriging_weights_3<hpgl::cov_model_t, true, hpgl::sugarbox_location_t>(
            hpgl::sugarbox_location_t(0, 0, 0), coords, cov, ok_weights, ok_var);
        CHECK(ok);
        double sum = 0.0;
        for (size_t i = 0; i < ok_weights.size(); ++i) sum += ok_weights[i];
        CHECK_CLOSE(sum, 1.0, 1e-9);
    }
    // Workspace public OK variant: same sum-to-1 contract.
    {
        std::vector<hpgl::sugarbox_location_t> coords(2, hpgl::sugarbox_location_t(0, 0, 0));
        coords[0] = hpgl::sugarbox_location_t(1, 0, 0);
        coords[1] = hpgl::sugarbox_location_t(2, 0, 0);
        hpgl::weight_calc_workspace_t ws;
        std::vector<hpgl::kriging_weight_t> ok_weights;
        double ok_var = 0.0;
        bool ok = hpgl::ok_kriging_weights_3_ws<hpgl::cov_model_t, true, hpgl::sugarbox_location_t>(
            hpgl::sugarbox_location_t(0, 0, 0), coords, cov, ok_weights, ok_var, ws);
        CHECK(ok);
        double sum = 0.0;
        for (size_t i = 0; i < ok_weights.size(); ++i) sum += ok_weights[i];
        CHECK_CLOSE(sum, 1.0, 1e-9);
    }
    // n=2: the SGS-OK calculator delegates to SK — weights and variance must
    // match sk_kriging_weights_3 exactly (and need not sum to 1).
    {
        std::vector<hpgl::sugarbox_location_t> coords(2, hpgl::sugarbox_location_t(0, 0, 0));
        coords[0] = hpgl::sugarbox_location_t(1, 0, 0);
        coords[1] = hpgl::sugarbox_location_t(2, 0, 0);
        std::vector<hpgl::kriging_weight_t> sk_weights;
        double sk_var = 0.0;
        bool sk_ok = hpgl::sk_kriging_weights_3<hpgl::cov_model_t, true, hpgl::sugarbox_location_t>(
            hpgl::sugarbox_location_t(0, 0, 0), coords, cov, sk_weights, sk_var);
        CHECK(sk_ok);
        hpgl::ok_sgs_weight_calculator_t sgs_wc;
        std::vector<hpgl::kriging_weight_t> sgs_weights;
        double sgs_var = 0.0;
        bool sgs_ok = sgs_wc(hpgl::sugarbox_location_t(0, 0, 0),
            coords, cov, sgs_weights, sgs_var);
        CHECK(sgs_ok);
        CHECK(sgs_weights.size() == sk_weights.size());
        for (size_t i = 0; i < sgs_weights.size(); ++i)
            CHECK_CLOSE(sgs_weights[i], sk_weights[i], 1e-9);
        CHECK_CLOSE(sgs_var, sk_var, 1e-9);
    }
    // n=4: the SGS-OK calculator applies full OK — weights sum to 1.
    {
        std::vector<hpgl::sugarbox_location_t> coords(4, hpgl::sugarbox_location_t(0, 0, 0));
        for (int i = 0; i < 4; ++i)
            coords[i] = hpgl::sugarbox_location_t(i + 1, 0, 0);
        hpgl::ok_sgs_weight_calculator_t sgs_wc;
        std::vector<hpgl::kriging_weight_t> sgs_weights;
        double sgs_var = 0.0;
        bool sgs_ok = sgs_wc(hpgl::sugarbox_location_t(0, 0, 0),
            coords, cov, sgs_weights, sgs_var);
        CHECK(sgs_ok);
        double sum = 0.0;
        for (size_t i = 0; i < sgs_weights.size(); ++i) sum += sgs_weights[i];
        CHECK_CLOSE(sum, 1.0, 1e-9);
    }
    // Pure-nugget n=1: public OK keeps the (kt3d) inflated variance 2×sill;
    // the SGS-OK calculator downgrades to SK → variance == sill (not 2×sill).
    {
        hpgl::cov_model_t nug_cov(hpgl::covariance_type_t::COV_SPHERICAL,
                                  ranges, angles, 1.0, 1.0);  // nugget == sill
        std::vector<hpgl::sugarbox_location_t> coords(1, hpgl::sugarbox_location_t(1, 0, 0));
        std::vector<hpgl::kriging_weight_t> ok_weights;
        double ok_var = 0.0;
        bool ok = hpgl::ok_kriging_weights_3<hpgl::cov_model_t, true, hpgl::sugarbox_location_t>(
            hpgl::sugarbox_location_t(0, 0, 0), coords, nug_cov, ok_weights, ok_var);
        CHECK(ok);
        CHECK_CLOSE(ok_var, 2.0, 1e-9);   // public OK: inflated (kt3d semantics)
        hpgl::ok_sgs_weight_calculator_t sgs_wc;
        std::vector<hpgl::kriging_weight_t> sgs_weights;
        double sgs_var = 0.0;
        bool sgs_ok = sgs_wc(hpgl::sugarbox_location_t(0, 0, 0),
            coords, nug_cov, sgs_weights, sgs_var);
        CHECK(sgs_ok);
        CHECK_CLOSE(sgs_var, 1.0, 1e-9);  // SGS downgrade: SK variance == sill
    }
}

// M-5: `#pragma omp cancel for` is a silent no-op in default builds (cancel-var
// ICV false), so user cancellation never stopped the kriging loop. The fix uses
// a shared atomic loop-stop flag checked per iteration. A progress handler that
// returns non-zero cancels; after cancellation the loop must stop early —
// leaving many output cells uninformed — instead of processing every cell.
static int cancel_progress_handler(char *, int, void *) { return 1; }  // cancel

void test_user_cancellation_stops_kriging_loop() {
    TEST("M-5: user cancellation stops the kriging loop in a default build");
    // Grid sized so the first per-thread batch flush (LP_BATCH_SIZE=1000 laps)
    // is guaranteed to happen before the whole grid is processed even on very
    // high-core machines: the first flush fires at the earliest thread reaching
    // 1000 laps, when at most num_threads×1000 cells have been touched.
    // num_threads is bounded by OMP max (~512) → 600000 > 512×1000.
    const int nx = 100, ny = 100, nz = 60;
    const int nodes = nx * ny * nz;
    std::vector<float> in_data(nodes);
    std::vector<unsigned char> in_mask(nodes, 0);
    // 70% informed — leaves ~180000 uninformed cells that kriging would fill.
    for (int i = 0; i < nodes; ++i) {
        if (i % 10 < 7) { in_data[i] = 1.0f; in_mask[i] = 1; }
    }
    std::vector<float> out_data(nodes, 0.0f);
    std::vector<unsigned char> out_mask(nodes, 0);
    hpgl_cont_masked_array_t in, out;
    in.m_data = in_data.data(); in.m_mask = in_mask.data();
    init_shape(in.m_shape, nx, ny, nz);
    out.m_data = out_data.data(); out.m_mask = out_mask.data();
    init_shape(out.m_shape, nx, ny, nz);

    hpgl_ok_params_t params;
    params.m_covariance_type = 0;  // COV_SPHERICAL
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 8;

    hpgl_set_progress_handler(cancel_progress_handler, nullptr);
    hpgl_ordinary_kriging(&in, &params, &out);
    hpgl_set_progress_handler(nullptr, nullptr);  // reset BEFORE any CHECK

    int informed = 0;
    for (int i = 0; i < nodes; ++i)
        if (out_mask[i]) ++informed;
    // Pre-fix: omp cancel is a no-op → all 15000 uninformed cells get kriged
    // (mask=1) → informed == nodes. Post-fix: the loop stops shortly after the
    // first batch flush observes the cancel → many cells remain uninformed.
    CHECK(informed < nodes);
}

#ifndef _WIN32
// Runs fn() with stderr redirected to a temp file and returns what was written.
static std::string capture_stderr_around(const std::function<void()> & fn)
{
    fflush(stderr);
    std::string path = make_temp_dir() + "/hpgl_stderr_capture.txt";
    int saved = dup(fileno(stderr));
    FILE * cap = fopen(path.c_str(), "w");
    if (cap != nullptr)
    {
        dup2(fileno(cap), fileno(stderr));
        fclose(cap);
    }
    fn();
    fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::string result;
    FILE * r = fopen(path.c_str(), "r");
    if (r != nullptr)
    {
        char buf[1024];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), r)) > 0)
            result.append(buf, n);
        fclose(r);
    }
    std::remove(path.c_str());
    return result;
}
#endif

// M-6: median IK previously emitted NO stderr failure signal when kriging
// systems failed (siblings cont_kriging/cokriging/SIS do). With sparse data,
// most nodes report KI_NO_NEIGHBOURS → the added signal must appear on stderr.
void test_median_ik_emits_stderr_failure_signal() {
    TEST("M-6: median IK emits stderr failure signal");
    const int cells = 16;            // 4x4x1
    const int cats = 2;
    std::vector<unsigned char> in_data(cells * cats, 0);
    std::vector<unsigned char> in_mask(cells * cats, 0);
    std::vector<unsigned char> out_data(cells * cats, 0);
    std::vector<unsigned char> out_mask(cells * cats, 0);
    in_mask[0] = 1;                  // corner datum (0,0)
    in_mask[15] = 1;                 // opposite corner (3,3)
    in_data[0] = 1;                  // category 0 indicator
    in_data[16 + 15] = 1;            // cell 15, indicator byte 1 → category 1

    hpgl_ind_masked_array_t in, out;
    in.m_data = in_data.data(); in.m_mask = in_mask.data();
    init_shape(in.m_shape, 4, 4, 1);
    in.m_indicator_count = cats;
    out.m_data = out_data.data(); out.m_mask = out_mask.data();
    init_shape(out.m_shape, 4, 4, 1);
    out.m_indicator_count = cats;

    hpgl_median_ik_params_t params;
    params.m_covariance_type = 0;  // COV_SPHERICAL
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 8;
    params.m_marginal_probs[0] = 0.5;
    params.m_marginal_probs[1] = 0.5;

#ifndef _WIN32
    std::string err = capture_stderr_around([&] {
        hpgl_median_ik(&in, &params, &out);
    });
    // With only 2 corner data and radius 1, the 12 middle cells have no
    // neighbour → the M-6 stderr failure signal must fire.
    CHECK(err.find("kriging failures") != std::string::npos);
#else
    hpgl_median_ik(&in, &params, &out);
    hpgl_kriging_stats_t stats = hpgl_get_kriging_stats();
    CHECK(stats.m_points_without_neighbours > 0);
#endif
}

// R-3 / M-11: the ndmin gate must count ORIGINAL conditioning data only
// (GSLIB sgsim semantics), not previously simulated nodes, and must fire even
// when previously simulated neighbours INFLATE the total count above ndmin.
// The round-1 regression test (single original datum) was VACUOUS — with one
// datum no node is ever simulated, so the inflation path was never exercised
// and the round-1 fix (which nested the original-count check inside the total
// gate, where original_count <= size always holds) was byte-identical to
// pre-fix. This test uses a 1×4 line with originals at 0 and 1 (min=2):
// node 2 has 2 original neighbours → simulated by both gates; node 3 has 1
// original neighbour (index 1) and — when node 2 was simulated earlier in the
// random path — a simulated neighbour too, so its TOTAL reaches 2 while its
// ORIGINAL count is 1. The fixed gate must SKIP node 3 (original < ndmin);
// the pre-fix/round-1 total gate simulated it whenever total >= 2.
void test_sgs_ndmin_skips_nodes_with_few_originals_despite_inflated_total() {
    TEST("R-3/M-11: SGS ndmin skips nodes with <ndmin ORIGINAL data even when simulated neighbours inflate the total");
    const int nx = 4, ny = 1, nz = 1;
    const int nodes = nx * ny * nz;
    // Non-vacuity guard: the divergence only manifests when node 2 (the node
    // that becomes a simulated neighbour) is processed BEFORE node 3. Use the
    // same seeded path generator the SGS kernel uses to select exactly the
    // seeds with that ordering, and assert at least one such seed exists —
    // otherwise the fixture is vacuous (the round-1 test's failure mode).
    bool saw_inflation_order = false;
    for (int64_t seed = 1; seed <= 40; ++seed)
    {
        hpgl::path_random_generator_t path(nodes, seed);
        int pos2 = -1, pos3 = -1;
        for (int i = 0; i < nodes; ++i)
        {
            int idx = path.next();
            if (idx == 2) pos2 = i;
            if (idx == 3) pos3 = i;
        }
        if (!(pos2 >= 0 && pos3 >= 0 && pos2 < pos3))
            continue;  // not an inflation-order seed — both gates behave identically
        saw_inflation_order = true;

        std::vector<float> data(nodes, 0.0f);
        std::vector<unsigned char> mask(nodes, 0);
        data[0] = 1.0f;  mask[0] = 1;
        data[1] = 2.0f;  mask[1] = 1;
        hpgl_cont_masked_array_t in;
        in.m_data = data.data();
        in.m_mask = mask.data();
        init_shape(in.m_shape, nx, ny, nz);

        hpgl_sgs_params_t params;
        params.m_covariance_type = 0;  // COV_SPHERICAL
        params.m_ranges[0] = 3; params.m_ranges[1] = 3; params.m_ranges[2] = 3;
        params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
        params.m_sill = 1.0;
        params.m_nugget = 0.0;
        params.m_radiuses[0] = 2; params.m_radiuses[1] = 0; params.m_radiuses[2] = 0;
        params.m_max_neighbours = 8;
        params.m_kriging_kind = 1;  // KRIG_SIMPLE
        params.m_seed = seed;
        params.m_min_neighbours = 2;

        double mean = 0.0;
        hpgl_sgs_simulation(&in, &params, nullptr, &mean, nullptr);

        // Originals preserved.
        CHECK(mask[0] == 1);
        CHECK(mask[1] == 1);
        // Node 2 has 2 ORIGINAL neighbours (indices 0,1) → simulated.
        CHECK(mask[2] == 1);
        // Node 3 has 1 ORIGINAL neighbour (index 1) < ndmin → must stay
        // unsimulated even though its TOTAL (index 1 + simulated node 2)
        // reached 2. The pre-fix/round-1 TOTAL gate simulated node 3 here —
        // the R-3 divergence the fix removes.
        CHECK(mask[3] == 0);
    }
    CHECK(saw_inflation_order);
}

// R-5: the SGS-OK <4-data downgraded estimate must have NO mean term (GSLIB
// sgsim lktype=0 zero-mean normal-score semantics). With a user mean of 100
// vs 0 and identical seeds on a config where every node kriges with <4 data
// (no failure fallback), the outputs must be IDENTICAL — the pre-fix code
// (single_mean_t(mean) + M-3 SK-downgraded weights) shifted every estimate by
// (1−Σλᵢ)·mean, making the n=4 mean-pull discontinuity user-visible.
void test_sgs_ok_downgraded_estimate_has_no_mean_term() {
    TEST("R-5: SGS-OK <4-data estimate has no (1-SumLambda)*mean term");
    const int nx = 5, ny = 1, nz = 1;  // 1×5 line
    const int nodes = nx * ny * nz;
    const int64_t seed = 42;

    auto run = [&](double user_mean, std::vector<float> & out_data,
                   std::vector<unsigned char> & out_mask) {
        std::vector<float> data(nodes, 0.0f);
        std::vector<unsigned char> mask(nodes, 0);
        data[0] = 1.0f;  mask[0] = 1;
        data[4] = 3.0f;  mask[4] = 1;
        hpgl_cont_masked_array_t in;
        in.m_data = data.data();
        in.m_mask = mask.data();
        init_shape(in.m_shape, nx, ny, nz);

        hpgl_sgs_params_t params;
        params.m_covariance_type = 0;  // COV_SPHERICAL
        params.m_ranges[0] = 3; params.m_ranges[1] = 3; params.m_ranges[2] = 3;
        params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
        params.m_sill = 1.0;
        params.m_nugget = 0.0;
        params.m_radiuses[0] = 2; params.m_radiuses[1] = 0; params.m_radiuses[2] = 0;
        params.m_max_neighbours = 8;
        params.m_kriging_kind = 0;    // KRIG_ORDINARY — SGS-OK path
        params.m_seed = seed;
        params.m_min_neighbours = 0;

        double mean = user_mean;
        hpgl_sgs_simulation(&in, &params, nullptr, &mean, nullptr);
        out_data = data;
        out_mask = mask;
    };

    std::vector<float> d100, d0;
    std::vector<unsigned char> m100, m0;
    run(100.0, d100, m100);
    run(0.0, d0, m0);

    // Every uninformed node has 1-2 ORIGINAL neighbours within radius 2 →
    // kriges on the <4-data SK-downgraded path (no failure fallback, so the
    // fallback mean — the only mean that legitimately differs — is never
    // used). With the R-5 fix the kriged mean is Σλᵢzᵢ regardless of the
    // user mean → identical outputs.
    for (int i = 0; i < nodes; ++i)
        CHECK(std::abs(d100[i] - d0[i]) < 1e-6);
}

// R-6: the plain-lookup pure-nugget fallback (2-M-32) must be BOUNDED. The
// covariance-threshold offset list for a pure-nugget model is empty, so the
// fallback fires per find(); pre-fix it iterated the FULL (2r+1)³ box per
// node — an effectively-infinite loop for large radii on sparse grids (the
// round-1 fix comment conflated the construction-time calc_cov_field scan
// with this find-time scan). The fix caps the per-direction window at 8
// cells: neighbours beyond the window are NOT admitted (bounded work),
// neighbours inside it still are (the 2-M-32 kriging behavior is preserved
// for realistic radii).
void test_plain_lookup_fallback_window_bounded() {
    TEST("R-6: plain-lookup pure-nugget fallback is bounded to a small window");
    hpgl::sugarbox_grid_t grid;
    grid.init(21, 21, 1);   // indices 0..440
    double ranges[3] = {3.0, 3.0, 3.0};
    double angles[3] = {0.0, 0.0, 0.0};
    hpgl::cov_model_t cov(hpgl::covariance_type_t::COV_SPHERICAL,
                          ranges, angles, 1.0, 1.0);  // pure nugget

    hpgl::neighbourhood_param_t nb;
    nb.set_radiuses(50, 50, 0);   // large legal radius
    nb.m_max_neighbours = 8;

    hpgl::neighbour_lookup_t<hpgl::sugarbox_grid_t, hpgl::cov_model_t> nl(&grid, &cov, nb);

    struct mask_pred {
        const unsigned char * m;
        bool operator()(hpgl::node_index_t i) const { return m[i] == 1; }
    };

    // Target (10,10). The out-of-window datum (1,1) is at offset (−9,−9).
    const int target = 10 * 21 + 10;
    std::vector<unsigned char> mask(441, 0);

    // 1) Out-of-window datum → NOT admitted (bounded). The pre-fix full-box
    // scan would have admitted it (radius 50 covers the whole 21×21 grid).
    {
        mask[1 * 21 + 1] = 1;
        std::vector<hpgl::node_index_t> indices;
        std::vector<hpgl::sugarbox_location_t> coords;
        hpgl::sugarbox_location_t node_coord;
        nl.find(target, mask_pred{&mask[0]}, node_coord, indices, coords);
        CHECK(indices.empty());
        mask[1 * 21 + 1] = 0;
    }
    // 2) In-window datum (offset (0,3)) → still admitted (fallback works).
    {
        mask[13 * 21 + 10] = 1;   // (10,13)
        std::vector<hpgl::node_index_t> indices;
        std::vector<hpgl::sugarbox_location_t> coords;
        hpgl::sugarbox_location_t node_coord;
        nl.find(target, mask_pred{&mask[0]}, node_coord, indices, coords);
        CHECK(indices.size() == 1);
        CHECK(indices[0] == 13 * 21 + 10);
    }
}

// R-11 / 2-M-35: hpgl_median_ik must validate marginal_probs. Pre-fix the
// pair was copied raw — a value outside [0,1] (e.g. marginal_probs[1]=5.0)
// silently produced all-1 output for direct-C callers (prob=5.0 →
// choose_indicator always returns 1). The fix mirrors the Python wrapper
// (geo.py:1879-1884 → validation.py:834-881): each value in [0,1] and the
// pair sums to ~1 (tolerance 0.01).
void test_median_ik_rejects_invalid_marginal_probs() {
    TEST("R-11/2-M-35: median_ik rejects marginal_probs outside [0,1] or not summing to 1");
    const int cells = 4;  // 2×2×1
    const int cats = 2;
    std::vector<unsigned char> in_data(cells * cats, 0);
    std::vector<unsigned char> in_mask(cells * cats, 0);
    std::vector<unsigned char> out_data(cells * cats, 0);
    std::vector<unsigned char> out_mask(cells * cats, 0);

    hpgl_ind_masked_array_t in, out;
    in.m_data = in_data.data(); in.m_mask = in_mask.data();
    init_shape(in.m_shape, 2, 2, 1);
    in.m_indicator_count = cats;
    out.m_data = out_data.data(); out.m_mask = out_mask.data();
    init_shape(out.m_shape, 2, 2, 1);
    out.m_indicator_count = cats;

    // 1) Value outside [0,1] → clear error (pre-fix: silent all-1 output).
    {
        hpgl_median_ik_params_t params;
        params.m_covariance_type = 0;
        params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
        params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
        params.m_sill = 1.0;
        params.m_nugget = 0.0;
        params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
        params.m_max_neighbours = 8;
        params.m_marginal_probs[0] = 0.5;
        params.m_marginal_probs[1] = 5.0;  // pre-fix: silent all-1 output
        hpgl_median_ik(&in, &params, &out);
        const char * msg = hpgl_get_last_exception_message();
        CHECK(msg != nullptr && strstr(msg, "marginal_probs") != nullptr);
    }
    // 2) Pair not summing to ~1 → clear error.
    {
        hpgl_median_ik_params_t params;
        params.m_covariance_type = 0;
        params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
        params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
        params.m_sill = 1.0;
        params.m_nugget = 0.0;
        params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
        params.m_max_neighbours = 8;
        params.m_marginal_probs[0] = 0.2;
        params.m_marginal_probs[1] = 0.5;  // sum 0.7 ≠ 1
        hpgl_median_ik(&in, &params, &out);
        const char * msg = hpgl_get_last_exception_message();
        CHECK(msg != nullptr && strstr(msg, "sum") != nullptr);
    }
    // 3) Valid pair still works (all-uninformed → all fallback to 0.5 →
    // category 0 → every output cell is written).
    {
        hpgl_median_ik_params_t params;
        params.m_covariance_type = 0;
        params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
        params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
        params.m_sill = 1.0;
        params.m_nugget = 0.0;
        params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
        params.m_max_neighbours = 8;
        params.m_marginal_probs[0] = 0.5;
        params.m_marginal_probs[1] = 0.5;
        hpgl_median_ik(&in, &params, &out);
        int informed = 0;
        for (int i = 0; i < cells; ++i)
            informed += (out_mask[i] ? 1 : 0);
        CHECK(informed == cells);  // no exception; every node processed
    }
}

// M-28: m_min_neighbours must be validated at the C API boundary. Pre-fix,
// min>max ⇒ every node skipped → fully-unsimulated SGS output with no error.
// The fix rejects min<0 and min>max with a clear error message.
void test_sgs_rejects_invalid_min_neighbours() {
    TEST("M-28: SGS rejects min_neighbours > max_neighbours");
    float data[4] = {1.0f, 0, 0, 0};
    unsigned char mask[4] = {1, 0, 0, 0};
    hpgl_cont_masked_array_t in;
    in.m_data = data;
    in.m_mask = mask;
    init_shape(in.m_shape, 2, 2, 1);

    hpgl_sgs_params_t params;
    params.m_covariance_type = 0;
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 3;
    params.m_kriging_kind = 1;
    params.m_seed = 42;
    params.m_min_neighbours = 5;  // > max

    double mean = 0.0;
    hpgl_sgs_simulation(&in, &params, nullptr, &mean, nullptr);
    const char * msg = hpgl_get_last_exception_message();
    CHECK(msg != nullptr && strstr(msg, "m_min_neighbours") != nullptr);
}

void test_sgs_rejects_negative_min_neighbours() {
    TEST("M-28: SGS rejects negative min_neighbours");
    float data[4] = {1.0f, 0, 0, 0};
    unsigned char mask[4] = {1, 0, 0, 0};
    hpgl_cont_masked_array_t in;
    in.m_data = data;
    in.m_mask = mask;
    init_shape(in.m_shape, 2, 2, 1);

    hpgl_sgs_params_t params;
    params.m_covariance_type = 0;
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 8;
    params.m_kriging_kind = 1;
    params.m_seed = 42;
    params.m_min_neighbours = -1;

    double mean = 0.0;
    hpgl_sgs_simulation(&in, &params, nullptr, &mean, nullptr);
    const char * msg = hpgl_get_last_exception_message();
    CHECK(msg != nullptr && strstr(msg, "m_min_neighbours") != nullptr);
}

// M-29 + 2-M-33: with max_neighbours=0 (unconditional simulation) every node
// takes the failure fallback. GSLIB draws N(gmean, 1.0) on the OK fallback
// (M-29: previously N(0,1) — the supplied mean was ignored), and the reported
// stats mean must divide by ALL simulated nodes (2-M-33: previously the
// success-only denominator forced m_mean=0 when every node fell back).
void test_sgs_ok_fallback_uses_gmean_and_stats_denominator() {
    TEST("M-29+2-M-33: OK-mode SGS fallback draws N(gmean,1) and stats mean uses all nodes");
    const int nx = 20, ny = 20, nz = 1;
    const int nodes = nx * ny * nz;
    std::vector<float> data(nodes, 0.0f);
    std::vector<unsigned char> mask(nodes, 0);
    mask[0] = 1;  // keep SGS running; radiuses 0 → no neighbours anywhere
    data[0] = 0.0f;
    hpgl_cont_masked_array_t in;
    in.m_data = data.data();
    in.m_mask = mask.data();
    init_shape(in.m_shape, nx, ny, nz);

    hpgl_sgs_params_t params;
    params.m_covariance_type = 0;
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 0; params.m_radiuses[1] = 0; params.m_radiuses[2] = 0;
    params.m_max_neighbours = 0;  // unconditional simulation — legal for SGS
    params.m_kriging_kind = 0;    // KRIG_ORDINARY — fallback path
    params.m_seed = 7;
    params.m_min_neighbours = 0;

    double mean = 5.0;  // user-supplied mean (gmean)
    hpgl_sgs_simulation(&in, &params, nullptr, &mean, nullptr);

    hpgl_kriging_stats_t stats = hpgl_get_kriging_stats();
    // All nodes fell back → m_points_calculated == 0; m_mean must still
    // approximate gmean=5.0 (M-29 fallback + 2-M-33 all-node denominator).
    // 400 samples of N(5,1): |mean − 5| < 0.15 with overwhelming probability.
    CHECK(stats.m_points_calculated == 0);
    CHECK(stats.m_mean > 4.0 && stats.m_mean < 6.0);
}

// M-31: kriging requires at least one neighbour — max_neighbours=0 (legal for
// SGS/SIS unconditional simulation) must be REJECTED on kriging entry points
// with a clear error instead of silently mean-filling / all-undefined output.
void test_kriging_rejects_zero_max_neighbours() {
    TEST("M-31: kriging rejects max_neighbours=0");
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    unsigned char in_mask[4] = {1, 1, 1, 1};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char out_mask[4] = {0, 0, 0, 0};
    hpgl_cont_masked_array_t in, out;
    in.m_data = in_data; in.m_mask = in_mask; init_shape(in.m_shape, 2, 2, 1);
    out.m_data = out_data; out.m_mask = out_mask; init_shape(out.m_shape, 2, 2, 1);

    hpgl_ok_params_t params;
    params.m_covariance_type = 0;
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 0;  // pre-fix: accepted → empty neighbourhood

    hpgl_ordinary_kriging(&in, &params, &out);
    const char * msg = hpgl_get_last_exception_message();
    CHECK(msg != nullptr && strstr(msg, "at least 1") != nullptr);
}

// 2-M-4: m_kriging_kind must be validated at the SGS C API boundary. Pre-fix
// every value ≠ KRIG_SIMPLE silently ran ordinary kriging. Values outside
// {KRIG_ORDINARY=0, KRIG_SIMPLE=1} must be rejected with a clear error.
void test_sgs_rejects_invalid_kriging_kind() {
    TEST("2-M-4: SGS rejects invalid kriging_kind");
    float data[4] = {1.0f, 0, 0, 0};
    unsigned char mask[4] = {1, 0, 0, 0};
    hpgl_cont_masked_array_t in;
    in.m_data = data;
    in.m_mask = mask;
    init_shape(in.m_shape, 2, 2, 1);

    hpgl_sgs_params_t params;
    params.m_covariance_type = 0;
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 8;
    params.m_kriging_kind = 99;  // invalid — pre-fix silently ran OK
    params.m_seed = 42;
    params.m_min_neighbours = 0;

    double mean = 0.0;
    hpgl_sgs_simulation(&in, &params, nullptr, &mean, nullptr);
    const char * msg = hpgl_get_last_exception_message();
    CHECK(msg != nullptr && strstr(msg, "kriging_kind") != nullptr);
}

// 2-M-32: cokriging uses the PLAIN neighbour lookup whose calc_cov_field offset
// list is covariance-threshold filtered — a pure-nugget model (nugget==sill,
// every h>0 covariance is 0) produced an EMPTY neighbourhood → every node
// mean-filled (26/26 KI_NO_NEIGHBOURS in the finding's repro). The plain
// lookup now mirrors the indexed lookup's radius-bounded pure-nugget fallback,
// so the same params produce kriged (KI_SUCCESS) values.
void test_cokriging_pure_nugget_not_mean_filled() {
    TEST("2-M-32: cokriging pure-nugget model kriges (not mean-fills)");
    hpgl::sugarbox_grid_t grid;
    grid.init(4, 4, 1);
    const int n = 16;
    std::vector<float> primary_data(n, 0.0f);
    std::vector<unsigned char> primary_mask(n, 0);
    primary_data[0] = 1.0f;  primary_mask[0] = 1;
    primary_data[1] = 3.0f;  primary_mask[1] = 1;
    primary_data[4] = 2.0f;  primary_mask[4] = 1;
    primary_data[15] = 4.0f; primary_mask[15] = 1;  // cover the far corner
    std::vector<float> secondary_data(n, 0.0f);
    std::vector<unsigned char> secondary_mask(n, 0);
    secondary_data[0] = 1.0f; secondary_mask[0] = 1;
    secondary_data[4] = 2.0f; secondary_mask[4] = 1;
    std::vector<float> output_data(n, 0.0f);
    std::vector<unsigned char> output_mask(n, 0);

    hpgl::cont_property_array_t primary(primary_data.data(), primary_mask.data(), n);
    hpgl::cont_property_array_t secondary(secondary_data.data(), secondary_mask.data(), n);
    hpgl::cont_property_array_t output(output_data.data(), output_mask.data(), n);

    hpgl::neighbourhood_param_t nb;
    nb.set_radiuses(2, 2, 0);
    nb.m_max_neighbours = 8;

    hpgl::covariance_param_t cp;
    cp.m_covariance_type = hpgl::covariance_type_t::COV_SPHERICAL;
    cp.set_ranges(3.0, 3.0, 3.0);
    cp.set_angles(0.0, 0.0, 0.0);
    cp.set_sill(1.0);
    cp.set_nugget(1.0);  // pure nugget: C(h>0) == 0

    hpgl::simple_cokriging_markI(grid, primary, secondary,
        0.0f, 0.0f, 1.0, 0.5, nb, cp, output);

    hpgl_kriging_stats_t stats = hpgl_get_kriging_stats();
    // Pre-fix: plain lookup threshold-filtered to nothing → 12/12 uninformed
    // nodes mean-filled (KI_NO_NEIGHBOURS, points_calculated == 0). Post-fix:
    // radius-bounded neighbours found → all uninformed nodes kriged.
    CHECK(stats.m_points_calculated > 0);
    CHECK(stats.m_points_without_neighbours == 0);
}

// 2-M-33 (median_ik): kriging_stats_t.m_mean must divide by all processed
// nodes, not success-only points_calculated. With every node falling back to
// the marginal (no neighbours), marginal_probs[1]=0.9 → every node outputs
// category 1 → m_mean must be 1.0, not 0.0 (pre-fix: points_calculated==0 →
// the ternary returned 0).
void test_median_ik_stats_mean_uses_all_processed() {
    TEST("2-M-33: median IK stats mean uses all-processed denominator");
    const int cells = 16;            // 4x4x1, all uninformed
    const int cats = 2;
    std::vector<unsigned char> in_data(cells * cats, 0);
    std::vector<unsigned char> in_mask(cells * cats, 0);
    std::vector<unsigned char> out_data(cells * cats, 0);
    std::vector<unsigned char> out_mask(cells * cats, 0);

    hpgl_ind_masked_array_t in, out;
    in.m_data = in_data.data(); in.m_mask = in_mask.data();
    init_shape(in.m_shape, 4, 4, 1);
    in.m_indicator_count = cats;
    out.m_data = out_data.data(); out.m_mask = out_mask.data();
    init_shape(out.m_shape, 4, 4, 1);
    out.m_indicator_count = cats;

    hpgl_median_ik_params_t params;
    params.m_covariance_type = 0;
    params.m_ranges[0] = 1; params.m_ranges[1] = 1; params.m_ranges[2] = 1;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;
    params.m_radiuses[0] = 1; params.m_radiuses[1] = 1; params.m_radiuses[2] = 1;
    params.m_max_neighbours = 8;
    params.m_marginal_probs[0] = 0.1;
    params.m_marginal_probs[1] = 0.9;

    hpgl_median_ik(&in, &params, &out);
    hpgl_kriging_stats_t stats = hpgl_get_kriging_stats();
    // No informed cells → every node falls back to marginal 0.9 → category 1.
    // Post-fix mean == 1.0 (16/16); pre-fix mean == 0.0 (points_calculated 0).
    CHECK(stats.m_points_calculated == 0);
    CHECK(stats.m_mean > 0.99);
}

// 2-M-2: hpgl_simple_kriging_weights was the ONLY kriging entry point
// without the MAX_NEIGHBOURS_UPPER_BOUND gate. Pre-fix a pathological
// neighbours_count (a) dereferenced the neighbour arrays up to count — a
// heap OOB read when count exceeds the arrays' actual length — and
// (b) allocated A = count² (100k pts → 160 GB) + O(count³) solve → OOM/DoS.
// The gate must reject count > 100000 with a clear error BEFORE any
// dereference, while a valid small count still works.
void test_simple_kriging_weights_rejects_huge_count() {
    TEST("2-M-2: simple_kriging_weights rejects neighbours_count > 100000");
    float center[3] = {0.0f, 0.0f, 0.0f};
    float nx[2] = {1.0f, 2.0f};
    float ny[2] = {0.0f, 0.0f};
    float nz[2] = {0.0f, 0.0f};
    float weights[2] = {0.0f, 0.0f};

    hpgl_cov_params_t params;
    params.m_covariance_type = 0;   // COV_SPHERICAL
    params.m_ranges[0] = 10; params.m_ranges[1] = 10; params.m_ranges[2] = 10;
    params.m_angles[0] = 0; params.m_angles[1] = 0; params.m_angles[2] = 0;
    params.m_sill = 1.0;
    params.m_nugget = 0.0;

    // Pathological count: the arrays have only 2 elements. Pre-fix this read
    // neighbours_x[i] for i < 100001 (heap OOB read) and allocated a 100001²
    // covariance matrix (~80 GB) before the solve.
    int rc = hpgl_simple_kriging_weights(center, nx, ny, nz, 100001, &params, weights);
    CHECK(rc == -1);
    const char * msg = hpgl_get_last_exception_message();
    CHECK(msg != nullptr && strstr(msg, "100000") != nullptr);

    // A valid small count still produces weights.
    rc = hpgl_simple_kriging_weights(center, nx, ny, nz, 2, &params, weights);
    CHECK(rc == 0);
    CHECK(std::isfinite(weights[0]) && std::isfinite(weights[1]));
}

// 2-M-9: cokriging Mark I/II gained per-node workspace reuse (the previous
// ~7 heap allocations per node — coords/indices/weights/values/A/b/A_backup —
// are now reused from a cokriging_ws_t). This test asserts correctness is
// unchanged: with an informed secondary, uninformed nodes are kriged
// (KI_SUCCESS), no singularity fallback, and all outputs are finite.
void test_cokriging_workspace_preserves_correctness() {
    TEST("2-M-9: cokriging Mark I workspace reuse preserves correctness");
    hpgl::sugarbox_grid_t grid;
    grid.init(4, 4, 1);
    const int n = 16;
    std::vector<float> primary_data(n, 0.0f);
    std::vector<unsigned char> primary_mask(n, 0);
    primary_data[0] = 1.0f;  primary_mask[0] = 1;
    primary_data[1] = 3.0f;  primary_mask[1] = 1;
    primary_data[4] = 2.0f;  primary_mask[4] = 1;
    std::vector<float> secondary_data(n, 0.0f);
    std::vector<unsigned char> secondary_mask(n, 0);
    secondary_data[0] = 1.0f; secondary_mask[0] = 1;
    secondary_data[4] = 2.0f; secondary_mask[4] = 1;
    std::vector<float> output_data(n, 0.0f);
    std::vector<unsigned char> output_mask(n, 0);

    hpgl::cont_property_array_t primary(primary_data.data(), primary_mask.data(), n);
    hpgl::cont_property_array_t secondary(secondary_data.data(), secondary_mask.data(), n);
    hpgl::cont_property_array_t output(output_data.data(), output_mask.data(), n);

    hpgl::neighbourhood_param_t nb;
    nb.set_radiuses(2, 2, 0);
    nb.m_max_neighbours = 8;

    hpgl::covariance_param_t cp;
    cp.m_covariance_type = hpgl::covariance_type_t::COV_EXPONENTIAL;
    cp.set_ranges(3.0, 3.0, 3.0);
    cp.set_angles(0.0, 0.0, 0.0);
    cp.set_sill(1.0);
    cp.set_nugget(0.0);

    hpgl::simple_cokriging_markI(grid, primary, secondary,
        0.0f, 0.0f, 1.0, 0.5, nb, cp, output);

    hpgl_kriging_stats_t stats = hpgl_get_kriging_stats();
    // Pre/post-fix identical: the workspace refactor is allocation-only.
    CHECK(stats.m_points_calculated > 0);
    CHECK(stats.m_points_singularity == 0);
    bool any_output = false;
    for (int i = 0; i < n; ++i) {
        if (output_mask[i]) {
            CHECK(std::isfinite(output_data[i]));
            any_output = true;
        }
    }
    CHECK(any_output);
}

// 2-M-15: the C++ property_array mask buffer is a separate allocation whose
// length may be smaller than the data size (a mismatched Python mask). The
// mask_size constructor arg bounds set_at/is_informed/delete_value_at so a
// smaller mask cannot cause a heap OOB write (0x01) or OOB read.
void test_property_array_mask_size_bounds() {
    TEST("2-M-15: property_array mask_size bounds mask access");
    float data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char mask[2] = {1, 0};   // mask smaller than data (4 cells)

    hpgl::cont_property_array_t prop(data, mask, 4, 2);  // mask_size = 2

    // Reads beyond the mask length must be treated as not-informed — pre-fix
    // is_informed(2)/is_informed(3) read mask[2]/mask[3] out of bounds.
    CHECK(prop.is_informed(0) == true);
    CHECK(prop.is_informed(1) == false);
    CHECK(prop.is_informed(2) == false);
    CHECK(prop.is_informed(3) == false);

    // Writes within the mask length still work.
    prop.set_at(1, 5.0f);
    CHECK(data[1] == 5.0f);
    CHECK(mask[1] == 1);

    // Legacy default (mask_size=0) keeps the pre-existing behavior.
    float data2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned char mask2[4] = {0, 0, 0, 0};
    hpgl::cont_property_array_t prop2(data2, mask2, 4);
    prop2.set_at(3, 7.0f);
    CHECK(data2[3] == 7.0f);
    CHECK(mask2[3] == 1);
}

// M-12: the clusterizer admission cap must bound the EFFECTIVE memory
// (pointer vector + worst-case fully-dense cluster_t objects), not just the
// pointer count. A 300³ grid at radius 1 has a cluster grid of 302³ ≈ 27.5M
// cells — legal under the old 100M pointer-vector cap but ~1.7 GB of lazy
// cluster_t heap objects when fully dense. It must be rejected with a clear
// memory-safe error instead of OOM.
void test_clusterizer_rejects_memory_unsafe_volume() {
    TEST("M-12: clusterizer rejects memory-unsafe cluster-grid volume");
    hpgl::sugarbox_grid_t grid;
    grid.init(300, 300, 300);
    hpgl::sugarbox_search_ellipsoid_t ell(1, 1, 1);

    bool threw = false;
    try {
        hpgl::clusterizer_t c(&grid, ell, 1);
    } catch (const hpgl::hpgl_exception & ex) {
        threw = true;
        CHECK(strstr(ex.what(), "memory-safe") != nullptr);
    }
    CHECK(threw);
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

    test_write_inc_file_byte_identity_remap_zero_values();
    test_write_inc_file_float_rejects_nonpositive_dims();
    test_ordinary_kriging_rejects_huge_max_neighbours();
    test_ordinary_kriging_rejects_zero_radius();
    test_property_writer_removes_tmp_on_failure();
    test_handler_setter_blocks_during_invocation();
    test_reentrant_handler_no_deadlock();
    test_read_inc_file_skips_midline_slash();
    test_read_inc_file_rejects_extra_tokens();
    test_read_inc_file_exact_size_with_terminator();

    // Production-check regression tests (F-04, F-20, F-21, I2-24, F-09,
    // F-19, F-22, F-14).
    test_gaussian_inverse_tail_saturation();
    test_order_relations_gslib_envelope();
    test_precalculated_covariance_exact_beyond_box();
    test_precalculated_covariance_throws_on_huge_volume();
    test_indexed_lookup_radius_bound();
    test_cokriging_stats_and_missing_secondary();
    test_sgs_min_neighbours_wired();

    // FIX convergence pass (PR-05 C++ part, PR-06, PR-07).
    test_indicator_kriging_rejects_zero_radius();
    test_covariance_field_exact_beyond_box();
    test_cokriging_rejects_huge_max_neighbours();

    // s6-fix-cpp-engines regression tests (M-3, M-5, M-6, M-11, M-28, M-29,
    // M-31, 2-M-4, 2-M-32, 2-M-33) + s9 round-2 re-fixes (R-3, R-5, R-6, R-11).
    test_ok_sgs_only_downgrades_under_4_neighbours();
    test_user_cancellation_stops_kriging_loop();
    test_median_ik_emits_stderr_failure_signal();
    test_sgs_ndmin_skips_nodes_with_few_originals_despite_inflated_total();
    test_sgs_rejects_invalid_min_neighbours();
    test_sgs_rejects_negative_min_neighbours();
    test_sgs_ok_fallback_uses_gmean_and_stats_denominator();
    test_kriging_rejects_zero_max_neighbours();
    test_sgs_rejects_invalid_kriging_kind();
    test_cokriging_pure_nugget_not_mean_filled();
    test_median_ik_stats_mean_uses_all_processed();
    test_sgs_ok_downgraded_estimate_has_no_mean_term();
    test_plain_lookup_fallback_window_bounded();
    test_median_ik_rejects_invalid_marginal_probs();

    // s6-fix-cpp-hardening regression tests (M-12, 2-M-2, 2-M-9, 2-M-15).
    test_simple_kriging_weights_rejects_huge_count();
    test_cokriging_workspace_preserves_correctness();
    test_property_array_mask_size_bounds();
    test_clusterizer_rejects_memory_unsafe_volume();

    std::printf("C++ unit tests: %d run, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
