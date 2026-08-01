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

    std::printf("C++ unit tests: %d run, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
