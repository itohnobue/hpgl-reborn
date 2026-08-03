/**
 * C++ regression tests for the cvariogram module fixes (F-06, F-08, F-37,
 * F-38, F-39, F-40, F-61, I2-25).
 *
 * Uses the same minimal assertion-based framework as test_hpgl_core.cpp.
 *
 * The cvariogram functions live in the hpgl_variogram shared library, which
 * hpgl_core_tests does not link. To keep this test self-contained the module
 * sources are compiled into this translation unit directly.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Compile the cvariogram module sources directly into this test binary.
#include "src/geo_bsd/_cvariogram/api.h"
#include "src/geo_bsd/_cvariogram/stack_layers.h"
#include "src/geo_bsd/_cvariogram/stack_layers.cpp"
#include "src/geo_bsd/_cvariogram/variograms.cpp"
#include "src/geo_bsd/_cvariogram/ellipsoid.cpp"

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

// ---- Helpers ----

// Builds a contiguous float_data_t of shape (nx, ny, 1) with the given fill.
static float_data_t make_layer(int nx, int ny, float fill)
{
    float_data_t L;
    std::vector<float> * buf = new std::vector<float>(static_cast<size_t>(nx) * ny, fill);
    L.m_data = buf->data();
    L.m_data_shape[0] = nx;
    L.m_data_shape[1] = ny;
    L.m_data_shape[2] = 1;
    L.m_data_strides[0] = ny;
    L.m_data_strides[1] = 1;
    L.m_data_strides[2] = 1;
    // Keep the buffer alive by stashing it in the unused padding-free tail of
    // the struct is not possible; instead leak is fine for a short test, but we
    // keep a registry to free at exit for cleanliness.
    static std::vector<std::vector<float> *> g_buffers;
    g_buffers.push_back(buf);
    return L;
}

// Builds a contiguous result of shape (nx, ny, nz).
static float_data_t make_result(int nx, int ny, int nz, float fill = 0.0f)
{
    float_data_t R;
    std::vector<float> * buf = new std::vector<float>(static_cast<size_t>(nx) * ny * nz, fill);
    R.m_data = buf->data();
    R.m_data_shape[0] = nx;
    R.m_data_shape[1] = ny;
    R.m_data_shape[2] = nz;
    R.m_data_strides[0] = ny * nz;
    R.m_data_strides[1] = nz;
    R.m_data_strides[2] = 1;
    static std::vector<std::vector<float> *> g_buffers;
    g_buffers.push_back(buf);
    return R;
}

// F-06: result x/y dims smaller than the layer-derived dims must be rejected
// instead of writing out of bounds (pre-fix: heap OOB WRITE / SIGSEGV).
void test_stack_layers_rejects_small_result_grid()
{
    TEST("F-06: stack_layers rejects result smaller than layer");
    cvar_clear_last_error();
    std::vector<float_data_t> layers;
    layers.push_back(make_layer(10, 5, 1.0f));
    int markers[1] = {1};
    float_data_t result = make_result(10, 4, 4);  // ny=5 layer vs ny=4 result
    stack_layers(layers, markers, 4, 1.0f, -99, result);
    CHECK(cvar_get_last_error() != nullptr);
    CHECK(std::strlen(cvar_get_last_error()) > 0);
}

// F-08: non-contiguous (sliced) layer arrays must be rejected instead of
// writing out of bounds into cumulative_k (pre-fix: map_index up to 88 > 25).
void test_stack_layers_rejects_non_contiguous_layer()
{
    TEST("F-08: stack_layers rejects non-contiguous (sliced) layer");
    cvar_clear_last_error();
    // Simulate a strided view: shape claims (5,5,1) but stride[0] = 10 (a
    // slice of a 10-wide parent buffer). map_index would reach 44 > 25.
    float_data_t sliced;
    std::vector<float> backing(100, 1.0f);
    sliced.m_data = backing.data();
    sliced.m_data_shape[0] = 5;
    sliced.m_data_shape[1] = 5;
    sliced.m_data_shape[2] = 1;
    sliced.m_data_strides[0] = 10;  // non-contiguous
    sliced.m_data_strides[1] = 1;
    sliced.m_data_strides[2] = 1;

    std::vector<float_data_t> layers;
    layers.push_back(sliced);
    int markers[1] = {1};
    float_data_t result = make_result(5, 5, 4);
    stack_layers(layers, markers, 4, 1.0f, -99, result);
    CHECK(std::strlen(cvar_get_last_error()) > 0);
}

// F-39: an exact-integer deposit must occupy exactly one cell (pre-fix:
// 1.0 filled cells 0 AND 1; 3x1.0 filled 4 cells).
// III-40: unwritten cells (top tail above the final surface) are now
// initialized to blank_value instead of keeping the caller's prefill.
void test_stack_layers_integer_deposit_fills_one_cell()
{
    TEST("F-39: 1.0 deposit occupies exactly one cell");
    cvar_clear_last_error();
    std::vector<float_data_t> layers;
    layers.push_back(make_layer(1, 1, 1.0f));
    int markers[1] = {1};
    float_data_t result = make_result(1, 1, 4);
    stack_layers(layers, markers, 4, 1.0f, -99, result);
    CHECK(std::strlen(cvar_get_last_error()) == 0);
    CHECK(result.m_data[0] == 1.0f);       // cell 0 marked
    CHECK(result.m_data[1] == -99.0f);     // cell 1 blank (pre-fix: 0.0 prefill, pre-F-39: 1.0)
    CHECK(result.m_data[2] == -99.0f);
    CHECK(result.m_data[3] == -99.0f);
}

void test_stack_layers_three_integer_deposits_three_cells()
{
    TEST("F-39: 3x1.0 deposits occupy exactly three cells");
    cvar_clear_last_error();
    std::vector<float_data_t> layers;
    layers.push_back(make_layer(1, 1, 1.0f));
    layers.push_back(make_layer(1, 1, 1.0f));
    layers.push_back(make_layer(1, 1, 1.0f));
    int markers[3] = {1, 2, 3};
    float_data_t result = make_result(1, 1, 4);
    stack_layers(layers, markers, 4, 1.0f, -99, result);
    CHECK(std::strlen(cvar_get_last_error()) == 0);
    CHECK(result.m_data[0] == 1.0f);
    CHECK(result.m_data[1] == 2.0f);
    CHECK(result.m_data[2] == 3.0f);
    CHECK(result.m_data[3] == -99.0f);     // III-40: top tail is blank_value, not 0.0 prefill
}

// I2-25: NaN thickness must be rejected with an error instead of producing
// FP->int cast UB (pre-fix: whole-column corruption / NaN poisoning).
void test_stack_layers_rejects_nan_thickness()
{
    TEST("I2-25: stack_layers rejects NaN thickness");
    cvar_clear_last_error();
    float_data_t nan_layer;
    std::vector<float> backing(25, 1.0f);
    backing[0] = std::numeric_limits<float>::quiet_NaN();
    nan_layer.m_data = backing.data();
    nan_layer.m_data_shape[0] = 5;
    nan_layer.m_data_shape[1] = 5;
    nan_layer.m_data_shape[2] = 1;
    nan_layer.m_data_strides[0] = 5;
    nan_layer.m_data_strides[1] = 1;
    nan_layer.m_data_strides[2] = 1;

    std::vector<float_data_t> layers;
    layers.push_back(nan_layer);
    int markers[1] = {1};
    float_data_t result = make_result(5, 5, 4);
    stack_layers(layers, markers, 4, 1.0f, -99, result);
    CHECK(std::strlen(cvar_get_last_error()) > 0);
}

void test_stack_layers_rejects_huge_thickness()
{
    TEST("I2-25: stack_layers rejects huge thickness (FP->int cast guard)");
    cvar_clear_last_error();
    std::vector<float_data_t> layers;
    layers.push_back(make_layer(1, 1, 1e10f));
    int markers[1] = {1};
    float_data_t result = make_result(1, 1, 4);
    stack_layers(layers, markers, 4, 1.0f, -99, result);
    CHECK(std::strlen(cvar_get_last_error()) > 0);
}

// F-37: cvar_clear_last_error must reset the module error so consecutive
// identical failures are not suppressed by the Python snapshot guard.
void test_cvar_clear_last_error_resets()
{
    TEST("F-37: cvar_clear_last_error clears the module error");
    cvar_clear_last_error();
    CHECK(std::strlen(cvar_get_last_error()) == 0);

    // Trigger an error path.
    variogram_search_template_t templ = {};
    float out[4] = {0, 0, 0, 0};
    // null data -> Null pointer argument error.
    calc_variograms(&templ, nullptr, out, 4, 100);
    CHECK(std::strlen(cvar_get_last_error()) > 0);

    cvar_clear_last_error();
    CHECK(std::strlen(cvar_get_last_error()) == 0);
}

// F-38: oversized lag_separation / num_lags must fail fast instead of looping
// ~1e11 times (pre-fix: effectively infinite hang).
void test_calc_variograms_rejects_oversized_window()
{
    TEST("F-38: calc_variograms rejects pathological lag window");
    cvar_clear_last_error();
    variogram_search_template_t templ = {};
    templ.m_lag_width = 1.0;
    templ.m_lag_separation = 1e6;   // pathological
    templ.m_tol_distance = 1.0;
    templ.m_num_lags = 10000;
    templ.m_first_lag_distance = 0.0;
    templ.m_ellipsoid.m_R1 = 10.0;
    templ.m_ellipsoid.m_R2 = 5.0;
    templ.m_ellipsoid.m_R3 = 3.0;
    templ.m_ellipsoid.m_direction1.m_data[0] = 1.0;
    templ.m_ellipsoid.m_direction2.m_data[1] = 1.0;
    templ.m_ellipsoid.m_direction3.m_data[2] = 1.0;

    float data_vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    unsigned char mask_vals[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    hard_data_t data = {};
    data.m_data = data_vals;
    data.m_mask = mask_vals;
    data.m_data_shape[0] = 2; data.m_data_shape[1] = 2; data.m_data_shape[2] = 2;
    data.m_mask_shape[0] = 2; data.m_mask_shape[1] = 2; data.m_mask_shape[2] = 2;
    data.m_data_strides[0] = 4; data.m_data_strides[1] = 2; data.m_data_strides[2] = 1;
    data.m_mask_strides[0] = 4; data.m_mask_strides[1] = 2; data.m_mask_strides[2] = 1;

    float out[4] = {0, 0, 0, 0};
    calc_variograms(&templ, &data, out, 4, 100);
    CHECK(std::strlen(cvar_get_last_error()) > 0);  // fails fast, no hang
}

// F-40: degenerate template (zero range) must set an error, not silently
// return all zeros.
void test_calc_variograms_rejects_zero_range()
{
    TEST("F-40: calc_variograms rejects zero ellipsoid range");
    cvar_clear_last_error();
    variogram_search_template_t templ = {};
    templ.m_lag_width = 1.0;
    templ.m_lag_separation = 2.0;
    templ.m_tol_distance = 1.0;
    templ.m_num_lags = 3;
    templ.m_first_lag_distance = 0.0;
    templ.m_ellipsoid.m_R1 = 10.0;
    templ.m_ellipsoid.m_R2 = 0.0;    // degenerate
    templ.m_ellipsoid.m_R3 = 3.0;
    templ.m_ellipsoid.m_direction1.m_data[0] = 1.0;
    templ.m_ellipsoid.m_direction2.m_data[1] = 1.0;
    templ.m_ellipsoid.m_direction3.m_data[2] = 1.0;

    float data_vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    unsigned char mask_vals[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    hard_data_t data = {};
    data.m_data = data_vals;
    data.m_mask = mask_vals;
    data.m_data_shape[0] = 2; data.m_data_shape[1] = 2; data.m_data_shape[2] = 2;
    data.m_mask_shape[0] = 2; data.m_mask_shape[1] = 2; data.m_mask_shape[2] = 2;
    data.m_data_strides[0] = 4; data.m_data_strides[1] = 2; data.m_data_strides[2] = 1;
    data.m_mask_strides[0] = 4; data.m_mask_strides[1] = 2; data.m_mask_strides[2] = 1;

    float out[3] = {0, 0, 0};
    calc_variograms(&templ, &data, out, 3, 100);
    CHECK(std::strlen(cvar_get_last_error()) > 0);
}

// F-40: is_in_tunnel with a zero range must set an error instead of silently
// returning false.
void test_is_in_tunnel_zero_range_sets_error()
{
    TEST("F-40: is_in_tunnel signals zero range");
    cvar_clear_last_error();
    variogram_search_template_t templ = {};
    templ.m_lag_width = 1.0;
    templ.m_lag_separation = 2.0;
    templ.m_tol_distance = 1.0;
    templ.m_num_lags = 3;
    templ.m_first_lag_distance = 0.0;
    templ.m_ellipsoid.m_R1 = 10.0;
    templ.m_ellipsoid.m_R2 = 0.0;
    templ.m_ellipsoid.m_R3 = 3.0;
    templ.m_ellipsoid.m_direction1.m_data[0] = 1.0;
    templ.m_ellipsoid.m_direction2.m_data[1] = 1.0;
    templ.m_ellipsoid.m_direction3.m_data[2] = 1.0;

    vector_t vec = {};
    vec.m_data[0] = 1.0;
    bool inside = is_in_tunnel(&templ, &vec);
    CHECK(!inside);
    CHECK(std::strlen(cvar_get_last_error()) > 0);
}

// F-61: calc_search_template_window must include the LagWidth/2 term so the
// window covers the full lag band (Python parity).
void test_search_template_window_includes_half_lag_width()
{
    TEST("F-61: search window includes LagWidth/2 term");
    cvar_clear_last_error();
    variogram_search_template_t templ = {};
    templ.m_lag_width = 4.0;
    templ.m_lag_separation = 10.0;
    templ.m_tol_distance = 1.0;
    templ.m_num_lags = 10;
    templ.m_first_lag_distance = 0.0;
    templ.m_ellipsoid.m_R1 = 1.0;
    templ.m_ellipsoid.m_R2 = 1.0;
    templ.m_ellipsoid.m_R3 = 1.0;
    templ.m_ellipsoid.m_direction1.m_data[0] = 1.0;
    templ.m_ellipsoid.m_direction2.m_data[1] = 1.0;
    templ.m_ellipsoid.m_direction3.m_data[2] = 1.0;

    search_template_window_t window;
    calc_search_template_window(&templ, &window);

    // Python reference: DI = D1 * (lag_sep*num_lags + first_lag + lag_width/2) * i
    // With i=1 and D1=(1,0,0): max_i = 10*10 + 0 + 4/2 = 102 plus DJ/DK contributions.
    // Pre-fix (missing LagWidth/2): max_i = 100 + DJ/DK.
    // The DJ/DK terms are at most +1 each along i when D2/D3 have an i component;
    // here D2=(0,1,0), D3=(0,0,1), so they contribute zero to i. Therefore the
    // i-extent alone proves the LagWidth/2 term. min_i = 0 comes from i=0.
    CHECK_CLOSE(window.m_max_i, 102.0, 1e-9);
    CHECK_CLOSE(window.m_min_i, 0.0, 1e-9);
}

// 2-M-3: grid-path percent sampling must be reproducible when a seed is
// supplied. Pre-fix the thread_local mt19937 was seeded from
// random_device^time and seed_rand_once() was a documented no-op — identical
// inputs produced different variograms. calc_variograms_seeded re-seeds the
// engine: same inputs + same seed → bit-identical output.
void test_calc_variograms_seeded_reproducible()
{
    TEST("2-M-3: seeded grid-path variogram is reproducible");
    cvar_clear_last_error();
    variogram_search_template_t templ = {};
    templ.m_lag_width = 1.0;
    templ.m_lag_separation = 1.0;
    templ.m_tol_distance = 1.0;
    templ.m_num_lags = 4;
    templ.m_first_lag_distance = 0.0;
    templ.m_ellipsoid.m_R1 = 10.0;
    templ.m_ellipsoid.m_R2 = 5.0;
    templ.m_ellipsoid.m_R3 = 3.0;
    templ.m_ellipsoid.m_direction1.m_data[0] = 1.0;
    templ.m_ellipsoid.m_direction2.m_data[1] = 1.0;
    templ.m_ellipsoid.m_direction3.m_data[2] = 1.0;

    const int nx = 5, ny = 5, nz = 1;
    const size_t n = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> data_vals(n);
    std::vector<unsigned char> mask_vals(n, 1);
    for (size_t i = 0; i < n; ++i)
        data_vals[i] = static_cast<float>(i + 1);

    hard_data_t data = {};
    data.m_data = data_vals.data();
    data.m_mask = mask_vals.data();
    data.m_data_shape[0] = nx; data.m_data_shape[1] = ny; data.m_data_shape[2] = nz;
    data.m_mask_shape[0] = nx; data.m_mask_shape[1] = ny; data.m_mask_shape[2] = nz;
    data.m_data_strides[0] = ny * nz; data.m_data_strides[1] = nz; data.m_data_strides[2] = 1;
    data.m_mask_strides[0] = ny * nz; data.m_mask_strides[1] = nz; data.m_mask_strides[2] = 1;

    const int percent = 50;   // exercises the RNG percent-sampling path
    float out1[4] = {0, 0, 0, 0};
    float out2[4] = {0, 0, 0, 0};
    float out3[4] = {0, 0, 0, 0};

    calc_variograms_seeded(&templ, &data, out1, 4, percent, 12345);
    calc_variograms_seeded(&templ, &data, out2, 4, percent, 12345);
    calc_variograms_seeded(&templ, &data, out3, 4, percent, 54321);

    CHECK(std::strlen(cvar_get_last_error()) == 0);
    // Same seed → bit-identical output (the reproducibility contract).
    CHECK(memcmp(out1, out2, sizeof(out1)) == 0);
    // Different seed → different sampled subset → (with overwhelming
    // probability) a different variogram.
    CHECK(memcmp(out1, out3, sizeof(out1)) != 0);
}

// II-15: the 64-bit seed must be honored in full. Pre-fix the seed was
// truncated to mt19937's 32-bit result_type, so seeds that differ only above
// 2^32 produced bit-identical variograms (seed=5 ≡ seed=2^32+5). Post-fix the
// full 64-bit seed is expanded through std::seed_seq, so every distinct seed
// value yields a distinct engine state and a distinct sampled subset.
void test_calc_variograms_seeded_full_64bit_seed()
{
    TEST("II-15: 64-bit seed honored in full (no truncation)");
    cvar_clear_last_error();
    variogram_search_template_t templ = {};
    templ.m_lag_width = 1.0;
    templ.m_lag_separation = 1.0;
    templ.m_tol_distance = 1.0;
    templ.m_num_lags = 4;
    templ.m_first_lag_distance = 0.0;
    templ.m_ellipsoid.m_R1 = 10.0;
    templ.m_ellipsoid.m_R2 = 5.0;
    templ.m_ellipsoid.m_R3 = 3.0;
    templ.m_ellipsoid.m_direction1.m_data[0] = 1.0;
    templ.m_ellipsoid.m_direction2.m_data[1] = 1.0;
    templ.m_ellipsoid.m_direction3.m_data[2] = 1.0;

    const int nx = 5, ny = 5, nz = 1;
    const size_t n = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> data_vals(n);
    std::vector<unsigned char> mask_vals(n, 1);
    for (size_t i = 0; i < n; ++i)
        data_vals[i] = static_cast<float>(i + 1);

    hard_data_t data = {};
    data.m_data = data_vals.data();
    data.m_mask = mask_vals.data();
    data.m_data_shape[0] = nx; data.m_data_shape[1] = ny; data.m_data_shape[2] = nz;
    data.m_mask_shape[0] = nx; data.m_mask_shape[1] = ny; data.m_mask_shape[2] = nz;
    data.m_data_strides[0] = ny * nz; data.m_data_strides[1] = nz; data.m_data_strides[2] = 1;
    data.m_mask_strides[0] = ny * nz; data.m_mask_strides[1] = nz; data.m_mask_strides[2] = 1;

    const int percent = 50;
    float out_small[4] = {0, 0, 0, 0};
    float out_big[4] = {0, 0, 0, 0};

    const uint64_t seed_small = 5;
    const uint64_t seed_big = (uint64_t(1) << 32) + 5;   // ≡ 5 mod 2^32
    calc_variograms_seeded(&templ, &data, out_small, 4, percent, seed_small);
    calc_variograms_seeded(&templ, &data, out_big, 4, percent, seed_big);

    CHECK(std::strlen(cvar_get_last_error()) == 0);
    // Distinct 64-bit seeds must NOT collide. Pre-fix: static_cast to
    // result_type truncated seed_big to 5 → memcmp == 0 → test FAILS.
    CHECK(memcmp(out_small, out_big, sizeof(out_small)) != 0);
}

// F-31: update_lags must not perform UB FP->int casts. A legal template with
// tiny lag_separation (validate_template only requires > 0) produces a
// quotient far beyond INT_MAX; pre-fix the (int)ceil/floor casts were UB
// ([conv.fpint]) — on x86-64 the positive-overflow cast saturates to INT_MIN
// (pair silently dropped), on arm64 to INT_MAX (pair counted in every lag).
// Post-fix the quotient is range-checked and clamped in double before any
// cast, so a wide lag_width/lag_sep band bins deterministically into every
// lag it covers.
void test_update_lags_no_fp_int_overflow_ub()
{
    TEST("F-31: update_lags guards FP->int overflow");
    cvar_clear_last_error();
    variogram_search_template_t templ = {};
    templ.m_lag_width = 100.0;      // wide band
    templ.m_lag_separation = 1e-9;  // tiny but legal (> 0)
    templ.m_first_lag_distance = 0.0;

    const int lag_count = 5;
    lag_statistics_t stats[5] = {};

    // dist = 1.7: min_q = (1.7 - 50)/1e-9 ≈ -4.8e10, max_q = (1.7 + 50)/1e-9
    // ≈ 5.2e10. Both far outside int range pre-fix. The band covers ALL lags
    // (each lag lies at i*1e-9, well inside [dist-50, dist+50]), so the
    // correct deterministic outcome is every lag counting the pair once.
    update_lags(&templ, stats, lag_count, 1.7, 2.0);

    CHECK(std::strlen(cvar_get_last_error()) == 0);
    for (int i = 0; i < lag_count; ++i)
        CHECK(stats[i].m_cov_count == 1);

    // Out-of-range band (dist beyond the last lag) must drop, not wrap.
    lag_statistics_t stats2[5] = {};
    cvar_clear_last_error();
    update_lags(&templ, stats2, lag_count, 1e6, 2.0);   // min_q ≈ 1e15 ≥ 5
    CHECK(std::strlen(cvar_get_last_error()) == 0);
    for (int i = 0; i < lag_count; ++i)
        CHECK(stats2[i].m_cov_count == 0);
}

// II-16: the grid path must validate hard_data member pointers before the
// nested loop dereferences data->m_data / data->m_mask. Pre-fix a null
// member SIGSEGV'd (uncatchable from Python); post-fix an error is set and
// the call returns cleanly.
void test_calc_variograms_rejects_null_member()
{
    TEST("II-16: calc_variograms rejects null data members");
    cvar_clear_last_error();
    variogram_search_template_t templ = {};
    templ.m_lag_width = 1.0;
    templ.m_lag_separation = 2.0;
    templ.m_tol_distance = 1.0;
    templ.m_num_lags = 3;
    templ.m_first_lag_distance = 0.0;
    templ.m_ellipsoid.m_R1 = 10.0;
    templ.m_ellipsoid.m_R2 = 5.0;
    templ.m_ellipsoid.m_R3 = 3.0;
    templ.m_ellipsoid.m_direction1.m_data[0] = 1.0;
    templ.m_ellipsoid.m_direction2.m_data[1] = 1.0;
    templ.m_ellipsoid.m_direction3.m_data[2] = 1.0;

    // All members null -> pre-fix SIGSEGV; post-fix clean error return.
    hard_data_t data = {};
    data.m_data_shape[0] = 2; data.m_data_shape[1] = 2; data.m_data_shape[2] = 2;
    data.m_mask_shape[0] = 2; data.m_mask_shape[1] = 2; data.m_mask_shape[2] = 2;
    data.m_data_strides[0] = 4; data.m_data_strides[1] = 2; data.m_data_strides[2] = 1;
    data.m_mask_strides[0] = 4; data.m_mask_strides[1] = 2; data.m_mask_strides[2] = 1;

    float out[3] = {0, 0, 0};
    calc_variograms(&templ, &data, out, 3, 100);
    CHECK(std::strlen(cvar_get_last_error()) > 0);
}

// II-56: validate_template must reject tol_distance <= 0 / NaN. Pre-fix NaN
// tol_distance silently produced an all-zero variogram (every is_in_tunnel
// test false) with no error; <= 0 degenerated the off-axis acceptance test.
void test_calc_variograms_rejects_bad_tol_distance()
{
    TEST("II-56: tol_distance <= 0 / NaN rejected");
    cvar_clear_last_error();
    variogram_search_template_t templ = {};
    templ.m_lag_width = 1.0;
    templ.m_lag_separation = 2.0;
    templ.m_tol_distance = 0.0;    // degenerate
    templ.m_num_lags = 3;
    templ.m_first_lag_distance = 0.0;
    templ.m_ellipsoid.m_R1 = 10.0;
    templ.m_ellipsoid.m_R2 = 5.0;
    templ.m_ellipsoid.m_R3 = 3.0;
    templ.m_ellipsoid.m_direction1.m_data[0] = 1.0;
    templ.m_ellipsoid.m_direction2.m_data[1] = 1.0;
    templ.m_ellipsoid.m_direction3.m_data[2] = 1.0;

    float data_vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    unsigned char mask_vals[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    hard_data_t data = {};
    data.m_data = data_vals;
    data.m_mask = mask_vals;
    data.m_data_shape[0] = 2; data.m_data_shape[1] = 2; data.m_data_shape[2] = 2;
    data.m_mask_shape[0] = 2; data.m_mask_shape[1] = 2; data.m_mask_shape[2] = 2;
    data.m_data_strides[0] = 4; data.m_data_strides[1] = 2; data.m_data_strides[2] = 1;
    data.m_mask_strides[0] = 4; data.m_mask_strides[1] = 2; data.m_mask_strides[2] = 1;

    float out[3] = {0, 0, 0};
    calc_variograms(&templ, &data, out, 3, 100);
    CHECK(std::strlen(cvar_get_last_error()) > 0);

    // NaN tol_distance must also be rejected.
    cvar_clear_last_error();
    templ.m_tol_distance = std::numeric_limits<double>::quiet_NaN();
    calc_variograms(&templ, &data, out, 3, 100);
    CHECK(std::strlen(cvar_get_last_error()) > 0);
}

// II-57: a zero-thickness layer must not be treated as erosion. Pre-fix the
// else branch (thickness > 0 is false) blanked [ceil(new_k), nz) — the ENTIRE
// column when the zero layer is first — silently changing the model vs the
// same model without the zero layer. Post-fix thickness==0 is a no-op.
void test_stack_layers_zero_thickness_no_erosion()
{
    TEST("II-57: zero-thickness layer does not erode");
    cvar_clear_last_error();
    // Baseline (probe Case C): deposit 2.0 marker 5 on nz=4.
    std::vector<float_data_t> base_layers;
    base_layers.push_back(make_layer(1, 1, 2.0f));
    int base_markers[1] = {5};
    float_data_t base_result = make_result(1, 1, 4);
    stack_layers(base_layers, base_markers, 4, 1.0f, -99, base_result);
    CHECK(std::strlen(cvar_get_last_error()) == 0);

    // Same model + a leading zero-thickness layer (probe Case B).
    // Pre-fix: the zero layer blanked [0, nz) first, then the deposit
    // rewrote [0,2) — leaving [2,4) at blank_value instead of the 7.0 the
    // probe's initial buffer held. Post-fix the zero layer is skipped, so
    // the deposit runs exactly as the baseline.
    std::vector<float_data_t> layers;
    layers.push_back(make_layer(1, 1, 0.0f));  // zero-thickness first
    layers.push_back(make_layer(1, 1, 2.0f));  // then the same deposit
    int markers[2] = {9, 5};
    float_data_t result = make_result(1, 1, 4);
    stack_layers(layers, markers, 4, 1.0f, -99, result);
    CHECK(std::strlen(cvar_get_last_error()) == 0);

    // The zero layer must not change the deposit footprint or markers.
    for (int k = 0; k < 4; ++k)
        CHECK(result.m_data[k] == base_result.m_data[k]);
    CHECK(result.m_data[0] == 5.0f);
    CHECK(result.m_data[1] == 5.0f);
    // Top tail (III-40) is blank_value either way — the point of II-57 is
    // that the zero layer did NOT blank the deposit region [0,2).
    CHECK(result.m_data[2] == -99.0f);
    CHECK(result.m_data[3] == -99.0f);
}

// III-40: every result cell must receive a defined value. Pre-fix top-tail
// cells above the final surface were left unwritten: a NaN-prefilled buffer
// triggered the Python wrapper's post-call NaN RuntimeError, and buffer reuse
// silently corrupted stale cells. Post-fix the result is initialized to
// blank_value so no cell depends on the caller's prefill.
void test_stack_layers_result_fully_initialized()
{
    TEST("III-40: stack_layers defines all result cells");
    cvar_clear_last_error();
    // Deposit 1.0 marker 7 on nz=4; pre-fill the buffer with NaN to
    // reproduce the spurious NaN RuntimeError (cvariogram.py:740-745).
    std::vector<float> nan_buf(4, std::numeric_limits<float>::quiet_NaN());
    float_data_t nan_result;
    nan_result.m_data = nan_buf.data();
    nan_result.m_data_shape[0] = 1; nan_result.m_data_shape[1] = 1; nan_result.m_data_shape[2] = 4;
    nan_result.m_data_strides[0] = 4; nan_result.m_data_strides[1] = 4; nan_result.m_data_strides[2] = 1;

    std::vector<float_data_t> layers;
    layers.push_back(make_layer(1, 1, 1.0f));
    int markers[1] = {7};
    stack_layers(layers, markers, 4, 1.0f, -99, nan_result);
    CHECK(std::strlen(cvar_get_last_error()) == 0);
    CHECK(nan_result.m_data[0] == 7.0f);
    // Every top-tail cell must be the defined blank_value — no NaN left.
    for (int k = 1; k < 4; ++k)
        CHECK(nan_result.m_data[k] == -99.0f);
}

// ---- Main ----

int main() {
    test_stack_layers_rejects_small_result_grid();
    test_stack_layers_rejects_non_contiguous_layer();
    test_stack_layers_integer_deposit_fills_one_cell();
    test_stack_layers_three_integer_deposits_three_cells();
    test_stack_layers_rejects_nan_thickness();
    test_stack_layers_rejects_huge_thickness();
    test_cvar_clear_last_error_resets();
    test_calc_variograms_rejects_oversized_window();
    test_calc_variograms_rejects_zero_range();
    test_is_in_tunnel_zero_range_sets_error();
    test_search_template_window_includes_half_lag_width();
    test_calc_variograms_seeded_reproducible();
    test_calc_variograms_seeded_full_64bit_seed();
    test_update_lags_no_fp_int_overflow_ub();
    test_calc_variograms_rejects_null_member();
    test_calc_variograms_rejects_bad_tol_distance();
    test_stack_layers_zero_thickness_no_erosion();
    test_stack_layers_result_fully_initialized();

    std::printf("C++ cvariogram tests: %d run, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
