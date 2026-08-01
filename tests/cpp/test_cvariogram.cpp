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
    CHECK(result.m_data[1] == 0.0f);       // cell 1 NOT marked (pre-fix: 1.0)
    CHECK(result.m_data[2] == 0.0f);
    CHECK(result.m_data[3] == 0.0f);
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
    CHECK(result.m_data[3] == 0.0f);       // pre-fix: 3.0f (4th cell over-fill)
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

    std::printf("C++ cvariogram tests: %d run, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
