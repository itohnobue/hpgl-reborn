cmake_minimum_required(VERSION 3.20)

# ============================================================================
# HPGL Test Suite
# ============================================================================

# Python Tests
if(HPGL_BUILD_PYTHON)
    # Pin the Python interpreter to the project venv when present so CTest
    # runs pytest against the lockfile-pinned environment instead of whatever
    # interpreter find_package(Python3) resolves to (F-56: ambient miniconda
    # base interpreter drift).
    set(_hpgl_test_python ${Python3_EXECUTABLE})
    if(EXISTS ${CMAKE_SOURCE_DIR}/.venv/bin/python)
        set(_hpgl_test_python ${CMAKE_SOURCE_DIR}/.venv/bin/python)
    elseif(WIN32 AND EXISTS ${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe)
        set(_hpgl_test_python ${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe)
    endif()
    # NOTE: no `-v` flag — test_ffi_contract.py::test_safe_load_library_path_escape_raises
    # monkeypatches pathlib.Path.__truediv__ globally; under pytest's verbose
    # nodeid reporting (pytest 9.0.2 AND 9.0.3) that leaks into cwd_relative_nodeid
    # and crashes pytest with INTERNALERROR ('str' object has no attribute 'parts').
    add_test(NAME hpgl_python_tests
        COMMAND ${_hpgl_test_python} -m pytest
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
endif()

# ============================================================================
# C++ Unit Tests (minimal, no external test framework)
# ============================================================================

# Build a standalone C++ test binary that verifies core numerical routines.
# Links against the hpgl_core library (same as Python extension but tested
# independently).
add_executable(hpgl_core_tests
    cpp/test_hpgl_core.cpp
)

target_include_directories(hpgl_core_tests PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/geo_bsd/hpgl
)
target_include_directories(hpgl_core_tests SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tnt_126
)

target_link_libraries(hpgl_core_tests PRIVATE hpgl_core)
# s6-fix-cpp-engines: test_hpgl_core.cpp now instantiates the header-only
# weight-calculator templates (ok_kriging_weights_3 / sk_kriging_weights_3),
# which call dpotrf_/dpotrs_ directly (solver_entry_point.h). The binary must
# link BLAS/LAPACK itself for the same reason hpgl_solver_tests does (F-59),
# since hpgl_core links them PRIVATE.
target_link_libraries(hpgl_core_tests PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

target_compile_features(hpgl_core_tests PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(hpgl_core_tests PRIVATE /W3)
else()
    target_compile_options(hpgl_core_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME hpgl_cpp_tests
    COMMAND hpgl_core_tests
)

# ============================================================================
# LAPACK solver entry-point tests
# ============================================================================
# test_solver_entry_point.cpp exercises lapack_spd_solve_1rhs/2rhs (SPD
# success, non-SPD gauss fallback, NaN/Inf rejection). The solver templates
# call dpotrf_/dpotrs_ directly (solver_entry_point.h), so the binary must
# link BLAS/LAPACK itself even though hpgl_core links them PRIVATE (F-59:
# previously orphaned — never built or run by CTest).
add_executable(hpgl_solver_tests
    cpp/test_solver_entry_point.cpp
)

target_include_directories(hpgl_solver_tests PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/geo_bsd/hpgl
)
target_include_directories(hpgl_solver_tests SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tnt_126
)

target_link_libraries(hpgl_solver_tests PRIVATE hpgl_core)
target_link_libraries(hpgl_solver_tests PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

target_compile_features(hpgl_solver_tests PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(hpgl_solver_tests PRIVATE /W3)
else()
    target_compile_options(hpgl_solver_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME hpgl_solver_entry_point_tests
    COMMAND hpgl_solver_tests
)

# ============================================================================
# cvariogram module regression tests
# ============================================================================
# test_cvariogram.cpp exercises the cvariogram module fixes (F-06, F-08, F-37,
# F-38, F-39, F-40, F-61, I2-25) — heap/contiguity OOB rejection, error clear,
# window caps, zero-range handling, and half-lag-width window extent. The file
# compiles the module sources (stack_layers.cpp / variograms.cpp / ellipsoid.cpp)
# directly into the test binary (see the file header), so it does NOT link
# hpgl_variogram or hpgl_core — it needs only the cvariogram include path.
# PR-03: previously orphaned — the 11 regression tests never built or ran
# under CTest, leaving the HIGH-severity F-06/F-08 memory-safety fixes without
# an automated gate.
add_executable(hpgl_variogram_tests
    cpp/test_cvariogram.cpp
)

target_include_directories(hpgl_variogram_tests PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/geo_bsd/_cvariogram
)

target_compile_features(hpgl_variogram_tests PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(hpgl_variogram_tests PRIVATE /W3)
else()
    target_compile_options(hpgl_variogram_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME hpgl_variogram_tests
    COMMAND hpgl_variogram_tests
)
