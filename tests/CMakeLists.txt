cmake_minimum_required(VERSION 3.20)

# ============================================================================
# HPGL Test Suite
# ============================================================================

# Python Tests
if(HPGL_BUILD_PYTHON)
    # Always register Python test via pytest (pytest availability checked at runtime)
    add_test(NAME hpgl_python_tests
        COMMAND ${Python3_EXECUTABLE} -m pytest -v
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

target_compile_features(hpgl_core_tests PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(hpgl_core_tests PRIVATE /W3)
else()
    target_compile_options(hpgl_core_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME hpgl_cpp_tests
    COMMAND hpgl_core_tests
)
