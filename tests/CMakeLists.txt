cmake_minimum_required(VERSION 3.20)

# ============================================================================
# HPGL Test Suite
# ============================================================================

# C++ Unit Tests
option(HPGL_BUILD_CPP_TESTS "Build C++ unit tests" OFF)

if(HPGL_BUILD_CPP_TESTS)
    enable_testing()

    # Test executable (paths relative to this CMakeLists.txt directory)
    add_executable(hpgl_test
        cpp/test_main.cpp
        cpp/test_kriging.cpp
        cpp/test_grid.cpp
        cpp/test_covariance.cpp
    )

    target_link_libraries(hpgl_test PRIVATE hpgl::core)

    add_test(NAME hpgl_cpp_tests COMMAND hpgl_test)
endif()

# Python Tests
if(HPGL_BUILD_PYTHON)
    # Always register Python test via pytest (pytest availability checked at runtime)
    add_test(NAME hpgl_python_tests
        COMMAND ${Python3_EXECUTABLE} -m pytest -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
endif()
