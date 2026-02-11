cmake_minimum_required(VERSION 3.20)

# ============================================================================
# HPGL Test Suite
# ============================================================================

# C++ Unit Tests
option(HPGL_BUILD_CPP_TESTS "Build C++ unit tests" OFF)

if(HPGL_BUILD_CPP_TESTS)
    enable_testing()

    # Test executable
    add_executable(hpgl_test
        tests/test_main.cpp
        tests/test_kriging.cpp
        tests/test_grid.cpp
        tests/test_covariance.cpp
    )

    target_link_libraries(hpgl_test PRIVATE hpgl::core)

    add_test(NAME hpgl_cpp_tests COMMAND hpgl_test)
endif()

# Python Tests
if(HPGL_BUILD_PYTHON)
    find_package(pytest QUIET)

    if(pytest_FOUND)
        add_test(NAME hpgl_python_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/python
        )
    else()
        message(STATUS "pytest not found - Python tests will not run")
        add_test(NAME hpgl_python_tests
            COMMAND ${Python3_EXECUTABLE} -c "import hpgl; print('HPGL import successful')"
        )
    endif()
endif()
