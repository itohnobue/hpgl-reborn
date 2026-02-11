cmake_minimum_required(VERSION 3.20)
project(HPGL VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Project Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(HPGL_BUILD_PYTHON "Build Python bindings" ON)
option(HPGL_BUILD_TESTS "Build tests" OFF)
option(HPGL_USE_OPENMP "Enable OpenMP parallelization" ON)
option(HPGL_USE_MKL "Use Intel MKL instead of OpenBLAS" OFF)
option(HPGL_FETCH_EIGEN "Fetch Eigen with FetchContent" ON)
option(HPGL_BUILD_SHARED "Build shared libraries" ON)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Dependencies
# ============================================================================

# OpenMP
if(HPGL_USE_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        message(STATUS "OpenMP found - version ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found - parallel algorithms will be disabled")
    endif()
endif()

# BLAS/LAPACK
if(HPGL_USE_MKL)
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        message(STATUS "Intel MKL found")
        set(BLAS_LIBRARIES MKL::MKL)
        set(LAPACK_LIBRARIES MKL::MKL)
    else()
        message(FATAL_ERROR "HPGL_USE_MKL is ON but Intel MKL not found")
    endif()
else()
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
        message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
    else()
        message(WARNING "BLAS/LAPACK not found - using custom implementation")
        set(BLAS_LIBRARIES "")
        set(LAPACK_LIBRARIES "")
    endif()
endif()

# Eigen (linear algebra)
if(HPGL_FETCH_EIGEN)
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_PROGRESS TRUE
    )
    # For Windows: make sure to pull content even if already present
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        message(STATUS "Fetching Eigen from GitLab...")
        FetchContent_Populate(eigen)
    else()
        message(STATUS "Eigen already populated at ${eigen_SOURCE_DIR}")
    endif()
    add_library(eigen INTERFACE)
    target_include_directories(eigen INTERFACE
        $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
    )
else()
    find_package(Eigen3 3.4 REQUIRED)
    add_library(eigen ALIAS Eigen3::Eigen)
endif()

# Python (for bindings)
if(HPGL_BUILD_PYTHON)
    find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
    message(STATUS "Python: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")

    # pybind11
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "pybind11: ${pybind11_VERSION}")
endif()

# ============================================================================
# Source Files
# ============================================================================

# HPGL core library sources
file(GLOB_RECURSE HPGL_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/hpgl/*.cpp
)

# Exclude precompiled header from non-MSVC builds
if(NOT MSVC)
    list(FILTER HPGL_CORE_SOURCES EXCLUDE REGEX ".*stdafx.cpp$")
endif()

# BsGTL library sources
file(GLOB_RECURSE HPGL_GSTL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bs_gstl/src/*.cpp
)

# Exclude precompiled header
if(NOT MSVC)
    list(FILTER HPGL_GSTL_SOURCES EXCLUDE REGEX ".*stdafx.cpp$")
endif()

# ============================================================================
# Library: hpgl_gstl
# ============================================================================

add_library(hpgl_gstl ${HPGL_GSTL_SOURCES})
target_include_directories(hpgl_gstl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/bs_gstl/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/tnt_126>
        $<INSTALL_INTERFACE:include>
)

add_library(hpgl::gstl ALIAS hpgl_gstl)

# ============================================================================
# Library: hpgl_core
# ============================================================================

add_library(hpgl_core ${HPGL_CORE_SOURCES})
add_library(hpgl::core ALIAS hpgl_core)

target_include_directories(hpgl_core
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/bs_gstl/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/tnt_126>
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(hpgl_core
    PUBLIC
        hpgl_gstl
        eigen
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(hpgl_core
        PRIVATE
            HPGL_DLL
            HPGL_EXPORTS
            _CRT_SECURE_NO_WARNINGS
    )
    # Windows-specific runtime library
    if(NOT HPGL_BUILD_SHARED)
        set_property(TARGET hpgl_core PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug:Debug>")
    endif()
else()
    target_compile_definitions(hpgl_core PUBLIC HPGL_SO)
endif()

# OpenMP support
if(HPGL_USE_OPENMP AND OPENMP_FOUND)
    target_link_libraries(hpgl_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(hpgl_core PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
    )
endif()

# ============================================================================
# Python Bindings
# ============================================================================

if(HPGL_BUILD_PYTHON)
    # Python binding sources
    file(GLOB_RECURSE HPGL_PYTHON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/_cvariogram/*.cpp
    )

    # Main C API file (for ctypes interface)
    set(HPGL_C_API_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/hpgl/api.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/hpgl/output.cpp
    )

    # Check if we have boost::python bindings (hpgl.pyd)
    set(HPGL_BOOST_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/hpgl.pyd")
    if(EXISTS "${HPGL_BOOST_PYTHON_DIR}/module.cpp")
        file(GLOB_RECURSE HPGL_BOOST_PYTHON_SOURCES
            ${HPGL_BOOST_PYTHON_DIR}/*.cpp
        )
    else()
        set(HPGL_BOOST_PYTHON_SOURCES "")
        message(STATUS "No boost::python bindings found (hpgl.pyd/module.cpp)")
    endif()

    # Create the Python extension module with pybind11
    # Note: HPGL uses ctypes for the main interface, so we create a minimal pybind11 wrapper
    pybind11_add_module(hpgl_py ${HPGL_BOOST_PYTHON_SOURCES})

    target_link_libraries(hpgl_py PRIVATE hpgl_core)

    # Include directories
    target_include_directories(hpgl_py PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd
    )

    # NumPy support
    target_compile_definitions(hpgl_py PRIVATE
        PY_ARRAY_UNIQUE_SYMBOL=hpgl_py_ARRAY_API
    )

    # For Windows: ensure correct DLL extension
    if(WIN32 AND NOT CYGWIN)
        set_target_properties(hpgl_py PROPERTIES
            PREFIX ""
            SUFFIX ".pyd"
        )
    endif()

    # Python module visibility
    if(UNIX AND NOT APPLE)
        target_link_libraries(hpgl_py PRIVATE "-Wl,--no-undefined")
    endif()
endif()

# ============================================================================
# Variogram Module (optional)
# ============================================================================

option(HPGL_BUILD_VARIOGRAM "Build cvariogram module" ON)
if(HPGL_BUILD_VARIOGRAM)
    file(GLOB HPGL_VARIOGRAM_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/_cvariogram/*.cpp
    )

    add_library(hpgl_variogram SHARED ${HPGL_VARIOGRAM_SOURCES})

    target_include_directories(hpgl_variogram PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/_cvariogram
    )

    if(WIN32)
        target_compile_definitions(hpgl_variogram PRIVATE CVAR_DLL CVAR_EXPORTS)
        set_target_properties(hpgl_variogram PROPERTIES
            PREFIX "_"
            SUFFIX "_d"
            OUTPUT_NAME "_cvariogram"
        )
    elseif(UNIX)
        # On Unix, Python expects underscore prefix for extension modules
        set_target_properties(hpgl_variogram PROPERTIES
            PREFIX "_"
            OUTPUT_NAME "_cvariogram"
        )
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

# Install core library
install(TARGETS hpgl_core hpgl_gstl
    EXPORT hpglTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers
install(DIRECTORY
    src/geo_bsd/hpgl/
    src/bs_gstl/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpgl
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install Python bindings
if(HPGL_BUILD_PYTHON)
    install(TARGETS hpgl_py
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/hpgl
    )

    # Install Python wrapper scripts
    install(FILES
        src/geo_bsd/__init__.py
        src/geo_bsd/geo.py
        src/geo_bsd/sgs.py
        src/geo_bsd/sis.py
        src/geo_bsd/variogram.py
        src/geo_bsd/validation.py
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/hpgl
    )
endif()

# Install variogram module
if(HPGL_BUILD_VARIOGRAM)
    install(TARGETS hpgl_variogram
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/hpgl
    )
endif()

# Install Eigen interface library (needed for exports)
install(TARGETS eigen
    EXPORT hpglTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create export targets
install(EXPORT hpglTargets
    FILE hpglTargets.cmake
    NAMESPACE hpgl::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpgl
)

# ============================================================================
# Testing
# ============================================================================

if(HPGL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "================== HPGL Configuration Summary ==================")
message(STATUS "Version:             ${PROJECT_VERSION}")
message(STATUS "Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix:      ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  HPGL_BUILD_PYTHON:     ${HPGL_BUILD_PYTHON}")
message(STATUS "  HPGL_BUILD_TESTS:      ${HPGL_BUILD_TESTS}")
message(STATUS "  HPGL_USE_OPENMP:       ${HPGL_USE_OPENMP}")
message(STATUS "  HPGL_USE_MKL:          ${HPGL_USE_MKL}")
message(STATUS "  HPGL_BUILD_VARIOGRAM:  ${HPGL_BUILD_VARIOGRAM}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  BLAS:                ${BLAS_LIBRARIES}")
message(STATUS "  LAPACK:              ${LAPACK_LIBRARIES}")
message(STATUS "  Eigen:               ${eigen_SOURCE_DIR}")
message(STATUS "  OpenMP:              ${OPENMP_FOUND}")
if(HPGL_BUILD_PYTHON)
    message(STATUS "  Python:              ${Python3_EXECUTABLE}")
    message(STATUS "  pybind11:            ${pybind11_VERSION}")
endif()
message(STATUS "====================================================================")
message(STATUS "")
