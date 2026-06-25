cmake_minimum_required(VERSION 3.20)

# ============================================================================
# BUILD SYSTEM NOTE
# This CMakeLists.txt is the canonical build system for HPGL.
# The MSVC vcxproj files and build.bat are secondary and MUST be manually
# synced when sources, flags, or dependencies change in this file.
# There is no automated sync between CMake and the Visual Studio project files.
# ============================================================================

# Read version from pyproject.toml (single source of truth).
# When built via scikit-build-core, SKBUILD_PROJECT_VERSION is set automatically.
# Fallback for standalone CMake builds reads pyproject.toml directly.
if(DEFINED SKBUILD_PROJECT_VERSION)
    set(_hpgl_version ${SKBUILD_PROJECT_VERSION})
else()
    file(STRINGS "${CMAKE_SOURCE_DIR}/pyproject.toml" _hpgl_ver_line REGEX "^version *= ")
    string(REGEX REPLACE "^version *= *\"([^\"]+)\".*" "\\1" _hpgl_version "${_hpgl_ver_line}")
endif()
project(HPGL VERSION ${_hpgl_version} LANGUAGES CXX)

# ============================================================================
# Project Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Link-Time Optimization (LTO/IPO) for Release builds.
# Equivalent to vcxproj Release configs that enable LTO via /GL and /LTCG.
include(CheckIPOSupported)
check_ipo_supported(RESULT _hpgl_ipo_supported OUTPUT _hpgl_ipo_output)
if(_hpgl_ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    message(STATUS "IPO/LTO enabled for Release builds")
else()
    message(STATUS "IPO/LTO not supported by this toolchain")
endif()

# Compiler warning flags (set per-target via target_compile_options)
# NOTE: Both CMake and MSVC vcxproj now use /W3 (Level3) for consistency.
if(MSVC)
    set(HPGL_COMPILE_WARNINGS /W3 /wd4100 /wd4127 /wd4514 /wd4710 /wd4820)
else()
    set(HPGL_COMPILE_WARNINGS -Wall -Wextra -Wpedantic)
    # TNT numeric library headers in src/tnt_126/ trigger -Wdeprecated-copy and
    # -Wc++11-narrowing on Apple Clang 14+. Suppress these for clean builds.
    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        list(APPEND HPGL_COMPILE_WARNINGS -Wno-deprecated-copy -Wno-c++11-narrowing)
    endif()
endif()

# Build options
option(HPGL_BUILD_PYTHON "Build Python bindings" ON)
option(HPGL_BUILD_TESTS "Build tests" OFF)
option(HPGL_USE_OPENMP "Enable OpenMP parallelization" ON)
option(HPGL_USE_MKL "Use Intel MKL instead of OpenBLAS" OFF)

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
# NOTE: MKL link order is now consistent across all vcxproj configurations:
#   mkl_core.lib;mkl_intel_lp64.lib;mkl_sequential.lib (all 4 configs)
# CMake delegates link order to FindMKL which resolves MKL::MKL target dependencies automatically.
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
        message(FATAL_ERROR "BLAS/LAPACK not found - one of these libraries is required. "
                            "Install BLAS/LAPACK packages for your platform or set HPGL_USE_MKL=ON.")
    endif()
endif()

# Eigen is not a direct dependency (no #include <Eigen/...> in HPGL source).
# The code uses LAPACK via the configured BLAS/LAPACK libraries above.

# Python (for bindings)
if(HPGL_BUILD_PYTHON)
    find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
    message(STATUS "Python: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")
endif()

# ============================================================================
# Source Files
# ============================================================================

# HPGL core library sources
file(GLOB_RECURSE HPGL_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/hpgl/*.cpp
)

# Exclude precompiled header from non-MSVC builds
if(NOT MSVC)
    list(FILTER HPGL_CORE_SOURCES EXCLUDE REGEX ".*stdafx.cpp$")
endif()

# ============================================================================
# Library: hpgl_core
# ============================================================================

add_library(hpgl_core SHARED ${HPGL_CORE_SOURCES})
add_library(hpgl::core ALIAS hpgl_core)

# Ensure consistent output name across platforms (Python bindings expect "hpgl")
set_target_properties(hpgl_core PROPERTIES
    OUTPUT_NAME "hpgl"
)

# Compiler warning flags applied at target scope
target_compile_options(hpgl_core PRIVATE ${HPGL_COMPILE_WARNINGS})

target_include_directories(hpgl_core
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/hpgl>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/tnt_126>
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

# BLAS/LAPACK: PUBLIC for MKL (consumers need the MKL target), PRIVATE otherwise
# (avoids baking raw library paths into export targets)
if(HPGL_USE_MKL AND MKL_FOUND)
    target_link_libraries(hpgl_core PUBLIC MKL::MKL)
else()
    target_link_libraries(hpgl_core PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# MKL compile definition (after target creation)
if(HPGL_USE_MKL AND MKL_FOUND)
    target_compile_definitions(hpgl_core PRIVATE HPGL_USE_MKL USE_INTEL_MKL)
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(hpgl_core
        PRIVATE
            HPGL_DLL
            HPGL_EXPORTS
            BS_EXPORTING_PLUGIN
            NOMINMAX
    )
    # Windows-specific runtime library
    # NOTE: hpgl_core is built as SHARED; when linking dynamically this default is appropriate.
    # To link statically, change add_library to STATIC and set runtime library:
    #   set_property(TARGET hpgl_core PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_compile_definitions(hpgl_core PUBLIC HPGL_SO)
endif()

# Security hardening flags (non-Debug builds only)
# Debug builds skip these to avoid interfering with debugging/sanitizers.
# Generator expressions ensure correct behavior for BOTH single-config
# (Make, Ninja) and multi-config (VS, Xcode) generators.
if(MSVC)
    # /sdl: Additional security checks (buffer overrun, strict const, etc.)
    # /guard:cf: Control Flow Guard (compile + link)
    target_compile_options(hpgl_core PRIVATE $<$<NOT:$<CONFIG:Debug>>:/sdl> $<$<NOT:$<CONFIG:Debug>>:/guard:cf>)
    target_link_options(hpgl_core PRIVATE $<$<NOT:$<CONFIG:Debug>>:/guard:cf>)
else()
    # -fstack-protector-strong: Stack canary protection
    # -ffile-prefix-map: Reproducible build paths
    # _FORTIFY_SOURCE=2: Compile-time buffer overflow detection
    # -Wl,-z,relro,-z,now: Full RELRO (read-only GOT, immediate binding)
    # -Wl,-z,noexecstack: Non-executable stack
    # -D_GLIBCXX_ASSERTIONS: libstdc++ runtime assertions (C++ only)
    target_compile_options(hpgl_core PRIVATE $<$<NOT:$<CONFIG:Debug>>:-fstack-protector-strong> $<$<NOT:$<CONFIG:Debug>>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>)
    target_compile_definitions(hpgl_core PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<PLATFORM_ID:Darwin>>>:_FORTIFY_SOURCE=2>)
    target_link_options(hpgl_core PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<PLATFORM_ID:Darwin>>>:-Wl,-z,relro,-z,now>)
    target_link_options(hpgl_core PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<PLATFORM_ID:Darwin>>>:-Wl,-z,noexecstack>)
    target_compile_definitions(hpgl_core PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<PLATFORM_ID:Darwin>>>:_GLIBCXX_ASSERTIONS>)
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

# ============================================================================
# Variogram Module (optional)
# ============================================================================

option(HPGL_BUILD_VARIOGRAM "Build cvariogram module" ON)
if(HPGL_BUILD_VARIOGRAM)
    file(GLOB HPGL_VARIOGRAM_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/_cvariogram/*.cpp
    )

    add_library(hpgl_variogram SHARED ${HPGL_VARIOGRAM_SOURCES})

    target_compile_options(hpgl_variogram PRIVATE ${HPGL_COMPILE_WARNINGS})

    target_include_directories(hpgl_variogram PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/_cvariogram
    )

    if(WIN32)
        # NOTE: CMake produces _cvariogram.dll for all configurations (Debug and Release).
        # The MSVC vcxproj (cvariogram.vcxproj) produces _cvariogram_d.dll for Debug
        # and _cvariogram.dll for Release. This is an intentional difference — the vcxproj
        # is the primary Windows build system and build.bat handles the Debug suffix naming.
        # CMake builds on Windows use the suffix-less name for both configurations.
        target_compile_definitions(hpgl_variogram PRIVATE CVAR_DLL CVAR_EXPORTS)
        set_target_properties(hpgl_variogram PROPERTIES
            PREFIX ""
            OUTPUT_NAME "_cvariogram"
        )
    elseif(UNIX)
        # On Unix, Python expects underscore prefix for extension modules
        set_target_properties(hpgl_variogram PROPERTIES
            PREFIX "_"
            OUTPUT_NAME "cvariogram"
        )
    endif()

    # Security hardening flags for variogram module (non-Debug builds only)
    # Generator expressions ensure correct behavior for BOTH single-config
    # (Make, Ninja) and multi-config (VS, Xcode) generators.
    if(MSVC)
        target_compile_options(hpgl_variogram PRIVATE $<$<NOT:$<CONFIG:Debug>>:/sdl> $<$<NOT:$<CONFIG:Debug>>:/guard:cf>)
        target_link_options(hpgl_variogram PRIVATE $<$<NOT:$<CONFIG:Debug>>:/guard:cf>)
    else()
        target_compile_options(hpgl_variogram PRIVATE $<$<NOT:$<CONFIG:Debug>>:-fstack-protector-strong> $<$<NOT:$<CONFIG:Debug>>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>)
        target_compile_definitions(hpgl_variogram PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<PLATFORM_ID:Darwin>>>:_FORTIFY_SOURCE=2>)
        target_link_options(hpgl_variogram PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<PLATFORM_ID:Darwin>>>:-Wl,-z,relro,-z,now>)
        target_link_options(hpgl_variogram PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<PLATFORM_ID:Darwin>>>:-Wl,-z,noexecstack>)
        target_compile_definitions(hpgl_variogram PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<PLATFORM_ID:Darwin>>>:_GLIBCXX_ASSERTIONS>)
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

# Install core library
# Native library must be co-located with Python wrapper scripts so that
# _safe_load_library() (which searches ref_path.parent) can find it.
install(TARGETS hpgl_core
    EXPORT hpglTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/hpgl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/hpgl
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/hpgl
)

# Install headers
install(DIRECTORY
    src/geo_bsd/hpgl/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpgl
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install Python wrapper scripts
if(HPGL_BUILD_PYTHON)
    install(FILES
        src/geo_bsd/__init__.py
        src/geo_bsd/cdf.py
        src/geo_bsd/cvariogram.py
        src/geo_bsd/geo.py
        src/geo_bsd/gtsim.py
        src/geo_bsd/hpgl_wrap.py
        src/geo_bsd/routines.py
        src/geo_bsd/sgs.py
        src/geo_bsd/sis.py
        src/geo_bsd/validation.py
        src/geo_bsd/variogram.py
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/hpgl
    )
endif()

# Install variogram module
if(HPGL_BUILD_VARIOGRAM)
    install(TARGETS hpgl_variogram
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/hpgl
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/hpgl
    )
endif()

# Create export targets
install(EXPORT hpglTargets
    FILE hpglTargets.cmake
    NAMESPACE hpgl::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpgl
)

# Generate hpglConfig.cmake so consumers can use find_package(hpgl)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hpglConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hpglConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hpgl"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hpglConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hpglConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/hpglConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hpgl"
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
message(STATUS "  Eigen:               not used (no Eigen headers in source)")
message(STATUS "  OpenMP:              ${OPENMP_FOUND}")
if(HPGL_BUILD_PYTHON)
    message(STATUS "  Python:              ${Python3_EXECUTABLE}")
endif()
message(STATUS "====================================================================")
message(STATUS "")
