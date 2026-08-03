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

# macOS deployment target: pin at the CMake level so EVERY build path (local
# builds, wheel builds) targets macOS 11.0 instead of the host-macOS version.
# Without this, a wheel built on macOS 15 tags macosx_15_0 and pip rejects it
# on macOS 11-14 (H-3). Overridable via -DCMAKE_OSX_DEPLOYMENT_TARGET=...
# (covers both undefined AND stale-empty cache entries).
if(APPLE AND (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL ""))
    set(CMAKE_OSX_DEPLOYMENT_TARGET 11.0 CACHE STRING "Minimum macOS deployment target" FORCE)
endif()

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
# NOTE: Both CMake and MSVC vcxproj suppress warnings at /W3 level but the
# suppression sets are intentionally different per build system:
#   CMake (MSVC): /wd4100 /wd4127 /wd4514 /wd4710 /wd4820
#   vcxproj:      4003;4244;4267;4018
# The CMake MSVC set targets pedantic warnings from C++ standards conformance
# checks in generic template code. The vcxproj set targets warnings from MSVC's
# own static analysis that don't apply to HPGL's specific code patterns.
# These sets are maintained independently — sync is not automatic.
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

# macOS: statically link libomp so the produced dylibs carry NO absolute
# Homebrew libomp runtime dependency — the wheel is then relocatable without
# delocate repair (H-3). Default ON on Apple, OFF elsewhere (other platforms
# link the system OpenMP runtime dynamically, which is the local convention).
if(APPLE)
    set(_hpgl_static_libomp_default ON)
else()
    set(_hpgl_static_libomp_default OFF)
endif()
option(HPGL_STATIC_LIBOMP "Statically link libomp on macOS for relocatable artifacts" ${_hpgl_static_libomp_default})

# R-26/R-31: stale-cache detection. The option() call above does NOT override a
# cached value, so a build directory configured before HPGL_STATIC_LIBOMP
# defaulted to ON silently keeps producing non-relocatable dylibs (absolute
# Homebrew libomp runtime dependency — the H-3 defect class) with no error.
# Warn loudly when macOS is configured with the option OFF so the stale-cache
# mechanism cannot silently disable the H-3 protection.
if(APPLE AND NOT HPGL_STATIC_LIBOMP)
    message(WARNING "HPGL_STATIC_LIBOMP is OFF on macOS: the produced dylibs will carry an absolute "
                    "Homebrew libomp runtime dependency and are NOT relocatable. This is usually a stale "
                    "CMake cache from before the option defaulted to ON — reconfigure with "
                    "-DHPGL_STATIC_LIBOMP=ON (or delete the cache dir) for self-contained artifacts.")
endif()

# BLAS vendor selection (only used when HPGL_USE_MKL=OFF).
# Set to pin a specific vendor: OpenBLAS, Intel10_64lp, Apple, etc.
# Leave empty to let CMake auto-detect.
set(BLA_VENDOR "" CACHE STRING "BLAS vendor to use (OpenBLAS, Intel10_64lp, Apple, etc.)")

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Dependencies
# ============================================================================

# OpenMP
if(HPGL_USE_OPENMP)
    # macOS/AppleClang: CMake's FindOpenMP does not search the Homebrew
    # prefix by default, so HPGL_USE_OPENMP=ON silently degraded to a
    # serial build on Apple Silicon/Intel even though libomp is installed
    # (production-check gap: wheels built on macOS shipped without OpenMP).
    # Point OpenMP_ROOT at the Homebrew libomp installation when present.
    # The probe is skipped for universal (multi-arch) builds: Homebrew
    # prefixes are single-arch, so an auto-picked libomp would force one
    # arch and break the other (F-2) — universal builds must set
    # OpenMP_ROOT explicitly per-arch in the packaging environment.
    if(APPLE AND NOT DEFINED OpenMP_ROOT)
        set(_hpgl_osx_archs "${CMAKE_OSX_ARCHITECTURES}")
        set(_hpgl_probe_prefixes "")
        if(NOT _hpgl_osx_archs)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
                set(_hpgl_probe_prefixes "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
            else()
                set(_hpgl_probe_prefixes "/usr/local/opt/libomp" "/opt/homebrew/opt/libomp")
            endif()
        elseif(NOT _hpgl_osx_archs MATCHES ";")
            # Single explicitly-listed arch: probe both prefixes, host-arch first.
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
                set(_hpgl_probe_prefixes "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
            else()
                set(_hpgl_probe_prefixes "/usr/local/opt/libomp" "/opt/homebrew/opt/libomp")
            endif()
        endif()
        foreach(_brew_prefix IN LISTS _hpgl_probe_prefixes)
            if(EXISTS "${_brew_prefix}/include/omp.h")
                set(OpenMP_ROOT "${_brew_prefix}" CACHE PATH "Homebrew libomp prefix" FORCE)
                message(STATUS "OpenMP: using Homebrew libomp at ${_brew_prefix}")
                break()
            endif()
        endforeach()
    endif()
    # HPGL_USE_OPENMP=ON is a HARD requirement: silently degrading to a serial
    # build produced numerically-divergent artifacts from the same commit on
    # different machines (2-M-21). Fail loudly at configure instead of
    # shipping a serial "high performance" library.
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found - version ${OpenMP_CXX_VERSION}")

    # macOS: prefer statically linking libomp.a so the built dylibs carry no
    # absolute /opt/homebrew/opt/libomp/lib/libomp.dylib load command (H-3).
    # Falls back to OpenMP::OpenMP_CXX (dynamic) when the archive is absent.
    set(_hpgl_omp_link OpenMP::OpenMP_CXX)
    set(_hpgl_omp_flags "")
    set(_hpgl_omp_include "")
    # R-21: records whether OpenMP is STATICALLY embedded in the built dylibs
    # (macOS wheel with libomp.a). The exported package config uses this to
    # skip find_dependency(OpenMP) — a static-libomp consumer needs no OpenMP
    # at configure time or runtime. Always set (FALSE default) so the config
    # template substitution is never empty.
    set(HPGL_OPENMP_STATIC FALSE)
    if(HPGL_STATIC_LIBOMP AND APPLE AND OpenMP_ROOT AND EXISTS "${OpenMP_ROOT}/lib/libomp.a")
        set(_hpgl_omp_flags ${OpenMP_CXX_FLAGS})
        # OpenMP_CXX_FLAGS is ONE string ("-Xclang -fopenmp"); splitting it is
        # required — passing it raw makes the compiler see a single quoted
        # argument "-Xclang -fopenmp", which clang++ rejects as "unknown
        # argument" (the exact failure the original OpenMP::OpenMP_CXX comment
        # documents). The include dir must be propagated manually too — the
        # OpenMP::OpenMP_CXX interface normally provides omp.h (source files
        # do #include <omp.h>).
        separate_arguments(_hpgl_omp_flags UNIX_COMMAND "${_hpgl_omp_flags}")
        set(_hpgl_omp_include "${OpenMP_ROOT}/include")
        set(_hpgl_omp_link "${OpenMP_ROOT}/lib/libomp.a")
        set(HPGL_OPENMP_STATIC TRUE)
        message(STATUS "OpenMP: statically linking libomp.a (relocatable artifacts)")

        # R-23: deployment-target guarantee. The static libomp.a objects carry
        # their own LC_BUILD_VERSION min-OS; if the archive was built for a
        # newer macOS than CMAKE_OSX_DEPLOYMENT_TARGET, the wheel's
        # macosx_11_0 tag is structurally unsound (embedded objects claim a
        # newer min-OS than the dylib). Warn at configure time (the escalation
        # probe found NO macOS 15+ API usage in Homebrew's libomp — 0
        # weak_import — so the wheel still runs on macOS 11+, but the
        # guarantee is fragile). The wheel verification gate additionally
        # asserts LC_BUILD_VERSION on the shipped dylibs (build.sh).
        execute_process(
            COMMAND otool -l "${OpenMP_ROOT}/lib/libomp.a"
            OUTPUT_VARIABLE _hpgl_libomp_otool
            ERROR_QUIET
        )
        string(REPLACE "\n" ";" _hpgl_libomp_otool_lines "${_hpgl_libomp_otool}")
        set(_hpgl_libomp_max "")
        foreach(_line IN LISTS _hpgl_libomp_otool_lines)
            string(STRIP "${_line}" _line_stripped)
            if(_line_stripped MATCHES "^minos[ \t]+([0-9]+\\.[0-9]+)")
                if(NOT _hpgl_libomp_max OR CMAKE_MATCH_1 VERSION_GREATER "${_hpgl_libomp_max}")
                    set(_hpgl_libomp_max "${CMAKE_MATCH_1}")
                endif()
            endif()
        endforeach()
        if(_hpgl_libomp_max AND CMAKE_OSX_DEPLOYMENT_TARGET
           AND _hpgl_libomp_max VERSION_GREATER CMAKE_OSX_DEPLOYMENT_TARGET)
            message(WARNING "Static libomp at ${OpenMP_ROOT}/lib/libomp.a was built for macOS ${_hpgl_libomp_max} "
                            "which is newer than the deployment target ${CMAKE_OSX_DEPLOYMENT_TARGET}. The produced "
                            "artifact embeds objects claiming min-OS ${_hpgl_libomp_max} inside a "
                            "macosx_${CMAKE_OSX_DEPLOYMENT_TARGET} wheel. Escalation probe found no macOS 15+ API "
                            "usage (0 weak_import), so the artifact still runs on the deployment target — but this "
                            "is fragile. Rebuild libomp with -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET} "
                            "for a structurally sound guarantee.")
        endif()
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
        # R-10: MKL is not OpenBLAS — the HPGL_USE_OPENBLAS define stays off.
        set(_hpgl_blas_is_openblas FALSE)
    else()
        message(FATAL_ERROR "HPGL_USE_MKL is ON but Intel MKL not found")
    endif()
else()
    if(BLA_VENDOR)
        set(BLA_VENDOR "${BLA_VENDOR}" CACHE STRING "" FORCE)
    endif()
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
        message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
        # Post-find vendor consistency: extract vendor names from found libraries
        # and warn if BLAS and LAPACK appear to come from different providers.
        set(_blas_vendor "")
        set(_lapack_vendor "")
        if(BLAS_LIBRARIES)
            list(GET BLAS_LIBRARIES 0 _blas_first)
        endif()
        if(LAPACK_LIBRARIES)
            list(GET LAPACK_LIBRARIES 0 _lapack_first)
        endif()
        if(_blas_first AND EXISTS "${_blas_first}")
            get_filename_component(_blas_name "${_blas_first}" NAME_WE)
            string(REGEX REPLACE "^lib" "" _blas_name "${_blas_name}")
            set(_blas_vendor "${_blas_name}")
        elseif(BLA_VENDOR)
            set(_blas_vendor "${BLA_VENDOR}")
        endif()
        if(_lapack_first AND EXISTS "${_lapack_first}")
            get_filename_component(_lapack_name "${_lapack_first}" NAME_WE)
            string(REGEX REPLACE "^lib" "" _lapack_name "${_lapack_name}")
            set(_lapack_vendor "${_lapack_name}")
        elseif(BLA_VENDOR)
            set(_lapack_vendor "${BLA_VENDOR}")
        endif()
        if(_blas_vendor AND _lapack_vendor AND NOT _blas_vendor MATCHES "${_lapack_vendor}" AND NOT _lapack_vendor MATCHES "${_blas_vendor}")
            message(WARNING "BLAS and LAPACK appear to come from different vendors: "
                            "BLAS=${_blas_vendor}, LAPACK=${_lapack_vendor}. "
                            "Mixed-vendor BLAS/LAPACK can cause ABI mismatches. "
                            "Set -DBLA_VENDOR=<vendor> to pin a specific provider.")
        endif()
        # R-10: HPGL_USE_OPENBLAS gates the OpenBLAS thread-control fast path
        # in indicator_kriging.h / cont_kriging.h / median_ik.cpp (M-32). It
        # must be defined ONLY when the build actually links OpenBLAS — the
        # openblas_* symbols it references are absent from other vendors, and
        # an unconditional define caused an undefined-symbol dlopen failure
        # with netlib/BLIS/FlexiBLAS/ATLAS. Detect via the library-name
        # vendor extraction above OR the explicit BLA_VENDOR cache var.
        if(_blas_vendor MATCHES "openblas" OR BLA_VENDOR MATCHES "OpenBLAS")
            set(_hpgl_blas_is_openblas TRUE)
        else()
            set(_hpgl_blas_is_openblas FALSE)
        endif()
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

# III-27: automated GLOB <-> vcxproj sync check. The MSVC project
# (src/msvc/hpgl.vcxproj) lists every source explicitly, so a new .cpp added
# to src/geo_bsd/hpgl/ is picked up by this GLOB on CMake builds but silently
# missing on MSVC builds (undefined symbols / missing functionality on
# Windows) — the header comment above says "no automated sync", which made
# the drift invisible until a Windows user hit it. Fail configure with a
# precise diff instead. stdafx.cpp is the precompiled-header TU, listed in
# the vcxproj with PrecompiledHeader=Create and filtered from non-MSVC GLOB
# results — excluded from the comparison on both sides.
set(_hpgl_vcxproj_path "${CMAKE_CURRENT_SOURCE_DIR}/src/msvc/hpgl.vcxproj")
if(EXISTS "${_hpgl_vcxproj_path}")
    file(STRINGS "${_hpgl_vcxproj_path}" _hpgl_vcx_lines REGEX "ClCompile Include=")
    set(_hpgl_vcx_sources "")
    foreach(_line IN LISTS _hpgl_vcx_lines)
        if(_line MATCHES "ClCompile Include=\"([^\"]+)\"")
            set(_hpgl_vcx_rel "${CMAKE_MATCH_1}")
            string(REPLACE "\\" "/" _hpgl_vcx_rel "${_hpgl_vcx_rel}")      # ..\geo_bsd\hpgl\x.cpp -> ../geo_bsd/hpgl/x.cpp
            string(REPLACE "../" "" _hpgl_vcx_rel "${_hpgl_vcx_rel}")      # geo_bsd/hpgl/x.cpp
            set(_hpgl_vcx_rel "src/${_hpgl_vcx_rel}")                      # src/geo_bsd/hpgl/x.cpp
            list(APPEND _hpgl_vcx_sources "${_hpgl_vcx_rel}")
        endif()
    endforeach()
    list(REMOVE_ITEM _hpgl_vcx_sources "src/geo_bsd/hpgl/stdafx.cpp")
    set(_hpgl_glob_relative "")
    foreach(_src IN LISTS HPGL_CORE_SOURCES)
        file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_src}")
        list(APPEND _hpgl_glob_relative "${_rel}")
    endforeach()
    list(REMOVE_ITEM _hpgl_glob_relative "src/geo_bsd/hpgl/stdafx.cpp")
    list(SORT _hpgl_vcx_sources)
    list(SORT _hpgl_glob_relative)
    if(NOT _hpgl_vcx_sources STREQUAL _hpgl_glob_relative)
        set(_hpgl_missing "")
        set(_hpgl_extra "")
        foreach(_vcx IN LISTS _hpgl_vcx_sources)
            list(FIND _hpgl_glob_relative "${_vcx}" _idx)
            if(_idx EQUAL -1)
                list(APPEND _hpgl_extra "${_vcx}")
            endif()
        endforeach()
        foreach(_glob IN LISTS _hpgl_glob_relative)
            list(FIND _hpgl_vcx_sources "${_glob}" _idx)
            if(_idx EQUAL -1)
                list(APPEND _hpgl_missing "${_glob}")
            endif()
        endforeach()
        message(FATAL_ERROR
            "GLOB <-> vcxproj sync check FAILED (src/msvc/hpgl.vcxproj):\n"
            "  In vcxproj but not in the source GLOB: ${_hpgl_extra}\n"
            "  In the source GLOB but not in vcxproj: ${_hpgl_missing}\n"
            "  Add/remove the corresponding <ClCompile Include=...> entry in src/msvc/hpgl.vcxproj.")
    endif()
    message(STATUS "vcxproj sync check: OK (${_hpgl_vcx_sources} matches source GLOB)")
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
# Version the shared library for standalone installs (C++ consumers get the
# SOVERSION symlink chain). Wheel builds (SKBUILD) skip VERSION/SOVERSION so
# the install emits a single plain libhpgl.dylib — the wheel assembler
# dereferences the versioned symlink chain into duplicate physical copies
# (H-3d / 2-M-25), and setting the properties to "" would emit the malformed
# name "libhpgl..dylib".
if(NOT SKBUILD)
    set_target_properties(hpgl_core PROPERTIES
        VERSION "${PROJECT_VERSION}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
    )
endif()

# II-25: Windows CMake builds. Two defects fixed:
#  1. No DEBUG_POSTFIX — Debug and Release both emitted hpgl.dll, so a Debug
#     build silently overwrote the Release DLL (name collision) and neither
#     was distinguishable; the vcxproj convention is hpgl_d.dll for Debug.
#  2. No deploy step — CMake's install() places DLLs under the install
#     prefix, which Python's _safe_load_library() (searches ref_path.parent)
#     never looks at. build.bat copies DLLs to src/geo_bsd; the CMake path
#     must do the same so CMake-built DLLs are findable at runtime.
#     PREFIX "" matches the MSVC default (no lib prefix) and the vcxproj
#     output names (hpgl.dll / hpgl_d.dll).
if(WIN32)
    set_target_properties(hpgl_core PROPERTIES
        PREFIX ""
        DEBUG_POSTFIX "_d"
    )
    add_custom_command(TARGET hpgl_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:hpgl_core>
            "${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/"
        COMMENT "Deploying hpgl DLL to src/geo_bsd (Python runtime location)"
    )
endif()

# Compiler warning flags applied at target scope
target_compile_options(hpgl_core PRIVATE ${HPGL_COMPILE_WARNINGS})

target_include_directories(hpgl_core
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/hpgl>
    PUBLIC
        $<INSTALL_INTERFACE:include>
)
# TNT is a third-party numeric library — use SYSTEM to suppress warnings
target_include_directories(hpgl_core SYSTEM
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/tnt_126>
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

# R-10: OpenBLAS thread-control fast path (M-32) — PRIVATE, because the define
# gates code compiled INTO hpgl_core (indicator_kriging.h / cont_kriging.h /
# median_ik.cpp); consumers compile the generic path through the same headers.
# Defined ONLY when the detected vendor is OpenBLAS (see the BLAS section).
if(_hpgl_blas_is_openblas)
    target_compile_definitions(hpgl_core PRIVATE HPGL_USE_OPENBLAS)
endif()

# RPATH configuration: embed BLAS/LAPACK library paths so that the built
# hpgl shared library can locate them at runtime without LD_LIBRARY_PATH.
# Only extracts directory paths from concrete library file names (skips
# -l flags and CMake interface targets).
#
# Framework bundles and system/SDK directories are EXCLUDED:
#  - macOS BLAS resolves to Accelerate.framework, which is referenced by its
#    absolute framework install name — no RPATH is needed. Baking the build
#    machine's Xcode SDK frameworks dir into LC_RPATH leaked a machine-specific
#    path into shipped artifacts (N4 fold-in into H-3).
#  - System search dirs (/usr/lib, /usr/local/lib, /System, /Library) are
#    already resolved by the OS default search paths; baking them into
#    INSTALL_RPATH made standalone installs non-relocatable (2-M-23).
set(_hpgl_rpath_dirs "")
function(_hpgl_collect_rpath_dirs lib_list)
    foreach(_lib ${lib_list})
        if("${_lib}" MATCHES "^-l" OR "${_lib}" MATCHES "^(optimized|debug)$")
            continue()
        endif()
        get_filename_component(_lib_dir "${_lib}" DIRECTORY)
        if(_lib_dir AND NOT _lib_dir STREQUAL "")
            if(_lib_dir MATCHES "\\.framework$" OR _lib_dir MATCHES "(/SDKs/|MacOSX\\.sdk|^/System/|^/Library/|^/usr/lib|^/usr/local/lib|/Applications/Xcode)")
                continue()
            endif()
            list(APPEND _hpgl_rpath_dirs "${_lib_dir}")
        endif()
    endforeach()
    if(_hpgl_rpath_dirs)
        list(REMOVE_DUPLICATES _hpgl_rpath_dirs)
        set(_hpgl_rpath_dirs "${_hpgl_rpath_dirs}" PARENT_SCOPE)
    endif()
endfunction()
_hpgl_collect_rpath_dirs("${BLAS_LIBRARIES}")
_hpgl_collect_rpath_dirs("${LAPACK_LIBRARIES}")
if(_hpgl_rpath_dirs)
    set_target_properties(hpgl_core PROPERTIES
        BUILD_RPATH "${_hpgl_rpath_dirs}"
        INSTALL_RPATH "${_hpgl_rpath_dirs}"
    )
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
if(HPGL_USE_OPENMP AND OpenMP_FOUND)
    if(_hpgl_omp_flags)
        # Static libomp (macOS, H-3): propagate the -fopenmp compile flags
        # (e.g. "-Xclang -fopenmp" on AppleClang) and the omp.h include dir,
        # and link the archive directly so no libomp.dylib runtime dependency
        # is recorded — the wheel is relocatable without bundling or
        # install_name_tool repair.
        target_compile_options(hpgl_core PUBLIC ${_hpgl_omp_flags})
        # R-22: wrap the Homebrew omp.h include dir in $<BUILD_INTERFACE:...>
        # ONLY — at install/export time it contributes nothing, so the
        # absolute build-machine path (/opt/homebrew/opt/libomp/include) no
        # longer leaks into the shipped hpglTargets.cmake. The existing
        # $<INSTALL_INTERFACE:include> (above) covers the packaged headers.
        target_include_directories(hpgl_core PUBLIC $<BUILD_INTERFACE:${_hpgl_omp_include}>)
        target_link_libraries(hpgl_core PRIVATE ${_hpgl_omp_link})
    else()
        # OpenMP::OpenMP_CXX propagates the compiler flags (e.g. -Xclang -fopenmp
        # on AppleClang) via its interface; adding ${OpenMP_CXX_FLAGS} manually
        # passed the flag string as ONE quoted argument ("-Xclang -fopenmp"),
        # which clang++ rejects as "unknown argument". The imported target is
        # sufficient and correct.
        target_link_libraries(hpgl_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
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
            # II-25: Debug builds emit _cvariogram_d.dll (matching the vcxproj
            # convention) so Debug/Release do not name-collide, and deploy the
            # built DLL to the Python runtime location.
            DEBUG_POSTFIX "_d"
        )
        add_custom_command(TARGET hpgl_variogram POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:hpgl_variogram>
                "${CMAKE_CURRENT_SOURCE_DIR}/src/geo_bsd/"
            COMMENT "Deploying _cvariogram DLL to src/geo_bsd (Python runtime location)"
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

    # OpenMP support for cvariogram module
    # The variogram computation contains an O(n²) double loop in
    # calc_variograms_from_point_set that benefits from parallelization.
    # OpenMP::OpenMP_CXX propagates the compile flags; see hpgl_core block.
    if(HPGL_USE_OPENMP AND OpenMP_FOUND)
        if(_hpgl_omp_flags)
            target_compile_options(hpgl_variogram PUBLIC ${_hpgl_omp_flags})
            # R-22: same BUILD_INTERFACE-only wrap as hpgl_core — keeps the
            # Homebrew include dir out of any exported target (hpgl_variogram
            # is not exported today, but the defect class is the same).
            target_include_directories(hpgl_variogram PUBLIC $<BUILD_INTERFACE:${_hpgl_omp_include}>)
            target_link_libraries(hpgl_variogram PRIVATE ${_hpgl_omp_link})
        else()
            target_link_libraries(hpgl_variogram PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()

    # Windows symbol export (M-17): cvar_clear_last_error is defined as plain
    # extern "C" without the CVAR_API macro in variograms.cpp, so on MSVC it
    # is not dllexport'ed by default and Python's hasattr() check silently
    # disables stale-error clearing. WINDOWS_EXPORT_ALL_SYMBOLS generates a
    # module-definition file at link time covering it (and the other cvar_*
    # exports) on fresh MSVC builds. Ignored on non-Windows platforms.
    set_target_properties(hpgl_variogram PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

# Python package destination for wrapper scripts AND native libraries.
# Wheel builds (scikit-build-core, detected via SKBUILD) install into the
# importable top-level package directory (${SKBUILD_PLATLIB_DIR} = wheel
# root = site-packages when installed), named after the source package
# (geo_bsd). Standalone installs keep the historical ${CMAKE_INSTALL_LIBDIR}
# layout. Native libraries MUST be co-located with the wrapper scripts so
# that _safe_load_library() (which searches ref_path.parent) can find them.
if(SKBUILD)
    # Wheel builds: scikit-build-core maps a plain RELATIVE destination to the
    # platlib tree (CMAKE_INSTALL_PREFIX == SKBUILD_PLATLIB_DIR), so the
    # installed location is identical to the old absolute
    # ${SKBUILD_PLATLIB_DIR}/geo_bsd — but install(EXPORT) then records
    # ${_IMPORT_PREFIX}/geo_bsd/... instead of the staging temp dir. The
    # shipped wheel's hpglTargets-release.cmake referenced a deleted
    # build-machine temp dir, making find_package(hpgl) against the installed
    # wheel a guaranteed FATAL_ERROR and leaking the build-machine path
    # (2-M-18). A relative destination fixes both.
    set(HPGL_PYTHON_DEST "geo_bsd")
    # Wheel builds additionally skip VERSION/SOVERSION (set conditionally at
    # target creation above) so the wheel ships a single plain libhpgl.dylib
    # — the wheel assembler dereferences the versioned symlink chain into
    # duplicate physical copies, breaking the "exactly one dylib per library"
    # verification gate (H-3d / 2-M-25).
else()
    set(HPGL_PYTHON_DEST ${CMAKE_INSTALL_LIBDIR}/hpgl)
endif()

# Install core library
# Native library must be co-located with Python wrapper scripts so that
# _safe_load_library() (which searches ref_path.parent) can find it.
install(TARGETS hpgl_core
    EXPORT hpglTargets
    ARCHIVE DESTINATION ${HPGL_PYTHON_DEST}
    LIBRARY DESTINATION ${HPGL_PYTHON_DEST}
    RUNTIME DESTINATION ${HPGL_PYTHON_DEST}
)

# Install headers
install(DIRECTORY
    src/geo_bsd/hpgl/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpgl
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install Python wrapper scripts
# NOTE: every module imported unconditionally by __init__.py / geo.py / sgs.py /
# sis.py / gtsim.py MUST be listed here (config.py and ffi_adapter.py were
# previously omitted — F-10: import hpgl failed with ModuleNotFoundError).
if(HPGL_BUILD_PYTHON)
    install(FILES
        src/geo_bsd/__init__.py
        src/geo_bsd/cdf.py
        src/geo_bsd/config.py
        src/geo_bsd/cvariogram.py
        src/geo_bsd/ffi_adapter.py
        src/geo_bsd/geo.py
        src/geo_bsd/gtsim.py
        src/geo_bsd/hpgl_wrap.py
        src/geo_bsd/routines.py
        src/geo_bsd/sgs.py
        src/geo_bsd/sis.py
        src/geo_bsd/validation.py
        src/geo_bsd/variogram.py
        DESTINATION ${HPGL_PYTHON_DEST}
    )
endif()

# Install variogram module
if(HPGL_BUILD_VARIOGRAM)
    install(TARGETS hpgl_variogram
        RUNTIME DESTINATION ${HPGL_PYTHON_DEST}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${HPGL_PYTHON_DEST}
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

# Normalize OpenMP status to a CMake boolean for hpglConfig.cmake.in.
# find_package(OpenMP) sets OpenMP_FOUND to TRUE (boolean), but the
# package config template gets a raw 0/1 integer substitution.
# Convert to explicit TRUE/FALSE for idiomatic CMake conditionals.
if(OpenMP_FOUND)
    set(HPGL_OPENMP_BOOL TRUE)
else()
    set(HPGL_OPENMP_BOOL FALSE)
endif()

# R-21: same normalization for the static-embed flag (set in the OpenMP
# section). TRUE when OpenMP is statically embedded (macOS libomp.a wheel) —
# the package config then skips find_dependency(OpenMP). Always defined
# (FALSE when HPGL_USE_OPENMP=OFF or dynamic OpenMP is linked), so the
# template substitution is never empty.
if(HPGL_OPENMP_STATIC)
    set(HPGL_OPENMP_STATIC_BOOL TRUE)
else()
    set(HPGL_OPENMP_STATIC_BOOL FALSE)
endif()

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
    # Guard against sdists built without the test suite (2-M-17): every
    # preset and build.sh sets HPGL_BUILD_TESTS=ON, so an unconditional
    # add_subdirectory(tests) hard-failed configure for sdist consumers whose
    # tarball omitted tests/. Skipping with a status message keeps preset
    # configures working even if tests/ is absent.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        enable_testing()
        add_subdirectory(tests)
    else()
        message(STATUS "HPGL_BUILD_TESTS is ON but tests/ is absent (e.g. sdist built without the test suite) — skipping the tests subdirectory")
    endif()
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
message(STATUS "  OpenMP:              ${OpenMP_FOUND}")
if(HPGL_BUILD_PYTHON)
    message(STATUS "  Python:              ${Python3_EXECUTABLE}")
endif()
message(STATUS "====================================================================")
message(STATUS "")
