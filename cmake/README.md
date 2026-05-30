# HPGL CMake Build Notes
# This directory was intended for custom CMake helper modules.
# However, HPGL Reborn uses CMake's built-in find modules:
#   - find_package(BLAS)    — built into CMake
#   - find_package(LAPACK)  — built into CMake
#   - find_package(Python3) — built into CMake
#   - find_package(MKL)     — Intel-provided, detected via CONFIG mode
#   - find_package(OpenMP)  — built into CMake
#
# No custom CMake modules are required for building HPGL.
# See the main CMakeLists.txt at the project root for full build configuration.
