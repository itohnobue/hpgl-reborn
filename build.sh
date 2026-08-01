#!/usr/bin/env bash
# HPGL Build Script (Linux/macOS)
# Rebuilds the HPGL project using CMake.
# Default: Release, Ninja (auto-detected) or Make fallback.
# Override with --config Debug|Release --preset NAME
#
# Usage:
#   ./build.sh                          # Release build with auto-detected generator
#   ./build.sh --config Debug           # Debug build
#   ./build.sh --preset macos-clang     # Use a CMake preset from CMakePresets.json

set -euo pipefail

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
BUILD_CONFIG="Release"
USE_PRESET=""

usage() {
    echo "Usage: $0 [--config Debug|Release] [--preset NAME] [--help]"
    echo ""
    echo "Options:"
    echo "  --config CONFIG    Build configuration: Debug or Release (default: Release)"
    echo "  --preset NAME      Use a CMake configure preset from CMakePresets.json"
    echo "                     (overrides --config and generator auto-detection)"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Release, auto-detected generator"
    echo "  $0 --config Debug               # Debug build"
    echo "  $0 --preset macos-clang         # Use macOS Clang preset"
    echo "  $0 --preset linux-ninja"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            BUILD_CONFIG="$2"
            shift 2
            ;;
        --preset)
            USE_PRESET="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate arguments
# ---------------------------------------------------------------------------
if [[ "$BUILD_CONFIG" != "Debug" && "$BUILD_CONFIG" != "Release" ]]; then
    echo "ERROR: --config must be Debug or Release, got '$BUILD_CONFIG'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
UNAME_S=$(uname -s)
case "$UNAME_S" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="macos" ;;
    *)
        echo "ERROR: Unsupported platform: $UNAME_S"
        exit 1
        ;;
esac

# Detect parallel job count
if command -v nproc &>/dev/null; then
    NPROC=$(nproc)
elif [[ "$UNAME_S" == "Darwin" ]]; then
    NPROC=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
else
    NPROC=4
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Print build header
# ---------------------------------------------------------------------------
echo "========================================"
echo "HPGL Build Script ($PLATFORM)"
echo "========================================"
echo ""
echo "Configuration: $BUILD_CONFIG"
if [[ -n "$USE_PRESET" ]]; then
    echo "CMake Preset:  $USE_PRESET"
fi
echo "Parallel Jobs: $NPROC"
echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [[ -n "$USE_PRESET" ]]; then
    # --- Preset-based build ---
    # NOTE: Do NOT pass -DCMAKE_BUILD_TYPE here — the preset's cacheVariables
    # control the build type. Passing a CLI -D override would silently replace
    # the preset's value and cause e.g. './build.sh --preset debug' to build
    # as Release (the BUILD_CONFIG default).
    echo "Configuring via preset '$USE_PRESET'..."
    cmake --preset "$USE_PRESET"

    echo "Building via preset '$USE_PRESET'..."
    cmake --build --preset "$USE_PRESET" --parallel "$NPROC"
else
    # --- Standard CMake build ---
    BUILD_DIR="${SCRIPT_DIR}/build/$(echo "$BUILD_CONFIG" | tr '[:upper:]' '[:lower:]')"

    mkdir -p "$BUILD_DIR"

    CMAKE_ARGS=(
        -S "$SCRIPT_DIR"
        -B "$BUILD_DIR"
        -DCMAKE_BUILD_TYPE="$BUILD_CONFIG"
        -DHPGL_BUILD_PYTHON=ON
        -DHPGL_BUILD_TESTS=ON
        -DHPGL_USE_OPENMP=ON
        -DHPGL_USE_MKL=OFF
    )

    # Auto-detect Ninja for faster builds
    if command -v ninja &>/dev/null; then
        CMAKE_ARGS+=(-G Ninja)
        echo "Generator: Ninja (auto-detected)"
    else
        echo "Generator: default (Make)"
    fi

    echo "Configuring..."
    cmake "${CMAKE_ARGS[@]}"

    echo ""
    echo "Building..."
    cmake --build "$BUILD_DIR" --parallel "$NPROC" --config "$BUILD_CONFIG"
fi

# ---------------------------------------------------------------------------
# Copy built libraries to runtime location (src/geo_bsd/)
# Python's _safe_load_library() searches ref_path.parent for the shared library.
# ---------------------------------------------------------------------------
RUNTIME_DIR="${SCRIPT_DIR}/src/geo_bsd"

# Detect shared library extension
if [[ "$PLATFORM" == "macos" ]]; then
    SHARED_EXT=".dylib"
else
    SHARED_EXT=".so"
fi

echo ""
echo "Copying libraries to runtime location..."

if [[ -n "$USE_PRESET" ]]; then
    # Preset builds: search under build/<preset-name>
    PRESET_BUILD_DIR="${SCRIPT_DIR}/build/${USE_PRESET}"
    if [[ -d "$PRESET_BUILD_DIR" ]]; then
        SEARCH_DIR="$PRESET_BUILD_DIR"
    else
        SEARCH_DIR="${SCRIPT_DIR}/build"
    fi
else
    SEARCH_DIR="$BUILD_DIR"
fi

# Copy main hpgl library
HPGL_LIB=$(find "$SEARCH_DIR" -maxdepth 3 \( -name "libhpgl${SHARED_EXT}" -o -name "hpgl${SHARED_EXT}" \) 2>/dev/null | head -1)
if [[ -n "$HPGL_LIB" ]]; then
    # Remove stale library variants that shadow the fresh build (F-03/I2-43):
    # the Python loader searches lib-{name} BEFORE {name}, so a leftover
    # libhpgl.dylib (or a stale cross-platform hpgl.so/hpgl.dylib) silently
    # wins at runtime over the freshly copied binary. Delete all variants,
    # then copy the fresh build so the search order cannot resolve to a stale
    # file.
    find "$RUNTIME_DIR" -maxdepth 1 \
        \( -name "libhpgl.*" -o -name "hpgl.so" -o -name "hpgl.dylib" \) \
        -delete 2>/dev/null || true
    cp "$HPGL_LIB" "${RUNTIME_DIR}/hpgl${SHARED_EXT}"
    echo "  $(basename "$HPGL_LIB") -> ${RUNTIME_DIR}/hpgl${SHARED_EXT}"
else
    echo "  WARNING: hpgl library not found in $SEARCH_DIR"
fi

# Copy variogram library
VARIO_LIB=$(find "$SEARCH_DIR" -maxdepth 3 -name "_cvariogram${SHARED_EXT}" 2>/dev/null | head -1)
if [[ -n "$VARIO_LIB" ]]; then
    # Remove stale lib_cvariogram.* and cross-platform leftovers (same
    # shadowing defect as the main library — repeat regression I2-43).
    find "$RUNTIME_DIR" -maxdepth 1 \
        \( -name "lib_cvariogram.*" -o -name "_cvariogram.so" -o -name "_cvariogram.dylib" \) \
        -delete 2>/dev/null || true
    cp "$VARIO_LIB" "${RUNTIME_DIR}/_cvariogram${SHARED_EXT}"
    echo "  $(basename "$VARIO_LIB") -> ${RUNTIME_DIR}/_cvariogram${SHARED_EXT}"
else
    echo "  WARNING: _cvariogram library not found in $SEARCH_DIR"
fi

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""

# Post-build smoke test: verify the FRESH native library loads via Python.
# _HAS_KRIGING_STATS is only True for builds exporting hpgl_get_kriging_stats —
# the stale Jun-27 binary predates that API and reports False (I2-47), so this
# assert catches a stale-lib shadow that a plain "does it load" check misses.
# The resolved library path is printed so a shadowing regression is visible.
# Use uv if available (project standard), fall back to system python.
echo "Smoke test: verifying library load..."
SMOKE_CMD="import sys; sys.path.insert(0, '${RUNTIME_DIR}/..'); from geo_bsd import hpgl_wrap; assert hpgl_wrap._HAS_KRIGING_STATS, 'stale library loaded (_HAS_KRIGING_STATS=False)'; print('  hpgl shared library loaded:', hpgl_wrap._hpgl_so._name)"
if command -v uv &>/dev/null; then
    if uv run python -c "$SMOKE_CMD" 2>/dev/null; then
        echo "  Smoke test: PASSED"
    else
        echo "  WARNING: Smoke test failed — library may not load at runtime."
        echo "  Check that all dependencies are installed and LD_LIBRARY_PATH is set."
    fi
else
    if python3 -c "$SMOKE_CMD" 2>/dev/null || python -c "$SMOKE_CMD" 2>/dev/null; then
        echo "  Smoke test: PASSED"
    else
        echo "  WARNING: Smoke test failed — library may not load at runtime."
        echo "  Check that all dependencies are installed and LD_LIBRARY_PATH is set."
    fi
fi
