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
USE_WHEEL=""

usage() {
    echo "Usage: $0 [--config Debug|Release] [--preset NAME] [--wheel] [--help]"
    echo ""
    echo "Options:"
    echo "  --config CONFIG    Build configuration: Debug or Release (default: Release)"
    echo "  --preset NAME      Use a CMake configure preset from CMakePresets.json"
    echo "                     (overrides --config and generator auto-detection)"
    echo "  --wheel            Build a self-contained wheel: python -m build + macOS"
    echo "                     delocate repair + relocatability verification gate +"
    echo "                     fresh-venv load smoke test (H-3). Requires the venv"
    echo "                     (or HPGL_PYTHON) to have scikit-build-core + build."
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Release, auto-detected generator"
    echo "  $0 --config Debug               # Debug build"
    echo "  $0 --preset macos-clang         # Use macOS Clang preset"
    echo "  $0 --preset linux-ninja"
    echo "  $0 --wheel                      # Self-contained wheel build"
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
        --wheel)
            USE_WHEEL="1"
            shift
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
# Wheel build mode (--wheel): self-contained wheel with repair + verification
# gate. Uses scikit-build-core via `python -m build`, which performs its own
# CMake configure/build in build/{wheel_tag}. Produces a wheel that is
# relocatable on any macOS 11+ machine: macOS dylibs are repaired with
# delocate (bundling any non-system deps), the verification gate asserts no
# absolute build-machine paths and a single dylib per library, and a fresh
# venv load smoke test loads the library by the name the wheel ships (H-3).
# ---------------------------------------------------------------------------
if [[ -n "$USE_WHEEL" ]]; then
    echo "========================================"
    echo "HPGL Wheel Build ($PLATFORM)"
    echo "========================================"
    echo "Parallel Jobs: $NPROC"

    PYTHON_BIN="${HPGL_PYTHON:-}"
    if [[ -z "$PYTHON_BIN" ]]; then
        if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
            PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
        elif command -v python3 &>/dev/null; then
            PYTHON_BIN="python3"
        else
            echo "ERROR: no Python found for the wheel build (set HPGL_PYTHON)" >&2
            exit 1
        fi
    fi
    echo "Python:    $PYTHON_BIN"

    echo ""
    echo "Building wheel..."
    (cd "$SCRIPT_DIR" && "$PYTHON_BIN" -m build --wheel --outdir dist) || {
        echo "ERROR: wheel build failed" >&2
        exit 1
    }

    WHEEL_FILE=$(ls -t "$SCRIPT_DIR"/dist/hpgl-*.whl 2>/dev/null | head -1)
    if [[ -z "$WHEEL_FILE" ]]; then
        echo "ERROR: no wheel produced in dist/" >&2
        exit 1
    fi
    echo "  wheel: $(basename "$WHEEL_FILE")"

    # macOS repair (H-3): delocate bundles any non-system dylibs into the
    # wheel, rewrites install names to @loader_path/... and ad-hoc re-signs
    # (mandatory on arm64). Skipped when delocate is unavailable — the
    # verification gate below still FAILS the build if the wheel is not
    # self-contained, so a silent skip cannot ship a broken wheel.
    if [[ "$PLATFORM" == "macos" ]] && "$PYTHON_BIN" -c "import delocate" 2>/dev/null; then
        echo "Repairing wheel with delocate..."
        "$PYTHON_BIN" -m delocate.delocate_wheel --require-archs "${DELOCATE_ARCHS:-arm64}" -w dist "$WHEEL_FILE" || {
            echo "ERROR: delocate-wheel repair failed" >&2
            exit 1
        }
        WHEEL_FILE=$(ls -t "$SCRIPT_DIR"/dist/hpgl-*.whl 2>/dev/null | head -1)
    elif [[ "$PLATFORM" == "macos" ]]; then
        echo "  NOTE: delocate not available — relying on static libomp + verification gate"
    fi

    # Verification gate (H-3d): FAILS the build on violation. Covers:
    #  - no absolute build-machine paths in dylib load commands (H-3)
    #  - no LC_RPATH entries (R-24) — regression guard for the N4 fold-in
    #  - LC_BUILD_VERSION min-OS <= deployment target (R-23)
    #  - no absolute build-machine paths in shipped .cmake files (R-22)
    #  - exactly one dylib per library, macosx_11_0 wheel tag (H-3d)
    echo "Verifying wheel relocatability..."
    "$PYTHON_BIN" - "$WHEEL_FILE" <<'PYEOF'
import re, subprocess, sys, tempfile, zipfile
from pathlib import Path

wheel = Path(sys.argv[1])
out = Path(tempfile.mkdtemp(prefix="hpgl-verify-"))
with zipfile.ZipFile(wheel) as z:
    z.extractall(out)

dylibs = sorted(out.rglob("*.dylib"))
if not dylibs:
    print("ERROR: no .dylib found in wheel", file=sys.stderr)
    sys.exit(1)

# R-23: must match CMAKE_OSX_DEPLOYMENT_TARGET / the macosx_11_0 wheel tag.
DEPLOYMENT_TARGET = 11.0
BAD = re.compile(r"/(opt/homebrew|usr/local|Users/|Applications/Xcode)")
failures: list[str] = []
seen: dict[str, list[str]] = {}

def fail(msg: str) -> None:
    failures.append(msg)

for d in dylibs:
    rel = str(d.relative_to(out))
    m = re.match(r"lib?([A-Za-z0-9_]+?)(?:\.\d+)?\.dylib$", d.name)
    key = m.group(1) if m else d.name
    seen.setdefault(key, []).append(rel)
    otool = subprocess.run(["otool", "-L", str(d)], capture_output=True, text=True)
    if otool.returncode != 0:
        fail(f"{rel}: otool -L failed: {otool.stderr.strip()}")
        continue
    for line in otool.stdout.splitlines()[1:]:
        dep = line.strip().split()[0]
        if BAD.search(dep):
            fail(f"{rel}: absolute build-machine dependency: {dep}")
        elif dep.startswith("/") and not dep.startswith(("/System/", "/usr/lib")):
            fail(f"{rel}: non-system absolute dependency: {dep}")

    # R-24/R-23: parse otool -l load commands — LC_RPATH entries and
    # LC_BUILD_VERSION minos. The old gate read otool -L only, so an LC_RPATH
    # regression (N4 fold-in: SDK path baked into the rpath) passed silently.
    otool_l = subprocess.run(["otool", "-l", str(d)], capture_output=True, text=True)
    if otool_l.returncode != 0:
        fail(f"{rel}: otool -l failed: {otool_l.stderr.strip()}")
        continue
    in_rpath = False
    minos = None
    for line in otool_l.stdout.splitlines():
        ls = line.strip()
        if ls.startswith("cmd LC_RPATH"):
            in_rpath = True
        elif in_rpath and ls.startswith("path "):
            rp = ls.split(None, 1)[1]
            if BAD.search(rp) or rp.startswith("/"):
                fail(f"{rel}: LC_RPATH entry: {rp}")
            in_rpath = False
        elif in_rpath and ls.startswith("cmd "):
            in_rpath = False
        if ls.startswith("minos "):
            try:
                minos = float(ls.split()[1])
            except (IndexError, ValueError):
                pass
    if minos is not None and minos > DEPLOYMENT_TARGET:
        fail(f"{rel}: LC_BUILD_VERSION minos {minos} exceeds deployment target {DEPLOYMENT_TARGET}")

# R-22: scan every shipped .cmake file for absolute build-machine paths — the
# export defect class (INTERFACE_INCLUDE_DIRECTORIES leaking
# /opt/homebrew/opt/libomp/include) is invisible to otool scans.
for cf in sorted(out.rglob("*.cmake")):
    rel = str(cf.relative_to(out))
    try:
        text = cf.read_text()
    except UnicodeDecodeError:
        continue
    for line in text.splitlines():
        if BAD.search(line):
            fail(f"{rel}: absolute build-machine path in exported CMake package: {line.strip()}")

for key, rels in seen.items():
    if len(rels) > 1:
        fail(f"multiple dylibs for library '{key}': {rels}")

if "macosx_11_0" not in wheel.name:
    fail(f"wheel tag is not macosx_11_0: {wheel.name}")

if failures:
    print("WHEEL VERIFY FAILED:", file=sys.stderr)
    for f in failures:
        print("  -", f, file=sys.stderr)
    sys.exit(1)
print(f"  WHEEL VERIFY PASSED: {len(dylibs)} dylib(s), no absolute build-machine paths, no LC_RPATH, minos <= {DEPLOYMENT_TARGET}, tag macosx_11_0")
PYEOF
    if [[ $? -ne 0 ]]; then
        echo "ERROR: wheel verification failed" >&2
        exit 1
    fi

    # Fresh-venv load smoke test (H-3e): installs the wheel and loads the
    # library by the name the wheel actually ships (libhpgl.dylib — the wheel
    # has no bare hpgl.dylib). This validates the artifact end users install,
    # which the dev smoke test (source-tree hpgl.dylib) cannot.
    echo "Smoke testing wheel in a fresh venv..."
    SMOKE_VENV=$(mktemp -d "${TMPDIR:-/tmp}/hpgl-smoke-XXXXXX")
    "$PYTHON_BIN" -m venv "$SMOKE_VENV" >/dev/null 2>&1 || {
        echo "ERROR: could not create smoke-test venv" >&2
        exit 1
    }
    "$SMOKE_VENV/bin/python" -m pip install --quiet "$WHEEL_FILE" || {
        echo "ERROR: wheel install into smoke venv failed" >&2
        exit 1
    }
    "$SMOKE_VENV/bin/python" -c "from geo_bsd import hpgl_wrap; assert hpgl_wrap._HAS_KRIGING_STATS, 'stale library loaded'; assert 'libhpgl' in str(hpgl_wrap._hpgl_so._name), 'loaded unexpected library: %s' % hpgl_wrap._hpgl_so._name; from geo_bsd import cvariogram; assert hasattr(cvariogram.cvar, 'cvar_clear_last_error'), '_cvariogram missing cvar_clear_last_error'; print('  wheel smoke test PASSED:', hpgl_wrap._hpgl_so._name)" || {
        echo "ERROR: wheel smoke test failed" >&2
        exit 1
    }

    echo ""
    echo "========================================"
    echo "Wheel build + repair + verify + smoke: ALL PASSED"
    echo "  wheel: $(basename "$WHEEL_FILE")"
    echo "========================================"
    exit 0
fi

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
    # 2-M-22: a missing library must FAIL the build — a broken artifact
    # previously reported success (WARNING + exit 0).
    echo "  ERROR: hpgl library not found in $SEARCH_DIR" >&2
    exit 1
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
    # 2-M-22: same as above — the variogram module is required.
    echo "  ERROR: _cvariogram library not found in $SEARCH_DIR" >&2
    exit 1
fi

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""

# Post-build smoke test: verify the FRESH native libraries load via Python.
# _HAS_KRIGING_STATS is only True for builds exporting hpgl_get_kriging_stats —
# the stale Jun-27 binary predates that API and reports False (I2-47), so this
# assert catches a stale-lib shadow that a plain "does it load" check misses.
# The _cvariogram module is loaded and its error-clear symbol checked too
# (2-M-22: the old gate never loaded _cvariogram, so a broken variogram module
# passed). The resolved library path is printed so a shadowing regression is
# visible. Use uv if available (project standard), fall back to system python.
# 2-M-22: a failed smoke test FAILS the build (exit 1) instead of WARNING +
# exit 0 — the smoke gate is the only check before packaging.
echo "Smoke test: verifying library load..."
SMOKE_CMD="import sys; sys.path.insert(0, '${RUNTIME_DIR}/..'); from geo_bsd import hpgl_wrap; assert hpgl_wrap._HAS_KRIGING_STATS, 'stale library loaded (_HAS_KRIGING_STATS=False)'; print('  hpgl shared library loaded:', hpgl_wrap._hpgl_so._name); from geo_bsd import cvariogram; assert hasattr(cvariogram.cvar, 'cvar_get_last_error'), '_cvariogram failed to load'; assert hasattr(cvariogram.cvar, 'cvar_clear_last_error'), '_cvariogram missing cvar_clear_last_error (stale build)'; print('  _cvariogram shared library loaded: OK')"
if command -v uv &>/dev/null; then
    if uv run python -c "$SMOKE_CMD" 2>/dev/null; then
        echo "  Smoke test: PASSED"
    else
        echo "  ERROR: Smoke test failed — the freshly built library does not load." >&2
        echo "  Check that all dependencies are installed and LD_LIBRARY_PATH is set." >&2
        exit 1
    fi
else
    if python3 -c "$SMOKE_CMD" 2>/dev/null || python -c "$SMOKE_CMD" 2>/dev/null; then
        echo "  Smoke test: PASSED"
    else
        echo "  ERROR: Smoke test failed — the freshly built library does not load." >&2
        echo "  Check that all dependencies are installed and LD_LIBRARY_PATH is set." >&2
        exit 1
    fi
fi

# R-26: relocatability check for the NON-wheel build path. The smoke test
# above verifies loadability/symbols but NOT that the runtime dylibs are
# self-contained: a build whose cache has HPGL_STATIC_LIBOMP=OFF (or a stale
# cache from before the option defaulted to ON) produces dylibs that carry an
# absolute /opt/homebrew/opt/libomp dependency, and the old gate PASSED them
# anyway (H-3 defect class undetectable). When the build is configured for
# static libomp, assert the runtime dylibs carry no absolute build-machine
# dependencies.
echo "Smoke test: checking runtime dylib relocatability..."
HPGL_STATIC_CACHE=$(grep -E "^HPGL_STATIC_LIBOMP:BOOL=" "$SEARCH_DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2 || true)
if [[ "$PLATFORM" == "macos" && "${HPGL_STATIC_CACHE:-ON}" == "ON" ]]; then
    RELOC_BAD=0
    for lib in "${RUNTIME_DIR}/hpgl${SHARED_EXT}" "${RUNTIME_DIR}/_cvariogram${SHARED_EXT}"; do
        if [[ ! -f "$lib" ]]; then
            echo "  WARNING: $lib not found — skipping relocatability check for it" >&2
            continue
        fi
        # Absolute deps under /opt/homebrew|/usr/local|/Users/|/Applications/Xcode
        # indicate a non-static (or stale-cache) build — the H-3 defect class.
        ABS=$(otool -L "$lib" 2>/dev/null | awk 'NR>1 {print $1}' | grep -E '^/(opt/homebrew|usr/local|Users/|Applications/Xcode)' || true)
        if [[ -n "$ABS" ]]; then
            echo "  ERROR: $lib carries absolute build-machine dependencies (HPGL_STATIC_LIBOMP=ON expected static embed):" >&2
            echo "$ABS" | sed 's/^/    /' >&2
            RELOC_BAD=1
        fi
    done
    if [[ $RELOC_BAD -ne 0 ]]; then
        echo "  ERROR: runtime dylib relocatability check FAILED — the build is not self-contained." >&2
        echo "  This is usually a stale CMake cache with HPGL_STATIC_LIBOMP=OFF; delete the cache dir and rebuild." >&2
        exit 1
    fi
    echo "  Relocatability check: PASSED (no absolute build-machine deps)"
else
    echo "  Relocatability check: SKIPPED (platform=$PLATFORM, HPGL_STATIC_LIBOMP=${HPGL_STATIC_CACHE:-unset})"
fi
