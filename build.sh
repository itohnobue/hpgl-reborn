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
    # F-32: exit code is an explicit argument — 0 for --help, 1 for
    # error-path invocations (unknown argument). The old usage() always
    # exited 0, so './build.sh --bogus' reported success (fail-non-zero
    # violation).
    exit "${1:-0}"
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
            usage 1
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
# --wheel is macOS-only (III-26). The pipeline — delocate repair, otool-based
# verification gate (LC_RPATH / LC_BUILD_VERSION / dylib uniqueness), dylib
# install-name rewriting — is macOS-specific, and the gate searches only
# *.dylib + requires a macosx_* tag, so --wheel on Linux failed
# deterministically AFTER doing a full build. Reject early with a clear
# platform message instead; Linux wheels are built with cibuildwheel
# (see pyproject.toml [tool.cibuildwheel.linux]).
# ---------------------------------------------------------------------------
if [[ -n "$USE_WHEEL" && "$PLATFORM" == "linux" ]]; then
    echo "ERROR: --wheel is macOS-only. The wheel pipeline (delocate repair, otool" >&2
    echo "  verification gate, dylib bundling) is macOS-specific, and the gate only" >&2
    echo "  inspects *.dylib artifacts with a macosx_* tag." >&2
    echo "  Build Linux wheels with cibuildwheel instead:" >&2
    echo "    python -m cibuildwheel --platform linux   (see pyproject.toml [tool.cibuildwheel.linux])" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# III-24 (build-side): Python 3.9 grammar smoke. pyproject.toml declares
# requires-python>=3.9 and ships py3-none wheels, but the test suite runs on
# the host interpreter (3.13), so post-3.9 syntax at module scope (PEP 604
# unions, etc.) shipped a wheel that crashes on import under 3.9 — with no CI
# to catch it. ast.parse with feature_version=(3, 9) rejects such syntax at
# parse time without needing a 3.9 interpreter. Runs on BOTH build paths
# (dev + wheel). Usage: py39_grammar_check <python-command...>
# ---------------------------------------------------------------------------
py39_grammar_check() {
    "$@" - "$SCRIPT_DIR/src/geo_bsd" <<'PY39EOF'
import ast
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
bad = []


def has_future_annotations(tree) -> bool:
    """True when the module carries `from __future__ import annotations`."""
    return any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in tree.body
    )


def bitor_in_annotation(tree) -> bool:
    """True when any annotation (AnnAssign / arg / return) uses a `X | Y`
    union. Without the future import this is a RUNTIME crash on Python 3.9
    (PEP 604 unions are grammatically valid BinOp(BitOr) in the 3.9 grammar,
    so ast.parse(feature_version=(3,9)) does NOT catch them — the annotation
    is evaluated at import time and `dict | None` raises TypeError)."""
    for node in ast.walk(tree):
        ann = None
        if isinstance(node, ast.AnnAssign):
            ann = node.annotation
        elif isinstance(node, ast.arg) and node.annotation is not None:
            ann = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            ann = node.returns
        if ann is None:
            continue
        for sub in ast.walk(ann):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                return True
    return False


for p in sorted(root.rglob("*.py")):
    if "__pycache__" in p.parts:
        continue
    src = p.read_text(encoding="utf-8")
    # 1. Real 3.9 grammar violations (match statements, parenthesized
    #    context managers, walrus in comprehension iterables, ...).
    try:
        tree = ast.parse(src, filename=str(p), feature_version=(3, 9))
    except SyntaxError as err:
        bad.append(f"{p}: syntax not valid under the 3.9 grammar: {err}")
        continue
    # 2. PEP 604 `X | Y` unions in annotations of a module WITHOUT the
    #    future import — the III-24 defect class (import-time TypeError on
    #    Python 3.9). The future import postpones annotation evaluation and
    #    is the project's established pattern (config/ffi_adapter/geo/
    #    hpgl_wrap/validation all carry it).
    if not has_future_annotations(tree) and bitor_in_annotation(tree):
        bad.append(f"{p}: uses `X | Y` union annotations without "
                   "`from __future__ import annotations` — crashes at import "
                   "on Python 3.9 (requires-python>=3.9).")
if bad:
    print("PY39 GRAMMAR CHECK FAILED:", file=sys.stderr)
    for b in bad:
        print("  -", b, file=sys.stderr)
    sys.exit(1)
print("  Py39 grammar check: OK (src/geo_bsd parses under Python 3.9 grammar, no unprotected PEP 604 unions)")
PY39EOF
}

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

    # III-24: fail before the (slow) CMake wheel build if any module breaks
    # the declared Python 3.9 grammar.
    echo "Checking Python 3.9 grammar compatibility of src/geo_bsd..."
    if ! py39_grammar_check "$PYTHON_BIN"; then
        echo "ERROR: Python 3.9 grammar check failed (see above)" >&2
        exit 1
    fi

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
    #  - LC_BUILD_VERSION min-OS <= deployment target (R-23, target read from
    #    the wheel's own macosx_* tag — II-24)
    #  - no absolute build-machine paths in shipped .cmake files (R-22)
    #  - exactly one dylib per library, macosx_* wheel tag (H-3d)
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

# R-23 (II-24): the deployment target is read from the wheel's own platform
# tag (macosx_<major>_<minor>_<arch>) — the single source scikit-build-core
# derives from the effective CMAKE_OSX_DEPLOYMENT_TARGET (which honors a
# CMAKE_ARGS="-DCMAKE_OSX_DEPLOYMENT_TARGET=..." override). The old gate
# hardcoded 11.0 and REJECTED a documented override (e.g. 12.0) even though
# CMakeLists.txt/pyproject.toml advertise overridability.
BAD = re.compile(r"/(opt/homebrew|usr/local|Users/|Applications/Xcode)")
failures: list[str] = []
seen: dict[str, list[str]] = {}

def fail(msg: str) -> None:
    failures.append(msg)

def lib_key(name: str) -> str:
    # F-36: canonicalize the dylib stem so versioned chains collapse to one
    # key. The old regex lib?([A-Za-z0-9_]+?)(?:\.\d+)?\.dylib$ only matched a
    # SINGLE version component, so libhpgl.2.0.2.dylib fell through to the
    # full filename while libhpgl.dylib and hpgl.dylib shared a key — a
    # partial symlink chain (libhpgl.dylib + libhpgl.2.0.2.dylib) passed the
    # uniqueness gate. Strip the lib prefix, then any trailing version
    # components, then the extension.
    stem = name
    if stem.startswith("lib"):
        stem = stem[3:]
    if stem.endswith(".dylib"):
        stem = stem[: -len(".dylib")]
    stem = re.sub(r"\.\d+(\.\d+)*$", "", stem)
    return stem

tag_m = re.search(r"macosx_(\d+)_(\d+)", wheel.name)
if not tag_m:
    fail(f"wheel tag is not macosx: {wheel.name}")
    DEPLOYMENT_TARGET = 11.0
else:
    DEPLOYMENT_TARGET = float(f"{tag_m.group(1)}.{tag_m.group(2)}")

for d in dylibs:
    rel = str(d.relative_to(out))
    key = lib_key(d.name)
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

if failures:
    print("WHEEL VERIFY FAILED:", file=sys.stderr)
    for f in failures:
        print("  -", f, file=sys.stderr)
    sys.exit(1)
print(f"  WHEEL VERIFY PASSED: {len(dylibs)} dylib(s), no absolute build-machine paths, no LC_RPATH, minos <= {DEPLOYMENT_TARGET}, tag macosx_{DEPLOYMENT_TARGET}")
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
    # II-28 (coordinate): the wheel's reported __version__ must match the
    # version in the wheel filename (which derives from pyproject.toml). The
    # source-tree __init__.py prefers pyproject.toml; the installed wheel has
    # no source tree, so it falls back to importlib.metadata — a stale
    # dist-info (the II-28 defect class) would disagree here and FAIL the
    # wheel smoke instead of shipping a mislabeled artifact.
    WHEEL_VERSION=$(basename "$WHEEL_FILE" | sed -E 's/^hpgl-([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
    "$SMOKE_VENV/bin/python" -c "from geo_bsd import hpgl_wrap; assert hpgl_wrap._HAS_KRIGING_STATS, 'stale library loaded'; assert 'libhpgl' in str(hpgl_wrap._hpgl_so._name), 'loaded unexpected library: %s' % hpgl_wrap._hpgl_so._name; from geo_bsd import cvariogram; assert hasattr(cvariogram.cvar, 'cvar_clear_last_error'), '_cvariogram missing cvar_clear_last_error'; import geo_bsd; assert geo_bsd.__version__ == '$WHEEL_VERSION', 'wheel __version__=%r != wheel filename version %r' % (geo_bsd.__version__, '$WHEEL_VERSION'); print('  wheel smoke test PASSED:', hpgl_wrap._hpgl_so._name, 'version', geo_bsd.__version__)" || {
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
    # II-22: --config is required for multi-config generators (VS/Xcode) —
    # they IGNORE the preset's CMAKE_BUILD_TYPE cacheVariable, so without it
    # the windows-msvc preset silently built Debug. Single-config generators
    # (Ninja/Make) ignore --config, so passing it unconditionally is safe.
    cmake --build --preset "$USE_PRESET" --parallel "$NPROC" --config "$BUILD_CONFIG"
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

# III-24 (build-side): Python 3.9 grammar smoke on the dev path too — the
# declared requires-python is >=3.9 (pyproject.toml) and the wheel ships
# py3-none, so post-3.9 module-scope syntax is a shipping defect regardless
# of the host interpreter.
echo "Smoke test: checking Python 3.9 grammar compatibility of src/geo_bsd..."
if command -v uv &>/dev/null; then
    if ! py39_grammar_check uv run python; then
        echo "  ERROR: Python 3.9 grammar check failed (see above)" >&2
        exit 1
    fi
else
    if ! py39_grammar_check python3; then
        echo "  ERROR: Python 3.9 grammar check failed (see above)" >&2
        exit 1
    fi
fi

# R-26 relocatability check for the NON-wheel build path. The smoke test
# above verifies loadability/symbols but NOT that the runtime libraries are
# self-contained.
#
# F-33 (macOS): the gate runs for EVERY macOS build regardless of the cached
# HPGL_STATIC_LIBOMP value. The old gate skipped when the cache held OFF —
# exactly the defect class its comment names (stale cache, or a deliberate
# OFF) — silently shipping non-relocatable dylibs with an absolute
# /opt/homebrew/opt/libomp dependency. The cache value is not evidence; the
# artifacts are. Hard-fail when a runtime dylib carries an absolute
# build-machine dependency.
#
# III-28 (Linux): the same defect class — non-system BLAS/LAPACK bakes an
# absolute RPATH/RUNPATH into the built .so, which is not relocatable on
# other machines. The CMake RPATH extraction (CMakeLists.txt) deliberately
# embeds non-system library dirs, so standalone installs must be gated here.
echo "Smoke test: checking runtime library relocatability..."
if [[ "$PLATFORM" == "macos" ]]; then
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
elif [[ "$PLATFORM" == "linux" ]]; then
    RELOC_BAD=0
    for lib in "${RUNTIME_DIR}/hpgl.so" "${RUNTIME_DIR}/_cvariogram.so"; do
        if [[ ! -f "$lib" ]]; then
            echo "  WARNING: $lib not found — skipping relocatability check for it" >&2
            continue
        fi
        # readelf -d reveals RPATH/RUNPATH entries and (rare) absolute NEEDED
        # sonames; fall back to ldd when binutils is absent. $ORIGIN markers
        # are relative and allowed — only absolute non-system dirs fail.
        # R-01 (III-28 gate): the sed pattern must use GNU BRE escaped parens
        # \(...\) and emit capture group 2 (the path list). The pre-fix
        # 's/.*(RPATH\|RUNPATH)[^[]*\[\(.*\)\]/\1/p' mis-parsed under GNU sed
        # (unescaped parens literal, `\|` top-level alternation) — single-entry
        # absolute RPATH AND RUNPATH both bypassed the gate, shipping
        # non-relocatable .so files with a false "PASSED". Verified against
        # GNU sed 4.10: the corrected pattern extracts /opt/... and
        # /usr/local/... paths and still ignores $ORIGIN/../lib markers.
        if command -v readelf &>/dev/null; then
            ABS=$(readelf -d "$lib" 2>/dev/null \
                | sed -n 's/.*\(RPATH\|RUNPATH\)[^[]*\[\(.*\)\]/\2/p' \
                | tr ':' '\n' \
                | grep -E '^/(opt|usr/local|home/|Users/|Applications)' || true)
            if [[ -z "$ABS" ]]; then
                ABS=$(readelf -d "$lib" 2>/dev/null | awk '/NEEDED/ {if ($NF ~ /^\[.*\//) {gsub(/[\[\]]/, "", $NF); print $NF}}' || true)
            fi
        else
            # R-01: ldd prints 'soname => /resolved/path (0x...)'; column 3 is
            # the resolved path, column 1 is the soname (never matches the
            # absolute-path filter, so the old '$1' fallback silently passed).
            ABS=$(ldd "$lib" 2>/dev/null | awk '{print $3}' | grep -E '^/(opt|usr/local|home/|Users/|Applications)' || true)
        fi
        if [[ -n "$ABS" ]]; then
            echo "  ERROR: $lib carries absolute non-system RPATH/RUNPATH (baked by a non-system BLAS/LAPACK build):" >&2
            echo "$ABS" | sort -u | sed 's/^/    /' >&2
            RELOC_BAD=1
        fi
    done
    if [[ $RELOC_BAD -ne 0 ]]; then
        echo "  ERROR: runtime .so relocatability check FAILED — the build is not self-contained." >&2
        echo "  Rebuild with system BLAS/LAPACK (or bundle them) so no absolute RPATH is embedded." >&2
        exit 1
    fi
    echo "  Relocatability check: PASSED (no absolute build-machine RPATH/RUNPATH)"
fi

# ---------------------------------------------------------------------------
# III-29: execute the registered CTest suite (C++ unit tests + Python test
# suite) in the build pipeline. Previously the 4 add_test registrations in
# tests/CMakeLists.txt were dead weight — nothing invoked ctest, so a
# compile-clean regression shipped with a green banner. Runs only when tests
# were configured into this build directory (CTestTestfile.cmake exists);
# the Python CTest target excludes slow tests and sets a timeout (III-31).
# ---------------------------------------------------------------------------
echo "Smoke test: running CTest suite..."
if [[ -n "$USE_PRESET" ]]; then
    CTestDir="$SEARCH_DIR"
else
    CTestDir="$BUILD_DIR"
fi
if [[ -f "$CTestDir/CTestTestfile.cmake" ]]; then
    if (cd "$CTestDir" && ctest --output-on-failure); then
        echo "  CTest suite: PASSED"
    else
        echo "  ERROR: CTest suite FAILED — see failures above" >&2
        exit 1
    fi
else
    echo "  CTest suite: SKIPPED (no CTestTestfile.cmake in $CTestDir — tests not configured)"
fi

echo ""
echo "========================================"
echo "All checks passed. HPGL build is ready."
echo "========================================"
echo ""
