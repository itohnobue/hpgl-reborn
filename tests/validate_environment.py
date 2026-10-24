#!/usr/bin/env python3
"""
Validate that the environment is ready for HPGL testing.

This script checks:
1. Python version
2. NumPy installation
3. HPGL library availability
4. Test dependencies

Run this before attempting to run the test suite.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version is 3.9+"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Found: Python {version.major}.{version.minor}.{version.micro}")

    if version >= (3, 9):
        print("  [PASS] Python version is acceptable (>= 3.9)")
        return True
    else:
        print("  [WARN] Python version < 3.9, some features may not work")
        return False


def check_numpy():
    """Check NumPy installation"""
    print("\nChecking NumPy...")
    try:
        import numpy

        print(f"  Found: NumPy {numpy.__version__}")

        # Parse version
        version_parts = numpy.__version__.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])

        # L-21: project declares numpy>=2.0,<3.0 (pyproject.toml); accept only 2.0+
        if (major, minor) >= (2, 0):
            print("  [PASS] NumPy version is acceptable (>= 2.0)")
            return True
        else:
            print(f"  [WARN] NumPy version {numpy.__version__} < 2.0, upgrade recommended")
            return False
    except ImportError:
        print("  [FAIL] NumPy not installed")
        print("  Install with: uv sync")
        return False


def check_pytest():
    """Check pytest installation"""
    print("\nChecking pytest...")
    try:
        import pytest

        print(f"  Found: pytest {pytest.__version__}")
        print("  [PASS] pytest is installed")
        return True
    except ImportError:
        print("  [WARN] pytest not installed")
        print("  Install with: uv sync --extra test")
        return False


def check_hpgl():
    """Check HPGL library availability"""
    print("\nChecking HPGL library...")

    # Add src/ to path so geo_bsd package can be imported
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))

    # F-546/F-565: `importlib.util.find_spec("geo_bsd")` only proves that a
    # module path exists — it never executes geo_bsd/__init__.py, so a missing,
    # stale or corrupt native library still printed
    # "[PASS] HPGL library imported successfully". Import the package for real:
    # that runs hpgl_wrap's ctypes load and raises OSError when the library
    # cannot be loaded.
    try:
        import geo_bsd
    except (ImportError, OSError) as exc:
        print(f"  [FAIL] Cannot import HPGL library: {exc}")
        print("\nPossible reasons:")
        print("  1. Build has not completed - run build.bat (Windows) or cmake (Linux)")
        print("  2. DLL/SO not in expected location (src/geo_bsd/)")
        print("  3. Missing dependencies (run: uv sync)")
        return False

    # F-565: report WHICH file ctypes actually resolved — a stale library
    # shadowing the fresh build is the failure class this check exists to catch,
    # and the only other trace of it is a loader warning nobody surfaces.
    loaded = getattr(getattr(geo_bsd, "hpgl_wrap", None), "_hpgl_so", None)
    loaded_path = getattr(loaded, "_name", None)
    if loaded_path:
        print(f"  [OK] HPGL native library: {Path(loaded_path).resolve()}")

    # __init__.py:41-49 swallows a failed _cvariogram load, so a green import
    # alone does not prove it loaded.
    if getattr(geo_bsd, "cvariogram", None) is None:
        print("  [FAIL] _cvariogram extension did not load")
        print("  Rebuild with HPGL_BUILD_VARIOGRAM=ON (see cmake options)")
        return False

    print("  [PASS] HPGL library imported successfully")
    return True


# II-20: platform-aware native-library name table, mirroring hpgl_wrap.py's
# lib_paths so check_build_files accepts the platform's ACTUAL extension
# (.dylib on macOS, .dll on Windows, .so on Linux) instead of only .dll/.so
# — which false-FAILed every healthy macOS build (hpgl.dylib was never
# checked). Keys follow sys.platform conventions; the fallback entry covers
# any other POSIX platform.
_NATIVE_LIB_GLOBS = {
    "win32": [
        "hpgl.dll", "libhpgl.dll", "hpgl_d.dll", "libhpgl_d.dll",
        "_cvariogram.dll", "lib_cvariogram.dll", "_cvariogram_d.dll", "lib_cvariogram_d.dll",
    ],
    "darwin": [
        "hpgl.dylib", "hpgl.so", "libhpgl.dylib", "libhpgl.so",
        "_cvariogram.dylib", "_cvariogram.so", "lib_cvariogram.dylib", "lib_cvariogram.so",
    ],
    "linux": [
        "hpgl.so", "libhpgl.so",
        "_cvariogram.so", "lib_cvariogram.so",
    ],
}


def _platform_lib_key() -> str:
    """Map sys.platform to the _NATIVE_LIB_GLOBS key."""
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _build_files_present(geo_bsd_dir: Path):
    """Return (hpgl_found, cvariogram_found) for the platform's library names.

    Pure and testable: the caller resolves the directory, this function only
    checks existence against the platform table.
    """
    platform_key = _platform_lib_key()
    names = _NATIVE_LIB_GLOBS[platform_key]
    hpgl_names = [n for n in names if n.startswith(("hpgl", "libhpgl"))]
    cvar_names = [n for n in names if n.startswith(("_cvariogram", "lib_cvariogram"))]
    hpgl_found = any((geo_bsd_dir / n).exists() for n in hpgl_names)
    cvar_found = any((geo_bsd_dir / n).exists() for n in cvar_names)
    return hpgl_found, cvar_found


def _report_lib_load(files):
    """Probe each candidate native library with a real load; True if any loads.

    F-565: ``Path.exists()`` passes for a truncated, wrong-architecture or
    stale library — only ``dlopen`` reports that. Every candidate found by the
    platform table is probed and its resolved path printed, because the
    loader's preference order (hpgl.dylib before libhpgl.dylib, see
    hpgl_wrap.py:280-315) decides which file ctypes actually uses.
    """
    import ctypes

    any_ok = False
    for f in files:
        try:
            ctypes.CDLL(str(f.resolve()))
        except OSError as exc:
            print(f"  [FAIL] {f.name} exists but does not load: {exc}")
            continue
        print(f"  [OK] {f.name} (loads: {f.resolve()})")
        any_ok = True
    return any_ok


def check_build_files():
    """Check for built extension files"""
    print("\nChecking for built native libraries...")

    project_root = Path(__file__).parent.parent
    geo_bsd_dir = project_root / "src" / "geo_bsd"

    # Look for native libraries with the platform's actual names/extensions
    # (II-20: .dylib on macOS, .dll on Windows, .so on Linux).
    platform_names = _NATIVE_LIB_GLOBS[_platform_lib_key()]
    hpgl_globs = [n for n in platform_names if n.startswith(("hpgl", "libhpgl"))]
    cvar_globs = [n for n in platform_names if n.startswith(("_cvariogram", "lib_cvariogram"))]
    dll_files, cvar_files = [], []
    hpgl_found, cvar_found = _build_files_present(geo_bsd_dir)
    if hpgl_found:
        dll_files = [geo_bsd_dir / n for n in hpgl_globs if (geo_bsd_dir / n).exists()]
    if cvar_found:
        cvar_files = [geo_bsd_dir / n for n in cvar_globs if (geo_bsd_dir / n).exists()]

    # F-565: existence is not loadability — probe every candidate found.
    if dll_files:
        hpgl_loads = _report_lib_load(dll_files)
    else:
        print(f"  [MISSING] hpgl library ({' / '.join(hpgl_globs)})")
        hpgl_loads = False

    if cvar_files:
        cvar_loads = _report_lib_load(cvar_files)
    else:
        print(f"  [MISSING] _cvariogram library ({' / '.join(cvar_globs)})")
        cvar_loads = False

    if hpgl_loads and cvar_loads:
        print("  [PASS] Native libraries exist and load")
        return True
    else:
        print("  [FAIL] Native libraries not found or not loadable")
        print("\n  Build with: build.bat (Windows) or cmake (Linux/macOS)")
        return False


def check_test_files():
    """Check test files exist"""
    print("\nChecking test files...")

    tests_dir = Path(__file__).parent / "python"
    # L-21: derive the manifest from the filesystem instead of maintaining a
    # stale hardcoded list (the old 18-file list silently passed with the
    # pin/FFI/s6 families missing from the checkout). Glob over test_*.py so
    # the check stays in sync as files are added/removed.
    test_files = ["conftest.py"] + sorted(
        p.name for p in tests_dir.glob("test_*.py") if p.is_file()
    )

    all_exist = True
    for test_file in test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            print(f"  [OK] {test_file}")
        else:
            print(f"  [MISSING] {test_file}")
            all_exist = False

    if all_exist:
        print("  [PASS] All test files present")
        return True
    else:
        print("  [FAIL] Some test files missing")
        return False


def main():
    """Run all checks"""
    print("=" * 60)
    print("HPGL Testing Environment Validation")
    print("=" * 60)

    results = {
        "Python version": check_python_version(),
        "NumPy": check_numpy(),
        "pytest": check_pytest(),
        "HPGL library": check_hpgl(),
        "Build files": check_build_files(),
        "Test files": check_test_files(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {check}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] Environment is ready for testing!")
        # N2-L24/N2-L27-adjacent: recommend the operational default-marker
        # command (no -v — unconditional verbose is the documented pytest
        # INTERNALERROR trigger — and slow tests are excluded from the
        # default suite by design).
        print("\nRun tests with: uv run pytest tests/python/ -m \"not slow\"")
        return 0
    elif results["HPGL library"] or results["Build files"]:
        print("\n[PARTIAL] Some checks failed")
        print("\nTests will be skipped if HPGL is not available.")
        return 1
    else:
        print("\n[NOT READY] Environment not ready for testing")
        print("\nPlease complete the build and install dependencies first.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
