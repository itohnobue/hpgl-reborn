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
import os
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
        version_parts = numpy.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if (major, minor) >= (1, 24):
            print("  [PASS] NumPy version is acceptable (>= 1.24)")
            return True
        else:
            print(f"  [WARN] NumPy version {numpy.__version__} < 1.24, upgrade recommended")
            return False
    except ImportError:
        print("  [FAIL] NumPy not installed")
        print("  Install with: pip install numpy>=1.24")
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
        print("  Install with: pip install pytest")
        return False

def check_hpgl():
    """Check HPGL library availability"""
    print("\nChecking HPGL library...")
    
    # Add src/geo_bsd to path
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src" / "geo_bsd"
    sys.path.insert(0, str(src_dir))
    
    try:
        import geo
        print("  [PASS] HPGL library imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Cannot import HPGL library: {e}")
        print("\nPossible reasons:")
        print("  1. Build has not completed - no Python extension available")
        print("  2. Python extension not in expected location")
        print("  3. Missing dependencies (NumPy, etc.)")
        print("\nTo fix:")
        print("  1. Complete the build process")
        print("  2. Verify .pyd or .so files exist in build/ directory")
        print("  3. Install required dependencies")
        return False

def check_build_files():
    """Check for built extension files"""
    print("\nChecking for built extension files...")
    
    project_root = Path(__file__).parent.parent
    build_dir = project_root / "build"
    
    # Look for Python extensions
    pyd_files = list(build_dir.glob("**/*.pyd"))
    so_files = list(build_dir.glob("**/*.so"))
    dll_files = list(build_dir.glob("**/*.dll"))
    
    if pyd_files or so_files:
        print(f"  Found {len(pyd_files)} .pyd files")
        print(f"  Found {len(so_files)} .so files")
        print("  [PASS] Extension files exist")
        return True
    else:
        print("  [FAIL] No Python extension files (.pyd or .so) found")
        print("\nThe build has not completed yet. Tests will be skipped.")
        return False

def check_test_files():
    """Check test files exist"""
    print("\nChecking test files...")
    
    tests_dir = Path(__file__).parent / "python"
    test_files = [
        "conftest.py",
        "test_kriging.py",
        "test_simulation.py",
        "test_numpy2_compat.py",
        "test_memory_leaks.py",
        "test_performance.py",
        "test_integration.py"
    ]
    
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
        "Test files": check_test_files()
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
        print("\nRun tests with: python tests/run_tests.py all")
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
