#!/usr/bin/env python3
"""
HPGL Test Runner

A comprehensive test runner for the HPGL (High Performance Geostatistics Library)
test suite. Supports multiple test categories, coverage reporting, and flexible
test filtering.

Usage:
    python run_tests.py [category] [options]

Categories:
    all           - Run all tests (default)
    kriging       - Kriging algorithm tests (OK, SK, IK)
    simulation    - Simulation algorithm tests (SGS, SIS)
    utilities     - Utility function tests
    classes       - Class and property tests
    edge          - Edge case and error handling tests
    legacy        - Migrated legacy tests
    numpy2        - NumPy 2.0+ compatibility tests
    memory        - Memory leak tests
    performance   - Performance benchmarks
    integration   - Integration workflow tests
    unit          - Tests marked with @pytest.mark.unit
    slow          - Tests marked with @pytest.mark.slow

Options:
    --cov         - Generate coverage report
    --cov-report  - Coverage report format (term, html, xml)
    -v, --verbose - Verbose output (shows individual test names)
    -s, --capture-no - Show print output during tests
    --slow        - Include slow-running tests
    -k EXP        - Filter tests by expression (passed to pytest)
    -x            - Stop on first failure
    --tb=STYLE    - Traceback style (long, short, line, no)
    --maxfail=N   - Stop after N failures
    -n N          - Run tests with N workers (requires pytest-xdist)

Examples:
    # Run all tests with coverage
    python run_tests.py all --cov

    # Run kriging tests with verbose output
    python run_tests.py kriging -v

    # Run specific test by expression
    python run_tests.py all -k "test_ok_basic"

    # Run memory tests only
    python run_tests.py memory

    # Run with pytest-xdist parallel execution
    python run_tests.py all -n 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def supports_color() -> bool:
    """Check if terminal supports color output."""
    # Check if we're on Windows
    if sys.platform == "win32":
        # Windows 10+ supports ANSI colors
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    # Unix-like systems typically support colors
    return True


def color_enabled() -> bool:
    """Check if color output should be enabled."""
    return supports_color() and sys.stdout.isatty()


def print_color(text: str, color: str = "") -> None:
    """Print colored text if colors are supported."""
    if color_enabled():
        print(f"{color}{text}{Colors.ENDC}")
    else:
        print(text)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_tests_dir() -> Path:
    """Get the tests directory path."""
    project_root = get_project_root()

    # Check for tests/python subdirectory
    python_tests_dir = project_root / "tests" / "python"
    if python_tests_dir.exists():
        return python_tests_dir

    # Fall back to tests directory
    tests_dir = project_root / "tests"
    if tests_dir.exists():
        return tests_dir

    # Current directory as fallback
    return Path(__file__).parent


def check_hpgl_available() -> tuple[bool, str]:
    """
    Check if HPGL library is available for import.

    Returns:
        Tuple of (is_available, message)
    """
    project_root = get_project_root()

    # Add potential source paths
    src_paths = [
        project_root / "src" / "geo_bsd",
        project_root / "build" / "lib" / "geo_bsd",
        project_root / "build" / "lib.win-amd64-3.11" / "geo_bsd",
    ]

    for src_path in src_paths:
        if src_path.exists():
            sys.path.insert(0, str(src_path))

    # Try to import
    try:
        import geo_bsd

        return True, "HPGL library loaded successfully"
    except ImportError as e:
        return False, f"HPGL library not available: {e}"


def get_pytest_command() -> str:
    """Get the pytest command to use."""
    # Use python -m pytest for consistency
    return f"{sys.executable} -m pytest"


def build_pytest_args(
    test_type: str,
    coverage: bool = False,
    cov_report: str = "term-missing",
    verbose: bool = False,
    show_output: bool = False,
    include_slow: bool = False,
    filter_expr: Optional[str] = None,
    stop_on_fail: bool = False,
    tb_style: str = "short",
    maxfail: Optional[int] = None,
    workers: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Build pytest command arguments.

    Args:
        test_type: Type/category of tests to run
        coverage: Generate coverage report
        cov_report: Coverage report format
        verbose: Verbose output
        show_output: Show print output during tests
        include_slow: Include slow-running tests
        filter_expr: Filter tests by expression
        stop_on_fail: Stop on first failure
        tb_style: Traceback style
        maxfail: Maximum failures before stopping
        workers: Number of parallel workers
        extra_args: Additional arguments to pass to pytest

    Returns:
        List of pytest command arguments
    """
    args = [sys.executable, "-m", "pytest"]

    # Add coverage if requested
    if coverage:
        args.extend(["--cov=geo_bsd", "--cov=geo", "--cov=sgs", "--cov=sis"])
        args.extend([f"--cov-report={cov_report}", "--cov-report=html"])

    # Verbosity
    if verbose:
        args.append("-vv")
    else:
        args.append("-v")

    # Show output
    if show_output:
        args.append("-s")

    # Stop on first failure
    if stop_on_fail:
        args.append("-x")

    # Traceback style
    args.append(f"--tb={tb_style}")

    # Max failures
    if maxfail is not None:
        args.append(f"--maxfail={maxfail}")

    # Parallel execution
    if workers is not None:
        args.extend(["-n", str(workers)])

    # Slow tests marker
    if include_slow:
        args.extend(["-m", "slow or not slow"])
    else:
        args.extend(["-m", "not slow"])

    # Filter expression
    if filter_expr:
        args.extend(["-k", filter_expr])

    # Get tests directory and determine target
    tests_dir = get_tests_dir()

    # Map test types to files/directories
    test_targets = {
        "all": str(tests_dir),
        "kriging": str(tests_dir / "test_kriging.py"),
        "simulation": str(tests_dir / "test_simulation.py"),
        "numpy2": str(tests_dir / "test_numpy2_compat.py"),
        "memory": str(tests_dir / "test_memory_leaks.py"),
        "performance": str(tests_dir / "test_performance.py"),
        "integration": str(tests_dir / "test_integration.py"),
        "utilities": str(tests_dir / "test_utilities.py") if (tests_dir / "test_utilities.py").exists() else str(tests_dir),
        "classes": str(tests_dir / "test_classes.py") if (tests_dir / "test_classes.py").exists() else str(tests_dir),
        "edge": str(tests_dir / "test_edge_cases.py") if (tests_dir / "test_edge_cases.py").exists() else str(tests_dir),
        "legacy": str(tests_dir / "test_legacy_migrated.py") if (tests_dir / "test_legacy_migrated.py").exists() else str(tests_dir),
        "unit": f"-m unit {tests_dir}",
        "slow": f"-m slow {tests_dir}",
    }

    # Get target for test type
    target = test_targets.get(test_type, str(tests_dir))
    args.extend(target.split())

    # Extra arguments
    if extra_args:
        args.extend(extra_args)

    return args


def print_header(text: str) -> None:
    """Print a formatted header."""
    width = 70
    print()
    print_color("=" * width, Colors.OKBLUE)
    print_color(f" {text} ".center(width), Colors.BOLD + Colors.OKBLUE)
    print_color("=" * width, Colors.OKBLUE)
    print()


def print_summary(test_type: str, result: subprocess.CompletedProcess[bytes]) -> None:
    """Print test execution summary."""
    width = 70

    if result.returncode == 0:
        print_color(f" [SUCCESS] {test_type.upper()} tests passed ".center(width, "="), Colors.OKGREEN + Colors.BOLD)
    else:
        print_color(f" [FAILED] {test_type.upper()} tests failed ".center(width, "="), Colors.FAIL + Colors.BOLD)

    print()


    if result.returncode == 0 and "--cov" in " ".join(result.args):
        print_color("Coverage report generated in htmlcov/ directory", Colors.OKCYAN)
        if sys.platform == "win32":
            print_color("  View with: start htmlcov/index.html", Colors.OKCYAN)
        else:
            print_color("  View with: open htmlcov/index.html", Colors.OKCYAN)
        print()


def run_tests(
    test_type: str = "all",
    coverage: bool = False,
    cov_report: str = "term-missing",
    verbose: bool = False,
    show_output: bool = False,
    include_slow: bool = False,
    filter_expr: Optional[str] = None,
    stop_on_fail: bool = False,
    tb_style: str = "short",
    maxfail: Optional[int] = None,
    workers: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    """
    Run HPGL tests.

    Args:
        test_type: Type of tests to run
        coverage: Generate coverage report
        cov_report: Coverage report format
        verbose: Enable verbose output
        show_output: Show print output during tests
        include_slow: Include slow-running tests
        filter_expr: Filter tests by expression
        stop_on_fail: Stop on first failure
        tb_style: Traceback style
        maxfail: Maximum failures before stopping
        workers: Number of parallel workers
        extra_args: Additional pytest arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    print_header(f"HPGL Test Runner - {test_type.upper()} Tests")

    # Check HPGL availability
    hpgl_available, hpgl_message = check_hpgl_available()
    print_color(f"HPGL Status: {hpgl_message}", Colors.OKGREEN if hpgl_available else Colors.WARNING)
    print()

    if not hpgl_available:
        print_color("Warning: HPGL library is not available. Tests will be skipped.", Colors.WARNING)
        print_color("To fix this:", Colors.WARNING)
        print_color("  1. Ensure HPGL has been built successfully", Colors.WARNING)
        print_color("  2. Check that the build output is in the expected location", Colors.WARNING)
        print()

    # Build pytest arguments
    pytest_args = build_pytest_args(
        test_type=test_type,
        coverage=coverage,
        cov_report=cov_report,
        verbose=verbose,
        show_output=show_output,
        include_slow=include_slow,
        filter_expr=filter_expr,
        stop_on_fail=stop_on_fail,
        tb_style=tb_style,
        maxfail=maxfail,
        workers=workers,
        extra_args=extra_args,
    )

    # Display command
    print_color(f"Running: {' '.join(pytest_args)}", Colors.OKCYAN)
    print()

    # Run tests
    project_root = get_project_root()
    result = subprocess.run(
        pytest_args,
        cwd=project_root,
        capture_output=False,
    )

    # Print summary
    print_summary(test_type, result)

    return result.returncode


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_tests",
        description="HPGL Test Runner - Run HPGL test suite with flexible options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py all --cov
  python run_tests.py kriging -v
  python run_tests.py all -k "test_ok_basic"
  python run_tests.py memory -v -s
  python run_tests.py all -n 4

For more information, see the documentation in TEST_README.md
        """,
    )

    parser.add_argument(
        "category",
        nargs="?",
        default="all",
        choices=[
            "all",
            "kriging",
            "simulation",
            "utilities",
            "classes",
            "edge",
            "legacy",
            "numpy2",
            "memory",
            "performance",
            "integration",
            "unit",
            "slow",
        ],
        help="Test category to run (default: all)",
    )

    parser.add_argument(
        "--cov",
        action="store_true",
        help="Generate coverage report",
    )

    parser.add_argument(
        "--cov-report",
        default="term-missing",
        choices=["term", "term-missing", "html", "xml", "json"],
        help="Coverage report format (default: term-missing)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (show individual test names)",
    )

    parser.add_argument(
        "-s", "--capture-no",
        action="store_true",
        help="Show print output during tests",
    )

    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow-running tests",
    )

    parser.add_argument(
        "-k",
        dest="filter_expr",
        metavar="EXPR",
        help="Filter tests by expression",
    )

    parser.add_argument(
        "-x",
        action="store_true",
        help="Stop on first failure",
    )

    parser.add_argument(
        "--tb",
        default="short",
        choices=["long", "short", "line", "no"],
        help="Traceback style (default: short)",
    )

    parser.add_argument(
        "--maxfail",
        type=int,
        metavar="N",
        help="Stop after N failures",
    )

    parser.add_argument(
        "-n",
        dest="workers",
        type=int,
        metavar="N",
        help="Number of parallel workers (requires pytest-xdist)",
    )

    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to pytest",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    return run_tests(
        test_type=args.category,
        coverage=args.cov,
        cov_report=args.cov_report,
        verbose=args.verbose,
        show_output=args.capture_no,
        include_slow=args.slow,
        filter_expr=args.filter_expr,
        stop_on_fail=args.x,
        tb_style=args.tb,
        maxfail=args.maxfail,
        workers=args.workers,
        extra_args=args.extra_args if args.extra_args else None,
    )


if __name__ == "__main__":
    sys.exit(main())
