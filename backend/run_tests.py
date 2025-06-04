#!/usr/bin/env python3
"""
Comprehensive test runner for Oreja backend.
Provides easy access to run different test suites with various configurations.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


class OrejaTestRunner:
    """Test runner for the Oreja backend."""
    
    def __init__(self):
        self.backend_dir = Path(__file__).parent
        self.project_root = self.backend_dir.parent
        
    def run_unit_tests(self, verbose=False, coverage=True):
        """Run unit tests only."""
        cmd = ["python", "-m", "pytest", "-m", "unit"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=.", "--cov-report=term-missing"])
        
        return self._run_command(cmd)
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests only."""
        cmd = ["python", "-m", "pytest", "-m", "integration"]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def run_api_tests(self, verbose=False):
        """Run API endpoint tests only."""
        cmd = ["python", "-m", "pytest", "-m", "api"]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def run_speaker_tests(self, verbose=False):
        """Run speaker recognition tests only."""
        cmd = ["python", "-m", "pytest", "-m", "speaker"]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def run_slow_tests(self, verbose=False):
        """Run performance and slow tests."""
        cmd = ["python", "-m", "pytest", "-m", "slow"]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def run_all_tests(self, verbose=False, coverage=True, html_report=False):
        """Run all tests."""
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=.", "--cov-report=term-missing"])
            
            if html_report:
                cmd.append("--cov-report=html")
        
        return self._run_command(cmd)
    
    def run_quick_tests(self, verbose=False):
        """Run only fast tests (exclude slow and integration)."""
        cmd = ["python", "-m", "pytest", "-m", "not slow and not integration"]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def run_parallel_tests(self, num_workers=4, verbose=False):
        """Run tests in parallel using pytest-xdist."""
        cmd = ["python", "-m", "pytest", "-n", str(num_workers)]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def generate_html_report(self):
        """Generate an HTML test report."""
        cmd = [
            "python", "-m", "pytest", 
            "--html=test_report.html", 
            "--self-contained-html",
            "--cov=.",
            "--cov-report=html:htmlcov"
        ]
        
        result = self._run_command(cmd)
        
        if result == 0:
            print(f"\n✅ HTML reports generated:")
            print(f"   Test Report: {self.backend_dir}/test_report.html")
            print(f"   Coverage Report: {self.backend_dir}/htmlcov/index.html")
        
        return result
    
    def install_test_dependencies(self):
        """Install test dependencies."""
        cmd = ["python", "-m", "pip", "install", "-r", "requirements_test.txt"]
        
        print("📦 Installing test dependencies...")
        result = self._run_command(cmd)
        
        if result == 0:
            print("✅ Test dependencies installed successfully")
        else:
            print("❌ Failed to install test dependencies")
        
        return result
    
    def check_test_environment(self):
        """Check if the test environment is properly set up."""
        print("🔍 Checking test environment...")
        
        # Check if test dependencies are installed
        try:
            import pytest
            import pytest_cov
            import pytest_asyncio
            print("✅ Core test dependencies available")
        except ImportError as e:
            print(f"❌ Missing test dependency: {e}")
            print("💡 Run: python run_tests.py --install-deps")
            return False
        
        # Check if main modules can be imported
        try:
            sys.path.insert(0, str(self.backend_dir))
            import server
            import speaker_embeddings
            print("✅ Main modules can be imported")
        except ImportError as e:
            print(f"❌ Cannot import main modules: {e}")
            return False
        
        # Check if test files exist
        test_files = [
            "tests/conftest.py",
            "tests/test_api_endpoints.py", 
            "tests/test_speaker_embeddings.py",
            "tests/test_utils.py"
        ]
        
        missing_files = []
        for test_file in test_files:
            if not (self.backend_dir / test_file).exists():
                missing_files.append(test_file)
        
        if missing_files:
            print(f"❌ Missing test files: {missing_files}")
            return False
        
        print("✅ Test files found")
        print("✅ Test environment is ready!")
        return True
    
    def _run_command(self, cmd):
        """Run a command and return exit code."""
        try:
            # Change to backend directory
            os.chdir(self.backend_dir)
            
            print(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            return result.returncode
            
        except KeyboardInterrupt:
            print("\n⚠️ Tests interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return 1


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Oreja Backend Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit --verbose         # Run unit tests with verbose output
  python run_tests.py --api --coverage         # Run API tests with coverage
  python run_tests.py --quick                  # Run only fast tests
  python run_tests.py --parallel --workers 8   # Run tests in parallel
  python run_tests.py --report                 # Generate HTML report
  python run_tests.py --check                  # Check test environment
        """
    )
    
    # Test selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--unit", action="store_true", help="Run unit tests only")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests only")
    test_group.add_argument("--api", action="store_true", help="Run API tests only")
    test_group.add_argument("--speaker", action="store_true", help="Run speaker tests only")
    test_group.add_argument("--slow", action="store_true", help="Run slow/performance tests")
    test_group.add_argument("--quick", action="store_true", help="Run only fast tests")
    
    # Test options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    
    # Utility options
    parser.add_argument("--report", action="store_true", help="Generate HTML test report")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--check", action="store_true", help="Check test environment")
    
    args = parser.parse_args()
    
    runner = OrejaTestRunner()
    
    # Handle utility commands
    if args.install_deps:
        return runner.install_test_dependencies()
    
    if args.check:
        if runner.check_test_environment():
            return 0
        else:
            return 1
    
    if args.report:
        return runner.generate_html_report()
    
    # Check environment before running tests
    if not runner.check_test_environment():
        print("\n💡 Fix the environment issues above before running tests")
        return 1
    
    # Determine coverage setting
    coverage = not args.no_coverage
    
    # Run appropriate test suite
    if args.parallel:
        return runner.run_parallel_tests(args.workers, args.verbose)
    elif args.unit:
        return runner.run_unit_tests(args.verbose, coverage)
    elif args.integration:
        return runner.run_integration_tests(args.verbose)
    elif args.api:
        return runner.run_api_tests(args.verbose)
    elif args.speaker:
        return runner.run_speaker_tests(args.verbose)
    elif args.slow:
        return runner.run_slow_tests(args.verbose)
    elif args.quick:
        return runner.run_quick_tests(args.verbose)
    elif args.all:
        return runner.run_all_tests(args.verbose, coverage, html_report=True)
    else:
        # Default: run quick tests
        print("🏃 Running quick test suite (use --all for complete tests)")
        return runner.run_quick_tests(args.verbose)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 