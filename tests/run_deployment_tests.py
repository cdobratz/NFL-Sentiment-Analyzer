"""
Deployment and Infrastructure Test Runner

Comprehensive test runner for all deployment and infrastructure tests.
Runs tests in logical order and provides detailed reporting.

Requirements covered: 8.1, 8.2, 8.3
"""

import pytest
import sys
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TestResult:
    """Test result data structure."""
    name: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: Optional[str] = None


@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    file: str
    description: str
    required: bool = True
    depends_on: List[str] = None


class DeploymentTestRunner:
    """Main test runner for deployment tests."""
    
    def __init__(self):
        self.test_suites = [
            TestSuite(
                name="docker_builds",
                file="test_docker_builds.py",
                description="Docker container build validation",
                required=True
            ),
            TestSuite(
                name="deployment_infrastructure",
                file="test_deployment_infrastructure.py",
                description="Multi-service orchestration and configuration",
                required=True
            ),
            TestSuite(
                name="smoke_tests",
                file="test_smoke_tests.py",
                description="Basic application health and functionality",
                required=True,
                depends_on=["docker_builds"]
            ),
            TestSuite(
                name="security_validation",
                file="test_security_validation.py",
                description="Security configuration and vulnerability checks",
                required=True
            ),
            TestSuite(
                name="performance_validation",
                file="test_performance_validation.py",
                description="Performance and scalability validation",
                required=False,
                depends_on=["smoke_tests"]
            )
        ]
        
        self.results: Dict[str, TestResult] = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, verbose: bool = False, fail_fast: bool = False) -> bool:
        """Run all deployment tests."""
        print("=" * 80)
        print("NFL Analyzer - Deployment and Infrastructure Tests")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = time.time()
        overall_success = True
        
        # Check prerequisites
        if not self._check_prerequisites():
            print("❌ Prerequisites not met. Aborting tests.")
            return False
        
        # Run test suites in order
        for suite in self.test_suites:
            if not self._should_run_suite(suite):
                continue
            
            print(f"Running {suite.name}: {suite.description}")
            print("-" * 60)
            
            success = self._run_test_suite(suite, verbose)
            
            if not success:
                overall_success = False
                if fail_fast and suite.required:
                    print(f"❌ Required test suite '{suite.name}' failed. Stopping execution.")
                    break
            
            print()
        
        self.end_time = time.time()
        self._print_summary()
        
        return overall_success
    
    def run_specific_test(self, test_name: str, verbose: bool = False) -> bool:
        """Run a specific test suite."""
        suite = next((s for s in self.test_suites if s.name == test_name), None)
        
        if not suite:
            print(f"❌ Test suite '{test_name}' not found.")
            available = [s.name for s in self.test_suites]
            print(f"Available test suites: {', '.join(available)}")
            return False
        
        print(f"Running specific test: {suite.name}")
        print("-" * 60)
        
        return self._run_test_suite(suite, verbose)
    
    def _check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        print("Checking prerequisites...")
        
        # Check if Docker is available
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("❌ Docker not available")
                return False
            print("✅ Docker available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Docker not available")
            return False
        
        # Check if docker-compose is available
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("❌ Docker Compose not available")
                return False
            print("✅ Docker Compose available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Docker Compose not available")
            return False
        
        # Check if required files exist
        required_files = [
            'Dockerfile',
            'docker-compose.yml',
            'requirements.txt'
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                print(f"❌ Required file not found: {file_path}")
                return False
            print(f"✅ Found {file_path}")
        
        print("✅ All prerequisites met")
        print()
        return True
    
    def _should_run_suite(self, suite: TestSuite) -> bool:
        """Check if a test suite should be run."""
        # Check dependencies
        if suite.depends_on:
            for dependency in suite.depends_on:
                if dependency not in self.results:
                    print(f"⏭️  Skipping {suite.name}: dependency {dependency} not run")
                    return False
                
                if self.results[dependency].status != 'passed':
                    print(f"⏭️  Skipping {suite.name}: dependency {dependency} failed")
                    return False
        
        return True
    
    def _run_test_suite(self, suite: TestSuite, verbose: bool = False) -> bool:
        """Run a single test suite."""
        test_file = Path("tests") / suite.file
        
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            self.results[suite.name] = TestResult(
                name=suite.name,
                status='error',
                duration=0,
                message=f"Test file not found: {test_file}"
            )
            return False
        
        # Prepare pytest arguments
        pytest_args = [
            str(test_file),
            "-v" if verbose else "-q",
            "--tb=short",
            "--no-header",
            "--disable-warnings"
        ]
        
        start_time = time.time()
        
        try:
            # Run pytest
            result = subprocess.run([
                sys.executable, "-m", "pytest"
            ] + pytest_args, 
            capture_output=True, text=True, timeout=600)
            
            duration = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                status = 'passed'
                print(f"✅ {suite.name} passed ({duration:.1f}s)")
                message = None
            elif result.returncode == 5:  # No tests collected
                status = 'skipped'
                print(f"⏭️  {suite.name} skipped - no tests collected ({duration:.1f}s)")
                message = "No tests collected"
            else:
                status = 'failed'
                print(f"❌ {suite.name} failed ({duration:.1f}s)")
                message = result.stdout + result.stderr
                
                if verbose:
                    print("Error output:")
                    print(result.stdout)
                    if result.stderr:
                        print("Stderr:")
                        print(result.stderr)
            
            self.results[suite.name] = TestResult(
                name=suite.name,
                status=status,
                duration=duration,
                message=message
            )
            
            return status == 'passed'
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"❌ {suite.name} timed out after {duration:.1f}s")
            
            self.results[suite.name] = TestResult(
                name=suite.name,
                status='error',
                duration=duration,
                message="Test execution timed out"
            )
            
            return False
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {suite.name} error: {e}")
            
            self.results[suite.name] = TestResult(
                name=suite.name,
                status='error',
                duration=duration,
                message=str(e)
            )
            
            return False
    
    def _print_summary(self):
        """Print test execution summary."""
        print("=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            print(f"Total execution time: {total_duration:.1f}s")
            print()
        
        # Count results
        passed = sum(1 for r in self.results.values() if r.status == 'passed')
        failed = sum(1 for r in self.results.values() if r.status == 'failed')
        skipped = sum(1 for r in self.results.values() if r.status == 'skipped')
        errors = sum(1 for r in self.results.values() if r.status == 'error')
        
        print(f"Results: {passed} passed, {failed} failed, {skipped} skipped, {errors} errors")
        print()
        
        # Print detailed results
        for suite in self.test_suites:
            if suite.name not in self.results:
                continue
            
            result = self.results[suite.name]
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'skipped': '⏭️ ',
                'error': '💥'
            }.get(result.status, '❓')
            
            required_text = " (required)" if suite.required else " (optional)"
            print(f"{status_icon} {result.name}: {result.status} ({result.duration:.1f}s){required_text}")
            
            if result.message and result.status in ['failed', 'error']:
                # Print first few lines of error message
                lines = result.message.split('\n')[:5]
                for line in lines:
                    if line.strip():
                        print(f"    {line}")
                if len(result.message.split('\n')) > 5:
                    print("    ...")
        
        print()
        
        # Overall result
        required_failed = any(
            self.results.get(s.name, TestResult('', 'error', 0)).status in ['failed', 'error']
            for s in self.test_suites if s.required
        )
        
        if required_failed:
            print("❌ DEPLOYMENT TESTS FAILED - Required tests failed")
            print("   Please fix the issues before deploying to production.")
        else:
            print("✅ DEPLOYMENT TESTS PASSED - Ready for production deployment")
            
            optional_issues = any(
                self.results.get(s.name, TestResult('', 'error', 0)).status in ['failed', 'error']
                for s in self.test_suites if not s.required
            )
            
            if optional_issues:
                print("⚠️  Some optional tests failed - consider investigating")
    
    def generate_report(self, output_file: str = "deployment_test_report.json"):
        """Generate JSON report of test results."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": self.end_time - self.start_time if self.start_time and self.end_time else 0,
            "summary": {
                "passed": sum(1 for r in self.results.values() if r.status == 'passed'),
                "failed": sum(1 for r in self.results.values() if r.status == 'failed'),
                "skipped": sum(1 for r in self.results.values() if r.status == 'skipped'),
                "errors": sum(1 for r in self.results.values() if r.status == 'error')
            },
            "results": {
                name: {
                    "status": result.status,
                    "duration": result.duration,
                    "message": result.message
                }
                for name, result in self.results.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Test report saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run deployment and infrastructure tests")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--fail-fast", "-x", action="store_true",
                       help="Stop on first failure")
    parser.add_argument("--test", "-t", type=str,
                       help="Run specific test suite")
    parser.add_argument("--report", "-r", type=str, default="deployment_test_report.json",
                       help="Output file for JSON report")
    
    args = parser.parse_args()
    
    runner = DeploymentTestRunner()
    
    if args.test:
        success = runner.run_specific_test(args.test, args.verbose)
    else:
        success = runner.run_all_tests(args.verbose, args.fail_fast)
    
    # Generate report
    runner.generate_report(args.report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()