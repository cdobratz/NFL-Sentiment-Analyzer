#!/usr/bin/env python3
"""
Test runner for API documentation and monitoring validation tests.

This script runs all tests related to:
- API documentation accuracy and completeness
- Rate limiting and authentication mechanisms  
- Load testing for API performance under high traffic
- Monitoring and alerting system validation

Requirements: 6.2, 6.3, 6.4
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import json


class TestRunner:
    """Test runner for API documentation and monitoring tests"""
    
    def __init__(self):
        self.test_files = [
            "tests/test_api_documentation_monitoring.py",
            "tests/test_load_performance.py", 
            "tests/test_monitoring_validation.py"
        ]
        self.results = {}
    
    def run_test_file(self, test_file: str, markers: List[str] = None) -> Dict[str, Any]:
        """Run a specific test file and return results"""
        print(f"\n{'='*60}")
        print(f"Running tests in {test_file}")
        print(f"{'='*60}")
        
        cmd = ["python", "-m", "pytest", test_file, "-v", "--tb=short"]
        
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Add coverage if available
        try:
            subprocess.run(["python", "-c", "import pytest_cov"], 
                         check=True, capture_output=True)
            cmd.extend(["--cov=app", "--cov-report=term-missing"])
        except subprocess.CalledProcessError:
            print("pytest-cov not available, skipping coverage")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        return {
            "file": test_file,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": end_time - start_time,
            "success": result.returncode == 0
        }
    
    def run_documentation_tests(self) -> Dict[str, Any]:
        """Run API documentation tests"""
        print("\n🔍 Running API Documentation Tests...")
        return self.run_test_file(
            "tests/test_api_documentation_monitoring.py::TestAPIDocumentation"
        )
    
    def run_rate_limiting_tests(self) -> Dict[str, Any]:
        """Run rate limiting tests"""
        print("\n⏱️  Running Rate Limiting Tests...")
        return self.run_test_file(
            "tests/test_api_documentation_monitoring.py::TestRateLimiting"
        )
    
    def run_authentication_tests(self) -> Dict[str, Any]:
        """Run authentication mechanism tests"""
        print("\n🔐 Running Authentication Tests...")
        return self.run_test_file(
            "tests/test_api_documentation_monitoring.py::TestAuthentication"
        )
    
    def run_load_tests(self) -> Dict[str, Any]:
        """Run load testing tests"""
        print("\n🚀 Running Load Testing Tests...")
        return self.run_test_file("tests/test_load_performance.py")
    
    def run_monitoring_tests(self) -> Dict[str, Any]:
        """Run monitoring validation tests"""
        print("\n📊 Running Monitoring Validation Tests...")
        return self.run_test_file("tests/test_monitoring_validation.py")
    
    def run_all_tests(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run all API documentation and monitoring tests"""
        print("🧪 Starting API Documentation and Monitoring Test Suite")
        print(f"Test files: {len(self.test_files)}")
        
        all_results = {
            "documentation": [],
            "rate_limiting": [],
            "authentication": [],
            "load_testing": [],
            "monitoring": [],
            "summary": {}
        }
        
        # Run documentation tests
        doc_result = self.run_documentation_tests()
        all_results["documentation"].append(doc_result)
        
        # Run rate limiting tests
        rate_result = self.run_rate_limiting_tests()
        all_results["rate_limiting"].append(rate_result)
        
        # Run authentication tests
        auth_result = self.run_authentication_tests()
        all_results["authentication"].append(auth_result)
        
        # Run load tests
        load_result = self.run_load_tests()
        all_results["load_testing"].append(load_result)
        
        # Run monitoring tests
        monitor_result = self.run_monitoring_tests()
        all_results["monitoring"].append(monitor_result)
        
        # Generate summary
        all_test_results = [
            doc_result, rate_result, auth_result, load_result, monitor_result
        ]
        
        total_tests = len(all_test_results)
        successful_tests = sum(1 for r in all_test_results if r["success"])
        total_duration = sum(r["duration"] for r in all_test_results)
        
        all_results["summary"] = {
            "total_test_suites": total_tests,
            "successful_suites": successful_tests,
            "failed_suites": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration
        }
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test results summary"""
        print(f"\n{'='*60}")
        print("📋 TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        
        summary = results["summary"]
        
        print(f"Total Test Suites: {summary['total_test_suites']}")
        print(f"Successful Suites: {summary['successful_suites']}")
        print(f"Failed Suites: {summary['failed_suites']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        
        # Print individual results
        test_categories = [
            ("documentation", "📚 API Documentation"),
            ("rate_limiting", "⏱️  Rate Limiting"),
            ("authentication", "🔐 Authentication"),
            ("load_testing", "🚀 Load Testing"),
            ("monitoring", "📊 Monitoring")
        ]
        
        print(f"\n{'Individual Test Results:'}")
        print("-" * 40)
        
        for category, display_name in test_categories:
            if category in results and results[category]:
                result = results[category][0]  # First (and only) result
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                duration = result["duration"]
                print(f"{display_name:<20} {status} ({duration:.2f}s)")
        
        # Print any failures
        failed_categories = []
        for category, display_name in test_categories:
            if category in results and results[category]:
                result = results[category][0]
                if not result["success"]:
                    failed_categories.append((category, display_name, result))
        
        if failed_categories:
            print(f"\n{'❌ FAILED TESTS:'}")
            print("-" * 40)
            
            for category, display_name, result in failed_categories:
                print(f"\n{display_name}:")
                if result["stderr"]:
                    print("STDERR:")
                    print(result["stderr"][:500] + "..." if len(result["stderr"]) > 500 else result["stderr"])
                if result["stdout"]:
                    # Extract just the failure summary
                    stdout_lines = result["stdout"].split('\n')
                    failure_lines = [line for line in stdout_lines if 'FAILED' in line or 'ERROR' in line]
                    if failure_lines:
                        print("FAILURES:")
                        for line in failure_lines[:5]:  # Show first 5 failures
                            print(f"  {line}")
    
    def run_quick_validation(self) -> bool:
        """Run quick validation tests for CI/CD"""
        print("🔍 Running Quick API Validation...")
        
        # Run the minimal tests that don't require full app initialization
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/test_api_documentation_monitoring_minimal.py", "-v"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("  ✅ All minimal validation tests PASSED")
            return True
        else:
            print("  ❌ Some validation tests FAILED")
            print("STDERR:", result.stderr[:200] if result.stderr else "None")
            return False


def main():
    """Main test runner function"""
    runner = TestRunner()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "quick":
            success = runner.run_quick_validation()
            sys.exit(0 if success else 1)
        
        elif command == "docs":
            result = runner.run_documentation_tests()
            print(f"Documentation tests: {'PASS' if result['success'] else 'FAIL'}")
            sys.exit(0 if result['success'] else 1)
        
        elif command == "rate-limit":
            result = runner.run_rate_limiting_tests()
            print(f"Rate limiting tests: {'PASS' if result['success'] else 'FAIL'}")
            sys.exit(0 if result['success'] else 1)
        
        elif command == "auth":
            result = runner.run_authentication_tests()
            print(f"Authentication tests: {'PASS' if result['success'] else 'FAIL'}")
            sys.exit(0 if result['success'] else 1)
        
        elif command == "load":
            result = runner.run_load_tests()
            print(f"Load tests: {'PASS' if result['success'] else 'FAIL'}")
            sys.exit(0 if result['success'] else 1)
        
        elif command == "monitoring":
            result = runner.run_monitoring_tests()
            print(f"Monitoring tests: {'PASS' if result['success'] else 'FAIL'}")
            sys.exit(0 if result['success'] else 1)
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: quick, docs, rate-limit, auth, load, monitoring")
            sys.exit(1)
    
    else:
        # Run all tests
        results = runner.run_all_tests()
        runner.print_summary(results)
        
        # Exit with appropriate code
        success = results["summary"]["failed_suites"] == 0
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()