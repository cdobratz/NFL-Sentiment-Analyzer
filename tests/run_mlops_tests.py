#!/usr/bin/env python3
"""
MLOps test runner script.
Runs comprehensive MLOps validation and testing suite.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import time
from datetime import datetime


class MLOpsTestRunner:
    """Test runner for MLOps validation and testing."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Test categories and their corresponding files
        self.test_categories = {
            "model_validation": [
                "tests/test_mlops_validation.py::TestModelValidation",
            ],
            "deployment": [
                "tests/test_mlops_deployment.py::TestModelDeploymentStrategies",
                "tests/test_mlops_deployment.py::TestABTestDeployment",
                "tests/test_mlops_deployment.py::TestModelRollback",
                "tests/test_mlops_deployment.py::TestDeploymentHealthMonitoring",
            ],
            "data_drift": [
                "tests/test_mlops_data_drift.py::TestDataDriftDetection",
                "tests/test_mlops_data_drift.py::TestPerformanceMonitoring",
                "tests/test_mlops_data_drift.py::TestRetrainingTriggers",
            ],
            "integration": [
                "tests/test_mlops_integration.py::TestMLOpsEndToEndWorkflow",
                "tests/test_mlops_integration.py::TestMLOpsServiceIntegration",
                "tests/test_mlops_integration.py::TestMLOpsErrorHandling",
            ],
            "pipeline_components": [
                "tests/test_mlops_validation.py::TestModelDeploymentAndRollback",
                "tests/test_mlops_validation.py::TestDataDriftDetection",
                "tests/test_mlops_validation.py::TestMLOpsPipelineIntegration",
            ]
        }
    
    def run_test_category(self, category: str, verbose: bool = False) -> Dict[str, Any]:
        """Run tests for a specific category."""
        if category not in self.test_categories:
            raise ValueError(f"Unknown test category: {category}")
        
        print(f"\n{'='*60}")
        print(f"Running {category.upper()} tests")
        print(f"{'='*60}")
        
        test_files = self.test_categories[category]
        category_results = {
            "category": category,
            "tests": [],
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0,
            "success": True
        }
        
        start_time = time.time()
        
        for test_file in test_files:
            print(f"\nRunning: {test_file}")
            result = self._run_pytest(test_file, verbose)
            category_results["tests"].append(result)
            
            # Aggregate results
            category_results["total_tests"] += result.get("total", 0)
            category_results["passed"] += result.get("passed", 0)
            category_results["failed"] += result.get("failed", 0)
            category_results["skipped"] += result.get("skipped", 0)
            
            if result.get("failed", 0) > 0:
                category_results["success"] = False
        
        category_results["duration"] = time.time() - start_time
        
        # Print category summary
        self._print_category_summary(category_results)
        
        return category_results
    
    def run_all_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all MLOps tests."""
        print("Starting comprehensive MLOps test suite...")
        print(f"Test execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        all_results = {
            "execution_time": datetime.now().isoformat(),
            "categories": {},
            "summary": {
                "total_categories": len(self.test_categories),
                "successful_categories": 0,
                "failed_categories": 0,
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "total_skipped": 0,
                "overall_success": True,
                "duration": 0
            }
        }
        
        # Run each test category
        for category in self.test_categories.keys():
            try:
                category_result = self.run_test_category(category, verbose)
                all_results["categories"][category] = category_result
                
                # Update summary
                if category_result["success"]:
                    all_results["summary"]["successful_categories"] += 1
                else:
                    all_results["summary"]["failed_categories"] += 1
                    all_results["summary"]["overall_success"] = False
                
                all_results["summary"]["total_tests"] += category_result["total_tests"]
                all_results["summary"]["total_passed"] += category_result["passed"]
                all_results["summary"]["total_failed"] += category_result["failed"]
                all_results["summary"]["total_skipped"] += category_result["skipped"]
                
            except Exception as e:
                print(f"Error running {category} tests: {e}")
                all_results["summary"]["failed_categories"] += 1
                all_results["summary"]["overall_success"] = False
        
        self.end_time = time.time()
        all_results["summary"]["duration"] = self.end_time - self.start_time
        
        # Print final summary
        self._print_final_summary(all_results)
        
        # Save results to file
        self._save_results(all_results)
        
        return all_results
    
    def run_specific_tests(self, test_patterns: List[str], verbose: bool = False) -> Dict[str, Any]:
        """Run specific tests matching the given patterns."""
        print(f"Running specific tests: {', '.join(test_patterns)}")
        
        results = {
            "patterns": test_patterns,
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "success": True,
                "duration": 0
            }
        }
        
        start_time = time.time()
        
        for pattern in test_patterns:
            print(f"\nRunning tests matching: {pattern}")
            result = self._run_pytest(pattern, verbose)
            results["tests"].append(result)
            
            # Aggregate results
            results["summary"]["total_tests"] += result.get("total", 0)
            results["summary"]["passed"] += result.get("passed", 0)
            results["summary"]["failed"] += result.get("failed", 0)
            results["summary"]["skipped"] += result.get("skipped", 0)
            
            if result.get("failed", 0) > 0:
                results["summary"]["success"] = False
        
        results["summary"]["duration"] = time.time() - start_time
        
        return results
    
    def _run_pytest(self, test_path: str, verbose: bool = False) -> Dict[str, Any]:
        """Run pytest for a specific test path."""
        cmd = ["python", "-m", "pytest", test_path, "--tb=short"]
        
        if verbose:
            cmd.append("-v")
        
        # Add coverage if available
        cmd.extend(["--cov=app/services/mlops", "--cov-report=term-missing"])
        
        # Add markers
        cmd.extend(["-m", "not external"])  # Skip tests requiring external services
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test file
            )
            
            # Parse pytest output
            output_lines = result.stdout.split('\n')
            
            # Extract test results from output
            test_result = {
                "test_path": test_path,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "success": result.returncode == 0
            }
            
            # Parse test counts from pytest output
            for line in output_lines:
                if "passed" in line or "failed" in line or "skipped" in line:
                    # Look for pattern like "5 passed, 2 failed, 1 skipped"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            test_result["passed"] = int(parts[i-1])
                        elif part == "failed" and i > 0:
                            test_result["failed"] = int(parts[i-1])
                        elif part == "skipped" and i > 0:
                            test_result["skipped"] = int(parts[i-1])
            
            test_result["total"] = test_result["passed"] + test_result["failed"] + test_result["skipped"]
            
            return test_result
            
        except subprocess.TimeoutExpired:
            return {
                "test_path": test_path,
                "exit_code": -1,
                "error": "Test execution timed out",
                "total": 0,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "success": False
            }
        except Exception as e:
            return {
                "test_path": test_path,
                "exit_code": -1,
                "error": str(e),
                "total": 0,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "success": False
            }
    
    def _print_category_summary(self, results: Dict[str, Any]):
        """Print summary for a test category."""
        print(f"\n{'-'*40}")
        print(f"Category: {results['category'].upper()}")
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']}")
        print(f"Failed: {results['failed']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Duration: {results['duration']:.2f}s")
        print(f"Status: {'✅ PASSED' if results['success'] else '❌ FAILED'}")
        print(f"{'-'*40}")
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final test execution summary."""
        summary = results["summary"]
        
        print(f"\n{'='*60}")
        print("MLOPS TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Execution Time: {results['execution_time']}")
        print(f"Total Duration: {summary['duration']:.2f}s")
        print(f"")
        print(f"Categories:")
        print(f"  Total: {summary['total_categories']}")
        print(f"  Successful: {summary['successful_categories']}")
        print(f"  Failed: {summary['failed_categories']}")
        print(f"")
        print(f"Tests:")
        print(f"  Total: {summary['total_tests']}")
        print(f"  Passed: {summary['total_passed']}")
        print(f"  Failed: {summary['total_failed']}")
        print(f"  Skipped: {summary['total_skipped']}")
        print(f"")
        
        if summary["overall_success"]:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED!")
            print("\nFailed Categories:")
            for category, result in results["categories"].items():
                if not result["success"]:
                    print(f"  - {category}: {result['failed']} failed tests")
        
        print(f"{'='*60}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"mlops_test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nTest results saved to: {results_file}")
        except Exception as e:
            print(f"Failed to save results: {e}")
    
    def validate_environment(self) -> bool:
        """Validate that the test environment is properly set up."""
        print("Validating test environment...")
        
        # Check if required packages are installed
        required_packages = [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "numpy",
            "pandas",
            "scikit-learn"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        # Check if test files exist
        test_files = [
            "tests/test_mlops_validation.py",
            "tests/test_mlops_deployment.py",
            "tests/test_mlops_data_drift.py",
            "tests/test_mlops_integration.py"
        ]
        
        missing_files = []
        for test_file in test_files:
            if not Path(test_file).exists():
                missing_files.append(test_file)
        
        if missing_files:
            print(f"❌ Missing test files: {', '.join(missing_files)}")
            return False
        
        print("✅ Test environment validation passed!")
        return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="MLOps Test Runner")
    parser.add_argument(
        "--category",
        choices=["model_validation", "deployment", "data_drift", "integration", "pipeline_components"],
        help="Run tests for a specific category"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all MLOps tests"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        help="Run specific test patterns"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--validate-env",
        action="store_true",
        help="Validate test environment only"
    )
    
    args = parser.parse_args()
    
    runner = MLOpsTestRunner()
    
    # Validate environment first
    if not runner.validate_environment():
        sys.exit(1)
    
    if args.validate_env:
        print("Environment validation completed successfully!")
        sys.exit(0)
    
    try:
        if args.all:
            results = runner.run_all_tests(verbose=args.verbose)
            sys.exit(0 if results["summary"]["overall_success"] else 1)
        
        elif args.category:
            results = runner.run_test_category(args.category, verbose=args.verbose)
            sys.exit(0 if results["success"] else 1)
        
        elif args.tests:
            results = runner.run_specific_tests(args.tests, verbose=args.verbose)
            sys.exit(0 if results["summary"]["success"] else 1)
        
        else:
            parser.print_help()
            print("\nExample usage:")
            print("  python tests/run_mlops_tests.py --all")
            print("  python tests/run_mlops_tests.py --category model_validation")
            print("  python tests/run_mlops_tests.py --tests tests/test_mlops_validation.py::TestModelValidation")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during test execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()