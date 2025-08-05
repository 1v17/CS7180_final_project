#!/usr/bin/env python3
"""
Script to run all tests in the tests folder except demo_scheduler.py
Usage: cd tests && python run_tests.py
       or from project root: python tests/run_tests.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def main():
    print("Running all tests except demo_scheduler.py...")
    print("=" * 50)
    
    # Get the directory containing this script (which is the tests directory)
    tests_dir = Path(__file__).parent
    script_dir = tests_dir.parent
    
    # Find all test files except demo_scheduler.py
    test_files = []
    for test_file in tests_dir.glob("test_*.py"):
        if test_file.name != "demo_scheduler.py":
            test_files.append(test_file)
    
    total_tests = len(test_files)
    passed_tests = 0
    failed_tests = 0
    
    print(f"Found {total_tests} test files to run:\n")
    
    for test_file in test_files:
        test_module = f"tests.{test_file.stem}"
        print(f"Running: {test_file.name}")
        print("-" * 30)
        
        try:
            # Run the test using unittest with proper encoding
            result = subprocess.run([
                sys.executable, "-m", "unittest", test_module
            ], capture_output=True, text=True, cwd=script_dir, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                print(f"✓ PASSED: {test_file.name}")
                passed_tests += 1
            else:
                print(f"✗ FAILED: {test_file.name}")
                if result.stderr:
                    print("STDERR:", result.stderr)
                if result.stdout:
                    print("STDOUT:", result.stdout)
                failed_tests += 1
                
        except Exception as e:
            print(f"✗ ERROR running {test_file.name}: {e}")
            failed_tests += 1
        
        print()
    
    # Summary
    print("=" * 50)
    print("TEST SUMMARY")
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("\nAll tests passed! 🎉")
        return 0
    else:
        print("\nSome tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
