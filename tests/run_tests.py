# -*- coding: utf-8 -*-
"""
Master Test Runner.
Executes all test suites across Normalization, Evaluation, Leakage Detection,
and Dialect Slang Edge Cases using standard Python unittest.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run_all_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=str(Path(__file__).parent), pattern="test_*.py")

    print("\n" + "=" * 80)
    print("  RUNNING ARABIC PAYMENTS EVALUATION TEST SUITE")
    print("=" * 80)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print("=" * 80)
    if result.wasSuccessful():
        print(f"  [SUCCESS] All {result.testsRun} tests passed successfully with 0 errors!")
        print("=" * 80 + "\n")
        return 0
    else:
        print(f"  [FAILURE] {len(result.failures)} failures, {len(result.errors)} errors detected.")
        print("=" * 80 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
