"""Run the test suite for :py:mod:`eof2`."""
import sys
from unittest import TestSuite, TestLoader, TextTestRunner

import solver_tests
import tools_tests


if __name__ == '__main__':
    tests_to_run = (
            TestLoader().loadTestsFromModule(solver_tests),
            TestLoader().loadTestsFromModule(tools_tests),
    )
    all_tests = TestSuite()
    all_tests.addTests(tests_to_run)
    TextTestRunner(sys.stdout, verbosity=3).run(all_tests)

