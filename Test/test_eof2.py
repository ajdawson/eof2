"""tests for eof2"""
import unittest
from unittest import TestCase

try:
    # Attempt to import cdms2. This is recommended for eof2 but not required.
    import cdms2
except ImportError:
    pass
import numpy as np

from testutils import generate_data, error, sign_matrix, identify
import eof2


class SolverTestCase(TestCase):

    def __str__(self):
        return 'EofSolver (NumPy) Solver'

    def setUp(self):
        self.sf, self.eofs, self.pcs = generate_data('numpy')
        self.eofobj = eof2.EofSolver(self.sf)

    def test_eofs(self):
        identify('EOFs')
        # Compute EOFs of the scalarfield.
        eofs = self.eofobj.eofs(neofs=2)
        # Coerce the sign of the computed EOFs to match the exact EOFs.
        sign = sign_matrix(self.eofs, eofs, itype='eof')
        eofs = sign * eofs
        # Compute the error in the EOFs.
        err = error(eofs, self.eofs)
        self.assertAlmostEqual(err, 0., places=2)

    def test_eofsAsCorrelation(self):
        identify('EOFs as correlation')
        eofs = self.eofobj.eofsAsCorrelation(neofs=2)
        # Compute reference EOFs.
        l = np.var(self.pcs, axis=0, ddof=1)
        s = np.std(self.sf, axis=0, ddof=1)
        e = s[np.newaxis] * eofs / np.sqrt(l[:, np.newaxis])
        # Coerce the sign of the computed EOFs.
        sign = sign_matrix(self.eofs, e, itype='eof')
        e = sign * e
        # Compute the error in the EOF correlations.
        err = error(e, self.eofs)
        self.assertAlmostEqual(err, 0., places=2)
        
    def test_pcs(self):
        identify('PCs')
        pcs = self.eofobj.pcs(npcs=2)
        # Coerce the sign of the computed PCs to match the exact PCs.
        sign = sign_matrix(self.pcs, pcs, itype='pc')
        pcs = sign * pcs
        # Compute the error in the PCs.
        err = error(pcs, self.pcs)
        self.assertAlmostEqual(err, 0., places=2)

    def test_eigenvalues(self):
        identify('eigenvalues (EOF variances)')
        evals = self.eofobj.eigenvalues(neigs=2)
        ref = np.var(self.pcs, axis=0, ddof=1)
        err = error(evals, ref)
        self.assertAlmostEqual(err, 0., places=2)


@unittest.skipIf('Eof' not in dir(eof2) or 'cdms2' not in dir(),
        'libaray component not available')
class CdmsWrapperTestCase(SolverTestCase):
    """Test the climate data management system wrapper."""

    def __str__(self):
        return 'Eof (cdms2) Solver'

    def setUp(self):
        self.sf, self.eofs, self.pcs = generate_data('cdms')
        self.eofobj = eof2.Eof(self.sf)


if __name__ == '__main__':
    import sys
    from unittest import TestLoader, TestSuite, TextTestRunner
    # Define sets of tests.
    basic_tests = TestLoader().loadTestsFromTestCase(SolverTestCase)
    cdms_tests = TestLoader().loadTestsFromTestCase(CdmsWrapperTestCase)
    # Determine which sets of tests will be run.
    tests_to_run = (
            basic_tests,
            cdms_tests,
    )
    # Create a test suite containing all the required tests.
    all_tests = TestSuite()
    all_tests.addTests(tests_to_run)
    # Run the test suite.
    TextTestRunner(sys.stdout, verbosity=3).run(all_tests)

