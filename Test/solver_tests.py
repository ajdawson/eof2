"""
Tests for EOF solvers :py:class:`eof2.EofSolver` and
:py:class:`eof2.Eof`.

"""
import unittest
from unittest import TestCase

try:
    import cdms2
except ImportError:
    pass
import numpy as np
import eof2

from testutils import generate_data, error, sign_matrix, identify


class SolverTestCase(TestCase):
    """Functionality of the :py:mod:`numpy` solver interface."""

    def __str__(self):
        return 'validating solution (numpy interface)'

    def setUp(self):
        self.sf, self.eofs, self.pcs = generate_data('numpy')
        self.eofobj = eof2.EofSolver(self.sf, ddof=0)

    def test_eofs(self):
        identify('eigenvectors')
        eofs = self.eofobj.eofs(neofs=2)
        # Coerce the sign of the computed EOFs to match the exact EOFs and
        # compute the error.
        signs = sign_matrix(self.eofs, eofs, transpose=False)
        eofs = eofs * signs[:, np.newaxis]
        err = error(eofs, self.eofs)
        self.assertAlmostEqual(err, 0., places=8)

    def test_eofsAsCorrelation(self):
        identify('eigenvectors (correlation)')
        eofs = self.eofobj.eofsAsCorrelation(neofs=2)
        # Convert these EOFs to the spatial EOF patterns. This is done using
        # the fact that correlation between the data and nth PC is given by:
        #   e_m l_m^(1/2) / s
        # where e_m is the m'th eigenvector (EOF), l is the m'th eigenvalue,
        # and s is the vector of standard deviations of the original
        # data set (see Wilks 2006).
        l = np.var(self.pcs, axis=0, ddof=1)
        s = np.std(self.sf, axis=0, ddof=1)
        e = s[np.newaxis] * eofs / np.sqrt(l[:, np.newaxis])
        # Coerce the sign of the computed EOFs to match the exact EOFs and
        # compute the error.
        signs = sign_matrix(self.eofs, e, transpose=False)
        e = e * signs[:, np.newaxis]
        err = error(e, self.eofs)
        self.assertAlmostEqual(err, 0., places=8)
        # Also verify that the range of the correlation coefficient is valid
        # (within acceptable tolerance).
        self.assertTrue((np.abs(eofs) <= 1.000000001).all())
        
    def test_pcs(self):
        identify('principal components')
        pcs = self.eofobj.pcs(npcs=2, pcscaling=0)
        # Coerce the sign of the computed PCs to match the exact PCs and
        # compute the error.
        sign = sign_matrix(self.pcs, pcs, transpose=True)
        pcs = sign * pcs
        err = error(pcs, self.pcs)
        self.assertAlmostEqual(err, 0., places=2)
        # PCs should be correlated with the exact PCs.
        c0 = np.corrcoef(pcs[:,0], self.pcs[:,0])[0,1]
        c1 = np.corrcoef(pcs[:,1], self.pcs[:,1])[0,1]
        self.assertAlmostEquals(c0, 1, places=8)
        self.assertAlmostEquals(c1, 1, places=8)
        # PCs should be un-correlated in time.
        c2 = np.corrcoef(pcs[:,0], pcs[:,1])[0,1]
        self.assertAlmostEquals(c2, 0, places=8)

    def test_eigenvalues(self):
        identify('eigenvalues')
        evals = self.eofobj.eigenvalues(neigs=2)
        ref = np.var(self.pcs, axis=0, ddof=0)
        err = error(evals, ref)
        self.assertAlmostEqual(err, 0., places=8)

    def test_reconstruction(self):
        identify("field reconstruction")
        rf = self.eofobj.reconstructedField(2)
        err = error(rf, self.sf)
        self.assertAlmostEqual(err, 0., places=2)

    def test_projection(self):
        identify("field projection")
        pf = self.eofobj.projectField(self.sf)
        pcs = self.eofobj.pcs()
        err = error(pf, pcs)
        self.assertAlmostEqual(err, 0., places=4)

    def test_projection_single(self):
        identify("single field projection")
        pf = self.eofobj.projectField(self.sf[12])
        pcs = self.eofobj.pcs()[12]
        err = error(pf, pcs)
        self.assertAlmostEqual(err, 0., places=3)


@unittest.skipIf('Eof' not in dir(eof2) or 'cdms2' not in dir(),
        'library component not available')
class SolverMetaDataTestCase(SolverTestCase):
    """Functionality of the :py:mod:`cdms2` enabled solver interface."""

    def __str__(self):
        return 'validating solution (cdms2 interface)'

    def setUp(self):
        self.sf, self.eofs, self.pcs = generate_data('cdms')
        self.eofobj = eof2.Eof(self.sf, weights="none", ddof=0)

