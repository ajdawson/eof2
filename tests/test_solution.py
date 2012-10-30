"""Test :py:mod:`eof2` computations against reference solutions."""
from nose import SkipTest
from nose.tools import assert_almost_equal, assert_true
import numpy as np

from eof2 import EofSolver
try:
    from eof2 import Eof
except ImportError:
    pass
from utils import error, sign_adjustments
from reference import reference_solution


class TestSolutionStandard(object):

    @classmethod
    def setup_class(cls):
        # Generate a scalar field and a corresponding EOF solution for the
        # standard numpy array interface.
        cls.sf, cls.eofs, cls.pcs = reference_solution('numpy')
        cls.neofs = cls.eofs.shape[0]
        # Create an EofSolver instance with the scalar field.
        cls.eofobj = EofSolver(cls.sf)
    
    def test_eofs(self):
        """EOFs match the reference solution?"""
        # Compute the EOFs using the solver.
        eofs = self.eofobj.eofs(neofs=self.neofs)
        # Coerce the sign of the EOFs so that they match the sign of the 
        # reference solution.
        eofs *= sign_adjustments(eofs, self.eofs)
        err = error(eofs, self.eofs)
        assert_almost_equal(err, 0.)
    
    def test_eofsascorrelation(self):
        """
        EOFs as correlation between PCs and input field match the
        reference solution?
        
        """
        # Compute the EOFs using the solver.
        eofs = self.eofobj.eofsAsCorrelation(neofs=self.neofs)
        # Adjust the EOFs so they are scaled as normal EOFs:
        #   c_m = e_m l_m^(0.5) / s
        # where c_m is the correlation between the m'th PC and the input
        # fields, e_m is the m'th EOF, l_m is the m'th eigenvalue and s is the
        # vector of standard deviations of the input fields (see Wilks, 2006,
        # Statistical Methods in the Atmospheric Sciences).
        l = np.var(self.pcs, axis=0, ddof=1)
        s = np.std(self.sf, axis=0, ddof=1)
        e = s[np.newaxis] * eofs / np.sqrt(l[:, np.newaxis])
        e *= sign_adjustments(e, self.eofs)
        err = error(e, self.eofs)
        assert_almost_equal(err, 0.)
        maxval = np.abs(eofs).max()
        assert_true(maxval <= 1.000000001)
    
    def test_eofsascovariance(self):
        """
        EOFs as covariance between PCs and input field match the
        reference solution?
        
        """
        # Compute the EOFs using the solver.
        eofs = self.eofobj.eofsAsCovariance(neofs=self.neofs)
        # Adjust the EOFs so they are scaled as normal EOFs:
        #   c_m = e_m l_m^(0.5) / s
        # where c_m is the correlation between the m'th PC and the input
        # fields, e_m is the m'th EOF, l_m is the m'th eigenvalue and s is the
        # vector of standard deviations of the input fields (see Wilks, 2006,
        # Statistical Methods in the Atmospheric Sciences). Since these are
        # covariance fields there is no need to multiply by the standard
        # deviation.
        l = np.var(self.pcs, axis=0, ddof=1)
        e = eofs / np.sqrt(l[:, np.newaxis])
        e *= sign_adjustments(e, self.eofs)
        err = error(e, self.eofs)
        assert_almost_equal(err, 0.)

    def test_pcs(self):
        """PCs match the reference solution?"""
        pcs = self.eofobj.pcs(npcs=self.neofs)
        pcs *= sign_adjustments(pcs.T, self.pcs.T).T
        err = error(pcs, self.pcs)
        assert_almost_equal(err, 0., places=2)
        # Check that cross correlations are zero.
        cpcs = np.corrcoef(pcs[:, 0], pcs[:, 1])[0,1]
        assert_almost_equal(cpcs, 0.)
        
    def test_eigenvalues(self):
        """Eigenvalues match PC variance from the reference solution?"""
        evals = self.eofobj.eigenvalues(neigs=self.neofs)
        ref = np.var(self.pcs, axis=0, ddof=1)
        err = error(evals, ref)
        assert_almost_equal(err, 0.)
        
    def test_projectfield(self):
        """Projected PCs match computed PCs?"""
        pf = self.eofobj.projectField(self.sf)
        pcs = self.eofobj.pcs()
        err = error(pf, pcs)
        assert_almost_equal(err, 0., places=2)
        
    def test_reconstructedfield(self):
        """Reconstructed field matches reference solution?"""
        rf = self.eofobj.reconstructedField(self.neofs)
        err = error(rf, self.sf)
        assert_almost_equal(err, 0., places=2)


class TestSolutionMetaData(TestSolutionStandard):
    
    @classmethod
    def setup_class(cls):
        # Generate a scalar field and a corresponding EOF solution for the
        # cdms2 interface.
        try:
            cls.sf, cls.eofs, cls.pcs = reference_solution('cdms2')
        except ValueError:
            raise SkipTest('library component not available')
        cls.neofs = cls.eofs.shape[0]
        # Create an Eof instance with the scalar field.
        cls.eofobj = Eof(cls.sf)
    
