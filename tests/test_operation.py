"""Test the operation of :py:mod:`eof2`.

These tests do not check for valid solutions, just for the correct
operation of the code.

"""
from nose import SkipTest
from nose.tools import raises, assert_true
import numpy as np

from eof2 import EofSolver, EofError
try:
    from eof2 import Eof
    import cdms2
except ImportError:
    pass
from reference import reference_solution


class TestOperationStandard(object):
    
    @classmethod
    def setup_class(cls):
        # Generate a scalar field and a corresponding EOF solution for the
        # standard numpy array interface.
        cls.sf, cls.eofs, cls.pcs = reference_solution('numpy')
        cls.neofs = cls.eofs.shape[0]
        # Create an EofSolver instance with the scalar field.
        cls.eofobj = EofSolver(cls.sf)
    
    def test_eofs_eofscaling_1(self):
        """EOFs divided by square-root of eigenvalue"""
        eofs1 = self.eofobj.eofs(neofs=self.neofs, eofscaling=1)

    def test_eofs_eofscaling_2(self):
        """EOFs multiplied by square-root of eigenvalue"""
        eofs2 = self.eofobj.eofs(neofs=self.neofs, eofscaling=2)
    
    def test_pcs_pcscaling_1(self):
        """PCs scaled to unit variance"""
        pcs1 = self.eofobj.pcs(npcs=self.neofs, pcscaling=1)
    
    def test_pcs_pcscaling_2(self):
        """PCs multiplied by square-root of eigenvalue"""
        pcs2 = self.eofobj.pcs(npcs=self.neofs, pcscaling=2)

    def test_variancefraction(self):
        """Variance fraction"""
        vfrac = self.eofobj.varianceFraction()

    def test_totalanomalyvariance(self):
        """Total variance"""
        var = self.eofobj.totalAnomalyVariance()

    def test_northtest_standard(self):
        """North's test"""
        errs = self.eofobj.northTest()

    def test_northtest_scaled(self):
        """North's test scaled to variance fraction"""
        errs = self.eofobj.northTest(vfscaled=True)

    def test_projectfield_full_shape(self):
        """Projection onto EOFs of field with same dimensions/shape."""
        pf = self.eofobj.projectField(self.sf, neofs=self.neofs)
        required_shape = (self.sf.shape[0], self.neofs)
        assert_true(pf.shape==required_shape)
    
    def test_projectfield_partial_shape(self):
        """Projection onto EOFs of field with different length time dimension."""
        pf = self.eofobj.projectField(self.sf[0:1], neofs=self.neofs)
        required_shape = (1, self.neofs)
        assert_true(pf.shape==required_shape)
    
    def test_projectfield_no_time(self):
        """Projection onto EOFs of field with no time dimension."""
        pf = self.eofobj.projectField(self.sf[0], neofs=self.neofs)
        required_shape = (self.neofs,)
        assert_true(pf.shape==required_shape)
    
    @raises(EofError)
    def test_projectfield_dimensions_invalid(self):
        """Projection onto EOFs of field with incorrect dimensionality."""
        pf = self.eofobj.projectField(self.sf[..., np.newaxis])
    
    @raises(EofError)
    def test_projectfield_shape_invalid(self):
        """Projection onto EOFs of field with incorrect shape."""
        pf = self.eofobj.projectField(self.sf[..., 1:])


class TestOperationMetaData(TestOperationStandard):
    
    @classmethod
    def setup_class(cls):
        # Generate a scalar field and a corresponding EOF solution for the
        # standard numpy array interface.
        cls.sf, cls.eofs, cls.pcs = reference_solution('cdms2')
        cls.neofs = cls.eofs.shape[0]
        # Create an Eof for the scalar field.
        cls.eofobj = Eof(cls.sf)
    
    @raises(EofError)
    def test_projectfield_dimensions_invalid(self):
        """Projection onto EOFs of field with incorrect dimensionality."""
        levax = cdms2.createAxis([0], id='level')
        var = cdms2.createVariable(
                cdms2.MV.reshape(self.sf, self.sf.shape+(1,)),
                axes=self.sf.getAxisList()+[levax], id=self.sf.id)
        pf = self.eofobj.projectField(var)
    
    def test_projectfield_cdms2_no_time(self):
        """
        Projection onto EOFs of variable with time dimension that will
        not be recognized by cdms2.
        
        """
        time = cdms2.createAxis(self.sf.getTime()[:], id='t')
        var = cdms2.createVariable(self.sf, id=self.sf.id)
        var.setAxis(0, time)
        pf = self.eofobj.projectField(var, neofs=self.neofs)
        required_shape = (self.sf.shape[0], self.neofs)
        assert_true(pf.shape==required_shape)

