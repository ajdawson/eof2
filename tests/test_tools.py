"""
"""
from nose import SkipTest
from nose.tools import raises, assert_almost_equal, assert_true
import numpy as np

from eof2 import EofSolver, EofError
import eof2.nptools
try:
    from eof2 import Eof
    import eof2.tools
    import cdms2
except ImportError:
    pass
from reference import reference_solution
from utils import error


class TestToolsStandard(object):

    @classmethod
    def setup_class(cls):
        # Generate a scalar field and a corresponding EOF solution for the
        # standard numpy array interface.
        cls.sf, cls.eofs, cls.pcs = reference_solution('numpy')
        # Create an EofSolver instance with the scalar field.
        cls.eofobj = EofSolver(cls.sf)
        cls.tools = {'covmap': eof2.nptools.covariance_map,
                'cormap': eof2.nptools.correlation_map}

    def test_covariancemap(self):
        """Covariance map matches EOFs as covariance?"""
        spc1 = self.eofobj.pcs(npcs=1, pcscaling=1)
        excov = self.tools['covmap'](spc1, self.sf)
        incov = self.eofobj.eofsAsCovariance(neofs=1)
        err = error(excov, incov)
        assert_almost_equal(err, 0.)

    def test_correlationmap(self):
        """Correlation map matches EOFs as correlation?"""
        spc1 = self.eofobj.pcs(npcs=1, pcscaling=1)
        excor = self.tools['cormap'](spc1, self.sf)
        incor = self.eofobj.eofsAsCorrelation(neofs=1)
        err = error(excor, incor)
        assert_almost_equal(err, 0.)
        maxval = np.abs(excor).max()
        assert_true(maxval <= 1.000000001)


class TestToolsMetaData(TestToolsStandard):
    
    @classmethod
    def setup_class(cls):
        # Generate a scalar field and a corresponding EOF solution for the
        # cdms2 interface.
        try:
            cls.sf, cls.eofs, cls.pcs = reference_solution('cdms2')
        except ValueError:
            raise SkipTest('library component not available')
        # Create an Eof instance with the scalar field.
        cls.eofobj = Eof(cls.sf)
        cls.tools = {'covmap': eof2.tools.covariance_map,
                'cormap': eof2.tools.correlation_map}


class TestToolsCrossInterface(object):
     
    @classmethod
    def setup_class(cls):
        # Generate a scalar field and a corresponding EOF solution for the
        # cdms2 interface.
        try:
            cls.sf, cls.eofs, cls.pcs = reference_solution('cdms2')
        except ValueError:
            raise SkipTest('library component not available')
        # Create an Eof instance with the scalar field.
        cls.eofobj = Eof(cls.sf)

    def test_covariancemap(self):
        """Covariance maps match across interfaces?"""
        spc1 = self.eofobj.pcs(npcs=1, pcscaling=1)
        npcov = eof2.nptools.covariance_map(spc1, self.sf)
        mdcov = eof2.tools.covariance_map(spc1, self.sf)
        err = error(npcov, mdcov)
        assert_almost_equal(err, 0.)

    def test_correlationmap(self):
        """Correlation maps match across interfaces?"""
        spc1 = self.eofobj.pcs(npcs=1, pcscaling=1)
        npcor = eof2.nptools.correlation_map(spc1, self.sf)
        mdcor = eof2.tools.correlation_map(spc1, self.sf)
        err = error(npcor, mdcor)
        assert_almost_equal(err, 0.)
        maxval = np.abs(npcor).max()
        assert_true(maxval <= 1.000000001)

