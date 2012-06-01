"""
Tests for tools in :py:mod:`eof2.nptools` and :py:mod:`eof2.tools`.

"""
import unittest
from unittest import TestCase

try:
    import cdms2
except ImportError:
    pass
import eof2

from testutils import generate_data, error, sign_matrix, identify


class ToolsTestCase(TestCase):
    """Functionality of extra tools (:py:mod:`numpy` interface)."""

    def __str__(self):
        return 'validating extra tools (numpy interface)'

    def setUp(self):
        self.sf, self.eofs, self.pcs = generate_data('numpy')
        self.eofobj = eof2.EofSolver(self.sf)
        self.covmap = eof2.nptools.covariance_map
        self.cormap = eof2.nptools.correlation_map

    def test_covariance(self):
        identify('covariance maps')
        # Get the standardized leading PC.
        spc1 = self.eofobj.pcs(npcs=1, pcscaling=1)
        # Compute covariance map using extra tools.
        excov = self.covmap(spc1, self.sf)
        # Compute the leading EOF expressed as covariance.
        incov = self.eofobj.eofsAsCovariance(neofs=1)
        # Compute the error.
        err = error(excov, incov)
        self.assertAlmostEqual(err, 0., places=2)

    def test_correlation(self):
        identify('correlation maps')
        # Get the standardized leading PC.
        spc1 = self.eofobj.pcs(npcs=1, pcscaling=1)
        # Compute covariance map using extra tools.
        excor = self.cormap(spc1, self.sf)
        # Compute the leading EOF expressed as covariance.
        incor = self.eofobj.eofsAsCorrelation(neofs=1)
        # Compute the error.
        err = error(excor, incor)
        self.assertAlmostEqual(err, 0., places=2)


@unittest.skipIf('Eof' not in dir(eof2) or 'cdms2' not in dir(),
        'library component not available')
class ToolsMetaDataTestCase(ToolsTestCase):
    """Functionality of extra tools (:py:mod:`cdms2` interface)."""

    def __str__(self):
        return 'validating extra tools (cdms2 interface)'

    def setUp(self):
        self.sf, self.eofs, self.pcs = generate_data('cdms')
        self.eofobj = eof2.Eof(self.sf)
        self.covmap = eof2.tools.covariance_map
        self.cormap = eof2.tools.correlation_map


@unittest.skipIf('Eof' not in dir(eof2) or 'cdms2' not in dir(),
        'library component not available')
class ToolsCrossInterfaceTestCase(TestCase):
    """Functionality of extra tools across interfaces."""

    def __str__(self):
        return "validating extra tools (cross-interfaces)"

    def setUp(self):
        self.sf, self.eofs, self.pcs = generate_data('cdms')
        self.eofobj = eof2.Eof(self.sf)
        self.covmap = eof2.tools.covariance_map
        self.cormap = eof2.tools.correlation_map

    def test_covariance(self):
        identify("covariance maps")
        spc1 = self.eofobj.pcs(npcs=1, pcscaling=1)
        npcov = eof2.nptools.covariance_map(spc1, self.sf)
        mdcov = eof2.tools.covariance_map(spc1, self.sf)
        err = error(mdcov, npcov)
        self.assertAlmostEqual(err, 0., places=2)

    def test_correlation(self):
        identify("correlation maps")
        spc1 = self.eofobj.pcs(npcs=1, pcscaling=1)
        npcor = eof2.nptools.correlation_map(spc1, self.sf)
        mdcor = eof2.tools.correlation_map(spc1, self.sf)
        err = error(mdcor, npcor)
        self.assertAlmostEqual(err, 0., places=2)

