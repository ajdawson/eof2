"""Test the eof2 package."""
import os
import sys
import unittest

import cdms2
import numpy

import eof2


class TestUtility(object):
    """
    A simple base class for test cases that provides utilitity methods.
    
    """

    def identify(self, name):
        """Print some information identifying a test."""
        print "TESTING [%s]" % name.upper()


class TestEofBase(TestUtility):
    """Base class for EOF tests."""

    def test_eofs(self):
        self.identify("eof maps")
        try:
            eofs = self.eofobj.eofs()
            eofs = self.eofobj.eofs(neofs=2)
            eofs = self.eofobj.eofs(eofscaling=1)
            eofs = self.eofobj.eofs(eofscaling=2)
            eofs = self.eofobj.eofs(eofscaling=2, neofs=1)
        except Exception, err:
            self.fail(err)

    def test_eofsAsCorrelation(self):
        self.identify("eof correlation maps")
        try:
            eofs = self.eofobj.eofsAsCorrelation()
            eofs = self.eofobj.eofsAsCorrelation(neofs=2)
        except Exception, err:
            self.fail(err)

    def test_pcs(self):
        self.identify("principal components")
        try:
            pcs = self.eofobj.pcs()
            pcs = self.eofobj.pcs(npcs=2)
            pcs = self.eofobj.pcs(pcscaling=1)
            pcs = self.eofobj.pcs(pcscaling=2)
            pcs = self.eofobj.pcs(pcscaling=2, npcs=1)
        except Exception, err:
            self.fail(err)

    def test_eigenvalues(self):
        self.identify("eigenvalues")
        try:
            eigs = self.eofobj.eigenvalues()
            eigs = self.eofobj.eigenvalues(neigs=2)
        except Exception, err:
            self.fail(err)

    def test_varianceFraction(self):
        self.identify("variance fraction")
        try:
            var = self.eofobj.varianceFraction()
            var = self.eofobj.varianceFraction(neigs=2)
        except Exception, err:
            self.fail(err)
            
    def test_totalAnomalyVariance(self):
        self.identify("total anomaly variance")
        try:
            var = self.eofobj.totalAnomalyVariance()
        except Exception, err:
            self.fail(err)

    def test_reconstructedField(self):
        self.identify("reconstructed field")
        try:
            rfield = self.eofobj.reconstructedField(neofs=2)
        except Exception, err:
            self.fail(err)

    def test_northTest(self):
        self.identify("north et al. typical errors")
        try:
            errs = self.eofobj.northTest()
            errs = self.eofobj.northTest(vfscaled=True)
            errs = self.eofobj.northTest(vfscaled=True, neigs=2)
        except Exception, err:
            self.fail(err)


class TestEofNumPy(TestEofBase, unittest.TestCase):
    """Test the EofNumPy object on its own."""

    def __str__(self):
        return "EOF TEST [NUMPY]           "

    def _init_scalar_field(self):
        """Create a time-evolving scalar field to do analysis on."""
        x = numpy.arange(0, 6.5, 0.5)
        y = x
        X, Y = numpy.meshgrid(x, y)
        h1 = 12. * (1.2 - 0.35 * numpy.sqrt((X-3) * (X-3) + \
                                            (Y-3) * (Y-3)))
        h1 = 1012. + h1 - numpy.mean(h1)
        h2 = 1022.8 - 3.6 * Y
        h3 = 1001.2 + 3.6 * X
        h1b = 1012. + (1012. - h1)
        h2b = 1012. + (1012. - h2)
        h3b = 1012. + (1012. - h3)
        H = numpy.zeros((6, 13, 13))
        H[0] = h1
        H[1] = h1b
        H[2] = h2
        H[3] = h2b
        H[4] = h3
        H[5] = h3b
        return H

    def setUp(self):
        """Initialize an EofNumPy object with a scalar field."""
        sf = self._init_scalar_field()
        self.eofobj = eof2.EofNumPy(sf)


class TestEof(TestEofBase, unittest.TestCase):
    """Test the Eof object with default arguments"""

    def __str__(self):
        return "EOF TEST [CDMS WRAPPER]    "

    def setUp(self):
        """Initialize an Eof object."""
        fpath = os.path.join(sys.prefix, "sample_data", "clt.nc")
        sf = cdms2.open(fpath)("clt")
        self.eofobj = eof2.Eof(sf)
    

class TestEofWeighted(TestEofBase, unittest.TestCase):
    """Test the Eof object with weighting."""

    def __str__(self):
        return "EOF TEST [WEIGHTED EOFS]   "

    def setUp(self):
        """Initialize an Eof object specifying default weighting."""
        fpath = os.path.join(sys.prefix, "sample_data", "clt.nc")
        sf = cdms2.open(fpath)("clt")
        self.eofobj = eof2.Eof(sf, weights="default")


class TestEofUncentered(TestEofBase, unittest.TestCase):
    """Test the Eof object with centering turned off."""

    def __str__(self):
        return "EOF TEST [UN-CENTERED EOFS]"

    def setUp(self):
        """Initialize an Eof object specifying not to remove the mean."""
        fpath = os.path.join(sys.prefix, "sample_data", "clt.nc")
        sf = cdms2.open(fpath)("clt")
        self.eofobj = eof2.Eof(sf, center=False, weights="none")


class TestEofInvalid(TestUtility, unittest.TestCase):
    """Test the Eof object with invalid input/combinations of options."""

    def __str__(self):
        return "INPUT TEST [INVALID]       "

    def setUp(self):
        pass

    def test_invalid_options(self):
        self.identify("invalid options")
        fpath = os.path.join(sys.prefix, "sample_data", "clt.nc")
        sf = cdms2.open(fpath)("clt", longitude=(0,0), squeeze=1)
        self.failUnlessRaises(eof2.EofError, eof2.Eof, sf,
                weights="area")

    def test_invalid_type(self):
        self.identify("invalid type")
        fpath = os.path.join(sys.prefix, "sample_data", "clt.nc")
        sf = cdms2.open(fpath)("clt").data
        self.failUnlessRaises(eof2.EofError, eof2.Eof, sf,
                weights="area")
    
    def test_invalid_time(self):
        self.identify("invalid time axis position")
        fpath = os.path.join(sys.prefix, "sample_data", "clt.nc")
        sf = cdms2.open(fpath)("clt").reorder("xyt")
        self.failUnlessRaises(eof2.EofError, eof2.Eof, sf,
                weights="area")

    def test_invalid_order(self):
        self.identify("invalid axis order with weighting")
        fpath = os.path.join(sys.prefix, "sample_data", "clt.nc")
        sf = cdms2.open(fpath)("u").reorder("txzy")
        self.failUnlessRaises(eof2.EofError, eof2.Eof, sf,
                weights="area")


if __name__ == "__main__":
    from unittest import TestLoader, TestSuite, TextTestRunner

    # Create test suites for each case.
    numpy_tests = TestLoader().loadTestsFromTestCase(TestEofNumPy)
    cdms_tests = TestLoader().loadTestsFromTestCase(TestEof)
    weighted_tests = TestLoader().loadTestsFromTestCase(TestEofWeighted)
    uncentered_tests = TestLoader().loadTestsFromTestCase(TestEofUncentered)
    invalid_tests = TestLoader().loadTestsFromTestCase(TestEofInvalid)
    
    # Define a list of the test suites that should be run.
    tests_to_run = (
            numpy_tests,
            cdms_tests,
            weighted_tests,
            uncentered_tests,
            invalid_tests,
    )

    # Create a suite containing all the individual tests.
    all_tests = TestSuite()
    all_tests.addTests(tests_to_run)

    # Run the tests.
    TextTestRunner(sys.stdout, verbosity=3).run(all_tests)

