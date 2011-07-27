"""Wrapper for using cdms2 variables with the EofNumPy class."""
# (c) Copyright 2010, 2011 Andrew Dawson. All Rights Reserved.
# 
# This file is part of eof2.
# 
# eof2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# eof2 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
# 
# You should have received a copy of the GNU General Public License
# along with eof2.  If not, see <http://www.gnu.org/licenses/>.
import cdms2
import numpy

from eofsolve import EofNumPy
from errors import EofError


class Eof(object):
    """EOF analysis cdms wrapper.
    
    EOF analysis of cdms2 variables. Outputs have sensible meta-data.
    
    """
    
    def __init__(self, dataset, weights='none', center=True, ddof=1):
        """Create an Eof object.
        
        Required argument:
        dataset -- cdms2 transient variable containing the data to be
            decomposed. Time must be the first dimension. Missing values
            are allowed provided that they are constant with time (ie.
            values of an oceanographic variable over land).
        
        Optional arguments:
        weights -- Sets the weighting method. Defaults to no weighting.
            The following weighting parameters are valid:

            'area'    - Square-root of grid cell area normalized by
                        total area. This is a standard weighting and
                        should be used if you are unsure.

            'cos_lat' - Square-root of cosine of latitude.

            'none'    - Equal weights for all grid points.

        center -- If True the mean along the first axis of the input
            data set (the time-mean) will be removed prior to analysis.
            If False the mean along the first axis will not be removed.
            This option should only be set to False if you know what you
            are doing and why. Defaults to True (mean is removed).
            the mean.
        ddof -- 'Delta degrees of freedom'. The divisor used to
            normalize the covariance matrix is N - ddof where N is the
            number of samples. Defaults to 1.

        Example:
        >>> from eof2 import Eof
        >>> eofobj = Eof(data, weights="default")

        """
        # Check that dataset is recognised by cdms2 as a variable.
        if not cdms2.isVariable(dataset):
            raise EofError("The input data must be a cdms2 variable.")
        # Store the time axis as an instance variable.
        self.timeax = dataset.getTime()
        # Verify that a time axis was found, getTime returns None when a
        # time axis is not found.
        if self.timeax is None:
            raise EofError("Time axis missing.")
        # Check the dimension order of the input, time must be the first
        # dimension.
        order = dataset.getOrder()
        if order[0] != "t":
            raise EofError("Time must be the first dimension.")
        # Verify the presence of at least one spatial dimension. The
        # instance variable channels will also be used as a partial axis
        # list when constructing meta-data. It contains the spatial
        # dimensions.
        self.channels = dataset.getAxisList()
        self.channels.remove(self.timeax)
        if len(self.channels) < 1:
            raise EofError("One or more spatial dimensions are required.")
        # Store the missing value attribute of the data set in an
        # instance variable so that it is recoverable later.
        self.missingValue = dataset.getMissing()
        # Extract the grid from the input cdms2 variable.
        grid = dataset.getGrid()
        # Either use the grid to compute appropriate weights or leave
        # unweighted. This decision depends on user input and the
        # presence of a grid in the input data.
        if grid is None or weights == "none":
            # Either no weighting was requested or no grid weights
            # are available.
            if weights != "none":
                # If weightting was requested but no grid is available then
                # raise an error.
                raise EofError("Automatic weighting requires a grid.")
            else:
                # Set wtarray to None, the appropriate argument for the
                # EofNumPy object when no weighting is required.
                wtarray = None
        else:
            # If automatic weighting needs to be done then we must check that
            # latitude and longitude are the right-most dimensions of the
            # input data. This allows the use of an array broadcast to create
            # a weights array. We should also determine if the latitude and
            # longitude axes are in reversed order ("xy"). If so this must be
            # taken into account when creating the weights array.
            if "x" in order and "y" in order:
                if order[-1] not in "xy" or order[-2] not in "xy":
                    raise EofError("Latitude and longitude must be the right-most dimensions when weighting is required.")
                xyreversed = order.index("x") < order.index("y") or False
            # Get weights from the grid.
            latw, lonw = grid.getWeights()
            # Modify the weights depending on the type of weighting
            # requested.
            if weights == "area":
                # Area weights are returned from the grid as default so no
                # modification is required.
                pass
            elif weights == "cos_lat" or weights == "default":
                # Take the square-root to transform the weights for the
                # 'cos_lat' setting ('default' is recognized for backwards
                # compatibility.
                # The weights are scaled such that the largest weight in
                # each of the latitude and longitude weight arrays is scaled
                # to unity.
                latw = latw / numpy.maximum.reduce(latw)
                lonw = lonw / numpy.maximum.reduce(lonw)
                latw = numpy.sqrt(latw)  # sqrt(cos(lat))
            else:
                # If the value of the 'weights' optional argument is not
                # recognised then an error is thrown alerting the user.
                raise EofError("invalid weights option: %s" % repr(weights))
            # Construct a 2d weight array by taking the outer product of
            # the latitude and longitude weights. This array is the same
            # shape as the latitude x longitude part of the input data.
            if xyreversed:
                wtarray = numpy.outer(lonw, latw)
            else:
                wtarray = numpy.outer(latw, lonw)
            if weights == "area":
                # If area weighting is specified then we normalize the
                # Computed weight array by its total area and take the
                # square root.
                wtarray /= wtarray.sum()
                wtarray = numpy.sqrt(wtarray)
            # Cast the wtarray to numpy.float32. This prevents the promotion
            # of 32-bit input to 64-bit on multiplication with the weight
            # array.
            wtarray = wtarray.astype(numpy.float32)
        # Create an EofNumpy object using appropriate arguments for this
        # data set. The object will be used for the decomposition and
        # for returning the results.
        self.eofobj = EofNumPy(dataset.data, missing=self.missingValue,
                          weights=wtarray, center=center, ddof=ddof)
        
    def pcs(self, pcscaling=0, npcs=None):
        """Principal components.
        
        Optional arguments:
        pcscaling -- Sets the scaling of the principal components. The
            following values are accepted:
            0 - Un-scaled principal components.
            1 - Principal components are divided by the square-root of
                their eigenvalues. This results in PCs with unit
                variance.
            2 - Principal components are multiplied by the square-root
                of their eigenvalues.
            Defaults to 0 (un-scaled principal components).
        npcs -- Number of principal components to return. Defaults to
            all principal components.

        Example 1:
        >>> pcs = eofobj.pcs() # All PCs, un-scaled.

        Example 2:
        >>> pcs = eofobj.pcs(npcs=3, pcscaling=1) # First 3 PCs, scaled.

        """
        pcs = self.eofobj.pcs(pcscaling, npcs)
        pcsax = cdms2.createAxis(range(pcs.shape[1]), id="pc")
        axlist = [self.timeax, pcsax]
        pcs = cdms2.createVariable(pcs, id="pcs", axes=axlist)
        pcs.name = "principal_components"
        pcs.long_name = "principal component time series"
        return pcs
    
    def eofs(self, eofscaling=0, neofs=None):
        """Emipirical orthogonal functions.
        
        Optional arguments:
        eofscaling -- Sets the scaling of the EOFs. The following values
            are accepted:
            0 - Un-scaled EOFs.
            1 - EOFs are divided by the square-root of their eigenvalues
            2 - EOFs are multiplied by the square-root of their
                eigenvalues.
            Defaults to 0 (un-scaled EOFs).
        neofs -- Number of EOFs to return. Defaults to all EOFs.

        Example 1:
        >>> # All EOFs, un-scaled.
        >>> eofs = eofobj.eofs()

        Example 2:
        >>> # First 3 EOFs, scaled.
        >>> eofs = eofobj.eofs(neofs=3, eofscaling=1)

        """
        eofs = self.eofobj.eofs(eofscaling, neofs)
        eofs[numpy.where(numpy.isnan(eofs))] = self.missingValue
        eofax = cdms2.createAxis(range(len(eofs)), id="eof")
        axlist = [eofax] + self.channels
        eofs = cdms2.createVariable(eofs, id="eofs", axes=axlist)
        eofs.name = "empirical_orthogonal_functions"
        eofs.long_name = "empirical orthogonal functions"
        eofs.setMissing(self.missingValue)
        return eofs
    
    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EOF.

        Optional argument:
        neigs -- Number of eigenvalues to return. Defaults to all
            eigenvalues.

        Example 1:
        >>> lambdas = eofobj.eigenvalues()

        Example 2:
        >>> lambda1 = eofobj.eigenvalues(neigs=1)
        
        """
        lambdas = self.eofobj.eigenvalues(neigs=neigs)
        eofax = cdms2.createAxis(range(len(lambdas)), id="eigenvalue")
        axlist = [eofax]
        lambdas = cdms2.createVariable(lambdas, id="eigenvalues", axes=axlist)
        lambdas.name = "eigenvalues"
        lambdas.long_name = "eigenvalues"
        return lambdas
        
    def eofsAsCorrelation(self, neofs=None):
        """EOFs scaled as the correlation of the PCs with orginal field.

        Optional argument:
        neofs -- Number of EOFs to return. Defaults to all EOFs.

        Example 1:
        >>> eofs = eofobj.eofsAsCorrelation()

        Example 2:
        >>> eof1 = eofobj.eofsAsCorrelation(neofs=1)
        
        """
        eofs = self.eofobj.eofsAsCorrelation(neofs)
        eofs[numpy.where(numpy.isnan(eofs))] = self.missingValue
        eofax = cdms2.createAxis(range(len(eofs)), id="eof")
        axlist = [eofax] + self.channels
        eofs = cdms2.createVariable(eofs, id="eofs_corr", axes=axlist)
        eofs.name = "empirical_orthogonal_functions"
        eofs.long_name = "correlation between principal components and data"
        eofs.setMissing(self.missingValue)
        return eofs
    
    def varianceFraction(self, neigs=None):
        """Fraction of the total variance explained by each mode.

        Optional argument:
        neigs -- Number of eigenvalues to return the fractional variance
            for. Defaults to all eigenvalues.

        Example 1:
        Get the fractional variance represented by each eigenvalue.
        >>> varfrac = eofobj.varianceFraction()

        Example 2:
        Get the fractional variance represented by the first 3
        eigenvalues.
        >>> varfrac = eofobj.VarianceFraction(neigs=3)
        
        """
        vfrac = self.eofobj.varianceFraction(neigs=neigs)
        eofax = cdms2.createAxis(range(len(vfrac)), id="eigenvalue")
        axlist = [eofax]
        vfrac = cdms2.createVariable(vfrac, id="variance", axes=axlist)
        vfrac.name = "variance_fraction"
        vfrac.long_name = "variance fraction"
        return vfrac
        
    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).
        
        Returns a scalar (not a cdms2 transient variable).

        Example:
        Get the total variance (sum of the eigenvalues).
        >>> var = eofobj.totalAnomalyVariance()
        
        """
        return self.eofobj.totalAnomalyVariance()
        
    def reconstructedField(self, neofs):
        """Reconstructed anomaly field based on a subset of EOFs.

        If automatic weighting was performed by the Eof object then the
        returned reconstructed field will be automatically un-weighted.
        Otherwise the returned reconstructed field will  be weighted in
        the same manner as the input to the Eof object.

        Argument:
        neofs -- Number of EOFs to use for the reconstruction.

        Example:
        Reconstruct the input field using 3 EOFs.
        >>> rfield = eofobj.reconstructedField(neofs=3)
        
        """
        rfield = self.eofobj.reconstructedField(neofs)
        rfield[numpy.where(numpy.isnan(rfield))] = self.missingValue
        axlist = [self.timeax] + self.channels
        rfield = cdms2.createVariable(rfield, id="rcon", axes=axlist)
        rfield.long_name = "reconstructed_field"
        rfield.setMissing(self.missingValue)
        return rfield
    
    def northTest(self, neigs=None, vfscaled=False):
        """Typical errors for eigenvalues.
        
        Uses the method of North et al. (1982) to compute the typical
        error for each eigenvalue. It is assumed that the number of
        times in the input data set is the same as the number of
        independent realizations. If this assumption is not valid then
        the results of this method may be inappropriate.
        
        Optional arguments:
        neigs -- Number of eigenvalues to return typical errors for.
            Defaults to typical errors for all eigenvalues.
        vfscaled -- If True scale the errors by the sum of the
            eigenvalues. This yields typical errors with the same scale
            as the values returned by the 'varianceFraction' method. If
            False then no scaling is done. Defaults to False (no
            scaling.)
        
        North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982,
            Sampling errors in the estimation of empirical orthogonal
            functions, Monthly Weather Review, 110, pages 669-706.

        Example 1:
        Get typical errors for all eigenvalues.
        >>> errs = eofobj.northTest()

        Example 2:
        Get typical errors for the first 3 eigenvalues and scale them
        by the sum of the eigenvalues.
        >>> errs = eofobj.northTest(neigs=3, vfscaled=True)
        
        """
        typerrs = self.eofobj.northTest(neigs=neigs, vfscaled=vfscaled)
        eofax = cdms2.createAxis(range(len(typerrs)), id="eigenvalue")
        axlist = [eofax]
        typerrs = cdms2.createVariable(typerrs, id="typical_errors", axes=axlist)
        typerrs.name = "typical_errors"
        typerrs.long_name = "north_typical_errors"
        return typerrs


if __name__ == "__main__":
    pass

