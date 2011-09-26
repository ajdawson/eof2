"""wrapper for using cdms2 variables with the EofNumPy class"""
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
import warnings

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
        # Generate an appropriate set of weights for the input dataset. There
        # are several weighting schemes. The 'area' weighting scheme requires
        # a latitude-longitude grid to be present, the 'cos_lat' scheme only
        # requires a latitude dimension.
        if weights == "none":
            # No weights requested, set the weight array to None.
            wtarray = None
        elif weights == "area":
            # Area weighting is requested, retrieve the grid weights from the
            # dataset and compute appropriate area weights.
            grid = dataset.getGrid()
            try:
                latw, lonw = grid.getWeights()
            except AttributeError:
                raise EofError("Automatic weighting with '%s' requires a grid." % weights)
            # Check that the last two dimensions are latitude and longitude
            # and if they are reversed.
            if order[-2:] not in ("xy", "yx"):
                raise EofError("'%s' weighting scheme requires latitude and longitude are the last dimensions." % weights)
            # Create the weights array from the latitude and longitude
            # weights.
            if order[-2:] == "yx":
                # Latitude before longitude.
                wtarray = numpy.outer(latw, lonw)
            else:
                # Longitude before latitude.
                wtarray = numpy.outer(lonw, latw)
            # Normalize by the sum of the weights and take the square-root.
            # These weights produce the same results as the Climate Data
            # Operators version 1.5.0.
            wtarray /= wtarray.sum()
            wtarray = numpy.sqrt(wtarray)
        elif weights == "cos_lat":
            # Square-root of cosine of latitude weights are requested, compute
            # the latitude weights from the dataset's latitude dimension. Get
            # the latitude values and compute the square-root of the cosine of
            # the latitude values.
            try:
                latvals = dataset.getLatitude()[:]
                latw = numpy.sqrt(numpy.cos(numpy.deg2rad(latvals)))
            except TypeError, AttributeError:
                raise EofError("'%s' weighting scheme requires a latitude dimension." % weights)
            # If 90 or -90 are in the latitude dimension then inaccurate
            # floating point representations may cause the weight to be NaN,
            # since the cosine of 90+dx or -90-dx will be negative, and hence
            # cannot be square-rooted. To safeguard against this we replace
            # NaNs with 0, which is the correct value for this case.
            latw[numpy.where(numpy.isnan(latw))] = 0.
            # The shape of the weights array depends on the position of the
            # latitude dimension and the presence of a longitude dimension.
            if "x" in order:
                # A longitude dimension is present, make sure latitude and
                # longitude are the last two dimensions.
                if order[-2:] not in ("xy", "yx"):
                    raise EofError("'%s' weighting scheme requires latitude and longitude are the last dimensions." % weights)
                # Find how many longitude values are present and create an
                # array of ones the same shape.
                nlons = len(dataset.getLongitude()[:])
                lonw = numpy.ones(nlons)
                # Create the weights array from the latitude and longitude
                # weights.
                if order[-2:] == "yx":
                    # Latitude before longitude.
                    wtarray = numpy.outer(latw, lonw)
                else:
                    # Longitude before latitude.
                    wtarray = numpy.outer(lonw, latw)
            else:
                # A longitude dimension is not present, make sure latitude is
                # the last dimension.
                if order[-1] != "y":
                    # Latitude must be last when no longitude.
                    raise EofError("'%s' weighting scheme requires latitude and longitude are the last dimensions." % weights)
                # Just use a weight array the same dimensionality as the
                # latitude dimension.
                wtarray = latw
        elif weights == "default":
            # The deprecated 'default' weighting scheme is requested. Issue a
            # warning and then produce weights with this normalization.
            warnings.warn("'default' weighting scheme is deprecated, use 'cos_lat'")
            # Retrieve the grid weights from the dataset and compute
            # appropriate area weights.
            grid = dataset.getGrid()
            try:
                latw, lonw = grid.getWeights
            except AttributeError:
                raise EofError("Automatic weighting with '%s' requires a grid." % weights)
            latw /= numpy.maximum.reduce(latw)
            lonw /= numpy.maximum.reduce(lonw)
            latw = numpy.sqrt(latw)
            # Check that the last two dimensions are latitude and longitude
            # and if they are reversed.
            if order[-2:] not in ("xy", "yx"):
                raise EofError("'%s' weighting scheme requires latitude and longitude are the last dimensions." % weights)
            if order[-2:] == "yx":
                wtarray = numpy.outer(latw, lonw)
            else:
                wtarray = numpy.outer(lonw, latw)
        # Cast the wtarray to numpy.float32. This prevents the promotion of
        # 32-bit input to 64-bit on multiplication with the weight array, this
        # will fail with a AttributeError exception if the weights array is
        # None, which it may be if no weighting was requested.
        try:
            wtarray = wtarray.astype(numpy.float32)
        except AttributeError:
            pass
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
        eofs = cdms2.createVariable(eofs, id="eofs", axes=axlist,
                fill_value=self.missingValue)
        eofs.name = "empirical_orthogonal_functions"
        eofs.long_name = "empirical orthogonal functions"
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
        eofs = cdms2.createVariable(eofs, id="eofs_corr", axes=axlist,
                fill_value=self.missingValue)
        eofs.name = "empirical_orthogonal_functions"
        eofs.long_name = "correlation between principal components and data"
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
        rfield = cdms2.createVariable(rfield, id="rcon", axes=axlist,
                fill_value=self.missingValue)
        rfield.long_name = "reconstructed_field"
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

    def getWeights(self):
        """Return the weights used for the analysis.
        
        Example:
        Get the 2D weights variable used for the analysis.
        >>> wgt = eofobj.getWeights()

        """
        weights = self.eofobj.getWeights()
        if weights is not None:
            axlist = self.channels[-2:]
            weights = cdms2.createVariable(weights, id="weights", axes=axlist)
            weights.name = "weights"
            weights.long_name = "grid_weights"
        return weights

    def projectField(self, field, neofs=None, eofscaling=0, weighted=True):
        """Project a field onto the EOFs.

        Argument:
        field -- A spatial or spatial-temporal field to project onto
            EOFs.

        Optional arguments:
        neofs -- Number of EOFs to return. Defaults to all EOFs.
        eofscaling -- Sets the scaling of the EOFs. The following values
            are accepted:
            0 - Un-scaled EOFs.
            1 - EOFs are divided by the square-root of their eigenvalues
            2 - EOFs are multiplied by the square-root of their
                eigenvalues.
            Defaults to 0 (un-scaled EOFs).
        weighted -- If True then weights are applied to the EOFs prior
            to the projection. If False then the EOFs are not weighted.
            Defaults to True (weighting is applied), this is the setting
            that should be used in most circumstances.       
        
        Example 1:
        Project a field onto all EOFs.
        >>> pcs = eofobj.projectField(field)

        Example 2:
        Project a field onto the three leading EOFs.
        >>> pcs = eofobj.projectField(field, neofs=3)
        
        """
        # Check to see if a time dimension is present in the input field. A
        # time dimension is not required but does need to be accounted for.
        if field.getTime() is None:
            notime = True
        else:
            notime = False
        # Compute the projected PCs.
        pcs = self.eofobj.projectField(field, missing=field.getMissing(),
                neofs=neofs, eofscaling=eofscaling, weighted=weighted,
                notime=notime)
        # Create an axis list, its contents depend on whether or not a time
        # axis was present in the input field.
        if notime:
            # No time axis, just use a PC axis.
            pcsax = cdms2.createAxis(range(pcs.shape[0]), id="pc")
            axlist = [pcsax]
        else:
            # A PC axis and a leading time axis.
            pcsax = cdms2.createAxis(range(pcs.shape[1]), id="pc")
            axlist = [field.getTime(), pcsax]
        # Apply meta data to the projected PCs.
        pcs = cdms2.createVariable(pcs, id="pcs", axes=axlist)
        pcs.name = "principal_components"
        pcs.long_name = "principal component time series"
        return pcs
    

if __name__ == "__main__":
    pass

