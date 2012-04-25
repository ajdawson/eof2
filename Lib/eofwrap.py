"""Meta-data preserving EOF analysis for :py:mod:`cdms2` variables."""
# (c) Copyright 2010-2012 Andrew Dawson. All Rights Reserved.
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

from eofsolve import EofSolver
from tools import weights_array
from errors import EofError, EofToolError


class Eof(object):
    """EOF analysis (meta-data enabled :py:mod:`cdms2` interface).""" 
    
    def __init__(self, dataset, weights='none', center=True, ddof=1):
        """Create an Eof object.
        
        **Argument:**

        *dataset*
            A :py:mod:`cdms2` variable containing the data to be
            analyzed. Time must be the first dimension. Missing values
            are allowed provided that they are constant with time (e.g.,
            values of an oceanographic field over land).
        
        **Optional arguments:**

        *weights*
            Sets the weighting method. The following values are
            accepted:

            * *"area"* : Square-root of grid cell area normalized by
              total area. Requires a latitude-longitude grid to be
              present in the input :py:mod:`cdms2` variable *dataset*.
              This is a fairly standard weighting strategy. If you are
              unsure which method to use and you have gridded data then
              this should be your first choice.

            * *"coslat"* : Square-root of cosine of latitude
              (*"cos_lat"* is also accepted). Requires a latitude
              dimension to be present in the input :py:mod:`cdms2`
              variable *dataset*.

            * *"none"* : Equal weights for all grid points (default).

            * *None* : Same as *"none"*.

             An array of weights may also be supplied instead of
             specifying a weighting method.

        *center*
            If *True*, the mean along the first axis of the input data
            set (the time-mean) will be removed prior to analysis. If
            *False*, the mean along the first axis will not be removed.
            Defaults to *True* (mean is removed). Generally this option
            should be set to *True* as the covariance interpretation
            relies on input data being anomalies with a time-mean of 0.
            A valid reson for turning this off would be if you have
            already generated an anomaly data set. Setting to *True* has
            the useful side-effect of propagating missing values along
            the time-dimension, ensuring the solver will work even if
            missing values occur at different locations at different
            times.

        *ddof*
            'Delta degrees of freedom'. The divisor used to normalize
            the covariance matrix is *N - ddof* where *N* is the
            number of samples. Defaults to *1*.

        **Examples:**

        EOF analysis with area-weighting for the input field:

        >>> from eof2 import Eof
        >>> eofobj = Eof(field, weights="area")

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
        if weights in ("none", None):
            # No weights requested, set the weight array to None.
            wtarray = None
        else:
            try:
                # Generate a weights array of the appropriate kind, with a
                # shape compatible with the data set.
                scheme = weights.lower()
                wtarray = weights_array(dataset, scheme=scheme)
            except AttributeError:
                # Weights is not a string, assume it is an array.
                wtarray = weights
            except EofToolError, err:
                # Weights is not recognized, raise an error.
                raise EofError(err)
        # Cast the wtarray to the same type as the dataset. This prevents the
        # promotion of 32-bit input to 64-bit on multiplication with the
        # weight array when not required. This will fail with a AttributeError
        # exception if the weights array is None, which it may be if no
        # weighting was requested.
        try:
            wtarray = wtarray.astype(dataset.dtype)
        except AttributeError:
            pass
        # Create an EofSolver object using appropriate arguments for this
        # data set. The object will be used for the decomposition and
        # for returning the results.
        self.eofobj = EofSolver(dataset.data, missing=self.missingValue,
                          weights=wtarray, center=center, ddof=ddof)
        
    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs).

        Returns the ordered PCs in a a :py:mod:`cdms2` variable.
        
        **Optional arguments:**

        *pcscaling*
            Set the scaling of the retrieved PCs. The following
            values are accepted:

            * *0* : Un-scaled principal components (default).
            * *1* : Principal components are scaled to unit variance
              (divided by the square-root of their eigenvalue).
            * *2* : Principal components are multiplied by the
              square-root of their eigenvalue.

        *npcs* : Number of principal components to retrieve. Defaults to
          all the principal components.

        **Examples:**

        All un-scaled PCs:

        >>> pcs = eofobj.pcs()

        First 3 PCs scaled to unit variance:

        >>> pcs = eofobj.pcs(npcs=3, pcscaling=1) 

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

        Returns a the ordered EOFs in a :py:mod:`cdms2` variable.
        
        **Optional arguments:**

        *eofscaling*
            Sets the scaling of the EOFs. The following values are
            accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their
              eigenvalues.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalues.
              
        *neofs* -- Number of EOFs to return. Defaults to all EOFs.
        
        **Examples:**

        All EOFs with no scaling:

        >>> eofs = eofobj.eofs()

        First 3 EOFs with scaling applied:

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

        Returns the ordered eigenvalues in a :py:mod:`cdms2` variable.

        **Optional argument:**
        
        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues.
        
        **Examples:**

        All eigenvalues:

        >>> lambdas = eofobj.eigenvalues()

        The first eigenvalue:

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
        """
        EOFs scaled as the correlation of the PCs with the original
        field.
        
        Returns the ordered correlation EOFs in a :py:mod:`cdms2`
        variable.
        
        **Optional argument:**
        
        *neofs*
            Number of EOFs to return. Defaults to all EOFs.
        
        **Note:**
        
        These are only the EOFs expressed as correlation and are not
        related to EOFs computed using the correlation matrix.

        **Examples:**

        All EOFs:

        >>> eofs = eofobj.eofsAsCorrelation()

        The leading EOF:

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
    
    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """
        EOFs scaled as the covariance of the PCs with the original
        field.

        Returns the ordered covariance EOFs in a :py:mod:`cdms2`
        variable.
        
        **Optional arguments:**
        
        *neofs*
            Number of EOFs to return. Defaults to all EOFs.
        
        *pcscaling*
            Set the scaling of the PCs used to compute covariance. The
            following values are accepted:

            * *0* : Un-scaled PCs.
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue) (default).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.

        **Examples:**
        
        All EOFs:

        >>> eofs = eofobj.eofsAsCovariance()

        The leading EOF:

        >>> eof1 = eofobj.eofsAsCovariance(neofs=1)

        The leading EOF using un-scaled PCs:

        >>> eof1 = eofobj.eofsAsCovariance(neofs=1, pcscaling=0)
        
        """
        eofs = self.eofobj.eofsAsCovariance(neofs, pcscaling)
        eofs[numpy.where(numpy.isnan(eofs))] = self.missingValue
        eofax = cdms2.createAxis(range(len(eofs)), id="eof")
        axlist = [eofax] + self.channels
        eofs = cdms2.createVariable(eofs, id="eofs_cov", axes=axlist,
                fill_value=self.missingValue)
        eofs.name = "empirical_orthogonal_functions"
        eofs.long_name = "covariance between principal components and data"
        return eofs
    
    def varianceFraction(self, neigs=None):
        """Fractional EOF variances.
        
        The fraction of the total variance explained by each EOF, a
        value between 0 and 1 inclusive, in a :py:mod:`cdms2` variable.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return the fractional variance for.
            Defaults to all eigenvalues.
        
        **Examples:**

        The fractional variance represented by each eigenvalue:

        >>> varfrac = eofobj.varianceFraction()

        The fractional variance represented by the first 3 eigenvalues:

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
        
        Returns a scalar (not a :py:mod:`cdms2` transient variable).

        **Examples:**

        >>> var = eofobj.totalAnomalyVariance()
        
        """
        return self.eofobj.totalAnomalyVariance()
        
    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the :py:class:`~eof2.Eof` instance
        then the returned reconstructed field will be automatically
        un-weighted. Otherwise the returned reconstructed field will
        be weighted in the same manner as the input to the
        :py:class:`~eof2.Eof` instance.
        
        **Argument:**
        
        *neofs*
            Number of EOFs to use for the reconstruction.
        
        **Examples:**

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
        
        Returns the typical error for each eigenvalue in a
        :py:mod:`cdms2` variable.

        The method of North et al. (1982) is used to compute the typical
        error for each eigenvalue. It is assumed that the number of
        times in the input data set is the same as the number of
        independent realizations. If this assumption is not valid then
        the result may be inappropriate.
        
        **Optional arguments:**
        
        *neigs*
            The number of eigenvalues to return typical errors for.
            Defaults to typical errors for all eigenvalues.
            
        *vfscaled*
            If *True* scale the errors by the sum of the eigenvalues.
            This yields typical errors with the same scale as the
            values returned by the
            :py:meth:`~eof2.Eof.varianceFraction` method. If *False*
            then no scaling is done. Defaults to *False* (no scaling).
        
        **References**

        North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982:
        "Sampling errors in the estimation of empirical orthogonal
        functions", *Monthly Weather Review*, **110**, pages 669-706.
        
        **Examples:**
        
        Typical errors for all eigenvalues:

        >>> errs = eofobj.northTest()

        Typical errors for the first 3 eigenvalues scaled by the sum of
        the eigenvalues:

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
        """Weights used for the analysis.
        
        **Examples:**

        The 2D weights variable used for the analysis:

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
        
        Returns projected time series in a :py:mod:`cdms2` variable.

        Given a field, projects it onto the EOFs to generate a
        corresponding set of time series in a :py:mod:`cdms2` variable.
        The field can be projected onto all the EOFs or just a subset.
        The field must have the same corresponding spatial dimensions
        (including missing values in the same places) as the original
        input to the :py:class:`~eof2.Eof` instance. The field may have
        a different length time dimension to the original input field
        (or no time dimension at all).
        
        **Argument:**
        
        *field*
            A field (:py:mod:`cdms2` variable) to project onto the EOFs.

        **Optional arguments:**

        *neofs*
            Number of EOFs to project onto. Defaults to all EOFs.

        *eofscaling*
            Set the scaling of the EOFs that are projected
            onto. The following values are accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their eigenvalue.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalue.

        *weighted*
            If *True* then the EOFs are weighted prior to projection. If
            *False* then no weighting is applied. Defaults to *True*
            (weighting is applied). Generally only the default setting
            should be used.

        **Examples:**

        Project a field onto all EOFs:

        >>> pcs = eofobj.projectField(field)

        Project fields onto the three leading EOFs:

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

