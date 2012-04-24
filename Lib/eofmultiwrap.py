"""
Meta-data preserving multiple EOF analysis for :py:mod:`cdms2`
variables.

"""
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

from eofmultisolve import MultipleEofSolver
from tools import weights_array
from errors import EofError, EofToolError


class MultipleEof(object):
    """Multiple EOF analysis (meta-data enabled `cdms2` interface)."""

    def __init__(self, *datasets, **kwargs):
        """Create a MultipleEof object.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**

        *\*datasets*
            One or more :py:mod:`cdms2` variables containing the data to
            be analyzed. Time must be the first dimension of each
            variable. Missing values are allowed provided that they are
            constant with time (e.g., values of an oceanographic field
            over land).
        
        **Optional arguments:**

        *weights*
            Sets the weighting method. The following values are
            accepted:

            * *'area'* : Square-root of grid cell area normalized by
              total area. Required a latitude-longitude grid to be
              present in the corresponding :py:mod:`cdms2` variable.
              This is a fairly standard weighting strategy. If you are
              unsure which method to use and you have gridded data then
              this should be your first choice.

            * *'coslat'* : Square-root of cosine of latitude
              (*'cos_lat'* is also accepted).

            * *'none'* : Equal weights for all grid points (default).

            * *None* : Same as *'none'*.

             A sequence of values may be passed to use different
             weighting for each data set. Arrays of weights may also
             be supplied instead of specifying a weighting method.

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

        EOF analysis with area-weighting for the input fields:

        >>> from eof2 import Eof
        >>> eofobj = Eof(field1, field2, weights="area")

        """
        # Handle keyword arguments manually.
        keywords = {'weights': 'none', 'center': True, 'ddof':1}
        for kwarg in kwargs:
            if kwarg not in keywords.keys():
                raise EofError('Invalid argument: %s.' % kwarg)
        weights = kwargs.get('weights', keywords['weights'])
        center = kwargs.get('center', keywords['center'])
        ddof = kwargs.get('ddof', keywords['ddof'])
        # Record the number of datasets.
        self._numdsets = len(datasets)
        # Ensure the weights are specified one per dataset.
        if weights in ('none', None, 'area', 'cos_lat', 'coslat'):
            weights = [weights] * self._numdsets
        elif len(weights) != self._numdsets:
            raise EofError('Number of weights and data sets differ.')
        # Record dimension information, missing values and compute weights.
        self._multitimeaxes = list()
        self._multichannels = list()
        self._multimissing = list()
        passweights = list()
        for dataset, weight in zip(datasets, weights):
            if not cdms2.isVariable(dataset):
                raise EofError('The input dataset must be a cdms2 variable.')
            # Ensure a time dimension exists.
            timeaxis = dataset.getTime()
            if timeaxis is None:
                raise EofError('Time axis missing.')
            self._multitimeaxes.append(timeaxis)
            # Ensure the time dimension is the first dimension.
            order = dataset.getOrder()
            if order[0] != 't':
                raise EofError('Time must be the first dimension.')
            # Record the other dimensions.
            channels = dataset.getAxisList()
            channels.remove(timeaxis)
            if len(channels) < 1:
                raise EofError('One or more spatial dimensions are required.')
            self._multichannels.append(channels)
            # Record the missing values.
            self._multimissing.append(dataset.getMissing())
            # Compute weights as required.
            if weight in ('none', None):
                passweights.append(None)
            else:
                try:
                    wtarray = weights_array(dataset, scheme=weight.lower())
                    passweights.append(wtarray)
                except AttributeError:
                    # Weight specification is not a string. Assume it is an array
                    # of weights.
                    passweights.append(weight)
                except EofToolError, err:
                    # Another error occured, raise it as an EOF error.
                    raise EofError(err)
        # Define a time axis as the time axis of the first dataset.
        self.timeax = self._multitimeaxes[0]
        # Create a MultipleEofSolver to do the computations.
        self.eofobj = MultipleEofSolver(*[d.data for d in datasets],
                missing=self._multimissing, weights=passweights,
                center=center, ddof=ddof)
            
    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs).

        Returns the ordered principal component time series expansions.
        
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
        """Empirical orthogonal functions (EOFs).
        
        Returns variables with the ordered EOFs along the first axis.
        
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
        eofset = self.eofobj.eofs(eofscaling, neofs)
        neofs = eofset[0].shape[0]
        eofax = cdms2.createAxis(range(neofs), id='eof')
        varset = list()
        for iset in xrange(self._numdsets):
            axlist = [eofax] + self._multichannels[iset]
            eofset[iset][numpy.where(numpy.isnan(eofset[iset]))] = self._multimissing[iset]
            eofset[iset] = cdms2.createVariable(eofset[iset], id='eofs', axes=axlist,
                    fill_value=self._multimissing[iset])
            eofset[iset].name = 'empirical_orthogonal_functions'
            eofset[iset].long_name = 'empirical orthogonal functions'
        return eofset

    def eigenvalues(self, neigs=None):
        """
        Eigenvalues (decreasing variances) associated with each EOF.
        each EOF.

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
        EOFs scaled as the correlation of the PCs with original field.
        
        **Optional argument:**
        
        *neofs*
            Number of EOFs to return. Defaults to all EOFs.
        
        **Examples:**

        All EOFs:

        >>> eofs = eofobj.eofsAsCorrelation()

        The leading EOF:

        >>> eof1 = eofobj.eofsAsCorrelation(neofs=1)
        
        """
        eofset = self.eofobj.eofsAsCorrelation(neofs)
        neofs = eofset[0].shape[0]
        eofax = cdms2.createAxis(range(neofs), id='eof')
        for iset in xrange(self._numdsets):
            axlist = [eofax] + self._multichannels[iset]
            eofset[iset][numpy.where(numpy.isnan(eofset[iset]))] = self._multimissing[iset]
            eofset[iset] = cdms2.createVariable(eofset[iset], id='eofs_corr', axes=axlist,
                    fill_value=self._multimissing[iset])
            eofset[iset].name = 'empirical_orthogonal_functions'
            eofset[iset].long_name = 'correlation between principal components and data'
        return eofset

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """
        EOFs scaled as the covariance of the PCs with original field.

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
        eofset = self.eofobj.eofsAsCovariance(neofs)
        neofs = eofset[0].shape[0]
        eofax = cdms2.createAxis(range(neofs), id='eof')
        for iset in xrange(self._numdsets):
            axlist = [eofax] + self._multichannels[iset]
            eofset[iset][numpy.where(numpy.isnan(eofset[iset]))] = self._multimissing[iset]
            eofset[iset] = cdms2.createVariable(eofset[iset], id='eofs_cov', axes=axlist,
                    fill_value=self._multimissing[iset])
            eofset[iset].name = 'empirical_orthogonal_functions'
            eofset[iset].long_name = 'covariance between principal components and data'
        return eofset

    def varianceFraction(self, neigs=None):
        """Fractional EOF variances.
        
        The fraction of the total variance explained by each EOF. This
        is a value between 0 and 1 inclusive.

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
        
        Returns a scalar (not a :py:mod:`cdms2` variable).

        **Examples:**
        
        >>> var = eofobj.totalAnomalyVariance()
        
        """
        return self.eofobj.totalAnomalyVariance()

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the :py:class:`~eof2.MultipleEof`
        instance then the returned reconstructed field will be
        automatically un-weighted. Otherwise the returned reconstructed
        field will  be weighted in the same manner as the input to the
        :py:class:`~eof2.EofSolver` instance.
        
        **Argument:**
        
        *neofs*
            Number of EOFs to use for the reconstruction.
        
        **Examples:**

        Reconstruct the input fields using 3 EOFs.

        >>> rfields = eofobj.reconstructedField(neofs=3)
        
        """
        rfset = self.eofobj.reconstructedField(neofs)
        for iset in xrange(self._numdsets):
            axlist = [self.timeax] + self._multichannels[iset]
            rfset[iset][numpy.where(numpy.isnan(rfset[iset]))] = self._multimissing[iset]
            rfset[iset] = cdms2.createVariable(rfset[iset], id='rcon', axes=axlist,
                    fill_value=self._multimissing[iset])
            rfset[iset].long_name = 'reconstructed_field'
        return rfset

    def northTest(self, neigs=None, vfscaled=False):
        """Typical errors for eigenvalues.
        
        Uses the method of North et al. (1982) to compute the typical
        error for each eigenvalue. It is assumed that the number of
        times in the input data set is the same as the number of
        independent realizations. If this assumption is not valid then
        the results of this method may be inappropriate.
        
        **Optional arguments:**
        
        *neigs*
            The number of eigenvalues to return typical errors for.
            Defaults to typical errors for all eigenvalues.
            
        *vfscaled*
            If *True* scale the errors by the sum of the eigenvalues.
            This yields typical errors with the same scale as the
            values returned by the
            :py:method:`~eof2.MultipleEofSolver.varianceFraction`
            method. If *False* then no scaling is done. Defaults to
            *False* (no scaling).
        
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

    def projectField(self, *fields, **kwargs):
        """Project a set of fields onto the EOFs.
        
        Given a set of fields, projects them onto the EOFs to generate
        a corresponding set of time series. Fields can be projected onto
        all the EOFs or just a subset. There must be the same number of 
        fields as were originally input into the
        :py:class:`~eof2.MultipleEofSolver` instance, and each field
        must have the same corresponding spatial dimensions (including
        missing values in the same places). The fields may have a
        different length time dimension to the original input fields (or
        no time dimension at all).
        
        **Argument:**
        
        *\*fields*
            One or more fields to project onto the EOFs. Must
            be the same as the number of fields used to initialize the
            `MultipleEofSolver`.

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
            If *True* then the EOFs are weighted prior to
            projection. If False the no weighting is applied. Defaults to
            True (weighting is applied). Generally only the default setting
            should be used.

        **Examples:**

        Project fields onto all EOFs:

        >>> pcs = eofobj.projectField(*fields)

        Project fields onto the three leading EOFs:

        >>> pcs = eofobj.projectField(*fields, , neofs=3)
        
        """
        multimissing = list()
        multitimeaxes = list()
        for field in fields:
            multimissing.append(field.getMissing())
            multitimeaxes.append(field.getTime())
        if None in multitimeaxes:
            notime = True
        else:
            notime = False
        pcs = self.eofobj.projectField([f.data for f in fields],
                missing=multimissing, neofs=neofs, eofscaling=eofscaling,
                weighted=weighted, notime=notime)
        # Create an axis list, its contents depend on whether or not a time
        # axis was present in the input field.
        if notime:
            # No time axis, just use a PC axis.
            pcsax = cdms2.createAxis(range(pcs.shape[0]), id="pc")
            axlist = [pcsax]
        else:
            # A PC axis and a leading time axis.
            pcsax = cdms2.createAxis(range(pcs.shape[1]), id="pc")
            axlist = [fields[0].getTime(), pcsax]
        # Apply meta data to the projected PCs.
        pcs = cdms2.createVariable(pcs, id="pcs", axes=axlist)
        pcs.name = "principal_components"
        pcs.long_name = "principal component time series"
        return pcs


if __name__ == '__main__':
    pass

