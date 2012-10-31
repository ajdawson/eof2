"""Multiple EOF analysis for :py:mod:`numpy` array data."""
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
import numpy as np
import numpy.ma as ma

from eofsolve import EofSolver
from errors import EofError


class MultipleEofSolver(object):
    """Multiple EOF analysis (:py:mod:`numpy` interface)."""

    def __init__(self, *datasets, **kwargs): 
        """Create a MultipleEofSolver object.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.
        
        **Arguments:**

        *\*datasets*
            One or more :py:class:`numpy.ndarray`s or
            :py:class:`numpy.ma.core.MasekdArray`s with two or more
            dimensions containing the data to be analysed. The first
            dimension of each array is assumed to represent time.
            Missing values are permitted, either in the form of masked
            arrays, or the value :py:attr:`numpy.nan`. Missing values
            must be constant with time (e.g., values of an oceanographic
            field over land).

        **Optional arguments:**

        *weights*
            A sequence of arrays of weights whose shapes are compatible
            with those of the input data sets. The weights can have the
            same shape as the input data set or a shape compatible with
            a an array broadcast operation (ie. the shape of the weights
            can match the rightmost parts of the shape of the input data
            set). If none of the input data sets require weighting then
            the single value *None* may be used. Defaults to *None* (no
            weighting for any data set).

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

        """

        # Define valid keyword arguments and their default values. This method
        # is required since Python 2.7 cannot accept a variable argument list
        # followed by a set of keyword arguments. For some reason both must be
        # variable.
        keywords = {"weights": None, "center": True, "ddof": 1}
        for kwarg in kwargs.keys():
            if kwarg not in keywords.keys():
                raise EofError("invalid argument: %s" % kwarg)
        weights = kwargs.get("weights", keywords["weights"])
        center = kwargs.get("center", keywords["center"])
        ddof = kwargs.get("ddof", keywords["ddof"])
        # Record the number of datasets provided.
        self._ndatasets = len(datasets)
        # Initialise instance variables dealing with dataset shapes.
        self._multirecords = list()
        self._multishapes = list()
        self._multislicers = list()
        self._multichannels = list()
        self._multidtypes = list()
        slicebegin = 0
        for dataset in datasets:
            records = dataset.shape[0]
            shape = dataset.shape[1:]
            channels = np.product(shape)
            slicer = slice(slicebegin, slicebegin+channels)
            slicebegin += channels
            self._multirecords.append(records)
            self._multishapes.append(shape)
            self._multislicers.append(slicer)
            self._multichannels.append(channels)
            self._multidtypes.append(dataset.dtype)
        # Check that all fields have the same time dimension.
        if not (np.array(self._multirecords) == self._multirecords[0]).all():
            raise EofError("all datasets must have the same first dimension")
        # Get the dtype that will be used for the data and weights. This will
        # be the 'highest' dtype of those passed.
        dtype = sorted(self._multidtypes, reverse=True)[0]
        # Form a full array to pass to the EOF solver consisting of all the
        # flat inputs.
        nt = self._multirecords[0]
        ns = self._multichannels.sum()
        dataset = ma.empty([nt, ns], dtype=dtype)
        for iset in xrange(self._ndatasets):
            slicer = self._multislicers[iset]
            channels = self._multichannels[iset]
            dataset[:, slicer] = datasets[iset].reshape([nt, channels])
        # Construct an array of weights the same shape as the data array.
        if weights is not None:
            if len(weights) != self._ndatasets:
                raise EofError("number of weights and datasets differs")
            if not filter(lambda i: False if i is None else True, weights):
                # If every entry in the weights list is None then just pass
                # None to the EofSolver __init__ method.
                warr = None
            else:
                # Construct a spatial weights array.
                warr = np.empty([1, ns], dtype=dtype)
                for iset in xrange(self._ndatasets):
                    slicer = self._multislicers[iset]
                    if weights[iset] is None:
                        # If this dataset has no weights use 1 for the weight
                        # of all elements.
                        warr[:, slicer] = 1.
                    else:
                        # Otherwise use the weights. These need to be
                        # conformed to the correct dimensions.
                        channels = self._multichannels[iset]
                        try:
                            warr[:, slicer] = np.broadcast_arrays(
                                    datasets[iset][0],
                                    weights[iset])[1].reshape([channels])
                        except ValueError:
                            raise EofError("weights are invalid")
        else:
            # Just pass None if none of the input datasets have associated
            # weights.
            warr = None
        # Create an EofSolver object to handle the computations.
        self._solver = EofSolver(dataset, weights=warr, center=center, ddof=1)

    def _unwrap(self, modes):
        """Split a returned mode field into component parts."""
        nmodes = modes.shape[0]
        modeset = [modes[:, slicer].reshape((nmodes,)+shape) \
                for slicer, shape in zip(self._multislicers, self._multishapes)]
        return modeset
        
    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs).

        Returns an array where the columns are the ordered PCs.
        
        **Optional arguments:**

        *pcscaling*
            Set the scaling of the retrieved PCs. The following
            values are accepted:

            * *0* : Un-scaled PCs (default).
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.

        *npcs* : Number of PCs to retrieve. Defaults to all the PCs.
        
        """
        return self._solver.pcs(pcscaling, npcs)

    def eofs(self, eofscaling=0, neofs=None):
        """Empirical orthogonal functions (EOFs).
        
        Returns arrays with the ordered EOFs along the first
        dimension.
        
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
        
        """
        modes = self._solver.eofs(eofscaling, neofs)
        return self._unwrap(modes)

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EOF.

        **Optional argument:**
        
        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues.
        
        """
        return self._solver.eigenvalues(neigs)

    def eofsAsCorrelation(self, neofs=None):
        """
        EOFs scaled as the correlation of the PCs with the original
        field.
        
        **Optional argument:**
        
        *neofs*
            Number of EOFs to return. Defaults to all EOFs.
        
        """
        modes = self._solver.eofsAsCorrelation(neofs)
        return self._unwrap(modes)

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """
        EOFs scaled as the covariance of the PCs with the original
        field.

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

        """
        modes = self._solver.eofsAsCovariance(neofs, pcscaling)
        return self._unwrap(modes)

    def varianceFraction(self, neigs=None):
        """Fractional EOF variances.
        
        The fraction of the total variance explained by each EOF. This
        is a value between 0 and 1 inclusive.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return the fractional variance for.
            Defaults to all eigenvalues.
        
        """
        return self._solver.varianceFraction(neigs)

    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).
        
        """
        return self._solver.totalAnomalyVariance()

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the
        :py:class:`~eof2.MultipleEofSolver` instance then the returned
        reconstructed field will be automatically un-weighted. Otherwise
        the returned reconstructed field will  be weighted in the same
        manner as the input to the
        :py:class:`~eof2.MultipleEofSolver` instance.
        
        **Argument:**
        
        *neofs*
            Number of EOFs to use for the reconstruction.
        
        """
        rf = self._solver.reconstructedField(neofs)
        return self._unwrap(rf)

    def northTest(self, neigs=None, vfscaled=False):
        """Typical errors for eigenvalues.
        
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
            :py:meth:`~eof2.MultipleEofSolver.varianceFraction`
            method. If *False* then no scaling is done. Defaults to
            *False* (no scaling).
        
        **References**

        North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982:
        "Sampling errors in the estimation of empirical orthogonal
        functions", *Monthly Weather Review*, **110**, pages 669-706.
        
        """
        return self._solver.northTest(neigs, vfscaled)

    def getWeights(self):
        """Weights used for the analysis."""
        w = self._solver.getWeights()
        return self._unwrap(w)

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
            One or more fields to project onto the EOFs. The number of
            fields must be the same as the number of fields used to
            initialize the :py:class:`~eof2.MultipleEofSolver`
            instance.

        **Optional arguments:**

        *missing*
            The missing value for all fields, or a list of the
            missing value for each field. If not supplied no particular
            value is assumed to be missing. Note that if 
            :py:attr:`numpy.nan` is used to represent missing values
            then this option does not need to be used as this case is
            handled automatically by the solver.

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

        *notime*
            If *True*, indicates that the input fields have no time
            dimension and should be treated as spatial data. If *False*
            then the first dimension of each input field will be assumed
            to be a time dimension. Defaults to *False* (a time
            dimension is assumed).
                
        """
        if len(fields) != self._ndatasets:
            raise EofError("number of fields differ from original input")
        # Handle keyword arguments manually. This works around an issue in
        # Python where defined keyword arguments cannot follow a variable
        # length regular argument list.
        keywords = {"neofs": None, "eofscaling": 0, "weighted": True,
                "notime": False}
        for kwarg in kwargs.keys():
            if kwarg not in keywords.keys():
                raise EofEorror("invalid argument: %s" % kwarg)
        neofs = kwargs.get("neofs", keywords["neofs"])
        eofscaling = kwargs.get("eofscaling", keywords["eofscaling"])
        weighted = kwargs.get("weighted", keywords["weighted"])
        notime = kwargs.get("notime", keywords["notime"])
        # Record shape information about the input fields.
        multirecords = list()
        multichannels = list()
        multidtypes = list()
        for iset, field in enumerate(fields):
            if notime:
                records = 0
                shape = field.shape
            else:
                records = field.shape[0]
                shape = field.shape[1:]
            channels = np.product(shape)
            if channels != self._multichannels[iset]:
                raise EofError("spatial dimensions do not match original fields")
            multirecords.append(records)
            multichannels.append(channels)
            multidtypes.append(field.dtype)
        # Check that all fields have the same time dimension.
        if not (np.array(multirecords) == multirecords[0]).all():
            raise EofError("all datasets must have the same first dimension")
        # Get the dtype that will be used for the data. This will be the
        # 'highest' dtype of those passed.
        dtype = sorted(multidtypes, reverse=True)[0]
        # Form a full array to pass to the EOF solver consisting of all the
        # combined flat inputs.
        nt = multirecords[0]
        ns = self._multichannels.sum()
        outdims = filter(None, [nt, ns])
        cfields = ma.empty(outdims, dtype=dtype)
        for iset in xrange(self._ndatasets):
            slicer = self._multislicers[iset]
            channels = self._multichannels[iset]
            dims = filter(None, [nt, channels])
            cfields[..., slicer] = fields[iset].reshape(dims)
        # Compute the projection using the EofSolver object.
        pcs = self._solver.projectField(cfields, neofs=neofs,
                eofscaling=eofscaling, weighted=weighted, notime=notime)
        return pcs


if __name__ == "__main__":
    pass 

