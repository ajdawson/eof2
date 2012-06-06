"""EOF analysis for :py:mod:`numpy` array data."""
# (c) Copyright 2000 Jon Saenz, Jesus Fernandez and Juan Zubillaga.
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
import warnings

from errors import EofError
from nptools import correlation_map, covariance_map


# New axis constant (actually a reference to *None* behind the scenes)
_NA = np.newaxis


class EofSolver(object):
    """EOF analysis (:py:mod:`numpy` interface)."""

    def __init__(self, dataset, weights=None, center=True, ddof=1):
        """Create an EofSolver object.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**
        
        *dataset*
            A :py:class:`numpy.ndarray` or
            :py:class:`numpy.ma.core.MasekdArray` with two or more
            dimensions containing the data to be analysed. The first
            dimension is assumed to represent time. Missing values are
            permitted, either in the form of a masked array, or the
            value :py:attr:`numpy.nan`. Missing values must be constant
            with time (e.g., values of an oceanographic field over
            land).
            
        **Optional arguments:**

        *weights*
            An array of weights whose shape is compatible with those of
            the input array *dataset*. The weights can have the same
            shape as the input data set or a shape compatible with an
            array broadcast operation (ie. the shape of the weights can
            can match the rightmost parts of the shape of the input
            array *dataset*). If the input array *dataset* does not
            require weighting then the value *None* may be used.
            Defaults to *None* (no weighting).

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
        # Store the input data in an instance variable.
        if dataset.ndim < 2:
            raise EofError("the input data set must be at least two dimensional")
        self.dataset = dataset.copy()
        # Check if the input is a masked array. If so fill it with NaN.
        try:
            self.dataset = self.dataset.filled(fill_value=np.nan)
        except AttributeError:
            pass
        # Store information about the shape/size of the input data.
        self.records = self.dataset.shape[0]
        self.originalshape = self.dataset.shape[1:]
        self.channels = np.product(self.originalshape)
        # Weight the data set according to weighting argument.
        if weights is not None:
            try:
                self.dataset = self.dataset * weights
                self.weights = weights
            except ValueError:
                raise EofError("weight array dimensions are incompatible")
            except TypeError:
                raise EofError("weights are not a valid type")
        else:
            self.weights = None
        # Remove the time mean of the input data unless explicitly told
        # not to by the "center" argument.
        self.centered = center
        if center:
            self.dataset = self._center(self.dataset)
        # Reshape to two dimensions (time, space) creating the design matrix.
        self.dataset = self.dataset.reshape([self.records, self.channels])
        # Find the indices of values that are not missing in one row. All the
        # rows will have missing values in the same places provided the
        # array was centered. If it wasn't then it is possible that some
        # missing values will be missed and the singular value decomposition
        # will produce not a number for everything.
        nonMissingIndex = np.where(np.isnan(self.dataset[0])==False)[0]
        # Remove missing values from the design matrix.
        dataNoMissing = self.dataset[:, nonMissingIndex]
        # Compute the singular value decomposition of the design matrix.
        A, Lh, E = np.linalg.svd(dataNoMissing, full_matrices=False)
        if np.any(np.isnan(A)):
            raise EofError("missing values encountered in SVD")
        # Singular values are the square-root of the eigenvalues of the
        # covariance matrix. Construct the eigenvalues appropriately and
        # normalize by N-ddof where N is the number of observations. This
        # corresponds to the eigenvalues of the normalized covariance matrix.
        self.ddof = ddof
        normfactor = float(self.records - self.ddof)
        self.L = Lh * Lh / normfactor
        # Store the number of eigenvalues (and hence EOFs) that were actually
        # computed.
        self.neofs = len(self.L)
        # Re-introduce missing values into the eigenvectors in the same places
        # as they exist in the input maps. Create an array of not-a-numbers
        # and then introduce data values where required. We have to use the
        # astype method to ensure the eigenvectors are the same type as the
        # input dataset since multiplication by np.NaN will promote to 64-bit.
        self.flatE = np.ones([self.neofs, self.channels],
                dtype=self.dataset.dtype) * np.NaN
        self.flatE = self.flatE.astype(self.dataset.dtype)
        self.flatE[:, nonMissingIndex] = E
        # Remove the scaling on the principal component time-series that is
        # implicitily introduced by using SVD instead of eigen-decomposition.
        # The PCs may be re-scaled later if required.
        self.P = A * Lh

    def _center(self, in_array):
        """Remove the mean of an array along the first dimension."""
        # Compute the mean along the first dimension.
        mean = in_array.mean(axis=0)
        # Return the input array with its mean along the first dimension
        # removed.
        return (in_array - mean)

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
        slicer = slice(0, npcs)
        if pcscaling == 0:
            # Do not scale.
            return self.P[:, slicer].copy()
        elif pcscaling == 1:
            # Divide by the square-root of the eigenvalue.
            return self.P[:, slicer] / np.sqrt(self.L[slicer])
        elif pcscaling == 2:
            # Multiply by the square root of the eigenvalue.
            return self.P[:, slicer] * np.sqrt(self.L[slicer])
        else:
            raise EofError("invalid PC scaling option: %s" % repr(pcscaling))

    def eofs(self, eofscaling=0, neofs=None):
        """Empirical orthogonal functions (EOFs).
        
        Returns an array with the ordered EOFs along the first
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
        slicer = slice(0, neofs)
        neofs = neofs or self.neofs
        if eofscaling == 0:
            # No modification. A copy needs to be returned in case it is
            # modified. If no copy is made the internally stored eigenvectors
            # could be modified unintentionally.
            rval = self.flatE[slicer].copy()
            return rval.reshape((neofs,) + self.originalshape)
        elif eofscaling == 1:
            # Divide by the square-root of the eigenvalues.
            rval = self.flatE[slicer] / np.sqrt(self.L[slicer])[:,_NA]
            return rval.reshape((neofs,) + self.originalshape)
        elif eofscaling == 2:
            # Multiply by the square-root of the eigenvalues.
            rval = self.flatE[slicer] * np.sqrt(self.L[slicer])[:,_NA]
            return rval.reshape((neofs,) + self.originalshape)
        else:
            raise EofError("invalid eof scaling option: %s" % repr(eofscaling))

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EOF.
        
        **Optional argument:**
        
        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues.
        
        """
        # Create a slicer and use it on the eigenvalue array. A copy must be
        # returned in case the slicer takes all elements, in which case a
        # reference to the eigenvalue array is returned. If this is modified
        # then the internal eigenvalues array would then be modified as well.
        slicer = slice(0, neigs)
        return self.L[slicer].copy()

    def eofsAsCorrelation(self, neofs=None):
        """
        EOFs scaled as the correlation of the PCs with the original
        field.
        
        **Optional argument:**
        
        *neofs*
            Number of EOFs to return. Defaults to all EOFs.
        
        """
        # Retrieve the specified number of PCs.
        pcs = self.pcs(npcs=neofs, pcscaling=1)
        # Compute the correlation of the PCs with the input field.
        c = correlation_map(pcs,
                self.dataset.reshape((self.records,)+self.originalshape))
        # The results of the correlation_map function will be a masked array.
        # For consistency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        return c.filled(fill_value=np.nan)

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
        pcs = self.pcs(npcs=neofs, pcscaling=pcscaling)
        # Divide the input data by the weighting (if any) before computing
        # the covariance maps.
        data = self.dataset.reshape((self.records,)+self.originalshape)
        if self.weights is not None:
            data /= self.weights
        c = covariance_map(pcs, data, ddof=self.ddof)
        # The results of the covariance_map function will be a masked array.
        # For consitsency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        return c.filled(fill_value=np.nan)
        
    def varianceFraction(self, neigs=None):
        """Fractional EOF variances.
        
        The fraction of the total variance explained by each EOF. This
        is a value between 0 and 1 inclusive.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return the fractional variance for.
            Defaults to all eigenvalues.
        
        """
        # Return the array of eigenvalues divided by the sum of the
        # eigenvalues.
        slicer = slice(0, neigs)
        return self.L[slicer] / self.L.sum()

    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).
        
        """
        # Return the sum of the eigenvalues.
        return self.L.sum()

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the :py:class:`~eof2.EofSolver`
        instance then the returned reconstructed field will be
        automatically un-weighted. Otherwise the returned reconstructed
        field will  be weighted in the same manner as the input to the
        :py:class:`~eof2.EofSolver` instance.
        
        **Argument:**
        
        *neofs*
            Number of EOFs to use for the reconstruction.
        
        """
        # Project principal components onto the EOFs to compute the
        # reconstructed field.
        rval = np.dot(self.P[:, :neofs], self.flatE[:neofs])
        # Reshape the reconstructed field so it has the same shape as the
        # input data set.
        rval = rval.reshape((self.records,) + self.originalshape)
        # Un-weight the reconstructed field if weighting was performed on
        # the input data set.
        if self.weights is not None:
            rval = rval / self.weights
        # Return the reconstructed field.
        return rval

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
            :py:meth:`~eof2.EofSolver.varianceFraction` method. If
            *False* then no scaling is done. Defaults to *False* (no
            scaling).
        
        **References**

        North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982:
        "Sampling errors in the estimation of empirical orthogonal
        functions", *Monthly Weather Review*, **110**, pages 669-706.
        
        """
        slicer = slice(0, neigs)
        # Compute the factor that multiplies the eigenvalues. The number of
        # records is assumed to be the number of realizations of the field.
        factor = np.sqrt(2.0 / self.records)
        # If requested, allow for scaling of the eigenvalues by the total
        # variance (sum of the eigenvalues).
        if vfscaled:
            factor /= self.L.sum()
        # Return the typical errors.
        return self.L[slicer] * factor

    def getWeights(self):
        """Weights used for the analysis."""
        return self.weights

    def projectField(self, field, neofs=None, eofscaling=0, weighted=True,
            notime=False):
        """Project a field onto the EOFs.
        
        Given a field, projects it onto the EOFs to generate a
        corresponding set of time series. The field can be projected
        onto all the EOFs or just a subset. The field must have the same
        corresponding spatial dimensions (including missing values in
        the same places) as the original input to the
        :py:class:`~eof2.MultipleEofSolver` instance. The field may have a
        different length time dimension to the original input field (or
        no time dimension at all).
        
        **Argument:**
        
        *field*
            A field to project onto the EOFs. The field should be
            contained in a :py:class:`numpy.ndarray` or a
            :py:class:`numpy.ma.core.MaskedArray`.

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

        *notime*
            If *True*, indicates that the input field has no time
            dimension and should be treated as spatial data. If *False*
            then the first dimension of the field will be assumed to be
            a time dimension. Defaults to *False* (a time dimension is
            assumed).
                
        """
        # Create a slice object for truncating the EOFs.
        slicer = slice(0, neofs)
        # If required, weight the dataset with the same weighting that was
        # used to compute the EOFs.
        field = field.copy()
        if weighted:
            wts = self.getWeights()
            if wts is not None:
                field = field * wts
        try:
            field = field.filled(fill_value=np.nan)
        except AttributeError:
            pass
        # Flatten the input field into [time, space] dimensionality unless it
        # is indicated that there is no time dimension present.
        if notime:
            channels = np.product(field.shape)
            field_flat = field.reshape([channels])
            nonMissingIndex = np.where(np.isnan(field_flat) == False)[0]
        else:
            records = field.shape[0]
            channels = np.product(field.shape[1:])
            field_flat = field.reshape([records, channels])
            nonMissingIndex = np.where(np.isnan(field_flat[0]) == False)[0]
        # Isolate the non-missing points.
        field_flat = field_flat[..., nonMissingIndex]
        # Remove missing values from the flat EOFs.
        eofNonMissingIndex = np.where(np.isnan(self.flatE[0]) == False)[0]
        if eofNonMissingIndex.shape != nonMissingIndex.shape or \
                (eofNonMissingIndex != nonMissingIndex).any():
            raise EofError("field and EOFs have different missing values")
        eofs_flat = self.flatE[slicer, eofNonMissingIndex]
        if eofscaling == 1:
            eofs_flat /= np.sqrt(self.L[slicer])[:,_NA]
        elif eofscaling == 2:
            eofs_flat *= np.sqrt(self.L[slicer])[:,_NA]
        # Project the field onto the EOFs using a matrix multiplication.
        projected_pcs = np.dot(field_flat, eofs_flat.T)
        return projected_pcs


# Create an alias *EofNumPy* for backward compatibility.
EofNumPy = EofSolver


if __name__ == "__main__":
    pass

