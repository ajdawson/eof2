"""EOF decomposition of NumPy arrays

This code is based on the svdeofs class from PyClimate. It has been
heavily re-worked to allow for missing values. Other features have been
added.

"""
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
import numpy
import warnings

from errors import EofError


# New axis constant (actually a reference to 'None' behind the scenes)
_NA = numpy.newaxis


class EofSolver(object):
    """EOF analysis object
    
    EOF analysis of NumPy arrays with missing data handling.

    """

    def __init__(self, dataset, missing=None, weights=None, center=True,
            ddof=1):
        """Create an EofSolver object.
        
        Arguments:
        dataset -- A NumPy array of two or more dimensions containing
            the data to be decomposed. Time must be the first dimension.
            Missing values are allowed provided that they are constant
            with time (ie. values of an oceanographic variable over
            land).
            
        Optional arguments:
        missing -- The missing value of the data set. Defaults to NaN.
            If the input data set has numpy.nan as its missing value
            then they will automatically be recognized and this option
            is not required.
        weights -- An array of weights whose shape is compatible with
            that of the input data set. the weights can be the same
            shape as the input data set or a shape compatible with a
            NumPy array broadcast operation (ie. the shape of the
            weights can match the rightmost parts of the shape of the
            input data set). Defaults to None (no weighting).
        center -- If True the mean along the first axis of the input
            data set (the time-mean) will be removed prior to analysis.
            If False the mean along the first axis will not be removed.
            This option should only be set to False if you know what you
            are doing and why. Defaults to True (mean is removed).
        ddof -- 'Delta degrees of freedom'. The divisor used to
            normalize the covariance matrix is N - ddof where N is the
            number of samples. Defaults to 1.

        """
        # Store the input data in an instance variable.
        if dataset.ndim < 2:
            raise EofError("The input data set must be at least two dimensional.")
        self.dataset = dataset.copy()
        # Replace missing values with NaN as this makes more sense when
        # handling numpy arrays.
        if missing is not None:
            self.dataset[numpy.where(dataset == missing)] = numpy.NaN
        # Store information about the shape/size of the input data.
        self.records = self.dataset.shape[0]
        self.originalshape = self.dataset.shape[1:]
        self.channels = numpy.multiply.reduce(self.originalshape)
        # Weight the data set according to weighting argument.
        if weights is not None:
            try:
                self.dataset = self.dataset * weights
                self.weights = weights
            except ValueError:
                raise EofError("Weight array dimensions are incompatible.")
            except TypeError:
                raise EofError("Weights are not a valid type.")
        else:
            self.weights = None
        # Remove the time mean of the input data unless explicitly told
        # not to by the 'center' argument.
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
        nonMissingIndex = numpy.where(numpy.isnan(self.dataset[0])==False)[0]
        # Remove missing values from the design matrix.
        dataNoMissing = self.dataset[:, nonMissingIndex]
        # Compute the singular value decomposition of the design matrix.
        A, Lh, E = numpy.linalg.svd(dataNoMissing, full_matrices=False)
        if numpy.any(numpy.isnan(A)):
            raise EofError("Missing values encountered in SVD.")
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
        self.flatE = numpy.ones([self.neofs, self.channels],
                dtype=self.dataset.dtype) * numpy.NaN
        self.flatE = self.flatE.astype(self.dataset.dtype)
        self.flatE[:, nonMissingIndex] = E
        # Remove the scaling on the principal component time-series that is
        # implicitily introduced by using SVD instead of eigen-decomposition.
        # The PCs may be re-scaled later if required.
        self.P = A * Lh

    def _center(self, in_array):
        """Returns the input array centered about the first axis.
        
        Required argument:
        in_array -- The array to centre.
        
        """
        # Compute the mean along the first dimension.
        mean = numpy.add.reduce(in_array) / float(len(in_array))
        # Return the input array with its mean along the first dimension
        # removed.
        return (in_array - mean)

    def pcs(self, pcscaling=0, npcs=None):
        """Returns the principal components.
        
        Returns a numpy array where columns are the ordered principal
        component time series expansions.
        
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

        """
        slicer = slice(0, npcs)
        if pcscaling == 0:
            # Do not scale.
            return self.P[:, slicer].copy()
        elif pcscaling == 1:
            # Divide by the square-root of the eigenvalue.
            return self.P[:, slicer] / numpy.sqrt(self.L[slicer])
        elif pcscaling == 2:
            # Multiply by the square root of the eigenvalue.
            return self.P[:, slicer] * numpy.sqrt(self.L[slicer])
        else:
            raise EofError("Invalid scaling option: %s." % repr(pcscaling))

    def eofs(self, eofscaling=0, neofs=None):
        """Returns the empirical orthogonal functions.
        
        Returns a NumPy array with the ordered empirical orthogonal
        functions along the first axis.
        
        Optional arguments:
        eofscaling -- Sets the scaling of the EOFs. The following values
            are accepted:
            0 - Un-scaled EOFs.
            1 - EOFs are divided by the square-root of their eigenvalues.
            2 - EOFs are multiplied by the square-root of their
                eigenvalues.
            Defaults to 0 (un-scaled EOFs).
        neofs -- Number of EOFs to return. Defaults to all EOFs.
        
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
            rval = self.flatE[slicer] / numpy.sqrt(self.L[slicer])[:,_NA]
            return rval.reshape((neofs,) + self.originalshape)
        elif eofscaling == 2:
            # Multiply by the square-root of the eigenvalues.
            rval = self.flatE[slicer] * numpy.sqrt(self.L[slicer])[:,_NA]
            return rval.reshape((neofs,) + self.originalshape)
        else:
            raise EofError("Invalid eof scaling option: %s." % repr(eofscaling))

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EOF.

        Optional argument:
        neigs -- Number of eigenvalues to return. Defaults to all
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
        EOFs scaled as the correlation of the PCs with original field.
        
        Optional argument:
        neofs -- Number of EOFs to return. Defaults to all EOFs.
        
        """
        # Correlation of the original dataset with the PCs is identical to:
        #   e_m l_m^(1/2) / s
        # where e_m is the m'th eigenvector (EOF), l is the m'th eigenvalue,
        # and s is the vector of standard deviations of the original
        # data set (see Wilks 2006).
        #
        # Retrieve the EOFs multiplied by the square-root of their eigenvalue.
        e = self.eofs(eofscaling=2, neofs=neofs)
        # Compute the standard deviation map of the input dataset.
        s = numpy.std(self.dataset.reshape(
                (self.records,) + self.originalshape), axis=0, ddof=self.ddof)
        # Compute the correlation maps, warnings are turned off to handle the
        # case where standard deviation is zero. This can happen easily if
        # cosine latitude weights are applied to a field with grid points at
        # 90N, which will be weighted by zero and hence not vary with time, or
        # more generally if the input dataset is constant in time at one or
        # more locations.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = e / s
        # Return the correlation maps.
        return c

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """
        EOFs scaled as the covariance of the PCs with original field.

        Optional arguments:
        neofs -- Number of EOFs to return. Defaults to all EOFs.
        pcscaling -- Sets the scaling of the principal components. The
            following values are accepted:
            0 - Un-scaled principal components.
            1 - Principal components are divided by the square-root of
                their eigenvalues. This results in PCs with unit
                variance.
            2 - Principal components are multiplied by the square-root
                of their eigenvalues.
            Defaults to 1 (standardised principal components).

        """
        if pcscaling not in (1, 2, 3):
            # An invalid PC scaling option was given.
            raise EofError("Invalid pc scaling option: %s." % repr(eofscaling))
        # Retrieve the EOFs expressed as correlation between PCs and the
        # original data.
        eofsc = self.eofsAsCorrelation(neofs=neofs)
        # Compute the standard deviation of the PCs. Retrieve the appropriate
        # number of eigenvalues. If the PCs are scaled by (multiplication of)
        # the square-root of their eigenvalue then their standard deviations
        # are given by the eigenvalues.
        pcstd = self.L[slice(0, neofs)]
        if pcscaling == 0:
            # If PCs are unscaled then their standard deviation is the
            # square-root of their eigenvalue.
            pcstd = numpy.sqrt(pcstd)
        elif pcscaling == 1:
            # If the PCs are scaled by (division of) the square-root of their
            # eigenvalue then their variance and standard deviation is 1.
            pcstd = numpy.ones_like(pcstd)
        # We shape the array of standard deviations so it can be broadcast
        # against the EOFs expressed as correlation of the PCs with the input
        # data.
        pcstd = pcstd.reshape([len(pcstd)] + [1] * len(self.originalshape))
        # Compute the standard deviation of the input data set time series.
        # This is reshaped into the spatial dimensions of the input data.
        if self.weights is not None:
            # If the input data was weighted then we should remove the
            # weighting before computing the standard deviation.
            datastd = numpy.std(
                    self.dataset.reshape(
                        (self.records,)+self.originalshape) / \
                    self.weights, axis=0, ddof=self.ddof)
        else:
            # If no weighting was used then the dataset does not need to be
            # adjusted.
            datastd = numpy.std(self.dataset, axis=0, ddof=self.ddof).reshape(
                    self.originalshape)
        # Multiply by the standard deviation of the PCs and data time series
        # at each point. This converts the correlation into covariance.
        eofsv = eofsc * datastd * pcstd
        # Return the EOFs expressed as covariance of PCs and the input data.
        return eofsv
        
    def varianceFraction(self, neigs=None):
        """
        Fraction of the total variance explained by each principal mode.

        Optional argument:
        neigs -- Number of eigenvalues to return the fractional variance
            for. Defaults to all eigenvalues.
        
        """
        # Return the array of eigenvalues divided by the sum of the
        # eigenvalues.
        slicer = slice(0, neigs)
        return self.L[slicer] / numpy.add.reduce(self.L)

    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).
        
        """
        # Return the sum of the eigenvalues.
        return numpy.add.reduce(self.L)

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the EofSolver object then the returned
        reconstructed field will be automatically un-weighted. Otherwise
        the returned reconstructed field will  be weighted in the same
        manner as the input to the EofSolver object.
        
        Argument:
        neofs -- Number fo EOFs to use for the reconstruction.
        
        """
        # Project principal components onto the EOFs to compute the
        # reconstructed field.
        rval = numpy.dot(self.P[:, :neofs], self.flatE[:neofs])
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
        """Returns typical errors for eigenvalues.
        
        Uses the method of North et al. (1982) to compute the typical
        error for each eigenvalue. It is assumed that the number of
        times in the input data set is the same as the number of
        independent realizations. If this assumption is not valid then
        the results of this method may be inappropriate.
        
        Optional arguments:
        neigs -- The number of eigenvalues to return typical errors for.
            Defaults to typical errors for all eigenvalues.
        vfscaled -- If True scale the errors by the sum of the
            eigenvalues. This yields typical errors with the same scale
            as the values returned by the 'varianceFraction' method. If
            False then no scaling is done. Defaults to no scaling.
        
        North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982,
            "Sampling errors in the estimation of empirical orthogonal
            functions", Monthly Weather Review, 110, pages 669-706.
        
        """
        slicer = slice(0, neigs)
        # Compute the factor that multiplies the eigenvalues. The number of
        # records is assumed to be the number of realizations of the field.
        factor = numpy.sqrt(2.0 / self.records)
        # If requested, allow for scaling of the eigenvalues by the total
        # variance (sum of the eigenvalues).
        if vfscaled:
            factor /= numpy.add.reduce(self.L)
        # Return the typical errors.
        return self.L[slicer] * factor

    def getWeights(self):
        """Return the weights used for the analysis."""
        return self.weights

    def projectField(self, field, missing=None, neofs=None, eofscaling=0,
            weighted=True, notime=False):
        """Project a field onto the EOFs.
        
        Projects a field onto the EOFs, or a subset of the EOFS,
        associated with this instance. The field must have the same
        spatial dimensions as the field used to initialize the EofSolver
        object, but may have a different length time dimension. Missing
        values must be in the same places as in the original field also.

        Argument:
        field -- NumPy array of data values to project onto EOFs. Must
            have the same dimensionality as the input to the EofSolver
            object except for the leading time dimension which may be
            any length.
        
        Optional arguments:
        missing -- The missing value of the data set. Defaults to NaN.
            If the input data set has numpy.nan as its missing value
            then they will automatically be recognized and this option
            is not required.
        neofs -- Number of EOFs to project onto. Defaults to all EOFs.
        eofscaling -- Sets the scaling of the EOFs. The following values
            are accepted:
            0 - Un-scaled EOFs.
            1 - EOFs are divided by the square-root of their eigenvalues.
            2 - EOFs are multiplied by the square-root of their
                eigenvalues.
            Defaults to 0 (un-scaled EOFs).
        weighted -- If True then the EOFs are weighted prior to the
            projection. If False then no weighting is applied. Defaults
            to True (weighting is applied).
        notime -- If True indicates that the input field has no time
            dimension and should be treated as spatial data. If False
            then the input field will be treated as spatial-temporal
            data. Defaults to False (spatial-temporal data).
        
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
        # Replace missing values with NaN as this makes more sense when
        # handling numpy arrays.
        if missing is not None:
            field[numpy.where(field == missing)] = numpy.NaN
        # Flatten the input field into [time, space] dimensionality unless it
        # is indicated that there is no time dimension present.
        if notime:
            channels = numpy.multiply.reduce(field.shape)
            field_flat = field.reshape([channels])
            nonMissingIndex = numpy.where(numpy.isnan(field_flat) == False)[0]
        else:
            records = field.shape[0]
            channels = numpy.multiply.reduce(field.shape[1:])
            field_flat = field.reshape([records, channels])
            nonMissingIndex = numpy.where(numpy.isnan(field_flat[0]) == False)[0]
        # Isolate the non-missing points.
        field_flat = field_flat[..., nonMissingIndex]
        # Remove missing values from the flat EOFs.
        eofNonMissingIndex = numpy.where(numpy.isnan(self.flatE[0]) == False)[0]
        if eofNonMissingIndex.shape != nonMissingIndex.shape or \
                (eofNonMissingIndex != nonMissingIndex).any():
            raise EofError("field and EOFs have different missing values")
        eofs_flat = self.flatE[slicer, eofNonMissingIndex]
        if eofscaling == 1:
            eofs_flat /= numpy.sqrt(self.L[slicer])[:,_NA]
        elif eofscaling == 2:
            eofs_flat *= numpy.sqrt(self.L[slicer])[:,_NA]
        # Project the field onto the EOFs using a matrix multiplication.
        projected_pcs = numpy.dot(field_flat, eofs_flat.T)
        return projected_pcs


# Create an alias 'EofNumPy' for backward compatibility.
EofNumPy = EofSolver


if __name__ == "__main__":
    pass

