"""EOF decomposition of NumPy arrays

This code is based on the svdeofs class from PyClimate. It has been
heavily re-worked to allow for missing values. Other features have been
added.

"""
# (c) Copyright 2000 Jon Saenz, Jesus Fernandez and Juan Zubillaga.
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
import numpy

from errors import EofError


# New axis constant (actually a reference to 'None' behind the scenes)
_NA = numpy.newaxis

class EofNumPy(object):
    """EOF analysis object
    
    EOF analysis of NumPy arrays with missing data handling.

    """

    def __init__(self, dataset, missing=None, weights=None, center=True,
            ddof=1):
        """Create an EofNumPy object.
        
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
        normfactor = float(self.records - ddof)
        self.L = Lh * Lh / normfactor
        # Store the number of eigenvalues (and hence EOFs) that were actually
        # computed.
        self.neofs = len(self.L)
        # Re-introduce missing values into the eigenvectors in the same places
        # as they exist in the input maps. Create an array of not-a-numbers
        # and then introduce data values where required.
        self.flatE = numpy.ones([self.neofs, self.channels]) * numpy.NaN
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
        # Construct a slice object.
        slicer = slice(0, neofs)
        neofs = neofs or self.neofs
        # If the input array was not centered then do that and use it as the
        # residuals here. Otherwise just use the dataset as the data residuals
        # in the correlation calculation.
        if not self.centered:
            residual = self._center(self.dataset)
        else:
            residual = self.dataset
        # Take a subset of the principal components and compute the residuals.
        # PCs are in columns.
        pcres = self._center(self.P[:,slicer])
        # Compute the standard deviation of the data points and the principal
        # components.
        datastd = numpy.std(self.dataset, axis=0, ddof=1)
        pcsstd = numpy.std(self.P[:,slicer], axis=0, ddof=1)
        # Create an array to store the EOFs.
        eofc = numpy.zeros([neofs, self.channels])
        # Loop over each EOF computing the correlation between the time-series
        # at each input data grid point and the associated PC.
        for i in xrange(len(pcsstd)):
            # Compute departures.
            depts = numpy.add.reduce(residual * pcres[:,i][:,_NA])
            # Compute the correlation coefficient.
            eofc[i] = depts / ((self.neofs - 1) * datastd * pcsstd[i])
        # Return the EOFs the same shape as the input data maps.
        return eofc.reshape((neofs,) + self.originalshape)

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

        If weights were passed to the EofNumPy object then the returned
        reconstructed field will be automatically un-weighted. Otherwise
        the returned reconstructed field will  be weighted in the same
        manner as the input to the EofNumPy object.
        
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
 
 
if __name__ == "__main__":
    pass

