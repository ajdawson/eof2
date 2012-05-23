"""
Supplementary tools for the :py:mod:`numpy` EOF analysis interface.

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
import numpy as np
import numpy.ma as ma

from errors import EofToolError


def _check_flat_center(pcs, field):
    """
    Check PCs and a field for shape compatibility, flatten both to 2D,
    and center along the first dimension.

    This set of operations is common to both covariance and correlation
    calculations.

    """
    # Get the number of times in the field.
    records = field.shape[0]
    if records != pcs.shape[0]:
        # Raise an error if the field has a different number of times to the
        # PCs provided.
        raise EofToolError("PCs and field must have the same first dimension")
    if len(pcs.shape) > 2:
        # Raise an error if the PCs are more than 2D.
        raise EofToolError("PCs must be 1D or 2D")
    # Check if the field is 1D.
    if len(field.shape) == 1:
        field_oned = True
        originalshape = tuple()
        channels = 1
    else:
        # Record the shape of the field and the number of spatial elements.
        originalshape = field.shape[1:]
        channels = np.product(originalshape)
    # Record the number of PCs.
    if len(pcs.shape) == 1:
        npcs = 1
        npcs_out = tuple()
    else:
        npcs = pcs.shape[1]
        npcs_out = (npcs,)
    # Create a flattened field so iterating over space is simple. Also do this
    # for the PCs to ensure they are 2D.
    field_flat = field.reshape([records, channels])
    pcs_flat = pcs.reshape([records, npcs])
    # Centre both the field and PCs in the time dimension.
    field_flat = field_flat - field_flat.mean(axis=0)
    pcs_flat = pcs_flat - pcs_flat.mean(axis=0)
    return pcs_flat, field_flat, npcs_out + originalshape


def correlation_map(pcs, field):
    """Correlation maps for a set of PCs and a spatial-temporal field.

    Given an array where the columns are PCs (e.g., as output from
    :py:meth:`eof2.EofSolve.pcs`) and an array containing a
    spatial-temporal where time is the first dimension, one correlation
    map per PC is computed.

    The field must have the same temporal dimension as the PCs. Any
    number of spatial dimensions (including zero) are allowed in the
    field and there can be any number of PCs.

    **Arguments:**

    *pcs*
        PCs as the columns of an array.

    *field*
        Spatial-temporal field with time as the first dimension.

    """
    # Check PCs and fields for validity, flatten the arrays ready for the
    # computation and remove the mean along the leading dimension.
    pcs_cent, field_cent, out_shape = _check_flat_center(pcs, field)
    # Compute the standard deviation of the PCs and the fields along the time
    # dimension (the leading dimension).
    pcs_std = pcs_cent.std(axis=0)
    field_std = field_cent.std(axis=0)
    # Set the divisor.
    div = np.float64(pcs_cent.shape[0])
    # Compute the correlation map.
    cor = ma.dot(field_cent.T, pcs_cent).T / div
    cor /= ma.outer(pcs_std, field_std)
    # Return the correlation with the appropriate shape.
    return cor.reshape(out_shape)


def covariance_map(pcs, field, ddof=1):
    """Covariance maps for a set of PCs and a spatial-temporal field.

    Given an array where the columns are PCs (e.g., as output from
    :py:meth:`eof2.EofSolve.pcs`) and an array containing a
    spatial-temporal where time is the first dimension, one covariance
    map per PC is computed.

    The field must have the same temporal dimension as the PCs. Any
    number of spatial dimensions (including zero) are allowed in the
    field and there can be any number of PCs.

    **Arguments:**

    *pcs*
        PCs as the columns of an array.

    *field*
        Spatial-temporal field with time as the first dimension.

    **Optional arguments:**

    *ddof*
        'Delta degrees of freedom'. The divisor used to normalize
        the covariance matrix is *N - ddof* where *N* is the
        number of samples. Defaults to *1*.

    """
    # Check PCs and fields for validity, flatten the arrays ready for the
    # computation and remove the mean along the leading dimension.
    pcs_cent, field_cent, out_shape = _check_flat_center(pcs, field)
    # Set the divisor according to the specified delta-degrees of freedom.
    div = np.float64(pcs_cent.shape[0] - ddof)
    # Compute the covariance map, making sure it has the appropriate shape.
    cov = (ma.dot(field_cent.T, pcs_cent).T / div).reshape(out_shape)
    return cov


if __name__ == "__main__":
    pass

