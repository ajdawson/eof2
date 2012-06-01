"""utilities for constructing tests"""
import numpy as np
try:
    import cdms2
except:
    pass


def scalar_field(container_type='numpy'):
    # Create spatial and temporal axes used to compute a dataset composed of
    # orthogonal patterns.
    t = np.linspace(0, 2. * np.pi, 100)
    x = y = np.linspace(-np.pi / 2., np.pi / 2., 15)
    # Construct two orthogonal spatial patterns.
    X, Y = np.meshgrid(x, y)
    Z1 = (X - X.mean()).ravel()
    Z2 = (np.cos(X) * np.cos(Y)).ravel()
    Z1 /= np.sqrt(np.sum(Z1 ** 2))
    Z2 /= np.sqrt(np.sum(Z2 ** 2))
    Z = np.array([Z1, Z2])
    # Construct two uncorrelated time series.
    M1 = np.cos(t)
    M2 = np.sin(t)
    M = np.array([M1, M2]).T
    # Create the scalar field as the product of the spatial patterns and the
    # time series, thus providing a dataset that can be decomposed with a
    # known result.
    D = np.dot(M, Z)
    return D.astype(np.float64), Z.astype(np.float64), M.astype(np.float64)


def generate_data(container_type='numpy'):
    D, Z, M = scalar_field()
    if container_type == 'numpy':
        return D, Z, M
    # Create meta data.
    time = cdms2.createAxis(np.arange(100), id='time')
    time.designateTime()
    time.standard_name = 'time'
    time.units = 'days since 2011-01-1 00:00:0.0'
    longitude = cdms2.createAxis(
        np.arange(0, 360, 360./225.), id='longitude')
    longitude.designateLongitude()
    longitude.standard_name = 'longitude'
    longitude.units = 'degrees_east'
    eof = cdms2.createAxis(range(1, 3), id='eof')
    eof.long_name = 'eof number'
    # Create a cdms2 variable with meta-data.
    sf = cdms2.createVariable(D, axes=(time, longitude),
            id='sf')
    eofs = cdms2.createVariable(Z, axes=(eof, longitude), 
            id='eofs')
    pcs = cdms2.createVariable(M, axes=(time, eof), id='pcs')
    return sf, eofs, pcs


def error(A1, A2):
    """Compute the error between two arrays.

    Computes RMSD normalized by the range of A2.

    """
    try:
        A1 = A1.asma()
        A2 = A2.asma()
    except AttributeError:
        pass
    return (np.sqrt((A1 - A2)**2).mean()) / (np.max(A2) - np.min(A2))


def sign_matrix(s1, s2, transpose=False):
    """
    Create a matrix of sign weights used for adjusting the signs of a
    set of EOFs or PCs to agree with a given set of EOFs or PCs.
    
    This is required since the sign of EOFs and PCs is arbitrary.
    
    **Arguments:**

    *s1*, *s2*
    Sets of EOFs or PCs.

    **Optional argument:**

    *transpose*
    If *True*, transpose the input arrays before computation of the
    sign. If *False* don't use the transpose. Defaults to *False*.
    
    """
    try:
        # cdms2 variables can be cast to masked arrays.
        s1 = s1.asma()
        s2 = s2.asma()
    except:
        pass
    if transpose:
        s1 = s1.T
        s2 = s2.T
    nmodes = s1.shape[0]
    signs = np.empty([nmodes], dtype=np.float64)
    for mode in xrange(nmodes):
        # For each mode, find a point that is non-zero in both inputs. The
        # sign of this value can then be compared to determine the relative
        # signs.
        i = 0
        try:
            while _close(s1[mode, i], 0) or _close(s2[mode, i], 0):
                i += 1
        except IndexError:
            raise ValueError("cannot determine sign due to zeros")
        if np.sign(s1[mode, i]) == np.sign(s2[mode, i]):
            signs[mode] = 1
        else:
            signs[mode] = -1
    return signs


def _close(a, b, rtol=1e-05, atol=1e-08):
    return abs(a - b) <= (atol + rtol * abs(b))


def identify(title):
    print '[%s] ' % title,

