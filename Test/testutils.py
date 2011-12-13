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
    # Create the scalr field as the product of the spatial patterns and the
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


def sign_matrix(s1, s2, itype='eof'):
    """
    Create a matrix of sign weights used for adjusting the signs of a
    set of EOFs or PCs to agree with a given set of EOFs or PCs.
    
    This is required since the sign of EOFs and PCs is arbitrary.
    
    Arguments:
    s1, s2 -- Sets of EOFs or PCs.

    Optional argument:
    itype -- The type of the input. Use 'eof' for EOFs and 'pc' for PCs.
            Defaults to 'eof'.

    """
    try:
        # Convert cdms2 variables to numpy masked arrays.
        s1 = s1.asma()
        s2 = s2.asma()
    except:
        pass
    if itype == 'eof':
        # EOFs are dimensioned (eof, space)
        sign = np.where(np.sign(s2[:, 0]) == np.sign(s1[:, 0]),
                1., -1.)[:, np.newaxis]
    elif itype == 'pc':
        # PCs are dimensioned (time, eof)
        sign = np.where(np.sign(s2[0]) == np.sign(s1[0]),
                1., -1.)[np.newaxis]
    return sign


def identify(title):
    print '[%s] ' % title,


