"""Generation of reference solutions for EOF analysis."""
import numpy as np
try:
    import cdms2
except ImportError:
    pass


def _construct_reference():
    # Create spatial and temporal axes used to compute a dataset composed of
    # orthogonal patterns.
    t = np.linspace(0., 2.*np.pi, 100)
    x = y = np.linspace(-np.pi/2., np.pi/2., 15)
    # Construct two orthogonal spatial patterns.
    X, Y = np.meshgrid(x, y)
    Z1 = (X - X.mean()).ravel()
    Z2 = (np.cos(X) * np.cos(Y)).ravel()
    Z1 /= np.sqrt(np.sum(Z1**2))
    Z2 /= np.sqrt(np.sum(Z2**2))
    Z = np.array([Z1, Z2])
    # Construct two uncorrelated time series.
    M1 = np.cos(t)
    M2 = np.sin(t)
    M = np.array([M1, M2]).T
    # Create the scalar field as the product of the spatial patterns and the
    # time series, thus providing a dataset that can be decomposed with a
    # known result.
    D = np.dot(M, Z)
    return D, Z, M


def reference_solution(container_type):
    """Generate a reference field and a corresponding EOF solution.
    
    **Argument:**
    
    *container_type*
        The type of the solution containers. Either 'numpy' for
        :py:mod:`numpy` arrays or 'cdms2' for :py:mod:`cdms2`.
    
    """
    sf, eofs, pcs = _construct_reference()
    if container_type.lower() == 'numpy':
        # Return the solution as-is for numpy containers.
        return sf, eofs, pcs
    # Create meta-data for cdms2 containers.
    try:
        time = cdms2.createAxis(np.arange(100), id='time')
        time.designateTime()
        time.units = 'days since 2011-01-01 00:00:0.0'
        longitude = cdms2.createAxis(np.arange(0., 360., 360./225.),
                id='longitude')
        longitude.designateLongitude()
        eof = cdms2.createAxis(range(2), id='eof')
        eof.long_name = 'eof number'
        sf = cdms2.createVariable(sf, axes=[time,longitude], id='sf')
        eofs = cdms2.createVariable(eofs, axes=[eof,longitude], id='eofs')
        pcs = cdms2.createVariable(pcs, axes=[time,eof], id='pcs')
    except NameError:
        raise ValueError("can't create cdms2 containers without cdms2")
    return sf, eofs, pcs


if __name__ == '__main__':
    pass
