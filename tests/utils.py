"""Tools used for testing :py:mod:`eof2`"""
import numpy as np


def _close(a, b, rtol=1e-05, atol=1e-08):
    """Check if two values are close."""
    return abs(a - b) <= (atol + rtol * abs(b))


def _tomasked(*args):
    """Convert a number of cdms2 variables to masked arrays.
    
    The conversion is safe, so if non-variables are passed they are
    just returned.
    
    """
    def _asma(a):
        try:
            a = a.asma()
        except AttributeError:
            pass
        return a
    return [_asma(a) for a in args]


def error(a, b):
    """Compute the error between two arrays.
    
    Computes the RMSD normalized by the range of the second input.
    
    """
    a, b = _tomasked(a, b)
    return np.sqrt((a - b)**2).mean() / (np.max(b) - np.min(b))


def sign_adjustments(eofset, refeofset):
    """Sign adjustments for EOFs/PCs.
    
    Create a matrix of sign weights used for adjusting the sign of a set
    of EOFs or PCs to the sign of a reference set.
    
    The first dimension is assumed to be modes.
    
    **Arguments:**
    
    *eofset*
        Set of EOFs.
    
    *refeofset*
        Reference set of EOFs.
    
    """
    if eofset.shape != refeofset.shape:
        raise ValueError('input set has different shape from reference set')
    eofset, refeofset = _tomasked(eofset, refeofset)
    nmodes = eofset.shape[0]
    signs = np.ones([nmodes])
    for mode in xrange(nmodes):
        i = 0
        try:
            while _close(eofset[mode,i], 0.) or _close(refeofset[mode,i], 0.):
                i += 1
        except IndexError:
            i = 0
        if np.sign(eofset[mode,i]) != np.sign(refeofset[mode,i]):
            signs[mode] = -1
    return signs[:, np.newaxis]


if __name__ == '__main__':
    pass
