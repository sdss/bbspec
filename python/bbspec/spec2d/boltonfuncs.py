import numpy as n
from scipy import special as sp

def sym_sqrt(a):
    """
    NAME: sym_sqrt

    PURPOSE: take 'square root' of a symmetric matrix via diagonalization

    USAGE: s = sym_sqrt(a)

    ARGUMENT: a: real symmetric square 2D ndarray

    RETURNS: s such that a = numpy.dot(s, s)

    WRITTEN: Adam S. Bolton, U. of Utah, 2009
    """
    w, v = n.linalg.eigh(a)
    w[w<0]=0 # Is this necessary to enforce eigenvalues positive definite???
    dm = n.diagflat(n.sqrt(w))
    return n.dot(v, n.dot(dm, n.transpose(v)))

def resolution_from_icov(icov):
    """
    Function to generate the 'resolution matrix' in the simplest
    (no unrelated crosstalk) Bolton & Schlegel 2010 sense.
    Works on dense matrices.  May not be suited for production-scale
    determination in a spectro extraction pipeline.

    Input argument is inverse covariance matrix array.
    If input is not 2D and symmetric, results will be unpredictable.

    WRITTEN: Adam S. Bolton, U. of Utah, 2009
    """
    sqrt_icov = sym_sqrt(icov)
    norm_vector = n.sum(sqrt_icov, axis=1)
    r_mat = n.outer(norm_vector**(-1), n.ones(norm_vector.size)) * sqrt_icov
    return r_mat

def bilinear(xinterp, yinterp, zbase, xbase=None, ybase=None):
    """
    Bilinear interpolation function.

    Arguments:
    xinterp, yinterp:
      x and y coordinates at which to interpolate
      (numpy ndarrays of any matching shape)
    zbase:
      Two-dimensional grid of values within which to interpolate.
    xbase, ybase:
      Baseline coordinates of the zbase image.
      should be monotonically increasing numpy 1D ndarrays, with
      lengths matched to the shape of zbase.
      If not set, these default to n.arange(nx) and n.arange(ny),
      where ny, nx = zbase.shape

    Returns:
      An array of interpolated values, with the same shape as xinterp.

    Written: Adam S. Bolton, U of Utah, 2010apr
    """
    ny, nx = zbase.shape
    if (xbase is None):
        xbase = n.arange(nx, dtype='float')
    if (ybase is None):
        ybase = n.arange(ny, dtype='float')
    # The digitize bins give the indices of the *upper* interpolating points:
    xhi = n.digitize(xinterp.flatten(), xbase)
    yhi = n.digitize(yinterp.flatten(), ybase)
    # Reassign to the highest bin if out of bounds above, and to the
    # lowest bin if out of bounds below, relying on the conventions
    # of numpy.digitize to return nx & ny or zero in these cases:
    xhi = xhi - (xhi == nx) + (xhi == 0)
    yhi = yhi - (yhi == ny) + (yhi == 0)
    # The indices of the *lower* bins:
    xlo = xhi - 1
    ylo = yhi - 1
    # The fractional positions within the bins:
    fracx = (xinterp.flatten() - xbase[xlo]) / (xbase[xhi] - xbase[xlo])
    fracy = (yinterp.flatten() - ybase[ylo]) / (ybase[yhi] - ybase[ylo])
    # The interpolation formula:
    zinterp = (zbase[ylo,xlo] * (1. - fracx) * (1. - fracy) +
               zbase[ylo,xhi] * fracx * (1. - fracy) +
               zbase[yhi,xlo] * (1. - fracx) * fracy +
               zbase[yhi,xhi] * fracx * fracy).reshape(xinterp.shape)
    return zinterp

def pgh(x, m=0, xc=0., sigma=1., dxlo=-0.5, dxhi=0.5):
    """
    Pixel-integrated (probabilist) Gauss-Hermite function.

    Arguments:
      x: pixel-center baseline array
      m: order of Hermite polynomial multiplying Gaussian core
      xc: sub-pixel position of Gaussian centroid relative to x baseline
      sigma: sigma parameter of Gaussian core in units of pixels
      dxlo, dxhi: pixel boundary low and high offsets, defauting to -/+0.5

    Written: Adam S. Bolton, U. of Utah, fall 2010
    """
    u = (x - xc) / sigma
    ulo = (x + dxlo - xc) / sigma
    uhi = (x + dxhi - xc) / sigma
    if m > 0:
        hiterm = - sp.hermitenorm(m-1)(uhi) * n.exp(-0.5 * uhi**2) / n.sqrt(2. * n.pi)
        loterm = - sp.hermitenorm(m-1)(ulo) * n.exp(-0.5 * ulo**2) / n.sqrt(2. * n.pi)
        return hiterm - loterm
    else:
        return 0.5 * (sp.erf(uhi/n.sqrt(2.)) - sp.erf(ulo/n.sqrt(2.)))

