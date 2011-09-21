#!/usr/bin/env python

"""
load_psf: Utility function for auto-recognizing a PSF type from a
fits file and returning the appropriate PSFBase subclass for that PSF.

Stephen Bailey, Summer 2011
"""

import pyfits
from bbspec.spec2d.psf import PSFGauss2D, PSFGaussHermite2D, PSFPixelated

def load_psf(filename):
    """
    Read a psf from a fits file, returning the appropriate subclass PSF object
    based upon the PSFTYPE keyword in the fits file
    """
    fx = pyfits.open(filename)
    psftype = fx[0].header['PSFTYPE']
    if psftype == 'GAUSS2D':
        return PSFGauss2D(filename)
    elif psftype == 'GAUSS-HERMITE':
        return PSFGaussHermite2D(filename)
    elif psftype == 'PCA-PIX':
        return PSFPixelated(filename)
    else:
        print "I don't know about PSFTYPE %s" % psftype
        raise NotImplementedError
