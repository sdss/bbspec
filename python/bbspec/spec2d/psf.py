#!/usr/bin/env python

"""
Classes and functions for manipulating 2D PSFs
"""

import numpy as N
import pyfits

class PSFBase(object):
    """Base class for 2D PSFs"""
    def __init__(self, filename):
        """
        Load PSF parameters from a file
        
        Loads:
            self.param[param_name][ispec, iflux]  #- PSF parameters
            self.npix_x   #- number of columns in the target image
            self.npix_y   #- number of rows in the target image
            self.nspec    #- number of spectra (fibers)
            self.nflux    #- number of flux bins per spectrum
            
        Subclasses of this class define the pix(ispec, iflux) method
        to access the projection of this psf into pixels.
        """
        
        fx = pyfits.open(filename)
        
        self.npix_x = fx[0].header['NPIX_X']
        self.npix_y = fx[0].header['NPIX_Y']
        self.nspec  = fx[0].header['NSPEC']
        self.nflux  = fx[0].header['NFLUX']
        self.param = dict()
        for i in range(len(fx)):
            name = fx[i].header['PSFPARAM']
            self.param[name] = fx[i].data
        fx.close()

    def pix(self, ispec, iflux):
        """
        Evaluate PSF for a given spectrum and flux bin
        
        returns xslice, yslice, pixels[yslice, xslice]
        """
        raise NotImplementedError
        
def load_psf(filename):
    """
    Read a psf from a fits file, returning the appropriate subclass PSF object
    based upon the PSFTYPE keyword in the fits file
    """
    fx = pyfits.open(filename)
    psftype = fx[0].header['PSFTYPE']
    if psftype == 'GAUSS2D':
        return PSFGauss2D(filename)
    else:
        print "I don't know about PSFTYPE %s" % psftype
        raise NotImplementedError

class PSFGauss2D(PSFBase):
    """Rotated 2D Gaussian PSF"""
    def __init__(self, filename):
        """
        Load PSF from file.  See PSFBase doc for details.
        """
        super(PSFGauss2D, self).__init__(filename)

    def pix(self, ispec, iflux):
        """
        Evaluate PSF for a given spectrum and flux bin
        
        returns xslice, yslice, pixels[yslice, xslice]
        """
        x0 = self.param['X'][ispec, iflux]
        y0 = self.param['Y'][ispec, iflux]
        amplitude = self.param['Amplitude'][ispec, iflux]
        sig0 = self.param['MajorAxis'][ispec, iflux]
        sig1 = self.param['MinorAxis'][ispec, iflux]
        theta = self.param['Angle'][ispec, iflux]

        #- Define sampling grid
        dxy = int(3*max(sig0, sig1))
        xmin, xmax = int(x0)-dxy, int(x0)+dxy
        ymin, ymax = int(y0)-dxy, int(y0)+dxy
        
        #- Clip to stay within image bounds
        xmin = max(0, xmin)
        xmax = min(xmax, self.npix_x-1)
        ymin = max(0, ymin)
        ymax = min(ymax, self.npix_y-1)
        
        #- Convert into grid of points to evaluate
        dx = N.arange(xmin, xmax+1) - x0
        dy = N.arange(ymin, ymax+1) - y0        
        dxx, dyy = map(N.ravel, N.meshgrid(dx, dy))

        #- Rotate coordinate systems
        cost = N.cos(theta)
        sint = N.sin(theta)
        xx =  dxx*cost + dyy*sint
        yy = -dxx*sint + dyy*cost

        #- Evaluate 2D gaussian in rotated system
        #- Fast, incorrect, non-integration for now
        norm = 2.5066282746310002  #- N.sqrt(2*N.pi)
        pix = amplitude * N.exp(-0.5*(xx/sig0)**2) * N.exp(-0.5*(yy/sig1)**2)
        pix /= sig0*norm
        pix /= sig1*norm
        
        xslice = slice(xmin, xmax+1)
        yslice = slice(ymin, ymax+1)
        nx = xslice.stop - xslice.start
        ny = yslice.stop - yslice.start
        return xslice, yslice, pix.reshape( (ny, nx) )
