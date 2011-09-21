#!/usr/bin/env python

"""
Pixelated 2D PSF

David Schlegel & Stephen Bailey, Summer 2011
"""

import numpy as N
import scipy.signal
import pyfits
from bbspec.spec2d.psf import PSFBase

#- Turn of complex -> real warnings in sinc interpolation
import warnings 
try:
	warnings.simplefilter("ignore", N.ComplexWarning)
except   AttributeError:
	pass
	
class PSFPixelated(PSFBase):
    """
    Pixelated PSF[ny, nx] = Ic[ny,nx] + x*Ix[ny, nx] + y*Iy[ny,nx] + ...
    """
    def __init__(self, filename):
        """
        Loads pixelated PSF parameters from a fits file
        """
        fx = pyfits.open(filename)
        
        self.npix_x = int(fx[0].header['NPIX_X'])
        self.npix_y = int(fx[0].header['NPIX_Y'])
        self.nspec  = int(fx[0].header['NSPEC'])
        self.nflux  = int(fx[0].header['NFLUX'])
        
        #- Read in positions and wavelength solutions from first 3 headers
        self._iflux = N.arange(self.nflux)  #- convenience index array
        self.param = dict(_iflux = N.tile(self._iflux, self.nspec).reshape(self.nspec, self.nflux))
        self.param['X'] = fx[0].data
        self.param['Y'] = fx[1].data
        
        #- check if X and Y are swapped
        if max(self.param['X'][0]) - min(self.param['X'][0]) > 2000:
            print 'swapping X and Y'
            self.param['X'] = fx[1].data
            self.param['Y'] = fx[0].data
        
        self.param['LogLam'] = fx[2].data
        
        #- Additional headers are a custom format for the pixelated psf
        self.nexp     = fx[3].data.view(N.ndarray)  #- icoeff xexp yexp
        self.xyscale  = fx[4].data.view(N.ndarray)  #- ifiber igroup x0 xscale y0 yscale
        self.psfimage = fx[5].data.view(N.ndarray)  #- [igroup, icoeff, iy, ix]
                
    def pix(self, ispec, iflux=None, loglam=None, y=None, xyrange=False):
        """
        Evaluate PSF for a given spectrum and flux bin
        
        returns xslice, yslice, pixels[yslice, xslice]
        """
        #- Get fiber group and scaling factors for this spectrum
        igroup = self.xyscale['IGROUP'][ispec]
        x0     = self.xyscale['X0'][ispec]
        xscale = self.xyscale['XSCALE'][ispec]
        y0     = self.xyscale['Y0'][ispec]
        yscale = self.xyscale['YSCALE'][ispec]
        
        #- Get x and y centroid for this spectrum and flux bin
        x = self.x(ispec, iflux=iflux, loglam=loglam, y=y)
        y = self.y(ispec, iflux=iflux, loglam=loglam, y=y)
        
        #- Rescale units
        xx = xscale * (x - x0)
        yy = yscale * (y - y0)
        
        #- Generate PSF image at (x,y)
        psfimage = N.zeros(self.psfimage.shape[2:4])
        for i in range(self.psfimage.shape[1]):
            nx = self.nexp['XEXP'][i]
            ny = self.nexp['YEXP'][i]
            psfimage += xx**nx * yy**ny * self.psfimage[igroup, i]
                                
        #- Sinc Interpolate
        dx = int(round(x)) - x
        dy = int(round(y)) - y
        psfimage = self._sincshift(psfimage, dx, dy)
        
        #- Check boundaries
        ny, nx = psfimage.shape
        ix = int(round(x))
        iy = int(round(y))
        
        xmin = max(0, ix-nx/2)
        xmax = min(self.npix_x, ix+nx/2+1)
        ymin = max(0, iy-ny/2)
        ymax = min(self.npix_y, iy+ny/2+1)
                
        if ix < nx/2:
            psfimage = psfimage[:, nx/2-ix:]
        if iy < ny/2:
            psfimage = psfimage[ny/2-iy:, :]
        
        if ix+nx/2+1 > self.npix_x:
            dx = self.npix_x - (ix+nx/2+1)
            psfimage = psfimage[:, :dx]
            
        if iy+ny/2+1 > self.npix_y:
            dy = self.npix_y - (iy+ny/2+1)
            psfimage = psfimage[:dy, :]
        
        xslice = slice(xmin, xmax)
        yslice = slice(ymin, ymax)
        
        if xyrange:
            return xslice, yslice, psfimage
        else:
            return psfimage

    #- Utility functions for sinc shifting pixelated PSF images
    def _sincfunc(self, x, dx):
        dampfac = 3.25
        if dx != 0.0:
            return N.exp( -((x+dx)/dampfac)**2 ) * N.sin( N.pi*(x+dx) ) / (N.pi * (x+dx))
        else:
            xx = N.zeros(len(x))
            xx[len(x)/2] = 1.0
            return xx

    def _sincshift(self, image, dx, dy): 
        
        #- DEBUG : Turn off shifting
        return image
        #- DEBUG : Turn off shifting
        
        sincrad = 10
        s = N.arange(-sincrad, sincrad+1)
        sincx = self._sincfunc(s, dx)

        #- If we're shifting just in x, do faster 1D convolution with wraparound
        #- WARNING: can introduce edge effects if PSF isn't nearly 0 at edge
        if abs(dy) < 1e-6:
            newimage = scipy.signal.convolve(image.ravel(), sincx, mode='same')
            return newimage.reshape(image.shape)
        
        sincy = self._sincfunc(s, dy)
        kernel = N.outer(sincy, sincx)
        newimage = scipy.signal.convolve2d(image, kernel, mode='same')
        return newimage

