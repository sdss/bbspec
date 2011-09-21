#!/usr/bin/env python

"""
Classes and functions for manipulating 2D PSFs
"""

import numpy as N
import pyfits
from bbspec.spec2d.psf import PSFBase

class PSFGauss2D(PSFBase):
    """Rotated 2D Gaussian PSF"""
    def __init__(self, filename):
        """
        Load PSF from file.  See PSFBase doc for details.
        """
        super(PSFGauss2D, self).__init__(filename)

    def pix(self, ispec, iflux=None, loglam=None, y=None, xyrange=False):
        """
        Evaluate PSF for a given spectrum and flux bin
        
        returns xslice, yslice, pixels[yslice, xslice]
        """
        #- Original code
        # x0 = self.param['X'][ispec, iflux]
        # y0 = self.param['Y'][ispec, iflux]
        # amplitude = self.param['Amplitude'][ispec, iflux]
        # sig0 = self.param['MajorAxis'][ispec, iflux]
        # sig1 = self.param['MinorAxis'][ispec, iflux]
        # theta = self.param['Angle'][ispec, iflux]
                
        #- Interpolate parameter grid
        x0 = self.eval_param('X', ispec=ispec, iflux=iflux, loglam=loglam, y=y)
        y0 = self.eval_param('Y', ispec=ispec, iflux=iflux, loglam=loglam, y=y)
        sig0 = self.eval_param('MajorAxis', ispec=ispec, iflux=iflux, loglam=loglam, y=y)
        sig1 = self.eval_param('MinorAxis', ispec=ispec, iflux=iflux, loglam=loglam, y=y)
        theta = self.eval_param('Angle', ispec=ispec, iflux=iflux, loglam=loglam, y=y)
        amplitude = self.eval_param('Amplitude', ispec=ispec, iflux=iflux, loglam=loglam, y=y)
        
        #- Define sampling grid
        dxy = int(5*max(sig0, sig1))
        
        # dxy = 11  #- DEBUG: match Ted's size
        
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

        #- Rotated Gaussian, formulas from
        #- http://en.wikipedia.org/wiki/Gaussian_function
        cost2 = N.cos(theta)**2
        sint2 = N.sin(theta)**2
        cos2t = N.cos(2*theta)
        sin2t = N.sin(2*theta)
        a = cost2/(2*sig0**2) + sint2/(2*sig1**2)
        b = sin2t/(4*sig0**2) - sin2t/(4*sig1**2)
        c = sint2/(2*sig0**2) + cost2/(2*sig1**2)
        pix = amplitude * N.exp(-(a*dxx**2 + 2*b*dxx*dyy + c*dyy**2))
        
        #- Renormalize to make amplitude = area
        norm = 2.5066282746310002  #- N.sqrt(2*N.pi)
        pix /= sig0 * norm
        pix /= sig1 * norm

        #- Original Calculation ---
        # #- Rotate coordinate systems
        # cost = N.cos(theta)
        # sint = N.sin(theta)
        # xx =  dxx*cost + dyy*sint
        # yy = -dxx*sint + dyy*cost
        # 
        # #- Evaluate 2D gaussian in rotated system
        # #- Fast, incorrect, non-integration for now
        # norm = 2.5066282746310002  #- N.sqrt(2*N.pi)
        # pix = amplitude * N.exp(-0.5*(xx/sig0)**2) * N.exp(-0.5*(yy/sig1)**2)
        # pix /= sig0*norm
        # pix /= sig1*norm
        #- End Original Calculation ---
        
        xslice = slice(xmin, xmax+1)
        yslice = slice(ymin, ymax+1)
        nx = xslice.stop - xslice.start
        ny = yslice.stop - yslice.start
        pix = pix.reshape( (ny, nx) )
        
        if xyrange:
            return xslice, yslice, pix
        else:
            return pix

