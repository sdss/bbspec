#!/usr/bin/env python

"""
Gauss-Hermite 2D PSF

Adam Bolton & Parul Pandey, Summer 2011
"""

import numpy as N
import pyfits
from bbspec.spec2d.psf import PSFBase
from bbspec.spec2d import pgh
from bbspec.spec2d.boltonfuncs import PGH

class PSFGaussHermite2D(PSFBase):
    """
    Pixel-integrated probabilist-form 2D Gauss-Hermite PSF.
    Currently hard-coded to fourth order.
    (A. Bolton, U. of Utah, 2011may)
    """
    def __init__(self, filename):
        """
        Load PSF from file.  See PSFBase doc for details.
        """
        super(PSFGaussHermite2D, self).__init__(filename)
        # Initialize the list of orders
        self.maxorder = 4
        self.pgh = PGH(self.maxorder)
        self.mvalues = []
        self.nvalues = []
        self.ordernames = []
        for thisord in range(self.maxorder+1):
            for thism in range(0, thisord+1):
                thisn = thisord - thism
                thisname = 'PGH(' + str(thism) + ',' + str(thisn) + ')'
                self.mvalues.append(thism)
                self.nvalues.append(thisn)
                self.ordernames.append(thisname)

        #- Pre-calculate the coefficients at all wavelengths
        waverange = N.power(10, self.param['LogLam'])
        for iorder, ordername in enumerate(self.ordernames):
            xname = 'expanded-' + ordername
            self.param[xname] = N.empty(waverange.shape)
            fitcoeff = self.param[self.ordernames[iorder]]
            for ispec in range(self.nspec):
                coeff  = N.poly1d(fitcoeff[ispec])
                self.param[xname][ispec] = coeff(waverange[ispec])
                
    def pix(self, ispec, iflux=None, loglam=None, y=None, xyrange=False):
        """
        Evaluate PSF for a given spectrum and flux bin (or loglam or y)
        
        returns xslice, yslice, pixels[yslice, xslice]
        """
        
        # Extract core parameters:
        xcen = self.x(ispec, iflux=iflux, loglam=loglam, y=y)
        ycen = self.y(ispec, iflux=iflux, loglam=loglam, y=y)
        sigma = self.eval_param('sigma', ispec, iflux=iflux, loglam=loglam, y=y)
        
        logwave = self.param['LogLam']
        # Set half-width and determine indexing variables:
        hw = int(N.ceil(5.*sigma))
        xcr = int(round(xcen))
        ycr = int(round(ycen))
        dxcen = xcen - xcr
        dycen = ycen - ycr
        xmin = max(xcr - hw, 0)
        xmax = min(xcr + hw, self.npix_x - 1)
        ymin = max(ycr - hw, 0)
        ymax = min(ycr + hw, self.npix_y - 1)
        xbase = N.arange(xmin - xcr, xmax - xcr + 1, dtype=float)
        ybase = N.arange(ymin - ycr, ymax - ycr + 1, dtype=float)
        nx, ny = len(xbase), len(ybase)
        # Don't go any further if we've got a null dimension:
        if (nx == 0) or (ny == 0):
            return slice(0, 0), slice(0, 0), N.zeros((0,0), dtype=float)
        # Build the output slices:
        xslice = slice(xmin, xmax+1)
        yslice = slice(ymin, ymax+1)

        # Build the 1D functions:
        # xfuncs = N.zeros((self.maxorder+1, nx), dtype=float)
        # yfuncs = N.zeros((self.maxorder+1, ny), dtype=float)
        # for iorder in range(self.maxorder+1):
        #     xfuncs[iorder] = pgh(xbase, m=iorder, xc=dxcen, sigma=sigma)
        #     yfuncs[iorder] = pgh(ybase, m=iorder, xc=dycen, sigma=sigma)
         
        #- Faster
        xfuncs = self.pgh.eval(xbase, xc=dxcen, sigma=sigma)
        yfuncs = self.pgh.eval(ybase, xc=dycen, sigma=sigma)
        
        # Build the output image:
        outimage = N.zeros((ny, nx), dtype=float)
        for i, name in enumerate(self.ordernames):
            xname = 'expanded-' + name
            c = self.eval_param(xname, ispec, iflux=iflux, loglam=loglam, y=y)
            img = N.outer(yfuncs[self.nvalues[i]],xfuncs[self.mvalues[i]])
            outimage += c * img

        # Pack up and return:
        if xyrange:
            return xslice, yslice, outimage
        else:
            return outimage
