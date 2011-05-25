#!/usr/bin/env python

"""
Classes and functions for manipulating 2D PSFs
"""

import numpy as N
import scipy.sparse
import pyfits
from bbspec.spec2d import pgh

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
        
        self.npix_x = int(fx[0].header['NPIX_X'])
        self.npix_y = int(fx[0].header['NPIX_Y'])
        self.nspec  = int(fx[0].header['NSPEC'])
        self.nflux  = int(fx[0].header['NFLUX'])
        self.param = dict()
        for i in range(len(fx)):
            name = fx[i].header['PSFPARAM'].strip()
            self.param[name] = fx[i].data
        fx.close()

    def pix(self, ispec, iflux):
        """
        Evaluate PSF for a given spectrum and flux bin
        
        returns xslice, yslice, pixels[yslice, xslice]
        """
        raise NotImplementedError
    
    def getA(self):
        """
        return dense matrix A[npix, nspec*nflux] where each column is the
        flattened pixel values for the PSF projected from a given flux bin.
        """
        A = N.zeros( (self.npix_y*self.npix_x, self.nspec*self.nflux) )
        tmp = N.zeros( (self.npix_y, self.npix_x) )
        for i in range(self.nspec):
            for j in range(self.nflux):
                xslice, yslice, pix = self.pix(i, j)
                tmp[yslice, xslice] = pix
                ij = i*self.nflux + j
                A[:, ij] = tmp.ravel()
                tmp[yslice, xslice] = 0.0
        
        return A

    def getSparseA(self):
        """
        Return a sparse version of A matrix (see getA() )
        
        Current implementation creates the full A and then uses that to
        make the sparse array.  Subsequent operations with sparse A are
        fast, but this creation step could be optimized.
        """
        A = self.getA()
        yy, xx = N.where(A != 0.0)
        vv = A[yy, xx]
        Ax = scipy.sparse.coo_matrix((vv, (yy, xx)), shape=A.shape)
        return Ax.tocsr()
        
    def spec2pix(self, spectra):
        """
        Project spectra through psf to product image
        
        Input: spectra[nspec, nflux]
        Output: image[npix_y, npix_x]
        """
        
        #- Check input dimensions
        nspec, nflux = spectra.shape
        if nspec != self.nspec or nflux != self.nflux:
            raise ValueError, "spectra dimensions [%d,%d] don't match PSF dimensions [%d,%d]" % \
                (nspec, nflux, self.nspec, self.nflux)
        
        #- Project spectra into the image
        image = N.zeros( (self.npix_y, self.npix_x) )
        for ispec in range(self.nspec):
            for iflux in range(self.nflux):
                if spectra[ispec, iflux] == 0.0:
                    continue

                xslice, yslice, pixels = self.pix(ispec, iflux)
                image[yslice, xslice] += pixels * spectra[ispec, iflux]
                
        return image
        
        
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

    def pix(self, ispec, iflux):
        """
        Evaluate PSF for a given spectrum and flux bin
        
        returns xslice, yslice, pixels[yslice, xslice]
        """
        # Extract core parameters:
        xcen = self.param['X'][ispec, iflux]
        ycen = self.param['Y'][ispec, iflux]
        sigma = self.param['sigma'][ispec, iflux]
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
        xfuncs = N.zeros((self.maxorder+1, nx), dtype=float)
        yfuncs = N.zeros((self.maxorder+1, ny), dtype=float)
        for iorder in range(self.maxorder+1):
            xfuncs[iorder] = pgh(xbase, m=iorder, xc=dxcen, sigma=sigma)
            yfuncs[iorder] = pgh(ybase, m=iorder, xc=dycen, sigma=sigma)
        # Build the output image:
        outimage = N.zeros((ny, nx), dtype=float)
        for iorder in range(len(self.ordernames)):
            outimage += self.param[self.ordernames[iorder]][ispec, iflux] * \
                        N.outer(yfuncs[self.nvalues[iorder]],
                                xfuncs[self.mvalues[iorder]])
        # Pack up and return:
        return xslice, yslice, outimage
