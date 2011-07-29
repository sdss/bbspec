#!/usr/bin/env python

"""
Classes and functions for manipulating 2D PSFs
"""

import numpy as N
import scipy.signal
import scipy.sparse
import pyfits
from bbspec.spec2d import pgh
from bbspec.spec2d.boltonfuncs import PGH
from time import time

#- Turn of complex -> real warnings in sinc interpolation
import warnings 
try:
	warnings.simplefilter("ignore", N.ComplexWarning)
except   AttributeError:
	pass
	
#- Checkpoint for timing tests
t0 = time()
def checkpoint(comment=None):
    global t0
    tx = time()
    if comment is not None:
        dt = tx - t0
        print "%-30s : %.5f" % (comment, dt)
    t0 = tx

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
    
    def x(self, ispec, iflux=None, loglam=None, y=None):
        """
        Return CCD X centroid of spectrum[ispec] at iflux, loglam, or y.
        
        If only ispec is given, return x for every flux bin of ispec.
        """
        x = self.param['X'][ispec]
        if iflux is not None:
            return x[iflux]
        elif loglam is not None:
            return N.interp(loglam, self.loglam(ispec), x)
        elif y is not None:
            return N.interp(y, self.y(ispec), x)
        else:
            return x
    
    def y(self, ispec, iflux=None, loglam=None):
        """
        Returns CCD Y centroid of spectrum[ispec] evaluated at iflux or loglam.
        
        If only ispec is given, return Y array for every flux bin of ispec.
        """
        y = self.param['Y'][ispec]
        if iflux is not None:
            return y[iflux]
        elif loglam is not None:
            return N.interp(loglam, self.loglam(ispec), y)
        else:
            return y

    def loglam(self, ispec, iflux=None, y=None):
        """
        Return log10(wavelength) of spectrum[ispec] evaluated at iflux or y.
        
        If only ispec is given, return Y array for every flux bin of ispec.
        """
        loglam = self.param['LogLam'][ispec]
        if iflux is not None:
            return loglam[iflux]
        elif y is not None:
            return N.interp(y, self.y(ispec), loglam)
        else:
            return loglam

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
        
    # def getAx(self):
    #     """
    #     Perhaps a more readable version of getA?
    #     """
    #     A = N.zeros( (self.nspec, self.nflux, self.npix_y, self.npix_x) )
    #     for i in range(self.nspec):
    #         for j in range(self.nflux):
    #             xslice, yslice, pix = self.pix(i, j)
    #             A[i, j, yslice, xslice] = pix
    # 
    #     return A.reshape((self.nspec*self.nflux, self.npix_y*self.npix_x)).T
        
    def getAsub(self, speclo, spechi, fluxlo, fluxhi, xlo, xhi, ylo, yhi):
        """
        Return A for a subset of spectra and flux bins
        """
        
        nspec = spechi - speclo
        nflux = fluxhi - fluxlo
        nx = xhi - xlo
        ny = yhi - ylo
        
        #- Generate A
        A = N.zeros( (ny*nx, nspec*nflux) )
        tmp = N.zeros( (self.npix_y, self.npix_x) )  #- full fat version
        for ispec in range(speclo, spechi):
            for iflux in range(fluxlo, fluxhi):
                #- Get subimage and index slices
                xslice, yslice, pix = self.pix(ispec, iflux)
                tmp[yslice, xslice] = pix
                
                #- put them into sub-region in A
                ij = (ispec-speclo)*nflux + (iflux-fluxlo)
                A[:, ij] = tmp[ylo:yhi, xlo:xhi].ravel()
                tmp[yslice, xslice] = 0.0
        
        return A
        
    def getSparseAsub(self, speclo, spechi, fluxlo, fluxhi, xlo, xhi, ylo, yhi):
        A = self.getAsub(speclo, spechi, fluxlo, fluxhi, xlo, xhi, ylo, yhi)
        yy, xx = N.where(A != 0.0)
        vv = A[yy, xx]
        Ax = scipy.sparse.coo_matrix((vv, (yy, xx)), shape=A.shape)
        return Ax.tocsr()
        
    def spec2pix(self, spectra):
        """
        Project spectra through psf to produce image
        
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
        
    def set_xyloglam(self, x, y, loglam):
        """
        Set PSF X, Y, and LogLam grids
        """
        if x.shape != y.shape or y.shape != loglam.shape:
            raise ValueError, 'Parameters must have same shape x%s y%s loglam%s' % \
                (str(x.shape), str(y.shape), str(loglam.shape))
        
        if x.shape[0] != self.nspec:
            raise  ValueError, 'x.shape[0] != nspec=%d' % self.nspec
        
        self.param['X'] = x
        self.param['Y'] = y
        self.param['LogLam'] = loglam
        
    def resample_wavelength(self, wmin, wmax, dloglam):
        """
        Resample PSF from wmin to wmax (Angstroms) on a logarithmic scale
        with dispersion dloglam.
        """
        loglam = N.arange(N.log10(wmin), N.log10(wmax)+dloglam/2, dloglam)
        nflux = len(loglam)
        x = N.zeros((self.nspec, nflux))
        
        
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

#-------------------------------------------------------------------------

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
        self.param = dict()
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
        
        
    def pix(self, ispec, iflux):
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
        x = self.param['X'][ispec, iflux]
        y = self.param['Y'][ispec, iflux]
        
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
        
        return xslice, yslice, psfimage

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

"""
from bbspec.spec2d.psf import load_psf
psf = load_psf('spBasisPSF-r1-00115982.fits')
image = N.zeros((500, 500))
xs, ys, pix = psf.pix(0,200)
image[ys, xs] = pix
imshow(image, interpolation='nearest')

"""

#-------------------------------------------------------------------------

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

        # if ispec == iflux == 0:
        #     #--- DEBUG ---
        #     import pdb; pdb.set_trace()
        #     #--- DEBUG ---

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
        return xslice, yslice, pix.reshape( (ny, nx) )

#-------------------------------------------------------------------------

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
        self.coeffVal = dict()
        for iorder in range(len(self.ordernames)):
            self.coeffVal[iorder] = N.empty(waverange.shape)
            fitcoeff = self.param[self.ordernames[iorder]]
            for ispec in range(self.nspec):
                coeff  = N.poly1d(fitcoeff[ispec])
                self.coeffVal[iorder][ispec] = coeff(waverange[ispec])
                
    def pix(self, ispec, iflux):
        """
        Evaluate PSF for a given spectrum and flux bin
        
        returns xslice, yslice, pixels[yslice, xslice]
        """
        # Extract core parameters:
        xcen = self.param['X'][ispec, iflux]
        ycen = self.param['Y'][ispec, iflux]
        sigma = self.param['sigma'][ispec, iflux]
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
        
        #for iorder in range(len(self.ordernames)):
        #    outimage += self.param[self.ordernames[iorder]][ispec, iflux] * \
        #             N.outer(yfuncs[self.nvalues[iorder]], xfuncs[self.mvalues[iorder]])

        # Corrected to include new fits file struture which does not store the actual value of PSF parameters but the 
        # polynomial coefficients from which they can be obtained.
        for iorder in range(len(self.ordernames)):
            coeffVal = self.coeffVal[iorder]  #- pre-cached coefficients
            outimage += coeffVal[ispec, iflux] * \
                        N.outer(yfuncs[self.nvalues[iorder]],xfuncs[self.mvalues[iorder]])

        # Pack up and return:
        return xslice, yslice, outimage
