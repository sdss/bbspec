#!/usr/bin/env python

"""
Base class for 2D PSFs

Stephen Bailey, Summer 2011
"""

import numpy as N
import scipy.sparse
import pyfits

from bbspec.util import MinMax

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
        
        #- Open fits file
        fx = pyfits.open(filename)
        
        #- Load basic dimensions
        self.npix_x = int(fx[0].header['NPIX_X'])
        self.npix_y = int(fx[0].header['NPIX_Y'])
        self.nspec  = int(fx[0].header['NSPEC'])
        self.nflux  = int(fx[0].header['NFLUX'])
        self._iflux = N.arange(self.nflux)  #- convenience index array

        #- Load individual parameters
        self.param = dict(_iflux = N.tile(self._iflux, self.nspec).reshape(self.nspec, self.nflux))
        for i in range(len(fx)):
            name = fx[i].header['PSFPARAM'].strip()
            self.param[name] = fx[i].data
        fx.close()

    #-------------------------------------------------------------------------
    #- Evaluate the PSF into pixels
    def pix(self, ispec, iflux, xyrange=False):
        """
        Evaluate PSF for a given spectrum and flux bin
        
        returns pixels[yslice, xslice] or
                xrange, yrange, pixels[yslice, xslice] if xyrange is True
        """
        raise NotImplementedError

    def xyrange(self, spec_range, flux_range, dx=8, dy=5):
        """
        Return recommended range of pixels which cover these spectra/fluxes:
        (xmin, xmax, ymin, ymax)
        
        spec_range = indices specmin, specmax
        flux_range = indices fluxmin, fluxmax
        dx, dy = amount of extra overlap
        """
        specmin, specmax = spec_range
        fluxmin, fluxmax = flux_range
        iiflux = range(fluxmin, fluxmax)

        def minmax(t):
            return t.min(), t.max()

        #- Find x/y min/max within range of spectra and fluxes
        xmin, xmax = minmax( self.x()[specmin:specmax, fluxmin:fluxmax] )
        ymin, ymax = minmax( self.y()[specmin:specmax, fluxmin:fluxmax] )

        #- Add borders, watching out for boundaries
        xmin = max(xmin-dx, 0)
        xmax = min(xmax+dx, self.npix_x)
        ymin = max(ymin-dy, 0)
        ymax = min(ymax+dy, self.npix_y)
        
        #+ Make them integers (could round instead of truncate...)
        xypix = map(int, (xmin, xmax, ymin, ymax))
        
        return xypix

    #-------------------------------------------------------------------------
    #- Resample PSF to a new wavelength grid
    def resample(self, loglam):
        """
        Resample psf to a new wavelength grid
        
        loglam: 1D array of log10(angstroms) to use for all spectra
        """
        assert loglam.ndim == 1
        
        #- Copy loglam, in case it is a slice of self.param['LogLam']
        loglam = N.array(loglam)
        
        #- Interpolate flux bins for this new loglam array
        orig_loglam = self.param['LogLam']      #- 2D[nspec, nflux]
            
        #- Resample parameters to this new grid
        self.nflux = loglam.size
        keys = self.param.keys()  #- Cache because loop modifies self.param
        for param in keys:
            #- Some PSF subclasses may keep parameters of other shapes; skip
            if self.param[param].shape != orig_loglam.shape:
                pass
                
            #- resample parameters to new grid
            p = self.param[param]
            tmp = N.zeros( (self.nspec, loglam.size) )
            for i in range(self.nspec):
                tmp[i] = N.interp(loglam, orig_loglam[i], p[i])
            self.param[param] = tmp
                
        #- _iflux should be just indices, not resampled
        self._iflux = N.arange(self.nflux)
        self.param['_iflux'] = N.tile(self._iflux, self.nspec).reshape(self.nspec, self.nflux)
    
    #-------------------------------------------------------------------------
    #- Shift PSF to a new x,y,loglam grid
    def shift_xy(self, x, y, loglam):        
        """
        Shift the x,y trace locations of this PSF, preserving the original
        loglam grid.  Inputs are 2D x, y, loglam grids, e.g. from an
        spCFrame file.  y is optionally 1D instead of 2D.
        """
        if y.ndim == 1:
            ny = len(y)
            y = N.tile(y, self.nspec).reshape(self.nspec, ny)
        assert x.shape == y.shape
        assert x.shape == loglam.shape
        
        for i in range(self.nspec):
            self.param['X'][i] = N.interp(self.loglam(i), loglam[i], x[i])
            self.param['Y'][i] = N.interp(self.loglam(i), loglam[i], y[i])
        
    
    #-------------------------------------------------------------------------
    #- Interpolate flux bins
    def _iflux2D(self, iflux=None, loglam=None, y=None):
        """
        Return a 2D interpolated iflux array [nspec, n]
        
        One of {iflux, loglam, y} must be 1D or 2D array
        
        TODO: Add more robust parameter checking
        """
        #- If iflux is not None, expand to 2D if needed and return that
        if iflux is not None:
            if iflux.ndim == 1:
                return N.tile(iflux, self.nspec).reshape(self.nspec, iflux.size)
            elif iflux.ndim == 2:
                return iflux
            else:
                raise ValueError, "iflux must be 1D or 2D array"
                
        #- Otherwise, interpolate loglam or y to iflux array
        elif loglam is not None:
            param = "LogLam"
            xx = loglam
        elif y is not None:
            param = "Y"
            xx = y
        else:
            raise ValueError, "iflux, loglam, or y required"
            
        result = list()
        xp = self.param[param]
        if xx.ndim == 1:
            for ispec in range(self.nspec):
                result.append(N.interp(xx, xp[ispec], self._iflux))
        elif xx.ndim == 2:
            assert xx.shape[0] == self.nspec
            for ispec in range(self.nspec):
                result.append(N.interp(xx[ispec], xp[ispec], self._iflux))
        else:
            raise ValueError, "loglam/y input must be 1D or 2D array"

        return N.array(result)

    #-------------------------------------------------------------------------
    #- Accessing PSF parameters
    def eval_param(self, param, ispec=None, iflux=None, loglam=None, y=None):
        """
        Interpolate parameter for spectrum ispec at flux bin iflux
        (possibly a float not int), loglam, or y.  If all 3 are None,
        return parameter array for ispec evaluated at all flux bins.
        
        Case ispec  iflux/loglam/y  Output
        -----------------------------
         1.  None       None          2D
         2.  None      scalar        Error
         3.  None        1D           2D
         4.  None        2D           2D
         5.  scalar     None          1D
         6.  scalar    scalar       scalar
         7.  scalar      1D           1D
         8.  scalar      2D          Error
        """
        
        #- Cases 1 and 5: iflux == loglam == y == None
        if iflux is None and loglam is None and y is None:
            if ispec is None:
                return self.param[param]
            else:
                return self.param[param][ispec]
           
        #- Case 2,3,4: ispec is None, iflux/loglam/y is not None
        #- Cases 2 (scalar iflux/loglam/y) should raise error along the way
        if ispec is None:
            iflux2D = self._iflux2D(iflux, loglam, y)
            result = list()
            p = self.param[param]
            for i in range(self.nspec):
                result.append(N.interp(iflux2D[i], self._iflux, p[i]))
            return N.array(result)
                        
        #- Cases 6, 7, 8: ispec is scalar and iflux/loglam, or y is given
        else:
            p = self.param[param][ispec]
            if iflux is not None:
                if type(iflux) == int:
                    return p[iflux]
                else:
                    return N.interp(iflux, self._iflux, p)
            elif loglam is not None:
                #--- DEBUG ---
                ### import pdb; pdb.set_trace()
                #--- DEBUG ---
                return N.interp(loglam, self.param['LogLam'][ispec], p)
            elif y is not None:
                return N.interp(y, self.param['Y'][ispec], p)
                
    def x(self, ispec=None, iflux=None, loglam=None, y=None):
        """
        Return CCD X centroid of spectrum[ispec] at iflux, loglam, or y.
        
        If only ispec is given, return x for every flux bin of ispec.
        """
        return self.eval_param('X', ispec=ispec, iflux=iflux, loglam=loglam, y=y)
    
    def y(self, ispec=None, iflux=None, loglam=None, y=None):
        """
        Returns CCD Y centroid of spectrum[ispec] evaluated at iflux or loglam.

        If y is not None, just return y (useful when passing y, iflux, and
        loglam without needing to check which one is not None).
        
        If only ispec is given, return Y array for every flux bin of ispec.
        """
        if y is not None:
            return y
        else:
            return self.eval_param('Y', ispec=ispec, iflux=iflux, loglam=loglam, y=y)

    def loglam(self, ispec=None, iflux=None, loglam=None, y=None):
        """
        Return log10(wavelength) of spectrum[ispec] evaluated at iflux or y.
        
        If only ispec is given, return Y array for every flux bin of ispec.
        """
        if loglam is not None:
            return loglam
        else:
            return self.eval_param('LogLam', ispec=ispec, iflux=iflux, loglam=loglam, y=y)
    
    def iflux(self, ispec=None, iflux=None, loglam=None, y=None):
        if iflux is not None:
            return iflux
        else:
            return self.eval_param('_iflux', ispec=ispec, iflux=iflux, loglam=loglam, y=y)
    
    #-------------------------------------------------------------------------
    #- Access the projection matrix A
    #- pix = A * flux
    
    def _reslice(self, xslice, yslice, data, xy_range):
        """
        shift xslice, yslice, and subsample data to match an xy_range,
        taking boundaries into account.
        """
    
        #- Check boundaries
        xmin, xmax, ymin, ymax = xy_range
        if xslice.start < xmin:
            dx = xmin - xslice.start
            xslice = slice(xmin, xslice.stop)
            data = data[:, dx:]
        if xslice.stop > xmax:
            dx = xslice.stop - xmax
            xslice = slice(xslice.start, xmax)
            data = data[:, :-dx]
        if yslice.start < ymin:
            dy = ymin - yslice.start
            yslice = slice(ymin, yslice.stop)
            data = data[dy:, :]
        if yslice.stop > ymax:
            dy = yslice.stop - ymax
            yslice = slice(yslice.start, ymax)
            data = data[:-dy, :]
            
        #- Shift slices    
        xslice = slice(xslice.start-xmin, xslice.stop-xmin)
        yslice = slice(yslice.start-ymin, yslice.stop-ymin)
    
        return xslice, yslice, data
        
    
    def getAx(self, spec_range, flux_range, pix_range):
        """
        Returns sparse projection matrix from flux to pixels
    
        Inputs:
            spec_range = (ispecmin, ispecmax)
            flux_range = (ifluxmin, ifluxmax)
            pix_range  = (xmin, xmax, ymin, ymax)
        """

        #- Matrix dimensions
        specmin, specmax = spec_range
        fluxmin, fluxmax = flux_range
        xmin, xmax, ymin, ymax = pix_range        
        nspec = specmax - specmin
        nflux = fluxmax - fluxmin
        nx = xmax - xmin
        ny = ymax - ymin
    
        #- Generate A
        A = N.zeros( (ny*nx, nspec*nflux) )
        tmp = N.zeros((ny, nx))
        for ispec in range(specmin, specmax):
            for iflux in range(fluxmin, fluxmax):
                #- Get subimage and index slices
                xslice, yslice, pix = self.pix(ispec, iflux, xyrange=True)
                xslice, yslice, pix = self._reslice(xslice, yslice, pix, pix_range)                
                tmp[yslice, xslice] = pix
            
                #- put them into sub-region in A
                ij = (ispec-specmin)*nflux + iflux-fluxmin
                A[:, ij] = tmp.ravel()
                tmp[yslice, xslice] = 0.0
    
        return scipy.sparse.csr_matrix(A)    
        
    def spec2pix(self, spectra, ispecmin=0, ifluxmin=0, xyrange=None):
        """
        Project spectra through psf to produce image
        
        spectra[nspec, nflux] : 2D array of spectra
        ispecmin, ifluxmin : lower indices of spectra and flux
        xyrange : pixels (xmin, xmax, ymin, ymax) to consider
        
        See also: PSFBase.xyrange(spec_range, flux_range)
        """

        nspec, nflux = spectra.shape

        if xyrange is None:
            spec_range = (ispecmin, ispecmin+nspec)
            flux_range = (ifluxmin, ifluxmin+nflux)
            xyrange = self.xyrange(spec_range, flux_range)
            
        xmin, xmax, ymin, ymax = xyrange
        nx = xmax-xmin
        ny = ymax-ymin
        
        #- Project spectra into the image
        image = N.zeros( (ny, nx) )
        for ispec in range(nspec):
            for iflux in range(nflux):
                if spectra[ispec, iflux] == 0.0:
                    continue

                xslice, yslice, pix = self.pix(ispec+ispecmin, iflux+ifluxmin, xyrange=True)
                xslice, yslice, pix = self._reslice(xslice, yslice, pix, xyrange)
                image[yslice, xslice] += pix * spectra[ispec, iflux]
                
        return image        
