#!/usr/bin/env python

"""
Spectra class to wrap extracted spectra, their resolution matrix, and
their projection back to CCD pixels.

Stephen Bailey, Summer 2011
"""

import sys
import os
import numpy as N
import pyfits
from bbspec.spec2d import ResolutionMatrix
from scipy import sparse

class Spectra(object):
    def __init__(self, flux, ivar, loglam,
                 R=None,
                 xflux=None, ixcovar=None, fullflux=None,
                 ispecmin=0, ifluxmin=0,
                 pix=None, xmin=0, ymin=0):
        """
        Initialize Spectra object

        Required inputs:
            flux   2D array of flux values [nspec, nflux]
            ivar   2D array of flux inverse variances [nspec, nflux]
            loglam 1D or 2D array of log10(wavelength) [nflux] or [nspec,nflux]

        Optional:
            R       : array of ResolutionMatrix objects [nspec]
            xflux   : deconvolved spectra [nspec,nflux]
            ixcovar : inverse covariance of xflux [nspec*nflux, nspec*nflux]
            ispecmin, ifluxmin : if these spectra are subset of the total,
                                 these indices give the offset from 0
                                 
        Use Spectra.blank(loglam) to create a blank Spectrum object
        
        TODO: Incorporate projected spectra -> pixels
        """
        #+ Add array consistency checks; raise exceptions for inconsistencies
        
        #- Basic array dimensions
        nspec, nflux = flux.shape
        self.nspec = nspec
        self.nflux = nflux
        
        #- Required inputs and defaults
        self.flux = flux
        self.ivar = ivar
        if loglam.ndim == 1:
            loglam = N.tile(loglam, nspec).reshape(nspec, nflux)
        self.loglam = loglam
        
        #- Starting indices default 0, but may be >0 to indicate subset
        self.ispecmin = ispecmin
        self.ifluxmin = ifluxmin
        
        #- Optional inputs; these could be None
        self.R = R
        self.xflux = xflux
        self.fullflux = fullflux
        self.ixcovar = ixcovar
        self.pix = pix
        self.xmin = xmin
        self.ymin = ymin
        
    def merge(self, spectra):
        """
        Merge these spectra with values from another Spectra object.
        For overlaps, assumes new Spectra are correct
        """
        i0 = spectra.ispecmin-self.ispecmin
        iispec = slice(i0, i0+spectra.nspec)
        
        i1 = spectra.ifluxmin-self.ifluxmin
        iiflux = slice(i1, i1+spectra.nflux)
        
        self.flux[iispec, iiflux] = spectra.flux
        self.ivar[iispec, iiflux] = spectra.ivar

        if self.xflux is not None:
            self.xflux[iispec, iiflux] = spectra.xflux

        for i in range(spectra.nspec):
            j = (spectra.ispecmin - self.ispecmin) + i
            if self.R[j] is None:
                full_range = self.ifluxmin, self.ifluxmin + self.nflux
                self.R[j] = ResolutionMatrix.blank(bandwidth=15, \
                            nflux=self.nflux, full_range=full_range)
                
            self.R[j].merge(spectra.R[i])
            
        if self.pix is None:
            if spectra.pix is not None:
                self.pix = spectra.pix.copy()
                self.xmin = spectra.xmin
                self.ymin = spectra.ymin
            else:
                pass
        elif spectra.pix is not None:
            xmin = min(self.xmin, spectra.xmin)
            ymin = min(self.ymin, spectra.ymin)
            xmax = max(self.xmax, spectra.xmax)
            ymax = max(self.ymax, spectra.ymax)
            nxtot = xmax-xmin+1
            nytot = ymax-ymin+1
            pix = N.zeros((nytot, nxtot))
            for spec in self, spectra:
                ny, nx = spec.pix.shape
                x0 = spec.xmin - xmin
                y0 = spec.ymin - ymin
                #- Add, not replace pixels
                pix[y0:y0+ny, x0:x0+nx] += spec.pix
            
            self.pix = pix
            self.xmin = xmin
            self.ymin = ymin
            
    @property
    def xmax(self):
        if self.pix is None:
            return None
        else:
            return self.xmin + self.pix.shape[1]
            
    @property
    def ymax(self):
        if self.pix is None:
            return None
        else:
            return self.ymin + self.pix.shape[0]
            
    @property
    def nx(self):
        if self.pix is None:
            return 0
        else:
            return self.pix.shape[1]

    @property
    def ny(self):
        if self.pix is None:
            return 0
        else:
            return self.pix.shape[0]
            
    def calc_model_image(self, psf, xyrange=None):
        """
        Calculate model image using psf object; fill in self.pix array
        
        xyrange = (xmin, xmax, ymin, ymax) in pixels
        """
        spec_range = (self.ispecmin, self.ispecmin+self.nspec)
        flux_range = (self.ifluxmin, self.ifluxmin+self.nflux)
        if xyrange is None:
            xyrange = psf.xyrange(spec_range, flux_range)
        xmin, xmax, ymin, ymax = xyrange
        self.pix = psf.spec2pix(self.xflux, xyrange=xyrange, \
                    ispecmin=self.ispecmin, ifluxmin=self.ifluxmin )
        self.xmin = xmin
        self.ymin = ymin
        
    def write(self, filename, comments=None, clobber=True):
        """
        Write spectra to a fits file
        """
        hdus = list()
        hdus.append( pyfits.PrimaryHDU(self.flux) )
        hdus.append( pyfits.ImageHDU(self.ivar) )
        hdus.append( pyfits.ImageHDU(self.loglam) )
        R = list()
        bandwidth = 15
        for i, Rx in enumerate(self.R):
            if Rx is not None:
                R.append(ResolutionMatrix.to_diagonals(Rx))
            else:
                R.append(N.zeros((bandwidth, self.nflux)))
        R = N.array(R)
        hdus.append( pyfits.ImageHDU(R) )
        
        #- Optional HDUs; data may be blank
        hdus.append( pyfits.ImageHDU(self.xflux) )
        hdus.append( pyfits.ImageHDU(self.ixcovar) )
        hdus.append( pyfits.ImageHDU(self.pix) )
        
        #- Add EXTNAME keywords
        hdus = pyfits.HDUList(hdus)
        hdus[0].update_ext_name('FLUX',   'Extracted re-convolved flux')
        hdus[1].update_ext_name('IVAR',   'Inverse variance of extracted re-convolved flux')
        hdus[2].update_ext_name('LOGLAM', 'log10(wavelength) of extracted spectra')
        hdus[3].update_ext_name('R',      'Resolution matrix')
        hdus[4].update_ext_name('XFLUX',  'Extracted deconvolved spectra')
        hdus[5].update_ext_name('IXCOVAR', 'Inverse Covar of extracted deconvolved flux')
        hdus[6].update_ext_name('PIX',    'Reconstructed CCD image')
        
        #- Add spectra/flux ranges
        hdus[0].header.update('ISPECMIN', self.ispecmin, 'Index of first spectrum')
        hdus[0].header.update('IFLUXMIN', self.ifluxmin, 'Index of first flux bin')
        hdus[0].header.update('NSPEC', self.nspec, 'Number of spectra')
        hdus[0].header.update('NFLUX', self.nflux, 'Number of flux bins')
        
        hdus[6].header.update('CRPIX1', 1, 'Reference pixel (1..n)')
        hdus[6].header.update('CRPIX2', 1, 'Reference pixel (1..n)')
        hdus[6].header.update('CRVAL1', self.xmin, 'Coordinate of first X pixel')
        hdus[6].header.update('CRVAL2', self.ymin, 'Coordinate of first Y pixel')
        
        if comments is not None:
            for c in comments:
                hdus[0].header.add_comment(c)
        
        hdus.writeto(filename, clobber=clobber)

    @staticmethod
    def coadd(spectra):
        """
        Given a list of spectra on the same wavelength grid, coadd them
        and return a new Spectra object with the effective ivar,
        resolution matrices R, etc.
        
        TODO: Not yet implemented
        """
        raise NotImplementedError
        
    @staticmethod
    def load(filename):
        """
        Load Spectrum object from a fits file
        """
        
        fx = pyfits.open(filename)
        flux = fx[0].data.astype('=f8')  #- convert to native endian
        ivar = fx[1].data.astype('=f8')
        loglam = fx[2].data.astype('=f8')
        
        #- spectral/flux ranges
        ispecmin = fx[0].header['ISPECMIN']
        ifluxmin = fx[0].header['IFLUXMIN']
        nspec, nflux = flux.shape
        
        #- Resolution matrix
        full_range = (ifluxmin, ifluxmin+nflux)
        good_range = (ifluxmin, ifluxmin+nflux)
        R = fx[3].data.astype('=f8')
        if R is not None:
            R = ResolutionMatrix.from_diagonals(R, full_range=full_range, good_range=good_range)
        
        #- Optional, these may be blank
        xflux = fx[4].data.astype('=f8')
        pix = fx[6].data.astype('=f8')
        xmin = fx[6].header['CRVAL1']
        ymin = fx[6].header['CRVAL2']
        
        fx.close()
        s = Spectra(loglam=loglam, flux=flux, ivar=ivar, R=R, xflux=xflux,
                     ispecmin=ispecmin, ifluxmin=ifluxmin, 
                     pix=pix, xmin=xmin, ymin=ymin)
        return s

    @staticmethod
    def blank(nspec, loglam, ispecmin=0, ifluxmin=0):
        """
        Create a blank spectra object
        
        nspec  : int, number of spectra
        loglam : 1D numpy array of log10(wavelengths) [nflux], or
                 2D numpy array of log10(wavelengths) [nspec, nflux]
        """
        if loglam.ndim == 1:
            nflux = loglam.size
            loglam = N.tile(loglam, nspec).reshape(nspec, nflux)
        else:
            n, nflux = loglam.shape
            assert n == nspec, 'loglam dimension mismatch %d != %d' % (n,nspec)
        
        flux = N.zeros((nspec, nflux))
        ivar = N.zeros((nspec, nflux))
        xflux = N.zeros((nspec, nflux))
        R = [None, ] * nspec
            
        return Spectra(flux, ivar, loglam, xflux = xflux, R=R,
                       ispecmin=ispecmin, ifluxmin=ifluxmin)
        
    @property
    def wavelength(self):
        return 10**self.loglam
        
    @property
    def w(self):
        return 10**self.loglam
        
        
        
