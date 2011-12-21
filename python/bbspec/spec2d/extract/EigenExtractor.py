#!/usr/bin/env python

"""
Extract using eigen decomposition.  Needs a better name.

Stephen Bailey, Summer 2011
"""

import sys

import numpy as N
import scipy.sparse
from scipy.sparse import spdiags
from time import time

from bbspec.spec2d.extract import BaseExtractor
from bbspec.spec2d import ResolutionMatrix
from bbspec.spec2d import Spectra

class EigenExtractor(BaseExtractor):
    def __init__(self, image, image_ivar, psf):
        BaseExtractor.__init__(self, image, image_ivar, psf)
    
    def extract(self, specminmax, fluxminmax=None, fluxstep=100, psfradius=10):
        """
        Extract a range of spectra and return a Spectrum object
        """
        
        #+ Sanity check on bounds vs. psf.nflux, psf.nspec, etc.
        
        if fluxminmax is None:
            fluxminmax = (0, self._psf.nflux)
            
        #- notational convenience shortcuts
        specmin, specmax = specminmax
        fluxmin, fluxmax = fluxminmax
        
        nspec = specmax - specmin
        loglam = self._psf.loglam()[specmin:specmax, fluxmin:fluxmax]
        spectra = Spectra.blank(nspec, loglam, ispecmin=specmin, ifluxmin=fluxmin)
        fluxstart = None
        
        for fluxlo in range(fluxmin, fluxmax-fluxstep+1, fluxstep):
            fluxhi = min(fluxlo+fluxstep, fluxmax)
            
            dlo = psfradius if fluxstart is None else len(fluxstart)
            dhi = psfradius
            
            fmin = max(0, fluxlo - dlo)
            fmax = min(fluxhi + dhi, fluxmax)
            
            s = self.subextract(specminmax, fluxminmax = (fmin, fmax),
                                fluxlohi = (fluxlo, fluxhi), fluxstart=fluxstart )
            spectra.merge(s)
            
            ## fluxstart = spectra.xflux[specmin:specmax, fluxhi-psfradius:fluxhi]
        
        return spectra
        
    def subextract(self, specminmax, fluxminmax, fluxlohi, fluxstart=None):
        """
        Extract a subregion, return a Spectrum object for that subregion
        
        specminmax : (min, max) range of spectra to extract
        fluxminmax : (min, max) full range of flux to extract
        fluxlohi   : (lo, hi) subregion of flux to trust; return spectra for this
        fluxstart  : array to constrain initial part of output flux to
                     ensure cascading consistency
        """
        timeit('quiet_start')
        specmin, specmax = specminmax
        fluxmin, fluxmax = fluxminmax
        fluxlo, fluxhi = fluxlohi

        #- Wavelength range to solve
        loglam = self._psf.loglam()[specmin:specmax, fluxlo:fluxhi]
        
        #- Find pixel boundaries
        # xypix = self._psf.xyrange(specminmax, fluxminmax, dx=8, dy=5)
        xypix = self._psf.xyrange(specminmax, fluxlohi, dx=8, dy=5)
        xmin, xmax, ymin, ymax = xypix
        
        #- Get inputs for solver
        A = self._psf.getAx(specminmax, fluxminmax, pix_range=xypix)
        pix = self._image[ymin:ymax, xmin:xmax]
        pix_ivar = self._image_ivar[ymin:ymax, xmin:xmax]
        ### timeit('A')
        
        #- Actually do the extraction
        xflux, flux, ivar, R = _solve(A, pix, pix_ivar, fluxstart=fluxstart)

        #- Reshape matrices
        nspec = specmax-specmin
        nflux = fluxmax - fluxmin
        xflux = xflux.reshape(nspec, nflux)
        flux  = flux.reshape(nspec, nflux)
        ivar  = ivar.reshape(nspec, nflux)

        #+ convert R into a ResolutionMatrix object
        R[R<1e-9] = 0.0  #- trim really small numbers so they don't enter
        Rx = list()
        for i in range(nspec):
            ii = slice(i*nflux, (i+1)*nflux)
            Rx.append(ResolutionMatrix.from_array(R[ii, ii], \
                full_range=fluxminmax, good_range=fluxlohi))
        
        #- Hang onto full flux range before trimming
        # fullflux = flux
        fullflux = None
        
        #- Get just the subset we trust
        dfluxlo = (fluxlo-fluxmin)
        nflux = (fluxhi-fluxlo)
        
        xflux = xflux[:, dfluxlo:dfluxlo+nflux]
        flux = flux[:, dfluxlo:dfluxlo+nflux]
        ivar = ivar[:, dfluxlo:dfluxlo+nflux]
        
        spectra = Spectra(flux, ivar, loglam, R=Rx,
                            xflux=xflux, fullflux=fullflux,
                            ispecmin=specmin, ifluxmin=fluxlo)
                          
                
        timeit('SubExtract [%d:%d, %d:%d]' % \
            (specminmax[0], specminmax[1], fluxminmax[0], fluxminmax[1]) )
        return spectra

tx = None
def timeit(name):
    ### return  #- don't print timing info
    global tx
    if name == 'start':
        tx = time()
        print '---'
    if name == 'quiet_start':
        tx = time()
    else:
        t = time()
        print '%-20s %.2f' % (name, t-tx)
        tx = t
        
def _solve(A, pix, pix_ivar, regularize=1e-3, fluxstart=None):
    """
    Solve pix[npix] = A[npix, nflux] * f[nflux] + noise[npix]

    A[npix, nflux] : projection matrix (possibly sparse)
    pix[ny, nx]
    pix_ivar[ny, nx]     : 1.0 / noise^2

    Return xflux, flux, ivar, R
    """

    #- Save input array shapes
    npix, nflux = A.shape
    ny, nx = pix.shape

    #- Flatten arrays
    pix = pix.ravel()
    pix_ivar = pix_ivar.ravel()

    #- Weight by inverse noise
    Ninv = scipy.sparse.spdiags(pix_ivar, [0,], npix, npix)
    Ninv_A = Ninv.dot(A)
    wpix = pix * pix_ivar

    #- Add regularization term to avoid singularity and excessive ringing
    if regularize > 0:
        I = regularize * scipy.sparse.identity(nflux)

        Ninv_Ax = scipy.sparse.vstack((Ninv_A, I))
        Ax = scipy.sparse.vstack((A, I))
        wpix = N.concatenate( (wpix, N.zeros(nflux)) )
    else:
        Ninv_Ax = Ninv_A
        Ax = A

    #- Get eigenvector decomposition
    Cinv = Ax.T.dot(Ninv_Ax).toarray()
    ### timeit('setup')
    w, V = N.linalg.eigh(Cinv)  #- w=eigenvalues. V=eigenvectors, 
    ### timeit('eigh')

    #- Find Cinv = QQ
    nw = len(w)
    Wx = spdiags(N.sqrt(w), 0, nw, nw)
    Q = V.dot( Wx.dot(V.T) )  #- dense multiply; kinda slow
    ### timeit('Q')
    
    #- Normalize Q -> R
    norm_vector = N.sum(Q, axis=1)    
    R = N.outer(norm_vector**(-1), N.ones(norm_vector.size)) * Q
    ### timeit('R')
    
    #- Get ivar from normalization vector (Bolton & Schlegel 2010 Eqn 13)
    ivar = norm_vector**2
    
    #- Brute inverse to get xflux
    ### C = N.linalg.inv(Cinv)
    ### xflux = C.dot(Ax.T.dot(wpix))
    
    #- Get extracted spectrum from eigenvectors
    Winv = spdiags(1.0/w, 0, nw, nw)
    xflux = V.dot(Winv.dot(V.T.dot(Ax.T.dot(wpix)).T))
    
    #- Convolve with resolution matrix
    flux = R.dot(xflux)
    ### timeit('flux')
        
    return xflux, flux, ivar, R
