#!/usr/bin/env python

"""
Classes for extraction.  When this grows too large, split into
a separate subdir product/module
"""

import sys
from time import time

import numpy as N
import scipy.sparse
from scipy.sparse import spdiags

from bbspec.spec2d import resolution_from_icov


class Extractor(object):
    """Base class for extraction objects to define interface"""
    def __init__(self, image, image_ivar, psf):
        """
        Create extraction object from input image, inverse variance ivar,
        and psf object.  Call extract() method to actually do the
        extraction.
        """
        self._image = image
        self._image_ivar = image_ivar
        self._psf = psf
        
        #- Create blank arrays to be filled
        self._spectra = N.zeros( (psf.nspec, psf.nflux) )
        self._ivar = N.zeros( (psf.nspec, psf.nflux) )
        self._deconvolved_spectra = N.zeros( (psf.nspec, psf.nflux) )
        self._model = N.zeros(image.shape)
        self._bkg_model = N.zeros(image.shape)
                
    def extract(self):
        """
        Perform the extraction and fill in output arrays.
        
        To do (maybe): Add options for sub-region extraction.
        
        Sub-classes should implement this.
        """
        raise NotImplementedError
                
    #- python properties are a way to implement read-only attributes
    #- for objects.  The following code is a way to put in the base
    #- class that all Extractors should have X.spectra, etc. without
    #- actually implementing it.  Subclasses should provide methods
    #- which actually fill in X._spectra, etc.
        
    @property
    def spectra(self):
        """
        Extracted spectra re-reconvolved with resolution matrix
        spectra[nspec, nflux]
        """
        return self._spectra
        
    @property
    def ivar(self):
        """
        Inverse variance of extracted reconvolved spectra
        ivar[nspec, nflux]
        """
        return self._ivar
        
    @property
    def resolution(self):
        """
        Resolution matrix (R) of extracted spectra
        resolution[n, n] where n = nspec * nflux
        """
        return self._resolution
        
    @property
    def deconvolved_spectra(self):
        """
        Deconvolved extracted spectra, before convolving with resolution matrix
        deconvolved_spectra[nspec, nflux]
        """
        return self._deconvolved_spectra
        
    @property
    def dspec_icov(self):
        """
        Inverse covariance of deconvolved spectra
        dspec_icov[n, n] where n = nspec * nflux
        """
        return self._dspec_icov

    @property
    def model(self):
        """
        Model image from extracted spectra
        """
        return self._model

    @property
    def bkg_model(self):
        """
        Model image from extracted spectra
        """
        return self._bkg_model

#-------------------------------------------------------------------------

_t0 = None
def _timeit(comment=None):
    
    ### return ###
    
    global _t0
    if _t0 is None:
        if comment is not None:
            print comment
            pass
        _t0 = time()
    else:
        t1 = time()
        if comment is not None:
            print '%-25s : %.1f' % (comment, t1-_t0)
            pass
        _t0 = t1

def _sparsify(A):
    yy, xx = N.where(A != 0.0)
    vv = A[yy, xx]
    Ax = scipy.sparse.coo_matrix((vv, (yy, xx)), shape=A.shape)
    return Ax


class SimpleExtractor(Extractor):
    """Simple brute-force non-optimized extraction algorithm"""
    def __init__(self, image, image_ivar, psf):
        """
        Create extraction object from input image, inverse variance ivar,
        and psf object.  Call extract() method to actually do the
        extraction.
        """ 
        #- Call Extractor.__init__ to actually do the initialization
        super(SimpleExtractor, self).__init__(image, image_ivar, psf)
        
    def _add_legendre(self, A, nx, ny, maxorder=5):
        """
        extend projection matrix A with terms for scattered light
        
        returns Ax[npix, nflux+maxorder+1]
        """
        #- Try legendre polynomials in x-direction only
        x = N.tile(N.linspace(-1, 1, nx), ny)
        
        B = N.zeros( (ny*nx, maxorder+1) )
        for i in range(maxorder+1):
            B[:, i] = scipy.special.legendre(i)(x)
            
        Ax = scipy.sparse.hstack( (A, B) )
        return Ax

    def extract_subregion(self, speclo, spechi, fluxlo, fluxhi, bkgorder=None,
        results_queue=None):
        """
        Extract just a subset of the spectra instead of all of them
        """

        #- Find which pixels are covered by these spectra
        psf = self._psf
        xlo = max(0, int(psf.param['X'][speclo, fluxlo:fluxhi].min()))
        xhi = min(psf.npix_x, int(psf.param['X'][spechi-1, fluxlo:fluxhi].max()))
        ylo = max(0, int(psf.param['Y'][speclo:spechi, fluxlo].min()))
        yhi = min(psf.npix_y, int(psf.param['Y'][speclo:spechi, fluxhi-1].max()))
        
        #- Experimental: Add some boundary in x and y
        xlo = max(0, xlo-8)
        xhi = min(xhi+8, psf.npix_x)
        
        ylo = max(0, ylo-5)
        yhi = min(yhi+5, psf.npix_y)
        
        nx = xhi - xlo
        ny = yhi - ylo

        #- Extract a larger range in wavelength, then trim down later
        dflo = min(fluxlo, 10)
        dfhi = min(psf.nflux-fluxhi, 10)
        
        nspec = spechi - speclo
        nflux = fluxhi - fluxlo
                
        #- Get sparse projection matrix A
        A = psf.getSparseAsub(speclo, spechi, fluxlo-dflo, fluxhi+dfhi, \
                              xlo, xhi, ylo, yhi)
        image = self._image[ylo:yhi, xlo:xhi]
        ivar = self._image_ivar[ylo:yhi, xlo:xhi]

        #- TEST : add background legendre polynomial
        if bkgorder >= 0:
            A = self._add_legendre(A, nx, ny, maxorder=bkgorder)

        #- Noise weight A and pixels
        #- Ni = (N)^-1
        Ni = spdiags(N.sqrt(ivar.ravel()), [0,], A.shape[0], A.shape[0])
        NiA = Ni.dot(A)
        p = (image*N.sqrt(ivar)).ravel()
        
        #- Spectra extend beyond pixels, so add a normalization term to
        #- avoid singularity: unconstrained spectra should -> 0
        npix, nfluxbins = A.shape
        I = 1e-6 * scipy.sparse.identity(nfluxbins)
        
        NiAx = scipy.sparse.vstack((NiA, I))
        Ax = scipy.sparse.vstack((A, I))
        p = N.concatenate( (p, N.zeros(nfluxbins)) )
        
        ATA = Ax.T.dot( NiAx )
        ATA = N.array(ATA.todense())
        try:
            ATAinv = N.linalg.inv(ATA)
        except N.linalg.LinAlgError, e:
            print >> sys.stderr, e
            print >> sys.stderr, "ERROR: Can't invert matrix"
            return
            
        xspec0 = N.array( ATAinv.dot( Ax.T.dot(p) ) )

        #- Project just the bkg model part
        if bkgorder > 0:
            bkg = N.zeros(xspec0.size)
            bkg[-bkgorder-1:] = xspec0.ravel()[-bkgorder-1:]
            bkg_model = A.dot(bkg).reshape(ny, nx)
            self._bkg_model[ylo:yhi, xlo:xhi] = bkg_model
                    
        xspec0 = xspec0.reshape( (nspec, nflux+dflo+dfhi) )
        
        #- Reconvolve flux by resolution
        R = resolution_from_icov(ATA)
        xspec1 = N.dot(R, xspec0.ravel()).reshape(xspec0.shape)

        #- Variance of extracted reconvolved spectrum
        Cx = N.dot(R, N.dot(ATAinv, R.T))  #- Bolton & Schelgel 2010 Eqn 15
        xspec_var = N.array([Cx[i,i] for i in range(Cx.shape[0])])
        xspec_ivar = (1.0/xspec_var).reshape(xspec1.shape)
        
        #- Project model, including bkg terms
        model = A.dot(xspec0.ravel()).reshape(ny, nx)
        
        #- Rename working variables for output
        ### self._spectra[speclo:spechi, fluxlo:fluxhi] = xspec1[:, dflo:dflo+nflux]
        ### self._ivar[speclo:spechi, fluxlo:fluxhi] = xspec_ivar[:, dflo:dflo+nflux]
        ### self._deconvolved_spectra[speclo:spechi, fluxlo:fluxhi] = xspec0[:, dflo:dflo+nflux]
        ### self._model[ylo:yhi, xlo:xhi] = model

        result = dict()
        result['spectra'] = xspec1[:, dflo:dflo+nflux]
        result['ivar'] = xspec_ivar[:, dflo:dflo+nflux]
        result['deconvolved_spectra'] = xspec0[:, dflo:dflo+nflux]
        result['image_model'] = model
        result['speclo'] = speclo
        result['spechi'] = spechi
        result['fluxlo'] = fluxlo
        result['fluxhi'] = fluxhi
        result['xlo'] = xlo
        result['xhi'] = xhi
        result['ylo'] = ylo
        result['yhi'] = yhi
        
        self.update_subregion(result)
        if results_queue is not None:
            results_queue.put(result)
        
        return result
                                
        #- What do I fill in for these?
        # n = psf.nspec * psf.nflux
        # self._dspec_icov = ATA.reshape( (n, n) )
        # self._resolution = R

    def update_subregion(self, results):
        """
        Fill in internal variables for spectra, model, etc. using the
        results dictionary returned by extract_subregion().  This is
        used internally by extract_subregion, and can be used externally
        as a parallel python callback.
        """
        
        xlo, xhi = results['xlo'], results['xhi']
        ylo, yhi = results['ylo'], results['yhi']
        speclo, spechi = results['speclo'], results['spechi']
        fluxlo, fluxhi = results['fluxlo'], results['fluxhi']
        
        ### print "updating [%d:%d, %d:%d]" % (ylo, yhi, xlo, xhi)
        
        self._model[ylo:yhi, xlo:xhi] = results['image_model']
        self._spectra[speclo:spechi, fluxlo:fluxhi] = results['spectra']
        self._deconvolved_spectra[speclo:spechi, fluxlo:fluxhi] = results['deconvolved_spectra']
        self._ivar[speclo:spechi, fluxlo:fluxhi] = results['ivar']        

    def extract(self):
        """Actually do the extraction"""
        print "Extracting"
        
        if N.all(self._image_ivar == 0.0):
            print >> sys.stderr, "ERROR: input ivar all 0"
            return

        #- shortcuts
        psf = self._psf
        image = self._image
        ivar = self._image_ivar
        sqrt_ivar = N.sqrt(ivar).ravel()
        p = (image.ravel() * sqrt_ivar)
            
        #- Generate the matrix to solve
        _timeit()
        Ax = psf.getSparseA()
        ### _timeit('Get Sparse A')
        
        #- construct sparse diagonal noise matrix
        ixy = N.arange(len(sqrt_ivar))
        npix = psf.npix_x * psf.npix_y
        Ni = scipy.sparse.coo_matrix( (sqrt_ivar, (ixy, ixy)), shape=(npix, npix) )
        
        NiAx = Ni.dot(Ax)
        ATAx = Ax.T.dot(NiAx)
        _timeit('Make ATAx')
        
        ATA = N.array(ATAx.todense())
        try:
            ATAinv = N.linalg.inv(ATA)   
        except N.linalg.LinAlgError, e:
            print >> sys.stderr, e
            print >> sys.stderr, "ERROR: Can't invert matrix"
            return
            
        _timeit('Invert ATA')

        #- Solve for flux
        xspec0 = N.array(ATAinv.dot(Ax.T.dot(p)))  #- Sparse        
        xspec0 = xspec0.reshape( (psf.nspec, psf.nflux) )
        _timeit('Project pixels to flux')
                
        #- Reconvolve flux by resolution
        #- Pulling out diagonal blocks doesn't seem to work (for numerical reasons?),
        #- so use full A.T A matrix
        R = resolution_from_icov(ATA)
        ### _timeit('Find R')
        xspec1 = N.dot(R, xspec0.ravel()).reshape( (psf.nspec, psf.nflux) )
        ### _timeit('Reconvolve')

        #- Variance of extracted reconvolved spectrum
        Cx = N.dot(R, N.dot(ATAinv, R.T))  #- Bolton & Schelgel 2010 Eqn 15
        xspec_ivar = (1.0/Cx.diagonal()).reshape(xspec1.shape)
        _timeit('Calculate Variance')
        
        #- Rename working variables for output
        self._spectra = xspec1
        self._ivar = xspec_ivar
        self._deconvolved_spectra = xspec0
        n = psf.nspec * psf.nflux
        self._dspec_icov = ATA.reshape( (n, n) )
        self._resolution = R
