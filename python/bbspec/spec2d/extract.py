#!/usr/bin/env python

"""
Classes for extraction.  When this grows too large, split into
a separate subdir product/module
"""

import sys
from time import time

import numpy as N

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
        
    def extract():
        """
        Perform the extraction and fill in output arrays.
        
        To do (maybe): Add options for sub-region extraction.
        
        Sub-classes should implement this.
        """
        raise NotImplementedError
        
    def _fill_dummy_values(self):
        """
        In the case of bad input, failed extraction, etc., define
        what the default bad extraction values should be.  For now,
        let's try zeros (instead of NaN or raising an exception)
        
        Note: this won't work for subregions
        """
        nspec = self._psf.nspec
        nflux = self._psf.nflux
        nx    = self._psf.npix_x
        ny    = self._psf.npix_y
        
        #- use the same zeros arrays to not waste memory
        blank_spec = N.zeros( (nspec, nflux) )
        self._spectra = blank_spec
        self._ivar = blank_spec
        self._deconvolved_spectra = blank_spec
        
        n = nspec*nflux
        blank_cov = N.zeros( (n,n) )
        self._resolution = blank_cov
        self._dspec_icov = blank_cov
        
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

#-------------------------------------------------------------------------

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

    def extract(self):
        """Actually do the extraction"""
        pass
        
        if N.all(self._image_ivar == 0.0):
            print >> sys.stderr, "ERROR: input ivar all 0"
            self._fill_dummy_values()
            return

        #- shortcuts
        psf = self._psf
        image = self._image
        ivar = self._image_ivar
            
        #- Generate the matrix to solve
        print "Generating A matrix"
        A = N.zeros( (psf.npix_y*psf.npix_x, psf.nspec*psf.nflux) )
        tmp = N.zeros( (psf.npix_y, psf.npix_x) )
        for i in range(psf.nspec):
            for j in range(psf.nflux):
                xslice, yslice, pix = psf.pix(i, j)
                tmp[yslice, xslice] = pix
                ij = i*psf.nflux + j
                A[:, ij] = tmp.ravel()
                tmp[yslice, xslice] = 0.0

        print 'Solving'
        t0 = time()
        sqrt_ivar = N.sqrt(ivar)
        p = (image * sqrt_ivar).ravel()
        ### NiA = N.dot(N.diag(N.sqrt(ivar).ravel()), A)
        NiA = (sqrt_ivar.ravel() * A.T).T  #- much faster
        
        ATA = N.dot(A.T, NiA)

        #- Directly solve by inverting ATA
        try:
            ATAinv = N.linalg.inv(ATA)   
        except N.linalg.LinAlgError:
            print >> sys.stderr, "ERROR: Can't invert matrix"
            self._fill_dummy_values()
            return
                     
        xspec0 = N.dot( ATAinv, N.dot(A.T, p) )
        
        #- Alternate: Solve by LU decomposition with call to N.linalg.solve
        #- Note: not exactly the same solution
        #- Note: doesn't create ATAinv, which we need for reconvolving
        # xspec0 = N.linalg.solve(ATA, N.dot(A.T, p))
        
        t1 = time()
        print '  --> %.1f seconds' % (t1-t0)
        
        xspec0 = xspec0.reshape( (psf.nspec, psf.nflux) )
        

        #- Reconvolve flux by resolution
        #- Pulling out diagonal blocks doesn't seem to work (for numerical reasons?),
        #- so use full A.T A matrix
        print "Reconvolving"
        R = resolution_from_icov(ATA)
        xspec1 = N.dot(R, xspec0.ravel()).reshape( (psf.nspec, psf.nflux) )

        #- Variance of extracted reconvolved spectrum
        Cx = N.dot(R, N.dot(ATAinv, R.T))  #- Bolton & Schelgel 2010 Eqn 15
        xspec_var = N.array([Cx[i,i] for i in range(Cx.shape[0])])
        xspec_ivar = (1.0/xspec_var).reshape(xspec1.shape)
        
        #- Rename working variables for output
        self._spectra = xspec1
        self._ivar = xspec_ivar
        self._deconvolved_spectra = xspec0
        n = psf.nspec * psf.nflux
        self._dspec_icov = ATA.reshape( (n, n) )
        self._resolution = R
