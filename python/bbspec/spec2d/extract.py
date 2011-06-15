#!/usr/bin/env python

"""
Classes for extraction.  When this grows too large, split into
a separate subdir product/module
"""

import sys
from time import time
import pyfits

import numpy as N
import scipy.sparse
from scipy.sparse import spdiags

from bbspec.spec2d import resolution_from_icov
from bbspec.spec2d import ResolutionMatrix

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
        
        #- These are big and may not be filled in for certain tests
        self._resolution = None
        self._dspec_icov = None
                
    def extract(self):
        """
        Perform the extraction and fill in output arrays.
        
        To do (maybe): Add options for sub-region extraction.
        
        Sub-classes should implement this.
        """
        raise NotImplementedError

    def writeto(self, filename, comments=None):
        """
        Write the results of this extraction to filename
        """
        print "Writing output to", filename
        hdus = list()
        hdus.append(pyfits.PrimaryHDU(data = self.spectra))             #- 0
        hdus.append(pyfits.ImageHDU(data = self.ivar))                  #- 1
        hdus.append(pyfits.ImageHDU(data = self.resolution))            #- 2
        hdus.append(pyfits.ImageHDU(data = self.deconvolved_spectra))   #- 3
        hdus.append(pyfits.ImageHDU(data = self.dspec_icov))            #- 4
        hdus.append(pyfits.ImageHDU(data = self.model))                 #- 5

        hdus = pyfits.HDUList(hdus)
        hdus[0].header.add_comment('Extracted re-convolved spectra')
        if comments is not None:
            for comment in comments:
                hdus[0].header.add_comment(comment)

        hdus[1].header.add_comment('Inverse variance of extracted re-convolved spectra')
        hdus[2].header.add_comment('Convolution kernel R')
        hdus[3].header.add_comment('Original extracted spectra')
        hdus[4].header.add_comment('Inverse covariance of original extracted spectra')
        hdus[5].header.add_comment('Reconstructed model image')

        hdus.writeto(filename, clobber=True)


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

class Boundaries(object):
    """
    Utility class for calculating pixel and spectral boundaries given
    a spectral range to extract
    """
    def __init__(self, psf, speclo, spechi, fluxlo, fluxhi):
        """
        Given a spectra range and fluxbin range to extract, create an
        object which defines:
        
        - Image pixel range to use [ymin:ymax, xmin:xmax]
        - Which pixel range will be valid for projection [ylo:yhi, xlo:xhi]
        - Spectral range to extract [specmin:specmax, fluxmin:fluxmax]
        - Spectra valid for final answer: inputs [speclo:spechi, fluxlo:fluxhi]
        
        Utility values, which could be derived from the others:
        - pixel boundaries: dylo, dyhi, dxlo, dxhi
        - spectra boundaries: dfluxlo, dfluxhi, dspeclo, dspechi
        
        BUGS:
        - Adds x boundaries assuming edge of bundles without cross talk
        """
    
        #- Inputs
        self.speclo = speclo
        self.spechi = spechi
        self.fluxlo = fluxlo
        self.fluxhi = fluxhi
    
        #- Find which pixels are at the center of these spectra
        cxlo = int(psf.param['X'][speclo,   fluxlo:fluxhi].min())
        cxhi = int(psf.param['X'][spechi-1, fluxlo:fluxhi].max())+1
        cylo = int(psf.param['Y'][speclo:spechi, fluxlo  ].min())
        cyhi = int(psf.param['Y'][speclo:spechi, fluxhi-1].max())+1

        #- Add pixel boundaries to fully cover spectra of interest
        dx = 8
        self.xlo = max(cxlo-dx, 0)
        self.xhi = min(cxhi+dx, psf.npix_x)
        
        #- Assume x boundaries at bundle edges without cross talk
        self.xmin = self.xlo
        self.xmax = self.xhi
        self.specmin = self.speclo
        self.specmax = self.spechi
        
        #- Explicitely calculate the boundary in case we change xlo==xmin, etc.
        self.dxlo = self.xlo  - self.xmin
        self.dxhi = self.xmax - self.xhi
        self.dspeclo = self.speclo  - self.specmin
        self.dspechi = self.specmax - self.spechi
        
        #- For y pixel extension, watch out for edges
        dy = 5
        self.dylo = min(dy, cylo)
        self.dyhi = min(dy, psf.npix_y-cyhi)
        
        self.ylo = cylo
        self.yhi = cyhi
        self.ymin = self.ylo - self.dylo
        self.ymax = self.yhi + self.dyhi

        #- Add extra flux bins to avoid edge effects,
        #- but don't exceed boundaries of the problems
        self.dfluxlo = min(2*dy, self.fluxlo)
        self.dfluxhi = min(2*dy, psf.nflux-self.fluxhi)
        self.fluxmin = self.fluxlo - self.dfluxlo
        self.fluxmax = self.fluxhi + self.dfluxhi

        #- Count things : full ranges
        self.nflux = self.fluxmax - self.fluxmin
        self.nspec = self.specmax - self.specmin
        self.nx = self.xmax - self.xmin
        self.ny = self.ymax - self.ymin
        
        #- Count things: subrange of good pixels/spectra
        self.nsubx = self.xhi - self.xlo
        self.nsuby = self.yhi - self.ylo
        self.nsubspec = self.spechi - self.speclo
        self.nsubflux = self.fluxhi - self.fluxlo
        
        #- Slice objects for convenient
        self.goodpix = (slice(self.ylo, self.yhi), slice(self.xlo, self.xhi))
        self.goodsubpix = (slice(self.dylo, self.dylo+self.nsuby),
                           slice(self.dxlo, self.dxlo+self.nsubx) )
                           
        self.goodspec = (slice(self.speclo, self.spechi),
                         slice(self.fluxlo, self.fluxhi) )
        self.goodsubspec = (slice(self.dspeclo, self.dspeclo+self.nsubspec),
                            slice(self.dfluxlo, self.dfluxlo+self.nsubflux) )
        
    def __str__(self):
        # print 'spec %d-%d flux %d-%d' % (speclo, spechi, flo, fhi)
        x = 'spec %d-%d  flux %3d-%3d  flux+ %3d-%3d' % (self.speclo,
            self.spechi, self.fluxlo, self.fluxhi, self.fluxmin, self.fluxmax)
        x += ' : x %2d-%3d  y %2d-%3d ' % (self.xlo, self.xhi, self.ymin, self.ymax)
        return x
    
#-------------------------------------------------------------------------

class SubExtractor(Extractor):
    """
    Extractor which pieces together extracted sub-regions
    """
    def __init__(self, image, image_ivar, psf):
        """
        Create extraction object from input image, inverse variance ivar,
        and psf object.  Call extract() method to actually do the
        extraction.
        """ 
        #- Call Extractor.__init__ to actually do the initialization
        Extractor.__init__(self, image, image_ivar, psf)
        
        #- setup resolution matrix
        self.R = list()
        for ispec in range(psf.nspec):
            # self.R.append(ResolutionMatrix(psf.nflux))
            self.R.append(ResolutionMatrix( (psf.nflux, psf.nflux) ))

    def extract(self, specmin=None, specmax=None, fluxmin=None, fluxmax=None, fluxstep=None):
        """
        Extract spectra within specified ranges
        """
        #- Default ranges
        psf = self._psf
        if specmin is None: specmin = 0
        if specmax is None: specmax = psf.nspec
        if fluxmin is None: fluxmin = 0
        if fluxmax is None: fluxmax = psf.nflux
        if fluxstep is None: fluxstep = 100
        
        #- Sanity check on boundaries
        fluxmin = max(fluxmin, 0)
        fluxmax = min(fluxmax, psf.nflux)
        
        #- Walk through subregions to extract
        _checkpoint()
        for fluxlo in range(fluxmin, fluxmax, fluxstep):
            
            #- Get bounaries of this sub-region
            fluxhi = min(fluxlo+fluxstep, fluxmax)
            if fluxlo > fluxhi: break
            B = Boundaries(psf, specmin, specmax, fluxlo, fluxhi)
            print B
                                    
            #- Get image and ivar cutouts
            pix  = self._image[B.ymin:B.ymax, B.xmin:B.xmax]
            ivar = self._image_ivar[B.ymin:B.ymax, B.xmin:B.xmax]

            #- Get sparse projection matrix A
            A = psf.getSparseAsub(
                B.specmin, B.specmax, B.fluxmin, B.fluxmax,
                B.xmin, B.xmax, B.ymin, B.ymax )
        
            #- Solve it!
            results = _solve(A, pix, ivar, boundaries=B)
            
            #- Save results
            self.store_results(results=results, boundaries=B)
                        
    def store_results(self, results, boundaries):
        """
        Save results from sub-region into full output arrays
        """
        B = boundaries  #- shorthand
        self._model[B.goodpix] = results['model'][B.goodsubpix]
        self._spectra[B.goodspec] = results['spectra'][B.goodsubspec]
        self._ivar[B.goodspec] = results['ivar'][B.goodsubspec]
        self._deconvolved_spectra[B.goodspec] = results['deconvolved_spectra'][B.goodsubspec]

        #- Get resolution matrix R cutouts
        #- WARNING: this will break if we add spectral boundaries
        #- in addition to flux boundaries, so add asserts to catch it
        assert(B.speclo == B.specmin)
        assert(B.spechi == B.specmax)
        R = results['resolution']            
        for ispec in range(B.nspec):
            #- Slice out a piece of R for this spectrum, avoiding edges
            ii = slice(ispec*B.nflux+B.dfluxlo/2, (ispec+1)*B.nflux-B.dfluxhi/2)
            Rx = R[ii, ii]
            jj = slice(B.fluxlo-B.dfluxlo/2, B.fluxlo-B.dfluxlo/2+Rx.shape[0])
            self.R[ispec][jj, jj] = R[ii, ii]
                        
    def expand_resolution(self):
        """
        Expand individual sparse resolution matrices into 3D matrix of
        diagonals for writing to a fits file.  Needs a better name.
        """
        bandwidth = 15
        nspec = self._psf.nspec
        nflux = self._psf.nflux
        self._resolution = N.zeros( (nspec, bandwidth, nflux), dtype=N.float32 )
        for ispec in range(nspec):
            if self.R[ispec].nnz > 0:
                self._resolution[ispec] = self.R[ispec].diagonals(bandwidth)
                                                    
#-------------------------------------------------------------------------
#- matrix solver

def _solve(A, pix, ivar, boundaries=None, regularize=1e-6):
    """
    Solve pix[npix] = A[npix, nflux] * f[nflux] + noise[npix]
    
    A[npix, nflux] : projection matrix (possibly sparse)
    ivar[npix]     : 1.0 / noise^2
    
    Results results dictionary with spectra, deconvolved_spectra, ...
    """
    
    #- Save input array shapes
    npix, nflux = A.shape
    ny, nx = pix.shape
    
    #- Flatten arrays
    pix = pix.ravel()
    ivar = ivar.ravel()

    #- Weight by inverse noise
    Ni = scipy.sparse.spdiags(ivar, [0,], npix, npix)
    NiA = Ni.dot(A)
    wpix = pix * ivar

    #- Add regularization term to avoid singularity and excessive ringing
    if regularize > 0:
        I = regularize * scipy.sparse.identity(nflux)
        
        #--- TEST ---
        # r = N.array(A.sum(axis=0))[0]
        # minr = r.max() * regularize
        # rx = (minr-r).clip(0.0) + 1e-6
        # I = scipy.sparse.dia_matrix( ((rx,), (0,)), shape=(nflux,nflux) )
                
        NiAx = scipy.sparse.vstack((NiA, I))
        Ax = scipy.sparse.vstack((A, I))
        wpix = N.concatenate( (wpix, N.zeros(nflux)) )
    else:
        NiAx = NiA
        Ax = A

    #--- DEBUG ---
    ### import pdb; pdb.set_trace()
    #--- DEBUG ---

    #- Invert ATA = (A.T N^-1 A)
    ATA = Ax.T.dot(NiAx).toarray()
    try:
        ATAinv = N.linalg.inv(ATA)
    except N.linalg.LinAlgError, e:
        print >> sys.stderr, e
        print >> sys.stderr, "ERROR: Can't invert matrix"
        
        results = dict()
        results['spectra'] = N.zeros(nflux)
        results['deconvolved_spectra'] = N.zeros(nflux)
        results['ivar'] = N.zeros(nflux)
        results['resolution'] = N.zeros( (nflux, nflux) )
        results['model'] = N.zeros( (ny, nx) )
        return results
        
    #- Find deconvolved spectrum solution
    deconvolved_flux = ATAinv.dot( Ax.T.dot(wpix) )
    
    #- Find resolution matrix R and reconvolved flux
    R = resolution_from_icov(ATA)
    flux = R.dot(deconvolved_flux)
    
    #- Test: try extracting subregions of R
    #- Fails : bad edge effects
    #- (or was that other bugs?  Don't give up on this yet)
    # R = N.zeros( (nflux, nflux) )
    # nspec = nflux/nfluxbins
    # for i in range(0, nflux, nspec):
    #     ii = slice(i, i+nfluxbins)
    #     R[ii, ii] = resolution_from_icov(ATA[ii, ii])
    # flux = R.dot(deconvolved_flux)

    #- Diagonalize inverse covariance matrix to get flux ivar
    Cx = R.dot( ATAinv.dot(R.T) )    #- Bolton & Schlegel 2010 Eqn 15
    flux_ivar = 1.0 / Cx.diagonal(0)
        
    #- Generate model image (drop regularization terms at end of array)
    model = A.dot(deconvolved_flux)[0:npix]
    
    #- Create dictionary of results
    results = dict()
    results['spectra'] = flux
    results['ivar'] = flux_ivar
    results['deconvolved_spectra'] = deconvolved_flux
    results['resolution'] = R
    results['model'] = model.reshape( (ny, nx) )
        
    #- Reshape if boundaries are given
    if boundaries is not None:
        B = boundaries
        for key in ('spectra', 'ivar', 'deconvolved_spectra'):
            results[key] = results[key].reshape( (B.nspec, B.nflux) )    

    return results

#-------------------------------------------------------------------------
#- Utility function for timing tests

_t0 = None
def _checkpoint(comment=None):
    global _t0
    if comment is not None:
        print '%-25s : %.1f' % (comment, time() - _t0)        
    _t0 = time()


