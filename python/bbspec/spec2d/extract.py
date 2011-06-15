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

class SubExtractor(Extractor):
    """
    Extractor which supports parallel extraction of sub-regions
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

    def extract(self, speclo=None, spechi=None, fluxlo=None, fluxhi=None, fluxstep=None):
        """
        Extract spectra within specified ranges
        """
        #- Default ranges
        psf = self._psf
        if speclo is None: speclo = 0
        if spechi is None: spechi = psf.nspec
        if fluxlo is None: fluxlo = 0
        if fluxhi is None: fluxhi = psf.nflux
        if fluxstep is None: fluxstep = 100
        nspec = spechi - speclo
        
        #- Sanity check on boundaries
        fluxlo = max(fluxlo, 0)
        fluxhi = min(fluxhi, psf.nflux)
        
        #- Walk through subregions to extract
        _checkpoint()
        for iflux in range(fluxlo, fluxhi, fluxstep):
            #- Define flux range for this cutout
            flo = iflux
            fhi = min(flo+fluxstep, fluxhi)
            
            #- Find which pixels are covered by these spectra
            xlo = int(psf.param['X'][speclo, flo:fhi].min())
            xhi = int(psf.param['X'][spechi-1, flo:fhi].max())
            ylo = int(psf.param['Y'][speclo:spechi, flo].min())
            yhi = int(psf.param['Y'][speclo:spechi, fhi-1].max())+1
        
            #- Add pixel boundaries to fully cover spectra of interest
            dx = 8
            xlo = max(xlo-dx, 0)
            xhi = min(xhi+dx, psf.npix_x)

            #- For y pixel extension, watch out for edges
            dy = 5
            dylo = min(dy, ylo)
            dyhi = min(dy, psf.npix_y-yhi)

            #- Add extra flux bins to avoid edge effects,
            #- but don't exceed boundaries of the problems
            dflo = min(2*dy, flo)
            dfhi = min(2*dy, psf.nflux-fhi)
            nflux = (fhi+dfhi) - (flo-dflo)
                        
            # print 'spec %d-%d flux %d-%d' % (speclo, spechi, flo, fhi)
            print 'spec %d-%d  flux %3d-%3d  flux+ %3d-%3d' % \
                (speclo, spechi, flo, fhi, flo-dflo, fhi+dfhi),
            print ' x %3d-%3d  y %3d-%3d ' % (xlo, xhi, ylo-dylo, yhi+dyhi)

            #--- DEBUG ---
            # import pdb; pdb.set_trace()
            #--- DEBUG ---

            #- Get sparse projection matrix A
            A = psf.getSparseAsub(speclo, spechi, flo-dflo, fhi+dfhi,
                xlo, xhi, ylo-dylo, yhi+dyhi)

            Anx = xhi-xlo
            Any = (yhi+dyhi) - (ylo-dylo)

            #- Get image and ivar cutouts
            pix = self._image[ylo-dylo:yhi+dyhi, xlo:xhi]
            ivar = self._image_ivar[ylo-dylo:yhi+dyhi, xlo:xhi]
        
            #- Solve it!
            results = _solve(A, pix, ivar, nfluxbins=nflux)

            #- Reshape
            results['spectra'] = results['spectra'].reshape( (nspec, nflux) )
            results['ivar'] = results['ivar'].reshape( results['spectra'].shape )
            results['deconvolved_spectra'] = results['deconvolved_spectra'].reshape( results['spectra'].shape )
            
            #- Put the results into the bigger picture
            self._model[ylo:yhi, xlo:xhi] = results['model'][dylo:dylo+(yhi-ylo), :]
            self._spectra[speclo:spechi, flo:fhi] = results['spectra'][:, dflo:dflo+(fhi-flo)]
            self._deconvolved_spectra[speclo:spechi, flo:fhi] = results['deconvolved_spectra'][:, dflo:dflo+(fhi-flo)]
            self._ivar[speclo:spechi, flo:fhi] = results['ivar'][:, dflo:dflo+(fhi-flo)]

            #--- DEBUG ---
            # import pylab as P
            # xspec = results['deconvolved_spectra']
            # x0 = N.zeros(xspec.shape)
            # x0[:, dflo:dflo+(fhi-flo)] = xspec[:, dflo:dflo+(fhi-flo)]
            # model = A.dot(xspec.ravel()).reshape( (Any, Anx) )
            # m0 = A.dot(x0.ravel()).reshape( (Any, Anx) )
            # import pdb; pdb.set_trace()
            #--- DEBUG ---

            #- Get resolution matrix R cutouts
            R = results['resolution']            
            for ispec in range(nspec):
                #- Slice out a piece of R for this spectrum, avoiding edges
                ii = slice(ispec*nflux+dflo/2, (ispec+1)*nflux-dfhi/2)
                Rx = R[ii, ii]
                jj = slice(flo-dflo/2, flo-dflo/2+Rx.shape[0])
                self.R[ispec][jj, jj] = R[ii, ii]
            
    def expand_resolution(self):
        bandwidth = 15
        nspec = self._psf.nspec
        nflux = self._psf.nflux
        self._resolution = N.zeros( (nspec, bandwidth, nflux), dtype=N.float32 )
        for ispec in range(nspec):
            if self.R[ispec].nnz > 0:
                self._resolution[ispec] = self.R[ispec].diagonals(bandwidth)
                                                    
#-------------------------------------------------------------------------
#- matrix solver

def _solve(A, pix, ivar, nfluxbins, regularize=1e-6):
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
        
    return results

#-------------------------------------------------------------------------
#- Utility function for timing tests

_t0 = None
def _checkpoint(comment=None):
    global _t0
    if comment is not None:
        print '%-25s : %.1f' % (comment, time() - _t0)        
    _t0 = time()


