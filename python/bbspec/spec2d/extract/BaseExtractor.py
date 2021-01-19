#!/usr/bin/env python

"""
Base class for extraction objects
"""

import sys
import pyfits
import numpy as N

class BaseExtractor(object):
    """Base class for extraction objects to define interface"""
    def __init__(self, image, image_ivar, psf):
        """
        Create extraction object from input image, inverse variance ivar,
        and psf object.  Call extract() method to actually do the
        extraction.
        """
        #- Keep arrays, ensuring native endian-ness for sparse matrix ops
        self._image = image.astype('=f8')
        self._image_ivar = image_ivar.astype('=f8')
        self._psf = psf
        
        #- Create blank arrays to be filled
        self._spectra = N.zeros( (psf.nspec, psf.nflux) )
        self._ivar    = N.zeros( (psf.nspec, psf.nflux) )
        self._deconvolved_spectra = N.zeros( (psf.nspec, psf.nflux) )
        self._model     = N.zeros(image.shape)
        self._bkg_model = N.zeros(image.shape)
        
        #- These are big and may not be filled in for certain tests
        self._resolution = None
        self._dspec_icov = None
                
    def extract(self):
        """
        Perform the extraction and fill in output arrays.
        
        Sub-classes should implement this.
        """
        raise NotImplementedError

    def writeto(self, filename, comments=None):
        """
        Write the results of this extraction to filename
        """
        print("Writing output to", filename)
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
    def psf(self):
        """Return PSF model object used for extraction"""
        return self._psf
        
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

