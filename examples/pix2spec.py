#!/usr/bin/env python

"""
Solve for spectra given an image and a PSF
"""

import sys
import os
import numpy as N
import pyfits
from time import time

from bbspec.spec2d.psf import load_psf
from bbspec.spec2d import resolution_from_icov

import optparse

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-i", "--input",  type="string",  help="input image")
parser.add_option("-p", "--psf",    type="string",  help="input psf")
parser.add_option("-o", "--output", type="string",  help="output extracted spectra")

opts, args = parser.parse_args()

#- Load the input data
print "Loading input image and PSF"
psf = load_psf(opts.psf)
image = pyfits.getdata(opts.input, 0)
ivar = pyfits.getdata(opts.input, 1)

#- Catch case where all input variances are 0.0
if N.all(ivar == 0.0):
    print >> sys.stderr, 'WARNING: All input variances are 0.0 -- writing dummy output'
    xspec0 = xspec1 = xspec_ivar = N.zeros( (psf.nspec, psf.nflux) )
    nx = psf.nspec * psf.nflux
    R = ATA = N.zeros( (nx, nx) )
else:
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
    p = image.ravel() * N.sqrt(ivar).ravel()

    NiA = N.dot(N.diag(N.sqrt(ivar).ravel()), A) 
    ATA = N.dot(A.T, NiA)
    try:
        ATAinv = N.linalg.inv(ATA)        
        xspec0 = N.dot( ATAinv, N.dot(A.T, p) ).reshape( (psf.nspec, psf.nflux) )
        t1 = time()
        print '  --> %.1f seconds' % (t1-t0)

        #- Reconvolve flux by resolution
        #- Pulling out diagonal blocks doesn't seem to work (for numerical reasons?),
        #- so use full A.T A matrix
        print "Reconvolving"
        R = resolution_from_icov(ATA)
        xspec1 = N.dot(R, xspec0.ravel()).reshape( (psf.nspec, psf.nflux) )

        t2 = time()
        ### print '  --> %.1f seconds' % (t2-t1)

        #- Variance of extracted reconvolved spectrum
        Cx = N.dot(R, N.dot(ATAinv, R.T))  #- Bolton & Schelgel 2010 Eqn 15
        xspec_var = N.array([Cx[i,i] for i in range(Cx.shape[0])])
        xspec_ivar = (1.0/xspec_var).reshape(xspec1.shape)
    except N.linalg.LinAlgError, e:
        print >> sys.stderr, "WARNING: LinAlgError -- writing dummy output"
        ### print >> sys.stderr, e
        xspec0 = xspec1 = xspec_ivar = N.zeros( (psf.nspec, psf.nflux) )
        nx = psf.nspec * psf.nflux
        R = ATA = N.zeros( (nx, nx) )
        
#- Output results
print "Writing output"
hdus = list()
hdus.append(pyfits.PrimaryHDU(data = xspec1))
hdus.append(pyfits.ImageHDU(data = xspec_ivar))
hdus.append(pyfits.ImageHDU(data = R))
hdus.append(pyfits.ImageHDU(data = xspec0))
nx = psf.nspec * psf.nflux
hdus.append(pyfits.ImageHDU(data = ATA.reshape( (nx, nx) )))

hdus = pyfits.HDUList(hdus)
hdus[0].header.add_comment('Extracted re-convolved spectra')
hdus[0].header.add_comment('Input image: %s' % opts.input)
hdus[0].header.add_comment('Input PSF: %s' % opts.psf)

hdus[1].header.add_comment('Inverse variance of extracted re-convolved spectra')
hdus[2].header.add_comment('Convolution kernel R')
hdus[3].header.add_comment('Original extracted spectra')
hdus[4].header.add_comment('Inverse covariance of original extracted spectra')

hdus.writeto(opts.output, clobber=True)






