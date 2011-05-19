#!/usr/bin/env python

"""
Plot results from PSF + spectra -> image -> extracted spectra tests
"""

import numpy as N
import pylab as P
import pyfits

from bbspec.spec2d.psf import load_psf

import optparse

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-s", "--spectra", type="string", help="input spectra")
parser.add_option("-x", "--xspec", type="string",  help="extracted spectra")
parser.add_option("-i", "--image", type="string",  help="image")
parser.add_option("-p", "--psf", type="string",  help="psf")

opts, args = parser.parse_args()

inspec = pyfits.getdata(opts.spectra)
nspec, nflux = inspec.shape

xspec = pyfits.getdata(opts.xspec, 0)   #- re-convolved spectra
R = pyfits.getdata(opts.xspec, 2)       #- re-convolving kernel
xspec0 = pyfits.getdata(opts.xspec, 3)  #- un-convolved spectra

#- Input spectra, convolved with resolution
cspec = N.dot(R, inspec.ravel()).reshape(inspec.shape)

#- Input image
image = pyfits.getdata(opts.image)

#- Model image
psf = load_psf(opts.psf)
model = N.zeros( (psf.npix_y, psf.npix_x) )
for i in range(nspec):
    for j in range(nflux):
        xslice, yslice, pix = psf.pix(i, j)
        model[yslice, xslice] += xspec0[i, j] * pix

#- Plots!
P.subplot(211)
for i in range(nspec):
    P.plot(xspec[i]+i*50, lw=2)
    P.plot(cspec[i]+i*50, 'k-')
P.title('Extracted Re-convolved Spectra')
    
vmin = N.min(image)
vmax = N.max(image)
opts = dict(interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)    

P.subplot(234)
P.imshow(image, **opts)
P.title('Input Image')

P.subplot(235)
P.imshow(model, **opts)
P.title('Model')

P.subplot(236)
P.imshow(10*(model-image), **opts)
P.title('10x Residuals')

P.show()
