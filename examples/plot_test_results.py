#!/usr/bin/env python

"""
Plot results from PSF + spectra -> image -> extracted spectra tests
"""

import numpy as N
import pylab as P
import pyfits

from bbspec.spec2d.psf import load_psf
from bbspec.spec2d import ResolutionMatrix

import optparse

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-s", "--spectra", type="string", help="input spectra")
parser.add_option("-x", "--xspec", type="string",  help="extracted spectra")
parser.add_option("-i", "--image", type="string",  help="input image")
parser.add_option("-p", "--psf", type="string",  help="psf")
parser.add_option("-m", "--model", type="string",  help="model image")

opts, args = parser.parse_args()

inspec = pyfits.getdata(opts.spectra)
nspec, nflux = inspec.shape

#- Cast type to get endianness correct for sparse matrix ops
inspec = inspec.astype(N.float64)

rspec = pyfits.getdata(opts.xspec, 0)   #- re-convolved spectra
rspec_ivar = pyfits.getdata(opts.xspec, 1)   #- re-convolved spectra inverse variance
R = pyfits.getdata(opts.xspec, 3)       #- re-convolving kernels
xspec = pyfits.getdata(opts.xspec, 4)  #- un-convolved spectra

#- Input spectra, convolved with resolution
### cspec = N.dot(R, inspec.ravel()).reshape(inspec.shape)
cspec = N.empty(inspec.shape)
for i in range(nspec):
    Rx = ResolutionMatrix.from_diagonals(R[i]).tocsr()
    cspec[i] = Rx.dot(inspec[i])
    
    #- Alternate rspec, computed from saved R and xspec
    ### rspec[i] = Rx.dot(xspec[i].astype(N.float64))

#- Load inputs 
image = pyfits.getdata(opts.image, 0)
image_ivar = pyfits.getdata(opts.image, 1)
psf = load_psf(opts.psf)

if opts.model is not None:
    model = pyfits.getdata(opts.model)
else:
    try:
        model = pyfits.getdata(opts.xspec, 5)
    except IndexError:
        print "WARNING: unable to load model image"
        model = N.zeros(image.shape)

#- Plots!
P.figure()
P.subplots_adjust(bottom=0.05)

#- Spectra
P.subplot(231)
for i in range(nspec):
    P.plot(rspec[i]+i*100, drawstyle='steps-mid')
    P.plot(cspec[i]+i*100, 'k-')
P.title('Spectra')

#- Flux chi
P.subplot(232)
chi = (cspec - rspec) * N.sqrt(rspec_ivar)
chi = chi.ravel()
P.hist(chi, 50, (-5, 5), histtype='stepfilled')
P.title('Spectra chi')
# P.xlabel('Spectra (data - fit) / error')
# print chi.mean(), chi.std()
ymin, ymax = P.ylim()
P.text(-4.5, ymax*0.9, "$\sigma=%.1f$" % chi.std())
P.xlim(-5, 5)

#- Pixel chi
P.subplot(233)
chi = ((model-image) * N.sqrt(image_ivar)).ravel()
P.hist(chi, 50, (-5, 5), histtype='stepfilled')
P.title('Pixel chi')
# P.xlabel('Image (data - fit) / error')
ymin, ymax = P.ylim()
P.text(-4.5, ymax*0.9, "$\sigma=%.1f$" % chi.std())
P.xlim(-5, 5)

#- Lower row of images    
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

P.savefig('blat.pdf')

P.show()
