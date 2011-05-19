#!/usr/bin/env python

"""
Convolve known input spectra with a PSF to generate a test output image
"""

import sys
import optparse

import numpy as N
import pyfits

from bbspec.spec2d.psf import load_psf

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-i", "--input",  type="string",  help="input spectra")
parser.add_option("-p", "--psf",    type="string",  help="input psf")
parser.add_option("-o", "--output", type="string",  help="output image")

opts, args = parser.parse_args()

if opts.input is None or opts.psf is None or opts.output is None:
    parser.print_help()
    print >> sys.stderr, "input, psf, and output are all required parameters"
    sys.exit(1)

spectra = pyfits.getdata(opts.input)
psf = load_psf(opts.psf)

#- Sanity check on dimensions
nspec, nflux = spectra.shape
if nspec != psf.nspec or nflux != psf.nflux:
    print >> sys.stderr, "Dimensions don't match:"
    print >> sys.stderr, "spectra.nspec = %d  psf.nspec = %d" % (nspec, psf.nspec)
    print >> sys.stderr, "spectra.nflux = %d  psf.nflux = %d" % (nspec, psf.nflux)
    sys.exit(1)

#- Project spectra -> image with PSF
image = N.zeros( (psf.npix_y, psf.npix_x) )
for ispec in range(psf.nspec):
    for iflux in range(psf.nflux):
        if spectra[ispec, iflux] == 0.0:
            continue
    
        xslice, yslice, pixels = psf.pix(ispec, iflux)
        image[yslice, xslice] += pixels * spectra[ispec, iflux]

#- Add noise
orig_image = image.copy()
read_noise = 2.0
var = image + read_noise**2
image += N.random.normal(scale=N.sqrt(var), size=image.shape)
ivar = 1.0 / var

#- Write output
hdus = list()
hdus.append(pyfits.PrimaryHDU(data=image))
hdus.append(pyfits.ImageHDU(data=ivar))
hdus.append(pyfits.ImageHDU(data=orig_image))

hdus = pyfits.HDUList(hdus)
hdus[0].header.add_comment('Noisy pixel data')
hdus[0].header.add_comment('Input spectra: %s' % opts.input)
hdus[0].header.add_comment('Input PSF: %s' % opts.psf)
hdus[1].header.add_comment('Pixel inverse variance')
hdus[2].header.add_comment('Pure image without noise')

hdus.writeto(opts.output, clobber=True)
    
        

