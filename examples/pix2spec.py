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

t0 = time()

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

#- Run extraction
from bbspec.spec2d.extract import SimpleExtractor
ex = SimpleExtractor(image, ivar, psf)
ex.extract()
        
#- Output results
print "Writing output"
hdus = list()
hdus.append(pyfits.PrimaryHDU(data = ex.spectra))
hdus.append(pyfits.ImageHDU(data = ex.ivar))
hdus.append(pyfits.ImageHDU(data = ex.resolution))
hdus.append(pyfits.ImageHDU(data = ex.deconvolved_spectra))
hdus.append(pyfits.ImageHDU(data = ex.dspec_icov))

hdus = pyfits.HDUList(hdus)
hdus[0].header.add_comment('Extracted re-convolved spectra')
hdus[0].header.add_comment('Input image: %s' % opts.input)
hdus[0].header.add_comment('Input PSF: %s' % opts.psf)

hdus[1].header.add_comment('Inverse variance of extracted re-convolved spectra')
hdus[2].header.add_comment('Convolution kernel R')
hdus[3].header.add_comment('Original extracted spectra')
hdus[4].header.add_comment('Inverse covariance of original extracted spectra')

hdus.writeto(opts.output, clobber=True)

dt = time() - t0
print '  --> %.1f sec' % dt





