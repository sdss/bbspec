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
from bbspec.spec2d.extract import SubExtractor
ex = SubExtractor(image, ivar, psf)
ex.extract()
ex.expand_resolution()
        
#- Output results
print "Writing output"
comments = list()
comments.append('Input image: %s' % opts.input)
comments.append('Input PSF: %s' % opts.psf)
ex.writeto(opts.output, comments=comments)

dt = time() - t0
print '  --> %.1f sec' % dt





