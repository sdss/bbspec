#!/usr/bin/env python

"""
Extract spectra from an image given a PSF, working in sub-regions

Hardcoded:
  - number of fibers per bundle
  - number of bundles
  - input options for testing
"""

import optparse
from multiprocessing import Process, Queue, cpu_count

import numpy as N

import pyfits

from bbspec.spec2d.psf import load_psf
from bbspec.spec2d.extract import SubExtractor

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-i", "--image",  type="string",  help="input image")
parser.add_option("-p", "--psf",    type="string",  help="input psf")
parser.add_option("-o", "--output", type="string",  help="output extracted spectra")
parser.add_option("-b", "--bundle", type="string",  help="comma separated list of bundles, 0-24")
parser.add_option("-f", "--fluxbins", type="string",  help="fmin,fmax,step : flux bin range and sub-region size")
parser.add_option("-P", "--parallel", action='store_true',  help="Parallelize calculation")
parser.add_option("--fibers_per_bundle", type="int", default=20, help="Number of fibers per bundle, default=%default")

### parser.add_option("-T", "--test",     action='store_true',  help="hook to try test code")
opts, args = parser.parse_args()

def parse_string_range(s):
    """
    e.g. "1,2,5-8,20" -> [1,2,5,6,7,8,20]
    
    modified from Sven Marnach,
    http://stackoverflow.com/questions/5704931/parse-string-of-integer-sets-with-intervals-to-list
    
    Feature/Bug: Only works with positive numbers
    """
    ranges = (x.split("-") for x in s.split(","))
    x = [i for r in ranges for i in range(int(r[0]), int(r[-1]) + 1)]
    return x
    
#--- TEST ---
#- Default options override while testing
if opts.image is None:
    opts.image = 'sdProc-r1-00115982.fits'
    print 'WARNING: Input image not set, using -i', opts.image
if opts.psf is None:
    opts.psf = 'spBasisPSF-PIX-r1-00115982.fits'
    print 'WARNING: Input psf not set, using -p', opts.psf
if opts.output is None:
    opts.output = 'spSpec-r1-00115982.fits'
    print 'WARNING: Output spectra not set, using -o', opts.output
if opts.fluxbins is None:
    opts.fluxbins = "3000,3500,20"    
    print "WARNING: Flux bin range not set, using -f", opts.fluxbins
if opts.bundle is None:
    opts.bundle = "0"
    print "WARNING: Bundle range not set, using -b", opts.bundle
#--- TEST ---

#- Load the input data
print "Loading input image and PSF"
psf = load_psf(opts.psf)
image = pyfits.getdata(opts.image, 0)
ivar = pyfits.getdata(opts.image, 1)

#- Parse bundle and flux bin options
opts.bundle = parse_string_range(opts.bundle)  #- "1,3-5" -> [1,3,4,5]
fmin,fmax,fstep = map(int, opts.fluxbins.split(','))

#- create extraction object
ex = SubExtractor(image, ivar, psf)

#- Loop over bundles and fibers
for b in opts.bundle:
    if b < 0 or b > 24:  #- Hardcode bundle range!
        print "WARNING: You are crazy, there is no bundle %d.  Try 0-24." % b
        continue

    ispecmin = b * opts.fibers_per_bundle
    ispecmax = ispecmin + opts.fibers_per_bundle
    jobs = list()
    ex.extract(ispecmin, ispecmax, fmin, fmax, fstep, parallel=opts.parallel)
            
#- Output results
print "Writing output"
ex.expand_resolution()
comments = list()
comments.append('Input image: %s' % opts.image)
comments.append('Input PSF: %s' % opts.psf)
ex.writeto(opts.output, comments=comments)
