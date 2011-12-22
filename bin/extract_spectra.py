#!/usr/bin/env python

"""
Extract spectra from an image given a PSF, working in sub-regions

Stephen Bailey, Summer 2011
"""

import os
import optparse
import multiprocessing as MP
from time import time, asctime
import numpy as N
import pyfits

from bbspec.spec2d.psf import load_psf
from bbspec.spec2d.extract import EigenExtractor
from bbspec.spec2d import Spectra
from bbspec.util import xmap

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-i", "--image",  type="string",  help="input image")
parser.add_option("-p", "--psf",    type="string",  help="input psf")
parser.add_option("-o", "--output", type="string",  help="output extracted spectra")
parser.add_option("-b", "--bundle", type="string",  help="comma separated list of bundles, 0-24")
parser.add_option("-f", "--fluxbins", type="string",  help="fmin,fmax,step : flux bin range and sub-region size")
parser.add_option("-F", "--framefile", type="string",  help="spCFrame file to use for trace solution")
parser.add_option("-w", "--wavelength", type="string",  help="wmin,wmax,dw : wavelength range and resolution")
parser.add_option("-L", "--loglam", type="string",  help="loglam_min,loglam_max,dloglam : log10 wavelength range and resolution")
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
    
#-------------------------------------------------------------------------

#- Load the input data
print "Loading input image and PSF", asctime()
psf = load_psf(opts.psf)
image = pyfits.getdata(opts.image, 0)
ivar = pyfits.getdata(opts.image, 1)

#- If framefile is given, shift PSF to that grid
if opts.framefile:
    frame_x = pyfits.getdata(opts.framefile, 7)
    frame_y = N.arange(frame_x.shape[1])
    frame_loglam = pyfits.getdata(opts.framefile, 3)
    psf.shift_xy(frame_x, frame_y, frame_loglam)

#- Parse bundle / spectra options
opts.bundle = parse_string_range(opts.bundle)  #- "1,3-5" -> [1,3,4,5]
bmin, bmax = min(opts.bundle), max(opts.bundle)
nspec = opts.fibers_per_bundle * (bmax-bmin+1)
specmin = opts.fibers_per_bundle * bmin
specmax = specmin + nspec
nspec = opts.fibers_per_bundle * (max(opts.bundle) - min(opts.bundle) + 1)
ispecmin = opts.fibers_per_bundle * min(opts.bundle)

#- Parse wavelength grid
if opts.loglam:
    loglam_min, loglam_max, dloglam = map(float, opts.loglam.split(','))
    loglam = N.arange(loglam_min, loglam_max, dloglam)
    psf.resample(loglam)
elif opts.wavelength:
    wmin, wmax, dw = map(float, opts.wavelength.split(','))
    w = N.arange(wmin, wmax, dw)
    psf.resample(N.log10(w))

#- Parse flux bin options
if opts.fluxbins is None:
    opts.fluxbins = fmin,fmax,fstep = 0, psf.nflux, 50
else:
    opts.fluxbins = fmin,fmax,fstep = map(int, opts.fluxbins.split(','))
    
loglam = psf.loglam()[specmin:specmax, fmin:fmax]

#- Extract a single bundle, write to temporary fits file
def extract_bundle(*args):
    global t0
    extractor, bundle, opts = args
    specmin = bundle * opts.fibers_per_bundle
    specmax = specmin + opts.fibers_per_bundle
    fmin,fmax,fstep = opts.fluxbins
    s = extractor.extract((specmin, specmax), (fmin, fmax), fstep)
    outfile = '%s_%02d' % (opts.output, bundle)
    s.calc_model_image(extractor.psf)
    ### print "Writing", outfile
    s.write(outfile)

#- create extraction object
extractor = EigenExtractor(image, ivar, psf)

#- Loop over bundles and fibers
print 'Starting extractions', asctime()
if opts.parallel:
    pool = MP.Pool()
    exopts = list()
    for b in opts.bundle:
        exopts.append((extractor, b, opts))
    # for s in pool.map(extract_bundle, exopts, chunksize=1):
    for s in xmap(extract_bundle, exopts, verbose=False):
        pass
        ### print 'got spec', time()-t0
        # spectra.merge(s)
        
else:
    for b in opts.bundle:
        s = extract_bundle( extractor, b, opts )

#- Merge separate bundles
print "Merging bundles", asctime()
spectra = Spectra.blank(nspec, loglam, ispecmin=ispecmin, ifluxmin=fmin)
for b in opts.bundle:
    ### print b
    bundle_file = '%s_%02d' % (opts.output, b)
    s = Spectra.load(bundle_file)
    os.remove(bundle_file)
    spectra.merge(s)
            
#- Calculate model image
print "Calculating model image", asctime()
xyrange = (0, image.shape[1], 0, image.shape[0]) #- xmin, xmax, ymin, ymax
spectra.calc_model_image(psf, xyrange=xyrange)
# spectra.pix = psf.spec2pix(spectra.xflux)
            
#- Output results
print "Writing output", asctime()
comments = list()
comments.append('Input image: %s' % opts.image)
comments.append('Input PSF: %s' % opts.psf)
spectra.write(opts.output, comments=comments)

print "Done", asctime()
