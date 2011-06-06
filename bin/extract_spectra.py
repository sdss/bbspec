#!/usr/bin/env python

"""
Extract spectra from an image given a PSF, working in sub-regions
"""

import optparse
import numpy as N

import pyfits

from bbspec.spec2d.psf import load_psf
from bbspec.spec2d.extract import SimpleExtractor

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-i", "--image",  type="string",  help="input image")
parser.add_option("-p", "--psf",    type="string",  help="input psf")
parser.add_option("-o", "--output", type="string",  help="output extracted spectra")
parser.add_option("-b", "--bundle", type="string",  help="comma separated list of bundles, 0-24")
parser.add_option("-f", "--fluxbins", type="string",  help="fmin,fmax,step : flux bin range and sub-region size")
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
    opts.fluxbins = "2000,2500,50"    
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

#- Set default flux bin range

#- Parse bundle and flux bin options
opts.bundle = parse_string_range(opts.bundle)  #- "1,3-5" -> [1,3,4,5]
fmin,fmax,fstep = map(int, opts.fluxbins.split(','))

#- Run extraction
from bbspec.spec2d.extract import SimpleExtractor
ex = SimpleExtractor(image, ivar, psf)

nspec_per_bundle = 20
for b in opts.bundle:
    ispecmin = b*nspec_per_bundle
    ispecmax = ispecmin + nspec_per_bundle
    for iflux in range(fmin,fmax,fstep):
        print "Bundle %2d, flux bins %4d - %4d" % (b, iflux, iflux+fstep)
        ex.extract_subregion(ispecmin, ispecmax, iflux, iflux+fstep)

#- Debug small output
# ys = slice(900, 1050)
# xs = slice(260, 410)
# pyfits.writeto('bkgfit.fits', image[ys, xs], clobber=True)
# pyfits.append('bkgfit.fits', ivar[ys, xs])
# pyfits.append('bkgfit.fits', ex1.model[ys, xs])
# pyfits.append('bkgfit.fits', ex2.model[ys, xs])
# pyfits.append('bkgfit.fits', ex2.bkg_model[ys, xs])

#--- DEBUG ---
### import pdb; pdb.set_trace()
#--- DEBUG ---

#- Output results
print "Writing output to", opts.output
hdus = list()
hdus.append(pyfits.PrimaryHDU(data = ex.spectra))       #- 0
hdus.append(pyfits.ImageHDU(data = ex.ivar))            #- 1

# hdus.append(pyfits.ImageHDU(data = ex.resolution))    #- 2
hdus.append(pyfits.ImageHDU(data = None))

hdus.append(pyfits.ImageHDU(data = ex.deconvolved_spectra)) #- 3

# hdus.append(pyfits.ImageHDU(data = ex.dspec_icov))    #- 4
hdus.append(pyfits.ImageHDU(data = None))

hdus.append(pyfits.ImageHDU(data = ex.model))           #- 5

hdus = pyfits.HDUList(hdus)
hdus[0].header.add_comment('Extracted re-convolved spectra')
hdus[0].header.add_comment('Input image: %s' % opts.image)
hdus[0].header.add_comment('Input PSF: %s' % opts.psf)

hdus[1].header.add_comment('Inverse variance of extracted re-convolved spectra')
hdus[2].header.add_comment('Convolution kernel R')
hdus[3].header.add_comment('Original extracted spectra')
hdus[4].header.add_comment('Inverse covariance of original extracted spectra')
hdus[5].header.add_comment('Reconstructed model image')

hdus.writeto(opts.output, clobber=True)