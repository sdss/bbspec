#!/usr/bin/env python

"""
Generate random test spectra to match a given PSF file
"""

import optparse
import sys

import numpy as N
import pyfits

from bbspec.spec2d.psf import PSFBase

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-p", "--psf", type="string",  help="input PSF to match")
parser.add_option("-o", "--output", type="string",  help="output spectra file")
opts, args = parser.parse_args()

if opts.psf is None or opts.output is None:
    parser.print_help()
    print "psf and output are required parameters"
    sys.exit(1)    

#- Load PSF
psf = PSFBase(opts.psf)

#- Allocate blank spectra array
spectra = N.zeros( (psf.nspec, psf.nflux) )

#- Add background continuua
x = N.arange(psf.nflux) * 2 * N.pi / psf.nflux
for i in range(psf.nspec):
    c = N.random.uniform(500)
    phase = N.random.uniform(N.pi)
    spectra[i] += c*N.abs(N.sin(x-phase))

#- Add delta functions
for test in range(100):
    i = N.random.randint(psf.nspec)
    j = N.random.randint(psf.nflux)
    c = N.random.uniform(0, 1000)
    spectra[i,j] += c
    
#- Write output
pyfits.writeto(opts.output, spectra, clobber=True)





