#!/usr/bin/env python

"""
Take PSF files with individual bundles, and merge them into a single file
"""

import sys
import numpy as N
import pyfits

import optparse

parser = optparse.OptionParser(usage = "%prog infiles... -o outfile.fits")
parser.add_option("-o", "--output", type="string",  help="output filename")
opts, args = parser.parse_args()

if opts.output is None:
    parser.print_help()
    print >> sys.stderr, 'ERROR: You must specify an output file (-o/--output)'
    sys.exit(1)

#- Get template from first input file
fx = pyfits.open(args[0])
del fx[0].header['BUNDLELO']
del fx[0].header['BUNDLEHI']

for filename in args[1:]:
    print filename
    fits = pyfits.open(filename)
    i0 = fits[0].header['BUNDLELO'] * 20
    i1 = (fits[0].header['BUNDLEHI']+1) * 20
    
    for ihdu in range(len(fits)):
        fx[ihdu].data[i0:i1] = fits[ihdu].data[i0:i1]
        
    fits.close()
        
fx.writeto(opts.output, clobber=True)


