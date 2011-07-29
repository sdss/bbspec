#!/usr/bin/env python

"""
Resample PSF to a new xy grid.

Inputs:
  - psf file
  - spCFrame file
  
Output:
  - psf file of the same format as the input psf, with the X, Y, and LogLam
    HDUs replaced with values from the spCFrame file.

Author:
    Stephen Bailey, LBNL, Summer 2011
"""

import sys
import numpy as N
import pyfits

import optparse

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-p", "--psf",        type="string", help="input PSF file")
parser.add_option("-o", "--output",     type="string", help="output PSF file")
parser.add_option("-f", "--frame",      type="string", help="input spCFrame file [optional]")
opts, args = parser.parse_args()

#- Required options
if opts.psf is None or opts.output is None:
    parser.print_help()
    print "ABORT: you must specify a psf and output file"
    sys.exit(1)

#- Load x and loglam vs. y=iflux positions from spCFrame file
frame_loglam = pyfits.getdata(opts.frame, 3)
frame_x = pyfits.getdata(opts.frame, 7)
nspec, ny = frame_x.shape
frame_y = N.tile(N.arange(ny), nspec).reshape((nspec, ny))

#- Open PSF file and replace X, Y, loglam data
psf = pyfits.open(opts.psf)
psf[0].data = frame_x
psf[1].data = frame_y
psf[2].data = frame_loglam

#- Write output psf file
psf.writeto(opts.output, output_verify='silentfix', clobber=True)
psf.close()



