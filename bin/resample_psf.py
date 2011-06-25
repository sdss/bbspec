#!/usr/bin/env python

"""
Resample PSF to a new grid from an spCFrame file

Stephen Bailey, LBNL, Summer 2011
"""

import sys
import numpy as N
import pyfits

import optparse

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-p", "--psf",        type="string", help="input PSF file")
parser.add_option("-o", "--output",     type="string", help="output PSF file")
parser.add_option("-f", "--frame",      type="string", help="input spCFrame file")
parser.add_option("-w", "--wavelength", type="string", help="wmin,wmax,n")
opts, args = parser.parse_args()

#- Required options
if opts.psf is None or opts.output is None:
    parser.print_help()
    print "ABORT: you must specify a psf and output file"
    sys.exit(1)

#- If wavelength range isn't given guess defaults from input psf
if opts.wavelength is None:
    nw = 3500
    wtmp = 10**pyfits.getdata(opts.psf, 2)
    if wtmp[0] < 4000:  #- blue camera
        wmin = N.log10(3500)
        wmax = N.log10(6500)
    else:
        wmin = N.log10(5500)
        wmax = N.log10(10500)        
else:
    #- Parse wavelength into log steps from wmin to wmax
    wtmp = opts.wavelength.split(',')
    wmin = float(wtmp[0])
    wmax = float(wtmp[1])
    nw   = int(wtmp[2])

#- Logarithmically sample wavelength range    
loglam = N.log10(N.logspace(N.log10(wmin), N.log10(wmax), nw))

#- Load spCFrame loglam and x, or use values from PSF
if opts.frame is not None:
    frame_loglam = pyfits.getdata(opts.frame, 3)
    frame_x = pyfits.getdata(opts.frame, 7)
else:
    frame_loglam = pyfits.getdata(opts.psf, 2)
    frame_x = pyfits.getdata(opts.psf, 0)

nspec, ny = frame_x.shape
frame_y = N.arange(ny)

#- Resample onto the requested wavelegth grid
psf_x = N.zeros( (nspec, nw) )
psf_y = N.zeros( (nspec, nw) )
psf_loglam = N.zeros( (nspec, nw) )

for i in range(nspec):
    psf_x[i] = N.interp(loglam, frame_loglam[i], frame_x[i])
    psf_y[i] = N.interp(loglam, frame_loglam[i], frame_y)
    psf_loglam[i] = loglam

#- Open PSF file and replace X, Y, loglam data
psf = pyfits.open(opts.psf)
psf[0].header.update('NFLUX', nw)
psf[0].data = psf_x
psf[1].data = psf_y
psf[2].data = psf_loglam

#- Write outputs
psf.writeto(opts.output, output_verify='silentfix', clobber=True)
psf.close()



