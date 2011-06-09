#!/usr/bin/env python

"""
View PSF cutouts for debugging
"""

import os.path
import numpy as N
import pylab as P

from bbspec.spec2d.psf import load_psf

import optparse

parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-p", "--png", help="save a png for each file", action="store_true")

opts, args = parser.parse_args()

#- cutout size smaller than psf cutout to emphasize core
cy = cx = 7

#- arrays of fibers and fluxes at which to evaluate the PSF
dfib = 20
dflux = 250
fibers = N.arange(10, 500, dfib)  #- bundle centers: 10, 30, 50, ... 490
fluxes = N.arange(500, 3501, dflux)

#- Loop over input files
for filename in args:
    psf = load_psf(filename)

    #- Create blank image for PSF cutouts
    nx = len(fibers) * cx
    ny = len(fluxes) * cy
    image = N.zeros( (ny, nx) )

    #- Fill in cutouts
    for i, fiber in enumerate(fibers):
        for j, flux in enumerate(fluxes):
            pix = psf.pix(fiber, flux)[2]
            xpix, ypix = pix.shape
            ix = (xpix-cx)/2
            iy = (ypix-cy)/2
        
            sx = slice(i*cx, (i+1)*cx)
            sy = slice(j*cy, (j+1)*cy)
            image[sy, sx] = pix[iy:iy+cy, ix:ix+cx]

    extent=(fibers[0]-dfib/2, fibers[-1]+dfib/2, fluxes[0]-dflux/2, fluxes[-1]+dflux/2)
        
    #- Plot!
    P.figure(figsize=(11, 7.5))
    P.subplots_adjust(bottom=0.10, top=0.95, left=0.10, right=0.95)
    P.imshow(image, interpolation='nearest', origin='lower', extent=extent, aspect='auto')
    P.xlabel('Fiber Number')
    P.ylabel('Flux row')
    P.title(filename)
    
    if opts.png:
        base, ext = os.path.splitext(filename)
        pngfile = base + '.png'
        P.savefig(pngfile, dpi=72)

P.show()
        
        

