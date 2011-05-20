# Quick visual check of Gauss-Hermite PSF object
# ASB 2011may

import numpy as n
import matplotlib as m
m.interactive(True)
from matplotlib import pyplot as p
from bbspec.spec2d import psf

f = 'spBasisPSF-r1-00122444.fits'

myargs = {'interpolation': 'nearest', 'origin': 'lower', 'hold': False}

PSFobj = psf.load_psf(f)

bigim = n.zeros((PSFobj.npix_y, PSFobj.npix_x), dtype=float)

for ispec in range(20):
    for iflux in range(200, 3800, 200):
        xs, ys, im = PSFobj.pix(ispec, iflux)
        bigim[ys,xs] += im


p.imshow(bigim, **myargs)

