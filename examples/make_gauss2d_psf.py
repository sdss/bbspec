#!/usr/bin/env python

"""
Write a toy PSF for a rotated 2D Gaussian
"""

import numpy as N
import pyfits

#- Hardcoded constants
class C:
    npix_x = 70    #- image dimensions
    nspec  = 7     #- number of spectra
    npix_y = 50    #- image dimensions
    nflux  = 50    #- number of flux bins per spectrum
    dx = 10        #- spectral spacing on the image

#- Testing single fiber extraction
### C.npix_x = 10
### C.nspec = 1

#- spatially varying PSF
def psf_params(x0, y0):
    """
    return spatially dependent PSF parameters:
    amplitude, major axis, minor axis, angle
    """
    a, b = 2.0, 2.0   #- ellipse major and minor axes in center
    ### a, b = 1.0, 1.0   #- ellipse major and minor axes in center
    c = 1.0           #- amplitude
    
    #- Transform to range -1 to 1
    x1 = (x0 - C.npix_x/2.0) / (C.npix_x/2.0)
    y1 = (y0 - C.npix_y/2.0) / (C.npix_y/2.0)
    # y1 = (y0 - C.nflux/2.0) / (C.nflux/2.0)
    
    #- major axis varies with position
    a += 1.0 * (N.abs(x1) + N.abs(y1))
    
    #- PSF angle varies with position
    theta = 0.25 * N.pi * N.sin(x1*N.pi/2) * N.sin(y1*N.pi/2)
    
    #- Test: fix theta
    ### theta *= 0.0
    
    #- convert non-varying parameters to arrays if input is array
    if isinstance(x0, N.ndarray):
        c = N.ones(x0.shape) * c
        b = N.ones(x0.shape) * b
    elif isinstance(y0, N.ndarray):
        c = N.ones(y0.shape) * c
        b = N.ones(y0.shape) * b
    
    return c, a, b, theta
    

def write_psf(filename):
    #- independent variable
    t = N.arange(C.nflux) + (C.npix_y - C.nflux)/2.0
    
    #- x is constant in y
    #- y matches z for now
    #- make up something for wavelength
    x = N.zeros( (C.nspec, C.nflux) )
    y = N.zeros( (C.nspec, C.nflux) )
    w = N.zeros( (C.nspec, C.nflux) )
    for i in range(C.nspec):
        x[i] = C.dx*(i+0.5)
        y[i] = t
        w[i] = N.logspace(N.log10(3500), N.log10(8500), C.nflux)
   
    #- PSF parameters
    c = N.zeros( (C.nspec, C.nflux) )  #- Amplitude  
    a = N.zeros( (C.nspec, C.nflux) )  #- Major axis
    b = N.zeros( (C.nspec, C.nflux) )  #- Minor axis
    t = N.zeros( (C.nspec, C.nflux) )  #- Rotation angle

    for i in range(C.nspec):
        cc, aa, bb, tt = psf_params(x[i], y[i])
        c[i] = cc
        a[i] = aa
        b[i] = bb
        t[i] = tt

    hdus = list()
    hdus.append(pyfits.PrimaryHDU(data=x))
    hdus.append(pyfits.ImageHDU(data=y))
    hdus.append(pyfits.ImageHDU(data=w))
    hdus.append(pyfits.ImageHDU(data=c))
    hdus.append(pyfits.ImageHDU(data=a))
    hdus.append(pyfits.ImageHDU(data=b))
    hdus.append(pyfits.ImageHDU(data=t))
    
    hdus = pyfits.HDUList(hdus)
    
    hdus[0].header.update('PSFTYPE', 'GAUSS2D', 'Rotated 2D Gaussian')
    hdus[0].header.update('NPIX_X', C.npix_x, 'number of image pixels in the X-direction')
    hdus[0].header.update('NPIX_Y', C.npix_y, 'number of image pixels in the Y-direction')
    hdus[0].header.update('NSPEC',  C.nspec,  'number of spectra [NAXIS2]')
    hdus[0].header.update('NFLUX',  C.nflux,  'number of flux bins per spectrum [NAXIS1]')
    
    hdus[0].header.update('PSFPARAM', 'X', 'X position as a function of flux bin')
    hdus[1].header.update('PSFPARAM', 'Y', 'Y position as function of flux bin')
    hdus[2].header.update('PSFPARAM', 'Wavelength', 'log10(wavelength [Angstroms])')
    hdus[3].header.update('PSFPARAM', 'Amplitude',  'Gaussian amplitude')
    hdus[4].header.update('PSFPARAM', 'MajorAxis',  'Gaussian major axis')
    hdus[5].header.update('PSFPARAM', 'MinorAxis',  'Gaussian minor axis')
    hdus[6].header.update('PSFPARAM', 'Angle',      'Gaussian rotation angle')
    
    hdus.writeto(filename, clobber=True)

#-----

if __name__ == '__main__':
    import optparse

    parser = optparse.OptionParser(usage = "%prog [options]")
    parser.add_option("-o", "--output", type="string",  help="output PSF file")

    opts, args = parser.parse_args()
    
    if opts.output is None:
        opts.output = 'psf_gauss2d.fits'
        
    print "Writing %s" % opts.output    
    write_psf(opts.output)
