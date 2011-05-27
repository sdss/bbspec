#!/usr/bin/env python

"""
Take PSF files with individual bundles, and merge them into a single file
"""

import sys
import numpy as N
import pyfits
import optparse

def blank_psf(template):
    #- Semi-hardcode dimensions
    nspec = 500
    nflux = template[0].header['NFLUX']
    data_shape = (nspec, nflux)
    
    #- Create HDUs with blank data
    hdus = list()
    hdus.append(pyfits.PrimaryHDU(data=N.zeros( data_shape )))
    for i in range(1, len(template)):
        hdus.append(pyfits.ImageHDU(data=N.zeros(data_shape)))

    #- Convert to pyfits HDUList object
    hdus = pyfits.HDUList(hdus)
    
    #- Update the headers
    hdus[0].header.update('NSPEC', nspec)
    hdus[0].header.update('NFLUX', nflux)
    
    for i in range(0, len(template)):
        std_keywords = hdus[i].header.keys()
        for keyword in template[i].header.keys():
            if keyword not in std_keywords:
                hdus[i].header.update(keyword, template[i].header[keyword])

    return hdus

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
hdus = blank_psf(fx)
fx.close()

#- Load each HDU
for filename in args[0:]:
    fits = pyfits.open(filename)
    i0 = fits[0].header['BUNDLELO'] * 20
    i1 = (fits[0].header['BUNDLEHI']+1) * 20
    
    for ihdu in range(len(fits)):
        ### print filename, ihdu, fits[ihdu].data.shape, hdus[ihdu].data.shape
        if fits[ihdu].data.shape == hdus[ihdu].data.shape:
            hdus[ihdu].data[i0:i1] = fits[ihdu].data[i0:i1]
        else:
            hdus[ihdu].data[i0:i1] = fits[ihdu].data
        
    fits.close()
        
hdus.writeto(opts.output, clobber=True)


