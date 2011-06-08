#!/usr/bin/env python
#PBS -j oe
#PBS -W umask=0022

"""
Calculate Gauss-Hermite PSF 
"""

import os
import sys

from bbspec.spec2d.arcmodel2D import arcmodel2D

if 'PBS_O_WORKDIR' in os.environ: os.chdir(os.environ['PBS_O_WORKDIR'])
else: print "$PBS_O_WORKDIR isn't set; staying where I am."

try:
    arcid = os.environ['ARCID']
    flatid = os.environ['FLATID']
    bundle = int(os.environ['BUNDLE'])
except KeyError: pass

print 'Calling model_arc for %s %s %d' % (arcid, flatid, bundle)
m = arcmodel2D(arcid,arcid)
m.setarc_flat(arcid,flatid)
print 'Writing %s, %s ' %  m.model_arc(bundle)
