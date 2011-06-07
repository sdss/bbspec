from bbspec.spec2d.arcmodel2D import arcmodel2D
import sys

"""
Wrapper script to call Parul's 2D arc PSF modeling code

Usage:
  python make-my-psf.py arcid flatid

where arcid and flatid are identifier strings
of the form, e.g., r1-XXXXXXXX
"""

arcid = sys.argv[1]
flatid = sys.argv[2]
i_bund = int(sys.argv[3])

m = arcmodel2D('.','.')
m.setarc_flat(arcid,flatid)
(f1, f2) = m.model_arc(i_bund)
print 'Wrote '  + f1
print 'Wrote '  + f2
