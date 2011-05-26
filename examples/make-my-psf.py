from bbspec.spec2d import func_arcmodel2D
import sys

"""
Wrapper script to call Parul's 2D arc PSF modeling code

Usage:
  python make-my-psf.py arcid flatid

where arcid and flatid are identifier strings
of the form, e.g., r1-XXXXXXXX
"""

(f1, f2) = func_arcmodel2D.model_arc(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
#(f1, f2) = func_arcmodel2D.model_arc(sys.argv[1], sys.argv[2])
print 'Wrote '  + f1
print 'Wrote '  + f2
