#- Utility classes/functions

import numpy as N

class MinMax(object):
    def __init__(self, xmin, xmax):
        """Utility class for wrapping min/max"""
        self._min = xmin
        self._max = xmax
        
    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def n(self):
        """max-min (e.g. if min,max are array ranges)"""
        return self._max - self._min

    @property
    def delta(self):
        """max-min"""
        return self._max - self._min
        
def minmax(x):
    """Return (xmin, xmax)"""
    if isinstance(x, N.ndarray):
        return x.min(), x.max()
    else:
        return min(x), max(x)
        