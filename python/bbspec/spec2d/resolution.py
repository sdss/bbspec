#!/usr/bin/env python

"""
Resolution matrix class
"""

import numpy as N
import scipy.sparse
from time import time
import sys

class ResolutionMatrix(scipy.sparse.dia_matrix):
    """
    Resolution Matrix for a single spectrum.
    
    Convert to CSR format before using for math: R = R.tocsr()
    """
    
    def __init__(self, diagonals=None, full_range=None, good_range=None):
        """
        Create ResolutionMatrix with diagonals[width, nflux].
        
        Optional inputs:
            full_range : (iflux_min, iflux_max) covered by this data
            good_range : (iflux_lo, iflux_hi) range to trust
            --> i.e. data may contain edge effects; good_range should be used
        """

        #- Initialize sparse matrix
        width, nflux = diagonals.shape
        if width%2 == 0:
            raise ValueError('diagonals width must be odd')
            
        offsets = list(range(-(width/2), (width/2)+1))
        super(ResolutionMatrix, self).__init__((diagonals, offsets), \
                shape=(nflux, nflux) )

        #- Default ranges start at 0 and cover everything
        if full_range is None:
            full_range = (0, nflux)
        if good_range is None:
            good_range = full_range

        #- Save ranges
        self.full_range = full_range
        self.good_range = good_range

    @property
    def nflux_full(self):
        return self.full_range[1] - self.full_range[0]

    @property
    def nflux_good(self):
        return self.good_range[1] - self.good_range[0]

    def merge(self, Rx):
        """
        Merge this resolution matrix with another.
        """
        #- Check ranges
        if self.full_range[0] > Rx.full_range[0] or \
            self.full_range[1] < Rx.full_range[1]:
            print(self.full_range, Rx.full_range, file=sys.stderr)
            raise ValueError('full_range of new matrix must be a subset of original')

        if self.data.shape[0] != Rx.data.shape[0]:
            raise ValueError("matrix bandwidths must agree (%d != %d)" % \
                (self.data.shape[0], Rx.data.shape[0]))
                
        start = Rx.full_range[0] - self.full_range[0]
        nx = Rx.full_range[1] - Rx.full_range[0]
        self.data[:, start:start+nx] = Rx.data.copy()

    @staticmethod
    def blank(bandwidth, nflux, full_range=None, good_range=None):
        d = N.zeros((bandwidth, nflux))
        return ResolutionMatrix(d, full_range=full_range, good_range=good_range)
  
    #--- IO methods ---
    @staticmethod
    def from_array(a, bandwidth=15, **kwargs):
        R = scipy.sparse.dia_matrix(a)
        b = int(bandwidth)/2
        ii = (-b <= R.offsets) & (R.offsets <= +b)
        if sum(ii) == bandwidth:
            diagonals = R.data[ii]
        else:
            nflux = R.data.shape[1]
            diagonals = N.zeros((bandwidth, nflux))
            diagonals[R.offsets[ii]+b] = R.data[ii]
            
        return ResolutionMatrix(diagonals, **kwargs)
    
    @staticmethod
    def from_diagonals(d, **kwargs):
        """
        Create a sparse ResolutionMatrix from a block of diagonals.
        
        Format is the same as for scipy.sparse.dia_matrix:
        
        input diagonals:
        d = [[1, 2, 3, 4],
             [2, 4, 6, 8],
             [9, 8, 7, 6]]
             
        become:
        r = [[2 8 0 0]
             [1 4 7 0]
             [0 2 6 6]
             [0 0 3 8]]
             
        if d is 3D, create a list of ResolutionMatrix objects
        """

        assert(d.ndim == 2 or d.ndim == 3)
        if d.ndim == 3:
            nR, bandwidth, nflux = d.shape
            return [ResolutionMatrix.from_diagonals(d[i], **kwargs) for i in range(nR)]
        else:
            return ResolutionMatrix(d, **kwargs)
        
    @staticmethod
    def to_diagonals(R):
        """
        returns diagonals within bandwidth from R as dense data array

        R is either a ResolutionMatrix, or an array of them
        
        BUGS (features!):
        - Assumes R is fully dense along main diagonal
        """
        if type(R) == list or type(R) == tuple:
            n = len(R)
            d = [ResolutionMatrix.to_diagonals(R[i]) for i in range(n)]
            return N.array(d)
        else:
            return R.data
                          

class ResolutionMatrixLIL(scipy.sparse.lil_matrix):
    """
    Resolution matrix for a single spectrum
    
    Convert to CSR before using for math: R.tocsr()
    """
        
    def __init__(self, data, full_range=None, good_range=None):
        """
        Initialize ResolutionMatrix
        
        Required inputs:
            data : square input matrix (dense or sparse)
        
        Optional inputs:
            full_range : (iflux_min, iflux_max) covered by this data
            good_range : (iflux_lo, iflux_hi) range to trust
            --> i.e. data may contain edge effects; good_range should be used
        """

        #- Initialize sparse matrix
        super(ResolutionMatrixLIL, self).__init__(data)

        if type(data) == list:
            return
        
        #- Input data should be square
        assert data.shape[0] == data.shape[1]
        
        #- Default ranges start at 0 and cover everything
        if full_range is None:
            full_range = (0, data.shape[0])
        if good_range is None:
            good_range = full_range

        #- Check ranges match data
        n = full_range[1] - full_range[0]
        assert data.shape == (n,n)
        
        #- Save ranges
        self.full_range = full_range
        self.good_range = good_range
        
    @property
    def nflux_full(self):
        return self.full_range[1] - self.full_range[0]

    @property
    def nflux_good(self):
        return self.good_range[1] - self.good_range[0]

    def merge(self, Rx):
        """
        Merge with another ResolutionMatrix Rx which is a subset of this one        
        """
        
        #- In rows, keep everything; in columns, keep good range
        iirow = slice(Rx.full_range[0] - self.full_range[0], \
                      Rx.full_range[1] - self.full_range[0] )
        iicol = slice(Rx.good_range[0] - self.good_range[0],
                      Rx.good_range[1] - self.good_range[0] )
              
        #- Create slices
        jjrow = slice(0, Rx.shape[0])
        ngood = Rx.good_range[1] - Rx.good_range[0]
        j0 = Rx.good_range[0]-Rx.full_range[0]
        jjcol = slice(j0, j0+ngood)

        #- Check edges
        if iirow.start < 0:
            drow = -iirow.start
            iirow = slice(0, iirow.stop)
            jjrow = slice(drow, jjrow.stop)
        n = self.full_range[1] - self.full_range[0]
        if iirow.stop > n:
            drow = iirow.stop - n
            iirow = slice(iirow.start, iirow.stop-drow)
            jjrow = slice(jjrow.start, jjrow.stop-drow)
        
        #- Merge
        ### self[iirow, iicol] = Rx[jjrow, jjcol]
        
        #- Much faster to merge via dense arrays (!)
        R = self.toarray()
        R[iirow, iicol] = Rx.toarray()[jjrow, jjcol]
        R = ResolutionMatrixLIL(R)
        self.data = R.data.copy()
        self.rows = R.rows.copy()        

    #--- IO methods ---
    @staticmethod
    def from_diagonals(d, **kwargs):
        """
        Create a sparse ResolutionMatrix from a block of diagonals.
        
        Format is the same as for scipy.sparse.dia_matrix:
        
        input diagonals:
        d = [[1, 2, 3, 4],
             [2, 4, 6, 8],
             [9, 8, 7, 6]]
             
        become:
        r = [[2 8 0 0]
             [1 4 7 0]
             [0 2 6 6]
             [0 0 3 8]]
             
        if d is 3D, create a list of ResolutionMatrix objects
        """

        assert(d.ndim == 2 or d.ndim == 3)
        if d.ndim == 3:
            nR, bandwidth, nflux = d.shape
            return [ResolutionMatrixLIL.from_diagonals(d[i], **kwargs) for i in range(nR)]
        else:
            bandwidth, nflux = d.shape
            offsets = list(range(-(bandwidth/2), (bandwidth/2)+1))
            R = scipy.sparse.dia_matrix( (d, offsets), shape=(nflux,nflux) )
            return ResolutionMatrixLIL(R, **kwargs)
        
    @staticmethod
    def to_diagonals(R, bandwidth=15):
        """
        returns diagonals within bandwidth from R as dense data array

        R is either a ResolutionMatrix, or an array of them
        
        BUGS (features!):
        - Assumes R is fully dense along main diagonal
        """
        if type(R) == list or type(R) == tuple:
            n = len(R)
            d = [ResolutionMatrixLIL.to_diagonals(R[i], bandwidth=bandwidth) for i in range(n)]
            return N.array(d)
        else:
            R = R.todia()
            b = int(bandwidth)/2
            ii = (-b <= R.offsets) & (R.offsets <= +b)
            if sum(ii) == bandwidth:
                return R.data[ii]
            else:
                nflux = R.data.shape[1]
                d = N.zeros((bandwidth, nflux))
                d[R.offsets[ii]+b] = R.data[ii]
                return d
                
        

if __name__ == '__main__':
    A = ResolutionMatrix( N.eye(50) ) 
    print("Input dimensions:", A.shape)
    A[0:2, 0:2] = 1
    A[3:5, 3:5] = 2
    A[10:12, 10:12] = 3
    A[0:40,20] = 5
    d = ResolutionMatrix.to_diagonals(A)
    print("Diagonal matrix dimensions:", d.shape)
    Ax = ResolutionMatrix.from_diagonals(d)
    print("Reconstructed sparse matrix dimensions:", Ax.shape)
