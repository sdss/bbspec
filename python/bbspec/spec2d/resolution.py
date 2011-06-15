#!/usr/bin/env python

"""
Resolution matrix class
"""

import numpy as N
import scipy.sparse

class ResolutionMatrix(scipy.sparse.lil_matrix):
    """
    Subclass of scipy.sparse.lil_matrix with added I/O functions
    to/from a dense block diagonal format, e.g for writing to fits.
    
    Convert to another type of sparse matrix, e.g. R.tocsr() before
    doing matrix algebra with this.
    """    
        
        
    #--- IO methods ---
    @staticmethod
    def from_diagonals(a):
        """
        Create a sparse ResolutionMatrix from a block of diagonals.
        
        Format is the same as for scipy.sparse.dia_matrix:
        
        input diagonals:
        a = [[1, 2, 3, 4],
             [2, 4, 6, 8],
             [9, 8, 7, 6]]
             
        become:
        r = [[2 8 0 0]
             [1 4 7 0]
             [0 2 6 6]
             [0 0 3 8]]
        """
        b, n = a.shape
        offsets = range(-(b/2), (b/2)+1)
        R = scipy.sparse.dia_matrix( (a, offsets), shape=(n,n) )
        return ResolutionMatrix(R)
        
    def diagonals(self, bandwidth=15):
        """
        return block array of diagonals
        
        Use ResolutionMatrix.from_diagonals() to recreate sparse array.
        Block diagonals follow format needed to make scipy.sparse.dia_matrix.
        """
        block = N.zeros((bandwidth, self.shape[0]))
        A = self.toarray()
        for i in range(bandwidth):
            x = A.diagonal(i-(bandwidth/2))
            if i - (bandwidth/2) <= 0:
                block[i, 0:len(x)] = x
            else:
                block[i, i-bandwidth/2:] = x
                
        return block

if __name__ == '__main__':
    A = ResolutionMatrix( (50, 50) ) 
    print "Input dimensions:", A.shape
    A[0:2, 0:2] = 1
    A[3:5, 3:5] = 2
    A[10:12, 10:12] = 3
    d = A.diagonals()
    print "Diagonal matrix dimensions:", d.shape
    Ax = ResolutionMatrix.from_diagonals(d)
    print "Reconstructed sparse matrix dimensions:", Ax.shape
    print "RMS of differences:", N.std((A-Ax).toarray())