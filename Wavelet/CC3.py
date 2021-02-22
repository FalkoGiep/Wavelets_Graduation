"""
Falko Giepmans, 05-08-2020

This is a code to calculate three-term connection coefficients for all(?) finite
supported discrete wavelet families on an unbounded interval.

This is a fix and an elaboration of the code provided by Manuels 
( https://github.com/manuels/db_cc3 ), where three-term connection coefficients
are calculated for Daubechies wavelets.

Most of this code is based on the equations given in ( Besora, 2004 ), which in
turn is based on the theory of ( Latto et al, 1992 ). 

References:
Besora, J., Galerkin Wavelet Method for Global Waves in 1D, 2004
Goedecker, S., Wavelets and their application for the solution of partial 
    differential equations in physics, 2009
Latto, A., Resnikoff, H., Tenenbaum, E., The evaluation of connection 
    coefficients of compactly supported wavelets, 1992

"""

import itertools
import warnings
import scipy

import numpy as np

from   scipy.special import factorial

# =============================================================================
# 
# =============================================================================

def moment(a, i, j):
    """
    The moment M^j_i for all(?) scaling functions (Goedecker, 2009)
    
    INPUT:
    a       :  np.array         Wavelet filter
    i       :  int
    j       :  int
    
    OUTPUT:
    M^j_i   :  float
    """
    N = a.size
    return (i-(int( N/2)-1))**j 


def threeterm_connection_coefficients(a, d1, d2, d3):
    """
    Calculates the three-term connection coefficients CC^(d1,d2,d3) of a wavelet.
    The three-term connection coefficients (CC) can be directly calculated for 
    d1 = 0. When d1 is not 0 integration by parts can be used to calculate the CC 
    of a summation of other CCs for which d1 = 0.
    
    INPUT:
    a       :  np.array         Wavelet filter
    d1,d2,d3:  int              Order of derivative
    
    OUTPUT:
    idx     :  lambda function  Plots (l,m) to correct index of CC
    indices :  np.array         non-zero (l,m) pairs of the connection coefficients
    CC      :  np.array         non-zero connection coefficients 
    """
    if d1 == 0:
        idx, indices, CC = fundamental_threeterm_connection_coefficients(a, d2, d3)
        return idx,indices, CC
    else:
        idx1,indices1,  CC1  = threeterm_connection_coefficients(a, d1-1, d2+1, d3)
        idx2,indices2,  CC2  = threeterm_connection_coefficients(a, d1-1, d2, d3+1)
        assert indices1 == indices2
        return idx1,indices1, -CC1 - CC2
    

def fundamental_threeterm_connection_coefficients(a, d2, d3):
    """
    Calculates the three-term connection coefficients CC^(0,d2,d3) of a wavelet.
    
    INPUT:
    a       :  np.array         Wavelet filter
    d2,d3   :  int              Order of derivative
    
    OUTPUT:
    idx     :  lambda function  Plots (l,m) to correct index of CC
    indices :  np.array         non-zero (l,m) pairs of the connection coefficients
    CC      :  np.array         non-zero connection coefficients 
    """
    N = a.size
    d = d2 + d3

    Tindices = list(set((l,m) for l,m in itertools.product(range(-(N-2), (N-2)+1), repeat=2)
                      if abs(l-m) < (N-1)))

    idx = lambda l,m: Tindices.index((l,m))


    if np.amax([d2,d3]) >= N/2:
        msg = 'Calculation of connection coefficients for {},{} > g = N/2 is invalid!'.format(d2,d3)
        warnings.warn(msg)


    """
    Eigen-value problem due to the scaling equations (Latto et al, 1992):
    """
    T = np.zeros([len(Tindices), len(Tindices)])
    
    for l,m in Tindices:
        for i,j,k in itertools.product(range(N), repeat=3):
            if (2*l+j-i, 2*m+k-i) not in Tindices:
                continue # skip the CC which are zero anyway
            T[idx(l,m), idx(2*l+j-i, 2*m+k-i)] += a[i]*a[j]*a[k]


    T -= 2**(1-d)*np.eye(len(Tindices)) 
    b = np.zeros([len(Tindices)])


    """
    The eigen-value problem above is rank-deficient, so extra equations are 
    needed to solve the system. We can use a adaptation of the moment equations
    to obtain d1+d2 extra homogenous equations (Latto et al, 1992).
    """
    M = np.zeros([d2, len(Tindices)])
    k = 0 if (d3 % 2) == 1 else 1
    
    for q in range(d2):
        for j in range(-(N-2), (N-2)+1):
            if (j, k) in Tindices:
                M[q, idx(j, k)] += moment(a, j, q)
    A = np.vstack([T,M])
    b = np.hstack([b, np.zeros([d2])])


    M = np.zeros([d3, len(Tindices)])
    j = 0 if (d2 % 2) == 1 else 1
    
    for q in range(d3):
        for k in range(-(N-2), (N-2)+1):
            if (j, k) in Tindices:
                M[q, idx(j, k)] += moment(a, k, q)
    A = np.vstack([A,M])
    b = np.hstack([b, np.zeros([d3])])

    
    """
    Since the eigenvector is determined up to a constant, we alse need a 
    normalization equation ( Latto et al, 1992):
    """
    M = np.zeros([1, len(Tindices)])
    for j,k in itertools.product(range(-(N-2), (N-2)+1), repeat=2):
        if (j, k) in Tindices:
            M[0, idx(j, k)] += moment(a, j, d2)*moment(a, k, d3)
            
    A = np.vstack([A,M])
    b = np.hstack([b, [factorial(d2)*factorial(d3)]])


    """
    A least squares algorithm is used to solve the over-determined system.
    One can also use np.linalg.lstsq with rcond = None. In my experience 
    however, np.linalg.lstsq does not always return residuals correctly.
    """
    CC, residuals, rank, singular_values = scipy.linalg.lstsq( A,b)

    if abs( residuals ) >= 10**-20:
        msg = 'Residue of lstsq algorithm is {:.2e}!'.format(residuals)
        warnings.warn(msg)

    return idx, Tindices, CC

