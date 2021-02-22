
import numpy as np
from scipy.interpolate import interp1d

import pywt
import random
import math
import copy
import pickle


def DyadicToArray( X, S ):
    # Initialize x,y and c rows
    xx = []
    yy = []
    cc = []
    
    # Find Smax
    Smax = 0
    for s in S:
        sm = max( abs( s ) )
        Smax = max( sm, Smax )
    ii = 0
    
    for s in S:
 
        for jj in range(0,len(s)):
            
            if not abs( s[jj] ) == 0.:
                xx.append( X[ii][jj] )
                yy.append(ii)
                cc.append( abs( s[jj] )/Smax )
                
        ii += 1
        
    return xx, yy, cc

# =============================================================================
# Object = TruncunateObject( Object, error)
# Returns Object (that can consist of lists in arrays in tuples in lists etc)
# with all numbers that are less than zero set to none
#
#INPUT:
#    Object (array, tuple, list or float)
#        Object of which its values should be filtered
#    error (float)
#        Threshold
#        
#OUTPUT:
#    Object (array, tuple, list or float)
#        Truncunated Object
#    
#CHANGED:
#    Object that is given is the same as the output one, so the original is changed
# =============================================================================

def TruncunateObject( Object, error ):
    if error == -1:
        return Object

    elif type( Object ) == tuple :     
        L = len(Object)
        Object = list(Object)
        
        for ii in range(0,L):
            Object[ii] = TruncunateObject( Object[ii], error )
            
        Object = tuple(Object)
        return Object
    elif type( Object ) == list:     
        L = len(Object)
        
        
        for ii in range(0,L):
            Object[ii] = TruncunateObject( Object[ii], error )
        return Object
        
    elif type( Object ) == np.ndarray:
        Object = np.where( abs( Object ) < error, Object*0, Object)
        return Object
        
    elif type( Object ) == float:
        if abs( Object ) < error:
            return 0.
        
        else:
            return Object
    

# =============================================================================
#        
# =============================================================================

def MakeX( Level ):
    Resolution  = 1/(2**Level)
    X_array     = np.arange( 0 , 1 + Resolution , Resolution )

    return X_array       

# =============================================================================
# 
# =============================================================================
# Makes x values for each of the wavelets and scaling functions in the coefficient array
def MakeXCoeffs( Coeffs, L):
    X = copy.deepcopy( Coeffs )
    Xarray = np.array([])
    for ii in range(0,len(X)):
        BB= len(X[ii])*2
        n = 0
        
        for jj in range(0, BB):
           if ii == 0:
               if jj % 2 == 0:
                   X[ii][n] = float(jj)/ float(BB)  
                   n += 1
                   Xarray = np.append( Xarray,float(jj)/ float(BB) )
           else:
               if jj % 2 == 1:
                   X[ii][n] = float(jj)/ float(BB)  
                   n += 1
                   Xarray = np.append( Xarray,float(jj)/ float(BB) )
   
    # Xarray = np.sort(Xarray)
    X[0]  = X[0] - X[0][0]
    return X, Xarray         
 
def MakeXArrayW(  L0, Lmax ):
    Xarray = np.array([])
    
    Nl = 2**L0
    BB = Nl*2
    for jj in range(0,BB):
        if jj % 2 == 0:
            Xarray = np.append( Xarray,float(jj)/ float(BB) )
            
    for ii in range(L0, Lmax):
        Nl = 2**ii
        BB = Nl*2
        
        for jj in range(0,BB):
            if jj % 2 == 1:
                Xarray = np.append( Xarray,float(jj)/ float(BB) )
    return Xarray 
 
def MakeSine( Level, Power ):
    X_array = MakeX( Level )
    Y_array = []
    
    for x in X_array:
        Y_array.append( math.sin( x*2*math.pi )**Power )
        
    X_array = X_array[0:-1]
    Y_array = Y_array[0:-1]
    return X_array, Y_array


def MakeDisCon( Level):
    X_array, Y_array = MakeSine(Level, 1)
    Y_array = np.zeros_like(Y_array)
    for ii in range(0, len(Y_array)):
        if 0.25 < X_array[ii] and X_array[ii] < 0.75:
            Y_array[ii] += 1
    return X_array, Y_array       
# =============================================================================
#  p = SmoothPolynominal( Order, Rand = [] )
#  Creates a interpolated function of a certain order based on a random or 
#  given signal on a dyadic grid
#
#  INPUT:
#      -Order (int)
#          Order of the function
#      -Rand (List) [Optional]
#          List of length 8(!) of coefficients that are used to create the 
#          signal
#  
#  OUTPUT:
#      -p (function)
#          Function that is made by interpolating the random values 
#          (or from Rand) with order "Order"
#  
#  CHANGES: 
#      -None
#    
# =============================================================================
def SmoothPolynominal( Order, Rand = [] ):
    X_arrayD = np.array([-3,-2.5,-2,-1.5,-1 , -0.75,    -0.5,   -0.25])
    xx = MakeX(3)
    X_arrayD = np.append(X_arrayD, xx)
    X_arrayD = np.append( X_arrayD, np.array([1.25, 1.5, 1.75, 2., 2.5, 3, 3.5]))
    

    L = len(xx) - 1
    if len(Rand) == 0:
        Yrand = []
        for ii in range( 0, L ):
            Yrand.append( random.random() )
        Yrand[-1] = Yrand[0]

    else:
        Yrand = Rand
    
    
    
    
    Y_arrayD = np.array( Yrand )
    Y_arrayD = np.append(Y_arrayD, Yrand)         
    Y_arrayD = np.append(Y_arrayD, Yrand )   
    
    tup = ( Y_arrayD[-2], Y_arrayD[1] )
    

    p = interp1d(X_arrayD, Y_arrayD, kind=Order, fill_value = tup)
    return p 
   
    

def i_C2List( i_C ):
    DofList   = []
    LevelList = []
    IndexList = []
    dof = 0
    for level in range(0,len(i_C)):
        for index in range(0,len(i_C[level])):
            if i_C[level][index] == 1:
                DofList.append( dof )
                LevelList.append( level )
                IndexList.append( index )
    
            dof += 1
            
    return DofList, LevelList, IndexList
