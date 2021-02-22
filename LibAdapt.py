
import copy
import bisect
import pywt
import numpy               as np

import Auxiliary.LibAux    as LibA

# =============================================================================
# 
# =============================================================================
    

def AdaptMain( w_S, i_S, Ana, slices, AdaptList ):
    w_W = np.matmul(Ana, w_S)
    wC  = pywt.array_to_coeffs(w_W, slices, output_format='wavedec')
    iC  = pywt.array_to_coeffs(i_S, slices, output_format='wavedec')
    
    iCold = copy.deepcopy(iC)
    iC, iCdel, iCadd, LAdapt = Adapt( iC, wC, AdaptList )
    ChangeDOFS( wC, iCold, iCdel, iCadd )

    
    i_S, slices = pywt.coeffs_to_array(iC)
    DofList     = LibA.i_S2Dof( i_S )
    
    return DofList, i_S, slices




# add new fluctuation step to the old coefficients
def GetNewSolution(iC, wCold, ddw ):
    
    counter = 0
    wC      = copy.deepcopy( iC )
    
    
    for ii in range(0,len(iC)):
        for jj in range(0,len(iC[ii])):
            if iC[ii][jj] == 1:
                wC[ii][jj] = ddw[counter] + wCold[ii][jj]
                counter += 1
    
    return wC

# Find the top coefficients (or leaves) of the dyadic tree
def FindTopCoeffs( iC ):
    iCmax = copy.deepcopy( iC )
    iCmax = LibA.TruncunateObject( iCmax, 10**16 )

    
    for ii in range(0, len(iC ) ):
        for jj in range(0, len( iC[ii]) ):
            
            iCij = iC[ii][jj]
            
            if iCij == 1:
                iCmax[ii][jj] = 1
                
                iCmax[ii - 1][int( jj / 2 )] = 0
                # iCmax[ii - 1][int( jj / 2 )] = 0
                
    return iCmax

# =============================================================================
# 
# =============================================================================
    
# Calculate the new index coefficient array and keep track of what is added/removed
def Adapt( iC, wC, AdaptList ):
    Ndel        = AdaptList[0]
    mintol      = AdaptList[1]
    maxtol      = AdaptList[2]
    MaxLevel    = AdaptList[3]
    
    iCdel = copy.deepcopy( iC )
    iCdel = LibA.TruncunateObject( iCdel, 10**16 )
    iCadd = copy.deepcopy( iCdel )
    
    AdaptLevel = 0
    iCmax = FindTopCoeffs( iC )
    
    if Ndel == -1:
        for ii in range(0, len(wC)):
            for jj in range(0, len(wC[ii])):
                if abs( wC[ii][jj] ) < mintol:
                    if iCmax[ii][jj] == 1:
                        if ii > 1:
                            iCdel[ii][jj] = 1
                            iC[ii][jj] = 0
                            
                            
        for ii in range(1, len(wC[:-1])):
            for jj in range(0, len(wC[ii])):
                if iCmax[ii][jj] == 1:
                    if abs( wC[ii][jj] ) > maxtol:
                        L = len(iCadd[ii+1])
                        if  not iC[ii + 1][2*jj - L] == 1:
                            if  not iC[ii + 1][2*jj - L + 1] == 1:
                                
                                iCadd[ii+ 1][2*jj  - L] = 1
                                iCadd[ii+ 1][2*jj  + 1- L] = 1
                                iC[ii+ 1][2*jj - L] = 1
                                iC[ii+ 1][2*jj + 1 - L] = 1
                                if ii > AdaptLevel:
                                    AdaptLevel = AdaptLevel + 1
    
    elif Ndel > 0:
        irow = list( -1*  np.ones((1,Ndel))[0] )
        jrow =  list(-1*  np.ones((1,Ndel)) [0] )
        crow =  list( 10**32*np.ones((1,Ndel))[0] ) 
        for ii in range(0, len(wC[0:-2])):
            for jj in range(0, len(wC[ii])):
                if iCmax[ii][jj] == 1:
                    if ii > 1:

                        index = bisect.bisect(crow,abs(wC[ii][jj]))
                        
                        if index < len(crow):

                            bisect.insort(crow,abs(wC[ii][jj]))
                            irow.insert(index, ii)
                            jrow.insert(index, jj)
                            
                            crow = crow[0:-1]
                            irow = irow[0:-1]
                            jrow = jrow[0:-1]
                            
                            
    uC = copy.deepcopy(iC)
    for ll in range( len(iC) ):
        uC[ll] = np.zeros_like( uC[ll] )
        for ii in range( len( iC[ll]) ):
            if iC[ll][ii] == 1:
                uC[ll][ii] = wC[ll][ii]
    return iC, uC

# Calculate the number that is associated to the Degree of freedom
def GetDOFS( iC ):
    wDOFS = []
    
    dof = 0
    for ii in range(0,len(iC)):
        for jj in range(0,len(iC[ii])):
            if iC[ii][jj] == 1:
                wDOFS.append( dof )
    
            dof += 1
            
    return wDOFS

# Add and remove the degree of freedoms in the Test and Trail basis
def ChangeDOFS( wC,  iCold, iCdel, iCadd ):
    
    wDOFS = GetDOFS( iCold )
    delDOFS = GetDOFS( iCdel )
    addDOFS = GetDOFS( iCadd )
    
    for d in delDOFS:
        i = np.where( np.array(wDOFS) == d )

        i = i[0][0]
        
        # Basis.pop( i )
        wDOFS.pop( i )
        
    for a in addDOFS:
        
        bisect.insort( wDOFS, a )
    
        i = np.where( np.array( wDOFS ) == a )
        i = i[0][0] - 1
        
        # Basis.add( i, a )

 
    print("Basis updated")