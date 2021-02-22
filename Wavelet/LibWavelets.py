import numpy as np
import pywt
import copy
import itertools
from Wavelet.CC3 import threeterm_connection_coefficients as cc3
from Wavelet.CC2 import twoterm_connection_coefficients as cc2
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import math
import pickle
import os.path as path

def OverlapMat( OverlapDB ):
    """
    

    Parameters
    ----------
    OverlapDB : dictonary
        database that tells which scaling functions/wavelets overlap with each
        other.

    Returns
    -------
    OverlapMat : np matrix
        matrix with overlapping lists in them.

    """
    N = len(OverlapDB )
    OverlapMat = {}
    for ii in range(0,N):
        for jj in range(0, N):
            OverlapMat[ii,jj] = np.intersect1d(OverlapDB[ii], OverlapDB[jj])
    return OverlapMat


def ULI( Coeffs, l0, i0 ):
    """
    Returns the equivalent index of the DOF at a higher level. This function
    is used to connect DOFs on different levels.

    Parameters
    ----------
    Coeffs : Coefficient Array
        Coefficient Array as is used in pywt
    l0 : level
        level of given DOF
    i0 : index
        Index of given DOF.

    Returns
    -------
    
        Equivalent index of the upper level

    """
    N1  = len(Coeffs[l0])
    N2  = len(Coeffs[l0+1])
    dln = math.log( N2/N1, 2 )

    return int(i0*2**dln)
    
def BasisInfo( Coeffs ):
    """
    Returns general information of the given wavelet basis which is defined
    by Coeffs.

    Parameters
    ----------
    Coeffs : Coefficient Array
        Coefficient Array as is used in pywt

    Returns
    -------
    DofList : list
        List with DOF numbering.
    LevelList : list
        List with level of DOF.
    IndexList : list
        List with index of DOF.
    Dof_C : Coefficient Array
        Coefficient Array with DOF numbering.

    """
    LevelList = []
    IndexList = []
    DofList   = []
    Dof_C = copy.deepcopy(Coeffs)

    if len(Dof_C) == 1:
        Dof_C.append(np.zeros_like(Dof_C[0]))
    else:
        Dof_C.append(np.zeros((1,2*len(Dof_C[-1])))[0] )

    dd = -1
    for ll in range(0, len(Dof_C)):
        Dof_C[ll] = Dof_C[ll].astype(int)
        for ii in range(0,len(Dof_C[ll])):
            dd += 1
            Dof_C[ll][ii] = int(dd)
                     
            if not ll == len(Dof_C)-1:
                DofList.append(dd)
                LevelList.append(ll)
                IndexList.append(ii)
    return DofList, LevelList, IndexList, Dof_C


def WaveValues( ll,ii, Coeffs, Wavelet ):
    """
    Function that calculates the values of wavelet (ll,ii) on the finest grid
    of Coeffs

    Parameters
    ----------
    ll : level
        level of given DOF
    ii : index
        Index of given DOF.
    Coeffs : Coefficient Array
        Coefficient Array as is used in pywt
    Wavelet : Wavelet 
        Wavelet object as is used in pywt.

    Returns
    -------
    FineData : np.array()
        list of wavelet values on finest level of Coeffs.

    """
    
    Coeffs[ll][ii] = 1
    FineData = pywt.waverec( Coeffs, Wavelet, mode = 'periodization') 
    Coeffs[ll][ii] = 0
    
    return FineData
    
def ScaleValues( ll,ii, Coeffs, Wavelet ):
    """
    Function that calculates the values of scaling function (ll,ii) on the 
    finest grid of Coeffs. If ll != 0 the equivalent scaling function is plotted

    Parameters
    ----------
    ll : level
        level of given DOF
    ii : index
        Index of given DOF.
    Coeffs : Coefficient Array
        Coefficient Array as is used in pywt
    Wavelet : Wavelet 
        Wavelet object as is used in pywt.

    Returns
    -------
    FineData : np.array()
        list of scaling function values on finest level of Coeffs.

    """
    Coeffs2 = copy.deepcopy(Coeffs)
    Coeffs2 = Coeffs2[(ll):]
    if ll > 0:
        Coeffs2.insert(0, Coeffs2[0])
    for jj in range(len(Coeffs2)):
        Coeffs2[jj] = np.zeros_like(Coeffs2[jj])

    Coeffs2[0][ii] = 1
    FineData = pywt.waverec( Coeffs2, Wavelet, mode = 'periodization') 
    Coeffs2[0][ii] = 0
    
    return FineData


def OverLap2( Coeffs, Wavelet, L0, Lmax ):
    """
    Calculates the overlapping DOFS. It is assumed that all DOFS only overlap 
    once. 

    Parameters
    ----------
    Coeffs : Coefficient Array
        Coefficient Array as is used in pywt
    Wavelet : Wavelet 
        Wavelet object as is used in pywt.

    Returns
    -------
    Overlap : list
        list of lists with the overlapping DOFS. Overlap[d1] is a list with all
        DOFS that overlap with d1.
    """
    fb2 = Wavelet.filter_bank
    ff = []
    for ii in range(0,len(fb2)):
        ff.append( np.ones_like(fb2[ii]) )
    Wavelet  = pywt.Wavelet( filter_bank = ff)
    
    
    Coeffs2   = GenerateCoeffsArray( L0,Lmax+4 )
    LevelList = []
    IndexList = []
    DofList   = []
    
    FineData = {}
    dd = -1
    for ll in range(0, len(Coeffs)):
        for ii in range(0,len(Coeffs[ll])):
           dd += 1
           
           DofList . append(dd)
           LevelList.append(ll)
           IndexList.append(ii)

           FineData[dd] = np.nonzero( ScaleValues( ll,ii, Coeffs2, Wavelet ) )

    Overlap = []
    
    for d1 in DofList:
        Overlap.append([])
        for d2 in DofList:
            if len( np.intersect1d( FineData[d1], FineData[d2]) ) > 0:
                    Overlap[-1].append(d2)
    
    
    nrl0 = 0
    for ii in Overlap[0]:
        if LevelList[ii] == 0:
            nrl0 += 1
    if nrl0 > len(Coeffs[0])-1:
        print("filter is too wide for basis!")
        exit
    return Overlap


def PlotOverlap( d, xC, OverlapDB, IndexList, LevelList, Coeffs, Wavelet ):
    """
    Function to plot the overlapping functions of dof d

    Parameters
    ----------
    d : int
        dof.
    xC : coeffs array
        coeffs array with corresponding x-positions.
    OverlapDB : dict
        overlapping database.
    IndexList : list
        list with indices.
    LevelList : list
        list with levels.
    Coeffs : coeffs array
        coeffs array of basis.
    Wavelet : pywt.wavelet
        wavelet object.

    Returns
    -------
    None.

    """
    Coeffs = copy.deepcopy(Coeffs)
    f, ax  = plt.subplots()
    f, ax2 = plt.subplots()
    f, ax3 = plt.subplots()
    if len(Coeffs) == 1:
        Coeffs.append(np.zeros_like(Coeffs[0]))
    else:
        Coeffs.append(np.zeros((1,2*len(Coeffs[-1])))[0] )

    for dd in OverlapDB[d]:
        if dd < len(LevelList):
            ii = IndexList[dd]
            ll = LevelList[dd]
            
            y = WaveValues( ll,ii, Coeffs, Wavelet ) + 2*ll

            ax2.plot(y, color = 'k')
        
            y = ScaleValues( ll,ii, Coeffs, Wavelet ) + 2*ll

            ax3.plot(y, color = 'k')
            
            ax.scatter( xC[ll][ii], ll, color = 'k')
    
    ll0 = LevelList[d]
    ii0 = IndexList[d]
    ax2.plot( WaveValues( ll0,ii0, Coeffs, Wavelet )+2*ll0, color = 'r' )
    ax3.plot( ScaleValues( ll0,ii0, Coeffs, Wavelet )+2*ll0, color = 'r' )
    ax.scatter( xC[ll0][ii0], ll0, color = 'r')

    
    
    
def FindIdx( i1,i2,N):
    """
    Returns the correct relative index of  i1 and i2 of a periodic list of 
    length N

    Parameters
    ----------
    i1 : index
    i2 : index
    N : length of list
        

    Returns
    -------
        Relative index.

    """
    absolute = [abs(i2 - i1), abs( ( i2 - N )- i1 ),abs( i2 - ( i1 - N ) )]
    signs    = [np.sign(i2 - i1), np.sign( ( i2 - N )- i1 ),np.sign( i2 - ( i1 - N ) )]
    ii  = np.argmin(absolute)
    return absolute[ii]*signs[ii]

# =============================================================================
# 
# =============================================================================
def CheckOverlap( l1,i1,l2,i2, l3, i3, OverlapDB, Dof_C):
    """
    check overlap of d1,d2,d3

    """
    
    Bool = True
    if not CheckOverlap2( l1,i1,l2,i2, OverlapDB, Dof_C):
            Bool = False
    elif not CheckOverlap2( l2,i2,l3,i3, OverlapDB, Dof_C):
            Bool = False
    elif not CheckOverlap2( l1,i1,l3,i3, OverlapDB, Dof_C):
            Bool = False
    return Bool

# =============================================================================
# 
# =============================================================================
def CheckOverlap2( l1,i1,l2,i2, OverlapDB, Dof_C):
    """
    check overlap of d1,d2

    """
    d1 = Dof_C[l1][i1]
    d2 = Dof_C[l2][i2]
    
    
    Bool = True
    if not d1 in OverlapDB[d2]:
            Bool = False

    return Bool

def MultiLeveledCC3( CClist, list1, list2, list3, Wavelet , OverlapDB, Dof_C,L0, DBsubCClist, rec):
    """
    Recursive calculations of the connection coefficients
    """

    rec += 1
    printbool = 0
    
    h = Wavelet.filter_bank[2]
    g = Wavelet.filter_bank[3]

    
    Lmid = int( len(h)/2 ) - 1
    
    l1, i1, w1 = list1
    l2, i2, w2 = list2
    l3, i3, w3 = list3
    

    
    OmegaList = np.zeros_like( CClist ).astype(float)
    if w1 == 1:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list1))
            
        for qq in range(len(g)):
            
            list1       = [ l1, 2*i1+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(g[qq]) + str(list1)) 
                
            if not g[qq] == 0: 
                OmegaList  += g[qq] * MultiLeveledCC3( CClist, list1, list2, list3, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)
        

        return OmegaList
    
    elif w2 == 1:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list2))
            
        for qq in range(len(g)):
            list2       = [ l2, 2*i2+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(g[qq]) + str(list2)) 
            
            if not g[qq] == 0: 
                OmegaList  += g[qq] * MultiLeveledCC3( CClist, list1, list2, list3, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)


        return OmegaList
    elif w3 == 1:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list3))
            
        for qq in range(len(g)):
            
            list3       = [ l3, 2*i3+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(g[qq]) + str(list3)) 
            if not g[qq] == 0: 
                OmegaList  += g[qq] * MultiLeveledCC3( CClist, list1, list2, list3, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)


        return OmegaList
    
    
    

    key    = ( l1, i1, w1,  l2, i2, w2,  l3, i3, w3 )
    Dof_C2 = copy.deepcopy(Dof_C)
    Dof_C2.append( np.zeros( ( 2*len(Dof_C2[-1]),1))[0] )
    Dof_C2 = Dof_C2[1:]
    
    lr2, ir2, way2, lr3, ir3, way3, w23 = Abs2Key2( list1, list2, list3, Dof_C2)
    key = (lr2, ir2, way2, lr3, ir3, way3)
    if key in DBsubCClist[0].keys():
        OmegaList = []
        for ii in DBsubCClist:
            OmegaList.append( ii[key] )
        return np.array( OmegaList )
    
    Lmax = max([l1,l2,l3])
 
    if l1 < Lmax:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list1))
            
        dL = 1
        for qq in range(len(h)):
            list1       = [ l1+dL, 2*i1+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(h[qq]) + str(list1)) 
            
            if not h[qq] == 0: 
                OmegaList  +=  h[qq] * MultiLeveledCC3( CClist, list1, list2, list3, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)

        for ii in range( len( DBsubCClist ) ):
            DBsubCClist[ii][key] = OmegaList[ii]
        return OmegaList
    
    elif l2 < Lmax:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list2))
        
        dL = 1
        for qq in range(len(h)):
            list2       = [ l2+dL, 2*i2+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(h[qq]) + str(list2)) 
            
            if not h[qq] == 0: 
                OmegaList  +=  h[qq] * MultiLeveledCC3( CClist, list1, list2, list3, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)

        for ii in range( len( DBsubCClist ) ):
            DBsubCClist[ii][key] = OmegaList[ii]
        return OmegaList
    
    elif l3 < Lmax:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list3))
            
        dL = 1
        for qq in range(len(h)):
            list3       = [ l3+dL, 2*i3+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(h[qq]) + str(list3)) 
            
            if not h[qq] == 0: 
                OmegaList  +=  h[qq] * MultiLeveledCC3( CClist, list1, list2, list3, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)

        for ii in range( len( DBsubCClist ) ):
            DBsubCClist[ii][key] = OmegaList[ii]
        return OmegaList


    Nmax =  2**(L0+Lmax) 
    idx1 = FindIdx( i1,i2,Nmax)
    idx2 = FindIdx( i1,i3, Nmax)


    for ii in range( 0, len(CClist) ):
        OmegaList[ii] = CClist[ii]( idx1, idx2)

    return OmegaList
    


def Abs2Rel(list2, l1, i1, Coeffs):
    """
    Calculates the relative level and index of DOF2 with respect to DOF1

    Parameters
    ----------
    list2 : list
        list of DOF2 with [level, index, waveletbool].
    l1 : index
        index of DOF1.
    i1 : level
        level of DOF1.
    Coeffs : Coefficient Array
        Coefficient Array as is used in pywt

    Returns
    -------
    list2n : list
        [relative level, relative index, waveletbool].

    """
    [l2, i2, w2] = list2

    # it is assumed that l2 > l1
    l2r = l2 - l1
    while not l2 == l1:
            i1 = ULI( Coeffs, l1, i1 )
            l1 += 1

    i2r  = FindIdx( i1, i2, len(Coeffs[l1]) )


    list2n = [l2r, i2r, w2]
    return  list2n
    
# =============================================================================
# 
# =============================================================================
def Abs2Key( list1, list2, Coeffs):
    """
    Returns the relative information of DOF1 and DOF2 that can be used to define
    a key for the CC database.

    Parameters
    ----------
    list1 : list
        list of DOF1 with [level, index, waveletbool]..
    list2 : list
        list of DOF2 with [level, index, waveletbool]..
    Coeffs : Coefficient Array
        Coefficient Array as is used in pywt

    Returns
    -------
    lr : int
        relative level.
    ir : int
        relative index.
    wr : bool
        bool that indicates whether the combination involves a scaling function
        if wr = 0, then DOF1 or DOF2 is a scaling function.
    way : int
        interger that indicates if the relative level/index is calculated of 
        DOF2 wrt DOF1 or the other way arround. if way = 1 we have DOF2 wrt DOF1
        and if way = -1 we have DOF1 wrt DOF2.

    """
    [l1, i1, w1] = list1
    [l2, i2, w2] = list2
    
    if l1 > l2:
        lr, ir,w  = Abs2Rel( list1, l2, i2, Coeffs )
        way    = -1
    else:
        lr, ir,w = Abs2Rel( list2, l1, i1, Coeffs )
        way    = 1
    wr = w1*w2

    return lr, ir, wr, way
    

def Abs2Key2( list1, list2, list3, Coeffs):
    
    lr2, ir2, wr2, way2 = Abs2Key( list1, list2, Coeffs)
    lr3, ir3, wr3, way3 = Abs2Key( list1, list3, Coeffs)
    key = (lr2, ir2, way2, lr3, ir3, way3, wr2*wr3)   
    return key
    
def CC3_Database( Wavelet, cc3list, BasisList, CC_DatabaseList = []):
    
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs] = BasisList

    DBsubCC = {}
 
    it = -1
    
    DBsubCCList     = []
    for ii in cc3list:
        DBsubCCList.append({})

    if not CC_DatabaseList:
        for ii in cc3list:
            CC_DatabaseList.append({})

    startTime = datetime.now()
    i1 = 0
    N1 = len(Dof_C[0]) # Number of scaling functions
    
    for l1 in range(len(Dof_C[:-1])):
        print( int( len(Dof_C[l1])/N1 ) )
        for i1 in range(0, int( len(Dof_C[l1])/N1 )):
            for jj,kk in itertools.product(range(len(DofList)), repeat=2):
                it += 1     
    
                l2 = LevelList[jj]
                i2 = IndexList[jj]
                l3 = LevelList[kk]
                i3 = IndexList[kk]
                
                w1 = 0 if l1 == 0 else 1 
                w2 = 0 if l2 == 0 else 1
                w3 = 0 if l3 == 0 else 1
                
                list1 = [l1, i1 , w1]
                list2 = [l2, i2 , w2]
                list3 = [l3, i3 , w3]
    
                key = Abs2Key2( list1, list2, list3, Coeffs)
                
                if CheckOverlap( l1,i1,l2,i2,l3,i3, OverlapDB, Dof_C):
                    
                    if not (key) in CC_DatabaseList[0].keys():
                        CC = MultiLeveledCC3( cc3list, list1, list2, list3, Wavelet, OverlapDB, Dof_C, L0,DBsubCCList ,0 )
                    
                        print(key, list1,list2,list3)
                        for ii in range(0, len(CC)):
                                CC_DatabaseList[ii][key] = CC[ii]
       
    print(datetime.now() - startTime)
    startTime = datetime.now()
    

    return CC_DatabaseList



def MultiLeveledCC2( CClist, list1, list2, Wavelet , OverlapDB, Dof_C,L0, DBsubCClist, rec):

    
    printbool = 0
    rec += 1
    h = np.array( Wavelet.filter_bank[2] ) 
    g = np.array( Wavelet.filter_bank[3] ) 
    
    
    Lmid = int( len(h)/2 ) - 1
    
    l1, i1, w1 = list1
    l2, i2, w2 = list2
    
    key  = ( l1, i1, w1,  l2, i2, w2 )
    if key in DBsubCClist[0].keys():
        OmegaList = []
        for ii in DBsubCClist:
            OmegaList.append( ii[key] )
        return np.array( OmegaList )
     


    OmegaList = np.zeros_like( CClist ).astype(float)
    if w1 == 1:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list1))
            
        for qq in range(len(g)):
            list1       = [ l1, 2*i1+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(g[qq]) + str(list1)) 
            
            OmegaList  += g[qq] * MultiLeveledCC2( CClist, list1, list2, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)
        
        for ii in range( len( DBsubCClist ) ):
            DBsubCClist[ii][key] = OmegaList[ii]
        return OmegaList
    elif w2 == 1:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list2))
            
        for qq in range(len(g)):
            list2       = [ l2, 2*i2+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(g[qq]) + str(list2)) 
            
            OmegaList  += g[qq] * MultiLeveledCC2( CClist, list1, list2, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)

        for ii in range( len( DBsubCClist ) ):
            DBsubCClist[ii][key] = OmegaList[ii]
        return OmegaList
   
    
    Lmax = max([l1,l2])
 
    
    if l1 < Lmax:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list1))
            
        """ if main scaling function we need to jump 2 levels, else 1"""
        dL = 1
        for qq in range(len(h)):

            list1       = [ l1+dL, 2*i1+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(h[qq]) + str(list1)) 
            
            OmegaList  +=  h[qq] * MultiLeveledCC2( CClist, list1, list2, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)
        
        for ii in range( len( DBsubCClist ) ):
            DBsubCClist[ii][key] = OmegaList[ii]
        return OmegaList
    
    elif l2 < Lmax:
        
        if printbool == 1:
            print( (rec-1)*'\t' + str(list2))
            
        dL = 1
        for qq in range(len(h)):
            list2 = [ l2+dL, 2*i2+qq-Lmid, 0 ]
            
            if printbool:
                print(rec*'\t' + "{:.2f}".format(h[qq]) + str(list2)) 
            
            OmegaList  +=  h[qq] * MultiLeveledCC2( CClist, list1, list2, Wavelet, OverlapDB, Dof_C ,L0, DBsubCClist, rec)

        for ii in range( len( DBsubCClist ) ):
            DBsubCClist[ii][key] = OmegaList[ii]
        return OmegaList


    # Number of scaling functions on max level
    Nmax =  2**(L0+Lmax) 
    idx1 = FindIdx( i1,i2, Nmax)

    for ii in range( 0, len(CClist) ):
        OmegaList[ii] = CClist[ii]( idx1 )
    
    if printbool:
        print(rec*'\t' + str(OmegaList) )
        print( rec*'\t' + str( [i1,i2, Nmax,idx1]))
    return OmegaList

def CC2_Database( Wavelet, cc2list, BasisList, CC_DatabaseList = []):
    
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs] = BasisList

 
    it = -1
    
    DBsubCCList     = []
    for ii in cc2list:
        DBsubCCList.append({})

    if not  CC_DatabaseList:
        for ii in cc2list:
            CC_DatabaseList.append({})
            
            
    startTime = datetime.now()
    
    for l1 in range(len(Dof_C[:-1])):
        
        for i1 in range( len(Dof_C[l1])):
            for jj in DofList:
                # print(l1,jj)
                it += 1     
    
                l2 = LevelList[jj]
                i2 = IndexList[jj]
                
                w1 = 0 if l1 == 0 else 1 
                w2 = 0 if l2 == 0 else 1
                
                list1 = [l1, i1, w1]
                list2 = [l2, i2, w2]
                
                key = Abs2Key(list1, list2, Dof_C)
     
                lr, ir, wr, way = key
                
                for way in [-1,1]:
                    key = ( lr, ir, wr, way)
                
                    if way == 1:
                        lista = list1
                        listb = list2
                    elif way == -1:
                        lista = list2
                        listb = list1
         
                    if CheckOverlap2( l1,i1,l2,i2, OverlapDB, Dof_C):
                        
                        if not (key ) in CC_DatabaseList[0].keys():
                            
                            CC = MultiLeveledCC2( cc2list, lista, listb, Wavelet, OverlapDB, Dof_C, L0, DBsubCCList,0 )

                            print(key, list1, list2)
                            for ii in range(0, len(CC)):
                                    CC_DatabaseList[ii][key] = CC[ii]
       
    print(datetime.now() - startTime)
    startTime = datetime.now()
    

    return CC_DatabaseList


# =============================================================================
# 
# =============================================================================


def cc3dict( h, d1,d2,d3 ):
    
    ccdict = {}
    idx, Tind, G = cc3(h,d1,d2,d3)
    ccdict = {}
    for ii in range(0,len(Tind)):
        if abs ( G[ii] ) > 10**-10:
            ccdict[Tind[ii]] = G[ii]
    
    func = lambda l,m: ccdict[(l,m)] if (l,m) in ccdict.keys() else 0
    return func

def Dict2Lam( Dict ):
    func = lambda key: Dict[(key)] if key in Dict.keys() else 0
    return func


def cc2dict( F ):

    func = lambda l: F[l+int(len(F)/2)] if (  -1<l+int(len(F)/2) and l+int(len(F)/2) < len(F) ) else 0
    return func


def CalculateDeslaurierWavelet( n ):
    
    DBname = 'db' + str(n)
#    DBname = 'coif' + str(n)

    hDB = pywt.Wavelet(DBname).filter_bank[0]    
    hDD = np.correlate( hDB, hDB, mode="full")

    for ii in range(0, len(hDD) ):
        if abs(hDD[ii]) < 10**(-14):
            hDD[ii] = 0.0
    # hDD = np.append( hDD, 0)
    # hDD = np.append( 0 , hDD)
    L   = len(hDD)
    N   = round( L/2 ) - 1
    
    lo_d = hDD
    
    hi_r = np.multiply( hDD, -1 )
    hi_r[N] = -1*hi_r[N]
    hi_d = np.zeros_like( lo_d )
    
    
    hi_d[N] = 1
    lo_r = hi_d
   
    
    
    hi_d    = np.append( 0, hi_d )
    lo_d    = np.append( lo_d, 0 ) 
    lo_r    = np.append( 0 , lo_r ) 
    hi_r    = np.append( hi_r, 0 )
    
    filter_bank = [ lo_r, hi_r, lo_d, hi_d]

    Wavelet         = pywt.Wavelet(name='dd' + str(n),  filter_bank = filter_bank )
    Wavelet.biorthogonal = True
    
    return Wavelet


# =============================================================================
#  Generate coefficient array
# =============================================================================
def GenerateCoeffsArray( L0, Lmax ):
    dummydata   = np.zeros( 2**Lmax )
    Coeffs      = pywt.wavedec(dummydata, 'db1', level = Lmax - L0)
    return Coeffs




# =============================================================================
# 
# =============================================================================

def MakeDerivativeMatrix( Filter, N, dx):

    D = np.zeros((N,N))
    
    

    Nmid = int(len(Filter)/2)

    xi = []
    for n in range(0,N):
        xi.append( np.arange(n - Nmid , n + Nmid + 1  ) % N )
    
    for ii in range(0, N):
            jrow = xi[ii]
            qq = -1
            for jj in jrow:
                qq += 1
                
                
                D[ii,jj] +=Filter[qq ]/dx
    return D



def UpLevelTest( list1, L0, Lamb, Wavelet, rec ):
    rec += 1
    [l1,i1,w1] = list1
    # s1 = 1 if scaling function else 0
    # if l1 == L0 and s1 ==1, we have a "main scaling funciton"
    Coeffs = GenerateCoeffsArray( L0, 20 )
    
    h = np.array( Wavelet.filter_bank[2] ) 
    g = np.array( Wavelet.filter_bank[3] ) 
    
    
    Lmid = int( len(h)/2 ) - 1
    
    
    if w1 == 0:
        y1 = ScaleValues(  l1, i1, Coeffs, Wavelet)
    else:
        y1 = WaveValues(  l1, i1, Coeffs, Wavelet)
        
    """ L0 =  L0+1,
    Wavelet and scaling function at level 0 are level 0 and 1 in code... compensate for that"""

    iu = ULI( Coeffs, l1, i1 )
    y2 = np.zeros_like(y1)
    if w1 == 1:
        for qq in range(len(g)):
            list1       = [ l1+1, 2*i1+qq-Lmid, 0 ]
            print(rec*'\t' + "{:.2f}".format(g[qq]) + str(list1)) 
            y2  += g[qq] * UpLevelTest( list1, L0, Lamb, Wavelet, rec  )
        
    elif l1 < Lamb:
        if l1 == 0:
            """ Main Scaling function, so add extra level """
            for qq in range(len(h)):
                list1       = [ l1+2, 2*i1+qq-Lmid, 0 ]
                # print("main S")
                print(rec*'\t' + "{:.2f}".format(h[qq]) + str(list1)) 
                y2  += h[qq] * UpLevelTest( list1, L0, Lamb, Wavelet, rec  )
        else:
            """ Regular Scaling function, so do not add extra level """
            for qq in range(len(h)):
                list1       = [ l1+1, 2*i1+qq-Lmid, 0 ]
                print(rec*'\t' + "{:.2f}".format(h[qq]) + str(list1)) 
                y2  += h[qq] * UpLevelTest( list1, L0, Lamb, Wavelet, rec  )
    else:
        y2 = y1
    
    if rec == 1:
        f, ax = plt.subplots()
        ax.plot(y1)
        ax.plot(y2)
    return y2

        
    
    

def CC2_Numerical( list1, list2, L0, Wavelet, d1, d2 ):
    
    
    Coeffs = GenerateCoeffsArray( L0, 17 )
    [l1,i1,w1] = list1
    [l2,i2,w2] = list2
    
    dx = 1/2**(L0 + max(l1,l2))
    # f, ax = plt.subplots()
    if w1 == 0:
        y1 = ScaleValues(  l1, i1, Coeffs, Wavelet)
    else:
        y1 = WaveValues(  l1, i1, Coeffs, Wavelet)
        
    if w2 == 0:
        y2 = ScaleValues(  l2, i2, Coeffs, Wavelet)
    else:
        y2 = WaveValues(  l2, i2, Coeffs, Wavelet)
        
    # ax.plot( y1 )
    # ax.plot( y2 )
    if d1 == 1:
        dy1 = [ (y1[ii+1] - y1[ii-1] )*len(y1)/2 for ii in range(len(y1)-1) ]
        dy1.append( (y1[1] - y1[-1]) *len(y1)/2)
        y1 = dy1

    if d2 == 1:
        dy2 = [ (y2[ii+1] - y2[ii-1] )*len(y1)/2 for ii in range(len(y2)-1) ]
        # print( y2 )
        dy2.append( (y2[1] - y2[-1]) *len(y1) /2)
        y2 = dy2
    if d2 == 2:
        dy2 = [ (y2[ii+1]+2*y2[ii] - y2[ii-1] )*len(y1)**2 for ii in range(len(y2)-1) ]
        # print( y2 )
        dy2.append( (-y2[1]+2*y2[0] - y2[-1]) *len(y1)**2)
        y2 = dy2
        
    CC = sum( np.multiply( y1, y2 )) / len(y1)* (dx)**(d1+d2 -1)
    
    
    # ax.plot( y1 )
    # ax.plot( y2 )
    return CC
        
def CC3_Numerical( list1, list2, list3, L0, Wavelet, d1, d2, d3 ):

    
    Coeffs = GenerateCoeffsArray( L0, 15 )
    [l1,i1,w1] = list1
    [l2,i2,w2] = list2
    [l3,i3,w3] = list3
    dx = 1/2**(L0 )
    if w1 == 0:
        y1 = np.array( ScaleValues(  l1, i1, Coeffs, Wavelet) )
    else:
        y1 = np.array( WaveValues(  l1, i1, Coeffs, Wavelet) )
        
    if w2 == 0:
        y2 = np.array( ScaleValues(  l2, i2, Coeffs, Wavelet) )
    else:
        y2 = np.array( WaveValues(  l2, i2, Coeffs, Wavelet) )
    if w3 == 0:
        y3 = np.array( ScaleValues(  l3, i3, Coeffs, Wavelet) )
    else:
        y3 = np.array( WaveValues(  l3, i3, Coeffs, Wavelet) )
        

    
    if d1 == 1:
        dy1 = [ (y1[ii+1] - y1[ii-1] )*len(y1)/2 for ii in range(len(y1)-1) ]
        dy1.append( (y1[1] - y1[-1])*len(y1) /2)
        y1 = np.array( dy1 )

    if d2 == 1:
        dy2 = [ (y2[ii+1] - y2[ii-1] )*len(y1)/2 for ii in range(len(y2)-1) ]
        # print( y2 )
        dy2.append( (y2[1] - y2[-1]) *len(y1)/2)
        y2 = np.array( dy2 )
    if d3 == 1:
        dy3 = [ (y3[ii+1] - y3[ii-1] )*len(y1)/2 for ii in range(len(y3)-1) ]
        # print( y2 )
        dy3.append( (y3[1] - y3[-1]) *len(y1)/2)
        y3 = np.array( dy3 )

    # f, ax = plt.subplots()        
    # ax.plot( y1 )
    # ax.plot( y2 )
    # ax.plot( y3 )
    CC = sum( np.multiply( y1, np.multiply( y2, y3) )) / len(y1) 
    return CC

def CheckCC3Num( CC3DB, d1,d2, d3, Wavelet, L0, threshold, OverlapDB, Dof_C, CClist ):
    
    for key in CC3DB.keys():
        lr2, ir2, way2, lr3, ir3, way3,  wr = key
        
        if wr == 0:
            l1 = 0
            w1 = 0
        else:
            l1 = 1
            w1 = 1
        i1 = 0
        
        if not way2 == -1:
            if not way3 == -1:
                l2 = l1 + lr2
                l3 = l1 + lr3
                i2 = i1 + ir2
                i3 = i1 + ir3
                
                if l2 ==0:
                    w2 = 0
                else:
                    w2 = 1
                if l3 ==0:
                    w3 = 0
                else:
                    w3 = 1
                
                list1 = [l1,i1,w1]
                list2 = [l2,i2,w2]
                list3 = [l3,i3,w3]
                
                Lm = (L0 + max(l1,l2,l3))*(d1+d2+d3-1)
                
                CCana = 2**Lm*CC3DB[key]
                CCnum = CC3_Numerical( list1, list2, list3, L0, Wavelet, d1, d2, d3 )
                
                if not abs( ( CCana - CCnum )/( CCnum + 10**(-3) )) < threshold: 
                    print(list1, list2, list3)
    
                    CCana2 = 2**Lm*MultiLeveledCC3( [CClist], list1, list2, list3, Wavelet , OverlapDB, Dof_C,L0, [{}], 0)[0]
                    print( key, "{:.4f}".format(CCana),  "{:.4f}".format(CCnum),  "{:.4f}".format(CCana2) )
 
        
def CheckCC3Num2( CC3DB, d1,d2, d3, Wavelet, threshold, Dof_C, CClist, BasisList ):
    
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs] = BasisList
    N = len(DofList)
    it = 0
    for ii in DofList:
        it += 1
        print(it/N)
        jjlist = np.intersect1d( DofList, OverlapDB[ii] )
        for jj in  jjlist :
            kklist = np.intersect1d( jjlist, OverlapDB[jj])
            for kk in kklist:
                
                l1 = LevelList[ii]
                l2 = LevelList[jj]
                l3 = LevelList[kk]
                
                i1 = IndexList[ii]
                i2 = IndexList[jj]
                i3 = IndexList[kk]
                
                w1 = 0 if l1 == 0 else 1
                w2 = 0 if l2 == 0 else 1
                w3 = 0 if l3 == 0 else 1
                
                list1 = [l1,i1,w1]
                list2 = [l2,i2,w2]
                list3 = [l3,i3,w3]
                # print(list1)
                # print(list2)
                # print(list3)
                key = Abs2Key2( list1, list2, list3, Coeffs )
            
            
    
    
                lr2, ir2, way2, lr3, ir3, way3,  wr = key
                
               
                        
                Lm = (L0 + max(l1,l2,l3))*(d1+d2+d3-1)
                
                CCana = 2**Lm*CC3DB(key)
                CCnum = CC3_Numerical( list1, list2, list3, L0, Wavelet, d1, d2, d3 )
                
                if not abs( ( CCana - CCnum )/( CCnum + 10**(-3) )) < threshold: 
                    print(list1, list2, list3)
    
                    CCana2 = 2**Lm*MultiLeveledCC3( [CClist], list1, list2, list3, Wavelet , OverlapDB, Dof_C,L0, [{}], 0)[0]
                    print( key, "{:.4f}".format(CCana),  "{:.4f}".format(CCnum),  "{:.4f}".format(CCana2) )
            
def CheckCC2Num2( CC2DB, d1,d2, Wavelet, threshold, Dof_C, CClist, BasisList ):
    
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs] = BasisList
    N = len(DofList)
    it = 0
    for ii in DofList:
        it += 1
        print(it/N)
        jjlist = np.intersect1d( DofList, OverlapDB[ii] )
        for jj in  jjlist :
          
                l1 = LevelList[ii]
                l2 = LevelList[jj]
                
                i1 = IndexList[ii]
                i2 = IndexList[jj]
                
                w1 = 0 if l1 == 0 else 1
                w2 = 0 if l2 == 0 else 1
                
                list1 = [l1,i1,w1]
                list2 = [l2,i2,w2]
                # print(list1)
                # print(list2)
                # print(list3)
                key = Abs2Key( list1, list2, Coeffs )
            
            
    
    
                lr2, ir2, way2,  wr = key
                
               
                        
                Lm = (L0 + max(l1,l2))*(d1+d2-1)
                
                CCana = 2**Lm*CC2DB(key)
                CCnum = CC2_Numerical( list1, list2, L0, Wavelet, d1, d2 )
                
                if not abs( ( CCana - CCnum )/( CCnum + 10**(-3) )) < threshold: 
                    print(list1, list2)
    
                    CCana2 = 2**Lm*MultiLeveledCC2( [CClist], list1, list2, Wavelet , OverlapDB, Dof_C,L0, [{}], 0)[0]
                    print( key, "{:.4f}".format(CCana),  "{:.4f}".format(CCnum),  "{:.4f}".format(CCana2) )                   
def CheckCC2Num( CC2DB, d1,d2, Wavelet, L0, threshold ):
    
    for key in CC2DB.keys():
        lr2, ir2, wr, way2 = key
        
        if wr == 0:
            l1 = 0
            w1 = 0
        else:
            l1 = 1
            w1 = 1
        i1 = 0
        
        if not way2 == -1:
                l2 = l1 + lr2
                i2 = i1 + ir2
                
                if l2 ==0:
                    w2 = 0
                else:
                    w2 = 1
                
                list1 = [l1,i1,w1]
                list2 = [l2,i2,w2]
                
                CCana = CC2DB[key]
                CCnum = CC2_Numerical( list1, list2,  L0, Wavelet, d1, d2, )
                
                if not abs( ( CCana - CCnum )/( CCnum + 10**(-3) )) < threshold: 
                    print( key, "{:.4f}".format(CCana),  "{:.4f}".format(CCnum) )
        
# =============================================================================
#         
# =============================================================================

def CCDmat( CC2DB, d1 ,d2, Wavelet, BasisList, i_list):
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs] = BasisList

    K = np.zeros( (len(i_list), len(i_list) )).astype(float)
    Kii = -1
    for ii in i_list:
        Kii    += 1
        i1      = IndexList[ii]
        l1      = LevelList[ii]
        w1      = 0 if l1 == 0 else 1
        list1   = [ l1, i1, w1 ]
    
        j_list  = np.intersect1d( i_list, OverlapDB[ii] ) 
    
        Kjj = -1
        for jj in j_list:
            Kjj     = int( np.argwhere( i_list == jj ) )
            i2      = IndexList[jj]
            l2      = LevelList[jj]
            w2      = 0 if l2 == 0 else 1
            # print(l2)
            maxl =  L0 
    
            list2           = [ l2, i2, w2 ]
            key             = Abs2Key(list1, list2, Dof_C)
            CC              = CC2DB(key)
            
            K[Kii,Kjj] +=  2**maxl*CC
            
    return K
# =============================================================================
#         
# =============================================================================
def CheckCC2Derivative( CC2DB, d1 ,d2, Wavelet,  A, BasisList, i_list, CC01):
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs] = BasisList
    CCDB2lam = Dict2Lam(CC2DB)
    CC2DBnum = {}
    K = np.zeros( (len(i_list), len(i_list) )).astype(float)
    K2 = np.zeros( (len(i_list), len(i_list) )).astype(float)
    Kii = -1
    for ii in i_list:
        Kii    += 1
        i1      = IndexList[ii]
        l1      = LevelList[ii]
        w1      = 0 if l1 == 0 else 1
        list1   = [ l1, i1, w1 ]
    
        j_list  = np.intersect1d( i_list, OverlapDB[ii] ) 
    
        Kjj = -1
        for jj in j_list:
            Kjj     = int( np.argwhere( i_list == jj ) )
            i2      = IndexList[jj]
            l2      = LevelList[jj]
            w2      = 0 if l2 == 0 else 1
            # print(l2)
            maxl =  L0 +0* max(l1,l2)
    
            list2           = [ l2, i2, w2 ]
            key             = Abs2Key(list1, list2, Dof_C)
            CC              = CCDB2lam(key)
            
            if not key in CC2DBnum.keys():
                if CheckOverlap2( l1,i1,l2,i2, OverlapDB, Dof_C):
                    CC2DBnum[key] = CC2_Numerical( list1, list2, L0, Wavelet, 0,1 )
                else:
                    CC2DBnum[key] = 0
            K2[Kii,Kjj] += 2**(maxl*(d1+d2-1))*CC2DBnum[key] 
            K[Kii,Kjj] +=  2**(maxl*(d1+d2-1))*CC
            
            a = K2[Kii,Kjj] 
            b = K[Kii,Kjj]
            
            if abs(( a - b )/( a + 10**-4)) > 0.001:
                Kcor = 2**maxl*MultiLeveledCC2([CC01], list1, list2, Wavelet, OverlapDB, Dof_C, L0, [{}], 0)[0]
                K[Kii,Kjj] = Kcor
                print( Kii,Kjj, list1, list2, key, "{:.4f}".format(a), "{:.4f}".format(b), Kcor)

         
    
    f, ax = plt.subplots()
    ax.matshow(K)
    f,ax = plt.subplots()
    ax.matshow(K2)
    f,ax = plt.subplots()
    ax.matshow(K2-K)
    f, ax2 = plt.subplots()
    dA_R = [(A[(ii+1)%len(A)]-A[ii-1])*len(A)/2 for ii in range(0,len(A))]
    ax2.plot( np.linspace(0,1,len(A)+1)[:-1], A)
    ax2.plot(np.linspace(0,1,len(dA_R)+1)[:-1], dA_R)
    ddA_R =[(dA_R[(ii+1)%len(dA_R)]-dA_R[ii-1])*len(dA_R)/2 for ii in range(0,len(dA_R))]


    A_C = pywt.wavedec(  A , Wavelet, mode = 'periodization', level = Lmax - L0 )    
    ACrow, slices = pywt.coeffs_to_array( A_C )
    dACrow = np.matmul(K , ACrow )
    dA_C = pywt.array_to_coeffs( dACrow, slices, output_format = 'wavedec' )
    dA_S = pywt.waverec( dA_C, Wavelet, mode = 'periodization' )
    ax2.plot( np.linspace(0,1,len(dA_S)+1)[:-1], dA_S)
    

def PlotDyadic( i_list, LevelList, IndexList, xC, ax ):

    xx = []
    yy = []
    for dd in i_list:
        ll = LevelList[dd]
        ii = IndexList[dd]
        
        xx.append(xC[ll][ii])
        yy.append(ll)
        
    ax.scatter(xx,yy)
    
def MakeCCDB(Wavelet, WaveName, CCdir, BasisList ):
    
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs] = BasisList
    h = np.array(Wavelet.filter_bank[2])
    CC010            = cc3dict(h, 0, 1, 0)
    CC001            = cc3dict(h, 0, 0, 1)
    CC100            = cc3dict(h, 1, 0, 0)
    CC011            = cc3dict(h, 0, 1, 1)
    CC000            = cc3dict(h, 0, 0, 0)
    CC111            = cc3dict(h, 1, 1, 1) 
    
    # Two-term CCs
    CC11             = cc2dict( -cc2(Wavelet,2 ) )
    CC01             = cc2dict(  cc2(Wavelet,1 ) )
    CC00             = cc2dict(  cc2(Wavelet,0 ) ) 
    
    CCarrays = [CC00, CC01, CC11, CC010, CC001, CC100, CC011, CC000, CC111]
    print("Calculated Primal Connection Coefficients")
    
    
    CCname                            = CCdir + '/'  +  WaveName + "_" + str(L0) + ".pickle"
    
    if path.exists(CCname):
        print("load existing database")
        with open(CCname, 'rb') as handle:
            [Lmax_old, CCDB2list, CCDB3list, OverlapDB ] = pickle.load(handle)
            BasisList = [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs]
    
        if Lmax_old < Lmax:
            
            OverlapDB   = OverLap2( Dof_C, Wavelet, L0, Lmax )
            BasisList = [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs]
            print("Extending old database")
            CCDB3list   = CC3_Database( Wavelet, [CC011, CC010, CC000], BasisList, CC_DatabaseList = CCDB3list )
            CCDB2list   = CC2_Database( Wavelet, [CC11, CC01, CC00], BasisList, CC_DatabaseList = CCDB2list )
            

    
            with open(CCname, 'wb') as handle:
                pickle.dump([Lmax, CCDB2list, CCDB3list, OverlapDB ], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    else:

        OverlapDB   = OverLap2( Dof_C, Wavelet, L0, Lmax )
        BasisList = [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs]
        CCDB3list   = CC3_Database( Wavelet, [CC011, CC010, CC000], BasisList )
        CCDB2list   = CC2_Database( Wavelet, [CC11, CC01, CC00], BasisList )
        
        with open(CCname, 'wb') as handle:
            pickle.dump([Lmax, CCDB2list, CCDB3list, OverlapDB ], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    print("Made CC database")
    return CCDB2list, CCDB3list, BasisList, CCarrays
    
            