
import copy
import pywt
import sys

import matplotlib.pyplot    as plt
import numpy                as np

from shutil                 import copyfile

from Wavelet.CC2 import twoterm_connection_coefficients as cc2

import Wavelet.LibNumerical as LibN
import Wavelet.LibWavelets  as LibW
import Auxiliary.LibAux     as LibA
import Auxiliary.SaveData   as SaveData
import Auxiliary.LibGeom    as LibG
import Auxiliary.LibMat     as LibM
import LibAdapt             as LibAd 

# =============================================================================
# =============================================================================
# # Initialization
# =============================================================================
# =============================================================================

import inputfile as i


# =============================================================================
# Make Coefficient Arrays
# =============================================================================

    
Coeffs      = LibW.GenerateCoeffsArray( i.L0, i.Lmax )
i_C         = copy.deepcopy( Coeffs )
i_list      = []
d = -1
for ll in range(0,i.LAdapt):
    for ii in range(0, len(i_C[ll])): 
        d           += 1
        i_C[ll][ii]  = 1
        i_list.append( d )


DofList, LevelList, IndexList, Dof_C = LibW.BasisInfo( Coeffs )

xC, xArray  = LibA.MakeXCoeffs( Coeffs, i.L )
x_S         = np.sort(xArray)

print("Made Coefficient Arrays")

BasisList   = [DofList, LevelList, IndexList, Dof_C, [], i.L0, i.Lmax, Coeffs]

# =============================================================================
#  Import Connection Coefficients
# =============================================================================

# derivative filter in the case for orthogonal wavelets
d_filter         = cc2(i.Wavelet,1 ) 

CCDB2list, CCDB3list, BasisList, CCarrays                       = LibW.MakeCCDB(i.Wavelet, i.WaveName, i.CCdir, BasisList )
[CC00, CC01, CC11, CC010, CC001, CC100, CC011, CC000, CC111]    = CCarrays
[CCDB3_011,CCDB3_010,CCDB3_000]                                 = CCDB3list
[CCDB2_11,CCDB2_01,CCDB2_00]                                    = CCDB2list
[DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs] = BasisList

D1max            = LibW.MakeDerivativeMatrix(d_filter, 2**(i.Lmax), 1/2**(i.Lmax)) 

CC_list          = [LibW.Dict2Lam(CCDB3_011), 
                    LibW.Dict2Lam(CCDB3_010), 
                    LibW.Dict2Lam(CCDB3_000), 
                    LibW.Dict2Lam(CCDB2_01), 
                    LibW.Dict2Lam(CCDB2_00), 
                    LibW.Dict2Lam(CCDB2_11), 
                    D1max]


# =============================================================================
# 
# =============================================================================
A_order       = 1
Area = LibG.Area( 'FixedPoly',i.L,1,A_order,x_S)
A    = Area.Arow
# Nmid  = int( 2**(i.Lmax - 1) )
# A.pop( Nmid )
# A.append( A[0])
    


Material = LibM.Material( i.MatName, i.MatParameters ) 

if i.MatName == 'PhaseField':
    Material.d        = np.zeros_like( x_S )
    Material.d_C      = copy.deepcopy(Coeffs)
    Material.H        = np.zeros_like( x_S )
    Material.Hold     = np.zeros_like( x_S )

# threshold = 0.2
# LibW.CheckCC3Num2( LibW.Dict2Lam( CCDB3_000 ) ,0,0,0, i.Wavelet, threshold, Dof_C, CC000, BasisList )
# print('check CC000')
# # LibW.CheckCC3Num2( LibW.Dict2Lam( CCDB3_010 ) ,0,1,0, i.Wavelet, threshold, Dof_C, CC010, BasisList )
# print('check CC010')
# # LibW.CheckCC3Num2( LibW.Dict2Lam( CCDB3_011 ) ,0,1,1, i.Wavelet, threshold, Dof_C, CC011, BasisList )
# print('check CC011')
# LibW.CheckCC3Num( CCDB3_000,0,0,0, i.Wavelet, i.L0,0.2, OverlapDB, Dof_C, CC000)
# print('CC000 Checked!')
# print('\n')
# LibW.CheckCC3Num( CCDB3_010,0,1,0, i.Wavelet, i.L0,0.2, OverlapDB, Dof_C, CC010)
# print('CC010 Checked!')
# print('\n')
# LibW.CheckCC3Num( CCDB3_011,0,1,1, i.Wavelet, i.L0, 0.2 , OverlapDB, Dof_C, CC011)
# print('CC011 Checked!')
# print('\n')
# LibW.CheckCC2Num( CCDB2_11, 1,1, i.Wavelet, i.L0, 0.2 )
# print('CC11 Checked!') 
# print('\n')
# LibW.CheckCC2Num2( LibW.Dict2Lam( CCDB2_01 ), 0,1, i.Wavelet, threshold, Dof_C, CC01, BasisList )
# # LibW.CheckCC2Num( CCDB2_01, 0,1, i.Wavelet, i.L0 , 0.2)
# print('CC01 Checked!')
# print('\n')
# LibW.CheckCC2Num( CCDB2_00, 0,0, i.Wavelet, i.L0, 0.2 )
# print('CC00 Checked!')

# LibW.CheckCC2Derivative( CCDB2_01, 0 ,1, i.Wavelet,  A, BasisList, i_list, CCDB2_01)

# =============================================================================
# 
# =============================================================================


u_C = copy.deepcopy(Coeffs)
u_S = pywt.waverec( Coeffs, i.Wavelet, mode='periodization' )



f, ax = plt.subplots(ncols = 3)
ie      = -1
Broken  = 0
for eMacro in i.erow:
    
    e_S = np.matmul( D1max, u_S ) + eMacro

    ie += 1
    print('/////' + str(ie) + '\\\\\\\\\ ' )
    print( eMacro )
    
    if Broken == 1:
        print('Specimen Broken!')
        break

    VarList     = [u_C, e_S, eMacro, i_C, i_list ] 
    u_S, u_C, e_S, fu, Kuu    = LibN.PhaseField_NR( BasisList, CC_list, VarList, Material, A, i.Wavelet )
    

    print(' Maximum Damage:' + str( max(Material.d)))
    
    Broken = 1 if max( Material.d) > 0.995 else 0
    
   
    

    ax[0].plot( u_S)
    ax[1].plot( e_S )
    ax[2].plot( Material.d )
    

# =============================================================================
# Adaptation
# =============================================================================
    if i.Adaptation:
        print("adapting basis")
        AdaptList = [i.Ndel, i.mintol, i.maxtol, i.Lmax]
        i_C, u_C  = LibAd.Adapt( i_C, u_C, AdaptList )
        
        i_list      = []
        d = -1
        for ll in range(0,len(i_C)):
            for ii in range(0, len(i_C[ll])): 
                d           += 1
                if i_C[ll][ii] == 1:
                    i_list.append( d )

# =============================================================================
# Saving results   
# =============================================================================
    
    print(len(i_list))
    
    print("Saving Results")
    ds_S, s_S = Material.f( e_S ) 
    ResultList = [ u_S, e_S, s_S, ds_S, A, Material.d, Material.H, i_C, Material.E, Material.l, Material.Gc]
    SaveData.SaveData( [ResultList], ['Results'], [i.MasterDir], 'PF' )
     
copyfile('inputfile.py', i.MasterDir + '/input.py')
