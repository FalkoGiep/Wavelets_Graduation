


import numpy             as np
import matplotlib.pyplot as plt
import pywt
import copy
from numpy.linalg        import solve, norm


import Wavelet.LibWavelets as LibW


# =============================================================================
# Matrix multiplications
# =============================================================================

def FindIdx( i1,i2,N):
    absolute = [abs(i2 - i1), abs( ( i2 - N )- i1 ),abs( i2 - ( i1 - N ) )]
    signs    = [np.sign(i2 - i1), np.sign( ( i2 - N )- i1 ),np.sign( i2 - ( i1 - N ) )]
    ii  = np.argmin(absolute)
    return absolute[ii]*signs[ii]
def coldot( Mat, Vec):
    M = np.zeros_like(Mat)
    
    for ii in range(0,len(M)):
        for jj in range(0,len(M[ii])):
            
            M[ii][jj] = Mat[ii][jj]*Vec[jj]
    return M

def HadamardShurdot(Vec1, Vec2 ):
    V = np.zeros_like(Vec1)
    for ii in range(0, len(Vec1)):
        V[ii] = Vec1[ii]*Vec2[ii]
    return V

        
# =============================================================================
# Construct K and f    
# =============================================================================

def MakeFd(  CC_list, Material, Wavelet, BasisList, VarList,OVrow, OVmat ):
    [u_C, e_S, eMacro, i_C, i_list ]                                       = VarList
    [CCDB3_011, CCDB3_010, CCDB3_000, CCDB2_01, CCDB2_00, CCDB2_11, D1max] = CC_list
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs]    = BasisList

    Gc = Material.Gc
    l  = Material.l
    d  = Material.d
    E  = Material.E
    H  = Material.Hold
    
    dgdd    = Material.dDegradation( d )
    dgddH   = np.multiply( dgdd, H )
    dgddH_C = pywt.wavedec( dgddH, Wavelet, mode='periodization', level = Lmax - L0 )
    
    f       = np.zeros( (1,len(i_list) ))[0].astype(float)
    fii     = -1
    for ii in i_list:
        fii    += 1
        
        i1      = IndexList[ii]
        l1      = LevelList[ii]
        w1      = 0 if l1 == 0 else 1
        list1   = [l1, i1, w1]
        
        j_list  = OVrow[ii]
        for jj in j_list:
            
            i2      = IndexList[jj]
            l2      = LevelList[jj]
            w2      = 0 if l2 == 0 else 1
            list2   = [ l2, i2, w2 ]
            
            key     = LibW.Abs2Key( list2, list1, Coeffs)

            Lnorm   = max(l1,l2) + L0
            f[fii] +=  2**(-Lnorm)*CCDB2_00( key )* Gc/2/l* Material.d_C[l2][i2]
            f[fii] +=  2**( Lnorm)*CCDB2_11( key )* Gc*2*l* Material.d_C[l2][i2]
            f[fii] +=  2**(-Lnorm)*CCDB2_00( key )* dgddH_C[l2][i2]
    
    return f
# =============================================================================
# 
# =============================================================================
def MakeKdd( CC_list, Material, Wavelet, BasisList, VarList, OVrow, OVmat ):
    [u_C, e_S, eMacro, i_C, i_list ]                                       = VarList
    [CCDB3_011, CCDB3_010, CCDB3_000, CCDB2_01, CCDB2_00, CCDB2_11, D1max] = CC_list
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs]    = BasisList
    
    Gc = Material.Gc
    l  = Material.l
    d  = Material.d
    E  = Material.E
    H  = Material.Hold

    ddgddd    = Material.ddDegradation( d )
    ddgdddH   = np.multiply( ddgddd, H )
    ddgdddH_C = pywt.wavedec( ddgdddH, Wavelet, mode = 'periodization', level = Lmax - L0 )

    K = np.zeros( (len(i_list), len(i_list) )).astype(float)
    Kii = -1
    for ii in i_list:
        Kii    += 1
        
        i1      = IndexList[ii]
        l1      = LevelList[ii]
        w1      = 0 if l1 == 0 else 1
        list1   = [l1, i1, w1]
    
        j_list  = OVrow[ii]
        for jj in j_list:
            Kjj     = int( np.argwhere( i_list == jj ) )
            
            i2      = IndexList[jj]
            l2      = LevelList[jj]
            w2      = 0 if l2 == 0 else 1
            list2   = [ l2, i2, w2 ]
            
            key     = LibW.Abs2Key( list2, list1, Coeffs)

            Lnorm       = max(l1,l2) + L0
            K[Kii,Kjj] += 2**( Lnorm*(-1))*CCDB2_00( key )*Gc/2/l
            K[Kii,Kjj] += 2**( Lnorm)*CCDB2_11( key )*Gc*2*l

            k_list  = OVmat[ii,jj]
            for kk in k_list:
                
                i3      = IndexList[kk]
                l3      = LevelList[kk]   
                w3      = 0 if l3 == 0 else 1
                list3   = [ l3, i3, w3 ]
                
                key     = LibW.Abs2Key2( list3, list2, list1, Coeffs)
               
                Lnorm       = max(l1,l2,l3) + L0
                K[Kii,Kjj] += 2**( Lnorm*(-1))*CCDB3_000( key )*ddgdddH_C[l3][i3]

    return K  


def MakeFu( CC_list, Material, A_S, Wavelet, BasisList, VarList, OVrow, OVmat ):
    [u_C, e_S, eMacro, i_C, i_list ]                                       = VarList
    [CCDB3_011, CCDB3_010, CCDB3_000, CCDB2_01, CCDB2_00, CCDB2_11, D1max] = CC_list
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs]    = BasisList
    
    Gc = Material.Gc
    l  = Material.l
    d  = Material.d
    E  = Material.E
    H  = Material.Hold
    
    g     = Material.Degradation( d )
    gEA   = np.multiply( g*E, A_S )
    gEA_C = pywt.wavedec( gEA, Wavelet, mode = 'periodization', level = Lmax - L0 )
    gEA_C2 = pywt.wavedec( gEA*eMacro, Wavelet, mode = 'periodization', level = Lmax - L0 )
    f1 = np.zeros( (1,len(i_list) ))[0].astype(float)
    f2 = np.zeros( (1,len(i_list) ))[0].astype(float)
    fii = -1
    for ii in i_list:
        fii    += 1
        i1      = IndexList[ii]
        l1      = LevelList[ii]
        w1      = 0 if l1 == 0 else 1
        list1   = [ l1, i1, w1 ]
        
        j_list  = OVrow[ii]
        for jj in j_list:
            
            i2      = IndexList[jj]
            l2      = LevelList[jj]
            w2      = 0 if l2 == 0 else 1
            list2   = [ l2, i2, w2 ]
            
            key      = LibW.Abs2Key( list2, list1, Coeffs)
            f2[fii] += CCDB2_01( key )*gEA_C2[l2][i2] 

            k_list   = OVmat[ii,jj]
            for kk in k_list:
                
               i3      = IndexList[kk]
               l3      = LevelList[kk]
               w3      = 0 if l3 == 0 else 1
               list3   = [l3,i3,w3]

               key     = LibW.Abs2Key2( list3, list2, list1, Coeffs)
               
               Lnorm    =  L0 + max(l1,l2,l3)
               
               f1[fii] += 2**(Lnorm)*CCDB3_011( key )*gEA_C[l3][i3]*u_C[l2][i2]

    fu = f1 + f2
    return fu

def MakeKuu( CC_list, Material, A_S, Wavelet, BasisList, VarList, OVrow, OVmat ):
    [u_C, e_S, eMacro, i_C, i_list ]                                       = VarList
    [CCDB3_011, CCDB3_010, CCDB3_000, CCDB2_01, CCDB2_00, CCDB2_11, D1max] = CC_list
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs]    = BasisList
    

    d  = Material.d
    E  = Material.E

    g     = Material.Degradation( d )
    gEA   = np.multiply( g*E, A_S )
    gEA_C = pywt.wavedec( gEA, Wavelet, mode = 'periodization', level = Lmax - L0 )

    K = np.zeros( (len(i_list), len(i_list) )).astype(float)
    
    Kii = -1
    for ii in i_list:
        Kii    += 1
        i1      = IndexList[ii]
        l1      = LevelList[ii]
        w1      = 0 if l1 == 0 else 1
        list1   = [ l1, i1, w1 ]
        
        j_list  = OVrow[ii]
        

        for jj in j_list:
            Kjj     = int( np.argwhere( i_list == jj ) )
            
            i2      = IndexList[jj]
            l2      = LevelList[jj]
            w2      = 0 if l2 == 0 else 1
            list2   = [ l2, i2, w2 ]
            
            
            k_list  = OVmat[ii,jj]
            for kk in k_list:
                
                i3      = IndexList[kk]
                l3      = LevelList[kk]   
                w3      = 0 if l3 == 0 else 1
                list3   = [l3,i3,w3]
                
                Lnorm   =  L0 + max(l1,l2,l3)
                
                key         = LibW.Abs2Key2( list3, list2, list1, Coeffs)
                K[Kii,Kjj] += 2**(Lnorm)*CCDB3_011( key )*gEA_C[l3][i3] 
        
    return K  



# =============================================================================
# 
# =============================================================================

def OV_Intersect( OverlapDB, i_list ):
    OVrow = {}
    for ii in i_list:
        OVrow[ii] =  np.intersect1d(i_list, OverlapDB[ii])
    
    OVmat = {}
    for ii in i_list:
        for jj in OVrow[ii]:
            OVmat[ii,jj] = np.intersect1d(OVrow[ii], OVrow[jj])
    

    
    return OVrow, OVmat

def PhaseField_NR( BasisList, CC_list, VarList, Material, A_S, Wavelet ):  

        
    [u_C, e_S, eMacro, i_C, i_list ]                                       = VarList
    [CCDB3_011, CCDB3_010, CCDB3_000, CCDB2_01, CCDB2_00, CCDB2_11, D1max] = CC_list
    [DofList, LevelList, IndexList, Dof_C, OverlapDB, L0, Lmax, Coeffs]    = BasisList

    OVrow, OVmat = OV_Intersect( OverlapDB, i_list )

    fd      = MakeFd(  CC_list, Material, Wavelet, BasisList, VarList, OVrow, OVmat )
    print('made fd')
    Kdd     = MakeKdd( CC_list, Material, Wavelet, BasisList, VarList, OVrow, OVmat )
    print('made Kdd')
    print("made d system")
    ddd     = solve( Kdd, -fd)
    print("solved d system")

    
    
    for jj in range( len( i_list ) ):
        dd                      = i_list[jj]
        ll                      = LevelList[dd]
        ii                      = IndexList[dd]
        Material.d_C[ll][ii]    += ddd[jj]

    Material.d  = pywt.waverec( Material.d_C, Wavelet, mode='periodization' )
    
    fu          = MakeFu(  CC_list, Material, A_S, Wavelet, BasisList, VarList, OVrow, OVmat )
    print('made fu')
    Kuu         = MakeKuu( CC_list, Material, A_S, Wavelet, BasisList, VarList, OVrow, OVmat )
    print('made Kuu')
    # Kuu[0,:]    = np.zeros_like(Kuu[0,:])
    # Kuu[0,0]    = 1
    # fu[0]       = 0

    print("made w system")
    ddu         = solve( Kuu, -fu)
    print("solved w system")


    # 
    for jj in range( len( i_list ) ):
        dd              = i_list[jj]
        ll              = LevelList[dd]
        ii              = IndexList[dd]
        u_C[ll][ii]     += ddu[jj]


    
    u_S         = pywt.waverec( u_C, Wavelet, mode='periodization' )
    u_S        -= u_S[0]
    
    du_S        = np.matmul( D1max, u_S )
    du_C        = pywt.wavedec( du_S, Wavelet, mode = 'periodization', level = Lmax - L0 )
    for ll in range(0,len(du_C)):
        for ii in range(0, len(i_C[ll])): 
            if not  i_C[ll][ii] == 1:
                du_C[ll][ii] = 0     
    du_S        = pywt.waverec( du_C, Wavelet, mode = 'periodization' )

    e_S         = du_S + eMacro
    
    
    # e_S = np.matmul( D1max, u_S) + eMacro
    
    # Calculate History on Lmax
    Material.UpdateHold( e_S )
    Material.UpdateH(    e_S )
    
    

    return u_S, u_C, e_S, fu, Kuu
    