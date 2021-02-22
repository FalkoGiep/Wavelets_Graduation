import Auxiliary.LibAux as LibA

import numpy as np
import pywt


# =============================================================================
# Numerical parameters
# =============================================================================


eMacro      = 0.005    # Macroscopic strain
Nstart      = 1
Nemac       = 1000
erow        = np.arange( eMacro*Nstart/( Nemac), eMacro,eMacro/Nemac )

L           = 1

# =============================================================================
# 
# =============================================================================

MatName         = 'PhaseField'
E               = 10**7     # E modulus
Gc              = 1
l               = 0.005*L
MatParameters   = [E, l, Gc]

# =============================================================================
# Wavelet Parameters
# =============================================================================
# Numerical model parameters

L0          = 5
Lmax        = 10
LAdapt      = 2


WaveName = 'bior3.3'
CCdir    = 'CCDB'

if not WaveName == 'dd':
    fb = pywt.Wavelet(WaveName).filter_bank
    fbn = []
    for ii in fb[:2]:
        fbn.append( list(np.array( ii ) / 2 **0.5 ) )
    for ii in fb[2:]:
        fbn.append( list(np.array( ii ) * 2 **0.5 ) )
    
    Wavelet = pywt.Wavelet( filter_bank = fbn)
    Wavelet2 = pywt.Wavelet( filter_bank = fb)
if WaveName == 'dd':
    orderW = 2
    Wavelet = LibA.CalculateDeslaurierWavelet(orderW)
    WaveName = WaveName + str(orderW)

Adaptation = 1
Ndel = -1
mintol = 10**(-9)
maxtol = 10**(-7)

MasterDir = 'Data'

