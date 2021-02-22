


import numpy as np
# =============================================================================
# Material struct
# =============================================================================
class Material:
    
    def __init__(self, Name, Parameters ):
        if Name == 'Linear':
            E           = Parameters[0]
            self.Name   = Name
            self.E      = E
            self.f      = self.LIN
        elif Name == 'MultiLinear':
            self.Name   = Name
            self.E1     = Parameters[0]
            self.E2     = Parameters[1]
            self.E3     = Parameters[2]
            
            self.ep1    = Parameters[3]
            self.ep2    = Parameters[4]
            self.f      = self.MLIN
        elif Name == 'Smooth':
            self.Name   = Name
            self.E      = Parameters[0]
            self.e0     = Parameters[1]
            self.f      = self.SMOOTH
        elif Name == 'PhaseField':
            self.Name   = Name
            self.E      = Parameters[0]
            self.l      = Parameters[1]
            self.Gc     = Parameters[2]
            self.dC     = []
            self.iCd    = []
            self.d      = []
            self.H      = []
            self.Hold   = []
            self.f      = self.PF

    # Linear material function
    def LIN(self, eA):
        dS = np.zeros_like(eA)
        S  = np.zeros_like(eA)
        for ii in range(0,len(eA)):
            e      = eA[ii]
            dS[ii] = self.E
            S[ii]  = self.E*e
        return  dS, S
    
    # Multi-linear material function
    def MLIN( self, eA):
        dS = np.zeros_like(eA)
        S  = np.zeros_like(eA)
        for ii in range(0,len(eA)):
            e = eA[ii]
            if e < self.ep1:
                dS[ii] = self.E1
                S[ii]  = self.E1*e       
            elif e < self.ep2:
                dS[ii] = self.E2
                S[ii]  = self.E1*self.ep1 + self.E2 *( e - self.ep1 )     
            else:
                dS[ii] = self.E3
                S[ii]  =  self.E1*self.ep1 + self.E2*( self.ep2 - self.ep1) + self.E3 *( e - self.ep1 - self.ep2 )    
        return dS, S
     
    # Non-linear smooth function
    def SMOOTH( self, eA ):
        dS = np.zeros_like(eA)
        S  = np.zeros_like(eA)
        for ii in range(0,len(eA)):
            e       = eA[ii] + self.e0
            S[ii]   = self.E* (e)**0.5 - self.E*(self.e0)**0.5
            dS[ii]  = 0.5*self.E*(  (e)**-0.5 )
        return dS, S
    
# =============================================================================
#     Phase Field stuff
# =============================================================================
    def PF( self, eA ):
        dS = np.zeros_like(eA)
        S  = np.zeros_like(eA)
        for ii in range(0,len(eA)):
            e = eA[ii]
            d = self.d[ii]
            
            dS[ii]  = self.E*self.Degradation(d)
            S[ii]   = dS[ii]*e
        return dS, S       
    
    
    def Degradation(self, d):
        return ((1-d)**2) + 10**-6
    
    def dDegradation(self, d):
        return -(2*(1-d))

    def ddDegradation(self, d):
        return 2 + 0*d
    
#    def Degradation(self, d):
#        s = 1/10
#        a1 = s*(1-d)**3
#        a2 = s*(1-d)**2
#        a3 = 3*(1-d)**2
#        a4 = 2*(1-d)**3
#        return a1-a2+a3-a4 
#    
#    def dDegradation(self, d):
#        s = 1/10
#        a1 = 3*s*(1-d)**2
#        a2 = 2*s*(1-d)
#        a3 = 2*3*(1-d)
#        a4 = 3*2*(1-d)**2
#        return -(a1-a2+a3-a4)
#    
#    def ddDegradation(self, d):
#        s = 1/10
#        a1 = -6*s*(1-d)
#        a2 = -2*s
#        a3 = -2*3
#        a4 = -2*3*2*(1-d)
#        return -(a1-a2+a3-a4)
    
    def UpdateH( self, e ):
        for ii in range(0, len(e)):
            if self.H[ii] < 0.5*e[ii]*self.E*e[ii]:
                self.H[ii] = 0.5*e[ii]*self.E*e[ii]
        return self.H
    
    def UpdateHold( self, e ):
        for ii in range(0, len(e)):
            if self.H[ii] < 0.5*e[ii]*self.E*e[ii]:
                self.Hold[ii] = 0.5*e[ii]*self.E*e[ii]
        return self.Hold
