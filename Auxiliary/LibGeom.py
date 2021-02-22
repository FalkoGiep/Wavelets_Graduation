
from Auxiliary.LibAux import SmoothPolynominal
import numpy as np
import math


# =============================================================================
# 
# =============================================================================
class Area:
    def __init__(self, AreaName, L, A0, Order, xgrid = np.arange(0,1,1/10000) ):
        self.L = L
        self.A0 = A0
        self.x  = xgrid
        if AreaName == 'RandPoly':
            p = SmoothPolynominal( 3 )
            self.p = p
            self.f = self.A
            
        elif AreaName == 'Constant':
            self.f = self.Const
            
        elif AreaName == 'FixedPoly':
            self.p = SmoothPolynominal( Order, Rand = [1, 1, 1, 1, 0, 1, 1, 1] )
            self.f = self.A
            
        elif AreaName == 'Sine':
            self.f = self.sine
            

            
        Arow   = []   
        for xx in self.x:
            Arow.append( self.f(xx))
        self.Arow = Arow
        
    def Const(self,x):
        return self.A0
    
    def sine(self,x):
        
        return ( ( math.sin( 2*math.pi*(x/self.L) ) )*self.A0 + 1.1*self.A0)
    
    def A(self, x):
        
        if x < 0:
            x = self.L + x
        if x > self.L:
            x = x - self.L
        
        return  self.A0* (self.p(x) + 1)