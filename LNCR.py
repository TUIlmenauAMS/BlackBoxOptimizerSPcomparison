"""
Local Non-Convexity Ratio computation, of an n-dimensional function f at an n-dimensional point x.

Gerald Schuller, August 2023
"""

import numpy as np
from midpoint_convexity_test import is_midpoint_convex
import matplotlib.pyplot as plt

def LNCR(f, x, diameter=1e-1):
   #x is the n-dimensional center point for the LNCR
   
   x = np.array(x)
   d= np.arange(1,11)/10*diameter #10 distances from the center point
   S=10
   R=np.zeros(d.shape)

   for i in range(len(d)): #distanc indices
      V=0
      #x=np.random.random(x.shape)
      for iter in range(10): #trials at a given distance
         delta=np.random.normal(size=x.shape)  
         delta= delta/np.linalg.norm(delta)*d[i] #normalizse
         #print("np.linalg.norm(delta)=", np.linalg.norm(delta))
         xi=is_midpoint_convex(f, x, delta)
         #print("xi=", xi)
         V+=xi #Number of convex points
      V=V/S #fraction of convex points
      R[i]=1-V #fraction of non-convex points
      
   plt.plot(d,R)
   plt.title("Local Nonconvex Ratio")
   plt.xlabel("Distance from center point")
   plt.ylabel("Nonconvex ratio R")
   plt.show()
   return d, R
      
if __name__ == '__main__':
   # Test the function
   def f(x):
       #return np.sum(np.array(x)**2)
       return np.sin(np.linalg.norm(np.array(x))*30)
       
       
   x = [0.0, 0.0]
   d,R=LNCR(f,x, diameter=1.0)

   #print("d,R=", d,R)



