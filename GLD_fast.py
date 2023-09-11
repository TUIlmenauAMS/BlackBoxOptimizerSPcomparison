#Program for optimizing (minimizing) using Google Brains Gradientless Descent GLD-fast.
#usage: coeffmin=gldfast(objfunction, coeffs, args=(X,))
#arguments: objfunction: name of objective function to minimize
#coeffs: Starting vector for the optimization, also determines the dimensionality of the input for objfunction
#Can be a tensor of any dimension!
#X: further arguments for objfunction
#returns: coeffmin: coeff vector which minimizes objfunction
#According to: https://arxiv.org/pdf/1911.06317.pdf
#Gerald Schuller, July 2020

import numpy as np

def objfunccomp(objfunction, coeffs, args, bounds):
   #compact objective function for optimization
   if bounds !=():
         for n in range(len(bounds)):
            coeffs[n]=np.clip(coeffs[n],bounds[n][0], bounds[n][1])
   if len(args)==0:
      X0=objfunction(coeffs)
   elif len(args)==1:
      X0=objfunction(coeffs, args[0])
   else:
      X0=objfunction(coeffs, args)
   return X0
   

def gldfast(objfunction, coeffs, args=(), bounds=(), iterations=1000, startingscale=1):
   #Arguments: objfunction: (char string) name of objective function to minimize
   #coeffs: array of the coefficients for which to optimize
   #args: Additional arguments for objfunction, must be a tuple! (in round parenthesis)
   #bounds: bounds for variables or coefficients, sequence (Tuple or list) of pairs of minima and maxima for each coefficient (optional)
   #This sequence can be shorter than the coefficent array, then it is only applied to the first coefficients that the bounds cover.
   #This can be used to include side conditions. Example: bounds=((0,1.5),(-1.57,1.57))
   #iterations: number of iterations for optimization
   #startingscale: scale for the standard deviation for the random numbers at the beginning of the iterations 
   #returns: 
   #coeffsmin: The coefficients which minimize objfunction
   
   #print("args=", args, "args[0]=", args[0])
   coeffs=np.array(coeffs) #turn into numpy array (if not already)
   sh=coeffs.shape #shape of objfunction input
   #Simple online optimization, using random directions optimization:
   #Initialization of the deviation vector:
   #print("coeffdeviation=", coeffdeviation)

   #Old values 0, starting point:
   
   f0=objfunccomp(objfunction, coeffs, args, bounds)
   print("Obj. function starting value:", f0)
   #Determine condition number bound Q, an estimation of the upper bound of the norm of a Gradient:
   Q=0.0
   #valleys of the objective function can be too narrow to be found randomly for Q, better guess.
   #"""
   for m in range(20): #make tests for condition number
      coeffvariation=np.random.normal(loc=np.zeros(coeffs.shape), scale=0.1)
      f1=objfunccomp(objfunction, coeffs+coeffvariation, args, bounds)
      if abs(f1-f0)/0.1 > Q:
          Q=abs(f1-f0)/0.1
   Q=np.clip(Q,2,10) #this seems to be a good range of Q values
   print("Condition number Q=", Q)
   #"""
   #Q=4.0 #trial and error
   #print("Condition number Q=", Q)
   K=int(round(np.log(4*np.sqrt(Q))))
   if K<1: 
      K=1
   H=int(round(len(coeffs)*Q*np.log(Q)))
   if H<1:
      H=1
   print("K=", K, "H=", H)
   fk=np.zeros(2*K)
   xk=np.zeros(2*K)
   R=startingscale
   
   print("start iterations loop")
   for t in range(1,iterations//(K)+1): #iterations roughly correspond to function evaluations
      if t%H==0:
         R=R/2
      r=pow(2.0,np.arange(-K,K))*R
      xk={}
      for k in range(2*K):
         xk[k]=coeffs+np.random.normal(loc=np.zeros(coeffs.shape), scale=r[k])
         fk[k]=objfunccomp(objfunction, xk[k], args, bounds) #test in this range
      k= np.argmin(fk) #choose min from those points (orig. f is not in these points, because r is not becoming zero!)
      #print("k=", k)
      coeffs=xk[k]
      if t%1==0:
         print("iteration=", t,"f=", fk[k])
         
   print("coeffs=", coeffs)
   #f=objfunccomp(objfunction, coeffs, args, bounds)
   f=fk[k]
   print("f=", f)
   #coeffdeviation=1.0  #possible preset
   return coeffs
   


#testing:
if __name__ == '__main__':
   import numpy as np
   import matplotlib.pyplot as plt
   
   """
   #Example: Bessel function with 2 variables:
   from  scipy.special import jv #Bessel function
   xmin= optimrandomdir('jv', np.array([1.0]), (1.0,))
   #xmin= optimrandomdir('np.linalg.norm', np.array([1.0, 1.0]) )
   print("xmin=", xmin)
   """
   
   #Example: Superposition with 2 time discrete sines, resulting in a discrete 1d function with local minima:
   def objfunc(x):
      #objective function with local minima to find the minimum for
      #arg: x
      #returns: y=f(x)
      #x=np.round(x) #for functions defined only on a discrete set
      #print("x=", x)
      N=8
      y=np.sin(x[0]*3.14/N*2)+np.sin(x[1]*3.14/N*7.5)
      #print("y=", y)
      return y
      
   gldfast(objfunc, [0.1, 0.1] , args=(), bounds=(), iterations=20, startingscale=1)
   
   
