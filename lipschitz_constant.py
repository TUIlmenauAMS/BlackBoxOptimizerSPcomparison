"""
From ChatGPT with "write a python program to estimate the Lipschitz constant of a function"
Adapted to multidimensional functions
Gerald Schuller, August 2023
"""
import numpy as np

def lipschitz_constant(f, x0, diameter=1e-3, num_samples=10):
    """
    Estimate the Lipschitz constant of a function f over the area x+diameter, where x0 is a multidimensional point.

    Parameters:
    - f: The function for which to estimate the Lipschitz constant.
    - x0, diameter: The region over which to estimate the Lipschitz constant.
    - num_samples: The number of samples to use for the estimation.

    Returns:
    - L: The estimated Lipschitz constant.
    """
    num_samples
    
    L = 0
    for i in range(num_samples):
         dx = np.random.normal(scale= diameter*np.ones(x0.shape)) #normal distributed random sample
         L_i = abs(f(x0+dx) - f(x0)) / np.linalg.norm(dx)
         L = max(L, L_i)
    return L

if __name__ == '__main__':
   # Example usage:
   f = lambda x: x**2
   x0=np.array(1) #x0 needs to be a numpy array
   diameter=1e-3
   L = lipschitz_constant(f, x0, diameter=diameter)
   print(f"Lipschitz constant of f over [{x0} +- {diameter}] is approximately {L:.2f}")


