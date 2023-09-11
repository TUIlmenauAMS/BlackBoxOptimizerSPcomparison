"""
Write a python function which tests if an n-dimensional function f is midpoint-convex at a given n-dimensional point x.

Gerald Schuller, August 2023
"""

import numpy as np

def is_midpoint_convex(f, x, delta):
    """
    Check if the function f is midpoint-convex at point x by perturbing x with delta to get y.

    Parameters:
    - f: function to evaluate
    - x: n-dimensional point at which to check midpoint convexity
    - delta: perturbation vector to generate the second point y

    Returns:
    - Boolean indicating if f is midpoint-convex at x using y = x + delta
    """
    # Convert x and delta to numpy arrays for vectorized operations
    x = np.array(x)
    y = x + np.array(delta)
    
    midpoint = (x + y) / 2
    #print("f(midpoint)=", f(midpoint))
    return f(midpoint) <= 0.5 * (f(x) + f(y))

if __name__ == '__main__':

   # Test the function
   def f(x):
       return np.sum(np.array(x)**2)

   x = [1.0, 2.0]
   delta = [0.5, -0.5]
   print(f"Is f(x) = sum(x^2) midpoint-convex at x = {x} with perturbation {delta}? {is_midpoint_convex(f, x, delta)}")

