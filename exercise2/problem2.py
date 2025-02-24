""" 
Course: FYS.420 Computational Physics 1
Name: Quy Le <quy.le@tuni.fi>
---------------------
Exercise 2 - Problem 2b: N-dimensional numerical gradient with O(h^4) derivative

This script implements numerical derivative technique for approximating first derivative and gradient
In addition, the script also contains test functions to ensure both functions work well.
"""

import numpy as np

def first_derivative(func, x, h):
    """Calculate the first derivative of func(x) using the derived O(h^4) formula derived in 2a.
    f'(x) = 1 / (12h) * [f(x - 2h) - 8f(x - h) + 8f(x + h) - f(x + 2h)]
    """
    coeff = 1 / (12 * h)
    sum_terms = func(x - 2 * h) - 8 * func(x - h) + 8 * func(x + h) - func(x + 2 * h)
    return coeff * sum_terms

def test_first_derivative():
    """Test the first_derivative function in both 1D and 3D cases."""
    # Test 1D case
    func = lambda x: x**2 + 2  
    dfunc = lambda x: 2 * x    
    x = 3                      
    h = 0.01                   
    dfunc_num = first_derivative(func, x, h)
    dfunc_analytical = dfunc(x)
    if abs(dfunc_num - dfunc_analytical) < 1e-6:
        print("In 1D case, first derivative works!")
    else:
        print("Mission failed")

    assert abs(dfunc_num - dfunc_analytical) < 1e-6

    # Test 3D case
    func = lambda X: X[0]**2 + X[1]**2 + X[2]**2  
    dfunc_dx = lambda X: 2 * X[0]                
    x = np.array([1.0, 2.0, 3.0])                
    h = 1e-4                            
    dfunc_dx_num = first_derivative(
        lambda t: func(np.array([t, x[1], x[2]])), x[0], h)
    dfunc_dx_analytical = dfunc_dx(x)
    if abs(dfunc_dx_num - dfunc_dx_analytical) < 1e-6:
        print("In 3D case, first derivative works!")
    else:
        print("Mission failed")
    assert abs(dfunc_dx_num - dfunc_dx_analytical) < 1e-6

def numerical_gradient(func, x, dx=0.01):
    """
    Calculate the gradient of function func(x) numerically using O(h^4) formula.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        def partial_func(xi):
            x_copy = x.copy()
            x_copy[i] = xi
            return func(x_copy)

        grad[i] = first_derivative(partial_func, x[i], dx)
    return grad

def func(x):
    """Test function: f(x) = cos(x1) + sin(xN) + sum(xi^3 for i=2 to N-1)."""
    N = x.shape[0]
    return np.cos(x[0]) + np.sin(x[-1]) + np.sum(x[1:N-1]**3)

def grad_func(x):
    """Analytical gradient of the test function for N = 3."""
    df_dx1 = -np.sin(x[0])
    df_dx3 = np.cos(x[2])
    df_dx2 = 3 * x[1]**2
    grad = np.array([df_dx1, df_dx2, df_dx3])
    return grad

def test_numerical_gradient():
    x = np.array([0.5, 1.0, 1.5])  
    grad_num = numerical_gradient(func, x, dx=1e-4)
    grad_analytical = grad_func(x)
    if np.allclose(grad_num, grad_analytical, atol=1e-6):
        print("Numerical gradient works!")
    else:
        print("Mission failed")

    assert np.allclose(grad_num, grad_analytical, atol=1e-6)

def main():
    test_first_derivative()
    test_numerical_gradient()

    x = np.array([0.5, 2.0, 1.5])  
    grad_num = numerical_gradient(func, x, dx=1e-4)
    grad_analytical = grad_func(x)
    print(f"Numerical gradient of function func at x: {grad_num}" )
    print(f"Analytical gradient of function func at x: {grad_analytical}")

if __name__ == "__main__":
    main()
