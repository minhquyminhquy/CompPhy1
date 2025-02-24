""" 
Course: FYS.420 Computational Physics
Name: Quy Le <quy.le@tuni.fi>
Student Id: 153118962
---------------------
Exercise 1 - Problem 3: Numerical derivatives

This script implements numerical derivative technique for approximating first and second derivative.
In addition, the script also contains test functions to ensure both functions work well.
"""

# PART A: First Derivative
def first_derivative(f, x, h):
    """
    Calculate the first derivative of function f(x).

    Parameters:
    f (function): Function f(x).
    x (float): Input value x for f(x).
    h (float): A small step size to approximate the second derivative.

    Returns:
    df_num (float): Approximate value of first derivative of f(x) at x.
    """
    f_x_plus_h = f(x+h)
    f_x_minus_h = f(x-h)
    df_num = (f_x_plus_h - f_x_minus_h) / (2*h) 
    return df_num

def test_first_derivative():
    """
    Test the function first_derivative by comparing its output with exact solution. 
    Print "Mission Completed" if works well, otherwise "Mission Failed".
    """
    def fun(x): return x**2 # Test function
    def dfun(x): return 2*x # Analytical first derivative

    h = 0.01
    epsilon = h / 10 # Tolerance for comparision

    xp = 1.0
    df_num = first_derivative(fun, xp, h)
    df_exact = dfun(xp)

    if abs(df_num-df_exact) < epsilon:
        print("First derivative: Mission Complete")
    else:
        print("First derivative: Mission Failed")

    assert abs(df_num-df_exact) < epsilon, "First derivative: Mission Failed"

# PART B: Second Derivative
def second_derivative(f, x, h):
    """
    Calculate the second derivative of function f(x).

    Parameters:
    f (function): Function f(x).
    x (float): Input value x for f(x).
    h (float): A small step size to approximate the second derivative.

    Returns:
    d2f_num (float): Approximate value of second derivative of f(x) at x.
    """
    f_x_plus_h = f(x+h)
    f_x_minus_h = f(x-h)
    f_x = f(x)
    d2f_num = (f_x_plus_h + f_x_minus_h - 2*f_x) / h**2 
    return d2f_num

def test_second_derivative():
    """
    Test the function second_derivative by comparing its output with exact solution.  
    Print "Mission Completed" if works well, otherwise "Mission Failed".
    """ 
    def fun(x): return x**2 # Test function
    def d2fun(x): return 2 # Analytical second derivative

    h = 0.01
    epsilon = h / 10 # Tolerance for comparison

    xp = 1.0
    df_num = first_derivative(fun,xp,h)
    df_exact = d2fun(xp)

    if abs(df_num-df_exact) < epsilon:
        print("Second derivative: Mission Complete")
    else:
        print("Second derivative: Mission Failed")

    assert abs(df_num-df_exact) < epsilon, "Second derivative: Mission Failed"

def main():
    test_first_derivative()
    test_second_derivative()

if __name__=="__main__":
    main()
