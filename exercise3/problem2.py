""" 
Course: FYS.420 Computational Physics 1
Name: Quy Le <quy.le@tuni.fi>
---------------------
Exercise 2 - Problem 2: Matrix vector product and “mystery matrix”

This script contains the derivation of derivative operator A and the result of calculating
first-order derivative with A on 4 different function and visualize them.
"""

import numpy as np
import matplotlib.pyplot as plt

"""
* Explanation of matrix A:
Matrix A is the operator that map function f into the first-order derivative of f. 
In this problem, matrix A map a column vector of x: x=[x_0, x_1,...,x_{N-1}] to a column vector b
that represent the first-order derivative of x. This can be present as A.x=b with x'=b.

* Derivativation of matrix A: 
- Shape: shape of x is (N, 1), shape of b is (N,1). Thus, shape of A is (N, N) so that
A(N,N) . x(N,1) = b(N,1)
- x=[x_0, x_1, x_2,...,x_{N-1}]. To approximate the derivative x_i', i.e b_i, we compute 
( x_{i+1} - x_{i-1} )/(2*h) with h = dx = x[1]-x[0]. For the boundary points, we use different ways:
    + At i=0, b_0 = (-3x_0 + 4x_1 - x_2) / (2h)
    + At i=N-1, b_{N-1} = x_{N-3} - 4x_{N-2} + 3x_{N-1} / (2h)
With this, now we form matrix A:
    - b_11 = b_0 = row_1 A . col_1 x <=> (-3x_0 + 4x_1 - x_2) / (2h) = row_1 A . [x_0, x_1, x_2,...,x_{49}]
    => row_1 A = [-3,4,-1,0...] * 1/(2h)
    - b_21 = b_1 = row_2 A . col_1 x <=> ( -x_0 + x_2 ) / (2h) = row_2 A . [x_0, x_1, x_2,...,x_{49}]
    => row_2 A = [-1, 0, 1, 0...] * 1/(2h)
    - b_31 = b_2 = row_3 A . col_1 x <=> ( -x_1 + x_3 ) / (2h) = row_3 A . [x_0, x_1, x_2,...,x_{49}]
    => row_3 A = [0, -1, 0, 1, 0...] * 1/(2h)
    - b_41 = b_3 = row_4 A . col_1 x <=> ( -x_2 + x_3 ) / (2h) = row_4 A . [x_0, x_1, x_2,...,x_{49}]
    => row_4 A = [0, 0, -1, 0, 1, 0...] * 1/(2h)
    This pattern continue until row_49 of A.
    - b_50_1 = b_49 = row_50 . cow_1 x <=> x_{47} - 4x_{48} + 3x_{49} / (2h) = row_50 A . [x_0, x_1, x_2,...,x_{49}]
    => row_50 A = [0..., 1, -4, 3]
The matrix A now looks like this
[
-3, 4,-1, 0, 0, 0...
-1, 0, 1, 0, 0, 0...
 0,-1, 0, 1, 0, 0...
 0, 0,-1, 0, 1, 0...
 0..., 1, -4, 3
] * 1/(2h)

"""
def make_matrix(x):
    """Generates operator matrix A from vector x"""
    N = len(x)
    h = x[1]-x[0]

    off_diag = np.ones((N-1,))

    # Fill in A with coefficient in from row_2 to row_49
    A = np.zeros((N,N)) - np.diag(off_diag,-1) + np.diag(off_diag,1)

    # Fill A with coefficient in row 1
    A[0,0:3] = [-3.0,4.0,-1.0] 

    # Fill A with coefficient in row 2
    A[-1,-3:] = [1.0,-4.0,3.0] 
    return A/(h*2)

def main():
    N = 50

    # domain of x when plug in functions
    grid = np.linspace(0,np.pi,N)

    # 4 different functions
    x1 = np.sin(grid)
    x2 = np.cos(grid)
    x3 = np.sin(grid) + np.cos(grid)
    x4 = 2*grid + 3

    # Compute derivative operator matrix A
    A = make_matrix(grid)
    
    # Compute first-order derivative of the 4 functions
    b1 = A@x1
    b2 = A@x2
    b3 = A@x3
    b4 = A@x4

    fig = plt.figure(figsize=(16,8))

    ax1 = fig.add_subplot(2, 2, 1)  
    ax2 = fig.add_subplot(2, 2, 2)  
    ax3 = fig.add_subplot(2, 2, 3) 
    ax4 = fig.add_subplot(2, 2, 4) 

    # Visualize the result of 4 different equations and their derivative
    ax1.plot(grid, x1, label='sin(x)')
    ax1.plot(grid, b1, label='A*sin(x)')
    ax1.legend()
    ax1.set_title('sin(x) and A*sin(x)')

    ax2.plot(grid, x2, label='cos(x)')
    ax2.plot(grid, b2, label='A*cos(x)')
    ax2.legend()
    ax2.set_title('cos(x) and A*cos(x)')

    ax3.plot(grid, x3, label='sin(x) + cos(x)')
    ax3.plot(grid, b3, label='A*(sin(x) + cos(x))')
    ax3.legend()
    ax3.set_title('sin(x) + cos(x) and A*(sin(x) + cos(x))')

    ax4.plot(grid, x4, label='2*x + 3')
    ax4.plot(grid, b4, label='A*(2*x + 3)')
    ax4.legend()
    ax4.set_title('2*x + 3 and A*(2*x + 3)')

    plt.tight_layout()
    plt.show()
    
if __name__=="__main__":
    main()



