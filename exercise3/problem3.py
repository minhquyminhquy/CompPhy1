""" 
Course: FYS.420 Computational Physics 1
Name: Quy Le <quy.le@tuni.fi>
---------------------
Exercise 3 - Problem 3: Simple eigenvalue solver using Power method

This script implement the eigenvalue solver using the Power method from lecture 3.
"""

"""
Computational physics 1

1. Add code to function 'largest_eig'
- use the power method to obtain the largest eigenvalue and the 
  corresponding eigenvector of the provided matrix

2. Compare the results with scipy's eigs
- this is provided, but you should use that to validating your 
  power method implementation
- they should yield the same, if not, then something is still wrong!!

3. Add code to function 'eigen_solver'
- obtain neigs number of largest eigenvalues and corresponding
  eigenvectors
- compare against scipy's results, which should match yours

Notice: 
  - np.dot(A,x), A.dot(x), A @ x could be helpful for performing 
    matrix operations with sparse matrices
  - numpy's toarray function might prove useful, and in general
    you could manage without it
  - for vv^T like product you might want to use np.outer(v,v)
"""


import numpy as np
from numpy.linalg import matrix_power, norm
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simpson

def make_matrix(grid):
    """Generates a sparse tridiagonal matrix based on the input grid."""
    grid_size = grid.shape[0]
    dx = grid[1]-grid[0]
    dx2 = dx*dx
    
    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size) - 1.0/(abs(grid)+2.0),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])

    return H0

def largest_eig(A,tol=1e-12,maxiters=100_000):
    """
    Implement Power Method to compute largest eigenvalue and its corresponding 
    eigenvector of given matrix A

    Parameters:
    A (numpy.ndarray) : The matrix to be computed.
    tol (float): The tolerance for convergence of the power method.
    maxiters (int): maximum number of iteration.
    
    Returns:
    eig (float): The largest eigenvalue
    evec (numpy.ndarray): The corresponding eigenvector
    """

    N = A.shape[0]
    eig = 0.0
    evec = np.ones(N)+abs(np.random.rand(N)) 

    # Power iteration loop
    for _ in range(maxiters):
      evec_new = (A @ evec) / norm(A @ evec)

      eig_new = norm(A @ evec_new) / norm(evec_new) 

      # Check for convergence
      if abs(eig_new-eig) < tol:
        return eig_new, evec_new
      
      eig = eig_new
      evec = evec_new

    return eig, evec

def test_largest_eig():
    """
    Tests the largest_eig function by comparing its result with the eigenvalue
    obtained using scipy's `eigsh` for two different grid sizes.
    """

    # Compare largest_eig estimate with scipy eigsh estimate 1st time
    x = np.linspace(-2,2,50)
    H1 = make_matrix(x)
    eigs, evecs = sla.eigsh(H1, k=1, which='LA')
    eig_val,eig_vec = largest_eig(H1)
    if (eig_val - eigs[0]) < 1e-5:
        print("largest_eig, 1st time: Mission completed")
    else:
        print("largest_eig, 1st time: Mission failed")

    assert (eig_val - eigs[0]) < 1e-5,"largest_eig: Mission failed"
    
    # Compare largest_eig estimate with scipy eigsh estimate 2nd time
    x = np.linspace(-3,3,101)
    H2 = make_matrix(x)    
    eigs, evecs = sla.eigsh(H2, k=1, which='LA')
    eig_val,eig_vec = largest_eig(H2)
    if (eig_val - eigs[0]) < 1e-5:
        print("largest_eig, 2nd time: Mission completed")
    else:
        print("largest_eig, 2nd time: Mission failed")

    assert (eig_val - eigs[0]) < 1e-5,"largest_eig: Mission failed" 

def eigen_solver(Ain,neigs=5,tol=1e-12):
    """
    Computes the #neigs largest eigenvalues and corresponding eigenvectors 
    of the matrix Ain using the power method.

    Parameters:
    Ain (numpy.ndarray): The matrix to be solved for eigenvalues.
    neigs (int): The number of eigenvalues 
    tol (float): The tolerance for convergence in the power method
    
    Returns:
    eigs (list): list of the `neigs` largest eigenvalues.
    evecs (list): list of the corresponding eigenvectors.
    """
    Amod = Ain.toarray()*1.0
    eigs = []
    evecs = []
    
    # Perform power iteration for the number of eigenvalues
    for _ in range(neigs):
        eig, evec = largest_eig(Amod, tol=tol)
        eigs.append(eig)
        evecs.append(evec)

        v_outer = np.outer(evec, evec)
        # Forming new matrix Amod by implementing equation (33)
        Amod -= eig * v_outer / np.dot(evec.T, evec)

    return eigs, evecs

def test_eigen_solver():
    """
    Tests the eigen_solver function by comparing its results with scipy's 
    `eigsh` function for multiple eigenvalues.
    """
    x = np.linspace(-2, 2, 50)
    H1 = make_matrix(x)
    
    neigs = 4
    eigs_scipy, evecs_scipy = sla.eigsh(H1, k=neigs, which='LA')
    eig_val, eig_vec = eigen_solver(H1, neigs=neigs)
    
    eig_val.sort()
    eigs_scipy.sort()
    
    for i in range(neigs):
        if abs(eig_val[i] - eigs_scipy[i]) < 1e-5:
            print(f"Eigenvalue {i+1}: Mission completed")
        else:
            print(f"Eigenvalue {i+1}: Mission failed")

    for i in range(neigs):
        assert abs(eig_val[i] - eigs_scipy[i]) < 1e-5,"largest_eig: Mission failed" 

   
def main():
    x = np.linspace(-5,5,101)
    H2 = make_matrix(x)    
    eigs, evecs = sla.eigsh(H2, k=1, which='LA')
    eig_val,eig_vec = largest_eig(H2)

    psi0 = evecs[:,0]
    norm_const = np.sqrt(simpson(abs(psi0)**2,x=x))
    psi0 = psi0/norm_const

    psi0_ = eig_vec*1.0
    norm_const = np.sqrt(simpson(abs(psi0_)**2,x=x))
    psi0_ = psi0_/norm_const

    plt.plot(x,abs(psi0)**2,label='scipy eig. vector squared')
    plt.plot(x,abs(psi0_)**2,'r--',label='largest_eig vector squared')
    plt.legend(loc=0)
    
    if abs(eig_val-eigs)<1e-6 and np.amax(abs(abs(psi0)**2-abs(psi0_)**2))<1e-2:
        print("Working fine")
    else:
        print("\nNOT yet working as expected!\n")

    plt.show()

if __name__=="__main__":
    main()
