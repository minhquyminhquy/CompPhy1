""" 
Course: FYS.420 Computational Physics 1
Name: Quy Le <quy.le@tuni.fi>
---------------------
Exercise 2 - Problem 4 : Root search
This script implement root search. It also includes a test case.
"""

import numpy as np

def index_search(f):
    """Finds the index i where f[i] <= 0 < f[i+1] using sort of binary search"""
    n_iters = 0
    lower = 0
    upper = len(f)
    while lower < upper:
        n_iters += 1
        mid = (lower + upper) // 2
        if f[mid] > 0:
            upper = mid
        else:
            lower = mid + 1
    i = lower - 1
    return i, n_iters

def generate_grid(r0, rmax, dim):
    """Generates a grid of points that are logarithmically spaced."""
    h = np.log(rmax / r0 + 1) / (dim - 1)
    x = np.zeros(dim)
    for i in range(1, dim):
        x[i] = r0 * (np.exp(i * h) - 1)
    return x

def test_index_search():
    """Tests the index_search function by generating a logarithmic grid and checking
    if the function correctly finds the index for random points within the grid."""
    r0 = 1e-4
    rmax = 50
    dim = 50
    x = generate_grid(r0, rmax, dim)
    xp_list = np.random.uniform(x[0], x[-1], size=5)
    
    for xp in xp_list:
        f = x - xp
        i, _ = index_search(f)
        expected_i = np.searchsorted(x, xp, side='right') - 1
        
        assert i == expected_i, f"Failed for xp={xp}: different values of i"
        assert x[i] <= xp, f"x[i] not less or equal to xp"

        if i + 1 < len(x):
            assert xp < x[i+1], f"xp not less than x[i+1]"
    print("Mission completed")

if __name__ == "__main__":
    test_index_search()