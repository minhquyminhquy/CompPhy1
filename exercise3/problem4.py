""" 
Course: FYS.420 Computational Physics 1
Name: Quy Le <quy.le@tuni.fi>
---------------------
Exercise 3 - Problem 4: 2D interpolation

The script also performs 2D interpolation on experimental data and compares the results with reference data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def load_and_prepare_data():
    """
    Load experimental data and grid points, reshape the data, and return.
    """
    xexp = np.loadtxt(r"x_grid.txt")
    yexp = np.loadtxt(r"y_grid.txt")
    Fexp = np.loadtxt(r"exp_data.txt").reshape([len(xexp), len(yexp)])

    return xexp, yexp, Fexp

def create_interpolator(xexp, yexp, Fexp):
    """
    Create and return a RectBivariateSpline interpolator for the experimental data.
    """
    return RectBivariateSpline(xexp, yexp, Fexp)

def generate_path(xexp, yexp):
    """
    Generate x_vals and y_vals for the path based on max x and y values.
    """
    y_max = np.max(yexp)
    x_max_y = (3*y_max)**(1/3) # inverse of y(x)
    x_max_x = np.max(xexp)
    x_max = min(x_max_y, x_max_x)

    x_vals = np.linspace(0, x_max, 100)
    y_vals = (1/3)*x_vals**3  

    return x_vals, y_vals

def compare_interpolation(F_interp, ref_data):
    """
    Compare the interpolated data with the reference data and print the result.
    """
    print("Interpolation close to reference:", np.allclose(F_interp, ref_data, rtol=1e-2))

def plot_results(X, Y, Fexp, x_vals, y_vals, F_interp, ref_data):
    """
    Generate the plots for the experimental data, path, and comparison with the reference data.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Experimental data, grid, and path
    ax1.contourf(X, Y, Fexp)
    ax1.scatter(X, Y, color='red', s=6)
    ax1.plot(x_vals, y_vals, 'b-', linewidth=4)
    ax1.set_title('Experimental Data and Path')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    # Interpolated vs Reference data
    ax2.plot(x_vals, F_interp, 'r-', label='Interpolated')
    ax2.plot(x_vals, ref_data, 'b--', label='Reference')
    ax2.set_title('Interpolation vs Reference')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$F$')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    xexp, yexp, Fexp = load_and_prepare_data()
    X, Y = np.meshgrid(xexp, yexp, indexing='ij')
    x_vals, y_vals = generate_path(xexp, yexp)
    interp = create_interpolator(xexp, yexp, Fexp)
    F_interp = interp(x_vals, y_vals, grid=False)
    ref_data = np.loadtxt(r"ref_interpolated.txt")
    compare_interpolation(F_interp, ref_data)
    plot_results(X, Y, Fexp, x_vals, y_vals, F_interp, ref_data)

if __name__ == "__main__":
    main()
