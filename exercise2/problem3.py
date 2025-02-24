""" 
Course: FYS.420 Computational Physics 1
Name: Quy Le <quy.le@tuni.fi>
---------------------
Exercise 2 - Problem 3 : Interpolation

This script compares 1D linear and Hermite cubic spline interpolations. This script also include 
a test case.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from linear_interp import interp1d 
from spline_class import spline     

x_points = np.linspace(0, 2*np.pi, 11)
y_points = np.sin(x_points)
x_true = np.linspace(0, 2*np.pi, 80)
y_true = np.sin(x_true)

# Figure 1: Compare our linear vs our cubic spline
y_linear = np.array([interp1d(x, y_points, x_points) for x in x_true])
spline1d = spline(dims=1, x=x_points, f=y_points)
y_cubic = spline1d.eval1d(x_true)

plt.figure(figsize=(10, 6))
plt.plot(x_true, y_true, '-', label='Ground truth sin(x)', linewidth=2)
plt.plot(x_true, y_linear, '--', label='Our linear', linewidth=1)
plt.plot(x_true, y_cubic, ':', label='Our Hermite', linewidth=1)
plt.scatter(x_points, y_points, color='green', label='Samples')
plt.title('Interpolation Methods Comparison')
plt.xlabel('$x$')
plt.ylabel('$sin(x)$')
plt.legend()


# Figure 2: Compare numpy vs scipy 
y_numpy = np.interp(x_true, x_points, y_points)
cs_scipy = CubicSpline(x_points, y_points, bc_type='natural')
y_scipy = cs_scipy(x_true)

plt.figure(figsize=(10, 6))
plt.plot(x_true, y_true, '-', label='Grouth truth sin(x)', linewidth=2)
plt.plot(x_true, y_numpy, '--', label='NumPy', linewidth=1)
plt.plot(x_true, y_scipy, ':', label='SciPy', linewidth=1)
plt.scatter(x_points, y_points, color='pink', label='Samples')
plt.title('Libraries Comparison')
plt.xlabel('$x$')
plt.ylabel('$sin(x)$')
plt.legend()


plt.show()