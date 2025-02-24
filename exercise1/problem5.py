""" 
Course: FYS.420 Computational Physics
Name: Quy Le <quy.le@tuni.fi>
Student Id: 153118962
---------------------
Exercise 1 - Problem 5: Numerics and plotting

This script plot the error of derivative approximation and integral approximation with respect to 
the width of intervals, i.e. grid spacing. The graph illustrate the convergence properties 
of the derivate function and the trapezoidal integration function.
"""

import numpy as np
import matplotlib.pyplot as plt
from exercise2.problem3 import first_derivative
from problem4a import trapezoid

# Plotting settings
plt.rcParams["legend.handlelength"] = 2
plt.rcParams["legend.numpoints"] = 1
plt.rcParams["font.size"] = 12

# Create 6x12 figure
fig = plt.figure(figsize=(6,12))

# Setup axes
ax = fig.add_subplot()

# Define the range for the number of points (N) in the grid and initialize arrays
N = np.arange(100)+3 # range [3, 102]
dx = np.zeros(N.shape)
f_int = np.zeros_like(dx) # Array storing the error of trapezoidal integration
f_der = 0.0*f_int # Array storing the error of numerical derivative

# Loop over the range of N to compute derivative and integral errors
for i in range(len(N)):
    x = np.linspace(0,np.pi/2,N[i])
    dx[i] = x[1]-x[0]
    # Absolute error in integral approximation
    f_int[i] = abs(1.0-trapezoid(np.sin(x),x)) 
    # Absolute error in derivative approximation
    f_der[i] = abs(np.cos(0.5)-first_derivative(np.sin,0.5,dx[i])) 


# add title to the plot
ax.set_title(r"Numerical Derivative vs Integration Error")

# plot and add label if legend desired
ax.plot(dx,f_int,label=r"Error in integral approximation of $f(x)=\sin(x)$")
ax.plot(dx,f_der,"--",label=r"Error in derivative approximation of $f(x)=\sin(x)$")


# include legend (with best location, i.e., loc=0)
ax.legend(loc=0)
# set axes labels and limits
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel(r"Error")
ax.set_xlim(x.min(), x.max())

# set x and y axis ticklabel precision (now one decimal for both)
ax.xaxis.set_major_formatter("{x:.1f}")
ax.yaxis.set_major_formatter("{x:.1f}")
fig.tight_layout(pad=1)

# save figure, e.g., as pdf
fig.savefig("testfile.pdf")
plt.show()
