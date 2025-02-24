"""
Module for simple linear interpolation

Assignment:
- Fill in the calculations for 1d, 2d and 3d linear interpolations
"""

import numpy as np
import matplotlib.pyplot as plt

l1 = lambda t: 1.0-t
l2 = lambda t: t

def get_index(x0,x):
    if (x0<=x[0]):
        i = 0
    elif (x0>=x[-2]):
        i = len(x)-2
    else:
        i = np.where(x<=x0)[0][-1]
    return i

def interp1d(x0,f,x):
    i = get_index(x0,x)
    dx = x[i+1] - x[i]
    t = (x0 - x[i]) / dx  
    # Interpolate between f[i] and f[i+1]
    s = f[i] * l1(t) + f[i+1] * l2(t)
    return s

def interp2d(x0,y0,f,x,y):
    i = get_index(x0,x)
    j = get_index(y0,y)
    dx = x[i+1] - x[i]
    dy = y[j+1] - y[j]
    tx = (x0 - x[i]) / dx
    ty = (y0 - y[j]) / dy
    # Interpolate along x for both y[j] and y[j+1]
    s_j = f[i, j] * l1(tx) + f[i+1, j] * l2(tx)
    s_j1 = f[i, j+1] * l1(tx) + f[i+1, j+1] * l2(tx)
    # Interpolate along y
    s = s_j * l1(ty) + s_j1 * l2(ty)
    return s

def interp3d(x0,y0,z0,f,x,y,z):
    i = get_index(x0,x)
    j = get_index(y0,y)
    k = get_index(z0,z)
    dx = x[i+1] - x[i]
    dy = y[j+1] - y[j]
    dz = z[k+1] - z[k]
    tx = (x0 - x[i]) / dx
    ty = (y0 - y[j]) / dy
    tz = (z0 - z[k]) / dz
    # Interpolate along x for all combinations of y and z
    c00 = f[i, j, k] * l1(tx) + f[i+1, j, k] * l2(tx)
    c01 = f[i, j, k+1] * l1(tx) + f[i+1, j, k+1] * l2(tx)
    c10 = f[i, j+1, k] * l1(tx) + f[i+1, j+1, k] * l2(tx)
    c11 = f[i, j+1, k+1] * l1(tx) + f[i+1, j+1, k+1] * l2(tx)
    # Interpolate along y for each z layer
    c0 = c00 * l1(ty) + c10 * l2(ty)
    c1 = c01 * l1(ty) + c11 * l2(ty)
    # Interpolate along z
    s = c0 * l1(tz) + c1 * l2(tz)
    return s

def test1d():
    x = np.linspace(0,np.pi,10)
    y = np.sin(x)
    xx = np.linspace(x[0],x[-1],100)
    yy = np.zeros_like(xx)
    for i in range(len(xx)):
        yy[i] = interp1d(xx[i],y,x)
    
    plt.plot(xx,yy)
    plt.plot(x,y,'o')

def test2d():
    fig=plt.figure()
    ax=fig.add_subplot(121)
    xo=np.linspace(0.0,3.0,11)
    yo=np.linspace(0.0,3.0,12)
    X,Y = np.meshgrid(xo,yo,indexing="ij")
    Zorig = (X+Y)*np.exp(-1.0*(X*X+Y*Y)**2)
    ax.pcolor(X,Y,Zorig)
    ax.set_title('original')

    ax2=fig.add_subplot(122)
    x = np.linspace(0.0,3.0,51)
    y = np.linspace(0.0,3.0,51)
    X,Y = np.meshgrid(x,y,indexing="ij")
    Z = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i,j] = interp2d(x[i],y[j],Zorig,xo,yo)
    ax2.pcolor(X,Y,Z)
    ax2.set_title('interpolated')

def test3d():
    x=np.linspace(0.0,3.0,10)
    y=np.linspace(0.0,3.0,11)
    z=np.linspace(0.0,3.0,10)
    X,Y,Z = np.meshgrid(x,y,z,indexing="ij")
    F = (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    X,Y = np.meshgrid(x,y,indexing="ij")
    fig3d=plt.figure()
    ax=fig3d.add_subplot(121)
    ax.pcolor(X,Y,F[...,int(len(z)/2)])
    ax.set_title('original (from 3D data)')

    ax2=fig3d.add_subplot(122)
    xi=np.linspace(0.0,3.0,50)
    yi=np.linspace(0.0,3.0,50)
    zi=np.linspace(0.0,3.0,50)
    X,Y = np.meshgrid(xi,yi,indexing="ij")
    Fi=np.zeros((len(xi),len(yi),len(zi)))
    for i in range(len(xi)):
        for j in range(len(yi)):
            for k in range(len(zi)):
                Fi[i,j,k] = interp3d(xi[i],yi[j],zi[k],F,x,y,z)
    ax2.pcolor(X,Y,Fi[...,int(len(z)/2)])
    ax2.set_title('linear interp. (from 3D data)')
    

def main():
    
    try:
        test1d()
    except:
        pass

    try:
        test2d()
    except:
        pass

    try:
        test3d()
    except:
        pass
    
    plt.show()

if __name__=="__main__":
    main()
