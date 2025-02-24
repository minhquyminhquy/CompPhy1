"""
Cubic hermite splines in 1d, 2d, and 3d

Intentionally unfinished :)

Related to Computational Physics 1
exercise 2 assignments at TAU.

By Ilkka Kylanpaa
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
Add basis functions p1,p2,q1,q2 here
"""

def p1(t):
    """Basis function for the left point's value (2t^3 - 3t^2 + 1)."""
    return 2 * t**3 - 3 * t**2 + 1

def p2(t):
    """Basis function for the right point's value (-2t^3 + 3t^2)."""
    return -2 * t**3 + 3 * t**2

def q1(t):
    """Basis function for the left point's derivative (t^3 - 2t^2 + t)."""
    return t**3 - 2 * t**2 + t

def q2(t):
    """Basis function for the right point's derivative (t^3 - t^2)."""
    return t**3 - t**2

def init_1d_spline(x,f,h):
    # now using complete boundary conditions
    # with forward/backward derivative
    # - natural boundary conditions commented
    a=np.zeros((len(x),))
    b=np.zeros((len(x),))
    c=np.zeros((len(x),))
    d=np.zeros((len(x),))
    fx=np.zeros((len(x),))

    # a[0]=1.0 # not needed
    b[0]=1.0

    # natural boundary conditions 
    #c[0]=0.5
    #d[0]=1.5*(f[1]-f[0])/(x[1]-x[0])

    # complete boundary conditions
    c[0]=0.0
    d[0]=(f[1]-f[0])/(x[1]-x[0])
    
    for i in range(1,len(x)-1):
        d[i]=6.0*(h[i]/h[i-1]-h[i-1]/h[i])*f[i]-6.0*h[i]/h[i-1]*f[i-1]+6.0*h[i-1]/h[i]*f[i+1]
        a[i]=2.0*h[i]
        b[i]=4.0*(h[i]+h[i-1])
        c[i]=2.0*h[i-1]        
    #end for

    
    b[-1]=1.0
    #c[-1]=1.0 # not needed

    # natural boundary conditions
    #a[-1]=0.5
    #d[-1]=1.5*(f[-1]-f[-2])/(x[-1]-x[-2])

    # complete boundary conditions
    a[-1]=0.0
    d[-1]=(f[-1]-f[-2])/(x[-1]-x[-2])
    
    # solve tridiagonal eq. A*f=d
    c[0]=c[0]/b[0]
    d[0]=d[0]/b[0]
    for i in range(1,len(x)-1):
        temp=b[i]-c[i-1]*a[i]
        c[i]=c[i]/temp
        d[i]=(d[i]-d[i-1]*a[i])/temp
    #end for
        
    fx[-1]=d[-1]
    for i in range(len(x)-2,-1,-1):
        fx[i]=d[i]-c[i]*fx[i+1]
    #end for
        
    return fx
# end function init_1d_spline


class spline:

    def __init__(self,*args,**kwargs):
        # Set up the spline's "ingredients" based on how many dimensions we have
        self.dims=kwargs['dims']
        # Simple 1D case
        if (self.dims==1):
            self.x=kwargs['x']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            # Calculate slopes at each point to make curve smooth
            self.fx=init_1d_spline(self.x,self.f,self.hx)

        # 2D case - like making a stretchy fabric over poles
        elif (self.dims==2):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
            # Make slope information for x direction, y direction, and diagonal
            self.fx=np.zeros(self.f.shape)
            self.fy=np.zeros(self.f.shape)
            self.fxy=np.zeros(self.f.shape)
             # Calculate slopes row by row and column by column
            for i in range(max([len(self.x),len(self.y)])):
                if (i<len(self.y)):
                    self.fx[:,i]=init_1d_spline(self.x,self.f[:,i],self.hx)
                if (i<len(self.x)):
                    self.fy[i,:]=init_1d_spline(self.y,self.f[i,:],self.hy)
            #end for
            for i in range(len(self.y)):
                self.fxy[:,i]=init_1d_spline(self.x,self.fy[:,i],self.hx)
            #end for

        # 3D Case: similar pattern to 2D but with extra z-axis stuff
        elif (self.dims==3):
            
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.z=kwargs['z']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
            self.hz=np.diff(self.z)
            self.fx=np.zeros(self.f.shape)
            self.fy=np.zeros(self.f.shape)
            self.fz=np.zeros(self.f.shape)
            self.fxy=np.zeros(self.f.shape)
            self.fxz=np.zeros(self.f.shape)
            self.fyz=np.zeros(self.f.shape)
            self.fxyz=np.zeros(self.f.shape)
            for i in range(max([len(self.x),len(self.y),len(self.z)])):
                for j in range(max([len(self.x),len(self.y),len(self.z)])):
                    if (i<len(self.y) and j<len(self.z)):
                        self.fx[:,i,j]=init_1d_spline(self.x,self.f[:,i,j],self.hx)
                    if (i<len(self.x) and j<len(self.z)):
                        self.fy[i,:,j]=init_1d_spline(self.y,self.f[i,:,j],self.hy)
                    if (i<len(self.x) and j<len(self.y)):
                        self.fz[i,j,:]=init_1d_spline(self.z,self.f[i,j,:],self.hz)
            #end for
            for i in range(max([len(self.x),len(self.y),len(self.z)])):
                for j in range(max([len(self.x),len(self.y),len(self.z)])):
                    if (i<len(self.y) and j<len(self.z)):
                        self.fxy[:,i,j]=init_1d_spline(self.x,self.fy[:,i,j],self.hx)
                    if (i<len(self.y) and j<len(self.z)):
                        self.fxz[:,i,j]=init_1d_spline(self.x,self.fz[:,i,j],self.hx)
                    if (i<len(self.x) and j<len(self.z)):
                        self.fyz[i,:,j]=init_1d_spline(self.y,self.fz[i,:,j],self.hy)
            #end for
            for i in range(len(self.y)):
                for j in range(len(self.z)):
                    self.fxyz[:,i,j]=init_1d_spline(self.x,self.fyz[:,i,j],self.hx)
            #end for
        else:
            print('Either dims is missing or specific dims is not available')
        #end if
            
    def eval1d(self,x):
        """Guess height between known points using bendy straw formula"""
        if np.isscalar(x):
            x=np.array([x])
        N=len(self.x)-1
        f=np.zeros((len(x),))
        ii=0
        for val in x:
            # Find which two points our value is between
            i=np.floor(np.where(self.x<=val)[0][-1]).astype(int)
            if i==N:
                f[ii]=self.f[i]
            else:
                t=(val-self.x[i])/self.hx[i]
                f[ii]=self.f[i]*p1(t)+self.f[i+1]*p2(t)+self.hx[i]*(self.fx[i]*q1(t)+self.fx[i+1]*q2(t))
            ii+=1

        return f
    #end eval1d

    # Similar to 1D but mixes info from 4 corners
    def eval2d(self,x,y):
        """Guess height on fabric between grid poles"""
        
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        f=np.zeros((len(x),len(y)))
        A=np.zeros((4,4))
        # Uses magic 4x4 number grid (A) to combine:
        # - Corner heights
        # - Left-right slopes
        # - Front-back slopes
        # - Twist slopes
        ii=0
        for valx in x:
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                u = (valx-self.x[i])/self.hx[i]
                v = (valy-self.y[j])/self.hy[j]
                pu = np.array([p1(u),p2(u),self.hx[i]*q1(u),self.hx[i]*q2(u)])
                pv = np.array([p1(v),p2(v),self.hy[j]*q1(v),self.hy[j]*q2(v)])
                A[0,:]=np.array([self.f[i,j],self.f[i,j+1],self.fy[i,j],self.fy[i,j+1]])
                A[1,:]=np.array([self.f[i+1,j],self.f[i+1,j+1],self.fy[i+1,j],self.fy[i+1,j+1]])
                A[2,:]=np.array([self.fx[i,j],self.fx[i,j+1],self.fxy[i,j],self.fxy[i,j+1]])
                A[3,:]=np.array([self.fx[i+1,j],self.fx[i+1,j+1],self.fxy[i+1,j],self.fxy[i+1,j+1]])           
                
                f[ii,jj]=np.dot(pu,np.dot(A,pv))
                jj+=1
            ii+=1
        return f
    #end eval2d

    def eval3d(self,x,y,z):
        """Guess value in 3D space between known points"""
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        if np.isscalar(z):
            z=np.array([z])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        Nz=len(self.z)-1
        f=np.zeros((len(x),len(y),len(z)))
        # Uses two 4x4 grids (A and B) to track:
        # - Corner values
        # - All possible slopes
        # - Mixes them with triple set of bendy formulas
        A=np.zeros((4,4))
        B=np.zeros((4,4))
        ii=0
        for valx in x:
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                kk=0
                for valz in z:
                    k=np.floor(np.where(self.z<=valz)[0][-1]).astype(int)
                    if (k==Nz):
                        k-=1
                    u = (valx-self.x[i])/self.hx[i]
                    v = (valy-self.y[j])/self.hy[j]
                    t = (valz-self.z[k])/self.hz[k]
                    pu = np.array([p1(u),p2(u),self.hx[i]*q1(u),self.hx[i]*q2(u)])
                    pv = np.array([p1(v),p2(v),self.hy[j]*q1(v),self.hy[j]*q2(v)])
                    pt = np.array([p1(t),p2(t),self.hz[k]*q1(t),self.hz[k]*q2(t)])
                    B[0,:]=np.array([self.f[i,j,k],self.f[i,j,k+1],self.fz[i,j,k],self.fz[i,j,k+1]])
                    B[1,:]=np.array([self.f[i+1,j,k],self.f[i+1,j,k+1],self.fz[i+1,j,k],self.fz[i+1,j,k+1]])
                    B[2,:]=np.array([self.fx[i,j,k],self.fx[i,j,k+1],self.fxz[i,j,k],self.fxz[i,j,k+1]])
                    B[3,:]=np.array([self.fx[i+1,j,k],self.fx[i+1,j,k+1],self.fxz[i+1,j,k],self.fxz[i+1,j,k+1]])
                    A[:,0]=np.dot(B,pt)
                    B[0,:]=np.array([self.f[i,j+1,k],self.f[i,j+1,k+1],self.fz[i,j+1,k],self.fz[i,j+1,k+1]])
                    B[1,:]=np.array([self.f[i+1,j+1,k],self.f[i+1,j+1,k+1],self.fz[i+1,j+1,k],self.fz[i+1,j+1,k+1]])
                    B[2,:]=np.array([self.fx[i,j+1,k],self.fx[i,j+1,k+1],self.fxz[i,j+1,k],self.fxz[i,j+1,k+1]])
                    B[3,:]=np.array([self.fx[i+1,j+1,k],self.fx[i+1,j+1,k+1],self.fxz[i+1,j+1,k],self.fxz[i+1,j+1,k+1]])
                    A[:,1]=np.dot(B,pt)

                    B[0,:]=np.array([self.fy[i,j,k],self.fy[i,j,k+1],self.fyz[i,j,k],self.fyz[i,j,k+1]])
                    B[1,:]=np.array([self.fy[i+1,j,k],self.fy[i+1,j,k+1],self.fyz[i+1,j,k],self.fyz[i+1,j,k+1]])
                    B[2,:]=np.array([self.fxy[i,j,k],self.fxy[i,j,k+1],self.fxyz[i,j,k],self.fxyz[i,j,k+1]])
                    B[3,:]=np.array([self.fxy[i+1,j,k],self.fxy[i+1,j,k+1],self.fxyz[i+1,j,k],self.fxyz[i+1,j,k+1]])
                    A[:,2]=np.dot(B,pt)
                    B[0,:]=np.array([self.fy[i,j+1,k],self.fy[i,j+1,k+1],self.fyz[i,j+1,k],self.fyz[i,j+1,k+1]])
                    B[1,:]=np.array([self.fy[i+1,j+1,k],self.fy[i+1,j+1,k+1],self.fyz[i+1,j+1,k],self.fyz[i+1,j+1,k+1]])
                    B[2,:]=np.array([self.fxy[i,j+1,k],self.fxy[i,j+1,k+1],self.fxyz[i,j+1,k],self.fxyz[i,j+1,k+1]])
                    B[3,:]=np.array([self.fxy[i+1,j+1,k],self.fxy[i+1,j+1,k+1],self.fxyz[i+1,j+1,k],self.fxyz[i+1,j+1,k+1]])
                    A[:,3]=np.dot(B,pt)
                
                    f[ii,jj,kk]=np.dot(pu,np.dot(A,pv))
                    kk+=1
                jj+=1
            ii+=1
        return f
    #end eval3d
#end class spline


    
def test1d():
    # 1d example
    x=np.linspace(0.,2.*np.pi,11)
    y=np.sin(x)
    spl1d=spline(x=x,f=y,dims=1)
    xx=np.linspace(0.,2.*np.pi,100)
    plt.figure()
    # function
    plt.plot(xx,np.sin(xx),'b',label='exact')
    plt.plot(x,y,'o',label='points')
    plt.plot(xx,spl1d.eval1d(xx),'r--',label='spline')
    plt.title('1d spline interpolation')
    plt.legend()

def test2d():
    # 2d example
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(221, projection='3d')
    ax2d2 = fig2d.add_subplot(222, projection='3d')
    ax2d3 = fig2d.add_subplot(223)
    ax2d4 = fig2d.add_subplot(224)

    x=np.linspace(-2.0,2.0,11)
    y=np.linspace(-2.0,2.0,11)
    X,Y = np.meshgrid(x,y,indexing="ij")
    Z = X*np.exp(-1.0*(X*X+Y*Y))
    ax2d.plot_wireframe(X,Y,Z)
    ax2d3.pcolor(X,Y,Z)
    ax2d.set_title('original')

    spl2d=spline(x=x,y=y,f=Z,dims=2)
    x=np.linspace(-2.0,2.0,51)
    y=np.linspace(-2.0,2.0,51)
    X,Y = np.meshgrid(x,y,indexing="ij")
    Z = spl2d.eval2d(x,y)
    
    ax2d2.plot_wireframe(X,Y,Z)
    ax2d4.pcolor(X,Y,Z)
    ax2d2.set_title('interpolated')

def test3d():
    # 3d example
    x=np.linspace(0.0,3.0,11)
    y=np.linspace(0.0,3.0,11)
    z=np.linspace(0.0,3.0,11)
    X,Y,Z = np.meshgrid(x,y,z,indexing="ij")
    F = (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    X,Y = np.meshgrid(x,y,indexing="ij")
    fig3d=plt.figure()
    ax=fig3d.add_subplot(121)
    ax.pcolor(X,Y,F[...,int(len(z)/2)])
    ax.set_title('original (from 3D data)')

    ax2=fig3d.add_subplot(122)
    spl3d=spline(x=x,y=y,z=z,f=F,dims=3)  
    x=np.linspace(0.0,3.0,51)
    y=np.linspace(0.0,3.0,51)
    z=np.linspace(0.0,3.0,51)
    X,Y = np.meshgrid(x,y,indexing="ij")
    F=spl3d.eval3d(x,y,z)
    ax2.pcolor(X,Y,F[...,int(len(z)/2)])
    ax2.set_title('spline interp. (from 3D data)')

    plt.show()
    
def main():
    plt.rcParams['pcolor.shading']='auto'
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
