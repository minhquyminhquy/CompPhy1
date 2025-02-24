import numpy as np

def read_example_xsf_density(filename):
    """
    Reading in structure and density information from an xsf-file
    
    More on the format at http://www.xcrysden.org/doc/XSF.html
    """
    lattice = []
    density = []
    grid = []
    shift = []
    i = 0
    start_reading = False
    with open(filename, 'r') as f:
        for line in f:
            if "END_DATAGRID_3D" in line:
                start_reading = False
            if start_reading and i==1:
                grid=np.array(line.split(),dtype=int)
            if start_reading and i==2:
                shift.append(np.array(line.split(),dtype=float))
            if start_reading and i==3:
                lattice.append(np.array(line.split(),dtype=float))
            if start_reading and i==4:
                lattice.append(np.array(line.split(),dtype=float))
            if start_reading and i==5:
                lattice.append(np.array(line.split(),dtype=float))
            if start_reading and i>5:            
                density.extend(np.array(line.split(),dtype=float))
            if start_reading and i>0:
                i=i+1
            if "DATAGRID_3D_UNKNOWN" in line:
                start_reading = True
                i=1
    
    rho = np.zeros((grid[0],grid[1],grid[2]))
    ii = 0
    # According to the XCrysDen file format information the values 
    # inside a datagrid are specified in column-major (i.e. FORTRAN) 
    # order so let's arrange our density accordingly
    for k in range(grid[2]):
        for j in range(grid[1]):        
            for i in range(grid[0]):
                rho[i,j,k] = density[ii]
                ii += 1

    # convert density to 1/Angstrom**3 from 1/Bohr**3, 
    # since the lattice is given in Angstrom units
    a0 = 0.52917721067
    a03 = a0*a0*a0
    rho /= a03

    # let's return the transpose of the lattice,
    # since in xsf format the rows are the lattice vectors
    # not the columns... and we now want that A=[a_1,a_2,a_3]
    return rho, np.array(lattice).T, grid


def main():
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid = read_example_xsf_density(filename)

    print('Real space lattice vectors in {}'.format(filename))
    print(lattice)
    
if __name__=="__main__":
    main()



