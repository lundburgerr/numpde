import scipy.linalg as la
import numpy as np

def fdm_poisson_2d_dense(N, k):
    '''A simple finite difference solver in 2d using a full matrix representation.'''

    # 1) Compute right hand side     
    
    # To define the grid we could use "linspace" as in Lab 1 to define subdivisions for the $x$ and $y$ axes.
    # But to make plotting easy and to vectorize the evaluation of the right-hand side $f$, we do something more fancy.
    # We define x and y coordinates for the grid using a "sparse grid" representation using the function 'ogrid'.
    # (Read the documentation for 'ogrid'!). Unfortunately, ogrid does not include the interval endpoints by 
    # default, but according to the numpy documentation, you can achieve this by multiplying your sampling number by
    # the pure imaginary number $i = \sqrt{-1}$  which is written as "1j" in Python code.
    # So simply speaking "(N+1)*1j" reads "include the end points" while (N+1) reads "exclude the end points".

    x,y = np.ogrid[0:1:(N+1)*1j, 0:1:(N+1)*1j]
    # Print x and y to see how they look like!
    #print(x)
    #print(y)
    
    # Evaluate f on the grid. 
    F_grid = 8 * (np.pi*k)**2 * np.sin(2*np.pi*k*x) * np.sin(2*np.pi*k*y)
    F_grid_inner = F_grid[1:N, 1:N]
    
    #You can print F_grid to verify that you got a 2 dimensional array
    # print(F_grid)
    
    # You can also plot F_grid now if you want :)
    # plot2D(x, y, F_grid, "f")
    
    # Now we define our rhs b by flattening out F, making it a 1 dimensional array of length (N+1)*(N+1). 
    F = F_grid_inner.ravel()  
    
    # 2) Create Matrix entries for unknowns associated with inner grid points. 
    
    # To translate the grid based double index into a proper numbering, we define a 
    # small mapping function, assuming a row-wise numbering. 
    # Drawing a picture of the grid and numbering the grid points in a row wise manner
    # helps to understand this mapping!
    def m(i,j):
        return i*(N-1) + j
    
    # Total number of unknowns is M = (N+1)*(N+1)
    M = (N-1)**2
    
    # Allocate a (full!) MxM matrix filled with zeros
    A = np.zeros((M,M))
    
    # Meshsize h
    h = 1/N
    hh = h*h
    
    # Compute matrix A entries by iterating over the *inner* grid points first.
    for i in range(0,N-1):      # i is the row number for the grid point
        for j in range(0,N-1):  # j is the column number for the grid point
            # Compute the index of the unknown at grid point (i,j). 
            # This is also the index of the row in matrix A we want to fill. 
            ri = m(i,j)       
            A[ri,m(i,j)] = 4/hh         # U_ij
            if i-1 >= 0:
                A[ri,m(i-1,j)] = -1/hh      # U_{i-1,j}
            if i+1 < N-1:
                A[ri,m(i+1,j)] = -1/hh      # U_{i+1,j}
            if j-1 >= 0:
                A[ri,m(i,j-1)] = -1/hh      # U_{i,j-1}
            if j+1 < N-1:
                A[ri,m(i,j+1)] = -1/hh      # U_{i,j+1}
    
    
    # 4) Solve linear systems
    # Solve linear algebra system 
    U = la.solve(A, F.T)
    
    # Reshape the flat solution vector U to make it a grid function
    U_grid_inner = U.reshape((N-1,N-1))
    U_grid = np.zeros((N+1,N+1))
    U_grid[1:N,1:N] = U_grid_inner
    
    # Return solution and x and y grid points for easy plotting
    return (x,y,F_grid,U_grid)