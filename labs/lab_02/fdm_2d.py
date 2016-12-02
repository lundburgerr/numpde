#%%
import numpy as np

import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve import spsolve

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot2D(X, Y, Z, title=""):
    # Define a new figure with given size an
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z,             
                           rstride=1, cstride=1, # Sampling rates for the x and y input data
                           cmap=cm.viridis)      # Use the new fancy colormap viridis
    
    # Set initial view angle
    ax.view_init(30, 225)
    
    # Set labels and show figure
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    plt.show()

def fdm_poisson_2d_dense(N, k):
    '''A simple finite difference solver in 2d using a full matrix representation.'''

    # 1) Compute right hand side 

    # Define a (sparse) grid representation
    x,y = np.ogrid[0:1:(N+1)*1j, 0:1:(N+1)*1j]

    # Evaluate f on the grid
    F_grid = np.sin(2*np.pi*k*x)*np.sin(2*np.pi*k*y)

    # Plot f if you want
    plot2D(x, y, F_grid, "f")
    
    # Define rhs b by flattening out F
    F = F_grid.ravel()
    
    # 2) Create Matrix entries for unknowns asscociated with inner grid points. 

    # Define map which maps a two-dimensional grid index to index of the unknown.
    # We assume a row-wise numbering. 
    def m(i,j):
        return i*(N+1) + j
    
    # Total number of unknowns is M = (N+1)*(N+1)
    M = (N+1)**2

    # Allocate a (full) MxM matrix filled with zeros
    A = np.zeros((M, M))

    # Meshsize h
    h = 1/N
    hh = h*h

    # Compute matrix A entries resulting by iterating
    # over rows and then colums of the grid indices.
    for i in range(1,N):
        for j in range(1,N):
            # Define row index related to unknown U_m(i,j)
            ri = m(i,j)
            A[ri,m(i,j)] = 4/hh   # U_ij
            A[ri,m(i-1,j)] = -1/hh  # U_{i-1,j}
            A[ri,m(i+1,j)] = -1/hh  # U_{i+1,j}
            A[ri,m(i,j-1)] = -1/hh  # U_{i,j-1}
            A[ri,m(i,j+1)] = -1/hh  # U_{i,j+1}

    # 3) Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    for i in [0, N]:
        for j in range(0,N+1):
            # Define row index related to unknown U_m(i,j)
            ri = m(i,j)
            A[ri,ri] = 1 # U_ij
            F[ri] = 0    # b_{i-1,j}

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for j in [0, N]:
        # Note we set corner points twice, change to range(1,N) to avoid 
        for i in range(0,N+1): 
            ri = m(i,j)
            A[ri,ri] = 1 # U_ij
            F[ri] = 0    # b_{i-1,j}

    # Uncomment to plot sparsity pattern of A
    plt.figure()    
    plt.spy(A, marker="+", markersize="5")
    
    # 4) Solve linear systems
    # Solve linear algebra system and return solution
    U = la.solve(A, F)
    U_grid = U.reshape((N+1,N+1))

    return (x,y,U_grid)

def fdm_poisson_2d_sparse(N, k):
    '''A simple finite difference solver in 2d using a sparse matrix representation.'''

    # 1) Compute right hand side 

    # Define a (sparse) grid representation
    x,y = np.ogrid[0:1:(N+1)*1j, 0:1:(N+1)*1j]

    # Evaluate f on the grid
    F_grid = np.sin(2*np.pi*k*x)*np.sin(2*np.pi*k*y)

    # Define rhs b by flattening out F
    F = F_grid.ravel()
    
    # 2) Create Matrix entries for unknowns asscociated with inner grid points. 

    # We assume a row-wise numbering. 
    def m(i,j):
        return i*(N+1) + j
    
    # Total number of unknowns is M = (N+1)*(N+1)
    M = (N+1)**2

    # Allocate a (sparse) MxM matrix filled with zeros
    A = sp.dok_matrix((M, M))

    # Meshsize h
    h = 1/N
    hh = h*h

    # Define map which maps a two-dimensional grid index to index of the unkwon.
    # Compute matrix A entries resulting by iterating
    # over rows and then colums of the grid indices.
    for i in range(1,N):
        for j in range(1,N):
            # Define row index related to unknown U_m(i,j)
            ri = m(i,j)
            A[ri,m(i,j)] = 4/hh   # U_ij
            A[ri,m(i-1,j)] = -1/hh  # U_{i-1,j}
            A[ri,m(i+1,j)] = -1/hh  # U_{i+1,j}
            A[ri,m(i,j-1)] = -1/hh  # U_{i,j-1}
            A[ri,m(i,j+1)] = -1/hh  # U_{i,j+1}

    # 3) Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    for i in [0, N]:
        for j in range(0,N+1):
            # Define row index related to unknown U_m(i,j)
            ri = m(i,j)
            A[ri,ri] = 1 # U_ij
            F[ri] = 0    # b_{i-1,j}

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for j in [0, N]:
        # Note we set corner points twice, change to range(1,N) to avoid 
        for i in range(0,N+1): 
            ri = m(i,j)
            A[ri,ri] = 1 # U_ij
            F[ri] = 0    # b_{i-1,j}

    # Now convert A to format which is more efficient for solving
    A_csr = A.tocsr()
    
    # 4) Solve linear systems
    # Solve linear algebra system and return solution
    U = spsolve(A_csr, F)
    U_grid = U.reshape((N+1,N+1))

    return (x, y, U_grid)


if __name__ == "__main__":

    k = 2
    for N in [8, 16, 32, 64]:
        print("Computing U with dense matrix representation ...")
        x,y, U_grid = fdm_poisson_2d_dense(N, k)
        print("done!")
        plot2D(x, y, U_grid, "Discrete solution U")
        
        print("Computing U with sparse matrix representation ...")
        x,y, U_grid = fdm_poisson_2d_sparse(N, k)
        print("done!")
        plot2D(x, y, U_grid, "Discrete solution U")    