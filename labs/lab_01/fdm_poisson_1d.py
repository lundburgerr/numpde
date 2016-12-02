import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def fdm_poisson_1d(N, bcs, pm):
    '''The simplest 1D diffusion implementation.'''

    # Gridsize
    h = 1.0/N

    # Define grid points
    x = np.linspace(0, 1, N+1)

    # Define zero matrix A of right size and insert
    # non zero entries
    A = np.zeros((N+1, N+1))
    
    eps = pm['eps']
    # Define tridiagonal part of A
    for i in range(1, N):
        A[i, i-1] = eps + h
        A[i, i] = -2*eps - h
        A[i, i+1] = eps
    
    print("Hello Again")
    # Compute rhs for f = sin(2 pi x)
    # F = -h**2*np.sin(2*np.pi*x)
    F = -h**2*x

    # Now adapt matrix and rhs according to bc data
    # Left boundary
    bc0 = bcs[0]
    if bc0[0] == "D":
        A[0, 0] = 1
        F[0] = bc0[1]
    elif bc0[0] == "N":
        # Apply a first order difference operator
        A[0, 0] = 1
        A[0, 1] = -1
        F[0] = h*bc0[1]
        # Should we add an improved variant?

    # Right boundary
    bc1 = bcs[1]
    if bc1[0] == "D":
        A[N, N] = 1
        F[N] = bc1[1]
    elif bc1[0] == "N":
        # Apply a first order difference operator
        A[N, N] = 1
        A[N, N-1] = -1
        F[N] = h*bc1[1]

    # Solve AU = F
    # (We will introduce a sparse solver when we look at 2D problems)
    U = la.solve(A, F)

    # Compute real solution and error at grid points
    x_hr = np.linspace(0, 1, N+1)
    u = 1/(2*np.pi)**2*np.sin(2*np.pi*x_hr)

    err = np.abs(u - U)
    print("Error |U - u|")
    print(err)
    print("Error max |U - u| ")
    print(err.max())

    # Clear figure first
    plt.clf()

    # Plot solution on a high resolution grid
    # plt.plot(x_hr, u, "+-b")

    # Plot discrete solution on chosen discretization grid
    plt.plot(x, U, "x-r")

    # Show figure (for non inline plotting)
    plt.show()

if __name__ == "__main__":

    # Number of subintervals = number of mesh points - 1
    N = 8
    # Boundary data described by type and data
    bc0 = ("D", 0)
#    bc0 = ("N", -1/(2*np.pi))

#    bc1 = ("N", 1/(2*np.pi))
    bc1 = ("D", 0)

    bcs = [bc0, bc1]
    parameters = {'eps' : 1.0e-2, 'fdo' :'-'}  

    print(parameters)
    # Solve for 4 levels
    for N in [4*2**N for N in range(1,6)]:
        print("HA")
        fdm_poisson_1d(N, bcs, parameters)