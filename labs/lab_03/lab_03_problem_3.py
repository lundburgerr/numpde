#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def AssembleStiffnessMatrix1D(x, a, k):
    # Number of intervals
    N = x.size-1
    # 1) Allocate and initiate matrix
    A = np.zeros((N+1,N+1))

    # 2) Compute volume contributions by iterating over intervals I_1 to I_N:
    for i in range(1,N+1):
        # Mesh  size
        h = x[i] - x[i-1]
        # Mid point
        m = (x[i-1] + x[i])/2
        # Compute local stiffness matrix
        A_loc = a(m)/h*np.array([[1, -1],[-1, 1]])
        # Write local matrix into global
        A[i-1, i-1] += A_loc[0, 0]
        A[i-1, i] += A_loc[0, 1]
        A[i, i-1] += A_loc[1, 0]
        A[i, i] += A_loc[1, 1]
    
    # 3) Compute boundary contributions
    # Neumann boundary on the left is already incorporated
    # Add Robin on the right
    A[N, N] += k(1)

    return A

def AssembleLoadVector1D(x, f, k, g_N, g_R):
    # Number of intervals
    N = x.size-1
    # 1) Allocate and initiate matrix
    b = np.zeros(N+1)
    # 2) Compute volume contributions by iterating over intervals I_1 to I_N:
    for i in range(1,N+1):
        # Mesh  size
        h = x[i] - x[i-1]
        b_loc = np.zeros(2)
        # Apply quadrature rule to int f phi_{i-1} and int f phi_{i}
        # Trapezoidal
        b_loc[0] = f(x[i-1])*h/2
        b_loc[1] = f(x[i])*h/2
        # Simpson
        # m = (x[i-1] + x[i])/2
        # b_loc[0] = (f(x[i-1]) + 4*f(m)*0.5)*h/6
        # b_loc[1] = (4*f(m)*0.5 + f(x[i]))*h/6
        
        b[i-1] += b_loc[0]
        b[i] += b_loc[1]

    # 3) Incorporate boundary values
    b[0] += g_N(0)
    b[N] += k(1)*g_R(1)*1

    return b

if __name__ == "__main__":
    
    # Define coefficients and rhs for PDE problem
    def a(x):
        return 1

    def kappa(x):
        return 1

    def g_N(x):
        return -1

    def g_R(x):
        return 3

    def f(x):
        return (2*np.pi)**2*np.cos(2*np.pi*x)

    def u_ex(x):
        return x + np.cos(2*np.pi*x)

    for N in  [4, 8, 16, 32, 64, 128, 256]:
        # Define nodes/mesh
        xn = np.linspace(0, 1, N)

        # Assemble matrix and rhs
        A = AssembleStiffnessMatrix1D(xn, a, kappa)
        b = AssembleLoadVector1D(xn, f, kappa, g_N, g_R)

        # Solve matrix system
        U = la.solve(A, b)

        # Plot solution

        plt.figure()
        x = np.linspace(0, 1, 10*N+1)
        plt.plot(x, u_ex(x), "-r")
        plt.plot(xn, U, "+-b")

    plt.show()
