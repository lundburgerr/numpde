# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:18:42 2016

@author: robin
"""
# Arrary and stuff 
import numpy as np
# Linear algebra solvers from scipy
import scipy.linalg as la
# Basic plotting routines from the matplotlib library 
import matplotlib.pyplot as plt


#define constant sigma
sigma = -1/(2*np.pi);
C = sigma + 1/(2*np.pi)

# Loop over different number of equally spaced subintervals
for N in [4, 8, 16, 32, 64]:
    # Mesh size
    h = 1/N #Important! In Python 2 you needed to write 1.0 to prevent integer divsion
    # Define N+1 grid points via linspace which is part of numpy now aliased as np 
    x = np.linspace(0,1,N+1)
    #print(x)
    
    # Define a (full) matrix filled with 0s.
    A = np.zeros((N, N))
    
    # Define tridiagonal part of A by for rows 1 to N-1
    for i in range(1, N-1):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1
        
    # Neumann condition
    A[0,0] = 1
    A[0,1] = -1
    
    # Right boundary
    A[N-1,N-2] = 1
    A[N-1,N-1] = -2
    
    #print(A)
    
    #Fill F matrix containing values from f(x) = sin(2*pi*x)
    F = np.sin(2*np.pi*x)*(-h**2)
    F = F[1:-1]
    F = np.append([sigma*h], F)
    
    #Solve system
    U = la.solve(A, F.T)
    U = np.append(U, [0])

    #Plot solution
    plt.plot(x, U)
    plt.hold('on')

plt.title("The solution to -u''=f for different N")
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend(["N=4", "N=8", "N=16", "N=32", "N=64"])
plt.show()


# Plot N=64 against theoretical solution
U_real = np.sin(2*np.pi*x)/(2*np.pi)**2 - C*x + C
plt.hold('off')
plt.plot(x, U, 'x-r')
plt.hold('on')
plt.plot(x, U_real)

plt.title("The solution to -u''=f for different N=64 and analytical solution")
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend(["N=64", "an. sol."])
plt.show()
