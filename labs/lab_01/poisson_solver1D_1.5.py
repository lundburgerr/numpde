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

# Loop over different number of equally spaced subintervals
sub_int = np.array([4, 8, 16, 32, 64])
E = np.zeros((len(sub_int),1))
iteration = 0
for N in sub_int:
    # Mesh size
    h = 1/N #Important! In Python 2 you needed to write 1.0 to prevent integer divsion
    # Define N+1 grid points via linspace which is part of numpy now aliased as np 
    x = np.linspace(0,1,N+1)
    
    # Define a (full) matrix filled with 0s.
    A = np.zeros((N-1, N-1))
    
    # Define tridiagonal part of A by for rows 1 to N-1
    for i in range(1, N-2):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1
        
    # Left boundary
    A[0,0] = -2
    A[0,1] = 1
    
    # Right boundary
    A[N-2,N-3] = 1
    A[N-2,N-2] = -2
    
    #print(A)
    
    #Fill F matrix containing values from f(x) = sin(2*pi*x)
    F = np.sin(2*np.pi*x)*(-h**2)
    F = F[1:-1]
    
    #Solve system
    U = la.solve(A, F.T)
    U = np.append([0],U)
    U = np.append(U, [0])
    U_real = np.sin(2*np.pi*x)/(2*np.pi)**2

    #Calculate error norm
    E[iteration] = max(np.absolute(U-U_real))
    
    iteration += 1

H = 1/sub_int
scale = min(E.T/H**2)
plt.loglog(H, E)
plt.hold('on')
plt.loglog(H, H**2*scale*1.2) #reference to check if Error is parallell to this

plt.title("loglog-plot of max-norm of error for numerical solution against step-size h")
#plt.ylabel('||E||')
plt.xlabel('h')
plt.legend(["||E||", "Ch^2"])
plt.show()

