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
sub_int = [4, 8, 16, 32, 64]
E = np.zeros((len(sub_int),1))
iteration = 0
for N in sub_int:
    # Mesh size
    h = 1/N #Important! In Python 2 you needed to write 1.0 to prevent integer divsion
    # Define N+1 grid points via linspace which is part of numpy now aliased as np 
    x = np.linspace(0,1,N+1)
    #print(x)
    
    # Define a (full) matrix filled with 0s.
    A = np.zeros((N, N))
    
    # Define tridiagonal part of A by for rows 1 to N-1
    for i in range(2, N):
        A[i, i-2] = -1
        A[i, i] = 1
    
    # Left boundary
    A[0,0] = 1
    A[1,1] = 1
    
    #print(A)
    
    #Fill F matrix containing values from f(x) = sin(2*pi*x)
    F = 2*h*np.ones((1,len(x)-1))[0]
    
    #Solve system
    U = la.solve(A, F.T)
    U = np.append([0], U)
    U_real = x

    #Calculate error norm
    E[iteration] = max(np.absolute(U-U_real))
    
    iteration += 1

EOC = np.log(E[:-1]/E[1:])/np.log(2)
print(EOC)

