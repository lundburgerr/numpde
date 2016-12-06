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
eps_values = [0.1, 0.01, 0.001]
for eps in eps_values:
    E = np.zeros((len(sub_int),1))
    iteration = 0
    for N in sub_int:
        # Mesh size
        h = 1/N #Important! In Python 2 you needed to write 1.0 to prevent integer divsion
        # Define N+1 grid points via linspace which is part of numpy now aliased as np 
        x = np.linspace(0,1,N+1)
        #print(x)
        
        #define constants
        C1 = -eps/h**2 - 1/(2*h)
        C2 = 2*eps/h**2
        C3 = -eps/h**2 + 1/(2*h)
        
        # Define a (full) matrix filled with 0s.
        A = np.zeros((N-1, N-1))
        
        # Define tridiagonal part of A by for rows 1 to N-1
        for i in range(1, N-2):
            A[i, i-1] = C1
            A[i, i] = C2
            A[i, i+1] = C3
            
        # Left Boundary
        A[0,0] = C2
        A[0,1] = C3
        
        # Right boundary
        A[N-2,N-3] = C1
        A[N-2,N-2] = C2
        
        #print(A)
        
        #Fill F matrix containing values from f(x) = sin(2*pi*x)
        F = np.ones((1,len(x)))[0]
        F = F[1:-1]
        
        #Solve system
        U = la.solve(A, F.T)
        U = np.append([0], U)
        U = np.append(U, [0])
        U_real = x-(np.exp((x-1)/eps) - np.exp(-1/eps))/(1-np.exp(-1/eps))
    
        #Calculate error norm
        E[iteration] = max(np.absolute(U-U_real))
        
        iteration += 1
        
        #Plot result
        plt.plot(x, U)
        plt.hold('on')

    #Plot analytical result
    plt.plot(x, U_real)
    plt.hold('off')
    plt.title("The solution to convection diffusion for eps={}".format(eps))
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend(["N=4", "N=8", "N=16", "N=32", "N=64", "an. sol."], bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()    
    
    #EOC = np.log(E[:-1]/E[1:])/np.log(2)
    #print(EOC)
    



