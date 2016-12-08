# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:18:42 2016

@author: robin
"""
# Arrary and stuff
import numpy as np
import plot2D as plt
import poisson_solver as ps

np.set_printoptions(suppress=True, precision=3)

# Loop over different number of equally spaced subintervals
N_values = [8, 16, 32, 64]
k = 2
iteration = 0
Err = np.zeros((len(N_values), 1))
for N in [8, 16, 32, 64]:
    #numerically calculate solution
    x,y,F_grid,U_grid = ps.fdm_poisson_2d_dense(N, k)
    plt.plot2D(x, y, U_grid, 'test')
    
    #Compare solution to analytical solution and calculate error
    U_real = np.sin(2*np.pi*k*x)*np.sin(2*np.pi*k*y)
    Err[iteration] = np.max(np.absolute(U_grid-U_real))
    iteration += 1
    print(np.max(np.abs(U_grid)))
    print("\n\n")

plt.plot2D(x, y, U_real, 'u real')
EOC = np.log(Err[:-1]/Err[1:])/np.log(2)
print(EOC)
    