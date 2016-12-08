# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:18:42 2016

@author: robin
"""
# Arrary and stuff
import numpy as np
import poisson_solver as ps
import time

np.set_printoptions(suppress=True, precision=3)

# Loop over different number of equally spaced subintervals
N_values = [8, 16, 32, 64, 128]
k = 2
iteration = 0
Err = np.zeros((len(N_values), 1))
times_dense = np.zeros((len(N_values),1))
times_sparse = np.zeros((len(N_values),1))
for N in N_values:
    #numerically calculate solution
    t0 = time.time()
    ps.fdm_poisson_2d_dense(N, k)
    times_dense[iteration] = time.time()-t0

    t0 = time.time()
    ps.fdm_poisson_2d_sparse(N, k)
    times_sparse[iteration] = time.time()-t0
    
    iteration += 1


print(times_dense)
print(times_sparse)