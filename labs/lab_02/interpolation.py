#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

import scipy.interpolate as ip

def plot_lagrange_basis(x_nodes):
    """ Plot the Lagrange nodal functions for given nodal points."""
    N = x_nodes.shape[0]
    nodal_values = np.ma.identity(N)

    # Create finer grid to print resulting functions
    xn = np.linspace(-2,2,100)
    fig = plt.figure()

    for i in range(N):
        L = ip.lagrange(x_nodes, nodal_values[i])
        plt.plot(x_nodes, L(x_nodes), "or")
        plt.plot(xn, L(xn), "-", label=("$L_{%d}$"%i))

    plt.legend() 
    plt.title("Lagrange basis for order %d" % (N-1))
    plt.xlabel("$x$")
    plt.ylabel("$\lambda_i(x)$")
    plt.xlim(-2.1, 2.1)
    plt.show()

def plot_lagrange_interpolant(x_nodes):
    """ Compute interpolant. (Make f an function argument)"""
    # Compute f at nodes 
    f = np.exp(-8*x_nodes**2)
    f_inter = ip.lagrange(x_nodes, f)

    # Plot f and f_inter at high resolution
    fig = plt.figure()    
    xn = np.linspace(-2,2,100)
    fn = np.exp(-8*xn**2)
    plt.plot(xn, fn, "r", label="$f = e^{-8x^2}$")

    # Plot f_inter at nodes values
    plt.plot(x_nodes, f_inter(x_nodes), "ob", label="\pi f")
    
    # Plot f_inter at high resolution
    plt.plot(xn, f_inter(xn), "--b", label="\pi f")
    plt.title("Lagrange interpolant $\pi^{%d}f$ for $f(x) = e^{-8x^2}$" % (N-1))
    plt.xlabel("$x$")
    plt.show()

def hatfun(x,i):
    e_i = np.zeros(len(x))
    print(e_i)
    e_i[i] = 1
    return ip.interp1d(x, e_i)

def interpolant(f, x, xe):
    import math
    f_nodes = f(x)
    f_xe = [ f_nodes[i]*hatfun(x, i)(xe) for i in range(0, len(x)) ]
    return math.fsum(f_xe)

if __name__ == "__main__":

    # print("Plotting interpolation results for uniformly distributed nodes.")
    # for N in [3,4, 7,8, 11,12]:
    #     x_nodes = np.linspace(-2,2,N)
    #     plot_lagrange_basis(x_nodes)
    #     plot_lagrange_interpolant(x_nodes)

    # Now repeat same experiment using Chebyshev nodes
    # print("Plotting interpolation results for Chebyshev nodes.")
    # for N in [3,4, 7,8, 11,12]:
    #     nodes = np.array(range(0, N+1))    
    #     print(nodes)
    #     x_nodes = 2*np.cos(nodes/N*np.pi)
    #     plot_lagrange_basis(x_nodes)
    #     plot_lagrange_interpolant(x_nodes)
    
    N = 4
    x = np.linspace(0, 1, N)
    print(x)
    for i in range(0, N):
        phi_i = hatfun(x,i)
        xn = np.linspace(0,1,10*N)
        plt.plot(xn, phi_i(xn))
    
    xe = 0.5
    def f_1(x):
        return np.sin(x)
    
    print(interpolant(f_1, x, xe))
    plt.plot(xn, interpolant(f_1, x, xn))
    