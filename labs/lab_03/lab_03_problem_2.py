#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

import scipy.interpolate as ip

# Define hat functions via np.piecewise
def hatfun(xn, i, x):
    N = len(xn) - 1
    if i == 0:
        return np.piecewise(x, 
                            [(xn[0] <= x) & (x <= xn[1])], 
                            [lambda x: (xn[1] - x)/(xn[1] - xn[0]), 0])
    elif i == N:     
        return np.piecewise(x, 
                            [(xn[N-1] <= x) & (x <=  xn[N])], 
                            [lambda x: (x - xn[i-1])/(xn[N] - xn[N-1]), 0])
    else:
        return np.piecewise(x, 
                            [(xn[i-1] <= x) & (x <= xn[i]),
                             (xn[i]   <= x) & (x <=  xn[i+1])], 
                             [lambda x: (x - xn[i-1])/(xn[i] - xn[i-1]),
                              lambda x: (xn[i+1] - x)/(xn[i+1] - xn[i]), 
                              0])

# Define hat functions assuming that x is vector of
# evaluation points and do a simply looping
def hatfun_simple(xn, i, x):
    y = np.zeros(x.size)
    N = xn.size-1
    for l in range(0, y.size):
        if i == 0:
            if  xn[0] <= x[l] and x[l] <= xn[1]:
                y[l] = (xn[1] - x[l])/(xn[1] - xn[0])
        elif i == N:
            if  xn[N-1] <= x[l] and x[l] <= xn[N]:
                y[l] = (x[l] - xn[N-1])/(xn[N] - xn[N-1])
        elif xn[i-1] <= x[l] and x[l] <= xn[i+1]:
            if  x[l] <= xn[i]:
                y[l] = (x[l] - xn[i-1])/(xn[i] - xn[i-1])
            else:
                y[l] = (xn[i+1] - x[l])/(xn[i+1] - xn[i])
    return y

def interp1d(f, xn, x):
    # Evaluate f at node points
    fn = f(xn)
    # Now write array [f(x_0)*phi(x,0), f(x_1)*phi(x,1), .... ]
    # where f(x_0)*phi(x,0) is either a scalar if x is scalar, otherwise
    # it is a vector
    l = np.array([fn[i]*hatfun(xn, i, x) for i in range(0,len(xn))])
    # Now sum of axis zero to compute resulting interpolant values
    return np.sum(l, axis=0)

def interp1d_simple(f, xn, x):
    # Implementation as in matlab
    pass


if __name__ == "__main__":
    
    # Define functions to interpolate
    def f_1(x):
        return x*np.sin(3*np.pi*x)

    def f_2(x):
        return 2-10*x

    def f_3(x):
        return x*(1-x)
    
    # Get colors, labels
    colors = ["r", "b", "g"] 
    labels = ["f_1", "f_2", "f_3"] 
    functions = [f_1, f_2, f_3]

    for N in [4, 7, 10]:

        # Define nodes
        xn = np.linspace(0, 1, N)

        # Finer sampling for plotting
        x = np.linspace(0, 1, 100)

        # Plot hat functions 
        plt.figure()
        for i in range(0,N):
            phi_i = hatfun(xn, i, x)
            plt.plot(x, phi_i, label=("$\phi_{%d}$"%i))
        plt.legend()

        # Get a new figure
        plt.figure()
        for f,c,l in zip(functions, colors, labels):
            # Plot them
            plt.plot(x, f(x), "-"+c, label="$"+l+"$")
            plt.plot(xn, interp1d(f, xn, xn), "o"+c)
            plt.plot(x, interp1d(f, xn, x), "--"+c, label="$\pi "+l+"$")
            plt.legend()

    plt.show()
