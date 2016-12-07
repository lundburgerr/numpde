# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:38:52 2016

@author: robin
"""
# We also need access to the colormaps for 3D plotting
from matplotlib import cm
# For 3D plotting
from mpl_toolkits.mplot3d import Axes3D
# Basic plotting routines from the matplotlib library 
import matplotlib.pyplot as plt

def plot2D(X, Y, Z, title=""):
    # Define a new figure with given size an
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z,             
                           rstride=1, cstride=1, # Sampling rates for the x and y input data
                           cmap=cm.viridis)      # Use the new fancy colormap viridis
    # Set initial view angle
    ax.view_init(30, 225)
    
    # Set labels and show figure
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    plt.show()