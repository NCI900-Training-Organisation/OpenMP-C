# =================================================================#
# Laplace-plot.py

# This program plots the surface of the solution of 2D Poisson equation

#.....................................................................#

# Produced for NCI Training
# Frederick Fung 2022
# 4527FD1D
#====================================================================#

import matplotlib.pyplot as plt
import numpy as np

def LaplacePlot(file):

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

  

    data = np.loadtxt(file, dtype = float)


    size = np.shape(data)[0]

    space = 1.0/size
    node_x = np.arange(0, 1, space)
    node_y = np.arange(0, 1, space)

    node_x, node_y = np.meshgrid(node_x, node_y)

  

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})    
    surf = ax.plot_surface(node_x, node_y, data,
            cmap='viridis',linewidth=0)
    
    
    
    # Include a colour bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    


LaplacePlot("laplace-soln.dat")