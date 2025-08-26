# =================================================================#
# Mandelbrot-plot.py

# This program plots the mandelbrot set
# For a faster I/O, it uses pandas routine to load the data.
# As such the input file needs to be csv.
# Note that better I/O could be attended by storing, parsing binary files.


#.....................................................................#

# Produced for NCI Training. 
# Frederick Fung 2022
# 4527FD1D
#====================================================================#



import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
from numpy import loadtxt
import pandas as pd

def plot_mandelbrot(file):
    

    start_load = time.time()

    data_set = pd.read_csv(file, header=None, index_col=False,dtype=float)
    data_set = data_set.values[:, :-1]


    end_load = time.time()
    print("time in loading ", end_load-start_load)
    start_plot = time.time()
    plt.figure(figsize=(10, 10))
    plt.imshow(data_set.T, interpolation="nearest")
    end_plot = time.time()
    print("time in plotting ", end_plot - start_plot)
    plt.show()



plot_mandelbrot("mandelbrot_set_100.csv")
plot_mandelbrot("mandelbrot_set_1000.csv")
#plot_mandelbrot("mandelbrot_set_10000.csv")
