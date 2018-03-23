import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn
import time
import scipy.special as scispec
import scipy.optimize as scopt

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Plot the Poisson Distribution - but this is actually not very useful, so don't worry about it
def poisson(landa, k_array):
    """
    Return pdf from the k_array
    :param landa: Poisson parameter
    :param k_array: Data Set which should be an integer
    :return: pdf over the entire k array
    """
    p = np.ones_like(k_array)
    for i in range(k_array.size):
        exp_term = np.exp(-1 * landa)
        power_term = landa ** k_array[i]
        factorial_term = 1
        for j in range(1, k_array[i]+ 1):
            factorial_term = factorial_term * j

    p[i] = exp_term * power_term / factorial_term

    return p


x = np.arange(0, 10, 1)
y = poisson(10, x)

factorial_term = 1
for j in range(1, 5):
    factorial_term = factorial_term * j

print(factorial_term)

