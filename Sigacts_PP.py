import pandas as pd
import math
import matplotlib
import numpy as np
import time
import functions as fn
import scipy
import scipy.special as scispec
import scipy.optimize as scopt

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Test extracting values from csv
"""Extract Data from csv"""  # Arbitrary Point Process Data
A = np.genfromtxt('SIGACTS_KH.csv', delimiter=',')  # Extract from csv using numpy
df = pd.read_csv('SIGACTS_KH.csv')  # Generates a DataFrame from csv - coal data

# Creating Boolean variables to indicate direct or indirect fire
direct_fire = df['category'] == "Direct Fire"
indirect_fire = df['category'] == "Indirect Fire"

# Taking in those boolean variables to obtain the arrays - remember to add the values
x_direct = df[direct_fire].values[:, 8]
y_direct = df[direct_fire].values[:, 9]
x_indirect = df[indirect_fire].values[:, 8]
y_indirect = df[indirect_fire].values[:, 9]

fig = plt.figure()
fig.canvas.set_window_title('Direct and Indirect Fire')  # Changing the title of the figure
incidence = fig.add_subplot(111)
direct_incidence = incidence.scatter(x_direct, y_direct, color='red', marker='.', s=0.1)
indirect_incidence = incidence.scatter(x_indirect, y_indirect, color='blue', marker='.', s=0.1)
incidence.legend([direct_incidence, indirect_incidence], ['Direct Fire', "Indirect Fire"])

plt.show()




