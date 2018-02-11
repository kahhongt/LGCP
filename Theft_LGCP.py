import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn
import scipy
import scipy.special as scispec
import scipy.optimize as scopt

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""Clean up data set from Chicago Crime Data - have to import directly from csv"""

"""Theft Crime in Chicago"""
theft_df = pd.read_csv('Theft_Clean.csv')  # generates dataframe from csv - zika data

year2017 = theft_df['Year'] == 2017  # set boolean variable for after 2008
half_year_point = theft_df['Date']
theft_df = theft_df[year2017]
theft_date = theft_df.values[:, 0]
theft_x = theft_df.values[:, 2]
theft_y = theft_df.values[:, 3]

theft_fig = plt.figure()
theft_fig.canvas.set_window_title('Chicago Theft Locations')
theft_spread = theft_fig.add_subplot(111)
theft_spread.scatter(theft_x, theft_y, color='black', marker='.', s=0.0001)

"""Create Histogram"""


