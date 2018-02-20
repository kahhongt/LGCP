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


# ------------------------------------------Start of Data Collection

# Aedes Occurrences in Brazil
aedes_df = pd.read_csv('Aedes_PP_Data.csv')  # generates dataframe from csv - zika data

# Setting boolean variables required for the data
brazil = aedes_df['COUNTRY'] == "Brazil"
taiwan = aedes_df['COUNTRY'] == "Taiwan"
aegyp = aedes_df['VECTOR'] == "Aedes aegypti"
albop = aedes_df['VECTOR'] == "Aedes albopictus"
year_2014 = aedes_df['YEAR'] == "2014"
year_2013 = aedes_df['YEAR'] == "2013"
year_2012 = aedes_df['YEAR'] == "2012"
year_2015 = aedes_df['YEAR'] == "2015"

# Extract data for Brazil and make sure to convert data type to float64
aedes_brazil = aedes_df[brazil]  # Extracting Brazil Data
aedes_brazil_2014 = aedes_df[brazil & year_2014]
aedes_brazil_2013 = aedes_df[brazil & year_2013]
aedes_brazil_2012 = aedes_df[brazil & year_2012]
aedes_brazil_2013_2014 = aedes_brazil_2013 & aedes_brazil_2014

# World Data from 2012 to 2014
aedes_world = aedes_df[year_2013 & year_2014]

x_2014 = aedes_brazil_2014.values[:, 5].astype('float64')
y_2014 = aedes_brazil_2014.values[:, 4].astype('float64')
x_2013 = aedes_brazil_2013.values[:, 5].astype('float64')
y_2013 = aedes_brazil_2013.values[:, 4].astype('float64')
x_2013_2014 = aedes_brazil_2013_2014.values[:, 5].astype('float64')
y_2013_2014 = aedes_brazil_2013_2014.values[:, 4].astype('float64')


# World Coordinates from 2012 to 2014
world_x = aedes_df.values[:, 5].astype('float64')
world_y = aedes_df.values[:, 4].astype('float64')

print(aedes_df[year_2015 & brazil])

# ------------------------------------------End of Data Collection

# Define Regression Space by specifying intervals and creating boolean variables for filter
# Note this is for 2014
maximum_x = -32.43
minimum_x = -72.79
maximum_y = 4.72
minimum_y = -32.21

# To allow for selection of range for regression, ignoring the presence of all other data points
x_upper = -43
x_lower = -63
y_upper = -2
y_lower = -22
x_window = (x_2013 > x_lower) & (x_2013 < x_upper)
y_window = (y_2013 > y_lower) & (y_2013 < y_upper)
x_within_window = x_2013[x_window & y_window]
y_within_window = y_2013[x_window & y_window]

print(x_within_window.shape)
print(y_within_window.shape)

# First conduct a regression on the 2014 data set
quads_on_side = 40  # define the number of quads along each dimension
# histo, x_edges, y_edges = np.histogram2d(theft_x, theft_y, bins=quads_on_side)  # create histogram
histo, y_edges, x_edges = np.histogram2d(y_within_window, x_within_window, bins=quads_on_side)
x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)  # creating mesh-grid for use
# x_mesh = x_mesh[:-1, :-1]  # Removing extra rows and columns due to edges
# y_mesh = y_mesh[:-1, :-1]
x_quad_all = fn.row_create(x_mesh)  # Creating the rows from the mesh
y_quad_all = fn.row_create(y_mesh)

# *** Centralising the coordinates to be at the centre of the quads
# Note that the quads will not be of equal length, depending on the data set
quad_length_x = (x_quad_all[-1] - x_quad_all[0]) / quads_on_side
quad_length_y = (y_quad_all[-1] - y_quad_all[0]) / quads_on_side
# x_quad_all = x_quad_all + 0.5 * quad_length_x
# y_quad_all = y_quad_all + 0.5 * quad_length_y
xy_quad_all = np.vstack((x_quad_all, y_quad_all))  # stacking the x and y coordinates vertically together
k_quad_all = fn.row_create(histo)  # histogram array

# For graphical plotting
x_mesh_centralise = x_quad_all.reshape(x_mesh.shape)
y_mesh_centralise = y_quad_all.reshape(y_mesh.shape)

# ------------------------------------------Start of Plotting Process of Point Patterns, Histogram and Posterior Mean

world_fig = plt.figure()
world_scatter = world_fig.add_subplot(111)
pp_world = world_scatter.scatter(world_x, world_y, marker='.', color='darkred', s=0.1)
world_scatter.set_title('Global Aedes Occurrences from 2013 to 2014')
world_scatter.set_xlabel('UTM Horizontal Coordinate')
world_scatter.set_ylabel('UTM Vertical Coordinate')


brazil_2014 = plt.figure()
brazil_scatter_2014 = brazil_2014.add_subplot(111)
pp_brazil_2014 = brazil_scatter_2014.scatter(x_2014, y_2014, marker='.', color='darkred', s=0.1)
brazil_scatter_2014.set_title('Aedes Occurrences in Brazil in 2014')
brazil_scatter_2014.set_xlabel('UTM Horizontal Coordinate')
brazil_scatter_2014.set_ylabel('UTM Vertical Coordinate')
brazil_scatter_2014.set_xlim(-75, -30)
brazil_scatter_2014.set_ylim(-35, 5)


brazil_2013 = plt.figure()
brazil_scatter_2013 = brazil_2013.add_subplot(111)
pp_brazil_2013 = brazil_scatter_2013.scatter(x_2013, y_2013, marker='.', color='darkred', s=0.1)
brazil_scatter_2013.set_title('Aedes Occurrences in Brazil in 2013')
brazil_scatter_2013.set_xlabel('UTM Horizontal Coordinate')
brazil_scatter_2013.set_ylabel('UTM Vertical Coordinate')
brazil_scatter_2013.set_xlim(-75, -30)
brazil_scatter_2013.set_ylim(-35, 5)

brazil_2013_window = plt.figure()
brazil_2013_w = brazil_2013_window.add_subplot(111)
pp_2013 = brazil_2013_w.scatter(x_within_window, y_within_window, marker='o', color='black', s=0.3)
# plt.legend([pp_2014, pp_2013], ["2014", "2013"])
brazil_2013_w.set_title('Brazil Aedes 2013 Window Scatter')
brazil_2013_w.set_xlabel('UTM Horizontal Coordinate')
brazil_2013_w.set_ylabel('UTM Vertical Coordinate')
# brazil_2013_w.set_xlim(x_lower, x_upper)
# brazil_2013_w.set_ylim(y_lower, y_upper)

# Data Histogram
brazil_histogram = plt.figure()
brazil_h_2013 = brazil_histogram.add_subplot(111)
brazil_h_2013.pcolor(x_mesh_centralise, y_mesh_centralise, histo, cmap='YlOrBr')
brazil_pp_2013 = brazil_h_2013.scatter(x_within_window, y_within_window, marker='o', color='black', s=0.3)
brazil_h_2013.set_title('Brazil Aedes 2013 Window Histogram')
brazil_h_2013.set_xlabel('UTM Horizontal Coordinate')
brazil_h_2013.set_ylabel('UTM Vertical Coordinate')
# brazil_h_2013.set_xlim(x_lower, x_upper)
# brazil_h_2013.set_ylim(y_lower, y_upper)
# plt.legend([brazil_pp_2013], ["2013"])


plt.show()
# plt.legend([pp_2014, pp_2013], ["2014", "2013"])

