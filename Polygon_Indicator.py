import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn
import time
import scipy.special as scispec
import scipy.optimize as scopt
import matplotlib.path as mpath

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""This Script tests out if a point falls inside a polygon, which will mainly be used for quadrat selection"""

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

# Extract data for Brazil and make sure to convert data type to float64
aedes_brazil = aedes_df[brazil]  # Extracting Brazil Data
aedes_brazil_2014 = aedes_df[brazil & year_2014]
aedes_brazil_2013 = aedes_df[brazil & year_2013]
aedes_brazil_2012 = aedes_df[brazil & year_2012]
aedes_brazil_2013_2014 = aedes_brazil_2013 & aedes_brazil_2014
x_2014 = aedes_brazil_2014.values[:, 5].astype('float64')
y_2014 = aedes_brazil_2014.values[:, 4].astype('float64')
x_2013 = aedes_brazil_2013.values[:, 5].astype('float64')
y_2013 = aedes_brazil_2013.values[:, 4].astype('float64')
x_2013_2014 = aedes_brazil_2013_2014.values[:, 5].astype('float64')
y_2013_2014 = aedes_brazil_2013_2014.values[:, 4].astype('float64')
# ------------------------------------------ End of Data Collection

# ------------------------------------------ Start of Scatter Point Set
# Define Scatter Point Boundary
x_upper_box = -35
x_lower_box = -65
y_upper_box = 0
y_lower_box = -30

# ChangeParam - select the year to be used
year = '2013'
if year == '2013':
    x = x_2013
    y = y_2013
elif year == '2014':
    x = x_2014
    y = y_2014
else:  # taking all years instead
    x = x_2013_2014
    y = y_2013_2014

# Define Boolean Variable for Scatter Points Selection
x_range_box = (x > x_lower_box) & (x < x_upper_box)
y_range_box = (y > y_lower_box) & (y < y_upper_box)

# Obtain the coordinates of points within the box, from a particular year
x_points = x[x_range_box & y_range_box]
y_points = y[x_range_box & y_range_box]

# ------------------------------------------ End of Scatter Point Set

# ------------------------------------------ Start of Regression Window Selection before Transformation
# Select regression window boundaries
# ChangeParam
center = (-50, -15)  # Create tuple for the center
radius = 8

# ChangeParam
point_select = 'circle'  # This is for selecting the regression window

if point_select == 'all':  # We bin everything that is in the box
    x_upper = x_upper_box
    x_lower = x_lower_box
    y_upper = y_upper_box
    y_lower = y_lower_box
elif point_select == 'manual':  # Check with max and min values above first
    x_upper = -43
    x_lower = -63
    y_upper = -2
    y_lower = -22
elif point_select == 'circle':  # Not really necessary
    x_upper = center[0] + radius
    x_lower = center[0] - radius
    y_upper = center[1] + radius
    y_lower = center[1] - radius
else:
    x_upper = max(x_points)
    x_lower = min(x_points)
    y_upper = max(y_points)
    y_lower = min(y_points)

# Create Boolean Variables
x_box = (x_points > x_lower) & (x_points < x_upper)
y_box = (y_points > y_lower) & (y_points < y_upper)

# Perform scatter point selection
x_within_box = x_points[x_box & y_box]
y_within_box = y_points[x_box & y_box]

# ------------------------------------------ End of Regression Window Selection before Transformation

# ------------------------------------------ Start of Performing Transformation

# Define the Center and Radius of the Square
# Note that the transformation of the scatter points will be about the center
xy_within_box = np.vstack((x_within_box, y_within_box))  # Create the sample points to be rotated

# Provide the optimal transformation matrix variables tabulated beforehand
# ChangeParam  - do we perform the desired transformation?
transform = 'yes'

if transform == 'yes':
    transform_matrix_array = np.array([0.30117594, 0.92893405, 0.65028918, -0.2277159])
elif transform == 'special':
    transform_matrix_array = np.array([-0.30117594, 0.92893405, -0.65028918, -0.2277159])
elif transform == 'line':
    transform_matrix_array = np.array([1, 1, 1, 1])
else:
    transform_matrix_array = np.array([1, 0, 0, 1])

frob_norm = fn.frob_norm(transform_matrix_array)

print('The optimal Transformation Matrix Variables are', transform_matrix_array)
print('The optimal Frobenius Norm is', frob_norm)

# ChangeParam - Conduct the transformation about the center of the regression window
transformed_xy_within_box = fn.transform_array(transform_matrix_array, xy_within_box, center)
x_points_trans = transformed_xy_within_box[0]
y_points_trans = transformed_xy_within_box[1]

# Obtain the maximum range in x and y in the transformed space - to define the regression window
# This is to maximise the number of selected quadrats
x_min = min(x_points_trans)
x_max = max(x_points_trans)
y_min = min(y_points_trans)
y_max = max(y_points_trans)


# First create a regression window with 20 x 20 quadrats before selecting the relevant quadrats
# ChangeParam
quads_on_side = 30  # define the number of quads along each dimension for the large regression window
k_mesh, y_edges, x_edges = np.histogram2d(y_points_trans, x_points_trans, bins=quads_on_side,
                                          range=[[y_min, y_max], [x_min, x_max]])
x_mesh_plot, y_mesh_plot = np.meshgrid(x_edges, y_edges)  # creating mesh-grid for use
x_mesh = x_mesh_plot[:-1, :-1]  # Removing extra rows and columns due to edges
y_mesh = y_mesh_plot[:-1, :-1]
x_quad = fn.row_create(x_mesh)  # Creating the rows from the mesh
y_quad = fn.row_create(y_mesh)
xy_quad = np.vstack((x_quad, y_quad))

k_quad = fn.row_create(k_mesh)

print('k_quad is', k_quad)
print('xy_quad is', xy_quad)

# Align the quadrats to the centers
# Realign the quad coordinates to the centers - shift centers by half a quad length on either dimension
quad_length_x = (x_upper - x_lower) / quads_on_side
quad_length_y = (y_upper - y_lower) / quads_on_side
x_quad = x_quad + (0.5 * quad_length_x)
y_quad = y_quad + (0.5 * quad_length_y)


# --------------------- Conduct binning into transformed space - the x and y quad lengths will be different
# Obtain the vertices after the transformation from the initial regression window
vertices = np.array([[x_lower, x_lower, x_upper, x_upper], [y_lower, y_upper, y_upper, y_lower]])
print('The Original Vertices are', vertices)

# Transform the vertices using the same transformation matrix
transformed_vertices = fn.transform_array(transform_matrix_array, vertices, center)
print('The Transformed Vertices are', transformed_vertices)

# Test out for points
polygon = mpath.Path(np.transpose(transformed_vertices))
polygon_indicator = polygon.contains_points(np.transpose(xy_quad), transform=None, radius=0.1)

x_quad_polygon = x_quad[polygon_indicator]
y_quad_polygon = y_quad[polygon_indicator]
xy_quad_polygon = np.vstack((x_quad_polygon, y_quad_polygon))
k_quad_polygon = k_quad[polygon_indicator]

print('The number of selected quadrats is', xy_quad_polygon.shape[1])

# Set up circle quad indicator to show which quads are within the Circular Regression Window
indicator_array = np.ones_like(k_quad) * polygon_indicator
indicator_mesh = indicator_array.reshape(k_mesh.shape)

# ------------------------------------------ End of Histogram Generation from Box
# Because this is now a rectangular box, I do not need to do realignment of quad centers

# Plot the Transformed Vertices and Original Vertices
plt.figure()
plt.scatter(vertices[0], vertices[1], color='black')
plt.scatter(transformed_vertices[0], transformed_vertices[1], color='darkorange')
plt.scatter(x_quad, y_quad, color='blue')
plt.scatter(x_quad_polygon, y_quad_polygon, color='red')

# Transform all scatter points in Brazil
brazil_scatter = np.vstack((x_points, y_points))
transformed_brazil_scatter = fn.transform_array(transform_matrix_array, brazil_scatter, center)
transformed_brazil_scatter_x = transformed_brazil_scatter[0]
transformed_brazil_scatter_y = transformed_brazil_scatter[1]

# Plot Indicator using the selected quadrats and all Brazil Points
histo_fig = plt.figure()
histo = histo_fig.add_subplot(111)
cmap = matplotlib.colors.ListedColormap(['white', 'orange'])
histo.pcolor(x_mesh_plot, y_mesh_plot, indicator_mesh, cmap=cmap, color='#ffffff')
histo.scatter(transformed_brazil_scatter_x, transformed_brazil_scatter_y, marker='.', color='black', s=0.3)
histo.set_title('Polygon Regression Window')
histo.set_xlabel('UTM Horizontal Coordinate')
histo.set_ylabel('UTM Vertical Coordinate')

# Plot Indicator using the selected quadrats and only regression points
indicator_fig = plt.figure()
indicator = indicator_fig.add_subplot(111)
cmap = matplotlib.colors.ListedColormap(['lightgrey', 'orange'])
indicator.pcolor(x_mesh_plot, y_mesh_plot, indicator_mesh, cmap=cmap, color='#ffffff')
indicator.scatter(x_points_trans, y_points_trans, marker='.', color='black', s=1.0)
rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='grey')
indicator.add_patch(rect)
indicator.set_title('Quadrilateral Regression Window')
indicator.set_xlabel('UTM Horizontal Coordinate')
indicator.set_ylabel('UTM Vertical Coordinate')

plt.show()