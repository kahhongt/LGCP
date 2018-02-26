import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scopt

"""Methodology for Conducting Gaussian Regression for 2-D"""
# 1.
# 1. Using Matern 3/2, calculate hyper-parameters for maximum likelihood (evidence)
# 2. Use Nelder-Mead first, then use the Hypercube to obtain the globally optimal hyper-parameters
# 3. Make sure you understand the matrix manipulation for the 2-D GP
# 4. Create an arbitrary transformation matrix which can also optimised before hyper-parameters
# 5. Using csv.read, import the 2-D point process data and plot it first


def mean_func_zero(c):  # Prior mean function taken as 0 for the entire sampling range
    if np.array([c.shape]).size == 1:
        mean_c = np.zeros(1)  # Make sure this is an array
    else:
        mean_c = np.zeros(c.shape[1])
    return mean_c  # Outputs a x and y coordinates, created from the mesh grid


def squared_exp_2d(sigma_exp, length_exp, x1, x2):  # Only for 2-D
    # Define horizontal and vertical dimensions of covariance matrix c
    if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1 and x1.size == x2.shape[0]:
        rows = 1
        columns = x2.shape[1]
    elif np.array([x2.shape]).size == 1 and np.array([x1.shape]).size != 1 and x2.size == x1.shape[0]:
        rows = x1.shape[1]
        columns = 1
    elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1 and x1.size == x2.size:
        rows = 1
        columns = 1
    else:
        rows = x1.shape[1]
        columns = x2.shape[1]

    c = np.zeros((rows, columns))

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1:
                diff = x1 - x2[:, j]
            elif np.array([x1.shape]).size != 1 and np.array([x2.shape]).size == 1:
                diff = x1[:, i] - x2
            elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1:
                diff = x1 - x2
            else:
                diff = x1[:, i] - x2[:, j]

            euclidean = np.sqrt(np.matmul(diff, np.transpose(diff)))
            exp_power = np.exp(-1 * (euclidean ** 2) * (length_exp ** -2))
            c[i, j] = (sigma_exp ** 2) * exp_power

    return c  # Note that this creates the covariance matrix directly


def matern_2d(v_value, sigma_matern, length_matern, x1, x2):  # there are only two variables in the matern function
    # Define horizontal and vertical dimensions of covariance matrix c
    if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1 and x1.size == x2.shape[0]:
        rows = 1
        columns = x2.shape[1]
    elif np.array([x2.shape]).size == 1 and np.array([x1.shape]).size != 1 and x2.size == x1.shape[0]:
        rows = x1.shape[1]
        columns = 1
    elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1 and x1.size == x2.size:
        rows = 1
        columns = 1
    else:
        rows = x1.shape[1]
        columns = x2.shape[1]

    c = np.zeros((rows, columns))

    if v_value == 1/2:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1:
                    diff = x1 - x2[:, j]
                elif np.array([x1.shape]).size != 1 and np.array([x2.shape]).size == 1:
                    diff = x1[:, i] - x2
                elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1:
                    diff = x1 - x2
                else:
                    diff = x1[:, i] - x2[:, j]

                euclidean = np.sqrt(np.matmul(diff, np.transpose(diff)))
                exp_term = np.exp(-1 * euclidean * (length_matern ** -1))
                c[i, j] = (sigma_matern ** 2) * exp_term

    if v_value == 3/2:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1:
                    diff = x1 - x2[:, j]
                elif np.array([x1.shape]).size != 1 and np.array([x2.shape]).size == 1:
                    diff = x1[:, i] - x2
                elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1:
                    diff = x1 - x2
                else:
                    diff = x1[:, i] - x2[:, j]

                euclidean = np.sqrt(np.matmul(diff, np.transpose(diff)))
                coefficient_term = (1 + np.sqrt(3) * euclidean * (length_matern ** -1))
                exp_term = np.exp(-1 * np.sqrt(3) * euclidean * (length_matern ** -1))
                c[i, j] = (sigma_matern ** 2) * coefficient_term * exp_term
    return c
# Both kernel functions take in numpy arrays of one row (create a single column first)


def mu_post(xy_next, c_auto, c_cross, mismatch):  # Posterior mean
    if c_cross.shape[1] != (np.linalg.inv(c_auto)).shape[0]:
        print('First Dimension Mismatch!')
    if (np.linalg.inv(c_auto)).shape[1] != (np.transpose(mismatch)).shape[0]:
        print('Second Dimension Mismatch!')
    else:
        mean_post = mean_func_zero(xy_next) + fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(mismatch))
        return mean_post


def cov_post(c_next_auto, c_cross, c_auto):  # Posterior Covariance
    c_post = c_next_auto - fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(c_cross))
    return c_post


def log_model_evidence(param, *args):  # Param includes both sigma and l, arg is passed as a pointer
    sigma = param[0]  # param is a tuple containing 2 things, which has already been defined in the function def
    length = param[1]
    noise = param[2]  # Over here we have defined each parameter in the tuple, include noise
    xy_coordinates = args[0]  # This argument is a constant passed into the function
    histogram_data = args[1]  # Have to enter histogram data as well
    matern_nu = args[2]  # Arbitrarily chosen v value
    prior_mu = mean_func_zero(xy_coordinates)  # This creates a matrix with 2 rows
    c_auto = matern_2d(matern_nu, sigma, length, xy_coordinates, xy_coordinates)
    # c_auto = squared_exp_2d(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    c_auto_noise = c_auto + c_noise  # Overall including noise, plus include any other combination
    model_fit = - 0.5 * fn.matmulmul(histogram_data - prior_mu, np.linalg.inv(c_auto_noise),
                                     np.transpose(histogram_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_auto_noise))
    model_constant = - 0.5 * len(histogram_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant
    return -log_model_evid  # We want to maximize the log-likelihood, meaning the min of negative log-likelihood


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
# ------------------------------------------End of Data Collection

# ------------------------------------------Start of Selective Binning

# *** Decide on the year to consider ***
year = 2013
if year == 2013:
    y_values, x_values = y_2013, x_2013
elif year == 2014:
    y_values, x_values = y_2014, x_2014
else:
    y_values, x_values = y_2013_2014, x_2013_2014  # Have to check this out! ***

# Define Regression Space by specifying intervals and creating boolean variables for filter
# Note this is for 2014
maximum_x = -32.43
minimum_x = -72.79
maximum_y = 4.72
minimum_y = -32.21

# To allow for selection of range for regression, ignoring the presence of all other data points
x_upper = -40
x_lower = -60
y_upper = 0
y_lower = -20
x_window = (x_values > x_lower) & (x_values < x_upper)
y_window = (y_values > y_lower) & (y_values < y_upper)
x_within_window = x_values[x_window & y_window]
y_within_window = y_values[x_window & y_window]

print(x_within_window.shape)
print(y_within_window.shape)

# First conduct a regression on the 2014 data set
quads_on_side = 20  # define the number of quads along each dimension
# histo, x_edges, y_edges = np.histogram2d(theft_x, theft_y, bins=quads_on_side)  # create histogram
histo, y_edges, x_edges = np.histogram2d(y_within_window, x_within_window, bins=quads_on_side)
x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)  # creating mesh-grid for use
x_mesh = x_mesh[:-1, :-1]  # Removing extra rows and columns due to edges
y_mesh = y_mesh[:-1, :-1]
x_quad_all = fn.row_create(x_mesh)  # Creating the rows from the mesh
y_quad_all = fn.row_create(y_mesh)

# *** Centralising the coordinates to be at the centre of the quads
# Note that the quads will not be of equal length, depending on the data set
quad_length_x = (x_quad_all[-1] - x_quad_all[0]) / quads_on_side
quad_length_y = (y_quad_all[-1] - y_quad_all[0]) / quads_on_side
x_quad_all = x_quad_all + 0.5 * quad_length_x
y_quad_all = y_quad_all + 0.5 * quad_length_y
xy_quad_all = np.vstack((x_quad_all, y_quad_all))  # stacking the x and y coordinates vertically together
k_quad_all = fn.row_create(histo)  # histogram array

# For graphical plotting
x_mesh_centralise_all = x_quad_all.reshape(x_mesh.shape)
y_mesh_centralise_all = y_quad_all.reshape(y_mesh.shape)

# ------------------------------------------End of Selective Binning

# ------------------------------------------Start of Zero Point Exclusion

# This is so as to account for boundaries whereby the probability of incidence is definitely zero in some areas
# of the map - such as on the sea, etc

# Plan is to exclude the points where the histogram is zero

# Create Boolean variable to identify only points with non-zero incidences
non_zero = (k_quad_all > -1)
x_quad_non_zero = x_quad_all[non_zero]
y_quad_non_zero = y_quad_all[non_zero]
k_quad_non_zero = k_quad_all[non_zero]
xy_quad_non_zero = np.vstack((x_quad_non_zero, y_quad_non_zero))

k_mesh = histo

# Another Boolean variable for the mesh shape
non_zero_mesh = (k_mesh > -1)
x_mesh_centralise_non_zero = x_mesh_centralise_all[non_zero_mesh]
y_mesh_centralise_non_zero = y_mesh_centralise_all[non_zero_mesh]

# ------------------------------------------End of Zero Point Exclusion

# ------------------------------------------Start of SELECTION FOR EXCLUSION OF ZERO POINTS

exclusion_sign = 'exclude'  # Toggle between exclusion(1) and inclusion(0) of 'out-of-boundary' points

if exclusion_sign == 'exclude':
    xy_quad = xy_quad_non_zero
    k_quad = k_quad_non_zero
    x_mesh_centralise = x_mesh_centralise_non_zero
    y_mesh_centralise = y_mesh_centralise_non_zero
else:
    xy_quad = xy_quad_all
    k_quad = k_quad_all
    x_mesh_centralise = x_mesh_centralise_all
    y_mesh_centralise = y_mesh_centralise_all

# ------------------------------------------End of SELECTION FOR EXCLUSION OF ZERO POINTS

# ------------------------------------------Start of Hyper-parameter Optimization

"""Initialise posterior mean and posterior covariance"""

mean_posterior = np.zeros(sampling_coord.shape[1])
cov_posterior = np.zeros(sampling_coord.shape[1])
# Prior mean tabulated from data set, not sampling points
prior_mean = mean_func_zero(xy_data_coord)  # should be one row of zeros even though data has two rows


"""Create auto-covariance matrix"""
C_dd = matern_2d(matern_v, sigma_optimal, length_optimal, xy_data_coord, xy_data_coord)
C_noise = np.eye(C_dd.shape[0]) * (noise_optimal ** 2)
C_dd_noise = C_dd + C_noise
prior_mismatch = histo - prior_mean


"""Tabulating the posterior mean and covariance for each sampling point"""
for i in range(sampling_coord.shape[1]):
    xy_star = sampling_coord[:, i]
    C_star_d = matern_2d(matern_v, sigma_optimal, length_optimal, xy_star, xy_data_coord)
    C_star_star = matern_2d(matern_v, sigma_optimal, length_optimal, xy_star, xy_star)
    mean_posterior[i] = mu_post(xy_star, C_dd_noise, C_star_d, prior_mismatch)
    cov_posterior[i] = cov_post(C_star_star, C_star_d, C_dd_noise)


"""Creating 2-D inputs for plotting surfaces"""
sampling_x_2d = sampling_x_row.reshape(intervals, intervals)
sampling_y_2d = sampling_y_row.reshape(intervals, intervals)
mean_posterior_2d = mean_posterior.reshape(intervals, intervals)
cov_posterior_2d = cov_posterior.reshape(intervals, intervals)


"""Plot Settings"""
fig_pp_data = plt.figure(1)

"""
data_original = fig_pp_data.add_subplot(221)  # Original Data Scatter
data_original.scatter(x, y, color='darkblue', marker='.')
data_original.set_title('Original Data Set')
data_original.set_xlabel('x-axis')
data_original.set_ylabel('y-axis')
data_original.grid(True)
"""


data_transform = fig_pp_data.add_subplot(121)  # Transformed Data Scatter
data_transform.scatter(x_transform, y_transform, color='darkblue', marker='.')
data_transform.set_title('Rotated Data Set' + '(Theta = %s) ' % theta)
data_transform.set_xlabel('x-axis')
data_transform.set_ylabel('y-axis')
data_transform.grid(True)

bin_plot = fig_pp_data.add_subplot(122, projection='3d')
bin_plot.scatter(xv_transform_row, yv_transform_row, histo, color='darkblue', marker='.')
bin_plot.set_title('3-D Binned Plot')
bin_plot.set_xlabel('x-axis')
bin_plot.set_ylabel('y-axis')
bin_plot.grid(True)

fig_pp_pred = plt.figure(2)

post_mean_plot = fig_pp_pred.add_subplot(221, projection='3d')
post_mean_plot.plot_surface(sampling_x_2d, sampling_y_2d, mean_posterior_2d, cmap='RdBu')
# post_mean_plot.plot_wireframe(sampling_x_2d, sampling_y_2d, mean_posterior_2d, color='darkblue')
post_mean_plot.set_title('Posterior Mean 3D-Plot')
post_mean_plot.set_xlabel('x-axis')
post_mean_plot.set_ylabel('y-axis')
post_mean_plot.set_zlabel('mean-axis')
post_mean_plot.grid(True)

post_cov_plot = fig_pp_pred.add_subplot(222, projection='3d')
post_cov_plot.plot_surface(sampling_x_2d, sampling_y_2d, cov_posterior_2d, cmap='RdBu')
post_cov_plot.set_title('Posterior Covariance 3D-Plot')
post_cov_plot.set_xlabel('x-axis')
post_cov_plot.set_ylabel('y-axis')
post_cov_plot.set_zlabel('covariance-axis')
post_cov_plot.grid(True)

post_mean_color = fig_pp_pred.add_subplot(223)
post_mean_color.pcolor(sampling_points_x, sampling_points_y, mean_posterior_2d, cmap='RdBu')
post_mean_color.set_title('Posterior Mean Color Map')
post_mean_color.set_xlabel('x-axis')
post_mean_color.set_ylabel('y-axis')
post_mean_color.grid(True)

post_cov_color = fig_pp_pred.add_subplot(224)
post_cov_color.pcolor(sampling_points_x, sampling_points_y, cov_posterior_2d, cmap='RdBu')
post_cov_color.set_title('Posterior Covariance Color Map')
post_cov_color.set_xlabel('x-axis')
post_cov_color.set_ylabel('y-axis')
post_cov_color.grid(True)

wireframes = plt.figure(3)

post_mean_wire = wireframes.add_subplot(121, projection='3d')
post_mean_wire.plot_surface(sampling_x_2d, sampling_y_2d, mean_posterior_2d, cmap='RdBu')
post_mean_wire.set_title('Posterior Mean')
post_mean_wire.set_xlabel('x-axis')
post_mean_wire.set_ylabel('y-axis')
post_mean_wire.grid(True)

post_cov_wire = wireframes.add_subplot(122, projection='3d')
post_cov_wire.plot_surface(sampling_x_2d, sampling_y_2d, cov_posterior_2d, cmap = 'RdBu')
post_cov_wire.set_title('Posterior Covariance')
post_cov_wire.set_xlabel('x-axis')
post_cov_wire.set_ylabel('y-axis')
post_cov_wire.grid(True)

plt.show()

