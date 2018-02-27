import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as scopt
import time

"""Methodology for Conducting Gaussian Regression for 2-D"""


def mean_func_zero(c):  # Prior mean function taken as 0 for the entire sampling range
    if np.array([c.shape]).size == 1:
        mean_c = np.zeros(1)  # Make sure this is an array
    else:
        mean_c = np.zeros(c.shape[1])
    return mean_c  # Outputs a x and y coordinates, created from the mesh grid


def mean_func_scalar(mean, c):  # Assume that the prior mean is a constant to be optimised
    if np.array([c.shape]).size == 1:
        mean_c = np.ones(1) * mean
    else:
        mean_c = np.ones(c.shape[1]) * mean
    return mean_c


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


# This is way faster than the function above beyond n=10
def fast_matern_2d(sigma_matern, length_matern, x1, x2):  # there are only two variables in the matern function
    """
    This is much much faster than iteration over every point beyond n = 10. This function takes advantage of the
    symmetry in the covariance matrix and allows for fast regeneration. For this function, v = 3/2
    :param sigma_matern: coefficient factor at the front
    :param length_matern: length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :return: Covariance matrix with matern kernel
    """
    # Note that this function only takes in 2-D coordinates, make sure there are 2 rows and n columns
    n = x1.shape[1]
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        cov_matrix[i, i] = sigma_matern ** 2
        for j in range(i + 1, n):
            diff = x1[:, i] - x2[:, j]
            euclidean = np.sqrt(np.matmul(diff, np.transpose(diff)))
            coefficient_term = (1 + np.sqrt(3) * euclidean * (length_matern ** -1))
            exp_term = np.exp(-1 * np.sqrt(3) * euclidean * (length_matern ** -1))
            cov_matrix[i, j] = (sigma_matern ** 2) * coefficient_term * exp_term
            cov_matrix[j, i] = cov_matrix[i, j]

    return cov_matrix


def mu_post(xy_next, c_auto, c_cross, mismatch):  # Posterior mean
    if c_cross.shape[1] != (np.linalg.inv(c_auto)).shape[0]:
        print('First Dimension Mismatch!')
    if (np.linalg.inv(c_auto)).shape[1] != (np.transpose(mismatch)).shape[0]:
        print('Second Dimension Mismatch!')
    else:
        mean_post = mean_func_zero(xy_next) + fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(mismatch))
        return mean_post


def var_post(c_next_auto, c_cross, c_auto):  # Posterior Covariance
    c_post = c_next_auto - fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(c_cross))
    return c_post


def log_gp_likelihood(param, *args):  # Param includes both sigma and l, arg is passed as a pointer
    """
    Function in format for optimization using Nelder-Mead Simplex Algorithm - change to include the scalar mean as a
    value to be optimised as well - total of 4 hyper-parameters to be optimised. Note that Matern v=3/2 is used in the
    fast_matern_2d function
    :param param: amplitude sigma, length scale and noise amplitude
    :param args: locations of quads, xy_quad and histogram values for each quad
    :return: the log-likelihood of the gaussian process
    """

    # Define parameters to be optimised
    sigma = param[0]  # param is a tuple containing 2 things, which has already been defined in the function def
    length = param[1]
    noise = param[2]  # Over here we have defined each parameter in the tuple, include noise
    mean = param[3]

    # Define arguments to be entered
    xy_coordinates = args[0]  # This argument is a constant passed into the function
    histogram_data = args[1]  # Have to enter histogram data as well

    # Tabulate prior mean as a scalar to be optimized
    prior_mu = mean_func_scalar(mean, xy_coordinates)  # This creates a matrix with 2 rows

    # Tabulate auto-covariance matrix using fast matern function plus noise
    c_auto = fast_matern_2d(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fronecker delta function
    c_overall = c_auto + c_noise

    # 3 components to log_gp_likelihood: model_fit, model_complexity and model_constant
    model_fit = - 0.5 * fn.matmulmul(histogram_data - prior_mu, np.linalg.inv(c_overall),
                                     np.transpose(histogram_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_overall))
    model_constant = - 0.5 * len(histogram_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant

    # Taking the minimum of the negative log_gp_likelihood, to obtain the maximum of log_gp_likelihood
    return -log_model_evid


# For the case where gaussian prior with zero mean is assumed
def log_gp_likelihood_zero_mean(param, *args):
    """
    Log marginal likelihood which is taken as the objective function for the optimization of the hyper-parameters,
    assuming a zero prior mean.
    :param param: sigma amplitude, length scale and noise
    :param args: coordinates of each quad and histogram data
    :return: the negative of the log marginal likelihood for optimization using Nelder-Mead/ DE
    """
    # Define parameters to be optimised
    sigma = param[0]  # param is a tuple containing 2 things, which has already been defined in the function def
    length = param[1]
    noise = param[2]  # Over here we have defined each parameter in the tuple, include noise

    # Define arguments to be entered
    xy_coordinates = args[0]  # This argument is a constant passed into the function
    histogram_data = args[1]  # Have to enter histogram data as well

    # Tabulate prior mean as a scalar to be optimized
    prior_mu = mean_func_zero(xy_coordinates)  # This creates a matrix with 2 rows

    # Tabulate auto-covariance matrix using fast matern function plus noise
    c_auto = fast_matern_2d(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fronecker delta function
    c_overall = c_auto + c_noise

    # 3 components to log_gp_likelihood: model_fit, model_complexity and model_constant
    model_fit = - 0.5 * fn.matmulmul(histogram_data - prior_mu, np.linalg.inv(c_overall),
                                     np.transpose(histogram_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_overall))
    model_constant = - 0.5 * len(histogram_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant

    # Taking the minimum of the negative log_gp_likelihood, to obtain the maximum of log_gp_likelihood
    return -log_model_evid


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
x_upper = -43
x_lower = -63
y_upper = -2
y_lower = -22
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

exclusion_sign = 'include'  # Toggle between exclusion(1) and inclusion(0) of 'out-of-boundary' points

if exclusion_sign == 'exclude':
    xy_quad = xy_quad_non_zero
    x_quad = x_quad_non_zero
    y_quad = y_quad_non_zero
    k_quad = k_quad_non_zero
    x_mesh_centralise = x_mesh_centralise_non_zero
    y_mesh_centralise = y_mesh_centralise_non_zero
else:
    xy_quad = xy_quad_all
    x_quad = x_quad_all
    y_quad = y_quad_all
    k_quad = k_quad_all
    x_mesh_centralise = x_mesh_centralise_all
    y_mesh_centralise = y_mesh_centralise_all

# ------------------------------------------End of SELECTION FOR EXCLUSION OF ZERO POINTS

# ------------------------------------------Start of Hyper-parameter Optimization

# Checking dimensions of histo and quad after selection of window and points above a certain threshold
print('The quad coordinates are ', xy_quad_all)
print('The shape of quad coordinates are ', xy_quad_all.shape)
print('The histogram array is ', k_quad)
print('The shape of histogram array is ', k_quad.shape)  # should be the square of the number of quads on side

# Initialise arguments to be entered into objective function
xyz_data = (xy_quad, k_quad)

# Check time for optimization process
start_opt = time.clock()

# Decide on optimization method
opt_method = 'differential_evolution'

# No bounds needed for Nelder-Mead Simplex Algorithm
if opt_method == 'nelder-mead':
    # Initialise parameters to be optimized - could have used Latin-Hypercube
    initial_param = np.array([20, 5, 5, 20])  # Sigma amplitude, length scale, noise amplitude and scalar mean
    solution = scopt.minimize(fun=log_gp_likelihood_zero_mean, args=xyz_data, x0=initial_param, method='Nelder-Mead')

# Differential Evolution Method - which can be shown to give the same result as Nelder-Mead
elif opt_method == 'differential_evolution':
    # boundary = [(20, 40), (0, 5), (0, 10), (20, 30)]  # if zero mean, the last element of tuple will not be used
    boundary = [(0, 30), (0, 10), (0, 10)]  # for zero mean
    solution = scopt.differential_evolution(func=log_gp_likelihood_zero_mean, bounds=boundary, args=xyz_data,
                                            init='latinhypercube')

# List optimal hyper-parameters
sigma_optimal = solution.x[0]
length_optimal = solution.x[1]
noise_optimal = solution.x[2]
# mean_optimal = solution.x[3]
print('Last function evaluation is ', solution.fun)
print('optimal sigma is ', sigma_optimal)
print('optimal length-scale is ', length_optimal)
print('optimal noise amplitude is ', noise_optimal)
# print('optimal scalar mean value is ', mean_optimal)

end_opt = time.clock()
time_opt = end_opt - start_opt
print(time_opt)
# ------------------------------------------End of Hyper-parameter Optimization

# ------------------------------------------Start of Sampling Points Creation

# Define number of points for y_*
intervals = 20

cut_decision = 'no'
if cut_decision == 'yes':
    # Define sampling points beyond the data set
    cut_off_x = (np.max(x_quad) - np.min(x_quad)) / 2
    cut_off_y = (np.max(y_quad) - np.min(y_quad)) / 2

else:
    cut_off_x = 0
    cut_off_y = 0

# Expressing posterior away from the data set by the cut-off values
sampling_points_x = np.linspace(np.min(x_quad) - cut_off_x, np.max(x_quad) + cut_off_x, intervals)
sampling_points_y = np.linspace(np.min(y_quad) - cut_off_y, np.max(y_quad) + cut_off_y, intervals)

# Centralising coordinates so that we tabulate values at centre of quad
sampling_half_length = 0.5 * (np.max(x_quad) - np.min(x_quad))
sampling_points_x = sampling_points_x + sampling_half_length
sampling_points_y = sampling_points_y + sampling_half_length

# Create iteration for coordinates using mesh-grid - for plotting
sampling_points_xmesh, sampling_points_ymesh = np.meshgrid(sampling_points_x, sampling_points_y)
sampling_x_row = fn.row_create(sampling_points_xmesh)
sampling_y_row = fn.row_create(sampling_points_ymesh)
sampling_xy = np.vstack((sampling_x_row, sampling_y_row))

# ------------------------------------------End of Sampling Points Creation

# ------------------------------------------Start of Posterior Tabulation

# Generate auto-covariance function from the data set
cov_dd = fast_matern_2d(sigma_optimal, length_optimal, xy_quad, xy_quad)
cov_noise = np.eye(cov_dd.shape[0]) * (noise_optimal ** 2)
cov_overall = cov_dd + cov_noise
prior_mean = mean_func_scalar(0, xy_quad)
prior_mismatch = k_quad - prior_mean

# Initialise mean_posterior and var_posterior array
mean_posterior = np.zeros(sampling_xy.shape[1])
var_posterior = np.zeros(sampling_xy.shape[1])

# Generate mean and covariance array
for i in range(sampling_xy.shape[1]):

    # Generate status output
    if i % 50 == 0:  # if i is a multiple of 50,
        print('Tabulating Prediction Point', i)

    # At each data point,
    xy_star = sampling_xy[:, i]
    cov_star_d = matern_2d(3/2, sigma_optimal, length_optimal, xy_star, xy_quad)  # Cross-covariance Matrix

    # auto-covariance between the same data point - adaptive function for both scalar and vectors
    cov_star_star = matern_2d(3/2, sigma_optimal, length_optimal, xy_star, xy_star)

    # Generate Posterior Mean and Variance
    mean_posterior[i] = mu_post(xy_star, cov_overall, cov_star_d, prior_mismatch)
    var_posterior[i] = var_post(cov_star_star, cov_star_d, cov_overall)


sampling_x_2d = sampling_x_row.reshape(intervals, intervals)
sampling_y_2d = sampling_y_row.reshape(intervals, intervals)
mean_posterior_2d = mean_posterior.reshape(intervals, intervals)
var_posterior_2d = var_posterior.reshape(intervals, intervals)
sd_posterior_2d = np.sqrt(var_posterior_2d)


# ------------------------------------------End of Posterior Tabulation

# ------------------------------------------Start of Plots
fig_post = plt.figure()
post_mean_color = fig_post.add_subplot(121)
post_mean_color.pcolor(sampling_points_x, sampling_points_y, mean_posterior_2d, cmap='YlOrBr')
post_mean_color.scatter(x_quad, y_quad, marker='o', color='black', s=0.3)
post_mean_color.set_title('Posterior Mean')
post_mean_color.set_xlabel('UTM Horizontal Coordinate')
post_mean_color.set_ylabel('UTM Vertical Coordinate')
# post_mean_color.grid(True)

post_sd_color = fig_post.add_subplot(122)
post_sd_color.pcolor(sampling_points_x, sampling_points_y, sd_posterior_2d, cmap='YlOrBr')
post_sd_color.scatter(x_quad, y_quad, marker='o', color='black', s=0.3)
post_sd_color.set_title('Posterior Standard Deviation')
post_sd_color.set_xlabel('UTM Horizontal Coordinate')
post_sd_color.set_ylabel('UTM Vertical Coordinate')
# post_cov_color.grid(True)

# ------------------------------------------End of Plots
plt.show()
