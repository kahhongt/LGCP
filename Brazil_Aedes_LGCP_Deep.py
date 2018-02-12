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


def poisson_cont(k, landa):  # to allow for non-integer k values
    numerator_p = np.power(landa, k) * np.exp(-1 * landa)
    denominator_p = scispec.gamma(k + 1)  # Generalised factorial function for non-integer k values
    # if argument into gamma function is 0, the output is a zero as well, but 0! = 1
    p = numerator_p / denominator_p
    return p


def poisson_product(k_array, landa_array):
    """Takes in 2 arrays of equal size, and takes product of poisson distributions"""
    quadrats = len(k_array)  # define the number of quadrats in total
    prob_array = np.zeros(quadrats)

    if landa_array.size == 1:
        for i in range(len(k_array)):
            prob_array[i] = poisson_cont(k_array[i], landa_array)
    else:
        if len(k_array) == len(landa_array):
            for i in range(len(prob_array)):
                prob_array[i] = poisson_cont(k_array[i], landa_array[i])
        else:
            print('Length Mismatch')
    p_likelihood = np.prod(prob_array)  # Taking combined product of distributions - leading to small values
    # Note output is a scalar (singular value)
    return p_likelihood  # Returns the non logarithmic version.


def log_special(array):
    """Taking an element-wise natural log of the array, retain array dimensions"""
    """with the condition that log(0) = 0, so there are no -inf elements"""
    log_array = np.zeros(array.size)
    for i in range(array.size):
        if array[i] == 0:
            log_array[i] = 0
        else:
            log_array[i] = np.log(array[i])
    return log_array


def mean_func_zero(c):  # Prior mean function taken as 0 for the entire sampling range
    if np.array([c.shape]).size == 1:
        mean_c = np.ones(1) * 0  # Make sure this is an array
    else:
        mean_c = np.ones(c.shape[1]) * 0
    return mean_c  # Outputs a x and y coordinates, created from the mesh grid


def mean_func_scalar(mean, c):  # Assume that the prior mean is a constant to be optimised
    if np.array([c.shape]).size == 1:
        mean_c = np.ones(1) * mean
    else:
        mean_c = np.ones(c.shape[1]) * mean
    return mean_c


def squared_exp_2d(sigma_exp, length_exp, x1, x2):  # Only for 2-D
    """
    Generates a covariance matrix using chosen hyper-parameters and coordinates to iterate over
    :param sigma_exp: coefficient factor
    :param length_exp: length scale
    :param x1: First set of coordinates to iterate over
    :param x2: Second set of coordinates to iterate over
    :return: Covariance Matrix with squared-exp kernel
    """
    # To allow the function to take in x1 and x2 of various dimensions
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
    """
    Creating the covariance matrix from chosen hyper-parameters and the coordinates the iterate over
    :param v_value: the matern factor miu: 1/2 or 3/2
    :param sigma_matern: coefficient factor at the front
    :param length_matern: length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :return: Covariance matrix with matern kernel
    """
    #  To allow the function to take in x1 and x2 of various dimensions
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


def log_model_evidence(param, *args):
    """
    ***NOTE THIS IS FOR STANDARD GP REGRESSION - DO NOT USE FOR LGCP. THIS FUNCTION ASSUMES THAT THE LATENT INTENSITY IS
    THE SAME AS THE DATA SET. HENCE, OVER HERE, WE TAKE (y_i - u_i) instead of (v_i - u_i) as the difference for the
    calculation of the euclidean

    :param param: sigma, length scale and noise hyper-parameters
    :param args: inputs into the function (from dataset and elsewhere)
    :return: The log-Model evidence
    """
    sigma = param[0]
    length = param[1]
    noise = param[2]  # Over here we have defined each parameter in the tuple, include noise
    scalar_mean = param[3]
    xy_coordinates = args[0]  # This argument is a constant passed into the function
    histogram_data = args[1]  # Have to enter histogram data as well
    prior_mu = mean_func_scalar(scalar_mean, xy_coordinates)  # This creates a matrix with 2 rows
    c_auto = fast_matern_2d(sigma, length, xy_coordinates, xy_coordinates)
    # c_auto = squared_exp_2d(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    c_auto_noise = c_auto + c_noise  # Overall including noise, plus include any other combination
    model_fit = - 0.5 * fn.matmulmul(histogram_data - prior_mu, np.linalg.inv(c_auto_noise),
                                     np.transpose(histogram_data - prior_mu))
    model_complexity = - 0.5 * (math.log(np.linalg.det(c_auto_noise)))
    model_constant = - 0.5 * len(histogram_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant
    return -log_model_evid  # We want to maximize the log-likelihood, meaning the min of negative log-likelihood


def log_integrand_without_v(param, *args):
    """
    1. Tabulates the log of the integrand, g(v), so that we can optimise for v_array and hyper-parameters
    The log of the integrand, log[g(v)] is used as log function is monotonically increasing - so they have the same
    optimal points - note we want to maximize the integrand
    2. Note here that because the LGCP model is doubly stochastic, the log-intensities are meant to be optimized]
    3. Kernel: Matern(3/2)
    :param param: v_array, hyperparameters - sigma, length scale and noise, prior scalar mean
    :param args: xy coordinates for iteration, data set k_array, matern factor value = 1/2 or 3/2
    :return: the log of the integrand, log[g(v)], so that we can optimise and find best hyperparameters and vhap
    """
    # Generate Matern Covariance Matrix
    # Enter parameters
    sigma = param[0]
    length = param[1]
    noise = param[2]
    scalar_mean = param[3]
    v_array = param[4:]  # Concatenate v_array behind the hyper-parameters

    # Enter Arguments
    xy_coordinates = args[0]
    k_array = args[1]
    prior_mean = mean_func_scalar(scalar_mean, xy_coordinates)
    c_auto = fast_matern_2d(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    cov_matrix = c_auto + c_noise

    """Generate Objective Function = log[g(v)]"""
    exp_term = -1 * np.sum(np.exp(v_array))
    product_term = np.matmul(v_array, np.transpose(k_array))
    det_term = -0.5 * np.log(2 * np.pi * np.linalg.det(cov_matrix))

    factorial_k = scispec.gamma(k_array + 1)
    factorial_term = - np.sum(np.log(factorial_k))

    v_difference = v_array - prior_mean
    euclidean_term = -0.5 * fn.matmulmul(v_difference, np.linalg.inv(cov_matrix), np.transpose(v_difference))

    """Summation of all terms change to correct form to find minimum point"""
    log_g = exp_term + product_term + det_term + factorial_term + euclidean_term
    log_g_minimization = -1 * log_g
    return log_g_minimization


def log_integrand_with_v(param, *args):
    """
    1. Tabulates the log of the integrand, g(v), so that we can optimise for the GP hyper-parameters given
    having optimised for the v_array. The v_array will now be entered as an argument into the objective function.
    The log of the integrand, log[g(v)] is used as log function is monotonically increasing - so they have the same
    optimal points - note we want to maximize the integrand

    2. Note here that because the LGCP model is doubly stochastic, the log-intensities are meant to be optimized]

    3. Kernel: Matern(3/2)
    :param param: v_array, hyperparameters - sigma, length scale and noise, prior scalar mean
    :param args: xy coordinates for iteration, data set k_array, matern factor value = 1/2 or 3/2
    :return: the log of the integrand, log[g(v)], so that we can optimise and find best hyperparameters and vhap

    *** Note that this objective function is currently problematic - advised to not use it ***
    """
    # Generate Matern Covariance Matrix
    # Enter parameters
    sigma = param[0]
    length = param[1]
    noise = param[2]
    scalar_mean = param[3]

    # Enter Arguments
    xy_coordinates = args[0]
    k_array = args[1]
    v_array = args[2]  # Note that this is refers to the optimised log-intensity array
    prior_mean = mean_func_scalar(scalar_mean, xy_coordinates)
    c_auto = fast_matern_2d(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    cov_matrix = c_auto + c_noise

    """Generate Objective Function = log[g(v)]"""
    exp_term = -1 * np.sum(np.exp(v_array))
    product_term = v_array * k_array
    det_term = -0.5 * np.log(2 * np.pi * np.linalg.det(cov_matrix))

    factorial_k = scispec.gamma(k_array + 1)
    factorial_term = - np.sum(np.log(factorial_k))

    v_difference = v_array - prior_mean
    euclidean_term = -0.5 * fn.matmulmul(v_difference, np.linalg.inv(cov_matrix), np.transpose(v_difference))

    """Summation of all terms change to correct form to find minimum point"""
    log_g = exp_term + product_term + det_term + factorial_term + euclidean_term
    log_g_minimization = -1 * log_g
    return log_g_minimization


def short_log_integrand_v(param, *args):
    """
    1. Shorter version that tabulates only the log of the GP prior behind the Poisson distribution. Includes only terms
    containing the covariance matrix elements that are made up of the kernel hyper-parameters
    2. Kernel: Matern(3/2)
    3. Assume a constant latent intensity, even at locations without any incidences
    :param param: v_array, hyperparameters - sigma, length scale and noise, prior scalar mean - array of 4 elements
    :param args: xy coordinates for input into the covariance function and the optimised v_array
    :return: the log of the GP Prior, log[N(prior mean, covariance matrix)]
    """
    # Generate Matern Covariance Matrix
    # Enter parameters
    sigma = param[0]
    length = param[1]
    noise = param[2]
    scalar_mean = param[3]

    # Enter Arguments
    xy_coordinates = args[0]
    v_array = args[1]  # Note that this is refers to the optimised log-intensity array

    # Set up inputs for generation of objective function
    prior_mean = mean_func_scalar(scalar_mean, xy_coordinates)
    c_auto = fast_matern_2d(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    cov_matrix = c_auto + c_noise

    """Generate Objective Function = log[g(v)]"""

    # Generate Determinant Term (after taking log)
    determinant = np.exp(np.linalg.slogdet(cov_matrix))[1]
    det_term = -0.5 * np.log(2 * np.pi * determinant)

    # Generate Euclidean Term (after taking log)
    v_difference = v_array - prior_mean
    inv_covariance_matrix = np.linalg.inv(cov_matrix)
    euclidean_term = -0.5 * fn.matmulmul(v_difference, inv_covariance_matrix, np.transpose(v_difference))

    """Summation of all terms change to correct form to find minimum point"""
    log_gp = det_term + euclidean_term
    log_gp_minimization = -1 * log_gp  # Make the function convex for minimization
    return log_gp_minimization


def log_likelihood(param, *args):
    """
    Considers only the log-likelihood of the Poisson distribution in front of the gaussian process to optimize
    latent values - note that there are no hyper-parameters here to consider. The log-likelhood is taken as
     the natural log is monotically increasing
    :param param: v_array containing the latent intensities
    :param args: k_array which is the data set
    :return: log of the combined poisson distributions
    """
    # Define parameters and arguments
    v_array = param
    k_array = args[0]

    # Generate Objective Function: log(P(D|v))
    exp_term = -1 * np.sum(np.exp(v_array))
    product_term = np.matmul(v_array, np.transpose(k_array))

    factorial_k = scispec.gamma(k_array + np.ones_like(k_array))
    factorial_term = - np.sum(np.log(factorial_k))  # summation of logs = log of product

    log_p_likelihood = exp_term + product_term + factorial_term
    log_p_likelihood_convex = -1 * log_p_likelihood
    return log_p_likelihood_convex


def gradient_log_likelihood(param, *args):
    """
    Construct gradient vector of the log-likelihood for optimization
    :param param: v_array (log of the latent intensities)
    :param args: k_array (the data set)
    :return: gradient vector of size n
    """
    # Define parameters and arguments
    v_array = param
    k_array = args[0]

    # Construct Gradient Vector
    exp_component = -1 * np.exp(v_array)
    k_component = k_array
    grad_vector = exp_component + k_component
    grad_vector_convex = -1 * grad_vector
    return grad_vector_convex


def hessianproduct_log_likelihood(param, *args):
    """
    Generates vector containing the hessian_product along each variable direction
    :param param: v_array containing the latent intensities
    :param args: tuple containing (k_array, p_array) - note this tuple taken into every function/derivative in the
    optimization
    :return: vector containing the hessian product, which is the hessian matrix multiplied by an arbitrary vector p
    """
    # Define parameters and arguments
    v_array = param
    p_array = args[1]
    # Generate Hessian Product without creating the hessian
    exp_v_array = np.exp(v_array)
    hessian_product = -1 * exp_v_array * p_array
    hessian_product_convex = -1 * hessian_product
    return hessian_product_convex


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
year = 2014
if year == 2013:
    y_values, x_values = y_2013, x_2013
elif year == 2014:
    y_values, x_values = y_2014, x_2014
else:
    y_values, x_values = y_2013_2014, x_2013_2014  # Have to check this out! ***

# Define Regression Space by specifying intervals and creating boolean variables for filter
maximum_x = -32.43
minimum_x = -72.79
maximum_y = 4.72
minimum_y = -32.21

# To allow for selection of range for regression, ignoring the presence of all other data points
x_upper = maximum_x
x_lower = minimum_x
y_upper = maximum_y
y_lower = minimum_y
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
x_quad = fn.row_create(x_mesh)  # Creating the rows from the mesh
y_quad = fn.row_create(y_mesh)

# *** Centralising the coordinates to be at the centre of the quads
# Note that the quads will not be of equal length, depending on the data set
quad_length_x = (x_quad[-1] - x_quad[0]) / quads_on_side
quad_length_y = (y_quad[-1] - y_quad[0]) / quads_on_side
x_quad = x_quad + 0.5 * quad_length_x
y_quad = y_quad + 0.5 * quad_length_y
xy_quad = np.vstack((x_quad, y_quad))  # stacking the x and y coordinates vertically together
k_quad = fn.row_create(histo)  # histogram array
x_mesh_centralise = x_quad.reshape(x_mesh.shape)
y_mesh_centralise = y_quad.reshape(y_mesh.shape)

# ------------------------------------------End of Selective Binning

# ------------------------------------------Start of Optimization of latent v_array using only the log-likelihood

start_v_opt = time.clock()
# Optimize using Gradient and Hessian Methods, instead of just using the Nelder-mead which depends
# solely on function evaluations. Gradient and Hessian provides the directions for the iteration, which speeds things
# up a lot

# Optimization using Newton-Conjugate-Gradient algorithm (method='Newton-CG') - note no hyperparameters here

# Initial starting point for optimization
initial_v_array = np.ones_like(k_quad)  # log-intensity has same dimensions as data points k_quad
initial_p_array = np.ones_like(k_quad)

# Tuple containing all arguments to be passed into objective function, jacobian and hessian, but we can specify
# which arguments will be used for each function
arguments_v = (k_quad, initial_p_array)

# Start Optimization Algorithm for latent intensities
v_solution = scopt.minimize(fun=log_likelihood, args=arguments_v, x0=initial_v_array, method='Newton-CG',
                            jac=gradient_log_likelihood, hessp=hessianproduct_log_likelihood,
                            options={'xtol': 0.001, 'disp': True, 'maxiter': 100000})

latent_v_array = v_solution.x  # v_array is the log of the latent intensity
lambda_quad = np.exp(latent_v_array)  # Taking the exponential of the log of the latent intensity
lambda_mesh = lambda_quad.reshape(x_mesh.shape)  # Create mesh from rows

print("Latent Intensity Values are ", lambda_quad)
print("Initial Data Points are ", k_quad)
print(latent_v_array.shape)
print(lambda_quad.shape)
print(k_quad.shape)

time_v_opt = time.clock() - start_v_opt
# ------------------------------------------Start of Optimization of GP Hyper-parameters
start_gp_opt = time.clock()
# Initialise Hyper-parameters for the Gaussian Process
initial_hyperparam = np.array([1, 1, 1, 1])

# Set up tuple for arguments
args_hyperparam = (xy_quad, latent_v_array)

# Start Optimization Algorithm for GP Hyperparameters
hyperparam_solution = scopt.minimize(fun=short_log_integrand_v, args=args_hyperparam, x0=initial_hyperparam,
                                     method='Nelder-Mead',
                                     options={'xatol': 0.1, 'fatol': 1, 'disp': True, 'maxfev': 1000})

# options={'xatol': 0.1, 'fatol': 1, 'disp': True, 'maxfev': 10000})
# No bounds needed for Nelder-Mead
# solution = scopt.minimize(fun=log_model_evidence, args=xyz_data, x0=initial_param, method='Nelder-Mead')
print(hyperparam_solution)

time_gp_opt = time.clock() - start_gp_opt

print('Time Taken for v optimization = ', time_v_opt)
print('TIme Taken for hyper-parameter optimization = ', time_gp_opt)

# ------------------------------------------End of Optimization of GP Hyper-parameters

# ------------------------------------------Start of Posterior Covariance Calculation
# Note Hessian = second derivative of the log[g(v)]
# Posterior Distribution follows N(v; v_hap, -1 * Hessian)

start_posterior_tab = time.clock()

# Extract optimized hyper-parameters
hyperparam_opt = hyperparam_solution.x
sigma_opt = hyperparam_opt[0]
length_opt = hyperparam_opt[1]
noise_opt = hyperparam_opt[2]
prior_mean_opt = hyperparam_opt[3]

# Generate prior covariance matrix with kronecker noise
cov_auto = fast_matern_2d(sigma_opt, length_opt, xy_quad, xy_quad)
cov_noise = (noise_opt ** 2) * np.eye(cov_auto.shape[0])
cov_overall = cov_auto + cov_noise

# Generate inverse of covariance matrix and set up the hessian matrix using symmetry
inv_cov_overall = np.linalg.inv(cov_overall)
inv_cov_diagonal_array = np.diag(inv_cov_overall)
hess_diagonal = -1 * (np.exp(latent_v_array) + inv_cov_diagonal_array)

# Initialise and generate hessian matrix
hess_matrix = np.zeros_like(inv_cov_overall)
hess_length = inv_cov_overall.shape[0]

# Fill in values
for i in range(hess_length):
    hess_matrix[i, i] = -1 * (np.exp(latent_v_array[i]) + inv_cov_overall[i, i])
    for j in range(i + 1, hess_length):
        hess_matrix[i, j] = -0.5 * (inv_cov_overall[i, j] + inv_cov_overall[j, i])
        hess_matrix[j, i] = hess_matrix[i, j]


# Generate Posterior Covariance Matrix of log-intensity v
posterior_cov_matrix = - hess_matrix

# Standard Deviation in terms of log-intensity v
posterior_sd_v_array = np.sqrt(np.diag(posterior_cov_matrix))  # This can then be plotted

# Taking into consideration 2 standard deviations away from the posterior mean
posterior_sd_v_upper = latent_v_array + (0.1 * posterior_sd_v_array)
posterior_sd_v_lower = latent_v_array - (0.1 * posterior_sd_v_array)

# Setting the boundary for the filling in-between *** Note that all possible values of lambda have to be positive
posterior_sd_lambda_upper = np.exp(posterior_sd_v_upper)
posterior_sd_lambda_lower = np.exp(posterior_sd_v_lower)

# Generate posterior standard deviation mesh for plotting ***
posterior_sd_v_mesh = posterior_sd_v_array.reshape(lambda_mesh.shape)

# Measure time taken for covariance matrix and final standard deviation tabulation
time_posterior_tab = time.clock() - start_posterior_tab

print('Time Taken for Posterior Tabulation = ', time_posterior_tab)

# ------------------------------------------End of Posterior Covariance Calculation

start_plotting = time.clock()

# ------------------------------------------Start of Plotting Process of Point Patterns, Histogram and Posterior Mean
brazil_fig = plt.figure()
brazil_fig.canvas.set_window_title('Brazil Aedes Occurrences')

brazil_scatter = brazil_fig.add_subplot(221)
# pp_2014 = brazil_scatter.scatter(x_2014, y_2014, marker='.', color='blue', s=0.1)
pp_2013 = brazil_scatter.scatter(x_2013, y_2013, marker='.', color='red', s=0.1)
# plt.legend([pp_2014, pp_2013], ["2014", "2013"])
brazil_scatter.set_xlim(x_lower, x_upper)
brazil_scatter.set_ylim(y_lower, y_upper)

brazil_histogram = brazil_fig.add_subplot(222)
brazil_histogram.pcolor(x_mesh_centralise, y_mesh_centralise, histo, cmap='RdBu')
# brazil_histogram.set_xlim(x_lower, x_upper)
# brazil_histogram.set_ylim(y_lower, y_upper)

brazil_lambda = brazil_fig.add_subplot(223)
brazil_lambda.pcolor(x_mesh_centralise, y_mesh_centralise, lambda_mesh, cmap='RdBu')
# brazil_lambda.set_xlim(x_lower, x_upper)
# brazil_lambda.set_ylim(y_lower, y_upper)

brazil_sd = brazil_fig.add_subplot(224)
brazil_sd.pcolor(x_mesh_centralise, y_mesh_centralise, posterior_sd_v_mesh, cmap='RdBu')

# Plotting the Posterior Covariance of intensity lambda
brazil_3d = plt.figure()
brazil_3d.canvas.set_window_title('Posterior Mean and Covariance in 3-D')
brazil_mean_3d = brazil_3d.add_subplot(121, projection='3d')
brazil_mean_3d.plot_surface(x_mesh, y_mesh, lambda_mesh, cmap='RdBu')
brazil_mean_3d.set_title('Posterior Mean')
brazil_mean_3d.set_xlabel('x-axis')
brazil_mean_3d.set_ylabel('y-axis')
brazil_mean_3d.grid(True)

# Plotting the Posterior Variance of log-intensity v, and not of lambda
brazil_mean_3d = brazil_3d.add_subplot(122, projection='3d')
brazil_mean_3d.plot_surface(x_mesh, y_mesh, posterior_sd_v_mesh, cmap='RdBu')
brazil_mean_3d.set_title('Posterior Standard Deviation')
brazil_mean_3d.set_xlabel('x-axis')
brazil_mean_3d.set_ylabel('y-axis')
brazil_mean_3d.grid(True)

# *** TAKE NOTE THAT POSTERIOR MEAN IS IN TERMS OF INTENSITY BUT STANDARD DEVIATION IS IN TERMS OF LOG-INTENSITY ***

# ------------------------------------------End of Plotting Process of Point Patterns, Histogram and Posteriors


# ------------------------------------------Start of 1-D Representation of 2-D Gaussian Process
# NOTE DOES NOT WORK FINDING THE ACTUAL INTENSITY POSTERIOR MEAN AND VARIANCE

# Involves creating an index so as to provide a representation of how the standard deviation varies for the location
# of each histogram data point

# Create an index to label each data point so as to make it 1-D
index = np.arange(0, lambda_mesh.size, 1)

# Bounds set at 2 standard deviations from the posterior mean of latent log-intensity v
upper_bound = posterior_sd_lambda_upper
lower_bound = posterior_sd_lambda_lower

brazil_1d = plt.figure()
brazil_1d.canvas.set_window_title('Brazil Reshaped to 1-D')

brazil_post = brazil_1d.add_subplot(111)
brazil_post.fill_between(index, lower_bound, upper_bound, color='lavender')
brazil_post.scatter(index, k_quad, color='darkblue', marker='x', s=1)
brazil_post.scatter(index, lambda_quad, color='darkred', marker='.', s=1)
brazil_post.set_title('Brazil Aedes Regression')
brazil_post.set_xlabel('Index of Histogram')
brazil_post.set_ylabel('Brazil Aedes Spread Posterior Distribution')

time_plotting = time.clock() - start_plotting
print('Time Taken for plotting graphs = ', time_plotting)

plt.show()

# ------------------------------------------End of 1-D Representation of 2-D Gaussian Process
