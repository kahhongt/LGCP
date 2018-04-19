import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn
import time
import scipy.special as scispec
import scipy.optimize as scopt
import matplotlib.path as mpath

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Over here, the binning process will have to adapt to the transformation of the polygon,
# instead of using the entire regression window W_T

""" For iterating when alpha is taken to be 1, then we iterate beta which determines the skew"""


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


def rational_quadratic_2d(alpha_rq, length_rq, x1, x2):
    """
    Rational Quadratic Coveriance function with 2 parameters to be optimized, using
    power alpha and length scale l. The Rational Quadratic Kernel is used to model the
    volatility of equity index returns, which is equivalent to a sum of Squared
    Exponential Kernels. This kernel is used to model multi-scale data

    This is a fast method of generating the rational quadratic kernel, by exploiting the symmetry
    of the covariance matrix
    :param alpha_rq: power and denominator
    :param length_rq: length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :return: Covariance matrix with Rational Quadratic Kernel
    """
    # Note that this function only takes in 2-D coordinates, make sure there are 2 rows and n columns
    n = x1.shape[1]
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        cov_matrix[i, i] = 1
        for j in range(i + 1, n):
            diff = x1[:, i] - x2[:, j]
            euclidean_squared = np.matmul(diff, np.transpose(diff))
            fraction_term = euclidean_squared / (2 * alpha_rq * (length_rq ** 2))
            cov_matrix[i, j] = (1 + fraction_term) ** (-1 * alpha_rq)
            cov_matrix[j, i] = cov_matrix[i, j]

    return cov_matrix


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


def fast_matern_1_2d(sigma_matern, length_matern, x1, x2):
    """
    Much faster method of obtaining the Matern v=1/2 covariance matrix by exploiting the symmetry of the
    covariance matrix. This is the once-differentiable (zero mean squared differentiable) matern
    :param sigma_matern: Coefficient at the front
    :param length_matern: Length scale
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
            exp_term = np.exp(-1 * euclidean * (length_matern ** -1))
            cov_matrix[i, j] = (sigma_matern ** 2) * exp_term
            cov_matrix[j, i] = cov_matrix[i, j]

    return cov_matrix


def fast_squared_exp_2d(sigma_exp, length_exp, x1, x2):  # there are only two variables in the matern function
    """
    This is much much faster than iteration over every point beyond n = 10. This function takes advantage of the
    symmetry in the covariance matrix and allows for fast regeneration.
    :param sigma_exp: coefficient factor at the front
    :param length_exp: length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :return: Covariance matrix with squared exponential kernel - indicating infinite differentiability
    """
    # Note that this function only takes in 2-D coordinates, make sure there are 2 rows and n columns
    n = x1.shape[1]
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        cov_matrix[i, i] = sigma_exp ** 2
        for j in range(i + 1, n):
            diff = x1[:, i] - x2[:, j]
            euclidean = np.sqrt(np.matmul(diff, np.transpose(diff)))
            exp_power = np.exp(-1 * (euclidean ** 2) * (length_exp ** -2))
            cov_matrix[i, j] = (sigma_exp ** 2) * exp_power
            cov_matrix[j, i] = cov_matrix[i, j]

    return cov_matrix


def fast_rational_quadratic_2d(alpha_rq, length_rq, x1, x2):
    """
    Rational Quadratic Coveriance function with 2 parameters to be optimized, using
    power alpha and length scale l. The Rational Quadratic Kernel is used to model the
    volatility of equity index returns, which is equivalent to a sum of Squared
    Exponential Kernels. This kernel is used to model multi-scale data

    This is a fast method of generating the rational quadratic kernel, by exploiting the symmetry
    of the covariance matrix
    :param alpha_rq: power and denominator
    :param length_rq: length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :return: Covariance matrix with Rational Quadratic Kernel
    """
    # Note that this function only takes in 2-D coordinates, make sure there are 2 rows and n columns
    n = x1.shape[1]
    covariance_matrix = np.zeros((n, n))
    for i in range(n):
        covariance_matrix[i, i] = 1
        for j in range(i + 1, n):
            diff = x1[:, i] - x2[:, j]
            euclidean_squared = np.matmul(diff, np.transpose(diff))
            fraction_term = euclidean_squared / (2 * alpha_rq * (length_rq ** 2))
            covariance_matrix[i, j] = (1 + fraction_term) ** (-1 * alpha_rq)
            covariance_matrix[j, i] = covariance_matrix[i, j]

    return covariance_matrix


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
    2. Kernel: Matern 3/2, Matern 1/2, Squared Exponential and Rational Quadratic Kernels
    3. Assume a constant latent intensity, even at locations without any incidences
    :param param: hyperparameters - sigma, length scale and noise, prior scalar mean - array of 4 elements
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
    kernel_choice = args[2]

    # The Covariance Matrix and Prior mean are created here as a component of the objective function
    prior_mean = mean_func_scalar(scalar_mean, xy_coordinates)

    # Select Kernel and Construct Covariance Matrix
    if kernel_choice == 'matern3':
        c_auto = fast_matern_2d(sigma, length, xy_coordinates, xy_coordinates)
    elif kernel_choice == 'matern1':
        c_auto = fast_matern_1_2d(sigma, length, xy_coordinates, xy_coordinates)
    elif kernel_choice == 'squared_exponential':
        c_auto = fast_squared_exp_2d(sigma, length, xy_coordinates, xy_coordinates)
    elif kernel_choice == 'rational_quad':
        c_auto = fast_rational_quadratic_2d(sigma, length, xy_coordinates, xy_coordinates)
    else:
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


def log_poisson_likelihood_opt(param, *args):
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
    # factorial_term = - np.sum(np.log(factorial_k))  # summation of logs = log of product
    factorial_term = - np.sum(fn.log_special(factorial_k))  # summation of logs = log of product

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


def short_log_integrand_data(param, *args):
    """
    1. Shorter version that tabulates only the log of the GP prior. Includes only terms
    containing the covariance matrix elements that are made up of the kernel hyper-parameters
    2. Kernel: Matern(3/2), Matern(1/2), Squared Exponential
    3. Assume a constant latent intensity, even at locations without any incidences
    :param param: hyperparameters - sigma, length scale and noise, prior scalar mean - array of 4 elements
    :param args: xy coordinates for input into the covariance function and the histogram
    :return: the log of the GP Prior, log[N(prior mean, covariance matrix)]
    """
    # Generate Matern Covariance Matrix
    # Enter parameters
    sigma = param[0]
    length = param[1]
    noise = param[2]
    scalar_mean = param[3]

    # Enter Arguments - entered as a tuple
    xy_coordinates = args[0]
    data_array = args[1]  # Note that this is refers to the optimised log-intensity array
    kernel = args[2]

    # Set up inputs for generation of objective function
    p_mean = mean_func_scalar(scalar_mean, xy_coordinates)

    # Change_Param - change kernel by setting cases
    if kernel == 'matern3':
        c_auto = fast_matern_2d(sigma, length, xy_coordinates, xy_coordinates)
    elif kernel == 'matern1':
        c_auto = fast_matern_1_2d(sigma, length, xy_coordinates, xy_coordinates)
    elif kernel == 'squared_exponential':
        c_auto = fast_squared_exp_2d(sigma, length, xy_coordinates, xy_coordinates)
    elif kernel == 'rational_quad':
        c_auto = fast_rational_quadratic_2d(sigma, length, xy_coordinates, xy_coordinates)
    else:  # Default kernel is matern1
        c_auto = np.eye(data_array.shape[1])
        print('Check for appropriate kernel')

    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    cov_matrix = c_auto + c_noise

    """Generate Objective Function = log[g(v)]"""
    # Generate Determinant Term (after taking log)
    determinant = np.exp(np.linalg.slogdet(cov_matrix))[1]
    det_term = -0.5 * np.log(2 * np.pi * determinant)

    # Generate Euclidean Term (after taking log)
    data_diff = data_array - p_mean
    inv_covariance_matrix = np.linalg.inv(cov_matrix)
    euclidean_term = -0.5 * fn.matmulmul(data_diff, inv_covariance_matrix, data_diff)

    """Summation of all terms change to correct form to find minimum point"""
    log_gp = det_term + euclidean_term
    log_gp_minimization = -1 * log_gp  # Make the function convex for minimization
    return log_gp_minimization


def short_log_integrand_data_rq(param, *args):
    """
    Optimization using the Rational Quadratic Kernel, with hyper-parameters alpha and
    length scale, while taking in coordinates and histo quad as inputs
    :param param: alpha, length_scale
    :param args: Coordinates and values of data points after taking the histogram
    :return: the negative of the marginal log likelihood (which we then have to minimize)
    """
    # Generate Rational Quadratic Covariance Matrix
    # Enter parameters
    alpha = param[0]
    length = param[1]
    noise = param[2]
    scalar_mean = param[3]

    # Enter Arguments
    xy_coordinates = args[0]
    data_array = args[1]  # Note that this is refers to the optimised log-intensity array

    # Set up inputs for generation of objective function
    p_mean = mean_func_scalar(scalar_mean, xy_coordinates)

    # Create Rational Quadratic Covariance Matrix including noise
    c_auto = fast_rational_quadratic_2d(alpha, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    cov_matrix = c_auto + c_noise

    # Generate Determinant Term (after taking log)
    determinant = np.exp(np.linalg.slogdet(cov_matrix))[1]
    det_term = -0.5 * np.log(2 * np.pi * determinant)

    # Generate Euclidean Term (after taking log)
    data_diff = data_array - p_mean
    inv_covariance_matrix = np.linalg.inv(cov_matrix)
    euclidean_term = -0.5 * fn.matmulmul(data_diff, inv_covariance_matrix, data_diff)

    """Summation of all terms change to correct form to find minimum point"""
    log_gp = det_term + euclidean_term
    log_gp_minimization = -1 * log_gp  # Make the function convex for minimization
    return log_gp_minimization


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


def linear_trans_opt(param, *args):
    """
    Computes the Log Marginal Likelihood using standard GP regression by first performing transformation of the data set
    :param param: transform_mat
    :param args:
    :return:
    """
    # Define arguments
    x_scatter = args[0]
    y_scatter = args[1]
    c = args[2]
    kernel = args[3]

    # Define parameters to be optimized - the matrix variables
    transform_mat = param

    # Begin transformation of the regression window
    xy_scatter = np.vstack((x_scatter, y_scatter))  # Create the sample points to be rotated
    xy_scatter_transformed = fn.transform_array(transform_mat, xy_scatter, c)
    x_points_trans = xy_scatter_transformed[0]
    y_points_trans = xy_scatter_transformed[1]

    # 1. Obtain the maximum range in x and y in the transformed space - to define the regression window
    x_down = min(x_points_trans)
    x_up = max(x_points_trans)
    y_down = min(y_points_trans)
    y_up = max(y_points_trans)

    # --------------------- Conduct binning into transformed space - the x and y quad lengths will be different

    # ChangeParam
    quads_on_side = 20  # define the number of quads along each dimension
    k_mesh, y_edges, x_edges = np.histogram2d(y_points_trans, x_points_trans, bins=quads_on_side,
                                              range=[[y_down, y_up], [x_down, x_up]])
    x_mesh_plot, y_mesh_plot = np.meshgrid(x_edges, y_edges)  # creating mesh-grid for use
    x_mesh = x_mesh_plot[:-1, :-1]  # Removing extra rows and columns due to edges
    y_mesh = y_mesh_plot[:-1, :-1]
    x_quad = fn.row_create(x_mesh)  # Creating the rows from the mesh
    y_quad = fn.row_create(y_mesh)
    xy_quad = np.vstack((x_quad, y_quad))
    k_quad = fn.row_create(k_mesh)

    # Start Optimization
    arguments = (xy_quad, k_quad, kernel)

    # Initialise kernel hyper-parameters - arbitrary value but should be as close to actual value as possible
    initial_hyperparameters = np.array([1, 1, 1, 1])

    # An optimization process is embedded within another optimization process
    solution = scopt.minimize(fun=short_log_integrand_data, args=arguments, x0=initial_hyperparameters,
                              method='Nelder-Mead',
                              options={'xatol': 1, 'fatol': 100, 'disp': True, 'maxfev': 500})

    print('Last function evaluation is ', solution.fun)  # This will be a negative value
    neg_log_likelihood = solution.fun  # We want to minimize the mirror image
    return neg_log_likelihood


def linear_trans_skinny_opt(param, *args):
    """
    Computes the Log Marginal Likelihood using standard GP regression by first performing transformation of the data set
    This finds the average log marginal likelihood instead of the combined log_likelhood, and will find this average
    while adapting to the number of quadrats in the regression window after transformation
    :param param: transform_mat - matrix variables to be optimized
    :param args: x and y coordinates of scatter points,, center, kernel type, and array containing vertices
    the right order
    :return: the average log likelihood by dividing total log likelihood with number of selected quadrats
    """
    # Define original required arguments
    xy_scatter = args[0]
    c = args[1]
    kernel = args[2]
    vertex_array = args[3]  # Have to be in the right order in the original mathematical space

    # Define parameters to be optimized - the matrix variables
    transform_mat = param

    # Begin transformation of the regression window
    xy_scatter_transformed = fn.transform_array(transform_mat, xy_scatter, c)
    x_points_trans = xy_scatter_transformed[0]
    y_points_trans = xy_scatter_transformed[1]

    # 1. Obtain the maximum range in x and y in the transformed space - to define the regression window
    x_down = min(x_points_trans)
    x_up = max(x_points_trans)
    y_down = min(y_points_trans)
    y_up = max(y_points_trans)

    # Conduct binning into transformed space - the x and y quad lengths will be different

    # ChangeParam
    quads_on_side = 20  # define the number of quads along each dimension
    k_mesh, y_edges, x_edges = np.histogram2d(y_points_trans, x_points_trans, bins=quads_on_side,
                                              range=[[y_down, y_up], [x_down, x_up]])
    x_mesh_plot, y_mesh_plot = np.meshgrid(x_edges, y_edges)  # creating mesh-grid for use
    x_mesh = x_mesh_plot[:-1, :-1]  # Removing extra rows and columns due to edges
    y_mesh = y_mesh_plot[:-1, :-1]
    x_quad = fn.row_create(x_mesh)  # Creating the rows from the mesh
    y_quad = fn.row_create(y_mesh)
    xy_quad = np.vstack((x_quad, y_quad))
    k_quad = fn.row_create(k_mesh)

    # Selection of quadrats that fall inside the polygon

    # Transform the vertices using the same transformation matrix
    transformed_vertices = fn.transform_array(transform_mat, vertex_array, center)

    # Create polygon and
    polygon = mpath.Path(np.transpose(transformed_vertices))
    polygon_indicator = polygon.contains_points(np.transpose(xy_quad), transform=None, radius=1.0)

    x_quad_polygon = x_quad[polygon_indicator]
    y_quad_polygon = y_quad[polygon_indicator]
    xy_quad_polygon = np.vstack((x_quad_polygon, y_quad_polygon))
    k_quad_polygon = k_quad[polygon_indicator]

    # Begin Optimization using selected quadrats
    arguments = (xy_quad_polygon, k_quad_polygon, kernel)

    # Initialise kernel hyper-parameters - arbitrary value but should be as close to actual value as possible
    initial_hyperparameters = np.array([1, 1, 1, 1])

    # An optimization process is embedded within another optimization process
    solution = scopt.minimize(fun=short_log_integrand_data, args=arguments, x0=initial_hyperparameters,
                              method='Nelder-Mead',
                              options={'xatol': 1, 'fatol': 100, 'disp': True, 'maxfev': 1000})

    positive_log_likelihood = solution.fun  # We want to minimize the mirror image
    selected_quadrats_n = k_quad_polygon.size
    avg_positive_log_likelihood = positive_log_likelihood / selected_quadrats_n
    print('Last function evaluation is ', solution.fun)  # This will be a negative value
    print('The number of selected quadrats inside polygon is', selected_quadrats_n)
    return avg_positive_log_likelihood


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
year = '2014'
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

# Perform scatter point selection within the regression window
x_within_box = x_points[x_box & y_box]
y_within_box = y_points[x_box & y_box]
xy_within_box = np.vstack((x_within_box, y_within_box))

# ------------------------------------------ Start the optimization process
# Try to use Latin Hypercube sampling to ensure global optimization

# Select kernel
ker = 'matern1'

# Define vertices in the original mathematical space
vertices = np.array([[x_lower, x_lower, x_upper, x_upper], [y_lower, y_upper, y_upper, y_lower]])

# Iterate each for each of the 4 matrix variables from 0 to 5
# ChangeParam
element_skew = np.arange(1, 6.1, 0.1)  # 4 values for each, and cubed
iterate_count = element_skew.size

# Initialise holding arrays
log_likelihood = np.zeros_like(element_skew)
avg_log_likelihood = np.zeros_like(element_skew)
selected_quadrats_n = np.zeros_like(element_skew)

# Initialise Array containing Frobenius Norm
frob_norm = np.zeros_like(element_skew)

start_iteration = time.clock()

# Over here, I am not trying to optimize for matrix variables, but just optimizing for the kernel
for i in range(iterate_count):
    # initial_mat_var = np.array([mat_element[a], 0, mat_element[c], mat_element[d]])
    start_iteration = time.clock()
    initial_mat_var = np.array([element_skew[i], 0, 0, element_skew[i]])
    frob_norm[i] = fn.frob_norm(initial_mat_var)
    print(' ------------- Start of Current Iteration', i+1)
    print('The Current Matrix Variables are', initial_mat_var)
    print('The Current Frobenius Norm is', frob_norm[i])

    xy_scatter_transformed = fn.transform_array(initial_mat_var, xy_within_box, center)
    x_points_trans = xy_scatter_transformed[0]
    y_points_trans = xy_scatter_transformed[1]

    # Obtain the maximum range in x and y in the transformed space
    # Transform the vertices
    transformed_vertices = fn.transform_array(initial_mat_var, vertices, center)

    x_down = min(transformed_vertices[0])
    x_up = max(transformed_vertices[0])
    y_down = min(transformed_vertices[1])
    y_up = max(transformed_vertices[1])

    # ChangeParam - create histogram in transformed space before quadrat selection
    quads_on_side = 20  # define the number of quads along each dimension
    k_mesh, y_edges, x_edges = np.histogram2d(y_points_trans, x_points_trans, bins=quads_on_side,
                                              range=[[y_down, y_up], [x_down, x_up]])
    x_mesh_plot, y_mesh_plot = np.meshgrid(x_edges, y_edges)  # creating mesh-grid for use
    x_mesh = x_mesh_plot[:-1, :-1]  # Removing extra rows and columns due to edges
    y_mesh = y_mesh_plot[:-1, :-1]
    x_quad = fn.row_create(x_mesh)  # Creating the rows from the mesh
    y_quad = fn.row_create(y_mesh)

    k_quad = fn.row_create(k_mesh)

    # Realign the quad coordinates to the centers - shift centers by half a quad length on either dimension
    quad_length_x = (x_upper - x_lower) / quads_on_side
    quad_length_y = (y_upper - y_lower) / quads_on_side
    x_quad = x_quad + (0.5 * quad_length_x)
    y_quad = y_quad + (0.5 * quad_length_y)
    xy_quad = np.vstack((x_quad, y_quad))

    # Create Polygon using the transformed_vertices
    polygon = mpath.Path(np.transpose(transformed_vertices))

    # Create Boolean array which is the polygon indicator
    polygon_indicator = polygon.contains_points(np.transpose(xy_quad), transform=None, radius=0.2)

    # Begin Quadrat Selection
    x_quad_polygon = x_quad[polygon_indicator]
    y_quad_polygon = y_quad[polygon_indicator]
    xy_quad_polygon = np.vstack((x_quad_polygon, y_quad_polygon))

    k_quad_polygon = k_quad[polygon_indicator]
    selected_quadrats_n[i] = k_quad_polygon.size
    print('The number of selected quadrats is', selected_quadrats_n[i])

    # Begin Optimization using selected quadrats
    arguments = (xy_quad_polygon, k_quad_polygon, ker)

    # Initialise kernel hyper-parameters
    initial_hyperparameters = np.array([1, 1, 1, 1])

    # Optimise for kernel hyperparameters
    solution = scopt.minimize(fun=short_log_integrand_data, args=arguments, x0=initial_hyperparameters,
                              method='Nelder-Mead',
                              options={'xatol': 1, 'fatol': 20, 'disp': True, 'maxfev': None})

    # Divide the log_likelihood by the number of selected quadrats
    # Taking the true negative value of the log_likelihoods
    log_likelihood[i] = -1 * solution.fun
    avg_log_likelihood[i] = log_likelihood[i] / selected_quadrats_n[i]

    end_iteration = time.clock()
    print('Time taken for this iteration is', end_iteration - start_iteration)
    print('The Log Likelihood is', log_likelihood[i])  # This will be a negative value
    print('The Average Log Likelihood is', avg_log_likelihood[i])


print('The 4-dimensional matrix containing Log Likelihood is', log_likelihood)  # This will be a negative value
print('The 4-dimensional matrix containing number of selected quadrats inside polygon is', selected_quadrats_n)
print('The 4-dimensional matrix containing average Log Likelihood is', avg_log_likelihood)
print('The 4-dimensional matrix containing frobenius norm is', frob_norm)

end_iteration = time.clock()
time_iteration = end_iteration - start_iteration
print('The time taken for iteration is', time_iteration)

# Select the optimal starting points, and the optimal matrix variables corresponding to greatest Log Likelihood
total_opt_index = np.argmax(log_likelihood)
max_total_likelihood = avg_log_likelihood[total_opt_index]
opt_selected_quadrats_total = selected_quadrats_n[total_opt_index]
total_opt_var = np.array([1, element_skew[total_opt_index], element_skew[total_opt_index], 1])


# This is for the average
avg_opt_index = np.argmax(avg_log_likelihood)
max_avg_likelihood = avg_log_likelihood[avg_opt_index]
avg_opt_var = np.array([1, element_skew[avg_opt_index], element_skew[avg_opt_index], 1])

# Record the optimal matrix variables in both cases
print('The globally-optimal matrix variables in terms of total are', total_opt_var)
print('The total globally-optimal log marginal likelihood is', max_total_likelihood)
print('The globally-optimal matrix variables in terms of average are', avg_opt_var)
print('The average globally-optimal log marginal likelihood is', max_avg_likelihood)


# Plot total likelihood against frobenius norm
likelihood_frob_fig = plt.figure()
likelihood_frob = likelihood_frob_fig.add_subplot(111)
likelihood_frob.plot(frob_norm, log_likelihood, color='black')
likelihood_frob.set_xlabel('Frobenius Norm')
likelihood_frob.set_ylabel('Combined Log Marginal Likelihood')

# Plot total likelihood against skew value
likelihood_skew_fig = plt.figure()
likelihood_skew = likelihood_skew_fig.add_subplot(111)
likelihood_skew.plot(element_skew, log_likelihood, color='black')
likelihood_skew.set_xlabel('Diagonal Entry - Alpha')
likelihood_skew.set_ylabel('Combined Log Marginal Likelihood')

# Plot average likelihood against frobenius norm
avg_likelihood_frob_fig = plt.figure()
avg_likelihood_frob = avg_likelihood_frob_fig.add_subplot(111)
avg_likelihood_frob.plot(frob_norm, avg_log_likelihood, color='black')
avg_likelihood_frob.set_xlabel('Frobenius Norm')
avg_likelihood_frob.set_ylabel('Average Log Marginal Likelihood')

# Plot average likelihood against frobenius norm
avg_likelihood_skew_fig = plt.figure()
avg_likelihood_skew = avg_likelihood_skew_fig.add_subplot(111)
avg_likelihood_skew.plot(element_skew, avg_log_likelihood, color='black')
avg_likelihood_skew.set_xlabel('Diagonal Entry - Alpha')
avg_likelihood_skew.set_ylabel('Average Log Marginal Likelihood')
avg_likelihood_skew_fig.savefig('Temporary_alpha_iterate.png')

plt.show()




