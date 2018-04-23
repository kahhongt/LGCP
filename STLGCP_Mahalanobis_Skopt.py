import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn
import time
import scipy.special as scispec
import scipy.optimize as scopt
import skopt as skp
from collections import Counter

matplotlib.use('TkAgg')
import matplotlib.cm as cmx
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
def fast_matern_3_2d(sigma_matern, length_matern, x1, x2):  # there are only two variables in the matern function
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


def fast_matern_3_1d(length_matern, x1, x2):  # there are only two variables in the matern function
    """
    This is much much faster than iteration over every point beyond n = 10. This function takes advantage of the
    symmetry in the covariance matrix and allows for fast regeneration. For this function, v = 3/2
    :param sigma_matern: coefficient factor at the front
    :param length_matern: length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :return: Covariance matrix with matern kernel
    """
    # Takes in 1D coordinates
    n = x1.size
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        cov_matrix[i, i] = 1
        for j in range(i + 1, n):
            diff = x1[i] - x2[j]
            euclidean = np.sqrt(diff ** 2)
            coefficient_term = (1 + np.sqrt(3) * euclidean * (length_matern ** -1))
            exp_term = np.exp(-1 * np.sqrt(3) * euclidean * (length_matern ** -1))
            cov_matrix[i, j] = 1 * coefficient_term * exp_term
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
    # Create cases to accommodate 1-D arrays
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


def fast_matern_1_1d(length_matern, x1, x2):
    """
    Much faster method of obtaining the Matern v=1/2 covariance matrix by exploiting the symmetry of the
    covariance matrix. This is the once-differentiable (zero mean squared differentiable) matern
    :param sigma_matern: Coefficient at the front
    :param length_matern: Length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :return: Covariance matrix with matern kernel
    """
    # Only takes in 1-D arrays
    n = x1.size
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        cov_matrix[i, i] = 1
        for j in range(i + 1, n):
            diff = x1[i] - x2[j]
            euclidean = np.sqrt(diff ** 2)
            exp_term = np.exp(-1 * euclidean * (length_matern ** -1))
            cov_matrix[i, j] = 1 * exp_term
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


def fast_squared_exp_1d(length_exp, x1, x2):  # there are only two variables in the matern function
    """
    This is much much faster than iteration over every point beyond n = 10. This function takes advantage of the
    symmetry in the covariance matrix and allows for fast regeneration.
    :param length_exp: length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :return: Covariance matrix with squared exponential kernel - indicating infinite differentiability
    """
    # Only take sin 1-D arrays
    n = x1.size
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        cov_matrix[i, i] = 1
        for j in range(i + 1, n):
            diff = x1[i] - x2[j]
            euclidean = np.sqrt(diff ** 2)
            exp_power = np.exp(-1 * (euclidean ** 2) * (length_exp ** -2))
            cov_matrix[i, j] = exp_power
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


def fast_rational_quadratic_1d(alpha_rq, length_rq, x1, x2):
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
    n = x1.size
    covariance_matrix = np.zeros((n, n))
    for i in range(n):
        covariance_matrix[i, i] = 1
        for j in range(i + 1, n):
            diff = x1[i] - x2[j]
            euclidean_squared = diff ** 2
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
    c_auto = fast_matern_3_2d(sigma, length, xy_coordinates, xy_coordinates)
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
    c_auto = fast_matern_3_2d(sigma, length, xy_coordinates, xy_coordinates)
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
    c_auto = fast_matern_3_2d(sigma, length, xy_coordinates, xy_coordinates)
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
        c_auto = fast_matern_3_2d(sigma, length, xy_coordinates, xy_coordinates)
    elif kernel_choice == 'matern1':
        c_auto = fast_matern_1_2d(sigma, length, xy_coordinates, xy_coordinates)
    elif kernel_choice == 'squared_exponential':
        c_auto = fast_squared_exp_2d(sigma, length, xy_coordinates, xy_coordinates)
    elif kernel_choice == 'rational_quad':
        c_auto = fast_rational_quadratic_2d(sigma, length, xy_coordinates, xy_coordinates)
    else:
        c_auto = fast_matern_3_2d(sigma, length, xy_coordinates, xy_coordinates)

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
    :param param: v_array containing the log latent intensities
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


def log_poisson_likelihood_large(param, *args):
    """
    Considers only the Poisson likelihood in front of the GP. The input parameter is the latent intensity
    instead of the log-latent intensity v
    :param param: lambda_array containing the latent intensities
    :param args: data set
    :return: the combined log poisson likelihood
    """
    v_array = param
    k_array = args[0]

    # Generate Objective Function: log(P(D|v))
    exp_term = -1 * np.sum(np.exp(v_array))
    product_term = sum(v_array * k_array)

    factorial_components = np.zeros_like(k_array)
    for i in range(k_array.size):
        factorial_components[i] = math.lgamma(k_array[0] + 1)

    factorial_term = -1 * sum(factorial_components)

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
    p_array = args[1]  # This is the arbitrary vector
    # Generate Hessian Product without creating the hessian
    exp_v_array = np.exp(v_array)
    hessian_product = -1 * exp_v_array * p_array
    hessian_product_convex = -1 * hessian_product
    return hessian_product_convex


def hessian_log_likelihood(param, *args):
    """
    Generates hessian matrix
    :param param: v_array containing the latent intensities
    :return: vector containing the hessian matrix
    """
    # Define parameters, no arguments here
    v_array = param

    # Generate Hessian Product without creating the hessian
    exp_v_array = np.exp(v_array)
    hessian_matrix = np.diag(-1 * exp_v_array)
    hessian_matrix_convex = -1 * hessian_matrix
    return hessian_matrix_convex


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
        c_auto = fast_matern_3_2d(sigma, length, xy_coordinates, xy_coordinates)
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

    # Define parameters to be optimized
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
    quads_on_side = 10  # define the number of quads along each dimension
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

    # Initialise kernel hyper-parameters - arbitrary value
    initial_hyperparameters = np.array([3, 2, 1, 1])

    # An optimization process is embedded within another optimization process
    solution = scopt.minimize(fun=short_log_integrand_data, args=arguments, x0=initial_hyperparameters,
                              method='Nelder-Mead',
                              options={'xatol': 1, 'fatol': 1, 'disp': True, 'maxfev': 1000})

    print('Last function evaluation is ', solution.fun)  # This will be a negative value
    neg_log_likelihood = -1 * solution.fun  # We want to minimize the mirror image
    return neg_log_likelihood


# ------------------------------------------ FUNCTIONS FOR SPATIAL TEMPORAL LGCP


def product_kernel_3d(sigma_p, length_space_p, length_time_p, xy_p, t_p, kernel_s, kernel_t):
    """
    Takes the product of a 2-D spatial kernel and 1-D time kernel and returns the covariance matrix with n x n,
    where n represents the total number of voxels
    *** Note that I can simply multiply the spatial and temporal matrices directly - element-wise
    Note that both spatial and temporal covariance matrices have the same dimensions due to
    the mesh grid that results in more repetitions in
    :param sigma_p: overall kernel amplitude
    :param length_space_p: length scale in spatial kernel
    :param length_time_p: length scale in time kernel
    :param xy_p: spatial coordinates with 2 rows
    :param t_p: time coordinates with only 1 row
    :param kernel_s: spatial kernel
    :param kernel_t: temporal kernel
    :return: auto-covariance matrix with n x n dimensions
    """
    # Generate spatial covariance matrix
    if kernel_s == 'matern1':
        spatial_cov_matrix = fast_matern_1_2d(sigma_p, length_space_p, xy_p, xy_p)
    elif kernel_s == 'matern3':
        spatial_cov_matrix = fast_matern_3_2d(sigma_p, length_space_p, xy_p, xy_p)
    elif kernel_s == 'squared_exponential':
        spatial_cov_matrix = fast_squared_exp_2d(sigma_p, length_space_p, xy_p, xy_p)
    elif kernel_s == 'rational_quad':
        spatial_cov_matrix = fast_rational_quadratic_2d(sigma_p, length_space_p, xy_p, xy_p)
    else:
        spatial_cov_matrix = np.zeros(t_p.size, t_p.size)
        print('No appropriate spatial kernel selected')

    # Generate temporal covariance matrix
    if kernel_t == 'matern1':
        temporal_cov_matrix = fast_matern_1_1d(length_time_p, t_p, t_p)
    elif kernel_t == 'matern3':
        temporal_cov_matrix = fast_matern_3_1d(length_time_p, t_p, t_p)
    elif kernel_t == 'squared_exponential':
        temporal_cov_matrix = fast_squared_exp_1d(length_time_p, t_p, t_p)
    else:
        temporal_cov_matrix = np.zeros(t_p.size, t_p.size)
        print('No appropriate temporal kernel selected')

    # Create overall kernel product covariance matrix
    kernel_product_matrix = spatial_cov_matrix * temporal_cov_matrix
    return kernel_product_matrix


def gp_likelihood_3d(param, *args):
    """
    Returns the Log_likelihood for the Spatial Temporal LGCP after obtaining the latent intensities
    :param param: hyperparameters - sigma, length scale and noise, prior scalar mean - array of 4 elements
    :param args: xy coordinates for input into the covariance function and the optimised v_array
    :return: the log of the GP Prior, log[N(prior mean, covariance matrix)]
    """
    # Generate Matern Covariance Matrix
    # Enter parameters - there are now 5 parameters to optimize
    sigma = param[0]
    length_space = param[1]
    length_time = param[2]
    noise = param[3]
    scalar_mean = param[4]
    # alpha = param[5]

    # There are 5 arguments to be entered
    xy_coord = args[0]
    t_coord = args[1]
    v_array = args[2]
    spatial_kernel = args[3]
    time_kernel = args[4]

    # Create prior mean array
    prior_mean = mean_func_scalar(scalar_mean, t_coord)

    # Construct spatial kernel
    c_auto = product_kernel_3d(sigma, length_space, length_time, xy_coord, t_coord,
                               spatial_kernel, time_kernel)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    cov_matrix = c_auto + c_noise

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


def gp_3d_mahalanobis(param, *args):
    """
    Returns the Log_likelihood for the Spatial Temporal LGCP after obtaining the latent intensities
    - using the Malahanobis distance metric
    :param param: hyperparameters - sigma, length scale and noise, prior scalar mean - array of 4 elements
    :param args: xy coordinates for input into the covariance function and the optimised v_array
    :return: the log of the GP Prior, log[N(prior mean, covariance matrix)]
    """
    # Generate Matern Covariance Matrix
    # Enter parameters - there are now 5 parameters to optimize
    sigma = param[0]
    length = param[1]
    noise = param[2]
    scalar_mean = param[3]
    matrix_tup = param[4:]  # Include the matrix array now - have to create the tuple beforehand

    # There are 3 arguments to be entered - use vstack to create 3 input rows - x, y and z
    xyt_coord = args[0]
    v_array = args[1]  # This is the optimized v_array
    kernel = args[2]  # Kernel chosen for the cases below

    # Create prior mean array
    prior_mean = mean_func_scalar(scalar_mean, xyt_coord[0])

    # Construct covariance function with the 3D kernel
    if kernel == 'matern1':
        c_auto = fast_matern_1_3d(sigma, length, xyt_coord, xyt_coord, matrix_tup)
    elif kernel == 'matern3':
        c_auto = fast_matern_3_3d(sigma, length, xyt_coord, xyt_coord, matrix_tup)
    elif kernel == 'squared_exponential':
        c_auto = fast_squared_exp_3d(sigma, length, xyt_coord, xyt_coord, matrix_tup)
    elif kernel == 'rational_quad':
        c_auto = fast_rational_quadratic_3d(sigma, length, xyt_coord, xyt_coord, matrix_tup)
    else:
        c_auto = np.eye(v_array.size)
        print('No appropriate kernel found')

    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    cov_matrix = c_auto + c_noise

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


# Below are functions for 3D kernels, and incorporates the mahalanobis distance in the distance metric

def fast_matern_1_3d(sigma_matern, length_matern, x1, x2, matrix_tuple):
    """
    Much faster method of obtaining the Matern v=1/2 covariance matrix by exploiting the symmetry of the
    covariance matrix. This is the once-differentiable (zero mean squared differentiable) matern
    3-DIMENSIONAL MATERN KERNEL WITH MAHALANOBIS DISTANCE METRIC
    :param sigma_matern: Coefficient at the front
    :param length_matern: Length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :param matrix_tuple: takes in a tuple containing the 6 distinct matrix variables
    :return: Covariance matrix with matern kernel
    """
    # Note that this function only takes in 3-D coordinates, make sure there are 3 rows and n columns
    # Takes into account the two spatial and one time dimensions

    # Count the number of columns - make sure to use vstack before using x1 and x2
    n = x1.shape[1]

    # Create Mahalanobis transformation matrix
    a = matrix_tuple[0]
    b = matrix_tuple[1]
    c = matrix_tuple[2]
    d = matrix_tuple[3]
    e = matrix_tuple[4]
    f = matrix_tuple[5]
    mat = np.array([[a, b, c], [b, d, e], [c, e, f]])

    cov_matrix = np.zeros((n, n))
    for i in range(n):
        cov_matrix[i, i] = sigma_matern ** 2
        for j in range(i + 1, n):
            diff = x1[:, i] - x2[:, j]

            # Implement mahalanobis distance metric
            array_product = fn.matmulmul(diff, mat, np.transpose(diff))
            euclidean = np.sqrt(array_product)

            exp_term = np.exp(-1 * euclidean * (1 / length_matern))
            cov_matrix[i, j] = (sigma_matern ** 2) * exp_term
            cov_matrix[j, i] = cov_matrix[i, j]

    return cov_matrix


def fast_matern_3_3d(sigma_matern, length_matern, x1, x2, matrix_tuple):  # there are only two variables in the matern function
    """
    This is much much faster than iteration over every point beyond n = 10. This function takes advantage of the
    symmetry in the covariance matrix and allows for fast regeneration. For this function, v = 3/2
    :param sigma_matern: coefficient factor at the front
    :param length_matern: length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :param matrix_tuple: tuple containing the 6 distinct matrix variables
    :return: Covariance matrix with matern kernel
    """
    # Note that this function only takes in 2-D coordinates, make sure there are 2 rows and n columns
    n = x1.shape[1]

    # Create Mahalanobis transformation matrix
    a = matrix_tuple[0]
    b = matrix_tuple[1]
    c = matrix_tuple[2]
    d = matrix_tuple[3]
    e = matrix_tuple[4]
    f = matrix_tuple[5]
    mat = np.array([[a, b, c], [b, d, e], [c, e, f]])

    cov_matrix = np.zeros((n, n))
    for i in range(n):
        cov_matrix[i, i] = sigma_matern ** 2
        for j in range(i + 1, n):
            diff = x1[:, i] - x2[:, j]

            # Implement mahalanobis distance metric
            array_product = fn.matmulmul(diff, mat, np.transpose(diff))
            euclidean = np.sqrt(array_product)

            coefficient_term = (1 + np.sqrt(3) * euclidean * (length_matern ** -1))
            exp_term = np.exp(-1 * np.sqrt(3) * euclidean * (length_matern ** -1))
            cov_matrix[i, j] = (sigma_matern ** 2) * coefficient_term * exp_term
            cov_matrix[j, i] = cov_matrix[i, j]

    return cov_matrix


def fast_squared_exp_3d(sigma_exp, length_exp, x1, x2, matrix_tuple):  # there are only two variables in the matern function
    """
    This is much much faster than iteration over every point beyond n = 10. This function takes advantage of the
    symmetry in the covariance matrix and allows for fast regeneration.
    :param sigma_exp: coefficient factor at the front
    :param length_exp: length scale
    :param x1: First set of coordinates for iteration
    :param x2: Second set of coordinates for iteration
    :param matrix_tuple: tuple containing the 6 distinct matrix variables
    :return: Covariance matrix with squared exponential kernel - indicating infinite differentiability
    """
    # Note that this function only takes in 2-D coordinates, make sure there are 2 rows and n columns
    n = x1.shape[1]

    # Create Mahalanobis transformation matrix
    a = matrix_tuple[0]
    b = matrix_tuple[1]
    c = matrix_tuple[2]
    d = matrix_tuple[3]
    e = matrix_tuple[4]
    f = matrix_tuple[5]
    mat = np.array([[a, b, c], [b, d, e], [c, e, f]])

    cov_matrix = np.zeros((n, n))
    for i in range(n):
        cov_matrix[i, i] = sigma_exp ** 2
        for j in range(i + 1, n):
            diff = x1[:, i] - x2[:, j]

            # Implement mahalanobis distance metric
            array_product = fn.matmulmul(diff, mat, np.transpose(diff))
            euclidean = np.sqrt(array_product)

            exp_power = np.exp(-1 * (euclidean ** 2) * (length_exp ** -2))
            cov_matrix[i, j] = (sigma_exp ** 2) * exp_power
            cov_matrix[j, i] = cov_matrix[i, j]

    return cov_matrix


def fast_rational_quadratic_3d(alpha_rq, length_rq, x1, x2, matrix_tuple):
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
    :param matrix_tuple: tuple containing the 6 distinct matrix variables
    :return: Covariance matrix with Rational Quadratic Kernel
    """
    # Note that this function only takes in 2-D coordinates, make sure there are 2 rows and n columns
    n = x1.shape[1]

    # Create Mahalanobis transformation matrix
    a = matrix_tuple[0]
    b = matrix_tuple[1]
    c = matrix_tuple[2]
    d = matrix_tuple[3]
    e = matrix_tuple[4]
    f = matrix_tuple[5]
    mat = np.array([[a, b, c], [b, d, e], [c, e, f]])

    covariance_matrix = np.zeros((n, n))
    for i in range(n):
        covariance_matrix[i, i] = 1
        for j in range(i + 1, n):
            diff = x1[:, i] - x2[:, j]

            # Implement mahalanobis distance metric
            array_product = fn.matmulmul(diff, mat, np.transpose(diff))

            # Combine components of kernel
            fraction_term = array_product / (2 * alpha_rq * (length_rq ** 2))
            covariance_matrix[i, j] = (1 + fraction_term) ** (-1 * alpha_rq)
            covariance_matrix[j, i] = covariance_matrix[i, j]

    return covariance_matrix


# ------------------------------------------ DATA COLLECTION STAGE
# Aedes Occurrences in Brazil
aedes_df = pd.read_csv('Aedes_PP_Data.csv')  # generates dataframe from csv - zika data

# Setting boolean variables required for the data
taiwan = aedes_df['COUNTRY'] == "Taiwan"
year_2014 = aedes_df['YEAR'] == "2014"
year_2013 = aedes_df['YEAR'] == "2013"
year_2012 = aedes_df['YEAR'] == "2012"

# Extract Data for Taiwan
aedes_taiwan = aedes_df[taiwan]  # Extract Taiwan data
x_taiwan = aedes_taiwan.values[:, 5].astype('float64')
y_taiwan = aedes_taiwan.values[:, 4].astype('float64')
year_taiwan = aedes_taiwan.values[:, 6].astype('float64')

# Define range of 3-D regression window
year_lower = 2003.5
year_upper = 2014
x_lower = 120.295
x_upper = 120.595
y_lower = 22.595
y_upper = 22.895

# Select Scatter Points within regression window

# Boolean arrays
year_range = (year_lower < year_taiwan) & (year_taiwan < year_upper)
x_range = (x_lower < x_taiwan) & (x_taiwan < x_upper)
y_range = (y_lower < y_taiwan) & (y_taiwan < y_upper)

# Selection of point using boolean arrays
x_taiwan_selected = x_taiwan[year_range & x_range & y_range]
y_taiwan_selected = y_taiwan[year_range & x_range & y_range]
t_taiwan_selected = year_taiwan[year_range & x_range & y_range]  # time is t - which is in years

print('The number of scatter points is', x_taiwan_selected.size)

# Create array for 3-D histogram clustering
xy_taiwan_selected = np.vstack((x_taiwan_selected, y_taiwan_selected))
xyt_taiwan_selected = np.vstack((xy_taiwan_selected, t_taiwan_selected))

# ChangeParam
vox_on_side = 10
k_mesh, xyt_edges = np.histogramdd(np.transpose(xyt_taiwan_selected), bins=(vox_on_side, vox_on_side, vox_on_side),
                                   range=((x_lower, x_upper), (y_lower, y_upper), (year_lower, year_upper)))

x_edges = xyt_edges[0][:-1]
y_edges = xyt_edges[1][:-1]
t_edges = xyt_edges[2][:-1]
x_mesh, y_mesh, t_mesh = np.meshgrid(x_edges, y_edges, t_edges)
x_vox = fn.row_create(x_mesh)
y_vox = fn.row_create(y_mesh)
t_vox = fn.row_create(t_mesh)
k_vox = fn.row_create(k_mesh)  # This is the original data set
xy_vox = np.vstack((x_vox, y_vox))
xyt_vox = np.vstack((xy_vox, t_vox))
print('k_vox shape is', k_vox.shape)
print("Initial Data Points are ", k_vox)

# Initialise arguments and parameters for optimization
# Arbitrary vector for optimization using the Newton-CG optimization algorithm
initial_p_array = np.ones_like(k_vox)

# Choose appropriate starting point for the optimization - it is reasonable to assume that a good starting point would
# be the log of the initial data values
initial_v_array = fn.log_special(k_vox)

# Tuple containing all arguments to be passed into objective function, jacobian and hessian, but we can specify
# which arguments will be used for each function
arguments_v = (k_vox, initial_p_array)

start_poisson_opt = time.clock()

# Start Optimization Algorithm for latent intensities - note this does not take into account the intensity locations
# ChangeParam
poisson_opt_method = 'Newton-CG'
if poisson_opt_method == 'Newton-CG':  # uses Jacobian and Hessian Product
    v_solution = scopt.minimize(fun=log_poisson_likelihood_large, args=arguments_v, x0=initial_v_array,
                                method='Newton-CG',
                                jac=gradient_log_likelihood,
                                hessp=hessianproduct_log_likelihood,
                                options={'xtol': 0.1, 'disp': True, 'maxiter': 10000000})

elif poisson_opt_method == 'Nelder-Mead':
    v_solution = scopt.minimize(fun=log_poisson_likelihood_large, args=arguments_v, x0=initial_v_array,
                                method='Nelder-Mead',
                                options={'disp': True, 'xatol': 0.01, 'fatol': 0.01, 'maxfev': None})

elif poisson_opt_method == 'BFGS':  # uses Jacobian only
    v_solution = scopt.minimize(fun=log_poisson_likelihood_large, args=arguments_v, x0=initial_v_array,
                                method='BFGS', jac=gradient_log_likelihood,
                                options={'disp': True, 'gtol': 0.00001, 'eps': 0.00000001,
                                         'return_all': False,
                                         'maxiter': None})

elif poisson_opt_method == 'dogleg':  # uses Jacobian and Hessian Matrix
    v_solution = scopt.minimize(fun=log_poisson_likelihood_large, args=arguments_v, x0=initial_v_array,
                                method='dogleg', jac=gradient_log_likelihood, hess=hessian_log_likelihood, options={})

latent_v_vox = v_solution.x  # v_array is the log of the latent intensity
p_likelihood = -1 * v_solution.fun
avg_p_likelihood = p_likelihood / k_vox.size
end_poisson_opt = time.clock()
print('The latent v array is', latent_v_vox)
print('Time taken for Poisson Optimization is', end_poisson_opt - start_poisson_opt)
print('The Maximum Log Likelihood is', p_likelihood)
print('The Average Log Likelihood is', avg_p_likelihood)
print('The Poisson optimization methods is', poisson_opt_method)
print('Latent Intensity Array Optimization Completed')

"""
index = np.arange(0, latent_v_vox.size, 1)
test_fig = plt.figure()
test = test_fig.add_subplot(111)
test.plot(index, latent_v_vox, color='black')
plt.show()
"""

# -------------------------------------------------------------------- START OF KERNEL OPTIMIZATION
start_gp_opt = time.clock()
# The parameters are sigma, length_space, length_time, noise, scalar_mean, and alpha
initial_kernel_scalar = 1
# maybe have a different start point for the matrix variables
initial_kernel_param = np.ones(4) * initial_kernel_scalar

# Have a different starting point for the matrix variables
initial_mat_param = np.array([1, 0, 0, 1, 0, 1])  # start off with the identity matrix
# initial_mat_param = np.array([1, 1, 1, 1, 1, 1])
# initial_mat_param = np.array([1, 1, 1, 1, 1, 1]) * 2

initial_all_param = np.append(initial_kernel_param, initial_mat_param)

# ChangeParam
ker = 'matern1'
opt_method = 'GP'
print('Kernel is', ker)
print('Optimizing Kernel Hyper-parameters...')
print('Vox per side is', vox_on_side)
print('Optimization method is', opt_method)

args_param = (xyt_vox, latent_v_vox, ker)  # tuple

kernel_bounds = (-5.0, 5.0)
kernel_bounds_m = sum(kernel_bounds) / 2
mahala_bounds_diag = (1.0, 2.0)
mahala_bounds_diag_m = sum(mahala_bounds_diag) / 2
mahala_bounds_skew = (0.0, 0.5)
mahala_bounds_skew_m = sum(mahala_bounds_skew) / 2

list_of_bounds = [kernel_bounds, kernel_bounds, kernel_bounds, kernel_bounds,
                  mahala_bounds_diag, mahala_bounds_skew, mahala_bounds_skew,
                  mahala_bounds_diag, mahala_bounds_skew, mahala_bounds_diag]

middle_of_bounds = [kernel_bounds_m, kernel_bounds_m, kernel_bounds_m, kernel_bounds_m,
                    mahala_bounds_diag_m, mahala_bounds_skew_m, mahala_bounds_skew_m,
                    mahala_bounds_diag_m, mahala_bounds_skew_m, mahala_bounds_diag_m]


# -------------------------------------------- CREATE NEW FUNCTION FOR GP OPTIMIZATION - BAYESIAN USING SKOPT
# Define new function here for definition - to be used for GP opt_method
def gp_3d_mahalanobis_skopt(param):
    """
    Returns the Log_likelihood for the Spatial Temporal LGCP after obtaining the latent intensities
    - using the Malahanobis distance metric
    - takes in arguments implicitly, as the skopt package does not easily allow arguments to be passed
    :param param: hyperparameters - sigma, length scale and noise, prior scalar mean - array of 4 elements
    :param args: xy coordinates for input into the covariance function and the optimised v_array
    :return: the log of the GP Prior, log[N(prior mean, covariance matrix)]
    """
    # Generate Matern Covariance Matrix
    # Enter parameters - there are now 5 parameters to optimize
    sigma = param[0]
    length = param[1]
    noise = param[2]
    scalar_mean = param[3]
    matrix_tup = param[4:]  # Include the matrix array now - have to create the tuple beforehand

    # There are 3 arguments to be entered - use vstack to create 3 input rows - x, y and z
    global args_param
    xyt_coord = args_param[0]
    v_array = args_param[1]  # This is the optimized v_array
    kernel = args_param[2]  # Kernel chosen for the cases below

    # Create prior mean array
    prior_mean = mean_func_scalar(scalar_mean, xyt_coord[0])

    # Construct covariance function with the 3D kernel
    if kernel == 'matern1':
        c_auto = fast_matern_1_3d(sigma, length, xyt_coord, xyt_coord, matrix_tup)
    elif kernel == 'matern3':
        c_auto = fast_matern_3_3d(sigma, length, xyt_coord, xyt_coord, matrix_tup)
    elif kernel == 'squared_exponential':
        c_auto = fast_squared_exp_3d(sigma, length, xyt_coord, xyt_coord, matrix_tup)
    elif kernel == 'rational_quad':
        c_auto = fast_rational_quadratic_3d(sigma, length, xyt_coord, xyt_coord, matrix_tup)
    else:
        c_auto = np.eye(v_array.size)
        print('No appropriate kernel found')

    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    cov_matrix = c_auto + c_noise

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

    if log_gp_minimization <= 0:
        log_gp_min = 1000000  # give excessively large value for me to ignore
    elif log_gp_minimization >= 1000000:
        log_gp_min = 10000
    else:
        log_gp_min = log_gp_minimization

    return log_gp_min


if opt_method == 'NM':
    print('Kernel Optimization using Nelder-Mead Function evaluation method')
    param_sol = scopt.minimize(fun=gp_3d_mahalanobis, args=args_param, x0=initial_all_param,
                               method='Nelder-Mead',
                               options={'xatol': 10, 'fatol': 300, 'disp': True, 'maxfev': 1000})
    func_optimal = param_sol.fun
elif opt_method == 'DE':
    print('Kernel Optimization using Differential Evolution')
    # Attempt to use differential evolution method as proposed by Stork K Price
    # Initialise Bounds for the parameters - in differential evolution, it takes a bound instead of starting point
    # The differential evolution uses the latin hypercube method - no gradient methods are used

    # The bound takes in a sequence of tuples
    b_u = 1
    b_l = -1
    param_bound = [(b_l, b_u), (b_l, b_u), (b_l, b_u), (b_l, b_u), (b_l, b_u),
                   (b_l, b_u), (b_l, b_u), (b_l, b_u), (b_l, b_u), (b_l, b_u)]
    param_sol = scopt.differential_evolution(func=gp_3d_mahalanobis, bounds=param_bound, args=args_param)
elif opt_method == 'GP':
    print('Performing Bayesian Optimization to find optimize for hyper-parameters')
    # Bayesian Optimization using Scikit-Optimize - Skopt
    # Note inputs are entered as lists instead of tuple
    # Decide bounds

    # Enter Arguments - which is args_param but must be in a list
    args_param = [xyt_vox, latent_v_vox, ker]  # This is a list to be entered into skp

    print('List of bounds is', list_of_bounds)

    # I have to enter arguments into the objective function itself
    """
    param_sol = skp.gp_minimize(func=gp_3d_mahalanobis_skopt,
                                dimensions=list_of_bounds,
                                base_estimator=None,  # This is Matern by default - within the GP Bayesian Optimization
                                n_calls=100,
                                n_random_starts=10,
                                acq_func='gp_hedge',
                                acq_optimizer='auto',
                                x0=None,  # Initial Input Points
                                y0=None,  # Initial Output Points
                                random_state=None,
                                verbose=True,
                                n_points=10000,
                                n_restarts_optimizer=5,
                                xi=0.01,
                                kappa=1.96,
                                noise='gaussian')
    """
    param_sol = skp.gp_minimize(func=gp_3d_mahalanobis_skopt,
                                dimensions=list_of_bounds,
                                verbose=True,
                                n_random_starts=5)
elif opt_method == 'DM':  # Random search by uniform sampling within the given bounds - which may be pretty good
    # Decide bounds
    print('Performing random search for minimum by uniform sampling within given bounds')

    # Enter Arguments - which is args_param but must be in a list
    args_param = [xyt_vox, latent_v_vox, ker]  # This is a list to be entered into skp

    param_sol = skp.dummy_minimize(func=gp_3d_mahalanobis_skopt,
                                   dimensions=list_of_bounds,
                                   n_calls=100)
else:
    print('No GP optimization method entered - Differential Evolution used by default')
    # The bound takes in a sequence of tuples
    b_u = 1
    b_l = -1
    param_bound = [(b_l, b_u), (b_l, b_u), (b_l, b_u), (b_l, b_u), (b_l, b_u),
                   (b_l, b_u), (b_l, b_u), (b_l, b_u), (b_l, b_u), (b_l, b_u)]
    param_sol = scopt.differential_evolution(func=gp_3d_mahalanobis, bounds=param_bound, args=args_param)


end_gp_opt = time.clock()
print('Time taken for kernel optimization is', end_gp_opt - start_gp_opt)
print('The optimal solution display is', param_sol)
print('The time taken for GP optimization is', end_gp_opt - start_gp_opt)

# List optimal hyper-parameters
sigma_optimal = param_sol.x[0]
length_optimal = param_sol.x[1]
noise_optimal = param_sol.x[2]
mean_optimal = param_sol.x[3]
matrix_var_optimal = param_sol.x[4:]
func_optimal = param_sol.fun

print('Optimal function evaluation is ', func_optimal)
print('optimal sigma is ', sigma_optimal)
print('optimal length-scale is ', length_optimal)
print('optimal noise amplitude is ', noise_optimal)
print('optimal scalar mean value is ', mean_optimal)
print('optimal matrix variables are', matrix_var_optimal)
print('Kernel is', ker)
print('The number of voxels per side is', vox_on_side)
print('GP Hyper-parameter Optimization Completed')
print('The starting kernel parameters are', initial_kernel_scalar)
print('The starting matrix parameters are', initial_mat_param)


# -------------------------------------------------------------------- END OF KERNEL OPTIMIZATION

"""
# -------------------------------------------------------------------- START POSTERIOR TABULATION
# Note Hessian = second derivative of the log[g(v)]
# Posterior Distribution follows N(v; v_hap, -1 * Hessian)
print('Performing Posterior Tabulation...')
start_posterior_tab = time.clock()

# Generate prior covariance matrix with kronecker noise
if ker == 'matern1':
    cov_auto = fast_matern_1_3d(sigma_optimal, length_optimal, xyt_vox, xyt_vox, matrix_var_optimal)
elif ker == 'matern3':
    cov_auto = fast_matern_3_3d(sigma_optimal, length_optimal, xyt_vox, xyt_vox, matrix_var_optimal)
elif ker == 'squared_exponential':
    cov_auto = fast_squared_exp_3d(sigma_optimal, length_optimal, xyt_vox, xyt_vox, matrix_var_optimal)
elif ker == 'rational_quad':
    cov_auto = fast_rational_quadratic_3d(sigma_optimal, length_optimal, xyt_vox, xyt_vox, matrix_var_optimal)
else:
    cov_auto = np.eye(latent_v_vox.size)
    print('No appropriate kernel chosen')

cov_noise = (noise_optimal ** 2) * np.eye(cov_auto.shape[0])  # Addition of noise
cov_overall = cov_auto + cov_noise

# Generate inverse of covariance matrix and set up the hessian matrix using symmetry
inv_cov_overall = np.linalg.inv(cov_overall)
inv_cov_diagonal_array = np.diag(inv_cov_overall)
hess_diagonal = -1 * (np.exp(latent_v_vox) + inv_cov_diagonal_array)

# Initialise and generate hessian matrix
hess_matrix = np.zeros_like(inv_cov_overall)
hess_length = inv_cov_overall.shape[0]

# Fill in values
for i in range(hess_length):
    hess_matrix[i, i] = -1 * (np.exp(latent_v_vox[i]) + inv_cov_overall[i, i])
    for j in range(i + 1, hess_length):
        hess_matrix[i, j] = -0.5 * (inv_cov_overall[i, j] + inv_cov_overall[j, i])
        hess_matrix[j, i] = hess_matrix[i, j]

# The hessian H of the log-likelihood at vhap is the negative of the Laplacian
hess_matrix = - hess_matrix

# Generate Posterior Covariance Matrix of log-intensity v *** Check this part
posterior_cov_matrix_v = np.linalg.inv(hess_matrix)
print('Posterior Covariance Matrix of v is ', posterior_cov_matrix_v)

print('Posterior Covariance Calculation Completed')
# ------------------------------------------------------------------- END POSTERIOR TABULATION

# ------------------------------------------------------------------- START CONVERSION INTO ARITHMETIC MEAN AND SD
print('Start conversion into arithmetic mean and standard deviation')
# Tabulation of Posterior Latent Intensity Mean
variance_v = np.diag(posterior_cov_matrix_v)
latent_intensity_mean = np.exp(latent_v_vox + 0.5 * variance_v)

# Tabulation of Posterior Latent Intensity Variance
latent_intensity_var = np.exp((2 * latent_v_vox) + variance_v) * (np.exp(variance_v) - 1)
latent_intensity_sd = np.sqrt(latent_intensity_var)

# Mesh Matrix containing posterior mean and standard deviation for plotting purposes
latent_intensity_mean_mesh = latent_intensity_mean.reshape(x_mesh.shape)
latent_intensity_sd_mesh = latent_intensity_sd.reshape(x_mesh.shape)
# Note that we cannot recreate the mesh after the zero points have been excluded


print('Log-Intensity Variances are ', variance_v)
print('Latent Intensity Values are ', latent_intensity_mean)
print('Latent Intensity Variances are ', latent_intensity_var)

# Measure time taken for covariance matrix and final standard deviation tabulation
time_posterior_tab = time.clock() - start_posterior_tab

print('Time Taken for Conversion into Latent Intensity = ', time_posterior_tab)

print('Latent Intensity Conversion Completed')
print('Kernel is', ker)

# ------------------------------------------ End of Conversion into Latent Intensity

# ------------------------------------------ Calculate the MSE from the Arithmetic Mean
mean_sq_error = fn.mean_squared_error(latent_intensity_mean, k_vox)
print('The Mean Squared Error is', mean_sq_error)
print('Last function evaluation is ', param_sol.fun)
print('optimal sigma is ', sigma_optimal)
print('optimal length-scale is ', length_optimal)
print('optimal noise amplitude is ', noise_optimal)
print('optimal scalar mean value is ', mean_optimal)
print('optimal matrix variables are', matrix_var_optimal)
print('Kernel is', ker)
print('The number of voxels per side is', vox_on_side)
print('GP Hyper-parameter Optimization Completed')
print('The starting kernel parameters are', initial_kernel_scalar)
print('The starting matrix parameters are', initial_mat_param)
print('Differential evolution bound is', np.array([b_l, b_u]))
"""