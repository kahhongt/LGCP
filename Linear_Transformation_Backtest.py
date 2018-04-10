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
# ------------------------------------------ End of Data Collection

# ------------------------------------------ Start of Scatter Point Set
# Define Scatter Point Boundary
x_upper_box = -35
x_lower_box = -65
y_upper_box = 0
y_lower_box = -30

# ChangeParam - select the year to be used
training_year = '2013'
if training_year == '2013':
    x = x_2013
    y = y_2013
elif training_year == '2014':
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

print(x_within_box.shape)

graph_2013_fig = plt.figure()
graph_2013 = graph_2013_fig.add_subplot(111)
graph_2013.scatter(x_within_box, y_within_box, color='black', s=5)  # note this is now for 2014
graph_2013.set_title('Scatter Plot for 2013')

# ------------------------------------------ End of Regression Window Selection before Transformation

# ------------------------------------------ Start of Performing Transformation
xy_within_box = np.vstack((x_within_box, y_within_box))  # Create the sample points to be rotated

# Initialise kernel hyperparameters
initial_hyperparameters = np.array([1, 1, 1, 1])

# ChangeParam *** IMPORTANT
# transform_matrix_array = np.array([0.3, 3.5, 3.5, 0.3])
# transform_matrix_array = np.array([1, 0, 0, 1])
transform_matrix_array = np.array([0.30, 0.93, 0.65, -0.23])


transformed_xy_within_box = fn.transform_array(transform_matrix_array, xy_within_box, center)
x_points_trans = transformed_xy_within_box[0]
y_points_trans = transformed_xy_within_box[1]

vertices = np.array([[x_lower, x_lower, x_upper, x_upper], [y_lower, y_upper, y_upper, y_lower]])
print('The Original Vertices are', vertices)

# Transform the vertices using the same transformation matrix
transformed_vertices = fn.transform_array(transform_matrix_array, vertices, center)
print('The Transformed Vertices are', transformed_vertices)

# 1. Obtain the maximum range in x and y in the transformed space
x_min = min(transformed_vertices[0])
x_max = max(transformed_vertices[0])
y_min = min(transformed_vertices[1])
y_max = max(transformed_vertices[1])

print('x range is', x_max - x_min)
print('y range is', y_max - y_min)  # Display regression window size at each iteration

# ChangeParam
quads_on_side = 20  # define the number of quads along each dimension
k_mesh, y_edges, x_edges = np.histogram2d(y_points_trans, x_points_trans, bins=quads_on_side,
                                          range=[[y_min, y_max], [x_min, x_max]])
x_mesh_plot, y_mesh_plot = np.meshgrid(x_edges, y_edges)  # creating mesh-grid for use
x_mesh = x_mesh_plot[:-1, :-1]  # Removing extra rows and columns due to edges
y_mesh = y_mesh_plot[:-1, :-1]
x_quad = fn.row_create(x_mesh)  # Creating the rows from the mesh
y_quad = fn.row_create(y_mesh)
xy_quad = np.vstack((x_quad, y_quad))

k_quad = fn.row_create(k_mesh)

# Kernel Optimization
ker = 'matern1'

# Start Optimization
arguments = (xy_quad, k_quad, ker)

# Check time taken for the optimization
start_opt = time.clock()

solution = scopt.minimize(fun=short_log_integrand_data, args=arguments, x0=initial_hyperparameters,
                          method='Nelder-Mead',
                          options={'xatol': 1, 'fatol': 1, 'disp': True, 'maxfev': 1000})

print('Last function evaluation is ', solution.fun)
sigma_optimal = solution.x[0]
length_optimal = solution.x[1]
noise_optimal = solution.x[2]
mean_optimal = solution.x[3]
print(solution)
print('Last function evaluation is ', solution.fun)
print('optimal sigma is ', sigma_optimal)
print('optimal length-scale is ', length_optimal)
print('optimal noise amplitude is ', noise_optimal)
print('optimal scalar mean value is ', mean_optimal)

#  REPEAT THE LOG MARGINAL LIKELIHOOD TABULATION USING 2014 VALUES

# ------------------------------------------Start of Sampling Points Creation

# Define number of points along each side of box containing the circle
# ChangeParam
intervals = 20

# Define the cut-off point beyond the circle - creating sampling points beyond data set
# ChangeParam
cut_decision = 'nothing'
if cut_decision == 'small_range':  # boundary exceeded by half an interval on each axis
    cut_off_x = (x_upper - x_lower) / (intervals * 2)
    cut_off_y = (y_upper - y_lower) / (intervals * 2)
    # intervals_final = intervals + 1

elif cut_decision == 'large_range':  # boundary exceeded by half the entire range on each axis
    cut_off_x = (x_upper - x_lower) / 4
    cut_off_y = (y_upper - y_lower) / 4
else:  # No inclusion of points beyond the circle
    cut_off_x = 0
    cut_off_y = 0

# Generate edges within the pre-defined range
sampling_points_x = np.linspace(x_min - cut_off_x, x_max + cut_off_x, intervals)
sampling_points_y = np.linspace(y_min - cut_off_y, y_max + cut_off_y, intervals)

# Create iteration for coordinates using mesh-grid - for plotting
sampling_points_xmesh, sampling_points_ymesh = np.meshgrid(sampling_points_x, sampling_points_y)
sampling_x_row = fn.row_create(sampling_points_xmesh)
sampling_y_row = fn.row_create(sampling_points_ymesh)
sampling_xy = np.vstack((sampling_x_row, sampling_y_row))

# ------------------------------------------End of Sampling Points Creation

# ------------------------------------------Start of Posterior Tabulation
start_posterior = time.clock()

# Create cases for kernel selection
if ker == 'matern1':
    cov_dd = fast_matern_1_2d(sigma_optimal, length_optimal, xy_quad, xy_quad)
elif ker == 'matern3':
    cov_dd = fast_matern_2d(sigma_optimal, length_optimal, xy_quad, xy_quad)
elif ker == 'squared_exponential':
    cov_dd = fast_squared_exp_2d(sigma_optimal, length_optimal, xy_quad, xy_quad)
elif ker == 'rational_quad':
    cov_dd = fast_rational_quadratic_2d(sigma_optimal, length_optimal, xy_quad, xy_quad)
else:  # No acceptable kernel defined
    cov_dd = np.eye(xy_quad.shape[1])  # Generate nonsensical identity matrix
    print('No acceptable kernel defined. Results should make no sense')

cov_noise = np.eye(cov_dd.shape[0]) * (noise_optimal ** 2)
cov_overall = cov_dd + cov_noise
prior_mean = mean_func_scalar(mean_optimal, xy_quad)
prior_mismatch = k_quad - prior_mean

# Initialise mean_posterior and var_posterior array
mean_posterior = np.zeros(sampling_xy.shape[1])
var_posterior = np.zeros(sampling_xy.shape[1])

# Generate mean and covariance array
for i in range(sampling_xy.shape[1]):

    # Generate status output
    if i % 100 == 0:  # if i is a multiple of 50,
        print('Tabulating Prediction Point', i)

    # Change_Param
    # At each data point,
    xy_star = sampling_xy[:, i]

    # Create cases for kernel selection
    if ker == 'matern1':
        cov_star_d = matern_2d(1/2, sigma_optimal, length_optimal, xy_star, xy_quad)
        cov_star_star = matern_2d(1 / 2, sigma_optimal, length_optimal, xy_star, xy_star)
    elif ker == 'matern3':
        cov_star_d = matern_2d(3 / 2, sigma_optimal, length_optimal, xy_star, xy_quad)
        cov_star_star = matern_2d(3 / 2, sigma_optimal, length_optimal, xy_star, xy_star)
    elif ker == 'squared_exponential':
        cov_star_d = squared_exp_2d(sigma_optimal, length_optimal, xy_star, xy_quad)
        cov_star_star = squared_exp_2d(sigma_optimal, length_optimal, xy_star, xy_star)
    elif ker == 'rational_quad':
        cov_star_d = rational_quadratic_2d(sigma_optimal, length_optimal, xy_star, xy_quad)
        cov_star_star = rational_quadratic_2d(sigma_optimal, length_optimal, xy_star, xy_star)
    else:
        cov_star_d = 0
        cov_star_star = 0
        print('No acceptable kernel entered')

    # Generate Posterior Mean and Variance
    mean_posterior[i] = mu_post(xy_star, cov_overall, cov_star_d, prior_mismatch)
    var_posterior[i] = var_post(cov_star_star, cov_star_d, cov_overall)

print('The posterior mean is', mean_posterior)
print('The posterior variance is', var_posterior)

# ------------------------------------------End of Posterior Tabulation

# ----------------------------------------------------------------------Extract 2014 Test set

# ChangeParam - select the year to be used - all the variables below would have changed
testing_year = '2014'
if testing_year == '2013':
    x = x_2013
    y = y_2013
elif testing_year == '2014':
    x = x_2014
    y = y_2014
else:  # taking all years instead
    x = x_2013_2014
    y = y_2013_2014

x_range_box = (x > x_lower_box) & (x < x_upper_box)
y_range_box = (y > y_lower_box) & (y < y_upper_box)
x_points = x[x_range_box & y_range_box]
y_points = y[x_range_box & y_range_box]

# Above is the whole thing containing Brazil

center = (-50, -15)  # Create tuple
radius = 8
xy_within_box = np.vstack((x_points, y_points))

# Draws on the transformation matrix from above
transformed_xy_within_box = fn.transform_array(transform_matrix_array, xy_within_box, center)
# Note that radius is not used here, only the center is being used
print('The number of transformed points is', transformed_xy_within_box.shape)
x_points_box = transformed_xy_within_box[0]
y_points_box = transformed_xy_within_box[1]

# ------------------------------------------ End of Performing Rotation

# ------------------------------------------Start of Selective Binning
# Note this is for 2014 - entire Brazil Data - note these are arbitrarily selected
# Maximum and minimum values of each coordinate for the scattered points
maximum_x = -32.43
minimum_x = -72.79
maximum_y = 4.72
minimum_y = -32.21

# Technically I can just bin everything first, then select the ones that I want

# Select regression window boundaries
# ChangeParam

x_window = (x_points_box > x_min) & (x_points_box < x_max)
y_window = (y_points_box > y_min) & (y_points_box < y_max)
x_within_window = x_points_box[x_window & y_window]
y_within_window = y_points_box[x_window & y_window]

# This is now the same regression window in the transformed space

print('Number of scatter points in box is', x_within_window.shape)  # Note this is after the transformation

# ------------------------------------------ End of Selective Binning into a Square

# ------------------------------------------ Start of Histogram Generation from Box
# First conduct a regression on the 2014 data set
# ChangeParam - same quads on side
k_mesh, y_edges, x_edges = np.histogram2d(y_within_window, x_within_window, bins=quads_on_side,
                                          range=[[y_min, y_max], [x_min, x_max]])  # this is the correct range
x_mesh_plot, y_mesh_plot = np.meshgrid(x_edges, y_edges)  # creating mesh-grid for use
x_mesh = x_mesh_plot[:-1, :-1]  # Removing extra rows and columns due to edges
y_mesh = y_mesh_plot[:-1, :-1]
x_quad = fn.row_create(x_mesh)  # Creating the rows from the mesh
y_quad = fn.row_create(y_mesh)

# ------------------------------------------ End of Histogram Generation from Box
xy_quad = np.vstack((x_quad, y_quad))
k_quad = fn.row_create(k_mesh)
print('The total number of quadrats used for regression is', k_quad.size)  # this is for 2014

# ------------------------------------------ End of Realignment of Quad Centers

# ------------------------------------------ Start calculating the Log Likelihood and MSE using 2014 testing data

# Change_Param - change kernel by setting cases
if ker == 'matern3':
    c_auto_test = fast_matern_2d(sigma_optimal, length_optimal, xy_quad, xy_quad)
elif ker == 'matern1':
    c_auto_test = fast_matern_1_2d(sigma_optimal, length_optimal, xy_quad, xy_quad)
elif ker == 'squared_exponential':
    c_auto_test = fast_squared_exp_2d(sigma_optimal, length_optimal, xy_quad, xy_quad)
elif ker == 'rational_quad':
    c_auto_test = fast_rational_quadratic_2d(sigma_optimal, length_optimal, xy_quad, xy_quad)
else:  # Default kernel is matern1
    c_auto_test = np.eye(k_quad.shape[1])
    print('Check for appropriate kernel')

c_noise_test = np.eye(c_auto_test.shape[0]) * (noise_optimal ** 2)  # Fro-necker delta function
cov_matrix_test = c_auto_test + c_noise_test

"""Generate Objective Function = log[g(v)]"""
# Generate Determinant Term (after taking log)
determinant_test = np.exp(np.linalg.slogdet(cov_matrix_test))[1]
det_term_test = -0.5 * np.log(2 * np.pi * determinant_test)

# Generate Euclidean Term (after taking log)
data_diff_test = k_quad - prior_mean  # this is the optimal prior mean
inv_covariance_matrix_test = np.linalg.inv(cov_matrix_test)
euclidean_term_test = -0.5 * fn.matmulmul(data_diff_test, inv_covariance_matrix_test, data_diff_test)

"""Summation of all terms change to correct form to find minimum point"""
log_gp_test = det_term_test + euclidean_term_test

print('The Log Marginal Likelihood is', log_gp_test)
print('The kernel used is', ker)
# Calculate Mean Squared Error
mse = fn.mean_squared_error(mean_posterior, k_quad)
print('The mean squared error is', mse)
print('The training year is', training_year)
print('The testing year is', testing_year)
print('The transformation matrix is', transform_matrix_array)
print('The number of quadrats selected is', k_quad.size)


graph_2014_fig = plt.figure()
graph_2014 = graph_2014_fig.add_subplot(111)
graph_2014.scatter(x_within_window, y_within_window, color='black', s=5)  # note this is now for 2014
graph_2014.set_title('Scatter Plot for 2014')

plt.show()



