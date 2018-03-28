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
        print('Check for Appropriate Kernel')

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


def rotation_likelihood_opt(param, *args):
    """
    Objective is to find the angle of rotation that gives the greatest log-likelihood, based on a
    standard GP regression. It would be a same assumption that the same optimal angle will be obtained using both
    standard GP regression and the LGCP. Over here, we do not need to tabulate the posterior so that saves time.

    We are taking the xy_data which is already boxed and a single year will be taken

    :param param: angle of rotation in degrees - note there is only one parameter to optimize
    :param args: xy_data, center, kernel form (this is a tuple), regression window
    :return: log marginal likelihood based on the standard GP process
    """
    angle = param

    # convert angle to radians
    radians = (angle / 180) * np.pi

    # Unpack Param Tuple
    center = args[0]
    kernel = args[1]
    n_quads = args[2]
    xy_coordinates = args[3]  # Make this a tuple, so it will be a tuple within a tuple
    regression_window = args[4]  # This is an array - x_upper, x_lower, y_upper and y_lower

    # Define regression window
    x_upper_box = regression_window[0]
    x_lower_box = regression_window[1]
    y_upper_box = regression_window[2]
    y_lower_box = regression_window[3]

    # Break up xy_coordinates into x and y
    x_coordinates = xy_coordinates[0]
    y_coordinates = xy_coordinates[1]

    # Define Boolean Variable for Scatter Points Selection
    x_range_box = (x_coordinates > x_lower_box) & (x_coordinates < x_upper_box)
    y_range_box = (y_coordinates > y_lower_box) & (y_coordinates < y_upper_box)

    # Obtain data points within the regression window
    x_coordinates = x_coordinates[x_range_box & y_range_box]
    y_coordinates = y_coordinates[x_range_box & y_range_box]

    # Stack x and y coordinates
    xy_within_box = np.vstack((x_coordinates, y_coordinates))

    # Perform rotation using simple steps
    rotation_mat = np.array([[np.cos(radians), - np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    rotation_mat = np.hstack((rotation_mat[0], rotation_mat[1]))
    print(rotation_mat.shape)
    x_within_box = xy_within_box[0] - center[0]
    y_within_box = xy_within_box[1] - center[1]
    xy_within_box = np.vstack((x_within_box, y_within_box))
    xy_within_box = np.matmul(rotation_mat, xy_within_box)
    rotated_x = xy_within_box[0] + center[0]
    rotated_y = xy_within_box[1] + center[1]

    # Create boolean variable
    x_window_w = (rotated_x > x_lower_box) & (rotated_x < x_upper_box)
    y_window_w = (rotated_y > y_lower_box) & (rotated_y < y_upper_box)
    x_window = rotated_x[x_window_w & y_window_w]
    y_window = rotated_y[x_window_w & y_window_w]

    # First conduct a regression on the 2014 data set
    # ChangeParam
    histo_f, y_edges_f, x_edges_f = np.histogram2d(y_window, x_window, bins=n_quads)
    x_mesh_plot_f, y_mesh_plot_f = np.meshgrid(x_edges_f, y_edges_f)  # creating mesh-grid for use
    x_mesh_f = x_mesh_plot_f[:-1, :-1]  # Removing extra rows and columns due to edges
    y_mesh_f = y_mesh_plot_f[:-1, :-1]
    x_quad_f = fn.row_create(x_mesh_f)  # Creating the rows from the mesh
    y_quad_f = fn.row_create(y_mesh_f)

    # Note that over here, we do not have to consider the alignment of quad centers

    # Stack x and y coordinates together
    xy_quad = np.vstack((x_quad_f, y_quad_f))
    # Create histogram array
    k_quad = fn.row_create(histo_f)

    # Being tabulating log marginal likelihood after optimizing for kernel hyper-parameters

    initial_hyperparam = np.array([3, 2, 1, 1])  # Note that this initial condition should be close to actual
    # Set up tuple for arguments
    args_hyperparam = (xy_quad, k_quad, kernel)

    # Start Optimization Algorithm
    hyperparam_solution = scopt.minimize(fun=short_log_integrand_data, args=args_hyperparam, x0=initial_hyperparam,
                                         method='Nelder-Mead',
                                         options={'xatol': 1, 'fatol': 1, 'disp': True, 'maxfev': 10000})

    # Extract Log_likelihood value
    neg_log_likelihood = hyperparam_solution.fun  # Eventually, we will have to minimize the negative log likelihood
    # Hence, this is actually an optimization nested within another optimization algorithm
    return neg_log_likelihood


# ------------------------------------------ Start of Data Collection

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

# ------------------------------------------ Start of defining scatter point boundary
# Define Scatter Point Boundary
x_upper_box = -35
x_lower_box = -65
y_upper_box = 0
y_lower_box = -30

# Define Boolean Variable for Scatter Points Selection
x_range_box = (x_2013 > x_lower_box) & (x_2013 < x_upper_box)
y_range_box = (y_2013 > y_lower_box) & (y_2013 < y_upper_box)

x_points = x_2013[x_range_box & y_range_box]
y_points = y_2013[x_range_box & y_range_box]

# ------------------------------------------ End of defining scatter point boundary


# Define arguments for calculating the Log Marginal Likelihood
# ChangeParam
c = np.array([-50, -15])
radius = 8
ker = 'rational_quad'
quads_on_side = 10
xy_points = np.vstack((x_points, y_points))  # This refers to all the points that are being rotated
# reg_limit = (-43, -63, -2, -22)  # x_upper, x_lower, y_upper, y_lower
# Define regression window which actually remains the same
x_upper = c[0] + radius
x_lower = c[0] - radius
y_upper = c[1] + radius
y_lower = c[1] - radius

# Starting iteration point for angle
# ChangeParam
angle_limit = 90
angle_array = np.arange(0, angle_limit+1, 1)
print('The angle array is ', angle_array)

# Initialise array to store log_likelihood_values
likelihood_array = np.zeros_like(angle_array)
print('The Initial likelihood array is ', likelihood_array)

start_likelihood_tab = time.clock()

# For each angle, re-tabulate the optimal hyper_parameters and calculate the log_likelihood
for i in range(angle_array.size):
    # Rotate Data Points that are beyond the regression window
    rotated_xy = fn.rotate_array_iterate(angle_array[i], xy_points, c)
    rotated_x = rotated_xy[0]
    rotated_y = rotated_xy[1]

    print('Rotated xy is ', rotated_xy)
    # Define regression limits
    # Create Boolean Variable
    x_window = (rotated_x > x_lower) & (rotated_x < x_upper)
    y_window = (rotated_y > y_lower) & (rotated_y < y_upper)
    x_within_window = rotated_x[x_window & y_window]
    y_within_window = rotated_y[x_window & y_window]
    # These are the coordinates of points within the regression window

    # Generate Histogram from the coordinates of points above
    histo, y_edges, x_edges = np.histogram2d(y_within_window, x_within_window, bins=quads_on_side)
    x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)  # creating mesh-grid for use
    x_mesh = x_mesh[:-1, :-1]  # Removing extra rows and columns due to edges
    y_mesh = y_mesh[:-1, :-1]
    x_quad = fn.row_create(x_mesh)  # Creating the rows from the mesh
    y_quad = fn.row_create(y_mesh)
    k_quad = fn.row_create(histo)
    xy_quad = np.vstack((x_quad, y_quad))

    # Initialise arguments for hyper-parameter optimization
    arguments = (xy_quad, k_quad, ker)

    # Initialise kernel hyper-parameters
    initial_hyperparameters = np.array([3, 2, 1, 1])

    print('The current angle of rotation is ', angle_array[i])
    # Start optimization
    solution = scopt.minimize(fun=short_log_integrand_data, args=arguments, x0=initial_hyperparameters,
                              method='Nelder-Mead',
                              options={'xatol': 1, 'fatol': 1, 'disp': True, 'maxfev': 1000})
    # Over here, not really concerned about value of hyper-parameters, just the likelihood
    likelihood_array[i] = -1 * solution.fun  # A log likelihood value for each i


angle_opt_index = np.argmax(likelihood_array)  # This gives the index of the maximum angle
angle_opt = angle_array[angle_opt_index]
print('The Log_likelihood Array is ', likelihood_array)
print('The Optimal Angle is ', angle_opt)

time_likelihood_tab = time.clock() - start_likelihood_tab
print('Time taken for Angle Iteration =', time_likelihood_tab)

rotated_xy_within_window = fn.rotate_array_iterate(angle_opt, xy_points, c)
x_rotated = rotated_xy_within_window[0]
y_rotated = rotated_xy_within_window[1]

# ------------------------------------------ Compute the Posterior using angle_opt
# Create Likelihood Array for Plotting
likelihood_array_plot = np.hstack((likelihood_array, likelihood_array[1:]))
angle_array_plot = np.arange(0, (2*angle_limit) + 1, 1)


# Quick plot for log likelihood versus angle in degrees
fig_likelihood_plot = plt.figure()
likelihood_plot = fig_likelihood_plot.add_subplot(111)
likelihood_plot.plot(angle_array, likelihood_array, color='black')
likelihood_plot.set_title('Plot of Log Marginal Likelihood against Rotation Angle')
likelihood_plot.set_xlabel('Rotation Angle in Degrees')
likelihood_plot.set_ylabel('Log Marginal Likelihood')

# Quick Plot for 2 periods
fig_likelihood_plot_2 = plt.figure()
likelihood_plot_2 = fig_likelihood_plot_2.add_subplot(111)
likelihood_plot_2.plot(angle_array_plot, likelihood_array_plot, color='black')
likelihood_plot_2.set_title('Plot of Log Marginal Likelihood against Rotation Angle')
likelihood_plot_2.set_xlabel('Rotation Angle in Degrees')
likelihood_plot_2.set_ylabel('Log Marginal Likelihood')

plt.show()

# ------------------------------------------ Start of rebuilding based on optimal angle


# ChangeParam
"""
fig_brazil_scatter = plt.figure()
brazil_scatter = fig_brazil_scatter.add_subplot(111)
# brazil_scatter.scatter(x_2014, y_2014, marker='.', color='blue', s=0.1)
brazil_scatter.scatter(x_2013, y_2013, marker='.', color='black', s=0.3)
# plt.legend([pp_2014, pp_2013], ["2014", "2013"])
brazil_scatter.set_title('Brazil 2013 Aedes Scatter')
# brazil_scatter.set_xlim(x_lower, x_upper)
# brazil_scatter.set_ylim(y_lower, y_upper)
brazil_scatter.set_xlabel('UTM Horizontal Coordinate')
brazil_scatter.set_ylabel('UTM Vertical Coordinate')


fig_brazil_histogram = plt.figure()
brazil_histogram = fig_brazil_histogram.add_subplot(111)
brazil_histogram.pcolor(x_mesh_plot, y_mesh_plot, histo, cmap='YlOrBr')
brazil_histogram.scatter(x_2013, y_2013, marker='.', color='black', s=0.3)
histogram_circle = plt.Circle((-50, -15), 11.3, fill=False, color='orange')
brazil_histogram.add_patch(histogram_circle)
brazil_histogram.set_title('Brazil 2013 Aedes Histogram')
# brazil_histogram.set_xlim(x_lower, x_upper)
# brazil_histogram.set_ylim(y_lower, y_upper)
brazil_histogram.set_xlabel('UTM Horizontal Coordinate')
brazil_histogram.set_ylabel('UTM Vertical Coordinate')


plt.show()

"""


