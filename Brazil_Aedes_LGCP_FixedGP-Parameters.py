import pandas as pd
import math
import matplotlib
import numpy as np
import time
import functions as fn
import scipy
import scipy.special as scispec
import scipy.optimize as scopt

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Preamble on LGCP
1. Used to infer non-stationary Poisson Processes. Need to guarantee positive definite covariance matrices.
2, A Cox Process is a non-homogeneous Poisson Point Process where 
the underlying intensity landa is itself a stochastic process
3. We assume that the log-intensity of the point process is a GP, and we have to take the exponential of the gaussian 
surface to obtain the latent intensity surface

We need to optimise by trying to maximise the posterior, which will give us the greatest possibility of
the latent intensity, given the data. Our goal is to predict log(landa) = v, not landa directly

LGCPs are not capable of producing predictions for any input vector, but only for inputs 
corresponding to the observations. The resulting values of v will then form the predicted mean of the model

The hessian is the the second derivative of the log-likelihood function
Unlike standard GP Regression. LGCPs are not capable of producing predictions for any input vector, but only for inputs
corresponding to the observations. The resulting values of v will then form the predicted mean of the model.
LGCP can only predict where there has been a data point

Proper Steps to follow:
1. Assume arbitrary covariance matrix parameters for the base case  - theta is fixed at an optimal value
2. Assume p(theta) is a delta function at those parameters
3. Goal here is to obtain v = log(landa), where landa is the actual intensity.
4. We are here predicting the posterior for v instead of landa
5. v follows a gaussian process, and there is only one optimal value set of values
6. m refers to the number of bins/quadrats
"""


# Define Poisson Distribution function for each quadrat - homogeneous Poisson
def poisson_discrete(k, landa):  # Takes in two parameters intensity landa and observation value k
    numerator = np.power(landa, k) * np.exp(-1 * landa)  # mean and covariance are both landa
    denominator = math.factorial(k)
    p = numerator / denominator
    return p


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
    p_likelihood = np.prod(prob_array)  # Taking combined product of distributions
    # Note output is a scalar (singular value)
    return p_likelihood  # Returns the non logarithmic version.


def log_special(array):
    """Taking an element-wise natural log of the array, retain array dimensions"""
    """with the condition that log(0) = 0, so there are no -inf elements"""
    log_array = np.zeros(array.size)
    for i in range(array.size):
        if array[i] == 0:
            log_array[i] = 0
        else:  # meaning that if the array element is not zero, then we can apply the natural log
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


# "2-dimensional squared exponential function that generates the covariance matrix"
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


# "2-dimensional matern kernel function that generates the covariance matrix directly"
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


"""Note the below is for standard gaussian process regression - so we will not be using them for Laplace"""
"""Eventually we will have to compare between the LGCP implementation and the standard GP regression"""


def mu_post(p_mean, xy_next, c_auto, c_cross, mismatch):  # Posterior mean
    if c_cross.shape[1] != c_auto.shape[1]:
        print('First Dimension Mismatch!')
    if c_auto.shape[0] != (np.transpose(mismatch)).shape[0]:
        print('Second Dimension Mismatch!')
    else:
        mean_post = mean_func_scalar(p_mean, xy_next) + \
                    fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(mismatch))
        return mean_post


def cov_post(c_next_auto, c_cross, c_auto):  # Posterior Covariance
    c_post = c_next_auto - fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(c_cross))
    return c_post


def log_model_evidence(param, *args):  # Param includes both sigma and l, arg is passed as a pointer
    sigma = param[0]  # param is a tuple containing 2 things, which has already been defined in the function def
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
    model_complexity = - 0.5 * math.log(np.linalg.det(c_auto_noise))
    model_constant = - 0.5 * len(histogram_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant
    return -log_model_evid  # We want to maximize the log-likelihood, meaning the min of negative log-likelihood


# Each region is assumed to not have a constant log-intensity, uni-variate gaussian distribution assumed
def posterior_num(v_array, k_array, gaussian_mean, cov_matrix):
    """numerator consists of product of poisson distribution and distribution of v given arbitrary hyper-parameters"""
    if len(k_array) == len(v_array):
        m = len(k_array)
        exp_v_array = np.exp(v_array)  # Discretization is now removed
        """Poisson Distribution Product"""
        poisson_dist_product = poisson_product(k_array, exp_v_array)  # Uses a generalised factorial - gamma function
        """Construct latter v distribution term"""
        v_distribution_coeff = (((2 * np.pi) ** m) * np.linalg.det(cov_matrix)) ** (-0.5)
        v_distribution_power = - 0.5 * fn.matmulmul(v_array - gaussian_mean, np.linalg.inv(cov_matrix),
                                                    np.transpose(v_array - gaussian_mean))
        v_distribution = v_distribution_coeff * np.exp(v_distribution_power)
        """Overall posterior numerator"""
        post_num_overall = poisson_dist_product * v_distribution

    else:
        print('k and v arrays do not have matching dimensions')

    return post_num_overall


# Generate posterior numerator function form for optimization
def posterior_numerator(param, *args):
    """We want to find the optimal v_array, so this is the parameter to be optimised"""
    v_array = param
    """args points to a tuple with 3 components, so have to create a tuple to contain them first"""
    k_array = args[0]
    gaussian_mean = args[1]
    covariance_matrix = args[2]
    posterior_numerator_value = posterior_num(k_array, v_array, gaussian_mean, covariance_matrix)
    return posterior_numerator_value
    # This is without regard to the size of the intervals


# Goal here is to estimate the denominator of the posterior
def laplace_approx(approx_func, approx_args, initial_param, approx_method):  # Note this is a general laplace approx
    """Finding an approximation to the integral of the function using Laplace's Approximation"""
    # Takes in a function and imposes a gaussian function over it
    # Measure uncertainty from actual value. As M increases, the approximation of the function by
    # a gaussian function gets better. Note that an unscaled gaussian function is used.
    """Tabulate the global maximum of the function - within certain boundaries - using latin hypercube"""
    solution_laplace = scopt.minimize(fun=approx_func, args=approx_args, x0=initial_param, method=approx_method)
    optimal_param_vect = solution_laplace.x  # location of maximum
    optimal_func_val = solution_laplace.fun  # value at maximum of function

    """
    Integral value can be approximated as product of 
    1. function valuation at maximum value
    2. Squareroot of 2pi / c, where c = the value of the hessian of the log of the function at the optimal location
    
    - Optimal location and maximum value of cost function are now tabulated
    
    """


# Shortcut optimization cost function to tabulate optimal v array
def posterior_num_cost(v_array, y_array, cov_matrix, gaussian_mean):  # values of v are only evaluated at locations of y
    exp_term = np.sum(np.exp(v_array))
    y_term = - 1 * np.matmul(v_array, np.transpose(y_array))
    determinant_cov = np.linalg.det(cov_matrix)
    ln_term = 0.5 * np.log(determinant_cov)
    data_diff = v_array - gaussian_mean
    euclidean_term = 0.5 * fn.matmulmul(data_diff, np.linalg.inv(cov_matrix), np.transpose(data_diff))
    p_cost = exp_term + y_term + ln_term + euclidean_term
    return p_cost  # I think this is a mistake


def posterior_num_cost_opt(param, *args):  # adapt original cost function for optimization
    v_array = param
    y_array = args[0]
    cov_matrix = args[1]
    gaussian_mean = args[2]
    p_cost_opt_inverted = posterior_num_cost(v_array, y_array, cov_matrix, gaussian_mean)
    return p_cost_opt_inverted


"""Over here, we assume the values of the hyper-parameters first, before optimizing the the v_array"""

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
x_2014 = aedes_brazil_2014.values[:, 5].astype('float64')
y_2014 = aedes_brazil_2014.values[:, 4].astype('float64')
x_2013 = aedes_brazil_2013.values[:, 5].astype('float64')
y_2013 = aedes_brazil_2013.values[:, 4].astype('float64')


# First conduct a regression on the 2014 data set
quads_on_side = 10  # define the number of quads along each dimension
# histo, x_edges, y_edges = np.histogram2d(theft_x, theft_y, bins=quads_on_side)  # create histogram
histo, y_edges, x_edges = np.histogram2d(y_2013, x_2013, bins=quads_on_side)
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


"""Generate auto-covariance matrix with noise - using arbitrary hyper-parameters first"""
sigma_arb = 3
length_arb = 5
noise_arb = 2
matern_v = 3/2
c_dd = matern_2d(matern_v, sigma_arb, length_arb, xy_quad, xy_quad)
c_dd_noise = c_dd + (noise_arb ** 2) * np.eye(c_dd.shape[0])  # Input to the posterior_numerator function

"""Testing out a squared exponential kernel"""
# c_dd = squared_exp_2d(sigma_arb, length_arb, xy_data_coord, xy_data_coord)
# c_dd_noise = c_dd + (noise_arb ** 2) * np.eye(c_dd.shape[0])

"""Generate conditions and values for optimization of v, given k"""
gp_mean_v = np.average(log_special(k_quad))  # assuming log(0) = 0, this is a simplified mean calculation
arguments = (k_quad, c_dd_noise, gp_mean_v)  # form of a tuple
initial_v = np.ones(k_quad.shape)  # histo is one long array

"""Tabulate optimal array of v, which is vhap"""
solution = scopt.minimize(fun=posterior_num_cost_opt, args=arguments, x0=initial_v, method='Nelder-Mead',
                          options={'xatol': 0.1, 'fatol': 10, 'disp': True, 'maxfev': 100000})
v_hap = solution.x  # generate optimal location of array v which generates the greatest posterior
landa_quad = np.reshape(np.exp(v_hap), histo.shape)

print(solution)

# Plot occurrences in Brazil (Scatter Plot)
brazil_fig = plt.figure()
brazil_fig.canvas.set_window_title('Brazil Aedes Occurrences')


x_lower = -70
x_upper = -35
y_lower = -40
y_upper = 10


brazil_scatter = brazil_fig.add_subplot(221)
pp_2013 = brazil_scatter.scatter(x_2013, y_2013, marker='.', color='red', s=0.01)
pp_2014 = brazil_scatter.scatter(x_2014, y_2014, marker='.', color='blue', s=0.01)
plt.legend([pp_2014, pp_2013], ["2014", "2013"])
brazil_scatter.set_xlim(x_lower, x_upper)
brazil_scatter.set_ylim(y_lower, y_upper)

brazil_histogram = brazil_fig.add_subplot(222)
brazil_histogram.pcolor(x_mesh, y_mesh, histo, cmap='Reds')
brazil_histogram.set_xlim(x_lower, x_upper)
brazil_histogram.set_ylim(y_lower, y_upper)

brazil_post_mean = brazil_fig.add_subplot(223)
brazil_post_mean.pcolor(x_mesh, y_mesh, landa_quad, cmap='Reds')
brazil_post_mean.set_xlim(x_lower, x_upper)
brazil_post_mean.set_ylim(y_lower, y_upper)

plt.show()



