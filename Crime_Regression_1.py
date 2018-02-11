import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn
import scipy
import scipy.special as scispec
import scipy.optimize as scopt

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        mean_c = np.ones(1) * 0  # meaning that if this is a 1-D array
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

    if v_value == 1 / 2:
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

    if v_value == 3 / 2:
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
    matern_nu = args[2]  # Arbitrarily chosen v value
    prior_mu = mean_func_scalar(scalar_mean, xy_coordinates)  # This creates a matrix with 2 rows
    c_auto = matern_2d(matern_nu, sigma, length, xy_coordinates, xy_coordinates)
    # c_auto = squared_exp_2d(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    c_auto_noise = c_auto + c_noise  # Overall including noise, plus include any other combination
    model_fit = - 0.5 * fn.matmulmul(histogram_data - prior_mu, np.linalg.inv(c_auto_noise),
                                     np.transpose(histogram_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_auto_noise))
    model_constant = - 0.5 * len(histogram_data) * math.log(2 * np.pi)
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
    return p_cost


def posterior_num_cost_opt(param, *args):  # adapt original cost function for optimization
    """
    Assuming an arbitrary covariance matrix at the start
    :param param: v_array
    :param args: data set y_array, arbitrary covariance matrix, and gaussian mean
    :return: generates the value of the numerator of the posterior
    """
    v_array = param
    y_array = args[0]
    cov_matrix = args[1]
    gaussian_mean = args[2]
    p_cost_opt_inverted = posterior_num_cost(v_array, y_array, cov_matrix, gaussian_mean)
    return p_cost_opt_inverted


"""Optimise hyper-parameters of the covariance as well"""


def posterior_cost_hyp(v_array, hyperparameters, k_array, xy_coordinates, gaussian_mean):
    """Generate covariance matrix inside function as we need to optimise hyper-parameters"""
    v = 3 / 2
    sigma = hyperparameters[0]
    length = hyperparameters[1]
    noise = hyperparameters[2]
    c_auto = matern_2d(v, sigma, length, xy_coordinates, xy_coordinates)
    c_auto_noise = c_auto + (noise ** 2) * np.eye(c_auto.shape[0])

    """Compute numerator of posterior"""
    exp_term = np.sum(np.exp(v_array))
    y_term = - 1 * np.matmul(v_array, np.transpose(k_array))
    determinant_cov = np.linalg.det(c_auto_noise)
    ln_term = 0.5 * np.log(determinant_cov)
    data_diff = v_array - gaussian_mean
    euclidean_term = 0.5 * fn.matmulmul(data_diff, np.linalg.inv(c_auto_noise), np.transpose(data_diff))
    p_cost = exp_term + y_term + ln_term + euclidean_term
    return p_cost


# Create suitable form for optimisation of v_array and hyper-parameters
def posterior_cost_hyp_opt(param, *args):  # param is an array, args is a tuple
    """
    :param param: optimal sigma, length scale, noise and v_array
    :param args: k_array, coordinates and gaussian mean
    :return: the numerator of the posterior, while taking the negative of it for minimal optimisation
    """
    sigma_optimal = param[0]
    length_optimal = param[1]
    noise_optimal = param[2]
    v_array = param[3:]  # The rest is v_array
    hyperparameters_optimal = np.array([sigma_optimal, length_optimal, noise_optimal])
    k_array = args[0]
    xy_coord = args[1]
    gp_mean = args[2]
    p_value = posterior_cost_hyp(v_array, hyperparameters_optimal, k_array, xy_coord, gp_mean)
    return -p_value  # We thus want to minimise the negative of the function value


"""Collate Data Points from PP_Data"""
# time_start = time.clock()  # Start computation time measurement

"""Extract Data from csv"""  # Arbitrary Point Process Data
A = np.genfromtxt('PP_Data_2D.csv', delimiter=',')  # Extract from csv using numpy
df = pd.read_csv('PP_Data_2D.csv')  # Generates a DataFrame from csv - coal data
x = np.ravel(df.values[0])
y = np.ravel(df.values[1])

print(x)
print(y)

"""Specify rotation matrix for data set"""
theta = 0 * np.pi  # Specify degree of rotation in a clockwise direction in radians
mat_transform = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
df_matrix = np.vstack((x, y))
df_transform = mat_transform * df_matrix
x_transform = np.ravel(df_transform[0])
y_transform = np.ravel(df_transform[1])

print(x_transform)
print(y_transform)
"""Bin point process data"""
bins_number = 4
histo, x_edges, y_edges = np.histogram2d(x_transform, y_transform, bins=bins_number)
xv_trans_data, yv_trans_data = np.meshgrid(x_edges, y_edges)
xv_trans_data = xv_trans_data[:-1, :-1]  # Removing the last bin edge and zero points to make dimensions consistent
yv_trans_data = yv_trans_data[:-1, :-1]  # Contains a square matrix
xv_trans_row = fn.row_create(xv_trans_data)  # Creates a row from the square matrix
yv_trans_row = fn.row_create(yv_trans_data)
histo_k_array = fn.row_create(histo)  # this is the k array
# xv_transform_row = xv_transform_row[histo != 0]  # Remove data point at histogram equal 0
# yv_transform_row = yv_transform_row[histo != 0]
# histo = histo[histo != 0]  # This is after putting them into rows

"""
Data point coordinates are now at bottom-left hand corner, coordinates of data points have
to be centralised to the centre of each quadrat
"""
# Centralizing coordinates for each quadrat
xv_trans_row = xv_trans_row + 0.5 * ((x_edges[-1] - x_edges[0]) / bins_number)
yv_trans_row = yv_trans_row + 0.5 * ((y_edges[-1] - y_edges[0]) / bins_number)

# Stack into 2 rows of many columns
xy_data_coord = np.vstack((xv_trans_row, yv_trans_row))  # location of all the data points

"""Generate conditions and values for optimization of v, given k"""
gp_mean_v = np.average(log_special(histo_k_array))  # assuming log(0) = 0, this is a simplified mean calculation
initial_v = np.ones(histo_k_array.shape)  # initial parameter for log of intensity
initial_hyperparam = np.array([3, 3, 3])  # sigma, length and noise (start of iteration)

"""Tabulate optimal array of v, and kernel hyperparameters"""
initial_v_hyperparam = np.concatenate((initial_hyperparam, initial_v), axis=0)  # concatenate hyperparameters, then v
arguments_all = (histo_k_array, xy_data_coord, gp_mean_v)  # form of a tuple with 3 elements
solution = scipy.optimize.minimize(fun=posterior_cost_hyp_opt, x0=initial_v_hyperparam, args=arguments_all,
                                   method='Nelder-Mead')

opt_param = solution.x
sigma_optimal = opt_param[0]
length_optimal = opt_param[1]
noise_optimal = opt_param[2]
opt_v_array = opt_param[3:]

"""What is the Hessian of the log posterior"""

"""Implement Predictive Bayesian Inference using the found hyper-parameters and latent values"""
# The Posterior distribution for log-intensity acts as a prior to the posterior of the occurrence of events


"""
index_array = np.zeros(histo_k_array.size)
for index in range(histo_k_array.size):
    index_array[index] = index

landa_hap = np.exp(v_hap)

print(poisson_product(histo_k_array, landa_hap))

plt.scatter(index_array, histo_k_array, marker='x', color='black')
plt.plot(index_array, v_hap, color='darkblue', linewidth=1)
plt.show()

"""

