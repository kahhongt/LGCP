import numpy as np
from numpy.lib import scimath
import time
import scipy.optimize as scopt
import matplotlib
matplotlib.use('TkAgg')


"""Simple Introductory Definitions"""


def array_summation(a):  # Computing the sum of a list
    q = 0  # Initialise z
    #  for i in x --- meaning for each element i in x
    for i in range(a.shape[0] + 1):  # Note range doesn't include the upper limit
        q = q + i
    return q


def solve_quadratic(a, b, c):
    """Use the General Equation"""
    real_term = (-b)/(2*a)
    imaginary_term = (np.lib.scimath.sqrt(b**2 - 4*a*c))/(2*a)
    root1 = real_term + imaginary_term
    root2 = real_term - imaginary_term
    return root1, root2


class Employee:

    def __init__(self, first, last, pay):  # This is a constructor which creates an instance of the class
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'

    def fullname(self):  # A method that takes in an instance of the class
        return '{} {}'.format(self.first, self.last)


emp_1 = Employee('Ivan', 'Payne', 100000)  # I have created an instance of an Employee class
emp_2 = Employee('Markus', 'Baxter', 50000)  # Used a method to create the email address
# print(emp_1.email)  # Methods always require parenthesis
# print(emp_2.email)
# print(emp_1.fullname())

"""Creating Switches"""


def linear(a, b):  # Takes in two rows
    a1 = np.matrix([a])  # Need to create the matrices first
    b1 = np.matrix([b])
    z = np.matmul(a1.transpose(), b1)  # Proper matrix multiplication
    return z


def f(x):
    return {
        'a': 1,
        'b': 2,
    }[x]


y = f('a')  # This assigns the value of y to be whatever defined in the function definition
# print(y)

# __init__ is used as a constructor in a class


def columnize(matrix):  # change a matrix into a column
    column = np.reshape(matrix, (matrix.size, 1))
    return column


def row_create(matrix):  # Generate a row from all the elements of a matrix
    row = np.ravel(np.reshape(matrix, (1, matrix.size)))
    return row


# Triple matrix Multiplication
def matmulmul(a1, b1, c1):
    matrix_product = np.matmul(np.matmul(a1, b1), c1)
    return matrix_product


# Return integer from datetime - using start and end dates
def dt_integer(dt_array, start, end):
    dt_int = dt_array.years + (dt_array.months / 12) + (dt_array.days / 365)
    return dt_int


# Generate inverse of matrix using cholesky decomposition - compare time taken with linagl.inv
def inverse_cholesky(matrix_a):
    l = np.linalg.cholesky(matrix_a)
    u = np.linalg.inv(l)
    inverse = np.matmul(u.transpose(), u)
    return inverse


# Generate artificial random stratified sample vectors using the Latin Hypercube principle
# As the number of guesses are arbitrarily chosen, it is less effective than differential evolution, but much faster
def initial_param_latin(bounds, guesses):  # guesses is an arbitrary input value
    final_vectors = np.zeros((len(bounds), guesses))
    while np.unique(final_vectors).size != final_vectors.size:  # While the all the elements are not unique, do the loop
        for i in range(final_vectors.shape[0]):
            for j in range(final_vectors.shape[1]):
                final_vectors[i, j] = np.random.randint(bounds[i][0] * 10, bounds[i][1] * 10) / 10
                # Generate random numbers with one decimal place
    return final_vectors


# Global Optimisation of Parameters attempt - generates the optimal parameters in the form of an array
def optimise_param(opt_func, opt_arg, opt_method, boundary, initial_param):

    # note that opt_arg is a tuple containing xy_data_coord, histo and matern_v
    if opt_method == 'nelder-mead':  # Uses only an arbitrary starting point
        # No bounds needed for Nelder-Mead
        # Have to check that all values are positive
        solution = scopt.minimize(fun=opt_func, args=opt_arg, x0=initial_param, method='Nelder-Mead')
        optimal_parameters = solution.x

    elif opt_method == 'latin-hypercube-de':  # everything is already taken as an input
        solution = scopt.differential_evolution(func=opt_func, bounds=boundary, args=opt_arg,
                                                init='latinhypercube')
        optimal_parameters = solution.x

    elif opt_method == 'latin-hypercube-manual':  # manual global optimization
        guesses = 10
        ind_parameters = np.zeros((len(boundary), guesses))
        ind_func = np.zeros(guesses)
        initial_param_stacked = initial_param_latin(boundary, guesses)  # using self-made function
        for i in range(len(boundary)):
            initial_param = initial_param_stacked[:, i]
            solution = scopt.minimize(fun=opt_func, args=opt_arg, x0=initial_param, method='Nelder-Mead')
            ind_parameters[:, i] = solution.x
            ind_func[i] = solution.fun
        opt_index = np.argmin(ind_func)
        optimal_parameters = ind_parameters[:, opt_index]

    return optimal_parameters


def mean_func_linear(grad, intercept, c):  # Should be the correct linear regression function
    # Create array for gradient component
    if np.array([c.shape]).size == 1:
        grad_c = np.arange(0, c.size, 1)
        linear_c = (np.ones(c.size) * intercept) + (grad * grad_c)
    else:
        grad_c = np.arange(0, c.shape[1], 1)
        linear_c = (np.ones(c.shape[1]) * intercept) + (grad * grad_c)
    return linear_c


"""Numerical Integration Methods"""
# MCMC Sampling, Laplace's Approximation, Bayesian Quadrature


def spatial_hessian(matrix):  # Generates the hessian with regards to location of elements on a matrix
    """Generate Hessian Matrix with finite differences - with multiple dimensions"""
    """Takes in any array/matrix and generates Hessian with 
    the dimensions (np.ndim(matrix), np.ndim(matrix), matrix.shape[0]. matrix.shape[1])"""

    matrix_grad = np.gradient(matrix)
    # Initialise Hessian - note the additional dimensions due to different direction of iteration
    hessian_matrix = np.zeros((np.ndim(matrix), np.ndim(matrix)) + matrix.shape)  # initialise correct dimensions
    for i, gradient_i in enumerate(matrix_grad):
        intermediate_grad = np.gradient(gradient_i)
        for j, gradient_ij in enumerate(intermediate_grad):
            hessian_matrix[i, j, :, :] = gradient_ij
    """Note the output will contain second derivatives in each dimensions (ii, ij, ji, jj), resulting in
    more dimensions in the hessian matrix"""
    return hessian_matrix


def jacobian(matrix):
    """Generate first derivative of a matrix - the jacobian"""
    matrix_grad = np.gradient(matrix)
    jacobian_matrix = np.zeros((np.ndim(matrix),) + matrix.shape)
    for i, gradient_i in enumerate(matrix_grad):
        jacobian_matrix[i, :, :] = gradient_i
    return jacobian_matrix


# Finding the inverse of a matrix of eigenvalues
def inv_eigenval(eigenvalue_matrix):
    eigenvalue_diag = np.diag(eigenvalue_matrix)
    inv_diag = 1 / eigenvalue_diag
    inv_eigenval_matrix = np.diag(inv_diag)
    return inv_eigenval_matrix


def mean_squared_error(regression_points, data_points):
    """
    Returns the mean squared error from the regression points and the data points.
    Have to make sure that both inputs are of the same size
    :param regression_points: predicted/ regressed values on the regression line
    :param data_points: actual observations
    :return: a single value which is the mean squared error
    """

    # Conduct check to see if they are of the same size
    if regression_points.size != data_points.size:
        print('Input arrays are not of the same size')
    else:
        n = regression_points.size
        error_vector = regression_points - data_points
        squared_error_vector = error_vector * error_vector
        mse = sum(squared_error_vector) / n

    return mse


def log_special(array):
    """
    If the element in the array is zero, then take the log as zero as well
    :param array: any data
    :return: log of the array
    """
    log_array = np.zeros_like(array)
    for i in range(array.size):
        if array[i] == 0:
            log_array[i] = 0
        else:
            log_array[i] = np.log(array[i])

    return log_array


def element_counter(element, array):
    """
    Counts the number of a particular element in an array
    :param element:
    :param array:
    :return: number of times the element appears in the array
    """
    # Initialise counter
    q = 0
    for i in range(array.size):
        if array[i] == element:
            q = q + 1
        else:
            q = q + 0

    return q


def rotate_array(angle_degrees, array, center):
    """
    Undergoes a 3-stage transformation
    1. Subtract the centre coordinate from each array point
    2. Perform rotation by multiplying with rotation matrix
    3. Add back the centre coordinate from each array point
    :param angle_degrees: Angle of Rotation
    :param array: x and y coordinates of points to be rotated
    :param center: center of rotation
    :return: The rotated coordinates
    """
    # Convert angle to radians
    radians = angle_degrees / 180 * np.pi

    # Start Subtracting Center, Rotate, then add back centre
    rotation_mat = np.array([[np.cos(radians), - np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    # rotation_mat = np.hstack((rotation_mat[0], rotation_mat[1]))
    # print(rotation_mat.shape)
    x_box = array[0] - center[0]
    y_box = array[1] - center[1]
    xy_box = np.vstack((x_box, y_box))
    xy_within_box = np.matmul(rotation_mat, xy_box)
    rotated_x = xy_within_box[0] + center[0]
    rotated_y = xy_within_box[1] + center[1]
    final_array = np.vstack((rotated_x, rotated_y))
    return final_array


def rotate_array_iterate(angle_degrees, array, center):
    """
    Undergoes a 3-stage transformation
    1. Subtract the centre coordinate from each array point
    2. Perform rotation by multiplying with rotation matrix
    3. Add back the centre coordinate from each array point
    :param angle_degrees: Angle of Rotation
    :param array: x and y coordinates of points to be rotated
    :param center: center of rotation
    :return: The rotated coordinates
    """
    # Convert angle to radians
    radians = angle_degrees / 180 * np.pi

    # Start Subtracting Center, Rotate, then add back centre
    rotation_mat = np.array([[np.cos(radians), - np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    x_box = array[0] - center[0]
    y_box = array[1] - center[1]
    xy_box = np.vstack((x_box, y_box))
    xy_within_box = np.matmul(rotation_mat, xy_box)
    rotated_x = xy_within_box[0] + center[0]
    rotated_y = xy_within_box[1] + center[1]
    final_array = np.vstack((rotated_x, rotated_y))
    return final_array


a = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
c = np.array([0, 0])
g = rotate_array_iterate(30, a, c)
print(g)

print(time.clock())








