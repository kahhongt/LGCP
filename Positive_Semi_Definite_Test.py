import pandas as pd
import math
import matplotlib
import numpy as np
import time
import functions as fn
import scipy
import scipy.special as scispec
import scipy.optimize as scopt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


"""Testing for Positive Semi Definite Matrices"""
m_1 = np.array([1.0846803387885862, 0.14482897477908155, 0.33745966422966023,
                1.3063136021938209, 0.047918247620715566, 1.038911858265929])

m_3 = np.array([1.147759315945861, 0.4364666324859926, 0.36122141052945933,
                1.5665971299780228, 0.35054853347625925, 1.908253765503943])

se = np.array([1.2104102367687857, 0.12777781473529062, 0.19271734392354523,
               1.745233331158524, 0.18934872826647065, 1.0257227969932683])

rq = np.array([1.2791241733080219, 0.2705424715127794, 0.15528741692787332,
               1.5067932347767945, 0.3795524985620355, 1.5899729828228477])

# Create function to convert them into matrices


def matrix_formation(mat):
    # Create new matrix fr
    """
    Generates the inverse of the covariance matrix
    :param mat: array containing 6 elements
    :return: inverse of the matrix
    """
    m11 = mat[0]
    m12 = mat[1]
    m13 = mat[2]
    m22 = mat[3]
    m23 = mat[4]
    m33 = mat[5]

    matrix = np.array([[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]])
    cov = np.linalg.inv(matrix)
    # cov_array = (cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2])
    return cov


m1_inv = matrix_formation(m_1)
m1_eig = np.linalg.eigvals(m1_inv)

m3_inv = matrix_formation(m_3)
m3_eig = np.linalg.eigvals(m3_inv)

se_inv = matrix_formation(se)
se_eig = np.linalg.eigvals(se_inv)

rq_inv = matrix_formation(rq)
rq_eig = np.linalg.eigvals(rq_inv)

print("The matern 1 covariance matrix is", m1_inv)
print("The eigenvalues are ", m1_eig)

print("The matern 3 covariance matrix is", m3_inv)
print("The eigenvalues are ", m3_eig)

print("The SE covariance matrix is", se_inv)
print("The eigenvalues are ", se_eig)

print("The RQ covariance matrix is", rq_inv)
print("The eigenvalues are ", rq_eig)

