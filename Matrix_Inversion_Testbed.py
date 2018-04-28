import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn


""""Simple Script to be used to invert matrices """

matern1 = np.array([1.2112,	0.3646,	0.2103,	1.8497,	0.0251,	1.911])
matern3 = np.array([1.4701,	0.3590,	0.2241,	1.0202,	0.1867,	1.7143])
se = np.array([1.5537,	0.0391,	0.1944,	1.2131,	0.2103,	1.5075])
rq = np.array([1.9614,	0.2571,	0.2423,	1.6139,	0.2780,	1.7074])

print(matern1[2])


def matrix_inv(mat):
    # Create new matrix from the tuple
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


print('Matern1 cov is', matrix_inv(matern1))
print('Matern3 cov is', matrix_inv(matern3))
print('SE cov is', matrix_inv(se))
print('RQ cov is', matrix_inv(rq))



