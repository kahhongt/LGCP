import numpy as np
from numpy.lib import scimath
import time
import scipy.optimize as scopt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import functions as fn
import matplotlib.pyplot as plt


element_skew = np.arange(0, 5.1, 0.1)

log_likelihood = np.array([-906.2718154, -737.87057425, -750.57136623,  -638.17940407,  -583.72531313,
                                 -469.50191331,  -390.67958355,  -338.09984975,  -182.17886111,   -50.27135909,
                                 -106.25591247 , -264.53596819,  -382.93078154,  -425.65429421,  -512.24870158,
                                 -577.63201103,  -628.14129032,  -662.26836522 , -711.63709127,  -753.82597955,
                                 -754.64916653 , -726.80606138,  -870.55210142,  -850.76837337, -1019.13770512,
                                 -1057.17327349,  -775.2081794,   -854.66191232,  -823.47489368,  -880.99785118,
                                 -828.43989189 , -905.27891038,  -895.30150105,  -775.2869835 ,  -946.75293085,
                                 -818.94849897,  -810.28210139,  -872.29208711,  -861.17500851,  -965.04503605,
                                 -876.88438356, -1180.01029623,  -877.5873107,   -880.12332591,  -874.21409145,
                                 -1995.87743173, -1995.52308011, -2019.5090879,  -2047.98988731, -1592.41914564,
                                 -1568.31451458])

quad_n = np.array([ 841.,  643.,  531.,  417., 329.,  243.,  177.,  117.,   61.,   17.,   30.,   76.,
                    120.,  150.,  186.,  216.,  240.,  262.,  286.,  306.,  320.,  348.,  364.,  382.,
                    394.,  410.,  422.,  436.,  448.,  464.,  464.,  484.,  490.,  502.,  510.,  518.,
                    526.,  534.,  542.,  552.,  552.,  564.,  568.,  576.,  580.,  588.,  592.,  600.,
                    604.,  610.,  610.])

avg_log_likelihood = np.array([-1.07761215, -1.14754366, -1.4135054,  -1.53040624, -1.77424107, -1.93210664,
                        -2.20722929, -2.88974231, -2.98653871, -2.95713877, -3.54186375, -3.48073642,
                        -3.19108985,-2.83769529, -2.75402528, -2.67422227, -2.61725538, -2.52774185,
                        -2.48824158, -2.46348359, -2.35827865, -2.08852316, -2.39162665, -2.22714234,
                        -2.58664392, -2.5784714, -1.83698621, -1.96023374, -1.8381136,  -1.89870227,
                        -1.7854308 , -1.87041097, -1.82714592, -1.54439638, -1.8563783,  -1.58098166,
                        -1.54046027, -1.63350578, -1.58888378, -1.74826999, -1.58855867, -2.09221684,
                        -1.54504808, -1.52799189, -1.50726567, -3.39434937, -3.37081601, -3.36584848,
                         -3.39071173, -2.61052319, -2.5710074])

frob_norm = np.array([1.41421,  1.42127,  1.44222,  1.47648, 1.52315,  1.58114,  1.64924,  1.72627,
                 1.81108,  1.90263,  2.,       2.10238,  2.20907,  2.31948,  2.43311,  2.54951,
                 2.66833,  2.78927,  2.91204,  3.03645,  3.16228,  3.28938,  3.4176,   3.54683,
                 3.67696,  3.80789,  3.93954,  4.07185,  4.20476,  4.3382 ,  4.47214,  4.60652,
                 4.74131,  4.87647,  5.01199,  5.14782 , 5.28394,  5.42033,  5.55698,  5.69386,
                 5.83095,  5.96825,  6.10574,  6.2434,   6.38122,  6.5192,   6.65733,  6.79559,
                 6.93397,  7.07248,  7.2111])


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
likelihood_skew.set_xlabel('Off-Diagonal Entry')
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
avg_likelihood_skew.set_xlabel('Off-Diagonal Entry')
avg_likelihood_skew.set_ylabel('Average Log Marginal Likelihood')

plt.show()
"""
The time taken for iteration is 259.83513800000037
The globally-optimal matrix variables in terms of total are [ 1.   0.9  0.9  1. ]
The total globally-optimal log marginal likelihood is -2.95713877008
The globally-optimal matrix variables in terms of average are [ 1.  0.  0.  1.]
The average globally-optimal log marginal likelihood is -1.07761214674
"""