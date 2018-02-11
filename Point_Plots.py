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
import matplotlib.ticker as ticker


"""SIGACTS DATA"""
# Test extracting values from csv
"""Extract Data from csv"""  # Arbitrary Point Process Data
df_sigacts = pd.read_csv('SIGACTS_KH.csv')  # Generates a DataFrame from csv - sigacts data

# Creating Boolean variables to indicate direct or indirect fire
direct_fire = df_sigacts['category'] == "Direct Fire"
indirect_fire = df_sigacts['category'] == "Indirect Fire"

# Taking in those boolean variables to obtain the arrays - remember to add the values
x_direct = df_sigacts[direct_fire].values[:, 8]
y_direct = df_sigacts[direct_fire].values[:, 9]
x_indirect = df_sigacts[indirect_fire].values[:, 8]
y_indirect = df_sigacts[indirect_fire].values[:, 9]

sigacts_fig = plt.figure()
sigacts_fig.canvas.set_window_title('Direct and Indirect Fire')  # Changing the title of the figure

d_incidence = sigacts_fig.add_subplot(211)
i_incidence = sigacts_fig.add_subplot(212)

direct_incidence = d_incidence.scatter(x_direct, y_direct, color='red', marker='.', s=0.1)
indirect_incidence = i_incidence.scatter(x_indirect, y_indirect, color='blue', marker='.', s=0.1)

# incidence.legend([direct_incidence, indirect_incidence], ['Direct Fire', "Indirect Fire"])

"""Zika Virus Spread"""
zika_df = pd.read_csv('Zika_PP_Data.csv')  # generates dataframe from csv - zika data
taiwan = zika_df['COUNTRY'] == "Taiwan"  # Create boolean variable
taiwan_main_island = zika_df['X'] > 120
taiwan_published = zika_df['SOURCE_TYPE'] == "published"
zika_taiwan_df = zika_df[taiwan & taiwan_main_island]  # Only take in values from the main island
taiwan_x = zika_taiwan_df.values[:, 6]
taiwan_y = zika_taiwan_df.values[:, 5]

taiwan_fig = plt.figure()
taiwan_fig.canvas.set_window_title('Taiwan Zika Outbreak')
taiwan_spread = taiwan_fig.add_subplot(111)
taiwan_spread.scatter(taiwan_x, taiwan_y, color='black', marker='.', s=0.1)

"""Theft Crime in Chicago"""
theft_df = pd.read_csv('Theft_Clean.csv')  # generates dataframe from csv - zika data
theft_date = theft_df.values[:, 0]
theft_x = theft_df.values[:, 2]
theft_y = theft_df.values[:, 3]

theft_fig = plt.figure()
theft_fig.canvas.set_window_title('Chicago Theft Locations')
theft_spread = theft_fig.add_subplot(111)
theft_spread.scatter(theft_x, theft_y, color='black', marker='.', s=0.0001)

print(zika_taiwan_df)

plt.show()


