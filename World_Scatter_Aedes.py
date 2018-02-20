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

# World Data from 2012 to 2014
aedes_world = aedes_df[year_2013 & year_2014]

x_2014 = aedes_brazil_2014.values[:, 5].astype('float64')
y_2014 = aedes_brazil_2014.values[:, 4].astype('float64')
x_2013 = aedes_brazil_2013.values[:, 5].astype('float64')
y_2013 = aedes_brazil_2013.values[:, 4].astype('float64')
x_2013_2014 = aedes_brazil_2013_2014.values[:, 5].astype('float64')
y_2013_2014 = aedes_brazil_2013_2014.values[:, 4].astype('float64')


# World Coordinates from 2012 to 2014
world_x = aedes_df.values[:, 5].astype('float64')
world_y = aedes_df.values[:, 4].astype('float64')

# ------------------------------------------End of Data Collection

# ------------------------------------------Start of Plotting Process of Point Patterns, Histogram and Posterior Mean

world_fig = plt.figure()
world_fig.canvas.set_window_title('Global Aedes Occurrences in period 2012-2014')

world_scatter = world_fig.add_subplot(111)
pp_world = world_scatter.scatter(world_x, world_y, marker='.', color='darkred', s=0.1)
world_scatter.set_xlabel('UTM Horizontal Coordinate')
world_scatter.set_ylabel('UTM Vertical Coordinate')
plt.show()
# plt.legend([pp_2014, pp_2013], ["2014", "2013"])

