# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:30:20 2022

@author: boris
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:18:03 2021

@author: boris
"""


import csv
import itertools
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from scipy.signal import savgol_filter
from scipy.signal import correlate
from scipy import special
import os
from scipy.optimize import curve_fit

#%%
def normalize_1(data):
    """Normalizes a vector to its maximum value"""
    normalized_data = np.abs(data)/max(data)
    return normalized_data

def readfile(file):
    with open(file) as current_file:
        d_tmp = pd.read_csv(current_file, sep = ',', header = None, engine = 'python', skiprows = range(2,17))
        d = pd.DataFrame()
        d['time'] = d_tmp.iloc[0,:]
        d['counts'] = d_tmp.iloc[1,:]
        d_trimmed = d.iloc[33000:37000, :].reset_index()
    return d_trimmed


# def readset():
#     data_files = os.listdir()
#     dct = {}
#     for file in data_files:
#         dct[file] = readfile(file)
#     return dct


