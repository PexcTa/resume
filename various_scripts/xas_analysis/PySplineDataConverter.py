# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:00:50 2022

@author: boris
"""

#%% LIBRARIES
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                                AutoMinorLocator, NullFormatter, ScalarFormatter)
# from scipy.signal import savgol_filter
# from scipy import special
import os

#%%

def convertData(file, energy_column, signal_column):
    with open(file) as current_data:
        fulldata = pd.read_csv(file, delim_whitespace = True, header = 0, engine = 'python')
        output = pd.DataFrame(fulldata.loc[:,energy_column])
        signal_column = pd.DataFrame(fulldata.loc[:, signal_column])
        output = pd.concat([output, signal_column], axis = 1)
        return output
        
full = convertData('Full.dat', 'e', 'flat')
full.to_csv('forspline.dat', index = False, header = False, sep = '\t')