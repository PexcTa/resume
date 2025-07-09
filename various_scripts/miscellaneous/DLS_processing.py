# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:19:18 2025

@author: boris
"""

import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, LogLocator)
#%% import the data
# set up for standard ascii files in .dat, .xyd or .xye formats
# make sure to run this block of code in a directory where you have data ONLY to avoid errors
# the files dictionary will contain the raw data 
files = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
        # skip directories
        continue
    with open(file):
        files[str(file)] = np.loadtxt(file,skiprows=2)
        
#%%
keys = ['tho2_milliq_4p3_day1', 'tho2_milliq_4p3_day28']
fig, axes = plt.subplots(len(keys), 1, figsize = (4, 4*len(keys)))
for i in range(len(keys)):
    x, y = files[keys[i]][:,0], files[keys[i]][:,1]
    widths = np.diff(x + [x[-1] * 10])
    x, y = files[keys[i]][:-1,0], files[keys[i]][:-1,1]
    axes[i].bar(x, y, widths, align='edge')
    axes[i].set_xscale('log')
    axes[i].xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
plt.show()

#%%
keys = ['tho2_milliq_hclo4_3p8_day1', 'tho2_milliq_hclo4_3p8_day28']
fig, axes = plt.subplots(len(keys), 1, figsize = (4, 4*len(keys)))
for i in range(len(keys)):
    x, y = files[keys[i]][:,0], files[keys[i]][:,1]
    widths = np.diff(x + [x[-1] * 10])
    x, y = files[keys[i]][:-1,0], files[keys[i]][:-1,1]
    axes[i].bar(x, y, widths, align='edge')
    axes[i].set_xscale('log')
    axes[i].xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
plt.show()
#%%
keys = ['ThO2_s1_pH3p8_milliQ_day2', 'ThO2_s1_pH3p8_milliQ_day14']
fig, axes = plt.subplots(len(keys), 1, figsize = (4, 4*len(keys)))
for i in range(len(keys)):
    x, y = files[keys[i]][:,0], files[keys[i]][:,1]
    widths = np.diff(x + [x[-1] * 10])
    x, y = files[keys[i]][:-1,0], files[keys[i]][:-1,1]
    axes[i].bar(x, y, widths, align='edge')
    axes[i].set_xscale('log')
    axes[i].xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
plt.show()

#%%
keys = ['ThO2_s5_25mM_oleate_day2', 'ThO2_s5_25mM_oleate_day14']
fig, axes = plt.subplots(len(keys), 1, figsize = (4, 4*len(keys)))
for i in range(len(keys)):
    x, y = files[keys[i]][:,0], files[keys[i]][:,1]
    widths = np.diff(x + [x[-1] * 10])
    x, y = files[keys[i]][:-1,0], files[keys[i]][:-1,1]
    axes[i].bar(x, y, widths, align='edge')
    axes[i].set_xscale('log')
    axes[i].xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
plt.show()