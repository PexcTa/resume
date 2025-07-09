# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 09:56:08 2022

@author: boris
"""

#%% LIBRARIES
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from scipy.signal import savgol_filter
from scipy import special
import os
#%%
def ReadFile(filename, header_list, skip=4):
    """Set this up with the header corresponding to actual paths in a readable form"""
    with open(filename) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, header = 0, names = header_list, engine = 'python', skiprows = lambda x: x in range(skip))
        return d
    
# test= ReadFile('nlink_1_3.dat', ['R', 'ON_1', 'OL_1', 'CL_1', 'Zr_1', 'OL_2', 'ON_2', 'OH', 'H2O'])

def ReadDire(header_list, skip=4):
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".dat"):
            data[file]=ReadFile(file, header_list)
    return data

names = ['R', 'ON_1', 'OL_1', 'CL_1', 'Zr_1', 'MS_1', 'MS_2', 'MS_3', 'OH']
dsre = ReadDire(names)

#%% Plot a Bunch of Data in R-space 

fig, ax = plt.subplots(figsize=(12,10))
ax.set_xlabel("R($\AA$)", fontsize = 24)
ax.set_ylabel("|\u03C7(R)|($\AA^{-3}$)", fontsize = 24)
ax.tick_params(axis='x', labelsize= 20)
ax.tick_params(axis='y', labelsize= 20)
ax.set_xlim([1,4.1])
ax.set_ylim([-0.5, 24])
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
ax.axhline(y=0, color = 'dimgrey', ls = '--')
ax.set_prop_cycle('color',[plt.cm.plasma(i) for i in np.linspace(0, 1, 7)])
ax.set_facecolor('gainsboro')
# data = dsim['NL12Im.dat']
# for col in data.columns[1:]:
#     ax.scatter(data['R'], data[col], s = 20, alpha = 0.5)
# ax.plot(data['R'], data.iloc[:,1:].sum(axis=1), color = 'black', linewidth=3)
ax.plot(dsim['NL4.dat']['R'], np.sqrt(dsre['NL4.dat'].iloc[:,1:].sum(axis=1)**2+dsim['NL4.dat'].iloc[:,1:].sum(axis=1)**2), linewidth = 3, label = '4-connected')
ax.plot(dsim['NL4.dat']['R'], np.sqrt(dsre['NL6.dat'].iloc[:,1:].sum(axis=1)**2+dsim['NL6.dat'].iloc[:,1:].sum(axis=1)**2), linewidth = 3, label = '6-connected')
ax.plot(dsim['NL4.dat']['R'], np.sqrt(dsre['NL8.dat'].iloc[:,1:].sum(axis=1)**2+dsim['NL8.dat'].iloc[:,1:].sum(axis=1)**2), linewidth = 3, label = '8-connected')
ax.plot(dsim['NL4.dat']['R'], np.sqrt(dsre['NL10.dat'].iloc[:,1:].sum(axis=1)**2+dsim['NL10.dat'].iloc[:,1:].sum(axis=1)**2), linewidth = 3, label = '10-connected')
ax.plot(dsim['NL4.dat']['R'], np.sqrt(dsre['NL12.dat'].iloc[:,1:].sum(axis=1)**2+dsim['NL12.dat'].iloc[:,1:].sum(axis=1)**2), linewidth = 3, label = '12-connected')
ax.legend(loc = 'upper right', fontsize=24)
# plt.savefig('ppNU1k_EmMaxs_vs_sqRind_plot.svg')

