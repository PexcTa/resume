# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:55:33 2022

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
def readTrace(file):
    with open(file) as current_file:
        d_tmp = pd.read_csv(current_file, sep = ',', header = None, engine = 'python', skiprows = range(2,16))
        d = pd.DataFrame()
        d['time'] = d_tmp.iloc[0,:]
        d['counts'] = d_tmp.iloc[1,:]
        d_trimmed = d.iloc[33000:, :].reset_index(drop=True)
        d_trimmed['time'] = d_trimmed['time']-d_trimmed['time'][np.argmax(d_trimmed['counts'])]
    return d_trimmed
def readFile(file):
    with open(file) as current_file:
        d_tmp = pd.read_csv(current_file, sep = ',', header = 0, engine = 'python')
    return d_tmp
def rebin(data, step):
    d_rebinned = pd.DataFrame(columns=['time', 'counts'])
    for i, j in zip(range(0, len(data)-step, step), range(step, len(data), step)):
        d_rebinned = d_rebinned.append({'time': np.average(data['time'].iloc[i:j]), 'counts':np.sum(data['counts'].iloc[i:j])}, ignore_index = True)
    return d_rebinned
def readset():
    data_files = os.listdir()
    dct = {}
    for file in data_files:
        dct[file] = readTrace(file)
    return dct
def saver(dictex):
    for key, val in dictex.items():
        val.to_csv("fits_without_rise_{}.csv".format(str(key)))

    with open("keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))
        
#%%
fig, ax = plt.subplots(figsize=(16,16))
ax.set_xlabel("Time (min)", fontsize = 36)
ax.set_ylabel("Normalized Intensity", fontsize = 36)
ax.tick_params(axis='x', labelsize= 30)
ax.tick_params(axis='y', labelsize= 30)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.set_xlim([5, 60])
ax.set_ylim([0.2, 1.05])

ax.set_prop_cycle(color = ['green', 'purple', 'blue', 'black'])
samples = ['ammonia', 'tea', 'pyridine', 'acetone']
for sample in samples:
    ax.plot(data['time']/60, normalize_1(data[sample]), linewidth = 3)
ax.legend(labels = ['Ammonia', 'Triethylamine', 'Pyridine', 'Acetone'], loc = 'lower right', fontsize = 30)
ax.axhline(y=0, color = 'dimgrey', ls = '--')
#%%
fig, ax = plt.subplots(figsize=(16,16))
ax.set_xlabel("Time (min)", fontsize = 36)
ax.set_ylabel("Intensity (CPS)", fontsize = 36)
ax.tick_params(axis='x', labelsize= 30)
ax.tick_params(axis='y', labelsize= 30)
ax.yaxis.offsetText.set_fontsize(25)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
# ax.set_xlim([5, 60])
# ax.set_ylim([0.2, 1.05])


sample = 'acetone'

ax.plot(data['time']/60, data[sample], linewidth = 3, color = 'black', label = 'Acetone')
ax.legend(loc = 'lower right', fontsize = 30)
ax.axhline(y=0, color = 'dimgrey', ls = '--')
    