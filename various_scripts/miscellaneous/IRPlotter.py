# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:05:14 2024

@author: boris
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
#%%
files = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
        # skip directories
        continue
    with open(file):
        if str(file).endswith('.spa'):
            files[str(file)[:-4]] = np.loadtxt(file, skiprows=20)
#%% 
def normalize(data, point='maximum', factor = 1):
    """
    Normalizes a vector to the maximum found within the vector.
    Negative values will be converted to positive ones.
    Parameters
    ----------
    data : array-like
    The array to normalize to its maximum.

    Returns
    -------
    normalized_data : array-like
    Normalized array of the same shape as passed in.

    """
    if point=='maximum':
        normalized_data = np.abs(data)/max(data)
    normalized_data *= factor
    return normalized_data
#%%
fig, ax = plt.subplots(figsize = (12,10))

for i in range(1,2):
    sig = f'3_ATR_{i}'
    ax.plot(files[sig][:,0], files[sig][:,1], label = sig)
ax.legend(loc = 'upper left', fontsize = 24)
ax.tick_params(axis = 'both', labelsize = 24)
    
ax.set_xlim([np.max(files[f'1_ATR_{i}'][:,0]), np.min(files[f'1_ATR_{i}'][:,0])])

#%% 
fs1 = 28
fs2 = 24
fig, ax = plt.subplots(figsize = (20, 18))
offsets = [0,0.75,0.6,0.55,0.52,0.51]
ax.set_prop_cycle('color',[plt.cm.gnuplot2(i) for i in np.linspace(0, 0.8, len(offsets))])
for i, dset in zip(range(6), files.keys()):
    ax.plot(files[dset][:,0], normalize(files[dset][:,1])-offsets[i]*i, linewidth = 3)
ax.set_xlabel('wavenumber, cm$^{-1}$', fontsize = fs1)
ax.set_ylabel('normalized absorption (a.u.)', fontsize = fs1)
ax.set_xlim([np.max(files[dset][:,0])+10, np.min(files[dset][:,0])-10])
# ax.set_xlim([1200, np.min(files[dset][:,0])-10])
ax.set_ylim([-2.5,2])
# ax.grid()
ax.set_yticks([])
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.tick_params(axis='x', labelsize = fs2)
ax.legend(labels = ['CePhos', 'ThPhos-4.8-1000$^{\circ}$C','ThPhos-7.5-1000$^{\circ}$C',
                    'ThPhos-4.8',
                    'ThPhos-7.5',
                    'ThPhos-4.8-6d'], loc = 'upper left', fontsize = fs1, ncol=2)
#%% 
fs1 = 24
fs2 = 18
fig, ax = plt.subplots(figsize = (12, 8))
offsets = [0,0.4]
ax.set_prop_cycle('color',['magenta', 'teal'])
for i, dset in zip(range(2), ['2_thphos', '3_thphos']):
    ax.plot(files[dset][:,0], normalize(files[dset][:,1])-offsets[i]*i, linewidth = 4)
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize = fs1)
ax.set_ylabel('Intensity (a.u.)', fontsize = fs1)
# ax.set_xlim([np.max(files[dset][:,0])+10, np.min(files[dset][:,0])-10])
# ax.set_xlim([1200, np.min(files[dset][:,0])-10])
ax.set_xlim([1300, 400])
ax.set_ylim([0,1.3])
ax.grid()
ax.set_yticks([])
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.tick_params(axis='x', labelsize = fs2)
ax.legend(labels = ['ThPhos-4.8-1000$^{\circ}$C','ThPhos-7.5-1000$^{\circ}$C'], frameon = False, loc = 'upper left', fontsize = fs1, ncol=1)
plt.tight_layout()
#%% 
fs1 = 28
fs2 = 24
fig, ax = plt.subplots(figsize = (10,10))
offsets = [0,0.7,0.6,0.6]
ax.set_prop_cycle('color',[plt.cm.gnuplot2(i) for i in (0, 0.56, 0.64, 0.80)])
for i, dset in zip(range(4), ['1_ce2403', '4_thphos', '5_thphos', '6_thphos']):
    ax.plot(files[dset][:,0], normalize(files[dset][:,1])-offsets[i]*i, linewidth = 3)
ax.set_xlabel('wavenumber, cm$^{-1}$', fontsize = fs1)
ax.set_ylabel('absorption (a.u.)', fontsize = fs1)
# ax.set_xlim([np.max(files[dset][:,0])+10, np.min(files[dset][:,0])-10])
ax.set_xlim([1500, np.min(files[dset][:,0])-10])
ax.set_ylim([-1.8,1.5])
ax.grid()
ax.set_yticks([])
ax.xaxis.set_major_locator(MultipleLocator(250))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.tick_params(axis='x', labelsize = fs2)
ax.legend(labels = ['CePhos', 
                    'ThPhos-4.8',
                    'ThPhos-7.5',
                    'ThPhos-4.8-6d'], loc = 'upper left', fontsize = fs2, ncol=2)