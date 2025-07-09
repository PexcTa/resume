# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:18:48 2025

based on 10.1016/j.nimb.2005.12.034

@author: boris
"""

import numpy as np
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)

def derivative(data):
    dRdwl = []
    dwl = 0.05 # standard for exafs
    for i in range(len(data[:,0])):
        dRdwl = np.gradient(data[:,-3], dwl)
    return dRdwl

chiqDict = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
    # skip directories
        continue
    with open(file):
        if str(file).endswith('.chiq'):
            chiqDict['headers'] = open(file).readlines()[37]
            chiqDict[file[:-5]] = np.loadtxt(file, dtype = float, skiprows=38)
#%%

samples = ['2415_T_merge_Phase_P']
labels = ['Sample 2415, P shells']

fig, ax = plt.subplots(figsize = (12, 6))
fs1 = 26
fs2 = 22

colors = np.linspace(0.3, 1, len(samples))
for i in range(len(samples)):
    data = chiqDict[samples[i]]
    x = data[:,0]
    y = data[:,-3]
    ax.plot(x,y-10*i, linewidth = 3.5, color = plt.cm.rainbow(colors[i]), label = labels[i])
    
ax.legend(loc = 'best', fontsize = fs2)
ax.tick_params(axis='both', labelsize= fs2)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_yticks([])
ax.set_xlim([3, 14])

ax.set_xlabel("k ($\AA^{-1}$)", fontsize = fs1) 
ax.set_ylabel("$\phi$(k) ($\AA^{-3}$)", fontsize = fs1) 

plt.show()

#%%

samples = ['2415_T_merge_Phase_P']
labels = ['Sample 2415, P shells']

fig, ax = plt.subplots(figsize = (12, 6))
fs1 = 26
fs2 = 22

colors = np.linspace(0.3, 1, len(samples))
for i in range(len(samples)):
    data = chiqDict[samples[i]]
    x = data[:,0]
    y = abs(derivative(data))
    low_cutoff = np.argmin(abs(x - 4))
    # ax.axvline(x = x[low_cutoff], color = 'black')
    y_cut, x_cut = y[low_cutoff:], x[low_cutoff:]
    maximum = np.max(y_cut)
    ax.plot(x,y-3*i, linewidth = 3.5, color = plt.cm.rainbow(colors[i]), label = labels[i])
    # ax.axvline(x = x_cut[np.argmin(abs(y_cut - maximum))], color = plt.cm.rainbow(colors[i]), linestyle = '--', linewidth = 2)
    # ax.annotate(f"{x_cut[np.argmin(abs(y_cut - maximum))]}", (x_cut[np.argmin(abs(y_cut - maximum))],y_cut[np.argmin(abs(y_cut - maximum))]-8),color = plt.cm.rainbow(colors[i]),bbox=dict(facecolor='gray', alpha=0.2), fontsize = fs2)
    
ax.legend(loc = 'best', fontsize = fs2)
ax.tick_params(axis='both', labelsize= fs2)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_yticks([])
ax.set_xlim([3, 14])

ax.axvline(x = 6.55, color = plt.cm.rainbow(colors[i]), linestyle = '--', linewidth = 2)

ax.set_xlabel("k ($\AA^{-1}$)", fontsize = fs1) 
ax.set_ylabel("d$\phi$(k)/dk ($\AA^{-4}$)", fontsize = fs1) 

plt.show()