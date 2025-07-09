# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:29:17 2021

@author: boris
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

#%%
def ReadData(file, fpeaks):
    with open(file) as current_file:
        d = np.loadtxt(current_file, delimiter = ';')
        return d

# def ReadAllData():
#     data_files = os.listdir()
#     data = {}
#     for file in data_files:
#         if file.endswith(".csv"):
#             data[file] = ReadData(file)
#     return data
n1s = ReadData('N1s.csv', 8)
zr3d = ReadData('Zr3d.csv', 8)

#%%
fig, ax = plt.subplots(figsize = (10,10))
ax.set_ylabel('Counts', fontsize = 20)


dset = CoSIM.copy()

ax.set_xlim(max(dset['BE']), min(dset['BE']))
ax.yaxis.set_major_locator(MultipleLocator(5000))

ax.yaxis.set_minor_locator(MultipleLocator(1000))
ax.tick_params(axis = 'y', labelsize = 16)

ax.scatter(dset['BE'], dset['Data'], s = 30, facecolors = 'none', color = 'darkmagenta', label = 'Co-SIM')

for peak in dset.columns[2:5]:
    ax.plot(dset['BE'], dset[peak], linewidth = 2, color = 'firebrick')
    plt.fill_between(dset['BE'], dset[peak], dset['BG'], color = 'firebrick', alpha = 0.3)
ax.plot(dset['BE'], dset.iloc[:,5], linewidth = 2, color = 'firebrick', label = 'CoO')
plt.fill_between(dset['BE'], dset.iloc[:,5], dset['BG'], color = 'firebrick', alpha = 0.3)


for peak in dset.columns[6:9]:
    ax.plot(dset['BE'], dset[peak], linewidth = 2, color = 'steelblue')
    plt.fill_between(dset['BE'], dset[peak], dset['BG'], color = 'steelblue', alpha = 0.3)
ax.plot(dset['BE'], dset.iloc[:,9], linewidth = 2, color = 'steelblue', label = 'Co(OH)$_2$')
plt.fill_between(dset['BE'], dset.iloc[:,9], dset['BG'], color = 'steelblue', alpha = 0.3)



ax.plot(dset['BE'], dset['BG'], linewidth = 2, linestyle = '--', color = 'black', label = 'Background')
ax.plot(dset['BE'], dset['Env'], linewidth = 3, color = 'red', label = 'Fit')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

ax.legend(loc = 'upper left', fontsize = 18)

difference = dset['Data'] - dset['Env']
frame2=fig.add_axes((0.125,0.05,0.775,0.1))
frame2.set_xlim(max(dset['BE']), min(dset['BE']))        
frame2.plot(dset['BE'],difference,color='magenta')
frame2.set_xlabel('Binding Energy, eV', fontsize = 20)
frame2.set_ylabel('$\Delta$I', fontsize = 20)
frame2.xaxis.set_major_locator(MultipleLocator(5))

frame2.xaxis.set_minor_locator(MultipleLocator(1))
frame2.tick_params(axis = 'x', labelsize = 16)
frame2.tick_params(axis = 'y', labelsize = 16)
plt.grid()


#%%
fig, ax = plt.subplots(figsize = (10,10))
ax.set_ylabel('Counts', fontsize = 24)

dset = zr3d.copy()
ax.set_xlim(max(dset[:,0]-1), min(dset[:,0]))
ax.plot(dset[:,0], dset[:,1], color = 'gray', linewidth = 1.5, label = 'Zr 3d')

for i in range(2,4):
    ax.plot(dset[:,0], dset[:,i], linewidth = 3, linestyle = '--', color = 'forestgreen')
    plt.fill_between(dset[:,0], dset[:,i], dset[:,-3], color = 'forestgreen', alpha = 0.3)
ax.plot(dset[:,0], dset[:,-3], linewidth = 2, color = 'black', label = 'Background')
ax.plot(dset[:,0], dset[:,-2], linewidth = 4, color = 'forestgreen', label = 'Fit')

ax.tick_params(axis = 'both', labelsize = 24)
ax.legend(loc = 'upper left', fontsize = 24)
frame2=fig.add_axes((0.125,0.05,0.775,0.1))
frame2.set_xlim(max(dset[:,0]-1), min(dset[:,0]))     
frame2.plot(dset[:,0],dset[:,-1],color='forestgreen')
frame2.set_xlabel('Binding Energy, eV', fontsize = 24)
frame2.set_ylabel('$\Delta$I', fontsize = 24)
frame2.xaxis.set_major_locator(MultipleLocator(5))
frame2.xaxis.set_minor_locator(MultipleLocator(1))
frame2.tick_params(axis = 'x', labelsize = 24)
frame2.tick_params(axis = 'y', labelsize = 24)
plt.grid()
