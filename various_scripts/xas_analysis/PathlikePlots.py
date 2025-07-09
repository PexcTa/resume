# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:19:57 2024

@author: boris
"""

import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter as sgf
from scipy import special
import os
from collections import OrderedDict

#%%
file1 = np.loadtxt("Kphase_Feff.dat", dtype = float, skiprows = 4)
file2 = np.loadtxt("Naphase_Feff.dat", dtype = float, skiprows = 4)
# file3 = np.loadtxt("thphosv2_paths_rmag.dat", dtype = float, skiprows = 4)
# file4 = np.loadtxt("thphosv2_paths_real.dat", dtype = float, skiprows = 4)
# file2 = np.loadtxt("dompaths_rmag.dat", dtype = float, skiprows = 4)

#%%

fig, ax = plt.subplots(figsize = (5,5))
fs1 = 16
fs2 = 12
ax.set_xlim(-0.1,6.1)
ax.set_ylabel("|\u03C7(R)|($\AA^{-3}$)", fontsize = fs1)
ax.set_xlabel("R($\AA$)", fontsize = fs1)
ax.tick_params('both', labelsize=fs2)
ax.plot(file2[:,0], file2[:,3], linewidth = 2.5, color = 'gray', alpha = 0.75)
# ax.plot(file1[:,0], file1[:,2], linewidth = 3, color = 'firebrick', alpha = 0.75)
# ax.plot(file1[:,0], file1[:,3], linewidth = 2.5, color = 'forestgreen', alpha = 0.5)
# ax.plot(file3[:,0], file3[:,5], linewidth = 3.5, color = 'black')
ax.set_prop_cycle('color', [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1, len(file2[0,:]))])
for i in range(4,len(file2[0,:])):
    ax.plot(file2[:,0], file2[:,i], linewidth = 1, linestyle = '--', alpha = 1)
# ax.legend(labels = ['Total', 'Np - O$_{ax}$', 'Np - O$_{eq}$', 'Np - C$_{ax}$', 'Np - O$_{dist}$','Np - K'], loc = 'upper right', ncol = 1, fontsize = fs2)
# ax.legend(labels = ['Data', 'Fit', 'Window', 'Th - O$_{1,1}$', 'Th - O$_{1,2}$', 'Th - O$_{1,3}$', 'Th - P$_{2}$', 'Th - P$_{3, 1}$','Th - P$_{3, 2}$','Th - Na$_{4}$',  'Th - Th$_{4}$'], 
           # loc = 'upper right', ncol = 2, fontsize = fs2)
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 1.75)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
# ax.set_ylim([-0.3, 7.6])
plt.show()
#%%

fig, ax = plt.subplots(figsize = (5,5))
fs1 = 16
fs2 = 12
ax.set_xlim(-0.1,6.1)
ax.set_ylabel("|\u03C7(R)|($\AA^{-3}$)", fontsize = fs1)
ax.set_xlabel("R($\AA$)", fontsize = fs1)
ax.tick_params('both', labelsize=fs2)
ax.plot(file1[:,0], file1[:,-1], linewidth = 2.5, color = 'gray', alpha = 0.75)
# ax.plot(file1[:,0], file1[:,2], linewidth = 3, color = 'firebrick', alpha = 0.75)
# ax.plot(file1[:,0], file1[:,3], linewidth = 2.5, color = 'forestgreen', alpha = 0.5)
# ax.plot(file3[:,0], file3[:,5], linewidth = 3.5, color = 'black')
ax.set_prop_cycle('color', [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1, len(file2[0,1:]))])
for i in range(1,len(file2[0,:-6])):
    ax.plot(file2[:,0], file2[:,i], linewidth = 1, linestyle = '--', alpha = 1)
# ax.legend(labels = ['Total', 'Np - O$_{ax}$', 'Np - O$_{eq}$', 'Np - C$_{ax}$', 'Np - O$_{dist}$','Np - K'], loc = 'upper right', ncol = 1, fontsize = fs2)
# ax.legend(labels = ['Data', 'Fit', 'Window', 'Th - O$_{1,1}$', 'Th - O$_{1,2}$', 'Th - O$_{1,3}$', 'Th - P$_{2}$', 'Th - P$_{3, 1}$','Th - P$_{3, 2}$','Th - Na$_{4}$',  'Th - Th$_{4}$'], 
           # loc = 'upper right', ncol = 2, fontsize = fs2)
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 1.75)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
# ax.set_ylim([-0.3, 7.6])
plt.show()
#%%

fig, ax = plt.subplots(figsize = (12,12))
fs1 = 28
fs2 = 24
ax.set_xlim(-0.1,6.1)
ax.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
ax.set_xlabel("R($\AA$)", fontsize = fs1)
ax.tick_params('both', labelsize=fs2)
ax.plot(file1[:,0], file1[:,1], linewidth = 3.5, alpha = 1, color = 'black')
# ax.plot(file4[:,0], file4[:,3], linewidth = 3.5, color = 'green')
# ax.plot(file4[:,0], file4[:,5], linewidth = 3.5, color = 'black')
ax.set_prop_cycle('color', [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1,  len(file1[0,:]))])
for i in range(1,len(file1[0,:])):
    ax.plot(file1[:,0], file1[:,i], linewidth = 2.5, linestyle = '--', alpha = 1)
# ax.legend(labels = ['Total', 'Pu - O$_{1,1}$', 'Pu - O$_{1,2}$', 'Pu - O$_{1,3}$', 'Pu - P$_{2}$','Pu - O$_{2}$', 'Pu - P$_{3}$', 'Pu - K$_{3}$', 'Pu - P$_{4}$','Pu - K$_{4}$', 'Pu - Pu'], loc = 'upper right', ncol = 2, fontsize = fs2-4)
# ax.legend(labels = ['Total', 'Th - O$_{1,1}$', 'Th - O$_{1,2}$', 'Th - O$_{1,3}$', 'Th - P$_{2}$', 'Th - P$_{3}$', 'Th - Th$_{4}$','Th - O$_{3}$',  'Th - O$_{1}$ - P$_{2}$', 'Th - O$_{1}$ - P$_{2}$','Th - O$_{1}$ - P$_{3}$','Th - O$_{1}$ - P$_{3}$','Th - O$_4$'], loc = 'upper right', ncol = 2, fontsize = fs2)
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 1.75)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
# ax.set_ylim([-0.3, 7.6])
plt.show()
#%%
fig, ax = plt.subplots(figsize = (12,8))
fs1 = 24
fs2 = 18
ax.set_xlim(-0.1,5.1)
ax.set_ylabel("|\u03C7(R)|($\AA^{-3}$)", fontsize = fs1)
ax.set_xlabel("R($\AA$)", fontsize = fs1)
ax.tick_params('both', labelsize=fs2)
ax.plot(file1[:,0], file1[:,1], linewidth = 3.5, color = 'gray', label = 'Data')
ax.plot(file1[:,0], file1[:,2], linewidth = 3.5, color = 'red', label = 'Fit 2')
ax.plot(file1[:,0], file1[:,3], linewidth = 3, linestyle = '--', color = 'green', label = 'Window')
ax.set_prop_cycle('color', [plt.cm.cividis(i) for i in np.linspace(0, 1, 5)])
ax.plot(file1[:,0], file1[:,4], linewidth = 3, linestyle = '-.', label = 'Pu-O$_{(1)}$')
ax.plot(file1[:,0], file1[:,5], linewidth = 3, linestyle = '-.', label = 'Pu-P$_{(2)}$')
ax.plot(file1[:,0], file1[:,6], linewidth = 3, linestyle = '-.', label = 'Pu-P$_{(3)}$')
ax.plot(file1[:,0], file1[:,7], linewidth = 3, linestyle = '-.', label = 'Pu-Pu$_{(4)}$')
ax.plot(file1[:,0], file1[:,8], linewidth = 3, linestyle = '-.', label = 'Pu-O$_{(5)}$')
# ax.set_prop_cycle('color', [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1, 11)])
# for i in range(1,12):
#     ax.plot(file2[:,0], file2[:,i], linewidth = 2.5, linestyle = '--', alpha = 1)
ax.legend(loc = 'upper left', fontsize = fs2)
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 1.75)
# ax.set_ylim([-0.3, 7.6])