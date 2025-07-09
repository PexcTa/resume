# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:03:05 2021

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
from scipy.signal import savgol_filter as sgf
from scipy import special
# import itertools
# from sympy import S, symbols, printing
# from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
#%%
def normalize_1(data):
    """Normalizes a vector to it's maximum value"""
    normalized_data = data/max(data)
    return normalized_data
def nm2WN(data_in_nm):
    return 10000000/data_in_nm
def WN2nm(data_in_WN):
    return 10000000/data_in_WN
def WN2ev(data_in_WN):
    return data_in_WN/8065.544
def ev2WN(data_in_ev):
    return 8065.544*data_in_ev
def nm2ev(data_in_nm):
    """
    Converts a vector of nm values to eV values

    Parameters
    ----------
    data_in_nm : array-like
        vector to be changed to eV

    Returns
    -------
    formula : array-like
        output vector changed to eV.

    """
    h = 4.135667516*10**(-15)
    c = 299792458
    return (h*c)/(data_in_nm*10**(-9))
def ev2nm(data_in_ev):
    h = 4.135667516*10**(-15)
    c = 299792458
    return (h*c)/(data_in_ev)/10**(-9)
def jac(wavelength, data):
    """
    Converts a spectrum recorded on nm scale to eV scale

    Parameters
    ----------
    wavelength : array-like
        wavelength vector in nanometers
    
    data : array-like
        data vector recorded against the nanometer scale

    Returns
    -------
    formula : array-like
        output, spectral data rescaled to the energy scale, written for eV
    """
    e = 1.602*10**(-19)
    h = 4.135667516*10**(-15)
    c = 299792458
    jf = (e*(wavelength*10**-9)**2)/h*c*10**9
    formula = np.multiply(data, jf)
    return formula
#%%
def ReadEEMapFile(filename):
    with open(filename) as current_file:
        dataset = np.loadtxt(current_file, delimiter = ',', dtype = 'str')
        dataset[0,0] = '0'
        dataset = dataset.astype(float)
    return dataset
def ReadFile(filename, background=True):
    with open(filename) as current_file:
        dataset = np.loadtxt(current_file, delimiter = ',', dtype = 'str')
        if background:
            dataset[0,0:2] = ['0', '0']
        else: 
            dataset[0,0] = '0'
        dataset = dataset.astype(float)
    return dataset
#%%
    
#%%
pyrMap1 = ReadEEMapFile('pyr_map1.csv')
pyrMap2 = ReadEEMapFile('pyr_map2.csv')
tftMap1 = ReadEEMapFile('tft_map1.csv')
tftMap2 = ReadEEMapFile('tft_map2.csv')
pyrEm = ReadFile('pyr_em.csv')
tftEm = ReadFile('tft_em.csv')
#%%
fig, ax = plt.subplots(2,3,figsize=(30,20))

dset = pyrMap1
x = dset[0,1:]
y = dset[1:,0]
z = dset[1:,1:]

for axes in ax.reshape(-1):
    axes.tick_params(axis='x', labelsize= 30)
    axes.tick_params(axis='y', labelsize= 30)

cs1 = ax[0,0].contourf(x,y,z,levels=20,colormap='cividis')
ax[0,0].set_xlabel('Excitation wavelength (nm)', fontsize = 40)
ax[0,0].set_ylabel('Emission wavelength (nm)', fontsize = 40)

ax[0,1].set_xlim((3.0, 2.1))
for i in range(1, len(x)):
    ax[0,1].plot(nm2ev(y), sgf(jac(y, z[:,i]), 9,3), linewidth = 4, label = f"{x[i]:.0f} nm exc.")    
ax[0,1].legend(loc = 'upper right', fontsize = 30)
ax[0,1].set_xlabel('Emission wavelength (nm)', fontsize = 40)
ax[0,1].set_ylabel('Intensity (cts)', fontsize = 40)

for i in range(1, len(x)):
    ax[0,2].scatter(nm2ev(x)[i], np.sum(np.multiply(nm2ev(y), jac(y, z[:,i]))/np.sum(jac(y, z[:,i]))), s = 500)
ax[0,2].set_xlabel('Excitation wavelength (nm)', fontsize = 40)
ax[0,2].set_ylabel('Spectral moment (nm)', fontsize = 40)


dset = tftMap1
x = dset[0,1:]
y = dset[1:,0]
z = dset[1:,1:]


cs1 = ax[1,0].contourf(x,y,z,levels=20,colormap='cividis')
ax[1,0].set_xlabel('Excitation wavelength (nm)', fontsize = 40)
ax[1,0].set_ylabel('Emission wavelength (nm)', fontsize = 40)

ax[1,1].set_xlim((3.0, 2.1))
for i in range(len(x)):
    ax[1,1].plot(nm2ev(y), sgf(jac(y, z[:,i]), 9,3), linewidth = 4, label = f"{x[i]:.0f} nm exc.")    
ax[1,1].legend(loc = 'upper right', fontsize = 30)
ax[1,1].set_xlabel('Emission wavelength (nm)', fontsize = 40)
ax[1,1].set_ylabel('Intensity (cts)', fontsize = 40)

for i in range(len(x)):
    ax[1,2].scatter(nm2ev(x)[i], np.sum(np.multiply(nm2ev(y), jac(y, z[:,i]))/np.sum(jac(y, z[:,i]))), s = 500)
ax[1,2].set_xlabel('Excitation wavelength (nm)', fontsize = 40)
ax[1,2].set_ylabel('Spectral moment (nm)', fontsize = 40)


plt.tight_layout()
#%%
acidMap = ReadEEMapFile('acidExMap.csv')
neutMap = ReadEEMapFile('neutExMap.csv')
acidEm = ReadFile('acidEm.csv')
neutEm = ReadFile('neutEm.csv')
#%%
fig, ax = plt.subplots(2,2,figsize=(20,20))
for axes in ax.reshape(-1):
    axes.tick_params(axis='x', labelsize= 30)
    axes.tick_params(axis='y', labelsize= 30)
    axes.yaxis.get_offset_text().set_fontsize(27)
    
cmap1 = matplotlib.cm.get_cmap('cool')
colors = np.linspace(0.1, 0.99, 5)

dset = neutEm
x = dset[:,0]
y = dset[:,1]
ax[0,0].plot(nm2ev(x), sgf(jac(x, y), 9,3), color = 'navy', linewidth = 4, label = "Neutral film")
ax[0,0].legend(loc = 'upper right', fontsize = 30)
ax[0,0].set_xlabel('Wavelength (nm)', fontsize = 40)
ax[0,0].set_ylabel('Emission Intensity (cts)', fontsize = 40)
ax[0,0].set_xlim((3.0, 1.9))
for i in range(len(neutMap[1:,0])):
    ax[0,0].axvline(nm2ev(neutMap[i+1,0]), color = cmap1(colors[i]), linestyle = '--', linewidth = 3)
    

dset = acidEm
x = dset[:,0]
y = dset[:,1]
ax[1,0].plot(nm2ev(x), sgf(jac(x, y), 9,3), color = 'crimson', linewidth = 4, label = "Acidified film")
ax[1,0].legend(loc = 'upper right', fontsize = 30)
ax[1,0].set_xlabel('Wavelength (nm)', fontsize = 40)
ax[1,0].set_ylabel('Emission Intensity (cts)', fontsize = 40)
ax[1,0].set_xlim((3.0, 1.9))
for i in range(len(neutMap[1:,0])):
    ax[1,0].axvline(nm2ev(neutMap[i+1,0]), color = cmap1(colors[i]), linestyle = '--', linewidth = 3)

dset = neutMap
x = dset[0,20:]
for i in range(len(dset[1:,0])):
    ax[0,1].plot(nm2ev(x), sgf(jac(x, dset[i+1, 20:]), 9,3), linewidth = 4, color = cmap1(colors[i]), label = f"PL at {nm2ev(dset[i+1,0]):.2f} eV")
    ax[0,1].legend(loc = 'upper left', fontsize = 30)
    ax[0,1].set_xlabel('Wavelength (nm)', fontsize = 40)
    ax[0,1].set_ylabel('Excitation Intensity (cts)', fontsize = 40)
    ax[0,1].set_xlim((4.5,3.1))
    
dset = acidMap
x = dset[0,20:]
for i in range(len(dset[1:,0])):
    ax[1,1].plot(nm2ev(x), sgf(jac(x, dset[i+1, 20:]), 9,3), linewidth = 4, color = cmap1(colors[i]), label = f"PL at {nm2ev(dset[i+1,0]):.2f} eV")
    ax[1,1].legend(loc = 'upper left', fontsize = 30)
    ax[1,1].set_xlabel('Wavelength (nm)', fontsize = 40)
    ax[1,1].set_ylabel('Excitation Intensity (cts)', fontsize = 40)
    ax[1,1].set_xlim((4.5,3.1))
    
    
plt.tight_layout()