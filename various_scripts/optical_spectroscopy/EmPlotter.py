# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:10:07 2022

@author: boris
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
import matplotlib
from sympy import S, symbols, printing
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter as sgf
import os

#%%
data = {}
for file in os.listdir():
    data[file] = np.loadtxt(open(file).readlines()[:-1], delimiter = '\t', skiprows=4, dtype=None)

#%%
def normalize_1(data):
    """Normalizes a vector to it's maximum value"""
    normalized_data = data/max(data)
    return normalized_data
def nm2WN(data_in_nm):
    return 10000000/data_in_nm
def WN2nm(data_in_WN):
    return 10000000/data_in_WN
def nm2ev(data_in_nm):
    h = 4.135667516*10**(-15)
    c = 299792458
    return (h*c)/(data_in_nm*10**(-9))
def ev2nm(data_in_ev):
    h = 4.135667516*10**(-15)
    c = 299792458
    return (h*c)/(data_in_ev)/10**(-9)
def jac(wavelength, data):
    e = 1.602*10**(-19)
    h = 4.135667516*10**(-15)
    c = 299792458
    jf = (e*(wavelength*10**-9)**2)/h*c*10**9
    return np.multiply(data, jf)


#%% plot a csv
fig, ax = plt.subplots(figsize=(12,12))
# ax.set_xlabel('Energy $(cm^{-1})$', fontsize = 20)
ax.set_xlabel('Energy (eV)', fontsize = 30)
ax.set_ylabel("Emission (cts)", fontsize = 30)
ax.yaxis.get_offset_text().set_fontsize(27)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.set_xlim([3.35, 1.90])
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
secax = ax.secondary_xaxis('top', functions=(ev2nm, nm2ev))
secax.tick_params(labelsize = 30)
secax.set_xlabel('Wavelength (nm)', fontsize = 30)
ax.axhline(y=0, color = 'dimgrey', ls = '--')
dset = dmso.copy()
labels=[0,0,'DMSO, OD = 0.06','DMSO, OD = 0.08','DMSO, OD = 0.12']
ax.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0.1, 0.4, 3)])
i = 4
while i > 1:
    ax.plot(nm2ev(dset[dset.columns[0]]), jac(dset[dset.columns[0]], dset[dset.columns[i]]-dset[dset.columns[1]]), linewidth = 3, label = labels[i])
    i -= 1
dset = tft.copy()
labels=[0,0,'TFT, OD = 0.06','TFT, OD = 0.08','TFT, OD = 0.12']
ax.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0.9, 0.6, 3)])
i = 4
while i > 1:
    ax.plot(nm2ev(dset[dset.columns[0]]), jac(dset[dset.columns[0]], dset[dset.columns[i]]-dset[dset.columns[1]]), linewidth = 3, label = labels[i])
    i -= 1
ax.legend(loc = 'upper right', fontsize = 24)
        
#%%
fig, ax = plt.subplots(figsize=(11,9))
# ax.set_xlabel('Energy $(cm^{-1})$', fontsize = 20)
ax.set_xlabel('Energy (eV)', fontsize = 24)
ax.set_ylabel("Normalized Emission (A.U.)", fontsize = 24)
ax.tick_params(axis='x', labelsize= 20)
ax.tick_params(axis='y', labelsize= 20)
ax.set_xlim([3.5, 1.80])
# ax.set_ylim([-0.1*10**8, 2*10**8])
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
secax = ax.secondary_xaxis('top', functions=(ev2nm, nm2ev))
secax.tick_params(labelsize = 20)
secax.set_xlabel('Wavelength (nm)', fontsize = 24, labelpad = 10)
# ax.set_facecolor('gray')
ax.set_prop_cycle('color', ['goldenrod', 'darkorange'])
for dset in [linker, isonu1k]:
    ax.plot(nm2ev(dset['wl']), normalize_1(jac(dset['wl'], dset['S'])), linewidth = 3)
    # ax.plot(nm2ev(dset['wl']), jac(dset['wl'], dset['S']), color = 'silver', linewidth = 5, linestyle = '--', label = 'NU-601, DMSO')
    # ax.plot(nm2ev(dset['wl']), jac(dset['wl'], dset['S']), color = 'firebrick', linewidth = 5, linestyle = '-.', label = 'NU-601, 1mM H$_2$SO$_4$')
# ax.annotate('x10000', (2.25, 0.3*10**7), color = 'crimson', fontsize = 30)
# ax.annotate('x40', (3.00, 0.75*10**7), color = 'ivory', fontsize = 30)
# ax.plot(nm2ev(nu6k['wl']), sgf(normalize_1(jac(nu6k['wl'], nu6k['nu601']-nu6k['bg350'])),9,2), color = 'purple', linewidth = 3, label = 'NU-601')
ax.legend(labels = ['Linker', 'NU-601 (mixed)'], loc = 'upper right', fontsize = 24)
ax.axhline(y=0, color = 'dimgrey', ls = '--')

#%%
fig, ax = plt.subplots(1,3,figsize = (18,6), sharey = True)
plt.setp(ax[1].get_yticklabels(), visible=False)
plt.setp(ax[2].get_yticklabels(), visible=False)
ax[0].set_ylabel('Intensity (A.U.)', fontsize = 32)
ax[0].tick_params(axis = 'y', labelsize = 24)
ax[0].tick_params(axis = 'x', labelsize = 24)
ax[1].tick_params(axis = 'x', labelsize = 24)
ax[2].tick_params(axis = 'x', labelsize = 24)
ax[0].set_ylim([-0.05, 1.5])
ax[0].set_xlim([360, 600])
ax[1].set_xlim([360, 600])
ax[2].set_xlim([360, 600])
# plt.subplots_adjust(wspace=0)
big_ax = fig.add_subplot(111)
big_ax.set_xticks([])
big_ax.set_facecolor('none')
big_ax.set_xlabel('Wavelength (nm)', fontsize = 32, labelpad = 35)
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

ax[0].set_prop_cycle('color', ['black', 'red'])
ax[0].plot(data['d1_hbapy_dmf.txt'][:,0], normalize_1(data['d1_hbapy_dmf.txt'][:,1]), linewidth = 3.5, linestyle = '--', label = 'HBAPy (DMF)')
ax[0].plot(data['et1_hbapy_etoac.txt'][:,0], normalize_1(data['et1_hbapy_etoac.txt'][:,1]), linewidth = 3.5, linestyle = '-', label = 'HBAPy (EtOAc)')
ax[0].legend(loc = 'upper right', fontsize = 20)

ax[1].set_prop_cycle('color', ['black', 'red'])
ax[1].plot(data['d4_et4tbapy_dmf.txt'][:,0], normalize_1(data['d4_et4tbapy_dmf.txt'][:,1]), linewidth = 3.5, linestyle = '--', label = 'Et$_4$TBAPy (DMF)')
ax[1].plot(data['e4_et4tbapy_etoac.txt'][:,0], normalize_1(data['e4_et4tbapy_etoac.txt'][:,1]), linewidth = 3.5, linestyle = '-', label = 'Et$_4$TBAPy (EtOAc)')
ax[1].legend(loc = 'upper right', fontsize = 20)

ax[2].set_prop_cycle('color', ['black', 'red'])
ax[2].plot(data['d6_isoh4tbapy_dmf.txt'][:,0], normalize_1(data['d6_isoh4tbapy_dmf.txt'][:,1]), linewidth = 3.5, linestyle = '--', label = 'iso-H$_4$TBAPy (DMF)')
ax[2].plot(data['et6_isoh4tbapy_etoac.txt'][:,0], normalize_1(data['et6_isoh4tbapy_etoac.txt'][:,1]), linewidth = 3.5, linestyle = '-', label = 'iso-H$_4$TBAPy (EtOAc)')
ax[2].legend(loc = 'upper right', fontsize = 20)

#%% 

def ReadData(filename, bg = False):
    with open(filename) as current_file:
        dataset = np.loadtxt(current_file, delimiter = ',', dtype = 'str')
        if bg:
            dataset[0,0:2] = ['0', '0']
        else:
            dataset[0,0] = '0'
        dataset = dataset.astype(float)
    return dataset 

def QY(datasets, standard, ref_QY = 1, rindex = [1.3, 1.3], labels = 'placeholder'):
    """

    Parameters
    ----------
    datasets : array-like of array-like
        datasets to compute the quantum yield for.
        datasets should have the 0,0 slot empty or useless.
        1:,0 should be wavelength.
        0,1: should be sample OD vector.
        the rest of the matrix should be intensity data.
    standard : array-like
        datasets to use as a standard
        dataset should have the 0,0 slot empty or useless.
        1:,0 should be wavelength.
        0,1: should be sample OD vector.
        the rest of the matrix should be intensity data.
    rindex : a vector of size 2 
        refractive indices to use for samples and standard, respectively. 
        The default is [1.3, 1.3].

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize = (12,8))
    ax.set_xlabel('OD (AU)', fontsize = 24)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_minor_locator(MultipleLocator(0.0025))
    ax.set_ylabel('Integrated Fluorescence', fontsize = 24)
    ax.yaxis.get_offset_text().set_fontsize(22)
    ax.tick_params(axis='x', labelsize= 20)
    ax.tick_params(axis='y', labelsize= 20)
    result_dict = {}
    areas = []
    od = standard[0,1:]
    for i in range(len(od)):
        areas.append(np.trapz(jac(standard[1:,0], standard[1:,i+1]), x = nm2ev(standard[1:,0])[::-1]))
    ax.scatter(od, areas, s = 75, facecolor = 'none', edgecolor = 'red', marker = 'd', linewidth = 3, label = labels)
    (st_m,st_b) = np.polyfit(od, areas, 1)
    x = np.linspace(0,0.1,1000)
    ax.plot(x, st_b + st_m*x, linestyle = '--', color = 'black', linewidth = 2.5, label='_nolegend_')
    colorList = np.linspace(0.05, 0.85, len(datasets))
    cmap = matplotlib.cm.get_cmap('inferno')
    for d in range(len(datasets)): 
        dataset = datasets[d]
        areas = []
        od = dataset[0,1:]
        c = colorList[d]
        for i in range(len(od)):
            areas.append(np.trapz(jac(dataset[1:,0], dataset[1:,i+1]), x = nm2ev(dataset[1:,0])[::-1]))
        ax.scatter(od, areas, s = 75, facecolor = 'none', edgecolor = cmap(c), linewidth = 3, label = labels)
        (m,b) = np.polyfit(od, areas, 1)
        x = np.linspace(0,0.1,1000)
        ax.plot(x, b + m*x, linestyle = '--', color = 'black', linewidth = 2.5, label='_nolegend_')
        result_dict[f'{labels[d+1]} QY'] = ref_QY * (m/st_m) * (rindex[0]**2/rindex[1]**2)
    ax.legend(labels, fontsize = 24)
    return result_dict


    
    
        
    
    