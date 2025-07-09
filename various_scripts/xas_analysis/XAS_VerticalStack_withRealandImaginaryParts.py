# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:54:46 2024

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
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter as sgf
from scipy import special
import os
from collections import OrderedDict
#%% DATA IMPORT tailored for demeter suite

def normalize_1(data):
    """Normalizes a vector to it's maximum value"""
    normalized_data = data/max(data)
    return normalized_data
def ReadSingleExafs_Real(file, skip=3):
    """
    Reads in a file and assigns it to a variable. 
    Will assign column names as 'energy' and 'int'.
    Skips a given number of lines.
    Input is a comma-separated data file. 
    Output is a Pandas Dataframe.
    """
    with open(file) as current_file:
        d = pd.read_csv(current_file, sep = '\s+', header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
def ReadSingleExafs_R(file, skip=3):
    """
    Reads in a file and assigns it to a variable. 
    Will assign column names as 'energy' and 'int'.
    Skips a given number of lines.
    Input is a comma-separated data file. 
    Output is a Pandas Dataframe.
    """
    with open(file) as current_file:
        d = pd.read_csv(current_file, sep = '\s+', header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
def ReadSingleExafs_k(file, skip=3):
    """
    Reads in a file and assigns it to a variable. 
    Will assign column names as 'energy' and 'int'.
    Skips a given number of lines.
    Input is a comma-separated data file. 
    Output is a Pandas Dataframe.
    """
    with open(file) as current_file:
        d = pd.read_csv(current_file, sep = '\s+', header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
def ReadMultiExafs(skip):
    """
    Reads all data in the current directory (only .dat files)
    Skips a given number of rows in each file.
    Output is a dict of Dataframes.
    """
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith("k.dat"):
            data[file] = ReadSingleExafs_k(file)
        if file.endswith("R.dat"):
            data[file] = ReadSingleExafs_R(file)
    return data

def ReadSingleXANESxmu(file, skip=40):
    with open(file) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, header = 0, names = ['E', 'xmu', 'bkg', 'pre_edge', 'post_edge', 'deriv', 'deriv_2', 'i0', 'chie'], engine = 'python', skiprows = lambda x: x in range(skip))
        return d
    
def ReadMultiXANESnor(skip):
    """
    Reads all data in the current directory (only .dat files)
    Skips a given number of rows in each file.
    Output is a dict of Dataframes.
    """
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".nor"):
            data[file] = ReadSingleXANESnor(file)
    return data

def ReadSingleXANESnor(file, skip=11):
    with open(file) as current_file:
        # names = ['E', 'norm', 'nbkg', 'flat', 'fbkg', 'nderiv', 'nderiv_2'], 
        d = pd.read_csv(current_file, delim_whitespace = True, header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
    
def ReadSingleEXAFSchir(file, skip = 38):
    with open(file) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, header = 0, names = ['R', 'real', 'imag', 'magn', 'pha', 'win', 'deriv_pha'], engine = 'python', skiprows = lambda x: x in range(skip))
        return d
    
    
def ReadMultiEXAFSchik(skip=38):
    """
    Reads all data in the current directory (only .chik files)
    Skips a given number of rows in each file.
    Output is a dict of Dataframes.
    """
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".chik"):
            data[file] = ReadSingleEXAFSchik(file)
    return data
    
def ReadSingleEXAFSchik(file, skip = 38):
    with open(file) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, header = 0, names = ['k', 'chi', 'chik', 'chik2', 'chik3', 'win', 'energy'], engine = 'python', skiprows = lambda x: x in range(skip))
        return d
data_im = {}
data_re = {}
data_rm = {}
data_ks = {}

for file in os.listdir():
    if file.endswith(".dat"):
        with open(file) as current_file:
            if file.endswith("imag.dat"):
                data_im[file]=ReadSingleExafs_Real(file)
            if file.endswith("real.dat"):
                data_re[file]=ReadSingleExafs_Real(file)
            if file.endswith("rmag.dat"):
                data_rm[file]=ReadSingleExafs_R(file)
            if file.endswith("kspa.dat"):
                data_ks[file]=ReadSingleExafs_k(file)
                
#%%
datadict_callers = ['global','global','global','global']
dataset_callers = ['Th01_pH4_NH3','Th01_pH6_NH3','Th01_pH8_NH3','Th01_pH10_NH3']
datalabels = ['ThO$_2$, pH = 4','ThO$_2$, pH = 6','ThO$_2$, pH = 8','ThO$_2$, pH = 10']
fitcolors = ['lightcoral', 'deepskyblue', 'royalblue', 'indigo']

fig, ax = plt.subplots(4, 1, figsize = (12,12), sharex=True)
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'darkgray'
realcolor = 'blue'
imagcolor = 'orange'
i=0

for dictcall, datacall in zip(datadict_callers, dataset_callers):
    r = data_re[dictcall+"_real.dat"]['r']
    k = data_ks[dictcall+'_kspa.dat']['k']
    dset_re = data_re[dictcall+'_real.dat'][datacall]
    fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
    dset_im = data_im[dictcall+'_imag.dat'][datacall]
    fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
    dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
    fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
    dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
    fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    # i = datadict_callers.index(dictcall)
    
    print(i)
    ax[i].axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
    ax[i].plot(r, dset_rm, color = rmagcolor, linewidth = 6, label = datalabels[i])
    # ax[i].plot(r, fit_rm, color = fitcolors[i], linewidth = 4.5, label = 'Fit')
    ax[i].plot(r, dset_re, color = fitcolors[i], linewidth = 4.5)
    ax[i].plot(r, dset_im, color = fitcolors[i], linestyle = '--', linewidth = 3)
    ax[i].legend(loc = 'upper right', fontsize = fs2)
    if i in [0,1]:
        ax[i].spines[['bottom']].set_visible(False)
    ax[i].tick_params(axis='both', labelsize= fs2)
    ax[i].yaxis.set_major_locator(MultipleLocator(5))
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    i+=1
    
ax[-1].set_xlabel("R($\AA$)", fontsize = fs1) 
ax[-1].set_xlim([0.5,5])
ax[1].set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
plt.subplots_adjust(hspace=0)

#%%
datadict_callers = ['puo2fit','puo2fit','puo2fit']
dataset_callers = ["Fit_to_PuOxide_full_from_Fit_123", "Fit_to_PuOxide_full_from_Fit_121", "Fit_to_PuOxide_full_from_Fit_122"]
datalabels = ['F1: No Cumulant', 'F2: Pu - O Cumulant', 'F3: Pu - Pu Cumulant']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(3, 1, figsize = (12,12), sharex=True)
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'darkgray'
realcolor = 'blue'
imagcolor = 'orange'
i = 0

for dictcall, datacall in zip(datadict_callers, dataset_callers):
    r = data_re[dictcall+"_real.dat"]['r']
    k = data_ks[dictcall+'_kspa.dat']['k']
    dset_re = data_re[dictcall+'_real.dat']['PuOxide_full']
    fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
    dset_im = data_im[dictcall+'_imag.dat']['PuOxide_full']
    fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
    dset_rm = data_rm[dictcall+'_rmag.dat']['PuOxide_full']
    fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
    dset_ks = data_ks[dictcall+'_kspa.dat']['PuOxide_full']
    fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    # i = datadict_callers.index(dictcall)
    
    print(i)
    ax[i].axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
    ax[i].plot(r, dset_rm, color = rmagcolor, linewidth = 6, label = datalabels[i])
    ax[i].plot(r, fit_rm, color = fitcolors[i], linewidth = 4.5)
    ax[i].plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3)
    ax[i].plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3)
    ax[i].legend(loc = 'upper right', fontsize = fs2)
    if i in [0,1]:
        ax[i].spines[['bottom']].set_visible(False)
    ax[i].tick_params(axis='both', labelsize= fs2)
    ax[i].yaxis.set_major_locator(MultipleLocator(5))
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    i+=1
    
    
ax[-1].set_xlabel("R($\AA$)", fontsize = fs1) 
ax[-1].set_xlim([0.5,5])
ax[1].set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
plt.subplots_adjust(hspace=0)

#%%
datadict_callers = ['thphos5c']
dataset_callers = ["ThPhos_ph48_v2_wrk_(MEE)"]
datalabels = ['ThPhos-5-1000']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(figsize = (12,12))
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'dimgrey'
realcolor = 'blue'
imagcolor = 'orange'


dictcall = datadict_callers[0]
datacall = dataset_callers[0]
r = data_re[dictcall+"_real.dat"]['r']
k = data_ks[dictcall+'_kspa.dat']['k']
dset_re = data_re[dictcall+'_real.dat'][datacall]
fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
dset_im = data_im[dictcall+'_imag.dat'][datacall]
fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    

ax.axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
ax.plot(r, dset_rm, color = rmagcolor, linewidth = 6, label = datalabels[0])
ax.plot(r, fit_rm, color = fitcolors[0], linewidth = 4.5, label = 'Fit to NaTh$_2$(PO$_4$)$_3$')
ax.plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
ax.plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3,label = 'FT Img. Part')
ax.legend(loc = 'upper right', fontsize = fs2)
ax.tick_params(axis='both', labelsize= fs2)
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    
    
ax.set_xlabel("R ($\AA$)", fontsize = fs1) 
ax.set_xlim([0.5,5])
ax.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
# plt.subplots_adjust(hspace=0)
plt.show()
#%%
datadict_callers = ['thphos5c']
dataset_callers = ["ThPhos_ph48_v2_wrk_(MEE)"]
datalabels = ['ThPhos-5-1000']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(figsize = (18,6))
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'dimgrey'
realcolor = 'blue'
imagcolor = 'orange'


dictcall = datadict_callers[0]
datacall = dataset_callers[0]
r = data_re[dictcall+"_real.dat"]['r']
k = data_ks[dictcall+'_kspa.dat']['k']
dset_re = data_re[dictcall+'_real.dat'][datacall]
fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
dset_im = data_im[dictcall+'_imag.dat'][datacall]
fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    

ax.axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
ax.plot(k, dset_ks, color = rmagcolor, linewidth = 6, label = datalabels[0])
ax.plot(k, fit_ks, color = fitcolors[0], linewidth = 4.5, label = 'Fit to NaTh$_2$(PO$_4$)$_3$')
# ax.plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
# ax.plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3,label = 'FT Img. Part')

ax.tick_params(axis='both', labelsize= fs2)
ax.yaxis.set_major_locator(MultipleLocator(2.5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.axvline(x=3, color = 'darkseagreen', linewidth = 3,ls = '--')
ax.axvline(x=11, color = 'darkseagreen', linewidth = 3, ls = '--', label = "Window")
ax.legend(loc = 'upper left', fontsize = fs2)
ax.set_xlabel("k($\AA^{-1}$)", fontsize = fs1) 
ax.set_xlim([-0.2,12.2])
ax.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fs1)
# plt.subplots_adjust(hspace=0)
plt.show()

#%%
datadict_callers = ['thphos8c']
dataset_callers = ["ThPhos_ph75_v2_wrk_(MEE)"]
datalabels = ['ThPhos-8-1000']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(figsize = (12,12))
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'dimgrey'
realcolor = 'blue'
imagcolor = 'orange'


dictcall = datadict_callers[0]
datacall = dataset_callers[0]
r = data_re[dictcall+"_real.dat"]['r']
k = data_ks[dictcall+'_kspa.dat']['k']
dset_re = data_re[dictcall+'_real.dat'][datacall]
fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
dset_im = data_im[dictcall+'_imag.dat'][datacall]
fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    

ax.axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
ax.plot(r, dset_rm, color = rmagcolor, linewidth = 6, label = datalabels[0])
ax.plot(r, fit_rm, color = fitcolors[0], linewidth = 4.5, label = 'Fit to Na$_2$Th(PO$_4$)$_2$')
ax.plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
ax.plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3,label = 'FT Img. Part')
ax.legend(loc = 'upper right', fontsize = fs2)
ax.tick_params(axis='both', labelsize= fs2)
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    
    
ax.set_xlabel("R ($\AA$)", fontsize = fs1) 
ax.set_xlim([0.5,5])
ax.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
# plt.subplots_adjust(hspace=0)
plt.show()
#%%
datadict_callers = ['thphos8c']
dataset_callers = ["ThPhos_ph75_v2_wrk_(MEE)"]
datalabels = ['ThPhos-8-1000']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(figsize = (18,6))
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'dimgrey'
realcolor = 'blue'
imagcolor = 'orange'


dictcall = datadict_callers[0]
datacall = dataset_callers[0]
r = data_re[dictcall+"_real.dat"]['r']
k = data_ks[dictcall+'_kspa.dat']['k']
dset_re = data_re[dictcall+'_real.dat'][datacall]
fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
dset_im = data_im[dictcall+'_imag.dat'][datacall]
fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    

ax.axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
ax.plot(k, dset_ks, color = rmagcolor, linewidth = 6, label = datalabels[0])
ax.plot(k, fit_ks, color = fitcolors[0], linewidth = 4.5, label = 'Fit to Na$_2$Th(PO$_4$)$_2$')
# ax.plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
# ax.plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3,label = 'FT Img. Part')

ax.tick_params(axis='both', labelsize= fs2)
ax.yaxis.set_major_locator(MultipleLocator(2.5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.axvline(x=3, color = 'darkseagreen', linewidth = 3,ls = '--')
ax.axvline(x=11, color = 'darkseagreen', linewidth = 3, ls = '--', label = "Window")
ax.legend(loc = 'lower left', fontsize = fs2)
ax.set_xlabel("k($\AA^{-1}$)", fontsize = fs1) 
ax.set_xlim([-0.2,12.2])
ax.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fs1)
# plt.subplots_adjust(hspace=0)
plt.show()
#%% ThPhos-5
datadict_callers = ['thphos5']
dataset_callers = ["ThPhos_pH48"]
datalabels = ['ThPhos-5']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(figsize = (12,12))
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'dimgrey'
realcolor = 'blue'
imagcolor = 'orange'


dictcall = datadict_callers[0]
datacall = dataset_callers[0]
r = data_re[dictcall+"_real.dat"]['r']
k = data_ks[dictcall+'_kspa.dat']['k']
dset_re = data_re[dictcall+'_real.dat'][datacall]
fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
dset_im = data_im[dictcall+'_imag.dat'][datacall]
fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    

ax.axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
ax.plot(r, dset_rm, color = rmagcolor, linewidth = 6, label = datalabels[0])
ax.plot(r, fit_rm, color = fitcolors[0], linewidth = 4.5, label = 'Fit to NaTh$_2$(PO$_4$)$_3$')
ax.plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
ax.plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3,label = 'FT Img. Part')
ax.legend(loc = 'upper right', fontsize = fs2)
ax.tick_params(axis='both', labelsize= fs2)
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    
    
ax.set_xlabel("R ($\AA$)", fontsize = fs1) 
ax.set_xlim([0.5,5])
ax.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
# plt.subplots_adjust(hspace=0)
plt.show()
#%%
datadict_callers = ['thphos5']
dataset_callers = ["ThPhos_pH48"]
datalabels = ['ThPhos-5']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(figsize = (18,6))
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'dimgrey'
realcolor = 'blue'
imagcolor = 'orange'


dictcall = datadict_callers[0]
datacall = dataset_callers[0]
r = data_re[dictcall+"_real.dat"]['r']
k = data_ks[dictcall+'_kspa.dat']['k']
dset_re = data_re[dictcall+'_real.dat'][datacall]
fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
dset_im = data_im[dictcall+'_imag.dat'][datacall]
fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    

ax.axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
ax.plot(k, dset_ks, color = rmagcolor, linewidth = 6, label = datalabels[0])
ax.plot(k, fit_ks, color = fitcolors[0], linewidth = 4.5, label = 'Fit to NaTh$_2$(PO$_4$)$_3$')
# ax.plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
# ax.plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3,label = 'FT Img. Part')

ax.tick_params(axis='both', labelsize= fs2)
ax.yaxis.set_major_locator(MultipleLocator(2.5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.axvline(x=3, color = 'darkseagreen', linewidth = 3,ls = '--')
ax.axvline(x=10.8, color = 'darkseagreen', linewidth = 3, ls = '--', label = "Window")
ax.legend(loc = 'upper left', fontsize = fs2)
ax.set_xlabel("k($\AA^{-1}$)", fontsize = fs1) 
ax.set_xlim([-0.2,12.2])
ax.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fs1)
# plt.subplots_adjust(hspace=0)
plt.show()

#%% ThPhos-8
datadict_callers = ['thphos8']
dataset_callers = ["ThPhos_pH75"]
datalabels = ['ThPhos-8']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(figsize = (12,12))
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'dimgrey'
realcolor = 'blue'
imagcolor = 'orange'


dictcall = datadict_callers[0]
datacall = dataset_callers[0]
r = data_re[dictcall+"_real.dat"]['r']
k = data_ks[dictcall+'_kspa.dat']['k']
dset_re = data_re[dictcall+'_real.dat'][datacall]
fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
dset_im = data_im[dictcall+'_imag.dat'][datacall]
fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    

ax.axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
ax.plot(r, dset_rm, color = rmagcolor, linewidth = 6, label = datalabels[0])
ax.plot(r, fit_rm, color = fitcolors[0], linewidth = 4.5, label = 'Fit to NaTh$_2$(PO$_4$)$_3$')
ax.plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
ax.plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3,label = 'FT Img. Part')
ax.legend(loc = 'upper right', fontsize = fs2)
ax.tick_params(axis='both', labelsize= fs2)
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    
    
ax.set_xlabel("R ($\AA$)", fontsize = fs1) 
ax.set_xlim([0.5,5])
ax.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
# plt.subplots_adjust(hspace=0)
plt.show()
#%%
datadict_callers = ['thphos5c']
dataset_callers = ["ThPhos_ph48_v2_wrk_(MEE)"]
datalabels = ['ThPhos-5-1000']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(figsize = (12,8))
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'dimgrey'
realcolor = 'blue'
imagcolor = 'orange'


dictcall = datadict_callers[0]
datacall = dataset_callers[0]
r = data_re[dictcall+"_real.dat"]['r']
k = data_ks[dictcall+'_kspa.dat']['k']
dset_re = data_re[dictcall+'_real.dat'][datacall]
fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
dset_im = data_im[dictcall+'_imag.dat'][datacall]
fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    

ax.axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
# ax.plot(k, dset_ks, color = rmagcolor, linewidth = 6, label = datalabels[0])
# ax.plot(k, fit_ks, color = fitcolors[0], linewidth = 4.5, label = 'Fit to NaTh$_2$(PO$_4$)$_3$')
# ax.plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
# ax.plot(r, fit_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
# ax.plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3,label = 'FT Img. Part')

ax.plot(r, dset_re, color = rmagcolor, linewidth = 6, label = datalabels[0])
ax.plot(r, fit_re, color = fitcolor, linewidth = 4.5, label = 'Fit')
# ax.plot(r, dset_re, color = rmagcolor, linewidth = 6, label = dataLabel0)
# ax.plot(r, fit_re, color = fitcolor, linewidth = 4.5, label = 'Fit')

ax.tick_params(axis='both', labelsize= fs2)
ax.yaxis.set_major_locator(MultipleLocator(2.5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.axvline(x=1., color = 'darkseagreen', linewidth = 3,ls = '--')
# ax.axvline(x=10.8, color = 'darkseagreen', linewidth = 3, ls = '--', label = "Window")
ax.legend(loc = 'best', fontsize = fs2)
ax.set_xlabel("R($\AA$)", fontsize = fs1) 
ax.set_xlim([0,5])
# ax.set_ylim([-7.9, 5.5])
ax.set_ylabel("Re[\u03C7(R)]($\AA^{-4}$)", fontsize = fs1)
# plt.subplots_adjust(hspace=0)
plt.show()
#%%
datadict_callers = ['thphos5c']
dataset_callers = ["ThPhos_ph48_v2_wrk_(MEE)"]
datalabels = ['ThPhos-5-1000']
fitcolors = ['red', 'blue', 'magenta']

fig, ax = plt.subplots(figsize = (12,8))
fs1 = 28
fs2 = 24
fitcolor = 'crimson'
rmagcolor = 'dimgrey'
realcolor = 'blue'
imagcolor = 'orange'


dictcall = datadict_callers[0]
datacall = dataset_callers[0]
r = data_re[dictcall+"_real.dat"]['r']
k = data_ks[dictcall+'_kspa.dat']['k']
dset_re = data_re[dictcall+'_real.dat'][datacall]
fit_re = data_re[dictcall+'_real.dat'][datacall+'_fit']
dset_im = data_im[dictcall+'_imag.dat'][datacall]
fit_im = data_im[dictcall+'_imag.dat'][datacall+'_fit']
dset_rm = data_rm[dictcall+'_rmag.dat'][datacall]
fit_rm = data_rm[dictcall+'_rmag.dat'][datacall+'_fit']
dset_ks = data_ks[dictcall+'_kspa.dat'][datacall]
fit_ks = data_ks[dictcall+'_kspa.dat'][datacall+'_fit']
    
    

ax.axhline(y = 0, color = 'grey', linestyle = '--', linewidth = 3.3)
# ax.plot(k, dset_ks, color = rmagcolor, linewidth = 6, label = datalabels[0])
# ax.plot(k, fit_ks, color = fitcolors[0], linewidth = 4.5, label = 'Fit to NaTh$_2$(PO$_4$)$_3$')
# ax.plot(r, dset_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
# ax.plot(r, fit_re, color = realcolor, linestyle = '--', linewidth = 3,label = 'FT Real Part')
# ax.plot(r, dset_im, color = imagcolor, linestyle = '--', linewidth = 3,label = 'FT Img. Part')

ax.plot(r, dset_im, color = rmagcolor, linewidth = 6, label = datalabels[0])
ax.plot(r, fit_im, color = fitcolor, linewidth = 4.5, label = 'Fit')
# ax.plot(r, dset_re, color = rmagcolor, linewidth = 6, label = dataLabel0)
# ax.plot(r, fit_re, color = fitcolor, linewidth = 4.5, label = 'Fit')

ax.tick_params(axis='both', labelsize= fs2)
ax.yaxis.set_major_locator(MultipleLocator(2.5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.axvline(x=1., color = 'darkseagreen', linewidth = 3,ls = '--')
# ax.axvline(x=10.8, color = 'darkseagreen', linewidth = 3, ls = '--', label = "Window")
ax.legend(loc = 'best', fontsize = fs2)
ax.set_xlabel("R($\AA$)", fontsize = fs1) 
ax.set_xlim([0,5])
# ax.set_ylim([-7.9, 5.5])
ax.set_ylabel("Im[\u03C7(R)]($\AA^{-4}$)", fontsize = fs1)
# plt.subplots_adjust(hspace=0)
plt.show()