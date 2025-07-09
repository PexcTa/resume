# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:49:46 2021

This is the plotting script written by Boris Kramar to plot data acquired on 
a diffuse reflectance instrument shimadzu UV 3600. It expects the input to be
any number of .txt files with two data columns separated by a comma.

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
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

#%% DATA INPUT
def ReadData(file, skip=2):
    """
    Reads in a file and assigns it to a variable. 
    Will assign column names as 'energy' and 'int'.
    Skips a given number of lines.
    Input is a comma-separated data file. 
    Output is a Pandas Dataframe.
    """
    # file = os.listdir()
    with open(file) as current_file:
        d = pd.read_csv(current_file, names = ['wl', 'R'], delim_whitespace = True, engine = 'python', skiprows = lambda x: x in range(skip))
        return d
def ReadAllData(skip=2):
    """
    Reads all data in the current directory (only .txt files)
    Skips a given number of rows in each file.
    Output is a dict of Dataframes.
    """
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".txt"):
            data[str(file)[:-4]] = ReadData(file,skip)
    return data
def RenameDictKeys(dictionary, new_names):
    """
    Renames dictionary keys to those given in new_names, keeps the order
    """
    res = {}
    for i in range(len(new_names)):
        res[new_names[i]] = dictionary[list(dictionary.keys())[i]]
    return res

# UVVIS_data = ReadAllData(2)
# Data = RenameDictKeys(UVVIS_data, ['pbz', 'pcn1', 'pcn3', 'pcn2'])

#%% CONVERSIONS

def nm2ev(data_in_nm):
    h = 4.135667516*10**(-15)
    c = 299792458
    return (h*c)/(data_in_nm*10**(-9))
def ev2nm(data_in_ev):
    h = 4.135667516*10**(-15)
    c = 299792458
    return (h*c)/(data_in_ev)/10**(-9)
def nm2WN(data_in_nm):
    return 10000000/data_in_nm
def WN2nm(data_in_WN):
    return 10000000/data_in_WN


#%% DATA PROCESSING
def normalize_1(data):
    """Normalizes a vector to its maximum value"""
    temp = np.zeros_like(data)
    for i in range(len(temp)):
        temp[i] = np.abs(data[i])
    normalized_data = temp/max(temp)
    return normalized_data
    
def KMT(data):
    transformed_data = ((100-data)**2)/2*data
    return transformed_data

def edgefit(data_x, data_y, low_limit, high_limit):
    subset = (data_x>low_limit) & (data_x<high_limit)
    xFit = data_x[subset]
    yFit = data_y[subset]
    linfit = np.polyfit(data_x, data_y, 1)
    return xFit*linfit[0]+linfit[1]

def derivative(data):
    dRdwl = []
    dwl = data['wl'][1] - data['wl'][0]
    for i in range(len(data['wl'])):
        dRdwl = np.gradient(data['R'], dwl)
    return dRdwl
    

#%%
AllDataDR = ReadAllData()
refl = {}
kubm = {}
derv = {}
for file in AllDataDR.keys():
    if 'KM' in str(file):
        kubm[file] = AllDataDR[file]
    elif 'd1' in str(file):
        derv[file] = AllDataDR[file]
    else:
        refl[file] = AllDataDR[file]
del file
# names = ['100um', '100mm',  '10mm','1m',  '1mm','250mm', '500mm',   'mph2o']
# refl = RenameDictKeys(refl, names)
# derv = RenameDictKeys(derv, names)
# kubm = RenameDictKeys(kubm, names)
#%% Plot Raw Data    


fig, ax = plt.subplots(figsize=(12,9))
# ax2 = ax.twinx()
# ax2.set_yticks([], [])
# ax.set_xlabel("Wavelength, nm", fontsize = 27)
ax.set_xlabel("Energy (eV)", fontsize = 24)
# ax.set_ylabel("Normalized Kubelka-Munk", fontsize = 27)
# ax.set_ylabel("Normalized dR/d$\lambda$", fontsize = 27)
ax.set_ylabel("Normalized Absorption (A.U.)", fontsize = 27)
# ax.set_ylabel("Normalized Emission (A.U.)", fontsize = 24)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
# ax.set_xlim([3.6,2.1])
# ax.set_ylim([0.2, 1.05])

# ax.set_xlim([3.1,2.8])
# ax.set_ylim([0.9, 1.05])
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

secax = ax.secondary_xaxis('top', functions=(ev2nm, nm2ev))
secax.tick_params(labelsize = 20)
secax.set_xlabel('Wavelength, nm', fontsize = 27, labelpad = 10)

names = list(refl.keys())[1:]



ax.set_prop_cycle('color', [plt.cm.inferno(i) for i in np.linspace(0, 1, 10)])
for name in names:
    ax.plot(nm2ev(refl[name]['wl']), sgf(normalize_1(refl[name]['R']-refl['blank_mecn']['R']),27,3), linewidth = 3.5)
# ax.set_prop_cycle('color', [plt.cm.viridis(i) for i in np.linspace(0.4, 0.8, 2)])
# for name in ['anatase_nano', 'rutile_nano']:
#     ax.plot(nm2ev(refl[name]['wl']), savgol_filter(normalize_1(refl[name]['R']),9,3), linewidth = 3.5)
# ax.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0.2, 0.8, 2)])
# for name in ['mr', 'hbapy']:
#     ax.plot(nm2ev(refl[name]['wl']), normalize_1(refl[name]['R']), linestyle = '--', linewidth = 3.5)

# ax.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0.1, 0.8, 3)])
# for name in ['tio2_ito', 'tio2_sio2', 'tio2_zro2']:
#     ax.plot(nm2ev(derv[name]['wl']), normalize_1(derv[name]['R']), linewidth = 3.5)
# ax.set_prop_cycle('color', [plt.cm.Greys(i) for i in np.linspace(0.5, 0.8, 2)])
# for name in ['anatase', 'rutile']:
#     ax.plot(nm2ev(derv[name]['wl']), normalize_1(derv[name]['R']), linestyle = '--', linewidth = 3.5)

ax.legend(labels = names, loc = 'upper right', fontsize=24)
# ax.axhline(y=0, color = 'dimgrey', ls = '--')
# plt.savefig('./img/ito_sio2_vs_standards.png')
#%% fancy multipanel 

d1 = '032223_bk241_1_NU901F'
d2 = '032223_bk241_2_NU901FF'

# one column
fig = plt.figure(figsize = (8,8))
gs = fig.add_gridspec(4, 1)
ax1 = fig.add_subplot(gs[:3, 0])
ax3 = fig.add_subplot(gs[3, 0], sharex = ax1)
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax3.get_xticklabels(), visible=False)

# two columns
# fig = plt.figure(figsize = (12,8))
# gs = fig.add_gridspec(4, 2)
# ax1 = fig.add_subplot(gs[:3, 0])
# ax2 = fig.add_subplot(gs[:3, 1], sharey = ax1)
# ax3 = fig.add_subplot(gs[3, 0], sharex = ax1)
# ax4 = fig.add_subplot(gs[3, 1], sharex = ax2, sharey = ax3)
# plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax2.get_xticklabels(), visible=False)

ax1.set_xlim([5.0, 1.5])
ax1.set_ylabel('Kubelka-Munk', fontsize = 32)
# ax1.plot(nm2ev(refl[d1+'.txt']['wl']), normalize_1(refl['nano_anatase.txt']['R']), linewidth = 4, linestyle = '-.', color = 'crimson', label = 'Anatase')
# ax1.plot(nm2ev(refl['nano_rutile.txt']['wl']), normalize_1(refl['nano_rutile.txt']['R']), linewidth = 4, linestyle = '--', color = 'limegreen', label = 'Rutile')
ax1.set_prop_cycle('color', [plt.cm.magma(i) for i in np.linspace(0.1, 0.8, 2)])
ax1.plot(nm2ev(refl[d1+'.txt']['wl']), sgf(normalize_1(refl[d1+'.txt']['R']),21,3), linewidth = 4,  label = 'NU-901-Formate')
ax1.plot(nm2ev(refl[d2+'.txt']['wl']), sgf(normalize_1(refl[d2+'.txt']['R']),21,3), linewidth = 4, label = 'NU-901-FF')
ax1.legend(fontsize = 20, loc = 'lower left')
ax1.axhline(y = 0, linestyle = '-', color = 'grey', alpha = 0.5)
secax = ax1.secondary_xaxis('top', functions=(ev2nm, nm2ev))
secax.tick_params(labelsize = 20)
secax.set_xlabel('Wavelength, nm', fontsize = 27, labelpad = 10)

# ax2.set_xlim([5.0, 1.5])
# ax2.set_yticklabels([])
# ax2.plot(nm2ev(refl['nano_anatase.txt']['wl']), normalize_1(refl['nano_anatase.txt']['R']), linewidth = 4, linestyle = '-.', color = 'crimson', label = 'Anatase')
# ax2.plot(nm2ev(refl['nano_rutile.txt']['wl']), normalize_1(refl['nano_rutile.txt']['R']), linewidth = 4, linestyle = '--', color = 'limegreen', label = 'Rutile')
# ax2.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0.1, 0.8, 2)])
# ax2.plot(nm2ev(refl['5nm_tio2_zro2_baselined.txt']['wl']), normalize_1(refl['5nm_tio2_zro2_baselined.txt']['R']), linewidth = 4,  label = '4 nm TiO$_2$@ZrO$_2$')
# ax2.plot(nm2ev(refl['1nm_tio2_zro2.txt']['wl']), normalize_1(refl['1nm_tio2_zro2.txt']['R']), linewidth = 4, label = '1 nm TiO$_2$@ZrO$_2$')
# ax2.legend(fontsize = 20, loc = 'upper right')
# ax2.axhline(y = 0, linestyle = '-', color = 'grey', alpha = 0.5)

ax3.set_xlim([5.0, 1.5])
ax3.set_ylabel('dA/d$\lambda$', fontsize = 32)
ax3.set_yticklabels([])
ax3.tick_params(axis = 'x', labelsize = 24)
ax3.set_xlabel('Energy (eV)', fontsize = 32)
# ax3.plot(nm2ev(derv['nano_anatase_d1.txt']['wl']), norma1lize_1(derv['nano_anatase_d1.txt']['R']), linewidth = 4, linestyle = '-.', color = 'crimson', label = 'Anatase')
# ax3.plot(nm2ev(derv['nano_rutile_d1.txt']['wl']), normalize_1(derv['nano_rutile_d1.txt']['R']), linewidth = 4, linestyle = '--', color = 'limegreen', label = 'Rutile')
ax3.set_prop_cycle('color', [plt.cm.magma(i) for i in np.linspace(0.1, 0.8, 2)])
ax3.plot(nm2ev(derv[d1+'_d1.txt']['wl']), normalize_1(derv[d1+'_d1.txt']['R']), linewidth = 4,  label = 'NU-901-Formate')
ax3.plot(nm2ev(derv[d2+'_d1.txt']['wl']), normalize_1(derv[d2+'_d1.txt']['R']), linewidth = 4, label = 'NU-901-FF')
ax3.axhline(y = 0, linestyle = '-', color = 'grey', alpha = 0.5)

# ax4.tick_params(axis = 'x', labelsize = '24')
# ax4.set_xlabel('Energy (eV)', fontsize = 32)
# ax4.plot(nm2ev(derv['nano_anatase_d1.txt']['wl']), normalize_1(derv['nano_anatase_d1.txt']['R']), linewidth = 4, linestyle = '-.', color = 'crimson', label = 'Anatase')
# ax4.plot(nm2ev(derv['nano_rutile_d1.txt']['wl']), normalize_1(derv['nano_rutile_d1.txt']['R']), linewidth = 4, linestyle = '--', color = 'limegreen', label = 'Rutile')
# ax4.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0.1, 0.8, 2)])
# ax4.plot(nm2ev(derv['5nm_tio2_zro2_baselined_d1.txt']['wl']), sgf(normalize_1(derv['5nm_tio2_zro2_baselined_d1.txt']['R']),29,3), linewidth = 4,  label = '4 nm TiO$_2$ on ZrO$_2$')
# ax4.plot(nm2ev(derv['1nm_tio2_zro2_d1.txt']['wl']), sgf(normalize_1(derv['1nm_tio2_zro2_d1.txt']['R']),29,3), linewidth = 4, label = '1 nm TiO$_2$ on ZrO$_2$')
# ax4.axhline(y = 0, linestyle = '-', color = 'grey', alpha = 0.5)

plt.tight_layout()
#%% Plot Normalized Data
fig, ax = plt.subplots(figsize=(10,8))
# ax.set_xlabel("Energy, $cm^{-1}$", fontsize = 20)
ax.set_xlabel('Energy $(cm^{-1})$', fontsize = 20)
ax.set_ylabel("Emission (counts)", fontsize = 20)
ax.tick_params(axis='x', labelsize= 15)
ax.tick_params(axis='y', labelsize= 15)
ax.set_xlim([25000,16000])
ax.xaxis.set_major_locator(MultipleLocator(1000))
ax.xaxis.set_minor_locator(MultipleLocator(200))
secax = ax.secondary_xaxis('top', functions=(WN2nm, nm2WN))
secax.tick_params(labelsize = 15)
secax.set_xlabel('Wavelength (nm)', fontsize = 20)
ax.axhline(y=0, color = 'dimgrey', ls = '--')

# dset = Data['NU1k.txt']
dset = NU1k
x = nm2WN(dset['energy'])
y = normalize_1(dset['int'])
# y -= y[500]
ax.plot(x, savgol_filter(y,9,3), color = 'goldenrod', linewidth = 3, label = 'NU-1000 (MeTHF)')
# dset = Co1c
# x = nm2WN(dset['wl'])
# y = dset['d1'] - dset['bg']
# # y -= y[500]
# ax.plot(x, savgol_filter(y,9,3), color = 'darkmagenta', linewidth = 3, label = 'Co-SIM (DMF)')
# dset = Ti1c
# x = nm2WN(dset['wl'])
# y = dset['d1'] - dset['bg']
# # y -= y[500]
# ax.plot(x, savgol_filter(y,9,3), color = 'slategrey', linewidth = 3, label = 'Ti-SIM (DMF)')
# dset = Ni1c
# x = nm2WN(dset['wl'])
# y = dset['d1'] - dset['bg']
# # y -= y[500]
# ax.plot(x, savgol_filter(y,9,3), color = 'teal', linewidth = 3, label = 'Ni-SIM (DMF)')

# dset = Data['BK23.txt']
# dset = Linker
# x = nm2WN(dset['wl'])
# y = dset['int']
# y -= y[500]
# ax.plot(x, savgol_filter(normalize_1(y),9,3), color = 'darkorange', linewidth = 3, linestyle = '--', label = 'NU-1000 in DMF')

ax.scatter(nm2WN(Linker['wl']), normalize_1(Linker['int']), s = 40, facecolors ='none',  edgecolors = 'crimson', marker = 'o', label = 'Linker (MeTHF)')
ax.legend(loc = 'upper right', fontsize = 18)
# plt.savefig('CoSIM_NU1000_linker_nm.svg')


#%% fit the derivative to a sum of gaussian peaks
import lmfit as lm
def MultipeakDerivativeFitting(data, peak_no, x0guess, theta, A_value = 1, fwhm_value =0.5, x_scale = 'eV', normalize = False, plotfit = True):
    def three_regions(energy, M, offset, **Pars):
        x = energy
        T_1 = np.zeros_like(energy)
        T_2 = np.zeros_like(energy)
        T_3 = np.zeros_like(energy)
        for m in range(1, M+1):
            if m == 1:
                A = Pars[f'A_{m}']
                x0 = Pars[f'x0_{m}']
                fwhm = Pars[f'fwhm_{m}']
                T_1 += A*np.exp(-0.5*((x - x0)/fwhm)**2) + offset
            elif m == 2:
                A = Pars[f'A_{m}']
                x0 = Pars[f'x0_{m}']
                fwhm = Pars[f'fwhm_{m}']
                T_2 += A*np.exp(-0.5*((x - x0)/fwhm)**2) + offset
            elif m == 3:
                A = Pars[f'A_{m}']
                x0 = Pars[f'x0_{m}']
                fwhm = Pars[f'fwhm_{m}']
                T_3 += A*np.exp(-0.5*((x - x0)/fwhm)**2) + offset
        return np.sum([T_1, T_2, T_3], axis = 0)
    def two_regions(energy, M, offset, **Pars):
        x = energy
        T_1 = np.zeros_like(x)
        T_2 = np.zeros_like(x)
        for m in range(1, M+1):
            if m == 1:
                A = Pars[f'A_{m}']
                x0 = Pars[f'x0_{m}']
                fwhm = Pars[f'fwhm_{m}']
                T_1 += A*np.exp(-0.5*((x - x0)/fwhm)**2) + offset
            elif m == 2:
                A = Pars[f'A_{m}']
                x0 = Pars[f'x0_{m}']
                fwhm = Pars[f'fwhm_{m}']
                T_2 += A*np.exp(-0.5*((x - x0)/fwhm)**2) + offset
        return np.sum([T_1, T_2], axis = 0)
    def one_region(energy, M, offset, **Pars):
        x = energy
        T_1 = np.zeros_like(energy)
        for m in range(1, M+1):
            if m == 1:
                A = Pars[f'A_{m}']
                x0 = Pars[f'x0_{m}']
                fwhm = Pars[f'fwhm_{m}']
                T_1 += A*np.exp(-0.5*((x - x0)/fwhm)**2) + offset
        return T_1
    M = peak_no
    print(f'There are {M} peaks being fit')
    if M == 1:
        model = lm.Model(one_region, independent_vars=['energy'])
    elif M == 2:
        model = lm.Model(two_regions, independent_vars=['energy'])
    elif M == 3:
        model = lm.Model(three_regions, independent_vars=['energy'])
    params = lm.Parameters()
    params.add('M', value=M, vary=False)
    for m in range(1, M+1):
        params.add(f'A_{m}', value=A_value, min = 0)
        params.add(f'x0_{m}', value=x0guess[m-1])
        params.add(f'fwhm_{m}', value=fwhm_value, min = 0)
    params.add('offset', value = theta[1])
    
    print('Model parameters constructed')
    sample = data[:,1]
    energy = data[:,0]
    print('Fitting...')
    result = model.fit(sample, energy = energy, params=params)
    print(result.fit_report())
    pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
    df_true = pd.DataFrame()
    if M == 1:
        results_full = result.eval(energy = energy)
        init_full = result.eval(energy=energy, params=params)
        df_true['trace1_fit'] = results_full
        df_true['trace1_init'] = init_full
    elif M == 2:
        results_full = result.eval(energy = energy)
        init_full = result.eval(energy=energy, params=params)
        df_true['trace1_fit'] = results_full
        df_true['trace1_init'] = init_full
    elif M == 3:
        results_full = result.eval(energy = energy)
        init_full = result.eval(energy=energy, params=params)
        df_true['trace1_fit'] = results_full
        df_true['trace1_init'] = init_full
    if plotfit:
        fig,ax = plt.subplots(figsize = [10,10])            
        ax.set_ylabel("Intensity (arb. units)", fontsize = 27)
        if normalize:
            ax.set_ylabel("Normalized Intensity", fontsize = 27)
        ax.set_yticklabels([])
        ax.set_xlim([max(energy), min(energy)])
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize= 24)
        ax.tick_params(axis='y', labelsize= 24)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel('energy, '+x_scale, fontsize = 27)
        ax.scatter(energy, sample, s = 60, facecolor = 'none', edgecolor = 'black', linewidth = 1)
        scatter_color_range = [plt.cm.viridis(i) for i in np.linspace(0.1, 0.9, M)]
        # for i in range(M):
        #     ax.plot(energy, df_true[f'trace{i+1}_fit'], color = scatter_color_range[i], linewidth = 3)
        ax.plot(energy, df_true[f'trace1_fit'], color = 'red', linewidth = 3)
    return pd_pars
        
test = MultipeakDerivativeFitting(jact_d1['rutile'], 2, x0guess = [3, 3], theta = (0.2, 0), A_value = 0.02, fwhm_value = 0.2, normalize = False, plotfit = True)

#%%
def LinearCombinationFitting(data, standards, x_scale = 'eV', normalize = True, plotfit = True):
    """
    
    Parameters
    ----------
    data : array-like
        dataset to fit to a sum of other datasets
    standards : list of array-like
        datasets to use for LCF
    normalize : boolean
        if true, normalize all to 1 (default True)
    plotfit : boolean
        if true, plot a simple figure. The default is True.

    Returns
    -------
    fitting parameters

    """
    energy = data[:,0]
    dset = normalize_1(np.array(data[:,1]))
    M = len(standards)
    normalized_standards = np.zeros((len(energy), M))
    for i in range(M):
        normalized_standards[:,i] = normalize_1(np.array(standards[i][:,1]))
    print(f'Chose {M} standards')
    def LCF(standards, M, **Pars):
        return np.sum([np.multiply(Pars[f'A_{m}'],standards[:,m-1]) for m in range(1,M+1)], axis = 0)
    # print(f'There are {M} peaks being fit')
    model = lm.Model(LCF)
    params = lm.Parameters()
    for m in range(1, M+1):
        params.add(f'A_{m}', value = 1/M, min = 0)
    params.add("M", value = M, vary = False)
    print('Model parameters constructed')
    print('Fitting...')
    result = model.fit(dset, standards = normalized_standards, M = M, params=params)
    print(result.fit_report())
    pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
    df_true = pd.DataFrame()
    results_full = result.eval(energy = energy)
    init_full = result.eval(energy=energy, params=params)
    df_true['trace_fit'] = results_full
    df_true['trace_init'] = init_full
    if plotfit:
        fig,ax = plt.subplots(figsize = [10,10])            
        ax.set_ylabel("Intensity (arb. units)", fontsize = 27)
        if normalize:
            ax.set_ylabel("Normalized Intensity", fontsize = 27)
        # ax.set_yticklabels([])
        ax.set_xlim([max(energy), min(energy)])
        # ax.set_yticks([])
        ax.tick_params(axis='x', labelsize= 24)
        ax.tick_params(axis='y', labelsize= 24)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel('energy, '+x_scale, fontsize = 27)
        ax.scatter(energy, dset, s = 60, facecolor = 'none', edgecolor = 'black', linewidth = 1)
        scatter_color_range = [plt.cm.viridis(i) for i in np.linspace(0.2, 0.8, M)]
        for i in range(M):
            ax.plot(standards[i][:,0], pd_pars.iloc[i, 1]*normalized_standards[:,i], color = scatter_color_range[i], linestyle = '--', linewidth = 2)
        ax.plot(energy, df_true['trace_fit'], color = 'red', linewidth = 3)
    return pd_pars
    
test = LinearCombinationFitting(jact_d1['tio2_sio2'], [jact_d1['anatase'], jact_d1['rutile'][:901]])