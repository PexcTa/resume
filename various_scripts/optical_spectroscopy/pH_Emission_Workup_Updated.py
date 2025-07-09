# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:10:07 2022

@author: boris
"""
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# import lmfit as lm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
# from sympy import S, symbols, printing
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter as sgf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
import os
import ast
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
def ReadFile(filename, background=True):
    with open(filename) as current_file:
        dataset = np.loadtxt(current_file, delimiter = ',', dtype = 'str')
        if background:
            dataset[0,0:2] = ['0', '0']
        else: 
            dataset[0,0] = '0'
        dataset = dataset.astype(float)
    return dataset

def ReadCurrentFolder():
    filelist = []
    for file in os.listdir():
        if file.endswith('.csv'):
            filelist.append(file)
    output = ReadFile(filelist[0])
    for file in filelist[1:]:
        temp = ReadFile(file)
        temp = temp[:,2:]
        output = np.concatenate([output, temp], axis = 1)
    wl_and_bg = output[:,:2]
    data = output[:,2:]
    i = np.argsort(data[0,:])
    data = data[:,i]
    output_sorted = np.concatenate([wl_and_bg, data], axis = 1)
    return output_sorted

def Export_BGSubtracted(dataset, name = 'default'):
    if name == 'default':
        name = 'last_bg_sub'
    dataset_shape = np.shape(dataset)
    output = np.zeros((dataset_shape[0], dataset_shape[1]-1))
    output[1:,0] = dataset[1:,0]
    output[0,1:] = dataset[0,2:]
    output[1:,1:] = np.transpose(np.array([np.subtract(dataset[1:,i], dataset[1:,1]) for i in range(2, len(output[0,1:])+2)]))
    np.savetxt(f"./exported/{name}.csv", output, delimiter=",")
    return output
#%%
mofPH = ReadCurrentFolder()

def phLuminescence_ContourPlot(dataset, energy_units = 'nm', colormap = 'inferno', normFactor = 1, bkgSub = True):
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_xlabel('Wavelength (nm)', fontsize = 36)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.set_ylabel('pH', fontsize = 36)
    x = dataset[1:,0]
    y = dataset[0,2:]
    if bkgSub:
        z = np.array([np.subtract(dataset[1:,i], dataset[1:,1]) for i in range(2, len(y)+2)])
    else:
        z = np.array([dataset[1:,i] for i in range(2, len(y)+2)])
    if energy_units == 'eV':
        z = np.array([jac(x, z[i,:]) for i in range(len(z[:,0]))])
        x = nm2ev(x)
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_xlim(max(x), min(x))
        ax.set_xlabel('Energy (eV)', fontsize = 36)
    z_normalized = z/normFactor
    cs1 = ax.contourf(x, y, z_normalized, 25, cmap = colormap)
    # ax.set_xlim(x[min(region)], x[max(region)])
    ax.tick_params(axis='x', labelsize= 30)
    ax.tick_params(axis='y', labelsize= 30)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(cs1, cax=cax, orientation='vertical')
    cbar.set_ticks([np.linspace(z_normalized.min().min(), z_normalized.max().max(), 10, dtype = int)])
    cax.tick_params(labelsize = 22)
    # cax.tick_params(labelright=False)  
    cax.set_ylabel('Counts per Unit O.D.', fontsize = 36)
    cax.yaxis.get_offset_text().set_fontsize(27)
    # cax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

phLuminescence_ContourPlot(mofPH, normFactor = 1, energy_units = 'eV')

# mofPH_Fix = np.delete(mofPH, 18, 1)
# phLuminescence_ContourPlot(mofPH_Fix, energy_units = 'eV')
#%% plot A File
import matplotlib.gridspec as gridspec
mofPH = ReadCurrentFolder()
fig = plt.figure(tight_layout=True, figsize=(24,12))
gs = gridspec.GridSpec(2, 2)
ax = fig.add_subplot(gs[:, 0])
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.set_ylabel('Intensity (A.U.)', fontsize = 36)
ax.tick_params(axis='x', labelsize= 30)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
ax.yaxis.get_offset_text().set_fontsize(27)
ax.tick_params(axis='y', labelsize= 30)
dataset = mofPH.copy()
x = dataset[2:,0] # wavelength
y = dataset[0,2:] # pH
od = dataset[1,2:]
z = np.array([np.subtract(dataset[2:,i], dataset[2:,1]) for i in range(2, len(y)+2)])
z = np.array([jac(x, z[i,:])/od[i] for i in range(len(z[:,0]))])
z = np.transpose(z)
x = nm2ev(x)
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.set_xlim(max(x), min(x))
ax.set_xlabel('Energy (eV)', fontsize = 36)
secax = ax.secondary_xaxis('top', functions=(ev2nm, nm2ev))
secax.tick_params(labelsize = 25)
secax.set_xlabel('Wavelength (nm)', fontsize = 30, labelpad = 10)
# ax.set_facecolor('gray')
ax.set_prop_cycle('color', [plt.cm.turbo(i) for i in np.linspace(1,0, len(y))])

choose_samples = [0,1,2,3,4,5,6,7,8]
for i in choose_samples:
    ax.plot(x, sgf(z[:,i],15,3), linewidth = 5, label = f"{y[i]:.2f}")
ax.legend(loc = 'upper right', fontsize = 28, ncol = 1)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.axvline(x = 2.48, linestyle = '--', linewidth = 3)
ax.axvspan(xmin = 2.48, xmax = 2.56, color = 'yellow')
ax.axvline(x = 2.56, linestyle = '--', linewidth = 3)


ax1 = fig.add_subplot(gs[0, 1])
for i in choose_samples:
    ax1.scatter(y[i], np.sum(np.multiply(x, z[:,i]))/np.sum(z[:,i]), facecolor = 'none', edgecolor = 'magenta', s = 300, linewidth = 5)


ax2 = fig.add_subplot(gs[1, 1])
for i in choose_samples:
    ax2.scatter(y[i], np.trapz(z[:,i]), facecolor = 'none', edgecolor = 'blue', s = 300, linewidth = 5)

ax1.set_ylabel('Spectral Moment', fontsize = 30)
ax2.set_ylabel('Relative PLQY', fontsize = 30)
ax2.set_xlabel('pH', fontsize = 30)
ax1.tick_params(axis='y', labelsize= 30)
ax1.set_xticklabels([])
ax2.yaxis.get_offset_text().set_fontsize(27)
ax2.tick_params(axis='both', labelsize= 30)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

#%%
def phIntegratedSpectra(dataset, energy_units = 'nm', grouping = 'halfUnit', colormap = 'RdBu', normFactor = 1, bkg = True, bkgSub = True):
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_xlabel('Wavelength (nm)', fontsize = 36)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.set_ylabel('Counts per Unit O.D.', fontsize = 36)
    ax.tick_params(axis='x', labelsize= 30)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.yaxis.get_offset_text().set_fontsize(27)
    ax.tick_params(axis='y', labelsize= 30)
    if bkg:
        x = dataset[1:,0] # wavelength
        y = dataset[0,2:] # pH
        if bkgSub:
            z = np.array([np.subtract(dataset[1:,i], dataset[1:,1]) for i in range(2, len(y)+2)])
        else:
            z = np.array([dataset[1:,i] for i in range(2, len(y)+2)])
    elif bkg == False:
        x = dataset[1:,0] # wavelength
        y = dataset[0,1:] # pH
        z = np.array([dataset[1:,i] for i in range(1, len(y)+1)])
    if energy_units == 'eV':
        z = np.array([jac(x, z[i,:]) for i in range(len(z[:,0]))])
        x = nm2ev(x)
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_xlim(max(x), min(x))
        ax.set_xlabel('Energy (eV)', fontsize = 36)
    z_normalized = z/normFactor
    print(np.shape(z_normalized))
    ax.set_facecolor('gray')
    if grouping == 'halfUnit':
        ph = 0
        step = 0.5
        colorrange = int(max(y)/step)
        ax.set_prop_cycle('color', [plt.cm.RdBu(i) for i in np.linspace(0, 1, colorrange)])
        while ph <= 10:
            label = f'{ph} to '
            idx_y_min = (np.abs(y - ph)).argmin()
            print(idx_y_min)
            ph += step
            label += f'{ph}'
            idx_y_max = (np.abs(y - ph)).argmin()
            print(idx_y_max)
            if idx_y_min == idx_y_max:
                continue
            signal = sgf(np.sum(z_normalized[idx_y_min:idx_y_max, :], axis = 0)/len(y[idx_y_min : idx_y_max]), 9, 3)
            ax.plot(x, signal, linewidth = 5, label = label)
        ax.legend(loc = 'upper right', fontsize = 24)
    if grouping == 'unit':
        ph = 0
        step = 1
        colorrange = int(max(y)/step)
        ax.set_prop_cycle('color', [plt.cm.RdBu(i) for i in np.linspace(0, 1, colorrange)])
        while ph <= 10:
            label = f'{ph} to '
            idx_y_min = (np.abs(y - ph)).argmin()
            print(idx_y_min)
            ph += step
            label += f'{ph}'
            idx_y_max = (np.abs(y - ph)).argmin()
            print(idx_y_max)
            if idx_y_min == idx_y_max:
                continue
            signal = sgf(np.sum(z_normalized[idx_y_min:idx_y_max, :], axis = 0)/len(y[idx_y_min : idx_y_max]), 9, 3)
            ax.plot(x, signal, linewidth = 5, label = label)
        ax.legend(loc = 'upper right', fontsize = 24)
    if type(grouping) == list:
        output = np.zeros(shape=(len(x), len(grouping)))
        output[:,0] = x
        colorrange = len(grouping) - 1
        ax.set_prop_cycle('color', [plt.cm.RdBu(i) for i in np.linspace(0, 1, colorrange)])
        for i in range(1, len(grouping)):
            label = f'{grouping[i-1]} to '
            idx_y_min = (np.abs(y - grouping[i-1])).argmin()
            label += f'{grouping[i]}'
            idx_y_max = (np.abs(y - grouping[i])).argmin()
            if idx_y_min == idx_y_max:
                continue
            signal = sgf(np.sum(z_normalized[idx_y_min:idx_y_max, :], axis = 0)/len(y[idx_y_min : idx_y_max]), 9, 3)
            output[:,i] = signal
            ax.plot(x, signal, linewidth = 5, label = label)
        ax.legend(loc = 'upper right', fontsize = 24)
    return output
    
avgs = phIntegratedSpectra(mofPH_Fix, 'eV', [0, 3.3, 5.7, 8.2, 8.5], bkg = True, bkgSub = True)

#%%
def IntegratedAreas(datasets, normFactors, label_list = 'abcdefghijklmnopqrst', colormap = 'inferno'):
    N = len(datasets)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.set_xlabel('pH', fontsize = 24)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_ylabel('Counts per Unit OD', fontsize = 24)
    ax.yaxis.get_offset_text().set_fontsize(22)
    ax.tick_params(axis='x', labelsize= 20)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.tick_params(axis='y', labelsize= 20)
    colorList = np.linspace(0.05, 0.85, N)
    cmap = matplotlib.cm.get_cmap(colormap)
    for n in range(N):
        dataset = datasets[n]
        c = colorList[n]
        x = dataset[1:,0] # wavelength
        y = dataset[0,2:] # pH
        z = np.array([np.subtract(dataset[1:,i], dataset[1:,1]) for i in range(2, len(y)+2)])
        ax.scatter(y, np.trapz(z), edgecolor = cmap(c), s = 75, linewidth = 3, facecolor = 'none', label = label_list[n])
    ax.legend(fontsize = 24, )
    
IntegratedAreas([bk207, bk197], [1,1], label_list = ['197', '207'], colormap = 'plasma')


#%% linear combination fitting
energy = test[:,0]
trace_to_fit = normalize_1(test[:,1])

en_comp = WN2ev(mofTheory[:,0])
ct_comp = mofTheory[:,3]
qd_comp = mofTheory[:,4]

interpol_ct = interp1d(en_comp, ct_comp)
interpol_qd = interp1d(en_comp, qd_comp)

ct_new = interpol_ct(energy+0.05)
qd_new = interpol_qd(energy-0.025)

standards = np.transpose(np.array([normalize_1(qd_new), normalize_1(ct_new)]))
M = 2

def LCF(standards, M, **Pars):
    return np.sum([np.multiply(Pars[f'A_{m}'],standards[:,m-1]) for m in range(1,M+1)], axis = 0)

model = lm.Model(LCF)
params = lm.Parameters()
for m in range(1, M+1):
    params.add(f'A_{m}', value = 1/M, min = 0)
params.add("M", value = M, vary = False)
print('Model parameters constructed')
print('Fitting...')
result = model.fit(trace_to_fit, standards = standards, M = M, params=params)
print(result.fit_report())
pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
results_full = result.eval(energy = energy)
fig,ax = plt.subplots(figsize = [6,6])            
# ax.set_ylabel("Normalized Intensity", fontsize = 27)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xlim([max(energy), min(energy)])
# ax.set_yticks([])
# ax.tick_params(axis='x', labelsize= 24)
# ax.tick_params(axis='y', labelsize= 24)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.set_xlabel('energy, '+x_scale, fontsize = 27)
cmap = matplotlib.cm.get_cmap('RdBu')
ax.scatter(energy, trace_to_fit, s = 60, facecolor = 'none', edgecolor = cmap(0), linewidth = 1)

ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax.plot(energy, pd_pars.iloc[0, 1]*standards[:,0], color = 'magenta', linestyle = '--', linewidth = 2)
ax.plot(energy, pd_pars.iloc[1, 1]*standards[:,1], color = 'lawngreen', linestyle = '--', linewidth = 2)
ax.plot(energy, results_full, color = 'black', linewidth = 3, alpha = 0.75)

aQD = np.trapz(pd_pars.iloc[0, 1]*standards[:,0], x = energy)
aCT = np.trapz(pd_pars.iloc[1, 1]*standards[:,1], x = energy)

ax.annotate(f"QD: {(aQD/(aQD+aCT)*100):.2f} % \nCT: {(aCT/(aQD+aCT)*100):.2f} %", (3, 0.3), fontsize = 18)

ax.legend(['Fit: QD', 'Fit: CT', 'Fit: Total', '1.50 to 3.30'], loc = 'upper right', fontsize = 18)
#%% linear combination fitting - With Gaussian BG
energy = test[:,0]
trace_to_fit = normalize_1(test[:,1])

en_comp = WN2ev(mofTheory[:,0])
ct_comp = mofTheory[:,3]
qd_comp = mofTheory[:,4]

interpol_ct = interp1d(en_comp, ct_comp)
interpol_qd = interp1d(en_comp, qd_comp)

ct_new = interpol_ct(energy+0.03)
qd_new = interpol_qd(energy+0.035)

standards = np.transpose(np.array([normalize_1(qd_new), normalize_1(ct_new)]))
standards = np.column_stack((energy, standards))

M = 2

def LCF(standards, M, AG, x0, fwhm, **Pars):
    linearSum = np.sum([np.multiply(Pars[f'A_{m}'],standards[:,m]) for m in range(1,M+1)], axis = 0)
    gaussian = (AG*np.exp(-(standards[:,0]-x0)**2/fwhm**2))
    return np.sum([linearSum, gaussian], axis = 0)
    

    
    
model = lm.Model(LCF)
params = lm.Parameters()
for m in range(1, M+1):
    params.add(f'A_{m}', value = 1/M, min = 0)
params.add("M", value = M, vary = False)
params.add("AG", value = 1)
params.add("x0", value = 3, min = 0, max = 5)
params.add("fwhm", value = 0.5, min = 0)

print('Model parameters constructed')
print('Fitting...')
result = model.fit(trace_to_fit, standards = standards, M = M, params=params)
print(result.fit_report())
pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
results_full = result.eval(energy = energy)
fig,ax = plt.subplots(figsize = [6,6])            
# ax.set_ylabel("Normalized Intensity", fontsize = 27)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xlim([max(energy), min(energy)])
# ax.set_yticks([])
# ax.tick_params(axis='x', labelsize= 24)
# ax.tick_params(axis='y', labelsize= 24)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.set_xlabel('energy, '+x_scale, fontsize = 27)
cmap = matplotlib.cm.get_cmap('RdBu')
ax.scatter(energy, trace_to_fit, s = 60, facecolor = 'none', edgecolor = cmap(0), linewidth = 1)

ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax.plot(energy, pd_pars.iloc[0, 1]*standards[:,1], color = 'magenta', linestyle = '--', linewidth = 2)
ax.plot(energy, pd_pars.iloc[1, 1]*standards[:,2], color = 'lawngreen', linestyle = '--', linewidth = 2)
ax.plot(energy, pd_pars.iloc[3, 1]*np.exp(-(standards[:,0]-pd_pars.iloc[4, 1])**2/pd_pars.iloc[5, 1]**2), color = 'sienna', linestyle = '--', linewidth = 2)
ax.plot(energy, results_full, color = 'black', linewidth = 3, alpha = 0.75)

aQD = np.trapz(pd_pars.iloc[0, 1]*standards[:,1], x = energy)
aCT = np.trapz(pd_pars.iloc[1, 1]*standards[:,2], x = energy)
aGA = np.trapz(pd_pars.iloc[3, 1]*np.exp(-(standards[:,0]-pd_pars.iloc[4, 1])**2/pd_pars.iloc[5, 1]**2), x = energy)

ax.annotate(f"QD: {(aQD/(aQD+aCT+aGA)*100):.2f} % \nCT: {(aCT/(aQD+aCT+aGA)*100):.2f} % \nGaus: {(aGA/(aQD+aCT+aGA)*100):.2f} %", (3, 0.3), fontsize = 18)

ax.legend(['Fit: QD', 'Fit: CT', 'Fit: Gaussian', 'Fit: Total', '1.50 to 3.30'], loc = 'upper right', fontsize = 18)

#%% plotting sads and saks


SAK = np.loadtxt('SAKs_slidings0p5.csv', delimiter = ',')
SAD = np.loadtxt('SADs_slidings0p5.csv', delimiter = ',')
DAT = np.loadtxt('sliding_s0p5.csv', delimiter = ',')

#%%
fig, axes = plt.subplots(16,1,sharex = True,figsize=(12,12))
plt.subplots_adjust(hspace=0)
big_ax = fig.add_subplot(111)
big_ax.set_facecolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
for i in range(0,16):
    axes[i].set_yticks([])
    axes[i].set_facecolor('gray')
    # axes[i].set_ylim([0, 3500000])
big_ax.set_xticks([])
big_ax.set_yticks([])
axes[15].set_xlim(370, 600)
axes[15].tick_params('x', labelsize = 24)
axes[15].set_xlabel('Wavelength (nm)', fontsize = 32)

x = SAD[:,0]

cmap1 = matplotlib.cm.get_cmap('RdBu')
cmap2 = matplotlib.cm.get_cmap('gnuplot2')
colors = np.linspace(0.01,0.99,16)

for i in range(16):
    y = DAT[:, i+2]
    axes[15-i].set_ylim([0, 3700000 - 250000*(11-i)])
    axes[15-i].plot(x, sgf(y,9,3), linewidth = 10, color = cmap1(colors[i]), alpha = 0.75)
    # axes[11-i].plot(x, SAD[:,1]*SAK[i,1], linestyle = '-.', color = cmap2(0))
    # axes[11-i].plot(x, SAD[:,2]*SAK[i,2], linestyle = '-.', color = cmap2(0.33))
    # axes[11-i].plot(x, SAD[:,3]*SAK[i,3], linestyle = '-.', color = cmap2(0.67))
    axes[15-i].plot(x, SAD[:,1]*SAK[i,1]+SAD[:,2]*SAK[i,2]+SAD[:,3]*SAK[i,3], linestyle = '--', linewidth = 3.5, color = "black")

#%%
fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]},figsize=(12,12))
sad = SAD.copy()
sak = SAK.copy()
ax[0].set_xlabel('Wavelength (nm)', fontsize = 36)
ax[0].set_ylabel('Intensity (cps)', fontsize = 36)
ax[0].tick_params(axis='x', labelsize= 30)
ax[0].tick_params(axis='y', labelsize= 30)
ax[0].xaxis.set_major_locator(MultipleLocator(50))
# ax[0].yaxis.set_ticklabels([])
ax[0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 4)])
for i in range(3):
    ax[0].plot(sad[:,0], sad[:,i+1], linewidth = 4)
ax[0].legend(labels =[ 'Species 1', 'Species 2', 'Species 3'], fontsize = 30)
ax[0].axhline(y=0, alpha = 0.75, linestyle = '--', color = 'black')

ax[1].set_xlabel('pH', fontsize = 36)
ax[1].set_ylabel('Species Fraction', fontsize = 36)
ax[1].tick_params(axis='x', labelsize= 30)
ax[1].tick_params(axis='y', labelsize= 30)
ax[1].xaxis.set_major_locator(MultipleLocator(1))
ax[1].set_xlim([0,10])
ax[1].axhline(y=0.5, linestyle = '-', color=  'gray')
ax[1].axhline(y=1.0, linestyle = '-', color=  'gray')
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax[1].yaxis.set_ticklabels([])
colors = [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 4)]
for i in range(3):
    ax[1].scatter(sak[:,0], sak[:,i+1], s = 200, linewidth = 4, facecolor = 'none', color = colors[i])

pKa1 = 2.02
pKa2 = 6.64

pH = np.linspace(-2,14,1000)
def henderson_hasselbalch_2(pka1, pka2, ph):
    D = (10**(-pH))**2 +  (10**(-pka1))*(10**(-pH)) + (10**(-pka1))*(10**(-pka2))
    a0 = 1/D*(10**(-pH))**2
    a1 = 1/D*(10**(-pka1))*(10**(-pH))
    a2 = 1/D*(10**(-pka1))*(10**(-pka2))
    return (a0,a1,a2)
def henderson_hasselbalch_3(pka1, pka2, pka3, ph):
    D = (10**(-pH))**3 +  (10**(-pka1))*(10**(-pH))**2 + (10**(-pka1))*(10**(-pka2))*(10**(-pH)) + (10**(-pka1))*(10**(-pka2))*(10**(-pka3))
    a0 = 1/D*(10**(-pH))**3
    a1 = 1/D*(10**(-pka1))*(10**(-pH))**2
    a2 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pH))
    a3 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pka3))
    return (a0,a1,a2,a3)
def henderson_hasselbalch_4(pka1, pka2, pka3, pka4, ph):
    D = (10**(-pH))**4 +  (10**(-pka1))*(10**(-pH))**3 + (10**(-pka1))*(10**(-pka2))*(10**(-pH))**2 + (10**(-pka1))*(10**(-pka2))*(10**(-pka3))*(10**(-pH)) + (10**(-pka1))*(10**(-pka2))*(10**(-pka3))*(10**(-pka4))
    a0 = 1/D*(10**(-pH))**4
    a1 = 1/D*(10**(-pka1))*(10**(-pH))**3
    a2 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pH))**2
    a3 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pka3))*(10**(-pH))
    a4 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pka3))*(10**(-pka4))
    return (a0,a1,a2,a3,a4)
fractions = henderson_hasselbalch_2(pKa1, pKa2, pH)
ax[1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 4)])
for i in range(len(fractions)):
    ax[1].plot(pH, fractions[i], linewidth = 2.5)
# ax[1].legend(labels =[ 'Species 1', 'Species 2', 'Species 3', 'Species 4'], fontsize = 30)
# ax[1].axhline(y=0, alpha = 0.75, linestyle = '--', color = 'black')
plt.tight_layout()
ax[0].yaxis.get_offset_text().set_fontsize(27)