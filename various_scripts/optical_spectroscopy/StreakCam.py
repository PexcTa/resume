# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:12:47 2021

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


#%% Processing Functions
def normalize_1(data):
    """Normalizes a vector to its maximum value"""
    normalized_data = np.abs(data)/max(data)
    return normalized_data

def FitSingleKineticWithRise(time_vector, int_vector, exp_no, initPar, bounds, tlims, ylims, logscale):
    if bounds == 0:
        bounds = (-np.inf, np.inf)
    time = time_vector
    x = time_vector
    y = int_vector
    nan_i = y.index[y.isnull()]
    y = y.dropna()
    x = x.drop(nan_i)
    # xplt = np.linspace(min(x), max(x), 50)
    if exp_no == 1:
        fittedParameters, pcov = curve_fit(TA_1exp, x, y, initPar, bounds = bounds)
        amp1, fwhm, tau1, t0, amp0 = fittedParameters
        yplt = TA_1exp(x, amp1, fwhm, tau1, t0, amp0)
        pars = ['Amp1', 'IRF FWHM', 'Tau1', 'Time Zero', 'Offset']
    if exp_no == 2:
        fittedParameters, pcov = curve_fit(TA_2exp, x, y, initPar, bounds = bounds)
        amp1, fwhm, tau1, t0, amp2, tau2, amp0 = fittedParameters
        yplt = TA_2exp(x, amp1, fwhm, tau1, t0, amp2, tau2, amp0)
        pars = ['Amp1', 'IRF FWHM', 'Tau1', 'Time Zero', 'Amp2', 'Tau2', 'Offset']
    if exp_no == 3:
        fittedParameters, pcov = curve_fit(TA_3exp, x, y, initPar, bounds = bounds)
        amp1, fwhm, tau1, t0, amp2, tau2, amp3, tau3, amp0 = fittedParameters
        yplt = TA_3exp(x, amp1, fwhm, tau1, t0, amp2, tau2, amp3, tau3, amp0)
        pars = ['Amp1', 'IRF FWHM', 'Tau1', 'Time Zero', 'Amp2', 'Tau2', 'Amp3', 'Tau3', 'Offset']
    if exp_no == 4:
        fittedParameters, pcov = curve_fit(TA_4exp, x, y, initPar, bounds = bounds)
        amp1, fwhm, tau1, t0, amp2, tau2, amp3, tau3,amp4, tau4, amp0 = fittedParameters
        yplt = TA_4exp(x, amp1, fwhm, tau1, t0, amp2, tau2, amp3, tau3,amp4, tau4, amp0)
        pars = ['Amp1', 'IRF FWHM', 'Tau1', 'Time Zero', 'Amp2', 'Tau2', 'Amp3', 'Tau3', 'Amp4', 'Tau4', 'Offset']
    fig, ax = plt.subplots(figsize=(10,8))
    resid = np.subtract(y, yplt)
    SE = np.square(resid) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(resid) / np.var(y))
    if max(tlims) > 100 and max(tlims) <= 500:
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
    if max(tlims) > 10 and max(tlims) <= 100:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    if max(tlims) <= 10:
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    if max(tlims) > 500 and max(tlims) <= 1000:
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
    if max(tlims) > 1000:
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
    elif tlims == 0: 
        pass
    ax.set_xlim(tlims)
    ax.set_xlabel("Time, ps", fontsize = 20)
    ax.set_ylabel("Intensity (counts)", fontsize = 20)
    if ylims == 0: 
        pass
    else:
        ax.set_ylim(ylims)
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
    ax.axhline(y=0, color = 'black', ls = '--')
    if logscale == 1:
        zr = next(x for x, val in enumerate(time) if val > 0)
        x = x[zr:]
        y = y[zr:]
        yplt = yplt[zr:]
        ax.scatter(x,y, color = 'black', s = 40, marker = "D", label = 'Data')
        ax.plot(x,yplt, color = 'purple', linewidth = 3, label = 'Fit')
        ax.scatter(x, np.subtract(y,yplt), color = 'magenta', s = 10, label = 'Residuals')
        ax.set_xscale('log')
    else:
        ax.scatter(x,y, color = 'black', s = 40, marker = "D", label = 'Data')
        ax.plot(x,yplt, color = 'purple', linewidth = 3, label = 'Fit')
        ax.scatter(x, np.subtract(y,yplt), color = 'magenta', s = 10, label = 'Residuals')
    ax.legend(loc = 'upper right')
    dummy = []
    for i in range(len(pars) - (2+exp_no)):
        dummy.append('0')
    if exp_no == 1:
        dummy.append('A1%:' + str(amp1/(amp1+amp0)))
        dummy.append('A0%:' + str(amp0/(amp1+amp0)))
    elif exp_no == 2:
        dummy.append('A1%:' + str(amp1/(amp1+amp2+amp0)))
        dummy.append('A2%:' + str(amp2/(amp1+amp2+amp0)))
        dummy.append('A0%:' + str(amp0/(amp1+amp2+amp0)))
    elif exp_no == 3:
        dummy.append('A1%:' + str(amp1/(amp1+amp2+amp3+amp0)))
        dummy.append('A2%:' + str(amp2/(amp1+amp2+amp3+amp0)))
        dummy.append('A3%:' + str(amp3/(amp1+amp2+amp3+amp0)))
        dummy.append('A0%:' + str(amp0/(amp1+amp2+amp3+amp0)))
    dummy.append(str(Rsquared))
    res = pd.DataFrame(
    {'parameter': pars,
     'value': list(fittedParameters),
     'sigma': list(np.sqrt(np.diag(pcov))),
     'R2': dummy,
    })
    return res

#%% Importing Functions

def ImportImage(file):
    with open(file) as current_file:
        d = pd.DataFrame(np.loadtxt(current_file, dtype=int))
        return d

def ImportFullKinetic(file):
    with open(file) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, names = ['time', 'cts'], engine = 'python')
        return d
    
def ImportFullSpectrum(file):
    with open(file) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, names = ['wl', 'cts'], engine = 'python')
        return d
def nm2WN(data_in_nm):
    return 10000000/data_in_nm
def WN2nm(data_in_WN):
    return 10000000/data_in_WN
def ImportDataset():
    images = {}
    kinetics = {}
    spectra = {}
    backgrounds = {}
    data_files = os.listdir()
    def XtractNumbers(string):
        nums = ''
        for symbol in string:
            if symbol in '0123456789':
                nums += symbol
        return nums
    for file in data_files:
        if file.endswith('.dat'):
            label = 'streak'+XtractNumbers(str(file))
            images[label] = ImportImage(file)
        if file.startswith('kinetic'):
            label = 'kin'+XtractNumbers(str(file))
            kinetics[label] = ImportFullKinetic(file)
        if file.startswith('background'):
            label = 'bg'+XtractNumbers(str(file))
            backgrounds[label] = ImportFullKinetic(file)
        if file.startswith('spectrum'):
            label = 'spec'+XtractNumbers(str(file))
            spectra[label] = ImportFullSpectrum(file)
    return(images, kinetics, backgrounds, spectra)

def saver(dictex):
    for key, val in dictex.items():
        val.to_csv("fits_with_rise_{}.csv".format(str(key)))

    with open("keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))
        
def loader():
    """Reading data from keys"""
    with open("keys.txt", "r") as f:
        keys = eval(f.read())

    dictex = {}    
    for key in keys:
        dictex[key] = pd.read_csv("data_{}.csv".format(str(key)))

    return dictex
#%%
Sample12 = ImportDataset()

#%% Plotting Things

dset = Sample12

def StreakPlot(wale, time, timescale, data, xlim, lvl, cmap, save):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_xlabel('Wavelength (nm)', fontsize = 20)
    ax.set_ylabel('Time ('+timescale+")", fontsize = 20)
    x = wale[-1::-1]
    y = time
    z = data
    cs1 = ax.contourf(x, y, z, lvl, cmap = cmap)
    if xlim == 0:
        ax.set_xlim([min(x), max(x)])
    else:
        ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(cs1, cax=cax, orientation='vertical')
    cbar.set_ticks([np.linspace(z.min().min(), z.max().max(), 10, dtype = int)])
    # if save == 1:
    #     plt.savefig(str(np.random.randint(1000))+'TAplotFig.svg')
    
StreakPlot(dset[3]['spec2800']['wl'], dset[1]['kin280']['time'], 'ps', dset[0]['streak280'], 0, 25, 'inferno', 0)
# StreakPlot(dset[3]['spec2']['wl'], dset[1]['kin2']['time'], 'ps', dset[0]['streak2'], 0, 25, 'inferno', 0)
#%%
s1k2 = (S1_nNU1k_tft[1]['kin2']['time'], S1_nNU1k_tft[1]['kin2']['cts'])
s2k2 = (S2_nNU1k_dmf[1]['kin2']['time'], S2_nNU1k_dmf[1]['kin2']['cts'])
s3k2 = (S3_TiSIM_tft[1]['kin2']['time'], S3_TiSIM_tft[1]['kin2']['cts'])
s4k2 = (S4_TiSIM_dmf[1]['kin2']['time'], S4_TiSIM_dmf[1]['kin2']['cts'])
s7k2 = (S7_NiSIM_tft[1]['kin2']['time'], S7_NiSIM_tft[1]['kin2']['cts'])
s8k2 = (S8_NiSIM_dmf[1]['kin2']['time'], S8_NiSIM_dmf[1]['kin2']['cts'])
s9k2 = (S9_CoSIM_tft[1]['kin2']['time'], S9_CoSIM_tft[1]['kin2']['cts'])
s10k2 = (S10_CoSIM_dmf[1]['kin2']['time'], S10_CoSIM_dmf[1]['kin2']['cts'])
Refk120 = (CdSe[1]['kin120']['time'], CdSe[1]['kin120']['cts']-np.average(CdSe[2]['bg120']['cts']))
AllKin2ps = [s1k2, s2k2, s3k2, s4k2, s7k2, s8k2, s9k2, s10k2]
#%%
s1k120 = (S1_nNU1k_tft[1]['kin120']['time'], S1_nNU1k_tft[1]['kin120']['cts']-np.average(S1_nNU1k_tft[2]['bg120']['cts']))
s2k120 = (S2_nNU1k_dmf[1]['kin120']['time'], S2_nNU1k_dmf[1]['kin120']['cts']-np.average(S2_nNU1k_dmf[2]['bg120']['cts']))
s3k120 = (S3_TiSIM_tft[1]['kin120']['time'], S3_TiSIM_tft[1]['kin120']['cts']-np.average(S3_TiSIM_tft[2]['bg120']['cts']))
s4k120 = (S4_TiSIM_dmf[1]['kin120']['time'], S4_TiSIM_dmf[1]['kin120']['cts']-np.average(S4_TiSIM_dmf[2]['bg120']['cts']))
s7k120 = (S7_NiSIM_tft[1]['kin120']['time'], S7_NiSIM_tft[1]['kin120']['cts']-np.average(S7_NiSIM_tft[2]['bg120']['cts']))
s8k120 = (S8_NiSIM_dmf[1]['kin120']['time'], S8_NiSIM_dmf[1]['kin120']['cts']-np.average(S8_NiSIM_dmf[2]['bg120']['cts']))
# s8k120_RM = (S8_NiSIM_dmf_RM[1]['kin120']['time'], S8_NiSIM_dmf_RM[1]['kin120']['cts']-np.average(S8_NiSIM_dmf_RM[2]['bg120']['cts']))
s9k120 = (S9_CoSIM_tft[1]['kin120']['time'], S9_CoSIM_tft[1]['kin120']['cts']-np.average(S9_CoSIM_tft[2]['bg120']['cts']))
s10k120 = (S10_CoSIM_dmf[1]['kin120']['time'], S10_CoSIM_dmf[1]['kin120']['cts']-np.average(S10_CoSIM_dmf[2]['bg120']['cts']))
Refk120 = (CdSe[1]['kin120']['time'], CdSe[1]['kin120']['cts']-np.average(CdSe[2]['bg120']['cts']))
# s10k120_RM = (S10_CoSIM_dmf_RM[1]['kin120']['time'], S10_CoSIM_dmf_RM[1]['kin120']['cts']-np.average(S10_CoSIM_dmf_RM[2]['bg120']['cts']))
AllKin120ps = [s1k120, s2k120, s3k120, s4k120, s7k120, s8k120, s9k120, s10k120]
#%%
s1s2 = (S1_nNU1k_tft[3]['spec2']['wl'], S1_nNU1k_tft[3]['spec2']['cts'])
s2s2 = (S2_nNU1k_dmf[3]['spec2']['wl'], S2_nNU1k_dmf[3]['spec2']['cts'])
s3s2 = (S3_TiSIM_tft[3]['spec2']['wl'], S3_TiSIM_tft[3]['spec2']['cts'])
s4s2 = (S4_TiSIM_dmf[3]['spec2']['wl'], S4_TiSIM_dmf[3]['spec2']['cts'])
s7s2 = (S7_NiSIM_tft[3]['spec2']['wl'], S7_NiSIM_tft[3]['spec2']['cts'])
s8s2 = (S8_NiSIM_dmf[3]['spec2']['wl'], S8_NiSIM_dmf[3]['spec2']['cts'])
s9s2 = (S9_CoSIM_tft[3]['spec2']['wl'], S9_CoSIM_tft[3]['spec2']['cts'])
s10s2 = (S10_CoSIM_dmf[3]['spec2']['wl'], S10_CoSIM_dmf[3]['spec2']['cts'])

AllSpec2 = [s1s2, s2s2, s3s2, s4s2, s7s2, s8s2, s9s2, s10s2]
#%%
s1s120 = (S1_nNU1k_tft[3]['spec120']['wl'], S1_nNU1k_tft[3]['spec120']['cts'])
s2s120 = (S2_nNU1k_dmf[3]['spec120']['wl'], S2_nNU1k_dmf[3]['spec120']['cts'])
s3s120 = (S3_TiSIM_tft[3]['spec120']['wl'], S3_TiSIM_tft[3]['spec120']['cts'])
s4s120 = (S4_TiSIM_dmf[3]['spec120']['wl'], S4_TiSIM_dmf[3]['spec120']['cts'])
s7s120 = (S7_NiSIM_tft[3]['spec120']['wl'], S7_NiSIM_tft[3]['spec120']['cts'])
s8s120 = (S8_NiSIM_dmf[3]['spec120']['wl'], S8_NiSIM_dmf[3]['spec120']['cts'])
s9s120 = (S9_CoSIM_tft[3]['spec120']['wl'], S9_CoSIM_tft[3]['spec120']['cts'])
s10s120 = (S10_CoSIM_dmf[3]['spec120']['wl'], S10_CoSIM_dmf[3]['spec120']['cts'])

AllSpec2 = [s1s2, s2s2, s3s2, s4s2, s7s2, s8s2, s9s2, s10s2]
#%%
s8s120_RM = (S8_NiSIM_dmf_RM[3]['spec120']['wl'], S8_NiSIM_dmf_RM[3]['spec120']['cts'])

s10s120_RM = (S10_CoSIM_dmf_RM[3]['spec120']['wl'], S10_CoSIM_dmf_RM[3]['spec120']['cts'])

# AllSpec120.append([s8s120_RM, s10s120_RM])
#%%
tlims = [0, 200]
# ylims = [0.05, -1.1]
fig, ax = plt.subplots(figsize=(10,8))
if max(tlims) > 100 and max(tlims) <= 500:
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
if max(tlims) > 10 and max(tlims) <= 100:
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
if max(tlims) <= 10:
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
if max(tlims) > 500 and max(tlims) <= 1000:
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(20))
if max(tlims) > 1000:
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.set_xlim(tlims)
ax.axhline(y=0, color = 'dimgrey', ls = '--')
# ax.axhline(y=0.5, color = 'dimgrey', ls = '--')
# ax.set_ylim([-0.1, 1.1])
ax.set_xlabel("Time (ps)", fontsize = 24)
# ax.set_ylabel("Normalized Intensity", fontsize = 20)
ax.tick_params(axis='x', labelsize= 22)
ax.tick_params(axis='y', right = False, left = False, labelleft = False, labelsize= 18)


ax.plot(s1k2[0], savgol_filter(normalize_1(s1k2[1]),5,2), color = 'goldenrod', linewidth = 4, label = 'NDC-NU-1000')
# ax.plot(dset[1]['kin120']['time'], normalize_1(dset[1]['kin120']['cts']), color = 'darkorange', linestyle = '--', linewidth = 3, label = 'FF-NU-1000 (RT)')
# ax.plot(dset[1]['kin12080']['time']-3.5, normalize_1(dset[1]['kin12080']['cts']), color = 'darksalmon', linestyle = '--', linewidth = 3, label = 'FF-NU-1000 (80K)')
ax.plot(s3k2[0]-37, savgol_filter(normalize_1(s3k2[1]),5,2), color = 'slategrey', linewidth = 4, label = 'Ti-SIM (DMF)')
ax.plot(s7k2[0]-60, savgol_filter(normalize_1(s7k2[1]),5,2), color = 'teal', linewidth = 4, label = 'Ni-SIM (DMF)')
# # ax.plot(s10k120[0]+1.5, savgol_filter(normalize_1(s10k120[1]),5,2), color = 'crimson', linewidth = 3, label = 'Ni-SIM 06/03')
ax.plot(s9k2[0]-27, savgol_filter(normalize_1(s9k2[1]),5,2), color = 'darkmagenta', linewidth = 4, label = 'Co-SIM (TFT)')
# ax.plot(s10k2[0]-25, savgol_filter(normalize_1(s10k2[1]),5,2), color = 'crimson', linewidth = 3, linestyle = '--', label = 'Co-SIM (DMF)')
# ax.plot(Refk120[0], savgol_filter(normalize_1(Refk120[1]),5,2), color = 'red', linewidth = 3, linestyle = '--', alpha = 0.5, label = 'CdSe')
# ax.legend(loc = 'lower right', fontsize = 16)

#%% Plot the Spectra, time-integrated
wlims = [350,700]
fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(wlims)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.axhline(y=0, color = 'dimgrey', ls = '--')
# ax.set_ylim([-0.1, 1.1])
ax.set_xlabel("Wavelength (nm)", fontsize = 20)
ax.set_ylabel("Normalized Intensity", fontsize = 20)
ax.tick_params(axis='x', labelsize= 18)
ax.tick_params(axis='y', labelsize= 18)
ax.plot(s2s2[0], savgol_filter(normalize_1(s2s2[1]),7,2), color = 'goldenrod', linewidth = 3, label = 'NDC-NU-1000')
ax.plot(s4s2[0], savgol_filter(normalize_1(s4s2[1]),7,2), color = 'slategrey', linewidth = 3, label = 'Ti-SIM')
ax.plot(s8s2[0], savgol_filter(normalize_1(s8s2[1]),7,2), color = 'teal', linewidth = 3, label = 'Ni-SIM')
ax.plot(s10s2[0], savgol_filter(normalize_1(s10s2[1]),7,2), color = 'darkmagenta', linewidth = 3, label = 'Co-SIM')
ax.legend(loc = 'upper right', fontsize = 16)
#%% Plot the spectra, follow the red shift
dset = Sample12

wlims = [400,525]
fig, ax = plt.subplots(figsize=(11,8))
ax.set_xlim(wlims)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.axhline(y=0, color = 'dimgrey', ls = '--')
# ax.set_ylim([0,25])
ax.set_xlabel("Wavelength (nm)", fontsize = 20)
ax.set_ylabel("Normalized Intensity", fontsize = 20)
ax.tick_params(axis='x', labelsize= 18)
ax.tick_params(axis='y', labelsize= 18)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
n = 15
ax.set_prop_cycle('color',[plt.cm.inferno(i) for i in np.linspace(0, 1, n)])
for i in range(0,480,32):
    ax.plot(dset[3]['spec2']['wl'][-1::-1], normalize_1(savgol_filter(dset[0]['streak2'].iloc[i,:], 101,2)), linewidth = 2)
    # ax.plot(dset[3]['spec2']['wl'][-1::-1], dset[0]['streak2'].iloc[i,:])
sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=0, vmax=2000))
# sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=0, vmax=120))
cbar = plt.colorbar(sm, pad = 0.01)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Time (ps)', fontsize = 18)

#%% Plot the spectra, follow the red shift - REBINNED

dset = S3_TiSIM_tft

wlims = [400,650]
fig, ax = plt.subplots(figsize=(11,8))
# ax.set_facecolor('silver')
ax.set_xlim(wlims)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.axhline(y=0, color = 'dimgrey', ls = '--')
ax.set_ylim([0,300])
ax.set_xlabel("Wavelength (nm)", fontsize = 24)
# ax.set_ylabel("Counts", fontsize = 24)
ax.set_ylabel(None)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
n = 14
ax.set_prop_cycle('color',[plt.cm.inferno(i) for i in np.linspace(0, 1, n)])
for i,j in zip(range(0,448,32), range(32,480,32)):
    ax.plot(dset[3]['spec2']['wl'][-1::-1], np.sum(dset[0]['streak2'].iloc[i:j,:]), linewidth = 2)
    # ax.plot(dset[3]['spec2']['wl'][-1::-1], dset[0]['streak2'].iloc[i,:])
sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=0, vmax=2000))
# sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=0, vmax=120))
cbar = plt.colorbar(sm, pad = 0.01)
cbar.ax.tick_params(labelsize=20)
cbar.set_label('Time (ps)', fontsize = 24)

#%% Kinetic fitting - it's finally here
def expdecay_1(data, amp1, tau1, t0, amp0):
    return amp1*np.exp(-(data+t0)/tau1) + amp0
def expdecay_2(data, amp1, tau1, amp2, tau2, t0, amp0):
    return amp1*np.exp(-(data+t0)/tau1) + amp2*np.exp(-(data+t0)/tau2) + amp0
def expdecay_3(data, amp1, tau1, amp2, tau2, amp3, tau3, t0, amp0):
    return amp1*np.exp(-(data+t0)/tau1) + amp2*np.exp(-(data+t0)/tau2) + amp3*np.exp(-(data+t0)/tau3) +amp0

initPar = [0.1, 40, 35, 0]

fits_120ps['s10k120'] = FitSingleKinetic(s10k120[0][110:], normalize_1(s10k120[1][110:]), 1, initPar, 0, [0,120], 0, 0)


#%%
def TA_4exp(data, amp1, fwhm, tau1, t0, amp2, tau2, amp3, tau3, amp4, tau4, amp0):
    return amp1*fwhm*np.exp((fwhm/(np.sqrt(2)*tau1))**2-(data-t0)/tau1)*(1-special.erf(fwhm/(np.sqrt(2)*tau1)-(data-t0)/(np.sqrt(2)*fwhm))) + \
        + amp2*fwhm*np.exp((fwhm/(np.sqrt(2)*tau2))**2-(data-t0)/tau2)*(1-special.erf(fwhm/(np.sqrt(2)*tau2)-(data-t0)/(np.sqrt(2)*fwhm))) + \
            + amp3*fwhm*np.exp((fwhm/(np.sqrt(2)*tau3))**2-(data-t0)/tau3)*(1-special.erf(fwhm/(np.sqrt(2)*tau3)-(data-t0)/(np.sqrt(2)*fwhm))) + \
                + amp4*fwhm*np.exp((fwhm/(np.sqrt(2)*tau4))**2-(data-t0)/tau4)*(1-special.erf(fwhm/(np.sqrt(2)*tau4)-(data-t0)/(np.sqrt(2)*fwhm))) + \
                    + amp0

def TA_3exp(data, amp1, fwhm, tau1, t0, amp2, tau2, amp3, tau3, amp0):
    return amp1*fwhm*np.exp((fwhm/(np.sqrt(2)*tau1))**2-(data-t0)/tau1)*(1-special.erf(fwhm/(np.sqrt(2)*tau1)-(data-t0)/(np.sqrt(2)*fwhm))) + \
        + amp2*fwhm*np.exp((fwhm/(np.sqrt(2)*tau2))**2-(data-t0)/tau2)*(1-special.erf(fwhm/(np.sqrt(2)*tau2)-(data-t0)/(np.sqrt(2)*fwhm))) + \
            + amp3*fwhm*np.exp((fwhm/(np.sqrt(2)*tau3))**2-(data-t0)/tau3)*(1-special.erf(fwhm/(np.sqrt(2)*tau3)-(data-t0)/(np.sqrt(2)*fwhm))) + \
                + amp0
                
def TA_2exp(data, amp1, fwhm, tau1, t0, amp2, tau2, amp0):
    return amp1*fwhm*np.exp((fwhm/(np.sqrt(2)*tau1))**2-(data-t0)/tau1)*(1-special.erf(fwhm/(np.sqrt(2)*tau1)-(data-t0)/(np.sqrt(2)*fwhm))) + \
        + amp2*fwhm*np.exp((fwhm/(np.sqrt(2)*tau2))**2-(data-t0)/tau2)*(1-special.erf(fwhm/(np.sqrt(2)*tau2)-(data-t0)/(np.sqrt(2)*fwhm))) + \
            + amp0
            
def TA_1exp(data, amp1, fwhm, tau1, t0, amp0):
    return amp1*fwhm*np.exp((fwhm/(np.sqrt(2)*tau1))**2-(data-t0)/tau1)*(1-special.erf(fwhm/(np.sqrt(2)*tau1)-(data-t0)/(np.sqrt(2)*fwhm))) + \
        + amp0
bounds2 = ([0, -np.inf,-np.inf,-np.inf,0,-np.inf, 0],
          [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,0.0001])

bounds3 = ([0, -np.inf,-np.inf,-np.inf,0,-np.inf,0,-np.inf,0],
          [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,0.00001])

initPar = [0.1, 20, 20, 100, 0.1, 200, 0.1, 1000, 0.00001]
fits_2ns_07262021['s10k2'] = FitSingleKineticWithRise(s10k2[0], s10k2[1], 3, initPar, bounds3, [0,2000], 0, 0)

#%%
fig, ax = plt.subplots()
