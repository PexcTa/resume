# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:42:49 2020

@author: boris

This script is currently only for:
    - fitting kinetic traces (one by one and globally)
    - visualisation of all kinds
and it assumes that some cropping, background subtraction and chirp correction have been 
performed beforehand. 

Background subtraction is trivial to add: average pre time-zero spectra and
subtract them out from the rest of the data. 

As of the time this script was written, the author isn't proficient enough 
with either python, mathematics, or underlying physics to incorporate chirp 
correction into this code. 

PLANS: adapt SVD analysis and add it to this code
integrate background sub
integrate chirp correction

CREDIT: Dr. Pyosang Kim for the kinetic fit functions
        Wasielewski Group for the MATLAB TA analysis suite which this is based on

"""
#%% LIBRARIES
# RUN THIS BLOCK OF CODE TO GET THE LIBRARIES

import pandas as pd
from functools import reduce
import numpy as np
# from bokeh.plotting import figure, output_file, show
import os
# import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
cmaps = OrderedDict()
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, FuncFormatter, 
                               AutoMinorLocator, NullFormatter, ScalarFormatter, 
                               LogFormatter, LogLocator, FixedFormatter)
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.signal import savgol_filter
from scipy import special

#%% RUN THIS BLOCK OF CODE AND GO TO THE NEXT SECTION
def saver(dictex):
    for key, val in dictex.items():
        val.to_csv("fits_with_rise_{}.csv".format(str(key)))

    with open("keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))

def read_file(title_path):
    """read_file reads a file given a path."""
    with open(title_path) as current_file:
        file = pd.read_csv(current_file, sep = ',', header = None)
    return file

def CheckFloat(s):
    """makes sure your data is a matrix of floats"""
    try: 
        float(s)
        return True
    except ValueError:
        return False
    
def normalize_1(data):
    """Normalizes a vector to it's maximum value. /
    All data must the of the same sign!"""
    mask = np.isnan(data)
    normalized_data = data[~mask]/max(data[~mask])
    if np.mean(data) < 0:
        normalized_data = np.abs(data[~mask])/max(np.abs(data[~mask]))
        normalized_data = np.negative(normalized_data)
    return normalized_data

def flip_sign(data):
    data = data.mul(-1)
    return data
    
def TAPlot(wale, time, data, xlim, ylim, lvl, linlog, cmap, save):
    if linlog == 0: 
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_xlabel('Wavelength, nm', fontsize = 24)
        ax.set_ylabel('Time, ps', fontsize = 24)
        x = wale
        y = time
        z = np.transpose(data)
        cs1 = ax.contourf(x, y, z, lvl, cmap = cmap)
        if ylim != 0:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([min(y), max(y)])
        if xlim == 0:
            ax.set_xlim([min(x), max(x)])
        else:
            ax.set_xlim(xlim)
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.tick_params(axis='x', labelsize= 24)
        ax.tick_params(axis='y', labelsize= 24)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(cs1, cax=cax, orientation='vertical')
        cbar.set_ticks([np.linspace(z.min().min(), z.max().max(), 10)])
    else:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.set_xlabel('Wavelength, nm', fontsize = 24)
        ax.set_ylabel('$Log_{10}t, ps$', fontsize = 24)
        x = wale
        ind = time > 0
        y = np.log10(time[ind])
        z = np.transpose(data)[ind]
        cs1 = ax.contourf(x, y, z, lvl, cmap = cmap)
        if ylim != 0:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([min(y), max(y)])
        if xlim == 0:
            ax.set_xlim([min(x), max(x)])
        else:
            ax.set_xlim(xlim)
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        # ax.yaxis.set_major_formatter(FixedFormatter(['dummy','0','1','2','3']))
        ax.tick_params(axis='x', labelsize= 24)
        ax.tick_params(axis='y', labelsize= 24)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(cs1, cax=cax, orientation='vertical')
        cbar.set_ticks([np.linspace(z.min().min(), z.max().max(), 6)])
        cax.set_yticklabels(['{:.3f}'.format(y) for y in np.linspace(z.min().min(), z.max().max(), 6)], fontsize = 20)
    if save == 1:
        plt.savefig(str(np.random.randint(1000))+'TAplotFig.svg')

def PlotSingleSlice(time_vector, time_chosen, wale_vector, data, save):
    i = pd.Index(time_vector).get_loc(time_chosen)
    slic = data[i]
    label = str(round(time_chosen)) + ' ps'
    fig, ax = plt.subplots(figsize=(10,8))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim([min(wale_vector), max(wale_vector)])
    ax.set_xlabel("Wavelength, nm", fontsize = 20)
    ax.set_ylabel("Intensity, OD", fontsize = 20)
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
    ax.axhline(y=0, color = 'black', ls = '--')
    ax.plot(wale_vector, slic, color = 'black', linewidth = 3, 
            label = label)
    ax.legend(loc = 'lower right')
    if save == 1:
        plt.savefig(str(np.random.randint(1000))+'SingleSliceFig.svg')
    
def PlotMultipleSlices(time_vector, times_chosen, wale_vector, data, xlim = 0, ylim = 0, save = 0):
    slic = pd.DataFrame()
    for t in times_chosen:
        i = pd.Index(time_vector).get_loc(t)
        if t < 10:
            label = str(round(t, 2)) + ' ps'
        elif t >= 10 and t < 100:
            label = str(round(t, 1)) + ' ps'
        else: 
            label = str(round(t)) + ' ps'
        slic[label] = data[i]
    fig, ax = plt.subplots(figsize=(10,8))
    n = len(times_chosen)
    ax.set_prop_cycle('color',[plt.cm.magma(i) for i in np.linspace(0, 1, n)])
    # ax.set_prop_cycle('color',['teal', 'darkorchid'])
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim([min(wale_vector), max(wale_vector)])
    ax.set_xlabel("Wavelength, nm", fontsize = 24)
    ax.set_ylabel(r"Intensity, $\Delta$OD", fontsize = 24)
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
    if xlim != 0:
        ax.set_xlim(xlim)
    if ylim != 0:
        ax.set_ylim(ylim)
    ax.axhline(y=0, color = 'black', ls = '--')
    for i in range(len(times_chosen)):
        ax.plot(wale_vector, slic[slic.columns[i]], linewidth = 3, label = slic.columns[i])
    ax.legend(loc = 'upper left', fontsize = 20, ncol = 2)
    if save == 1:
        plt.savefig(str(np.random.randint(1000))+'MultipleSlicesFig.svg')

def PlotSingleTrace(wale_vector, wale_chosen, time_vector, data, tlims, ylims, norm, save):
    i = pd.Index(wale_vector).get_loc(wale_chosen)
    kins = data.iloc[i]
    nan_i = kins.index[kins.isnull()]
    if norm == 1:
        kins = kins.dropna()
        kins = normalize_1(kins)
        time_vector = time_vector.drop(nan_i)
    label = str(round(wale_chosen)) + ' nm'
    fig, ax = plt.subplots(figsize=(10,8))
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
    ax.set_xlim(tlims)
    if ylims == 0: 
        pass
    else:
        ax.set_ylim(ylims)
    if norm == 1:
        ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("Time, ps", fontsize = 20)
    ax.set_ylabel("Intensity, OD", fontsize = 20)
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
    ax.axhline(y=0, color = 'black', ls = '--')
    ax.scatter(time_vector, kins, color = 'black', s = 40, marker = "D",
            label = label)
    ax.legend(loc = 'upper right')
    if save == 1:
        plt.savefig(str(np.random.randint(1000))+'SingleTraceFig.svg')


def PlotMultipleTraces(wale_vector, wales_chosen, time_vector, data, tlims, ylims, norm, save):
    trac = pd.DataFrame()
    txs = pd.DataFrame()
    for wale in wales_chosen:
        print(wale)
        i = pd.Index(wale_vector).get_loc(wale)
        print(i)
        kins = data.iloc[i]
        print(kins)
        tx = time_vector
        print(tx)
        nan_i = kins.index[kins.isnull()]
        if norm == 1:
            kins = kins.dropna()
            kins = normalize_1(kins)
            tx = time_vector.drop(nan_i)
        label = str(round(wale)) + ' nm'
        trac[label] = kins
        txs[label] = tx
    fig, ax = plt.subplots(figsize=(10,8))
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
    ax.set_xlim(tlims)
    if ylims == 0: 
        pass
    else:
        ax.set_ylim(ylims)
    ax.set_xlabel("Time, ps", fontsize = 20)
    ax.set_ylabel("Intensity, OD", fontsize = 20)
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
    ax.axhline(y=0, color = 'black', ls = '--')
    for i in range(len(wales_chosen)):
        yy = trac[trac.columns[i]]
        tx = txs[txs.columns[i]]
        ax.scatter(tx, yy, s = 40, marker = "D", label = trac.columns[i])
    ax.legend(loc = 'upper right', fontsize = 18)
    if save == 1:
        plt.savefig(str(np.random.randint(1000))+'MultipleTracesFig.svg')

def FitSingleKinetic(time_vector, wale_vector, wl, data, exp_no, initPar, bounds, tlims, ylims, logscale, save):
    if bounds == 0:
        bounds = (-np.inf, np.inf)
    x = time_vector
    y = data.iloc[pd.Index(wale_vector).get_loc(wl)]
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
    if tlims != 0:
        ax.set_xlim(tlims)
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
    else:
        ax.set_ylim(tlims)
    ax.set_xlabel("Time, ps", fontsize = 20)
    ax.set_ylabel("Intensity, OD", fontsize = 20)
    if ylims == 0: 
        pass
    else:
        ax.set_ylim(ylims)
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
    ax.axhline(y=0, color = 'black', ls = '--')
    if logscale == 1:
        zr = next(x for x, val in enumerate(time_vector) if val > 0)
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
    
    ax.legend(loc = 'upper right', fontsize = 18)

    # print(fittedParameters)
    # print(np.sqrt(np.diag(pcov)))
    dummy = []
    for i in range(len(pars) - (2+exp_no)):
        dummy.append('0')
    if exp_no == 1:
        dummy.append('A1%:' + str(amp1/(amp1+amp0)))
        dummy.append('A0%:' + str(amp0/(amp1+amp0)))
    elif exp_no == 2:
        dummy.append('A1 is rise!')
        dummy.append('A2%:' + str(amp2/(amp2+amp0)))
        dummy.append('A0%:' + str(amp0/(amp2+amp0)))
    elif exp_no == 3:
        dummy.append('A1 is rise!')
        dummy.append('A2%:' + str(amp2/(amp2+amp3+amp0)))
        dummy.append('A3%:' + str(amp3/(amp2+amp3+amp0)))
        dummy.append('A0%:' + str(amp0/(amp2+amp3+amp0)))
    elif exp_no == 4:
        dummy.append('A1 is rise!')
        dummy.append('A2%:' + str(amp2/(amp2+amp3+amp4+amp0)))
        dummy.append('A3%:' + str(amp3/(amp2+amp3+amp4+amp0)))
        dummy.append('A4%:' + str(amp4/(amp2+amp3+amp4+amp0)))
        dummy.append('A0%:' + str(amp0/(amp2+amp3+amp4+amp0)))
    dummy.append(str(Rsquared))
    res = pd.DataFrame(
    {'parameter': pars,
     'value': list(fittedParameters),
     'sigma': list(np.sqrt(np.diag(pcov))),
     'R2': dummy,
    })
    if save == 1:
        plt.savefig(str(np.round(wl))+'SingleKineticFit'+str(np.random.randint(1000))+'.svg')
        res.to_csv(str(np.round(wl))+'fit.csv', index = False)
    return res

def SaveATrace(wale_vector, wale_chosen, time_vector, data, norm):
    i = pd.Index(wale_vector).get_loc(wale_chosen)
    kins = data.iloc[i]
    nan_i = kins.index[kins.isnull()]
    if norm == 1:
        kins = kins.dropna()
        kins = normalize_1(kins)
        time_vector = time_vector.drop(nan_i)
    df = pd.DataFrame({
        'time': time_vector,
        'int': kins,
        })
    return df
def saver(dictex):
    for key, val in dictex.items():
        val.to_csv("fits_750_{}.csv".format(str(key)))

    with open("keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))
#%% PRE-PROCESSING
dsets = []
labels = []
def readSolsticeData(file):
    with open(file) as current_file:
        data = pd.read_csv(current_file, header = None, delim_whitespace = True, engine = 'python')
        return data

for i in range(len(os.listdir())):
    if os.listdir()[i].endswith('.dat'):
        dsets.append(readSolsticeData(os.listdir()[i]))
        labels.append(str(os.listdir()[i]))


#%% check alignment
fig, ax = plt.subplots(figsize=(10,8)) 
ax.set_xlim(-1, 10)
# ax.set_ylim(-0.01, 0.02)
a = 10
b = 20
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, 10)])
for dset in dsets[a:b]:
    ax.plot(dset.iloc[:,0], dset.iloc[:,1575])
# ax.plot(avg.iloc[:,0], avg.iloc[:, 1550])

# labels.append('average')
ax.legend(loc = 'upper right', labels = labels[a:b], ncol = 1)
#%%
avg = pd.concat(dsets).mean(level=0)
del [dsets]
zero_index = 20
pre_zero = np.mean(avg.iloc[0:zero_index, 1:])
cor = avg.copy()
cor = cor.sub(pre_zero.squeeze(), axis = 1)
cor.iloc[:,0] = avg.iloc[:,0]
wale = np.linspace(469.5, 740.6, 1600)

nu1kff_1octanol_3mw_halfOD = cor.copy()
#%% READ THE DATA IN
# Enter the filename
rawdata = nu1kff_1octanol_3mw_halfOD.copy()
# Get rid of the useless information rows 
rawdata = rawdata[rawdata.applymap(CheckFloat).all(axis=1)].astype(float)
# Extract data matrix, wl and time
time = rawdata.iloc[:,0]
# wale = wale
data = np.transpose(rawdata.iloc[0:,1:])
# data = flip_sign(data)
# This assumes you have wale on y-axis and time on x-axis.
# the TAPlot function will transpose your data to make it work. 
#%% Plot the Surface
# recommended lvl: 15 for obvious contours, 50+ for smooth
# recommended cmaps: magma, plasma, inferno, viridis, cividis
# args: wl vec., t vec., data matrix, xlim (0 for full), contour lvl, linlog, cmap, save
# 
TAPlot(wale[160:], time, data.iloc[160:, :], [497, 740.6], [-0.5, 1.5], 15, 1, 'magma', 0)

#%% Plot the Time Slices
# Arguments: time vector, time point index in the format time.loc(axis=0)[INDEX]
# wavelength vector, data matrix, save 
# PlotSingleSlice(time, time.loc(axis=0)[40], wale, data,0)
PlotMultipleSlices(time, time.loc(axis=0)[33, 50, 90, 120, 320, 350, 400, 425, 500, 550, 600, 625], wale[160:], data.iloc[160:,:], [497, 740.6], [-0.007, 0.008])
# PlotMultipleSlices(time, time.loc(axis=0)[60, 83, 93, 100, 120, 140], wale, [440, 799], data,0)



#%% Plot the Kinetics
# PlotSingleTrace(wale, wale.loc(axis=0)[25], time, data, [-50, 3000], [-1,1],1,0)
# norm_codmf_se = SaveATrace(wale, wale.loc(axis=0)[25], time, data, 1)
# wale_vector, wales_chosen, time_vector, data, tlims, ylims, norm, save
PlotMultipleTraces(wale, wale[[350, 950, 1150, 1590]], time, data, [-1,1500],0, 0,0)
#%% Fit the Kinetics

# Define the functions; edit if you know what you're doing
# By default these are standard exponential decay sums convoluted with IRF
# These functions will be called by FitSingleKinetic which currently accepts 3 exp max
# but should be very easy to edit
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
#%%
par_bounds = ([-np.inf, 0.1, 0, -np.inf, 0 ,0.5, 0, 25, -np.inf, 1000, 0],
              [0,  0.9, 25,  np.inf,  np.inf, 1100, np.inf, 3000, np.inf, 250000, np.inf])
initPar = [0.007, 0.27, 1.8, 0.55, 0.003, 10, 0.003, 480, 0]
fit_os = FitSingleKinetic(time_Os, wale_Os, wale_Os.loc(axis=0)[35], data_Os, \
                        3, initPar, 0,\
                        [0.1, 3000], [-0.005, 0.005], \
                        1,0)
# time vector, wavelength vector, chosen wavelength by index, data matrix, 
    # number of exponentials, initial parameters, Xlimits, Ylimits, 
    # normalization (1 = yes, 0 = no), save results (1 = yes, 0 = no)
# pass wl as wale.loc(axis=0)[index]

#%% Convert Files For Optimus
# Enter the filename
for file in os.listdir():  
    headr = '%FILENAME=\n%DATATYPE=TAVIS\n%NUMBERSCANS=1\n%TIMESCALE=ps\n%TIMELIST=\n%WAVELENGTHLIST=\n%INTENSITYMATRIX='
    filename = str(file)
    dum = [0,0]
    rawdata = read_file(file)
    rawdata = rawdata[rawdata.applymap(CheckFloat).all(axis=1)].astype(float)
    time = rawdata.iloc[0][1:]
    time = np.transpose(time)
    wale = rawdata[0][1:]
    data = np.transpose(rawdata.iloc[1:,1:])
    np.savetxt(str(file)[:5]+'TIME', np.matrix(time.astype(float)), delimiter = ' ', fmt = '%.2f')
    np.savetxt(str(file)[:5]+'WALE', np.matrix(wale.astype(float)), delimiter = ' ', fmt = '%.2f')
    np.savetxt(str(file)[:5]+'INTE', data.astype(float), delimiter = ' ', fmt = '%.8f')
    np.savetxt(str(file)[:5]+'final.ana', dum, header=headr)

#%% Convert Files for Glotaran!
filename = 'co1c_Cf3t_clean.csv'
# def GlotaranConversion(filename):
rawdata = read_file(filename)
# Get rid of the useless information rows 
rawdata = rawdata[rawdata.applymap(CheckFloat).all(axis=1)].astype(float)
# Extract data matrix, wl and time
time = rawdata.iloc[0][10:]
wale = rawdata[0][1:]
data = rawdata.iloc[1:,10:]
file1 = open('co1c_CC_glotaran.ascii', 'a')
file1.write("# This is the "+filename+" converted to Glotaran Format\n")
file1.write("\n")
file1.write("Time explicit\n")
file1.write("Intervalnr "+str(len(time))+"\n")
for item0 in time:
    file1.write(str(item0)+" ")
file1.write("\n")
for item1 in wale:
    idx = wale[wale == item1].index.values[0]
    file1.write(str(item1)+" ")
    for item2 in data.iloc[idx-1]:
        file1.write(str(item2)+" ")
    file1.write("\n")
file1.close()
# GlotaranConversion(filename)
        
#%%
i = 11
co1c_cf3t_trace = data_Cocf3t.iloc[pd.Index(wale_Cocf3t).get_loc(wale_Cocf3t.loc(axis=0)[i])]
co1c_dmf_trace = data_codmf.iloc[pd.Index(wale_codmf).get_loc(wale_codmf.loc(axis=0)[i])]
# ni1c_dmf_580nm = data.iloc[pd.Index(wale).get_loc(wale.loc(axis=0)[122])]
ni1c_dmf_trace = data_NiDMF.iloc[pd.Index(wale_NiDMF).get_loc(wale_NiDMF.loc(axis=0)[i])]
ni1c_cf3t_trace = data_Nicf3t.iloc[pd.Index(wale_Nicf3t).get_loc(wale_Nicf3t.loc(axis=0)[i])]
ti1c_cf3t_trace = data_Ticf3t.iloc[pd.Index(wale_Ticf3t).get_loc(wale_Ticf3t.loc(axis=0)[i])]
bare_trace = data.iloc[pd.Index(wale).get_loc(wale.loc(axis=0)[i])]
zn1c_dmf_trace = data_zndmf.iloc[pd.Index(wale_zndmf).get_loc(wale_zndmf.loc(axis=0)[i])]
#%%
i = 32
co1c_cf3t_trace_2 = data_Cocf3t.iloc[pd.Index(wale_Cocf3t).get_loc(wale_Cocf3t.loc(axis=0)[i])]
co1c_dmf_trace_2 = data_codmf.iloc[pd.Index(wale_codmf).get_loc(wale_codmf.loc(axis=0)[i])]
# ni1c_dmf_580nm = data.iloc[pd.Index(wale).get_loc(wale.loc(axis=0)[122])]
ni1c_dmf_trace_2 = data_NiDMF.iloc[pd.Index(wale_NiDMF).get_loc(wale_NiDMF.loc(axis=0)[i])]
ni1c_cf3t_trace_2 = data_Nicf3t.iloc[pd.Index(wale_Nicf3t).get_loc(wale_Nicf3t.loc(axis=0)[i])]
ti1c_cf3t_trace_2 = data_Ticf3t.iloc[pd.Index(wale_Ticf3t).get_loc(wale_Ticf3t.loc(axis=0)[i])]
bare_trace_2 = data.iloc[pd.Index(wale).get_loc(wale.loc(axis=0)[i])]
# zn1c_dmf_trace = data_zndmf.iloc[pd.Index(wale_zndmf).get_loc(wale_zndmf.loc(axis=0)[i])]
#%%
i = 145
co1c_cf3t_trace_3 = data_Cocf3t.iloc[pd.Index(wale_Cocf3t).get_loc(wale_Cocf3t.loc(axis=0)[i])]
co1c_dmf_trace_3 = data_codmf.iloc[pd.Index(wale_codmf).get_loc(wale_codmf.loc(axis=0)[i])]
# ni1c_dmf_580nm = data.iloc[pd.Index(wale).get_loc(wale.loc(axis=0)[122])]
ni1c_dmf_trace_3 = data_NiDMF.iloc[pd.Index(wale_NiDMF).get_loc(wale_NiDMF.loc(axis=0)[i])]
ni1c_cf3t_trace_3 = data_Nicf3t.iloc[pd.Index(wale_Nicf3t).get_loc(wale_Nicf3t.loc(axis=0)[i])]
ti1c_cf3t_trace_3 = data_Ticf3t.iloc[pd.Index(wale_Ticf3t).get_loc(wale_Ticf3t.loc(axis=0)[i])]
bare_trace_3 = data.iloc[pd.Index(wale).get_loc(wale.loc(axis=0)[i])]
# zn1c_dmf_trace = data_zndmf.iloc[pd.Index(wale_zndmf).get_loc(wale_zndmf.loc(axis=0)[i])]
#%%
i = 264
co1c_cf3t_trace_4 = data_Cocf3t.iloc[pd.Index(wale_Cocf3t).get_loc(wale_Cocf3t.loc(axis=0)[i])]
co1c_dmf_trace_4 = data_codmf.iloc[pd.Index(wale_codmf).get_loc(wale_codmf.loc(axis=0)[i])]
# ni1c_dmf_580nm = data.iloc[pd.Index(wale).get_loc(wale.loc(axis=0)[122])]
ni1c_dmf_trace_4 = data_NiDMF.iloc[pd.Index(wale_NiDMF).get_loc(wale_NiDMF.loc(axis=0)[i])]
ni1c_cf3t_trace_4 = data_Nicf3t.iloc[pd.Index(wale_Nicf3t).get_loc(wale_Nicf3t.loc(axis=0)[i])]
ti1c_cf3t_trace_4 = data_Ticf3t.iloc[pd.Index(wale_Ticf3t).get_loc(wale_Ticf3t.loc(axis=0)[i])]
bare_trace_4 = data.iloc[pd.Index(wale).get_loc(wale.loc(axis=0)[i])]
# zn1c_dmf_trace = data_zndmf.iloc[pd.Index(wale_zndmf).get_loc(wale_zndmf.loc(axis=0)[i])]
#%%
# tlims = [0, 2500]
# ylims = [0.05, -1.1]
fig, ax = plt.subplots(figsize=(10,8))
# if max(tlims) > 100 and max(tlims) <= 500:
#     ax.xaxis.set_major_locator(MultipleLocator(50))
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#     ax.xaxis.set_minor_locator(MultipleLocator(10))
# if max(tlims) > 10 and max(tlims) <= 100:
#     ax.xaxis.set_major_locator(MultipleLocator(10))
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#     ax.xaxis.set_minor_locator(MultipleLocator(1))
# if max(tlims) <= 10:
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# if max(tlims) > 500 and max(tlims) <= 1000:
#     ax.xaxis.set_major_locator(MultipleLocator(100))
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#     ax.xaxis.set_minor_locator(MultipleLocator(20))
# if max(tlims) > 1000:
#     ax.xaxis.set_major_locator(MultipleLocator(500))
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#     ax.xaxis.set_minor_locator(MultipleLocator(100))
# ax.set_xlim(tlims)
ax.set_xlim([10**-1, 10**3.3])
ax.set_ylim([-0.005, 0.01])
# ax.set_ylim([-0.0005, 0.008])
ax.set_xlabel("Time, ps", fontsize = 24)
ax.set_ylabel(r"Intensity, $\Delta$OD", fontsize = 24)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.axhline(y=0, color = 'black', ls = '--')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Not Smoothed - Line
# ax.plot(time_zndmf[2:], normalize_1(zn1c_dmf_trace[1:]), color = 'lightcoral', linewidth = 3, label = 'Zn-SIM')
# ax.plot(time_Ticf3t[1:], normalize_1(ti1c_cf3t_trace[1:]), color = 'slategray', linewidth = 3, label = 'Ti-SIM')
# ax.plot(time_NiDMF[1:], normalize_1(ni1c_dmf_trace[1:]), color = 'teal', linewidth = 3, label = 'Ni-SIM')
# # ax.plot(time_NiDMF[2:], normalize_1(ni1c_dmf_trace[2:]), color = 'cyan', linewidth = 3, label = 'Ni-SIM in DMF, 750 nm')
# ax.plot(time_Cocf3t[1:], normalize_1(co1c_cf3t_trace), color = 'darkmagenta', linewidth = 3, label = 'Co-SIM')
# ax.plot(time[1:],normalize_1(bare_trace), color='goldenrod',linewidth=3,label='NDC-NU-1000')

# Not Smoothed - Scatter
# ax.scatter(time_zndmf[1:], normalize_1(zn1c_dmf_trace[1:]), color = 'lightcoral', s = 25,marker = 'D', label = 'Zn-SIM')
# ax.scatter(time_Ticf3t[1:], normalize_1(ti1c_cf3t_trace[1:]), color = 'slategray', s = 25,marker = 'o', label = 'Ti-SIM')
# ax.scatter(time_NiDMF[1:], normalize_1(ni1c_dmf_trace[1:]), color = 'teal', s = 25,marker = 'v', label = 'Ni-SIM')
# ax.scatter(time_Cocf3t[1:], normalize_1(co1c_cf3t_trace), color = 'purple', s = 25,marker = '^', label = 'Co-SIM')
# ax.scatter(time[1:],normalize_1(bare_trace), color='darkgoldenrod',s = 25, marker = 's',label='NU-1000-NDC')

# Smoothed with Sav-Gol Filter - Line
# SGw = 5
# SGs = 2
# # ax.plot(time_zndmf[2:], savgol_filter(normalize_1(zn1c_dmf_trace[1:]),SGw, SGs), color = 'steelblue', linewidth = 3, label = 'Zn-SIM')
# ax.plot(time[2:], savgol_filter(normalize_1(bare_trace),SGw, SGs), color='goldenrod',linewidth=3,label='NDC-NU-1000')
# ax.plot(time_Ticf3t[1:], savgol_filter(normalize_1(ti1c_cf3t_trace[1:]),SGw, SGs), color = 'slategray', linewidth = 3, label = 'Ti-SIM')
# ax.plot(time_Nicf3t[2:], savgol_filter(normalize_1(ni1c_cf3t_trace[1:]),SGw, SGs), color = 'teal', linewidth = 3, label = 'Ni-SIM')
# # ax.plot(time_NiDMF[2:], normalize_1(ni1c_dmf_trace[2:]), color = 'cyan', linewidth = 3, label = 'Ni-SIM in DMF, 750 nm')
# ax.plot(time_Cocf3t[1:]-1.5, savgol_filter(normalize_1(co1c_cf3t_trace),SGw, SGs), color = 'darkmagenta', linewidth = 3, label = 'Co-SIM')


# Smoothed with Sav-Gol Filter - Scatter

# ax.scatter(time[66:],bare_trace[66:],color='navy',s = 25, marker = 's',label='NU-1000-NDC, 450 nm',facecolors = 'none',linewidth = 1.5)
# ax.scatter(time[66:],bare_trace_2[66:],color='royalblue',s = 25, marker = 's',label='NU-1000-NDC, 475 nm',facecolors = 'none',linewidth = 1.5)
ax.scatter(time[66:],bare_trace_3[66:],color='goldenrod',s = 25, marker = 's',label='NU-1000-NDC, 600 nm',facecolors = 'none',linewidth = 1.5)
# ax.scatter(time[66:],bare_trace_4[66:],color='darkorange',s = 25, marker = 's',label='NU-1000-NDC, 750 nm',facecolors = 'none',linewidth = 1.5)
# ax.plot(time[66:], TA_2exp(time[66:], *fit_nu1k_450['value']), color = 'black', linewidth = 2)
# ax.plot(time[66:], TA_2exp(time[66:], *fit_nu1k_475['value']), color='black',linewidth=2)
ax.plot(time[66:], TA_4exp(time[66:], *fit_nu1k['value']), color='black',linewidth=2)
# ax.plot(time[66:], TA_3exp(time[66:], *fit_nu1k_750['value']), color='black',linewidth=2)

# ax.scatter(time_Ticf3t[80:], ti1c_cf3t_trace[80:], color = 'navy', s = 25,marker = 'o',facecolors = 'none',linewidth = 1.5, label = 'Ti-SIM, 450 nm')
# ax.scatter(time_Ticf3t[80:], ti1c_cf3t_trace_2[80:], color = 'royalblue', s = 25,marker = 'o',facecolors = 'none',linewidth = 1.5, label = 'Ti-SIM, 475 nm')
ax.scatter(time_Ticf3t[80:], ti1c_cf3t_trace_3[80:], color = 'slategrey', s = 25,marker = 'o',facecolors = 'none',linewidth = 1.5, label = 'Ti-SIM, 600 nm')
# ax.scatter(time_Ticf3t[80:], ti1c_cf3t_trace_4[80:], color = 'darkorange', s = 25,marker = 'o',facecolors = 'none',linewidth = 1.5, label = 'Ti-SIM, 750 nm')
# ax.plot(time_Ticf3t[80:], TA_3exp(time_Ticf3t[80:], *fit_tisim_450['value']), color = 'black', linewidth = 2)
# ax.plot(time_Ticf3t[80:], TA_3exp(time_Ticf3t[80:], *fit_tisim_475['value']), color='black',linewidth=2)
ax.plot(time_Ticf3t[80:], TA_4exp(time_Ticf3t[80:], *fit_tisim_600['value']), color='black',linewidth=2)
# ax.plot(time_Ticf3t[80:], TA_3exp(time_Ticf3t[80:], *fit_tisim_750['value']), color='black',linewidth=2)

# ax.scatter(time_Nicf3t[78:], ni1c_cf3t_trace[78:], color = 'navy', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Ni-SIM, 450 nm')
# ax.scatter(time_Nicf3t[78:], ni1c_cf3t_trace_2[78:], color = 'royalblue', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Ni-SIM, 475 nm')
ax.scatter(time_Nicf3t[78:], ni1c_cf3t_trace_3[78:], color = 'teal', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Ni-SIM, 600 nm')
# ax.scatter(time_Nicf3t[78:], ni1c_cf3t_trace_4[78:], color = 'darkorange', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Ni-SIM, 750 nm')
# ax.plot(time_Nicf3t[78:], TA_3exp(time_Nicf3t[78:], *fit_nisim_tft_450['value']), color='black',linewidth=2)
# ax.plot(time_Nicf3t[78:], TA_3exp(time_Nicf3t[78:], *fit_nisim_tft_475['value']), color='black',linewidth=2)
ax.plot(time_Nicf3t[78:], TA_3exp(time_Nicf3t[78:], *fit_Nicf3t['value']), color='black',linewidth=2)
# ax.plot(time_Nicf3t[78:], TA_3exp(time_Nicf3t[78:], *fit_nisim_tft_750['value']), color='black',linewidth=2)

# ax.scatter(time_NiDMF[60:], ni1c_dmf_trace[60:], color = 'navy', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Ni-SIM, 450 nm')
# ax.scatter(time_NiDMF[60:], ni1c_dmf_trace_2[60:], color = 'royalblue', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Ni-SIM, 475 nm')
# ax.scatter(time_NiDMF[60:], ni1c_dmf_trace_3[60:], color = 'darkorchid', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Ni-SIM, 600 nm')
# ax.scatter(time_NiDMF[60:], ni1c_dmf_trace_4[60:], color = 'darkorange', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Ni-SIM, 750 nm')
# ax.plot(time_NiDMF[60:], TA_3exp(time_NiDMF[60:], *fit_nisim_dmf_450['value']), color='black',linewidth=2)
# ax.plot(time_NiDMF[60:], TA_3exp(time_NiDMF[60:], *fit_nisim_dmf_475['value']), color='black',linewidth=2)
# ax.plot(time_NiDMF[60:], TA_3exp(time_NiDMF[60:], *fit_nisim_dmf_600['value']), color='black',linewidth=2)
# ax.plot(time_NiDMF[60:], TA_3exp(time_NiDMF[60:], *fit_nisim_dmf_750['value']), color='black',linewidth=2)


# ax.scatter(time_Cocf3t[80:], co1c_cf3t_trace[80:], color = 'indigo', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Co-SIM, 450 nm')
# ax.scatter(time_Cocf3t[80:], co1c_cf3t_trace_2[80:], color = 'royalblue', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Co-SIM, 475 nm')
ax.scatter(time_Cocf3t[80:], co1c_cf3t_trace_3[80:], color = 'darkmagenta', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Co-SIM, 600 nm')
# ax.scatter(time_Cocf3t[80:], co1c_cf3t_trace_4[80:], color = 'darkorange', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Co-SIM, 750 nm')
# ax.plot(time_Cocf3t[80:], TA_3exp(time_Cocf3t[80:], *fit_cosim_tft_450['value']), color='black',linewidth=2)
# ax.plot(time_Cocf3t[80:], TA_3exp(time_Cocf3t[80:], *fit_cosim_tft_475['value']), color='black',linewidth=2)
ax.plot(time_Cocf3t[80:], TA_4exp(time_Cocf3t[80:], *fit_Cocf3t_2['value']), color='black',linewidth=2)
# ax.plot(time_Cocf3t[80:], TA_3exp(time_Cocf3t[80:], *fit_cosim_tft_750['value']), color='black',linewidth=2)

# ax.scatter(time_codmf[80:], co1c_cf3t_trace[80:], color = 'navy', s = 25,marker = 'v',facecolors = 'none',linewidth = 1.5, label = 'Co-SIM, 450 nm')



# ax.scatter(time_codmf[83:], co1c_dmf_trace[83:], color = 'navy', s = 25,marker = '^',facecolors = 'none',linewidth = 1.5, label = 'Co-SIM, 450 nm')
# ax.scatter(time_codmf[83:], co1c_dmf_trace_2[83:], color = 'royalblue', s = 25,marker = '^',facecolors = 'none',linewidth = 1.5, label = 'Co-SIM, 475 nm')
# ax.scatter(time_codmf[83:], co1c_dmf_trace_3[83:], color = 'darkorchid', s = 25,marker = '^',facecolors = 'none',linewidth = 1.5, label = 'Co-SIM, 600 nm')
# ax.scatter(time_codmf[83:], co1c_dmf_trace_4[83:], color = 'darkorange', s = 25,marker = '^',facecolors = 'none',linewidth = 1.5, label = 'Co-SIM, 750 nm')
# ax.plot(time_codmf[83:], TA_3exp(time_codmf[83:], *fit_cosim_dmf_450['value']), color='black',linewidth=2)
# ax.plot(time_codmf[83:], TA_3exp(time_codmf[83:], *fit_cosim_dmf_475['value']), color='black',linewidth=2)
# ax.plot(time_codmf[83:], TA_3exp(time_codmf[83:], *fit_cosim_dmf_600['value']), color='black',linewidth=2)
# ax.plot(time_codmf[83:], TA_3exp(time_codmf[83:], *fit_cosim_dmf_750['value']), color='black',linewidth=2)


ax.set_xscale('log')
ax.legend(loc = 'lower right', fontsize = 24)
# plt.savefig('FigureX_750nmkinetics_allsamples_raw_lim50.svg')


#%%
# dset = data

# wlims = [440,500]
# fig, ax = plt.subplots(figsize=(11,8))
# ax.set_xlim(wlims)
# ax.xaxis.set_major_locator(MultipleLocator(50))
# ax.xaxis.set_minor_locator(MultipleLocator(10))
# ax.axhline(y=0, color = 'dimgrey', ls = '--')
# ax.set_ylim([0, 10])
# ax.set_xlabel("Wavelength (nm)", fontsize = 20)
# ax.set_ylabel("Counts", fontsize = 20)
# ax.tick_params(axis='x', labelsize= 18)
# ax.tick_params(axis='y', labelsize= 18)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# n = 15
# ax.set_prop_cycle('color',[plt.cm.magma(i) for i in np.linspace(0, 1, n)])
# for i in range(0,240,16):
#     ax.plot(wale, dset.iloc[i,:])
#     # ax.plot(dset[3]['spec2']['wl'][-1::-1], dset[0]['streak2'].iloc[i,:])
# # sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=0, vmax=1000))
# sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=0, vmax=60))
# cbar = plt.colorbar(sm, pad = 0.01)
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Time (ps)', fontsize = 18)

def BlueSlices(wale_vector, data, ylim=0, xlim=0):
    dset = data.copy()
    fig, ax = plt.subplots(figsize=(11,8))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.set_xlim([440,525])
    ax.set_xlabel("Wavelength, nm", fontsize = 24)
    ax.set_ylabel("Intensity, OD", fontsize = 24)
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
    if xlim != 0:
        ax.set_xlim(xlim)
    if ylim != 0:
        ax.set_ylim(ylim)
    ax.axhline(y=0, color = 'black', ls = '--')
    n = 34
    ax.set_prop_cycle('color',[plt.cm.magma(i) for i in np.linspace(0, 1, n)])
    for i in range(91, 329, 7):
        ax.plot(wale_vector, dset.iloc[:,i], linewidth = 3)
    sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=1, vmax=3000))
    cbar = plt.colorbar(sm, pad = 0.01)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Time (ps)', fontsize = 24)
    ax.axvline(x=450, color = 'indigo', ls = '-.', linewidth = 4, alpha = 0.75)
    ax.axvline(x=475, color = 'blue', ls = '-.', linewidth = 4, alpha = 0.75)
    # ax.legend(loc = 'lower right', fontsize = 18)
    
BlueSlices(wale_Cocf3t, data_Cocf3t, [ -0.015, 0.001])

#%% global analysis WORKS
# def GlobalParallelAnalysis(time_vector, wale_vector, wales_chosen, intensity_matrix, exp_no):
#     kins = []
#     for wavelength in wales_chosen:
#         i = pd.Index(wale).get_loc(wavelength)
#         kinetic = np.transpose(intensity_matrix).iloc[i]
#         kins.append(kinetic)
#     dictionary = {}
#     def multiexp_function(params):
#         func = params['offset'] + [params[f'amp1{i}']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params[f'tau{i}']))**2-(x-params['timezero'])/params[f'tau{i}'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params[f'tau{i}'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) for i in range(exp_no)]
#         return func
#     for x in range(1,len(kins)+1):
#         dictionary["trace{0}".format(x)] = 


from lmfit import minimize, Parameters, fit_report
import numpy as np

# residual function to minimize
def fit_function_3exp_2traces(params, x=None, dat1=None, dat2=None):
    trace1 = params['amp11']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau1']))**2-(x-params['timezero'])/params['tau1'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau1'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
        + params['amp12']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau2']))**2-(x-params['timezero'])/params['tau2'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau2'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
            + params['amp13']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau3']))**2-(x-params['timezero'])/params['tau3'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau3'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
                + params['offset']
    trace2 = params['amp21']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau1']))**2-(x-params['timezero'])/params['tau1'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau1'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
        + params['amp22']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau2']))**2-(x-params['timezero'])/params['tau2'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau2'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
            + params['amp23']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau3']))**2-(x-params['timezero'])/params['tau3'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau3'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
                + params['offset']
    # trace3 = params['amp31']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau1']))**2-(x-params['timezero'])/params['tau1'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau1'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
    #     + params['amp32']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau2']))**2-(x-params['timezero'])/params['tau2'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau2'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
    #         + params['amp33']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau3']))**2-(x-params['timezero'])/params['tau3'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau3'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
    #             + params['offset']   
    resid1 = dat1 - trace1
    resid2 = dat2 - trace2
    # resid3 = dat3 - trace3
    return np.concatenate((resid1, resid2))
def fit_function_2exp_2traces(params, x=None, dat1=None, dat2=None):
    trace1 = params['amp11']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau1']))**2-(x-params['timezero'])/params['tau1'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau1'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
        + params['amp12']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau2']))**2-(x-params['timezero'])/params['tau2'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau2'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
            + params['offset']
    trace2 = params['amp21']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau1']))**2-(x-params['timezero'])/params['tau1'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau1'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
        + params['amp22']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau2']))**2-(x-params['timezero'])/params['tau2'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau2'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
            + params['offset']
    # trace3 = params['amp31']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau1']))**2-(x-params['timezero'])/params['tau1'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau1'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
    #     + params['amp32']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau2']))**2-(x-params['timezero'])/params['tau2'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau2'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
    #         + params['amp33']*params['fwhm']*np.exp((params['fwhm']/(np.sqrt(2)*params['tau3']))**2-(x-params['timezero'])/params['tau3'])*(1-special.erf(params['fwhm']/(np.sqrt(2)*params['tau3'])-(x-params['timezero'])/(np.sqrt(2)*params['fwhm']))) + \
    #             + params['offset']   
    resid1 = dat1 - trace1
    resid2 = dat2 - trace2
    # resid3 = dat3 - trace3
    return np.concatenate((resid1, resid2))

# setup fit parameters
params = Parameters()
params.add('amp11', value=-0.001)
params.add('amp12', value=-0.001)
params.add('amp13', value=-0.001)
params.add('amp21', value=0.001)
params.add('amp22', value=0.001)
params.add('amp23', value=0.001)
# params.add('amp31', value=0.001)
# params.add('amp32', value=0.001)
# params.add('amp33', value=0.001)
params.add('tau1', value=1)
params.add('tau2', value=10)
params.add('tau3', value=100)
params.add('fwhm', value = 0.1)
params.add('offset', value=0)
params.add('timezero', value=0)


# setup data files
wales_chosen = wale[[292, 1559]]
kins = []
for wavelength in wales_chosen:
    i = pd.Index(wale).get_loc(wavelength)
    kinetic = data.iloc[i]
    kins.append(kinetic)

x = time.copy()
y1 = kins[0]
y2 = kins[1]
# y3 = kins[2]

# fit
out = minimize(fit_function_3exp_2traces, params, kws={"x": x, "dat1": y1, "dat2": y2})
print(fit_report(out))
#%% global analysis WORKS
fig, ax = plt.subplots(figsize=(11,8))
# ax.xaxis.set_major_locator(MultipleLocator(10))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.set_xlim([-1,1400])
ax.set_xlabel("Time, ps", fontsize = 24)
ax.set_ylabel("Amplitude, mOD", fontsize = 24)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.axhline(y=0, color = 'black', ls = '--')
for i in range(len(kins)):
    ax.scatter(x, kins[i])
    ax.plot(x, TA_3exp(x, out.params[f'amp{i+1}1'],out.params['fwhm'],out.params['tau1'],out.params['timezero'],out.params[f'amp{i+1}2'],out.params['tau2'],out.params[f'amp{i+1}3'],out.params['tau3'],out.params['offset']), color='black', linewidth = 3)


# ax.legend(loc = 'lower right', fontsize = 18)

#%%
dictionary = {}
for x in range(10):
    dictionary["string{0}".format(x)] = "Hello"