# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:18:03 2021

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

#%%
def normalize_1(data):
    """Normalizes a vector to its maximum value"""
    normalized_data = np.abs(data)/max(data)
    return normalized_data
def readfile(file):
    with open(file) as current_file:
        d_tmp = pd.read_csv(current_file, sep = ',', header = None, engine = 'python', skiprows = range(2,16))
        d = pd.DataFrame()
        d['time'] = d_tmp.iloc[0,:]
        d['counts'] = d_tmp.iloc[1,:]
        d_trimmed = d.iloc[33000:, :].reset_index(drop=True)
        d_trimmed['time'] = d_trimmed['time']-d_trimmed['time'][np.argmax(d_trimmed['counts'])]
    return d_trimmed
def rebin(data, step):
    d_rebinned = pd.DataFrame(columns=['time', 'counts'])
    for i, j in zip(range(0, len(data)-step, step), range(step, len(data), step)):
        d_rebinned = d_rebinned.append({'time': np.average(data['time'].iloc[i:j]), 'counts':np.sum(data['counts'].iloc[i:j])}, ignore_index = True)
    return d_rebinned
def readset():
    data_files = os.listdir()
    dct = {}
    for file in data_files:
        dct[file] = readfile(file)
    return dct
def saver(dictex):
    for key, val in dictex.items():
        val.to_csv("fits_without_rise_{}.csv".format(str(key)))

    with open("keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))


#%%
fig, ax = plt.subplots(figsize=(12,8))

# ax.xaxis.set_major_locator(MultipleLocator(0.2))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax.xaxis.set_minor_locator(MultipleLocator(0.2))

ax.set_xlim([-100,1100])
# ax.set_xlim([10**-0.4, 10**3.1])
# ax.set_ylim([-0.005, 0.01])
# ax.set_ylim([-0.0005, 0.008])
ax.set_xlabel("Time (ns)", fontsize = 30)
ax.set_ylabel("Normalized Intensity", fontsize = 30)
ax.tick_params(axis='x', labelsize= 25)
ax.tick_params(axis='y', labelsize= 25)
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.axhline(y=0, color = 'dimgrey', ls = '--')
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# dset = irf_r
# ax.set_prop_cycle('color',[plt.cm.inferno(i) for i in np.linspace(0, 1, 5)])
ax.scatter(irf_r['time'], normalize_1(irf_r['counts']), color = 'slategray', facecolors = 'none', linewidth = 0.5, s = 80, label = "IRF")
ax.scatter(ruco_mof_r['time'], normalize_1(ruco_mof_r['counts']-np.average(ruco_mof_r['counts'].iloc[:100])), color = 'red', facecolors = 'none', linewidth = 2,s = 80, alpha = 0.15, marker = '^', label = "Rubpycyclooct@MIL-96")
# ax.scatter(RuCO_dye_r['time'], normalize_1(RuCO_dye_r['counts']), s = 40, label = "Ru-cyclooct dye")
# ax.scatter(RuDO_MOF_r['time'], normalize_1(RuDO_MOF_r['counts']), s = 40, label = "Ru-dodecyl MOF")
# ax.scatter(rudo_dye_r['time'], normalize_1(rudo_dye_r['counts']-np.average(rudo_dye_r['counts'].iloc[:100])), color = 'darkseagreen', facecolors = 'none', s = 40, marker = '^', label = "Rubpydodec, dilute")
# ax.scatter(rudo_dye_2r['time'], normalize_1(rudo_dye_2r['counts']-np.average(rudo_dye_2r['counts'].iloc[:100])), color = 'darkgreen', facecolors = 'none', s = 40, marker = 'd', label = "Rubpydodec, conc.")
ax.scatter(rudo_mof_r['time'], normalize_1(rudo_mof_r['counts']-np.average(rudo_mof_r['counts'].iloc[:100])), color = 'lawngreen', facecolors = 'none', linewidth = 2,s = 80,alpha = 0.75, label = "Rubpydodec@MIL-96")
ax.scatter(ruoc_mof_r['time'], normalize_1(ruoc_mof_r['counts']-np.average(ruoc_mof_r['counts'].iloc[:100])), color = 'blue', facecolors = 'none', linewidth = 2,s = 80,alpha = 0.75, marker = 'd', label = "Rubpyoct@MIL-96")
# ax.scatter(RuOC_dye_r['time'], normalize_1(RuOC_dye_r['counts']), s = 40, label = "Ru-octyl dye")

# ax.scatter(s9['time'], normalize_1(s9['counts']), s = 40, label = "DMSO")
# ax.scatter(s10['time'], normalize_1(s10['counts']), s = 40, label = "1mM TFA$_{DMSO}$")
# ax.scatter(s10a['time'], normalize_1(s10a['counts']), s = 40, label = "1M TFA$_{DMSO}$")
# ax.scatter(s11['time'], normalize_1(s11['counts']), s = 40, label = "0.1M LiNO$_{3, DMSO}$")
# ax.scatter(s12['time'], normalize_1(s12['counts']), s = 40, label = "1mM LiNO$_{3, DMSO}$")
# ax.scatter(s4['time'], normalize_1(s4['counts']), s = 40, label = "1mM HNO$_3")
# ax.scatter(s3['time'], normalize_1(s3['counts']), s = 40, label = "0.1M HNO$_3")
# ax.scatter(s2['time'], normalize_1(s2['counts']), s = 40, label = "1M HNO$_3$")
# ax.scatter(s5['time'], normalize_1(s5['counts']), s = 40, label = "10uM NaOH")
# ax.scatter(s6['time'], normalize_1(s6['counts']), s = 40, label = "0.1M LiNO$_3$")
# ax.scatter(s7['time'], normalize_1(s7['counts']), s = 40, label = "1mM LiNO$_3$")


# ax.plot(dset['440_MeTHF.csv']['time']-1659.4, dset['440_MeTHF.csv']['counts'])
# ax.semilogx(dset['440_MeTHF.csv']['time'], dset['440_MeTHF.csv']['counts'])
ax.legend(loc = 'upper right', fontsize = 25)
# ax.set_xscale('log')
# plt.savefig('FigureX_750nmkinetics_allsamples_raw_lim50.svg')

#%% Fitting without rise
def fitdecay_multipleplots(time_vector, int_vectors, exp_no, initPar, bounds, tlims, ylims, logscale):
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
    if max(tlims) > 500 and max(tlims) <= 2000:
        ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
    if max(tlims) > 2000:
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
    elif tlims == 0: 
        pass
    ax.set_xlim(tlims)
    ax.set_xlabel("Time, ps", fontsize = 24)
    ax.set_ylabel("Intensity (arb. units)", fontsize = 24)
    if ylims == 0: 
        pass
    else:
        ax.set_ylim(ylims)
    ax.tick_params(axis='x', labelsize= 20)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.axhline(y=0, color = 'black', ls = '--')
    for int_vector in int_vectors:
        if bounds == 0:
            bounds = (-np.inf, np.inf)
        time = time_vector
        x_full = time_vector
        y_full = int_vector
        nan_i = y_full.index[y_full.isnull()]
        y_full = y_full.dropna()
        x_full = x_full.drop(nan_i)
        # need to find max and fit decay only
        x = x_full[np.argmax(y_full):]
        y = y_full[np.argmax(y_full):]
        # xplt = np.linspace(min(x), max(x), 50)
        if exp_no == 1:
            fittedParameters, pcov = curve_fit(expdecay_1, x, y, initPar, bounds = bounds)
            amp1, tau1, t0, amp0 = fittedParameters
            yplt = expdecay_1(x, amp1, tau1, t0, amp0)
            pars = ['Amp1', 'Tau1', 'Time Zero', 'Offset']
        if exp_no == 2:
            fittedParameters, pcov = curve_fit(expdecay_2, x, y, initPar, bounds = bounds)
            amp1, tau1, amp2, tau2, t0, amp0 = fittedParameters
            yplt = expdecay_2(x, amp1,  tau1, amp2, tau2, t0, amp0)
            pars = ['Amp1', 'Tau1',  'Amp2', 'Tau2','Time Zero', 'Offset']
        if exp_no == 3:
            fittedParameters, pcov = curve_fit(expdecay_3, x, y, initPar, bounds = bounds)
            amp1, tau1, t0, amp2, tau2, amp3, tau3, amp0 = fittedParameters
            yplt = expdecay_3(x, amp1,tau1, t0, amp2, tau2, amp3, tau3, amp0)
            pars = ['Amp1', 'Tau1', 'Time Zero', 'Amp2', 'Tau2', 'Amp3', 'Tau3', 'Offset']
        resid = np.subtract(y, yplt)
        SE = np.square(resid) # squared errors
        MSE = np.mean(SE) # mean squared errors
        RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(resid) / np.var(y))
        if logscale == 1:
            zr = next(x for x, val in enumerate(time) if val > 0)
            x = x[zr:]
            y = y[zr:]
            yplt = yplt[zr:]
            ax.scatter(x_full,y_full, color = 'black', facecolors='none', s = 40, alpha = 0.4, marker = "D", label = 'Data')
            ax.plot(x,yplt, color = 'darkcyan', linewidth = 3, label = 'Fit')
            ax.scatter(x, np.subtract(y,yplt), color = 'magenta', s = 10, label = 'Residuals')
            ax.set_xscale('log')
            ax.legend(loc = 'lower left', fontsize = 24)
        else:
            ax.scatter(x_full,y_full, color = 'black', facecolors='none', s = 40, alpha = 0.4, marker = "D", label = 'Data')
            ax.plot(x,yplt, color = 'darkcyan', linewidth = 3, label = 'Fit')
            ax.scatter(x, np.subtract(y,yplt), color = 'magenta', s = 10, label = 'Residuals')
            ax.legend(loc = 'upper right', fontsize = 24)
    # dummy = []
    # for i in range(len(pars) - (2+exp_no)):
    #     dummy.append('0')
    # if exp_no == 1:
    #     dummy.append('A1%:' + str(amp1/(amp1+amp0)))
    #     dummy.append('A0%:' + str(amp0/(amp1+amp0)))
    # elif exp_no == 2:
    #     dummy.append('A1%:' + str(amp1/(amp1+amp2+amp0)))
    #     dummy.append('A2%:' + str(amp2/(amp1+amp2+amp0)))
    #     dummy.append('A0%:' + str(amp0/(amp1+amp2+amp0)))
    # elif exp_no == 3:
    #     dummy.append('A1%:' + str(amp1/(amp1+amp2+amp3+amp0)))
    #     dummy.append('A2%:' + str(amp2/(amp1+amp2+amp3+amp0)))
    #     dummy.append('A3%:' + str(amp3/(amp1+amp2+amp3+amp0)))
    #     dummy.append('A0%:' + str(amp0/(amp1+amp2+amp3+amp0)))
    # dummy.append(str(Rsquared))
    # res = pd.DataFrame(
    # {'parameter': pars,
    #  'value': list(fittedParameters),
    #  'sigma': list(np.sqrt(np.diag(pcov))),
    #  'R2': dummy,
    # })
    # return res

def expdecay_1(data, amp1, tau1, t0, amp0):
    return amp1*np.exp(-(data+t0)/tau1) + amp0
def expdecay_2(data, amp1, tau1, amp2, tau2, t0, amp0):
    return amp1*np.exp(-(data+t0)/tau1) + amp2*np.exp(-(data+t0)/tau2) + amp0
def expdecay_3(data, amp1, tau1, amp2, tau2, amp3, tau3, t0, amp0):
    return amp1*np.exp(-(data+t0)/tau1) + amp2*np.exp(-(data+t0)/tau2) + amp3*np.exp(-(data+t0)/tau3) +amp0
# bounds1 = ([0,   0 ,   -np.inf,    0],
#           [np.inf   ,np.inf   ,np.inf,np.inf])
bounds2 = ([0,  0  ,0   ,0,   -50,    0],
          [np.inf   ,1000   ,np.inf, np.inf ,100,0.00001])
initPar = [1000, 500, 1000,50, 0, 0]
var = ruco_mof_r.copy()
tlims = [-100, 1100]
# tlims = [10**2, 10**3.1]
final_fits_norise['Rubpycyclooct@MIL96'] = fitdecay(var['time'], var['counts']-np.average(var['counts'][0:10]), 2, initPar, bounds2, tlims, 0, 0)
#%%
def fitdecay_multipleplots(time_vector, int_vectors, exp_no, initPar, bounds, tlims, ylims, logscale):
    print(tlims)
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
    if max(tlims) > 500 and max(tlims) <= 2000:
        ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
    if max(tlims) > 2000:
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
    elif tlims == 0: 
        pass
    ax.set_xlim(tlims)
    ax.set_xlabel("Time, ps", fontsize = 24)
    ax.set_ylabel("Normalized Intensity (arb. units)", fontsize = 24)
    if ylims == 0: 
        pass
    else:
        ax.set_ylim(ylims)
    ax.tick_params(axis='x', labelsize= 20)
    ax.tick_params(axis='y', labelsize= 20)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    # ax.set_yticklabels([])
    # ax.set_yticks([])
    ax.axhline(y=0, color = 'black', ls = '--')
    ax.plot(irf_r['time'], normalize_1(irf_r['counts']), color = 'slategray', linewidth = 1,alpha = 0.75, label = "IRF")
    # ax.set_prop_cycle(color=['red','green','blue'], marker=['d', 'o', '^'])
    # ax.set_prop_cycle('marker', ['d', 'o', '^'])
    labels = ['Rubpydodec, dilute', 'Rubpydodec, conc.', 'Rubpydodec@MIL-96']
    colors = ['dodgerblue', 'orange', 'green', 'black', 'saddlebrown', 'darkgreen']
    markers = ['^', 'o', 'd']
    linestyles = ['-', '--', '-.']
    i = 0
    for int_vector in int_vectors:
        if bounds == 0:
            bounds = (-np.inf, np.inf)
        time = time_vector
        x_full = time_vector
        y_full = int_vector
        nan_i = y_full.index[y_full.isnull()]
        y_full = y_full.dropna()
        x_full = x_full.drop(nan_i)
        # need to find max and fit decay only
        x = x_full[np.argmax(y_full):]
        y = normalize_1(y_full[np.argmax(y_full):])
        # xplt = np.linspace(min(x), max(x), 50)
        if exp_no == 1:
            fittedParameters, pcov = curve_fit(expdecay_1, x, y, initPar, bounds = bounds)
            amp1, tau1, t0, amp0 = fittedParameters
            yplt = expdecay_1(x, amp1, tau1, t0, amp0)
            pars = ['Amp1', 'Tau1', 'Time Zero', 'Offset']
        if exp_no == 2:
            fittedParameters, pcov = curve_fit(expdecay_2, x, y, initPar, bounds = bounds)
            amp1, tau1, amp2, tau2, t0, amp0 = fittedParameters
            yplt = expdecay_2(x, amp1,  tau1, amp2, tau2, t0, amp0)
            pars = ['Amp1', 'Tau1',  'Amp2', 'Tau2','Time Zero', 'Offset']
        if exp_no == 3:
            fittedParameters, pcov = curve_fit(expdecay_3, x, y, initPar, bounds = bounds)
            amp1, tau1, t0, amp2, tau2, amp3, tau3, amp0 = fittedParameters
            yplt = expdecay_3(x, amp1,tau1, t0, amp2, tau2, amp3, tau3, amp0)
            pars = ['Amp1', 'Tau1', 'Time Zero', 'Amp2', 'Tau2', 'Amp3', 'Tau3', 'Offset']
        resid = np.subtract(y, yplt)
        SE = np.square(resid) # squared errors
        MSE = np.mean(SE) # mean squared errors
        RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(resid) / np.var(y))
        if logscale == 1:
            zr = next(x for x, val in enumerate(time) if val > 0)
            x_full = x_full[zr:]
            y_full = y_full[zr:]
            yplt = yplt[zr:]
            x = x[zr:]
            ax.scatter(x_full,normalize_1(y_full),s=40, color = colors[i], marker=markers[i], facecolors = 'none', alpha = 0.3)
            ax.plot(x,yplt,color = colors[i+3],linewidth = 3, label = labels[i])
            ax.set_xscale('log')
            i+=1
        else:
            # ax.scatter(x_full,normalize_1(y_full),s=120, color = colors[i], marker=markers[i], facecolors = 'none', alpha = 0.75)
            ax.plot(x_full+3,normalize_1(y_full), color = colors[i],alpha = 0.3)
            ax.plot(x+3,yplt,color = colors[i+3],linewidth = 5, linestyle = linestyles[i], label = labels[i])
            i+=1
    ax.legend(loc = 'upper right', fontsize = 24)
    # ax.legend(loc = 'lower left', fontsize = 24)
    # dummy = []
    # for i in range(len(pars) - (2+exp_no)):
    #     dummy.append('0')
    # if exp_no == 1:
    #     dummy.append('A1%:' + str(amp1/(amp1+amp0)))
    #     dummy.append('A0%:' + str(amp0/(amp1+amp0)))
    # elif exp_no == 2:
    #     dummy.append('A1%:' + str(amp1/(amp1+amp2+amp0)))
    #     dummy.append('A2%:' + str(amp2/(amp1+amp2+amp0)))
    #     dummy.append('A0%:' + str(amp0/(amp1+amp2+amp0)))
    # elif exp_no == 3:
    #     dummy.append('A1%:' + str(amp1/(amp1+amp2+amp3+amp0)))
    #     dummy.append('A2%:' + str(amp2/(amp1+amp2+amp3+amp0)))
    #     dummy.append('A3%:' + str(amp3/(amp1+amp2+amp3+amp0)))
    #     dummy.append('A0%:' + str(amp0/(amp1+amp2+amp3+amp0)))
    # dummy.append(str(Rsquared))
    # res = pd.DataFrame(
    # {'parameter': pars,
    #  'value': list(fittedParameters),
    #  'sigma': list(np.sqrt(np.diag(pcov))),
    #  'R2': dummy,
    # })
    # return res
var = [rudo_dye_r, rudo_dye_2r, rudo_mof_r]
bounds2 = ([0,  0  ,0   ,0,   -50,    0],
          [np.inf   ,1000   ,np.inf, np.inf ,100,0.00001])
initPar = [1, 500, 1,50, 0, 0]
fitdecay_multipleplots(var[0]['time'], [i['counts']-np.average(i['counts'][0:10]) for i in var], 2, initPar, bounds2, tlims, 0, 0)
#%% Fitting With Rise
def FitSingleKineticWithRise(time_vector, int_vector, exp_no, initPar, bounds, tlims, ylims, logscale):
    if bounds == 0:
        bounds = (-np.inf, np.inf)
    time = time_vector
    x = time_vector
    y = int_vector
    nan_i = y.index[y.isnull()]
    y = y.dropna()
    x = x.drop(nan_i)
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
    if max(tlims) > 500 and max(tlims) <= 2000:
        ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
    if max(tlims) > 2000:
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
    elif tlims == 0: 
        pass
    ax.set_xlim(tlims)
    ax.set_xlabel("Time, ns", fontsize = 24)
    ax.set_ylabel("Intensity (arb. units)", fontsize = 24)
    if ylims == 0: 
        pass
    else:
        ax.set_ylim(ylims)
    ax.tick_params(axis='x', labelsize= 20)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.axhline(y=0, color = 'black', ls = '--')
    if logscale == 1:
        zr = next(x for x, val in enumerate(time) if val > 0)
        x = x[zr:]
        y = y[zr:]
        yplt = yplt[zr:]
        ax.scatter(x,y, color = 'black', facecolors = 'none', s = 40, marker = "D", label = 'Data')
        ax.plot(x,yplt, color = 'purple', linewidth = 3, label = 'Fit')
        ax.scatter(x, np.subtract(y,yplt), color = 'magenta', s = 10, label = 'Residuals')
        ax.set_xscale('log')
        ax.legend(loc = 'lower left', fontsize = 24)
    else:
        ax.scatter(x,y, color = 'black', facecolors = 'none',s = 40, marker = "D", label = 'Data')
        ax.plot(x,yplt, color = 'purple', linewidth = 3, label = 'Fit')
        ax.scatter(x, np.subtract(y,yplt), color = 'magenta', s = 10, label = 'Residuals')
        ax.legend(loc = 'upper right', fontsize = 24)
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
    return res

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
        
# bounds1 = ([0, 0,-np.inf,-np.inf,0,],
#           [np.inf,2,np.inf,np.inf,np.inf])

bounds2 = ([0,   0  ,0   ,-np.inf,   0,   0,    0],
          [np.inf   ,np.inf   ,np.inf,   np.inf,np.inf,np.inf,np.inf])

# bounds3 = ([0, -np.inf,-np.inf,-np.inf,0,-np.inf,0,-np.inf,0],
#           [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,0.00001])

initPar = [10, 0.4, 300, 0, 1000, 30, 1000]
var = ruoc_mof_r
tlims = [-50, 100]
# tlims = [10**0.0001, 10**3.1]
final_fits['RuOC_MOF'] = FitSingleKineticWithRise(var['time'], var['counts']-np.average(var['counts'][0:10]), 2, initPar, bounds2, tlims, 0, 0)


initPar = []
#%%
ruco_mof_rr = rebin(ruco_mof, 100)
rudo_mof_rr = rebin(rudo_mof, 100)
ruoc_mof_rr = rebin(ruoc_mof, 100)
irf_rr = rebin(irf, 100)