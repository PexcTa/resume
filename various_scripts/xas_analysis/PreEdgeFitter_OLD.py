# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:37:36 2022

@author: boris
"""

import csv
import itertools
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from scipy.signal import savgol_filter
from scipy.signal import correlate
from scipy import special
import os
from scipy.optimize import curve_fit
from lmfit import Model

#%%
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

def ReadSingleXANESnor(file, skip=40):
    with open(file) as current_file:
        # names = ['E', 'norm', 'nbkg', 'flat', 'fbkg', 'nderiv', 'nderiv_2'], 
        d = pd.read_csv(current_file, delim_whitespace = True, header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
#%% Read in the data (assumes an Athena .nor file with a bunch of normalized XANES data in columns)
data = ReadSingleXANESnor('marked.nor', skip = 12)
    
#%% Set up a dataframe which will house the results
FitResults = pd.DataFrame(columns = ['Variable'])
FitResults['Variable'] = ['FWHM', 'A1 Amp', 'A2 Amp', 'A2+3 Amp', 'A3 Amp', 'B Amp', 'C Amp', 'A1 E0', 'A2 E0', 'A2+3 E0', 'A3 E0', 'B E0', 'C E0']
#%% BLOCK A0 - DEFINE FIT-SPECIFIC FUNCTIONS
def pseudovoigt(E, meow, a, w, e):
    return a*(meow*(w**2 / (w**2 + (2*E - 2*e)**2)) + (1-meow)*(np.exp(-(E-e)**2/(2*w**2))))

def TiK_PreEdge_TotalFit_PseudoVoigt50pcEqualFWHM_PVEdge(E, a1, w1, e1, a2, e2, a3, e3, a4,e4, a5, e5, a6, w6, e6, meow):
    return a1*(0.5*(w1**2 / (w1**2 + (2*E - 2*e1)**2)) + 0.5*(np.exp(-(E-e1)**2/(2*w1**2)))) + \
            a2*(0.5*(w1**2 / (w1**2 + (2*E - 2*e2)**2)) + 0.5*(np.exp(-(E-e2)**2/(2*w1**2)))) + \
            a3*(0.5*(w1**2 / (w1**2 + (2*E - 2*e3)**2)) + 0.5*(np.exp(-(E-e3)**2/(2*w1**2)))) + \
            a4*(0.5*(w1**2 / (w1**2 + (2*E - 2*e4)**2)) + 0.5*(np.exp(-(E-e4)**2/(2*w1**2)))) + \
            a5*(0.5*(w1**2 / (w1**2 + (2*E - 2*e5)**2)) + 0.5*(np.exp(-(E-e5)**2/(2*w1**2)))) + \
            a6*(meow*(w6**2 / (w6**2 + (2*E - 2*e6)**2)) + (1-meow)*(np.exp(-(E-e6)**2/(2*w6**2))))
#%% BLOCK A1 - STANDARD MODEL - A1 A2 A3 B C Edge peaks, WORKS
dataset = data.copy()
for label in data.columns[1:]:

    start_Index = 45
    end_Index = 140
    energy = dataset['energy'][start_Index:end_Index]
    signal = dataset[label][start_Index:end_Index]
    
    FitModel1 = Model(TiK_PreEdge_TotalFit_PseudoVoigt50pcEqualFWHM_PVEdge)
    parameters = FitModel1.make_params()
    FitModel1.set_param_hint('a1', value=0.3, min=0.01, max = 1)
    FitModel1.set_param_hint('a2', value=0.3, min=0.01, max = 1)
    FitModel1.set_param_hint('a3', value=0.3, min=0.01, max = 1)
    FitModel1.set_param_hint('a4', value=0.3, min=0.01, max = 1)
    FitModel1.set_param_hint('a5', value=1, min=0)
    FitModel1.set_param_hint('a6', value=1, min=0)
    FitModel1.set_param_hint('e1', value=4968.5)
    FitModel1.set_param_hint('e2', value=4970.5)
    FitModel1.set_param_hint('e3', value=4972)
    FitModel1.set_param_hint('e4', value=4974)
    FitModel1.set_param_hint('e5', value=4979)
    FitModel1.set_param_hint('e6', value=4985)
    FitModel1.set_param_hint('meow', value=0.5)
    FitModel1.set_param_hint('w1', value=0.5, min = 0, max = 3.0)
    FitModel1.set_param_hint('w6', value=3)
    result0 = FitModel1.fit(signal, E=energy)
    
    
    fig, ax = plt.subplots(1,3,figsize=(36,12))
    
    ax[0].xaxis.set_ticklabels([])
    ax[0].tick_params(axis='y', labelsize= 24)
    ax[0].set_xlim(4960, max(energy))
    ax[0].set_ylabel("Normalized x\u03bc(E)", fontsize = 24)
    ax[0].xaxis.set_major_locator(MultipleLocator(5))
    ax[0].xaxis.set_minor_locator(MultipleLocator(1))
    ax[0].scatter(energy, signal, facecolors = 'none', color = 'black', label = 'data')
    ax[0].plot(energy, result0.best_fit, '-', linewidth = 3, color = 'navy', label = 'fit')
    ax[0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
    ax[0].plot(energy, pseudovoigt(energy, result0.params['meow'].value, result0.params['a6'].value, result0.params['w6'].value, result0.params['e6'].value), linewidth = 2, linestyle = '--', label = 'Edge')
    ax[0].plot(energy, pseudovoigt(energy, 0.5, result0.params['a1'].value, result0.params['w1'].value, result0.params['e1'].value), linewidth = 2, linestyle = '--', label = 'A1')
    ax[0].plot(energy, pseudovoigt(energy, 0.5,result0.params['a2'].value, result0.params['w1'].value, result0.params['e2'].value), linewidth = 2, linestyle = '--', label = 'A2-1')
    ax[0].plot(energy, pseudovoigt(energy, 0.5,result0.params['a3'].value, result0.params['w1'].value, result0.params['e3'].value), linewidth = 2, linestyle = '--', label = 'A2-2')
    ax[0].plot(energy, pseudovoigt(energy, 0.5,result0.params['a4'].value, result0.params['w1'].value, result0.params['e4'].value), linewidth = 2, linestyle = '--', label = 'A3')
    ax[0].plot(energy, pseudovoigt(energy, 0.5,result0.params['a5'].value, result0.params['w1'].value, result0.params['e5'].value), linewidth = 2, linestyle = '--', label = 'B1')
    ax[0].legend(loc = 'upper left', fontsize = 24)
    difference = signal - result0.best_fit
    frame10=fig.add_axes((0.125,0.05,0.2275,0.075))
    frame10.set_xlim(4960, max(energy))       
    frame10.plot(energy,difference,color='magenta', label='Residuals')
    frame10.legend(loc = 'lower left', fontsize = 20)
    frame10.set_xlabel('Energy, eV', fontsize = 24)
    frame10.set_ylabel('$\Delta\mu$', fontsize = 24)
    frame10.xaxis.set_major_locator(MultipleLocator(5))
    frame10.axhline(y = 0, color = 'slategray', linewidth = 1.5, linestyle = '--', alpha = 0.5)
    frame10.xaxis.set_minor_locator(MultipleLocator(1))
    frame10.tick_params(axis = 'x', labelsize = 24)
    frame10.tick_params(axis='y', labelsize= 18)
    
    
    with open(label+'_4960_to_'+str(int(max(energy)))+'_eV.txt', 'w') as fh:
        fh.write(result0.fit_report())
        
        
    start_Index = 50
    end_Index = 140
    energy = dataset['energy'][start_Index:end_Index]
    signal = dataset[label][start_Index:end_Index]
    result1 = FitModel1.fit(signal, E=energy)
    
    ax[1].xaxis.set_ticklabels([])
    ax[1].tick_params(axis='y', labelsize= 24)
    ax[1].set_xlim(4960, max(energy))
    ax[1].set_ylabel("Normalized x\u03bc(E)", fontsize = 24)
    ax[1].xaxis.set_major_locator(MultipleLocator(5))
    ax[1].xaxis.set_minor_locator(MultipleLocator(1))
    ax[1].scatter(energy, signal, facecolors = 'none', color = 'black', label = 'data')
    ax[1].plot(energy, result1.best_fit, '-', linewidth = 3, color = 'navy', label = 'fit')
    ax[1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
    ax[1].plot(energy, pseudovoigt(energy, result1.params['meow'].value, result1.params['a6'].value, result1.params['w6'].value, result1.params['e6'].value), linewidth = 2, linestyle = '--', label = 'Edge')
    ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a1'].value, result1.params['w1'].value, result1.params['e1'].value), linewidth = 2, linestyle = '--', label = 'A1')
    ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a2'].value, result1.params['w1'].value, result1.params['e2'].value), linewidth = 2, linestyle = '--', label = 'A2-1')
    ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a3'].value, result1.params['w1'].value, result1.params['e3'].value), linewidth = 2, linestyle = '--', label = 'A2-2')
    ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a4'].value, result1.params['w1'].value, result1.params['e4'].value), linewidth = 2, linestyle = '--', label = 'A3')
    ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a5'].value, result1.params['w1'].value, result1.params['e5'].value), linewidth = 2, linestyle = '--', label = 'B1')
    difference = signal - result1.best_fit
    frame11=fig.add_axes((0.3987,0.05,0.2275,0.075))
    frame11.set_xlim(4960, max(energy))       
    frame11.set_xlabel('Energy, eV', fontsize = 24)
    frame11.set_ylabel('$\Delta\mu$', fontsize = 24)
    frame11.plot(energy,difference,color='magenta', label='Residuals')
    frame11.xaxis.set_major_locator(MultipleLocator(5))
    frame11.axhline(y = 0, color = 'slategray', linewidth = 1.5, linestyle = '--', alpha = 0.5)
    frame11.xaxis.set_minor_locator(MultipleLocator(1))
    frame11.tick_params(axis = 'x', labelsize = 24)
    frame11.tick_params(axis='y', labelsize= 18)
    
    with open(label+'_4960_to_'+str(int(max(energy)))+'_eV.txt', 'w') as fh:
        fh.write(result1.fit_report())
    
    start_Index = 50
    end_Index = 145
    energy = dataset['energy'][start_Index:end_Index]
    signal = dataset[label][start_Index:end_Index]
    result2 = FitModel1.fit(signal, E=energy)
    
    
    
    ax[2].xaxis.set_ticklabels([])
    ax[2].tick_params(axis='y', labelsize= 24)
    ax[2].set_xlim(4960, max(energy))
    ax[2].set_ylabel("Normalized x\u03bc(E)", fontsize = 24)
    ax[2].xaxis.set_major_locator(MultipleLocator(5))
    ax[2].xaxis.set_minor_locator(MultipleLocator(1))
    ax[2].scatter(energy, signal, facecolors = 'none', color = 'black', label = 'data')
    ax[2].plot(energy, result2.best_fit, '-', linewidth = 3, color = 'navy', label = 'fit')
    ax[2].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
    ax[2].plot(energy, pseudovoigt(energy, result2.params['meow'].value, result2.params['a6'].value, result2.params['w6'].value, result2.params['e6'].value), linewidth = 2, linestyle = '--', label = 'Edge')
    ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a1'].value, result2.params['w1'].value, result2.params['e1'].value), linewidth = 2, linestyle = '--', label = 'A1')
    ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a2'].value, result2.params['w1'].value, result2.params['e2'].value), linewidth = 2, linestyle = '--', label = 'A2-1')
    ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a3'].value, result2.params['w1'].value, result2.params['e3'].value), linewidth = 2, linestyle = '--', label = 'A2-2')
    ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a4'].value, result2.params['w1'].value, result2.params['e4'].value), linewidth = 2, linestyle = '--', label = 'A3')
    ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a5'].value, result2.params['w1'].value, result2.params['e5'].value), linewidth = 2, linestyle = '--', label = 'B1')
    difference = signal - result2.best_fit
    frame12=fig.add_axes((0.6723,0.05,0.2275,0.075))
    frame12.set_xlim(4960, max(energy))       
    frame12.set_xlabel('Energy, eV', fontsize = 24)
    frame12.plot(energy,difference,color='magenta', label='Residuals')
    frame12.set_ylabel('$\Delta\mu$', fontsize = 24)
    frame12.xaxis.set_major_locator(MultipleLocator(5))
    frame12.axhline(y = 0, color = 'slategray', linewidth = 1.5, linestyle = '--', alpha = 0.5)
    frame12.xaxis.set_minor_locator(MultipleLocator(1))
    frame12.tick_params(axis = 'x', labelsize = 24)
    frame12.tick_params(axis='y', labelsize= 18)
    
    
    
    with open(label+'_4960_to_'+str(int(max(energy)))+'_eV.txt', 'w') as fh:
        fh.write(result2.fit_report())
    plt.savefig(f"FitImage_{label}.png")

    LastResult = {}
    LastResult['Avg FWHM'] = np.average([result0.params['w1'].value, result1.params['w1'].value, result2.params['w1'].value])
    LastResult['Avg FWHM Error'] = np.sqrt(np.sum([result0.params['w1'].stderr**2, result1.params['w1'].stderr**2, result2.params['w1'].stderr**2])/9)
    
    LastResult['Avg A-peak Amp Sum'] = np.average([np.sum([result0.params['a1'].value, result0.params['a2'].value, result0.params['a3'].value,result0.params['a4'].value]), \
                                                      np.sum([result1.params['a1'].value, result1.params['a2'].value, result1.params['a3'].value, result1.params['a4'].value]),\
                                                          np.sum([result2.params['a1'].value, result2.params['a2'].value, result2.params['a3'].value,result2.params['a4'].value])])
    LastResult['Avg A-peak Amp Sum Error'] = np.sqrt(np.sum([result0.params['a1'].stderr**2, result1.params['a1'].stderr**2, result2.params['a1'].stderr**2, \
                                                             result0.params['a2'].stderr**2, result1.params['a2'].stderr**2, result2.params['a2'].stderr**2, \
                                                                 result0.params['a3'].stderr**2, result1.params['a3'].stderr**2, result2.params['a3'].stderr**2, \
                                                                     result0.params['a4'].stderr**2, result1.params['a4'].stderr**2, result2.params['a4'].stderr**2])/144)
    LastResult['Avg A1 Peak Share'] = np.average([result0.params['a1'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a1'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a1'].value/LastResult['Avg A-peak Amp Sum']])
    LastResult['Avg A1 Peak Share Error'] = LastResult['Avg A1 Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a1'].stderr/result0.params['a1'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a1'].stderr/result1.params['a1'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a1'].stderr/result2.params['a1'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg A2 Peak Share'] = np.average([result0.params['a2'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a2'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a2'].value/LastResult['Avg A-peak Amp Sum']])
    LastResult['Avg A2 Peak Share Error'] = LastResult['Avg A2 Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a2'].stderr/result0.params['a2'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a2'].stderr/result1.params['a2'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a2'].stderr/result2.params['a2'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg A2+3 Peak Share'] = np.average([(result0.params['a2'].value+result0.params['a3'].value)/LastResult['Avg A-peak Amp Sum'],(result1.params['a2'].value+result1.params['a3'].value)/LastResult['Avg A-peak Amp Sum'],(result2.params['a2'].value+result2.params['a3'].value)/LastResult['Avg A-peak Amp Sum']])
    LastResult['Avg A2+3 Peak Share Error'] = LastResult['Avg A2+3 Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a2'].stderr/result0.params['a2'].value)**2,(result0.params['a3'].stderr/result0.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a2'].stderr/result1.params['a2'].value)**2,(result1.params['a3'].stderr/result1.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a2'].stderr/result2.params['a2'].value)**2,(result2.params['a3'].stderr/result2.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg A3 Peak Share'] = np.average([result0.params['a3'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a3'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a3'].value/LastResult['Avg A-peak Amp Sum']])
    LastResult['Avg A3 Peak Share Error'] = LastResult['Avg A3 Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a3'].stderr/result0.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a3'].stderr/result1.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a3'].stderr/result2.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg B Peak Share'] = np.average([result0.params['a4'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a4'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a4'].value/LastResult['Avg A-peak Amp Sum']]) 
    LastResult['Avg B Peak Share Error'] = LastResult['Avg B Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a4'].stderr/result0.params['a4'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a4'].stderr/result1.params['a4'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a4'].stderr/result2.params['a4'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg C Peak to Amp Sum'] = np.average([result0.params['a5'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a5'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a5'].value/LastResult['Avg A-peak Amp Sum']])
    LastResult['Avg C Peak to Amp Sum Error'] = LastResult['Avg C Peak to Amp Sum']*np.average([np.sqrt(np.sum([(result0.params['a5'].stderr/result0.params['a5'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a5'].stderr/result1.params['a5'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a5'].stderr/result2.params['a5'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg A1 E0'] = np.average([result0.params['e1'].value, result1.params['e1'].value, result2.params['e1'].value])
    LastResult['Avg A1 E0 Error'] = np.sqrt(np.sum([result0.params['e1'].stderr**2, result1.params['e1'].stderr**2, result2.params['e1'].stderr**2])/9)
    LastResult['Avg A2 E0'] = np.average([result0.params['e2'].value, result1.params['e2'].value, result2.params['e2'].value])
    LastResult['Avg A2 E0 Error'] = np.sqrt(np.sum([result0.params['e2'].stderr**2, result1.params['e2'].stderr**2, result2.params['e2'].stderr**2])/9)
    LastResult['Avg A2+3 E0'] = np.average([result0.params['e2'].value, result1.params['e2'].value, result2.params['e2'].value, result0.params['e3'].value, result1.params['e3'].value, result2.params['e3'].value])
    LastResult['Avg A2+3 E0 Error'] = np.sqrt(np.sum([result0.params['e2'].stderr**2, result1.params['e2'].stderr**2, result2.params['e2'].stderr**2,result0.params['e3'].stderr**2, result1.params['e3'].stderr**2, result2.params['e3'].stderr**2])/36)
    LastResult['Avg A3 E0'] = np.average([result0.params['e3'].value, result1.params['e3'].value, result2.params['e3'].value])
    LastResult['Avg A3 E0 Error'] = np.sqrt(np.sum([result0.params['e3'].stderr**2, result1.params['e3'].stderr**2, result2.params['e3'].stderr**2])/9)
    LastResult['Avg B E0'] = np.average([result0.params['e4'].value, result1.params['e4'].value, result2.params['e4'].value])
    LastResult['Avg B E0 Error'] = np.sqrt(np.sum([result0.params['e4'].stderr**2, result1.params['e4'].stderr**2, result2.params['e4'].stderr**2])/9)
    LastResult['Avg C E0'] = np.average([result0.params['e5'].value, result1.params['e5'].value, result2.params['e5'].value])
    LastResult['Avg C E0 Error'] = np.sqrt(np.sum([result0.params['e5'].stderr**2, result1.params['e5'].stderr**2, result2.params['e5'].stderr**2])/9)
    FitResults[label+"_value"] = [LastResult['Avg FWHM'], \
                                  LastResult['Avg A1 Peak Share'], \
                                LastResult['Avg A2 Peak Share'], \
                                    LastResult['Avg A2+3 Peak Share'], \
                                    LastResult['Avg A3 Peak Share'], \
                                    LastResult['Avg B Peak Share'], \
                                        LastResult['Avg C Peak to Amp Sum'], \
                                        LastResult['Avg A1 E0'], \
                                           LastResult['Avg A2 E0'], \
                                               LastResult['Avg A2+3 E0'], \
                                                   LastResult['Avg A3 E0'], \
                                                   LastResult['Avg B E0'], \
                                                       LastResult['Avg C E0']]
        
    FitResults[label+"_error"] = [LastResult['Avg FWHM Error'], \
                                  LastResult['Avg A1 Peak Share Error'], \
                                LastResult['Avg A2 Peak Share Error'], \
                                    LastResult['Avg A2+3 Peak Share Error'], \
                                    LastResult['Avg A3 Peak Share Error'], \
                                    LastResult['Avg B Peak Share Error'], \
                                        LastResult['Avg C Peak to Amp Sum Error'], \
                                        LastResult['Avg A1 E0 Error'], \
                                           LastResult['Avg A2 E0 Error'], \
                                               LastResult['Avg A2+3 E0 Error'], \
                                                   LastResult['Avg A3 E0 Error'], \
                                                   LastResult['Avg B E0 Error'], \
                                                       LastResult['Avg C E0 Error']]
    
FitResults.to_csv("Fitresults_Table.csv", index = False)


#%% BLOCK A2 - get pH BORDER REGIONS FROM REFs - WORKS
results_table = RESULTS_ref4nm.copy()
ph_regionBorders = pd.DataFrame(columns = ['Variable'])
ph_regionBorders['Variable'] = ['FWHM', 'A1 Amp', 'A2 Amp', 'A2+3 Amp', 'A3 Amp', 'B Amp', 'C Amp', 'A1 E0', 'A2 E0', 'A2+3 E0', 'A3 E0', 'B E0', 'C E0']
for column in results_table.columns[1:]:
    if "dry" in str(column) or "H2O" in str(column):
        continue
    if str(column).endswith('value'):
        ph_regionBorders[str(column)[:-6]+"_low"] = results_table[column] - results_table[str(column)[:-6]+"_error"]
        ph_regionBorders[str(column)[:-6]+"_high"] = results_table[column] + results_table[str(column)[:-6]+"_error"]
        
#%% BLOCK A3 - for PLOTTING FWHM SHARES in STANDARD MODEL
Labels = ['OCV', '50 mA',  '100 mA', 'OCV \npost 100 mA', '250 mA', 'OCV \npost 250 mA']

fig, ax = plt.subplots(figsize = (12,12))
rowIndex = FitResults[FitResults['Variable'] == 'FWHM'].index.values[0]

ax.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, len(Labels)+1)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax.errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
# comment the section below OUT if plotting REFERENCE DATA
ax.axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
ax.axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
ax.axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
ax.axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
ax.axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
ax.axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
ax.axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
ax.axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
ax.axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)

x_ticks_labels = Labels
ax.set_xticklabels(x_ticks_labels, rotation=45, fontsize=32)
ax.tick_params(axis='y', labelsize= 32)
ax.set_ylabel('pseudo-Voigt Average FWHM (eV)', fontsize = 32)

#%% BLOCK A4 - for PLOTTING PEAK SHARES in STANDARD MODEL
fig, ax = plt.subplots(2,3,figsize = (36,24))

rowIndex = FitResults[FitResults['Variable'] == 'A1 Amp'].index.values[0]
ax[0,0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,0].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[0,0].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[0,0].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[0,0].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, facecolor='crimson', alpha=0.35)
ax[0,0].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[0,0].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[0,0].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, facecolor='turquoise', alpha=0.35)
ax[0,0].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[0,0].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[0,0].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, facecolor='indigo', alpha=0.35)
ax[0,0].xaxis.set_ticklabels([])
ax[0,0].tick_params(axis='y', labelsize= 36)
ax[0,0].set_ylabel('Average Share of Peak A1 (%)', fontsize = 36)
ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

rowIndex = FitResults[FitResults['Variable'] == 'A2 Amp'].index.values[0]
ax[0,1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,1].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[0,1].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[0,1].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[0,1].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, facecolor='crimson', alpha=0.35)
ax[0,1].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[0,1].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[0,1].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, facecolor='turquoise', alpha=0.35)
ax[0,1].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[0,1].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[0,1].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, facecolor='indigo', alpha=0.35)
ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0,1].xaxis.set_ticklabels([])
ax[0,1].tick_params(axis='y', labelsize= 36)
ax[0,1].set_ylabel('Average Share of Peak A2 (%)', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'A3 Amp'].index.values[0]
ax[0,2].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,2].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[0,2].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[0,2].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[0,2].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, facecolor='crimson', alpha=0.35)
ax[0,2].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[0,2].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[0,2].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, facecolor='turquoise', alpha=0.35)
ax[0,2].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[0,2].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[0,2].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, facecolor='indigo', alpha=0.35)
ax[0,2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0,2].xaxis.set_ticklabels([])
ax[0,2].tick_params(axis='y', labelsize= 36)
ax[0,2].set_ylabel('Average Share of Peak A3 (%)', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'A2+3 Amp'].index.values[0]
ax[1,0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,0].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[1,0].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,0].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,0].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, facecolor='crimson', alpha=0.35)
ax[1,0].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,0].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,0].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, facecolor='turquoise', alpha=0.35)
ax[1,0].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,0].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,0].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, facecolor='indigo', alpha=0.35)
ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
x_ticks_labels = Labels
ax[1,0].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,0].tick_params(axis='y', labelsize= 36)
ax[1,0].set_ylabel('Average Proportion of Sum Peaks A2-3 (%)', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'B Amp'].index.values[0]
ax[1,1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,1].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[1,1].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,1].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,1].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, facecolor='crimson', alpha=0.35)
ax[1,1].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,1].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,1].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, facecolor='turquoise', alpha=0.35)
ax[1,1].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,1].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,1].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, facecolor='indigo', alpha=0.35)
ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
x_ticks_labels = Labels
ax[1,1].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,1].tick_params(axis='y', labelsize= 36)
ax[1,1].set_ylabel('Average Proportion of Peak B (%)', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'C Amp'].index.values[0]
ax[1,2].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,2].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[1,2].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,2].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,2].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph1_merge_high'][rowIndex]*100, facecolor='crimson', alpha=0.35)
ax[1,2].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,2].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,2].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph7merge_high'][rowIndex]*100, facecolor='turquoise', alpha=0.35)
ax[1,2].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,2].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,2].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex]*100, ph_regionBorders['4nm_ph10_merge_high'][rowIndex]*100, facecolor='indigo', alpha=0.35)
ax[1,2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
x_ticks_labels = Labels
ax[1,2].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,2].tick_params(axis='y', labelsize= 36)
ax[1,2].set_ylabel('Average Proportion of Peak C (%)', fontsize = 36)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.25, 
                    hspace=0.05)
#%% BLOCK A5 - for PLOTTING PEAK POSITIONS in STANDARD MODEL
fig, ax = plt.subplots(2,3,figsize = (36,24))

rowIndex = FitResults[FitResults['Variable'] == 'A1 E0'].index.values[0]
ax[0,0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,0].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
ax[0,0].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
ax[0,0].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
ax[0,0].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
ax[0,0].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
ax[0,0].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
ax[0,0].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
ax[0,0].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
ax[0,0].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
ax[0,0].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[0,0].xaxis.set_ticklabels([])
ax[0,0].tick_params(axis='y', labelsize= 36)
ax[0,0].set_ylabel('Average Position of Peak A1', fontsize = 36)
ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

rowIndex = FitResults[FitResults['Variable'] == 'A2 E0'].index.values[0]
ax[0,1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,1].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
ax[0,1].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
ax[0,1].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
ax[0,1].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
ax[0,1].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
ax[0,1].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
ax[0,1].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
ax[0,1].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
ax[0,1].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
ax[0,1].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0,1].xaxis.set_ticklabels([])
ax[0,1].tick_params(axis='y', labelsize= 36)
ax[0,1].set_ylabel('Average Position of Peak A2', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'A3 E0'].index.values[0]
ax[0,2].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,2].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
ax[0,2].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
ax[0,2].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
ax[0,2].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
ax[0,2].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
ax[0,2].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
ax[0,2].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
ax[0,2].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
ax[0,2].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
ax[0,2].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[0,2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0,2].xaxis.set_ticklabels([])
ax[0,2].tick_params(axis='y', labelsize= 36)
ax[0,2].set_ylabel('Average Position of Peak A3', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'A2+3 E0'].index.values[0]
ax[1,0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,0].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
ax[1,0].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
ax[1,0].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
ax[1,0].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
ax[1,0].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
ax[1,0].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
ax[1,0].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
ax[1,0].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
ax[1,0].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
ax[1,0].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
x_ticks_labels = Labels
ax[1,0].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,0].tick_params(axis='y', labelsize= 36)
ax[1,0].set_ylabel('Average Position of Sum Peaks A2-3', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'B E0'].index.values[0]
ax[1,1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,1].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
ax[1,1].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
ax[1,1].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
ax[1,1].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
ax[1,1].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
ax[1,1].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
ax[1,1].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
ax[1,1].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
ax[1,1].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
ax[1,1].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
x_ticks_labels = Labels
ax[1,1].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,1].tick_params(axis='y', labelsize= 36)
ax[1,1].set_ylabel('Average Position of Peak B', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'C E0'].index.values[0]
ax[1,2].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,2].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
ax[1,2].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
ax[1,2].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
ax[1,2].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
ax[1,2].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
ax[1,2].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
ax[1,2].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
ax[1,2].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
ax[1,2].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
ax[1,2].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[1,2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
x_ticks_labels = Labels
ax[1,2].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,2].tick_params(axis='y', labelsize= 36)
ax[1,2].set_ylabel('Average Position of Peak C', fontsize = 36)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.05)

     
#%% DEAD SPACE





#%%
FitResults = pd.DataFrame(columns = ['Variable'])
FitResults['Variable'] = ['FWHM', 'A1 Amp', 'A2 Amp', 'B Amp', 'C Amp', 'A1 E0', 'A2 E0', 'B E0', 'C E0']
#%% modified model, 1 nm cell samples
def pseudovoigt(E, meow, a, w, e):
    return a*(meow*(w**2 / (w**2 + (2*E - 2*e)**2)) + (1-meow)*(np.exp(-(E-e)**2/(2*w**2))))

# def TiK_PreEdge_TotalFit_PseudoVoigt50pcEqualFWHM_PVEdge(E, a1, w1, e1, a2, e2, a4,e4, a5, e5, a6, w6, e6, meow):
#     return a1*(0.5*(w1**2 / (w1**2 + (2*E - 2*e1)**2)) + 0.5*(np.exp(-(E-e1)**2/(2*w1**2)))) + \
#             a2*(0.5*(w1**2 / (w1**2 + (2*E - 2*e2)**2)) + 0.5*(np.exp(-(E-e2)**2/(2*w1**2)))) + \
#             a4*(0.5*(w1**2 / (w1**2 + (2*E - 2*e4)**2)) + 0.5*(np.exp(-(E-e4)**2/(2*w1**2)))) + \
#             a5*(0.5*(w1**2 / (w1**2 + (2*E - 2*e5)**2)) + 0.5*(np.exp(-(E-e5)**2/(2*w1**2)))) + \
#             a6*(meow*(w6**2 / (w6**2 + (2*E - 2*e6)**2)) + (1-meow)*(np.exp(-(E-e6)**2/(2*w6**2))))
            
def TiK_PreEdge_TotalFit_PseudoVoigt50pcEqualFWHM_PVEdge(E, a1, w1, w2, e1, a2, e2, a4,e4, a5, e5, a6, w6, e6, meow):
    return a1*(0.5*(w1**2 / (w1**2 + (2*E - 2*e1)**2)) + 0.5*(np.exp(-(E-e1)**2/(2*w1**2)))) + \
            a2*(0.5*(w2**2 / (w2**2 + (2*E - 2*e2)**2)) + 0.5*(np.exp(-(E-e2)**2/(2*w2**2)))) + \
            a4*(0.5*(w1**2 / (w1**2 + (2*E - 2*e4)**2)) + 0.5*(np.exp(-(E-e4)**2/(2*w1**2)))) + \
            a5*(0.5*(w1**2 / (w1**2 + (2*E - 2*e5)**2)) + 0.5*(np.exp(-(E-e5)**2/(2*w1**2)))) + \
            a6*(meow*(w6**2 / (w6**2 + (2*E - 2*e6)**2)) + (1-meow)*(np.exp(-(E-e6)**2/(2*w6**2))))
            
dataset = data.copy()
for label in data.columns[1:]:

    start_Index = 50
    end_Index = 138
    energy = dataset['energy'][start_Index:end_Index]
    signal = dataset[label][start_Index:end_Index]
    
    FitModel1 = Model(TiK_PreEdge_TotalFit_PseudoVoigt50pcEqualFWHM_PVEdge)
    parameters = FitModel1.make_params()
    FitModel1.set_param_hint('a1', value=0.3, min=0.01, max = 2)
    FitModel1.set_param_hint('a2', value=0.3, min=0.01, max = 2)
    # FitModel1.set_param_hint('a3', value=0.3, min=0.01, max = 1)
    FitModel1.set_param_hint('a4', value=0.3, min=0.01, max = 2)
    FitModel1.set_param_hint('a5', value=1, min=0, max = 2)
    FitModel1.set_param_hint('a6', value=1, min=0)
    FitModel1.set_param_hint('e1', value=4968.5, min=4966, max = 4975)
    FitModel1.set_param_hint('e2', value=4970.5, min=4966, max = 4975)
    # FitModel1.set_param_hint('e3', value=4972)
    FitModel1.set_param_hint('e4', value=4974, min=4966, max = 4975)
    FitModel1.set_param_hint('e5', value=4979)
    FitModel1.set_param_hint('e6', value=4985)
    FitModel1.set_param_hint('meow', value=0.5)
    FitModel1.set_param_hint('w1', value=0.5, min = 0, max = 3.0)
    FitModel1.set_param_hint('w2', value=0.5, min = 0, max = 3.0)
    FitModel1.set_param_hint('w6', value=3)
    result0 = FitModel1.fit(signal, E=energy)
    
    
    fig, ax = plt.subplots(1,3,figsize=(36,12))
    
    ax[0].xaxis.set_ticklabels([])
    ax[0].tick_params(axis='y', labelsize= 24)
    ax[0].set_xlim(4960, max(energy))
    ax[0].set_ylabel("Normalized x\u03bc(E)", fontsize = 24)
    ax[0].xaxis.set_major_locator(MultipleLocator(5))
    ax[0].xaxis.set_minor_locator(MultipleLocator(1))
    ax[0].scatter(energy, signal, facecolors = 'none', color = 'black', label = 'data')
    ax[0].plot(energy, result0.best_fit, '-', linewidth = 3, color = 'navy', label = 'fit')
    ax[0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
    ax[0].plot(energy, pseudovoigt(energy, result0.params['meow'].value, result0.params['a6'].value, result0.params['w6'].value, result0.params['e6'].value), linewidth = 2, linestyle = '--', label = 'Edge')
    ax[0].plot(energy, pseudovoigt(energy, 0.5, result0.params['a1'].value, result0.params['w1'].value, result0.params['e1'].value), linewidth = 2, linestyle = '--', label = 'A1')
    ax[0].plot(energy, pseudovoigt(energy, 0.5,result0.params['a2'].value, result0.params['w2'].value, result0.params['e2'].value), linewidth = 2, linestyle = '--', label = 'A2-1')
    # ax[0].plot(energy, pseudovoigt(energy, 0.5,result0.params['a3'].value, result0.params['w1'].value, result0.params['e3'].value), linewidth = 2, linestyle = '--', label = 'A2-2')
    ax[0].plot(energy, pseudovoigt(energy, 0.5,result0.params['a4'].value, result0.params['w1'].value, result0.params['e4'].value), linewidth = 2, linestyle = '--', label = 'A3')
    ax[0].plot(energy, pseudovoigt(energy, 0.5,result0.params['a5'].value, result0.params['w1'].value, result0.params['e5'].value), linewidth = 2, linestyle = '--', label = 'B1')
    ax[0].legend(loc = 'upper left', fontsize = 24)
    difference = signal - result0.best_fit
    frame10=fig.add_axes((0.125,0.05,0.2275,0.075))
    frame10.set_xlim(4960, max(energy))       
    frame10.plot(energy,difference,color='magenta', label='Residuals')
    frame10.legend(loc = 'lower left', fontsize = 20)
    frame10.set_xlabel('Energy, eV', fontsize = 24)
    frame10.set_ylabel('$\Delta\mu$', fontsize = 24)
    frame10.xaxis.set_major_locator(MultipleLocator(5))
    frame10.axhline(y = 0, color = 'slategray', linewidth = 1.5, linestyle = '--', alpha = 0.5)
    frame10.xaxis.set_minor_locator(MultipleLocator(1))
    frame10.tick_params(axis = 'x', labelsize = 24)
    frame10.tick_params(axis='y', labelsize= 18)
    
    
    with open(label+'_4960_to_'+str(int(max(energy)))+'_eV.txt', 'w') as fh:
        fh.write(result0.fit_report())
        
        
    start_Index = 50
    end_Index = 140
    energy = dataset['energy'][start_Index:end_Index]
    signal = dataset[label][start_Index:end_Index]
    result1 = FitModel1.fit(signal, E=energy)
    
    ax[1].xaxis.set_ticklabels([])
    ax[1].tick_params(axis='y', labelsize= 24)
    ax[1].set_xlim(4960, max(energy))
    ax[1].set_ylabel("Normalized x\u03bc(E)", fontsize = 24)
    ax[1].xaxis.set_major_locator(MultipleLocator(5))
    ax[1].xaxis.set_minor_locator(MultipleLocator(1))
    ax[1].scatter(energy, signal, facecolors = 'none', color = 'black', label = 'data')
    ax[1].plot(energy, result1.best_fit, '-', linewidth = 3, color = 'navy', label = 'fit')
    ax[1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
    ax[1].plot(energy, pseudovoigt(energy, result1.params['meow'].value, result1.params['a6'].value, result1.params['w6'].value, result1.params['e6'].value), linewidth = 2, linestyle = '--', label = 'Edge')
    ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a1'].value, result1.params['w1'].value, result1.params['e1'].value), linewidth = 2, linestyle = '--', label = 'A1')
    ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a2'].value, result1.params['w2'].value, result1.params['e2'].value), linewidth = 2, linestyle = '--', label = 'A2-1')
    # ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a3'].value, result1.params['w1'].value, result1.params['e3'].value), linewidth = 2, linestyle = '--', label = 'A2-2')
    ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a4'].value, result1.params['w1'].value, result1.params['e4'].value), linewidth = 2, linestyle = '--', label = 'A3')
    ax[1].plot(energy, pseudovoigt(energy, 0.5,result1.params['a5'].value, result1.params['w1'].value, result1.params['e5'].value), linewidth = 2, linestyle = '--', label = 'B1')
    difference = signal - result1.best_fit
    frame11=fig.add_axes((0.3987,0.05,0.2275,0.075))
    frame11.set_xlim(4960, max(energy))       
    frame11.set_xlabel('Energy, eV', fontsize = 24)
    frame11.set_ylabel('$\Delta\mu$', fontsize = 24)
    frame11.plot(energy,difference,color='magenta', label='Residuals')
    frame11.xaxis.set_major_locator(MultipleLocator(5))
    frame11.axhline(y = 0, color = 'slategray', linewidth = 1.5, linestyle = '--', alpha = 0.5)
    frame11.xaxis.set_minor_locator(MultipleLocator(1))
    frame11.tick_params(axis = 'x', labelsize = 24)
    frame11.tick_params(axis='y', labelsize= 18)
    
    with open(label+'_4960_to_'+str(int(max(energy)))+'_eV.txt', 'w') as fh:
        fh.write(result1.fit_report())
    
    start_Index = 50
    end_Index = 141
    energy = dataset['energy'][start_Index:end_Index]
    signal = dataset[label][start_Index:end_Index]
    result2 = FitModel1.fit(signal, E=energy)
    
    
    
    ax[2].xaxis.set_ticklabels([])
    ax[2].tick_params(axis='y', labelsize= 24)
    ax[2].set_xlim(4960, max(energy))
    ax[2].set_ylabel("Normalized x\u03bc(E)", fontsize = 24)
    ax[2].xaxis.set_major_locator(MultipleLocator(5))
    ax[2].xaxis.set_minor_locator(MultipleLocator(1))
    ax[2].scatter(energy, signal, facecolors = 'none', color = 'black', label = 'data')
    ax[2].plot(energy, result2.best_fit, '-', linewidth = 3, color = 'navy', label = 'fit')
    ax[2].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
    ax[2].plot(energy, pseudovoigt(energy, result2.params['meow'].value, result2.params['a6'].value, result2.params['w6'].value, result2.params['e6'].value), linewidth = 2, linestyle = '--', label = 'Edge')
    ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a1'].value, result2.params['w1'].value, result2.params['e1'].value), linewidth = 2, linestyle = '--', label = 'A1')
    ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a2'].value, result2.params['w2'].value, result2.params['e2'].value), linewidth = 2, linestyle = '--', label = 'A2-1')
    # ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a3'].value, result2.params['w1'].value, result2.params['e3'].value), linewidth = 2, linestyle = '--', label = 'A2-2')
    ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a4'].value, result2.params['w1'].value, result2.params['e4'].value), linewidth = 2, linestyle = '--', label = 'A3')
    ax[2].plot(energy, pseudovoigt(energy, 0.5,result2.params['a5'].value, result2.params['w1'].value, result2.params['e5'].value), linewidth = 2, linestyle = '--', label = 'B1')
    difference = signal - result2.best_fit
    frame12=fig.add_axes((0.6723,0.05,0.2275,0.075))
    frame12.set_xlim(4960, max(energy))       
    frame12.set_xlabel('Energy, eV', fontsize = 24)
    frame12.plot(energy,difference,color='magenta', label='Residuals')
    frame12.set_ylabel('$\Delta\mu$', fontsize = 24)
    frame12.xaxis.set_major_locator(MultipleLocator(5))
    frame12.axhline(y = 0, color = 'slategray', linewidth = 1.5, linestyle = '--', alpha = 0.5)
    frame12.xaxis.set_minor_locator(MultipleLocator(1))
    frame12.tick_params(axis = 'x', labelsize = 24)
    frame12.tick_params(axis='y', labelsize= 18)
    
    
    
    with open(label+'_4960_to_'+str(int(max(energy)))+'_eV.txt', 'w') as fh:
        fh.write(result2.fit_report())

    print(f"reporting results for {label}")
    LastResult = {}
    LastResult['Avg FWHM'] = np.average([result0.params['w1'].value, result1.params['w1'].value, result2.params['w1'].value])
    LastResult['Avg FWHM Error'] = np.sqrt(np.sum([result0.params['w1'].stderr**2, result1.params['w1'].stderr**2, result2.params['w1'].stderr**2])/9)
    
    LastResult['Avg A-peak Amp Sum'] = np.average([np.sum([result0.params['a1'].value, result0.params['a2'].value, result0.params['a4'].value]), \
                                                      np.sum([result1.params['a1'].value, result1.params['a2'].value,  result1.params['a4'].value]),\
                                                          np.sum([result2.params['a1'].value, result2.params['a2'].value, result2.params['a4'].value])])
    LastResult['Avg A-peak Amp Sum Error'] = np.sqrt(np.sum([result0.params['a1'].stderr**2, result1.params['a1'].stderr**2, result2.params['a1'].stderr**2, \
                                                             result0.params['a2'].stderr**2, result1.params['a2'].stderr**2, result2.params['a2'].stderr**2, \
                                                                     result0.params['a4'].stderr**2, result1.params['a4'].stderr**2, result2.params['a4'].stderr**2])/81)
    LastResult['Avg A1 Peak Share'] = np.average([result0.params['a1'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a1'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a1'].value/LastResult['Avg A-peak Amp Sum']])
    LastResult['Avg A1 Peak Share Error'] = LastResult['Avg A1 Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a1'].stderr/result0.params['a1'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a1'].stderr/result1.params['a1'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a1'].stderr/result2.params['a1'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg A2 Peak Share'] = np.average([result0.params['a2'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a2'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a2'].value/LastResult['Avg A-peak Amp Sum']])
    LastResult['Avg A2 Peak Share Error'] = LastResult['Avg A2 Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a2'].stderr/result0.params['a2'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a2'].stderr/result1.params['a2'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a2'].stderr/result2.params['a2'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    # LastResult['Avg A2+3 Peak Share'] = np.average([(result0.params['a2'].value+result0.params['a3'].value)/LastResult['Avg A-peak Amp Sum'],(result1.params['a2'].value+result1.params['a3'].value)/LastResult['Avg A-peak Amp Sum'],(result2.params['a2'].value+result2.params['a3'].value)/LastResult['Avg A-peak Amp Sum']])
    # LastResult['Avg A2+3 Peak Share Error'] = LastResult['Avg A2+3 Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a2'].stderr/result0.params['a2'].value)**2,(result0.params['a3'].stderr/result0.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a2'].stderr/result1.params['a2'].value)**2,(result1.params['a3'].stderr/result1.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a2'].stderr/result2.params['a2'].value)**2,(result2.params['a3'].stderr/result2.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    # LastResult['Avg A3 Peak Share'] = np.average([result0.params['a3'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a3'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a3'].value/LastResult['Avg A-peak Amp Sum']])
    # LastResult['Avg A3 Peak Share Error'] = LastResult['Avg A3 Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a3'].stderr/result0.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a3'].stderr/result1.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a3'].stderr/result2.params['a3'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg B Peak Share'] = np.average([result0.params['a4'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a4'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a4'].value/LastResult['Avg A-peak Amp Sum']]) 
    LastResult['Avg B Peak Share Error'] = LastResult['Avg B Peak Share']*np.average([np.sqrt(np.sum([(result0.params['a4'].stderr/result0.params['a4'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a4'].stderr/result1.params['a4'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a4'].stderr/result2.params['a4'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg C Peak to Amp Sum'] = np.average([result0.params['a5'].value/LastResult['Avg A-peak Amp Sum'],result1.params['a5'].value/LastResult['Avg A-peak Amp Sum'],result2.params['a5'].value/LastResult['Avg A-peak Amp Sum']])
    LastResult['Avg C Peak to Amp Sum Error'] = LastResult['Avg C Peak to Amp Sum']*np.average([np.sqrt(np.sum([(result0.params['a5'].stderr/result0.params['a5'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result1.params['a5'].stderr/result1.params['a5'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2])), np.sqrt(np.sum([(result2.params['a5'].stderr/result2.params['a5'].value)**2, (LastResult['Avg A-peak Amp Sum Error']/LastResult['Avg A-peak Amp Sum'])**2]))])
    LastResult['Avg A1 E0'] = np.average([result0.params['e1'].value, result1.params['e1'].value, result2.params['e1'].value])
    LastResult['Avg A1 E0 Error'] = np.sqrt(np.sum([result0.params['e1'].stderr**2, result1.params['e1'].stderr**2, result2.params['e1'].stderr**2])/9)
    LastResult['Avg A2 E0'] = np.average([result0.params['e2'].value, result1.params['e2'].value, result2.params['e2'].value])
    LastResult['Avg A2 E0 Error'] = np.sqrt(np.sum([result0.params['e2'].stderr**2, result1.params['e2'].stderr**2, result2.params['e2'].stderr**2])/9)
    # LastResult['Avg A2+3 E0'] = np.average([result0.params['e2'].value, result1.params['e2'].value, result2.params['e2'].value, result0.params['e3'].value, result1.params['e3'].value, result2.params['e3'].value])
    # LastResult['Avg A2+3 E0 Error'] = np.sqrt(np.sum([result0.params['e2'].stderr**2, result1.params['e2'].stderr**2, result2.params['e2'].stderr**2,result0.params['e3'].stderr**2, result1.params['e3'].stderr**2, result2.params['e3'].stderr**2])/36)
    # LastResult['Avg A3 E0'] = np.average([result0.params['e3'].value, result1.params['e3'].value, result2.params['e3'].value])
    # LastResult['Avg A3 E0 Error'] = np.sqrt(np.sum([result0.params['e3'].stderr**2, result1.params['e3'].stderr**2, result2.params['e3'].stderr**2])/9)
    LastResult['Avg B E0'] = np.average([result0.params['e4'].value, result1.params['e4'].value, result2.params['e4'].value])
    LastResult['Avg B E0 Error'] = np.sqrt(np.sum([result0.params['e4'].stderr**2, result1.params['e4'].stderr**2, result2.params['e4'].stderr**2])/9)
    LastResult['Avg C E0'] = np.average([result0.params['e5'].value, result1.params['e5'].value, result2.params['e5'].value])
    LastResult['Avg C E0 Error'] = np.sqrt(np.sum([result0.params['e5'].stderr**2, result1.params['e5'].stderr**2, result2.params['e5'].stderr**2])/9)
    FitResults[label+"_value"] = [LastResult['Avg FWHM'], \
                                  LastResult['Avg A1 Peak Share'], \
                                LastResult['Avg A2 Peak Share'], \
                                    LastResult['Avg B Peak Share'], \
                                        LastResult['Avg C Peak to Amp Sum'], \
                                        LastResult['Avg A1 E0'], \
                                           LastResult['Avg A2 E0'], \
                                                   LastResult['Avg B E0'], \
                                                       LastResult['Avg C E0']]
        
    FitResults[label+"_error"] = [LastResult['Avg FWHM Error'], \
                                  LastResult['Avg A1 Peak Share Error'], \
                                LastResult['Avg A2 Peak Share Error'], \
                                    LastResult['Avg B Peak Share Error'], \
                                        LastResult['Avg C Peak to Amp Sum Error'], \
                                        LastResult['Avg A1 E0 Error'], \
                                           LastResult['Avg A2 E0 Error'], \
                                                   LastResult['Avg B E0 Error'], \
                                                       LastResult['Avg C E0 Error']]
    print(f"FitResults filled successfully for {label}")
    
FitResults.to_csv("Fitresults_Table.csv", index = False)

#%% FWHM
Labels = ['Cell 1, OCV', 'Cell 1, 50 mA',  'Cell 1, OCV \npost 50 mA', 'Cell 1, 100 mA', 'Cell 1, OCV \npost 100 mA', 'Cell 1, 250 mA', 'Cell 1, OCV \npost 250 mA']

fig, ax = plt.subplots(figsize = (12,12))
rowIndex = FitResults[FitResults['Variable'] == 'FWHM'].index.values[0]

ax.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax.errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
ax.axhline(y = ph_regionBorders['1nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
ax.axhline(y = ph_regionBorders['1nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
ax.axhspan(ph_regionBorders['1nm_ph1_merge_low'][rowIndex], ph_regionBorders['1nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
ax.axhline(y = ph_regionBorders['1nm_ph7_merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
ax.axhline(y = ph_regionBorders['1nm_ph7_merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
ax.axhspan(ph_regionBorders['1nm_ph7_merge_low'][rowIndex], ph_regionBorders['1nm_ph7_merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
ax.axhline(y = ph_regionBorders['1nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
ax.axhline(y = ph_regionBorders['1nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
ax.axhspan(ph_regionBorders['1nm_ph10_merge_low'][rowIndex], ph_regionBorders['1nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
x_ticks_labels = Labels
ax.set_xticklabels(x_ticks_labels, rotation=45, fontsize=24)
ax.tick_params(axis='y', labelsize= 24)
ax.set_ylabel('pseudo-Voigt Average FWHM (eV)', fontsize = 24)

#%% Peak Shares
fig, ax = plt.subplots(2,2,figsize = (24,24))

rowIndex = FitResults[FitResults['Variable'] == 'A1 Amp'].index.values[0]
ax[0,0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,0].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[0,0].axhline(y = ph_regionBorders['1nm_ph1_merge_low'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[0,0].axhline(y = ph_regionBorders['1nm_ph1_merge_high'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[0,0].axhspan(ph_regionBorders['1nm_ph1_merge_low'][rowIndex]*100, ph_regionBorders['1nm_ph1_merge_high'][rowIndex]*100, facecolor='crimson', alpha=0.35)
ax[0,0].axhline(y = ph_regionBorders['1nm_ph7_merge_low'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[0,0].axhline(y = ph_regionBorders['1nm_ph7_merge_high'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[0,0].axhspan(ph_regionBorders['1nm_ph7_merge_low'][rowIndex]*100, ph_regionBorders['1nm_ph7_merge_high'][rowIndex]*100, facecolor='turquoise', alpha=0.35)
ax[0,0].axhline(y = ph_regionBorders['1nm_ph10_merge_low'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[0,0].axhline(y = ph_regionBorders['1nm_ph10_merge_high'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[0,0].axhspan(ph_regionBorders['1nm_ph10_merge_low'][rowIndex]*100, ph_regionBorders['1nm_ph10_merge_high'][rowIndex]*100, facecolor='indigo', alpha=0.35)
ax[0,0].xaxis.set_ticklabels([])
ax[0,0].tick_params(axis='y', labelsize= 36)
ax[0,0].set_ylabel('Average Share of Peak A1 (%)', fontsize = 36)
ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

rowIndexA = FitResults[FitResults['Variable'] == 'A2 Amp'].index.values[0]
# rowIndexB = ph_regionBorders[ph_regionBorders['Variable'] == 'A2+3 Amp'].index.values[0]
# rowIndexB = ph_regionBorders[ph_regionBorders['Variable'] == 'A2 Amp'].index.values[0]
rowIndexB = ph_regionBorders[ph_regionBorders['Variable'] == 'A3 Amp'].index.values[0]
ax[0,1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,1].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[0,1].axhline(y = ph_regionBorders['1nm_ph1_merge_low'][rowIndexB]*100, linestyle = '--', color = 'crimson')
ax[0,1].axhline(y = ph_regionBorders['1nm_ph1_merge_high'][rowIndexB]*100, linestyle = '--', color = 'crimson')
ax[0,1].axhspan(ph_regionBorders['1nm_ph1_merge_low'][rowIndexB]*100, ph_regionBorders['1nm_ph1_merge_high'][rowIndexB]*100, facecolor='crimson', alpha=0.35)
ax[0,1].axhline(y = ph_regionBorders['1nm_ph7_merge_low'][rowIndexB]*100, linestyle = '--', color = 'turquoise')
ax[0,1].axhline(y = ph_regionBorders['1nm_ph7_merge_high'][rowIndexB]*100, linestyle = '--', color = 'turquoise')
ax[0,1].axhspan(ph_regionBorders['1nm_ph7_merge_low'][rowIndexB], ph_regionBorders['1nm_ph7_merge_high'][rowIndexB]*100, facecolor='turquoise', alpha=0.35)
ax[0,1].axhline(y = ph_regionBorders['1nm_ph10_merge_low'][rowIndexB]*100, linestyle = '--', color = 'indigo')
ax[0,1].axhline(y = ph_regionBorders['1nm_ph10_merge_high'][rowIndexB]*100, linestyle = '--', color = 'indigo')
ax[0,1].axhspan(ph_regionBorders['1nm_ph10_merge_low'][rowIndexB]*100, ph_regionBorders['1nm_ph10_merge_high'][rowIndexB]*100, facecolor='indigo', alpha=0.35)
ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0,1].xaxis.set_ticklabels([])
ax[0,1].tick_params(axis='y', labelsize= 36)
ax[0,1].set_ylabel('Average Share of Peak A2 (%)', fontsize = 36)

# rowIndex = FitResults[FitResults['Variable'] == 'A3 Amp'].index.values[0]
# ax[0,2].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
# for label in FitResults.columns[1:]:
#     if str(label).endswith('value'):
#         ax[0,2].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
# ax[0,2].axhline(y = ph_regionBorders['1nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
# ax[0,2].axhline(y = ph_regionBorders['1nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
# ax[0,2].axhspan(ph_regionBorders['1nm_ph1_merge_low'][rowIndex], ph_regionBorders['1nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
# ax[0,2].axhline(y = ph_regionBorders['1nm_ph7_merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[0,2].axhline(y = ph_regionBorders['1nm_ph7_merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[0,2].axhspan(ph_regionBorders['1nm_ph7_merge_low'][rowIndex], ph_regionBorders['1nm_ph7_merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
# ax[0,2].axhline(y = ph_regionBorders['1nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
# ax[0,2].axhline(y = ph_regionBorders['1nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
# ax[0,2].axhspan(ph_regionBorders['1nm_ph10_merge_low'][rowIndex], ph_regionBorders['1nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
# ax[0,2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax[0,2].xaxis.set_ticklabels([])
# ax[0,2].tick_params(axis='y', labelsize= 36)
# ax[0,2].set_ylabel('Average Share of Peak A3 (%)', fontsize = 36)

# rowIndex = FitResults[FitResults['Variable'] == 'A2+3 Amp'].index.values[0]
# ax[1,0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
# for label in FitResults.columns[1:]:
#     if str(label).endswith('value'):
#         ax[1,0].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
# ax[1,0].axhline(y = ph_regionBorders['1nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
# ax[1,0].axhline(y = ph_regionBorders['1nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
# ax[1,0].axhspan(ph_regionBorders['1nm_ph1_merge_low'][rowIndex], ph_regionBorders['1nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
# ax[1,0].axhline(y = ph_regionBorders['1nm_ph7_merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[1,0].axhline(y = ph_regionBorders['1nm_ph7_merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[1,0].axhspan(ph_regionBorders['1nm_ph7_merge_low'][rowIndex], ph_regionBorders['1nm_ph7_merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
# ax[1,0].axhline(y = ph_regionBorders['1nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
# ax[1,0].axhline(y = ph_regionBorders['1nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
# ax[1,0].axhspan(ph_regionBorders['1nm_ph10_merge_low'][rowIndex], ph_regionBorders['1nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
# ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# x_ticks_labels = Labels
# ax[1,0].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
# ax[1,0].tick_params(axis='y', labelsize= 36)
# ax[1,0].set_ylabel('Average Proportion of Sum Peaks A2-3 (%)', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'B Amp'].index.values[0]
ax[1,0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,0].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[1,0].axhline(y = ph_regionBorders['1nm_ph1_merge_low'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,0].axhline(y = ph_regionBorders['1nm_ph1_merge_high'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,0].axhspan(ph_regionBorders['1nm_ph1_merge_low'][rowIndex]*100, ph_regionBorders['1nm_ph1_merge_high'][rowIndex]*100, facecolor='crimson', alpha=0.35)
ax[1,0].axhline(y = ph_regionBorders['1nm_ph7_merge_low'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,0].axhline(y = ph_regionBorders['1nm_ph7_merge_high'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,0].axhspan(ph_regionBorders['1nm_ph7_merge_low'][rowIndex]*100, ph_regionBorders['1nm_ph7_merge_high'][rowIndex]*100, facecolor='turquoise', alpha=0.35)
ax[1,0].axhline(y = ph_regionBorders['1nm_ph10_merge_low'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,0].axhline(y = ph_regionBorders['1nm_ph10_merge_high'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,0].axhspan(ph_regionBorders['1nm_ph10_merge_low'][rowIndex]*100, ph_regionBorders['1nm_ph10_merge_high'][rowIndex]*100, facecolor='indigo', alpha=0.35)
ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
x_ticks_labels = Labels
ax[1,0].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,0].tick_params(axis='y', labelsize= 36)
ax[1,0].set_ylabel('Average Proportion of Peak B (%)', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'C Amp'].index.values[0]
ax[1,1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 8)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,1].errorbar([label], FitResults[label][rowIndex]*100, yerr = FitResults[str(label)[:-6]+'_error'][rowIndex]*100, capsize = 12.5, marker = 'o', markersize = 25)
ax[1,1].axhline(y = ph_regionBorders['1nm_ph1_merge_low'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,1].axhline(y = ph_regionBorders['1nm_ph1_merge_high'][rowIndex]*100, linestyle = '--', color = 'crimson')
ax[1,1].axhspan(ph_regionBorders['1nm_ph1_merge_low'][rowIndex]*100, ph_regionBorders['1nm_ph1_merge_high'][rowIndex]*100, facecolor='crimson', alpha=0.35)
ax[1,1].axhline(y = ph_regionBorders['1nm_ph7_merge_low'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,1].axhline(y = ph_regionBorders['1nm_ph7_merge_high'][rowIndex]*100, linestyle = '--', color = 'turquoise')
ax[1,1].axhspan(ph_regionBorders['1nm_ph7_merge_low'][rowIndex]*100, ph_regionBorders['1nm_ph7_merge_high'][rowIndex]*100, facecolor='turquoise', alpha=0.35)
ax[1,1].axhline(y = ph_regionBorders['1nm_ph10_merge_low'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,1].axhline(y = ph_regionBorders['1nm_ph10_merge_high'][rowIndex]*100, linestyle = '--', color = 'indigo')
ax[1,1].axhspan(ph_regionBorders['1nm_ph10_merge_low'][rowIndex]*100, ph_regionBorders['1nm_ph10_merge_high'][rowIndex]*100, facecolor='indigo', alpha=0.35)
ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
x_ticks_labels = Labels
ax[1,1].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,1].tick_params(axis='y', labelsize= 36)
ax[1,1].set_ylabel('Average Proportion of Peak C (%)', fontsize = 36)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.25, 
                    hspace=0.05)
#%% E0 Positions
fig, ax = plt.subplots(2,3,figsize = (36,24))

rowIndex = FitResults[FitResults['Variable'] == 'A1 E0'].index.values[0]
ax[0,0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,0].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
# ax[0,0].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
# ax[0,0].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
# ax[0,0].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
# ax[0,0].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[0,0].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[0,0].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
# ax[0,0].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
# ax[0,0].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
# ax[0,0].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[0,0].xaxis.set_ticklabels([])
ax[0,0].tick_params(axis='y', labelsize= 36)
ax[0,0].set_ylabel('Average Position of Peak A1', fontsize = 36)
ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

rowIndex = FitResults[FitResults['Variable'] == 'A2 E0'].index.values[0]
ax[0,1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,1].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
# ax[0,1].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
# ax[0,1].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
# ax[0,1].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
# ax[0,1].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[0,1].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[0,1].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
# ax[0,1].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
# ax[0,1].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
# ax[0,1].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0,1].xaxis.set_ticklabels([])
ax[0,1].tick_params(axis='y', labelsize= 36)
ax[0,1].set_ylabel('Average Position of Peak A2', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'A3 E0'].index.values[0]
ax[0,2].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[0,2].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
# ax[0,2].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
# ax[0,2].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
# ax[0,2].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
# ax[0,2].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[0,2].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[0,2].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
# ax[0,2].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
# ax[0,2].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
# ax[0,2].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[0,2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0,2].xaxis.set_ticklabels([])
ax[0,2].tick_params(axis='y', labelsize= 36)
ax[0,2].set_ylabel('Average Position of Peak A3', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'A2+3 E0'].index.values[0]
ax[1,0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,0].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
# ax[1,0].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
# ax[1,0].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
# ax[1,0].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
# ax[1,0].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[1,0].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[1,0].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
# ax[1,0].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
# ax[1,0].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
# ax[1,0].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
x_ticks_labels = Labels
ax[1,0].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,0].tick_params(axis='y', labelsize= 36)
ax[1,0].set_ylabel('Average Position of Sum Peaks A2-3', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'B E0'].index.values[0]
ax[1,1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,1].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
# ax[1,1].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
# ax[1,1].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
# ax[1,1].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
# ax[1,1].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[1,1].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[1,1].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
# ax[1,1].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
# ax[1,1].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
# ax[1,1].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
x_ticks_labels = Labels
ax[1,1].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,1].tick_params(axis='y', labelsize= 36)
ax[1,1].set_ylabel('Average Position of Peak B', fontsize = 36)

rowIndex = FitResults[FitResults['Variable'] == 'C E0'].index.values[0]
ax[1,2].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, 7)])
for label in FitResults.columns[1:]:
    if str(label).endswith('value'):
        ax[1,2].errorbar([label], FitResults[label][rowIndex], yerr = FitResults[str(label)[:-6]+'_error'][rowIndex], capsize = 12.5, marker = 'o', markersize = 25)
# ax[1,2].axhline(y = ph_regionBorders['4nm_ph1_merge_low'][rowIndex], linestyle = '--', color = 'crimson')
# ax[1,2].axhline(y = ph_regionBorders['4nm_ph1_merge_high'][rowIndex], linestyle = '--', color = 'crimson')
# ax[1,2].axhspan(ph_regionBorders['4nm_ph1_merge_low'][rowIndex], ph_regionBorders['4nm_ph1_merge_high'][rowIndex], facecolor='crimson', alpha=0.35)
# ax[1,2].axhline(y = ph_regionBorders['4nm_ph7merge_low'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[1,2].axhline(y = ph_regionBorders['4nm_ph7merge_high'][rowIndex], linestyle = '--', color = 'turquoise')
# ax[1,2].axhspan(ph_regionBorders['4nm_ph7merge_low'][rowIndex], ph_regionBorders['4nm_ph7merge_high'][rowIndex], facecolor='turquoise', alpha=0.35)
# ax[1,2].axhline(y = ph_regionBorders['4nm_ph10_merge_low'][rowIndex], linestyle = '--', color = 'indigo')
# ax[1,2].axhline(y = ph_regionBorders['4nm_ph10_merge_high'][rowIndex], linestyle = '--', color = 'indigo')
# ax[1,2].axhspan(ph_regionBorders['4nm_ph10_merge_low'][rowIndex], ph_regionBorders['4nm_ph10_merge_high'][rowIndex], facecolor='indigo', alpha=0.35)
ax[1,2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
x_ticks_labels = Labels
ax[1,2].set_xticklabels(x_ticks_labels, rotation=45, fontsize=36)
ax[1,2].tick_params(axis='y', labelsize= 36)
ax[1,2].set_ylabel('Average Position of Peak C', fontsize = 36)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.05)