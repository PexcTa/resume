# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:01:15 2022

@author: boris
"""
#%% LIBRARIES
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from scipy.signal import savgol_filter
from scipy import special
# import itertools
# from sympy import S, symbols, printing
from scipy.optimize import curve_fit
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
            data[file] = ReadData(file,skip)
    return data
def RenameDictKeys(dictionary, new_names):
    """
    Renames dictionary keys to those given in new_names, keeps the order
    """
    res = {}
    for i in range(len(new_names)):
        res[new_names[i]] = dictionary[list(dictionary.keys())[i]]
    return res

# UVVIS_data = ReadAllData()
# Data = RenameDictKeys(UVVIS_data, ['100uL', '25uL', '50uL', '75uL'])
#%% compute van de Hulst ADT Qext fraction

def CalculateRadiusFromCylinderVolume(diameter, length):
    volume = math.pi*((0.5*diameter)**2)*length
    radius = np.cbrt(volume*0.75/math.pi)
    return radius

radius = CalculateRadiusFromCylinderVolume(0.5, 1.7)

def ComputeQExtinction(radius, rindexParticle, rindexSolvent, wavelength):
    phase = [(4*math.pi*radius*(rindexParticle - rindexSolvent))/(i/1000) for i in wavelength]
    qext = [2-(4/p)*np.sin(p)+(4/p**2)*(1-np.cos(p)) for p in phase]
    return (phase, qext)

# def ComputeQExtinctionJac(radius, rindexParticle, rindexSolvent, wavelength):
#     radius = nm2ev(radius*1000)
#     wavelength = nm2ev(wavelength)
#     phase = [(4*math.pi*radius*(rindexParticle - rindexSolvent))/(i) for i in wavelength]
#     qext = [2-(4/p)*np.sin(p)+(4/p**2)*(1-np.cos(p)) for p in phase]
#     return (phase, qext)

(p, Q) = ComputeQExtinction(radius, 1.6, 1.326, sample['wl'])

#%% aux functions

def normalize_1(data):
    """Normalizes a vector to its maximum value"""
    normalized_data = np.abs(data)/max(data)
    return normalized_data

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
def jac(wavelength, data):
    e = 1.602*10**(-19)
    h = 4.135667516*10**(-15)
    c = 299792458
    jf = (e*(wavelength*10**-9)**2)/h*c*10**9
    return np.multiply(data, jf)

def gaus1(x,a1,mu1,sigma1,a0):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)+a0)

def gaus2(x,a1,mu1,sigma1,a2,mu2,sigma2,a0):
    return a1*np.exp(-(x-mu1)**2/(2*sigma2**2)+a0)  + \
        a2*np.exp(-(x-mu2)**2/(2*sigma1**2)+a0)
        
def gaus3(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a0):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)+a0) + \
        a2*np.exp(-(x-mu2)**2/(2*sigma2**2)+a0) + \
        a3*np.exp(-(x-mu3)**2/(2*sigma3**2)+a0)

def gaus4(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a0):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)+a0)+ \
        a2*np.exp(-(x-mu2)**2/(2*sigma2**2)+a0) + \
        a3*np.exp(-(x-mu3)**2/(2*sigma3**2)+a0) + \
        a4*np.exp(-(x-mu4)**2/(2*sigma4**2)+a0) 

def gaus5(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,a0):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)+a0) + \
        a2*np.exp(-(x-mu2)**2/(2*sigma2**2)+a0) + \
        a3*np.exp(-(x-mu3)**2/(2*sigma3**2)+a0) + \
        a4*np.exp(-(x-mu4)**2/(2*sigma4**2)+a0) + \
        a5*np.exp(-(x-mu5)**2/(2*sigma5**2)+a0) 

def gaus6(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,a6,mu6,sigma6,a0):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)+a0) + \
        a2*np.exp(-(x-mu2)**2/(2*sigma2**2)+a0) + \
        a3*np.exp(-(x-mu3)**2/(2*sigma3**2)+a0) + \
        a4*np.exp(-(x-mu4)**2/(2*sigma4**2)+a0) + \
        a5*np.exp(-(x-mu5)**2/(2*sigma5**2)+a0) + \
        a6*np.exp(-(x-mu6)**2/(2*sigma6**2)+a0) 

def gaus2_withVDH(X_and_Qext,a1,mu1,sigma1,a2,mu2,sigma2,aQext):
    x, Qext = X_and_Qext
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + np.multiply(aQext,Qext)

def gaus3_withVDH(X_and_Qext,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,aQext):
    x, Qext = X_and_Qext
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + \
        a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + \
        a3*np.exp(-(x-mu3)**2/(2*sigma3**2)) + np.multiply(aQext,Qext)

def gaus4_withVDH(X_and_Qext,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,aQext):
    x, Qext = X_and_Qext
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2))+ \
        a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + \
        a3*np.exp(-(x-mu3)**2/(2*sigma3**2)) + \
        a4*np.exp(-(x-mu4)**2/(2*sigma4**2)) + np.multiply(aQext,Qext)

def gaus5_withVDH(X_and_Qext,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,aQext):
    x, Qext = X_and_Qext
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + \
        a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + \
        a3*np.exp(-(x-mu3)**2/(2*sigma3**2)) + \
        a4*np.exp(-(x-mu4)**2/(2*sigma4**2)) + \
        a5*np.exp(-(x-mu5)**2/(2*sigma5**2)) + np.multiply(aQext,Qext)

def gaus6_withVDH(X_and_Qext,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,a6,mu6,sigma6,aQext):
    x, Qext = X_and_Qext
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + \
        a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + \
        a3*np.exp(-(x-mu3)**2/(2*sigma3**2)) + \
        a4*np.exp(-(x-mu4)**2/(2*sigma4**2)) + \
        a5*np.exp(-(x-mu5)**2/(2*sigma5**2)) + \
        a6*np.exp(-(x-mu6)**2/(2*sigma6**2)) + np.multiply(aQext,Qext)
#%%

def FitGaussians(wavelength, data, gaus_no, initPar, bounds, label, save):
    x = wavelength
    y = data
    fig, ax = plt.subplots(figsize=(10,8))
    if bounds == 0:
        bounds = (-np.inf, np.inf)
    if gaus_no == 1:
        fittedParameters, pcov = curve_fit(gaus1, x, y, initPar, bounds = bounds)
        a1, mu1, sigma1, a0 = fittedParameters
        yplt = gaus1(x, a1, mu1, sigma1, 0)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Offset']
    if gaus_no == 2:
        fittedParameters, pcov = curve_fit(gaus2, x, y, initPar, bounds = bounds)
        a1, mu1, sigma1, a2, mu2, sigma2, a0 = fittedParameters
        yplt = gaus2(x, a1, mu1, sigma1, a2, mu2, sigma2, a0)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Offset']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
    if gaus_no == 3:
        fittedParameters, pcov = curve_fit(gaus3, x, y, initPar, bounds = bounds)
        a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a0 = fittedParameters
        yplt = gaus3(x, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a0)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Offset']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
    if gaus_no == 4: 
        fittedParameters, pcov = curve_fit(gaus4, x, y, initPar, bounds = bounds)
        a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4= fittedParameters
        yplt = gaus4(x, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Amp4', 'Mean4', 'Sigma4']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, gaus1(x, a4, mu4, sigma4, 0), ls = '-.', alpha = 0.75, color = 'dodgerblue')
    if gaus_no == 5: 
        fittedParameters, pcov = curve_fit(gaus5, x, y, initPar, bounds = bounds)
        a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5 = fittedParameters
        yplt = gaus5(x, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4, a5, mu5, sigma5)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Amp4', 'Mean4', 'Sigma4', 'Amp5', 'Mean5', 'Sigma5']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, gaus1(x, a4, mu4, sigma4, 0), ls = '-.', alpha = 0.75, color = 'dodgerblue')
        ax.plot(x, gaus1(x, a5, mu5, sigma5, 0), ls = '-.', alpha = 0.75, color = 'slateblue')
    if gaus_no == 6: 
        fittedParameters, pcov = curve_fit(gaus6, x, y, initPar, bounds = bounds)
        a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,a6,mu6,sigma6 = fittedParameters
        yplt = gaus6(x, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4, a5, mu5, sigma5,a6,mu6,sigma6)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Amp4', 'Mean4', 'Sigma4', 'Amp5', 'Mean5', 'Sigma5', 'Amp6', 'Mean6', 'Sigma6']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, gaus1(x, a4, mu4, sigma4, 0), ls = '-.', alpha = 0.75, color = 'dodgerblue')
        ax.plot(x, gaus1(x, a5, mu5, sigma5, 0), ls = '-.', alpha = 0.75, color = 'slateblue')
        ax.plot(x, gaus1(x, a6, mu6, sigma6, 0), ls = '-.', alpha = 0.75, color = 'darkorchid')
    resid = np.subtract(y, yplt)
    SE = np.square(resid) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(resid) / np.var(y))
    ax.set_xlim([max(x), min(x)])
    ax.set_xlabel("Energy, $cm^{-1}$", fontsize = 20)
    ax.set_ylabel("Normalized Fluorescence", fontsize = 20)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(MultipleLocator(200))
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
    ax.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0,0))
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    ax.axhline(y=0, color = 'black', ls = '--')
    ax.scatter(x,y, color = 'black', s = 40, marker = "D", label = label)
    ax.plot(x,yplt, color = 'purple', linewidth = 3, label = 'Fit')
    ax.scatter(x, np.subtract(y,yplt), color = 'magenta', s = 10, label = 'Residuals')
    ax.legend(loc = 'upper right')
    # print(fittedParameters)
    # print(np.sqrt(np.diag(pcov)))
    dummy = []
    for i in range(len(pars) - 1):
        dummy.append('0')
    dummy.append(str(Rsquared))
    res = pd.DataFrame(
    {'parameter': pars,
     'value': list(fittedParameters),
     'sigma': list(np.sqrt(np.diag(pcov))),
     'R2': dummy,
    })
    if save == 1:
        plt.savefig(str(label)+'GaussFit_'+str(gaus_no)+'gaus.svg')
        res.to_csv(str(label)+'Gaussfit_'+str(gaus_no)+'gaus.csv', index = False)
    return res

def FitGaussiansWithBackground(wavelength, Qext, data, gaus_no, initPar, bounds, label, save):
    x = nm2ev(wavelength)
    y = normalize_1(jac(wavelength, data))
    # jQext = jac(wavelength, Qext)
    X_and_Qext = (x, Qext)
    fig, ax = plt.subplots(figsize=(10,8))
    if bounds == 0:
        bounds = (-np.inf, np.inf)
    # if gaus_no == 1:
    #     fittedParameters, pcov = curve_fit(gaus1, x, y, initPar, bounds = bounds)
    #     a1, mu1, sigma1, a0 = fittedParameters
    #     yplt = gaus1(x, a1, mu1, sigma1, 0)
    #     pars = ['Amp1', 'Mean1', 'Sigma1', 'Offset']
    if gaus_no == 2:
        fittedParameters, pcov = curve_fit(gaus2_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1, mu1, sigma1, a2, mu2, sigma2, aQext = fittedParameters
        yplt = gaus2_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, aQext)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    if gaus_no == 3:
        fittedParameters, pcov = curve_fit(gaus3_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, aQext = fittedParameters
        yplt = gaus3_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, aQext)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    if gaus_no == 4: 
        fittedParameters, pcov = curve_fit(gaus4_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,aQext= fittedParameters
        yplt = gaus4_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Amp4', 'Mean4', 'Sigma4', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, gaus1(x, a4, mu4, sigma4, 0), ls = '-.', alpha = 0.75, color = 'dodgerblue')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    if gaus_no == 5: 
        fittedParameters, pcov = curve_fit(gaus5_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,aQext = fittedParameters
        yplt = gaus5_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4, a5, mu5, sigma5)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Amp4', 'Mean4', 'Sigma4', 'Amp5', 'Mean5', 'Sigma5', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, gaus1(x, a4, mu4, sigma4, 0), ls = '-.', alpha = 0.75, color = 'dodgerblue')
        ax.plot(x, gaus1(x, a5, mu5, sigma5, 0), ls = '-.', alpha = 0.75, color = 'slateblue')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    if gaus_no == 6: 
        fittedParameters, pcov = curve_fit(gaus6_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,a6,mu6,sigma6,aQext = fittedParameters
        yplt = gaus6_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4, a5, mu5, sigma5,a6,mu6,sigma6)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Amp4', 'Mean4', 'Sigma4', 'Amp5', 'Mean5', 'Sigma5', 'Amp6', 'Mean6', 'Sigma6', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, gaus1(x, a4, mu4, sigma4, 0), ls = '-.', alpha = 0.75, color = 'dodgerblue')
        ax.plot(x, gaus1(x, a5, mu5, sigma5, 0), ls = '-.', alpha = 0.75, color = 'slateblue')
        ax.plot(x, gaus1(x, a6, mu6, sigma6, 0), ls = '-.', alpha = 0.75, color = 'darkorchid')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    resid = np.subtract(y, yplt)
    SE = np.square(resid) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(resid) / np.var(y))
    ax.set_xlim([max(x), min(x)])
    ax.set_xlabel("Wavelength (eV)", fontsize = 24)
    ax.set_ylabel("Normalized Absorption", fontsize = 24)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    # secax = ax.secondary_xaxis('top', functions=(WN2nm, nm2WN))
    # secax.tick_params(labelsize = 20)
    # secax.set_xlabel('Wavelength (nm)', fontsize = 24, labelpad = 10)
    ax.tick_params(axis='x', labelsize= 20)
    ax.tick_params(axis='y', labelsize= 20)
    # ax.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0,0))
    # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    # ax.axhline(y=0, color = 'black', ls = '--')
    ax.scatter(x,y, color = 'black', s = 40, marker = "D", label = label)
    ax.plot(x,yplt, color = 'purple', linewidth = 3, label = 'Fit')
    ax.scatter(x, np.subtract(y,yplt), color = 'magenta', s = 10, label = 'Residuals')
    ax.legend(loc = 'upper right', fontsize=24)
    # print(fittedParameters)
    # print(np.sqrt(np.diag(pcov)))
    dummy = []
    for i in range(len(pars) - 1):
        dummy.append('0')
    dummy.append(str(Rsquared))
    res = pd.DataFrame(
    {'parameter': pars,
     'value': list(fittedParameters),
     'sigma': list(np.sqrt(np.diag(pcov))),
     'R2': dummy,
    })
    if save == 1:
        plt.savefig(str(label)+'GaussFit_'+str(gaus_no)+'gaus.svg')
        res.to_csv(str(label)+'Gaussfit_'+str(gaus_no)+'gaus.csv', index = False)
    return res

x = sample['wl'][75:]
y = sample['R'][75:]
# initPar = [0.1, 31250, 1000, 0.1, 23800, 1000, 0.01]
initPar = [1, 3.87, 0.1, 1, 2.95, 0.1, 0.01]
test = FitGaussiansWithBackground(x, Q[75:], y, 2, initPar, 0, 'NU-1000', 0)
# FitGaussians(x, normalize_1(y), 2, initPar, 0, 'NU-1000', 0)


#%% alternative approach: fit van de hulst background in wavelength space, then subtract it and fit the rest to
#   gauusians in energy space

def FitGaussiansWithBackground(wavelength, Qext, data, gaus_no, initPar, bounds, label, save):
    x = wavelength
    y = normalize_1(data)
    X_and_Qext = (x, Qext)
    fig, ax = plt.subplots(figsize=(10,8))
    if bounds == 0:
        bounds = (-np.inf, np.inf)
    # if gaus_no == 1:
    #     fittedParameters, pcov = curve_fit(gaus1, x, y, initPar, bounds = bounds)
    #     a1, mu1, sigma1, a0 = fittedParameters
    #     yplt = gaus1(x, a1, mu1, sigma1, 0)
    #     pars = ['Amp1', 'Mean1', 'Sigma1', 'Offset']
    if gaus_no == 2:
        fittedParameters, pcov = curve_fit(gaus2_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1, mu1, sigma1, a2, mu2, sigma2, aQext = fittedParameters
        yplt = gaus2_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, aQext)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    if gaus_no == 3:
        fittedParameters, pcov = curve_fit(gaus3_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, aQext = fittedParameters
        yplt = gaus3_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, aQext)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    if gaus_no == 4: 
        fittedParameters, pcov = curve_fit(gaus4_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,aQext= fittedParameters
        yplt = gaus4_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Amp4', 'Mean4', 'Sigma4', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, gaus1(x, a4, mu4, sigma4, 0), ls = '-.', alpha = 0.75, color = 'dodgerblue')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    if gaus_no == 5: 
        fittedParameters, pcov = curve_fit(gaus5_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,aQext = fittedParameters
        yplt = gaus5_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4, a5, mu5, sigma5)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Amp4', 'Mean4', 'Sigma4', 'Amp5', 'Mean5', 'Sigma5', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, gaus1(x, a4, mu4, sigma4, 0), ls = '-.', alpha = 0.75, color = 'dodgerblue')
        ax.plot(x, gaus1(x, a5, mu5, sigma5, 0), ls = '-.', alpha = 0.75, color = 'slateblue')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    if gaus_no == 6: 
        fittedParameters, pcov = curve_fit(gaus6_withVDH, (x,Qext), y, initPar, bounds = bounds)
        a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,a6,mu6,sigma6,aQext = fittedParameters
        yplt = gaus6_withVDH(X_and_Qext, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4, a5, mu5, sigma5,a6,mu6,sigma6)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Amp3', 'Mean3', 'Sigma3', 'Amp4', 'Mean4', 'Sigma4', 'Amp5', 'Mean5', 'Sigma5', 'Amp6', 'Mean6', 'Sigma6', 'AmpQext']
        ax.plot(x, gaus1(x, a1, mu1, sigma1, 0), ls = '-.', alpha = 0.75, color = 'mediumaquamarine')
        ax.plot(x, gaus1(x, a2, mu2, sigma2, 0), ls = '-.', alpha = 0.75, color = 'darkturquoise')
        ax.plot(x, gaus1(x, a3, mu3, sigma3, 0), ls = '-.', alpha = 0.75, color = 'darkcyan')
        ax.plot(x, gaus1(x, a4, mu4, sigma4, 0), ls = '-.', alpha = 0.75, color = 'dodgerblue')
        ax.plot(x, gaus1(x, a5, mu5, sigma5, 0), ls = '-.', alpha = 0.75, color = 'slateblue')
        ax.plot(x, gaus1(x, a6, mu6, sigma6, 0), ls = '-.', alpha = 0.75, color = 'darkorchid')
        ax.plot(x, np.multiply(aQext,Qext), ls = '--', alpha = 0.75, color = 'slategrey')
    resid = np.subtract(y, yplt)
    SE = np.square(resid) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(resid) / np.var(y))
    ax.set_xlim([max(x), min(x)])
    ax.set_xlabel("Wavelength (eV)", fontsize = 24)
    ax.set_ylabel("Normalized Absorption", fontsize = 24)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    # secax = ax.secondary_xaxis('top', functions=(WN2nm, nm2WN))
    # secax.tick_params(labelsize = 20)
    # secax.set_xlabel('Wavelength (nm)', fontsize = 24, labelpad = 10)
    ax.tick_params(axis='x', labelsize= 20)
    ax.tick_params(axis='y', labelsize= 20)
    # ax.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0,0))
    # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    # ax.axhline(y=0, color = 'black', ls = '--')
    ax.scatter(x,y, color = 'black', s = 40, marker = "D", label = label)
    ax.plot(x,yplt, color = 'purple', linewidth = 3, label = 'Fit')
    ax.plot(x,y-np.multiply(Qext,aQext), color = 'orange', linewidth = 3, label = 'True')
    ax.scatter(x, np.subtract(y,yplt), color = 'magenta', s = 10, label = 'Residuals')
    ax.legend(loc = 'upper right', fontsize=24)
    # print(fittedParameters)
    # print(np.sqrt(np.diag(pcov)))
    dummy = []
    for i in range(len(pars) - 1):
        dummy.append('0')
    dummy.append(str(Rsquared))
    res = pd.DataFrame(
    {'parameter': pars,
     'value': list(fittedParameters),
     'sigma': list(np.sqrt(np.diag(pcov))),
     'R2': dummy,
    })
    if save == 1:
        plt.savefig(str(label)+'GaussFit_'+str(gaus_no)+'gaus.svg')
        res.to_csv(str(label)+'Gaussfit_'+str(gaus_no)+'gaus.csv', index = False)
    return res

x = sample['wl'][75:]
y = sample['R'][75:]
# initPar = [0.1, 31250, 1000, 0.1, 23800, 1000, 0.01]

lower_bound = [0, 300, 25, 0, 400, 25, 0]
upper_bound = [np.inf, 400, 100, np.inf, 500, 100,  np.inf]
bounds= (lower_bound, upper_bound)

initPar = [1, 320, 50, 1, 420, 50, 0.01]
test = FitGaussiansWithBackground(x, Q[75:], y, 2, initPar, bounds, 'NU-1000', 0)
# FitGaussians(x, normalize_1(y), 2, initPar, 0, 'NU-1000', 0)