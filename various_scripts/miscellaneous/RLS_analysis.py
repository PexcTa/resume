# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:31:05 2022

@author: boris
"""

import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from sympy import S, symbols, printing
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter as sgf
import os
import ast
from lmfit import Model

#%%
def ReadData(filename):
    with open(filename) as current_file:
        d = pd.read_csv(current_file, sep = ',', header = 0, engine = 'python')
    return d
def ReadAbsData(filename, skip=2):
    with open(filename) as current_file:
        d = pd.read_csv(current_file, names = ['wl', 'R'], delim_whitespace = True, engine = 'python', skiprows = lambda x: x in range(skip))
        return d
def readPhDict(filename):
    with open(filename) as current_file:
        dataframe = pd.read_csv(current_file, names = ['id', 'ph'], sep = ',', engine = 'python')
        dataframe.set_index('id',inplace=True)
        dictionary = dataframe.to_dict()['ph']
        return dictionary
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
def saver(dictex):
    for key, val in dictex.items():
        val.to_csv("data_{}.csv".format(str(key)))
    with open("keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))
        
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

def gaus2_withVDH_andRLS(X_and_Qext,a1,mu1,sigma1,a2,mu2,sigma2,aQext,a,b):
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
s1r1 = ReadData('s1_over_r1.csv')
s1c = ReadData('s1_corrected.csv')
absDict= {}
for file in os.listdir('./absorption'):
    absDict[file] = ReadAbsData('./absorption/'+str(file))
absDictRebinned= {}
for key in absDict.keys():
    step = 2
    x = np.array([])
    y = np.array([])
    for i,j in zip(range(0,1000,step), range(step,1002,step)):
        x = np.append(x, absDict[key]['wl'][i])
        y = np.append(y, np.sum(absDict[key]['R'].iloc[i:j]))
    temp = np.transpose(np.concatenate((x[None,:],y[None,:]),axis=0))
    absDictRebinned[str(key)[0]+str(key)[5]] = pd.DataFrame(temp, columns = ['wl', 'E'])
del (file, key, x, y, temp, step, i, j)
#%% 
fig, ax = plt.subplots(figsize=(12,12))
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.tick_params(axis='x', labelsize= 25)
ax.tick_params(axis='y', labelsize= 25)
ax.axhline(y=0, color = 'dimgrey', ls = '--')
ax.set_ylabel('RLS Intensity (CPS)', fontsize = 30)
ax.set_facecolor('gray')
ax.set_xlabel('Wavelength (nm)', fontsize = 30)
ax.set_prop_cycle('color',[plt.cm.inferno(i) for i in np.linspace(0, 1, 7)])
for column in s1c.columns[2:]:
    ax.plot(s1c['wl'], s1c[column]-s1c['bg'], label = str(column), linewidth = 3, )
    
    
#%%
dset = "p6"

fittingStartPoint1 = 199
fittingEndPoint1 = 255

fittingStartPoint2 = 310
fittingEndPoint2 = 340

fittingStartPoint3 = 260
fittingEndPoint3 = 285

def Linear(rls, a, b):
    return a*rls + b
ext1 = np.array(absDictRebinned[dset]['E'][fittingStartPoint1:fittingEndPoint1])
rls1 = np.array(s1r1[dset][fittingStartPoint1:fittingEndPoint1]-s1r1['bg'][fittingStartPoint1:fittingEndPoint1])
y1 = ext1
linModel1 = Model(Linear)
params1 = linModel1.make_params()
result1 = linModel1.fit(y1,rls = rls1, a = 0.000000001, b = 0.00005)
print(result1.fit_report())


ext2 = np.array(absDictRebinned[dset]['E'][fittingStartPoint2:fittingEndPoint2])
rls2 = np.array(s1r1[dset][fittingStartPoint2:fittingEndPoint2]-s1r1['bg'][fittingStartPoint2:fittingEndPoint2])
y2 = ext2
linModel2 = Model(Linear)
params2 = linModel2.make_params()
result2 = linModel2.fit(y2,rls = rls2, a = 0.000000001, b = 0.00005)
print(result2.fit_report())

ext3 = np.array(absDictRebinned[dset]['E'][fittingStartPoint3:fittingEndPoint3])
rls3 = np.array(s1r1[dset][fittingStartPoint3:fittingEndPoint3]-s1r1['bg'][fittingStartPoint3:fittingEndPoint3])
y3 = ext3
linModel3 = Model(Linear)
params3 = linModel3.make_params()
result3 = linModel3.fit(y3,rls = rls3, a = 0.000000001, b = 0.00005)
print(result3.fit_report())

plotStartIndex = 200

fig, ax = plt.subplots(figsize=(16,16))
ax.scatter(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1], absDictRebinned[dset]['E'][plotStartIndex:], s = 30, facecolors = 'none', color = 'black')
ax.ylim = (min(absDictRebinned[dset]['E'][plotStartIndex:]), max(absDictRebinned[dset]['E'][plotStartIndex:]))
ax.xlim = (min(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1]), max(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1]))
ax.set_xlabel('Scattering (CPS)', fontsize = 36)
ax.set_ylabel('Extinction (AU)', fontsize = 36)
ax.tick_params(axis='x', labelsize= 28)
ax.tick_params(axis='y', labelsize= 28)
for i in np.linspace(199,499,13):
    #vertical grid lines
    # ax.axvline(x = s1r1[dset][i])
    # ax.annotate(str(s1r1['wl'][i]),(s1r1[dset][i], (max(ax.ylim)+min(ax.ylim))/2-((max(ax.ylim)+min(ax.ylim))/2)*0.02*i/100), fontsize=20)
    #horisontal grid lines
    ax.axhline(y = absDictRebinned[dset]['E'][i], linestyle = '-.', alpha = 0.75)
    ax.annotate(str(int(absDictRebinned[dset]['wl'][i]))+" nm",((max(ax.xlim)+min(ax.xlim))/6,absDictRebinned[dset]['E'][i]), fontsize=20)

ax.plot(rls1, result1.best_fit, '-', linewidth = 6, color = 'navy')
ax.plot(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1], Linear(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1], result1.params['a'], result1.params['b']), linestyle = '--', linewidth = 3, color = 'slategrey')
ax.annotate("a = "+f"{result1.params['a'].value:.2e}"+", b = "+f"{result1.params['b'].value:.2f}", (1*10**(7), 0.760), fontsize = 22)

ax.plot(rls2, result2.best_fit, '-', linewidth = 6, color = 'slateblue')
ax.plot(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1], Linear(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1], result2.params['a'], result2.params['b']), linestyle = '--', linewidth = 3, color = 'slategrey')
ax.annotate("a = "+f"{result2.params['a'].value:.3e}"+", b = "+f"{result2.params['b'].value:.3f}", (1.25*10**(7), 0.685), fontsize = 22)

ax.plot(rls3, result3.best_fit, '-', linewidth = 6, color = 'darkturquoise')
ax.plot(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1], Linear(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1], result3.params['a'], result3.params['b']), linestyle = '--', linewidth = 3, color = 'slategrey')
ax.annotate("a = "+f"{result3.params['a'].value:.3e}"+", b = "+f"{result3.params['b'].value:.3f}", (1.50*10**(7), 0.730), fontsize = 22)

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

(p, Q) = ComputeQExtinction(radius, 1.6, 1.414, absDictRebinned['p1']['wl'])

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

def FitGaussiansWithVDHandRLS(wavelength, Qext, a, b, rls, data, gaus_no, initPar, bounds, label, save):
    x = nm2ev(wavelength)
    y = normalize_1(jac(wavelength, data))
    jQext = jac(wavelength, Qext)
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

#%%
def gaus2_withVDH_andRLS(x,Qext,rls,a,b,a1,mu1,sigma1,a2,mu2,sigma2,ampQext,ampRLS):
    return a1*np.exp((-(x-mu1)**2)/(2*sigma1**2)) + a2*np.exp((-(x-mu2)**2)/(2*sigma2**2)) + np.multiply(ampQext,Q) + ampRLS*(np.multiply(rls,a) + b)


# startIndex = 0
ExtinctionModel = Model(gaus2_withVDH_andRLS, independent_vars=['x', 'Qext', 'rls', 'a', 'b'])
params = ExtinctionModel.make_params(
    a1 = 0.1, 
    mu1 = 400,
    sigma1 = 10, 
    a2 = 0.1, 
    mu2 = 320, 
    sigma2 = 10, 
    ampQext = 1, 
    ampRLS = 10,
    )
ExtinctionModel.set_param_hint('mu1', min=380, max=430)
ExtinctionModel.set_param_hint('mu2', min=300, max=330)
ExtinctionModel.set_param_hint('sigma1', min=1, max=100)
ExtinctionModel.set_param_hint('sigma2', min=1, max=100)
ExtinctionModel.set_param_hint('a1', min=0)
ExtinctionModel.set_param_hint('a2', min=0)
ExtinctionModel.set_param_hint('ampQext', min=0)
ExtinctionModel.set_param_hint('ampRLS', min=0)
y = np.array(absDictRebinned[dset]['E'])
result0 = ExtinctionModel.fit(y,params,
                        x = np.array(absDictRebinned[dset]['wl']),
                        Qext = np.array(Q), 
                        rls = np.array((s1r1[dset]-s1r1['bg']))[:-1],
                        a = -1.83*10**(-9), 
                        b = 0.81)
print(result0.fit_report())
fig, ax = plt.subplots(figsize=(16,16))
x = np.array(absDictRebinned[dset]['wl'])
ax.plot(x,y)
ax.plot(np.array(absDictRebinned[dset]['wl']), result0.best_fit, '-', linewidth = 3, color = 'slateblue')
# ax.plot(x, resonantLightScattering(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1], absDictRebinned[dset]['E'][plotStartIndex:], -1.83*10**(-9), 0.81, 1))
ax.plot(x,np.multiply(result0.params['ampQext'],Q))
ax.plot(x,result0.params["ampRLS"]*(np.multiply(np.array((s1r1[dset]-s1r1['bg']))[:-1],-1.83*10**(-9)) +  0.81))
ax.plot(x,gaus1(x,result0.params['a1'],result0.params['mu1'],result0.params['sigma1'],0))
ax.plot(x,gaus1(x,result0.params['a2'],result0.params['mu2'],result0.params['sigma2'],0))
#%%
def resonantLightScattering(rls, ext, a, b, Ls):
    return np.add(np.multiply(a, rls), b) * np.add(1, np.multiply(2.3026,ext)*Ls)

plotStartIndex = 0

fig, ax = plt.subplots(figsize = (12,12))
for Ls in np.linspace (0.9, 1.1, 21):
    y = resonantLightScattering(s1r1[dset][plotStartIndex:-1]-s1r1['bg'][plotStartIndex:-1], absDictRebinned[dset]['E'][plotStartIndex:], -1.83*10**(-9), 0.81, Ls)
    # jac_y = jac(absDictRebinned[dset]['wl'][plotStartIndex:], y)
    # jac_x = nm2ev(absDictRebinned[dset]['wl'][plotStartIndex:])
    ax.plot(absDictRebinned[dset]['wl'][plotStartIndex:], y, label = str(Ls))
    # ax.plot(jac_x, jac_y, label = str(Ls))
ax.legend()

#%%
def gaus2_withVDH_andRLS(x,Qext,rls,a,b,a1,mu1,sigma1,a2,mu2,sigma2,ampQext,ampRLS):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + np.multiply(ampQext,Qext) + ampRLS*(np.add(np.multiply(rls,a),b)) 

fig, ax = plt.subplots(figsize = (12,12))
ax.plot(absDictRebinned[dset]['wl'], gaus2_withVDH_andRLS(x=absDictRebinned[dset]['wl'], Qext=Q, rls=s1r1[dset]-s1r1['bg'], a=4.299*10**(-10), b=0.255, a1=0.1, mu1=400, sigma1=10, a2=0.1, mu2=320, sigma2=10, ampQext=1, ampRLS=10)[:-1])

test = gaus2_withVDH_andRLS(x=absDictRebinned[dset]['wl'], Qext=Q, rls=s1r1[dset]-s1r1['bg'], a=4.299*10**(-10), b=0.255, a1=0.1, mu1=400, sigma1=10, a2=0.1, mu2=320, sigma2=10, ampQext=1, ampRLS=10)[:-1]
