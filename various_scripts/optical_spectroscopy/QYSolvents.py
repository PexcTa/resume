# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:36:12 2020

@author: boris
"""
import csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from sympy import S, symbols, printing
from scipy.optimize import curve_fit

import os
#%% Define normalization function
def normalize_1(data):
    """Normalizes a vector to it's maximum value"""
    normalized_data = data/max(data)
    return normalized_data

def ReadQYData(filepath):
    with open(filepath) as current_file:
        file = pd.read_csv(current_file, sep = ",", header = 0, engine = 'python')
        return file

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def ReadAllData():
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".csv"):
            data[file] = ReadQYData(file)
    return data

def ComputeAreas(data):
    areas = {}
    for key in data.keys():
        dset = data[key]
        temp = []
        for column in ['d1', 'd2', 'd3', 'd4', 'd5']:
            temp.append(np.trapz(dset[column], dx = 1) - np.trapz(dset['bg'], dx = 1))
            areas[key] = temp
    return areas

def GetAbsorptionValues(data, rng):
    data_files = os.listdir()
    abs_val = []
    abs_dict = {}
    for file in data_files:
        if file.endswith(".txt"):
            with open(file) as current_file:
                temp = pd.read_csv(current_file, delim_whitespace = True, header = None, skiprows = 2, engine = 'python')
                abs_val.append(temp.iloc[40,1] - temp.iloc[350,1])
                # abs_val.append(temp.iloc[40,1])
    for i in range(rng):
        abs_dict[list(data.keys())[i]] = list(chunks(abs_val,5))[i]
    return abs_dict

def GetAbsRawData(): 
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".txt"):
            with open(file) as current_file:
                data[file] = pd.read_csv(current_file, skiprows = 2, header = None, delim_whitespace = True, engine = 'python')
    return data

def nm2ev(data_in_nm):
    '''
    Parameters
    ----------
    data_in_nm : array of int or float
        Converts the energy scale from nanometers to eV

    Returns : array of the same size, converted
    -------
    None.
    '''
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
#%%
data = ReadAllData()
#%%
areas = ComputeAreas(data)
absor = GetAbsorptionValues(data, 31)
#%%
abs_raw = GetAbsRawData()
def RenameDictKeys(dictionary, new_names):
    res = {}
    for i in range(10):
        res[new_names[i]] = dictionary[list(dictionary.keys())[i]]
    return res
abs_raw = RenameDictKeys(abs_raw, ['cou102_d1','cou102_d2','cou102_d3','cou102_d4','cou102_d5',
                             'cou151_d1','cou151_d2','cou151_d3','cou151_d4','cou151_d5'])
#%% correct baseline individually. 
def GetAbsVal(data):
    temp = pd.DataFrame()
    for index in [10,20,30,40]:
        abs_vector = []
        for key in abs_raw: 
            wl = int(abs_raw[key].iloc[index,0])
            if '102' in str(key):
                lab = 'cou' + '_102_' +str(wl)
                abs_vector.append(abs_raw[key].iloc[index,1])
                if len(abs_vector) == 5:
                    temp[str(lab)] = abs_vector
                    abs_vector = []
                print(abs_vector)
            elif '151' in str(key):
                lab = 'cou' + '_151_' +str(wl)
                abs_vector.append(abs_raw[key].iloc[index,1])
                if len(abs_vector) == 5:
                    temp[str(lab)] = abs_vector
                    abs_vector = []
                print(abs_vector)
    return temp
absor = GetAbsVal(abs_raw)

#%%
# labels = ['d-NU-1000 (CF3T)', 'd-NU-1000 (DMF)', 'd-Ni-SIM-1c (CF3T)', \
#           'd-Ni-SIM-1c (DMF)', 'd-Ti-SIM-1c (CF3T)', 'd-Ti-SIM-1c (DMF)',\
#           'd-Ti-SIM-1c (DMSO)', 'd-Ti-SIM-1c (EtOH)','d-Ti-SIM-1c (hep)',\
#           'd-Ti-SIM-1c (pyr)', 'd-Ti-SIM-1c (water)', 'pp-NU-1000 (1) (CF3T)',\
#           'pp-NU-1000 (1) (DMF)', 'pp-NU-1000 (2) (DMF)', 'pp-NU-1000 (2) (water)',\
#           'pp-NU-1000-scc (2) (CF3T)', 'pp-NU-1000-scc (2) (DMF)', 'pp-NU-1000-vac (2) (DMF)',\
#             'pp-Co-SIM-1c (CF3T)', 'pp-Co-SIM-1c (DMF)',\
#           'pp-Ti-SIM-1c (CF3T)', 'pp-Ti-SIM-1c (DMF)',\
#           'Coumarin 102' , 'Coumarin 151', 'Coumarin 30',\
#           'pp-Ni-SIM-1c (CF3T)', 'pp-Ni-SIM-1c (DMF)','pp-Ni-SIM-2c (CF3T)',\
#            'pp-Ni-SIM-2c (DMF)','pp-Ti-SIM-2c (CF3T)', 'pp-Ti-SIM-2c (DMF)']

def RenameDictKeys(dictionary, new_names):
    res = {}
    for i in range(12):
        res[new_names[i]] = dictionary[list(dictionary.keys())[i]]
    return res
data = RenameDictKeys(data, ['cou102_d1','cou102_d2','cou102_d3','cou102_d4','cou102_d5',
                             'cou151_d1','cou151_d2','cou151_d3','cou151_d4','cou151_d5',
                             'bg102','bg151'])
#%%
def CorrectBG(data, *args):
    corr_data = {}
    for df in data.keys(): 
        corr_data[df] = pd.DataFrame()
        for arg in args:
            bgkey = 'bg'+str(arg)
            if str(arg) in str(df):
                corr_data[df] = np.subtract(data[df],data[bgkey])
        corr_data[df]['wl'] = data[df]['wl']
    for arg in args:
        bgkey = 'bg'+str(arg)
        corr_data.pop(bgkey, None)
    return corr_data
corr_data = CorrectBG(data, 102, 151)        
    

#%%
def ComputeAreas(data):
    areas = {}
    for key in data:
        for column in data[key].columns[1:]:
            wale = int(column)+20
            id_wl = data[key][data[key]['wl'] == wale].index.values[0]
            wl_vector = nm2WN(data[key]['wl'][id_wl:])
            dt_vector = data[key][column][id_wl:]
            label = str(key)+'_'+str(column)
            areas[label] = np.trapz(y = dt_vector, x = wl_vector[::-1])
    temp = pd.DataFrame()
    
    for col in ['360','370','380','390']:
        area_vector = []
        for key in areas:
            if '102' in str(key) and str(col) in str(key):
                lab  = 'cou'+'_102_'+str(col)
                area_vector.append(areas[key])
                if len(area_vector) == 5:
                    temp[str(lab)] = area_vector
                    area_vector = []
                print(area_vector)
            elif '151' in str(key) and str(col) in str(key):
                lab  = 'cou'+'_151_'+str(col)
                area_vector.append(areas[key])
                if len(area_vector) == 5:
                    temp[str(lab)] = area_vector
                    area_vector = []
                print(area_vector)
    return temp
areas = ComputeAreas(corr_data)
#%% Overlaid Average Emission
def PlotLinearQYFits(areas_dict, absor_dict, labels):
    fig, ax = plt.subplots(figsize = (10,6.6))
    for key in labels:
        x = absor_dict[key]
        y = areas_dict[key]
        coeff, covar = np.polyfit(x,y,1, cov= True)
        a = coeff[0]
        b = coeff[1]
        ye = np.sqrt(covar[1][1])
        ax.scatter(x, y, s = 15, label = str(key))
        ax.plot(np.unique(x), np.unique(x)*a + b, label = None, linewidth = 0.5)
        # ax.errorbar(np.unique(x), np.unique(x)*a + b, yerr = ye, ecolor = 'black', fmt = 'none', capsize = 2, label = None)
    ax.legend(loc = 'upper left')
    ax.set_xlim([0, 0.08])
    ax.set_xlabel("Absorbance", fontsize = 20)
    ax.set_ylabel("Integrated Fluorescence", fontsize = 20)
def ComputeLinearQYFits(areas_dict, absor_dict, labels):
    fits = {}
    for key in labels:
        x = absor_dict[key]
        y = areas_dict[key]
        coeff, covar = np.polyfit(x, y, 1, cov = True)
        # r-squared
        p = np.poly1d(coeff)
        # fit values, and mean
        yhat = p(x)                         # or [p(z) for z in x]
        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
        sqrR= ssreg / sstot
        error = np.sqrt(np.diag(covar))
        fits[key] = [coeff, error, sqrR]
    return fits
def CalculateQY(QYRef, Ref, Sample, RefSolvent, SampleSolvent, fits, solvents):
    x = solvents[solvents['Solvent']==SampleSolvent].index.values
    RindX = solvents['Rind'][x]
    xref = solvents[solvents['Solvent']==RefSolvent].index.values
    RindRef = solvents['Rind'][xref]
    GradRef = fits[Ref][0][0]
    GradRefError = fits[Ref][1][0]/GradRef
    GradX = fits[Sample][0][0]
    GradXError = fits[Sample][1][0]/GradX
    QYX = QYRef*(GradX/GradRef)*((RindX.iloc[0]**2) / (RindRef.iloc[0]**2))
    QYE = (np.sqrt(GradRefError**2 + GradXError**2))*QYX
    return [QYX, QYE]
#%%
labs_dNiTi = ['d-NU-1000 (CF3T)', 'd-NU-1000 (DMF)', 'd-Ni-SIM-1c (CF3T)', 'd-Ni-SIM-1c (DMF)',\
          'd-Ti-SIM-1c (CF3T)', 'd-Ti-SIM-1c (DMF)']
labs_only_dTi = [
          'd-Ti-SIM-1c (CF3T)', 'd-Ti-SIM-1c (DMF)', 'd-Ti-SIM-1c (DMSO)', 'd-Ti-SIM-1c (EtOH)',\
                 'd-Ti-SIM-1c (pyr)', 'd-Ti-SIM-1c (water)', 'd-Ti-SIM-1c (hep)']
labs_only_ppNi = ['pp-NU-1000-scc (2) (CF3T)', 'pp-NU-1000-scc (2) (DMF)', \
                  'pp-Ni-SIM-2c (CF3T)', 'pp-Ni-SIM-2c (DMF)',\
                      'pp-Ni-SIM-1c (CF3T)', 'pp-Ni-SIM-1c (DMF)']
labs_only_ppCo = ['pp-NU-1000-scc (2) (CF3T)', 'pp-NU-1000-scc (2) (DMF)', \
                  'pp-Co-SIM-1c (CF3T)', 'pp-Co-SIM-1c (DMF)']
labs_only_ppTi = ['pp-NU-1000-scc (2) (CF3T)', 'pp-NU-1000-scc (2) (DMF)',\
          'pp-Ti-SIM-1c (CF3T)', 'pp-Ti-SIM-1c (DMF)','pp-Ti-SIM-2c (CF3T)', 'pp-Ti-SIM-2c (DMF)']
labs_onlybase = ['d-NU-1000 (CF3T)', 'd-NU-1000 (DMF)', 'pp-NU-1000 (1) (CF3T)', \
                 'pp-NU-1000-scc (2) (CF3T)', 'pp-NU-1000-scc (2) (DMF)', 'pp-NU-1000-vac (2) (DMF)',\
                     'pp-NU-1000 (1) (DMF)', 'pp-NU-1000 (2) (DMF)', 'pp-NU-1000 (2) (water)']
labs_ppNiTiCo_all = ['pp-NU-1000-scc (2) (CF3T)', 'pp-NU-1000-scc (2) (DMF)', \
               'pp-Ni-SIM-1c (CF3T)', 'pp-Ni-SIM-1c (DMF)',\
                   'pp-Ti-SIM-1c (CF3T)', 'pp-Ti-SIM-1c (DMF)',\
                             'pp-Ni-SIM-2c (CF3T)', 'pp-Ni-SIM-2c (DMF)',\
                                 'pp-Co-SIM-1c (CF3T)', 'pp-Co-SIM-1c (DMF)',\
                            'pp-Ti-SIM-2c (CF3T)', 'pp-Ti-SIM-2c (DMF)']
labs_ppNiTiCo_1c = ['pp-NU-1000-scc (2) (CF3T)', 'pp-NU-1000-scc (2) (DMF)', \
               'pp-Ni-SIM-1c (CF3T)', 'pp-Ni-SIM-1c (DMF)',\
                   'pp-Ti-SIM-1c (CF3T)', 'pp-Ti-SIM-1c (DMF)',\
                'pp-Co-SIM-1c (CF3T)', 'pp-Co-SIM-1c (DMF)']
labs_coumarins = ['Coumarin 30', 'Coumarin 102', 'Coumarin 151']
labs_ppnu1k2 = [
                 'pp-NU-1000-scc (2) (CF3T)', 'pp-NU-1000-scc (2) (DMF)', 'pp-NU-1000-vac (2) (DMF)','pp-NU-1000 (2) (DMF)', 'pp-NU-1000 (2) (water)']
#%%

PlotLinearQYFits(areas, absor, labs_onlybase)
fits = ComputeLinearQYFits(areas, absor, labels)
#%%
solvents = pd.read_csv('solvents.csv', sep = ';')




#%%
def ComputeLinearQYFit(areas, absor):
    x = absor
    y = areas
    coeff, covar = np.polyfit(x, y, 1, cov = True)
    # r-squared
    p = np.poly1d(coeff)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    sqrR= ssreg / sstot
    error = np.sqrt(np.diag(covar))
    return [coeff, error, sqrR]


label = 'cou_151_390'
cou_151_390fit = ComputeLinearQYFit(areas[label], absor[label])


#%%
fig, ax = plt.subplots(figsize = (10,6.6))
for k in ['cou_102_390','cou_151_390']:
    x = absor[k]
    y = areas[k]
    coeff, covar = np.polyfit(x,y,1, cov= True)
    a = coeff[0]
    b = coeff[1]
# ye = np.sqrt(covar[1][1])
    ax.scatter(x, y, s = 15, label = str(k))
    ax.plot(np.unique(x), np.unique(x)*a + b, label = None, linewidth = 0.5)
# ax.errorbar(np.unique(x), np.unique(x)*a + b, yerr = ye, ecolor = 'black', fmt = 'none', capsize = 2, label = None)
ax.legend(loc = 'upper left')
# ax.set_xlim([0, 0.08])
ax.set_xlabel("Absorbance", fontsize = 20)
ax.set_ylabel("Integrated Fluorescence", fontsize = 20)
#%%

def CalculateQY(QYRef, Xfit, Reffit, RindX, RindRef):
    GradRef = Reffit[0][0]
    GradRefError = Reffit[1][0]/GradRef
    GradX = Xfit[0][0]
    GradXError = Xfit[1][0]/GradX
    QYX = QYRef*(GradX/GradRef)*((RindX**2) / (RindRef**2))
    QYE = (np.sqrt(GradRefError**2 + GradXError**2))*QYX
    return [QYX, QYE]

print(CalculateQY(0.76, cou_151_390fit, cou_102_390fit, 1,1))
#%%
# slvs = ['trifluorotoluene', 'dimethylformamide', 'trifluorotoluene',\
        # 'dimethylformamide', 'trifluorotoluene', 'dimethylformamide',]
slvs_all = ['trifluorotoluene', 'dimethylformamide', 'trifluorotoluene',\
        'dimethylformamide', 'trifluorotoluene', 'dimethylformamide',\
        'dimethylsulfoxide', 'ethanol', 'heptane', 'pyridine', 'water',\
        'trifluorotoluene', 'dimethylformamide', 'dimethylformamide',\
        'water', 'trifluorotoluene','dimethylformamide','dimethylformamide',\
            'trifluorotoluene','dimethylformamide', 'trifluorotoluene',\
        'dimethylformamide', 'methanol', 'methanol',\
                'acetonitrile', 'trifluorotoluene',\
        'dimethylformamide', 'trifluorotoluene', 'dimethylformamide',\
        'trifluorotoluene', 'dimethylformamide']
# slvs = ['trifluorotoluene', 'dimethylformamide', 'dimethylsulfoxide',\
#         'ethanol', 'pyridine', 'water', 'dimethylformamide', 'water']
yields = {}
for label, solvent in zip(labels, slvs_all):
    yields[label] = CalculateQY(0.67, 'Coumarin 30', label, 'acetonitrile', solvent, fits, solvents)
#%%
with open('yields_corr_11242020.txt', 'w') as f:
    print(yields, file=f)
#%%
def smoothTriangle(labels, dataset, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smooth_data = {}
    for label in labels: 
        smoothed=[]
        spectrum = dataset[label]['d1']
        for i in range(degree, len(spectrum) - degree * 2):
            point=spectrum[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point)/np.sum(triangle))
        smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
        while len(smoothed) < len(spectrum):
            smoothed.append(smoothed[-1])
        temp_df = pd.DataFrame()
        temp_df['wl'] = data[label]['wl']
        temp_df['d1'] = smoothed
        smooth_data[label] = temp_df
    return smooth_data


    
def PlotNormFluor(data, labels, xlim, ylim, save):
    fig, ax = plt.subplots(figsize = (10,8))
    ax.axhline(y=0, color = 'black', ls = '--')
    for label in labels:
        x = nm2WN(data[label]['wl'])
        ax.plot(x, normalize_1(data[label]['d1']), label = label, linewidth = 3)
    if xlim != 0:
        ax.set_xlim(xlim)
    # dxlim = max(xlim) - min(xlim)
    # if dxlim >= 100:
    #     ax.xaxis.set_major_locator(MultipleLocator(20))
    #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    #     ax.xaxis.set_minor_locator(MultipleLocator(5))
    # elif dxlim < 100 and dxlim >= 50:
    #     ax.xaxis.set_major_locator(MultipleLocator(10))
    #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    #     ax.xaxis.set_minor_locator(MultipleLocator(2))
    # elif dxlim < 50:
    #     ax.xaxis.set_major_locator(MultipleLocator(5))
    #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    #     ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim([max(x), min(x)])
    ax.set_xlabel("$Wavenumbers (cm^{-1})$", fontsize = 20)
    secax = ax.secondary_xaxis('top', functions=(WN2nm, nm2WN))
    secax.tick_params(labelsize = 18)
    secax.set_xlabel('wavelength, nm', fontsize = 20)
    if ylim == 0:
        ax.set_ylim([-0.05,1.1])
    else:
        ax.set_ylim(ylim)
    ax.set_ylabel("Normalized Fluorescence", fontsize = 20)
    ax.legend(loc = 'upper right')
    if save == 1:
        plt.savefig(str(np.random.randint(1000))+'picture.svg')  
        
        
def PlotAbsorption(data, labels, xlim, ylim, norm, save):
    fig, ax = plt.subplots(figsize = (10, 6.6))
    ax.axhline(y=0, color = 'black', ls = '--')
    if xlim != 0:
        ax.set_xlim(xlim)
        dxlim = max(xlim) - min(xlim)
        if dxlim >= 100:
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
        elif dxlim < 100 and dxlim >= 50:
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(2))
        elif dxlim < 50:
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
    if norm == 1:
        for label in labels: 
            ax.plot(data[label][0], normalize_1(data[label][1]), label = label, linewidth = 3)
        ax.set_ylim([-0.05, 1.1])
        ax.set_ylabel("Normalized Absorbance", fontsize = 20)
    else:
        for label in labels: 
            ax.plot(data[label][0], data[label][1], label = label, linewidth = 3)
        ax.set_ylim(ylim)
        ax.set_ylabel("Absorbance, OD", fontsize = 20)
    ax.legend(loc = 'upper right')
    if save == 1:
        plt.savefig(str(np.random.randint(1000))+'picture.svg')  
#%%
# smoothed = smoothTriangle(labels, data, 3)
PlotNormFluor(data, labs_only_dTi, 0, 0 , 0)


#%%
def FindMaxima(data, labels):
    em_peaks = dict()
    for label in labels:
        index = data[label][['d1']].idxmax()
        em_peaks[label] = int(data[label]['wl'][index])
    return em_peaks

em_peaks = FindMaxima(data, labels)

def FindRatios(data, labels):
    ratios={}
    for label in labels: 
        index450 = data[label]['d1'].loc(axis = 0)[40]
        index480 = data[label]['d1'].loc(axis = 0)[70]
        ratio = index450/index480
        ratios[label] = ratio
    return ratios

def FitGaussians(wavelength, data, gaus_no, initPar, label, save):
    x = wavelength
    y = data
    if gaus_no == 1:
        fittedParameters, pcov = curve_fit(gaus1, x, y, initPar)
        a1, mu1, sigma1, a0 = fittedParameters
        yplt = gaus1(x, a1, mu1, sigma1, a0)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Offset']
    if gaus_no == 2:
        fittedParameters, pcov = curve_fit(gaus2, x, y, initPar)
        a1, mu1, sigma1, a2, mu2, sigma2, a0 = fittedParameters
        yplt = gaus2(x, a1, mu1, sigma1, a2, mu2, sigma2, a0)
        pars = ['Amp1', 'Mean1', 'Sigma1', 'Amp2', 'Mean2', 'Sigma2', 'Offset']
    fig, ax = plt.subplots(figsize=(10,8))
    resid = np.subtract(y, yplt)
    SE = np.square(resid) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(resid) / np.var(y))
    ax.set_xlim([max(x), min(x)])
    ax.set_xlabel("Energy, eV", fontsize = 20)
    ax.set_ylabel("Normalized Fluorescence", fontsize = 20)
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
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
        plt.savefig(str(label)+'GaussFit'+str(np.random.randint(1000))+'.svg')
        res.to_csv(str(label)+'Gaussfit.csv', index = False)
    return res
    
def nm2ev(data_in_nm):
    h = 4.135667516*10**(-15)
    c = 299792458
    return (h*c)/(data_in_nm*10**(-9))

#%%
def gaus1(x,a1,mu1,sigma1,a0):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a0

# def gaus2(x,a1,mu1,sigma1,a2,mu2,sigma2,a0):
    # return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + a0
def gaus2(x,a1,mu1,sigma1,a2,mu2,sigma2,a0):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + a0

def lorz2(x,a1,mu1,gamma1,a2,mu2,gamma2,a0):
    return a1*(gamma1/((x-mu1)**2+gamma1**2)) + a2*(gamma2/((x-mu2)**2+gamma2**2)) + a0


initPar = [1, 2.75, 0.1, 1, 2.55, 0.1, 0.05]
energy = nm2ev(data['d-Ti-SIM-1c (DMF)']['wl'])
signal = normalize_1(data['d-Ti-SIM-1c (DMF)']['d1'])
# signal = smoothed['d-Ti-SIM-1c (DMF)']['d1']

fit_dmf = FitGaussians(energy, signal, 2, initPar, 'd-Ti-SIM-1c (DMF)', 0)
#%% Overlaid Average Emission

fig, ax = plt.subplots(figsize = (10,6.6))
# ax.xaxis.set_major_locator(MultipleLocator(50))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# # For the minor ticks, use no labels; default NullFormatter.
# # ax.xaxis.set_minor_locator(MultipleLocator(0.2))

ax.set_ylim([435, 495])
# ax.set_xlim([])
ax.set_xlabel("Refractive Index", fontsize = 20)
# Set the yticks as well. A little more complex
# ax.set_ylim([0,1.1])
# ax.set_yticks(yticks)
ax.set_ylabel("Fluorescence Peak Maximum, nm", fontsize = 20)
for label, solvent in zip(labs_onlyTi, slvs):
    ax.scatter(solvents['Rind'][solvents[solvents['Solvent']==solvent].index.values],em_peaks[label],s=10000*yields[label], label = label)
ax.legend(loc = 'upper right')
# plt.savefig('linker_absemi.svg')

