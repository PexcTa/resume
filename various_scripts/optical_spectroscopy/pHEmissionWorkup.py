# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:57:50 2022

@author: boris
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:10:07 2022

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import ast

#%%
colnames = []
colnames.append('wl')
colnames.append('bg')
temp = np.zeros(12)
for i in range(10):
    colnames.append("r"+str(i)+'b')
colnames = np.row_stack([temp, colnames])    

del(i,temp)


#%%
def ReadData(filename):
    with open(filename) as current_file:
        d = pd.read_csv(current_file, sep = ',', header = 0, engine = 'python')
    return d
def ReadAbsData(filename, skip=2):
    with open(filename) as current_file:
        d = pd.read_csv(current_file, names = ['wl', 'R'], delim_whitespace = True, engine = 'python', skiprows = lambda x: x in range(skip))
        return d
def readPhDict(filename, deutero=False):
    with open(filename) as current_file:
        dataframe = pd.read_csv(current_file, names = ['id', 'ph'], sep = ',', engine = 'python')
        dataframe.set_index('id',inplace=True)
        dictionary = dataframe.to_dict()['ph']
        if deutero == True:
            for key in dictionary.keys():
                dictionary[key] += 0.44
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
def saver(dictex):
    for key, val in dictex.items():
        val.to_csv("data_{}.csv".format(str(key)))
    with open("keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))
#%% Import Data - Average Into Equally-Spaced Bins
def ReadAllData():
    data_dict = {}
    for file in os.listdir():
        if file.endswith('.csv'):
            data_dict[file] = ReadData(file)
    return data_dict

rawdata = ReadAllData()
ph_data = readPhDict('ph.txt', True)
normalization_factor = 1
# outliers = ['w2b', 'w3b', 'w4b','w2c', 'w3c', 'w4c']
# ph_corr = {}
# for key in ph_dict.keys():
#     ph_corr[key] = ph_dict[key]+0.44
# average in units of pH: EVERY UNIT
ph = 0.0
phstep = 0.5
data_dict = rawdata.copy()
ph_dict = ph_data.copy()
data_averaged_by_pH_halfunit = {}
while ph <= 10:
    temp = pd.DataFrame()
    for dataset in data_dict.keys():
        # if dataset in ['s_day1.csv','s_day2.csv','t_day1.csv','t_day2.csv']:
        #     continue
        dset = data_dict[dataset]
        # print(dset.columns)
        for column in dset.columns[2:]:
            # print(str(column))
            if str(column) in outliers:
                continue
            if ph <= ph_dict[column] <= (ph+phstep):
                cln_series = pd.DataFrame(dset[column] - dset['bg'], columns = [str(column)])
                # cln_series = pd.DataFrame(dset[column], columns = [str(column)])
                temp = pd.concat([temp, cln_series], axis = 1)
    avg = temp.agg(np.mean, 1)
    if math.isnan(np.mean(avg)) == False:
        std = temp.agg(np.std, 1)
        output = pd.concat([dset['wl'], avg/normalization_factor, std/normalization_factor], axis = 1)
        output.columns = ['wl', 'avg', 'stdev']
        key = '%.1f' % ph +' to '+ '%.1f' % (ph+phstep)
        data_averaged_by_pH_halfunit[key] = output
    ph+=phstep
#%% Import Data - Average with a Sliding Window
# for key in ph_dict.keys():
#     ph_corr[key] = ph_dict[key]+0.44
# average in units of pH: EVERY UNIT
ph = 0
phstep = 1
ph_dict = ph_data.copy()
data_averaged_by_pH_sliding = {}
while ph <= 10:
    temp = pd.DataFrame()
    for dataset in data_dict.keys():
        # if dataset in ['s_day1.csv','s_day2.csv','t_day1.csv','t_day2.csv']:
        #     continue
        dset = data_dict[dataset]
        # print(dset.columns)
        for column in dset.columns[2:]:
            # print(str(column))
            if str(column) in outliers:
                continue
            if ph <= ph_dict[column] <= (ph+phstep*2):
                cln_series = pd.DataFrame(dset[column] - dset['bg'], columns = [str(column)])
                # cln_series = pd.DataFrame(dset[column], columns = [str(column)])
                temp = pd.concat([temp, cln_series], axis = 1)
    avg = temp.agg(np.mean, 1)
    if math.isnan(np.mean(avg)) == False:
        std = temp.agg(np.std, 1)
        output = pd.concat([dset['wl'], avg/normalization_factor, std/normalization_factor], axis = 1)
        output.columns = ['wl', 'avg', 'stdev']
        key = '%.2f' % ph +' to '+ '%.2f' % (ph+phstep*2)
        data_averaged_by_pH_sliding[key] = output
    ph+=phstep
#%% average in units of pH: NODE PKA
def ReadAllData():
    data_dict = {}
    for file in os.listdir():
        if file.endswith('.csv'):
            data_dict[file] = ReadData(file)
    return data_dict

data = ReadAllData()
# ph_data = readPhDict('phdict.txt')
# ph_corr = {}
# for key in ph_dict.keys():
#     ph_corr[key] = ph_dict[key]+0.44
# average in units of pH: EVERY UNIT
ph = 0
data_dict = data.copy()
ph_dict = ph_data.copy()
data_averaged_by_pH_node = {}
i = 0
while i < 5:
    if i == 0:
        lim=(0.00,1)
    elif i == 1:
        lim=(1,3.3)
    elif i == 2:
        lim = (3.3, 5.75)
    elif i == 3:
        lim = (5.75, 8.20)
    elif i == 4:
        lim = (8.20, 9.5)
    temp = pd.DataFrame()
    for dataset in data_dict.keys():
        if dataset in []:
            continue
        dset = data_dict[dataset]
        for column in dset.columns[2:]:
            if str(column) in outliers:
                continue
            if min(lim) <= ph_dict[column] <= max(lim):
                cln_series = pd.DataFrame(dset[column] - dset['bg'], columns = [str(column)])
                # cln_series = pd.DataFrame(dset[column], columns = [str(column)])
                temp = pd.concat([temp, cln_series], axis = 1)
    avg = temp.agg(np.mean, 1)
    if math.isnan(np.mean(avg)) == False:
        std = temp.agg(np.std, 1)
        output = pd.concat([dset['wl'], avg/normalization_factor, std/normalization_factor], axis = 1)
        output.columns = ['wl', 'avg', 'stdev']
        key = str(min(lim))+' to '+str(max(lim))
        data_averaged_by_pH_node[key] = output
    i+=1
#%% average in units of pH: HENDERSON-HESSELBALCH MAXIMA
def ReadAllData():
    data_dict = {}
    for file in os.listdir():
        if file.endswith('.csv'):
            data_dict[file] = ReadData(file)
    return data_dict

data = ReadAllData()
# ph_data = readPhDict('phdict.txt')
# ph_corr = {}
# for key in ph_dict.keys():
#     ph_corr[key] = ph_dict[key]+0.44
# average in units of pH: EVERY UNIT
ph = 0
data_dict = data.copy()
ph_dict = ph_data.copy()
data_averaged_by_pH_HHM = {}
i = 0
while i < 5:
    if i == 0:
        lim=(0.00,1)
    elif i == 1:
        lim=(1,2.1)
    elif i == 2:
        lim = (2.1, 4.5)
    elif i == 3:
        lim = (4.5, 7.0)
    elif i == 4:
        lim = (7.0, 9.5)
    temp = pd.DataFrame()
    for dataset in data_dict.keys():
        if dataset in []:
            continue
        dset = data_dict[dataset]
        for column in dset.columns[2:]:
            if str(column) in outliers:
                continue
            if min(lim) <= ph_dict[column] <= max(lim):
                cln_series = pd.DataFrame(dset[column] - dset['bg'], columns = [str(column)])
                # cln_series = pd.DataFrame(dset[column], columns = [str(column)])
                temp = pd.concat([temp, cln_series], axis = 1)
    avg = temp.agg(np.mean, 1)
    if math.isnan(np.mean(avg)) == False:
        std = temp.agg(np.std, 1)
        output = pd.concat([dset['wl'], avg/normalization_factor, std/normalization_factor], axis = 1)
        output.columns = ['wl', 'avg', 'stdev']
        key = str(min(lim))+' to '+str(max(lim))
        data_averaged_by_pH_HHM[key] = output
    i+=1
                            
#%%
fig, ax = plt.subplots(figsize=(13,12))
# ax.set_xlabel('Energy $(cm^{-1})$', fontsize = 20)
ax.set_xlabel('Energy (eV)', fontsize = 30)
ax.set_ylabel("Emission (A.U. per Unit OD)", fontsize = 30)
# ax.set_ylabel("Normalized Emission (A.U.)", fontsize = 30)
ax.tick_params(axis='x', labelsize= 27)
ax.tick_params(axis='y', labelsize= 27)
ax.ticklabel_format(axis='y', style='sci', scilimits = (0,0))
ax.set_xlim([3.84, 2.3])
# ax.set_xlim([3.0, 1.9])
ax.set_ylim([-0.01*10**5, 2*10**5])

ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
secax = ax.secondary_xaxis('top', functions=(ev2nm, nm2ev))
secax.tick_params(labelsize = 27)
secax.set_xlabel('Wavelength (nm)', fontsize = 30, labelpad = 10)


# ax.set_prop_cycle('label',[labeldict.keys[i] for i in np.linspace(0, 1, 11)])
# ax.set_facecolor('silver')
# colors = ['goldenrod', 'darkorange', 'green', 'firebrick']
# ax.set_prop_cycle('color', ['goldenrod', 'darkorange', 'green', 'firebrick'])
dset = data_averaged_by_pH_halfunit.copy()
ax.set_prop_cycle('color',[plt.cm.turbo(i) for i in np.linspace(1,0, len(list(dset.keys())))])
# ph = pHnukM.copy()
for key in dset.keys():
    # if key in ['0.00 to 1.00']:
    #     continue
    data = dset[key]
    x = nm2ev(data['wl'])
    y = sgf(jac(data['wl'], data['avg']),27,3)
    # y = jac(data['wl'], data[0])
    if math.isnan(y[0]) == True:
        continue
    err = jac(data['wl'], data['stdev'])
    # err = jac(data['wl'], data[1])
    ax.plot(x,y,linewidth = 5,label = str(key))
    ax.fill_between(x, y-err, y+err, alpha = 0.2)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(bbox_to_anchor=(1, 0.5), facecolor='silver', framealpha=1, loc = 'center left', fontsize = 26, ncol = 3)
ax.legend(loc = 'upper right', fontsize = 24)
ax.axhline(y=0, color = 'dimgrey', ls = '--')
ax.yaxis.get_offset_text().set_fontsize(27)
# ax.axvline(x=3.0, color = 'red', ls = '-.')
# ax.axvline(x=3.1, color = 'red', ls = '-.')
# ax.annotate("Very Strong Raman \npeak here", (3.09, 4*10**5), fontsize = 22)
# ax.annotate("n = " + str(len(data_dict.keys())*10-len(outliers)), (2.75, 1.5*10**6), fontsize = 25)


#%% averaging and bgsub with one background
# raw_dset = pd.concat([data.copy(), data.copy().iloc[:,2:]], axis = 1)
raw_dset = ReadData('extras.csv')
i = 2
cln_dset = pd.DataFrame(raw_dset.iloc[:,0], columns = ['wl'])
while i < len(raw_dset.columns):
    # AVERAGE EVERY TWO STARTING FROM iTH COLUMN
    avg_series = pd.concat([raw_dset.iloc[:,i:i+1]], axis=1).agg(np.mean, 1)
    # SUBTRACT BACKGROUND WHICH IS IN 1st (python counting) COLUMN
    cln_series = pd.DataFrame(avg_series - raw_dset.iloc[:,1], columns = [str(raw_dset.columns[i][:4])])
    # cln_series = pd.DataFrame(avg_series, columns = [str(raw_dset.columns[i])+"_avg"])
    # std_series = pd.DataFrame(pd.concat([raw_dset.iloc[:,i:i+1]], axis=1).agg(np.std, 1), columns = [str(raw_dset.columns[i])+'_std'])
    cln_dset = pd.concat([cln_dset, cln_series], axis = 1)
    i += 1
# nrm_dset = cln_dset.copy()
# for col in nrm_dset.columns[1:]:
#     identificator = str(col)[:2]
#     absorption = ReadAbsData("./abs/"+identificator+".txt")
#     nrm_dset[col] = nrm_dset[col]/absorption.iloc[203,1]
# nr1_dset = nrm_dset.copy()
# highest_in_all_data = nrm_dset.max().max()
# for col in nr1_dset.iloc[:,1:]:
#     nr1_dset[col] = nr1_dset[col]/highest_in_all_data
del [avg_series, cln_series, i]
#%% averaging and bgsub with individual backgrounds
raw_dset = ReadData('day1.csv')
i = 2
cln_dset = pd.DataFrame(raw_dset.iloc[:,0], columns = ['wl'])
while i < 13:
    avg_series = pd.concat([raw_dset.iloc[:,i:i+1]], axis=1).agg(np.mean, 1)
    cln_series = pd.DataFrame(avg_series - raw_dset.iloc[:,i-1], columns = [str(raw_dset.columns[i])])
    std_series = pd.DataFrame(pd.concat([raw_dset.iloc[:,i:i+1]], axis=1).agg(np.std, 1), columns = [str(raw_dset.columns[i])+'_std'])
    cln_dset = pd.concat([cln_dset, cln_series, std_series], axis = 1)
    i += 1
nrm_dset = cln_dset.copy()
for col in nrm_dset.columns[1:]:
    identificator = str(col)[:2]
    absorption = ReadAbsData("./abs_data/"+identificator+".txt")
    nrm_dset[col] = nrm_dset[col]/absorption.iloc[80,1]
del [avg_series, cln_series, std_series, i, identificator, absorption]
#%%
fig, ax = plt.subplots(figsize=(12,12))
# ax.set_xlabel('Energy $(cm^{-1})$', fontsize = 20)
ax.set_xlabel('Energy (eV)', fontsize = 30)
ax.set_ylabel("Emission (cts)", fontsize = 30)
# ax.set_ylabel("Normalized Emission (A.U.)", fontsize = 30)
ax.tick_params(axis='x', labelsize= 25)
ax.tick_params(axis='y', labelsize= 25)
ax.ticklabel_format(axis='y', style='sci', scilimits = (0,0))
# ax.set_xlim([4.00, 2.15])
# ax.set_xlim([3.00, 1.9])
ax.set_xlim([3.35, 2.0])

ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
secax = ax.secondary_xaxis('top', functions=(ev2nm, nm2ev))
secax.tick_params(labelsize = 25)
secax.set_xlabel('Wavelength (nm)', fontsize = 30, labelpad = 10)
ax.set_prop_cycle('color',[plt.cm.magma(i) for i in np.linspace(0, 0.8, 5)])
# ax.set_prop_cycle('label',[labeldict.keys[i] for i in np.linspace(0, 1, 11)])
# ax.set_facecolor('gainsboro')
# colors = ['goldenrod', 'darkorange', 'green', 'firebrick']
# ax.set_prop_cycle('color', ['goldenrod', 'darkorange', 'green', 'firebrick'])

dset = cln_dset.copy()
# ph = pHnukM.copy()

for col in dset.columns[1:]:
    # if ph[col] in [8.09]:
    #     continue
    x = nm2ev(dset['wl'])
    y = sgf(jac(dset['wl'], dset[col]), 9,3)
    # ax.plot(x,sgf(y,5,2),linewidth = 5)
    # err = jac(dset['wl'], dset[str(col)[:-4]+"_std"])
    # ax.errorbar(x, y, yerr = err, linewidth = 3, errorevery = 5, capsize = 12.5, alpha = 0.75)
    # ax.fill_between(x, y - err, y + err, alpha=0.5)
    ax.plot(x,y,linewidth = 3,label = concs[col])
# ax.annotate('0.1M LiNO$_3$', (2.25, 4*10**6), fontsize = 24)
ax.legend(loc = 'upper right', fontsize = 28, ncol = 1)
# ax.legend(labels = ['Initial', '2hr at 333K', '3hr at 393K', '18hr at 393K', '5 min under air', '3 days under air'], loc = 'upper right', fontsize = 28, ncol = 1)
ax.axhline(y=0, color = 'dimgrey', ls = '--')


#%% re-order the ph-dictionary (will only work in python 3.7+)
for key in outliers:
    ph_dict.pop(key)
sorted_ph_dict = {k: v for k, v in sorted(ph_dict.items(), key=lambda item: item[1])}
# convert to a list, create the dataframe for output
sorted_keys = list(sorted_ph_dict.keys())
sorted_dataframe = pd.DataFrame()
sorted_dataframe = pd.concat([sorted_dataframe, data_dict['k_day1.csv']['wl']], axis = 1)
# take a key and look for in in data-dict (dict of dataframes)
for key in sorted_keys:
    for dataframe in data_dict.keys():
        if key in data_dict[dataframe].columns:
            sorted_dataframe = pd.concat([sorted_dataframe, data_dict[dataframe][key]-data_dict[dataframe]['bg']], axis = 1)
        else:
            continue
names = list(sorted_ph_dict.values())
for i in range(1,len(names)):
    if names[i] == names[i-1]:
        names[i] += 0.001
    elif names[i] == names[i-2]:
        names[i] += 0.002
sorted_dataframe.columns = ['wl'] + names
# elyte_high = sorted_dataframe.copy()

# sorted_dataframe.to_csv('./exported_Data/sortedTotalData.csv')
#%% scan through successive datasets and remove outliers 
# decides whether to remove a spectrum based on total area AND singlet intensity
dataset = sorted_dataframe.copy()
dataset_no_outliers = pd.DataFrame()
dataset_no_outliers = pd.concat([dataset_no_outliers, dataset['wl']], axis = 1)
i = 4
j = 3
dataset_no_outliers = pd.concat([dataset_no_outliers, dataset.iloc[:,1:j]], axis = 1)
while i < len(dataset.columns):
    last_column_area = np.trapz(dataset.iloc[1:, j], dx = 1, axis = 0)
    current_column_area = np.trapz(dataset.iloc[1:, i], dx = 1, axis = 0)
    last_ct_intensity = dataset.iloc[52, j] + dataset.iloc[53, j]
    current_ct_intensity = dataset.iloc[52, i] + dataset.iloc[53, i]
    last_singlet_peak_intensity = dataset.iloc[12, j] + dataset.iloc[13, j]
    current_singlet_peak_intensity = dataset.iloc[12, i] + dataset.iloc[13, i]
    if current_column_area > last_column_area and current_singlet_peak_intensity > last_singlet_peak_intensity:
    # if current_ct_intensity > last_ct_intensity and current_singlet_peak_intensity > last_singlet_peak_intensity:
    # if current_singlet_peak_intensity > last_singlet_peak_intensity:
        # print((current_column_area - last_column_area, current_singlet_peak_intensity - last_singlet_peak_intensity))
        dataset_no_outliers = pd.concat([dataset_no_outliers, dataset.iloc[:,i]], axis = 1)
        j = i
        i += 1
    else: 
        i += 1
#%%
fig, ax = plt.subplots(figsize=(12,12))
# ax.set_xlabel('Energy $(cm^{-1})$', fontsize = 20)
ax.set_xlabel('Energy (eV)', fontsize = 30)
ax.set_ylabel("Emission (cts)", fontsize = 30)
# ax.set_ylabel("Normalized Emission (A.U.)", fontsize = 30)
ax.tick_params(axis='x', labelsize= 25)
ax.tick_params(axis='y', labelsize= 25)
ax.ticklabel_format(axis='y', style='sci', scilimits = (0,0))
# ax.set_xlim([4.00, 2.15])
# ax.set_xlim([3.00, 1.9])
# ax.set_ylim([-1.1, 2.7])
ax.set_prop_cycle('color', [plt.cm.RdBu(i) for i in np.linspace(0, 1, len(dataset_no_outliers.columns[1:]))])
for col in dataset_no_outliers.columns[1:]:
    ax.plot(dataset_no_outliers['wl'], dataset_no_outliers[col], linewidth = 3, label = col)
ax.legend()
    
# dataset_no_outliers.to_csv('./exported/noOutliers_11102022.csv')

#%% integrate areas and plot vs ph
data_array = np.array(sorted_dataframe.iloc[1:, 1:])
areas = np.trapz(data_array, dx = 1, axis = 0)

fig, ax = plt.subplots(figsize=(12,12))
ax.set_xlabel('pH', fontsize = 30)
ax.set_ylabel('Integrated Fluorescence', fontsize = 30)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.scatter(sorted_dataframe.columns[1:], areas, s = 100)
#%% go through spectra ten at a time to determine outliers
start = 96
end = start + 5
fig, ax = plt.subplots(figsize=(12,12))
for i in range(start, end):
    ax.plot(sorted_dataframe['wl'], sorted_dataframe[sorted_dataframe.columns[i]], label = list(sorted_ph_dict.keys())[i-1])
ax.legend(fontsize = 35)
#%% export dataset with no outliers
outliers = ['h4e', 'h8b', 'h9c', 'h7c', 'k9e', 'h8c', 'h6c', 'k4c', 'h5c', 'k7d']
sorted_ph_dict = {k: v for k, v in sorted(ph_dict.items(), key=lambda item: item[1])}
for key in outliers:
    sorted_ph_dict.pop(key)
# convert to a list, create the dataframe for output
sorted_keys = list(sorted_ph_dict.keys())
sorted_dataframe = pd.DataFrame()
sorted_dataframe = pd.concat([sorted_dataframe, data_dict['h_day1.csv']['wl']], axis = 1)
# take a key and look for in in data-dict (dict of dataframes)

for key in sorted_keys:
    # if key in outliers:
    #     continue
    for dataframe in data_dict.keys():
        if key in data_dict[dataframe].columns:
            sorted_dataframe = pd.concat([sorted_dataframe, data_dict[dataframe][key]-data_dict[dataframe]['bg']], axis = 1)
        else:
            continue
names = list(sorted_ph_dict.values())

for i in range(1,len(names)):
    if names[i] == names[i-1]:
        names[i] += 0.001
    elif names[i] == names[i-2]:
        names[i] += 0.002
sorted_dataframe.columns = ['wl'] + names

sorted_dataframe.to_csv('./exported/sortedData_Clean_112822.csv')
#%% export averaged dataset - ignore errorbars
to_export = data_averaged_by_pH_sliding.copy()
sorted_dataframe_avg = pd.DataFrame()
sorted_dataframe_avg = pd.concat([sorted_dataframe_avg, to_export['0.00 to 2.00']['wl']], axis = 1)
names = ['wl']
# this dictionary should already come with keys going in the right order
for key in to_export.keys():
    key_low = float(key[:4])
    key_high = float(key[8:])
    print((key_low, key_high))
    names.append('%.2f' % ((key_low+key_high)/2))
    sorted_dataframe_avg = pd.concat([sorted_dataframe_avg, to_export[key]['avg']], axis = 1)
sorted_dataframe_avg.columns = names

sorted_dataframe_avg.to_csv('./exported/sliding_s1.csv')
    
#%% plot matrix as a contour plot of emission vs ph
fig, ax = plt.subplots(figsize=(12,12))
ax.set_xlabel('Wavelength (nm)', fontsize = 36)
ax.set_ylabel('pH', fontsize = 36)
x = sorted_dataframe_avg['wl'][1:]
y = sorted_dataframe_avg.columns[1:]
z = np.transpose(sorted_dataframe_avg.iloc[1:, 1:])
cs1 = ax.contourf(x, y, z, 25, cmap = 'gnuplot2')
# ax.set_xlim(x[min(region)], x[max(region)])
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.tick_params(axis='x', labelsize= 30)
ax.tick_params(axis='y', labelsize= 30)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(cs1, cax=cax, orientation='vertical')
cbar.set_ticks([np.linspace(z.min().min(), z.max().max(), 10, dtype = int)])
cax.tick_params(labelsize = 25)
#%% plot excitation
fig, ax = plt.subplots(figsize=(15,12))
ax.set_xlabel('Energy (eV)', fontsize = 30)
# ax.set_ylabel("Emission (cts)", fontsize = 30)
ax.set_ylabel("Normalized Excitation (A.U.)", fontsize = 30)
ax.tick_params(axis='x', labelsize= 25)
ax.tick_params(axis='y', labelsize= 25)
ax.ticklabel_format(axis='y', style='sci', scilimits = (0,0))
ax.set_xlim([4.15, 2.40])
# ax.set_xlim([3.00, 1.9])
# ax.set_ylim([-1.1, 2.7])
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
secax = ax.secondary_xaxis('top', functions=(ev2nm, nm2ev))
secax.tick_params(labelsize = 25)
secax.set_xlabel('Wavelength (nm)', fontsize = 30, labelpad = 10)
# ax.set_prop_cycle('color',[plt.cm.RdBu(i) for i in np.linspace(0, 1, 7)])
# ax.set_facecolor('gray')
# colors = ['goldenrod', 'darkorange', 'green', 'firebrick']
ax.set_prop_cycle('color', ['crimson', 'magenta', 'indigo', 'dodgerblue'])

dset = d2o.copy()
# ph = pHnukM.copy()
i = 0
while i < len(exdata.columns):
    print(i)
    x = nm2ev(dset.iloc[:,i])
    y = normalize_1(jac(dset.iloc[:,i+1], dset.iloc[:,i]))
    ax.plot(x, y, linewidth = 5)
    i += 2
ax.legend(labels = ['pH=2.5, $\lambda_{em}$=500','pH=2.5, $\lambda_{em}$=460','pH=8.0, $\lambda_{em}$=500','pH=8.0, $\lambda_{em}$=460'],  framealpha=1, loc = 'upper right', fontsize = 25, ncol = 1)
ax.axhline(y=0, color = 'dimgrey', ls = '--')



#%% re-order the ph-dictionary (will only work in python 3.7+)
sorted_ph_dict = {k: v for k, v in sorted(ph_dict.items(), key=lambda item: item[1])}
# convert to a list, create the dataframe for output
sorted_keys = list(sorted_ph_dict.keys())
sorted_dataframe = pd.DataFrame()
sorted_dataframe = pd.concat([sorted_dataframe, data_dict['a_day1.csv']['wl']], axis = 1)
# take a key and look for in in data-dict (dict of dataframes)
for key in sorted_keys:
    for dataframe in data_dict.keys():
        if key in data_dict[dataframe].columns:
            sorted_dataframe = pd.concat([sorted_dataframe, data_dict[dataframe][key]-data_dict[dataframe]['bg']], axis = 1)
        else:
            continue
names = list(sorted_ph_dict.values())
for i in range(1,len(names)):
    if names[i] == names[i-1]:
        names[i] += 0.001
    elif names[i] == names[i-2]:
        names[i] += 0.002
sorted_dataframe.columns = ['wl'] + names

sorted_dataframe.to_csv('./sortedTotalData.csv')
#%% integrate areas and plot vs ph
fig, ax = plt.subplots(figsize=(16,4))
ax.set_xlabel('pH', fontsize = 30)
ax.set_ylabel('$\int$ I${_F}$ $d \lambda$ per AU', fontsize = 30)
ax.tick_params(axis='x', labelsize= 27)
ax.tick_params(axis='y', labelsize= 27)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
for sorted_dataframe in [elyte_low, elyte_high]:
    data_array = np.array(sorted_dataframe.iloc[1:, 1:])
    areas = np.trapz(data_array, dx = 1, axis = 0)
    if sorted_dataframe is elyte_low:
        ax.scatter(sorted_dataframe.columns[1:], areas/0.190, s = 100, color = 'turquoise', marker = '^')
    elif sorted_dataframe is elyte_high:
        ax.scatter(sorted_dataframe.columns[1:], areas/0.029, s = 100, color = 'indigo')
ax.legend(labels = ['0.001M NaNO$_3$', '1M NaNO$_3$'], loc = 'upper left', fontsize = 26)
ax.yaxis.get_offset_text().set_fontsize(27)

#%% plot species associated spectra and kinetics

def readSAK(filename):
    with open(filename) as current_file:
        d = pd.read_csv(current_file, sep = ',', header = 0, engine = 'python')
        headr = ['pH']
        for i in range(len(d.columns)-1):
            headr.append(f"trace{i}")
        d.columns = headr
    return d
def readSAD(filename):
    with open(filename) as current_file:
        d = pd.read_csv(current_file, sep = ',', header = 0, engine = 'python')
        headr = ['wl']
        for i in range(len(d.columns)-1):
            headr.append(f"spec{i}")
        d.columns = headr
    return d

SAK = readSAK('SAKs_slidings0p75.csv')
SAD = readSAD('SADs_slidings0p75.csv')
#%%
sad = SAD.copy()
sak = SAK.copy()
fig, ax = plt.subplots(1,2,figsize=(26,12))

ax[0].set_xlabel('Wavelength (nm)', fontsize = 36)
ax[0].set_ylabel('Fluorescence Intensity', fontsize = 36)
ax[0].tick_params(axis='x', labelsize= 30)
ax[0].tick_params(axis='y', labelsize= 30)
ax[0].xaxis.set_major_locator(MultipleLocator(50))
ax[0].yaxis.set_ticklabels([])
ax[0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, len(sad.columns))])
for column in sad.columns[1:]:
    ax[0].plot(sad['wl'], sad[column], linewidth = 3.5)
ax[0].legend(labels =[ 'Species 1', 'Species 2', 'Species 3'], fontsize = 30)
ax[0].axhline(y=0, alpha = 0.75, linestyle = '--', color = 'black')

ax[1].set_xlabel('pH', fontsize = 36)
ax[1].set_ylabel('Species Fraction', fontsize = 36)
ax[1].tick_params(axis='x', labelsize= 30)
ax[1].tick_params(axis='y', labelsize= 30)
ax[1].xaxis.set_major_locator(MultipleLocator(1))
ax[1].set_xlim([0,10])
# ax[1].yaxis.set_ticklabels([])
colors = [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, len(sad.columns))]
for i in range(len(sak.columns[1:])):
    ax[1].scatter(sak['pH'], sak[sak.columns[i+1]], s = 200, linewidth = 4, facecolor = 'none', color = colors[i])

pKa1 = 2.07
pKa2 = 6.76

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
ax[1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, len(sad.columns))])
for i in range(len(fractions)):
    ax[1].plot(pH, fractions[i], linewidth = 2)
# ax[1].legend(labels =[ 'Species 1', 'Species 2', 'Species 3', 'Species 4'], fontsize = 30)
# ax[1].axhline(y=0, alpha = 0.75, linestyle = '--', color = 'black')

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.05)

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
ax[0].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, len(sad.columns))])
for column in sad.columns[1:]:
    ax[0].plot(sad['wl'], sad[column], linewidth = 4)
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
colors = [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, len(sad.columns))]
for i in range(len(sak.columns[1:])):
    ax[1].scatter(sak['pH'], sak[sak.columns[i+1]], s = 200, linewidth = 4, facecolor = 'none', color = colors[i])

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
ax[1].set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 1, len(sad.columns))])
for i in range(len(fractions)):
    ax[1].plot(pH, fractions[i], linewidth = 2.5)
# ax[1].legend(labels =[ 'Species 1', 'Species 2', 'Species 3', 'Species 4'], fontsize = 30)
# ax[1].axhline(y=0, alpha = 0.75, linestyle = '--', color = 'black')
plt.tight_layout()
ax[0].yaxis.get_offset_text().set_fontsize(27)

# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.2, 
#                     hspace=0.05)

#%%
rawdata = np.loadtxt('sortedTotalData.csv', delimiter=',')
ph = rawdata[0, 1:]
wl = rawdata[1:, 0]
cts = rawdata[1:, 1:]
cts_ev = np.zeros_like(cts)
norm = 1
for i in range(len(cts[0,:])):
    cts_ev[:,i] = np.abs(jac(wl, cts[:,i]))
wl_ev = nm2ev(wl)
fig, ax = plt.subplots(figsize = (10,10))
ax.set_xlim([3.8, 2.35])
ax.set_ylim([0, 1.05])
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


idx_ph_1 = (np.abs(ph - 3.60)).argmin()
idx_ph_2 = (np.abs(ph - 5.85)).argmin()
idx_ph_3 = (np.abs(ph - 8.20)).argmin()
ax.set_prop_cycle('color', [plt.cm.RdBu(i) for i in np.linspace(0.1, 0.9, 3)])
ax.set_facecolor('gray')
iq1 = [np.trapz(cts_ev[i, :idx_ph_1], dx = 1) for i in range(261)]
ax.plot(wl_ev[36:], normalize_1(sgf(np.divide(iq1, len(ph[ph<3.60])*norm),9,3)[36:]), linewidth = 5, label = 'pH < 3.60')
iq2 = [np.trapz(cts_ev[i, idx_ph_1:idx_ph_2], dx = 1) for i in range(261)]
ax.plot(wl_ev[36:], normalize_1(sgf(np.divide(iq2, len(ph[np.where((ph>3.60)*(ph<5.85))])*norm),9,3)[36:]), linewidth = 5, label = '3.60 < pH < 5.85')
iq3 = [np.trapz(cts_ev[i, idx_ph_2:idx_ph_3], dx = 1) for i in range(261)]
ax.plot(wl_ev[36:], normalize_1(sgf(np.divide(iq3, len(ph[np.where((ph>5.85)*(ph<8.20))])*norm),9,3)[36:]), linewidth = 5, label = '5.85 < pH < 8.20')
# iq4 = [np.trapz(cts_ev[i, idx_ph_3:], dx = 1) for i in range(261)]
# ax.plot(wl_ev, np.divide(iq4, len(ph[ph>8.2])*norm), linewidth = 5)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.set_ylabel('Intensity (counts per A.U.)', fontsize = 27)
ax.set_xlabel('Energy (eV)', fontsize = 27)
ax.yaxis.get_offset_text().set_fontsize(24)
ax.legend(loc = 'lower right', fontsize = 27)

# plt.tight_layout()
#%%
rawdata = np.loadtxt('sortedTotalData_noOutliers.csv', delimiter = ',')
ph = rawdata[0, 1:]
wl = rawdata[1:, 0]
cts = rawdata[1:, 1:]
cts_ev = np.zeros_like(cts)
norm = 0.03
for i in range(len(cts[0,:])):
    cts_ev[:,i] = jac(wl, cts[:,i])
wl_ev = nm2ev(wl)
fig, ax = plt.subplots(figsize = (10,10))
ax.set_xlim([3.35, 2.15])

idx_ph_1 = (np.abs(ph - 3.60)).argmin()
idx_ph_2 = (np.abs(ph - 5.85)).argmin()
idx_ph_3 = (np.abs(ph - 8.20)).argmin()
ax.set_prop_cycle('color', [plt.cm.RdBu(i) for i in np.linspace(0.1, 0.9, 4)])

iq1 = [np.trapz(cts_ev[i, :idx_ph_1], dx = 1) for i in range(231)]
ax.plot(wl_ev, sgf(np.divide(iq1, len(ph[ph<3.60])*norm),5,2), linewidth = 5, label = 'pH < 3.60')
iq2 = [np.trapz(cts_ev[i, idx_ph_1:idx_ph_2], dx = 1) for i in range(231)]
ax.plot(wl_ev, sgf(np.divide(iq2, len(ph[np.where((ph>3.60)*(ph<5.85))])*norm),5,2), linewidth = 5, label = '3.60 < pH < 5.85')
iq3 = [np.trapz(cts_ev[i, idx_ph_2:idx_ph_3], dx = 1) for i in range(231)]
ax.plot(wl_ev, sgf(np.divide(iq3, len(ph[np.where((ph>5.85)*(ph<8.20))])*norm),5,2), linewidth = 5, label = '5.85 < pH < 8.20')
iq4 = [np.trapz(cts_ev[i, idx_ph_3:], dx = 1) for i in range(231)]
ax.plot(wl_ev, sgf(np.divide(iq4, len(ph[ph>8.2])*norm),5,2), linewidth = 5, label = 'pH > 8.20')
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.set_ylabel('Intensity (counts per A.U.)', fontsize = 27)
ax.set_xlabel('Energy (eV)', fontsize = 27)
ax.yaxis.get_offset_text().set_fontsize(24)
ax.legend(loc = 'upper right', fontsize = 27)