# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:36:59 2024

@author: boris
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)

from scipy.signal import savgol_filter as sgf
from scipy.signal import lombscargle
from scipy.signal import windows
import os
from collections import OrderedDict

#%%
chirDict = OrderedDict()
for file in os.listdir():
    with open(file):
        if str(file).endswith('.chir'):
            chirDict['headers'] = open(file).readlines()[37]
            chirDict[file] = np.loadtxt(file, dtype = float, skiprows=38)
            
chikDict = OrderedDict()
for file in os.listdir():
    with open(file):
        if str(file).endswith('.chik'):
            chikDict['headers'] = open(file).readlines()[37]
            chikDict[file] = np.loadtxt(file, dtype = float, skiprows=38)
            
#%% aux func

def hamming_window(data, k, klims):
    id_min = (np.abs(k - klims[0])).argmin()
    id_max = (np.abs(k - klims[1])).argmin()
    windowed_section = data[id_min:id_max]
    length = len(windowed_section)
    window_insert = windows.hamming(length, sym = False)
    window = np.zeros_like(data)
    window[id_min:id_max] = window_insert
    windowed_data = window*data
    return windowed_data

def hamming_lineshape(k, klims):
    id_min = (np.abs(k - klims[0])).argmin()
    id_max = (np.abs(k - klims[1])).argmin()
    windowed_section = k[id_min:id_max]
    length = len(windowed_section)
    window_insert = windows.hamming(length, sym = False)
    window = np.zeros_like(k)
    window[id_min:id_max] = window_insert
    return window

def KB_window(data, k, klims, beta = 30):
    id_min = (np.abs(k - klims[0])).argmin()
    id_max = (np.abs(k - klims[1])).argmin()
    windowed_section = data[id_min:id_max]
    length = len(windowed_section)
    try:
        window_insert = windows.kaiser_bessel_derived(length, beta, sym = True)
    except ValueError:
        id_min += 1
        windowed_section = data[id_min:id_max]
        length = len(windowed_section)
        window_insert = windows.kaiser_bessel_derived(length, beta, sym = True)
    window = np.zeros_like(data)
    window[id_min:id_max] = window_insert
    windowed_data = window*data
    return windowed_data

def KB_lineshape(k, klims, beta = 30):
    id_min = (np.abs(k - klims[0])).argmin()
    id_max = (np.abs(k - klims[1])).argmin()
    windowed_section = k[id_min:id_max]
    length = len(windowed_section)
    try:
        window_insert = windows.kaiser_bessel_derived(length, beta, sym = True)
    except ValueError:
        id_min += 1
        windowed_section = k[id_min:id_max]
        length = len(windowed_section)
        window_insert = windows.kaiser_bessel_derived(length, beta, sym = True)
    window = np.zeros_like(k)
    window[id_min:id_max] = window_insert
    return window

def hann_window(data, k, klims):
    id_min = (np.abs(k - klims[0])).argmin()
    id_max = (np.abs(k - klims[1])).argmin()
    windowed_section = data[id_min:id_max]
    length = len(windowed_section)
    window_insert = windows.hann(length, sym = False)
    window = np.zeros_like(data)
    window[id_min:id_max] = window_insert
    windowed_data = window*data
    return windowed_data

def hann_lineshape(k, klims):
    id_min = (np.abs(k - klims[0])).argmin()
    id_max = (np.abs(k - klims[1])).argmin()
    windowed_section = k[id_min:id_max]
    length = len(windowed_section)
    window_insert = windows.hann(length, sym = False)
    window = np.zeros_like(k)
    window[id_min:id_max] = window_insert
    return window

def box_window(data, k, klims):
    id_min = (np.abs(k - klims[0])).argmin()
    id_max = (np.abs(k - klims[1])).argmin()
    windowed_section = data[id_min:id_max]
    length = len(windowed_section)
    window_insert = windows.boxcar(length, sym = False)
    window = np.zeros_like(data)
    window[id_min:id_max] = window_insert
    windowed_data = window*data
    return windowed_data

#%%
tag = 'NpMg_2421_OlderData'

title = tag
k_tag = f'{tag}'+".chik"
r_tag = f'{tag}'+".chir"


k_cutout = [0.0, 11]
k_limits = [3, 10]
n_comps = 5

data = chikDict[k_tag][:,4][np.logical_and(chikDict[k_tag][:,0] > k_cutout[0], chikDict[k_tag][:,0] < k_cutout[1])]
fig = plt.figure(layout = 'constrained', figsize = (32,18))
fs1 = 28
fs2 = 24
fig.suptitle(f'PCA Results for {title}', fontsize = 48, y = 1.05)


from matplotlib.gridspec import GridSpec
gs = GridSpec(6, 4, figure=fig)

file = list(chikDict.keys())[-1]
k = chikDict[file][:,0][np.logical_and(chikDict[file][:,0] > k_cutout[0], chikDict[file][:,0] < k_cutout[1])]
k = np.array(k, float)
R = chirDict[r_tag][:,0]

raw_data = np.vstack([chikDict[file][:,4][np.logical_and(chikDict[file][:,0] > k_cutout[0], chikDict[file][:,0] < k_cutout[1])] for file in list(chikDict.keys())[1:]])
raw_data = np.array(raw_data, dtype=float)
raw_data = raw_data.T
pca_data = np.zeros_like(raw_data)
for i in range(len(raw_data[0,:])):
    pca_data[:,i] = box_window(raw_data[:,i], k, k_limits)


from sklearn.decomposition import PCA

pca = PCA(n_components=n_comps)
comps = pca.fit_transform(pca_data)
evr = pca.explained_variance_ratio_
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(np.linspace(1, len(evr), len(evr)), evr, s = 100, color = 'crimson', label = 'Explained Variance Ratio')
ax1.plot(np.linspace(1, len(evr), len(evr)), evr, color = 'dodgerblue', linewidth = 3, linestyle = '--')
ax1.legend(loc = 'upper right', fontsize = fs1)
ax1.tick_params(axis='both', labelsize= fs2)
ax1.set_ylabel("EVR", fontsize = fs1)
ax1.set_xlabel("Component", fontsize = fs1)

for i in range(1, 6):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(k, comps[:,i-1], linewidth = 3, color = 'crimson', label = f'component {i} ({evr[i-1]:.2f})')
    ax.set_xlim([k_limits[0]-1, k_limits[1]+1])
    ax.tick_params(axis='both', labelsize= fs2)
    ax.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fs1)
    if i == 5:
        ax.set_xlabel("k($\AA^{-1}$)", fontsize = fs1)
    # ax.legend(loc = 'upper left', fontsize = fs1)
    
from scipy import interpolate
N = 65536
rdfs = np.zeros([5000,n_comps])
Rhat = np.linspace(R[1],np.max(R)*2,5000)
kint = np.linspace(np.min(k), np.max(k), N)

for i in range(n_comps):
    interpolant = interpolate.interp1d(k, comps[:,i], kind="linear")
    g = interpolant(kint)
    g = g - np.mean(g)
    Ghat = lombscargle(kint, g, freqs=Rhat, normalize=True)
    rdfs[:,i] = Ghat


for i in range(1, 6):
    ax = fig.add_subplot(gs[i, 1])
    ax.plot(Rhat/2, rdfs[:,i-1], linewidth = 3, color = 'crimson', label = f'comp. {i} ({evr[i-1]:.2f})')
    ax.set_xlim([0, 5.2])
    ax.tick_params(axis='both', labelsize= fs2)
    ax.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
    if i == 5:
        ax.set_xlabel("R($\AA$)", fontsize = fs1)
    ax.legend(loc = 'upper right', fontsize = fs1)
    
ax2 = fig.add_subplot(gs[:3, 2:])

import lmfit as lm
import pandas as pd

def LCF(standards, M, **Pars):
    return np.sum([np.multiply(Pars[f'A_{m}'],standards[:,m]) for m in range(0,M)], axis = 0)

model = lm.Model(LCF)
params = lm.Parameters()
M = 5
for m in range(0, M):
    params.add(f'A_{m}', value = 1/M)
params.add("M", value = M, vary = False)
print('Model parameters constructed')
print('Fitting...')
result = model.fit(data, standards = comps[:,:M], M = M, params=params)
print(result.fit_report())
pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
results_full = result.eval(k = k)

ax2.scatter(k, data, s = 75, facecolor = 'none', edgecolor = 'black', linewidth = 1.5, label = 'Data')
# ax2.plot(k, hann_lineshape(k, k_limits)*np.max(hann_window(data, k, k_limits))*1.5, linewidth = 3, color = 'blue', label = 'Window')
ax2.axvline(x=k_limits[0], linewidth = 3, color = 'blue')
ax2.axvline(x=k_limits[1], linewidth = 3, color = 'blue', label = 'Window')
ax2.plot(k, results_full, color = 'red', linewidth = 4, alpha = 1, label = 'Fit')
ax2.set_prop_cycle('color', [plt.cm.cool(i) for i in np.linspace(0, 1, M)])
ax2.plot(k, pd_pars.iloc[0, 1]*comps[:,0], linestyle = '--', linewidth = 3, label = 'comp. 1')
# label = f'comp. 1 ({np.abs(pd_pars.iloc[0, 1])/np.sum(np.abs(pd_pars.iloc[:M, 1])):.2f})'
ax2.plot(k, pd_pars.iloc[1, 1]*comps[:,1], linestyle = '--', linewidth = 3, label = 'comp. 2')
ax2.plot(k, pd_pars.iloc[2, 1]*comps[:,2], linestyle = '--', linewidth = 3, label = 'comp. 3')
ax2.plot(k, pd_pars.iloc[3, 1]*comps[:,3], linestyle = '--', linewidth = 3, label = 'comp. 4')
ax2.plot(k, pd_pars.iloc[4, 1]*comps[:,4], linestyle = '--', linewidth = 3, label = 'comp. 5')
ax2.set_xlim([k_cutout[0], k_cutout[1]])
ax2.tick_params(axis='both', labelsize= fs2)
ax2.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fs1)
ax2.set_xlabel("k($\AA^{-1}$)", fontsize = fs1)
ax2.legend(loc = 'lower left', ncol = 4,  fontsize = fs1)


ax3 = fig.add_subplot(gs[3:, 2:])

windowed_data = hann_window(data, k, k_limits)
interpolant = interpolate.interp1d(k, windowed_data, kind="linear")
gData = interpolant(kint)
gData = gData - np.mean(gData)
data_rdf = lombscargle(kint, gData, freqs=Rhat, normalize=True)
data_rdf_unnorm = lombscargle(kint, gData, freqs=Rhat, normalize=False)

def LCF_R(standards, M, **Pars):
    return np.sum([np.multiply(Pars[f'B_{m}'],standards[:,m]) for m in range(0,M)], axis = 0)

model_R = lm.Model(LCF_R)
params_R = lm.Parameters()

for m in range(0, M):
    params_R.add(f'B_{m}', value = 1/(1+m**2), min = 0)
    
params_R.add("M", value = M, vary = False)
print('Model parameters constructed')
print('Fitting...')
result_r = model_R.fit(data_rdf, standards = rdfs, M = M, params=params_R)
pd_pars_r = pd.DataFrame([(p.name, p.value, p.stderr) for p in result_r.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
results_r_full = result_r.eval(Rhat = Rhat)

ax3.scatter(Rhat[0::10]/2, data_rdf[0::10], s = 75, facecolor = 'none', edgecolor = 'black', linewidth = 1.5)
ax3.set_prop_cycle('color', [plt.cm.cool(i) for i in np.linspace(0, 1, 5)])
ax3.plot(Rhat/2, pd_pars_r.iloc[0, 1]*rdfs[:,0], linestyle = '--', linewidth = 3, label = f'comp. 1 ({pd_pars_r.iloc[0, 1]/np.sum(pd_pars_r.iloc[:5, 1]):.2f})')
ax3.plot(Rhat/2, pd_pars_r.iloc[1, 1]*rdfs[:,1], linestyle = '--', linewidth = 3, label = f'comp. 2 ({pd_pars_r.iloc[1, 1]/np.sum(pd_pars_r.iloc[:5, 1]):.2f})')
ax3.plot(Rhat/2, pd_pars_r.iloc[2, 1]*rdfs[:,2], linestyle = '--', linewidth = 3, label = f'comp. 3 ({pd_pars_r.iloc[2, 1]/np.sum(pd_pars_r.iloc[:5, 1]):.2f})')
ax3.plot(Rhat/2, pd_pars_r.iloc[3, 1]*rdfs[:,3], linestyle = '--', linewidth = 3, label = f'comp. 4 ({pd_pars_r.iloc[3, 1]/np.sum(pd_pars_r.iloc[:5, 1]):.2f})')
ax3.plot(Rhat/2, pd_pars_r.iloc[4, 1]*rdfs[:,4], linestyle = '--', linewidth = 3, label = f'comp. 5 ({pd_pars_r.iloc[4, 1]/np.sum(pd_pars_r.iloc[:5, 1]):.2f})')
ax3.plot(Rhat/2, results_r_full, color = 'red', linewidth = 4, alpha = 0.75)
ax3.set_xlim([0, 5.2])
ax3.tick_params(axis='both', labelsize= fs2)
ax3.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
ax3.set_xlabel("R($\AA$)", fontsize = fs1)
ax3.legend(loc = 'upper right', fontsize = fs1)

plt.show()

#%% compare two
tag = 'NpMg_2421_OlderData'
sec_tag = 'NpMg_300_T'

title = tag
k_tag = f'{tag}'+".chik"
r_tag = f'{tag}'+".chir"


k_cutout = [0.0, 11]
k_limits = [2, 11]
n_comps = 5

data = chikDict[k_tag][:,4][np.logical_and(chikDict[k_tag][:,0] > k_cutout[0], chikDict[k_tag][:,0] < k_cutout[1])]
fig = plt.figure(layout = 'constrained', figsize = (32,18))
fs1 = 28
fs2 = 24
fig.suptitle(f'PCA Results for {title}', fontsize = 48, y = 1.05)


from matplotlib.gridspec import GridSpec
gs = GridSpec(6, 4, figure=fig)

file = list(chikDict.keys())[-1]
k = chikDict[file][:,0][np.logical_and(chikDict[file][:,0] > k_cutout[0], chikDict[file][:,0] < k_cutout[1])]
k = np.array(k, float)
R = chirDict[r_tag][:,0]

raw_data = np.vstack([chikDict[file][:,4][np.logical_and(chikDict[file][:,0] > k_cutout[0], chikDict[file][:,0] < k_cutout[1])] for file in list(chikDict.keys())[1:]])
raw_data = np.array(raw_data, dtype=float)
raw_data = raw_data.T
pca_data = np.zeros_like(raw_data)
for i in range(len(raw_data[0,:])):
    pca_data[:,i] = KB_window(raw_data[:,i], k, k_limits)


from sklearn.decomposition import PCA

pca = PCA(n_components=n_comps)
comps = pca.fit_transform(pca_data)
evr = pca.explained_variance_ratio_
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(np.linspace(1, len(evr), len(evr)), evr, s = 100, color = 'crimson', label = 'Explained Variance Ratio')
ax1.plot(np.linspace(1, len(evr), len(evr)), evr, color = 'dodgerblue', linewidth = 3, linestyle = '--')
ax1.legend(loc = 'upper right', fontsize = fs1)
ax1.tick_params(axis='both', labelsize= fs2)
ax1.set_ylabel("EVR", fontsize = fs1)
ax1.set_xlabel("Component", fontsize = fs1)

for i in range(1, 6):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(k, comps[:,i-1], linewidth = 3, color = 'crimson', label = f'component {i} ({evr[i-1]:.2f})')
    ax.set_xlim([k_limits[0]-1, k_limits[1]+1])
    ax.tick_params(axis='both', labelsize= fs2)
    ax.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fs1)
    if i == 5:
        ax.set_xlabel("k($\AA^{-1}$)", fontsize = fs1)
    # ax.legend(loc = 'upper left', fontsize = fs1)
    
from scipy import interpolate
N = 65536
rdfs = np.zeros([5000,n_comps])
Rhat = np.linspace(R[1],np.max(R)*2,5000)
kint = np.linspace(np.min(k), np.max(k), N)

for i in range(n_comps):
    interpolant = interpolate.interp1d(k, comps[:,i], kind="linear")
    g = interpolant(kint)
    g = g - np.mean(g)
    Ghat = lombscargle(kint, g, freqs=Rhat, normalize=True)
    rdfs[:,i] = Ghat


for i in range(1, 6):
    ax = fig.add_subplot(gs[i, 1])
    ax.plot(Rhat/2, rdfs[:,i-1], linewidth = 3, color = 'crimson', label = f'comp. {i} ({evr[i-1]:.2f})')
    ax.set_xlim([0, 5.2])
    ax.tick_params(axis='both', labelsize= fs2)
    ax.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
    if i == 5:
        ax.set_xlabel("R($\AA$)", fontsize = fs1)
    ax.legend(loc = 'upper right', fontsize = fs1)
    
ax2 = fig.add_subplot(gs[:3, 2:])

import lmfit as lm
import pandas as pd

def LCF(standards, M, **Pars):
    return np.sum([np.multiply(Pars[f'A_{m}'],standards[:,m]) for m in range(0,M)], axis = 0)

model = lm.Model(LCF)
params = lm.Parameters()
M = 5
for m in range(0, M):
    params.add(f'A_{m}', value = 1/M)
params.add("M", value = M, vary = False)
print('Model parameters constructed')
print('Fitting...')
result = model.fit(data, standards = comps[:,:M], M = M, params=params)
print(result.fit_report())
pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
results_full = result.eval(k = k)

ax2.scatter(k, data, s = 75, facecolor = 'none', edgecolor = 'black', linewidth = 1.5, label = 'Data')
# ax2.plot(k, hann_lineshape(k, k_limits)*np.max(hann_window(data, k, k_limits))*1.5, linewidth = 3, color = 'blue', label = 'Window')
ax2.axvline(x=k_limits[0], linewidth = 3, color = 'blue')
ax2.axvline(x=k_limits[1], linewidth = 3, color = 'blue', label = 'Window')
ax2.plot(k, results_full, color = 'red', linewidth = 4, alpha = 1, label = 'Fit')
ax2.set_prop_cycle('color', [plt.cm.cool(i) for i in np.linspace(0, 1, M)])
ax2.plot(k, pd_pars.iloc[0, 1]*comps[:,0], linestyle = '--', linewidth = 3, label = 'comp. 1')
# label = f'comp. 1 ({np.abs(pd_pars.iloc[0, 1])/np.sum(np.abs(pd_pars.iloc[:M, 1])):.2f})'
ax2.plot(k, pd_pars.iloc[1, 1]*comps[:,1], linestyle = '--', linewidth = 3, label = 'comp. 2')
ax2.plot(k, pd_pars.iloc[2, 1]*comps[:,2], linestyle = '--', linewidth = 3, label = 'comp. 3')
ax2.plot(k, pd_pars.iloc[3, 1]*comps[:,3], linestyle = '--', linewidth = 3, label = 'comp. 4')
ax2.plot(k, pd_pars.iloc[4, 1]*comps[:,4], linestyle = '--', linewidth = 3, label = 'comp. 5')
ax2.set_xlim([k_cutout[0], k_cutout[1]])
ax2.tick_params(axis='both', labelsize= fs2)
ax2.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fs1)
ax2.set_xlabel("k($\AA^{-1}$)", fontsize = fs1)
ax2.legend(loc = 'lower left', ncol = 4,  fontsize = fs1)


ax3 = fig.add_subplot(gs[3:, 2:])
k_tag2 = f'{sec_tag}'+".chik"
r_tag2 = f'{sec_tag}'+".chir"
data2 = chikDict[k_tag2][:,4][np.logical_and(chikDict[k_tag2][:,0] > k_cutout[0], chikDict[k_tag2][:,0] < k_cutout[1])]
# k2 = chikDict[file][:,0][np.logical_and(chikDict[file][:,0] > k_cutout[0], chikDict[file][:,0] < k_cutout[1])]
for m in range(0, M):
    params.add(f'A_{m}', value = 1/M)
params.add("M", value = M, vary = False)
print('Model parameters constructed')
print('Fitting...')
result = model.fit(data2, standards = comps[:,:M], M = M, params=params)
print(result.fit_report())
pd_pars2 = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
results_full2 = result.eval(k = k)

ax3.scatter(k, data2, s = 75, facecolor = 'none', edgecolor = 'black', linewidth = 1.5, label = 'Data')
ax3.plot(k, KB_lineshape(k, k_limits)*np.max(hann_window(data, k, k_limits))*1.5, linewidth = 3, color = 'blue', label = 'Window')
# ax3.axvline(x=k_limits[0], linewidth = 3, color = 'blue')
# ax3.axvline(x=k_limits[1], linewidth = 3, color = 'blue', label = 'Window')
ax3.plot(k, results_full2, color = 'red', linewidth = 4, alpha = 1, label = 'Fit')
ax3.set_prop_cycle('color', [plt.cm.cool(i) for i in np.linspace(0, 1, M)])
ax3.plot(k, pd_pars2.iloc[0, 1]*comps[:,0], linestyle = '--', linewidth = 3, label = 'comp. 1')
# label = f'comp. 1 ({np.abs(pd_pars.iloc[0, 1])/np.sum(np.abs(pd_pars.iloc[:M, 1])):.2f})'
ax3.plot(k, pd_pars2.iloc[1, 1]*comps[:,1], linestyle = '--', linewidth = 3, label = 'comp. 2')
ax3.plot(k, pd_pars2.iloc[2, 1]*comps[:,2], linestyle = '--', linewidth = 3, label = 'comp. 3')
ax3.plot(k, pd_pars2.iloc[3, 1]*comps[:,3], linestyle = '--', linewidth = 3, label = 'comp. 4')
ax3.plot(k, pd_pars2.iloc[4, 1]*comps[:,4], linestyle = '--', linewidth = 3, label = 'comp. 5')
ax3.set_xlim([k_cutout[0], k_cutout[1]])
ax3.tick_params(axis='both', labelsize= fs2)
ax3.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fs1)
ax3.set_xlabel("k($\AA^{-1}$)", fontsize = fs1)
ax3.legend(loc = 'lower left', ncol = 4,  fontsize = fs1)

plt.show()
#%%

def comp_out_k1(comps, comp_number):
    col1 = k.copy()
    col2 = np.divide(comps[:,comp_number],col1**3)
    out = np.vstack([col1,col2]).T
    return out

comp3 = comp_out_k1(comps, 2)
np.savetxt('NpCarbPCA_c2.dat', comp2, delimiter = ',')