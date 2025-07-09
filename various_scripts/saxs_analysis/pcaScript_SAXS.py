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
saxsDict = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
    # skip directories
        continue
    if str(file).endswith('.json'):
        continue
    if str(file).endswith('.xlsx'):
        continue
    if str(file).endswith('.pptx'):
        continue
    with open(file):
        if str(file).endswith('lqhq.dat'):
            key = 's'+str(file)[4:9]+'_lqhq'
            saxsDict[key] = np.loadtxt(file, skiprows = 4)
        else:
            continue
def normalize(data, point='maximum', factor = 1):
    """
    Normalizes a vector to the maximum found within the vector.
    Negative values will be converted to positive ones.
    Parameters
    ----------
    data : array-like
    The array to normalize to its maximum.

    Returns
    -------
    normalized_data : array-like
    Normalized array of the same shape as passed in.

    """
    if point=='maximum':
        normalized_data = np.abs(data)/max(data)
    normalized_data *= factor
    return normalized_data

def derivative(data):
    out = np.zeros_like(data)
    out[:,0] = data[:,0]
    # dq = np.mean(np.diff(data[:,0])) # standard for exafs
    dIdq = np.gradient(data[:,1], data[:,0])
    out[:,1] = np.abs(dIdq)
    return out

def interpolator(data, Npts):
    out = np.zeros((Npts, 2))
    x, y = data[:,0], data[:,1]
    x_ = np.linspace(np.min(x), np.max(x), Npts)
    y_ = np.interp(x_, x, y)
    out[:,0], out[:,1] = x_, y_
    return out

saxsDict_intp = OrderedDict()
for key in saxsDict.keys():
    if key == 'headers':
        continue
    saxsDict_intp[key+'_intp'] = interpolator(saxsDict[key], 50000)
    
saxsDict_derv = OrderedDict()
for key in saxsDict_intp.keys():
    saxsDict_derv[key+'_d1'] = derivative(saxsDict_intp[key])
    
saxsDict_intp_n = OrderedDict()
for key in saxsDict_intp.keys():
    if key == 'headers':
        continue
    saxsDict_intp_n[key+'_n'] = np.zeros_like(saxsDict_intp[key])
    saxsDict_intp_n[key+'_n'][:,0] = saxsDict_intp[key][:,0]
    saxsDict_intp_n[key+'_n'][:,1] = normalize(saxsDict_intp[key][:,1])

    
#%%

import seaborn as sns
fig, ax = plt.subplots(figsize = (10, 10))
ax.set(xscale = 'log', yscale = 'log')

dset = saxsDict_intp

for key, i in zip(dset.keys(), range(len(dset.keys()))):
    if key == 'headers':
        continue
    sns.scatterplot(x = dset[key][:,0], y = np.multiply(100**i, dset[key][:,1]), ax=ax)
    

plt.show()
#%%



dataDict = saxsDict_intp

q_cutout = [0.004, 1]
q_limits = [0, 1]
n_comps = 6

fig = plt.figure(layout = 'constrained', figsize = (32,18))
fs1 = 28
fs2 = 24



from matplotlib.gridspec import GridSpec
gs = GridSpec(6, 3, figure=fig)

file = list(dataDict.keys())[-1]
q = dataDict[file][:,0][np.logical_and(dataDict[file][:,0] > q_cutout[0], dataDict[file][:,0] < q_cutout[1])]
q = np.array(q, float)
# I = dataDict[title][:,0]

raw_data = np.vstack([dataDict[file][:,1][np.logical_and(dataDict[file][:,0] > q_cutout[0], dataDict[file][:,0] < q_cutout[1])] for file in list(dataDict.keys())])
raw_data = np.array(raw_data, dtype=float)
raw_data = raw_data.T
pca_data = np.zeros_like(raw_data)
for i in range(len(raw_data[0,:])):
    pca_data[:,i] = np.log10(raw_data[:,i])

    
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
pca_data = imputer.fit_transform(pca_data)

pca = PCA(n_components=n_comps)
comps = pca.fit_transform(pca_data)
evr = pca.explained_variance_ratio_
ax1 = fig.add_subplot(gs[0, :1])
ax1.scatter(np.linspace(1, len(evr), len(evr)), evr, s = 100, color = 'crimson', label = 'Explained Variance Ratio')
ax1.plot(np.linspace(1, len(evr), len(evr)), evr, color = 'dodgerblue', linewidth = 3, linestyle = '--')
ax1.legend(loc = 'upper right', fontsize = fs1)
ax1.tick_params(axis='both', labelsize= fs2)
ax1.set_ylabel("EVR", fontsize = fs1)
ax1.set_xlabel("Component", fontsize = fs1)

for i in range(1, 6):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(q, comps[:,i-1], linewidth = 3, color = 'crimson', label = f'component {i} ({evr[i-1]:.2f})')
    # ax.set_xlim([q_limits[0], q_limits[1]])
    ax.tick_params(axis='both', labelsize= fs2)
    ax.set_ylabel("I (a.u.)", fontsize = fs1)
    ax.set_xscale('log')
    if i == 5:
        ax.set_xlabel("k($\AA^{-1}$)", fontsize = fs1)
    ax.legend(loc = 'upper left', fontsize = fs1)
    
    
    
which_sample = '2203' 

tag_i = 's_'+which_sample+'_lqhq_intp'
tag_d = 's_'+which_sample+'_lqhq_intp_d1'

    
ax2 = fig.add_subplot(gs[:3, 1:])

fig.suptitle(f'PCA Results for sample {which_sample}', fontsize = 48, y = 1.05)
data = dataDict[tag_i][:,1][np.logical_and(dataDict[tag_i][:,0] > q_cutout[0], dataDict[tag_i][:,0] < q_cutout[1])]
data = np.log10(data)

import lmfit as lm
import pandas as pd

def LCF(standards, M, **Pars):
    return np.sum([np.multiply(Pars[f'A_{m}'],standards[:,m]) for m in range(0,M)], axis = 0) + Pars['fudge']

model = lm.Model(LCF)
params = lm.Parameters()
M = 6
for m in range(0, M):
    params.add(f'A_{m}', value = 1/M)
params.add("M", value = M, vary = False)
params.add("fudge", value = 0)
print('Model parameters constructed')
print('Fitting...')
result = model.fit(data, standards = comps[:,:M], M = M, params=params)
print(result.fit_report())
pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
results_full = result.eval(q = q)

ax2.scatter(q, data, s = 75, facecolor = 'none', edgecolor = 'black', linewidth = 1.5, label = 'Data')
# ax2.plot(k, hann_lineshape(k, k_limits)*np.max(hann_window(data, k, k_limits))*1.5, linewidth = 3, color = 'blue', label = 'Window')
ax2.plot(q, results_full, color = 'red', linewidth = 4, alpha = 1, label = 'Fit')
ax2.set_prop_cycle('color', [plt.cm.cool(i) for i in np.linspace(0, 1, M)])
# ax2.plot(q, pd_pars.iloc[0, 1]*comps[:,0], linestyle = '--', linewidth = 3, label = f'comp. 1 ({np.abs(pd_pars.iloc[0, 1])/np.sum(np.abs(pd_pars.iloc[:M, 1])):.2f})')
# ax2.plot(q, pd_pars.iloc[1, 1]*comps[:,1], linestyle = '--', linewidth = 3, label = f'comp. 2 ({np.abs(pd_pars.iloc[1, 1])/np.sum(np.abs(pd_pars.iloc[:M, 1])):.2f})')
# ax2.plot(q, pd_pars.iloc[2, 1]*comps[:,2], linestyle = '--', linewidth = 3, label = f'comp. 3 ({np.abs(pd_pars.iloc[2, 1])/np.sum(np.abs(pd_pars.iloc[:M, 1])):.2f})')
# ax2.plot(q, pd_pars.iloc[3, 1]*comps[:,3], linestyle = '--', linewidth = 3, label = f'comp. 4 ({np.abs(pd_pars.iloc[3, 1])/np.sum(np.abs(pd_pars.iloc[:M, 1])):.2f})')
for i in range(0, M):
    ax2.plot(q, pd_pars.iloc[i, 1]*comps[:,i], linestyle = '--', linewidth = 3)
    # ax2.plot(q, pd_pars.iloc[1, 1]*comps[:,1], linestyle = '--', linewidth = 3)
    # ax2.plot(q, pd_pars.iloc[2, 1]*comps[:,2], linestyle = '--', linewidth = 3)
    # ax2.plot(q, pd_pars.iloc[3, 1]*comps[:,3], linestyle = '--', linewidth = 3)
# ax.set_xscale('log')
# ax2.plot(q, pd_pars.iloc[4, 1]*comps[:,4], linestyle = '--', linewidth = 3)
# ax2.plot(q, pd_pars.iloc[5, 1]*comps[:,5], linestyle = '--', linewidth = 3, label = f'comp. 5 ({np.abs(pd_pars.iloc[4, 1])/np.sum(np.abs(pd_pars.iloc[:M, 1])):.2f})')
ax2.set_xscale('log')
# ax2.set_xlim([k_cutout[0], k_cutout[1]])
# ax2.tick_params(axis='both', labelsize= fs2)
# ax2.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fs1)
# ax2.set_xlabel("k($\AA^{-1}$)", fontsize = fs1)
ax2.legend(loc = 'lower left', ncol = 2,  fontsize = fs1)

ax3 = fig.add_subplot(gs[3:, 1:])
dataDict = saxsDict_derv.copy()

data = dataDict[tag_d][:,1][np.logical_and(dataDict[tag_d][:,0] > q_cutout[0], dataDict[tag_d][:,0] < q_cutout[1])]
data = np.log10(data)
ax3.scatter(q, data, s = 75, facecolor = 'none', edgecolor = 'black', linewidth = 1.5, label = 'Derivative')
ax3.set_xscale('log')
ax3.legend(loc = 'lower left', ncol = 2,  fontsize = fs1)
#%%