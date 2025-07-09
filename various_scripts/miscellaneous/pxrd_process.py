# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:09:52 2024

@author: boris
"""

import numpy as np
import os

import lmfit as lm
import matplotlib.pyplot as plt
import lmfit as lm
import pandas as pd
from collections import OrderedDict
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy import sparse
from scipy.sparse.linalg import spsolve

from lmfit import (Minimizer, conf_interval, conf_interval2d, create_params,
                   report_ci, report_fit)
#%% import the data
# set up for standard ascii files in .dat, .xyd or .xye formats
# make sure to run this block of code in a directory where you have data ONLY to avoid errors
# the files dictionary will contain the raw data 
files = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
        # skip directories
        continue
    if str(file).endswith('.png'):
        continue
    with open(file):
        files[str(file)] = np.loadtxt(file)
#%%
# define a couple helpful functions
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

def qconv(two_theta, wavelength):
    return 4*np.pi*np.sin((two_theta/2)*(np.pi/180))/wavelength

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z


def residual_fit_peaks(Pars, x, peaks, data):
    N = len(peaks)
    res = {}
    for i in range(1, N+1):
        res[f'P_{i}'] = np.zeros_like(x)
        meow = Pars[f'meow_{i}']
        a = Pars[f'a_{i}']
        w = Pars[f'w_{i}']
        e = x[int(peaks[i-1])]
        res[f'P_{i}'] = a*(meow*(w**2 / (w**2 + (2*x - 2*e)**2)) + (1-meow)*(np.exp(-(x-e)**2/(2*w**2))))
    total = np.sum([res[f'P_{i}'] for i in range(1, N+1)], axis = 0)
    return total - windowed_data
#%% if available, use a standard to get instrumental broadening
std_keys = ['LaB6_150.0001.dat']
ref_keys = ['capillar-0.3.0008.dat']
fs1 = 28
fs2 = 24

wl = 0.75

fig, axs = plt.subplots(3,2, figsize = (24, 12))

ax = axs.flatten()
ax[0].tick_params('both', labelsize = fs2)
ax[0].set_xlabel("Q ($\AA^{-1}$)", fontsize = fs1)
ax[0].set_ylabel("Intensity", fontsize = fs1)
ax[0].set_prop_cycle('color', ['red', 'blue'])
for key in std_keys:
    dset = files[key]
    q = qconv(dset[:,0], wl)
    y = dset[:,1]
    ax[0].plot(q,y)
ax[0].set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 0.95, 3)])
for key in ref_keys:
    dset = files[key]
    q = qconv(dset[:,0], wl)
    y = dset[:,1]
    ax[0].plot(q,y)
ax[0].legend(labels = std_keys+ref_keys, loc = 'upper right', fontsize = fs1)  
ax[0].axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
ax[0].axhline(y = 1, linestyle = '--', color = 'gray', linewidth = 0.5)

rawdset = files['LaB6_150.0001.dat']
rawback = files['capillar-0.3.0008.dat']

dset_xdata = rawdset[:,0]
dset_ydata = rawdset[:,1]
back_xdata = rawback[:,0]
back_ydata = rawback[:,1]

backmax = np.argmax(back_ydata)+0 # if bg max coincides with a peak, shift this a little
backnorm = normalize(back_ydata, 'maximum', 1) 

scalefactor = dset_ydata[backmax]/backnorm[backmax]
dsetnorm = dset_ydata/scalefactor

if len(dsetnorm)-len(backnorm) != 0:
    end = abs(len(dsetnorm) - len(backnorm))
    if len(dsetnorm) > len(backnorm):
        dsetnorm = dsetnorm[:-end]
        dset_xdata = dset_xdata[:-end]
    elif len(backnorm) > len(dsetnorm):
        backnorm = backnorm[end:]
        back_xdata = back_xdata[end:]
        
dset_bgsub = dsetnorm - backnorm
for i in range(len(dset_bgsub)):
    if dset_bgsub[i] < 0:
        dset_bgsub[i] = 0
        
ax[1].xaxis.set_major_locator(MultipleLocator(5))
ax[1].xaxis.set_minor_locator(MultipleLocator(1))
ax[1].tick_params('both', labelsize = fs2)
ax[1].set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax[1].set_ylabel("Intensity", fontsize = fs1)
ax[1].set_prop_cycle('color', ['black', 'red', 'blue', 'green'])
ax[1].plot(back_xdata, backnorm, linestyle = '-.', label = 'Background')
ax[1].plot(dset_xdata, dsetnorm, label ='Raw Data')
ax[1].plot(back_xdata, dset_bgsub, label ='Bkg-subtracted')
ax[1].legend(loc = 'upper right', fontsize = fs1)  
ax[1].axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
ax[1].axhline(y = 1, linestyle = '--', color = 'gray', linewidth = 0.5)


dset_xdata = back_xdata
dset_ydata = dset_bgsub

clndset = dset_ydata - baseline_als(dset_ydata, 10**5.5, 0.001, 100)

ax[2].xaxis.set_major_locator(MultipleLocator(5))
ax[2].xaxis.set_minor_locator(MultipleLocator(1))
ax[2].tick_params('both', labelsize = fs2)
ax[2].set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax[2].set_ylabel("Intensity", fontsize = fs1)
ax[2].set_prop_cycle('color', ['black', 'red', 'blue', 'green'])
ax[2].plot(dset_xdata, dset_ydata)
ax[2].plot(dset_xdata, clndset)
ax[2].legend(labels = ['Original data', 'Subtracted data'], loc = 'upper right', fontsize = fs1)  
ax[2].axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)

# carry over the variables from last section
region = [8, 55] # define the region in which to search for peaks
x0 = dset_xdata # the full 2theta axis
reference = True

# clean_to_use = clndset
clean_to_use = clndset

# window the data and the 2theta based on the region 
idx_x_min = (np.abs(x0 - region[0])).argmin()
idx_x_max = (np.abs(x0 - region[1])).argmin()
windowed_data = clean_to_use[idx_x_min:idx_x_max]
x1 = x0[idx_x_min:idx_x_max]

# find the peak positions
threshold = 8
from scipy.signal import find_peaks
peaks, _ = find_peaks(windowed_data, height=threshold, distance = 1)
# the distance variable to find_peaks defines peak separation 
# it helps when peaks are broad

# check the peak positions
ax[3].xaxis.set_major_locator(MultipleLocator(5))
ax[3].xaxis.set_minor_locator(MultipleLocator(1))
ax[3].tick_params('both', labelsize = fs2)
ax[3].set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax[3].set_ylabel("Intensity", fontsize = fs1)
ax[3].set_prop_cycle('color', ['black', 'red', 'blue', 'green'])
ax[3].axhline(y = threshold, linestyle = '--', linewidth = 0.5, color = 'lightgrey')
ax[3].axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
ax[3].plot(x0, clean_to_use, linewidth = 2, label = 'Clean data')
ax[3].plot(x1, windowed_data, linewidth = 2, label = 'Scherrer analysis region')
ax[3].legend(loc = 'upper right', fontsize = fs1)
ax[3].scatter(x1[peaks], windowed_data[peaks], s = 50, color = 'black')
    
params = lm.Parameters()
N = len(peaks)
for i in range(1, N+1):
    params.add(f'a_{i}', value = 1, min = 0)
    params.add(f'w_{i}', value=0.05, min = 0)
    params.add(f'meow_{i}', value=0.5, min = 0, max = 1)
    
mini = lm.Minimizer(residual_fit_peaks, 
                    params, fcn_args=(x1, peaks, ), fcn_kws={'data': windowed_data})
out = mini.leastsq()

fit = residual_fit_peaks(out.params, x1, peaks, windowed_data)

for p in out.params:
    out.params[p].stderr = abs(out.params[p].value * 0.1)
    
pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in out.params.values()], 
                        columns=('name', 'best-fit value', 'standard error'))
fwhms = []
for i in range(1, len(pd_pars.iloc[:,0]), 3):
    if str(pd_pars.iloc[i,0]):
        fwhms.append(pd_pars.iloc[i,1]*2*(np.pi/180))

stdFWHM = np.average(fwhms)

test = residual_fit_peaks(out.params, x1, peaks, windowed_data) + windowed_data

ax[4].xaxis.set_major_locator(MultipleLocator(5))
ax[4].xaxis.set_minor_locator(MultipleLocator(1))
ax[4].tick_params('both', labelsize = fs2)
ax[4].set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax[4].set_ylabel("Intensity", fontsize = fs1)
ax[4].set_prop_cycle('color', ['black', 'red', 'blue', 'green'])
ax[4].plot(x1, windowed_data)
ax[4].plot(x1, test)
ax[4].legend(labels = ['data', 'fit'], fontsize = fs1, loc = 'upper right')
ax[4].scatter(x1[peaks], windowed_data[peaks], s = 100)

x2 = np.log(1/(np.cos((x1[peaks]/2)*(np.pi/180))))
# fwhms = [np.log(ci[f'w_{i+1}'][1][1]*2*(np.pi/180)) for i in range(len(peaks))]
# fwhm_errors = np.zeros([2,len(peaks)])
# fwhm_errors[0,:] = [np.log(ci[f'w_{i+1}'][0][1]*2*(np.pi/180)) for i in range(len(peaks))]
# fwhm_errors[1,:] = [np.log(ci[f'w_{i+1}'][2][1]*2*(np.pi/180)) for i in range(len(peaks))]
# fwhm_errors = abs(fwhm_errors - fwhms)


z2 = np.polyfit(x2, np.log(fwhms), 1, cov = True)

ax[5].scatter(x2, np.log(fwhms), s = 100, color = 'black')
# ax[5].errorbar(x2, fwhms, yerr = fwhm_errors, ls = 'none', color = 'black', linewidth = 1)
ax[5].tick_params('both', labelsize = fs2)
ax[5].set_xlabel(r'ln($\frac{1}{cos\theta}$)', fontsize = fs1)
ax[5].set_ylabel(r"ln$\beta$", fontsize = fs1)
ax[5].plot(x2, z2[0][0]*x2 + z2[0][1], linestyle = '--', linewidth = 3, color = 'red', label = 'linear')
ax[5].legend(fontsize = fs1, loc = 'lower right')
# intercept = z2[0][1]
# K = 0.89
# lam = 0.075
# L = (K*lam)/np.exp(intercept) 
# Lerr = (K*lam)/np.exp(z2[1][1,1])
# ax[5].annotate(f'size = {L:.2f} $\pm$ {Lerr:.2f} nm', [0.01, -5.3], color = 'red', fontsize = fs2)

print(f'the instrumental broadening is: {stdFWHM:.5f}')

plt.tight_layout()
plt.show()
# MOST VARIABLES IN THIS SECTION WILL BE OVERWRITTEN LATER


#%% from your files dictionary, select keys to plot for a quick view
dat_keys = ['Kuz_BK_0425_Ce1_phos5_7K.0013.dat',
            'Kuz_BK_0425_Ce1_phos5_21K.0014.dat',
            'Kuzenkova_Ce_phos_10to15.0055(1).dat']
ref_keys = []
fs1 = 18
fs2 = 14

wl = 0.75
#'nath2phos_0p74_full.xye','NaTh$_2$(PO$_4$)$_3$ cif', 
fig, ax = plt.subplots(figsize = (8,1.25*(len(dat_keys)+len(ref_keys))))
# ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax.xaxis.set_minor_locator(MultipleLocator(0.2))
# ax.set_xlim([3, 35])
ax.tick_params('both', labelsize = fs2)
ax.set_yticklabels([])
# ax.set_xlabel("Q ($\AA^{-1}$)", fontsize = fs1)
ax.set_xlabel("2theta (degrees)", fontsize = fs1)
ax.set_ylabel("Intensity", fontsize = fs1)
# ax.set_prop_cycle('color', ['red', 'blue', 'green', 'magenta'])
i = len(dat_keys)*0.75
for key in dat_keys:
    dset = files[key]
    # q = qconv(dset[:,0], 0.74)
    q = dset[:,0]
    y = normalize(dset[:,1])+i
    ax.plot(q,y)
    i-= 0.75
ax.set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0.1, 0.8, 4)])

for key in ref_keys:
    dset = files[key]
    # q = qconv(dset[:,0], 1.54)
    q = dset[:,0]
    y = normalize(dset[:,1])
    ax.plot(q,y)
    i-=0.75
ax.set_xlim([5, 55])
ax.set_ylim([0,3])
ax.legend(labels = dat_keys+ref_keys, loc = 'upper right', fontsize = 12)  
# ax.legend(labels = dat_keys+ref_keys, loc = 'upper right', fontsize = fs1)  
# ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
# ax.axhline(y = 1, linestyle = '--', color = 'gray', linewidth = 0.5)
# ax.set_xlim([0, 250])
# ax.plot(data2['2 Theta'], data2['NI'], color = 'red', linewidth = 3, label = 'NU-901')
# ax.plot(traces_n1['wl'], traces_n1['BK80'], color = 'blue', linewidth = 3, label = '80 (phase-pure)')
# ax.plot(traces_n1['wl'], traces_n1['BK83'], color = 'green', linewidth = 3, label = '83 (Ti-SIM-1c)')
# ax.plot(traces_n1['wl'], traces_n1['BK84'], color = 'cyan', linewidth = 3, label = '84 (Zn-SIM-1c)')

plt.tight_layout()
plt.show()


#%% Background Subtraction via the measured background function
rawdset = files['Kuzenkova_Th_phos_1M_48.0002.dat']
rawback = files['capillar-0.3.0008.dat']
dset_lambda = 0.74

dset_xdata = rawdset[:,0]
dset_ydata = rawdset[:,1]
back_xdata = rawback[:,0]
back_ydata = rawback[:,1]

backmax = np.argmax(back_ydata)+40 # if bg max coincides with a peak, shift this a little
backnorm = normalize(back_ydata, 'maximum', 1) 

scalefactor = dset_ydata[backmax]/backnorm[backmax]
dsetnorm = dset_ydata/scalefactor

if len(dsetnorm)-len(backnorm) != 0:
    end = abs(len(dsetnorm) - len(backnorm))
    if len(dsetnorm) > len(backnorm):
        dsetnorm = dsetnorm[:-end]
        dset_xdata = dset_xdata[:-end]
    elif len(backnorm) > len(dsetnorm):
        backnorm = backnorm[end:]
        back_xdata = back_xdata[end:]
        
dset_bgsub = dsetnorm - backnorm
for i in range(len(dset_bgsub)):
    if dset_bgsub[i] < 0:
        dset_bgsub[i] = 0
        
fig, ax = plt.subplots(figsize = (16,6))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax.set_ylabel("Intensity", fontsize = fs1)
ax.set_prop_cycle('color', ['black', 'red', 'blue', 'green'])
ax.plot(back_xdata, backnorm, linestyle = '-.', label = 'Background')
ax.plot(dset_xdata, dsetnorm, label ='Raw Data')
ax.plot(back_xdata, dset_bgsub, label ='Bkg-subtracted')
ax.legend(loc = 'upper right', fontsize = fs1)  
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
ax.axhline(y = 1, linestyle = '--', color = 'gray', linewidth = 0.5)
ax.axvline(x = back_xdata[backmax])
plt.tight_layout()
plt.show()
#%% Background Subtraction via ALS (Eilers and Boelens method)

dset_lambda = 0.74
fs1 = 28
fs2 = 24

# DID YOU SUBTRACT THE BACKGROUND?
bg_subtracted = False

# IF NOT, RUN ALS ON RAW DATA FILE

key = 'Plakhova_Th_1MPO4_1_0.74A_0022.dat'
rawdset = files[key]

if bg_subtracted:
    dset_xdata = back_xdata
    dset_ydata = dset_bgsub
else:
    dset_xdata = rawdset[:,0]
    dset_ydata = rawdset[:,1]


# Background subtraction section
# Asymmetric Least Squares Smoothing

from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z
clndset = dset_ydata - baseline_als(dset_ydata, 10**7, 0.004, 100)

fig, ax = plt.subplots(figsize = (16,6))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax.set_ylabel("Intensity", fontsize = fs1)
ax.set_prop_cycle('color', ['black', 'red', 'blue', 'green'])
ax.plot(dset_xdata, dset_ydata)
ax.plot(dset_xdata, clndset)
ax.legend(labels = ['Original data', 'Subtracted data'], loc = 'upper right', fontsize = fs1)  
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)

plt.tight_layout()
plt.show()
np.savetxt('CLEAN_'+key, np.vstack((dset_xdata, clndset)).T)

#%% Plot Clean Pattern Versus the cif file if you have one
ref_keys = ['puo2sim_0p75_full.xye']
fs1 = 28
fs2 = 24

fig, ax = plt.subplots(figsize = (16,6))
# ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax.xaxis.set_minor_locator(MultipleLocator(0.2))
# ax.set_xlim([2,15])
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel("Q ($\AA^{-1}$)", fontsize = fs1)
ax.set_ylabel("Norm. Intensity", fontsize = fs1)
ax.set_prop_cycle('color', ['red', 'blue'])
ax.plot(qconv(dset_xdata, 0.74), normalize(clndset))
# ax.set_xlim([40, 250])
for key in ref_keys:
    dset = files[key]
    q = qconv(dset[:,0], 0.75)
    y = normalize(dset[:,1])
    ax.plot(q,y)
ax.legend(labels = ['Clean data', 'Reference'], loc = 'upper right', fontsize = fs1)  
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
ax.axhline(y = 1, linestyle = '--', color = 'gray', linewidth = 0.5)
# ax.set_xlim([0, 250])
# ax.plot(data2['2 Theta'], data2['NI'], color = 'red', linewidth = 3, label = 'NU-901')
# ax.plot(traces_n1['wl'], traces_n1['BK80'], color = 'blue', linewidth = 3, label = '80 (phase-pure)')
# ax.plot(traces_n1['wl'], traces_n1['BK83'], color = 'green', linewidth = 3, label = '83 (Ti-SIM-1c)')
# ax.plot(traces_n1['wl'], traces_n1['BK84'], color = 'cyan', linewidth = 3, label = '84 (Zn-SIM-1c)')

plt.tight_layout()
# plt.savefig('NU1k_NU901_overlaid.svg')

#%% Scherrer analysis
# carry over the variables from last section
region = [8, 50] # define the region in which to search for peaks
x0 = dset_xdata # the full 2theta axis
reference = True
ref_keys = ['puo2sim_1p54_full.xye']
# clean_to_use = clndset
clean_to_use = clndset

# window the data and the 2theta based on the region 
idx_x_min = (np.abs(x0 - region[0])).argmin()
idx_x_max = (np.abs(x0 - region[1])).argmin()
windowed_data = clean_to_use[idx_x_min:idx_x_max]
x1 = x0[idx_x_min:idx_x_max]

# find the peak positions
threshold = 0.018
from scipy.signal import find_peaks
peaks, _ = find_peaks(windowed_data, height=threshold, distance = 117)
# the distance variable to find_peaks defines peak separation 
# it helps when peaks are broad

# check the peak positions
fig, ax = plt.subplots(figsize = (16,6))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.tick_params('both', labelsize = fs2)
ax.set_yticklabels([])
ax.set_xlim([1.3, 7.1])
# ax.set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax.set_xlabel("Q ($\AA^{-1}$)", fontsize = fs1)
ax.set_ylabel("Intensity", fontsize = fs1)
ax.set_prop_cycle('color', ['blue', 'black', 'red', 'green'])
ax.axhline(y = threshold, linestyle = '--', linewidth = 0.5, color = 'lightgrey')
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
for key in ref_keys:
    dset = files[key]
    q = qconv(dset[:,0], 1.54)
    y = normalize(dset[:,1])
    ax.plot(q,y,alpha = 0.75,label='PuO$_2$ (sim.)')
ax.plot(qconv(x0,0.74), clean_to_use, linewidth = 3, label = 'n-PuO$_2$')
ax.plot(qconv(x1,0.74), windowed_data, linewidth = 2, label = 'Scherrer analysis region')
ax.scatter(qconv(x1[peaks],0.74), windowed_data[peaks], s = 50, color = 'black', label = 'Peaks')
ax.legend(loc = 'upper right', fontsize = fs1)

    
plt.tight_layout()




#%% if needed calculate confidence interwals on the FWHM values
# please note that it will take a few minutes because it's numerical

params = lm.Parameters()
N = len(peaks)
for i in range(1, N+1):
    params.add(f'a_{i}', value = 1, min = 0)
    params.add(f'w_{i}', value=0.05, min = 0)
    params.add(f'meow_{i}', value=0.5, min = 0, max = 1)
    
mini = lm.Minimizer(residual_fit_peaks, 
                    params, fcn_args=(x1, peaks, ), fcn_kws={'data': windowed_data})
out = mini.leastsq()

fit = residual_fit_peaks(out.params, x1, peaks, windowed_data)
report_fit(out)

for p in out.params:
    out.params[p].stderr = abs(out.params[p].value * 0.1)
    
pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in out.params.values()], 
                        columns=('name', 'best-fit value', 'standard error'))

fwhms = []
fwhm_errors = []
for i in range(1, len(pd_pars.iloc[:,0]), 3):
    if str(pd_pars.iloc[i,0]):
        fwhms.append(pd_pars.iloc[i,1]*2*(np.pi/180)-stdFWHM)
        fwhm_errors.append(pd_pars.iloc[i,2]*2*(np.pi/180))

# ci_man, tr = conf_interval(mini, out, trace=True, sigmas=[1])

# report_ci(ci_man)
#%% SCHERRER PLOT

test = residual_fit_peaks(out.params, x1, peaks, windowed_data) + windowed_data

fig, ax = plt.subplots(1,2,figsize = (16,6))
ax[0].xaxis.set_major_locator(MultipleLocator(5))
ax[0].xaxis.set_minor_locator(MultipleLocator(1))
ax[0].tick_params('both', labelsize = fs2)
ax[0].set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax[0].set_ylabel("Intensity", fontsize = fs1)
ax[0].set_prop_cycle('color', ['black', 'red', 'blue', 'green'])
ax[0].plot(x1, windowed_data)
ax[0].plot(x1, test)
ax[0].legend(labels = ['data', 'fit'], fontsize = fs1, loc = 'upper right')
ax[0].scatter(x1[peaks], windowed_data[peaks], s = 100)

x2 = np.log(1/(np.cos((x1[peaks]/2)*(np.pi/180))))
fwhm_errors_log = np.divide(fwhm_errors, fwhms)
fwhms_log = np.log(fwhms)


# fwhms = [np.log(ci[f'w_{i+1}'][1][1]*2*(np.pi/180)) for i in range(len(peaks))]
# fwhm_errors = np.zeros([2,len(peaks)])
# fwhm_errors[0,:] = [np.log(ci[f'w_{i+1}'][0][1]*2*(np.pi/180)) for i in range(len(peaks))]
# fwhm_errors[1,:] = [np.log(ci[f'w_{i+1}'][2][1]*2*(np.pi/180)) for i in range(len(peaks))]
# fwhm_errors = abs(fwhm_errors - fwhms)


z2 = np.polyfit(x2, fwhms_log, 1, cov = True)

ax[1].scatter(x2, fwhms_log, s = 100, color = 'black')
ax[1].errorbar(x2, fwhms_log, yerr = fwhm_errors_log, ls = 'none', color = 'black', linewidth = 1)
ax[1].tick_params('both', labelsize = fs2)
ax[1].set_xlabel(r'ln($\frac{1}{cos\theta}$)', fontsize = fs1)
ax[1].set_ylabel(r"ln$\beta$", fontsize = fs1)
ax[1].plot(x2, z2[0][0]*x2 + z2[0][1], linestyle = '--', linewidth = 3, color = 'red', label = 'linear')
ax[1].legend(fontsize = fs1, loc = 'lower right')
intercept = z2[0][1]
K = 0.89
lam = 0.075
L = (K*lam)/np.exp(intercept) 
Lerr = ((K*lam)/np.exp(intercept))*z2[1][1,1]
# ax[1].annotate(f'size = {L:.2f} $\pm$ {Lerr:.2f} nm', [0.02, -3.25], color = 'red', fontsize = fs2)


plt.tight_layout()


#%% WILLIAMSON-HALL
test = residual_fit_peaks(out.params, x1, peaks, windowed_data) + windowed_data

fig, ax = plt.subplots(1,2,figsize = (16,6))
ax[0].xaxis.set_major_locator(MultipleLocator(5))
ax[0].xaxis.set_minor_locator(MultipleLocator(1))
ax[0].tick_params('both', labelsize = fs2)
ax[0].set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax[0].set_ylabel("Intensity", fontsize = fs1)
ax[0].set_prop_cycle('color', ['black', 'red', 'blue', 'green'])
ax[0].plot(x1, windowed_data)
ax[0].plot(x1, test)
ax[0].legend(labels = ['data', 'fit'], fontsize = fs1, loc = 'upper right')
ax[0].scatter(x1[peaks], windowed_data[peaks], s = 100)

x2 = 4 * (np.sin((x1[peaks]/2)*(np.pi/180)))
y2 = fwhms * (np.cos((x1[peaks]/2)*(np.pi/180)))
y2_error = fwhm_errors * (np.cos((x1[peaks]/2)*(np.pi/180)))
# fwhms = [np.log(ci[f'w_{i+1}'][1][1]*2*(np.pi/180)) for i in range(len(peaks))]
# fwhm_errors = np.zeros([2,len(peaks)])
# fwhm_errors[0,:] = [np.log(ci[f'w_{i+1}'][0][1]*2*(np.pi/180)) for i in range(len(peaks))]
# fwhm_errors[1,:] = [np.log(ci[f'w_{i+1}'][2][1]*2*(np.pi/180)) for i in range(len(peaks))]
# fwhm_errors = abs(fwhm_errors - fwhms)


z2 = np.polyfit(x2, y2, 1, cov = True)

ax[1].scatter(x2, y2, s = 100, color = 'black')
ax[1].errorbar(x2, y2, yerr = y2_error, ls = 'none', color = 'black', linewidth = 1)
ax[1].tick_params('both', labelsize = fs2)
ax[1].set_xlabel(r'4$\epsilon sin\theta_{hkl}$', fontsize = fs1)
ax[1].set_ylabel(r"$\beta cos\theta_{hkl}$", fontsize = fs1)
ax[1].plot(x2, z2[0][0]*x2 + z2[0][1], linestyle = '--', linewidth = 3, color = 'red', label = 'linear')
ax[1].legend(fontsize = fs1, loc = 'lower right')
slope = z2[0][0]
intercept = z2[0][1]
K = 0.89
lam = 0.075
Dv = (K*lam)/intercept
Dverr = ((K*lam)/(intercept**2)) * z2[1][1,1] 
sloperr = z2[1][0,0]
ax[1].annotate(f'size = {Dv:.2f} $\pm$ {Dverr:.2f} nm \nstrain = {slope:.5f} $\pm$ {sloperr:.5f}', 
               [0.6, 0.03], color = 'red', fontsize = fs2)


plt.tight_layout()

#%% plot the reference pattern and planes
# use something like mercury to figure out what they are
# the actual analysis will rely on plane assignments

refkey = 'puo2sim_1p54_full.xye'
wl_sample = 0.75
wl_ref = 1.54

fs1 = 28
fs2 = 24

fig, ax = plt.subplots(figsize = (16,6))
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel("Q ($\AA^{-1}$)", fontsize = fs1)
ax.set_ylabel("Intensity", fontsize = fs1)

q_ref = qconv(files[refkey][:,0], wl_ref)
s_ref = normalize(files[refkey][:,1])
ax.plot(q_ref, s_ref, color = 'blue', linewidth = 2, label = 'Ref. Pattern')

ax.axhline(y = threshold, linestyle = '--', linewidth = 0.5, color = 'lightgrey')
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
ax.plot(qconv(x0, wl_sample), clean_to_use, color = 'black', linewidth = 3, label = 'Clean data')
ax.plot(qconv(x1, wl_sample), windowed_data, color = 'crimson', linewidth = 3.5, label = 'Scherrer analysis region')
ax.legend(loc = 'upper right', fontsize = fs1)
ax.scatter(qconv(x1[peaks], wl_sample), windowed_data[peaks], s = 50, color = 'black')
    
# so far have to do this manually 
positions = [(100, 0.8), (130, 0.45), (188, 0.6), (225, 0.6), (280, 0.17), (310, 0.3), (355, 0.25), (385, 0.25), (465, 0.35)]
planes = [111, 200, 220, 311, 400, 331, 422, 511, 531]
cell_constant = 5.396 # this does not support non cubic structures yet

for plane, position in zip(planes, positions):
    ax.annotate(str(plane), position, color = 'blue', fontsize = fs2, rotation = 90)

plt.tight_layout()


#%%
ref = 'Na_phase.xye'
refdata = files[ref]
fs1 = 28
fs2 = 24
region = [3,50]
datlambda = 0.75

fig, ax = plt.subplots(3,1,figsize = (16,6), sharex=True)

# ax.set_xlabel("Q ($\AA^{-1}$)", fontsize = fs1)
# ax.set_ylabel("Intensity", fontsize = fs1)
# ax.set_prop_cycle('color', ['red', 'blue', 'green'])

x1 = (np.abs(np10_x - region[0])).argmin()
x2 = (np.abs(np10_x - region[1])).argmin()

ax[0].plot(qconv(np10ca_x[x1:x2], datlambda), normalize(np10ca_y[x1:x2]), linewidth = 3, color = 'crimson', label = 'Ca$_{0.5}$NpO$_2$CO$_3$')
ax[1].plot(qconv(np10_x[x1:x2], datlambda), normalize(np10_y[x1:x2]), linewidth = 3, color = 'royalblue', label = 'KNpO$_2$CO$_3$')
# ax[2].plot(qconv(thphos_1_x[x1:x2], datlambda), normalize(thphos_1_y[x1:x2]), linewidth = 3, color = 'darkmagenta', label = 'ThPhos, pH = 4.8')
ax[2].plot(qconv(refdata[:,0], 1.54), normalize(refdata[:,1]), color = 'black', label = 'теор. NaNpO$_2$CO$_3$ $\cdot$ 3H$_2$O')
for axi in ax:
    axi.tick_params('both', labelsize = fs2)
    axi.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
    axi.legend(loc = 'upper right', fontsize = fs1)  
    axi.set_yticklabels([])
# ax.axhline(y = 1, linestyle = '--', color = 'gray', linewidth = 0.5)
# ax.set_xlim([0, 250])
# ax.plot(data2['2 Theta'], data2['NI'], color = 'red', linewidth = 3, label = 'NU-901')
# ax.plot(traces_n1['wl'], traces_n1['BK80'], color = 'blue', linewidth = 3, label = '80 (phase-pure)')
# ax.plot(traces_n1['wl'], traces_n1['BK83'], color = 'green', linewidth = 3, label = '83 (Ti-SIM-1c)')
# ax.plot(traces_n1['wl'], traces_n1['BK84'], color = 'cyan', linewidth = 3, label = '84 (Zn-SIM-1c)')
ax[-1].set_xlabel("Q ($\AA^{-1}$)", fontsize = fs1) 
ax[-1].set_xlim([0.4, 6.1])
# ax[1].set_ylabel("Intensity (a.u.)", fontsize = fs1)

plt.subplots_adjust(hspace=0)

# plt.tight_layout()
#%%
ref = 'nath2phos_1p54_full.xye'
refdata = files[ref]
fs1 = 28
fs2 = 24
region = [3,50]
datlambda = 0.74

fig, ax = plt.subplots(4,1,figsize = (16,8), sharex=True)

# ax.set_xlabel("Q ($\AA^{-1}$)", fontsize = fs1)
# ax.set_ylabel("Intensity", fontsize = fs1)
# ax.set_prop_cycle('color', ['red', 'blue', 'green'])

x1 = (np.abs(puphos_x - region[0])).argmin()
x2 = (np.abs(puphos_x - region[1])).argmin()

ax[0].plot(qconv(puphos_x[x1:x2], datlambda), normalize(puphos_y[x1:x2]), linewidth = 3, color = 'crimson', label = 'PuPhos')
ax[1].plot(qconv(thphos4_x[x1:x2], datlambda), normalize(thphos4_y[x1:x2]), linewidth = 3, color = 'royalblue', label = 'ThPhos, pH = 7.5')
ax[2].plot(qconv(thphos1_x[x1:x2], datlambda), normalize(thphos1_y[x1:x2]), linewidth = 3, color = 'darkmagenta', label = 'ThPhos, pH = 4.8')
ax[3].plot(qconv(refdata[:,0], 1.54), normalize(refdata[:,1]), color = 'black', label = 'теор. NaTh$_2$(PO$_4$)$_3$')
for axi in ax:
    axi.tick_params('both', labelsize = fs2)
    axi.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
    axi.legend(loc = 'upper right', fontsize = fs1)  
    axi.set_yticklabels([])
# ax.axhline(y = 1, linestyle = '--', color = 'gray', linewidth = 0.5)
ax[-1].set_xlabel("Q ($\AA^{-1}$)", fontsize = fs1) 
ax[-1].set_xlim([0.75, 7.1])
# ax[1].set_ylabel("Intensity (a.u.)", fontsize = fs1)

plt.subplots_adjust(hspace=0)

# plt.tight_layout()
#%% plot a pattern based on d/n values
# nlambda= 2dsintheta
dn_file = 'Th4PO4P2O7_dobs_Int.txt'
dns = np.loadtxt(dn_file, dtype = 'float', skiprows=0)

#%%
def xconv(data, wavelength):
    """
    Converts d/n values into 2theta
    
    Arguments
    -------
    data: a vector of d/n values
    wavelength: the wavelength of the x-ray source
    the data and wavelength must both be in angstrom or both in nm

    Returns
    -------
    out: a vector of 2theta values in degrees

    """
    # d/n = lambda/(2sintheta)
    # sintheta = (n*lambda)/(2d)
    # 2theta = 2*arcsin((d/n)^(-1) * lambda/2)
    out = 2 * (np.arcsin((1/data) * wavelength/2)) * (180/np.pi)
    return out
    


#%%
fig, ax = plt.subplots(6,1,figsize = (10, 18), sharex=True)
fs1 = 28
fs2 = 24
# ax.set_prop_cycle('color', ['red', 'blue', 'green'])
# ax.set_xlabel(r'2$\theta$ (deg.)', fontsize = fs1)
# ax.set_ylabel('norm. int.', fontsize = fs1)
# ax.set_yticks([])
# ax.tick_params('x', labelsize = fs2)
# ax.set_xlim([2.5, 35])

legendary = [['CsTh$_2$(PO$_4$)$_3$'], 
             ['BaTh(PO$_4$)$_2$'], 
             ['CuTh$_2$(PO$_4$)$_3$'],
             ['Na$_2$Th(PO$_4$)$_2$'],
             ['NaTh$_2$(PO$_4$)$_3$'],
             ['ThPHOS (4.8)']]

for label, axid in zip(('ThPhos1_clean_2theta_Int_0p74.txt', 
              'nath2phos_0p74_full.xye', 
              'na2thphos_0p74_full.xye',
              'cuth2phos_0p74_full.xye')[::-1], range(2,6)):
    dset = files[label]
    ax[axid].plot(dset[:,0], normalize(dset[:,1]), color = 'black', linewidth = 1.5)
    ax[axid].set_yticks([])
    if axid == 5:
        ax[axid].set_xlabel(r'2$\theta$ (deg.)', fontsize = fs1)
        ax[axid].set_xlim([2.5, 35])
        ax[axid].tick_params(labelsize = fs2)
        ax[axid].plot(dset[:,0], normalize(dset[:,1]), color = 'blue', linewidth = 1.5)
        ax[axid].axvline(x = 6.9, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 9.1, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 9.9, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 12.8, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 13.7, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 14.6, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 15.15, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 17.8, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 19, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 19.8, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 22.2, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 22.8, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 24.2, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 25, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 25.7, color = 'red', linestyle = '--')
    ax[axid].legend(labels = legendary[axid], loc = 'upper right', fontsize = fs2)
    ax[axid].axvline(x = 6.9, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 9.1, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 9.9, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 12.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 13.7, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 14.6, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 15.15, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 17.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 19, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 19.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 22.2, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 22.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 24.2, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 25, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 25.7, color = 'red', linestyle = '--')

for label, axid in zip(('CsThPhos_Brandel2002_2theta_Int_1p54.txt',
              'BaThPhos_Brandel2002_2theta_Int_1p54.txt'), range(2)):
    dset = files[label]
    ax[axid].bar(dset[:,0]*0.74/1.54, dset[:,1], width = 0.2, color = 'black')
    ax[axid].set_yticks([])
    ax[axid].legend(labels = legendary[axid], loc = 'upper right', fontsize = fs2)
    ax[axid].axvline(x = 6.9, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 9.1, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 9.9, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 12.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 13.7, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 14.6, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 15.15, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 17.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 19, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 19.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 22.2, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 22.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 24.2, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 25, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 25.7, color = 'red', linestyle = '--')
              
plt.subplots_adjust(hspace=0)
plt.tight_layout()
#%%
fig, ax = plt.subplots(6,1,figsize = (10, 18), sharex=True)
fs1 = 28
fs2 = 24
# ax.set_prop_cycle('color', ['red', 'blue', 'green'])
# ax.set_xlabel(r'2$\theta$ (deg.)', fontsize = fs1)
# ax.set_ylabel('norm. int.', fontsize = fs1)
# ax.set_yticks([])
# ax.tick_params('x', labelsize = fs2)
# ax.set_xlim([2.5, 35])

legendary = [['Th$_4$(PO$_4$)$_4$SiO$_4$'], 
             ['ThFPO$_4$ $\cdot$ H$_2$O'], 
             ['Th(OH)PO$_4$'],
             ['Th$_2$O(PO$_4$)$_2$'],
             ['Th$_2$(PO$_4$)$_2$HPO$_4$ $\cdot$ H$_2$O'],
             ['Th$_4$(PO$_4$)$_4$P$_2$O$_7$']]


for label, axid in zip(('ThSio4Phos_Brandel2002_2theta_Int_1p54.txt',
                        'ThFPhos_Brandel2002_2theta_Int_1p54.txt',
                        'ThOHPO4_Brandel2001_2theta_Int_1p54.txt',
                        'DTOP_Brandel2001_2theta_Int_1p54.txt',
              'BaThPhos_Brandel2002_2theta_Int_1p54.txt',
              'Th4(PO4)4P2O7_0p74_bars.txt'), range(6)):
    dset = files[label]
    if axid == 5:
        ax[axid].bar(dset[:,0], dset[:,1], width = 0.2, color = 'black')
        ax[axid].set_xlabel(r'2$\theta$ (deg.)', fontsize = fs1)
        ax[axid].set_xlim([2.5, 35])
        ax[axid].tick_params(labelsize = fs2)
    else:
        ax[axid].bar(dset[:,0]*0.74/1.54, dset[:,1], width = 0.2, color = 'black')
    ax[axid].set_yticks([])
    ax[axid].legend(labels = legendary[axid], loc = 'upper right', fontsize = fs2)
    ax[axid].axvline(x = 6.9, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 9.1, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 9.9, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 12.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 13.7, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 14.6, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 15.15, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 17.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 19, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 19.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 22.2, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 22.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 24.2, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 25, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 25.7, color = 'red', linestyle = '--')
plt.subplots_adjust(hspace=0)
plt.tight_layout()
#%%
fig, ax = plt.subplots(9,1,figsize = (10, 18), sharex=True)
fs1 = 28
fs2 = 24

legendary = [['Na$_3$PO$_4$'], 
             ['ThO$_2$'], 
             [r'$\beta$-TPD'],
             [r'$\alpha$-TPD'],
             ['NaZr$_2$(PO$_4$)$_3$ (rhomb. NZP)'],
             ['Na$_2$Th(PO$_4$)$_2$'],
             ['NaTh$_2$(PO$_4$)$_3$'],
             ['Sample TP-2'],
             ['Sample TP-1']]

for label, axid in zip(('na3po4_0p74_full.xye', 
              'tho2_0p74_full.xye', 
              'betaTPD_0p74.xye',
              'alphaTPD_0p74.xye',
              'nazr2phos_0p74_full.xye',
              'na2thphos_0p74_full.xye',
              'nath2phos_0p74_full.xye',
              'ThPhos4_clean_2theta_Int_0p74.txt',
              'ThPhos1_clean_2theta_Int_0p74.txt'), range(9)):
    dset = files[label]
    ax[axid].plot(dset[:,0], normalize(dset[:,1]), color = 'black', linewidth = 1.5)
    ax[axid].set_yticks([])
    if axid == 8:
        ax[axid].set_xlabel(r'2$\theta$ (deg.)', fontsize = fs1)
        ax[axid].set_xlim([2.5, 35])
        ax[axid].tick_params(labelsize = fs2)
        # ax[axid].plot(dset[:,0], normalize(dset[:,1]), color = 'black', linewidth = 1.5)
        ax[axid].axvline(x = 6.9, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 9.1, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 9.9, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 12.8, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 13.7, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 14.6, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 15.15, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 17.8, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 19, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 19.8, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 22.2, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 22.8, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 24.2, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 25, color = 'red', linestyle = '--')
        ax[axid].axvline(x = 25.7, color = 'red', linestyle = '--')
    ax[axid].legend(labels = legendary[axid], loc = 'upper right', fontsize = fs2)
    ax[axid].axvline(x = 6.9, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 9.1, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 9.9, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 12.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 13.7, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 14.6, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 15.15, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 17.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 19, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 19.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 22.2, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 22.8, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 24.2, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 25, color = 'red', linestyle = '--')
    ax[axid].axvline(x = 25.7, color = 'red', linestyle = '--')
              
plt.subplots_adjust(hspace=0)
plt.tight_layout()

#%%
dset = files['ThPhos1_clean_2theta_Int_0p74.txt']
fs1 = 28
fs2 = 24
fig, ax = plt.subplots(figsize = (16,6))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax.set_ylabel("Intensity", fontsize = fs1)
ax.set_prop_cycle('color', ['black', 'red', 'blue', 'green'])
ax.plot(dset[:,0]*(1.54/0.74), dset[:,1])
# ax.plot(dset_xdata, clndset)
ax.legend(labels = ['ThPhos pH = 4.8 (wl = 0.154 nm)'], loc = 'upper right', fontsize = fs1)  
ax.axhline(y = 0, linestyle = '--', color = 'gray', linewidth = 0.5)
ax.set_xlim([0,65])

plt.tight_layout()
#%% OverlaidComparisons
files = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
        # skip directories
        continue
    with open(file):
        files[str(file)] = np.loadtxt(file)
fs1 = 20
fs2 = 14
fig, ax = plt.subplots(figsize = (12,4))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax.set_ylabel("Intensity", fontsize = fs1)
ax.set_xlim([4,50])
ax.set_prop_cycle('color', ['blue', 'orange','green','red'])
for key in files.keys():
    ax.plot(files[key][:,0], files[key][:,1], alpha = 1, linewidth = 1)
# ax.legend(labels = ['ThPhos-5-Na-1000', 'ThPhos-6-1000', 'ThPhos-5-1000', 'ThPhos-8-1000'], fontsize = fs1, frameon=False)
ax.set_prop_cycle('color', ['blue', 'orange','green','red'])
for key in files.keys():
    ax.plot(files[key][:,0], files[key][:,1], alpha = 0.45, linewidth = 3)
    print(key)
plt.tight_layout()
plt.show()

#%% OverlaidComparisons
files = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
        # skip directories
        continue
    with open(file):
        files[str(file)] = np.loadtxt(file)
fs1 = 20
fs2 = 14
fig, ax = plt.subplots(figsize = (12,4))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax.set_ylabel("Intensity", fontsize = fs1)
ax.set_xlim([4,50])
ax.set_prop_cycle('color', ['blue', 'orange','green','red','cyan'])
for key in list(files.keys()):
    if key == 'ThPhos1_clean_2theta_Int_0p74.txt':
        ax.plot(files[key][:,0]*0.75/0.74, files[key][:,1], alpha = 1, linewidth = 1)
    else:
        ax.plot(files[key][:,0], files[key][:,1], alpha = 1, linewidth = 1)
ax.legend(labels = ['ThPhos-4.8 (new)', 'ThPhos-4.8-Na', 'ThPhos-5.4', 'ThPhos-6.4', 'ThPhos-4.8 (old)'], fontsize = fs1, frameon=False)
ax.set_prop_cycle('color', ['blue', 'orange','green','red','cyan'])
for key in list(files.keys()):
    if key == 'ThPhos1_clean_2theta_Int_0p74.txt':
        ax.plot(files[key][:,0]*0.75/0.74, files[key][:,1], alpha = 0.45, linewidth = 3)
    else:
        ax.plot(files[key][:,0], files[key][:,1], alpha = 0.45, linewidth = 3)
    print(key)

plt.tight_layout()
plt.show()
#%% OverlaidComparisons
files = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
        # skip directories
        continue
    with open(file):
        files[str(file)] = np.loadtxt(file)
fs1 = 20
fs2 = 14
fig, ax = plt.subplots(figsize = (12,4))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel(r"2$\theta$ (deg.)", fontsize = fs1)
ax.set_ylabel("Intensity", fontsize = fs1)
ax.set_xlim([4,50])
ax.set_ylim([-100, 850])
ax.set_prop_cycle('color', ['blue', 'orange','green','red','cyan'])
for key in list(files.keys()):
    if key == 'ThPhos4_clean_2theta_Int_0p74.txt':
        ax.plot(files[key][:,0]*0.75/0.74, files[key][:,1], alpha = 1, linewidth = 1)
    else:
        ax.plot(files[key][:,0], files[key][:,1], alpha = 1, linewidth = 1)
ax.legend(labels = ['ThPhos-7.5 (new)', 'ThPhos-7.5 (old)'], fontsize = fs1, frameon=False)
ax.set_prop_cycle('color', ['blue', 'orange','green','red','cyan'])
for key in list(files.keys()):
    if key == 'ThPhos4_clean_2theta_Int_0p74.txt':
        ax.plot(files[key][:,0]*0.75/0.74, files[key][:,1], alpha = 0.45, linewidth = 3)
    else:
        ax.plot(files[key][:,0], files[key][:,1], alpha = 0.45, linewidth = 3)
    print(key)

plt.tight_layout()
plt.show()