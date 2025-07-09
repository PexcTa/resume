# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:31:37 2024

@author: boris
"""

#%% LIBRARIES
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.gridspec as gs
from scipy.signal import savgol_filter as sgf
from scipy import special
from scipy.stats import linregress
from collections import OrderedDict
import os

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
#%% import the data
# experimental EXAFS and the transformations
ipolEXAFS = np.loadtxt('ipolEXAFS.dat', dtype = float)
expBFT = np.loadtxt('expBFT.dat', dtype = float)
expFT = np.loadtxt('expFT.dat', dtype = float)
expWT = np.loadtxt('expWT.dat', dtype = float)

# EXAFS corresponding to final structure and the transformations
calcEXAFS = np.loadtxt('calcEXAFS.dat', dtype = float)
calcBFT = np.loadtxt('calcBFT.dat', dtype = float)
calcFT = np.loadtxt('calcFT.dat', dtype = float)
calcWT = np.loadtxt('calcWT.dat', dtype = float)
#%% Define some fitting constants that you used

kpower = 3
rmin = 1.6
rmax = 5.0


#%% Do stacked plot: k-space,  FT, back FT
fig, ax = plt.subplots(3,1,figsize=(10,15), sharex=False)
fontSize1 = 24
fontSize2 = 20


dset_0 = ipolEXAFS
dset_1 = calcEXAFS
ax[0].plot(dset_0[:, 0], dset_0[:,1]*(dset_0[:, 0]**kpower), color = 'darkgrey', linewidth = 5.5)
ax[0].plot(dset_1[:, 0], dset_1[:,1]*(dset_1[:, 0]**kpower), color = 'crimson', linewidth = 4.5)
ax[0].set_xlim(np.min(dset_0[:, 0]),np.max(dset_0[:, 0]))
ax[0].set_ylim([-7,7])
dset_0 = expFT
dset_1 = calcFT
ax[1].plot(dset_0[:, 0], dset_0[:,-1], color = 'darkgrey', linewidth = 5.5, label = 'experimental')
ax[1].plot(dset_1[:, 0], dset_1[:,-1], color = 'crimson', linewidth = 4.5, label = 'calculated')
dset_0 = expBFT
dset_1 = calcBFT
ax[2].plot(dset_0[:, 0], dset_0[:,1], color = 'darkgrey', linewidth = 5.5)
ax[2].plot(dset_1[:, 0], dset_1[:,1], color = 'crimson', linewidth = 4.5)
ax[2].set_xlim(np.min(dset_0[:, 0]),np.max(dset_0[:, 0]))
# ax[1].axvline(x = 1.3, linestyle = '--', color = 'seagreen')
# ax[1].axvline(x = 5.5, linestyle = '--', color = 'seagreen', label = 'window')
# ax[0].legend(loc = 'upper right', fontsize = fontSize1)
ax[1].legend(loc = 'best', fontsize = fontSize1)
# ax[2].legend(loc = 'upper right', fontsize = fontSize1)

for axis in ax:
    axis.tick_params(axis = 'both', labelsize = fontSize2)

ax[0].set_xlabel("k($\AA^{-1}$)", fontsize = fontSize1)
ax[1].set_xlabel("R($\AA$)", fontsize = fontSize1)
ax[2].set_xlabel("k($\AA^{-1}$)", fontsize = fontSize1)

ax[1].set_xlim([0,rmax+0.49])

ax[1].axvline(x = rmin, linestyle = '--', color = 'seagreen')
ax[1].axvline(x = rmax, linestyle = '--', color = 'seagreen', label = 'window')

# 
ax[0].set_ylabel(f"k$^{kpower}$\u03C7(k)($\AA^{{{-kpower}}}$)", fontsize = fontSize1)
ax[1].set_ylabel(f"|\u03C7(R)|($\AA^{{{-(1+kpower)}}}$)", fontsize = fontSize1)
ax[2].set_ylabel(f"Re[$\chi$(q)]($\AA^{{{-kpower}}}$)", fontsize = fontSize1)

plt.tight_layout()
plt.show()
#%% Do contour map of experimental wavelet transform
fig, bx = plt.subplots(figsize = (7.5,15))
fontSize1 = 36
fontSize2 = 28

nLevels = 10
stepLevels = 4

klen = len(ipolEXAFS[:,0])
rlen = len(expFT[:,0])

k = expWT[:klen,0]
r = expWT[:,1]
r = r[0:-1:rlen]
z = expWT[:,-1].reshape(klen,rlen)

levels = np.linspace(z.min(), z.max(), nLevels)
levelsIndex = np.round(np.linspace(1, len(levels) - 1, int(nLevels/stepLevels))).astype(int)
bnt = bx.contourf(k, r, z, levels=levels,
            extend='both',cmap=plt.cm.inferno)
bx.set_ylabel("R($\AA$)", fontsize = fontSize1)
bx.set_xlabel("k($\AA^{-1}$)", fontsize = fontSize1)
bx.tick_params(axis='both', labelsize= fontSize1)
bx.set_ylim([r.min(), rmax+0.49])
bx.set_xlim([k.min(), k.max()-0.01])
bx.contour(k, r, z, levels=levels[levelsIndex], colors='black', linewidths=2.5)

bx.xaxis.set_major_locator(MultipleLocator(2))
bx.xaxis.set_minor_locator(MultipleLocator(1))
bx.yaxis.set_major_locator(MultipleLocator(0.5))
bx.yaxis.set_minor_locator(MultipleLocator(0.25))

bx.set_title('experimental', fontsize = fontSize1)
plt.tight_layout()
plt.show()
#%% Do contour map of the calculated wavelet transform

fig, bx = plt.subplots(figsize = (7.5,15))
fontSize1 = 36
fontSize2 = 28

nLevels = 10
stepLevels = 4

klen = len(calcBFT[:,0])
rlen = len(calcFT[:,0])

k = calcWT[:klen,0]
r = calcWT[:,1]
r = r[0:-1:rlen]
z = calcWT[:,-1].reshape(klen,rlen)

levels = np.linspace(z.min(), z.max(), nLevels)
levelsIndex = np.round(np.linspace(1, len(levels) - 1, int(nLevels/stepLevels))).astype(int)
bnt = bx.contourf(k, r, z, levels=levels,
            extend='both',cmap=plt.cm.inferno)
bx.set_ylabel("R($\AA$)", fontsize = fontSize1)
bx.set_xlabel("k($\AA^{-1}$)", fontsize = fontSize1)
bx.tick_params(axis='both', labelsize= fontSize1)
bx.set_ylim([r.min(), rmax+0.49])
bx.set_xlim([k.min(), k.max()-0.01])
bx.contour(k, r, z, levels=levels[levelsIndex], colors='black', linewidths=2.5)
bx.xaxis.set_major_locator(MultipleLocator(2))
bx.xaxis.set_minor_locator(MultipleLocator(1))
bx.yaxis.set_major_locator(MultipleLocator(0.5))
bx.yaxis.set_minor_locator(MultipleLocator(0.25))
bx.set_title('calculated', fontsize = fontSize1)
plt.tight_layout()
plt.show()
#%% Do contour map of the difference between experimental and calculated wavelet transforms!
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.colormaps.register(cmap=newcmap)

    return newcmap

from mpl_toolkits.axes_grid1 import make_axes_locatable
k = expWT[:klen,0]
r = expWT[:,1]
r = r[0:-1:rlen]
z1 = expWT[:,-1].reshape(klen,rlen)
k = calcWT[:klen,0]
r = calcWT[:,1]
r = r[0:-1:rlen]
z2 = calcWT[:,-1].reshape(klen,rlen)

zDiff = (np.subtract(z1, z2))/((np.max(z1)+np.max(z2))/2) * 100

cmap = plt.cm.seismic

# shiftmap = shiftedColorMap(cmap, 0, (1-np.max(zDiff)/(np.max(zDiff)+np.abs(np.min(zDiff)))), 1, 'shiftmap')
# 
fig, bx = plt.subplots(figsize = (8,15))
fontSize1 = 36
fontSize2 = 28

nLevels = 10
stepLevels = 1

levels = np.linspace(zDiff.min(), zDiff.max(), nLevels)
# levelsIndex = np.round(np.linspace(1, len(levels) - 1, int(nLevels/stepLevels))).astype(int)
bnt = bx.contourf(k, r, zDiff, levels=levels,extend='both',cmap=shiftmap)
bx.set_ylabel("R($\AA$)", fontsize = fontSize1)
bx.set_xlabel("k($\AA^{-1}$)", fontsize = fontSize1)
bx.tick_params(axis='both', labelsize= fontSize1)
bx.set_ylim([r.min(), rmax+0.49])
bx.set_xlim([k.min(), k.max()-0.01])
divider = make_axes_locatable(bx)
bax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(bnt, cax=bax, orientation='vertical')
cbar.set_ticks(np.linspace(np.min(zDiff), np.max(zDiff), nLevels, dtype = int))
bax.tick_params(labelsize = fontSize1)
# bax.set_title('% diff', fontsize = fontSize1)
bx.xaxis.set_major_locator(MultipleLocator(2))
bx.xaxis.set_minor_locator(MultipleLocator(1))
bx.yaxis.set_major_locator(MultipleLocator(0.5))
bx.yaxis.set_minor_locator(MultipleLocator(0.25))

# bx.axhline(y = rmin, linestyle = '--', color = 'seagreen')
# bx.axhline(y = rmax, linestyle = '--', color = 'seagreen')

bx.set_title('difference (%)', fontsize = fontSize1)
plt.tight_layout()
plt.show()
#%%
rdf_headers = np.loadtxt('rdf.dat', dtype = 'str')[0]
rdf = np.loadtxt('rdf.dat', skiprows=1)

fig, ax = plt.subplots(figsize = (15,5))

fontSize1 = 42
fontSize2 = 36
ax.set_ylabel("Frequency", fontsize = fontSize1)
ax.set_xlabel("R($\AA$)", fontsize = fontSize1)
ax.set_yticks([])
ax.tick_params(axis='x', labelsize= fontSize1)

# colors = [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1, len(rdf_headers))]
colors = ['darkgrey', 'crimson', 'crimson', 'darkgrey']
# colors = ['darkgrey', 'goldenrod','crimson', 'deepskyblue' ]

# colors = ['crimson', 'deepskyblue' , 'darkgrey', 'goldenrod']
# colors = ['crimson', 'darkviolet' , 'darkgrey', 'goldenrod']
for i in range(1, len(rdf_headers)):
    ax.bar(rdf[:,0], rdf[:,i], width=np.mean(np.diff(rdf[:,0])), align='center', color = colors[i-1], alpha = 0.75, label = rdf_headers[i])

ax.axvline(x = rmin, linestyle = '--', color = 'seagreen')
ax.axvline(x = rmax, linestyle = '--', color = 'seagreen')

ax.legend(loc = 'best', fontsize = fontSize2, frameon = False, ncol = 2)
ax.set_xlim([1, rmax + 0.49])

plt.tight_layout()
plt.show()

#%% fit a distribution 

# pd_pars_last = pd_pars.copy()

# need to rebin, plot as peaks, find peaks, fit a gaussian to each one
# get the standard deviation and be happy
from scipy.signal import find_peaks
import lmfit as lm
def rebin(a, shape):
    # @jfs on StackOverflow
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
def residual_fit_peaks(Pars, x, peaks, data):
    N = len(peaks)
    res = {}
    for i in range(1, N+1):
        res[f'P_{i}'] = np.zeros_like(x)
        a = Pars[f'a_{i}']
        w = Pars[f'w_{i}']
        e = x[int(peaks[i-1])] + Pars[f'de_{i}']
        res[f'P_{i}'] = a*(np.exp(-(x-e)**2/(2*w**2)))
    total = np.sum([res[f'P_{i}'] for i in range(1, N+1)], axis = 0)
    return total - data

# CHOOSE THE NUMBER OF THE COLUMN FROM RDF_HEADERS
colno = 4
region = (1.3, 5.54)

# window the data and the 2theta based on the region 
idx_x_min = (np.abs(rdf[:,0] - region[0])).argmin()
idx_x_max = (np.abs(rdf[:,0] - region[1])).argmin()
rdf_reduced = rdf[idx_x_min:idx_x_max, :]


rdf_element = np.vstack([rdf_reduced[:,0],rdf_reduced[:,colno]]).T
rdf_rebin = rebin(rdf_element, (int(len(rdf_reduced[:,0])/3),2))

fig, ax = plt.subplots(figsize = (12,4))

fontSize1 = 24
fontSize2 = 20
ax.set_ylabel("Frequency", fontsize = fontSize1)
ax.set_xlabel("R($\AA$)", fontsize = fontSize1)
# ax.set_yticks([1])
ax.tick_params(axis='both', labelsize= fontSize2)
for i in range(1, len(rdf_headers)):
    ax.bar(rdf[:,0], rdf[:,i], width=np.mean(np.diff(rdf[:,0])), align='center', color = colors[i-1], alpha = 0.2)
ax.bar(rdf_element[:,0], rdf_element[:,1], width=np.mean(np.diff(rdf[:,0])), align='center', color = colors[colno-1], alpha = 0.75, label = rdf_headers[colno]+" Original RDF")
ax.plot(rdf_rebin[:,0], rdf_rebin[:,1], color = 'black', linewidth = 4, label = 'Rebinned RDF')

threshold = 5
peaks, _ = find_peaks(rdf_rebin[:,1], height=threshold, distance = 5)
print(f'{len(peaks)} peaks found!')

params = lm.Parameters()
N = len(peaks)
for i in range(1, N+1):
    params.add(f'a_{i}', value = 10, min = 0)
    params.add(f'w_{i}', value=0.008, min = 0)
    params.add(f'de_{i}', value=0.01, min = 0)
    
mini = lm.Minimizer(residual_fit_peaks, 
                    params, fcn_args=(rdf_rebin[:,0], peaks, ), fcn_kws={'data': rdf_rebin[:,1]})
out = mini.leastsq()

fit = residual_fit_peaks(out.params, rdf_rebin[:,0], peaks, rdf_rebin[:,1])
lm.report_fit(out)

for p in out.params:
    out.params[p].stderr = abs(out.params[p].value * 0.1)
    
pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in out.params.values()], 
                        columns=('name', 'best-fit value', 'standard error'))

fit = residual_fit_peaks(out.params, rdf_rebin[:,0], peaks, rdf_rebin[:,1]) + rdf_rebin[:,1]
ax.plot(rdf_rebin[:,0], fit, color = 'crimson', linewidth = 4, label = 'Fitted RDF')
ax.set_xlim([1.5, rmax+0.51])
ax.set_ylim([0, np.max(rdf_element)+0.5*np.max(rdf_element)])
yann_coord = np.zeros_like(peaks)
yann_coord = (5, 20, 24, 7, 24, 60, 42, 24,7)
x_shifts = np.zeros_like(peaks)
x_shifts = (-0.6, -0.6, 0.1, +0.1, -0.6, -0.4, -0.25, 0, 0,0)
for i in range(N):
    ax.annotate(f"$\sigma^2$ = {pd_pars.iloc[1+3*i, 1]**2:.4f}\nR = {rdf_rebin[:,0][peaks[i]]+pd_pars.iloc[2+3*i, 1]**2:.3f}",
                xy = (rdf_rebin[:,0][peaks[i]]+x_shifts[i], yann_coord[i]), 
                fontsize = fontSize2-4,
                bbox=dict(facecolor='gray', alpha=0.2))

# ax.legend(loc = 'upper left', fontsize = fontSize2)

plt.tight_layout()

#%% simulated RDF's from initial structure; generated with Ovito
RDF1, RDF2, RDF3 = np.loadtxt('pRDF_start_NpO', skiprows = 2), np.loadtxt('pRDF_start_NpC', skiprows = 2),np.loadtxt('pRDF_start_NpMg', skiprows = 2)
fig, ax = plt.subplots(figsize = (15,5))

fontSize1 = 42
fontSize2 = 36
ax.set_ylabel("Frequency", fontsize = fontSize1)
ax.set_xlabel("R($\AA$)", fontsize = fontSize1)
ax.set_yticks([])
ax.tick_params(axis='x', labelsize= fontSize1)

# colors = [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1, len(rdf_headers))]
colors = ['darkgrey', 'crimson', 'goldenrod', 'darkviolet']

# this section to plot the K2CePO4 RDFs
# ax.bar(RDF3[:,0], RDF3[:,-1], 0.1, align='center', color = colors[0], alpha = 0.75, label = 'Pu-Pu')
# ax.bar(RDF1[:,0], RDF1[:,7], 0.1, align='center', color = colors[1], alpha = 0.75, label = 'Pu-O')
# ax.bar(RDF2[:,0], RDF2[:,9], 0.1, align='center', color = colors[2], alpha = 0.75, label = 'Pu-P')
# ax.bar(RDF3[:,0], RDF3[:,4], 0.1, align='center', color = colors[3], alpha = 0.75, label = 'Pu-K')

# this section to plot the Na2ThPO4 RDFs
ax.bar(RDF3[:,0], RDF3[:,3], 0.05, align='center', color = colors[0], alpha = 0.75, label = 'Np-Np')
ax.bar(RDF1[:,0], RDF1[:,2], 0.05, align='center', color = colors[1], alpha = 0.75, label = 'Np-O')
ax.bar(RDF2[:,0], RDF2[:,2], 0.05, align='center', color = colors[2], alpha = 0.75, label = 'Np-C')
ax.bar(RDF3[:,0], RDF3[:,2], 0.05, align='center', color = colors[3], alpha = 0.75, label = 'Np-Mg')

ax.legend(loc = 'best', fontsize = fontSize2-6, ncol = 2, frameon=False)
ax.set_xlim([1, rmax + 0.49])
ax.axvline(x = rmin, linestyle = '--', color = 'seagreen')
ax.axvline(x = rmax, linestyle = '--', color = 'seagreen')
plt.tight_layout()
plt.show()

#%% evax residuals
residuals = np.loadtxt('output.dat', skiprows = 1)
fig, ax = plt.subplots(figsize = (5,15))
ax.plot(residuals[:,0], residuals[:,1], marker = 'o', markersize = 10, color = 'black')

fontSize1 = 42
fontSize2 = 36
ax.set_ylabel("Residuals", fontsize = fontSize1)
ax.set_xlabel("R($\AA$)", fontsize = fontSize1)

ax.tick_params(axis='x', labelsize= fontSize1)




plt.tight_layout()

plt.show()
