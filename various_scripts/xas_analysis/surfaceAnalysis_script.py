# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:43:51 2024

@author: boris
"""

#%% LIBRARIES
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
import matplotlib.gridspec as gs
from scipy.signal import savgol_filter as sgf
from scipy import special
from scipy.stats import linregress
from collections import OrderedDict
import os

#%%
norDict = OrderedDict()
for file in os.listdir():
    with open(file):
        if str(file).endswith('.nor'):
            norDict['headers'] = open(file).readlines()[37]
            norDict[file] = np.loadtxt(file, dtype = float, skiprows=38)
            
#%%
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

def xcut(energy, data, region_deltas):
    enot = energy[np.argmax(data)]
    region = [enot - np.abs(region_deltas[0]), enot + np.abs(region_deltas[1])]
    idx_x_min = (np.abs(energy - region[0])).argmin()
    idx_x_max = (np.abs(energy - region[1])).argmin()
    return np.vstack([energy[idx_x_min:idx_x_max],data[idx_x_min:idx_x_max]]).T

xtest = norDict['Th01_pH4_NH3.nor'][:,0]
ytest = norDict['Th01_pH4_NH3.nor'][:,3]

ctest = xcut(xtest, ytest, [-10, 100])

#%%
ref = 'Th01_pH4_NH3.nor'
sample = 'Th01_pH6_NH3.nor'
margins_cut = [-45, 150]
fs1 = 24
fs2 = 20

refcut = xcut(norDict[ref][:,0], norDict[ref][:,1], margins_cut)
datcut = xcut(norDict[sample][:,0], norDict[sample][:,1], margins_cut)


from scipy import interpolate
N = 65536
en_int = np.linspace(np.min(refcut[:,0])+10, np.max(refcut[:,0])-15, N)
interpolant = interpolate.interp1d(refcut[:,0], refcut[:,1], kind="linear")
g_ref = interpolant(en_int)
interpolant = interpolate.interp1d(datcut[:,0], datcut[:,1], kind="linear")
g_dat = interpolant(en_int)

# from scipy.signal import resample
# g_ref = resample(refcut[:,1], len(refcut[:,1]))
# g_dat = resample(datcut[:,1], len(refcut[:,1]))
# ref_amp = np.max(g_ref)
# g_dat = normalize(g_dat, 'maximum', ref_amp)



diffcut = (g_dat-g_ref)/np.max(g_ref) * 100

fig, ax1 = plt.subplots(figsize = (12,8))
ax1.set_xlabel('Energy (eV)', fontsize = fs1)
ax1.set_ylabel("$\Delta$$\mu$(E) (%)", fontsize = fs1)
# ax.yaxis.get_offset_text().set_fontsize(27)
ax1.tick_params(axis='both', labelsize= fs2)
# ax.set_xlim([3.35, 1.90])
# ax.xaxis.set_major_locator(MultipleLocator(0.25))
# ax.xaxis.set_minor_locator(MultipleLocator(0.05))


ax2 = ax1.twinx()
ax2.spines['right'].set_color('firebrick')
ax2.tick_params(axis='both', labelsize = fs2, labelcolor = 'firebrick')
ax2.set_ylabel('normalized $\mu$(E)', fontsize = fs1, color = 'firebrick')
ax2.plot(refcut[:,0],refcut[:,1],color='firebrick',alpha=0.8,linewidth = 2.5, label = f'ref: {ref[:-4]}')
ax2.plot(datcut[:,0],datcut[:,1],color='deepskyblue',alpha=0.8,linewidth = 2.5, label = f'dat: {sample[:-4]}')
ax2.legend(loc = 'lower right', fontsize = fs1)


# ax1.plot(refcut[:,0], diffcut, linewidth = 3.5, color = 'black', label = 'Difference')
ax1.plot(en_int, diffcut, linewidth = 3.5, color = 'black', label = 'dat - ref')
ax1.legend(loc = 'upper right', fontsize = fs1)
ax1.axhline(y = 0, linestyle = '--', color = 'black')
cmap = plt.get_cmap('RdBu')
# ax1.fill_between(refcut[:,0], 0, diffcut, where=diffcut>0, facecolor=cmap(0.66), alpha = 0.65, interpolate=True)
# ax1.fill_between(refcut[:,0], 0, diffcut, where=diffcut<=0, facecolor=cmap(0.33), alpha = 0.65,interpolate=True)
ax1.fill_between(en_int, 0, diffcut, where=diffcut>0, facecolor=cmap(0.7), alpha = 0.8, interpolate=True)
ax1.fill_between(en_int, 0, diffcut, where=diffcut<=0, facecolor=cmap(0.3), alpha = 0.8,interpolate=True)
integral = np.trapz(y = diffcut, x = en_int)
ax1.annotate(f'integral = {integral:.2f}',
            xy=(.5, .8), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=fs2)

plt.tight_layout()