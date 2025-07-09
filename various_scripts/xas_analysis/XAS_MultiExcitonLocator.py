# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:10:45 2024

@author: boris
"""

# Manual Multiexciton Correction
#%% LIBRARIES
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter as sgf
from scipy import special
import os
#%%
def ReadSingleXANESnor(file, skip=11):
    with open(file) as current_file:
        # names = ['E', 'norm', 'nbkg', 'flat', 'fbkg', 'nderiv', 'nderiv_2'], 
        d = pd.read_csv(current_file, delim_whitespace = True, header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
def energy_to_k(E0, Ex):
    planck = 6.626 * 10 ** (-34)
    c = 2.998 * 10 ** (8)
    me = 9.109 * 10 ** (-31)
    ev_in_j = 6.242e18
    return (((8*(np.pi**2)*me*(np.abs(Ex-E0))/ev_in_j)/(planck**2))**(0.5))/10**(10)


def k_to_energy(E0, k):
    planck = 6.626 * 10 ** (-34)
    c = 2.998 * 10 ** (8)
    me = 9.109 * 10 ** (-31)
    ev_in_j = 6.242e18
    return E0 + (((k*10**10)**2 * planck**2)/(8*(np.pi**2)*me))*ev_in_j  
#%%
file1 = ReadSingleXANESnor('ThPO4_pH4p8.nor', skip = 37)
sample1 = np.vstack([[file1.e, file1.flat]]).T
bkg1 = np.vstack([[file1.e, file1.fbkg]]).T

#%%
# k_ME for Th multiexciton is ca. 9.95 w/ base of 0.10 inverse A

def normplot(data, E0, k_ME, region=False):
    fig, ax = plt.subplots(figsize = (10,5))
    for i in range(len(data)):
        d = data[i]
        x = d[:,0]
        y = d[:,1]
        if region != False:
            idx_x_min = (np.abs(x - region[0])).argmin()
            idx_x_max = (np.abs(x - region[1])).argmin()
            x = x[idx_x_min:idx_x_max]
            y = y[idx_x_min:idx_x_max]
        ax.axhline(y = 1, linestyle = '--', linewidth = 1.5, color = 'black')
        ax.axvline(x = np.min(x), linestyle = '--', linewidth = 1.5, color = 'red')
        ax.axvline(x = np.max(x), linestyle = '--', linewidth = 1.5, color = 'red')
        ax.axvline(x = k_to_energy(E0, k_ME+0.1), linestyle = '-', linewidth = 1, alpha = 0.50, color = 'yellow')
        ax.axvline(x = k_to_energy(E0, k_ME-0.1), linestyle = '-', linewidth = 1, alpha = 0.50, color = 'yellow')
        ax.axvspan(xmin = k_to_energy(E0, k_ME-0.1), xmax = k_to_energy(E0, k_ME+0.1), alpha = 0.45,color='yellow')
        ax.plot(x,y,linewidth = 3,label = str(i))    
        ax.legend(loc = 'upper right')
        
normplot([sample1, bkg1], 16291.922, 9.95, (16450, 17000))
    