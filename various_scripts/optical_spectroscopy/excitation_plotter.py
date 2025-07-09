# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:40:52 2023

@author: boris
"""

#%% LIBRARIES
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from scipy.signal import savgol_filter as sgf
from scipy import special
import os
#%%
keyref = "wl1,k2_ex380,wl2,k2_ex425,wl3,k2_ex450,wl4,k2_ex500,wl5,h4_ex380,wl6,h4_ex425,wl7,h4_ex450,wl8,h4_ex500"

data = np.genfromtxt('excitation.csv', delimiter=',')[:,:]


fig, ax = plt.subplots(1,2,figsize = (20,10))

ax[0].set_xlim(min(data[:,6]), max(data[:,6]))
ax[1].set_xlim(min(data[:,6]), max(data[:,6]))


ax[0].set_xlabel('Wavelength (nm)', fontsize = 32)
ax[0].set_ylabel('Counts (CPS)', fontsize = 32)
ax[0].tick_params(axis='both', labelsize=24)
ax[0].set_prop_cycle('color', [plt.cm.RdBu(i) for i in np.linspace(0, 0.35, 4)])
for i in [0,2,4,6]:
    ax[0].plot(data[:,i], data[:,i+1], linewidth = 5)
ax[0].legend(['380 nm', '425 nm', '450 nm', '500 nm'], fontsize = 32)
    

ax[1].set_xlabel('Wavelength (nm)', fontsize = 32)
ax[1].set_ylabel('Counts (CPS)', fontsize = 32)
ax[1].tick_params(axis='both', labelsize=24)
ax[1].set_prop_cycle('color', [plt.cm.RdBu(i) for i in np.linspace(0.65, 0.99, 4)])
for i in [8, 10, 12, 14]:
    ax[1].plot(data[:,i], data[:,i+1], linewidth = 5)
ax[1].legend(['380 nm', '425 nm', '450 nm', '500 nm'], fontsize = 32)


plt.tight_layout()