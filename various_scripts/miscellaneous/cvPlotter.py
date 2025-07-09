# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:57:46 2023

@author: boris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sgf

#%%

dset = np.loadtxt('cv_deposition.csv', delimiter = ';')

#%%
fig, ax = plt.subplots(figsize = (10,10))
ax.set_prop_cycle('color',[plt.cm.inferno(i) for i in np.linspace(0, 0.95, 30)])
for i in range(0, len(dset[0,:]), 2):
    ax.plot(dset[:,i]+0.22, sgf(dset[:,i+1],9,3))
    
ax.set_xlabel("Potential vs SHE", fontsize = 27)
ax.set_ylabel("Current", fontsize = 27)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
    
