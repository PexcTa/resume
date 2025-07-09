# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:25:08 2024

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
from scipy.signal import savgol_filter
from scipy import special
from scipy.stats import cauchy
import os
#%%
def ReadFile(filename, header_list, skip=4):
    """Set this up with the header corresponding to actual paths in a readable form"""
    with open(filename) as current_file:
        d = pd.read_csv(current_file, sep = '\s+', header = 0, index_col = False, names = header_list, engine = 'python', skiprows = lambda x: x in range(skip))
        return d
    
# test= ReadFile('nlink_1_3.dat', ['R', 'ON_1', 'OL_1', 'CL_1', 'Zr_1', 'OL_2', 'ON_2', 'OH', 'H2O'])

def ReadDire(header_list, skip=4):
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".dat"):
            data[file]=ReadFile(file, header_list)
    return data

names = ['x','data']
dsre = ReadDire(names)
#%%
k = np.array(dsre['ThPhos_PathSum_kspa.dat'].iloc[:,0])
oscil = np.array(dsre['ThPhos_PathSum_kspa.dat'].iloc[:,1])

meank = np.mean(oscil)
noise = np.random.normal(meank, 1.5, np.shape(oscil)[0])
noise = np.multiply(noise, np.random.rand(np.shape(oscil)[0]))
noise = np.multiply(noise, 0.002*k**3)

spike = 0.75*cauchy.pdf(k, 9.95, 0.04)
fig, ax = plt.subplots(figsize = (15,5))
ax.plot(k, oscil)
# ax.plot(k, noise)
ax.plot(k, oscil+noise+spike)
# ax.plot(k, spike)

#%%
chik3 = oscil
chik1 = np.zeros_like(chik3)
for i in range(len(chik3)):
    if chik3[i] == 0:
        chik1[i] = 0
    else:
        chik1[i] = chik3[i]/k[i]**3
output = np.vstack([k, chik3, chik1]).T
np.savetxt('ThPhos_PathSum_k.dat', output, fmt='%1.5f', delimiter=',')
#%%