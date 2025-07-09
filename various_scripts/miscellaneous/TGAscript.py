# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 09:41:26 2021

@author: boris
"""
#%% Import Libraries

import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from scipy.signal import savgol_filter
from scipy import special
import os


#%% Processing
def normend(data):
    return (data/np.min(data))*100


#%% Import Data

def ImportTGACurve(file):
    """Assumes Blank subtracted on the instrument. Assumes the first two rows are trash
    Time is in seconds, mass in mg, temps in celsius"""
    with open(file) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, skiprows = 2, names = ['index','time','mass','Tsam','Tref'])
        return d
    
bk125_tfa = ImportTGACurve('BK125_FUiO66_TFA_07212021_BlankSubtracted.txt')
#%% Plot Data
        
fig, ax = plt.subplots(figsize=(10,8))
# ax.set_xlabel('Time (s)', fontsize = 20)
ax.set_xlabel('Temperature ($^\circ$C)', fontsize = 24)
ax.set_ylabel("Normalized Mass %", fontsize = 24)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.set_xlim([30,600])
# ax.set_ylim([-0.01, 0.16])
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(25))
# for c in range(4,14,2):
#     ax.axhline(y=c, color = 'dimgrey', ls = '--', alpha = 0.5)
# ax.axvline(x=1, color = 'darkseagreen', ls = '--')

d1 = bk123_ref.copy()
d2 = bk123_tfa.copy()
d3 = bk124_tfa.copy()
d4 = bk125_tfa.copy()


ax.plot(d1['Tsam'], normend(d1['mass']), color = 'black', linewidth = 3, label = 'UiO-66')
ax.plot(d2['Tsam'], normend(d2['mass']), color = 'red', linewidth = 3, label = 'UiO-66-TFA')
ax.plot(d3['Tsam'], normend(d3['mass']), color = 'orange', linewidth = 3, label = 'NH2-UiO-66-TFA')
ax.plot(d4['Tsam'], normend(d4['mass']), color = 'purple', linewidth = 3, label = 'F-UiO-66-TFA')
# ax.annotate("Initial: "+str(round(max(d1['mass']),4))+" mg", (100, 2.25), fontsize = 14)
# ax.annotate("Final: "+str(round(min(d1['mass']), 4))+" mg", (450, 1.25), fontsize = 14)
# ax.annotate("Total Loss: "+str(round(100*(max(d1['mass'])-min(d1['mass']))/max(d1['mass']), 2))+ ' %', (350, 9), fontsize = 15)
# ax.annotate("6.24x H$_2$O ", (350, 8.5), fontsize = 15)
# ax.plot(d2['time'], data_R['Fit'], color = 'crimson', linewidth = 3, label = 'Fit')
# ax.scatter(TiFit_R['R'], np.subtract(TiFit_R['Data'],TiFit_R['Fit']),  s=10, facecolors = None, edgecolors = 'magenta',  marker = 'o', label = 'Residuals')
# ax.axvline(x=3.4, color = 'darkseagreen', ls = '--', label = 'Window')
ax.legend(loc = 'upper right', fontsize=24)
# plt.savefig('ppNU1k_EmMaxs_vs_sqRind_plot.svg')