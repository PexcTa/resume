# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:24:05 2020

@author: boris
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
#%% Define normalization function
def normalize_1(data):
    """Normalizes a vector to it's maximum value"""
    normalized_data = data/max(data)
    return normalized_data



#%% 
def read_in_files():
    data = {}
    for file in os.listdir():
        if file.endswith(".xyd"):
            with open(file) as curfile:
                data[str(file)] = pd.read_csv(curfile,delim_whitespace=True,names=["2Theta", "Intensity"])
        elif file.endswith(".xye"):
            with open(file) as curfile:
                data[str(file)] = pd.read_csv(curfile,delim_whitespace=True,names=["2Theta", 'Random', "Intensity"],skiprows=1) 
    return data
        
rawdata = read_in_files()
print(rawdata.keys())

def RenameDictKeys(dictionary, new_names):
    res = {}
    for i in range(len(rawdata.keys())):
        res[new_names[i]] = dictionary[list(dictionary.keys())[i]]
    return res
rawdata = RenameDictKeys(rawdata, ['313_1', 'iso', 'nu1ksim', '313_2', '313sim'])

normdata = {}
for key in rawdata.keys():
    normdata[key] = pd.DataFrame()
    normdata[key] = rawdata[key]['2Theta']
    normdata[key] = pd.concat([normdata[key], normalize_1(rawdata[key]['Intensity'])], axis = 1)
    normdata[key].columns = ['2Theta', 'NI']
del key    


#%% Plot Stacked Patterns In One Big Figure
fig, ax = plt.subplots(figsize = (12,6))

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))

# Change the x-axis
ax.set_xlim([1.5, 15])
# ax.set_xticks(xticks)
ax.set_xlabel("2 Theta (degrees)", fontsize = 24)

ax.get_yaxis().set_ticks([])
ax.tick_params(axis='x', labelsize= 24)
# Set the yticks as well. A little more complex
ax.set_ylim([0.5,3])

ax.set_ylabel("Normalized Intensity (A.U.)", fontsize = 24)
labs = ['313_2', '313_1', '313sim']
ax.set_prop_cycle('color',[plt.cm.gnuplot2(i) for i in np.linspace(0.8, 0, len(labs))])

i = 0
while i < len(labs):
    ax.plot(normdata[labs[i]]['2Theta'], normdata[labs[i]]['NI']+(len(labs)-i)/2, linewidth = 3)
    i+=1

ax.legend(labels = ['UMCM-313 Sim.', 'UMCM-313, batch 1', 'UMCM-313, batch 2'][::-1], loc = 'upper right', ncol = 1, fontsize = 24)
# ax.axvline(x = 2.57, linestyle = '--', color = 'red', linewidth = 2.5)
# ax.axvline(x = 5.16, linestyle = '--', color = 'red', linewidth = 2.5)
# ax.axvline(x = 7.45, linestyle = '--', color = 'red', linewidth = 2.5)
# ax.axvline(x = 2.56, linestyle = '--', color = 'red', linewidth = 2.5)
plt.tight_layout()
# ax.legend(labels = ['0.1mM NatBuO', '10mM NatBuO', '1mM LiNO$_3$ in DMSO', '1mM NatBuO', 'untreated'], loc = 'upper right', fontsize = 24)
# plt.savefig('driedfromacetone.svg')

#%% Overlaid PXRD Patterns

fig, ax = plt.subplots(figsize = (10,6.6))
# Tell the interactive backend to plot the figure (not much to be seen here)
# xticks = np.arange(0,15,step = 1)
# yticks = np.linspace(0,1,11)

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(0.2))

# Change the x-axis
ax.set_xlim([2,15])
# ax.set_xticks(xticks)
ax.set_xlabel("2 Theta (degrees)", fontsize = 20)
# Set the yticks as well. A little more complex
# ax.set_ylim([0,1.1])
# ax.set_yticks(yticks)
ax.set_ylabel("Normalized Intensity", fontsize = 20)
ax.plot(normdata['bk145_nu601.xyd']['2Theta'], normdata['bk145_nu601.xyd']['NI'], color = 'black', linewidth = 3, label = 'NU-1000')
# ax.plot(data2['2 Theta'], data2['NI'], color = 'red', linewidth = 3, label = 'NU-901')
# ax.plot(traces_n1['wl'], traces_n1['BK80'], color = 'blue', linewidth = 3, label = '80 (phase-pure)')
# ax.plot(traces_n1['wl'], traces_n1['BK83'], color = 'green', linewidth = 3, label = '83 (Ti-SIM-1c)')
# ax.plot(traces_n1['wl'], traces_n1['BK84'], color = 'cyan', linewidth = 3, label = '84 (Zn-SIM-1c)')
ax.legend(loc = 'upper right')
# plt.savefig('NU1k_NU901_overlaid.svg')

#%% Stacked PXRD Patterns
# fig = plt.subplots(nrows = 2,  figsize = (10,6.6), sharex = True, sharey = True)


# fig, (axb, ax1, ax2) = plt.subplots(
#         nrows=2, ncols=1, sharex=True, sharey=True, 
#         )
# outer_grid = fig.add_gridspec(2,1, hspace=0)



fig = plt.figure(figsize = (12,12))
axb = fig.add_subplot(111)    # The big subplot
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# ax3 = fig.add_subplot(413)
# ax4 = fig.add_subplot(414)

# Turn off axis lines and ticks of the big subplot
axb.spines['top'].set_color('none')
axb.spines['bottom'].set_color('none')
axb.spines['left'].set_color('none')
axb.spines['right'].set_color('none')
axb.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
# Hide x labels and tick labels for all but bottom plot.

ax2.xaxis.set_major_locator(MultipleLocator(1))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# For the minor ticks, use no labels; default NullFormatter.
ax2.xaxis.set_minor_locator(MultipleLocator(0.2))

# Change the x-axis
for ax in [ax1, ax2]:
    ax.set_xlim([4,30])
# ax.set_xticks(xticks)
axb.set_xlabel("2 Theta (degrees)", fontsize = 18, labelpad = 8)
axb.set_ylabel("Normalized Intensity", fontsize = 18, labelpad = 8)

# these will go from the top of the figure in the same order you call axn.plot
ax1.plot(normdata['zndipyni_ndc_mof_06072021.xyd']['2Theta'], normdata['zndipyni_ndc_mof_06072021.xyd']['NI'], linewidth = 3, color = 'black', label = 'Simulated')
ax2.plot(normdata['simulatedPXRD.xye']['2Theta'], normdata['simulatedPXRD.xye']['NI'],  linewidth = 3, color = 'goldenrod', label = 'Zn-MOF')
# ax3.plot(normdata['znmof_cotm.xyd']['2Theta'], normdata['znmof_cotm.xyd']['Normalized Intensity'],  linewidth = 3, color = 'darkmagenta', label = 'Zn-MOF - Cobalt')
# ax4.plot(normdata['znmof_postsorb.xyd']['2Theta'], normdata['znmof_postsorb.xyd']['Normalized Intensity'],  linewidth = 3, color = 'goldenrod', label = 'Zn-MOF after activation')

# ax1.axvline(x=5.16, color = 'dimgrey', ls = '--')
# ax1.axvline(x=7.44, color = 'dimgrey', ls = '--')
# ax2.axvline(x=5.16, color = 'dimgrey', ls = '--')
# ax2.axvline(x=7.44, color = 'dimgrey', ls = '--')
# ax.plot(traces_n1['wl'], traces_n1['BK80'], color = 'blue', linewidth = 3, label = '80 (phase-pure)')
# ax.plot(traces_n1['wl'], traces_n1['BK83'], color = 'green', linewidth = 3, label = '83 (Ti-SIM-1c)')
# ax.plot(traces_n1['wl'], traces_n1['BK84'], color = 'cyan', linewidth = 3, label = '84 (Zn-SIM-1c)')
for ax in [ax1, ax2]:
    ax.label_outer()
    ax.tick_params(axis='y', labelsize= 16)
ax2.tick_params(axis='x', labelsize= 16)
fig.legend(loc = 'upper right', fontsize = 20)
# plt.savefig('stacked.svg')



#%%
fig = plt.figure()

fig.add_subplot(231)
ax1 = fig.add_subplot(2, 3, 1)  # equivalent but more general

fig.add_subplot(232, frameon=False)  # subplot with no frame
fig.add_subplot(233, projection='polar')  # polar subplot
# fig.add_subplot(234, sharex=ax1)  # subplot sharing x-axis with ax1
# fig.add_subplot(235, facecolor="red")  # red subplot

# ax1.remove()  # delete ax1 from the figure
# fig.add_subplot(ax1)  # add ax1 back to the figure