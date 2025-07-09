# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:45:54 2021
My EXAFS and XANES plotting script
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
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter as sgf
from scipy import special
import os

#%% DATA IMPORT tailored for demeter suite

def normalize_1(data):
    """Normalizes a vector to it's maximum value"""
    normalized_data = data/max(data)
    return normalized_data
def ReadSingleExafs_Real(file, skip=3):
    """
    Reads in a file and assigns it to a variable. 
    Will assign column names as 'energy' and 'int'.
    Skips a given number of lines.
    Input is a comma-separated data file. 
    Output is a Pandas Dataframe.
    """
    with open(file) as current_file:
        d = pd.read_csv(current_file, sep = '\s+', header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
def ReadSingleExafs_R(file, skip=3):
    """
    Reads in a file and assigns it to a variable. 
    Will assign column names as 'energy' and 'int'.
    Skips a given number of lines.
    Input is a comma-separated data file. 
    Output is a Pandas Dataframe.
    """
    with open(file) as current_file:
        d = pd.read_csv(current_file, sep = '\s+', header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
def ReadSingleExafs_k(file, skip=3):
    """
    Reads in a file and assigns it to a variable. 
    Will assign column names as 'energy' and 'int'.
    Skips a given number of lines.
    Input is a comma-separated data file. 
    Output is a Pandas Dataframe.
    """
    with open(file) as current_file:
        d = pd.read_csv(current_file, sep = '\s+', header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
def ReadMultiExafs(skip):
    """
    Reads all data in the current directory (only .dat files)
    Skips a given number of rows in each file.
    Output is a dict of Dataframes.
    """
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith("k.dat"):
            data[file] = ReadSingleExafs_k(file)
        if file.endswith("R.dat"):
            data[file] = ReadSingleExafs_R(file)
    return data

def ReadSingleXANESxmu(file, skip=40):
    with open(file) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, header = 0, names = ['E', 'xmu', 'bkg', 'pre_edge', 'post_edge', 'deriv', 'deriv_2', 'i0', 'chie'], engine = 'python', skiprows = lambda x: x in range(skip))
        return d
    
def ReadMultiXANESnor(skip):
    """
    Reads all data in the current directory (only .dat files)
    Skips a given number of rows in each file.
    Output is a dict of Dataframes.
    """
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".nor"):
            data[file] = ReadSingleXANESnor(file)
    return data

def ReadSingleXANESnor(file, skip=11):
    with open(file) as current_file:
        # names = ['E', 'norm', 'nbkg', 'flat', 'fbkg', 'nderiv', 'nderiv_2'], 
        d = pd.read_csv(current_file, delim_whitespace = True, header = [0], engine = 'python', skiprows = lambda x: x in range(skip))
        colnames = d.columns[1:]
        d = d.drop(d.columns[-1], axis=1)
        d.columns = colnames
        return d
    
def ReadSingleEXAFSchir(file, skip = 38):
    with open(file) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, header = 0, names = ['R', 'real', 'imag', 'magn', 'pha', 'win', 'deriv_pha'], engine = 'python', skiprows = lambda x: x in range(skip))
        return d
    
    
def ReadMultiEXAFSchik(skip=38):
    """
    Reads all data in the current directory (only .chik files)
    Skips a given number of rows in each file.
    Output is a dict of Dataframes.
    """
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".chik"):
            data[file] = ReadSingleEXAFSchik(file)
    return data
    
def ReadSingleEXAFSchik(file, skip = 38):
    with open(file) as current_file:
        d = pd.read_csv(current_file, delim_whitespace = True, header = 0, names = ['k', 'chi', 'chik', 'chik2', 'chik3', 'win', 'energy'], engine = 'python', skiprows = lambda x: x in range(skip))
        return d
#%% Import Data
data_R = ReadSingleExafs_R('Rmag.dat')
data_Re = ReadSingleExafs_Real('Rreal.dat')
data_k = ReadSingleExafs_k('kspace.dat')
data_Im = ReadSingleExafs_Real('Rimag.dat')
data_postpyro = {"mag": data_R,
                "real":data_Re,
                "imag":data_Im,
                "kay":data_k}
#%% Import Data
data_im = {}
data_re = {}
data_rm = {}
data_ks = {}

for file in os.listdir():
    if file.endswith(".dat"):
        with open(file) as current_file:
            if file.endswith("imag.dat"):
                data_im[file]=ReadSingleExafs_Real(file)
            if file.endswith("real.dat"):
                data_re[file]=ReadSingleExafs_Real(file)
            if file.endswith("rmag.dat"):
                data_rm[file]=ReadSingleExafs_R(file)
            if file.endswith("kspa.dat"):
                data_ks[file]=ReadSingleExafs_k(file)
                
#%% custom imports

zro2 = ReadSingleXANESnor('xanesNorm.nor', skip = 12)
sio2_4 = ReadSingleXANESnor('tio2sio2_refs.nor', skip = 11)
sio2_1 = ReadSingleXANESnor('tio2sio2_refs_1nm.nor', skip = 11)
#%% Plot Raw Data and Fit in R-space 

fig, ax = plt.subplots(figsize=(13,10))
ax.set_xlabel("R($\AA$)", fontsize = 24)
ax.set_ylabel("Norm|\u03C7(R)|($\AA^{-3}$)", fontsize = 24)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.set_xlim([0,5])
# ax.set_ylim([-5,8])
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
# ax.axhline(y=0, color = 'dimgrey', ls = '--')
# ax.axvline(x=1.16, color = 'darkseagreen', linewidth = 2.5,ls = '--')
ax.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0, 0.8, 4)])
ax.plot(0,0)
ax.scatter(data_rm['mosim_rspa.dat']['r'], normalize_1(data_rm['mosim_rspa.dat']['MoSx_SIM_RT_Air']), color = 'royalblue', s = 60, facecolors = 'none',linewidth = 2.5, label = 'MoSx-SIM')
ax.plot(data_rm['mosim_rspa.dat']['r'], normalize_1(data_rm['mosim_rspa.dat']['MoSx_SIM_RT_Air_fit']),  linewidth = 3.5, label = 'MoSx-SIM Fit')
ax.scatter(data_rm['moaim_rspa.dat']['r'], normalize_1(data_rm['moaim_rspa.dat']['SA_MoSx_AIM_FF_RT_KBW']), color = 'crimson', s = 60, facecolors = 'none',linewidth = 2.5, label = 'MoSx-AIM')
ax.plot(data_rm['moaim_rspa.dat']['r'], normalize_1(data_rm['moaim_rspa.dat']['SA_MoSx_AIM_FF_RT_KBW_fit']),  linewidth = 3.5, label = 'MoSx-AIM Fit')
ax.plot(0,0)
# ax.plot(data_Im['R'], data_Im['Data'], color = 'peru', linewidth=1.5, label = 'UiO-66-d, imaginary')

# ax.plot(data_Im['R'], data_Im['Fit'], color = 'crimson',linewidth=3, label = 'As prepared')
# ax.plot(data_Im['R'], data_Im['Fit'], color = 'crimson', linewidth = 3, label = 'Fit')
# ax.scatter(TiFit_R['R'], np.subtract(TiFit_R['Data'],TiFit_R['Fit']),  s=10, facecolors = None, edgecolors = 'magenta',  marker = 'o', label = 'Residuals')
# ax.axvline(x=3.60, color = 'darkseagreen', linewidth = 2.5, ls = '--')
ax.legend(loc = 'upper right', fontsize=24)
# plt.savefig('UiO66_Freefit_R.png')
#%% Plot Raw Data and Real FT Part in R-space 

fig, ax = plt.subplots(figsize=(13,10))
ax.set_xlabel("R($\AA$)", fontsize = 24)
# ax.set_ylabel("|\u03C7(R)|($\AA^{-3}$)", fontsize = 20)
ax.set_ylabel("Im[\u03C7(R)]($\AA^{-4}$)", fontsize = 24)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.set_xlim([0,5])
# ax.set_ylim([-0.01, 0.16])
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
# ax.axhline(y=0, color = 'dimgrey', ls = '--')
ax.axvline(x=1.16, color = 'darkseagreen', ls = '--')
ax.scatter(data_Im['R'], data_Im['Data'], color = 'darkgray', s = 40, facecolors = 'none',linewidth = 3, label = 'As prepared')
ax.plot(data_Im['R'], data_Im['Fit'], color = 'royalblue', linewidth = 3, label = 'Fit')
# ax.scatter(TiFit_R['R'], np.subtract(TiFit_R['Data'],TiFit_R['Fit']),  s=10, facecolors = None, edgecolors = 'magenta',  marker = 'o', label = 'Residuals')
ax.axvline(x=3.60, color = 'darkseagreen', ls = '--')
ax.legend(loc = 'upper right', fontsize=24)
# plt.savefig('UiO66_Freefit_Img.png')

#%% Plot a Bunch of Data in R-space , Normalized

fig, ax = plt.subplots(figsize=(12,12))

fs1 = 28
fs2 = 24
ax.set_xlabel("R($\AA$)", fontsize = fs1)
# ax.set_ylabel("Norm[|\u03C7(R)|]($\AA^{-3}$)", fontsize = fs1)
ax.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
ax.tick_params(axis='x', labelsize= fs2)
ax.tick_params(axis='y', labelsize= fs2)
ax.set_xlim([0,5])
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
ax.axhline(y=0, color = 'dimgrey', ls = '--')
ax.set_xticks([])
# ax.axvline(x=1, color = 'darkseagreen', ls = '--')
# ax.plot(tixafs['R'], normalize_1(tixafs['magn']), color = 'gray', linewidth = 4, linestyle = '--', label = 'Ti Foil')
# ax.plot(data_rm['r1_rmag.dat']['r'], normalize_1(data_rm['r1_rmag.dat']['Ti_R1_001.dat']), color = 'red', linewidth = 4, linestyle = '-.', label = 'Anatase')
# ax.plot(rtxafs['R'], normalize_1(rtxafs['magn']), color = 'limegreen', linewidth = 4, linestyle = '--', label = 'Rutile')
# ax.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0.1, 0.8, 2)])
# ax.plot(data_rm['ti1_rmag.dat']['r'], normalize_1(data_rm['ti1_rmag.dat']['Ti1merged_Rbkg1']), linewidth = 4, label = '4 nm TiO$_2$@ZrO$_2$, pwdr.')
# ax.plot(data_rm['ti3_rmag.dat']['r'], normalize_1(data_rm['ti3_rmag.dat']['Ti3_merged']), linewidth = 4, label = '1 nm TiO$_2$@ZrO$_2$, pwdr.')

ax.plot(tixafs['R'], tixafs['magn'], color = 'gray', linewidth = 4, linestyle = '--', label = 'Ti Foil')
ax.plot(data_rm['r1_rmag.dat']['r'], data_rm['r1_rmag.dat']['Ti_R1_001.dat'], color = 'red', linewidth = 4, linestyle = '-.', label = 'Anatase')
ax.plot(rtxafs['R'], rtxafs['magn'], color = 'limegreen', linewidth = 4, linestyle = '--', label = 'Rutile')
ax.set_prop_cycle('color', [plt.cm.magma(i) for i in np.linspace(0.1, 0.8, 2)])
ax.plot(tio2sio24nm_xafs['R'], tio2sio24nm_xafs['magn'],  linewidth = 4,  label = 'Rutile')
ax.plot(tio2sio21nm_xafs['R'], tio2sio21nm_xafs['magn'],  linewidth = 4,  label = 'Rutile')
# ax.plot(data_rm['ti1_rmag.dat']['r'], data_rm['ti1_rmag.dat']['Ti1merged_Rbkg1'], linewidth = 4, label = '4 nm TiO$_2$@ZrO$_2$, pwdr.')
# ax.plot(data_rm['ti3_rmag.dat']['r'], data_rm['ti3_rmag.dat']['Ti3_merged'], linewidth = 4, label = '1 nm TiO$_2$@ZrO$_2$, pwdr.')
# ax.scatter(foil_R['R'], normalize_1(foil_R['magn'])*2, color = 'darkgrey',facecolors = 'none', linewidth = 2.5, alpha = 0.7, label = 'Fe foil')
# ax.scatter(porph_R['R'], normalize_1(porph_R['magn']), color = 'indigo', facecolors = 'none', linewidth = 2.5, alpha = 0.5,  label = 'Fe porphyrin')
# ax.plot(pre_R['R'], normalize_1(pre_R['magn']), color = 'teal', linewidth = 3, label = 'As prepared')
# ax.plot(post_R['R'], normalize_1(post_R['magn']), color = 'crimson', linewidth = 3, label = 'Pyrolyzed')
# ax.plot(feo['R'], normalize_1(feo['magn']), color = 'firebrick', linestyle = '--', linewidth = 2.5, alpha = 0.7, label = 'FeO')
# ax.plot(fe2o3['R'], normalize_1(fe2o3['magn']), color = 'forestgreen', linestyle = '--', linewidth = 2.5, alpha = 0.7, label = 'Fe$_2$O$_3$')

# ax.plot(NiOEXAFS_R['R'], NiOEXAFS_R['magn'], color = 'black', linewidth = 3, label = 'NiO')
# ax.plot(NiSIMEXAFS_R['R'], NiSIMEXAFS_R['magn'], color = 'teal', linewidth = 3, label = 'Ni-SIM')


# ax.scatter(TiFit_R['R'], np.subtract(TiFit_R['Data'],TiFit_R['Fit']),  s=10, facecolors = None, edgecolors = 'magenta',  marker = 'o', label = 'Residuals')
# ax.axvline(x=3.4, color = 'darkseagreen', ls = '--', label = 'Window')
# ax.legend(loc = 'upper right', fontsize=24)
# plt.savefig('ppNU1k_EmMaxs_vs_sqRind_plot.svg')
plt.tight_layout()

#%% Plot Raw Data and Fit in k-space 
fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlabel("k($\AA^{-1}$)", fontsize = 24)
ax.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = 24)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.set_xlim([0,12])
ax.set_ylim([-7.5, 7.5])
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
# ax.axhline(y=0, color = 'dimgrey', ls = '--')
ax.axvline(x=2, color = 'darkseagreen', ls = '--')
# ax.plot(data_k['k'], data_k['Data'], color = 'darkmagenta', linewidth = 3, label = 'Co-SIM')
ax.scatter(data_prepyro['kay']['k'], data_prepyro['kay']['Data'],  s=40, linewidth = 3, facecolors = 'none', edgecolors = 'darkgray',  label = 'As prepared')
ax.plot(data_prepyro['kay']['k'], data_prepyro['kay']['Fit'], color = 'crimson', linewidth = 3, label = 'Fit')
# ax.scatter(TiFit_R['R'], np.subtract(TiFit_R['Data'],TiFit_R['Fit']),  s=10, facecolors = None, edgecolors = 'magenta',  marker = 'o', label = 'Residuals')
ax.axvline(x=9, color = 'darkseagreen', ls = '--')
ax.legend(loc = 'upper right', fontsize=24)
# plt.savefig('UiO66_Freefit_k.png')


#%% Plot a Bunch of Data in k-space 

fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlabel("k($\AA^{-1}$)", fontsize = 24)
ax.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = 24)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.set_xlim([0,12])
# ax.set_ylim([-40, 40])
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
# ax.axhline(y=0, color = 'dimgrey', ls = '--')
# ax.axvline(x=1, color = 'darkseagreen', ls = '--')
ax.plot(kscans['postG.chik']['k'], kscans['postG.chik']['chik3'], color = 'blue', linewidth = 3, label = 'Raw data')
ax.plot(kscans['postDG.chik']['k'], kscans['postDG.chik']['chik3'], color = 'black', linewidth = 3, label = 'Deglitched')
# ax.plot(CoEXAFS_k['k'], CoEXAFS_k['chik3'], color = 'black', linewidth = 3, label = 'Co foil')
# ax.scatter(TiFit_R['R'], np.subtract(TiFit_R['Data'],TiFit_R['Fit']),  s=10, facecolors = None, edgecolors = 'magenta',  marker = 'o', label = 'Residuals')
# ax.axvline(x=3.4, color = 'darkseagreen', ls = '--', label = 'Window')
ax.legend(loc = 'upper right', fontsize=24)
# plt.savefig('ppNU1k_EmMaxs_vs_sqRind_plot.svg')

#%% Plot Normalized XANES
fig, ax = plt.subplots(figsize = (12,8))
# ax.set_xlabel("Energy (keV)", fontsize = 32)
# ax.set_ylabel("d/dx\u03bc(E)", fontsize = 32)
# ax.set_ylabel("Normalized x\u03bc(E)", fontsize = 32)
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.axhline(y = 0, ls = '--', color = 'dimgrey')
ax.set_xlim([8.3, 8.45])
# ax.set_ylim([1.0, 1.45])
# ax.xaxis.set_major_locator(MultipleLocator(0.01))
# ax.xaxis.set_minor_locator(MultipleLocator(0.0025))
xanes = data.copy()
cmap = plt.cm.gist_ncar
labels = ['NU-1000, RT', 'NU-1000, 60$^\circ$C']

ax.plot(xanes['energy']/1000, xanes['_Ref_Ni2P_TM'],  linewidth = 4, color='gray', linestyle = '--',  label = 'Ni Foil')
ax.plot(xanes['energy']/1000, xanes['NiO_TM'], linewidth = 4,  color = 'black', linestyle = '-.',  label = 'NiO')
ax.plot(xanes['energy']/1000, xanes['Ni2P_TM'], linewidth = 4,  color = 'purple', linestyle = '-',label = 'Ni$_2$P')
ax.set_prop_cycle('color', [plt.cm.gnuplot2(i) for i in np.linspace(0.1, 0.8, 2)])
ax.plot(zro2['energy']/1000, zro2['Ti4nm_merged'], linewidth = 4, label = '4nm TiO$_2$@ZrO$_2$, pwdr.')
ax.plot(zro2['energy']/1000, sgf(zro2['Ti2_merged'],9,3), linewidth = 4, label = '4nm TiO$_2$@ZrO$_2$, mbrn.')

plt.tight_layout()
#%% Plot Normalized XANES on a Cool Layout
fig = plt.figure(figsize = (16,8))
gs = fig.add_gridspec(2,4)
ax1 = fig.add_subplot(gs[:, 0:3])
ax2 = fig.add_subplot(gs[0, 3])
ax3 = fig.add_subplot(gs[1, 3])
ax1.set_xlabel("Energy (keV)", fontsize = 32)
ax3.set_xlabel("Energy (keV)", fontsize = 32)
# ax1.set_ylabel("d/dx\u03bc(E)", fontsize = 32)
ax1.set_ylabel("Normalized x\u03bc(E)", fontsize = 32)
ax1.tick_params(axis='x', labelsize= 24)
ax1.tick_params(axis='y', labelsize= 24)
ax2.tick_params(axis='x', labelsize= 24)
ax2.set_yticks([])
ax3.tick_params(axis='x', labelsize= 24)
ax3.set_yticks([])
# ax1.axhline(y = 0, ls = '--', color = 'dimgrey')

# ax.set_ylim([1.0, 1.45])
# ax.xaxis.set_major_locator(MultipleLocator(0.01))
# ax.xaxis.set_minor_locator(MultipleLocator(0.0025))

# ax.axvline(x = 4.966, ls = '-.', color = 'indianred')

# ax.annotate('Ti K edge', (4.967, 1), fontsize = 14)
# ax.annotate('4966 eV', (4.967, 0.95), fontsize = 14)


# d1 = TiXANES.copy()
# d2 = TiO2XANES.copy()
# d3 = TiSIMXANES.copy()
# ax.plot(scans['energy']/1000, scans[''],   linewidth = 3, linestyle = '--', label = 'Ex-situ 1 nm shell')
ax1.set_xlim([4.95, 5.15])
ax1.xaxis.set_major_locator(MultipleLocator(0.02))
ax1.plot(zro2['energy']/1000, zro2['TiFoil_Calibred'],  linewidth = 4, color='gray', linestyle = '--',  label = 'Ti Foil')
ax1.plot(zro2['energy']/1000, zro2['Ti_R1_001.dat'], linewidth = 4,  color = 'crimson', linestyle = '-.',  label = 'Anatase')
ax1.plot(zro2['energy']/1000, zro2['Ti_R2_001.dat'], linewidth = 4,  color = 'limegreen', linestyle = '--',label = 'Rutile')
ax1.set_prop_cycle('color', [plt.cm.magma(i) for i in np.linspace(0.1, 0.8, 2)])
# ax1.plot(zro2['energy']/1000, zro2['Ti4nm_merged'], linewidth = 4, label = '4nm TiO$_2$@ZrO$_2$, pwdr.')
# ax.plot(zro2['energy']/1000, sgf(zro2['Ti2_merged'],9,3), linewidth = 4, label = '4nm TiO$_2$@ZrO$_2$, mbrn.')
# ax1.plot(zro2['energy']/1000, sgf(zro2['Ti3_merged'],9,3), linewidth = 4, label = '1nm TiO$_2$@ZrO$_2$, pwdr.')
ax1.plot(sio2_4['energy']/1000+0.0005, sgf(sio2_4['4nm_dry_merge'],9,3), linewidth = 4, label = '4nm TiO$_2$@SiO$_2$, pwdr.')
ax1.plot(sio2_1['energy']/1000+0.0005, sgf(sio2_1['1nm_Dry_merge'],9,3), linewidth = 4, label = '1nm TiO$_2$@SiO$_2$, pwdr.')

ax2.set_xlim([4.98, 5.01])
ax2.xaxis.set_major_locator(MultipleLocator(0.01))
ax2.set_ylim([0.9, 1.4])
ax2.plot(zro2['energy']/1000, zro2['TiFoil_Calibred'],  linewidth = 4, color='gray', linestyle = '--', alpha = 0.5,  label = 'Ti Foil')
ax2.plot(zro2['energy']/1000, zro2['Ti_R1_001.dat'], linewidth = 4,  color = 'crimson', linestyle = '-.', alpha = 0.5, label = 'Anatase')
ax2.plot(zro2['energy']/1000, zro2['Ti_R2_001.dat'], linewidth = 4,  color = 'limegreen', linestyle = '--',label = 'Rutile')
ax2.set_prop_cycle('color', [plt.cm.magma(i) for i in np.linspace(0.1, 0.8, 2)])
# ax2.plot(zro2['energy']/1000, zro2['Ti4nm_merged'], linewidth = 4, label = '4nm TiO$_2$@ZrO$_2$, pwdr.')
# ax.plot(zro2['energy']/1000, sgf(zro2['Ti2_merged'],9,3), linewidth = 4, label = '4nm TiO$_2$@ZrO$_2$, mbrn.')
# ax2.plot(zro2['energy']/1000, sgf(zro2['Ti3_merged'],9,3), linewidth = 4, label = '1nm TiO$_2$@ZrO$_2$, pwdr.')
ax2.plot(sio2_4['energy']/1000+0.0005, sgf(sio2_4['4nm_dry_merge'],9,3), linewidth = 4, label = '4nm TiO$_2$@SiO$_2$, pwdr.')
ax2.plot(sio2_1['energy']/1000+0.0005, sgf(sio2_1['1nm_Dry_merge'],9,3), linewidth = 4, label = '1nm TiO$_2$@SiO$_2$, pwdr.')

ax3.set_xlim([4.96, 4.98])
ax3.xaxis.set_major_locator(MultipleLocator(0.01))
ax3.set_ylim([0.0, 0.3])
ax3.plot(zro2['energy']/1000, zro2['TiFoil_Calibred'],  linewidth = 4, color='gray', linestyle = '--', alpha = 0.5,  label = 'Ti Foil')
ax3.plot(zro2['energy']/1000, zro2['Ti_R1_001.dat'], linewidth = 4,  color = 'crimson', linestyle = '-.', alpha = 0.5, label = 'Anatase')
ax3.plot(zro2['energy']/1000, zro2['Ti_R2_001.dat'], linewidth = 4,  color = 'limegreen', linestyle = '--',label = 'Rutile')
ax3.set_prop_cycle('color', [plt.cm.magma(i) for i in np.linspace(0.1, 0.8, 2)])
# ax3.plot(zro2['energy']/1000, zro2['Ti4nm_merged'], linewidth = 4, label = '4nm TiO$_2$@ZrO$_2$, pwdr.')
# ax.plot(zro2['energy']/1000, sgf(zro2['Ti2_merged'],9,3), linewidth = 4, label = '4nm TiO$_2$@ZrO$_2$, mbrn.')
# ax3.plot(zro2['energy']/1000, sgf(zro2['Ti3_merged'],9,3), linewidth = 4, label = '1nm TiO$_2$@ZrO$_2$, pwdr.')
ax3.plot(sio2_4['energy']/1000+0.0005, sgf(sio2_4['4nm_dry_merge'],9,3), linewidth = 4, label = '4nm TiO$_2$@SiO$_2$, pwdr.')
ax3.plot(sio2_1['energy']/1000+0.0005, sgf(sio2_1['1nm_Dry_merge'],9,3), linewidth = 4, label = '1nm TiO$_2$@SiO$_2$, pwdr.')

ax1.legend(loc = 'lower right',  fontsize = 28)
ax2.axvline(x=4.987, color = 'black', linewidth = 2.5,ls = ':')
ax2.axvline(x=4.992, color = 'black', linewidth = 2.5,ls = ':')
ax2.axvline(x=5.004, color = 'black', linewidth = 2.5,ls = ':')
# ax.plot(xanes['energy']/1000, xanes['TiFoil_Calibred'],  linewidth = 4, color='gray', linestyle = '--', alpha = 0.25, label = 'Ti Foil')
# ax.plot(xanes['energy']/1000, xanes['Ti_R1_001.dat'], linewidth = 4,  color = 'crimson', linestyle = '-.', alpha = 0.25, label = 'Anatase')
plt.tight_layout()
#%% That one time that i was exporting data for someone who can't use anything but excel fml
Glitch_data_for_plotting_k = pd.DataFrame()
for key in kscans.keys():
    temp = pd.DataFrame(data={'_k': kscans[key]['k'], '_chik3':kscans[key]['chik3']})
    Glitch_data_for_plotting_k = pd.concat([Glitch_data_for_plotting_k, temp], axis = 1)
    
# exafs_data_for_plotting = pd.DataFrame()
# for key in [fe2o3, feo, foil_R, porph_R, post_R, pre_R]:
#     temp = pd.DataFrame(data={'_R': key['R'], '_Rmag':key['magn']})
#     exafs_data_for_plotting = pd.concat([exafs_data_for_plotting, temp], axis = 1)
    
#%% Plot quad figure - EXAFS
data_im = {}
data_re = {}
data_rm = {}
data_ks = {}

for file in os.listdir():
    if os.path.isdir(file):
    # skip directories
        continue
    if file.endswith(".dat"):
        with open(file) as current_file:
            if file.endswith("imag.dat"):
                data_im[file]=ReadSingleExafs_Real(file)
            if file.endswith("real.dat"):
                data_re[file]=ReadSingleExafs_Real(file)
            if file.endswith("rmag.dat"):
                data_rm[file]=ReadSingleExafs_R(file)
            if file.endswith("kspa.dat"):
                data_ks[file]=ReadSingleExafs_k(file)

#%%
fig, axs = plt.subplots(2,2,figsize=(24,18))

datadict_caller = "npmg_300"
dataset_caller = "NpMg_300_T"
#['Fit_to_'+dataset_caller+'_from_Fit_122']
r = data_re[datadict_caller+"_real.dat"]['r']
k = data_ks[datadict_caller+'_kspa.dat']['k']
dset_re = data_re[datadict_caller+'_real.dat'][dataset_caller]
dset_im = data_im[datadict_caller+'_imag.dat'][dataset_caller]
dset_rm = data_rm[datadict_caller+'_rmag.dat'][dataset_caller]
dset_ks = data_ks[datadict_caller+'_kspa.dat'][dataset_caller]

fit_re = data_re[datadict_caller+'_real.dat'][dataset_caller+'_fit']
fit_im = data_im[datadict_caller+'_imag.dat'][dataset_caller+'_fit']
fit_rm = data_rm[datadict_caller+'_rmag.dat'][dataset_caller+'_fit']
fit_ks = data_ks[datadict_caller+'_kspa.dat'][dataset_caller+'_fit']

# fit_re = data_re[datadict_caller+'_real.dat']['Fit_to_'+dataset_caller+'_from_Fit_37']
# fit_im = data_im[datadict_caller+'_imag.dat']['Fit_to_'+dataset_caller+'_from_Fit_37']
# fit_rm = data_rm[datadict_caller+'_rmag.dat']['Fit_to_'+dataset_caller+'_from_Fit_37']
# fit_ks = data_ks[datadict_caller+'_kspa.dat']['Fit_to_'+dataset_caller+'_from_Fit_37']

k_window = (3.16, 11.8)
r_window = (1.0, 4.39)

dataLabel0 = "Mg$_{0.5}$NpO$_2$CO$_3$, 300$^{\circ}$C"
# dataLabel0 = "1nm TiO$_2$ on ZrO$_2$, powder"
# fitcolor = 'firebrick'
fitcolor = 'crimson'
datacolor = 'slategray'

axs[0,0].set_xlabel("R($\AA$)", fontsize = 28)
axs[0,0].set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = 28)
axs[0,0].tick_params(axis='x', labelsize= 28)
axs[0,0].tick_params(axis='y', labelsize= 28)
axs[0,0].set_xlim([0,5])
# ax.set_ylim([-5,8])
axs[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[0,0].xaxis.set_major_locator(MultipleLocator(1))
axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.2))
axs[0,0].axhline(y=0, color = 'dimgrey', ls = '--')
axs[0,0].axvline(x=r_window[0], color = 'darkseagreen', linewidth = 4,ls = '--')
# axs[0,0].scatter(r, dset_rm, color = datacolor, s = 40, facecolors = 'none',linewidth = 3, label = dataLabel0)
axs[0,0].plot(r, dset_rm, color = datacolor, linewidth = 6, label = dataLabel0)
axs[0,0].plot(r, fit_rm, color = fitcolor, linewidth = 4.5, label = 'Fit')
axs[0,0].axvline(x=r_window[1], color = 'darkseagreen', linewidth = 4, ls = '--')
axs[0,0].legend(loc = 'best', fontsize=24)

axs[0,1].set_xlabel("R($\AA$)", fontsize = 28)
axs[0,1].set_ylabel("Re[\u03C7(R)]($\AA^{-4}$)", fontsize = 28)
axs[0,1].tick_params(axis='x', labelsize= 28)
axs[0,1].tick_params(axis='y', labelsize= 28)
axs[0,1].set_xlim([0,5])
# ax.set_ylim([-5,8])
axs[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[0,1].xaxis.set_major_locator(MultipleLocator(1))
axs[0,1].xaxis.set_minor_locator(MultipleLocator(0.2))
axs[0,1].axhline(y=0, color = 'dimgrey', ls = '--')
axs[0,1].axvline(x=r_window[0], color = 'darkseagreen', linewidth = 4,ls = '--')
# axs[0,1].scatter(r, dset_re, color = datacolor, s = 40, facecolors = 'none',linewidth = 3, label = dataLabel0)
axs[0,1].plot(r, dset_re, color = datacolor, linewidth = 6, label = dataLabel0)
axs[0,1].plot(r, fit_re, color = fitcolor, linewidth = 4.5, label = 'Fit')
axs[0,1].axvline(x=r_window[1], color = 'darkseagreen', linewidth = 4, ls = '--')
axs[0,1].legend(loc = 'best', fontsize=24)

axs[1,0].set_xlabel("k($\AA^{-1}$)", fontsize = 28)
axs[1,0].set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = 28)
axs[1,0].tick_params(axis='x', labelsize= 28)
axs[1,0].tick_params(axis='y', labelsize= 28)
axs[1,0].set_xlim([np.min(k_window)-1,np.max(k_window)+1])
axs[1,0].set_ylim([-10,15])
axs[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[1,0].xaxis.set_major_locator(MultipleLocator(1))
axs[1,0].xaxis.set_minor_locator(MultipleLocator(0.2))
axs[1,0].axhline(y=0, color = 'dimgrey', ls = '--')
axs[1,0].axvline(x=k_window[0], color = 'darkseagreen', linewidth = 4,ls = '--')
# axs[1,0].scatter(k, dset_ks, color = datacolor, s = 40, facecolors = 'none',linewidth = 3, label = dataLabel0)
axs[1,0].plot(k, dset_ks, color = datacolor, linewidth = 6, label = dataLabel0)
axs[1,0].plot(k, fit_ks, color = fitcolor, linewidth = 4.5, label = 'Fit')
axs[1,0].axvline(x=k_window[1], color = 'darkseagreen', linewidth = 4, ls = '--')
axs[1,0].legend(loc = 'best', fontsize=24)

axs[1,1].set_xlabel("R($\AA$)", fontsize = 28)
axs[1,1].set_ylabel("Im[\u03C7(R)]($\AA^{-4}$)", fontsize = 28)
axs[1,1].tick_params(axis='x', labelsize= 28)
axs[1,1].tick_params(axis='y', labelsize= 28)
axs[1,1].set_xlim([0,5])
# ax.set_ylim([-5,8])
axs[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[1,1].xaxis.set_major_locator(MultipleLocator(1))
axs[1,1].xaxis.set_minor_locator(MultipleLocator(0.2))
axs[1,1].axhline(y=0, color = 'dimgrey', ls = '--')
axs[1,1].axvline(x=r_window[0], color = 'darkseagreen', linewidth = 4,ls = '--')
# axs[1,1].scatter(r, dset_im, color = datacolor, s = 40, facecolors = 'none',linewidth = 3, label = dataLabel0)
axs[1,1].plot(r, dset_im, color = datacolor, linewidth = 6, label = dataLabel0)
axs[1,1].plot(r, fit_im, color = fitcolor, linewidth = 4.5, label = 'Fit')
axs[1,1].axvline(x=r_window[1], color = 'darkseagreen', linewidth = 4, ls = '--')
axs[1,1].legend(loc = 'best', fontsize=24)

plt.tight_layout()
plt.show()
#%% Plot duo figure XANES and EXAFS

fig, axs = plt.subplots(2,2,figsize=(24,20))

e = xanes['energy']
r01 = data_re['nu1kff_real.dat']['r']
r10 = data_re['nu1kffcl_real.dat']['r']
r11 = data_re['nu1kf_real.dat']['r']
dset_rm_01a = data_rm['nu1kff_rmag.dat']['NU1000FF_RT']
fit_rm_01a = data_rm['nu1kff_rmag.dat']['NU1000FF_RT_fit']
dset_rm_01b = data_rm['nu1kff_rmag.dat']['NU1000FF_125']
fit_rm_01b = data_rm['nu1kff_rmag.dat']['NU1000FF_125_fit']

dset_rm_10a = data_rm['nu1kffcl_rmag.dat']['NU1000FFCl']
fit_rm_10a = data_rm['nu1kffcl_rmag.dat']['NU1000FFCl_fit']
dset_rm_10b = data_rm['nu1kffcl_rmag.dat']['NU1000FFCl_125']
fit_rm_10b = data_rm['nu1kffcl_rmag.dat']['NU1000FFCl_125_fit']
dset_rm_10c = data_rm['rh_rmag.dat']['NU1kFF_Cl_RH50']
fit_rm_10c = data_rm['rh_rmag.dat']['NU1kFF_Cl_RH50_fit']


dset_rm_11a = data_rm['nu1kf_rmag.dat']['NU1000F_RT']
fit_rm_11a = data_rm['nu1kf_rmag.dat']['NU1000F_RT_fit']
dset_rm_11b = data_rm['nu1kf_rmag.dat']['NU1000F_125']
fit_rm_11b = data_rm['nu1kf_rmag.dat']['NU1000F_125_fit']

# dataLabel0 = "NU-1000-FF-Cl, 125$^\circ$C"
dataLabel0 = "NU-1000-FF-Cl, RT"
fitcolor = 'royalblue'
datacolor = 'darkgray'
fontsize1 = 28
fontsize2 = 24

axs[0,0].set_xlabel("Energy (keV)", fontsize = fontsize1)
axs[0,0].set_ylabel("Normalized x\u03bc(E)", fontsize = fontsize1)
axs[0,0].tick_params(axis='x', labelsize= fontsize1)
axs[0,0].tick_params(axis='y', labelsize= fontsize1)
axs[0,0].set_xlim([17.985, 18.130])
axs[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[0,0].set_prop_cycle('color', ['red',  'magenta','royalblue','darkred', 'darkmagenta',  'darkblue'])
# for column in xanes.columns[1:]:
#     if str(column) != 'NU1kFF_Cl_RH50':
#         axs[0,0].plot(e/1000, xanes[column], linewidth = 2.5, alpha = 0.7)
axs[0,0].plot(e/1000, xanes['NU1000FF_RT'], linewidth = 2.5, alpha = 0.7)
axs[0,0].plot(e/1000, xanes['NU1000F_RT'], linewidth = 2.5, alpha = 0.7)
axs[0,0].plot(e/1000, xanes['NU1000FFCl'], linewidth = 2.5, alpha = 0.7)
axs[0,0].plot(e/1000, xanes['NU1000FF_125'], linewidth = 2.5, alpha = 0.7)
axs[0,0].plot(e/1000, xanes['NU1000F_125'], linewidth = 2.5, alpha = 0.7)
axs[0,0].plot(e/1000, xanes['NU1000FFCl_125'], linewidth = 2.5, alpha = 0.7)
axs[0,0].plot(e/1000, xanes['NU1kFF_Cl_RH50'], linewidth = 2.5, alpha = 0.7)
axs[0,0].legend(labels = ['As-synthesized NU-1000', 'As-synthesized NU-1000-Formate', 'As-synthesized NU-1000-Cl', 'Dehydrated NU-1000', 'Dehydrated NU-1000-Formate','Dehydrated NU-1000-Cl', 'Dehydrated NU-1000-Cl Post 45% RH'], loc = 'lower right', fontsize = fontsize2)
axs[0,0].axhline(y = 0, ls = '--', color = 'dimgrey')

axs[0,1].set_xlabel("R($\AA$)", fontsize = fontsize1)
axs[0,1].set_ylabel("|\u03C7(R)|($\AA^{-3}$)", fontsize = fontsize1)
axs[0,1].tick_params(axis='x', labelsize= fontsize1)
axs[0,1].tick_params(axis='y', labelsize= fontsize1)
axs[0,1].set_xlim([0,5])
axs[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.set_ylim([-5,8])
axs[0,1].xaxis.set_major_locator(MultipleLocator(1))
axs[0,1].xaxis.set_minor_locator(MultipleLocator(0.2))
axs[0,1].axhline(y=0, color = 'dimgrey', ls = '--')
axs[0,1].axvline(x=1.21, color = 'darkseagreen', linewidth = 2.5,ls = '--')
axs[0,1].scatter(r, dset_rm_01a, color = 'red', s = 100, facecolors = 'none',linewidth = 3, label='As-synthesized NU-1000')
axs[0,1].scatter(r, dset_rm_01b, color = 'darkred', s = 100, marker = 'd', facecolors = 'none',linewidth = 3, label = 'Dehydrated NU-1000')
axs[0,1].plot(r, fit_rm_01a, color = 'black', linewidth = 2.5, alpha = 0.5)
axs[0,1].plot(r, fit_rm_01b, color = 'black', linewidth = 2.5, alpha = 0.5)
axs[0,1].axvline(x=3.62, color = 'darkseagreen', linewidth = 2.5, ls = '--')
axs[0,1].legend(loc = 'upper left', fontsize=fontsize2)

axs[1,0].set_xlabel("R($\AA$)", fontsize = fontsize1)
axs[1,0].set_ylabel("|\u03C7(R)|($\AA^{-3}$)", fontsize = fontsize1)
axs[1,0].tick_params(axis='x', labelsize= fontsize1)
axs[1,0].tick_params(axis='y', labelsize= fontsize1)
axs[1,0].set_xlim([0,5])
axs[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.set_ylim([-5,8])
axs[1,0].xaxis.set_major_locator(MultipleLocator(1))
axs[1,0].xaxis.set_minor_locator(MultipleLocator(0.2))
axs[1,0].axhline(y=0, color = 'dimgrey', ls = '--')
axs[1,0].axvline(x=1.21, color = 'darkseagreen', linewidth = 2.5,ls = '--')
axs[1,0].scatter(r, dset_rm_10a, color = 'magenta', s = 100, facecolors = 'none',linewidth = 3, label = 'As-synthesized NU-1000-Cl')
axs[1,0].scatter(r, dset_rm_10b, color = 'darkmagenta', s = 100, marker = 'd',facecolors = 'none',linewidth = 3, label = 'Dehydrated NU-1000-Cl')
# axs[1,0].scatter(r, dset_rm_10c, color = 'darkorchid', s = 100, marker = '^',facecolors = 'none',linewidth = 3, label = 'Dehydrated NU-1000-Cl Post 45% RH')
axs[1,0].plot(r, fit_rm_10a, color = 'black', linewidth = 2.5, alpha = 0.5)
axs[1,0].plot(r, fit_rm_10b, color = 'black', linewidth = 2.5, alpha = 0.5)
# axs[1,0].plot(r, fit_rm_10c, color = 'black', linewidth = 2.5, alpha = 0.5)
axs[1,0].axvline(x=3.62, color = 'darkseagreen', linewidth = 2.5, ls = '--')
axs[1,0].legend(loc = 'upper left', fontsize=fontsize2)

axs[1,1].set_xlabel("R($\AA$)", fontsize = fontsize1)
axs[1,1].set_ylabel("|\u03C7(R)|($\AA^{-3}$)", fontsize = fontsize1)
axs[1,1].tick_params(axis='x', labelsize= fontsize1)
axs[1,1].tick_params(axis='y', labelsize= fontsize1)
axs[1,1].set_xlim([0,5])
axs[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.set_ylim([-5,8])
axs[1,1].xaxis.set_major_locator(MultipleLocator(1))
axs[1,1].xaxis.set_minor_locator(MultipleLocator(0.2))
axs[1,1].axhline(y=0, color = 'dimgrey', ls = '--')
axs[1,1].axvline(x=1.21, color = 'darkseagreen', linewidth = 2.5,ls = '--')
axs[1,1].scatter(r, dset_rm_11a, color = 'royalblue', s = 100, facecolors = 'none',linewidth = 3, label = 'As-synthesized NU-1000-Formate')
axs[1,1].scatter(r, dset_rm_11b, color = 'darkblue', s = 100, marker = 'd',facecolors = 'none',linewidth = 3, label = 'Dehydrated NU-1000-Formate')
axs[1,1].plot(r, fit_rm_11a, color = 'black', linewidth = 2.5, alpha = 0.5)
axs[1,1].plot(r, fit_rm_11b, color = 'black', linewidth = 2.5, alpha = 0.5)
axs[1,1].axvline(x=3.62, color = 'darkseagreen', linewidth = 2.5, ls = '--')
axs[1,1].legend(loc = 'upper left', fontsize=fontsize2)

#%% Plot Normalized XANES and LCF Results Next To It
fig = plt.figure(figsize = (24,12))
gs = fig.add_gridspec(5, 2)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[:4, 1])
ax3 = fig.add_subplot(gs[4, 1], sharex = ax2)
# ax.set_xlabel("Energy (keV)", fontsize = 32)
# ax.set_ylabel("d/dx\u03bc(E)", fontsize = 32)
ax1.set_ylabel("Normalized x\u03bc(E)", fontsize = 36)
ax1.set_xlabel("Energy (keV)", fontsize = 36)
ax1.tick_params(axis='x', labelsize= 32)
ax1.tick_params(axis='y', labelsize= 32)
ax1.axhline(y = 0, ls = '--', color = 'dimgrey')
ax1.set_xlim([8.32, 8.38])

ax1.set_prop_cycle('color', ['gray', 'black', 'purple', 'red', 'green'])
ax1.plot(xanes['energy']/1000, xanes['_Ref_Ni2P_TM'],  linewidth = 4, linestyle = '--',  label = 'Ni Foil')
ax1.plot(xanes['energy']/1000, xanes['NiO_TM'], linewidth = 4, linestyle = '-',  label = 'NiO')
ax1.plot(xanes['energy']/1000, xanes['Ni2P_TM'], linewidth = 4,  linestyle = '-',label = 'Ni$_2$P')
ax1.plot(xanes['energy']/1000, sgf(xanes['NiAIM'],9,3), linewidth = 4, label = 'Ni-AIM')
ax1.plot(xanes['energy']/1000, sgf(xanes['NiP-NU1k'],9,3), linewidth = 4, label = 'NU-1000-Ni$_x$P$_y$')
ax1.legend(loc = 'lower right',  fontsize = 36)

ax2.set_ylabel("Normalized x\u03bc(E)", fontsize = 36)
ax2.set_xlabel("Energy (keV)", fontsize = 32)
ax2.tick_params(axis='x', labelsize= 28)
ax2.tick_params(axis='y', labelsize= 28)
ax2.axhline(y = 0, ls = '--', color = 'dimgrey')
ax2.set_xlim([8.315, 8.45])

ax2.set_prop_cycle('color', ['black', 'purple', 'red', 'blue'])
ax2.plot(lcf['energy']/1000, sgf(lcf['NiO_TM'],9,3), linewidth = 4, label = 'NiO')
ax2.plot(lcf['energy']/1000, lcf['NiAIM'], linewidth = 4, label = 'Ni$_2$P')
ax2.plot(lcf['energy']/1000, lcf['Ni2P_TM'], linewidth = 4, label = 'Ni-AIM')
ax2.scatter(lcf['energy']/1000, sgf(lcf['data'],9,3), linewidth = 2, s = 200, edgecolor = 'green', facecolor = 'none', label = 'NU-1000-Ni$_x$P$_y$')
ax2.plot(lcf['energy']/1000, sgf(lcf['fit'],9,3), linewidth = 4, label = 'NU-1000-Ni$_x$P$_y$ Fit')
ax2.legend(loc = 'upper right',  fontsize = 28, ncol = 1)

ax3.set_ylabel("$\Delta\u03bc(E)$", fontsize = 32)
ax3.set_xlabel("Energy (keV)", fontsize = 32)
ax3.tick_params(axis='x', labelsize= 28)
ax3.tick_params(axis='y', labelsize= 28)
ax3.axhline(y = 0, ls = '--', color = 'dimgrey')
ax3.set_xlim([8.315, 8.45])

ax3.plot(lcf['energy']/1000, sgf(lcf['data']-lcf['fit'],9,3), linewidth = 4)

# ax.axvline(x=4.987, color = 'black', linewidth = 2.5,ls = ':')
# ax.axvline(x=4.992, color = 'black', linewidth = 2.5,ls = ':')
# ax.axvline(x=5.004, color = 'black', linewidth = 2.5,ls = ':')
# ax.plot(xanes['energy']/1000, xanes['TiFoil_Calibred'],  linewidth = 4, color='gray', linestyle = '--', alpha = 0.25, label = 'Ti Foil')
# ax.plot(xanes['energy']/1000, xanes['Ti_R1_001.dat'], linewidth = 4,  color = 'crimson', linestyle = '-.', alpha = 0.25, label = 'Anatase')
plt.tight_layout()

#%%
from collections import OrderedDict
norDict = OrderedDict()
for file in os.listdir():
    with open(file):
        if str(file).endswith('.nor'):
            norDict['headers'] = open(file).readlines()[37]
            norDict[file] = np.loadtxt(file, dtype = float, skiprows=38)
eNotValues = [18056.8153426812, 18054.620555345, 16289.7705780986, 16290.1871063271, 16290.4349782758, 16290.291464637]
eNotDict = OrderedDict(zip(list(norDict.keys())[1:], eNotValues))
labels = ['PuO$_2$', 'PuPhos', 'ThO$_2$ (cr)', 'ThO$_2$ (2.5nm)', 'ThPhos (pH 4.8)', 'ThPhos (pH 7.5)']
labelDict = OrderedDict(zip(list(norDict.keys())[1:], labels))
colors = ['black', 'deepskyblue', 'brown', 'salmon', 'crimson', 'orange']
colorDict = OrderedDict(zip(list(norDict.keys())[1:], colors))
fig, ax1 = plt.subplots(figsize = (12,8))
fs1 = 30
fs2 = 24

ax1.set_ylabel("Normalized x\u03bc(E)", fontsize = fs1)
ax1.set_xlabel("E - E$_0$ (keV)", fontsize = fs1)
ax1.tick_params(axis='x', labelsize= fs2)
ax1.tick_params(axis='y', labelsize= fs2)
ax1.axhline(y = 0, ls = '--', color = 'dimgrey')
ax1.set_xlim([-0.2, 0.2])

plotList = (list(eNotDict.keys())[0],list(eNotDict.keys())[2],list(eNotDict.keys())[3])
# plotList = list(eNotDict.keys())[2:]
# ax1.set_prop_cycle('color', ['black', 'deepskyblue', 'brown', 'salmon', 'crimson', 'orange'])
for key in plotList:
    ax1.plot((norDict[key][:,0]-eNotDict[key])/1000, norDict[key][:,3], color = colorDict[key], linewidth = 4, linestyle = '-', alpha = 0.75, label = labelDict[key])
ax1.legend(loc = 'upper left',  fontsize = fs2)

plt.tight_layout()


#%% 26072024
data_im = {}
data_re = {}
data_rm = {}
data_ks = {}

for file in os.listdir():
    if file.endswith(".dat"):
        with open(file) as current_file:
            if file.endswith("imag.dat"):
                data_im[file]=ReadSingleExafs_Real(file)
            if file.endswith("real.dat"):
                data_re[file]=ReadSingleExafs_Real(file)
            if file.endswith("rmag.dat"):
                data_rm[file]=ReadSingleExafs_R(file)
            if file.endswith("kspa.dat"):
                data_ks[file]=ReadSingleExafs_k(file)


fig, axs = plt.subplots(3,2,figsize=(30,27))

datadict_callers = ['2421_NaMetal', '2421_MgMetal', '2421_NpMetal']
dataset_caller = "SA_2421_NpMg_Fl"
#['Fit_to_'+dataset_caller+'_from_Fit_122']
irange = [0,1,2]
labels = ['2421 (+Na)', '2421 (+Mg)', '2421 (+Np)']

for datadict_caller, i in zip(datadict_callers, irange):
    r = data_re[datadict_caller+"_real.dat"]['r']
    k = data_ks[datadict_caller+'_kspa.dat']['k']
    dset_re = data_re[datadict_caller+'_real.dat'][dataset_caller]
    fit_re = data_re[datadict_caller+'_real.dat'][dataset_caller+'_fit']
    dset_im = data_im[datadict_caller+'_imag.dat'][dataset_caller]
    fit_im = data_im[datadict_caller+'_imag.dat'][dataset_caller+'_fit']
    dset_rm = data_rm[datadict_caller+'_rmag.dat'][dataset_caller]
    fit_rm = data_rm[datadict_caller+'_rmag.dat'][dataset_caller+'_fit']
    dset_ks = data_ks[datadict_caller+'_kspa.dat'][dataset_caller]
    fit_ks = data_ks[datadict_caller+'_kspa.dat'][dataset_caller+'_fit']
    k_window = (3.0, 12)
    r_window = (1, 4.69)

    dataLabel0 = "Mg$_{0.5}$NpO$_2$CO$_3$"
    # dataLabel0 = "1nm TiO$_2$ on ZrO$_2$, powder"
    # fitcolor = 'firebrick'
    fitcolor = 'crimson'
    datacolor = 'darkgray'

    axs[i,0].set_xlabel("R($\AA$)", fontsize = 28)
    axs[i,0].set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = 28)
    axs[i,0].tick_params(axis='x', labelsize= 28)
    axs[i,0].tick_params(axis='y', labelsize= 28)
    axs[i,0].set_xlim([0,5])
    # ax.set_ylim([-5,8])
    axs[i,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[i,0].xaxis.set_major_locator(MultipleLocator(1))
    axs[i,0].xaxis.set_minor_locator(MultipleLocator(0.2))
    axs[i,0].axhline(y=0, color = 'dimgrey', ls = '--')
    axs[i,0].axvline(x=r_window[0], color = 'darkseagreen', linewidth = 4,ls = '--')
    # axs[i,0].scatter(r, dset_rm, color = datacolor, s = 40, facecolors = 'none',linewidth = 3, label = dataLabel0)
    # axs[i,0].plot(r, dset_re, color = datacolor, linewidth = 4)
    axs[i,0].plot(r, fit_re, color = 'indigo', linestyle = '--',linewidth = 3, label = 'Fit (Re(FT))')
    # axs[i,0].plot(r, dset_im, color = datacolor, linewidth = 4)
    axs[i,0].plot(r, fit_im, color = 'orange', linestyle = '--',linewidth = 3, label = 'Fit (Im(FT))')
    axs[i,0].plot(r, dset_rm, color = datacolor, linewidth = 6.5, label = labels[i])
    axs[i,0].plot(r, fit_rm, color = fitcolor, linewidth = 5, label = 'Fit (|FT|)')
    axs[i,0].axvline(x=r_window[1], color = 'darkseagreen', linewidth = 4, ls = '--')
    axs[i,0].legend(loc = 'upper right', fontsize=28)
    
    axs[i,1].set_xlabel("k($\AA^{-1}$)", fontsize = 28)
    axs[i,1].set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = 28)
    axs[i,1].tick_params(axis='x', labelsize= 28)
    axs[i,1].tick_params(axis='y', labelsize= 28)
    axs[i,1].set_xlim([np.min(k),15])
    axs[i,1].set_ylim([-10,15])
    axs[i,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[i,1].xaxis.set_major_locator(MultipleLocator(1))
    axs[i,1].xaxis.set_minor_locator(MultipleLocator(0.2))
    axs[i,1].axhline(y=0, color = 'dimgrey', ls = '--')
    axs[i,1].axvline(x=k_window[0], color = 'darkseagreen', linewidth = 4,ls = '--')
    # axs[i,1].scatter(k, dset_ks, color = datacolor, s = 40, facecolors = 'none',linewidth = 3, label = dataLabel0)
    axs[i,1].plot(k, dset_ks, color = datacolor, linewidth = 6.5, label = labels[i])
    axs[i,1].plot(k, fit_ks, color = fitcolor, linewidth = 5, label = 'Fit')
    axs[i,1].axvline(x=k_window[1], color = 'darkseagreen', linewidth = 4, ls = '--')
    axs[i,1].legend(loc = 'upper right', fontsize=28)

plt.tight_layout()