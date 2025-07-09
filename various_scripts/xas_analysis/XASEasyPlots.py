# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:04:07 2024

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
from collections import OrderedDict
#%% Import the data
chirDict = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
    # skip directories
        continue
    with open(file):
        if str(file).endswith('.chir'):
            chirDict['headers'] = open(file).readlines()[37]
            chirDict[file] = np.loadtxt(file, dtype = float, skiprows=38)

# grfDict = OrderedDict()
# for file in os.listdir():
#     with open(file):
#         if str(file).endswith('.grf'):
#             grfDict[file] = np.loadtxt(file, dtype = float, skiprows=38)
# grfDict['headers'] = ['k', 'chi']

chikDict = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
    # skip directories
        continue
    with open(file):
        if str(file).endswith('.chik'):
            chikDict['headers'] = open(file).readlines()[37]
            chikDict[file] = np.loadtxt(file, dtype = float, skiprows=38)

norDict = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
    # skip directories
        continue
    with open(file):
        if str(file).endswith('.nor'):
            norDict['headers'] = open(file).readlines()[37]
            norDict[file] = np.loadtxt(file, dtype = float, skiprows=38)
            
chiqDict = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
    # skip directories
        continue
    with open(file):
        if str(file).endswith('.chiq'):
            chiqDict['headers'] = open(file).readlines()[37]
            chiqDict[file] = np.loadtxt(file, dtype = float, skiprows=38)

#%%
def normalize(data, factor=1):
    """Normalizes a vector to it's maximum value"""
    normalized_data = factor*data/max(data)
    return normalized_data

def pdfXafsOverlaid(pdf_dset, xafs_dset, phaseShift=0.5, rLim=10):
    fig, ax = plt.subplots(figsize = (8,8))
    pdf_x, pdf_y = pdf_dset[:,0], pdf_dset[:,1]
    xafs_x, xafs_yM, xafs_yR = xafs_dset[:,0], xafs_dset[:,3], xafs_dset[:,1]
    ax.plot(pdf_x, normalize(pdf_y, 1), color = 'red', alpha = 0.8, linewidth = 3, label = 'PDF')
    ax.plot(xafs_x+phaseShift, normalize(xafs_yM, 1), color = 'blue', alpha = 0.5,linewidth = 3,  label = 'EXAFS FT magnitude')
    ax.plot(xafs_x+phaseShift, normalize(xafs_yR, 1), color = 'blue', linewidth = 2,linestyle='--',  label = 'EXAFS FT real')
    ax.set_xlim([0, rLim])
    ax.set_yticks([])
    ax.tick_params(axis = 'x', labelsize = 24)
    ax.set_xlabel('R', fontsize = 24)
    ax.legend(loc = 'lower right', fontsize = 20)
    plt.tight_layout()

# pdf = "pdf_thphos75.grf"
# xafs = 'ThPhos_pH75_MEE_.chir'
# pdfXafsOverlaid(grfDict[pdf], chirDict[xafs], 0.13)
#%% 
def quadPlot(nordict, norkeys, chikdict, chikkeys, chirdict, chirkeys, stack_gap=[0,0,0,0]):
    nordata = [norDict[key] for key in norkeys]
    chikdata = [chikDict[key] for key in chikkeys]
    chirdata = [chirDict[key] for key in chirkeys]
    fs1 = 18
    fs2 = 16
    fig, ax = plt.subplots(2,2,figsize = (16,9))
    axflat = ax.flatten()
    for axis in axflat:
        axis.tick_params('both', labelsize=fs2)
    for i in range(4):
        if stack_gap[i] != 0:
            axflat[i].set_yticks([])
    count = 0
    axflat[0].set_xlabel('energy, keV', fontsize = fs1)
    axflat[0].set_ylabel("flat $x\u03bc$(E)", fontsize = fs1)
    for norfile in nordata:
        nor_x, nor_y = norfile[:,0]/1000, norfile[:,3]
        axflat[0].plot(nor_x, nor_y+np.max(nor_y)*stack_gap[0]*count, linewidth = 3)
        if count == 0:  
            edgeTip = np.argmax(nor_y)
            axflat[0].set_xlim(nor_x[edgeTip]-75/1000, nor_x[edgeTip]+75/1000)
        count+=1
    axflat[0].legend(labels = [str(key)[:-4] for key in norkeys], loc = 'upper left', fontsize = fs2)
    count = 0
    axflat[1].set_xlabel('energy, keV', fontsize = fs1)
    axflat[1].set_ylabel("flat $x\u03bc$(E)", fontsize = fs1)
    for norfile in nordata:
        nor_x, nor_y = norfile[:,0]/1000, norfile[:,3]
        axflat[1].plot(nor_x, nor_y+np.max(nor_y)*stack_gap[1]*count, linewidth = 3)
        if count == 0:  
            edgeTip = np.argmax(nor_y)
            axflat[1].set_xlim(nor_x[edgeTip]+75/1000, np.max(nor_x))
        count+=1
        axflat[1].set_ylim(1-stack_gap[1], 1+np.max(nor_y)*stack_gap[1]*count)
    count = 0
    axflat[2].set_xlabel("k($\AA^{-1}$)", fontsize = fs1)
    axflat[2].set_ylabel("k$^{3}$\u03C7(k)($\AA^{-4}$)", fontsize = fs1)
    for chikfile in chikdata:
        chik_x, chik_y = chikfile[:,0], chikfile[:,3]
        axflat[2].plot(chik_x, normalize(chik_y)+1*stack_gap[2]*count, linewidth = 3)
        count+=1
    # axflat[2].legend(labels = [str(key)[:-5] for key in chikkeys], loc = 'lower right', fontsize = fs2)
    count = 0
    axflat[3].set_xlim(-0.1,5.1)
    axflat[3].set_ylabel("|\u03C7(R)|($\AA^{-3}$)", fontsize = fs1)
    axflat[3].set_xlabel("R($\AA$)", fontsize = fs1)
    for chirfile in chirdata:
        chir_x, chir_y = chirfile[:,0], chirfile[:,3]
        axflat[3].plot(chir_x, normalize(chir_y)+1*stack_gap[3]*count, linewidth = 3)
        count+=1

quadPlot(norDict, ['NpBlue_Ca.nor', 'NpCO3Np10Ca.nor', 'NpFreshMg.nor', 'NpFreshMn.nor'], 
          chikDict,['NpBlue_Ca.chik', 'NpCO3Np10Ca.chik','NpFreshMg.chik', 'NpFreshMn.chik'], 
          chirDict, ['NpBlue_Ca.chir', 'NpCO3Np10Ca.chir','NpFreshMg.chir', 'NpFreshMn.chir'],
          [0.2, 0.05, 1, 0.5])

#%%
def kStackPlot(chikdict, chikkeys, stack_gap = 0):
    chikdata = [chikDict[key] for key in chikkeys]
    fs1 = 18
    fs2 = 16
    fig, ax = plt.subplots(figsize = (9,16))
    count = 0
    ax.set_xlim(-0.1,5.1)
    ax.set_xlabel("k($\AA^{-1}$)", fontsize = fs1)
    ax.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-4}$)", fontsize = fs1)
    ax.tick_params('both', labelsize=fs2)
    for chikfile in chikdata:
        chik_x, chik_y = chikfile[:,0], chikfile[:,3]
        ax.plot(chik_x, chik_y+1*stack_gap*count, linewidth = 3)
        count+=1
    ax.set_ylim(-2, 1+np.max(chik_y)*stack_gap*count)
    ax.legend(labels = [str(key)[:-5] for key in chikkeys], loc = 'lower center', ncol = 2, fontsize = fs2)
    
kStackPlot(chikDict, ['Th01_pH10_NH3_.chik', 'Th01_pH4_NH3_MEE_.chik', 'Th01_pH6_NH3_MEE_.chik', 'Th01_pH8_NH3_MEE_.chik', 'ThO2-1000C.chik', 'ThO2-3MNaOH.chik'], 1)

#%%
def RStackPlot(chirdict, chirkeys, stack_gap = 0):
    chirdata = [chirDict[key] for key in chirkeys]
    fs1 = 24
    fs2 = 18
    fig, ax = plt.subplots(figsize = (12,12))
    count = 0
    ax.set_ylabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fs1)
    ax.set_xlabel("R($\AA$)", fontsize = fs1)
    ax.tick_params('both', labelsize=fs2)
    ax.set_yticks([])
    colors = ['cyan','magenta','orange','gray']
    # ax.set_prop_cycle('color', ['black', 'red', 'darkorange', 'darkmagenta'])
    for color, chirfile in zip(colors, chirdata):
        chir_x, chir_y = chirfile[:,0], chirfile[:,3]
        # ax.plot(chir_x, normalize(chir_y)+len(chirdata)*stack_gap-stack_gap*(count+1), linewidth = 3)
        ax.scatter(chir_x, chir_y+len(chirdata)*stack_gap-stack_gap*(count+1), s = 100, facecolor = color, edgecolor = 'black', alpha = 1, linewidth = 1.5)
        count+=1
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 5)
    ax.legend(labels = chirkeys, loc = 'upper right', ncol = 1, fontsize = fs1)
    
RStackPlot(chirDict, 
           ['Th01_pH4_NH3.chir', 'Th01_pH6_NH3.chir', 'Th01_pH8_NH3.chir', 'Th01_pH10_NH3.chir', 'ThO2-1000C.chir'],
           0)

#%%
def muStackPlot(nordict, norkeys, stack_gap = 0):
    nordata = [norDict[key] for key in norkeys]
    fs1 = 18
    fs2 = 16
    fig, ax = plt.subplots(1,2,figsize = (18,16))
    axflat = ax.flatten()
    count = 0
    axflat[0].set_xlabel('energy, keV', fontsize = fs1)
    axflat[0].set_ylabel("flat $x\u03bc$(E)", fontsize = fs1)
    axflat[0].tick_params('both', labelsize=fs2)
    for norfile in nordata:
        nor_x, nor_y = norfile[:,0]/1000, norfile[:,3]
        axflat[0].plot(nor_x, nor_y+1*stack_gap*count, linewidth = 3)
        if count == 0:  
            edgeTip = np.argmax(nor_y)
            axflat[0].set_xlim(nor_x[edgeTip]-75/1000, nor_x[edgeTip]+75/1000)
        count+=1
    axflat[0].set_ylim(-0.75, 1+np.max(nor_y)*stack_gap*(count))
    axflat[0].legend(labels = [str(key)[:-5] for key in norkeys], loc = 'lower center', ncol = 2, fontsize = fs2)
    
muStackPlot(norDict, 
           ['Th01_pH10_NH3_MEE_.nor', 'Th01_pH4_NH3_MEE_.nor', 'Th01_pH6_NH3_MEE_.nor', 'Th01_pH8_NH3_MEE_.nor', 'ThO2-1000C.nor', 'ThO2-3MNaOH.nor'],
           0.3)

#%% pretty xanes
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