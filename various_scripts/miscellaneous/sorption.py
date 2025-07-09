# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:10:02 2020

@author: boris
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FixedLocator)
import os
from numpy import genfromtxt
from scipy.stats import linregress as linreg
#%%
def ReadData(file):
    with open(file) as current_file:
        d = pd.read_csv(current_file, sep = ',', names = ['x', 'y'], engine = 'python')
        return d
def ReadAllData():
    data_files = os.listdir()
    data = {}
    for file in data_files:
        if file.endswith(".csv"):
            data[file] = ReadData(file)
            data[file] = data[file][data[file].applymap(CheckFloat).all(axis=1)].astype(float)
    return data
def CheckFloat(s):
    """makes sure your data is a matrix of floats"""
    try: 
        float(s)
        return True
    except ValueError:
        return False
#%%
AllData = ReadAllData()
isotherma = {}
pores = {}
for file in AllData.keys():
    if 'isotherm' in str(file):
        isotherma[file] = AllData[file]
    else:
        pores[file] = AllData[file]
del file
#%%
def StackIsothermaPlots(samples, samplecolors, labels):
    fig1, ax1 = plt.subplots(figsize=(10,8))
    ax1.set_xlabel("Relative Pressure, P/P$_{0}$", fontsize = 32)
    ax1.set_ylabel("Quantity Adsorbed, cm$^{3}$/g", fontsize = 32)
    # for i in range(int(max(reference['y'])//200)):
    #     ax1.axhline(y=200*i, color = 'lightgrey', ls = '--')
    ax1.tick_params(axis='x', labelsize= 24)
    ax1.tick_params(axis='y', labelsize= 24)
    i = 0
    for data in samples:
        xa = data['x'].iloc[0:data['x'].idxmax()]
        ya = data['y'].iloc[0:data['x'].idxmax()]
        xd = data['x'].iloc[data['x'].idxmax():-1]
        yd = data['y'].iloc[data['x'].idxmax():-1]
        if labels[i] =='NU-1000-F$_4$':
            print('yes')
            ya *= 20
            yd *= 20
        ax1.scatter(xd,yd,s=75,marker='o',facecolors = 'none', edgecolors = samplecolors[i], linewidth = 1)
        ax1.scatter(xa,ya,s=75,marker='o',facecolors = samplecolors[i], label = labels[i])
        i += 1
    
    
    # for i in range(len(samples)):
    #     f = samples[i]
    #     ax1.scatter(f['x'][1:], f['y'][1:], s=60, marker='o', c = samplecolors[i])
    #     ax1.plot(f['x'][1:], f['y'][1:], linewidth = 3, color = samplecolors[i], label = labels[i])
    # ax1.scatter(reference['x'][1:], reference['y'][1:], s=60, marker = 'd', c = 'black')
    # ax1.plot(reference['x'][1:], reference['y'][1:], linewidth = 3, color = 'black', label = labels[-1])
    # ax1.set_ylim([-50, max(reference['y'])+max(reference['y'])*0.1])
    ax1.set_xlim([-0.05,1.05])
    ax1.legend(fontsize = 20, loc = 'lower right')
    
def StackPorePlots(samples, samplecolors, labels):
    fig1, ax1 = plt.subplots(figsize=(10,8))
    ax1.set_xlabel(r"Pore Width, $\AA$", fontsize = 32)
    ax1.set_ylabel("dV/dlogW", fontsize = 32)
    ax1.tick_params(axis='x', labelsize= 24)
    ax1.tick_params(axis='y', labelsize= 24)
    for i in range(len(samples)):
        f = samples[i]
        ax1.scatter(f['x'][1:], f['y'][1:], s=60, marker='o', c = samplecolors[i])
        ax1.plot(f['x'][1:], f['y'][1:], linewidth = 3, color = samplecolors[i], label = labels[i])
    # ax1.scatter(reference['x'][1:], reference['y'][1:], s=60, marker = 'd', c = 'black')
    # ax1.plot(reference['x'][1:], reference['y'][1:], linewidth = 3, color = 'black', label = labels[-1])
    # ax1.set_ylim([0, max(reference['y'])+max(reference['y'])*0.1])
    ax1.set_xlim([5, 40])
    ax1.legend(fontsize = 20, loc = 'upper left')

# StackIsothermaPlots([isotherma['isotherm1k.csv'], isotherma['isotherm1kF.csv'], isotherma['isotherm_f4.csv'],isotherma['isotherm601.csv']], ['purple', 'magenta', 'orange', 'indigo'], ['NU-1000', 'NU-1000-Formate', 'NU-1000-F$_4$', 'NU-601'])
# StackPorePlots([pores['pores1k.csv'], pores['pores1kF.csv'], pores['pores_f4.csv'],pores['pores601.csv']], 
#                ['purple', 'magenta', 'orange', 'indigo'], 
#                ['NU-1000', 'NU-1000-Formate', 'NU-1000-F$_4$', 'NU-601'])
#%%
def plot1isotherm(data, save):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_xlabel("Relative Pressure, P/P$_{0}$", fontsize = 20)
    ax.set_ylabel("Quantity Adsorbed, cm$^{3}$/g", fontsize = 20)
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
    xa = data['x'].iloc[0:data['x'].idxmax()]
    ya = data['y'].iloc[0:data['x'].idxmax()]
    xd = data['x'].iloc[data['x'].idxmax():-1]
    yd = data['y'].iloc[data['x'].idxmax():-1]
    ax.scatter(xd,yd,s=75,marker='o',facecolors = 'none', edgecolors = 'royalblue', linewidth = 1, label = 'Desorption')
    ax.scatter(xa,ya,s=75,marker='o',facecolors = 'dodgerblue', label = 'Adsorption')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc = 'lower right', fontsize = 18)
    if save ==1:
        plt.savefig('singleisotherm.png')
def plot1poredist(data, save):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_xlabel(r"Pore Width, $\AA$", fontsize = 20)
    ax.set_ylabel("dV/dlog(W)", fontsize = 20)
    ax.tick_params(axis='x', labelsize= 18)
    ax.tick_params(axis='y', labelsize= 18)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    x = data['x']
    y = data['y']
    ax.set_xlim([min(x),40])
    ax.plot(x,y,linewidth=4,color='royalblue')
    if save ==1:
        plt.savefig('singleporedist.png')
        
#%%
plot1isotherm(AllData['isotherm145.csv'], 0)
# plot1poredist(AllData['pores145.csv'], 0)


#%%
fig, ax = plt.subplots(figsize = (8,6))
x = [0, 4, 22]
y = [0.30, 0.30 + 6.91 + 5.88, 0.30 + 6.91 + 38.8]
colors = ['green', 'orange', 'red']
for i in range(3):
    ax.scatter(x[i],y[i],s = 300,color = colors[i])
ax.set_xlim(-2,24)
xmajorlocator = FixedLocator(x)
ymajorlocator = FixedLocator(y)
ax.xaxis.set_major_locator(xmajorlocator)
ax.yaxis.set_major_locator(ymajorlocator)

ax.tick_params(labelsize = 24)

ax.set_ylabel('HOMO - HOMO Gap (eV)', fontsize = 24)
ax.set_xlabel('# Protons Removed', fontsize = 24)

plt.tight_layout()

#%%
data = np.genfromtxt('isotherma.csv', delimiter = ';')

fig, ax = plt.subplots(figsize = (12,9))
ax.set_prop_cycle('color', [plt.cm.winter(i) for i in np.linspace(0,1,3)])
cmap = matplotlib.cm.get_cmap('winter')
ax.plot(data[:,0][41:61], data[:,1][41:61], marker = 'o', ms = 10, linewidth = 3)
# ax.scatter(data[:,0][61:], data[:,1][61:], marker = 'o',facecolor='none', edgecolor=cmap(0.33),s = 100, linewidth = 3)
ax.plot(data[:,2][40:59], data[:,3][40:59], marker = 'o', ms = 10, linewidth = 3)
# ax.scatter(data[:,2][61:], data[:,3][61:], marker = 'o',facecolor='none', edgecolor=cmap(0.66),s = 100, linewidth = 3)
ax.plot(data[:,4][30:49], data[:,5][30:49], marker = 'o', ms = 10, linewidth = 3)
# ax.scatter(data[:,4][61:], data[:,5][61:], marker = 'o',facecolor='none', edgecolor=cmap(0.99),s = 100, linewidth = 3)
# ax.plot(data[:,6], data[:,7], marker = 'o', linewidth = 3)
# ax.plot(data[:,8], data[:,9], marker = 'o', linewidth = 3)
# ax.plot(data[:,10], data[:,11], marker = 'o', linewidth = 3)
# ax.plot(data[:,12], data[:,13], marker = 'o', linewidth = 3)
# ax.plot(data[:,14], data[:,15], marker = 'o', linewidth = 3)

ax.tick_params(labelsize = 24)

ax.set_ylabel('Quantity Adsorbed (mmol/g)', fontsize = 24)
ax.set_xlabel('Absolute Pressure (mbar)', fontsize = 24)

ax.legend(loc = 'lower right', ncol = 1, labels = ['Stirred 1-3 days', 'Stirred 4-7 days', 'Stirred 10-13 days'], fontsize = 22)

plt.tight_layout()

#%% t-plot



def HarkinsJuraEqn(Pterm):
    t = np.sqrt(13.99/(0.034 - np.log(Pterm)))
    return t

def mmolToVol(data):
    return data*22.4

data = np.genfromtxt('isotherma.csv', delimiter = ';')

fig, ax = plt.subplots(1,2,figsize = (18,9),sharey=True)
ax[0].set_prop_cycle('color', [plt.cm.winter(i) for i in np.linspace(0,1,3)])
ax[1].set_prop_cycle('color', [plt.cm.winter(i) for i in np.linspace(0,1,3)])
ax[0].plot(data[:,0][41:61], mmolToVol(data[:,1][41:61]), marker = 'o', ms = 20, linewidth = 3)
ax[0].plot(data[:,2][40:59], mmolToVol(data[:,3][40:59]), marker = 'o', ms = 20, linewidth = 3)
ax[0].plot(data[:,4][30:49], mmolToVol(data[:,5][30:49]), marker = 'o', ms = 20, linewidth = 3)
ax[1].plot(HarkinsJuraEqn(data[:,0][41:61]), mmolToVol(data[:,1][41:61]), marker = 'o', alpha = 0.5,  ms = 20, linewidth = 3)
ax[1].plot(HarkinsJuraEqn(data[:,2][40:59]), mmolToVol(data[:,3][40:59]), marker = 'o', alpha = 0.5,ms = 20, linewidth = 3)
ax[1].plot(HarkinsJuraEqn(data[:,4][30:49]), mmolToVol(data[:,5][30:49]), marker = 'o', alpha = 0.5,ms = 20, linewidth = 3)

ax[0].set_ylim((-1000, 18000))
ax[0].tick_params(labelsize = 24)
ax[1].tick_params(labelsize = 24)
ax[0].set_ylabel('Quantity Adsorbed (mL STP/g)', fontsize = 24)
ax[0].set_xlabel('Absolute Pressure (mbar)', fontsize = 24)
ax[1].set_xlabel('t($\AA$)', fontsize = 24)

a = linreg(HarkinsJuraEqn(data[:,0][52:61]), mmolToVol(data[:,1][52:61]))
ax[1].plot(HarkinsJuraEqn(data[:,0][41:61]), HarkinsJuraEqn(data[:,0][41:61])*a[0]+a[1], color='red', linewidth = 2.5)
ax[1].annotate(f"Slope: {a[0]:.1f}, y$_0$: {a[1]:.0f}, R$^2$: {a[2]:.3f}", (6, 17000), fontsize=20)

b = linreg(HarkinsJuraEqn(data[:,2][49:59]), mmolToVol(data[:,3][49:59]))
ax[1].plot(HarkinsJuraEqn(data[:,2][40:59]), HarkinsJuraEqn(data[:,2][40:59])*b[0]+b[1], color='red', linewidth = 2.5)
ax[1].annotate(f"Slope: {b[0]:.1f}, y$_0$: {b[1]:.0f}, R$^2$: {b[2]:.3f}", (6, 15000), fontsize=20)

c = linreg(HarkinsJuraEqn(data[:,4][31:49]), mmolToVol(data[:,5][31:49]))
ax[1].plot(HarkinsJuraEqn(data[:,4][30:49]), HarkinsJuraEqn(data[:,4][30:49])*c[0]+c[1], color='red', linewidth = 2.5)
ax[1].annotate(f"Slope: {c[0]:.1f}, y$_0$: {c[1]:.0f}, R$^2$: {c[2]:.3f}", (6, 5000), fontsize=20)

ax[0].legend(loc = 'lower right', ncol = 1, labels = ['Stirred 1-3 days', 'Stirred 4-7 days', 'Stirred 10-13 days'], fontsize = 22)
ax[0].axvline(x = 0.26, linestyle = '--', color = 'black', linewidth = 2.5)
ax[0].annotate("P/P$_0$ ~ 0.26", (0.27, 500), fontsize=20)
plt.tight_layout()
    
#%%
data = np.genfromtxt('pores.csv', delimiter = ';')

fig, ax = plt.subplots(figsize = (12,9))
ax.set_prop_cycle('color', [plt.cm.winter(i) for i in np.linspace(0,1,3)])
ax.plot(data[:,0], data[:,1], marker = 'o', ms = 10, linewidth = 3)
ax.plot(data[:,2], data[:,3], marker = 'o', ms = 10, linewidth = 3)
ax.plot(data[:,4], data[:,5], marker = 'o', ms = 10, linewidth = 3)
# ax.plot(data[:,6], data[:,7], marker = 'o', linewidth = 3)
# ax.plot(data[:,8], data[:,9], marker = 'o', linewidth = 3)
# ax.plot(data[:,10], data[:,11], marker = 'o', linewidth = 3)
# ax.plot(data[:,12], data[:,13], marker = 'o', linewidth = 3)
# ax.plot(data[:,14], data[:,15], marker = 'o', linewidth = 3)

ax.tick_params(labelsize = 24)
ax.set_xlim([8,40])

ax.set_ylabel("dV/dlog(W)", fontsize = 24)
ax.set_xlabel(r"Pore Width, $\AA$", fontsize = 24)

ax.legend(loc = 'upper left', ncol = 1, labels = ['Stirred 1-3 days', 'Stirred 4-7 days', 'Stirred 10-13 days'], fontsize = 22)

plt.tight_layout()

#%% plot in molecules
sorpdata = np.genfromtxt('isotherms.csv', delimiter = ',')

mof808 = 1363.71

def converter(data_in_mmol_g, mofMr):
    # returns values converted to molecules per zr6 node
    molesMof = 1/mofMr
    molesZrNodes = molesMof
    numberZrNodes = molesZrNodes * 6.022*10**23
    numberGasMolecules = data_in_mmol_g * 6.022*10**23 * 1e-3
    output = numberGasMolecules / numberZrNodes
    return output
    

fig, ax = plt.subplots(figsize = (12,9))


# ax.set_prop_cycle('color', [plt.cm.Set1(i) for i in np.linspace(0,1,3)])
i = np.nanargmax(sorpdata[:,0])
ax.plot(sorpdata[:,0][i:], converter(sorpdata[:,1],mof808)[i:], marker = 'o', markersize = 10, color = 'lightblue', linewidth = 3)
ax.plot(sorpdata[:,0][:i+1], converter(sorpdata[:,1],mof808)[:i+1], marker = 'o', markersize = 10, color = 'blue', linewidth = 3)

i = np.nanargmax(sorpdata[:,18])
ax.plot(sorpdata[:,18][i:], converter(sorpdata[:,19],mof808)[i:], marker = 'o', markersize = 10, color = 'pink', linewidth = 3)
ax.plot(sorpdata[:,18][:i+1], converter(sorpdata[:,19],mof808)[:i+1], marker = 'o', markersize = 10, color = 'red', linewidth = 3)

i = np.nanargmax(sorpdata[:,20])
ax.plot(sorpdata[:,20][i:], converter(sorpdata[:,21],mof808)[i:], marker = 'o', markersize = 10, color = 'thistle', linewidth = 3)
ax.plot(sorpdata[:,20][:i+1], converter(sorpdata[:,21],mof808)[:i+1], marker = 'o', markersize = 10, color = 'darkviolet', linewidth = 3)



ax.tick_params(labelsize = 24)

ax.set_ylabel('Quantity Adsorbed (CO$_2$ per Zr$_6$O$_8$)', fontsize = 24)
ax.set_xlabel('Absolute Pressure (mbar)', fontsize = 24)

ax.legend(loc = 'upper left', ncol = 1, labels = ['MOF-808-Cl Adsorption @ 298K', 'MOF-808-Cl Desorption @ 298K', 'MOF-808-FF Adsorption @ 298K ', 'MOF-808-FF Desorption @ 298K ', 'MOF-808-FF Adsorption @ 318K ', 'MOF-808-FF Desorption @ 318K '], fontsize = 16)
ax.set_xlim(0,100)
ax.set_ylim(0,1)
# ax.axvline(0.422, linestyle = '--')
plt.tight_layout()


#%% 
V1 = 15000 * 10**(-6) 
t1 = 10 * 10**(-9)

V2 = 16000 * 10**(-6) 
t2 = 12.5 * 10**(-9)