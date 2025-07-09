# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:01:06 2024
This script calculates a few things about the nanoscaling effects in oxides
Adapted from Romanchuk et al. J. Synchrotron Rad. (2022). 29, 288–294
and a few other sources
@author: boris
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
#%% generate a plot for CN (Metal-Metal) vs Particle Size

def CNSize(dim, num, thi = 1, maxN = 12, size = [np.log10(0.5), 2], data=False, model="Jentys_N1"):
    """
    1. Take the size of the cluster (or NP) and the unit cell
    2. Calculate the number of unit cells per cluster
    3. Calculate the number of metal atoms in the cluster.
    4. Apply the 
    Inputs:
        dim: Unit cell dimensions, in angstrom
        num: Metal atoms per unit cell
        thi: Shell thickness for the core-shell model from AYuR paper
        model: by default, Jentys_N1
    Returns
    a plot of average and effective CNs vs particle size
    -------
    None.
    """
    fig, ax = plt.subplots(figsize = (10,10))
    # calculate the general relationship for CN (avg)
    pradius = np.logspace(size[0], size[1], 50) # in nm
    psize = (4/3)*np.pi*pradius**3
    if len(dim) == 1:
        dim = np.repeat(dim, 3)
    udimm = dim[0]*dim[1]*dim[2]*0.001
    cells = np.divide(psize, udimm)
    atoms = np.multiply(cells, num)
    if model=="Jentys_N1":
        a,b,c,d = 8.981, 9.640, 3.026, 1462.61
        CNavg = (a*atoms)/(np.add(atoms, b)) + (c*atoms)/(np.add(atoms, d))
    # ax.plot(pradius*2, CNavg, marker = 'o', markersize = 10, linewidth = 3, color = 'black', label = 'CN$_{avg}$')
    coreradius = pradius - thi
    coresize = (4/3)*np.pi*coreradius**3
    corecells = np.divide(coresize, udimm)
    coreatoms = np.multiply(corecells, num)
    CNeff = maxN * np.divide(coreatoms, atoms)
    ax.plot(np.where(CNeff>3, pradius*2, None), np.where(CNeff>3, CNeff, None), linewidth = 3, marker = 'o', markersize = 10, color = 'green', label = 'CN$_{eff}$')
    keys = list(data.keys())
    colors = ['red', 'blue', 'magenta', 'cyan']
    for key, color in zip(keys, colors):
        ax.axhline(y = data[key][0], linestyle = '-.', linewidth = 2.5, color = color, label = str(key))
        # ax.axhspan(ymin = data[key][0] - data[key][1], ymax = data[key][0] + data[key][1], color = color, alpha = 0.1)
    ax.set_xlim(0.5, 100)
    ax.set_xscale('log')
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_ylabel('CN$_{Me-Me}$', fontsize = 24)
    ax.set_xlabel('Particle Diameter (nm)', fontsize = 24)
    ax.tick_params(axis = 'both', labelsize = 24)
    ax.axhline(y = maxN, linestyle = '--', linewidth = 2.5, color = 'gray', label = 'bulk CN')
    ax.legend(loc = 'lower right', fontsize = 24)
    plt.tight_layout()
    plt.show()
    return atoms, pradius*2
test1, test2 = CNSize((5.396, 5.396, 5.396), 4, 0.4, 12, size = [np.log10(0.4), 2],
       data = {'n-PuO$_2$, fit 1': [6.7], 
               'n-PuO$_2$, fit 2': [6.9], 
               'n-PuO$_2$, fit 3': [8.4]})

#%% generate a plot for coordination numbers from particle diameter, written for ThO2
# both the Jentys (hard crystal) model and the Romanchuk model
def CNSize(dim, num, maxN = 12, data=False, model="Jentys_N1"):
    """
    1. Take the size of the cluster (or NP) and the unit cell
    2. Calculate the number of unit cells per cluster
    3. Calculate the number of metal atoms in the cluster.
    4. Apply the 
    Inputs:
        dim: Unit cell dimensions, in angstrom
        num: Metal atoms per unit cell
        thi: Shell thickness for the core-shell model from AYuR paper
        model: by default, Jentys_N1
    Returns
    a plot of average and effective CNs vs particle size
    -------
    None.
    """
    fig, ax = plt.subplots(figsize = (10,10))
    # calculate the general relationship for CN (avg)
    pradius = np.logspace(np.log10(0.25), np.log10(12.5), 50) # in nm
    psize = (4/3)*np.pi*pradius**3
    if len(dim) == 1:
        dim = np.repeat(dim, 3)
    udimm = dim[0]*dim[1]*dim[2]*0.001
    cells = np.divide(psize, udimm)
    atoms = np.multiply(cells, num)
    if model=="Jentys_N1":
        a,b,c,d = 8.981, 9.640, 3.026, 1462.61
        CNavg = (a*atoms)/(np.add(atoms, b)) + (c*atoms)/(np.add(atoms, d))
    # ax.plot(pradius*2, CNavg, marker = 'o', markersize = 10, linewidth = 3, color = 'black', label = 'CN$_{avg}$')
    thicknesses = (dim[0]*0.1, dim[0]*0.2, dim[0]*0.3)
    colors = plt.cm.gnuplot2(np.linspace(0.2, 0.8, len(thicknesses)))
    for thi, color in zip(thicknesses, colors): 
        coreradius = pradius - thi
        coresize = (4/3)*np.pi*coreradius**3
        corecells = np.divide(coresize, udimm)
        coreatoms = np.multiply(corecells, num)
        CNeff = maxN * np.divide(coreatoms, atoms)
        ax.plot(np.where(CNeff>1, pradius*2, None), np.where(CNeff>1, CNeff, None), linewidth = 3, marker = 'o', markersize = 10, color = color, label = 'CN$_{eff}$ ' + f'{thi:.1f}'+' nm shell')
    ax.plot(np.where(CNavg>3, pradius*2, None), np.where(CNavg>3, CNavg, None), linewidth = 3, marker = 'o', markersize = 10, color = 'black', label = 'CN$_{avg}$')
    keys = list(data.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(keys)))
    for key, color in zip(keys, colors):
        # ax.axhline(y = data[key][0], linestyle = '-.', linewidth = 2.5, color = color, label = str(key))
        ax.axvline(x = data[key][0], linestyle = '-.', linewidth = 2.5, color = color, label = str(key))
        # ax.axhspan(ymin = data[key][0] - data[key][1], ymax = data[key][0] + data[key][1], color = color, alpha = 0.1)
    # ax.set_xlim(0.5, 100)
    ax.set_xscale('log')
    major_ticks = np.arange(0, 13, 1)
    minor_ticks = np.arange(0, 13, 0.2)
    # ax.set_xticks(major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_ylabel('CN$_{Me-Me}$', fontsize = 24)
    ax.set_xlabel('Particle Diameter (nm)', fontsize = 24)
    ax.tick_params(axis = 'both', labelsize = 24)
    ax.axhline(y = maxN, linestyle = '--', linewidth = 2.5, color = 'gray', label = 'bulk CN')
    ax.legend(loc = 'best', fontsize = 24)
    plt.tight_layout()
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.6)
    plt.show()
    return atoms, pradius*2
test1, test2 = CNSize((5.434, 5.434, 5.434), 4, 12,
       data = {'NpO$_2$ d. \n(Scherrer)': [11]})

#%% plot number of atoms vs particle size, total and surface
# from matplotlib.ticker import LogLocator
def AtNSize(dim, Z, size, sizerange = [np.log10(0.5), 2]):
    """
    written for fluorite, double check

    Parameters
    ----------
    dim : array of len 3
        dimensions of the unit cell
    Z : int
        total number of METAL atoms per unit cell
    size : array of len 2
        lower and higher margins on the particle size (diameter of the sphere)

    Returns
    -------
    

    """
    fig, ax = plt.subplots(figsize = (10,10))
    # calculate the general relationship for CN (avg)
    pradius = np.linspace(sizerange[0]/2, sizerange[1]/2, 200) # in nm
    psize = (4/3)*np.pi*pradius**3
    if len(dim) == 1:
        dim = np.repeat(dim, 3)
    udimm = dim[0]*dim[1]*dim[2]*0.001
    cells = psize/udimm
    atoms = cells*Z
    
    coreradius = pradius - 0.5*(dim[2]*0.1)
    coresize = (4/3)*np.pi*coreradius**3
    shellsize = psize - coresize
    shellcells = shellsize/udimm
    shellatoms = np.multiply(shellcells, Z)
    shellratio = (shellatoms/atoms)*100
    
    ax.set_xlim(np.min(pradius*2), np.max(pradius*2))
    # ax.set_xscale('log')
    ax.spines['left'].set_color('indigo')
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_ylabel('Total Unit Cells', fontsize = 24, color = 'indigo')
    ax.set_xlabel('Particle Diameter (nm)', fontsize = 24)
    ax.tick_params(axis = 'y', labelsize = 24, labelcolor = 'indigo')
    ax.tick_params(axis = 'x', labelsize = 24)
    ax.scatter(pradius*2, cells, marker = 'o', s = 100, linewidth = 1, edgecolor = 'indigo', facecolor = 'none')
    ax.set_yscale('log')
    
    ax.axvline(x = size)
    
    ax2 = ax.twinx()
    ax2.spines['right'].set_color('firebrick')
    ax2.tick_params(axis='both', labelsize = 24, labelcolor = 'firebrick')
    ax2.set_ylabel('Atoms on the surface (%)', fontsize = 24, color = 'firebrick')
    ax2.scatter(pradius*2, shellratio, marker = '^', s = 100, linewidth = 1, edgecolor = 'firebrick', facecolor = 'none')
    
    plt.tight_layout()
    plt.show()
    
    idx = (np.abs(pradius - size/2)).argmin()
    print(f'a particle {size} nm in diameter has {atoms[idx]:.0f} metal atoms, {cells[idx]:.0f} unit cells and {shellratio[idx]:.1f}% surface metal sites')
    return atoms/Z, shellatoms, pradius*2

a,s,d = AtNSize((5.592, 5.592, 5.592), 4, 1.7, (0.5592, 20))

#%% plot number of atoms vs particle size, total and surface, core-shell particle
from matplotlib.ticker import LogLocator
def AtNSize(dim, Z,  thickness, size = [np.log10(0.5), 2]):
    """
    

    Parameters
    ----------
    dim : array of len 3
        dimensions of the unit cell
    Z : int
        total number of atoms per unit cell
    size : array of len 2
        lower and higher margins on the particle size (diameter of the sphere)

    Returns
    -------
    

    """
    fig, ax = plt.subplots(figsize = (10,10))
    # calculate the general relationship for CN (avg)
    pradius = np.logspace(size[0], size[1], 50) # in nm
    psize = (4/3)*np.pi*pradius**3
    if len(dim) == 1:
        dim = np.repeat(dim, 3)
    udimm = dim[0]*dim[1]*dim[2]*0.001
    cells = psize/udimm
    atoms = cells*Z
    
    surface_thi = (np.mean(dim)*0.1)
    print(f"surface thickness approximated as {surface_thi:.2f} nm")
    
    # calculate number of all atoms in shell
    coreradius = pradius - thickness
    coresize = (4/3)*np.pi*coreradius**3
    shellsize_bulk = psize - coresize
    shellcells = shellsize_bulk/udimm
    shellatoms = np.multiply(shellcells, Z)
    
    # calculate number of atoms in the surface unit cell
    bulkradius = pradius - surface_thi
    bulksize = (4/3)*np.pi*bulkradius**3
    shellsize_surface = psize - bulksize
    surfacecells = shellsize_surface/udimm
    surfaceatoms = np.multiply(surfacecells, Z)
    
    shellratio = (surfaceatoms/shellatoms)*100
    
    ax.set_xlim(1, np.max(pradius*2))
    ax.set_xscale('log')
    ax.spines['left'].set_color('indigo')
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_ylabel('Total Atoms', fontsize = 24, color = 'indigo')
    ax.set_xlabel('Particle Diameter (nm)', fontsize = 24)
    ax.tick_params(axis = 'y', labelsize = 24, labelcolor = 'indigo')
    ax.tick_params(axis = 'x', labelsize = 24)
    ax.scatter(pradius*2, atoms, marker = 'o', s = 100, linewidth = 1, edgecolor = 'indigo', facecolor = 'none')
    ax.set_yscale('log')
    
    ax.axvline(x = thickness * 2)
    
    ax2 = ax.twinx()
    ax2.spines['right'].set_color('firebrick')
    ax2.tick_params(axis='both', labelsize = 24, labelcolor = 'firebrick')
    ax2.set_ylabel('Atoms on the surface (%)', fontsize = 24, color = 'firebrick')
    ax2.scatter(pradius*2, shellratio, marker = '^', s = 100, linewidth = 1, edgecolor = 'firebrick', facecolor = 'none')
    
    plt.tight_layout()
    plt.show()
    return coreradius, bulkradius, pradius*2

a,s,d = AtNSize((3.799, 3.799, 9.509), 4, 4, [np.log10(1), 7])

#%% one nanoparticle from slab
import pandas as pd
filename = 'tho2_xyz_sorted_slab.csv'
slab = pd.read_csv(filename, names = ['atom', 'x', 'y', 'z', 'dist'])
slab_ = np.array(slab.iloc[:, 1:])

diameter = 
cutoff = diameter/2

nanosphere = slab.iloc[slab_[:,3]<=cutoff, :-1]

# check that it's oxide terminated and increase cutoff if it is not
# while nanosphere.iloc[-1,0] == 'Th':
#     cutoff += 0.01
#     nanosphere = slab.iloc[slab_[:,3]<=cutoff, :-1]
    
compound = 'ThO2'
number_a = len(slab.iloc[slab_[:,3]<=cutoff, -1])

print(int(number_a), compound,  nanosphere.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compound}_{diameter:.1f}ang_1sphere_{number_a}atoms.xyz', 'w'))
#%% one shell from slab
import pandas as pd
filename = 'Ni3P2O8_slab_sorted.csv'
slab = pd.read_csv(filename, names = ['atom', 'x', 'y', 'z', 'dist'])
slab_ = np.array(slab.iloc[:, 1:])

diameter = 30
offset = 2
diameter += offset
thickness = 5
cutoff_low = diameter/2
cutoff_high = cutoff_low + thickness

nanoshell = slab[(slab.dist >= cutoff_low) & (slab.dist <= cutoff_high)]
nanoshell = nanoshell.iloc[:,:-1]

# shell_index = np.where((slab_[:,3]>=cutoff_low) & (slab_[:,3] <= cutoff_high))
# nanosphere = slab.iloc[slab_[np.min(shell_index):np.max(shell_index),3], :-1]

# shell_index = np.where(np.logical_and(slab.iloc[:,4]>=cutoff_low, slab.iloc[:,4]<=cutoff_high))
    
compound = 'Ni3P2O8'
number_a = nanoshell.shape[0]

print(int(number_a), compound,  nanoshell.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compound}_{diameter:.1f}ang_1shell_{number_a}atoms.xyz', 'w'))

#%% combine two files
file1 = nanosphere.copy()
file2 = nanoshell.copy()

total = pd.concat([file1, file2], ignore_index = True)
number_a = file1.shape[0]+file2.shape[0]

compounds ="Ni2P_Ni3P2O8"
diameter = 2*np.max([np.sqrt(total.iloc[i,1]**2+total.iloc[i,2]**2+total.iloc[i,3]**2) for i in range(number_a)])

print(int(number_a), compounds,  total.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compounds}_{diameter:.1f}ang_coreshell_{number_a}atoms.xyz', 'w'))

#%% multiple nanoparticles from slab
import pandas as pd
filename = 'tho2_xyz_sorted_slab.csv'
slab = pd.read_csv(filename, names = ['atom', 'x', 'y', 'z', 'dist'])
slab_ = np.array(slab.iloc[:, 1:])

diameter = 38
cutoff = diameter/2

nanosphere = slab.iloc[slab_[:,3]<=cutoff, :-1]

# check that it's oxide terminated and increase cutoff if it is not
# while nanosphere.iloc[-1,0] == 'Th':
#     cutoff += 0.01
#     nanosphere = slab.iloc[slab_[:,3]<=cutoff, :-1]
    
compound = 'ThO2'
number_a = len(slab.iloc[slab_[:,3]<=cutoff, -1])

print(int(number_a), compound,  nanosphere.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compound}_{diameter:.1f}ang_1sphere_{number_a}atoms.xyz', 'w'))

translation_distance = diameter*2
total_atoms = 2000

number_particles = round(total_atoms/number_a)

multipliers = ((0,0,1), (0,1,0), (1,0,0), 
               (0, 0, -1), (0,-1,0), (-1, 0, 0),
               (0,1,1), (1,1,0), (1,1,0))

spheres = nanosphere.copy()
for i in range(number_particles):
    m = multipliers[i]
    ns = nanosphere.copy()
    ns.iloc[:,1] += m[0]*translation_distance
    ns.iloc[:,2] += m[1]*translation_distance
    ns.iloc[:,3] += m[2]*translation_distance
    spheres = pd.concat([spheres,ns])
    
number_a = len(spheres.iloc[:,1])

print(int(number_a), compound,  spheres.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compound}_{diameter:.1f}ang_{number_particles+1}spheres_{number_a}atoms.xyz', 'w'))

#%% nanoparticle and clusters from slab
# option 1: set proportion by moles
# option 2: equal proportions by Th weight

# option 1 

parts_NP = 1
parts_clusters = 1

import pandas as pd
filename = 'tho2_xyz_sorted_slab.csv'
slab = pd.read_csv(filename, names = ['atom', 'x', 'y', 'z', 'dist'])
slab_ = np.array(slab.iloc[:, 1:])

diameter = 19.4
cutoff = diameter/2

nanosphere = slab.iloc[slab_[:,3]<=cutoff, :-1]

# check that it's oxide terminated and increase cutoff if it is not
while nanosphere.iloc[-1,0] == 'Th':
    cutoff += 0.01
    nanosphere = slab.iloc[slab_[:,3]<=cutoff, :-1]
    
compound = 'ThO2'
number_a = len(slab.iloc[slab_[:,3]<=cutoff, -1])

print(int(number_a), compound,  nanosphere.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compound}_{diameter:.1f}ang_1sphere_{number_a}atoms.xyz', 'w'))

translation_distance = diameter*2
total_atoms = 1000

number_particles = round(total_atoms/number_a)

multipliers = ((0,0,1), (0,1,0), (1,0,0), (0, 0, -1), (0,-1,0), (-1, 0, 0))

spheres = nanosphere.copy()
for i in range(number_particles):
    m = multipliers[i]
    ns = nanosphere.copy()
    ns.iloc[:,1] += m[0]*translation_distance
    ns.iloc[:,2] += m[1]*translation_distance
    ns.iloc[:,3] += m[2]*translation_distance
    spheres = pd.concat([spheres,ns])
    
number_a = len(spheres.iloc[:,1])

print(int(number_a), compound,  spheres.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compound}_{diameter:.1f}ang_{number_particles+1}spheres_{number_a}atoms.xyz', 'w'))

number_clusters = round(number_particles * (parts_clusters/parts_NP)) + 1

cluster_file = 'th6cluster_xyz_sorted_slab.csv'
cluster = pd.read_csv(cluster_file, names = ['atom', 'x', 'y', 'z', 'dist'])
cluster_ = cluster.iloc[:, :-1]

# create coordinates for clusters
from itertools import product
t = range(3)
res = set(product(set(t),repeat = 3))
res_ = list(res)
for i in range(len(res_)):
    res_[i] = np.subtract(res_[i], (1, 1, 1))*0.5
to_rm = np.array([0., 0., 0.])
res_ = [ x for x in res_ if not (x==to_rm).all()]

spheres_clusters = spheres.copy()
for i in range(number_clusters):
    m = res_[i]
    ct = cluster_.copy()
    ct.iloc[:,1] += m[0]*translation_distance
    ct.iloc[:,2] += m[1]*translation_distance
    ct.iloc[:,3] += m[2]*translation_distance
    spheres_clusters = pd.concat([spheres_clusters, ct])
    
number_a = len(spheres_clusters.iloc[:,1])
    
print(int(number_a), compound+"_Th6O8Ox",  spheres_clusters.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compound}_{diameter:.1f}ang_{number_particles+1}spheres_{number_clusters}clusters_{number_a}atoms.xyz', 'w'))

#%% nanoparticle and clusters from slab
# option 1: set proportion by moles
# option 2: equal proportions by Th weight

# option 2 

parts_NP = 1
parts_clusters = 1

import pandas as pd
filename = 'tho2_xyz_sorted_slab.csv'
slab = pd.read_csv(filename, names = ['atom', 'x', 'y', 'z', 'dist'])
slab_ = np.array(slab.iloc[:, 1:])

diameter = 38
cutoff = diameter/2

nanosphere = slab.iloc[slab_[:,3]<=cutoff, :-1]

# check that it's oxide terminated and increase cutoff if it is not
while nanosphere.iloc[-1,0] == 'Th':
    cutoff += 0.01
    nanosphere = slab.iloc[slab_[:,3]<=cutoff, :-1]
    
compound = 'ThO2'
number_a = len(slab.iloc[slab_[:,3]<=cutoff, -1])
number_particles = 0

translation_distance = diameter*2

print(int(number_a), compound,  nanosphere.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compound}_{diameter:.1f}ang_1sphere_{number_a}atoms.xyz', 'w'))

Th_count = nanosphere[nanosphere['atom']=='Th'].shape[0]

cluster_size = 6

number_clusters = round(Th_count/cluster_size)
if number_clusters > 26:
    number_clusters = 26

cluster_file = 'th6cluster_xyz_sorted_slab.csv'
cluster = pd.read_csv(cluster_file, names = ['atom', 'x', 'y', 'z', 'dist'])
cluster_ = cluster.iloc[:, :-1]

# create coordinates for clusters
from itertools import product
t = range(3)
res = set(product(set(t),repeat = 3))
res_ = list(res)
for i in range(len(res_)):
    res_[i] = np.subtract(res_[i], (1, 1, 1))*0.5
to_rm = np.array([0., 0., 0.])
res_ = [ x for x in res_ if not (x==to_rm).all()]

spheres_clusters = nanosphere.copy()
for i in range(number_clusters):
    m = res_[i]
    ct = cluster_.copy()
    ct.iloc[:,1] += m[0]*translation_distance
    ct.iloc[:,2] += m[1]*translation_distance
    ct.iloc[:,3] += m[2]*translation_distance
    spheres_clusters = pd.concat([spheres_clusters, ct])
    
number_a = len(spheres_clusters.iloc[:,1])
    
print(int(number_a), compound+"_Th6O8Ox",  spheres_clusters.to_string(index=False, header=False), sep='\n', 
      file=open(f'nano_{compound}_{diameter:.1f}ang_{number_particles+1}spheres_{number_clusters}clusters_{number_a}atoms.xyz', 'w'))