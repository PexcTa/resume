# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:36:17 2022

@author: boris
"""

import csv
import itertools
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
from scipy.signal import correlate, convolve
from scipy import special
import os
from scipy.optimize import curve_fit
from scipy.integrate import quad, dblquad
from scipy.signal import savgol_filter as sgf
from collections import OrderedDict
import lmfit as lm

#%% import and processing functions
def normalize_1(data):
    """
    Normalizes a vector to the maximum found within the vector.
    Negative values will be converted to positive ones.
    Parameters
    ----------
    data : array-like
    The array to normalize to its maximum.

    Returns
    -------
    normalized_data : array-like
    Normalized array of the same shape as passed in.

    """
    normalized_data = np.abs(data)/max(data)
    return normalized_data
def normalize_any(data, factor):
    """
    Normalizes a vector to the maximum found within the vector.
    Negative values will be converted to positive ones.
    Parameters
    ----------
    data : array-like
    The array to normalize to its maximum.

    Returns
    -------
    normalized_data : array-like
    Normalized array of the same shape as passed in.

    """
    normalized_data = np.abs(data)/max(data)
    normalized_data *= factor
    return normalized_data
def nm2WN(data_in_nm):
    return 10000000/data_in_nm
def WN2nm(data_in_WN):
    return 10000000/data_in_WN
def nm2ev(data_in_nm):
    h = 4.135667516*10**(-15)
    c = 299792458
    return (h*c)/(data_in_nm*10**(-9))
def ev2nm(data_in_ev):
    h = 4.135667516*10**(-15)
    c = 299792458
    return (h*c)/(data_in_ev)/10**(-9)
def jac(wavelength, data):
    e = 1.602*10**(-19)
    h = 4.135667516*10**(-15)
    c = 299792458
    jf = (e*(wavelength*10**-9)**2)/h*c*10**9
    return np.multiply(data, jf)


#%% data import
def StreakCamImageImport(image, time, wavelength):
    """
    Import streak camera data collected at the CNM facilities. 
    
    Parameters
    ----------
    image: string for filename
        string pointing to an mxn matrix with detector counts;
    time: string for filename
        string pointing to matrix with a time vector of length m
    wavelength: string for filename
        string pointing to matrix with a wavelength vector of length n
        
    Returns
    -------
    A (m+1)x(n+1) matrix with numerical x,y,z data for wavelength, time, counts.
    
    For the specific setup at CNM, the z data on wavelength axis was always from high to low. This function has a flip to accomodate.
    """
    with open(image) as current_file:
        z_image = np.loadtxt(current_file, dtype=int)
    with open(time) as current_file:
        tm = np.loadtxt(current_file, dtype=float)
    with open(wavelength) as current_file:
        wl = np.loadtxt(current_file, dtype=float)
    data = np.zeros((len(tm)+1, len(wl)+1))
    data[1:,0] = tm[:]
    data[0,1:] = wl[:]
    data[1:, 1:] = np.flip(z_image[:,:], axis = 1)
    return data
# 
s18_low = StreakCamImageImport('s18_150uw.dat', 'time_10ns.csv', 'wavelength.csv')
s18_hi = StreakCamImageImport('s18_4470uw.dat', 'time_10ns.csv', 'wavelength.csv')
# s14_2 = StreakCamImageImport('s14_1550uw.dat', 'time_50ns.csv', 'wavelength.csv')
# s14_3 = StreakCamImageImport('s14_2770uw.dat', 'time_50ns.csv', 'wavelength.csv')
# s14_4 = StreakCamImageImport('s14_3940uw.dat', 'time_50ns.csv', 'wavelength.csv')

#%% Plotting Functions - Rewritten Nov. 15 2022
def streakPlot(data, region=False, timescale='ns', lvl=20, cmap='inferno'):
    """

    Parameters
    ----------
    data : array-like
        first row must be wavelength
        first column must be time
        [0,0] position is empty
        [1:, 1:] must be data
    region : tuple
        borders of the wavelength region of interest. 
        The default is False (The entire region will be taken)
    timescale : string
        units of the time axis. The default is 'ns'.
    lvl : int
        Number of levels in the contour plot. The default is 20.
    cmap : string
    Colormap to use for the contour plot. The default is 'inferno'.

    Returns
    -------
    PLots a contour map of the detector response.
    None.

    """
    fig, ax = plt.subplots(figsize=(12,10))
    ax.set_xlabel('Wavelength (nm)', fontsize = 27)
    ax.set_ylabel('Time ('+timescale+")", fontsize = 27)
    x = data[0,1:]
    y = data[1:,0]
    z = data[1:,1:]
    if region != False:
        idx_x_min = (np.abs(x - region[0])).argmin()
        idx_x_max = (np.abs(x - region[1])).argmin()
        x = x[idx_x_min:idx_x_max]
        z = z[:, idx_x_min:idx_x_max]
    cs1 = ax.contourf(x, y, z, lvl, cmap = cmap)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(cs1, cax=cax, orientation='vertical')
    cbar.set_ticks([np.linspace(z.min().min(), z.max().max(), 10, dtype = int)])
    cax.tick_params(labelsize = 20)
    cax.set_title('Cts', fontsize = 24)
    
def followShift(data, region=False, nbins = 10, maxtime = False):
    """
    
    Parameters
    ----------
    data : array-like
        first row must be wavelength
        first column must be time
        [0,0] position is empty
        [1:, 1:] must be data
    region : tuple
        borders of the wavelength region of interest. 
        The default is False (The entire region will be taken)
    nbins : TYPE, optional
        number of bins to put the z data into. counts will be summed up along time. The default is 10.
    maxtime : TYPE, optional
        maximum time of interest to go out to. 
        The default is False (The full time range will be taken)
    cmap : TYPE, optional
        Colormap to use. The default is 'inferno'.

    Returns
    -------
    Plots a figure that shows evolution of fluorescence spectrum in time.
    Does not return anything else.

    """
    fig, ax = plt.subplots(figsize=(12,10))
    x = data[0,1:]
    y = data[1:,0]
    z = data[1:,1:]
    if region != False:
        idx_x_min = (np.abs(x - region[0])).argmin()
        idx_x_max = (np.abs(x - region[1])).argmin()
        x = x[idx_x_min:idx_x_max]
        z = z[:, idx_x_min:idx_x_max]
    if maxtime != False:
        idx_y_max = (np.abs(y - maxtime)).argmin()
        y = y[0:idx_y_max]
        z = z[0:idx_y_max, :]
    ax.set_xlim([min(x), max(x)])
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.axhline(y=0, color = 'dimgrey', ls = '--')
    ax.set_xlabel("Wavelength (nm)", fontsize = 27)
    ax.set_ylabel("Counts", fontsize = 27)
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    step = int(len(y)/nbins+1)
    ax.set_prop_cycle('color',[plt.cm.inferno(i) for i in np.linspace(0, 0.95, nbins)])
    for i,j in zip(range(0,len(y)-step,step), range(step,len(y),step)):
        ax.plot(x, np.sum(z[i:j,:], axis = 0), linewidth = 3)
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=0, vmax=max(y)))
    cbar = plt.colorbar(sm, pad = 0.01)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Time (ns)', fontsize = 24) 
    
def kinetics(data_tuple, timeshifts=False, labels=["placeholder"], timescale='ns', logscale = False, maxtime = False, region = False, normalize = True):
    """
    This function is used to plot kinetics integrated over a specific region of the overall streak matrix.

    Parameters
    ----------
    data_tuple : tuple of array-like
        A tuple of matrices. In each matrix:
        first row must be wavelength
        first column must be time
        [0,0] position is empty
        [1:, 1:] must be data
    timeshifts : array-like, 1d
        A list of floats to shift the signal in time for alignment purposes. 
        The default is False - the time axis is not shifted.
    labels : array-like, 1d
        A list of labels to use for the legend. The default is ["placeholder"].
    timescale : string, optional
        units of the time axis. The default is 'ns'.
    logscale : boolean, optional
        If True, sets the y-axis to logarithmic scale. The default is False.
    maxtime : float, optional
        If non-False, will cut time axis at the specified float value. The default is False.
    region : tuple of integers, optional
        If non-False, will cut the wavelength axis to the specified region. The default is False.

    Returns
    -------
    None.

    """
    if type(data_tuple) == OrderedDict:
        if labels == ['placeholder']:
            labels = data_tuple.keys()
        data_tuple = tuple(data_tuple.values())
    fig, ax = plt.subplots(figsize=(10,10))
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
    ax.set_xlabel('Time ('+timescale+")", fontsize = 27)
    ax.set_ylabel("Normalized Intensity", fontsize = 27)
    ax.set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, len(data_tuple))])
    i = 0
    if timeshifts == False:
        timeshifts = np.zeros(len(data_tuple))
    for dataset in data_tuple:
        x = dataset[0,1:]
        y = dataset[1:,0]
        z = dataset[1:,1:]
        if region != False:
            idx_x_min = (np.abs(x - region[0])).argmin()
            idx_x_max = (np.abs(x - region[1])).argmin()
            x = x[idx_x_min:idx_x_max]
            z = z[:, idx_x_min:idx_x_max]
        if maxtime != False:
            idx_y_max = (np.abs(y - maxtime)).argmin()
            y = y[0:idx_y_max]
            z = z[0:idx_y_max, :]
            ax.set_xlim([0, maxtime])
        elif maxtime == False:
            maxtime = max(y)
            ax.set_xlim([0, maxtime])
        if maxtime > 0.1 and maxtime <= 0.5:
            ax.xaxis.set_major_locator(MultipleLocator(0.05))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        if maxtime > 0.01 and maxtime <= 0.1:
            ax.xaxis.set_major_locator(MultipleLocator(0.01))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.001))
        if maxtime <= 0.01:
            ax.xaxis.set_major_locator(MultipleLocator(0.001))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        if maxtime > 0.5 and maxtime <= 1:
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        if maxtime > 1 and maxtime <= 5:
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        if maxtime > 5:
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        if maxtime >= 10:
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        if maxtime >= 25:
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        if normalize == True:
            kinetic_signal = normalize_1(np.sum(z, axis=1))  
            ax.plot(y+timeshifts[i], kinetic_signal, linewidth = 3)
            i+=1
        elif normalize == False:
            kinetic_signal = np.sum(z, axis=1)
            ax.plot(y+timeshifts[i], kinetic_signal, linewidth = 3)
            ax.set_ylabel("Intensity (cts)", fontsize = 27)
            i+=1
    ax.legend(labels = labels, loc = 'upper right', fontsize = 27)
    if logscale != False:
        ax.set_yscale('log')
    ax.axhline(y=0, color = 'dimgrey', ls = '--')


def spectra(data_tuple, labels=["placeholder"], units = 'ev', maxtime = False, region = False, normalize = True):
    """
    A function to plot spectral lineshapes integrated over the entire time range. 
    As-written, returns smoothed spectra (sav. gol. 5,2)
    Plots on energy axis with Jacobian transformation applied 

    Parameters
    ----------
    data_tuple : tuple of array-like
        A tuple of matrices. In each matrix:
        first row must be wavelength
        first column must be time
        [0,0] position is empty
        [1:, 1:] must be data
    labels : array-like, 1d
        A list of labels to use for the legend. The default is ["placeholder"].
    maxtime : float, optional
        If non-False, will cut time axis at the specified float value. The default is False.
    region : tuple of integers, optional
        If non-False, will cut the wavelength axis to the specified region. The default is False.
    normalize : boolean, optional
        If True, normalizes the spectra to the maximum (found within the bounds imposed by region and maxtime). The default is True.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_ylabel("Intensity (arb. units)", fontsize = 27)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, len(data_tuple))])
    for dataset in data_tuple:
         x = dataset[0,1:]
         y = dataset[1:,0]
         z = dataset[1:,1:]
         if region != False:
             idx_x_min = (np.abs(x - region[0])).argmin()
             idx_x_max = (np.abs(x - region[1])).argmin()
             x = x[idx_x_min:idx_x_max]
             z = z[:, idx_x_min:idx_x_max]
         if maxtime != False:
             idx_y_max = (np.abs(y - maxtime)).argmin()
             y = y[0:idx_y_max]
             z = z[0:idx_y_max, :]
         elif maxtime == False:
             maxtime = max(y)
         if normalize == True:
             z_sum = normalize_1(np.sum(z, axis=0))
         elif normalize == False:
             z_sum = np.sum(z, axis=0)
             ax.set_ylabel("Intensity (cts)", fontsize = 27)
         if units == 'ev':
             spectral = savgol_filter(normalize_1(jac(x, z_sum)), 5, 2)
             ax.plot(nm2ev(x), spectral, linewidth = 4)
             ax.set_xlim(nm2ev(min(x)), nm2ev(max(x)))
             ax.xaxis.set_major_locator(MultipleLocator(0.2))
             ax.xaxis.set_minor_locator(MultipleLocator(0.05))
             ax.set_xlabel("Energy (eV)", fontsize = 27)
             secax = ax.secondary_xaxis('top', functions=(ev2nm, nm2ev))
             secax.tick_params(labelsize = 20)
             secax.set_xlabel('Wavelength (nm)', fontsize = 24, labelpad = 10)
         elif units == 'nm':
             spectral = savgol_filter(normalize_1(z_sum), 5, 2)
             ax.plot(x, spectral, linewidth = 4)
             ax.set_xlim([min(x), max(x)])
             ax.xaxis.set_major_locator(MultipleLocator(50))
             ax.xaxis.set_minor_locator(MultipleLocator(10))
             ax.set_xlabel("Wavelength (nm)", fontsize = 27)
             secax = ax.secondary_xaxis('top', functions=(nm2ev, ev2nm))
             secax.tick_params(labelsize = 20)
             secax.set_xlabel('Energy (eV)', fontsize = 24, labelpad = 10)
    ax.legend(labels = labels, loc = 'upper right', fontsize = 27)
    ax.axhline(y=0, color = 'dimgrey', ls = '--')
    plt.tight_layout()
    return np.vstack([x,spectral])

def TraceExtractor(data, regions, normalize = False, normfactors = 'default', plotTraces = False):
    """
    Takes a data matrix and a tuple of wavelengths to return the kinetic traces closest to the wavelengths specified.
    If plotTraces set to True, will also return a labeled plot.
    Parameters
    ----------
    data : array-like
        first row must be wavelength
        first column must be time
        [0,0] position is empty
        [1:, 1:] must be data
    regions : list of tuples
        each tuple must contain two int values defining regions of interest

    Returns
    -------
    traces : TYPE
        DESCRIPTION.

    """
    NCol = len(regions)
    x = data[0,1:]
    y = data[1:,0]
    z = data[1:,1:]
    traces = np.zeros((len(y), NCol+1), dtype = float)
    traces[:,0] = y
    i = 1
    if normfactors == 'default':
        normfactors = np.ones(len(regions))
    for r in range(len(regions)): 
        region = regions[r]
        position1 = (np.abs(x - min(region))).argmin()
        position2 = (np.abs(x - max(region))).argmin()
        trace = np.sum(z[:,position1:position2], axis=1)
        if normalize != False:
            trace = normalize_any(trace, normfactors[r])
        traces[:,i] = trace
        i+=1
    if plotTraces:
        fig,ax = plt.subplots(figsize = [10,10])            
        ax.set_ylabel("Intensity (arb. units)", fontsize = 27)
        if normalize:
            ax.set_ylabel("Normalized Intensity", fontsize = 27)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize= 24)
        ax.tick_params(axis='y', labelsize= 24)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel('Time, ns', fontsize = 27)
        maxtime = max(y)
        if maxtime > 0.1 and maxtime <= 0.5:
            ax.xaxis.set_major_locator(MultipleLocator(0.05))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        if maxtime > 0.01 and maxtime <= 0.1:
            ax.xaxis.set_major_locator(MultipleLocator(0.01))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.001))
        if maxtime <= 0.01:
            ax.xaxis.set_major_locator(MultipleLocator(0.001))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        if maxtime > 0.5 and maxtime <= 1:
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        if maxtime > 1 and maxtime <= 5:
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        if maxtime > 5:
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        if maxtime >= 10:
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        if maxtime >= 25:
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        labels = [str(min(region)) + " to " + str(max(region)) for region in regions]
        ax.set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, len(labels))])
        for i in range(1,len(traces[0,1:])+1):
            ax.plot(y, traces[:, i], linewidth = 4)
        ax.set_yscale('log')
        ax.legend(labels = labels, loc = 'upper right', fontsize = 27)
    return traces

def moment(data_tuple, labels=["placeholder"], y_units = 'ns', maxtime = False, region = False):
    """
    a function to plot the spectral moment of spectrally resolved TRPL data
    
    Parameters
    ----------
    data_tuple : tuple of array-like
        A tuple of matrices. In each matrix:
        first row must be wavelength
        first column must be time
        [0,0] position is empty
        [1:, 1:] must be data
    labels : array-like, 1d
        A list of labels to use for the legend. The default is ["placeholder"].
    maxtime : float, optional
        If non-False, will cut time axis at the specified float value. The default is False.
    region : tuple of integers, optional
        If non-False, will cut the wavelength axis to the specified region. The default is False.
    normalize : boolean, optional
        If True, normalizes the spectra to the maximum (found within the bounds imposed by region and maxtime). The default is True.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(10,10))
    # ax.set_ylabel("Intensity (arb. units)", fontsize = 27)
    # ax.set_yticklabels([])
    # ax.set_yticks([])
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, len(data_tuple))])
    ax.set_ylabel('Fluor. moment (eV)', fontsize = 27)
    for dataset in data_tuple:
        x = dataset[0,1:]
        y = dataset[1:,0]
        z = dataset[1:,1:]
        if region != False:
            idx_x_min = (np.abs(x - region[0])).argmin()
            idx_x_max = (np.abs(x - region[1])).argmin()
            x = x[idx_x_min:idx_x_max]
            z = z[:, idx_x_min:idx_x_max]
        if maxtime != False:
            idx_y_max = (np.abs(y - maxtime)).argmin()
            y = y[0:idx_y_max]
            z = z[0:idx_y_max, :]
        elif maxtime == False:
            maxtime = max(y)
        moment = np.zeros_like(y)
        i = 0
        while i < len(y):
            moment[i] = np.sum(np.multiply(nm2ev(x), z[i,:]))/np.sum(z[i,:])
            i+=1
        ax.plot(y, moment, linewidth = 4)
        ax.set_xlim(0, 50)
        ax.set_ylim(2.0, 3.5)
        # ax.xaxis.set_major_locator(MultipleLocator(0.2))
        # ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_xlabel(f"Time, {y_units}", fontsize = 27)
    ax.legend(labels = labels, loc = 'upper right', fontsize = 27)
    ax.axhline(y=0, color = 'dimgrey', ls = '--')
#%% where we watch
# streakPlot(s18_0, region = (400, 660), cmap = 'gnuplot2')
# followShift(s17_150, region = (400, 660), nbins = 8, maxtime = False)
# followShift(s17_3230, region = (400, 660), nbins = 8, maxtime = False)
# moment([s14_0, s14_1, s14_2, s14_3, s14_4], region = (370, 600))
spectrals = spectra([S20[key] for key in S20.keys()], 
                    labels=[f'{label:.3f}' for label in S20_Report['Excitation Probabilities']], 
                    units = 'ev', maxtime = False, region = [400, 660], 
                    normalize = True)
# test3230 = spectra([s17_3230], labels=["placeholder"], units = 'nm', maxtime = False, region = False, normalize = True)


#%% where we fit

def GlobalFitting(data, regions, timelim, exp_number, initial_tau, theta = (0,0.1), A_value = 1, timescale = 'ns', normalize = False, logscale = True, plotfit = True):
    """
    Takes the data matrix, extracts several regions (wavelength-wise) and integrates signal over time, then fits the signal decay to a sum of exponential functions. 
    The fitting equation is a convolution of a single Gaussian response function to multiple exponential decay functions.
    Only meant to be used to model first order decay processes.
    
    
    Parameters
    ----------
    data : array-like
        first row must be wavelength
        first column must be time
        [0,0] position is empty
        [1:, 1:] must be data
    regions : list of tuples
        each tuple must contain two int values defining regions of interest
    timelim : int
        maximum time to for plotting purposes.
        the kinetics will be fit to full range regardless of the timelim value
    exp_number : int
        number of exponentials to use when the function is built
        generally, more regions will require more exponentials
    initial_tau : list of float
        initial guesses for your tau values. length must equal exp_number
    theta : tuple
        a tuple with time zero and Gaussian IRF full-width half maximum values
        theta[0] describes the time zero, i.e. signal rise point
        theta[1] describes the IRF, i.e. pulse width
        The default is (0,0.1).
    A_value : int
        Starting value for amplitude pre-factors to use in the exponential sum.
        Recommended value is on the order of maximum signal attained.
        The default is 1.
    timescale : string
        Specify time units to use in plotting. The default is 'ns'.
    normalize : boolean
        If True, kinetics will be normalized to max. value of 1. The default is False.
    logscale : boolean
        If True, kinetics will be plotted on logarithmic y-scale. The default is True.
    plotfit : boolean
        If True, data and fit will be plotted. The default is True.

    Returns
    -------
    pd_pars : pandas Data Frame
        A Dataframe containing the fitted values and associated uncertainties.

    """
    
    def three_regions(time, M, N, fwhm, t0, **Pars):
        x = time
        T_1 = np.zeros_like(time)
        T_2 = np.zeros_like(time)
        T_3 = np.zeros_like(time)
        for m in range(1, M+1):
            for n in range(1, N+1):
                if m == 1:
                    A = Pars[f'A_{m}_{n}']
                    tau = Pars[f'tau_{n}']
                    T_1 += A*fwhm*np.exp((fwhm/(np.sqrt(2)*tau))**2-(x-t0)/tau)*(1-special.erf(fwhm/(np.sqrt(2)*tau)-(x-t0)/(np.sqrt(2)*fwhm)))
                elif m == 2:
                    A = Pars[f'A_{m}_{n}']
                    tau = Pars[f'tau_{n}']
                    T_2 += A*fwhm*np.exp((fwhm/(np.sqrt(2)*tau))**2-(x-t0)/tau)*(1-special.erf(fwhm/(np.sqrt(2)*tau)-(x-t0)/(np.sqrt(2)*fwhm)))
                elif m == 3:
                    A = Pars[f'A_{m}_{n}']
                    tau = Pars[f'tau_{n}']
                    T_3 += A*fwhm*np.exp((fwhm/(np.sqrt(2)*tau))**2-(x-t0)/tau)*(1-special.erf(fwhm/(np.sqrt(2)*tau)-(x-t0)/(np.sqrt(2)*fwhm)))
        return np.concatenate([T_1, T_2, T_3])
    def two_regions(time, M, N, fwhm, t0, **Pars):
        x = time
        T_1 = np.zeros_like(time)
        T_2 = np.zeros_like(time)
        for m in range(1, M+1):
            for n in range(1, N+1):
                if m == 1:
                    A = Pars[f'A_{m}_{n}']
                    tau = Pars[f'tau_{n}']
                    T_1 += A*fwhm*np.exp((fwhm/(np.sqrt(2)*tau))**2-(x-t0)/tau)*(1-special.erf(fwhm/(np.sqrt(2)*tau)-(x-t0)/(np.sqrt(2)*fwhm)))
                elif m == 2:
                    A = Pars[f'A_{m}_{n}']
                    tau = Pars[f'tau_{n}']
                    T_2 += A*fwhm*np.exp((fwhm/(np.sqrt(2)*tau))**2-(x-t0)/tau)*(1-special.erf(fwhm/(np.sqrt(2)*tau)-(x-t0)/(np.sqrt(2)*fwhm)))
        return np.concatenate([T_1, T_2])
    M = len(regions)
    print(f'There are {M} regions selected: {regions}')
    N = exp_number
    print(f'Fitting each region to {N} exponentials')
    if M == 2:
        model = lm.Model(two_regions, independent_vars=['time'])
    elif M == 3:
        model = lm.Model(three_regions, independent_vars=['time'])
    params = lm.Parameters()
    params.add('M', value=M, vary=False) # ms and ns shouldn't vary, they just
    params.add('N', value=N, vary=False) # let us decide when to truncate our sum
    tauGuess = initial_tau
    print(f'Using the following initial pars for tau: {tauGuess}')
    for m in range(1, M+1):
        for n in range(1, N+1):
            params.add(f'A_{m}_{n}', value=A_value, min = 0)
    for n in range(1, N+1):
        params.add(f'tau_{n}', value = tauGuess[n-1], min = 0)
    params.add('t0', value = theta[0])
    params.add('fwhm', value = theta[1])
    print('Model parameters constructed')
    traces = TraceExtractor(data, regions, normalize, plotTraces = False)
    samples = np.concatenate([traces[:,i] for i in range(1, len(traces[0,1:])+1)])
    time = traces[:,0]
    print('Fitting...')
    result = model.fit(samples, time = time, params=params)
    print(result.fit_report())
    pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
    df_true = pd.DataFrame()
    if M == 2:
        results_full = result.eval(time = time).reshape((2,-1))
        init_full = result.eval(time=time, params=params).reshape((2,-1))
        df_true['trace1_fit'] = results_full[0]
        df_true['trace2_fit'] = results_full[1]
        df_true['trace1_init'] = init_full[0]
        df_true['trace2_init'] = init_full[1]
    elif M == 3:
        results_full = result.eval(time = time).reshape((3,-1))
        init_full = result.eval(time=time, params=params).reshape((3,-1))
        df_true['trace1_fit'] = results_full[0]
        df_true['trace2_fit'] = results_full[1]
        df_true['trace1_init'] = init_full[0]
        df_true['trace2_init'] = init_full[1]
        df_true['trace3_fit'] = results_full[2]
        df_true['trace3_init'] = results_full[2]
    if plotfit:
        fig,ax = plt.subplots(figsize = [10,10])            
        ax.set_ylabel("Intensity (arb. units)", fontsize = 27)
        if normalize:
            ax.set_ylabel("Normalized Intensity", fontsize = 27)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize= 24)
        ax.tick_params(axis='y', labelsize= 24)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel('Time, '+timescale, fontsize = 27)
        scatter_color_range = [plt.cm.inferno(i) for i in np.linspace(0.1, 0.9, len(traces[0,1:])+1)]
        maxtime = timelim
        if maxtime > 0.1 and maxtime <= 0.5:
                ax.xaxis.set_major_locator(MultipleLocator(0.05))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        if maxtime > 0.01 and maxtime <= 0.1:
            ax.xaxis.set_major_locator(MultipleLocator(0.01))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.001))
        if maxtime <= 0.01:
            ax.xaxis.set_major_locator(MultipleLocator(0.001))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        if maxtime > 0.5 and maxtime <= 1:
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        if maxtime > 1 and maxtime <= 5:
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        if maxtime > 5:
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        if maxtime >= 10:
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        if maxtime >= 25:
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        for i in range(1, len(traces[0,1:])+1):
            ax.scatter(time, traces[:,i], s = 60, facecolor = 'none', edgecolor = scatter_color_range[i], linewidth = 2)
        ax.legend(labels=[str(min(region)) + " to " + str(max(region)) for region in regions], fontsize= 27)
        ax.plot(time, df_true['trace1_fit'], color = 'black', linewidth = 3)
        ax.plot(time, df_true['trace2_fit'], color = 'black', linewidth = 3)
        if M == 3:
            ax.plot(time, df_true['trace3_fit'], color = 'black', linewidth = 3)
        if logscale:
            ax.set_yscale('log')
            ax.set_ylim(bottom = 10**(0))
            if normalize:
                ax.set_ylim(bottom = 10**(-3))
    return pd_pars

# test_s2 = GlobalFitting(s2, [(380, 410), (425,500), (510, 600)], 50, 4, [0.05, 0.5, 2.7, 24], (5, 0.2), 200, 'ns', normalize=False, logscale = True, plotfit = True)
#%%

def ImportFolderandSortbyPower(time, wavelength):
    """
    This function is meant for importing a collection of data taken with
    constant wavelength and time but variable power. Returns the data as an 
    ordered dictionary

    Parameters
    ----------
    time: string for filename
        string pointing to matrix with a time vector of length m
    wavelength: string for filename
        string pointing to matrix with a wavelength vector of length n
        
    Returns
    -------
    An OrderedDict of (m+1)x(n+1) matrices with numerical x,y,z data for wavelength, time, counts.
    Relies on StreakCamImageImport function defined earlier.
    List keys ranked with lowest power first.

    """
    # import the data
    result = {}
    for file in os.listdir():
        if file.endswith(".dat"):
            label = str(file)[:-4]
            result[label] = StreakCamImageImport(file, time, wavelength)
    reordered = OrderedDict()
    powers = []
    # section below isolates the power from the filename string
    for key in result.keys():
        powers.append(int(key[4:-2]))
        keyHandle1 = key[:4]
        keyHandle2 = key[-2:]
    # sort the powers
    powers.sort()
    # reinsert keys into ordered dictionary
    for power in powers:
        reordered[power] = result[keyHandle1+str(power)+keyHandle2]
    return reordered
S18 = ImportFolderandSortbyPower('time_10ns.csv', 'wavelength.csv')

#%%
def PDkinetics(data_tuple, align, timeshifts=False, labels=["placeholder"], timescale='ns', linearized = False, assumed_tau = 1, logscale_y = False, logscale_x = False, mintime = False, maxtime = False, region = False, normalize = True):
    """
    This function is used to plot kinetics integrated over a specific region of the overall streak matrix.

    Parameters
    ----------
    data_tuple : tuple of array-like
        A tuple of matrices. In each matrix:
        first row must be wavelength
        first column must be time
        [0,0] position is empty
        [1:, 1:] must be data
    timeshifts : array-like, 1d
        A list of floats to shift the signal in time for alignment purposes. 
        The default is False - the time axis is not shifted.
    labels : array-like, 1d
        A list of labels to use for the legend. The default is ["placeholder"].
    timescale : string, optional
        units of the time axis. The default is 'ns'.
    logscale : boolean, optional
        If True, sets the y-axis to logarithmic scale. The default is False.
    maxtime : float, optional
        If non-False, will cut time axis at the specified float value. The default is False.
    region : tuple of integers, optional
        If non-False, will cut the wavelength axis to the specified region. The default is False.

    Returns
    -------
    None.

    """
    exampleDataset = data_tuple[list(data_tuple.keys())[0]]
    exampleTime = exampleDataset[1:,0]
    outputArray = np.zeros((len(exampleTime), 1+len(data_tuple.keys())))
    if type(data_tuple) == OrderedDict:
        data_tuple = tuple(data_tuple.values())
        if labels == ['placeholder']:
            labels = data_tuple.keys()
    if timeshifts == False:
        timeshifts = np.zeros(len(data_tuple))
    i = 0
    for dataset in data_tuple:
        x = dataset[0,1:]
        y = dataset[1:,0]
        # print("check 1")
        # print(len(y))
        z = dataset[1:,1:]
        if region != False:
            idx_x_min = (np.abs(x - region[0])).argmin()
            idx_x_max = (np.abs(x - region[1])).argmin()
            x = x[idx_x_min:idx_x_max]
            z = z[:, idx_x_min:idx_x_max]
        if maxtime != False:
            idx_y_max = (np.abs(y - maxtime)).argmin()
            y = y[:idx_y_max]
            z = z[:idx_y_max, :]
        if normalize == True:
            kinetic_signal = normalize_1(np.sum(z, axis=1))  
            yAxisLabel = 'Normalized Intensity'
        else:
            kinetic_signal = np.sum(z, axis=1)
            yAxisLabel = 'Intensity (A.U.)'
        outputArray[:,i+1] = kinetic_signal
        i+=1
    outputArray[:,0] = y
    if align:
        t = outputArray[:,0]
        unaligned = outputArray[:,1:]
        m,n = unaligned.shape
        median = np.median(unaligned.argmax(0))
        col_shifts = median - unaligned.argmax(0)
        row_idx = (np.arange(m)[:,None]-col_shifts).clip(min=0,max=m-1)
        row_idx = np.array(row_idx, dtype = int)
        aligned_out = unaligned[row_idx,np.arange(n)]
        dset = np.concatenate([t[:,None], aligned_out], axis=1)
    else:
        dset = outputArray.copy()
    t = dset[:,0]
    fig, ax = plt.subplots(figsize=(12,10))
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
    ax.set_xlabel('Time ('+timescale+")", fontsize = 27)
    if maxtime != False:
        ax.set_xlim(left=  None, right = maxtime)
        if mintime != False:
            ax.set_xlim([mintime, maxtime])
    elif maxtime == False:
        ax.set_xlim([0, max(y)])
    ax.set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, len(data_tuple))])
    if max(y) > 0.1 and max(y) <= 0.5:
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    if max(y) > 0.01 and max(y) <= 0.1:
        ax.xaxis.set_major_locator(MultipleLocator(0.01))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    if max(y) <= 0.01:
        ax.xaxis.set_major_locator(MultipleLocator(0.001))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    if max(y) > 0.5 and max(y) <= 1:
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    if max(y) > 1 and max(y) <= 5:
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    if max(y) > 5:
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    if max(y) >= 10:
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    if max(y) >= 25:
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    i = 1
    while i <= len(data_tuple):
        kinetic_signal = dset[:,i]
        ax.plot(t, kinetic_signal, linewidth = 3)
        ax.set_ylabel(f"{yAxisLabel}", fontsize = 27)
        i+=1
    ncol = 1
    if len(labels) >= 8:
        ncol = 3
    ax.legend(labels = labels, ncol = ncol, loc = 'upper right', fontsize = 20)
    if logscale_y != False:
        print('y_scale is logarithmic')
        ax.set_yscale('log')
    if logscale_x != False:
        print('x_scale is logarithmic')
        ax.set_xscale('log')
        ax.set_xlim([10**(-.2),10**(1.1)])
    ax.axhline(y=0, color = 'dimgrey', ls = '--')
    return dset
    

S18traces = PDkinetics(S18, 
                       labels = [f"{list(S18.keys())[i]}" for i in range(12)], 
                       align = True,
                       logscale_y = True, 
                       logscale_x = False, 
                       normalize = True, 
                       region = (400, 660))
#labels = [str(key) for key in S14.keys()
#%% Test for Annihilation
def AnniTest(data_tuple, tau, amps, align = True, timeshifts=False, labels=["placeholder"], timescale='ns', linearized = False, assumed_tau = 1, logscale_y = False, logscale_x = False, mintime = False, maxtime = False, region = False, normalize = True):
    """
    This function is used to plot kinetics integrated over a specific region of the overall streak matrix.

    Parameters
    ----------
    data_tuple : tuple of array-like
        A tuple of matrices. In each matrix:
        first row must be wavelength
        first column must be time
        [0,0] position is empty
        [1:, 1:] must be data
    tau : a vector of tau values to use
    timeshifts : array-like, 1d
        A list of floats to shift the signal in time for alignment purposes. 
        The default is False - the time axis is not shifted.
    labels : array-like, 1d
        A list of labels to use for the legend. The default is ["placeholder"].
    timescale : string, optional
        units of the time axis. The default is 'ns'.
    logscale : boolean, optional
        If True, sets the y-axis to logarithmic scale. The default is False.
    maxtime : float, optional
        If non-False, will cut time axis at the specified float value. The default is False.
    region : tuple of integers, optional
        If non-False, will cut the wavelength axis to the specified region. The default is False.

    Returns
    -------
    None.

    """
    exampleDataset = data_tuple[list(data_tuple.keys())[0]]
    exampleTime = exampleDataset[1:,0]
    outputArray = np.zeros((len(exampleTime), 1+len(data_tuple.keys())))
    if type(data_tuple) == OrderedDict:
        data_tuple = tuple(data_tuple.values())
        if labels == ['placeholder']:
            labels = data_tuple.keys()
    if timeshifts == False:
        timeshifts = np.zeros(len(data_tuple))
    i = 0
    for dataset in data_tuple:
        x = dataset[0,1:]
        y = dataset[1:,0]
        # print("check 1")
        # print(len(y))
        z = dataset[1:,1:]
        if region != False:
            idx_x_min = (np.abs(x - region[0])).argmin()
            idx_x_max = (np.abs(x - region[1])).argmin()
            x = x[idx_x_min:idx_x_max]
            z = z[:, idx_x_min:idx_x_max]
        if maxtime != False:
            idx_y_max = (np.abs(y - maxtime)).argmin()
            y = y[:idx_y_max]
            z = z[:idx_y_max, :]
        if normalize == True:
            kinetic_signal = normalize_1(np.sum(z, axis=1))  
            yAxisLabel = 'Normalized Intensity'
        else:
            kinetic_signal = np.sum(z, axis=1)
            yAxisLabel = 'Intensity (A.U.)'
        outputArray[:,i+1] = kinetic_signal
        i+=1
    outputArray[:,0] = y
    if align:
        t = outputArray[:,0]
        unaligned = outputArray[:,1:]
        m,n = unaligned.shape
        median = np.median(unaligned.argmax(0))
        col_shifts = median - unaligned.argmax(0)
        row_idx = (np.arange(m)[:,None]-col_shifts).clip(min=0,max=m-1)
        row_idx = np.array(row_idx, dtype = int)
        aligned_out = unaligned[row_idx,np.arange(n)]
        dset = np.concatenate([t[:,None], aligned_out], axis=1)
    else:
        dset = outputArray.copy()
    t1 = dset[:,0]
    t2 = dset[:,0]
    x1 = np.exp(t1/tau[0])
    x2 = np.exp(t2/tau[1])
    x3 = np.exp(t2/tau[2])
    # tRoot = np.sqrt(dset[:,0])
    # fig = plt.figure(figsize=plt.figaspect(0.1)*3)
    # ax = fig.add_subplot(projection='3d')
    fig,ax = plt.subplots(3,1,figsize = (10,10), sharey=False)
    ax[0].tick_params(axis='both', labelsize= 24)
    ax[1].tick_params(axis='both', labelsize= 24)
    ax[2].tick_params(axis='both', labelsize= 24)
    # ax.tick_params(axis='z', labelsize= 16)
    ax[0].set_xlabel(r'exp(t/${\tau}_1$)', fontsize = 24, labelpad = 10)
    ax[1].set_xlabel(r'exp(t/${\tau}_2$)', fontsize = 24, labelpad = 10)
    ax[2].set_xlabel(r'exp(t/${\tau}_{avg}$)', fontsize = 24, labelpad = 10)
    # ax.set_xlabel(r"t$^{0.5}$", fontsize = 16)
    ax[0].set_ylabel("1/n(t)", fontsize = 24)
    ax[1].set_ylabel("1/n(t)", fontsize = 24)
    ax[2].set_ylabel("1/n(t)", fontsize = 24)
    # if maxtime != False:
    #     ax.set_xlim(left=  None, right = maxtime)
    #     if mintime != False:
    #         ax.set_xlim([mintime, maxtime])
    # elif maxtime == False:
    #     ax.set_xlim([0, max(t)])
    ax[0].set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, len(data_tuple))])
    ax[1].set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, len(data_tuple))])
    ax[2].set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, len(data_tuple))])
    i = 1
    while i <= len(data_tuple):
        kinetic_signal = (1/dset[:,i])
        # ks1 = 1/(np.multiply(amps[i-1].iloc[0], np.exp(-t1/tau[0])))
        # ks2 = 1/(np.multiply(amps[i-1].iloc[1], np.exp(-t2/tau[1])))
        ax[0].scatter(x1[45:], sgf(kinetic_signal[45:], 9,3))
        ax[1].scatter(x2[45:], sgf(kinetic_signal[45:], 9,3))
        ax[2].scatter(x3[45:], sgf(kinetic_signal[45:], 9,3))
        # ax.scatter(tRoot[40:], sgf(kinetic_signal[40:], 9,3))
        i+=1
    ncol = 1
    if len(labels) >= 8:
        ncol = 3
    # ax[0].legend(labels = labels, ncol = 3, loc = 'upper left', fontsize = 18)
    # ax.set_box_aspect((np.ptp(t2[45:]), np.ptp(t1[45:]), np.ptp(kinetic_signal[45:])))  # aspect ratio is 1:1:1 in data space
    ax[0].set_xlim([2, 500])
    ax[1].set_xlim([785, 227142])
    ax[2].set_xlim([11, 1400])
    # ax[1].set_xscale('log')
    ax[0].set_ylim([0, 0.5])
    ax[1].set_ylim([0, 0.015])
    ax[2].set_ylim([0, 0.04])
    plt.tight_layout()
    return np.transpose(np.vstack([x1, x2, x3, kinetic_signal]))
    

S18AnniLines = AnniTest(S18, tau = [1.5, 0.15, 0.40], 
                        amps = [s18_results_09062023[key]['best-fit value'][1:3] for key in list(s18_results_09062023.keys())], 
                       labels = [f"{S18_Report['Excitation Probabilities'][i]:.3f}" for i in range(12)], 
                       align = False,
                       logscale_y = False, 
                       logscale_x = False, 
                       normalize = False, 
                       region = (400, 660))
#labels = [str(key) for key in S14.keys()
#%% Align Normalized Kinetics at the Max
def alignAndGetDeltaTraces(unaligned_dataset, scaling = 'none', labels=["placeholder"], timescale='ns', standard='low', xlim='default', logscale_y = False, logscale_x = False):
    unaligned = unaligned_dataset[:,1:]
    t = unaligned_dataset[:,0]
    m,n = unaligned.shape
    median = np.median(unaligned.argmax(0))
    col_shifts = median - unaligned.argmax(0)
    row_idx = (np.arange(m)[:,None]-col_shifts).clip(min=0,max=m-1)
    row_idx = np.array(row_idx, dtype = int)
    aligned_out = unaligned[row_idx,np.arange(n)]
    if standard == 'low':
        stdTrace = aligned_out[:,0]
    elif standard == 'high':
        stdTrace = aligned_out[:,-1]
    if scaling != 'none':
        for i in range(len(aligned_out[0,:])):
            aligned_out[:,i] *= scaling[i]
    fig = plt.figure(figsize = (10,13.3))
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[:3, 0])
    ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.tick_params(axis='x', labelsize= 24)
    # ax.tick_params(axis='y', labelsize= 24)
    ax2.set_xlabel('Time ('+timescale+")", fontsize = 27)
    ax1.set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, n)])
    for i in range(len(aligned_out[0,:])):
        ax1.plot(t, aligned_out[:,i], linewidth = 3)
    ax1.set_ylabel("Norm[I]", fontsize = 27)
    ax1.tick_params(axis='y', labelsize= 24)
    ax2.set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, n)])
    for i in range(len(aligned_out[0,:])):
        ax2.plot(t, aligned_out[:,i]-stdTrace, linewidth = 3)
    ax2.set_ylabel("$\Delta$Norm[I]", fontsize = 27)
    ax2.tick_params(axis='y', labelsize= 24)
    ncol = 1
    if len(labels) >= 10:
        ncol = 2
    ax1.legend(labels = labels, ncol = ncol, loc = 'upper right', fontsize = 24)
    if logscale_y != False:
        print('y_scale is logarithmic')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    if logscale_x != False:
        print('x_scale is logarithmic')
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    if xlim!='default':
        ax1.set_xlim(xlim)
    ax1.axhline(y=0, color = 'dimgrey', ls = '--')
    ax2.axhline(y=0, color = 'dimgrey', ls = '--')
    delta_traces = np.subtract(aligned_out,stdTrace[:,np.newaxis])
    return (aligned_out, delta_traces, t, stdTrace[:,np.newaxis])

S20_Deltas = alignAndGetDeltaTraces(S20traces, 
                              labels = [str(label)[:4]+r' $\mu$J/cm$^2$' for label in S20_Report['Fluence (uJ/cm2)']], 
                              timescale='ns', 
                              standard='low',
                              xlim = [0.5,10],
                              logscale_y = False, 
                              logscale_x = True)

#%% Working with the Delta Traces


# Clip the Data at Maximum - It's OK because these are delta traces
deltas = np.abs(NormDeltas[1][:,1:]) # exclude the zero trace
t = NormDeltas[2] # get the time axis back
traces_ToFit = deltas[:,2:]
N = traces_ToFit.shape[1]
maxPoint = np.argmax(traces_ToFit[:,-1])
endPoint = (np.abs(t - 1.5)).argmin()
clippedTraces = np.concatenate([[traces_ToFit[maxPoint:,i]]for i in range(N)], axis = 1)
t_clipped = t[maxPoint:]

# Build the Model with Lmfit
def natefunc(x, A, t0, tau, gamma):
    S = (A*np.exp(-(x - t0)/tau))/(1+A*gamma*tau*(1-np.exp(-(x - t0)/tau)))
    return S

def natefunc_(x, N, **Pars):
    res = {}
    tau = Pars['tau']
    t0 = Pars['t0']
    for i in range(1, N+1):
        res[f'T_{i}'] = np.zeros_like(x)
        gamma = Pars[f'y_{i}']
        A = Pars[f'A_{i}']
        res[f'T_{i}'] = (A*np.exp(-(x - t0)/tau))/(1+A*gamma*tau*(1-np.exp(-(x - t0)/tau)))
    return np.concatenate([res[f'T_{i}'] for i in range(1, N+1)])
model = lm.Model(natefunc_, independent_vars=['x', 'N'])
params = lm.Parameters()
params.add('t0', value = 0.8)
params.add('tau', value = 0.3)
for i in range(1, N+1):
    params.add(f'y_{i}', value = 0)
    params.add(f'A_{i}', value=0.4)

result = model.fit(clippedTraces, x = t_clipped, N=N, params=params)
print(result.fit_report())

results_full = result.eval(x = t_clipped).reshape((N,-1)).transpose()
samples = np.reshape(clippedTraces, (N, -1)).transpose()

fig, ax = plt.subplots(figsize = (10,10))
ax.set_xscale('log')
for i in range(N):
    ax.plot(t_clipped, samples[:,i], linewidth = 4, alpha = 0.5)
    ax.plot(t_clipped, results_full[:,i], linewidth = 3, color = 'black')

#%% Brian/Michael's convolution code
from numpy import fft

def gauss(t, A, t0, w):
    G = A * np.exp(-(t-t0)**2/(2*w**2)) / (w*np.sqrt(2*np.pi))
    return G
# convolve population dynamics curve d with Gaussian curve g
def conv(t, t0, d, g):
# zero pad curves d and g to 2**N for N time delays
    np2 = int(np.ceil(np.log2(t.size*2)))
    i0 = np.absolute(t-t0).argmin()
    dp = np.concatenate((d, np.zeros(2**np2-d.size)))
    gp = np.concatenate((g, np.zeros(2**np2-g.size)))
# do the convolution
    C = np.real(fft.ifft(fft.fft(dp)*fft.fft(gp)))
# normalize the result
    dt = np.average(t[1:]-t[:-1])
    gs = np.sum(gp*dt)
    C *= dt/gs
# select the subset of the output corresponding to the actual curve, i.e undo the zero padding
    C = C[i0:i0+t.size]
    return C
# This marks the end of Brian's code
def exp2dsine_noconv(t, A1, A2, tau1, tau2, Baseline, t0, Amp1, t1, w1, phase1, A, fwhm):
    t_extend = np.append((t[0]-(np.arange(1,int(len(t)*0.2))*(t[1]-t[0])))[::-1],t)
    t_full = np.append(t_extend,(t[-1]+(np.arange(1,int(len(t)*0.2))*(t[1]-t[0]))))
    y_exp = Baseline + (A1 * np.exp(-(t_full-t0)/tau1)) + (A2 * np.exp(-(t_full-t0)/tau2))
    y_sine = (Amp1 * np.exp(-t_full/t1) * np.sin((2 * np.pi * t_full/w1) + phase1))
    y_sine[t_full < t0] = 0
    y_exp[t_full < t0] = Baseline
    convolution_out = conv(t_full, t0, y_exp + y_sine, gauss(t_full, A, t0, fwhm))
    #return y_exp + y_sine
    #return conv(t, t0, y_exp, gauss(t, A, t0, fwhm))
    return convolution_out[np.where(t_full == t[0])[0][0]:np.where(t_full == t[-1])[0][0] + 1]

#%% convolution: Tyler's Recipe
def Annihilation(time, fwhm, **Pars):
    x = time
    res = {}
    A1 = Pars['A1']
    A2 = Pars['A2']
    tau1 = Pars['tau1']
    tau2 = Pars['tau2']
    t0 = Pars['t0']

#%%
def QuickPDKineticsFit(data_dictionary, region, timelim, initial_tau, fwhm = 0.1, A_value = 1, timescale = 'ns', normalize = False, normfactors = 'default', logscale = True, plotfit = True):
    N = len(data_dictionary.keys())
    powers = list(data_dictionary.keys())
    example_dataset = data_dictionary[powers[0]]
    time = example_dataset[1:,0]
    for key in powers:
        traces = TraceExtractor(data_dictionary[key], region, normalize, normfactors = normfactors, plotTraces = False)
        if key == powers[0]:
            samples = traces[:,1]
        else:
            samples = np.concatenate([samples, traces[:,1]])
    print(np.shape(samples))
    def SEMG_Model(time, N, fwhm, **Pars):
        x = time
        res = {}
        # single exp, single exp, bimolecular, irf
        for i in range(1, N+1):
            res[f'T_{i}'] = np.zeros_like(time)
        for i in range(1, N+1):
            A1 = Pars[f'A1_{i}']
            A2 = Pars[f'A2_{i}']
            A3 = Pars[f'A3_{i}']
            tau1 = Pars[f'tau1_{i}']
            tau2 = Pars[f'tau2_{i}']
            tau3 = Pars[f'tau3_{i}']
            t0 = Pars[f't0_{i}']
            # beta1 = Pars[f'beta1_{i}']
            # beta2 = Pars[f'beta2_{i}']
            # gamma = Pars[f'y0_{i}']
            # AG = Pars[f'AG_{i}']
            func1 = A1*fwhm*np.exp((fwhm/(np.sqrt(2)*tau1))**2-(x-t0)/tau1)*(1-special.erf(fwhm/(np.sqrt(2)*tau1)-(x-t0)/(np.sqrt(2)*fwhm)))
            func2 = A2*fwhm*np.exp((fwhm/(np.sqrt(2)*tau2))**2-(x-t0)/tau2)*(1-special.erf(fwhm/(np.sqrt(2)*tau2)-(x-t0)/(np.sqrt(2)*fwhm)))
            func3 = A3*fwhm*np.exp((fwhm/(np.sqrt(2)*tau3))**2-(x-t0)/tau3)*(1-special.erf(fwhm/(np.sqrt(2)*tau3)-(x-t0)/(np.sqrt(2)*fwhm)))
            funcSum = func1+func2+func3
            res[f'T_{i}'] = funcSum
        return np.concatenate([res[f'T_{i}'] for i in range(1, N+1)])
    def SEMG_Model_Anni(time, N, fwhm, **Pars):
        x = time
        res = {}
        # stretch exponentially modified gaussian
        for i in range(1, N+1):
            res[f'T_{i}'] = np.zeros_like(time)
        for i in range(1, N+1):
            A1 = Pars[f'A1_{i}']
            tau1 = Pars[f'tau1_{i}']
            t0 = Pars[f't0_{i}']
            beta = Pars[f'beta_{i}']
            # A0 = Pars[f'A0_{i}']
            gamma = Pars[f'y0_{i}']
            res[f'T_{i}'] = A1/(2*tau1) * np.exp(((1/(2*tau1))*(2*t0 + (fwhm**2)/tau1) - 2*x))**beta * special.erfc((t0+(fwhm**2)/tau1-x)/(np.sqrt(2)*fwhm)) + (A1*np.exp((-(x))/tau1))/(1+A1*gamma*tau1*(1-np.exp((-(x))/tau1)))
        return np.concatenate([res[f'T_{i}'] for i in range(1, N+1)])
    def SEMG_Model_Anni2(time, N, fwhm, **Pars):
        x = time
        res = {}
        # stretch exponentially modified gaussian
        for i in range(1, N+1):
            res[f'T_{i}'] = np.zeros_like(time)
        for i in range(1, N+1):
            A1 = Pars[f'A1_{i}']
            AG = Pars[f'AG_{i}']
            A2 = Pars[f'A2_{i}']
            tau1 = Pars[f'tau1_{i}']
            tau2 = Pars[f'tau2_{i}']
            t0 = Pars[f't0_{i}']
            beta = Pars[f'beta_{i}']
            # A0 = Pars[f'A0_{i}']
            gamma = Pars[f'y0_{i}']
            func1 = (AG*np.exp(-(x-t0)**2/fwhm**2))
            func2 = (A2*np.exp(-(x-t0)/tau2)**beta)
            func3 = A1*(1-gamma)/(np.exp((x-t0)/tau1) - gamma) 
            func4 = convolve(func1, func2)
            func5 = convolve(func4, func3)
            res[f'T_{i}'] = func5[:len(time)]
        return np.concatenate([res[f'T_{i}'] for i in range(1, N+1)])
    
        
    print(f'Fitting a power series ({N} datasets) to single Gaussian stretched exp. decays')
    model = lm.Model(SEMG_Model, independent_vars=['time'])
    params = lm.Parameters()
    params.add('N', value=N, vary=False) # sets a limit on the sum in the model
    tauGuess = initial_tau
    print(f'Using the following initial pars for tau: {tauGuess}')
    for i in range(1, N+1):
        params.add(f'A1_{i}', value=A_value[0], min = 0)
        params.add(f'A2_{i}', value=A_value[1], min = 0)
        params.add(f'A3_{i}', value=A_value[2], min = 0)
        # params.add(f'A0_{i}', value=0, min = 0)
        params.add(f'tau1_{i}', value = tauGuess[0], min = 0.1, max = 10)
        params.add(f'tau2_{i}', value = tauGuess[1], min = 0.1, max = 10)
        params.add(f'tau3_{i}', value = tauGuess[2], min = 0.1, max = 10)
        params.add(f't0_{i}', value = 0.75, min = 0, max = 1)
        # params.add(f'beta1_{i}', value = 0.5, min = 0, max = 1)
        # params.add(f'beta2_{i}', value = 0.5, min = 0, max = 1)
        # params.add(f'y0_{i}', value = 0, min = 0, max = 1)
    params.add('fwhm', value = fwhm, min = 0.01, max = 1)
    print('Model parameters constructed')
    print('Fitting...')
    result = model.fit(samples, time = time, params=params)
    print(result.fit_report())
    pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))
    df_true = pd.DataFrame()
    results_full = result.eval(time = time).reshape((N,-1))
    init_full = result.eval(time=time, params=params).reshape((N,-1))
    samples = np.transpose(np.reshape(samples, (N, -1)))
    # print(np.shape(samples))
    for i in range(1,N+1):
        df_true[f'trace{i}_fit'] = results_full[i-1]
        df_true[f'trace{i}_init'] = init_full[i-1]
    if plotfit:
        fig,ax = plt.subplots(figsize = [10,10])            
        ax.set_ylabel("Intensity (arb. units)", fontsize = 27)
        if normalize:
            ax.set_ylabel("Normalized Intensity", fontsize = 27)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize= 24)
        ax.tick_params(axis='y', labelsize= 24)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel('Time, '+timescale, fontsize = 27)
        scatter_color_range = [plt.cm.inferno(i) for i in np.linspace(0.1, 0.9, len(samples[0,:]))]
        maxtime = timelim
        if maxtime > 0.1 and maxtime <= 0.5:
                ax.xaxis.set_major_locator(MultipleLocator(0.05))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        if maxtime > 0.01 and maxtime <= 0.1:
            ax.xaxis.set_major_locator(MultipleLocator(0.01))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.001))
        if maxtime <= 0.01:
            ax.xaxis.set_major_locator(MultipleLocator(0.001))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        if maxtime > 0.5 and maxtime <= 1:
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        if maxtime > 1 and maxtime <= 5:
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        if maxtime > 5:
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        if maxtime >= 10:
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        if maxtime >= 25:
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        # print(np.shape(time))
        # print(np.shape(samples))
        for i in range(len(samples[0,:])):
            ax.scatter(time, samples[:,i], s = 60, facecolor = 'none', edgecolor = scatter_color_range[i], linewidth = 2)
        ax.legend(labels=powers, fontsize= 27)
        for i in range(1, N+1):
            ax.plot(time, df_true[f'trace{i}_fit'], color = 'black', linewidth = 3)
        if logscale:
            ax.set_yscale('log')
            ax.set_ylim(bottom = 10**(0))
            if normalize:
                ax.set_ylim(bottom = 10**(-3), top = 10**(0.1))
    return pd_pars



test2 = QuickPDKineticsFit({150: S17[150]}, [(400,500)], 10, initial_tau = (0.5, 1, 3), fwhm = 0.25, normalize = True, logscale = True, A_value = (1, 1, 1))


#%% testing functions: 3 exp fit on a single trace!
test = TraceExtractor(S17[510], [(400,500)], normalize = True, normfactors = 'default', plotTraces = False)
trace = test[:,1]
time = test[:,0]

def noAnni(time, fwhm, **Pars):
    x = time
    # single exp, single exp, bimolecular, irf
    A1 = Pars['A1']
    A2 = Pars['A2']
    A3 = Pars['A3']
    tau1 = Pars['tau1']
    tau2 = Pars['tau2']
    tau3 = Pars['tau3']
    t0 = Pars['t0']
    func1 = A1*fwhm*np.exp((fwhm/(np.sqrt(2)*tau1))**2-(x-t0)/tau1)*(1-special.erf(fwhm/(np.sqrt(2)*tau1)-(x-t0)/(np.sqrt(2)*fwhm)))
    func2 = A2*fwhm*np.exp((fwhm/(np.sqrt(2)*tau2))**2-(x-t0)/tau2)*(1-special.erf(fwhm/(np.sqrt(2)*tau2)-(x-t0)/(np.sqrt(2)*fwhm)))
    func3 = A3*fwhm*np.exp((fwhm/(np.sqrt(2)*tau3))**2-(x-t0)/tau3)*(1-special.erf(fwhm/(np.sqrt(2)*tau3)-(x-t0)/(np.sqrt(2)*fwhm)))
    funcSum = func1+func2+func3
    res = funcSum
    return res
    
model = lm.Model(noAnni, independent_vars=['time'])
params = lm.Parameters()
params.add('A1', value=13, min = 0)
params.add('A2', value=1, min = 0)
params.add('A3', value=0.7, min = 0)
params.add('tau1', value = 0.2, min = 0.1, max = 10)
params.add('tau2', value = 0.5, min = 0.1, max = 10)
params.add('tau3', value = 1.5, min = 0.1, max = 10)
params.add('t0', value = 0.8, min = 0, max = 1)
params.add('fwhm', value = 0.067, min = 0.01, max = 1)
print('Model parameters constructed')
print('Fitting...')
result = model.fit(trace, time = time, params=params)
print(result.fit_report())

pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))

results_full = result.eval(time = time)
init_full = result.eval(time=time, params=params)

fig,ax = plt.subplots(figsize = [10,10])            
ax.set_ylabel("Intensity (arb. units)", fontsize = 27)
ax.set_ylabel("Normalized Intensity", fontsize = 27)
ax.set_yticklabels([])
ax.set_yticks([])
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_xlabel('Time, ns', fontsize = 27)
ax.set_yscale('log')
ax.set_ylim(bottom = 10**(-3), top = 10**(0.1))
ax.scatter(time, trace, s = 60, facecolor = 'none', edgecolor = 'black', linewidth = 2, label = 'Data')
ax.plot(time, results_full, color = 'red', linewidth = 3, label = 'Fit')
ax.plot(time, init_full, color = 'blue', linewidth = 3, label = 'Init')

ax.legend(loc = 'upper right', fontsize = 24)


#%% testing functions: annihilation!!!
test = TraceExtractor(S17[3060], [(400,500)], normalize = True, normfactors = 'default', plotTraces = False)
trace = test[:,1]
time = test[:,0]
def gauss(t, A, t0, w):
    G = A * np.exp(-(t-t0)**2/(2*w**2)) / (w*np.sqrt(2*np.pi))
    return G

def conv(t, t0, d, g):
# zero pad curves d and g to 2**N for N time delays
    np2 = int(np.ceil(np.log2(t.size*2)))
    i0 = np.absolute(t-t0).argmin()
    dp = np.concatenate((d, np.zeros(2**np2-d.size)))
    gp = np.concatenate((g, np.zeros(2**np2-g.size)))
# do the convolution
    C = np.real(fft.ifft(fft.fft(dp)*fft.fft(gp)))
# normalize the result
    dt = np.average(t[1:]-t[:-1])
    gs = np.sum(gp*dt)
    C *= dt/gs
# select the subset of the output corresponding to the actual curve, i.e undo the zero padding
    C = C[i0:i0+t.size]
    return C

def Anni(time, gamma, fwhm, **Pars):
    x = time
    # single exp, single exp, bimolecular, irf
    A1 = Pars['A1']
    A2 = Pars['A2']
    A3 = Pars['A3']
    A4 = Pars['A4']
    tau1 = Pars['tau1']
    tau2 = Pars['tau2']
    tau3 = Pars['tau3']
    tau4 = Pars['tau4']
    t0 = Pars['t0']
    Heaviside = np.zeros_like(x)
    gaussian = gauss(x, A1, t0, fwhm)
    pos = (np.abs(x - t0)).argmin()
    Heaviside[pos:] += 1
    intermediate = np.multiply(Heaviside, (A1*np.exp(-(x-t0)/tau1))/(1+A1*gamma*tau1*(1-np.exp(-(x-t0)/tau1))))
    func1 = conv(x, t0, intermediate, gaussian)
    func2 = A2*fwhm*np.exp((fwhm/(np.sqrt(2)*tau2))**2-(x-t0)/tau2)*(1-special.erf(fwhm/(np.sqrt(2)*tau2)-(x-t0)/(np.sqrt(2)*fwhm)))
    func3 = A3*fwhm*np.exp((fwhm/(np.sqrt(2)*tau3))**2-(x-t0)/tau3)*(1-special.erf(fwhm/(np.sqrt(2)*tau3)-(x-t0)/(np.sqrt(2)*fwhm)))
    func4 = A4*fwhm*np.exp((fwhm/(np.sqrt(2)*tau4))**2-(x-t0)/tau4)*(1-special.erf(fwhm/(np.sqrt(2)*tau4)-(x-t0)/(np.sqrt(2)*fwhm)))
    funcSum = func1+func2+func3+func4
    res = funcSum
    return res
    
model = lm.Model(Anni, independent_vars=['time'])
params = lm.Parameters()
params.add('A1', value=1, min = 0)
params.add('A2', value=13, min = 0)
params.add('A3', value=4.5, min = 0)
params.add('A4', value=0.4, min = 0)
params.add('tau1', value = 0.05, min = 0.05, max = 10)
params.add('tau2', value = 0.1, min = 0.05, max = 10)
params.add('tau3', value = 0.2, min = 0.05, max = 10)
params.add('tau4', value = 1.5, min = 0.05, max = 10)
params.add('t0', value = 0.83, min = 0, max = 1)
params.add('fwhm', value = 0.05, min = 0.01, max = 1)
params.add('gamma', value = 10000, min = 0.0)
print('Model parameters constructed')
print('Fitting...')
result = model.fit(trace, time = time, params=params)
print(result.fit_report())

pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))

results_full = result.eval(time = time)
init_full = result.eval(time=time, params=params)

fig,ax = plt.subplots(figsize = [10,10])            
ax.set_ylabel("Intensity (arb. units)", fontsize = 27)
ax.set_ylabel("Normalized Intensity", fontsize = 27)
ax.set_yticklabels([])
ax.set_yticks([])
ax.tick_params(axis='x', labelsize= 24)
ax.tick_params(axis='y', labelsize= 24)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_xlabel('Time, ns', fontsize = 27)
ax.set_yscale('log')
ax.set_ylim(bottom = 10**(-3), top = 10**(0.2))
ax.scatter(time, trace, s = 60, facecolor = 'none', edgecolor = 'black', linewidth = 2, label = 'Data')
ax.plot(time, results_full, color = 'red', linewidth = 3, label = 'Fit')
ax.plot(time, init_full, color = 'blue', linewidth = 3, label = 'Init')

ax.legend(loc = 'upper right', fontsize = 24)
#%% excitation probability

def ExcitationPowerReport(powers, OD, epsilon, gausDiameter, nmax, cellVolume, exWl = 400, PL = 10, reprate = 2000):
    """

    Parameters
    ----------
    powers : array-like
        a 1d array with excitation powers in microwatts
    OD : float
        optical density of the sample studied, unitless
    epsilon : int or float
        "molar" absorptivity, in L/(mol*cm)
    gausDiameter : int or float
        pump beam FWHM, in micrometers
    nmax : int
        number of excitations to compute probabilities for
        note: large values of nmax will lead to large reports
    cellVolume : float
        unit cell volume in angstroms cubed
    exWl : int
        excitation wavelength in nanometers. The default is 400.
    PL : int or float
        cuvette path length in millimeters, default is 10
    reprate : int
        pulsed laser repetition rate in Hz. The default is 2000.

    Returns
    -------
    report:  pandas DataFrame
        contains the initial powers, energies, excitation probabilities and average predicted excitation density on the basis of the molar absorptivity

    """
    # set up an output dataframe
    report = pd.DataFrame()
    # generate energies per pulse in SI
    energy_per_pulse = [float(i)*10**(-6)/reprate for i in powers]
    report['Power (uW)'] = powers
    report['Energy Per Pulse (nJ)'] = np.multiply(energy_per_pulse, 10**9)
    # calculate the attenuation coefficient for beam in SI
    alpha = OD*np.log(10)/(PL*10**(3)) # micron^-1
    # set up the constants
    hPlanck = 6.62606896*10**(-34) # SI
    cSpeed = 2.99792458*10**8 # SI
    # maximum number of excitations per formula unit (int)
    # in NU-1000, formula unit has 2 linkers
    nmax = nmax # excitations per unit cell
    # approximate absorption cross section from the molar absorptivity
    sigma = 3.82*10**(-21)*epsilon*10**(8) # um2 if epsilon in L/(mol*cm)
    print(f'the estimated sigma is {sigma} um2')
    numberPumpPhotons = [(energy*exWl*10**(-9))/(hPlanck*cSpeed) for energy in energy_per_pulse]
    report['Number of Photons'] = numberPumpPhotons
    # compute the fluences in SI
    fluences_SI = [i/(2*np.pi*(0.5*(gausDiameter*10**(-6))**2)) for i in energy_per_pulse]
    # compute the fluences in microjoules per cm2 (standard)
    fluences_uJ_over_cm2 = [i*10**6/10**8 for i in fluences_SI]
    report['Fluence (uJ/cm2)'] = fluences_uJ_over_cm2
    # compute gaussian waist
    w = (gausDiameter)/np.sqrt(2*np.log(2)); # micron
    print(f'the Gaussian beam waist is {w} um')
    # compute the beam profile
    def HPu(r, z):
        # uses the values of w and alpha computed before - those are constants
        # r is radial angle, z is depth in propagation direction
        return (2/(np.pi*w**2)) * np.exp(-2*(r**2)/(w**2)) * np.exp(-alpha*z) # per um2
    hIntegral = dblquad(lambda r,z: HPu(r,z) * 2*np.pi*r, a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf)[0]
    print(f'H-integral is equal to {hIntegral} um^(-2)')
    photonFluence = [i * dblquad(lambda r, z: HPu(r, z), a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf)[0] for i in numberPumpPhotons]
    report['Fluence (photons/um2)'] = photonFluence
    # the above quantity should be photons per um2
    report['Fluence (photons/cm2)'] = np.multiply(photonFluence,10**8)
    excitationProbabilities = [i * sigma*10**(-8) * dblquad(lambda r, z: HPu(r, z), a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf)[0] for i in numberPumpPhotons]
    report['Excitation Probabilities'] = excitationProbabilities
    excitationProbability_00 = [i * sigma * HPu(0,0) for i in numberPumpPhotons]
    # excitationProbability_00 = [i * sigma * dblquad(lambda r, z: HPu(r, z), a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf)[0] for i in numberPumpPhotons]
    report['Neh (0,0)'] = excitationProbability_00
    # units of the above quantity... not sure
    # for n in range(nmax+1):
    #     Probability_n = [(dblquad(lambda r, z: (i) * HPu(r, z) * sigma, a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf)[0]**n * np.exp(-dblquad(lambda r, z: (i) * HPu(r, z) * sigma, a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf)[0]))/np.math.factorial(n) for i in numberPumpPhotons]
    #     report[f'P_{n}'] = Probability_n

    for n in range(nmax+1):
        Probability_n = [(p**n * np.exp(-p))/np.math.factorial(n) for p in excitationProbabilities]
        report[f'P_{n}'] = Probability_n
    # for n in range(nmax+1):
    #     Probability_n_avg = [lambda r,z: i * 2*np.pi*r for i in Probability_n]
    #     report[f'<P>_{n}'] = Probability_n_avg   
    result = []
    for p in excitationProbabilities:
        result.append(np.sum([[n*(p**n * np.exp(-p))/np.math.factorial(n) for n in range(nmax+1)]]))
    report['Average Excitation Probability (per cell)'] = result
    # the above value is per unit cell
    # you may need it per cm3
    cells = ((1e-2)**3)/(cellVolume*(1e-10)**3) # cells per cm3
    report['Average Exciton Density'] = np.multiply(result,cells)
    return report
S19_wattages = list(S19.keys())
S19_Report = ExcitationPowerReport(S19_wattages, 0.394, 75000, 142, nmax = 6, cellVolume = 22391, PL=10)


#%% excitation probability v2
S18_wattages = list(S18.keys())
def ExcitationPowerReport_v2(powers,  OD, epsilon, gausDiameter, nmax, exWl = 400, PL = 10, reprate = 2000):
    """

    Parameters
    ----------
    powers : array-like
        a 1d array with excitation powers in microwatts
    OD : float
        optical density of the sample studied, unitless
    epsilon : int or float
        "molar" absorptivity, in L/(mol*cm)
    gausDiameter : int or float
        pump beam FWHM, in micrometers
    nmax : int
        number of excitations to compute probabilities for
        note: large values of nmax will lead to large reports
    exWl : int
        excitation wavelength in nanometers. The default is 400.
    PL : int or float
        cuvette path length in millimeters, default is 10
    reprate : int
        pulsed laser repetition rate in Hz. The default is 2000.

    Returns
    -------
    report:  pandas DataFrame
        contains the initial powers, energies, excitation probabilities and average predicted excitation density on the basis of the molar absorptivity

    """
    # set up an output dataframe
    report = pd.DataFrame()
    # generate energies per pulse in SI
    energy_per_pulse = [float(i)*10**(-6)/reprate for i in powers]
    # add POWERS to REPORT
    report['Power (uW)'] = powers
    # add ENERGIES to REPORT
    report['Energy Per Pulse (nJ)'] = np.multiply(energy_per_pulse, 10**9)
    # set up the constants
    hPlanck = 6.62606896*10**(-34) # SI
    cSpeed = 2.99792458*10**8 # SI
    nmax = nmax # excitations per formula unit (must be consistent with epsilon and formula unit)
    # approximate absorption cross section from the molar absorptivity
    print(f'the max number of excitations per formula unit is {nmax}')
    sigma = 3.82*10**(-21)*epsilon*10**(8) # um2 if epsilon in L/(mol*cm)
    # calculate the attenuation coefficient for beam in SI
    alpha = OD*np.log(10)/(PL*10**(3)) # micron^-1
    print(f'the estimated sigma is {sigma} um2')
    # calculate the number of pump photons and add to REPORT
    numberPumpPhotons = [(energy*exWl*10**(-9))/(hPlanck*cSpeed) for energy in energy_per_pulse]
    report['Number of Photons'] = numberPumpPhotons
    # compute the fluences in SI
    fluences_SI = [i/(2*np.pi*(0.5*(gausDiameter*10**(-6))**2)) for i in energy_per_pulse]
    # compute the fluences in microjoules per cm2 (standard) and REPORT
    fluences_uJ_over_cm2 = [i*10**6/10**8 for i in fluences_SI]
    report['Fluence (uJ/cm2)'] = fluences_uJ_over_cm2
    # compute gaussian waist
    w = (gausDiameter)/np.sqrt(2*np.log(2)); # micron
    print(f'the Gaussian beam waist is {w} um')
    # compute the beam profile
    HPu = lambda r,z: (2/(np.pi*w**2)) * np.exp(-2*(r**2)/(w**2)) * np.exp(-alpha*z) # per um2
    # compute photon fluences
    FPu = lambda r,z,E: Num(E) * HPu(r,z) # generic function
    FPu_00 = [i * HPu(0,0) for i in numberPumpPhotons] # at incidence
    FPu_rz = [dblquad(lambda r,z: i * HPu(r,z), a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf)[0] for i in numberPumpPhotons]
    report['Fluence-0 (photons)'] = FPu_00
    report['Fluence-rz (photons)'] = FPu_rz
    # compute the number of photons as a generic function
    Num = lambda E: (E*exWl*10**(-9))/(hPlanck*cSpeed)

    excitationProbabilities_00 = [sigma * FPu(0,0,E) for E in energy_per_pulse]
    report['Excitation Probabilities - 00'] = excitationProbabilities_00
    NEh = lambda r,z,E: sigma * FPu(r,z,E)
    NEh_rz = [dblquad(lambda r,z,E: sigma * NEh(r,z,E), a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf, args = (E,))[0] for E in energy_per_pulse]
    report['Excitation Probabilities - rz'] = NEh_rz
    # excitationProbability_00 = [i * sigma * dblquad(lambda r, z: HPu(r, z), a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf)[0] for i in numberPumpPhotons]
    Pn = lambda r,z,E,n: (NEh(r,z,E)**n * np.exp(-NEh(r,z,E)))/np.math.factorial(n)
    for n in range(nmax+1):
        Probability_00 = [Pn(0,0,E,n) for E in energy_per_pulse]
        report[f'P_{n}_00'] = Probability_00
    for n in range(nmax+1):
        Probability_rz = [(p**n * np.exp(-p))/np.math.factorial(n) for p in NEh_rz]
        report[f'P_{n}_rz'] = Probability_rz
    for n in range(nmax+1):
        Probability_n_avg = [dblquad(lambda r,z,E,n : (2*np.pi*r*Pn(r,z,E,n))/(2*np.pi*r), a = 0, b = PL*10**(3), gfun = 0, hfun = np.inf, args = (E,n,))[0] for E in energy_per_pulse]
        report[f'<P>_{n}'] = Probability_n_avg   
    result = []
    for p in NEh_rz:
        # Pavg = np.sum([n*Pn(0,0,E,n) for n in range(nmax+1)])
        Pavg = np.sum([n*(p**n * np.exp(-p))/np.math.factorial(n) for n in range(nmax+1)])
        result.append(Pavg)
    report['<Neh>'] = result
    return report

S18_Report = ExcitationPowerReport_v2(S18_wattages, 0.472, 75000, 142, nmax = 6, PL=10)
#%% plot report probability histograms
df = S17_Report_nmax6_v2[['P_0_00', 'P_1_00','P_2_00','P_3_00','P_4_00','P_5_00','P_6_00']]
fig, axes = plt.subplots(3,3,sharex=True, sharey=True,figsize=(9,9))
axes[0,0].bar(np.linspace(0,6,7),df.iloc[0,:])
axes[0,1].bar(np.linspace(0,6,7),df.iloc[1,:])
axes[0,2].bar(np.linspace(0,6,7),df.iloc[2,:])
axes[1,0].bar(np.linspace(0,6,7),df.iloc[3,:])
axes[1,1].bar(np.linspace(0,6,7),df.iloc[4,:])
axes[1,2].bar(np.linspace(0,6,7),df.iloc[5,:])
axes[2,0].bar(np.linspace(0,6,7),df.iloc[6,:])
axes[2,1].bar(np.linspace(0,6,7),df.iloc[7,:])
axes[2,2].bar(np.linspace(0,6,7),df.iloc[8,:])
axes = axes.flatten()
for ax in axes:
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
fig.text(0.5, -0.03, '# excitations per unit cell', ha='center', fontsize = 27)
fig.text(-0.03, 0.5, 'probability', va='center', rotation='vertical', fontsize = 27)
    
plt.tight_layout()

#%% crude approximation of the exciton density
def beerslaw_nu1000(powers, OD, epsilon = 75000, exWl = 400, gausDiameter = 142, PL = 10, reprate = 2000):
    NAvogadro = 6.0221408*10**(23)
    pseudoconc_unitcell = OD/(epsilon*PL*0.1)
    pseudoconc_cc = pseudoconc_unitcell * 1000
    print(f'pseudo concentration is {pseudoconc_unitcell} unit cells per liter')
    absorber_density_unc = pseudoconc_cc * NAvogadro # in unitcells/cc
    absorber_density_lnk = absorber_density_unc * 6 # in linkers/cc
    energy_per_pulse = [float(i)*10**(-6)/reprate for i in powers]
    hPlanck = 6.62606896*10**(-34) # SI
    cSpeed = 2.99792458*10**8 # SI
    # photonsIn = [(energy*exWl*10**(-9))/(hPlanck*cSpeed) for energy in energy_per_pulse]
    T_ratio = 10**(-OD)
    fluences_SI = [i/(2*np.pi*(0.5*(gausDiameter*10**(-6))**2)) for i in energy_per_pulse]
    photonFluenceIn = [(energy*exWl*10**(-9))/(hPlanck*cSpeed) for energy in fluences_SI]
    photonsOut = [T_ratio * numberPhotons for numberPhotons in photonFluenceIn]
    photonsAbsorbed = np.subtract(photonFluenceIn,photonsOut)
    excitonDensity_perUnitCell = [num/absorber_density_unc for num in photonsAbsorbed]
    excitonDensity_perLinker = [num/absorber_density_lnk for num in photonsAbsorbed]
    report = pd.DataFrame()
    report['Powers (uW)'] = powers
    report['Energies (nJ)'] = np.multiply(energy_per_pulse, 10**9)
    report['Photon Fluence In'] = photonFluenceIn
    report[f'Photon Fluence Out (A = {OD})'] = photonsOut
    report['Photons Absorbed'] = photonsAbsorbed
    report['Excitation Density (unitcell)'] = excitonDensity_perUnitCell
    report['Excitation Density (per absorber)'] = excitonDensity_perLinker
    return report

s17_BeersLaw = beerslaw_nu1000(S17_wattages, 0.472)

