# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:16:55 2024

@author: boris
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pycwt as wavelet
from scipy.signal import windows
#%%
def normalize_1(data):
    """Normalizes a vector to it's maximum value"""
    normalized_data = data/np.max(data)
    return normalized_data

def hann_lineshape(k, klims):
    id_min = (np.abs(k - klims[0])).argmin()
    id_max = (np.abs(k - klims[1])).argmin()
    windowed_section = k[id_min:id_max]
    length = len(windowed_section)
    window_insert = windows.hann(length, sym = False)
    window = np.zeros_like(k)
    window[id_min:id_max] = window_insert
    return window
#%%
fstring = 'ThO2_NaOH_40C_2nm'
kfile = f'{fstring}.chik'
rfile = f'{fstring}.chir'

kData = np.loadtxt(kfile, dtype = float, skiprows = 38)
rData = np.loadtxt(rfile, dtype = float, skiprows = 38)

#%%
# athena .chik format files have the header below
# #  k chi chik chik2 chik3 win energy
# athena exports the raw k-space and the hanning window with pre-set parameters in athena
# interestingly, the scipy hann window can achieve similar results even with default pars
# how is the window actually applied in athena?

k = kData[:,0]
chik1, chik2, chik3 = kData[:,2], kData[:,3], kData[:,4]
w = kData[:,5]
r = rData[:,0]
chir3 = rData[:,3]
k0 = np.min(k)
k_mask = np.nonzero(w)
dk = np.mean(np.diff(k[np.min(k_mask):np.max(k_mask)]))
i = chik1
N = i.size
dat = i*w


center = 2
fwhm = 1
mother = wavelet.Morlet(center) # define wavelet central frequency
# NOTE: individual paths are best resolved when central freq is about 2R
# NOTE: change central frequency depending on what R you want to resolve best
# now define parameters for Morlet wavelet transform
# see Torrence, C.; Compo, G. P. A practical guide to wavelet analysis. Bulletin of the American Meteorological society 1998, 79 (1), 61-78.

s0 = 2 * dk  # Starting scale, in this case 2 * k-spacing in the chosen k-range
dj = 1 / 64  # sub-octaves per octaves, increase denominator for better resolution
J = 11 / dj  # keep numerator high enough to cover the entire range of interest

# pycwt some denoising options like below
# doesn't seem to matter for exafs but try it
# alpha, _, _ = wavelet.ar1(i)  # Lag-1 autocorrelation for red noise

# PERFORM TRANSFORM
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat, dk, dj, s0, J,
                                                      mother)

iwave = wavelet.icwt(wave, scales, dk, dj, mother) # inverse transform to check
power = (np.abs(wave)) ** 2 # the actual power spectrum we want
period = freqs * np.pi # the R-space; we multiply freqs by pi because this is exafs
# power /= (scales[:, None]) # rescale power spectrum to enhance high R features
fft_power = np.abs(fft)**2
# sometimes the transform has high-frequency artifacts
# they are outside the region of interest
# but they throw the color scale off
# so we cut the period and power spectra to a max R = rLimit
# any relevant exafs is usually under R = 5 so we are in the clear
rLimit = 7
idx_rLimit = (np.abs(period - rLimit)).argmin()
period = period[idx_rLimit:]
power = power[idx_rLimit:,:]

# Prepare the figure
plt.close('all')
plt.ioff()
figprops = dict(figsize=(12, 12))
fig = plt.figure(**figprops)
fontSize1 = 24
fontSize2 = 18
rRange = [0.5,5]
kRange = [np.min(k), np.max(k)]

ax = plt.axes([0, 0.76, 0.7, 0.2])
ax.plot(k, i, 'k', linewidth=2.5, label = 'k-space')
ax.plot(k, -1*iwave, '-', linewidth=2, color='red', label = 'Inverse WT')
# ax.plot(k, w*np.max(i)*1.5, 'green', linewidth=1.5, label = 'Window')
# ax.set_title('Original EXAFS and the WT Inverse Transform', fontsize = fontSize1)
ax.set_xticks([])
ax.set_xlim(kRange)
ax.set_ylabel("k $^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fontSize1)
ax.tick_params(axis='both', labelsize= fontSize1)
ax.legend(loc = 'best', fontsize = fontSize2)

bx = plt.axes([0, 0, 0.7, 0.75])
nLevels = 45
stepLevels = 4
levels = np.linspace(power.min(), power.max(), nLevels)
levelsIndex = np.round(np.linspace(1, len(levels) - 1, int(nLevels/stepLevels))).astype(int)
bnt = bx.contourf(k, period, power, levels=levels,
            extend='both',cmap=plt.cm.YlOrRd)
# bx.set_title('b) Wavelet Power Spectrum ({})'.format(mother.name), fontsize=fontSize1)
bx.set_ylabel("R ($\AA$)", fontsize = fontSize1)
bx.set_xlabel("k ($\AA^{-1}$)", fontsize = fontSize1)
bx.tick_params(axis='both', labelsize= fontSize1)
bx.set_ylim([rRange[0], rRange[1]-0.01])
bx.set_xlim(kRange)
bx.annotate(f'Morlet $\omega$ = {center}', (0.5, rRange[1]-0.3), color='black', fontsize = fontSize1)
# plt.colorbar(bnt, ax=bx, location='bottom')
bx.contour(k, period, power, levels=levels[levelsIndex], colors='black', linewidths=2.5)

cx = plt.axes([0.71, 0.0, 0.2, 0.75])
cx.set_yticklabels([])
cx.tick_params(axis='both', labelsize= fontSize1)
cx.set_xlabel("|\u03C7(R)| ($\AA^{-4}$)", fontsize = fontSize1)
cx.set_ylim([rRange[0], rRange[1]-0.01])
cx.plot(chir3/np.max(chir3),r, 'k', linewidth=2.5, label = 'R-space')
proj = [np.sum(power[i,:]) for i in range(len(period[:,None]))]
cx.plot(fft_power/np.max(fft_power), fftfreqs*np.pi, linewidth=2, color='red', label = 'FFT')
cx.legend(loc = 'upper right', fontsize = fontSize2)
# cx.plot(fftpower/10,fftfreqs*np.pi)

# cx.set_title('c) Global Wavelet Spectrum')
# cx.set_ylim([period.min(), period.max()])



plt.show()
#%%



fstring = 'ThO2_NaOH_40C_2nm'
kfile = f'{fstring}.chik'
rfile = f'{fstring}.chir'
kData = np.loadtxt(kfile, dtype = float, skiprows = 38)
rData = np.loadtxt(rfile, dtype = float, skiprows = 38)


# athena .chik format files have the header below
# #  k chi chik chik2 chik3 win energy
# athena exports the raw k-space and the hanning window with pre-set parameters in athena
# interestingly, the scipy hann window can achieve similar results even with default pars
# how is the window actually applied in athena?
# datfile = np.loadtxt('ThPhos_pH75.chik', dtype = float, skiprows = 38)
# datfile = np.loadtxt('k3ce_model_kspa.dat', dtype = float, skiprows = 4)

# k = datfile[:,0]
k, chi, chik1, chik2, chik3 = kData[:,0], kData[:,1], kData[:,2], kData[:,3], kData[:,4]
# chik3 = datfile[:,1]

# r = rData[:,0]
# chir3 = rData[:,3]
k0 = np.min(k)


i = chik3
N = i.size


w = kData[:,5]
# w = hann_lineshape(k, (0,15))
k_mask = np.nonzero(w)
dk = np.mean(np.diff(k[np.min(k_mask):np.max(k_mask)]))
dat = i*w


center = 10
mother = wavelet.Morlet(center) # define wavelet central frequency
# NOTE: individual paths are best resolved when central freq is about 2R
# NOTE: change central frequency depending on what R you want to resolve best
# now define parameters for Morlet wavelet transform
# see Torrence, C.; Compo, G. P. A practical guide to wavelet analysis. Bulletin of the American Meteorological society 1998, 79 (1), 61-78.

s0 = 2 * dk  # Starting scale, in this case 2 * k-spacing in the chosen k-range
dj = 1 / 64  # sub-octaves per octaves, increase denominator for better resolution
J = 11 / dj  # keep numerator high enough to cover the entire range of interest

# pycwt some denoising options like below
# doesn't seem to matter for exafs but try it
# alpha, _, _ = wavelet.ar1(i)  # Lag-1 autocorrelation for red noise

# PERFORM TRANSFORM
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat, dk, dj, s0, J,
                                                      mother)

iwave = wavelet.icwt(wave, scales, dk, dj, mother) # inverse transform to check
power = (np.abs(wave)) ** 2 # the actual power spectrum we want
period = freqs * np.pi # the R-space; we multiply freqs by pi because this is exafs
power /= (scales[:, None]) # rescale power spectrum to enhance high R features

# sometimes the transform has high-frequency artifacts
# they are outside the region of interest
# but they throw the color scale off
# so we cut the period and power spectra to a max R = rLimit
# any relevant exafs is usually under R = 5 so we are in the clear
rLimit = 7
idx_rLimit = (np.abs(period - rLimit)).argmin()
period = period[idx_rLimit:]
power = power[idx_rLimit:,:]

# Prepare the figure
plt.close('all')
plt.ioff()
figprops = dict(figsize=(8, 12))
fig = plt.figure(**figprops)
fontSize1 = 28
fontSize2 = 24
rRange = [0.5, 5.0]
kRange = [np.min(k),12.2]

# ax = plt.axes([0, 0.76, 0.7, 0.2])
# ax.plot(k, i, 'k', linewidth=2.5, label = 'k-space')
# ax.plot(k, -1*iwave, '-', linewidth=2, color='red', label = 'Inverse WT')
# ax.plot(k, w*np.max(i)*1.5, 'green', linewidth=1.5, label = 'Window')
# # ax.set_title('Original EXAFS and the WT Inverse Transform', fontsize = fontSize1)
# ax.set_xticks([])
# ax.set_xlim(kRange)
# ax.set_ylabel("k$^{3}$\u03C7(k)($\AA^{-3}$)", fontsize = fontSize1)
# ax.tick_params(axis='both', labelsize= fontSize1)
# ax.legend(loc = 'upper right', fontsize = fontSize2)

bx = plt.axes([0, 0, 0.7, 0.75])
nLevels = 10
stepLevels = 4
levels = np.linspace(power.min(), power.max(), nLevels)
levelsIndex = np.round(np.linspace(1, len(levels) - 1, int(nLevels/stepLevels))).astype(int)
bnt = bx.contourf(k, period, power, levels=levels,
            extend='both',cmap=plt.cm.inferno)
# bx.set_title('b) Wavelet Power Spectrum ({})'.format(mother.name), fontsize=fontSize1)
bx.set_ylabel("R ($\AA$)", fontsize = fontSize1)
bx.set_xlabel("k ($\AA^{-1}$)", fontsize = fontSize1)
bx.tick_params(axis='both', labelsize= fontSize2)
bx.set_ylim([rRange[0], rRange[1]-0.01])
bx.set_xlim(kRange)
bx.xaxis.set_ticks(np.arange(min(k), max(k)+1, 2))
# bx.annotate(f'Morlet $\omega$ = {center}', (0.5, rRange[1]-0.3), color='black', fontsize = fontSize1)
# plt.colorbar(bnt, ax=bx, location='bottom')
# bx.contour(k, period, power, levels=levels[levelsIndex], colors='black', linewidths=2.5)

# cx = plt.axes([0.71, 0.0, 0.2, 0.75])
# cx.set_yticklabels([])
# cx.tick_params(axis='both', labelsize= fontSize1)
# cx.set_xlabel("|\u03C7(R)|($\AA^{-4}$)", fontsize = fontSize1)
# cx.set_ylim([rRange[0], rRange[1]-0.01])
# cx.plot(chir3/np.max(chir3),r, 'k', linewidth=2.5, label = 'R-space')
# proj = [np.sum(power[i,:]) for i in range(len(period[:,None]))]
# cx.plot(proj/np.max(proj), period, linewidth=2, color='red', label = 'WT Projection')
# cx.legend(loc = 'upper right', fontsize = fontSize2)
# cx.plot(fftpower/10,fftfreqs*np.pi)

# cx.set_title('c) Global Wavelet Spectrum')
# cx.set_ylim([period.min(), period.max()])



plt.show()

#%% Do contour map of the difference between experimental and calculated wavelet transforms!
import matplotlib

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.colormaps.register(cmap=newcmap)

    return newcmap

from mpl_toolkits.axes_grid1 import make_axes_locatable

# k_pu, k_th, p_pu, p_th, w_pu, w_th

from scipy.interpolate import interp2d

N = 1000
new_k, new_r = np.linspace(np.min(k1), np.max(k1), N), np.linspace(np.min(r1), np.max(r1), N)
f = interp2d(k1, r1, w1, kind='cubic')
new_w_pu = normalize_1(f(new_k, new_r))
f = interp2d(k2, r2, w2, kind='cubic')
new_w_th = normalize_1(f(new_k, new_r))

zDiff = (np.subtract(new_w_pu, new_w_th))/((np.max(new_w_pu)+np.max(new_w_th))/2) * 100


cmap = plt.cm.seismic
shiftmap = shiftedColorMap(cmap, 0, (1-np.max(zDiff)/(np.max(zDiff)+np.abs(np.min(zDiff)))), 1, 'shiftmap')

fig, bx = plt.subplots(figsize = (15,15))
fontSize1 = 36
fontSize2 = 28

nLevels = 10
stepLevels = 1

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
levels = np.linspace(zDiff.min(), zDiff.max(), nLevels)
levelsIndex = np.round(np.linspace(1, len(levels) - 1, int(nLevels/stepLevels))).astype(int)
bnt = bx.contourf(new_k, new_r, zDiff, levels=levels,extend='both',cmap=shiftmap)
bx.set_ylabel("R ($\AA$)", fontsize = fontSize1)
bx.set_xlabel("k ($\AA^{-1}$)", fontsize = fontSize1)
bx.tick_params(axis='both', labelsize= fontSize1)
bx.set_ylim([0, 5.5-0.01])
bx.set_xlim([0, 13.5-0.01])
divider = make_axes_locatable(bx)
bax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(bnt, cax=bax, orientation='vertical')
cbar.set_ticks(np.linspace(np.min(zDiff), np.max(zDiff), nLevels, dtype = int))
bax.tick_params(labelsize = fontSize1)
bax.set_title('% diff', fontsize = fontSize1)
bx.xaxis.set_major_locator(MultipleLocator(2))
bx.xaxis.set_minor_locator(MultipleLocator(1))
bx.yaxis.set_major_locator(MultipleLocator(0.5))
bx.yaxis.set_minor_locator(MultipleLocator(0.25))

plt.tight_layout()

plt.savefig('difference.png')