# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:16:55 2024

@author: boris
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pywt

#%%

file = np.loadtxt('feo_xafs.chik', dtype = float, skiprows = 39)
# athena .chik format files have the header below
# #  k chi chik chik2 chik3 win energy
# athena exports the raw k-space and the hanning window with pre-set parameters in athena
# interestingly, the scipy hann window can achieve similar results even with default pars
# how is the window actually applied in athena

k = file[:,0]
i = file[:,4]
w = file[:,5]

def hanning(k1, k2, k):
    mid = k1 + np.abs(k2 - k1)/2
    window = 1/mid * np.cos(np.pi*k/mid)**2
    for i in range(len(window)):
        if k[i] <= k1 or k[i] >= k2:
            window[i] = 0
    return window

test = hanning(3,11,k)

#%%
window = signal.windows.hann(len(k))

fig, ax = plt.subplots()

ax.plot(k, i)
ax.plot(k, i*window)
ax.plot(k, i*w)


#%%
wavelet = "cmor7-0.5"
# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=100)
# scales = [1 * 2**(j*0.125) for j in range(64)]
sampling_period = np.diff(k).mean()
cwtmatr, freqs = pywt.cwt(i*w, widths, wavelet, sampling_period=sampling_period)
# absolute take absolute value of complex result
cwtmatr = np.abs(cwtmatr[:-1, :-1])

fig, ax = plt.subplots(figsize = (10,10))
pcm = ax.pcolormesh(k, freqs*2*np.pi, cwtmatr)
ax.set_ylim(0,10)
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Distance")

fig.colorbar(pcm, ax=ax)

#%%
output = np.zeros_like(cwtmatr[:,0])
for i in range(len(output)):
    output[i] = np.sum(cwtmatr[i,:])