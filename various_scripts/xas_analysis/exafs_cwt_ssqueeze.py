# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:16:55 2024

@author: boris
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from ssqueezepy import ssq_cwt, ssq_stft
from ssqueezepy.experimental import scale_to_freq

def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    plt.show()
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='turbo')
    plt.show()
#%%

file = np.loadtxt('thpo4_ph4p8_resplined_k3.chik', dtype = float, skiprows = 39)
# athena .chik format files have the header below
# #  k chi chik chik2 chik3 win energy
# athena exports the raw k-space and the hanning window with pre-set parameters in athena
# interestingly, the scipy hann window can achieve similar results even with default pars
# how is the window actually applied in athena

k = file[:,0]
i = file[:,4]
w = file[:,5]


#%%
window = signal.windows.hann(len(k))

fig, ax = plt.subplots()

ax.plot(k, i)
# ax.plot(k, i*window)
ax.plot(k, i*w)

#%%
Twxo, Wxo, *_ = ssq_cwt(i*w, ('morlet', {'mu': 3.5}), scales = 'linear', fs = 1/np.mean(np.diff(k)))
viz(k, Twxo, Wxo)

#%%# With units #######################################
from ssqueezepy import Wavelet, cwt, stft, imshow

fs = np.max(k)

wavelet = Wavelet(('morlet', {'mu': 8}))
Wx, scales = cwt(i*window, wavelet, fs=fs)
# Sx = stft(i*w)[::-1]

freqs_cwt = scale_to_freq(scales, wavelet, len(k), fs=fs)*2*np.pi
# freqs_stft = np.linspace(1, 0, len(Sx)) * fs/2

# ikw = dict(abs=1, xticks=k, xlabel="Time [sec]", ylabel="Frequency [Hz]")
# imshow(Wx, **ikw, yticks=freqs_cwt)
# imshow(Sx, **ikw, yticks=freqs_stft)

fig, ax = plt.subplots(figsize = (10,10))
pcm = ax.pcolormesh(k, freqs_cwt-np.min(freqs_cwt), np.abs(Wx))
# ax.set_ylim(0,6)
ax.set_xlabel("Wavenumber")
ax.set_ylabel("Distance")

fig.colorbar(pcm, ax=ax)
#%%
output = np.zeros_like(np.abs(Wx)[:,0])
for i in range(len(output)):
    output[i] = np.sum(np.abs(Wx)[i,:])