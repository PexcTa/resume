# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:05:55 2024

@author: boris
"""

#%%
import numpy as np

#%%


def energy_to_k(E0, Ex):
    planck = 6.626 * 10 ** (-34)
    c = 2.998 * 10 ** (8)
    me = 9.109 * 10 ** (-31)
    ev_in_j = 6.242e18
    return (((8*(np.pi**2)*me*(np.abs(Ex-E0))/ev_in_j)/(planck**2))**(0.5))/10**(10)


def k_to_energy(E0, k):
    planck = 6.626 * 10 ** (-34)
    c = 2.998 * 10 ** (8)
    me = 9.109 * 10 ** (-31)
    ev_in_j = 6.242e18
    return E0 + (((k*10**10)**2 * planck**2)/(8*(np.pi**2)*me))*ev_in_j