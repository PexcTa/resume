# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:06:14 2022

@author: boris
"""
import numpy as np
import matplotlib.pyplot as plt
def normalize_1(data):
    """Normalizes a vector to its maximum value"""
    normalized_data = np.abs(data)/max(data)
    return normalized_data
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
#%%
pKa1 = 1
pKa2 = 3.30
pKa3 = 5.70
pKa4 = 8.20

pH = np.linspace(-2,14,1000)


def henderson_hasselbalch_2(pka1, pka2, ph):
    D = (10**(-pH))**2 +  (10**(-pka1))*(10**(-pH)) + (10**(-pka1))*(10**(-pka2))
    a0 = 1/D*(10**(-pH))**2
    a1 = 1/D*(10**(-pka1))*(10**(-pH))
    a2 = 1/D*(10**(-pka1))*(10**(-pka2))
    return (a0,a1,a2)


def henderson_hasselbalch_3(pka1, pka2, pka3, ph):
    D = (10**(-pH))**3 +  (10**(-pka1))*(10**(-pH))**2 + (10**(-pka1))*(10**(-pka2))*(10**(-pH)) + (10**(-pka1))*(10**(-pka2))*(10**(-pka3))
    a0 = 1/D*(10**(-pH))**3
    a1 = 1/D*(10**(-pka1))*(10**(-pH))**2
    a2 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pH))
    a3 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pka3))
    return (a0,a1,a2,a3)

def henderson_hasselbalch_4(pka1, pka2, pka3, pka4, ph):
    D = (10**(-pH))**4 +  (10**(-pka1))*(10**(-pH))**3 + (10**(-pka1))*(10**(-pka2))*(10**(-pH))**2 + (10**(-pka1))*(10**(-pka2))*(10**(-pka3))*(10**(-pH)) + (10**(-pka1))*(10**(-pka2))*(10**(-pka3))*(10**(-pka4))
    a0 = 1/D*(10**(-pH))**4
    a1 = 1/D*(10**(-pka1))*(10**(-pH))**3
    a2 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pH))**2
    a3 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pka3))*(10**(-pH))
    a4 = 1/D*(10**(-pka1))*(10**(-pka2))*(10**(-pka3))*(10**(-pka4))
    return (a0,a1,a2,a3,a4)

fractions = henderson_hasselbalch_3(pKa2, pKa3, pKa4, pH)
#%%
fig, ax = plt.subplots(figsize=(16,6))

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
# ax.axhline(y=0, color = 'dimgrey', ls = '--')
ax.set_xlim([0,12])
ax.set_ylim([-0.05,1.35])
ax.set_xlabel("pH", fontsize = 28)
ax.set_ylabel("Fraction", fontsize = 28)
ax.set_facecolor('gray')
ax.tick_params(axis='x', labelsize= 25)
ax.tick_params(axis='y', labelsize= 25)
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.plot(pH, np.zeros_like(pH), linewidth = 6, color = 'black', label = 'Protonation States')
ax.set_prop_cycle('color',[plt.cm.RdBu(i) for i in np.linspace(0, 1, 4)])
ax.grid(color='dimgray')
for i in range(4):
    ax.plot(pH, fractions[i], linewidth = 6, label='_nolegend_')

# ax.set_prop_cycle('color',[plt.cm.summer(i) for i in np.linspace(0.1, 0.9, 3)])
D = (10**(-pH))**3 +  (10**(-pKa2))*(10**(-pH))**2 + (10**(-pKa2))*(10**(-pKa3))*(10**(-pH)) + (10**(-pKa2))*(10**(-pKa3))*(10**(-pKa4))
a0 = 1/D*(10**(-pH))**3
result = 1-(1-a0)**4
# ax.plot(pH, result, linestyle = '--', linewidth = 6,  label = '$\hatP_1$')
# a1 = a1 = 1/D*(10**(-pKa2))*(10**(-pH))**2
# result = 1-(1-a1-a0)**4
# ax.plot(pH, result, linestyle = '--', linewidth = 6, label = '$\hatP_2$')
# a2 = 1/D*(10**(-pKa2))*(10**(-pKa3))*(10**(-pH))
# result = 1-(1-a2-a1-a0)**4
# ax.plot(pH, result, linestyle = '--', linewidth = 6, label = '$\hatP_3$')


ax.plot(pH, result, linestyle = '--', linewidth = 6, color = 'seagreen', label = '$\hat{P}$')
ax.legend(loc = 'upper center', fontsize = 25, ncol = 4)


#%%
index2 = (np.abs(pH - 5.94)).argmin()
index3 = (np.abs(pH - 7.61)).argmin()
index4 = (np.abs(pH - 7.66)).argmin()

print(fractions[1][index2], fractions[2][index2], fractions[3][index2])