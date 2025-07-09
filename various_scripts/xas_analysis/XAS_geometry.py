# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:34:06 2025

@author: boris
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
# CARBONATE
geom = {}
# get the side that does not change as geom['o_cut']


geom['ac_ref'] = 3.014
geom['ao_ref'] = 2.481
geom['co_ref'] = 1.355
geom['aco_ref'] = 54.31
geom['aco_ref'] *= (np.pi/180)

geom['o_cut'] = geom['co_ref'] * np.sin(geom['aco_ref'])

# get the relationship between dR(absorber-o) and dR(absorber-c)

dRao = [0]
geom['ao_sys'] = [geom['ao_ref']]
geom['ac_sys'] = [geom['ac_ref']-np.sqrt(geom['co_ref']**2-geom['o_cut']**2)]
i = 0
while max(dRao) <= 0.3:
    dRao.append(0.005*i)
    ao_i = geom['ao_ref'] + dRao[i]
    geom['ao_sys'].append(ao_i)
    ac_i = np.sqrt(ao_i**2 - geom['o_cut']**2)
    geom['ac_sys'].append(ac_i)
    i+=1
    
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(dRao, np.subtract(geom['ac_sys'],geom['ac_ref']-np.sqrt(geom['co_ref']**2-geom['o_cut']**2)), linewidth = 7, color = 'red', alpha = 0.5)
coefficients = np.polyfit(dRao, np.subtract(geom['ac_sys'],geom['ac_ref']-np.sqrt(geom['co_ref']**2-geom['o_cut']**2)), 1)
print("Linear Fit Coefficients:", coefficients)

# Create polynomial function
p = np.poly1d(coefficients)

plt.plot(dRao, p(dRao), label='Linear Fit', linewidth = 3.5, linestyle = '--', color='black')
plt.show()

