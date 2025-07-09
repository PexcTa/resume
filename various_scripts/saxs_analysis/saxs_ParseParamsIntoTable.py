# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:58:19 2025

@author: boris
"""



#%%

import pandas as pd
import re

filestring= 's1700_fit_recal_v3.txt'

def get_param_value(params, param_type, num, default='0'):
    """Helper function to get parameter value or return default if not found"""
    for item in params[param_type]:
        if item[0] == num:
            return item[1] if len(item) == 2 else item[1:3]
    return [default, default] if param_type in ['rg', 'B', 'G'] else default

# Read the file
with open(filestring, 'r') as f:
    lines = f.readlines()

# Initialize storage with all possible parameters up to level 4
params = {
    'power': [],  # Will store (num, value)
    'rg': [],     # Will store (num, value, error)
    'B': [],      # Will store (num, value, error)
    'G': [],      # Will store (num, value, error)
    'special': {} # For scale and background
}

# Pattern for numbered parameters
param_pattern = re.compile(r'^\s*(power|rg|B|G)(\d+)\s*,')
# Pattern for special parameters
special_pattern = re.compile(r'^\s*(scale|background)\s*,')

for line in lines:
    line = line.strip()
    if not line or line.startswith(('sasview_parameter_values', 'model_name', 'multiplicity')):
        continue
    
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 3:
        continue
    
    param_name = parts[0]
    
    # Check for special parameters first
    special_match = special_pattern.match(param_name + ',')
    if special_match:
        param_type = special_match.group(1)
        value = parts[2]
        params['special'][param_type] = value
        continue
    
    # Check for numbered parameters
    param_match = param_pattern.match(param_name + ',')
    if not param_match:
        continue
    
    param_type = param_match.group(1)
    param_num = int(param_match.group(2))
    value = parts[2]
    error = parts[3] if len(parts) > 3 and parts[3] else '0'  # Default to '0' if missing
    
    if param_type == 'power':
        params['power'].append((param_num, value))
    else:
        params[param_type].append((param_num, value, error))

# Sort all parameters by their number (except special ones)
for key in ['power', 'rg', 'B', 'G']:
    params[key].sort(key=lambda x: x[0])

# Build the DataFrame with all possible parameters up to level 4
data = {}

# Add power parameters (p1-p4)
for num in range(1, 5):
    value = get_param_value(params, 'power', num)
    data[f'p{num}'] = [value]

# Add rg parameters (rg2-rg4, skip rg1)
for num in range(2, 5):
    value, error = get_param_value(params, 'rg', num)
    data[f'rg{num}'] = [value]
    data[f'rg{num}_err'] = [error]

# Add special parameters (scale and background)
if 'scale' in params['special']:
    data['scale'] = [params['special']['scale']]
else:
    data['scale'] = ['0']

if 'background' in params['special']:
    data['background'] = [params['special']['background']]
else:
    data['background'] = ['0']

# Add B parameters (B1-B4)
for num in range(1, 5):
    value, error = get_param_value(params, 'B', num)
    data[f'B{num}'] = [value]
    data[f'B{num}_err'] = [error]

# Add G parameters (G2-G4, skip G1)
for num in range(2, 5):
    value, error = get_param_value(params, 'G', num)
    data[f'G{num}'] = [value]
    data[f'G{num}_err'] = [error]

# Create DataFrame
df = pd.DataFrame(data)

# Reorder columns to put scale and background before B1
cols = list(df.columns)
# Find where B columns start
b_start = next((i for i, col in enumerate(cols) if col.startswith('B')), len(cols))
# Insert special params before B columns
for param in ['background', 'scale']:
    if param in cols:
        cols.remove(param)
        cols.insert(b_start - 1, param)

df = df[cols]
df = df.astype(float)

# Display the result
print(df)

#%% Reconstruct the Curve

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erf

def beaucage1lv(q, Pars):
    scale, G, RgSub, B, P, bkg = Pars[0], Pars[1], Pars[2], Pars[3], Pars[4], Pars[5]
    term = bkg + scale*(G*np.exp((-(q*RgSub)**2)/3) + (B) * (((erf(q*RgSub/np.sqrt(6)))**3)/q)**P)
    return term

def beaucage_3lv(q, k, G1, Rg1, B1, P1, G2, Rg2, B2, P2, G3, Rg3, B3, P3, scale, bkg):
    term1 = G1*np.exp((-(q*Rg1)**2)/3) + (B1*np.exp((-(q*Rg2)**2)/3)) * (((erf((q*k*Rg1)/np.sqrt(6)))**3)/q)**P1
    term2 = G2*np.exp((-(q*Rg2)**2)/3) + (B2*np.exp((-(q*Rg3)**2)/3)) * (((erf((q*k*Rg2)/np.sqrt(6)))**3)/q)**P2
    term3 = G3*np.exp((-(q*Rg3)**2)/3) + (B3) * (((erf((q*k*Rg3)/np.sqrt(6)))**3)/q)**P3
    return scale*(term1+term2+term3)+bkg

# def beaucage3lvFull(q):
#     top = df.B1[0]*np.exp((-(q*1500)**2)/3) * (1/(q*(erf(q*RgSub/np.sqrt(6)))**(-3)))**P
#     full = top + mid1 + mid2 + bot
#     full *= df.scale[0]
#     full += df.background[0]

# def pwrLaw(q, Pars):
#     scale, Rg, B, P, bkg = Pars[0], Pars[1], Pars[2], Pars[3], Pars[4]
#     term = bkg + (B*np.exp((-(q*Rg)**2)/3)) * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**P
#     return term
# 
# dset = 'sas1700_q_Iabs.dat'
df = df.astype(float)
file = np.loadtxt('sas1700_q_Iabs_recal.dat')
q = file[:,0]
Iexp = file[:,1]

fig, ax = plt.subplots(figsize = (6,6))
ax.loglog(q, Iexp)

test = beaucage_3lv(q, 1, 0, 1500, df.B1[0], df.p1[0], 
                    df.G2[0], df.rg2[0], df.B2[0], df.p2[0],
                    df.G3[0], df.rg3[0], df.B3[0], df.p3[0],
                    df.scale[0], df.background[0])

pars3 = [float(i) for i in [df.scale[0], df.G3[0], df.rg3[0], df.B3[0], df.p3[0], df.background[0]]]
Ifit3 = beaucage1lv(q, pars3)

# ax.loglog(q, Ifit4)
ax.loglog(q, Ifit3)
# ax.loglog(q, Ifit2)
# ax.loglog(q, Ifit1)
# ax.loglog(q, Ifit3+Ifit2+Ifit1)
ax.loglog(q, test)

ax.set_ylim([min(Iexp), max(Iexp)])

plt.show()

np.savetxt('sas1700_q_Ifit_Primary_3lv.dat', np.vstack([q, Ifit3]).T)
# np.savetxt('sas1700_q_Ifit_Mid.dat', np.vstack([q, Ifit3]).T)

#%% Integrate the Porod Invariate
# lowQ = np.trapezoid(Ifit4*np.pow(q, 2), x = q)
Qinv = np.trapezoid(Ifit3*np.pow(q, 2), x = q)
print(Qinv)

#%% determine the volume fraction
import numpy as np

file = np.loadtxt('sas1700_q_Iabs_recal.dat')
q = file[:,0]
Iexp = file[:,1]

totalQinv = np.trapezoid(Iexp*np.pow(q,2), x = q)
rho1 = 6.61e-5
rho2 = 1.2e-8
A = (totalQinv/1e8)/(2 * np.pi**2 * (rho1-rho2)**2)

phi1 = (1-np.sqrt(1 - 4*A))/2
phi2 = (1+np.sqrt(1 - 4*A))/2

print(phi1, phi2)

#%% link B to G
import numpy as np 

def link(G1, Rg1, p):
    q1 = (1/Rg1)*np.sqrt(3*p/2)
    B1 = G1*np.exp(-((q1*Rg1)**2)/3)*q1**p
    return B1


# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Parameters
medians = [2.11, 1.57, 18.2]
sigmas = [0.303, 0.426, 0.346]

# Calculate mu and sigma
fig, ax = plt.subplots(figsize = (5,5))


for i in range(len(medians)):
    median, sigma = 2*medians[i], sigmas[i]
    mu = np.log(median)
    x = np.linspace(0, median * 5, 100)  # Adjust the range as needed
    pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2 * sigma**2))
    ax.plot(x, pdf)
ax.set_xlabel("Particle Diameter, $\AA$")
ax.set_ylabel("PDF")
ax.set_xscale('log')

plt.grid(True)
plt.show()
