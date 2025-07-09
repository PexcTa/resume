# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:40:27 2023

@author: boris
"""
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import pandas as pd

#%% Define the fitting functions

def gauss(t, A, t0, w):
    # gaussian_function
    G = A * np.exp(-(t-t0)**2/(2*w**2)) / (w*np.sqrt(2*np.pi))
    return G
# convolve population dynamics curve d with Gaussian curve g

def conv(t, t0, d, g):
# zero pad curves d and g to 2**N for N time delays
    np2 = int(np.ceil(np.log2(t.size*2))) # 
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

def FullFunc_2(t, A0, A1, A2, AG, t0, tau1, tau2, gamma, fwhm):
    # Extend the time
    t_extend = np.append((t[0]-(np.arange(1,int(len(t)*0.2))*(t[1]-t[0])))[::-1],t)
    t_full = np.append(t_extend,(t[-1]+(np.arange(1,int(len(t)*0.2))*(t[1]-t[0]))))
    x = t_full.copy()
    # Define Exponential Components
    # y_exp_1 = A0 + (A1 * np.exp(-(x-t0)/tau1))
    y_exp_1 = A0 + (A1*np.exp(-(x - t0)/tau1))/(1+A1*gamma*tau1*(1-np.exp(-(x - t0)/tau1)))
    y_exp_2 = (A2*np.exp(-(x - t0)/tau2))/(1+A2*gamma*tau2*(1-np.exp(-(x - t0)/tau2)))
    y_exp_1[t_full < t0] = A0
    y_exp_2[t_full < t0] = A0
    convolution_out = conv(t_full, t0, y_exp_1 + y_exp_2, gauss(t_full, AG, t0, fwhm))
    #return y_exp + y_sine
    #return conv(t, t0, y_exp, gauss(t, A, t0, fwhm))
    return convolution_out[np.where(t_full == t[0])[0][0]:np.where(t_full == t[-1])[0][0] + 1]

#%% visualizer
# plot normalized curves to find the point where annihilation starts
fig, ax = plt.subplots(figsize =( 10,10))
for k in range(1,6):
    ax.plot(S17traces[:,0], S17traces[:,k])

#%% The S17 Dataset
# The first 4 traces don't have power dependence
# Assume no annihilation happens
# Set gamma to 0 and don't allow it to vary
# Fit the first Four Traces and get estimates on tau1, tau2
# Try setting those estimates as restraints when fitting the rest of the traces
# unlock gamma

S17_ConvFit_bounds = {}
S17_ConvFit_bounds['tau1 upper'] = np.mean([results_08212023['150']['best-fit value'][5],
                                     results_08212023['510']['best-fit value'][5],
                                     results_08212023['980']['best-fit value'][5],
                                     results_08212023['1520']['best-fit value'][5]]) + np.std([results_08212023['150']['best-fit value'][5],
                                     results_08212023['510']['best-fit value'][5],
                                     results_08212023['980']['best-fit value'][5],
                                     results_08212023['1520']['best-fit value'][5]])
S17_ConvFit_bounds['tau1 lower'] = np.mean([results_08212023['150']['best-fit value'][5],
                                     results_08212023['510']['best-fit value'][5],
                                     results_08212023['980']['best-fit value'][5],
                                     results_08212023['1520']['best-fit value'][5]]) - np.std([results_08212023['150']['best-fit value'][5],
                                     results_08212023['510']['best-fit value'][5],
                                     results_08212023['980']['best-fit value'][5],
                                     results_08212023['1520']['best-fit value'][5]])
S17_ConvFit_bounds['tau2 upper'] = np.mean([results_08212023['150']['best-fit value'][6],
                                     results_08212023['510']['best-fit value'][6],
                                     results_08212023['980']['best-fit value'][6],
                                     results_08212023['1520']['best-fit value'][6]]) + np.std([results_08212023['150']['best-fit value'][6],
                                     results_08212023['510']['best-fit value'][6],
                                     results_08212023['980']['best-fit value'][6],
                                     results_08212023['1520']['best-fit value'][6]])
S17_ConvFit_bounds['tau2 lower'] = np.mean([results_08212023['150']['best-fit value'][6],
                                     results_08212023['510']['best-fit value'][6],
                                     results_08212023['980']['best-fit value'][6],
                                     results_08212023['1520']['best-fit value'][6]]) - np.std([results_08212023['150']['best-fit value'][6],
                                     results_08212023['510']['best-fit value'][6],
                                     results_08212023['980']['best-fit value'][6],
                                     results_08212023['1520']['best-fit value'][6]])

#%% Unit Cell Volume
a = 39.587e-10
b = 39.587e-10
c = 16.499e-10
V = a*b*c
cells = ((1e-2)**3)/V

qy = 0.25


#%% 
import lmfit as lm

normalFlag = 1
i = 4

if normalFlag:
    testTrace = S17traces[:,i+1]
else:
    testTrace = S17traces[:,i+1]*cells*S17_Report_nmax6['Average Excitation Probability'][i]*qy

model = lm.Model(FullFunc_2, independent_vars = ['t'])
params = lm.Parameters()
params.add('A0', value = 0, min = 0)
params.add('A1', value = 0.4, min = 0)
params.add('A2', value = 1, min = 0)
params.add('AG', value = 10000, min = 0)
params.add('t0', value = 0.8, min=0)
# params.add('tau1', value = 1.6, min=S17_ConvFit_bounds['tau1 lower'], max=S17_ConvFit_bounds['tau1 upper'])
# params.add('tau2', value = 0.18, min=S17_ConvFit_bounds['tau2 lower'], max=S17_ConvFit_bounds['tau2 upper'])
params.add('tau1', value = 1.45, vary=False)
params.add('tau2', value = 0.15, vary=False)
params.add('gamma', value = 3, min=0)
params.add('fwhm', value = 0.1, min=0.05, max=0.15)


# params.add('A3', value = 1, min = 0)
# params.add('tau3', value = 0.4, min=0)

result = model.fit(testTrace, t = t, params=params)
print(result.fit_report())

results_full = result.eval(t=t)
init_full = result.eval(t=t, params=params)
pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                            columns=('name', 'best-fit value', 'standard error'))

fig = plt.figure(figsize = (10,13.3))
gs = fig.add_gridspec(4, 1)
ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.scatter(t, testTrace, facecolor = 'none', edgecolor = 'black', linewidth = 3, label = 'Data')
ax1.plot(t, init_full, color = 'blue', linewidth = 3,label = 'Initial')
ax1.plot(t, results_full, color = 'red', linewidth = 3,label = 'Fit')
ax1.tick_params(axis='x', labelsize= 24)
ax1.tick_params(axis='y', labelsize= 24)
ax2.tick_params(axis='x', labelsize= 24)
ax2.tick_params(axis='y', labelsize= 24)
ax1.legend(loc = 'upper right', fontsize = 24)

ax2.plot(t, testTrace - results_full, color = 'black', linewidth = 3, 
         label = f'residuals (avg: {np.mean(testTrace-results_full):.3f})')

ax2.legend(loc = 'upper right', fontsize = 24)
ax2.axhline(color = 'gray', linestyle = '--', linewidth = 2.5)

plt.tight_layout()
# results_08212023[f'{S17_wattages[i]}'] = pd_pars
print(f'this was {S17_wattages[i]}')


#%% three-dimensional annihilation
i = 4
print(f'this was {S17_wattages[i]}')
factor = cells*S17_Report_nmax6['Average Excitation Probability'][i]
print(f'emissive exciton density equals about {factor:.2e} excitons per cc')
gamma = 2.16
gamma_0 = gamma*np.power(10, -np.log10(factor))
print(f'adjusted gamma equals about {gamma_0:.2e} cc/ns')
gamma_1 = gamma_0*1e9
print(f'standardized gamma equals about {gamma_1:.2e} cc/s')
diffcoeff = gamma_1/(8*np.pi*1e-7) #cm2/s
print(f'standardized diffusion coefficient equals {diffcoeff:.3f} cm2/s')
difflength = np.sqrt(diffcoeff*1e-4*1.45e-9)
print(f'standardized diffusion length equals {1e9*difflength:.3f} nm, assuming isotropic 3D motion')

#%% one-dimensional annihilation
i = 8
print(f'this was {S17_wattages[i]}')
factor = cells*S17_Report_nmax6['Average Excitation Probability'][i]*qy
print(f'emissive exciton density equals about {factor:.2e} excitons per cc')
gamma = 30.06
gamma_0 = gamma*np.power(10, -np.log10(factor))
print(f'adjusted gamma equals about {gamma_0:.2e} cc/ns')
gamma_1 = gamma_0*1e9
print(f'standardized gamma equals about {gamma_1:.2e} cc/s')
diffcoeff = gamma_1/(8*np.pi*1e-7) #cm2/s
print(f'standardized diffusion coefficient equals {diffcoeff:.3f} cm2/s')
difflength = np.sqrt(diffcoeff*1e-4*0.18e-9)
print(f'standardized diffusion length equals {1e9*difflength:.3f} nm')