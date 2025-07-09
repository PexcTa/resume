# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:38:55 2022

@author: boris
"""

import numpy as np
from pandas import Series
from scipy import special
from lmfit import Model, Parameters, report_fit, minimize

def decay(t, N, tau):
    return N*np.exp(-t/tau)

t = np.linspace(0, 5, num=1000)
np.random.seed(2021)
data = decay(t, 7, 3) + np.random.randn(t.size)

model = Model(decay, independent_vars=['t'])
result = model.fit(data, t=t, N=10, tau=1)

print(result.values)

result.plot()

#%%
def TA_3exp(data, amp1, fwhm, tau1, t0, amp2, tau2, amp3, tau3, amp0):
    return amp1*fwhm*np.exp((fwhm/(np.sqrt(2)*tau1))**2-(data-t0)/tau1)*(1-special.erf(fwhm/(np.sqrt(2)*tau1)-(data-t0)/(np.sqrt(2)*fwhm))) + \
        + amp2*fwhm*np.exp((fwhm/(np.sqrt(2)*tau2))**2-(data-t0)/tau2)*(1-special.erf(fwhm/(np.sqrt(2)*tau2)-(data-t0)/(np.sqrt(2)*fwhm))) + \
            + amp3*fwhm*np.exp((fwhm/(np.sqrt(2)*tau3))**2-(data-t0)/tau3)*(1-special.erf(fwhm/(np.sqrt(2)*tau3)-(data-t0)/(np.sqrt(2)*fwhm))) + \
                + amp0
                
def TA_3exp_dataset(params, i, x):
    """Calculate exponential parameters for the dataset."""
    amp1 = params[f'amp1_{i+1}']
    amp2 = params[f'amp2_{i+1}']
    amp3 = params[f'amp3_{i+1}']
    tau1 = params[f'tau1_{i+1}']
    tau2 = params[f'tau2_{i+1}']
    tau3 = params[f'tau3_{i+1}']
    fwhm = params[f'fwhm_{i+1}']
    t0 = params[f't0_{i+1}']
    amp0 = params[f'amp0_{i+1}']
    return TA_3exp(data, amp1, fwhm, tau1, t0, amp2, tau2, amp3, tau3, amp0)

def objective(params, x, data):
    """Calculate total residual for fits of exponential sums to several data sets."""
    ndata, _ = data.shape
    resid = 0.0*data[:]
    for i in range(ndata):
        resid[i, :] = data[i, :] - TA_3exp_dataset(params, i, x)
    return resid.flatten()

fit_params = Parameters()
for iy, y in enumerate(data):
    fit_params.add(f'amp1_{iy+1}', value=0.001, min = 0.0)
    fit_params.add(f'amp2_{iy+1}', value=0.001, min = 0.0)
    fit_params.add(f'amp3_{iy+1}', value=0.001, min = 0.0)
    fit_params.add(f'tau1_{iy+1}', value=1, min=0.0, max=10)
    fit_params.add(f'tau2_{iy+1}', value=10, min=0.0, max=100)
    fit_params.add(f'tau3_{iy+1}', value=100, min=0.0, max=1000)
    fit_params.add(f'fwhm_{iy+1}', value=0)
    fit_params.add(f't0_{iy+1}', value=0.3)
    fit_params.add(f'amp0_{iy+1}', value=0.3, min=0.0)
    
for iy in [2]:
    fit_params[f'tau1_{iy}'].expr = 'tau1_1'
    fit_params[f'tau1_{iy}'].expr = 'tau2_1'
    fit_params[f'tau1_{iy}'].expr = 'tau3_1'
    fit_params[f'tau1_{iy}'].expr = 'fwhm_1'
    fit_params[f'tau1_{iy}'].expr = 't0_1'
    fit_params[f'tau1_{iy}'].expr = 'amp0_1'
    
out = minimize(objective, fit_params, args=(time, data))
report_fit(out.params)


#%%
