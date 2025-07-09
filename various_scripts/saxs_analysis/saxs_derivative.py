# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:34:55 2024

@author: boris
"""

import numpy as np
from scipy.special import erf
import lmfit as lm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import OrderedDict
import os
import pandas as pd
from matplotlib.ticker import LogLocator, LogFormatter,FuncFormatter, MultipleLocator
#%%
def normalize(data, point='maximum', factor = 100):
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
    if point=='maximum':
        normalized_data = np.abs(data)/max(data)
    normalized_data *= factor
    return normalized_data


#%%
saxs1 = OrderedDict()
saxsN = OrderedDict()
for file in os.listdir():
    if os.path.isdir(file):
    # skip directories
        continue
    if str(file).endswith('.json'):
        continue
    if str(file).endswith('.xlsx'):
        continue
    if str(file).endswith('.pptx'):
        continue
    with open(file):
        if str(file).endswith('norm_nmA.dat'):
            key = str(file)[:-4]
            saxsN[key] = np.loadtxt(file, skiprows = 4)
            saxsN[key][:,1] = normalize(saxsN[key][:,1])
        elif str(file).endswith('.dat'):
            key = str(file)[:-4]
            saxs1[key] = np.loadtxt(file, skiprows = 4)
        else:
            continue
#%%
labels = {}
with open("sampleDict") as f:
    for line in f:
       (key, val) = line.strip().split(';')
       labels[int(key)] = val
#%% 
def derivative(data):
    out = np.zeros_like(data)
    out[:,0] = data[:,0]
    # dq = np.mean(np.diff(data[:,0])) # standard for exafs
    dIdq = np.gradient(data[:,1], data[:,0])
    out[:,1] = np.abs(dIdq)
    return out

def interpolator(data, Npts):
    out = np.zeros((Npts, 2))
    x, y = data[:,0], data[:,1]
    x_ = np.linspace(np.min(x), np.max(x), Npts)
    y_ = np.interp(x_, x, y)
    out[:,0], out[:,1] = x_, y_
    return out
    
#%%
saxs_intp = OrderedDict()
for key in saxs1.keys():
    saxs_intp[key+'_intp'] = interpolator(saxs1[key], 50000)
    
saxs_derv = OrderedDict()
for key in saxs_intp.keys():
    saxs_derv[key+'_d1'] = derivative(saxs_intp[key])
    
saxs_derv_raw = OrderedDict()
for key in saxs1.keys():
    saxs_derv_raw[key+'_d1'] = derivative(saxs1[key])
    
#%%
fig, ax = plt.subplots(figsize = (10, 10))
ax.set(xscale = 'log', yscale = 'log')

dset = saxs_derv_raw

for key, i in zip(dset.keys(), range(len(dset.keys()))):
    sns.scatterplot(x = dset[key][:,0], y = np.multiply(100**i, dset[key][:,1]), ax=ax)
    

plt.show()
#%%
fig, ax = plt.subplots(figsize = (7, 4))
ax.set(xscale = 'log', yscale = 'log')

dset = saxs_derv_raw['sas_2202_restitched_nmA_d1']
dset_ = saxs_derv_raw['sas_2223_nmA_d1']

sns.scatterplot(x = dset[:,0], y = np.multiply(1, dset[:,1]), ax=ax, label = '$\dfrac{dI}{dq}$ (sample 2202)')
sns.scatterplot(x = dset_[:,0], y = np.multiply(1, dset_[:,1]), ax=ax, label = '$\dfrac{dI}{dq}$ (sample 2223)')

ax.set_xlabel('q ($\AA^{-1}$)')
ax.set_ylabel('Scattering Intensity')

# tem_mean = 36.0
# tem_var = 27.050

# q_crit = 2*np.pi/tem_mean * (1-1.6*(np.sqrt(tem_var)/tem_mean))

# ax.axvline(x = q_crit, color = 'red', label = 'q$_c$ (TEM)')
ax.legend(loc = 'best')

plt.show()

#%% fit the derivative to multiple linear sections
fig, ax = plt.subplots(figsize = (7, 4))
# ax.set(xscale = 'log', yscale = 'log')

dset = saxs_derv_raw['sas_2203_nmA_d1']

# sns.scatterplot(x = dset[:,0], y = np.multiply(1, dset[:,1]), ax=ax, label = '$\dfrac{dI}{dq}$ (sample 2203)')


ax.set_xlabel('q ($\AA^{-1}$)')
ax.set_ylabel('Scattering Intensity')


ax.legend(loc = 'best')

region = [-1.4, -0.35]
ax.axvline(x = region[0], linestyle = '--')
ax.axvline(x = region[1], linestyle = '--')

q = dset[:,0]

idx_highq = (np.abs(np.log10(q) - region[1])).argmin()
idx_lowq = (np.abs(np.log10(q) - region[0])).argmin()

q_ = np.log10(q[idx_lowq:idx_highq])
dIdq = np.log10(dset[idx_lowq:idx_highq,1])

sns.scatterplot(x = np.log10(dset[:,0]), y = np.log10(np.multiply(1, dset[:,1])), ax=ax, label = '$\dfrac{dI}{dq}$ (sample 2203)')
sns.scatterplot(x = q_, y = dIdq, ax = ax, color = 'red')

# def two_lines(x, m1, b1, qc, m2, b2):
#     y1, y2 = np.zeros_like(x), np.zeros_like(x)
#     y1[x<qc] = x[x<qc]*m1 + b1
#     y2[x>=qc] = x[x>=qc]*m2 + b2
#     result = np.concat([y1[x<qc],y2[x>=qc]])
#     return result

def two_lines(x, m1, b1, qc, m2, b2):
    result = []
    for xi in x:
        if xi < qc: 
            result.append(xi*m1+b1)
        elif xi == qc:
            result.append((xi*m2+b2+xi*m1+b1)/2)
        else:
            result.append(xi*m2+b2)
    return result

model= lm.Model(two_lines, independent_vars = ['x'])
params = lm.Parameters()
params.add('qc', value = -1.3, min = region[0], max = region[1], vary = True)
params.add('m1', value = -2)
params.add('b1', value = 0)
params.add('m2', value = -2)
params.add('b2', value = 0)

result = model.fit(dIdq, x = q_, params=params, method = 'leastsq')
print(result.fit_report())

fit = result.eval(x = q_)
init = result.eval(x = q_, params=params)

sns.lineplot(x = q_, y = fit, ax=ax, label = 'fitted')
sns.lineplot(x = q_, y = init, ax=ax, label = 'initial')

plt.show()

#%%
import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

xData = q_
yData = dIdq


def func(xArray, breakpoint, slopeA, offsetA, slopeB, offsetB):
    returnArray = []
    for x in xArray:
        if x < breakpoint:
            returnArray.append(slopeA * x + offsetA)
        else:
            returnArray.append(slopeB * x + offsetB)
    return returnArray


# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = func(xData, *parameterTuple)
    return numpy.sum((yData - val) ** 2.0)


def generate_Initial_Parameters():
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)
    slope = 10.0 * (maxY - minY) / (maxX - minX) # times 10 for safety margin

    parameterBounds = []
    parameterBounds.append([minX, maxX]) # search bounds for breakpoint
    parameterBounds.append([-slope, slope]) # search bounds for slopeA
    parameterBounds.append([minY, maxY]) # search bounds for offsetA
    parameterBounds.append([-slope, slope]) # search bounds for slopeB
    parameterBounds.append([minY, maxY]) # search bounds for offsetB


    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

# by default, differential_evolution completes by calling curve_fit() using parameter bounds
geneticParameters = generate_Initial_Parameters()

# call curve_fit without passing bounds from genetic algorithm
fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters)
print('Parameters:', fittedParameters)
print()

modelPredictions = func(xData, *fittedParameters) 

absError = modelPredictions - yData

SE = numpy.square(absError) # squared errors
MSE = numpy.mean(SE) # mean squared errors
RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))

print()
print('RMSE:', RMSE)
print('R-squared:', Rsquared)

print()


##########################################################
# graphics output section
def ModelAndScatterPlot(graphWidth, graphHeight):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    # first the raw data as a scatter plot
    axes.plot(xData, yData,  'D')

    # create data for the fitted equation plot
    xModel = numpy.linspace(min(xData), max(xData))
    yModel = func(xModel, *fittedParameters)

    # now the model as a line plot
    axes.plot(xModel, yModel)

    axes.set_xlabel('X Data') # X axis data label
    axes.set_ylabel('Y Data') # Y axis data label

    plt.show()
    plt.close('all') # clean up after using pyplot

graphWidth = 800
graphHeight = 600
ModelAndScatterPlot(graphWidth, graphHeight)

