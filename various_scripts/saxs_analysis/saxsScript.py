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
    with open(file):
        if str(file).endswith('lqhq.dat'):
            key = str(file)
            saxs1[key] = np.loadtxt(file, skiprows = 4)
        elif str(file).endswith('normI.dat'):
            key = str(file)
            saxsN[key] = np.loadtxt(file, skiprows = 4)
            saxsN[key][:,1] = normalize(saxsN[key][:,1])

#%%
labels = {}
with open("sampleDict") as f:
    for line in f:
       (key, val) = line.strip().split(';')
       labels[int(key)] = val
# labels_rus = {}
# with open("sampleDict_rus") as f:
#     for line in f:
#        (key, val) = line.strip().split(';')
#        labels[int(key)] = val
#%% plot log(I) vs q

group1 = [2202, 2205, 2206, 2225] # the NaOH samples annealed at 40 C, viridis
group2 = [2203, 2210, 2211, 2212, 2222, 2221, 2213] # the 0.1M and 0.01M X-ray(am) ThO2, gnuplot2
group3 = [2204, 2223] # the 150C samples, winter
group4 = [2401, 2402, 2403] # the phosphates, autumn

tags = group3

fig, ax = plt.subplots(figsize = (10,10))

fs1 = 24
fs2 = 20
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel('log $q$ (nm$^{-1}$)', fontsize = fs1)
ax.set_ylabel('log $I$ (a.u.)', fontsize = fs1)


colors = [plt.cm.autumn(i) for i in np.linspace(0, 0.9, len(tags))]

for i in range(len(tags)):
    dset = [value for key, value in saxs1.items() if str(tags[i]) in key][0]
    # ax.semilogy(dset[:,0], dset[:,1], 'o-', color = colors[i], markersize = 15, markerfacecolor='none',linewidth = 2.5, label = labels[tags[i]])
    ax.loglog(dset[:,0], dset[:,1], 'o-', color = colors[i], markersize = 15, markerfacecolor='none',linewidth = 2.5, label = labels[tags[i]])
ax.legend(loc = 'upper right', fontsize = fs2)
ax.set_xlim(0.03, 13)

plt.tight_layout()
#%% plot multiple spectra with guinier axes

tags = group2
which = 2210
stretch = (0.002,0.02)
fit_region = (0.008, 0.016)
ylims = [-6,1]
anPos = -2

if tags == group1:
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 0.9, len(tags))]
elif tags == group2:
    colors = [plt.cm.gnuplot2(i) for i in np.linspace(0, 0.9, len(tags))]
elif tags == group3:
    colors = [plt.cm.winter(i) for i in np.linspace(0, 0.9, len(tags))]


j = tags.index(which)


from matplotlib.gridspec import GridSpec
gs = GridSpec(5, 2, figure=fig)

fig = plt.figure(figsize = (20,10))

fs1 = 28
fs2 = 24
ax1 = fig.add_subplot(gs[:,0])

ax1.tick_params('both', labelsize = fs2)
ax1.set_xlabel('q$^2$ (nm$^{-2}$)', fontsize = fs1)
ax1.set_ylabel('ln (I)', fontsize = fs1)

for i in range(len(tags)):
    dset = [value for key, value in saxsN.items() if str(tags[i]) in key][0]
    ax1.plot(dset[:,0]**2, np.log(dset[:,1]), 'o-', color = colors[i],  markerfacecolor='none',linewidth = 2.5, label = labels[tags[i]])
ax1.legend(loc = 'upper right', fontsize = fs2)
# ax1.set_yticks([])
ax1.axvline(x = stretch[0])
ax1.axvline(x = stretch[1])


ax2 = fig.add_subplot(gs[:4, 1])
# ax2.tick_params('y', labelsize = fs2)
guinier_result = pd.DataFrame(columns=('name', 'best-fit value', 'standard error'))
# ax2.set_ylabel('ln (I)', fontsize = fs1)
ax2.set_xlim([stretch[0] - np.max(stretch)*0.1, stretch[1] + np.max(stretch)*0.1])
for i in range(len(tags)):
    dset = [value for key, value in saxsN.items() if str(tags[i]) in key][0]
    if i == j:
        ax2.plot(dset[:,0]**2, np.log(dset[:,1]), 'o', color = colors[i], markersize = 30,markerfacecolor='none',markeredgewidth = 2, label = labels[tags[i]])
        x = dset[:,0]**2
        idx_x_min = (np.abs(x - fit_region[0])).argmin()
        idx_x_max = (np.abs(x - fit_region[1])).argmin()
        x = x[idx_x_min:idx_x_max]
        y = np.log(dset[:,1][idx_x_min:idx_x_max])
        z, cov = np.polyfit(x, y, 1, cov = True)
        Rg = np.sqrt(np.abs(z[0]*3))
        qRg = np.sqrt(np.max(x))*Rg
        guinier_result = pd.DataFrame([['grad', 'y0', 'Rg', 'qRg'], 
                                       [z[0], z[1], Rg, qRg], 
                                       [cov[0,0], cov[1,1], cov[0,0]/(2*np.sqrt(np.abs(z[0])*3)), cov[0,0]/(2*np.sqrt(np.abs(z[0])*3))*np.sqrt(np.max(x))]],index=('name', 'best-fit value', 'variance'))
        ax2.plot(x, x*z[0]+z[1], 'r-', linewidth = 2.5, label = 'Fit')
        ax2.annotate(f'R$_g$ = {Rg:.2f}\nqR$_g$ = {qRg:.2f}', [x.min(), anPos], fontsize = fs2)
    else:
        ax2.plot(dset[:,0]**2, np.log(dset[:,1]), 'o', color = colors[i], markersize = 30, markerfacecolor='none',markeredgewidth = 1, alpha=0.15)
ax2.legend(loc = 'upper right', fontsize = fs2)
ax2.set_ylim(ylims)
ax2.tick_params(axis = 'y', labelsize = fs2)
# ax2.set_yticks([])
ax2.xaxis.set_major_locator(MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.set_xticklabels([])

ax3 = fig.add_subplot(gs[4,1])
ax3.set_xlabel('q$^2$ (nm$^{-2}$)', fontsize = fs1)
ax3.xaxis.set_major_locator(MultipleLocator(0.1))
ax3.xaxis.set_major_formatter(FuncFormatter("{:.1f}".format))
ax3.xaxis.set_minor_locator(MultipleLocator(0.05))
ax3.axhline(y = 0, linestyle = '--', linewidth = 1.5, color = 'grey')
dset = [value for key, value in saxsN.items() if str(tags[j]) in key][0]
x = dset[:,0]**2
ax3.plot(x, np.log(dset[:,1]) - (x*z[0]+z[1]), '-', color = colors[j])
idx_x_min = (np.abs(x - fit_region[0])).argmin()
idx_x_max = (np.abs(x - fit_region[1])).argmin()
x = x[idx_x_min:idx_x_max]
ax3.plot(x, np.log(dset[:,1])[idx_x_min:idx_x_max] - (x*z[0]+z[1]), '-', linewidth = 3, color = 'red', label = 'Residuals')
# y = np.log(dset[:,1][idx_x_min:idx_x_max])
ax3.tick_params(axis = 'both', labelsize = fs2)
ax3.set_ylim([-0.1, 0.1])
ax3.set_xlim([stretch[0] - np.max(stretch)*0.1, stretch[1] + np.max(stretch)*0.1])
ax3.legend(loc = 'upper left', fontsize = fs2)

plt.tight_layout()

#%% plot multiple spectra with log log axes

tags = group2
which = 2221
stretch = (0.01, 0.04)

if tags == group1:
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 0.9, len(tags))]
elif tags == group2:
    colors = [plt.cm.gnuplot2(i) for i in np.linspace(0, 0.9, len(tags))]
elif tags == group3:
    colors = [plt.cm.winter(i) for i in np.linspace(0, 0.9, len(tags))]


j = tags.index(which)

fig, ax = plt.subplots(1,2,figsize = (20,10), sharey = True)

fs1 = 28
fs2 = 24
ax[0].tick_params('both', labelsize = fs2)
ax[0].set_xlabel('ln (q)', fontsize = fs1)
ax[0].set_ylabel('ln (I)', fontsize = fs1)

for i in range(len(tags)):
    dset = [value for key, value in saxs1.items() if str(tags[i]) in key][0]
    ax[0].loglog(dset[:,0], normalize(dset[:,1]), 'o-', color = colors[i], base = np.e, markerfacecolor='none',linewidth = 2.5, label = labels[tags[i]])
ax[0].legend(loc = 'upper right', fontsize = fs2)
# ax[0].set_yticks([])
# ax[0].axvline(x = stretch[0])
# ax[0].axvline(x = stretch[1])

ax[1].tick_params('both', labelsize = fs2)
ax[1].set_xlabel('ln (q)', fontsize = fs1)
# ax[1].set_ylabel('ln (I)', fontsize = fs1)

for i in range(len(tags)):
    dset = [value for key, value in saxs1.items() if str(tags[i]) in key][0]
    if i == j:
        ax[1].loglog(dset[:,0], normalize(dset[:,1]), 'o', color = colors[i], base = np.e, markersize = 25,
                     markerfacecolor='none',markeredgewidth = 2, label = labels[tags[i]])
    else:
        ax[1].loglog(dset[:,0], normalize(dset[:,1]), 'o', color = colors[i], base = np.e, markersize = 25, 
                     markerfacecolor='none',markeredgewidth = 1,alpha=0.05)
# ax[1].set_xlim([stretch[0] - np.max(stretch)*0.1, stretch[1] + np.max(stretch)*0.1])
# ax[1].set_ylim(10, 10**8)
# ax[1].set_yticks([])
ax[1].legend(loc = 'upper right', fontsize = fs2)
from matplotlib.ticker import LogLocator, LogFormatter
for axis in ax:
    axis.xaxis.set_major_locator(LogLocator(base = 10, numticks=15))
    axis.xaxis.set_major_formatter(LogFormatter(base = 10, labelOnlyBase=True))
    axis.yaxis.set_major_formatter(LogFormatter(base = 10, labelOnlyBase=False))

plt.tight_layout()
#%% plot the Porod law 

tags = group2
which = 2221
stretch = (0.01, 0.04)

if tags == group1:
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 0.9, len(tags))]
elif tags == group2:
    colors = [plt.cm.gnuplot2(i) for i in np.linspace(0, 0.9, len(tags))]
elif tags == group3:
    colors = [plt.cm.winter(i) for i in np.linspace(0, 0.9, len(tags))]


j = tags.index(which)

fig, ax = plt.subplots(1,2,figsize = (20,10))

fs1 = 28
fs2 = 24
ax[0].tick_params('both', labelsize = fs2)
ax[0].set_xlabel('ln (q)', fontsize = fs1)
ax[0].set_ylabel('ln (I)', fontsize = fs1)

for i in range(len(tags)):
    dset = [value for key, value in saxs1.items() if str(tags[i]) in key][0]
    ax[0].loglog(dset[:,0], normalize(dset[:,1]), 'o-', color = colors[i], base = np.e, markerfacecolor='none',linewidth = 2.5, label = labels[tags[i]])
ax[0].legend(loc = 'upper right', fontsize = fs2)
# ax[0].set_yticks([])
# ax[0].axvline(x = stretch[0])
# ax[0].axvline(x = stretch[1])

ax[1].tick_params('both', labelsize = fs2)
ax[1].set_xlabel('ln (q)', fontsize = fs1)
# ax[1].set_ylabel('ln (I)', fontsize = fs1)

for i in range(len(tags)):
    dset = [value for key, value in saxs1.items() if str(tags[i]) in key][0]
    if i == j:
        ax[1].loglog(dset[:,0]**4, dset[:,1] * dset[:,0]**4, 'o', color = colors[i], base = np.e, markersize = 25,
                     markerfacecolor='none',markeredgewidth = 2, label = labels[tags[i]])
    else:
        ax[1].loglog(dset[:,0]**4, dset[:,1] * dset[:,0]**4, 'o', color = colors[i], base = np.e, markersize = 25, 
                     markerfacecolor='none',markeredgewidth = 1,alpha=0.01)
# ax[1].set_xlim([stretch[0] - np.max(stretch)*0.1, stretch[1] + np.max(stretch)*0.1])
# ax[1].set_ylim(10, 10**8)
# ax[1].set_yticks([])
ax[1].legend(loc = 'upper left', fontsize = fs2)
from matplotlib.ticker import LogLocator, LogFormatter
for axis in ax:
    axis.xaxis.set_major_formatter(LogFormatter(base = np.e))
    axis.yaxis.set_major_formatter(LogFormatter(base = np.e))

plt.tight_layout()

#%% beaucage fit programming
which = 2401
dset = [value for key, value in saxs1.items() if str(which) in key][0]

lowqlimit = 0.04
highqlimit = 1.9

from scipy.special import erf, gamma
import lmfit as lm

def lognorm(x, N, mu, s, p):
    # adapted from the SASfit manual
    # N = scaling parameter
    # mu = distribution center (size)
    # s = sigma
    # p = shape parameter (between 0 and 1)
    term1 = np.sqrt(2*np.pi) * s * mu**(1-p) * np.exp(0.5*((1-p)**2)*s**2)
    term2 = N *  (1/x**p) * np.exp((-np.log(x/mu)**2)/(2*s**2))
    return term2/term1
#%%
def beaucage_A(q, z, P, Gs, Rs, Bs, Ps = 4):
    Rsub = Rs
    df = P
    Rg = ((((2*np.sqrt(5/3)*Rs)**2)*z**(2/df))/((1+2/df)*(2+2/df)))**(1/2)
    term1 = (Gs*z)*np.exp((-(q*Rg)**2)/3)
    B = ((Gs*z*df)/(Rg**df))*gamma(df/2)
    term2 = (B*np.exp((-(q*Rsub)**2)/3)) * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**(P)
    term3 = Gs*np.exp((-(q*Rs)**2)/3)
    term4 = Bs * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**(Ps)
    return term1+term2+term3+term4

def beaucage_B(q, z, P, Gs, Rs, Bs, Ps, N, s, p):
    Rs_range = np.linspace(1, 100, 1000)
    Rs_dist = lognorm(Rs_range, N, Rs, s, p)
    df = P
    Rg = np.sum([((((2*np.sqrt(5/3)*Rss)**2)*z**(2/df))/((1+2/df)*(2+2/df)))**(1/2)*Irss for Rss, Irss in zip(Rs_range, Rs_dist)])
    term1 = (Gs*z)*np.exp((-(q*Rg)**2)/3)
    B = ((Gs*z*df)/(Rg**df))*gamma(df/2)
    term2 = np.sum([Irss*(B*np.exp((-(q*Rss)**2)/3)) * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**(P) for Rss, Irss in zip(Rs_range, Rs_dist)], axis = 0)
    term3 = np.sum([Irss*Gs*np.exp((-(q*Rss)**2)/3) for Rss, Irss in zip(Rs_range, Rs_dist)])
    term4 = Bs * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**(Ps)
    return term1+term2+term3+term4

def beaucage(q, G, Rg, B, Rsub, P, Gs, Rs, Bs, Ps):
    term1 = G*np.exp((-(q*Rg)**2)/3)
    term2 = (B*np.exp((-(q*Rsub)**2)/3)) * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**P
    term3 = Gs*np.exp((-(q*Rs)**2)/3)
    term4 = Bs * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**Ps
    return term1+term2+term3+term4

def beaucage_3lv(q, k, G1, Rg1, B1, P1, G2, Rg2, B2, P2, G3, Rg3, B3, P3):
    term1 = G1*np.exp((-(q*Rg1)**2)/3) + (B1*np.exp((-(q*Rg2)**2)/3)) * (((erf((q*k*Rg1)/np.sqrt(6)))**3)/q)**P1
    term2 = G2*np.exp((-(q*Rg2)**2)/3) + (B2*np.exp((-(q*Rg3)**2)/3)) * (((erf((q*k*Rg2)/np.sqrt(6)))**3)/q)**P2
    term3 = G3*np.exp((-(q*Rg3)**2)/3) + (B3*np.exp((-(q*Rg3)**2)/3)) * (((erf((q*k*Rg3)/np.sqrt(6)))**3)/q)**P3
    return term1+term2+term3

model = lm.Model(beaucage_B, independent_vars=['q'])

q = dset[:,0]
sig = dset[:,1]
idx_highq = (np.abs(q - highqlimit)).argmin()
idx_lowq = (np.abs(q - lowqlimit)).argmin()
q_fit = q[idx_lowq:idx_highq]
sig_fit = sig[idx_lowq:idx_highq]

# weights = np.geomspace(1,1.e4,len(q_fit))
weights = np.ones_like(q_fit)

params = lm.Parameters()
params.add('z', value = 5, min = 2, max = 300)
params.add('P', value = 1, min = 1, max = 3)
params.add('Gs', value = 20000, min = 0, max = 1.e10)
params.add('Rs', value = 30, min = 4, max = 50)
params.add('Bs', value = 1, min = 0, max = 20)
params.add('Ps', value = 3.99, min = 3, max = 4, vary = True)
params.add('N', value = 1, min = 0, max = 1, vary = False)
params.add('s', value = 0.05, min = 0.01, max = 0.5, vary = True)
params.add('p', value = 1, min = 1, max = 4, vary = False)

result = model.fit(sig_fit, q = q_fit, params=params, weights=weights, method = 'leastsq')
print(result.fit_report())

fit = result.eval(q = q_fit)
init = result.eval(q = q_fit, params=params)

fig, ax = plt.subplots(figsize = (10,10))

fs1 = 24
fs2 = 20
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel('q (nm$^{-1}$)', fontsize = fs1)
ax.set_ylabel('I (a.u.)', fontsize = fs1)

ax.loglog(q, sig, linewidth = 4, color = 'gray', label = f'data, sample {which}')
ax.loglog(q_fit, fit, linewidth = 3, color = 'crimson', label = 'Beaucage fit')
ax.loglog(q_fit, init, linewidth = 3, linestyle = '--', color = 'blue', label = 'Initial pars')
ax.axvline(x = highqlimit, linewidth = 2.5, linestyle = '-.', color = 'black')
ax.axvline(x = lowqlimit, linewidth = 2.5, linestyle = '-.', color = 'black')

ax.set_xlim(0.03, 12.5)

pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                        columns=('name', 'best-fit value', 'standard error'))

def beaucage_B_terms(q, params):
    z, P, Gs, Rs, Bs, Ps, N, s, p = params
    Rs_range = np.linspace(1, 100, 1000)
    Rs_dist = lognorm(Rs_range, N, Rs, s, p)
    df = P
    Rg = np.sum([((((2*np.sqrt(5/3)*Rss)**2)*z**(2/df))/((1+2/df)*(2+2/df)))**(1/2)*Irss for Rss, Irss in zip(Rs_range, Rs_dist)])
    term1 = (Gs*z)*np.exp((-(q*Rg)**2)/3)
    B = ((Gs*z*df)/(Rg**df))*gamma(df/2)
    term2 = np.sum([Irss*(B*np.exp((-(q*Rss)**2)/3)) * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**(P) for Rss, Irss in zip(Rs_range, Rs_dist)], axis = 0)
    term3 = np.sum([Irss*Gs*np.exp((-(q*Rss)**2)/3) for Rss, Irss in zip(Rs_range, Rs_dist)])
    term4 = Bs * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**(Ps)
    print(Rg, B)
    return (term1, term2, term3, term4)

terms = beaucage_B_terms(q, list(pd_pars['best-fit value']))
ax.set_prop_cycle('color', ['chocolate', 'chartreuse', 'steelblue', 'hotpink'])
ax.loglog(q, terms[0], linewidth = 2.5, linestyle = '-.', label = 'GP $_{agg}$')
ax.loglog(q, terms[1], linewidth = 2.5, linestyle = '-.', label = 'GP $_{pp}$')
ax.loglog(q, terms[2], linewidth = 2.5, linestyle = '-.', label = 'Guinier $_{pp}$')
ax.loglog(q, terms[3], linewidth = 2.5, linestyle = '-.', label = 'Porod $_{pp}$')
ax.set_ylim(1.e-2, 1.e7)

ax.legend(loc = 'upper right', fontsize = fs2)
plt.tight_layout()
#%% pretty figure
which = 2222
dset = [value for key, value in saxsN.items() if str(which) in key][0]

highqlimit = 10
q = dset[:,0]
sig = dset[:,1]

fig, ax = plt.subplots(figsize = (12,12))

fs1 = 32
fs2 = 28
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel('q (nm$^{-1}$)', fontsize = fs1)
ax.set_ylabel('I (a.u.)', fontsize = fs1)

ax.loglog(q, sig, linewidth = 4, color = 'gray', alpha = 0.3)
# ax.loglog(q_fit, sig_fit, linewidth = 4, color = 'gray', label = labels[which])
# ax.loglog(q_fit, fit, linewidth = 3, color = 'crimson', label = 'Beaucage fit')
# ax.axvline(x = highqlimit, linewidth = 2.5, linestyle = '-.', color = 'black', label = 'Cutoff')
ax.legend(loc = 'upper right', fontsize = fs2)
ax.set_xlim(0.03, 12.5)

plt.tight_layout()

#%% beaucage 3 level
which = 2210
dset = [value for key, value in saxsN.items() if str(which) in key][0]

highqlimit = 10




def beaucage_3lv(q, k, G1, Rg1, B1, P1, G2, Rg2, B2, P2, G3, Rg3, B3, P3):
    term1 = G1*np.exp((-(q*Rg1)**2)/3) + (B1*np.exp((-(q*Rg2)**2)/3)) * (((erf((q*k*Rg1)/np.sqrt(6)))**3)/q)**P1
    term2 = G2*np.exp((-(q*Rg2)**2)/3) + (B2*np.exp((-(q*Rg3)**2)/3)) * (((erf((q*k*Rg2)/np.sqrt(6)))**3)/q)**P2
    term3 = G3*np.exp((-(q*Rg3)**2)/3) + (B3*np.exp((-(q*Rg3)**2)/3)) * (((erf((q*k*Rg3)/np.sqrt(6)))**3)/q)**P3
    return term1+term2+term3

model = lm.Model(beaucage_3lv, independent_vars=['q'])

q = dset[:,0]
sig = dset[:,1]
idx_highq = (np.abs(q - highqlimit)).argmin()
q_fit = q[:idx_highq]
sig_fit = sig[:idx_highq]

weights = np.geomspace(1,1.e4,len(q_fit))
# weights = np.ones_like(q_fit)

params = lm.Parameters()
params.add('k', value = 1, vary = True, min = 0.95, max = 1.05)
params.add('G1', value = 100, min = 0, max = 1.e10, vary = False)
params.add('Rg1', value = (2*np.pi)/np.min(q_fit), min = 0, max = 2*(2*np.pi)/np.min(q_fit))
params.add('B1', value = 0.00125, min = 0, max = 100)
params.add('P1', value = 3, min = 0, max = 5)
params.add('G2', value = 100, min = 0, max = 1.e10)
params.add('Rg2', value = 10, min = 0, max = 20)
params.add('B2', value = 0.00125, min = 0, max = 100)
params.add('P2', value = 4, min = 0, max = 5, vary = False)
params.add('G3', value = 100, min = 0, max = 1.e10)
params.add('Rg3', value = (2*np.pi)/np.max(q_fit), min = (2*np.pi)/np.max(q_fit)/2, max = 2)
params.add('B3', value = 0.0000125, min = 0, max = 100)
params.add('P3', value = 4, min = 0, max = 5, vary = False)


result = model.fit(sig_fit, q = q_fit,params=params, weights=weights, method = 'leastsq')
print(result.fit_report())

pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
                        columns=('name', 'best-fit value', 'standard error'))

fit = result.eval(q = q_fit)
init = result.eval(q = q_fit, params=params)

fig, ax = plt.subplots(figsize = (10,10))

fs1 = 24
fs2 = 20
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel('q (nm$^{-1}$)', fontsize = fs1)
ax.set_ylabel('I (a.u.)', fontsize = fs1)

ax.loglog(q, sig, linewidth = 4, color = 'gray', label = f'data, sample {which}')
ax.loglog(q_fit, fit, linewidth = 3, color = 'crimson', label = 'Beaucage fit')
ax.loglog(q_fit, init, linewidth = 3, linestyle = '--', color = 'blue', label = 'Initial pars')
ax.axvline(x = highqlimit, linewidth = 2.5, linestyle = '-.', color = 'black', label = 'Cutoff')
ax.legend(loc = 'upper right', fontsize = fs2)
ax.set_xlim(0.03, 12.5)

plt.tight_layout()

beaucage_3lv_fits[f'{which}_3lv_qmax{highqlimit}'] = pd_pars
#%% beaucage testing
def lognorm(x, N, mu, s, p):
    # adapted from the SASfit manual
    # N = scaling parameter
    # mu = distribution center (size)
    # s = sigma
    # p = shape parameter (between 0 and 1)
    term1 = np.sqrt(2*np.pi) * s * mu**(1-p) * np.exp(0.5*((1-p)**2)*s**2)
    term2 = N *  (1/x**p) * np.exp((-np.log(x/mu)**2)/(2*s**2))
    return term2/term1

def beaucage_rs(q, G, Rg, B, Rsub, P, Gs, Bs, Ps):
    term1 = G*np.exp((-(q*Rg)**2)/3)
    term2 = (B*np.exp((-(q*Rsub)**2)/3)) * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**P
    term3 = Gs*np.exp((-(q*Rsub)**2)/3)
    term4 = Bs * (((erf((q*Rg)/np.sqrt(6)))**3)/q)**Ps
    return term1+term2+term3+term4

def beaucage_rs_lognorm(q, G, Rg, B, Rsub, P,  Gs, Bs, Ps, N, s, p):
    Rsub_range = np.linspace(Rg/1000, Rg*2, 100)
    Rsub_dist = lognorm(Rsub_range, N, Rsub, s, p)
    ans = np.sum([beaucage_rs(q, G, Rg, B, Rsub, P, Gs, Bs, Ps) for Rsub in Rsub_dist], axis = 0)
    return ans
    

which = 2401
dset = [value for key, value in saxsN.items() if str(which) in key][0]

highqlimit = 9

q = dset[:,0]
sig = dset[:,1]
idx_highq = (np.abs(q - highqlimit)).argmin()
q_fit = q[:idx_highq]
sig_fit = sig[:idx_highq]

model = lm.Model(beaucage_rs_lognorm, independent_vars=['q'])

fitFlag = 0
    
params = lm.Parameters()
params.add('G', value = 1, min = 0, max = 1.e10)
params.add('Rg', value = (2*np.pi)/np.min(q_fit), min = 1, max = 3*(2*np.pi)/np.min(q_fit))
params.add('B', value = 12, min = 0, max = 100)
params.add('Rsub', value = 30, min = 1, max = 60)
params.add('P', value = 3, min = 0, max = 5)
params.add('Gs', value = 1, min = 0, max = 1.e10)
# params.add('Rs', value = 3, min = 1, max = 10)
params.add('Bs', value = 11, min = 0, max = 10000)
params.add('Ps', value = 3, min = 0, max = 5, vary = False)
params.add('N', value = 4, min = 0, max = 1.e10)
params.add('s', value = 0.1, min = 0.01, max = 0.1, vary = False)
params.add('p', value = 1, vary = False)

if fitFlag == 1:
    result = model.fit(sig_fit, q = q_fit,params=params, method = 'leastsq')
    print(result.fit_report())
    pd_pars = pd.DataFrame([(p.name, p.value, p.stderr) for p in result.params.values()], 
    columns=('name', 'best-fit value', 'standard error'))
    fit = result.eval(q = q_fit)

init = result.eval(q = q_fit, params=params)

fig, ax = plt.subplots(figsize = (10,10))

fs1 = 24
fs2 = 20
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel('q (nm$^{-1}$)', fontsize = fs1)
ax.set_ylabel('I (a.u.)', fontsize = fs1)

ax.loglog(q, sig, linewidth = 4, color = 'gray', label = f'data, sample {which}')
ax.loglog(q_fit, fit, linewidth = 3, color = 'crimson', label = 'Beaucage fit')
ax.loglog(q_fit, init, linewidth = 3, linestyle = '--', color = 'blue', label = 'Initial pars')
ax.axvline(x = highqlimit, linewidth = 2.5, linestyle = '-.', color = 'black', label = 'Cutoff')
ax.legend(loc = 'upper right', fontsize = fs2)
ax.set_xlim(0.03, 12.5)

plt.tight_layout()

#%%
def beaucage_rs_lognorm(q, G=1, Rg=300, B=0.01, Rsub=30, P=4,  Gs=1, Bs=0.01, Ps=4, N=1, s=0.05, p=1):
    Rsub_range = np.linspace(Rg/1000, Rg, len(q))
    Rsub_dist = lognorm(Rsub_range, N, Rsub, s, p)
    ans = beaucage_rs(q, G, Rg, B, Rsub_dist, P, Gs, Bs, Ps)
    return Rsub_dist

test = beaucage_rs_lognorm(q_fit)
plt.plot(test)

#%%
def rg_sphere(Rg, error_Rg):
    # returns the DIAMETER of a sphere based on given Rg
    # units are the same as passed
    # Rg = R * (3/5)**(0.5)
    d = 2*Rg/(np.sqrt(3/5))
    e = 2*error_Rg/(np.sqrt(3/5))
    return(d, e)

def rg_lamella_random(Rg, error_Rg):
    t = Rg*np.sqrt(12)
    e = error_Rg*np.sqrt(12)
    print(t, e)
    return t, e

def rg_cylinder(Rg1, Rg2):
    # returns thickness and length of a cylinder based on given Rg1, Rg2
    # units are the same as passed
    # Rg1 = R/2**(0.5)
    # Rg2 = (L**2/12 + R**2/2)**0.5
    T = 2*Rg1*np.sqrt(2)
    L = np.sqrt(12*(Rg2**2 - T**2/8))
    return(T,L)

#%% plot sasview output
datfile1 = np.loadtxt('2401_data.txt', skiprows = 1)
fitfile1 = np.loadtxt('2401_fit.txt', skiprows = 1)
datfile2 = np.loadtxt('2402_data.txt', skiprows = 1)
fitfile2 = np.loadtxt('2402_fit.txt', skiprows = 1)
datfile3 = np.loadtxt('2403_data.txt', skiprows = 1)
fitfile3 = np.loadtxt('2403_fit.txt', skiprows = 1)

fig, ax = plt.subplots(figsize = (10,10))

fs1 = 26
fs2 = 22
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel('log $q$ ($\AA^{-1}$)', fontsize = fs1)
ax.set_ylabel('log $I$ (a.u.)', fontsize = fs1)


ax.loglog(datfile3[:,0], datfile3[:,1], 'o-', color='black', markersize = 15, markerfacecolor='none', linewidth = 2.5, alpha = 0.5)
ax.loglog(fitfile3[:,0], fitfile3[:,1], '-', color='crimson', linewidth = 3.5)


ax.legend(labels = ['Sample TP-1c', 'Fit'], loc = 'upper right', fontsize = fs1)
# ax.set_xlim(0.03, 13)

plt.tight_layout()

#%% vertical stack figure

group1 = [2202, 2205, 2206, 2225] # the NaOH samples annealed at 40 C, viridis
group2 = [2203, 2210, 2211, 2212, 2222, 2221, 2213] # the 0.1M and 0.01M X-ray(am) ThO2, gnuplot2
group3 = [2204, 2223] # the 150C samples, winter
group4 = [2401, 2402, 2403] # the phosphates, autumn

tags = group2[1:-2]+[group2[0]]

N = len(tags)
fs1 = 26
fs2 = 22
fig, axes = plt.subplots(N,1,sharex = True,figsize=(12,12))
plt.subplots_adjust(hspace=0)
big_ax = fig.add_subplot(111)
big_ax.set_facecolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
for i in range(0,N):
    axes[i].set_yticks([])
    # axes[i].set_facecolor('gray')
    # axes[i].set_ylim([0, 3500000])
big_ax.set_xticks([])
big_ax.set_yticks([])
# axes[N-1].set_xlim(370, 600)
axes[N-1].tick_params('x', labelsize = fs2)
axes[N-1].set_xlabel('log $q$ (nm$^{-1}$)', fontsize = fs1)
big_ax.set_ylabel('log $I$ (a.u.)', fontsize = fs1)


cmap1 = matplotlib.cm.get_cmap('viridis')

colors = np.linspace(0,0.8,N)

for i in range(N):
    dset = [value for key, value in saxs1.items() if str(tags[i]) in key][0]
    x = dset[:,0]
    y = dset[:,1]
    # axes[15-i].set_ylim([0, 3700000 - 250000*(11-i)])
    # print(labels[i])
    axes[N-(i+1)].loglog(x, y, 'o-', markersize = 7.5, markerfacecolor='none', linewidth = 2.5, alpha = 0.5, color = cmap1(colors[i]), label = labels[tags[i]])
    axes[N-(i+1)].legend(loc = 'upper right', fontsize = fs2, framealpha = 0)
    axes[N-(i+1)].set_yticks([])
    axes[N-(i+1)].spines[['top']].set_visible(False)
    
    
#%%
group2 = [2204, 2223, 2203, 2210, 2211, 2212, 2222, 2221, 2213]

tags = group2
which = 2213
fig, ax = plt.subplots(figsize = (10,10))

fs1 = 24
fs2 = 20
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel('log $q$ (A$^{-1}$)', fontsize = fs1)
ax.set_ylabel('log $I$ (a.u.)', fontsize = fs1)


colors = [plt.cm.gnuplot2(i) for i in np.linspace(0, 0.9, len(tags))]

for i in range(len(tags)):
    dset = [value for key, value in saxs1.items() if str(tags[i]) in key][0]
    if tags[i] == which:
        ax.loglog(dset[:,0]/10, normalize(dset[:,1]), '-', color = colors[i], markersize = 15, markerfacecolor='none',linewidth = 4.5, label = labels[tags[i]])
    else:
        ax.loglog(dset[:,0]/10, normalize(dset[:,1]), '-', color = colors[i], markersize = 15, markerfacecolor='none',linewidth = 3.0, alpha = 0.1)
ax.legend(loc = 'upper right', fontsize = fs2)

ax.axvline(x = 0.45)
# ax.set_xlim(0.03, 2)

plt.tight_layout()

#%%
def qconv(two_theta, wavelength):
    return 4*np.pi*np.sin((two_theta/2)*(np.pi/180))/wavelength

def qtod(q):
    return (2*np.pi)/q

# q111 = qconv(27.613, 1.54)
# q200 = qconv(31.900, 1.54)
# q220 = qconv(45.964, 1.54)

#%% G/B analysis
from scipy.special import gamma
B_values = [0.055459, 47.546, 101.05, 1167]
G_values = [2.471e5, 6.6875e7, 1.225e7, 2.1041e6]
Rg_values = [39.646, 71.292, 37.888, 12.134]

def apmf(G, Rg, df):
    return np.multiply(G_values, df)/np.power(Rg, df) * gamma(df/2)

test = apmf(G_values, Rg_values, 2)