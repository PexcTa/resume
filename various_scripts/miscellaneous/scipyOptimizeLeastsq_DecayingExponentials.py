# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:19:30 2023

@author: boris
"""

#%%
import numpy as np
from scipy.optimize import curve_fit
from scipy import special
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# The two-dimensional domain of the fit.
ymin, ymax, ny = 400, 800, 400
xmin, xmax, nx = 0.1, 3.5, 350
x, y = np.logspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)


def normalize_n(data, n):
    """Normalizes data to the specified value"""
    normalized_data = (data/np.amax(data))/(1/n)
    return normalized_data
#%% construct simulated TA dataset
fA = normalize_n(scipy.stats.norm(loc = 450, scale = 50/2).pdf(Y) * np.exp(-(X)/10), -2)
fB = normalize_n(scipy.stats.norm(loc = 600, scale = 200/2).pdf(Y) * np.exp(-(X)/1000), 0.75)
fC = normalize_n(scipy.stats.norm(loc = 750, scale = 100/2).pdf(Y) * np.exp(-(X)/25), 3)
fS = scipy.stats.norm(1, 0.5).cdf(X) * (fA+fB+fC)
fS += 0.1*np.random.randn(*fS.shape)
fig, ax = plt.subplots(figsize = (10,10))
cset = ax.contourf(Y,np.log10(X), fS, cmap='plasma')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(cset, cax=cax, orientation='vertical')
cbar.set_ticks([np.linspace(fS.min().min(), fS.max().max(), 6)])
cax.set_yticklabels(['{:.3f}'.format(y) for y in np.linspace(fS.min().min(), fS.max().max(), 6)])
plt.show()

#%%
u, s, v = np.linalg.svd(fS)

fig, ax = plt.subplots(1,3,figsize = (12,4))
for i in range(3):
    ax[0].plot(y, u[:,i])
ax[1].bar(x = range(20), height = s[:20])
for i in range(3):
    ax[2].plot(np.log10(x), v[i,:])

#%% 
# Our function to fit is going to be a sum of one-dimensional gaussians convoluted with decaying exponentials
def dataFunc(x, fwhm, x0, tau, A):
    return A*fwhm*np.exp((fwhm/(np.sqrt(2)*tau))**2-(x-x0)/tau)*(1-special.erf(fwhm/(np.sqrt(2)*tau)-(x-x0)/(np.sqrt(2)*fwhm)))
# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _dataFunc(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
       arr += dataFunc(x, *args[i*4:i*4+4])
    return arr

# Initial guesses to the fit parameters.
guess_prms = [(0.5, 1, 10, 0),
              (0.5, 1, 1000, 0),
              (0.5, 1, 25, 0)]
# Flatten the initial guess parameter list.
p0 = [p for prms in guess_prms for p in prms]
#%%
# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
xdata = np.vstack((Y.ravel(), X.ravel()))
#%%
# Do the fit, using our custom _gaussian function which understands our
# flattened (ravelled) ordering of the data points.
popt, pcov = curve_fit(_dataFunc, xdata, fS.ravel(), p0)
fit = np.zeros(fS.shape)
for i in range(len(popt)//4):
    fit += dataFunc(X, *popt[i*4:i*4+4])
print('Fitted parameters:')
print(popt)

rms = np.sqrt(np.mean((fS - fit)**2))
print('RMS residual =', rms)
#%%
# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, fit, cmap='plasma')
cset = ax.contourf(X, Y, fS-fit, zdir='z', cmap='plasma')
ax.set_zlim(-4,np.max(fit))
plt.show()

# Plot the test data as a 2D image and the fit as overlaid contours.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(fS, origin='lower', cmap='plasma',
          extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(Y, X, fit, colors='w')
plt.show()

#%%
fig, ax = plt.subplots(figsize = (10,10))
cset = ax.contourf(Y,np.log10(X), fS, cmap='plasma')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(cset, cax=cax, orientation='vertical')
cbar.set_ticks([np.linspace(fS.min().min(), fS.max().max(), 6)])
cax.set_yticklabels(['{:.3f}'.format(y) for y in np.linspace(fS.min().min(), fS.max().max(), 6)])
plt.show()