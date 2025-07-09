r"""
Definition
----------

This model employs the empirical multiple level unified Exponential/Power-law
fit method developed by Beaucage. Four functions are included so that 1, 2, 3,
or 4 levels can be used. In addition a 0 level has been added which simply
calculates

.. math::

    I(q) = \text{scale} / q + \text{background}

The Beaucage method is able to reasonably approximate the scattering from
many different types of particles, including fractal clusters, random coils
(Debye equation), ellipsoidal particles, etc.

The model works best for mass fractal systems characterized by Porod exponents
between 5/3 and 3. It should not be used for surface fractal systems. Hammouda
(2010) has pointed out a deficiency in the way this model handles the
transitioning between the Guinier and Porod regimes and which can create
artefacts that appear as kinks in the fitted model function.

Also see the :ref:`guinier-porod` model.

The empirical fit function is:

.. math::

    I(q) = \text{background}
    + \sum_{i=1}^N \Bigl[
        G_i \exp\Bigl(-\frac{q^2R_{gi}^2}{3}\Bigr)
       + B_i \exp\Bigl(-\frac{q^2R_{g(i+1)}^2}{3}\Bigr)
             \Bigl(\frac{1}{q_i^*}\Bigr)^{P_i} \Bigr]

where

.. math::

    q_i^* = q \left[\operatorname{erf}
            \left(\frac{q R_{gi}}{\sqrt{6}}\right)
        \right]^{-3}


For each level, the four parameters $G_i$, $R_{gi}$, $B_i$ and $P_i$ must
be chosen.  Beaucage has an additional factor $k$ in the definition of
$q_i^*$ which is ignored here.

For example, to approximate the scattering from random coils (Debye equation),
set $R_{gi}$ as the Guinier radius, $P_i = 2$, and $B_i = 2 G_i / R_{gi}$

See the references for further information on choosing the parameters.

For 2D data: The 2D scattering intensity is calculated in the same way as 1D,
where the $q$ vector is defined as

.. math::

    q = \sqrt{q_x^2 + q_y^2}


References
----------

#. G Beaucage, *J. Appl. Cryst.*, 28 (1995) 717-728
#. G Beaucage, *J. Appl. Cryst.*, 29 (1996) 134-146
#. B Hammouda, *Analysis of the Beaucage model*,
   *J. Appl. Cryst.*, (2010), 43, 1474-1478

Authorship and Verification
----------------------------

* **Author:**
* **Last Modified by:**
* **Last Reviewed by:**
"""

from __future__ import division

import numpy as np
from numpy import inf, exp, sqrt, errstate
from scipy.special import erf, gamma

category = "shape-independent"
name = "unified_power_Rg"
title = "Unified Power Rg"
description = """
        The Beaucage model employs the empirical multiple level unified
        Exponential/Power-law fit method developed by G. Beaucage. Four functions
        are included so that 1, 2, 3, or 4 levels can be used.
        """

# pylint: disable=bad-whitespace, line-too-long
parameters = [
    ["level",     "",     1,      [1, 6], "", "Level number"],
    ["rg[level]", "Ang",  15.8,   [0, inf], "", "Radius of gyration"],
    ["power[level]", "",  4,      [-inf, inf], "", "Power"],
    ["B[level]",  "1/cm", 4.5e-6, [-inf, inf], "", ""],
    ["G[level]",  "1/cm", 400,    [0, inf], "", ""],
    ]
# pylint: enable=bad-whitespace, line-too-long

def Iq(q, level, rg, power, B, G):
    """Return I(q) for unified power Rg model."""
    level = int(level + 0.5)
    
    result_dict = {}
    
    if level == 0:
        with errstate(divide='ignore'):
            return 1./q

    with errstate(divide='ignore', invalid='ignore'):
        result = np.zeros(q.shape, 'd')
        for i in range(level):
            exp_now = exp(-(q*rg[i])**2/3.)
            pow_now = (erf(q*rg[i]/sqrt(6.))**3/q)**power[i]
            if i < level-1:
                exp_next = exp(-(q*rg[i+1])**2/3.)
            else:
                exp_next = 1
            result += G[i]*exp_now + B[i]*exp_next*pow_now
            result_dict[f'G{i+1}'] = G[i]*exp_now
            result_dict[f'B{i+1}'] = B[i]*exp_next*pow_now
            
    result[q == 0] = np.sum(G[:level])
    return result, result_dict

Iq.vectorized = True

def random():
    """Return a random parameter set for the model."""
    level = np.minimum(np.random.poisson(0.5) + 1, 6)
    n = level
    power = np.random.uniform(1.6, 3, n)
    rg = 10**np.random.uniform(1, 5, n)
    G = np.random.uniform(0.1, 10, n)**2 * 10**np.random.uniform(0.3, 3, n)
    B = G * power / rg**power * gamma(power/2)
    scale = 10**np.random.uniform(1, 4)
    pars = dict(
        #background=0,
        scale=scale,
        level=level,
    )
    pars.update(("power%d"%(k+1), v) for k, v in enumerate(power))
    pars.update(("rg%d"%(k+1), v) for k, v in enumerate(rg))
    pars.update(("B%d"%(k+1), v) for k, v in enumerate(B))
    pars.update(("G%d"%(k+1), v) for k, v in enumerate(G))
    return pars

# multi-shell models want demo parameters
demo = dict(
    level=2,
    rg=[15.8, 21],
    power=[4, 2],
    B=[4.5e-6, 0.0006],
    G=[400, 3],
    scale=1.,
    background=0.,
)

#%% sample 2401 (ThPhos-5)
data = np.loadtxt('2401_data.txt', skiprows=1)
fit = np.loadtxt('2401_fit.txt', skiprows=1)
resid = np.loadtxt('2401_resid.txt', skiprows=1)


q = data[:,0]
lvl = 2
scale = 657.76
res, sections = Iq(q, lvl, [7.9489, 122.29], [4, 3.8896], [1e-6, 2.4446e-6], [0.001406, 39.718])

res2, sections2 = Iq(q, 1, [2.2], [4], [4.5e-6], [0.1])


def pwrlaw(q, scale, alpha):
    return scale * q ** (-alpha)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (12, 12))
fs1 = 28
fs2 = 24
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel('log $q$ ($\AA^{-1}$)', fontsize = fs1)
ax.set_ylabel('log $I$ (a.u.)', fontsize = fs1)

lowQpwr = 2
bkg = pwrlaw(q, 1.275, lowQpwr)

fit_lowq = np.min(fit[:,0])
idx_lowq = (np.abs(q - fit_lowq)).argmin()

ax.loglog(q, data[:,1], 'o-', color='black', markersize = 15, markerfacecolor='none', linewidth = 2.5, alpha = 0.5, label = 'ThPhos-5')
ax.loglog(fit[:,0], fit[:,1], '-', color='crimson', linewidth = 3.5, label = 'Unified fit')
ax.loglog(q[:idx_lowq], bkg[:idx_lowq], '--', color='indigo', linewidth = 3.5, alpha = 1, label = 'Power Law')
ax.loglog(q, bkg, '--', color='indigo', linewidth = 3.5, alpha = 0.2)

# ax.loglog(q, scale*2*sections['G1'], '-', color='forestgreen', linewidth = 3.5, label = 'Guinier Level 1')
# ax.loglog(q, scale*2*sections['B1'], '-', color='yellowgreen', linewidth = 3.5, label = 'Porod Level 1')

# ax.loglog(q, scale*sections['G2'], '-', color='darkblue', linewidth = 3.5, label = 'Guinier Level 2')
# ax.loglog(q, scale*sections['B2'], '-', color='indigo', linewidth = 3.5, label = 'Porod Level 2')


ax.set_ylim((np.min(data[:,1]), np.max(data[:,1])+5*np.max(data[:,1])))

ax.legend(loc = 'upper right', fontsize = fs1)
plt.tight_layout()

#%% sample 2402 (ThPhos-8)
data = np.loadtxt('2402_data.txt', skiprows=1)
fit = np.loadtxt('2402_fit.txt', skiprows=1)
resid = np.loadtxt('2402_resid.txt', skiprows=1)


q = data[:,0]
lvl = 2
scale = 657.76
res, sections = Iq(q, lvl, [10.226, 55.946], [4, 4], [4.5e-6, 0.013354], [0.010716, 6640.5])

res2, sections2 = Iq(q, 1, [2.2], [4], [4.5e-6], [0.1])


def pwrlaw(q, scale, alpha):
    return scale * q ** (-alpha)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (12, 12))
fs1 = 28
fs2 = 24
ax.tick_params('both', labelsize = fs2)
ax.set_xlabel('log $q$ ($\AA^{-1}$)', fontsize = fs1)
ax.set_ylabel('log $I$ (a.u.)', fontsize = fs1)

lowQpwr = 3.15
bkg = pwrlaw(q, 0.0016, lowQpwr)

fit_lowq = np.min(fit[:,0])
idx_lowq = (np.abs(q - fit_lowq)).argmin()

ax.loglog(q, data[:,1], 'o-', color='black', markersize = 15, markerfacecolor='none', linewidth = 2.5, alpha = 0.5, label = 'ThPhos-8')
ax.loglog(fit[:,0], fit[:,1], '-', color='crimson', linewidth = 3.5, label = 'Unified fit')
ax.loglog(q[:idx_lowq], bkg[:idx_lowq], '--', color='indigo', linewidth = 3.5, alpha = 1, label = 'Power Law')
ax.loglog(q, bkg, '--', color='indigo', linewidth = 3.5, alpha = 0.2)

# ax.loglog(q, scale*2*sections['G1'], '-', color='forestgreen', linewidth = 3.5, label = 'Guinier Level 1')
# ax.loglog(q, scale*2*sections['B1'], '-', color='yellowgreen', linewidth = 3.5, label = 'Porod Level 1')

# ax.loglog(q, scale*sections['G2'], '-', color='darkblue', linewidth = 3.5, label = 'Guinier Level 2')
# ax.loglog(q, scale*sections['B2'], '-', color='indigo', linewidth = 3.5, label = 'Porod Level 2')


ax.set_ylim((np.min(data[:,1]), np.max(data[:,1])+5*np.max(data[:,1])))

ax.legend(loc = 'upper right', fontsize = fs1)
plt.tight_layout()