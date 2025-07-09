# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:16:30 2024

@author: boris
"""

from scipy.optimize import lsq_linear
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

#%%

def PosNormal(mean, sigma):
    x = np.random.normal(mean,sigma,1)
    return(x if x>=0 else PosNormal(mean,sigma))
#%%
edsTP1 = np.array([6.75, 17.21, 13.01, 63.03])
edsTP1_err = np.array([1.18, 0.62, 1.15, 0.95])
edsTP2_1 = np.array([8.58, 24.31, 13.60, 53.51])
edsTP2_1_err = np.array([8.00, 6.20, 4.89, 4.77])
edsTP2_2 = np.array([14.28, 20.08, 10.89, 54.76])
edsTP2_2_err = np.array([5.05, 2.85, 1.96, 2.36])
edsTP2_3 = np.array([12.12, 21.29, 11.76, 54.85])
edsTP2_3_err = np.array([3.18, 1.84, 1.12, 2.88])
# edsTP2_4 = np.array([12.12, 21.29, 11.76, 54.85])
# edsTP2_4_err = np.array([3.18, 1.84, 1.12, 2.88])

edsTP1_abs = np.array([0.3,	1.00, 0.76, 3.66])
edsTP1_abs_err = np.array([0.07, 0.04, 0.07, 0.05])

edsThPhos8 = np.array([12.40, 7.73, 13.52, 65.85])
edsThPhos8_err = np.array([1.06, 1.48, 1.13, 1.57])

data = edsThPhos8
erro = edsThPhos8_err
# 
samples = np.zeros(shape = [50000, 4], dtype = float)
# for i in range(50000):
#     samples[i,0] = PosNormal(data[0], erro[0])
samples[:,:3] = np.random.normal(data[:3], erro[:3], (50000,3))
for i in range(50000):
    samples[i,3] = np.subtract(100, np.sum(samples[i,:3]))

fig, ax = plt.subplots(1, 4, figsize = (16,4))

for axis, i in zip(ax, range(4)):
    axis.hist(samples[:,i], bins = 100)
    
plt.show()

#%%
# NaTh2(PO4)3, ThO2, NaPO4, Th4(PO4)4P207
solver_phos1=np.array([[5.56, 11.11, 16.67, 66.67],
                      [0, 33.33, 0, 66.67],
                      [37.50, 0, 12.50, 50.00],
                      [0, 0, 0, 100]])

# Na2Th(PO4)2, ThO2, NaPO4, Th4(PO4)4P207
solver_phos2=np.array([[15.38, 7.69, 15.38, 61.54],
                      [0, 33.33, 0, 66.67],
                      [37.50, 0, 12.50, 50.00],
                      [0, 0, 0, 100]])

# Na2Th(PO4)2, NaTh2(PO4)3, ThO2, nothing
solver_phos3=np.array([[15.38, 7.69, 15.38, 61.54],
                      [5.56, 11.11, 16.67, 66.67],
                      [0, 33.33, 0, 66.67],
                      [0, 0, 0, 0]])

solver_phos1_abs = np.array([[1, 2, 3, 12],
                      [0, 1, 0, 2],
                      [3, 0, 1, 4],
                      [0, 0, 0, 1]]) 


n = solver_phos3.shape[1]

A = solver_phos3.T

num_samples = samples.shape[0]
output = np.zeros(shape = [num_samples, 4])

for i in range(num_samples):
    b = samples[i,:]
    res = lsq_linear(A, b, bounds=np.array([(0.,np.inf) for i in range(n)]).T, lsmr_tol='auto', verbose=0)
    y = res.x
    output[i,:] = y
    
#%% 
normalized_output = np.zeros_like(output)
for i in range(output.shape[0]):
    for j in range(4):
        normalized_output[i,j] = output[i,j]/np.sum(output[i,:])
        # normalized_output[i,j] = output[i,j]/1

    
#%%
# np.random.seed(18475)

eds = data
eds_ = erro
solver = solver_phos1
labels0 = ['Na', 'Th', 'P', 'O']
labels1 = ['Na Content, %', 'Th Content, %', 'P Content, %', 'O Content, %']
# labels2 = ['NaTh$_2$(PO$_4$)$_3$', 'ThO$_2$', 'Na$_3$PO$_4$', 'Th$_4$(PO$_4$)$_4$(P$_2$O$_7$)']
labels2 = ['NaTh$_2$(PO$_4$)$_3$', 'ThO$_2$', 'Na$_3$PO$_4$', 'H$_2$O']
# labels3 = ['Na$_2$Th(PO$_4$)$_2$', 'ThO$_2$', 'Na$_3$PO$_4$', 'Th$_4$(PO$_4$)$_4$(P$_2$O$_7$)']
labels3 = ['Na$_2$Th(PO$_4$)$_2$', 'ThO$_2$', 'Na$_3$PO$_4$', 'H$_2$O']
labels4 = ['Na$_2$Th(PO$_4$)$_2$', 'NaTh$_2$(PO$_4$)$_3$', 'ThO$_2$', 'null']

colors1 = ['brown', 'darkmagenta', 'orange', 'red']
colors2 = ['darkcyan', 'gray', 'forestgreen', 'black']
colors3 = ['darkblue', 'gray', 'forestgreen', 'black']
fig = plt.figure(layout = 'constrained', figsize = (32,18))
fs1 = 28
fs2 = 24
# fig.suptitle(f'EDS ThPhos @ pH = 7.5 | Given: Na {eds[0]}$\pm${eds_[0]}%, Th {eds[1]}$\pm${eds_[1]}%, P {eds[2]}$\pm${eds_[2]}%, O {eds[3]}$\pm${eds_[3]}%', fontsize = 42, y = 1.05)
gs = GridSpec(5, 4, figure=fig)

ax1 = fig.add_subplot(gs[0,0])
ax2,ax3,ax4 = fig.add_subplot(gs[1,0]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[3,0])
ax1.hist(samples[:,0], bins = 100, color = 'brown', alpha = 0.75)
ax2.hist(samples[:,1], bins = 100, color = 'darkmagenta', alpha = 0.75)
ax3.hist(samples[:,2], bins = 100, color = 'orange', alpha = 0.75)
ax4.hist(samples[:,3], bins = 100, color = 'red', alpha = 0.75)
for ax, i in zip([ax1, ax2, ax3, ax4], range(len(labels1))):
    ax.tick_params(axis='both', labelsize= fs2)
    ax.set_yticks([])
    ax.set_xlabel(labels1[i], fontsize = fs1)
    
    
plot_sample = np.random.choice(range(samples.shape[0]), 25)
ax5 = fig.add_subplot(gs[0:2, 1:3])


    
# for i in plot_sample:
#     ax5.scatter(labels0, samples[i,:], facecolor = 'none', edgecolor = 'black', s = 100, linewidth = 0.5)

count = 0
for i in plot_sample:
    solution = normalized_output[i, :]
    percentage = np.zeros_like(solution)
    for j in range(4):
        percentage[j] = np.sum(np.multiply(solution, solver[:,j]))
    count+=1
    if count != 24:
        ax5.plot(labels0, percentage, color = 'darkgray', alpha = 0.25, linewidth = 5, label = '_')
    else:
        ax5.plot(labels0, percentage, color = 'darkgray', alpha = 0.25, linewidth = 5, label = 'Sample Solutions')
    ax5.legend(fontsize = fs2, loc = 'upper left')
    
for i in range(4):   
    ax5.bar(x = labels0[i], height = eds[i], linewidth = 3, color = colors1[i], 
            width = 0.2)
    ax5.errorbar(labels0[i], eds[i], yerr = eds_[i], color='Black', elinewidth=2.5,capthick=2.5, errorevery=1, alpha=1, ms=4, capsize = 5)
    ax5.set_ylabel("Content (%)", fontsize = fs1)
    ax5.tick_params(axis='both', labelsize= fs2)

num_solutions = len(plot_sample)
groups = range(num_solutions)
values = np.zeros(shape=[len(solution), num_solutions])
for i in groups:
    values[:,i] = normalized_output[plot_sample[i],:]
ax6 = fig.add_subplot(gs[2:4, 1:3])
ax6.set_ylim(0, 1.2)
ax6.set_xlim(-0.9, 24.9)
ax6.set_ylabel("Proportion", fontsize = fs1)
ax6.set_xlabel("Sample Solutions", fontsize = fs1)
ax6.tick_params(axis='both', labelsize= fs2)
for i in range(values.shape[0]):
    ax6.bar(groups, values[i], bottom = np.sum(values[:i], axis = 0), label = labels4[i], color = colors3[i])
ax6.legend(loc = 'upper center', fontsize = fs2, ncol = 4)
ax6.xaxis.set_major_locator(MultipleLocator(1))
ax6.axhline(y = 1, linewidth = 3, linestyle = '--', color = 'red')
ax6.axhline(y = 0, linewidth = 3, linestyle = '--', color = 'red')

ax7 = fig.add_subplot(gs[0,-1])
ax8,ax9,ax10 = fig.add_subplot(gs[1,-1]),fig.add_subplot(gs[2,-1]),fig.add_subplot(gs[3,-1])
ax7.hist(normalized_output[:,0], bins = 100, color = colors3[0], alpha = 0.75)
ax8.hist(normalized_output[:,1], bins = 100, color = 'gray', alpha = 0.75)
ax9.hist(normalized_output[:,2], bins = 100, color = 'forestgreen', alpha = 0.75)
ax10.hist(normalized_output[:,3], bins = 100, color = 'black', alpha = 0.75)
for ax, i in zip([ax7, ax8, ax9, ax10], range(len(labels2))):
    ax.tick_params(axis='both', labelsize= fs2)
    # ax.set_yticks([])
    ax.set_xlabel(f'{labels4[i]} share', fontsize = fs1)
    ax.set_ylabel('Frequency', fontsize = fs1)
    
# plt.tight_layout()
plt.show()

#%% solve a binary phase
element_number = 3
ratios_1 = [2, 1, 2]
ratios_2 = [1, 2, 3]
comp = np.linspace(0,1,1000)

traces = np.zeros(shape = (len(comp), element_number), dtype = float)
parts = np.zeros(shape = (len(comp), element_number), dtype = float)

for i in range(element_number):
    traces[:,i] = ratios_1[i]*comp + ratios_2[i]*comp[::-1]
for j in range(len(comp)):
    for i in range(element_number):
        parts[j,i] = (traces[j,i]/np.sum(traces[j,:]))*100
        
experimental = [12.53, 7.97, 13.99]
exp_norm = [(i/sum(experimental))*100 for i in experimental]
labels = ['Na %', 'Th %', 'P %']
fig, ax = plt.subplots(figsize = (12,8))
for i in range(element_number):
    ax.plot(comp, parts[:,i], label = labels[i])
