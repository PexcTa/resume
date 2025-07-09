# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:49:27 2025

@author: boris
"""

import pandas, numpy, os

#%%

# keys = [f'220{i}' for i in range(1, 7)]
# keys += [f'22{i}' for i in range(10, 14)]
# keys += [f'22{i}' for i in range(21, 26)]
keys = ['1703', '1715', '1716', '2223']

# keys += ['lupolen', 'water', 'waterCitrate', 'kapton']

files = list(os.listdir())

rows_list = []
for key in keys:
    for file in files:
        if key in str(file):
            with open(file):
                data = numpy.loadtxt(file, dtype = float, skiprows=1)
                value = 0.03
                id_q = (numpy.abs(data[:,0] - value)).argmin()
                rows_list.append({'key': key, 'q': data[id_q, 0], 'int': data[id_q, 1]})
                files.remove(file)
        else:
            continue
                
        
intensities = pandas.DataFrame(rows_list)

#%%
source = numpy.loadtxt('calfactors_current.txt', dtype = 'str', skiprows = 1)
calibration_factors = dict(zip(source[:,0], [float(i) for i in source[:,3]]))   

#%%

keys = [f'220{i}' for i in range(1, 7)]
keys += [f'22{i}' for i in range(10, 14)]
keys += [f'22{i}' for i in range(21, 26)]
keys += [f'22{i}' for i in range(21, 26)]
keys += [f'22{i}' for i in range(21, 26)]
keys += [f'170{i}' for i in range(10)]
keys += ['1712d', '1713d']
keys += [f'17{i}' for i in range(10,25)]
keys += ['lupolen']


files = list(os.listdir())

rows_list = []
for key in keys:
    for file in files:
        if key in str(file):
            with open(file):
                data = numpy.loadtxt(file, dtype = float, skiprows=1)
                new = numpy.zeros_like(data)
                new[:,0] = data[:,0]
                factor = calibration_factors[key]
                new[:,1] = numpy.multiply(data[:,1], factor)
                numpy.savetxt(f'sas{key}_q_Iabs_recal.dat', new)
                files.remove(file)
#%%
                
files = list(os.listdir())
keys = ['sas1700_q_Iabs_av.dat']

rows_list = []
for key in keys:
    for file in files:
        if key in str(file):
            with open(file):
                data = numpy.loadtxt(file, dtype = float, skiprows=1)
                new = numpy.zeros_like(data[:,:-1])
                new[:,0] = data[:,0]
                factor = 0.000149973
                new[:,1] = numpy.divide(data[:,1], factor)
                numpy.savetxt(f'{key}_Uncab.dat', new)
                files.remove(file)