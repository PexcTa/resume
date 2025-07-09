# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:41:15 2022

@author: boris
"""
import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter, ScalarFormatter)
radius = 0.5*1.468 # in millimeters
height = 5 # in millimeters


sio2_particle_radius = 0.001 # in nanometers
tio2_shell_thickness = np.linspace(2.5,12.5,10)
total_particle_radius = np.add(tio2_shell_thickness, sio2_particle_radius)

sio2_density = 2.65 # in grams/cm3
tio2_density = 4.23

sio2_molar_mass = 60.08 # in grams/mole
tio2_molar_mass = 79.87

tio2_volume_per_particle = 1.33*math.pi*np.subtract(total_particle_radius**3, sio2_particle_radius**3) * (10**(-9)) / (10**(-2))
tio2_mass_per_particle = tio2_volume_per_particle * tio2_density
sio2_mass_per_particle = sio2_density * 1.33 * math.pi * sio2_particle_radius**3 * (10**(-9)) / (10**(-2))
total_mass_per_particle = np.add(tio2_mass_per_particle, sio2_mass_per_particle)
tio2_mass_fraction = np.divide(tio2_mass_per_particle, total_mass_per_particle)
sio2_mass_fraction = np.subtract(1, tio2_mass_fraction)
tio2_molar_fraction = np.divide(tio2_mass_fraction, tio2_molar_mass)
sio2_molar_fraction = np.divide(sio2_mass_fraction, sio2_molar_mass)
sio2_molar_fraction_normalized = np.divide(sio2_molar_fraction,sio2_molar_fraction)
tio2_molar_fraction_normalized = np.divide(tio2_molar_fraction,sio2_molar_fraction)
ti_molar_fraction_normalized = np.multiply(tio2_molar_fraction_normalized,47.87/79.87)
particle_molar_mass = np.multiply(sio2_molar_mass, sio2_molar_fraction_normalized)+np.multiply(tio2_molar_mass, tio2_molar_fraction_normalized)

cylinder_volume = (math.pi * (0.1*radius)**2 * 0.1*height) # in cm cubed
cylinder_volume *= 0.001 # in L

cylinder_volume = 0.001

required_concentration = 0.005 # in M
required_moles_Ti = cylinder_volume * required_concentration

required_moles_of_particles = np.divide(required_moles_Ti, ti_molar_fraction_normalized)
required_mass_of_particles = np.multiply(required_moles_of_particles, particle_molar_mass) * 1000 # in mg

fig, ax = plt.subplots(figsize = (12,12))

x = tio2_shell_thickness
y = required_mass_of_particles
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.tick_params(axis='x', labelsize= 25)
ax.tick_params(axis='y', labelsize= 25)
ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.set_xlabel('Thickness of the TiO$_2$ shell, nm', fontsize = 28)
ax.set_xlabel('Radius of the TiO$_2$ partlice, nm', fontsize = 28)
ax.set_ylabel('Mass of sample per reference, mg', fontsize = 28)
ax.grid()
# ax.annotate("Stock volume: "+str(round(cylinder_volume*1000,2))+ " mL", (3,2.5), fontsize = 25)
plt.scatter(x,y, s=120, color='orange')


#%% pure tio2

#capillary_parameters
radius = 0.5*1.468 # in millimeters
height = 5 # in millimeters



tio2_particle_radius = np.linspace(2.5,12.5,10) # in nanometers
tio2_particle_radius *= 10**(-9) # in meters
tio2_density = 4.23 # g/cm3
tio2_molar_mass = 79.87

tio2_volume_per_particle = (1.33*math.pi*tio2_particle_radius**3)/(10**(-2)) # in cm3
tio2_mass_per_particle = tio2_volume_per_particle * tio2_density # in g
tio2_mass_fraction = np.divide(tio2_mass_per_particle, total_mass_per_particle)
sio2_mass_fraction = np.subtract(1, tio2_mass_fraction)
tio2_molar_fraction = np.divide(tio2_mass_fraction, tio2_molar_mass)
sio2_molar_fraction = np.divide(sio2_mass_fraction, sio2_molar_mass)
sio2_molar_fraction_normalized = np.divide(sio2_molar_fraction,sio2_molar_fraction)
tio2_molar_fraction_normalized = np.divide(tio2_molar_fraction,sio2_molar_fraction)
ti_molar_fraction_normalized = np.multiply(tio2_molar_fraction_normalized,47.87/79.87)
particle_molar_mass = np.multiply(sio2_molar_mass, sio2_molar_fraction_normalized)+np.multiply(tio2_molar_mass, tio2_molar_fraction_normalized)

cylinder_volume = (math.pi * (0.1*radius)**2 * 0.1*height) # in cm cubed
cylinder_volume *= 0.001 # in L

cylinder_volume = 0.001

required_concentration = 0.005 # in M
required_moles_Ti = cylinder_volume * required_concentration

required_moles_of_particles = np.divide(required_moles_Ti, ti_molar_fraction_normalized)
required_mass_of_particles = np.multiply(required_moles_of_particles, particle_molar_mass) * 1000 # in mg

fig, ax = plt.subplots(figsize = (12,12))

x = tio2_shell_thickness
y = required_mass_of_particles
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.tick_params(axis='x', labelsize= 25)
ax.tick_params(axis='y', labelsize= 25)
ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.set_xlabel('Thickness of the TiO$_2$ shell, nm', fontsize = 28)
ax.set_xlabel('Radius of the TiO$_2$ partlice, nm', fontsize = 28)
ax.set_ylabel('Mass of sample per reference, mg', fontsize = 28)
ax.grid()
# ax.annotate("Stock volume: "+str(round(cylinder_volume*1000,2))+ " mL", (3,2.5), fontsize = 25)
plt.scatter(x,y, s=120, color='orange')


#%% Fraction of Atoms on the Surface
# SiO2
# assume 20 nm SiO2
# assume 1 -1 0 rutile plane 
results = []

import numpy as np
import matplotlib.pyplot as plt

core_radius = 10e-9

def SphereSurface(radius):
    return 4*np.pi*radius**2
def SphereVolume(radius):
    return (4/3) * np.pi * radius**3

tio2_molar_mass = 79.866 # g/mol 

# tio2 unit cell density
(a, b, c) = (4.61e-10, 4.61e-10, 2.97e-10)
cell_volume = a*b*c
# unit cell has 2 Ti atoms, 4 O atoms
cell_density = (((79.866*2)/6.022e23)/cell_volume)/1000 # kg/m3
# plane has 2 ti atoms
plane_area = 6.521e-10 * 2.973e-10 # meters

# shell thickness = 4 nm
shell_thickness = 4e-9
Inner = SphereSurface(core_radius)
Outer = SphereSurface(core_radius+shell_thickness)
ShellVolume = SphereVolume(core_radius + shell_thickness) - SphereVolume(core_radius)
tio2_mass = ShellVolume * cell_density
tio2_moles = tio2_mass*10e3/tio2_molar_mass
tio2_bulk_atoms = tio2_moles * 6.022e23

InnerAtoms = 2*Inner/plane_area 
OuterAtoms = 2*Outer/plane_area

InnerFraction = InnerAtoms/tio2_bulk_atoms
OuterFraction = OuterAtoms/tio2_bulk_atoms

results.append(InnerFraction)
results.append(OuterFraction)

# shell thickness = 4 nm
shell_thickness = 1e-9
Inner = SphereSurface(core_radius)
Outer = SphereSurface(core_radius+shell_thickness)
ShellVolume = SphereVolume(core_radius + shell_thickness) - SphereVolume(core_radius)
tio2_mass = ShellVolume * cell_density
tio2_moles = tio2_mass*10e3/tio2_molar_mass
tio2_bulk_atoms = tio2_moles * 6.022e23

InnerAtoms = 2*Inner/plane_area 
OuterAtoms = 2*Outer/plane_area

InnerFraction = InnerAtoms/tio2_bulk_atoms
OuterFraction = OuterAtoms/tio2_bulk_atoms

results.append(InnerFraction)
results.append(OuterFraction)

fig, ax = plt.subplots(figsize = (6,6))
ax.bar(['4 nm, inner','4 nm, outer','1 nm, inner','1 nm, outer' ], height = np.multiply(results, 100), width = 0.5, facecolor = 'none', edgecolor = 'black', linewidth = 4)
ax.set_xlabel('Shell Surface', fontsize = 20)
ax.tick_params(axis = 'both', labelsize = 20, rotation = 45)
ax.set_ylabel('% Atoms on Surface', fontsize = 20)
