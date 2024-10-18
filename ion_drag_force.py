#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:33:30 2024
@author: Luki
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.integrate import quad as integrate
from scipy.constants import k, m_e, e, epsilon_0
from scipy.special import erf
import optuna
from functools import partial

# Constants
eV_K = 11606
sigma_neon = 10**(-18)  # m^2
sigma_argon = 2.3 * 10**(-18)  # m^2
u = 1.660539066 * 10**(-27)  # atomic mass unit in kg
m_neon = 20.1797 * u  # neon mass in kg
m_argon = 39.948 * u  # neon mass in kg

# Functions for interpolation Pustilnik et al.

def T_e_interpolation(x, I):
    C = [7.13, 7.06, 6.98, 5.5]
    D = [1.23, 0.75, 0.77, 1.59]
    y_data = np.add(C, np.divide(D, I))
    x_data = [20, 40, 60, 100]
    coef = np.polyfit(x_data, y_data, 3)
    poly1d_fn = np.poly1d(coef)
    return poly1d_fn(x)

def n_e_interpolation(x, I):
    A = [1.92, 2.75, 3.15, 4.01]
    B = [-0.38, -0.42, -0.34, 0.047]
    y_data = np.add(np.multiply(A, I), np.multiply(B, I**2))
    x_data = [20, 40, 60, 100]
    coef = np.polyfit(x_data, y_data, 1)
    poly1d_fn = np.poly1d(coef)
    return poly1d_fn(x)

def e_field(x, I):
    F = [1.97, 2.11, 2.07, 1.94]
    G = [0.14, 0.072, 0.098, 0.12]
    y_data = np.add(F, np.divide(G, I**2))
    x_data = [20, 40, 60, 100]
    coef = np.polyfit(x_data, y_data, 1)
    poly1d_fn = np.poly1d(coef)
    return poly1d_fn(x)

# Variable Parameters
I = 1.5  # mA
p = np.array([15, 20, 25, 30, 40, 100])  # Pa
z = np.array([0.2] * len(p))  # Charge potential adjsutable 0.3 +- 0.1 NEON, 0.4 +-1 ARGON; Antonova et. al.
epstein = [1.26] * len(p)  # Neutral damping Epstein coefficient NEON = 1.44; ARGON = 1.26!
a = (3.4 / 2) * 10**(-6)  # Micrometer particle radius
n_d = np.array([.1] * len(p)) * 10**11  #? Dust number density in m^-3; not sure about this value

# Calculations
E_0_calc = [e_field(15, I), e_field(20, I), e_field(25, I), e_field(30, I), e_field(40, I), e_field(100, I)]
E_0 = np.multiply(E_0_calc, 100)  # V/m
T_e = np.array([T_e_interpolation(15, I), T_e_interpolation(20, I), T_e_interpolation(25, I), T_e_interpolation(30, I), T_e_interpolation(40, I), T_e_interpolation(100, I)])
T_n = 0.025  # eV
l_i = np.divide(T_n * k, p * sigma_argon)
T_i = (np.multiply(2 / 9 * np.abs(np.multiply(E_0, 1)) * e / k, l_i) + 0.03)
n_e0 = np.multiply([n_e_interpolation(15, I), n_e_interpolation(20, I), n_e_interpolation(25, I), n_e_interpolation(30, I), n_e_interpolation(40, I), n_e_interpolation(100, I)], 10**14)
n_0 = p / (k * T_n * eV_K) * 10**(-6)  # cm^-3
m_d = 4 / 3 * np.pi * a**3 * 1574  # Dust mass

# Other Equations
v_tn = np.sqrt(8 * k * T_n * eV_K / (np.pi * m_argon))
v_ti = np.sqrt(8 * k * T_i * eV_K / (np.pi * m_argon))
Z_d = 4 * np.pi * epsilon_0 * k * T_e * eV_K * a * z / (e**2)
n_i0 = np.add(n_e0, np.multiply(Z_d, n_d))

# Debye lengths
debye_De = np.sqrt(epsilon_0 * k * T_e * eV_K / (n_e0 * e**2))
debye_Di = np.sqrt(epsilon_0 * k * T_i * eV_K / (n_i0 * e**2))
debye_D = np.divide(np.multiply(debye_De, debye_Di), np.sqrt(debye_De**2 + debye_Di**2))

# Integration for ion drag force
def integration_function(x, debye_Di_val, roh_0_val):
    return 2 * np.exp(-x) * np.log((2 * debye_Di_val * x + roh_0_val) / (2 * a * x + roh_0_val))

roh_0 = np.divide(Z_d , T_i*11606) * e**2 / (4 * np.pi * epsilon_0 * k)
#
integrated_f = np.array([
    integrate(lambda x: integration_function(x, debye_Di[i], roh_0[i]), 0, np.inf)[0]
    for i in range(len(p))
])

'''    Modified Frost Formular    '''
A = 0.0168 #NEON = 0.0321
B = 0.007  #NEON = 0.012
C = 1.238  #NEON = 1.181
EN = (np.divide(E_0, 100)/n_0)*(10**17) #10^17 from Vcm^2 to Td
M = A * np.abs((1 + np.abs((B * EN)**C))**(-1/(2*C))) * EN
v_ti2 = np.sqrt(k * T_i * 11606 / m_argon)
u_i = M*v_ti2

# Electric and ion drag forces
F_e = Z_d * e * E_0
F_i = np.multiply(n_i0,((8*np.sqrt(2*np.pi))/3) * m_argon * (v_ti) * (u_i) * (a**2 + a*roh_0/2 +(roh_0**2) * integrated_f/4))

# Particle velocity
factor = np.multiply(epstein, (4 / 3) * np.pi * a**2 * m_argon * v_tn * (p / (T_n * eV_K * k)))
v_d = (F_e + F_i) / factor

# Plotting
fig, ax = plt.subplots(dpi=600)
fig.set_size_inches(4, 3)
ax.plot(p, np.abs(F_e) * 10**14, linestyle='solid', marker='^', color='#00429d', linewidth=.75)
ax.plot(p, F_i * 10**14, linestyle='solid', marker='x', color='#00cc00', linewidth=.75)
ax.legend(['$F_e x 10^{-14}$', '$F_i x 10^{-14}$'])
ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
plt.show()

fig, ax = plt.subplots(dpi=600)
fig.set_size_inches(4, 3)
ax.scatter(p, v_d*1000, marker='x', linestyle='solid', color='#00cc00', linewidth=.7)
ax.legend(['Theory1 $v_{group}$'])
plt.xlabel('Pressure [Pa]')
plt.ylabel('$v_{mean}$ [mm/s]')
ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
plt.show()
