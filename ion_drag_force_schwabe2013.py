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
from scipy.optimize import curve_fit, fsolve
import os

# Constants
eV_K = 11606
sigma_neon = 10**(-18)  # m^2
sigma_argon = 2.3 * 10**(-18)  # m^2
u = 1.660539066 * 10**(-27)  # atomic mass unit in kg

atomic_mass_neon = 20.1797
atomic_mass_argon = 39.948

m_neon = atomic_mass_neon * u  # neon mass in kg
m_argon = atomic_mass_argon * u  # neon mass in kg

# Argon Data Interpolated from Zobnin unpublished Data (measured by Langmur Probe in Pk-4)
# Load the JSON file
file_path = "argon_interpolation/argon_params.json"
# Results file path in the same directory
result_path = os.path.join("json_files/", "mean_v_results.json")

with open(file_path, "r") as file:
    data = json.load(file)
# Read existing data or initialize empty dictionary if file doesn't exist
with open(result_path, 'r') as file:
    results = json.load(file)

# Extract data for a given current and pressure range
def extract_plasma_data(data, current, pressure_range):
    # Ensure the selected current is valid
    if current not in data:
        raise ValueError(f"Invalid current selected. Available options: {list(data.keys())}")
    # Extract the data for the selected current
    pressure_data = np.array(data[current]["Pressure (Pa)"])
    E_0 = np.array(data[current]["E"])  # Electric field strength
    T_e = np.array(data[current]["T"])  # Electron temperature
    n_e0 = np.array(data[current]["n"])  # Electron number density

    # Interpolate to match the requested pressure range
    E_0_interp = np.interp(pressure_range, pressure_data, E_0)
    T_e_interp = np.interp(pressure_range, pressure_data, T_e)
    n_e0_interp = np.interp(pressure_range, pressure_data, n_e0)

    return E_0_interp, T_e_interp, n_e0_interp

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

def model_schwabe(gas_type, I, polarity, E_multiplier, ne_multiplier, Te_multiplier, z_multiplier, e=e):

    # Model:
    model = "Schwabe2013"
    display_plots = False
    #
    charge_depletion = 1
    T_e_argon_neon_translation = 0.45
       
    selected_current = str(I)+"mA"
    #
    p = np.array([15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120])  # Pa
    
    if gas_type == "Neon":
        #p = [15,   20,   25,   30,   40,   50,  60,  70,  80,  90, 100, 120]
        z = [0.37, 0.36, 0.36, 0.33, .31, .29, .26, .26, .26, .26, .26, .3]
        epstein = [1.44] * len(p)  # Neutral damping Epstein coefficient, euqal in Neon and Argon
    else:
        #z = [.58, .49, .45, .43, .42, .41, .4, .38, .37, .36, .35, .35]  # Charge potential adjsutable 0.3 +- 0.1 NEON, 0.4 +-1 ARGON; Antonova et. al.
        #ref_p = [15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        z = [ .59, .52, .4, .4, .4, .4, .4, .4, .37, .36, .34, .32]
        epstein = [1.44] * len(p)  # Neutral damping Epstein coefficient, equal in Neon and Argon
    z = np.array(z)*z_multiplier    
        
    a = (3.4 / 2) * 10**(-6)  # Micrometer particle radius
    
    if gas_type == "Argon" and I == 1.5:
        n_d = np.array([0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.4, 0.4, 0.6, 0.7, 0.8]) * 10**11
    elif gas_type == "Argon" and I == 1:
        n_d = np.array([0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.4, 0.4, 0.6, 0.7, 0.8]) * 10**11
    elif gas_type == "Neon":
                    #p= [ 15,   20,   25,   30,  40, 50,  60, 70,  80,  90,  100, 120] # Pa
        n_d = np.array([0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8]) * 10**11
    # Extract the data for the selected current
    try:
        E_0_argon, T_e_argon, n_e0_argon = extract_plasma_data(data, selected_current, p)
        E_0_argon = np.multiply(E_0_argon, -100*E_multiplier)  # V/m
        n_e0_argon = n_e0_argon * ne_multiplier
        T_e_argon = T_e_argon * Te_multiplier
    except ValueError as error_data:
        print(error_data)
    
    # Calculations Neon
    E_0_calc = [e_field(15, I), e_field(20, I), e_field(25, I), e_field(30, I), e_field(40, I), e_field(50, I), e_field(60, I), e_field(70, I), e_field(80, I), e_field(90, I), e_field(100, I), e_field(120, I)]
    E_0 = np.multiply(E_0_calc, -100*E_multiplier)  # V/m
    T_e = np.multiply(np.array([T_e_interpolation(15, I), T_e_interpolation(20, I), T_e_interpolation(25, I), T_e_interpolation(30, I), T_e_interpolation(40, I), T_e_interpolation(50, I), T_e_interpolation(60, I), T_e_interpolation(70, I), T_e_interpolation(80, I), T_e_interpolation(90, I), T_e_interpolation(100, I), T_e_interpolation(120, I)]), Te_multiplier)
    n_e0 = np.multiply([n_e_interpolation(15, I), n_e_interpolation(20, I), n_e_interpolation(25, I), n_e_interpolation(30, I), n_e_interpolation(40, I), n_e_interpolation(50, I), n_e_interpolation(60, I), n_e_interpolation(70, I), n_e_interpolation(80, I), n_e_interpolation(90, I), n_e_interpolation(100, I), n_e_interpolation(120, I)], ne_multiplier * 10**14)

    if gas_type == "Argon":
        T_e_argon = T_e * Te_multiplier * T_e_argon_neon_translation
        E_0_argon = E_0
    else:
        T_e = T_e * Te_multiplier

    
    T_n = 0.025  # eV
    if gas_type == "Neon":
        l_i = np.divide(T_n  * eV_K * k, p * sigma_neon)
        T_i = (2 / 9 * np.abs(E_0) * e / k * l_i) + 0.025 * eV_K
    else:
        l_i = np.divide(T_n  * eV_K * k, p * sigma_argon)
        T_i = (2 / 9 * np.abs(E_0_argon) * e / k * l_i) + 0.025 * eV_K
    n_0 = p / (k * T_n * eV_K) * 10**(-6)  # cm^-3
    m_d = 4 / 3 * np.pi * a**3 * 1574  # Dust mass
    
    # Other Equations
    if gas_type == "Neon":
        v_tn = np.sqrt(8 * k * T_n * eV_K / (np.pi * m_neon))
        v_ti = np.sqrt(8 * k * T_i / (np.pi * m_neon))
        Z_d = 4 * np.pi * epsilon_0 * k * T_e * eV_K * a * z / (e**2)
        n_i0 = np.add(n_e0, np.multiply(Z_d, n_d))
    else:
        v_tn = np.sqrt(8 * k * T_n * eV_K / (np.pi * m_argon))
        v_ti = np.sqrt(8 * k * T_i / (np.pi * m_argon))
        Z_d = 4 * np.pi * epsilon_0 * k * T_e_argon * eV_K * a * z / (e**2)
        n_i0 = np.add(n_e0_argon, np.multiply(Z_d, n_d))
    
    '''    Charge depletion adjustment    '''
    # New z from Havnes Parameter Physics Reports 421 (2005) 1 – 103#
    def oml_func_p0(x):
        return np.sqrt(m_e/m_neon)*(1+x*tau[i]) - np.sqrt(tau[i]) * np.exp(-x)
    def oml_func(x):
        return np.sqrt(m_e/m_neon)*(1+x*tau[i])*(1+P[i]) - np.sqrt(tau[i]) * np.exp(-x)
    
    '''    Havnes Parameter    '''
    if gas_type == 'Neon':
        P = np.multiply(np.multiply(695*(1.3/2),T_e),np.divide(n_d,n_i0))
        P2 = np.multiply(Z_d/z,np.divide(n_d,n_i0))
        tau = np.divide(T_e,T_i)
    else:
        P = np.multiply(np.multiply(695*(1.3/2),T_e_argon),np.divide(n_d,n_i0))
        P2 = np.multiply(Z_d/z,np.divide(n_d,n_i0))
        tau = np.divide(T_e_argon,T_i)
        
    #
    z_depl = []
    Z_d_0 = Z_d
    #
    if charge_depletion == 1:   
        for i in range(len(p)):
            root_p0 = fsolve(oml_func_p0, 0.4)
            root = fsolve(oml_func, 0.4)
            z_depl = np.append(z_depl,(((100 / root_p0) *root)/100) *z[i])
            if gas_type == 'Neon':
                Z_d[i] = ((4 * np.pi * epsilon_0 * k * T_e[i] * eV_K * a * z_depl[i]) / (e**2))
            else:
                Z_d[i] = ((4 * np.pi * epsilon_0 * k * T_e_argon[i] * eV_K * a * z_depl[i]) / (e**2))
        if gas_type == 'Neon':
            n_i0 = np.add(n_e0, np.multiply(Z_d, n_d)) #m^-3
        else:
            n_i0 = np.add(n_e0_argon, np.multiply(Z_d, n_d)) #m^-3
    
    # Debye lengths
    if gas_type == "Neon":
        debye_De = np.sqrt(epsilon_0 * k * T_e * eV_K / (n_e0 * e**2))
        debye_Di = np.sqrt(epsilon_0 * k * T_i / (n_i0 * e**2))
        debye_D = np.divide(np.multiply(debye_De, debye_Di), np.sqrt(debye_De**2 + debye_Di**2))
    else:
        debye_De = np.sqrt(epsilon_0 * k * T_e_argon * eV_K / (n_e0_argon * e**2))
        debye_Di = np.sqrt(epsilon_0 * k * T_i / (n_i0 * e**2))
        debye_D = np.divide(np.multiply(debye_De, debye_Di), np.sqrt(debye_De**2 + debye_Di**2))
    
    '''    Modified Frost Formular    '''
    if gas_type == "Neon":
        A = 0.0321
        B = 0.012
        C = 1.181
        EN = (np.divide(-E_0, 100)/n_0)*(10**17) #10^17 from Vcm^2 to Td
        M = A * np.abs((1 + np.abs((B * EN)**C))**(-1/(2*C))) * EN
        v_ti2 = np.sqrt(k * T_i / m_neon)
        u_i = M*v_ti2
    else:
        A = 0.0168
        B = 0.007
        C = 1.238
        EN = (np.divide(-E_0_argon, 100)/n_0)*(10**17) #10^17 from Vcm^2 to Td
        M = A * np.abs((1 + np.abs((B * EN)**C))**(-1/(2*C))) * EN
        v_ti2 = np.sqrt(k * T_i / m_argon)
        u_i = M*v_ti2
    
    
    if gas_type == "Neon":
        roh_0 = Z_d * e**2 / (2 * np.pi * epsilon_0 * m_neon * v_ti**2)
        debye_u = np.sqrt(debye_De**2/(1+(2*k*T_e*eV_K/(m_neon*u_i**2)))) + a**2 
    else:
        roh_0 = Z_d * e**2 / (2 * np.pi * epsilon_0 * m_argon * v_ti**2)
        debye_u = np.sqrt(debye_De**2/(1+(2*T_e_argon*eV_K/(m_argon*u_i**2)))) + a**2 
    
    
    if gas_type == "Neon":
        v_boom = np.sqrt(k*T_e*eV_K/m_neon)
        nue = np.sqrt((8*T_n*k*eV_K/(np.pi*m_neon)) + u_i**2 * (1+ ((u_i/v_boom)/(0.6 + 0.05*np.log(atomic_mass_neon) + (debye_De/(5*a))*(np.sqrt(T_i/T_e)-.1)))**3))
        debye_nue = np.sqrt(debye_De**2/(1+(2*k*T_e*eV_K/(m_neon*nue**2)))) + a**2      #?
        roh_0_nue = Z_d * e**2 / (2 * np.pi * epsilon_0 * m_neon * nue**2)
    else:
        v_boom = np.sqrt(k*T_e_argon*eV_K/m_argon)
        nue = np.sqrt((8*T_n*k*eV_K/(np.pi*m_argon)) + u_i**2 * (1+ ((u_i/v_boom)/(0.6 + 0.05*np.log(atomic_mass_argon) + (debye_De/(5*a))*(np.sqrt(T_i/T_e_argon)-.1)))**3))
        debye_nue = np.sqrt(debye_De**2/(1+(2*k*T_e_argon*eV_K/(m_argon*nue**2)))) + a**2 #?
        roh_0_nue = Z_d * e**2 / (2 * np.pi * epsilon_0 * m_argon * nue**2)
    
    '''    Scattering Parameter, Khrapak DOI: 10.1103/PhysRevE.66.046414     '''
    #beta_T = roh_0/debye_Di
    beta_T = roh_0_nue/debye_nue
    beta_T2 = roh_0/debye_D
    #beta_T2 = np.divide(Z_d * e**2, (v_ti**2)*m_argon) / (4 * np.pi * epsilon_0 * debye_D)
    
    coulomb_logarithm = np.log((roh_0_nue + debye_nue)/(roh_0_nue + a))
    x = debye_nue/l_i
    K = x * np.arctan(x) + (np.sqrt(np.pi/2) - 1) * (x**2/(1+x**2)) - np.sqrt(np.pi/2) * np.log(1+x**2)
    
    if display_plots == True:
        #PLOT K and Coulomb Logarythm
        # Plot the results
        fig, ax = plt.subplots(dpi=150)
        plt.plot(p, (K), color='blue', marker='o')
        ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.7)
        plt.tight_layout()
        plt.show()
        fig, ax = plt.subplots(dpi=150)
        plt.plot(p, (coulomb_logarithm), color='red')
        ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.7)
        plt.tight_layout()
        plt.show()
    #
    
    if I == 1.5:
        I_var = "1p5mA"
    else:
        I_var = "1mA"
    
    if polarity == "neg":
        pol_var = "negative"
    else:
        pol_var = "positive"
    
    # Electric and ion drag forces
    if gas_type == "Neon":
        F_e = Z_d * e * E_0
        sigma_scatter = np.pi * a**2 * (1+roh_0/a)
        F_i = n_i0 * m_neon * u_i * v_ti * (sigma_scatter + np.pi * roh_0**2 * (coulomb_logarithm + K))
    else:
        F_e = Z_d * e * E_0_argon
        sigma_scatter = np.pi * a**2 * (1+roh_0/a)
        F_i = n_i0 * m_argon * u_i * v_ti * (sigma_scatter + np.pi * roh_0**2 * (coulomb_logarithm + K))
        
    # Particle velocity
    if gas_type == "Neon":
        v_error = np.array(results[f"{gas_type}"][I_var]["v_mean_error_mm"][pol_var])
        p_error = np.array(results[f"{gas_type}"][I_var]["pressure"])
        factor = np.multiply(epstein, (4 / 3) * np.pi * a**2 * m_neon * v_tn * (p / (T_n * eV_K * k)))
        v_d = (F_e + F_i) / factor
        z_error = abs(factor*e/((4*np.pi*epsilon_0)*k*a*T_e*eV_K*E_0))
        F_i_error = np.zeros(len(p))
        # Calculate F_i_error and z_error using proper indexing
        for i in range(len(p_error)-1):
            position = np.where(p_error[i] == p)[0]  # Get the index of the matching pressure
            if position.size > 0:  # Check if the position array is not empty
                F_i_error[position] = v_error[i] / 1000 * factor[position]  # Update only if the position exists 
                z_error[position] = abs(z_error[position] * (v_error[i] / 1000))
    else:
        v_error = np.array(results[f"{gas_type}"][I_var]["v_mean_error_mm"][pol_var])
        p_error = np.array(results[f"{gas_type}"][I_var]["pressure"])
        factor = np.multiply(epstein, (4 / 3) * np.pi * a**2 * m_argon * v_tn * (p / (T_n * eV_K * k)))
        v_d = (F_e + F_i) / factor
        z_error = abs(factor*e/((4*np.pi*epsilon_0)*k*a*T_e_argon*eV_K*E_0_argon))
        F_i_error = np.zeros(len(p))
        # Calculate F_i_error and z_error using proper indexing
        for i in range(len(p_error)-1):
            position = np.where(p_error[i] == p)[0]  # Get the index of the matching pressure
            if position.size > 0:  # Check if the position array is not empty
                F_i_error[position] = v_error[i] / 1000 * factor[position]  # Update only if the position exists
                z_error[position] = abs(z_error[position] * (v_error[i] / 1000))
    z_error[z_error > 1] = 0
    dZ_d_error = (Z_d/z)*z_error
    beta_T_error = (beta_T / Z_d) * dZ_d_error

    # Define the fitting model: v = c1 * p**(-1) + c2 * p**(-2) + c3 * p**(-3)
    def inverse_power_model(p, c0, c1, c2, c3):
        return c0 + c1 * p**(-1) + c2 * p**(-2) + c3 * p**(-3)
    
    # Perform curve fitting for the velocity-pressure relationship
    try:
        popt, pcov = curve_fit(inverse_power_model, p, (v_d * -1000), absolute_sigma=True) #, sigma=2
        c0, c1, c2, c3 = popt
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        popt, pcov = [0, 0, 0, 0], None
    
    # Generate a smooth line for plotting the fit
    # Generate a smooth line for plotting the fit
    if gas_type == 'Argon':
        pressure_range = np.linspace(12, 120, 500)  # Start from 0.1 to avoid division by zero
    else:
        pressure_range = np.linspace(19, 120, 500)  # Start from 0.1 to avoid division by zero
    fit = inverse_power_model(pressure_range, *popt)
    
    if display_plots == True:
        # Plot the results
        fig, ax = plt.subplots(dpi=150)
        fig.set_size_inches(6, 3)
        plt.title(f"Fit Theory: {gas_type} {I} mA " + polarity, fontsize=10)
        plt.xlabel('Pressure [Pa]', fontsize=9)
        plt.ylabel('$v_{mean}$ [mm/s]', fontsize=9)
        plt.plot(pressure_range, fit, 'g--', label=f'Fit: $c_0 + c_1/p + c_2/p^2 + c_3/p^3$', linewidth=0.8)
        plt.scatter(p, (v_d * -1000), color='blue', marker='o', label='Data', s=10)
        #
        ax.legend(fontsize=8)
        ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.7)
        plt.xlim(10, 130)
        plt.ylim(-10,60)# * .6, np.max(np.abs(v_d * -1000)) * 1.2)
        plt.tight_layout()
        plt.show()
    
        # Display fitted parameters
        print(f"Fitted coefficients: c0={c0:.3f}, c1={c1:.3f}, c2={c2:.3f}, c3={c3:.3f}")
    
    if display_plots == True:
        # Plotting Force 
        fig, ax = plt.subplots(dpi=150)
        fig.set_size_inches(6, 3)
        ax.plot(p, np.abs(F_e) * 10**14, linestyle='solid', marker='^', color='#00429d', linewidth=.75)
        ax.plot(p, F_i * 10**14, linestyle='solid', marker='x', color='#00cc00', linewidth=.75)
        ax.legend(['$F_e x 10^{-14}$', '$F_i x 10^{-14}$'])
        ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
        plt.title(gas_type + " " + str(I) + "mA " + polarity)
        plt.show()
    
    if gas_type == "Neon":
        t_var = (np.pi*a**2*n_i0*m_neon*v_ti**2)
        x = (u_i/v_ti)
        y = (F_i/t_var)
    else:
        t_var =(np.pi*a**2*n_i0*m_argon*v_ti**2)
        x = (u_i/v_ti) 
        y = (F_i/t_var)
    
    if display_plots == True:    
        # Plotting Force/faktor 
        fig, ax = plt.subplots(dpi=300)
        fig.set_size_inches(6, 3)
        ax.plot(x, y, linestyle='solid', marker='x', color='#00cc00', linewidth=.75)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(['$F_i/c*u_i$'])
        plt.xlabel('$u_i / v_{ti}$', fontsize=9)
        plt.ylabel('$F_i / \pi a^2 n_i m_i v_{ti}^2$', fontsize=9)
        ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
        plt.title(gas_type + " " + str(I) + "mA " + polarity + " Hutchinson/Khrapak")
        plt.show()
    
    # Prepare the data to be stored in the JSON file
    if gas_type == "Neon":
        data_to_store = {
            polarity: {
            "p": p.tolist(),
            "F_i": F_i.tolist(),
            "F_i_error": F_i_error.tolist(),
            "factor": factor.tolist(),
            "F_e": F_e.tolist(),
            "v_d_theory": (v_d * -1000).tolist(),
            "v_d_fit": fit.tolist(),
            "p_fit": pressure_range.tolist(),
            "E_0": E_0.tolist(),
            "T_e": T_e.tolist(),
            "n_e": n_e0.tolist(),
            "z": z_depl.tolist(),
            "dz": z_error.tolist(),
            "dZ_d": dZ_d_error.tolist(),
            "beta_T": beta_T.tolist(),
            "beta_T_error": beta_T_error.tolist(),
            "textbook_graph_F_x": x.tolist(),
            "textbook_graph_F_y": y.tolist(),
            "textbook_var": t_var.tolist(),
            "F_e/F_i": (abs(F_e)/F_i).tolist(),
            "F_n/F_e": (abs(factor*v_d)/abs(F_e)).tolist(),
            "F_n/F_i": (abs(factor*v_d)/abs(F_i)).tolist()
            }
        }
    else:
        data_to_store = {
            polarity: {
            "p": p.tolist(),
            "F_i": F_i.tolist(),
            "F_i_error": F_i_error.tolist(),
            "factor": factor.tolist(),
            "F_e": F_e.tolist(),
            "v_d_theory": (v_d * -1000).tolist(),
            "v_d_fit": fit.tolist(),
            "p_fit": pressure_range.tolist(),
            "E_0": E_0_argon.tolist(),
            "T_e": T_e_argon.tolist(),
            "n_e": n_e0_argon.tolist(),
            "z": z_depl.tolist(),
            "dz": z_error.tolist(),
            "dZ_d": dZ_d_error.tolist(),
            "beta_T": beta_T.tolist(),
            "beta_T_error": beta_T_error.tolist(),
            "textbook_graph_F_x": x.tolist(),
            "textbook_graph_F_y": y.tolist(),
            "textbook_var": t_var.tolist(),
            "F_e/F_i": (abs(F_e)/F_i).tolist(),
            "F_n/F_e": (abs(factor*v_d)/abs(F_e)).tolist(),
            "F_n/F_i": (abs(factor*v_d)/abs(F_i)).tolist()
            }
        }
    
    if '.' in selected_current:
        selected_current = selected_current.replace('.', 'p')
    # Create the filename based on gas type and selected current
    filename = f"{gas_type}_{selected_current}_{model}.json"
    filepath = f"json_files/theory/{filename}"
    
    # Check if the file already exists
    if os.path.exists(filepath):
        # Load existing data
        with open(filepath, "r") as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}
    
    # Update the data (replace if the same polarity exists, otherwise append)
    existing_data.update(data_to_store)
    
    # Save the updated data back to the JSON file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure the directory exists
    with open(filepath, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
