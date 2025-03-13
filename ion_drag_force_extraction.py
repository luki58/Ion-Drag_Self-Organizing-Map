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

m_neon = 20.1797 * u  # neon mass in kg
m_argon = 39.948 * u  # neon mass in kg

# Argon Data Interpolated from Zobnin unpublished Data (measured by Langmur Probe in Pk-4)
# Load the JSON file

file_path = "argon_interpolation/argon_params.json"

with open(file_path, "r") as file:
    data = json.load(file)
    
# Results file path in the same directory
result_path = os.path.join("json_files/", "mean_v_results.json")

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


def function_all(gas_type, I, polarity):
    
    if gas_type == "Argon" and I == 1.5 and polarity == "neg":
        E_multiplier = 1.
        ne_multiplier = .6
        T_e_multiplier = 1.1
    elif gas_type == "Argon" and I == 1.5 and polarity == "pos":
        E_multiplier = .8
        ne_multiplier = .6
        T_e_multiplier = 1.1
    elif gas_type == "Argon" and I == 1 and polarity == "pos":
        E_multiplier = 0.95
        ne_multiplier = .8
        T_e_multiplier = 1.3
    elif gas_type == "Argon" and I == 1 and polarity == "neg":
        E_multiplier = 0.95
        ne_multiplier = .8
        T_e_multiplier = 1.3
    elif gas_type == "Neon" and I == 1 and polarity == "pos":
        E_multiplier = .9
        ne_multiplier = 1
        T_e_multiplier = .85
    elif gas_type == "Neon" and I == 1 and polarity == "neg":
        E_multiplier = 1.2
        ne_multiplier = 1
        T_e_multiplier = 1
    elif gas_type == "Neon" and I == 1.5 and polarity == "pos":
        E_multiplier = 1.1
        ne_multiplier = .7
        T_e_multiplier = .8
    else:
        E_multiplier = 1.1
        ne_multiplier = .7
        T_e_multiplier = 0.8
        
    charge_depletion = 1
    T_e_argon_neon_translation = 0.45
    selected_current = str(I)+"mA"
    #
    if I == 1.5:
        I_var = "1p5mA"
        I_var2 = "1.5mA"
    else:
        I_var = "1mA"
        I_var2 = "1mA"
    
    if polarity == "neg":
        pol_var = "negative"
    else:
        pol_var = "positive"
    
    # Pressure
    p = np.array(results[gas_type][I_var]["pressure"])
    # Ion drag forces extraction
    exp_data = results[gas_type][I_var]["v_mean_mm"][pol_var]
    exp_error = results[gas_type][I_var]["v_mean_error_mm"][pol_var]
    sorted_pairs = sorted(zip(p, exp_data))
    sorted_p, sorted_exp_data = zip(*sorted_pairs)
    sorted_pairs2 = sorted(zip(p, exp_error))
    sorted_p, sorted_exp_error = zip(*sorted_pairs2)
    p = np.array(sorted_p)
    exp_data = np.array(sorted_exp_data)
    exp_error = np.array(sorted_exp_error)
    
    ref_p = [12,    15,    18,   20,   23,   30,   40,   50,  60,  70,  80,  90, 100, 110, 120]
    z_neon = [0.37, 0.37, 0.37, 0.36, 0.36, 0.33, .31, .29, .26, .26, .26, .26, .26, .26, .3]
    #z_neon = [0.3] * len(z_neon)
    # ref_p = [12,   15, 18,   20,  23, 30, 40, 50, 60, 70, 80,  90, 100, 110, 120]
    z_argon = [.66, .59, .56, .52, .47, .4, .4, .4, .4, .4, .4, .37, .36, .34, .32] 
    ref_n_d = np.array([0.06, 0.06, 0.06, 0.07, 0.09, 0.1, 0.1, 0.2, 0.3, 0.4, .5, .6, .7, .8, .8]) * 10**11 

    if gas_type == "Neon":
        z = []
        for i in p:
            z = np.append(z, z_neon[ref_p.index(i)])
    else:
        z = []
        for i in p:
            z = np.append(z, z_argon[ref_p.index(i)])
    
    n_d = []
    for n in p:
        n_d = np.append(n_d, ref_n_d[ref_p.index(n)])
        
    a = (3.4 / 2) * 10**(-6)  # Micrometer particle radius
    epstein = [1.44] * len(p)  # Neutral damping Epstein coefficient, equal in Neon and Argon.
        
    # Extract the data for the selected current
    try:
        E_0_argon, T_e_argon, n_e0_argon = extract_plasma_data(data, selected_current, p)
        E_0_argon = np.multiply(E_0_argon, -100*E_multiplier)  # V/m
        n_e0_argon = n_e0_argon * ne_multiplier
    except ValueError as error_data:
        print(error_data)
    
    # Calculations Neon
    E_0_calc = T_e = n_e0 = []
    for p_var in p:
        E_0_calc = np.append(E_0_calc, e_field(p_var, I))
        T_e = np.append(T_e, T_e_interpolation(p_var, I))
        T_e_argon = T_e * T_e_multiplier * T_e_argon_neon_translation
        n_e0 =  np.append(n_e0,  n_e_interpolation(p_var, I))
    E_0 = np.multiply(E_0_calc, -100*E_multiplier)  # V/m
    n_e0 = np.multiply(n_e0,  ne_multiplier * 10**14)
    #
    T_n = 0.025  # eV
    if gas_type == "Neon":
        l_i = np.divide(T_n  * eV_K * k, p * sigma_neon)
        T_i = (np.multiply(2 / 9 * np.abs(np.multiply(E_0, 1)) * e / k, l_i) + 0.025 * eV_K)
    else:
        l_i = np.divide(T_n  * eV_K * k, p * sigma_argon)
        T_i = (np.multiply(2 / 9 * np.abs(np.multiply(E_0_argon, 1)) * e / k, l_i) + 0.025 * eV_K)
    n_0 = p / (k * T_n * eV_K) * 10**(-6)  # cm^-3
    
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
        F_e = Z_d * e * E_0
        factor = np.multiply(epstein, (4 / 3) * np.pi * a**2 * m_neon * v_tn * (p / (T_n * eV_K * k)))
        F_i = abs(F_e) - abs(exp_data) * factor/1000
        F_i_error = (exp_error)* factor/1000
        graph_value = np.pi * a**2 * n_i0 * m_neon * v_ti**2
        y = F_i/graph_value
        y_error = F_i_error/graph_value
        x = (u_i/v_ti) 
        z_error = abs(factor*e/((4*np.pi*epsilon_0)*k*a*T_e*eV_K*E_0))*(exp_error/1000)
    else:
        F_e = Z_d * e * E_0_argon
        factor = np.multiply(epstein, (4 / 3) * np.pi * a**2 * m_argon * v_tn * (p / (T_n * eV_K * k)))
        F_i = abs(F_e) - abs(exp_data) * factor/1000
        F_i_error = (exp_error)* factor/1000
        graph_value = np.pi * a**2 * n_i0 * m_argon * v_ti**2
        y = F_i/graph_value
        y_error = F_i_error/graph_value
        x = (u_i/v_ti)
        z_error = abs(factor*e/((4*np.pi*epsilon_0)*k*a*T_e_argon*eV_K*E_0_argon))*(exp_error/1000)
    
    # EXP Data save
    # Prepare the data to be stored in the JSON file
    data_to_store = {
        polarity: {
            "P": p.tolist(),
            "F_i": F_i.tolist(),
            "F_i_error": F_i_error.tolist(),
            "textbook_var": graph_value.tolist(),
            "textbook_x": x.tolist(),
            "textbook_y": y.tolist(),
            "textbook_y_error": y_error.tolist(),
            "F_n/F_e": (abs(factor*exp_data/1000)/F_e).tolist(),
            "F_n/F_i": (abs(factor*exp_data/1000)/F_i).tolist(),
            "F_e/F_i": (abs(F_e)/F_i).tolist(),
            "z": z_depl.tolist(),
            "dz": z_error.tolist(),
        }
    }
    
    # Define filename and file path
    filename = f"{gas_type}_{I_var}_exp.json"
    filepath = os.path.join("json_files", "exp", filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Check if the file exists and load existing data if it does
    if os.path.exists(filepath):
        with open(filepath, "r") as json_file:
            try:
                existing_data = json.load(json_file)
            except json.JSONDecodeError:
                existing_data = {}  # Handle corrupted JSON files gracefully
    else:
        existing_data = {}
    
    # Update the data: replace if the same polarity exists, otherwise merge
    existing_data.update(data_to_store)
    
    # Save the updated data back to the JSON file
    with open(filepath, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
    print(f"Saved data for {gas_type}, {I}mA, {polarity}")
    
    if gas_type == "Argon" and polarity == "pos":
        store_plasma_data(ref_p, n_e0_argon, T_e_argon, E_0_argon, I_var2)
        print(f"Saved plasma data for {gas_type}, {I}mA")
    elif gas_type == "Neon" and polarity == "pos":
        store_plasma_data(ref_p, n_e0, T_e, E_0, I_var2)
        print(f"Saved plasma data for {gas_type}, {I}mA")

def store_plasma_data(ref_p, n_e, T_e, E_0, current, file_name = "argon_interpolation/exp_argon_params.json"):

    # Prepare data to store
    current_data = {
        "Pressure (Pa)": list(ref_p),
        "n": list(n_e),
        "T": list(T_e),
        "E": list(E_0)
    }
    
    
    # Check if the file already exists
    if os.path.exists(file_name):
        # Load existing data
        with open(file_name, "r") as file:
            data = json.load(file)
    else:
        # Initialize a new data structure
        data = {}
        
    # Add the new data for the specified current
    data[current] = current_data

    # Save the updated data to the JSON file
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Data for {current} successfully stored in {file_name}.")

# Possible values for each variable
gas_types = ["Argon", "Neon"]
currents = [1, 1.5]
polarities = ["pos", "neg"]
# Iterate over all combinations
for gas_type in gas_types:
    for I in currents:
        for polarity in polarities:
            function_all(gas_type, I, polarity)