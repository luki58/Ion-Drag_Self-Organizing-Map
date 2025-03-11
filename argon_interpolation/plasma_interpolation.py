# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:46:47 2024

@author: Lukas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import os

def store_fitted_data(fit_results, current, pressure_fit_range, fit = "Logarithmic" , file_name="argon_params.json"):
    """Parameters:
    - fit_results: Dictionary containing the fitted data for n, T, E.
    - current: Discharge current (e.g., "1mA", "2mA").
    - pressure_fit_range: Pressure range used for evaluation.
    - file_name: Name of the JSON file to store the data.
    """
    # Prepare data to store
    current_data = {
        "Pressure (Pa)": list(pressure_fit_range),
        "n": list(fit_results["Linear"]["n_fit"]* (10)**14),
        "T": list(fit_results[fit]["T_fit"]),
        "E": list(fit_results[fit]["E_fit"])
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

def plot_stored_data(data_points_topolot_T, data_points_topolot_n, file_name="argon_params.json"):
    p_data = [10, 20, 40, 60]
    # Load data from the JSON file
    try:
        with open(file_name, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return
    
    # Initialize subplots
    plt.figure(figsize=(10, 7), dpi=300)

    # Colors for plotting
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))

    # Subplot 1: Electron Density (n)
    plt.subplot(3, 1, 1)
    for i, (current, current_data) in enumerate(data.items()):
        pressure = np.array(current_data["Pressure (Pa)"])
        n_e = np.array(current_data["n"])
        plt.plot(pressure, n_e * (10)**(-14), label=f"{current}", color=colors[i])
    plt.scatter(p_data, data_points_topolot_n[0], color='b', marker='o', label='Data 1mA', s=20)
    plt.scatter(p_data[1:], data_points_topolot_n[1], color='r', marker='x', label='Data 2mA', s=20)
    plt.title("Electron Density ($n_e$)")
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("$n_e [m^{-3}]$ ")
    plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)

    # Subplot 2: Electron Temperature (T)
    plt.subplot(3, 1, 2)
    for i, (current, current_data) in enumerate(data.items()):
        pressure = np.array(current_data["Pressure (Pa)"])
        T_e = np.array(current_data["T"])
        plt.plot(pressure, T_e, label=f"{current}", color=colors[i])
    plt.scatter(p_data, data_points_topolot_T[0], color='b', marker='o', label='Data 1mA', s=20)
    plt.scatter(p_data[1:], data_points_topolot_T[1], color='r', marker='x', label='Data 2mA', s=20)
    plt.title("Electron Temperature ($T_e$)")
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("$T_e$ (eV)")
    plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)

    # Subplot 3: Electric Field (E)
    plt.subplot(3, 1, 3)
    for i, (current, current_data) in enumerate(data.items()):
        pressure = np.array(current_data["Pressure (Pa)"])
        E = np.array(current_data["E"])
        plt.plot(pressure, E, label=f"{current}", color=colors[i])
    #plt.scatter(p_data, data_points_topolot[0], color='b', marker='o', label='Data 1mA', s=20)
    #plt.scatter(p_data[1:], data_points_topolot[1], color='r', marker='x', label='Data 2mA', s=20)
    plt.title("Electric Field (E)")
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("E (V/cm)")
    plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()


def print_fitted_models(fit_results):
    """Print the equations for the fitted models."""
    print("\nFitted Model Equations:\n")

    for fit_name, result in fit_results.items():
        if fit_name == "Power Law":
            a, b = result["E_fit_params"]
            print(f"Power Law Model for E: E(P) = {a:.4f} * P^{b:.4f}")

# Define possible fitting models for the data

def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def cubic_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

def power_law_model(x, a, b):
    return a * x**b

def logarithmic_model(x, a, b):
    return a * np.log(x) + b

argon_df = {
    "P_Pa": [10, 20, 40, 60],  # Pressures in Pascal
    "1mA_n": [2.81 , 2.93, 4.01, 4.46],  # Electron density (10^8 cm^-3) 10Pa = 2.81, 20PA = 2.43 original, 40Pa = 4.26 [#CORRECTED]
    "1mA_T": [4.23, 4.41, 4.62, 4.65],  # MODEL DATA
    "1mA_E": [1.98, 2.265, 3.05, 3.310],  # Electric field (V/cm) #CORRECTED
    "2mA_n": [5.43, 8.14, 7.82],  # Electron density (10^8 cm^-3) 20 - 40 Pa #CORRECTED
    "2mA_T": [4.3, 4.58, 4.6],  # MODEL DATA
    "2mA_E": [3.55, 3.45, 4.15],  # MODEL DATA
}

data_points_topolot_T = [[4.13, 4.555, 4.795, 5.23],[4.94, 5.08, 4.37]] #"1mA_T" & "2mA_T" # Electron temperature (eV)

data_points_topolot_n = [[2.81 , 4.43, 4.01, 3.26],[7.43, 8.14, 6.22]]

data_points_topolot = [[1.58, 1.855, 2.395, 4.10],[2.0, 2.53, 4.38]] #Electric field (V/cm)

# Fit models to the data and evaluate fits for 1mA

current = "1mA"

if current == "1mA":
    x_data = np.array(argon_df["P_Pa"])
    x_data_n = np.array(argon_df["P_Pa"][1:])
    y_data_n = np.array(argon_df["1mA_n"][1:])
    y_data_T = np.array(argon_df["1mA_T"])
    y_data_E = np.array(argon_df["1mA_E"]) * 0.965
elif current == "2mA":
    x_data_n = np.array(argon_df["P_Pa"][1:])
    y_data_n = np.array(argon_df["2mA_n"])
    y_data_T = np.array(argon_df["2mA_T"])
    y_data_E = np.array(argon_df["2mA_E"]) * 1.18
elif current == "1p5mA":
    x_data_n = np.array(argon_df["P_Pa"][1:])
    y_data_n = 0.5 * (np.array(argon_df["1mA_n"][1:]) + np.array(argon_df["2mA_n"]))  # Interpolated n np.array(argon_df["P_Pa"][1:])
    y_data_T = 0.5 * (np.array(argon_df["1mA_T"][1:]) + np.array(argon_df["2mA_T"]))
    y_data_E = 0.5 * (np.array(argon_df["1mA_E"][1:]) + np.array(argon_df["2mA_E"])) * 1.15
else:
    print("Wrong Input")

# Fit models

fits = {
    "Linear": linear_model,
    "Quadratic": quadratic_model,
    "Power Law": power_law_model,
    "Logarithmic": logarithmic_model,
    #"Cubic": cubic_model,
    #"Exponential": exponential_model,
}

fit_results = {}
pressure_fit_range = np.linspace(10, 120, 23)  # Pressure range for evaluation

for fit_name, model in fits.items():
    try:
        if current == "1mA":
            # Fit models for each parameter
            n_params, _ = curve_fit(model, x_data_n, y_data_n)
            T_params, _ = curve_fit(model, x_data, y_data_T)
            E_params, _ = curve_fit(model, x_data, y_data_E)
        else:
            n_params, _ = curve_fit(model, x_data_n, y_data_n)
            T_params, _ = curve_fit(model, x_data_n, y_data_T)
            E_params, _ = curve_fit(model, x_data_n, y_data_E)

        # Evaluate the models over the pressure range
        fit_results[fit_name] = {
            "n_fit": model(pressure_fit_range, *n_params),
            "T_fit": model(pressure_fit_range, *T_params),
            "E_fit": model(pressure_fit_range, *E_params),
            "E_fit_params": E_params 
        }
    except Exception as e:
        print(f"Failed to fit {fit_name} model: {e}")



if 'p' in current:
    current = current.replace('p', '.')

# Visualize the fits
plt.figure(figsize=(8, 6), dpi=150)

# Electron density (n_e)
plt.subplot(3, 1, 1)
for fit_name, result in fit_results.items():
    plt.plot(pressure_fit_range, result["n_fit"], label=f"{fit_name} Fit")
plt.scatter(x_data_n, y_data_n, label="Measured Data", color='black')
plt.title(f"Electron Density (n_e) Fits - {current}")
plt.xlabel("Pressure (Pa)")
plt.ylabel("n_e (10^8 cm^-3)")
plt.legend()
plt.grid()

# Electron temperature (T_e)
if not current == "1mA":
    x_data = x_data_n
    
plt.subplot(3, 1, 2)
for fit_name, result in fit_results.items():
    plt.plot(pressure_fit_range, result["T_fit"], label=f"{fit_name} Fit")
plt.scatter(x_data, y_data_T, label="Measured Data", color='black')
plt.title(f"Electron Temperature (T_e) Fits - {current}")
plt.xlabel("Pressure (Pa)")
plt.ylabel("T_e (eV)")
plt.legend()
plt.grid()

# Electric field (E)
plt.subplot(3, 1, 3)
for fit_name, result in fit_results.items():
    plt.plot(pressure_fit_range, result["E_fit"], label=f"{fit_name} Fit")
plt.scatter(x_data, y_data_E, label="Measured Data", color='black')
plt.title(f"Electric Field (E) Fits - {current}")
plt.xlabel("Pressure (Pa)")
plt.ylabel("E (V/cm)")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()

#

print_fitted_models(fit_results)

store_fitted_data(fit_results, current, pressure_fit_range, file_name="argon_params.json")
plot_stored_data(data_points_topolot_T, data_points_topolot_n, "argon_params.json")

#end