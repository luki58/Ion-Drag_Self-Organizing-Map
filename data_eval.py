# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:58:37 2024

@author: Luki
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit

#%% v mean plots

json_folder = "json_files/Argon/1mA"
# json_folder = "json_files/Argon/"
file_list = [os.path.join(json_folder, img) for img in os.listdir(json_folder) if img.endswith(".json")]

dataset = []
for file in file_list:
    json_file = open(file, 'r')
    json_data = json.load(json_file)
    dataset.append(json_data)

with open(json_folder.split('/')[0] + "/theory/" + json_folder.split('/')[1] + "_" + json_folder.split('/')[2] + ".json", "r") as file:
    theory_data = json.load(file)

if json_folder.split('/')[1] == 'Argon' or json_folder.split('/')[1] == 'Neon':
    v_mean_pos = np.array(())
    pressure_pos = np.array(())
    v_error_pos = np.array(())
    #
    v_mean_neg = np.array(())
    pressure_neg = np.array(())
    v_error_neg = np.array(())
    
    for data in dataset:
        vel = data['velocity']
        polarity = data['polarity']
        if polarity == 'pos':
            v_mean_pos = np.append(v_mean_pos, np.mean(vel)*1000)
            v_error_pos = np.append(v_error_pos, np.std(np.array(vel))*1000)
            pressure_pos = np.append(pressure_pos, data['pressure'])
        else:
            v_mean_neg = np.append(v_mean_neg, np.mean(vel)*1000)
            v_error_neg = np.append(v_error_neg, np.std(np.array(vel))*1000)
            pressure_neg = np.append(pressure_neg, data['pressure'])

# Define the fitting model: v = c1 * p**(-1) + c2 * p**(-2) + c3 * p**(-3)
def inverse_power_model(p, c0, c1, c2, c3):
    return c1 * p**(-1) + c2 * p**(-2) + c3 * p**(-3)

# Fit model for positive polarity
popt_pos, pcov_pos = curve_fit(inverse_power_model, pressure_pos, v_mean_pos, sigma=v_error_pos, absolute_sigma=True)
c0_pos, c1_pos, c2_pos, c3_pos = popt_pos

# Fit model for negative polarity
popt_neg, pcov_neg = curve_fit(inverse_power_model, pressure_neg, v_mean_neg, sigma=v_error_neg, absolute_sigma=True)
c0_neg, c1_neg, c2_neg, c3_neg = popt_neg

# Generate a smooth line for plotting the fit
if json_folder.split('/')[1] == 'Argon':
    pressure_range = np.linspace(12, 120, 500)  # Start from 0.1 to avoid division by zero
else:
    pressure_range = np.linspace(19, 120, 500)  # Start from 0.1 to avoid division by zero
fit_pos = inverse_power_model(pressure_range, *popt_pos)
fit_neg = inverse_power_model(pressure_range, *popt_neg)

print("Polynomials (POS): c0 = " + str(c0_pos) + ", c1 = " + str(c1_pos) + ", c2 = "+ str(c2_pos) + ", c3 = " + str(c3_pos))
print("Polynomials (NEG): c0 = " + str(c0_neg) + ", c1 = " + str(c1_neg) + ", c2 = "+ str(c2_neg) + ", c3 = " + str(c3_pos))

# Plotting the results + FIT
plt.figure(dpi=500)
plt.errorbar(pressure_neg, v_mean_neg, yerr=v_error_neg, fmt='d', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='g', label='Negative Polarity')
plt.errorbar(pressure_pos, v_mean_pos, yerr=v_error_pos, fmt='o', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='b', label='Positive Polarity')
# Plot the nonlinear fit lines
plt.plot(pressure_range, fit_neg, 'g--', label=f'Negative Fit', linewidth=.7)
plt.plot(pressure_range, fit_pos, 'b--', label=f'Positive Fit', linewidth=.7)
# Labels, title, and legend
plt.xlabel('Pressure [Pa]')
plt.ylabel('$v_{mean}$ [mm/s]')
plt.grid(color='gray', linestyle='-', linewidth=0.2)
plt.title(str(json_folder.split('/')[1]) + ' ' + str(json_folder.split('/')[2]) + " Experimental Fit")
plt.legend(loc='upper right')
plt.xlim(0, 130)
#plt.ylim(0, 40)
plt.show()

# Plotting the results + Theory FIT
plt.figure(dpi=500)
plt.errorbar(pressure_neg, v_mean_neg, yerr=v_error_neg, fmt='d', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='g', label='Negative Polarity')
plt.errorbar(pressure_pos, v_mean_pos, yerr=v_error_pos, fmt='o', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='b', label='Positive Polarity')
# Plot the nonlinear fit lines
plt.plot(theory_data["pos"]["p_fit"], theory_data["pos"]["v_d_fit"], '--', color='cyan', label=f'Theory Model Pos', linewidth=.7)
plt.plot(theory_data["neg"]["p_fit"], theory_data["neg"]["v_d_fit"], '--', color='lime', label=f'Theory Model Neg', linewidth=.7)
# Labels, title, and legend
plt.xlabel('Pressure [Pa]')
plt.ylabel('$v_{mean}$ [mm/s]')
plt.grid(color='gray', linestyle='-', linewidth=0.2)
plt.title(str(json_folder.split('/')[1]) + ' ' + str(json_folder.split('/')[2]) + " Theory Fit")
plt.legend(loc='upper right')
plt.xlim(0, 130)
#plt.ylim(0, 40)
plt.show()

#%%
gas_type = "Neon"
p = np.array([15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120])  # Pa
# Plotting ION-DRAG FORCE
plt.figure(dpi=600)
path = json_folder.split('/')[0] + "/theory/"
for file in os.listdir(path):
    if file.split('_')[0] == gas_type:
        json_file = open(path+file, 'r')
        json_data = json.load(json_file)
        if file.split('_')[1].split('.')[0] == "1mA":
            fmt_list ='d'
        else:
            fmt_list ='s' 
        plt.errorbar(p, np.array(json_data["pos"]["F_i"])*(10**(13)), yerr=np.array(json_data["pos"]["F_i"])*(0.05*10**(13)), fmt=fmt_list, color='red', label='$F_i^{+}$ ' + file.split('_')[1].split('.')[0], linewidth=.7, markersize=2.5, capsize=2, ecolor='black')
        plt.errorbar(p, np.array(json_data["neg"]["F_i"])*(10**(13)), yerr=np.array(json_data["neg"]["F_i"])*(0.05*10**(13)), fmt=fmt_list, color='blue', label='$F_i^{-}$ ' + file.split('_')[1].split('.')[0], linewidth=.7, markersize=2, capsize=2, ecolor='black')
# Labels, title, and legend
plt.xlabel('Pressure [Pa]')
plt.ylabel('$F_i \cdot 10^{-13}$ [N]')
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.legend(loc='upper right')
plt.title(gas_type)
plt.xlim(0, 130)
plt.show()
#%%
# Plotting Charge Number z
plt.figure(figsize=(5, 3), dpi=600)
path = json_folder.split('/')[0] + "/theory/"
color_list = ["b", "r", "g", "cyan"]
i=0
for file in os.listdir(path):
    json_file = open(path+file, 'r')
    json_data = json.load(json_file)
    if file.split('_')[1].split('.')[0] == "1mA":
        plt.errorbar(p, np.array(json_data["pos"]["z"]), yerr=.01, fmt='d', color=color_list[i], label='z ' + file.split('_')[0].split('.')[0], linewidth=.7, markersize=3, capsize=2, mfc='w', ecolor='black')
        i+=1
# Labels, title, and legend
plt.xlabel('Pressure [Pa]')
plt.ylabel('z')
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.legend(loc='upper right')
#plt.title(gas_type)
plt.xlim(0, 130)
plt.show()
#%%
# Load JSON files

file_paths = [
    "json_files/theory/Argon_1mA.json",
    "json_files/theory/Argon_1p5mA.json",
    "json_files/theory/Neon_1mA.json",
    "json_files/theory/Neon_1p5mA.json"
]

data = {}
for path in file_paths:
    with open(path, 'r') as f:
        data[path.split('/')[-1].replace('.json', '')] = json.load(f)

# Example data structure, assuming T_e, n_e, and E are available in files
pressures = np.array([15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120])  # Pa

# Correctly extract T_e, n_e0, and E_0 for plotting
def extract_correct_params(data, keys):
    params = {key: [] for key in keys}
    for key in keys:
        if key in data:
            params[key] = data[key][:len(pressures)]  # Extracting data corresponding to pressures
    return params

# Updated keys for extraction
keys = ["n_e0", "T_e", "E_0"]

# Parse data for each dataset
parsed_data_correct = {}
for dataset_name, content in data.items():
    parsed_data_correct[dataset_name] = {
        "pos": extract_correct_params(content.get("pos", {}), keys),
        "neg": extract_correct_params(content.get("neg", {}), keys)
    }

# Define the experimental data
argon_df = {
    "P_Pa": [10, 20, 40, 60],  # Pressures in Pascal
    "1mA_n": [2.81, 4.43, 4.01, 3.26],  # Electron density (10^8 cm^-3)
    "1mA_T": [4.13, 4.555, 4.795, 5.23],  # Temperature (eV)
    "1mA_E": [1.58, 1.855, 2.395, 4.10],  # Electric field (V/cm)
    "2mA_n": [np.nan, 7.43, 8.14, 6.22],  # Electron density (10^8 cm^-3)
    "2mA_T": [np.nan, 4.94, 5.08, 4.37],  # Temperature (eV)
    "2mA_E": [np.nan, 2.0, 2.53, 4.38],  # Electric field (V/cm)
}

# Plotting T_e, n_e0, and E_0 in separate subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 15), dpi=600)

# Titles and y-labels for each subplot
parameters = ["n_e0", "T_e", "E_0"]
titles = [r"$n_{e0}$ vs Pressure", r"$T_e$ vs Pressure", r"$E_0$ vs Pressure"]
y_labels = [r"$n_{e0}$ ($\mathrm{m^{-3}}$)", r"$T_e$ (eV)", r"$E_0$ ($\mathrm{V/m}$)"]

# Plot each parameter in a separate subplot
for i, param in enumerate(parameters):
    ax = axes[i]
    for dataset_name, dataset in parsed_data_correct.items():
        for polarity, params in dataset.items():
            linestyle = "solid" if polarity == "pos" else "dashed"
            color = "orange" if "Argon" in dataset_name and param == "n_e0" else None  # Highlight Argon n_e0
            ax.plot(
                pressures, 
                params[param], 
                linestyle=linestyle, 
                label=f"{param} ({polarity}, {dataset_name})",
                color=color
            )
    # Add experimental data
    if param == "T_e":
        ax.scatter(argon_df["P_Pa"], argon_df["1mA_T"], color="blue", marker="o", label="1mA Experimental T_e")
        ax.scatter(argon_df["P_Pa"], argon_df["2mA_T"], color="blue", marker="x", label="2mA Experimental T_e")
    elif param == "n_e0":
        ax.scatter(argon_df["P_Pa"], np.array(argon_df["1mA_n"])*(10)**14, color="green", marker="o", label="1mA Experimental n_e0")
        ax.scatter(argon_df["P_Pa"], np.array(argon_df["2mA_n"])*(10)**14, color="green", marker="x", label="2mA Experimental n_e0")
    elif param == "E_0":
        ax.scatter(argon_df["P_Pa"], np.array(argon_df["1mA_E"])*(-100), color="red", marker="o", label="1mA Experimental E_0")
        ax.scatter(argon_df["P_Pa"], np.array(argon_df["2mA_E"])*(-100), color="red", marker="x", label="2mA Experimental E_0")

    ax.set_title(titles[i])
    ax.set_xlabel("Pressure (Pa)")
    ax.set_ylabel(y_labels[i])
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    ax.grid(True)

plt.tight_layout()
plt.show()
#%%
# Plotting scattering parameter \beta_T
plt.figure(figsize=(5, 3), dpi=600)
path = json_folder.split('/')[0] + "/theory/"
color_list = ["b", "r", "g", "cyan"]
i=0
for file in os.listdir(path):
    json_file = open(path+file, 'r')
    json_data = json.load(json_file)
    if file.split('_')[0].split('.')[0] == "Argon":
        plt.errorbar(pressures, np.array(json_data["pos"]["beta_T"]), yerr=np.array(json_data["pos"]["beta_T"])*0.05, fmt='d', color=color_list[i], linewidth=.5, markersize=4, capsize=2, mfc='w', ecolor='black')
        plt.errorbar(pressures, np.array(json_data["neg"]["beta_T"]), yerr=np.array(json_data["neg"]["beta_T"])*0.05, fmt='d', color=color_list[i], label=r'$\beta_T$ ' + file.split('_')[0].split('.')[0] + ' pos & neg', linewidth=.5, markersize=4, capsize=2, mfc='w', ecolor='black')
        i+=1
    else:
        plt.errorbar(pressures, np.array(json_data["pos"]["beta_T"]), yerr=np.array(json_data["pos"]["beta_T"])*0.05, fmt='o', color=color_list[i], linewidth=.7, markersize=3, capsize=2, mfc='w', ecolor='black')
        plt.errorbar(pressures, np.array(json_data["neg"]["beta_T"]), yerr=np.array(json_data["neg"]["beta_T"])*0.05, fmt='o', color=color_list[i], label=r'$\beta_T$ ' + file.split('_')[0].split('.')[0] + ' pos & neg', linewidth=.7, markersize=3, capsize=2, mfc='w', ecolor='black')
        i+=1
# Labels, title, and legend
plt.xlabel('Pressure [Pa]')
plt.ylabel(r'Scattering Parameter $\beta_T$')
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.legend(loc='upper right')
plt.show()