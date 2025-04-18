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
from collections import defaultdict

#%% v mean plots

gastype = "Neon" #Argon
current = "1p5mA"  #1p5mA

off_plot = False

json_folder = f"json_files/{gastype}/{current}"
        
# json_folder = "json_files/Argon/"
file_list = [os.path.join(json_folder, img) for img in os.listdir(json_folder) if img.endswith(".json")]

if json_folder == "json_files/Argon/1mA":
    file_list = file_list[12:]

dataset = []
for file in file_list:
    json_file = open(file, 'r')
    json_data = json.load(json_file)
    dataset.append(json_data)

with open(json_folder.split('/')[0] + "/theory/" + json_folder.split('/')[1] + "_" + json_folder.split('/')[2] + "_Khrapak0405_strong.json", "r") as file:
    theory_data_strong = json.load(file)
with open(json_folder.split('/')[0] + "/theory/" + json_folder.split('/')[1] + "_" + json_folder.split('/')[2] + "_Khrapak0405_weak.json", "r") as file:
    theory_data_weak = json.load(file)
with open(json_folder.split('/')[0] + "/theory/" + json_folder.split('/')[1] + "_" + json_folder.split('/')[2] + "_Schwabe2013.json", "r") as file:
    theory_schwabe2013 = json.load(file)

v_mean_pos = np.array(())
pressure_pos = np.array(())
v_error_pos = np.array(())
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

# Results file path in the same directory
result_path = os.path.join("json_files/", "mean_v_results.json")

# Read existing data or initialize empty dictionary if file doesn't exist
try:
    with open(result_path, 'r') as file:
        results = json.load(file)
except FileNotFoundError:
    results = {}

# Define the keys for gas type and current if they don't exist
if gastype not in results:
    results[gastype] = {}

if current not in results[gastype]:
    results[gastype][current] = {}

# Check if results already exist in the file
if 'v_mean_mm' not in results[gastype][current] or 'v_mean_error_mm' not in results[gastype][current]:
    # Update the specific gas type and current setting with new data
    results[gastype][current].update({
        'v_mean_mm': {
            'positive': v_mean_pos.tolist(),
            'negative': v_mean_neg.tolist(),
        },
        'v_mean_error_mm': {
            'positive': v_error_pos.tolist(),
            'negative': v_error_neg.tolist()
        },
        'pressure': pressure_pos.tolist()
    })

    # Write the updated results back to the file
    with open(result_path, 'w') as file:
        json.dump(results, file, indent=4)

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
#
if off_plot == True:
    # Plotting the results + FIT
    plt.figure(dpi=150)
    plt.errorbar(pressure_neg, v_mean_neg, yerr=v_error_neg, fmt='d', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='#004D40', label='Negative Polarity')
    plt.errorbar(pressure_pos, v_mean_pos, yerr=v_error_pos, fmt='o', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='#1E88E5', label='Positive Polarity')
    # Plot the nonlinear fit lines
    plt.plot(pressure_range, fit_neg, '--', color="#004D40", label='Negative Fit', linewidth=.7)
    plt.plot(pressure_range, fit_pos, '--', color="#1E88E5", label='Positive Fit', linewidth=.7)
    # Labels, title, and legend
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('$v_{mean}$ [mm/s]')
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    plt.title(str(json_folder.split('/')[1]) + ' ' + str(json_folder.split('/')[2]) + " Experimental Fit")
    plt.legend(loc='upper right')
    plt.xlim(0, 140)
    plt.ylim(0, 60)
    plt.show()

# Plotting the results + Theory FIT
plt.figure(figsize=(6, 3.5), dpi=400)
if current == "1mA":
    plt.errorbar(pressure_neg, v_mean_neg, yerr=v_error_neg, fmt='s', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='#004D40', label='Exp. Data (-)')
    plt.errorbar(pressure_pos, v_mean_pos, yerr=v_error_pos, fmt='s', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='#1E88E5', label='Exp. Data (+)')
else:
    plt.errorbar(pressure_neg, v_mean_neg, yerr=v_error_neg, fmt='d', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='#004D40', label='Exp. Data (-)')
    plt.errorbar(pressure_pos, v_mean_pos, yerr=v_error_pos, fmt='d', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='#1E88E5', label='Exp. Data (+)')
# Plot the nonlinear fit lines
#plt.plot(theory_data_strong["pos"]["p_fit"], theory_data_strong["pos"]["v_d_fit"], '--', color='#D81B60', label='$F_{id}^{strong}$', linewidth=.7)
#plt.plot(theory_data_strong["neg"]["p_fit"], theory_data_strong["neg"]["v_d_fit"], '-.', color='#D81B60', label='Krapak Model $F_i^{strong}$ (-)', linewidth=.7)
plt.plot(theory_data_weak["neg"]["p_fit"], theory_data_weak["neg"]["v_d_fit"], '--', color='#5DD9C9', label='$F_{id}^{weak,int} \,\,\,\,\,\,$ (-)', linewidth=.7)
plt.plot(theory_data_weak["pos"]["p_fit"], theory_data_weak["pos"]["v_d_fit"], '--', color='grey', label='$F_{id}^{weak,int} \,\,\,\,\,\,$ (+)', linewidth=.7)

#plt.plot(theory_data_weak["neg"]["p_fit"], theory_data_weak["neg"]["v_d_fit"], '-.', color='#5DD9C9', label='Krapak Model $F_i^{weak}$ (-)', linewidth=.7)
plt.plot(theory_schwabe2013["neg"]["p_fit"], theory_schwabe2013["neg"]["v_d_fit"], '-.', color='#FFC107', label='$F_{id}^{hybrid} \,\,\,\,\,\,\,\,\,\,\,\,$ (-)', linewidth=.7)
plt.plot(theory_schwabe2013["pos"]["p_fit"], theory_schwabe2013["pos"]["v_d_fit"], '-.', color='grey', label='$F_{id}^{hybrid} \,\,\,\,\,\,\,\,\,\,\,\,$ (+)', linewidth=.7)
# Labels, title, and legend
plt.xlabel('Pressure [Pa]')
plt.ylabel('$v_{mean}$ [mm/s]')
plt.grid(color='gray', linestyle='-', linewidth=0.2)
#plt.title(str(json_folder.split('/')[1]) + ' ' + str(json_folder.split('/')[2]) + " Theory Fit")
plt.legend(loc='upper right')
plt.xlim(0, 140)
plt.ylim(0, 60)
plt.show()

#%% F_i exp

# =============================================================================
# F_i measured
# =============================================================================

gas_type = "Neon"
current = "1p5mA"

file_paths_fi = [
    "json_files/exp/Argon_1p5mA_exp.json",
    "json_files/exp/Neon_1p5mA_exp.json"
]
file_path2 = [
    "json_files/exp/Argon_1mA_exp.json",
    "json_files/exp/Neon_1mA_exp.json"
]

file_path_theory = [
    "json_files/theory/Argon_1mA_Khrapak0405_weak.json",
    "json_files/theory/Neon_1mA_Khrapak0405_weak.json",
    "json_files/theory/Argon_1mA_Schwabe2013.json",
    "json_files/theory/Neon_1mA_Schwabe2013.json"
]

file_path_theory_1p5mA = [
    "json_files/theory/Argon_1p5mA_Khrapak0405_weak.json",
    "json_files/theory/Neon_1p5mA_Khrapak0405_weak.json",
    "json_files/theory/Argon_1p5mA_Schwabe2013.json",
    "json_files/theory/Neon_1p5mA_Schwabe2013.json"
]

# Corrected logic for current selection
if current == "1p5mA": 
    file_paths_fi = [
        "json_files/exp/Argon_1p5mA_exp.json",
        "json_files/exp/Neon_1p5mA_exp.json"
    ]
    file_path_theory = file_path_theory_1p5mA
else:
    file_paths_fi = file_path2

# Define fmt_list (adjust markers as needed)
fmt_list = ['o', 's', 'D', '^', 'v']  # Example markers

# Define colors for theoretical predictions
color_kinetic = "#FFC107"  # Yellow for F^kin_id
color_weak = "#5DD9C9"  # Cyan for F^weak_id

p = np.array([15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120])

plt.figure(figsize=(6, 3.5), dpi=400)

### EXP DATA PLOT ###
for file_path in file_paths_fi:
    file_name = os.path.basename(file_path)  # Extract just the filename

    if file_name.startswith(gas_type):  # Check gas type
        with open(file_path, 'r') as json_file:
            json_data = json.load(json_file)

        # Extract current value from filename
        current_value = file_name.split('_')[1].replace('p', '.').replace('mA', ' mA')

        # Plot negative ion-drag force (-)
        plt.errorbar(
            json_data["pos"]["P"], 
            np.abs(np.array(json_data["neg"]["F_i"])) * 10**13, 
            yerr=np.array(json_data["neg"]["F_i_error"]) * 10**13, 
            fmt=fmt_list[1], color='#004D40', 
            label='$F_{id}$' + f' (-)  {current_value}', 
            linewidth=0.5, markersize=2.5, capsize=2, ecolor='black'
        )

        # Plot negative ion-drag force (+)
        plt.errorbar(
            json_data["pos"]["P"], 
            np.abs(np.array(json_data["pos"]["F_i"])) * 10**13, 
            yerr=np.array(json_data["pos"]["F_i_error"]) * 10**13, 
            fmt=fmt_list[2], color='#1E88E5', 
            label='$F_{id}$' + f' (+) {current_value}', 
            linewidth=0.5, markersize=2.5, capsize=2, ecolor='black'
        )

### THEORY DATA PLOT ###
for file_path in file_path_theory:
    file_name = os.path.basename(file_path)  # Extract just the filename

    if file_name.startswith(gas_type):  # Check gas type
        with open(file_path, 'r') as json_file:
            json_data = json.load(json_file)

        # Extract current value from filename
        current_value = current.replace('p', '.').replace('mA', ' mA')

        # Determine color based on theory type
        if "Khrapak0405_weak" in file_name:
            theory_label = "$F_{id}^{weak}$"
            theory_color = color_weak
        else:  # Schwabe2013 is assumed to be kinetic
            theory_label = "$F_{id}^{hybrid}$"
            theory_color = color_kinetic

        # Plot theoretical negative ion-drag force
        pol="neg"
        plt.errorbar(
            p, 
            np.abs(np.array(json_data[pol]["F_i"])) * 10**13, 
            yerr=np.array(json_data[pol]["F_i_error"]) * 10**13, 
            fmt='none', 
            label=theory_label, 
            linewidth=0.7, markersize=2.5, capsize=2, ecolor=theory_color
        )
        plt.plot(p, np.abs(np.array(json_data[pol]["F_i"])) * 10**13, 
                 linewidth=0.7, linestyle="--", color=theory_color)

# Labels, title, and legend
plt.xlabel('Pressure [Pa]')
plt.ylabel('$F_{id} \\cdot 10^{-13}$ [N]')
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.legend(loc='upper right')
#plt.title(f"{gas_type}")
plt.xlim(0, 140)
plt.ylim(-0.1, 2.9)
plt.show()

#%% z

# =============================================================================
# z
# =============================================================================

plt.figure(figsize=(4.5, 2.5), dpi=500)

file_paths = [
    "json_files/exp/Argon_1p5mA_exp.json",
    "json_files/exp/Neon_1p5mA_exp.json",
    
    "json_files/exp/Argon_1mA_exp.json",
    "json_files/exp/Neon_1mA_exp.json"
]

polarity = "neg"
color_list = {"Argon": "black", "Neon": "black"}
label_list = {"Argon": "Argon", "Neon": "Neon"}

# Data storage
aggregated_data = {"Argon": defaultdict(list), "Neon": defaultdict(list)}

# Process files and aggregate data
for file in file_paths:
    with open(file, 'r') as json_file:
        json_data = json.load(json_file)
        gas_type = file.split('/')[2].split('_')[0]  # Extract gas type (Argon or Neon)
        P_values = np.array(json_data[polarity]["P"])
        z_values = np.array(json_data[polarity]["z"])
        dz_values = np.array(json_data[polarity]["dz"])
        
        for P, z, dz in zip(P_values, z_values, dz_values):
            aggregated_data[gas_type][P].append((z, dz))

# Compute mean values for duplicate P
processed_data = {"Argon": [], "Neon": []}
for gas in aggregated_data:
    for P in sorted(aggregated_data[gas].keys()):
        z_vals, dz_vals = zip(*aggregated_data[gas][P])
        mean_z = np.mean(z_vals)
        mean_dz = np.mean(dz_vals)
        processed_data[gas].append((P, mean_z, mean_dz))

# Plot data
for gas in processed_data:
    P_vals, z_vals, dz_vals = zip(*processed_data[gas])
    plt.errorbar(P_vals, z_vals, yerr=dz_vals, fmt='x' if gas == "Argon" else '^',
                 color=color_list[gas], label=label_list[gas], linewidth=0.6, markersize=3.5, capsize=2, ecolor='black')

# Theoretical Values
plt.axhspan(.2, .4, color="#004D40", linewidth=.3, alpha=0.1, hatch='/', label="_nolegend_")
plt.axhspan(.3, .5, color='#D81B60', linewidth=.3, alpha=0.1, hatch='\\', label="_nolegend_")

# Labels, title, and legend
plt.xlabel('Pressure [Pa]')
plt.ylabel('$z_{depleted}$')
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.legend(loc='upper right')
plt.xlim(0, 130)
plt.ylim(0.1, .9)
plt.show()

#store
# Convert defaultdict to a normal dict for JSON serialization
processed_data_serializable = {gas: [{"P": P, "z": z, "dz": dz} for P, z, dz in processed_data[gas]] for gas in processed_data}

# Save the data to a JSON file
with open("json_files/exp/charge.json", "w") as f:
    json.dump(processed_data_serializable, f, indent=4)
    
#%% DCP

# =============================================================================
# Discharge Parameters
# =============================================================================

file_paths = [
    "json_files/theory/Neon_1mA_Schwabe2013.json",
    "json_files/theory/Neon_1p5mA_Schwabe2013.json",
    "json_files/theory/Argon_1mA_Schwabe2013.json",
    "json_files/theory/Argon_1p5mA_Schwabe2013.json"
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
keys = ["n_e", "T_e", "E_0"]

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
fig, axes = plt.subplots(3, 1, figsize=(8, 10), dpi=600)

# Titles and y-labels for each subplot
parameters = ["n_e", "T_e", "E_0"]
titles = [r"$n_{e}$ vs Pressure", r"$T_e$ vs Pressure", r"$E_0$ vs Pressure"]
y_labels = [r"$n_{e}$ ($\cdot 10 ^{15} \, \mathrm{m^{-3}}$)", r"$T_e$ (eV)", r"$E_0$ ($\mathrm{V/m}$)"]

y_axis_limits = {
    'n_e': (0, 1.3*10**15),  # Replace with your actual parameter names and limits
    'T_e': (0, 10),
    'E_0': (-600,0)
}

for i, param in enumerate(parameters):
    ax = axes[i]
    
    # Apply y-axis limits if defined for the current parameter
    if param in y_axis_limits:
        ax.set_ylim(y_axis_limits[param])
    
    for dataset_name, dataset in parsed_data_correct.items():
        for polarity, params in dataset.items():
            if polarity == "pos":
                linestyle = "solid"

                # Extract current dataset value and modify label if necessary
                current_value = dataset_name.split('_')[1]
                label_value = "1.5mA" if current_value == "1p5mA" else current_value
                
                # Initialize color and linestyle based on conditions
                if "Argon" in dataset_name and current_value == "1mA":
                    color = "#D81B60"
                elif "Argon" in dataset_name and current_value == "1p5mA":
                    color = "#5DD9C9"
                elif "Neon" in dataset_name and current_value == "1mA":
                    color = "#D81B60"
                    linestyle = '--'
                elif "Neon" in dataset_name and current_value == "1p5mA":
                    color = "#5DD9C9"
                    linestyle = '--'
                else:
                    color = "black"  # Default color
                
                ax.plot(
                    pressures, 
                    params[param], 
                    linestyle=linestyle, 
                    label=f"${param}$ ({dataset_name.split('_')[0]}, {label_value})",
                    color=color
                )

    # Add experimental data
    if param == "T_e":
        ax.errorbar(argon_df["P_Pa"], argon_df["1mA_T"], yerr=2, color="#1E88E5", capsize=3, fmt="^", label="$T_e^{*}$ (Argon, 1mA)", mfc="w")
        ax.errorbar(argon_df["P_Pa"], argon_df["2mA_T"], yerr=2, color="#1E88E5", capsize=3, fmt="x", label="$T_e^{*}$ (Argon, 2mA)")
    elif param == "n_e":
        ax.errorbar(argon_df["P_Pa"], np.array(argon_df["1mA_n"])*(10)**14, yerr=1*(10)**14, color="#1E88E5", capsize=3, fmt="^", label="$n_e^{*}$ (Argon, 1mA)", mfc="w")
        ax.errorbar(argon_df["P_Pa"], np.array(argon_df["2mA_n"])*(10)**14, yerr=1*(10)**14, color="#1E88E5", capsize=3, fmt="x", label="$n_e^{*}$ (Argon, 2mA)")
    elif param == "E_0":
        ax.errorbar(argon_df["P_Pa"], np.array(argon_df["1mA_E"])*(-100), yerr=50, color="#1E88E5", fmt="^", capsize=3, label="$E_0^{*}$ (Argon, 1mA)", mfc="w")
        ax.errorbar(argon_df["P_Pa"], np.array(argon_df["2mA_E"])*(-100), yerr=50, color="#1E88E5", fmt="x", capsize=3, label="$E_0^{*}$ (Argon, 1mA)")

    #ax.set_title(titles[i])
    ax.set_xlabel("Pressure (Pa)")
    ax.set_ylabel(y_labels[i])
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1),fontsize=11)
    ax.grid(True, color="grey", linestyle="--", linewidth=0.2)

plt.tight_layout()
plt.show()
#%% \beta

# =============================================================================
# Scattering parameter \beta
# =============================================================================

plt.figure(figsize=(5.5, 3), dpi=400)

gas_type = "Argon"
polarity = "neg"

file_paths = [
    os.path.join(json_folder.split('/')[0] + "/theory/", filename)
    for filename in os.listdir(json_folder.split('/')[0] + "/theory/")
    if not filename.startswith('._')  # Excludes macOS system files
       and not filename.endswith("strong.json")  # Excludes files ending with 'strong.json'
       and filename.startswith(gas_type)  # Includes only files that start with 'Neon'
]

#error 
for gas in processed_data:
    P_vals, z_vals, dz_vals = zip(*processed_data[gas])
    #P_vals, z_vals, dz_vals

color_list = ["#D81B60", "#5DD9C9", "#D81B60", "#5DD9C9", "#5DD9C9", "#FFC107","#D81B60", "#5DD9C9", "#FFC107", "#D81B60", "#5DD9C9", "#FFC107"]
marker_list = ["s", "^", "d", "o", "s", "^", "d", "o","s", "^", "d", "o", "s", "^", "d", "o"]
i=0

for file in file_paths:
    json_file = open(file, 'r')
    json_data = json.load(json_file)
    if file.split('/')[2].split('_')[0] == gas_type and file.split('/')[2].split('_')[1] == '1p5mA':
        if file.split('/')[2].split('_')[2][:7] == "Khrapak":
            label_text = r"$\beta_T^{weak,int}$ (1.5 mA)"
        else:
            label_text = r"$\beta_T^{hybrid} \,\,\,\,\,\,$ (1.5 mA)"
        plt.errorbar(pressures, np.array(json_data[polarity]["beta_T"]), yerr=np.array(json_data[polarity]["beta_T_error"]), fmt="^", color=color_list[i], linewidth=.7, markersize=4, capsize=2, ecolor='black', mfc='w', label=label_text)
        #plt.errorbar(pressures, np.array(json_data["neg"]["beta_T"]), yerr=np.array(json_data["neg"]["beta_T"])*0.05, fmt=marker_list[i], color=color_list[i], label=r'$\beta_T$ ' + file.split('/')[2].split('_')[0].split('.')[0] + ' pos & neg', linewidth=.7, markersize=4, capsize=2, ecolor='black', mfc='w')
        i+=1
    elif file.split('/')[2].split('_')[0] == gas_type and file.split('/')[2].split('_')[1] == '1mA':
        if file.split('/')[2].split('_')[2][:7] == "Khrapak":
            label_text = r"$\beta_T^{weak,int}$ (1 mA)"
        else:
            label_text = r"$\beta_T^{hybrid} \,\,\,\,\,\,$ (1 mA)"
        plt.errorbar(pressures, np.array(json_data[polarity]["beta_T"]), yerr=np.array(json_data[polarity]["beta_T_error"]), fmt="x", color=color_list[i], linewidth=.7, markersize=4, capsize=2, ecolor='black', label=label_text)#, mfc='w')
        #plt.errorbar(pressures, np.array(json_data["neg"]["beta_T"]), yerr=np.array(json_data["neg"]["beta_T"])*0.05, fmt=marker_list[i], color=color_list[i], label=r'$\beta_T$ ' + file.split('/')[2].split('_')[0].split('.')[0] + ' pos & neg', linewidth=.7, markersize=4, capsize=2, ecolor='black', mfc='w')
        i+=1

# Labels, title, and legend
plt.xlabel('Pressure [Pa]')
plt.ylabel(r'Scattering Parameter $\beta_T$')
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.legend(loc='upper right')
#plt.title(gas_type)
plt.ylim( 0 , 2.8)
plt.show()
#%% log(F_i)

# =============================================================================
# Comparison F_i [log_scale]
# =============================================================================

textbook_fi = open("json_files/textbook_fi.json", 'r')
json_data_textbook_fi = json.load(textbook_fi)

# Plotting scattering parameter \beta_T
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=400)

split = "gas-wt"
gas_type = "Argon"
polarity = "neg"

file_paths_fi = [
    "json_files/exp/Argon_1mA_exp.json",
    "json_files/exp/Argon_1p5mA_exp.json",
    "json_files/exp/Neon_1mA_exp.json",
    "json_files/exp/Neon_1p5mA_exp.json"
]

file_paths = [
    os.path.join(json_folder.split('/')[0] + "/theory/", f)
    for f in os.listdir(json_folder.split('/')[0] + "/theory/")
    if f != '.DS_Store' and f != '._.DS_Store'
        if f.split('_')[0] == gas_type and f.split('_')[-1] != 'strong.json' #andf.split('_')[2] == 'Schwabe2013.json' #and f.split('_')[3] != 'strong.json'
]

if split == "all":
    color_list = ["r", "g", "b", "r", "g", "b", "r", "g", "b", "r", "g", "b"]
    marker_list = ["s", "s", "s", "d", "d", "d", "^", "^", "^", "o", "o", "o"] 
elif split == "theory":
    color_list = ["b", "b", "b", "b"]# "g", "b", "r"]
    marker_list = ["s", "d", "^", "o"]
elif split == "gas":
    color_list = ["#D81B60", "#5DD9C9", "#FFC107", "#D81B60", "#5DD9C9", "#FFC107"]# "g", "b", "r"]
    marker_list = ["s", "s", "s", "d", "d", "d"]
    label_list = ["Hutchinson", "$F_{id}^{strong} \, (1mA)$", "$F_{id}^{weak,int}$", "$F_{id}^{kin}$", "$F_{id}^{strong} \, (1.5mA)$", "$F_{id}^{weak,int}$", "$F_{id}^{kin}$"]
elif split == "gas-wt":
    color_list = ["#5DD9C9", "#FFC107", "#5DD9C9", "#FFC107"]# "g", "b", "r"]
    marker_list = ["s", "s", "d", "d"]
    label_list = ["Hutchinson", "$F_{id}^{weak,int} \, (1mA)$", "$F_{id}^{hybrid}$", "$F_{id}^{weak,int} \, (1.5mA)$", "$F_{id}^{hybrid}$", "Exp. Data (-)", "Exp. Data (+)"]

# MODEL DATA
i = 0
for file in file_paths:
    json_file = open(file, 'r')
    json_data = json.load(json_file)
    y_error = np.array(json_data[polarity]["F_i_error"])/np.array(json_data[polarity]["textbook_var"])
    #ax.plot(json_data[polarity]["textbook_graph_F_x"], json_data[polarity]["textbook_graph_F_y"], color=color_list[i], label=file.split('_')[3].split('.')[0], linewidth=.85, linestyle="--")
    ax.errorbar(json_data[polarity]["textbook_graph_F_x"], json_data[polarity]["textbook_graph_F_y"], yerr=0, color=color_list[i], fmt=marker_list[i], label=label_list[i+1], markersize=4, linewidth=.6, mfc='w', capsize=0)
    i+=1
# EXP DATA
for file in file_paths_fi:
    if file.split('/')[2].split('_')[0] == gas_type and file.split('/')[2].split('_')[1] == "1mA":
        json_file = open(file, 'r')
        json_data = json.load(json_file)
        y = np.array(json_data["pos"]["textbook_y"])
        y_error = np.array(json_data["pos"]["textbook_y_error"])
        ax.errorbar(json_data["pos"]["textbook_x"], y, yerr=y_error, color='black', fmt='o', label=label_list[-1], markersize=3, linewidth=.6, mfc='w', capsize=1)
        y = np.array(json_data["neg"]["textbook_y"])
        y_error = np.array(json_data["neg"]["textbook_y_error"])
        ax.errorbar(json_data["neg"]["textbook_x"], y, yerr=y_error, color='black', fmt='x', label=label_list[-2], markersize=3, linewidth=.6, mfc='w', capsize=1)
    if file.split('/')[2].split('_')[0] == gas_type and file.split('/')[2].split('_')[1] == "1p5mA":  
        json_file = open(file, 'r')
        json_data = json.load(json_file)
        y = np.array(json_data["pos"]["textbook_y"])
        y_error = np.array(json_data["pos"]["textbook_y_error"])
        ax.errorbar(json_data["pos"]["textbook_x"], y, yerr=y_error, color='black', fmt='x', label=label_list[-2], markersize=3, linewidth=.6, mfc='w', capsize=1)
        y = np.array(json_data["neg"]["textbook_y"])
        y_error = np.array(json_data["neg"]["textbook_y_error"])
        ax.errorbar(json_data["neg"]["textbook_x"], y, yerr=y_error, color='black', fmt='o', label=label_list[-1], markersize=3, linewidth=.6, mfc='w', capsize=1)

# HUTCHINSON KHRAPAK   
ax.plot(json_data_textbook_fi["x"], json_data_textbook_fi["y"], color="black", label="Hutchinson", linewidth=.85, linestyle="--")

# Setting log scale for both axes
ax.set_yscale('log')
ax.set_xscale('log')

# Setting axis limits
ax.set_xlim(0.04, 18)
ax.set_ylim(10, 10000)

# Labeling axes
plt.xlabel('$u_i / v_{th,i}$', fontsize=10)
plt.ylabel('$F_{id} / \pi a^2 n_i m_i v_{th,i}^2$', fontsize=10)


# Adding grid, legend, and title
ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)

# Get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Display legend with only the first three entries
plt.legend(handles, label_list, loc='upper right')

#plt.title("Model Comparison " + gas_type)

# Displaying the plot
plt.show()