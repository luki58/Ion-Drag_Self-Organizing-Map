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

json_folder = "json_files/Neon/1p5mA"
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
pressure_range = np.linspace(12, 120, 500)  # Start from 0.1 to avoid division by zero
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
gas_type = "Argon"
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
# Plotting Charge Number
plt.figure(dpi=500)
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
plt.title(gas_type)
plt.xlim(0, 130)
plt.show()
    
    