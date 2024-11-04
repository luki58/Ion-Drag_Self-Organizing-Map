# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:58:37 2024

@author: Sali
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json



#%% v mean plots

json_folder = "json_files/Argon/1p5mA"
# json_folder = "json_files/Argon/"
file_list = [os.path.join(json_folder, img) for img in os.listdir(json_folder) if img.endswith(".json")]

dataset = []
for file in file_list:
    json_file = open(file, 'r')
    json_data = json.load(json_file)
    dataset.append(json_data)


if json_folder.split('/')[2] == '1p5mA':
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
else:
    v_mean = np.array(())
    pressure = np.array(())
    v_error = np.array(())
    
    for data in dataset:
        vel = data['velocity']
        v_mean = np.append(v_mean, np.mean(vel))
        v_error = np.append(v_error, np.std(np.array(vel)))
        pressure = np.append(pressure, data['pressure'])
    
    v_mean_pos = v_mean[1::2]*1000
    v_error_pos = v_error[1::2]*1000
    pressure_pos = pressure[1::2]
    
    v_mean_neg = v_mean[0::2]*1000
    v_error_neg = v_error[0::2]*1000
    pressure_neg = pressure[0::2]

colors = ['g','b']
plt.figure(dpi=500)
plt.errorbar(pressure_neg, v_mean_neg, v_error_neg, fmt='d', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='g', label='negative polarity')
plt.errorbar(pressure_pos, v_mean_pos, v_error_pos, fmt='o', color='r', linewidth=1, markersize=3, capsize=2, mfc='w', ecolor='b', label='positive polarity')
plt.xlabel('Pressure [Pa]')
plt.ylabel('$v_{mean}$ [mm/s]')
plt.grid(color='gray', linestyle='-', linewidth=.2)
plt.title(str(json_folder.split('/')[1]) + ' ' + str(json_folder.split('/')[2]))
plt.legend(loc='upper right')
plt.xlim(0,130)
plt.show()

#%% alphas

#Neon
pressure = np.array([120, 100, 90, 70, 60, 50, 40])
alpha_neg = np.array([0.0133, 0.0156, 0.0283, 0.029, 0.032, 0.0102, 0.0059])
alpha_pos = np.array([0.0031, 0.0079, 0.0179, 0.0245, 0.0251, 0.0044, 0.001]) 

#Argon
# pressure = np.array([120,110,100,90,80,70,60,50,40,30,23,15])
# alpha_neg = np.array([0.0022, 0.0108, 0.001, 0.0036, 0.006, 0.0038, 0.0041, 0.0045, 0.0104, 0.0191, 0.0066, 0.0132])
# alpha_pos = np.array([0.0011, 0.0061, 0.0018, 0.001, 0.001, 0.0047, 0.0035, 0, 0.0097, 0.0136, 0.0067, 0.0083])


plt.figure(dpi=500)

plt.plot(pressure, alpha_neg, 'x', c='blue')
plt.plot(pressure, alpha_neg, '--', c='blue', linewidth=0.5, label='negative polarity')

plt.plot(pressure, alpha_pos, 'x', c='red')
plt.plot(pressure, alpha_pos, '--', c='red', linewidth=0.5, label='positive polarity')

plt.ylabel('alpha')
plt.xlabel('pressure [Pa]')
plt.title(str(json_folder.split('/')[1]) + ' ' + str(json_folder.split('/')[2]))
plt.legend(loc='upper right')
plt.show()



