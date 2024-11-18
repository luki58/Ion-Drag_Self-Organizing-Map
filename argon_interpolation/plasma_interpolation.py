# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:46:47 2024

@author: Lukas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

    "1mA_n": [2.81 , 4.43, 4.01, 3.26],  # Electron density (10^8 cm^-3) 10Pa = 2.81

    "1mA_T": [4.13, 4.555, 4.795, 5.23],  # Electron temperature (eV)

    "1mA_E": [1.58, 1.855, 2.395, 4.10],  # Electric field (V/cm)

    "2mA_n": [7.43, 8.14, 6.22],  # Electron density (10^8 cm^-3)

    "2mA_T": [4.94, 5.08, 4.37],  # Electron temperature (eV)

    "2mA_E": [2.0, 2.53, 4.38],  # Electric field (V/cm)

}



# Fit models to the data and evaluate fits for 1mA

x_data = np.array(argon_df["P_Pa"])
x_data_n = np.array(argon_df["P_Pa"][1:])
#y_data_n = np.array(argon_df["1mA_n"][1:])
#y_data_T = np.array(argon_df["1mA_T"])
#y_data_E = np.array(argon_df["1mA_E"])

y_data_n = np.array(argon_df["2mA_n"])
y_data_T = np.array(argon_df["2mA_T"])
y_data_E = np.array(argon_df["2mA_E"])

# Fit models

fits = {
    "Linear": linear_model,
    #"Quadratic": quadratic_model,
    "Power Law": power_law_model,
    "Logarithmic": logarithmic_model,
    #"Cubic": cubic_model,
    #"Exponential": exponential_model,
}



fit_results = {}
pressure_fit_range = np.linspace(10, 100, 50)  # Pressure range for evaluation


for fit_name, model in fits.items():
    try:
        # Fit models for each parameter
        n_params, _ = curve_fit(model, x_data_n, y_data_n)
        T_params, _ = curve_fit(model, x_data_n, y_data_T)
        E_params, _ = curve_fit(model, x_data_n, y_data_E)

        # Evaluate the models over the pressure range
        fit_results[fit_name] = {
            "n_fit": model(pressure_fit_range, *n_params),
            "T_fit": model(pressure_fit_range, *T_params),
            "E_fit": model(pressure_fit_range, *E_params),
        }

    except Exception as e:
        print(f"Failed to fit {fit_name} model: {e}")



# Visualize the fits

plt.figure(figsize=(16, 12), dpi=400)

# Electron density (n_e)

plt.subplot(3, 1, 1)

for fit_name, result in fit_results.items():
    plt.plot(pressure_fit_range, result["n_fit"], label=f"{fit_name} Fit")
plt.scatter(x_data_n, y_data_n, label="Measured Data", color='black')
plt.title("Electron Density (n_e) Fits")
plt.xlabel("Pressure (Pa)")
plt.ylabel("n_e (10^8 cm^-3)")
plt.legend()
plt.grid()



# Electron temperature (T_e)

plt.subplot(3, 1, 2)

for fit_name, result in fit_results.items():
    plt.plot(pressure_fit_range, result["T_fit"], label=f"{fit_name} Fit")
plt.scatter(x_data_n, y_data_T, label="Measured Data", color='black')
plt.title("Electron Temperature (T_e) Fits")
plt.xlabel("Pressure (Pa)")
plt.ylabel("T_e (eV)")
plt.legend()
plt.grid()



# Electric field (E)

plt.subplot(3, 1, 3)

for fit_name, result in fit_results.items():
    plt.plot(pressure_fit_range, result["E_fit"], label=f"{fit_name} Fit")
plt.scatter(x_data_n, y_data_E, label="Measured Data", color='black')
plt.title("Electric Field (E) Fits")
plt.xlabel("Pressure (Pa)")
plt.ylabel("E (V/cm)")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()