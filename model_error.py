# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 14:50:13 2025

@author: Lukas
"""
import json
import os
import numpy as np

def load_data(gastype, current):
    """
    Loads experimental data and theoretical models for a given gas type and current.
    
    Parameters:
    - gastype (str): The gas type (e.g., "Neon" or "Argon").
    - current (str): The current setting (e.g., "1mA" or "1p5mA").
    
    Returns:
    - v_mean_pos (np.array): Mean velocity (positive polarity)
    - v_error_pos (np.array): Velocity standard deviation (positive polarity)
    - pressure_pos (np.array): Pressure values (positive polarity)
    - v_mean_neg (np.array): Mean velocity (negative polarity)
    - v_error_neg (np.array): Velocity standard deviation (negative polarity)
    - pressure_neg (np.array): Pressure values (negative polarity)
    - theory_data_weak (dict): Theoretical weak model data
    - theory_schwabe2013 (dict): Theoretical Schwabe model data
    """
    if current == 1.0:
        current = "1mA"
    else:
        current = "1p5mA"

    json_folder = f"json_files/{gastype}/{current}"
    
    # Load experimental data files
    file_list = [os.path.join(json_folder, img) for img in os.listdir(json_folder) if img.endswith(".json")]
    
    if json_folder == "json_files/Argon/1mA":
        file_list = file_list[12:]
    
    dataset = []
    for file in file_list:
        json_file = open(file, 'r')
        json_data = json.load(json_file)
        dataset.append(json_data)

    with open(json_folder.split('/')[0] + "/theory/" + json_folder.split('/')[1] + "_" + json_folder.split('/')[2] + "_Khrapak0405_weak.json", "r") as file:
        theory_data_weak = json.load(file)
    with open(json_folder.split('/')[0] + "/theory/" + json_folder.split('/')[1] + "_" + json_folder.split('/')[2] + "_Schwabe2013.json", "r") as file:
        theory_schwabe2013 = json.load(file)

    # Initialize arrays
    v_mean_pos, pressure_pos, v_error_pos = np.array([]), np.array([]), np.array([])
    v_mean_neg, pressure_neg, v_error_neg = np.array([]), np.array([]), np.array([])

    # Process dataset
    for data in dataset:
        vel = np.array(data['velocity'])
        polarity = data['polarity']
        mean_velocity = np.mean(vel) * 1000  # Convert to mm/s
        velocity_std = np.std(vel) * 1000

        if polarity == 'pos':
            v_mean_pos = np.append(v_mean_pos, mean_velocity)
            v_error_pos = np.append(v_error_pos, velocity_std)
            pressure_pos = np.append(pressure_pos, data['pressure'])
        else:
            v_mean_neg = np.append(v_mean_neg, mean_velocity)
            v_error_neg = np.append(v_error_neg, velocity_std)
            pressure_neg = np.append(pressure_neg, data['pressure'])

    return (v_mean_pos, v_error_pos, pressure_pos, v_mean_neg, v_error_neg, pressure_neg, theory_data_weak, theory_schwabe2013)

def load_exp_froces(gastype, current):
    if current == 1.0:
        current = "1mA"
    else:
        current = "1p5mA"
        
    json_data = os.path.join("json_files", "exp", f"{gastype}_{current}_exp.json")
    
    with open(json_data, "r") as file:
        Fi_data_exp = json.load(file)

    # Initialize arrays
    Fi_exp_pos = np.array(Fi_data_exp["pos"]["F_i"])
    pressure_pos = np.array(Fi_data_exp["pos"]["P"])
    Fi_exp_error_pos = np.array(Fi_data_exp["pos"]["F_i_error"])
    
    Fi_exp_neg = np.array(Fi_data_exp["neg"]["F_i"])
    pressure_neg = np.array(Fi_data_exp["neg"]["P"])
    Fi_exp_error_neg = np.array(Fi_data_exp["neg"]["F_i_error"])

    return (Fi_exp_pos, Fi_exp_error_pos, pressure_pos, Fi_exp_neg, Fi_exp_error_neg, pressure_neg)

def compute_model_error(gastype, current, model):
    """
    Computes the mean squared error (MSE) between experimental velocity data 
    and a theoretical model for a given gas type and current.

    Parameters:
    - gastype (str): The gas type (e.g., "Neon" or "Argon").
    - current (str): The current setting (e.g., "1mA" or "1p5mA").
    - model (str): Theoretical model to compare ("weak" or "schwabe").

    Returns:
    - dict: Contains the error values for both positive and negative polarities.
    """

    # Load data
    (v_mean_pos, v_error_pos, pressure_pos, 
     v_mean_neg, v_error_neg, pressure_neg, 
     theory_data_weak, theory_schwabe2013) = load_data(gastype, current)
    #
    (Fi_exp_pos, Fi_exp_error_pos, p_fi_pos, 
     Fi_exp_neg, Fi_exp_error_neg, p_fi_neg) = load_exp_froces(gastype, current)

    # Select model
    theory_data = theory_data_weak if model == "Khrapak" else theory_schwabe2013

    def mse(v_mean, pressure, theory):
        if len(v_mean) == 0 or len(pressure) == 0:
            return np.nan  # Handle empty cases
        v_theory = np.interp(pressure, theory["p_fit"], theory["v_d_fit"])
        return np.mean((v_mean - v_theory) ** 2)
    
    def mse_fi(Fi_exp, p_fi, theory):
        if len(Fi_exp) == 0 or len(p_fi) == 0 or len(theory["p"]) == 0:
            return np.nan  # Handle empty cases
    
        # Convert to NumPy arrays to avoid structure mismatches
        p_fi = np.asarray(p_fi)
        Fi_exp = np.asarray(Fi_exp)
    
        theory_p = np.asarray(theory["p"])
        theory_F_i = np.asarray(theory["F_i"])
    
        # Ensure p_fi and Fi_exp have the same shape before applying indexing
        if p_fi.shape != Fi_exp.shape:
            min_length = min(len(p_fi), len(Fi_exp))
            p_fi = p_fi[:min_length]
            Fi_exp = Fi_exp[:min_length]
    
        # Mask: Keep only pressures that exist within the theoretical range
        valid_idx = (p_fi >= min(theory_p)) & (p_fi <= max(theory_p))
    
        # Ensure valid_idx shape matches Fi_exp before applying
        if len(valid_idx) != len(Fi_exp):
            return np.nan  # Avoid applying incorrect boolean masks
    
        # Apply the filter
        p_fi_filtered = p_fi[valid_idx]
        Fi_exp_filtered = Fi_exp[valid_idx]
    
        if len(p_fi_filtered) == 0:  # If no valid data remains, return NaN
            return np.nan
    
        # Interpolate theoretical force values at these specific experimental pressures
        Fi_theory_filtered = np.interp(p_fi_filtered, theory_p, theory_F_i)
    
        # Compute mean squared error
        return np.mean((abs(Fi_exp_filtered) - abs(Fi_theory_filtered)) ** 2)

    # Compute error
    error_neg = mse(v_mean_neg, pressure_neg, theory_data["neg"])
    error_pos = mse(v_mean_pos, pressure_pos, theory_data["pos"])
    error_fi_neg = mse_fi(Fi_exp_neg, pressure_neg, theory_data["neg"])
    error_fi_pos = mse_fi(Fi_exp_pos, pressure_pos, theory_data["pos"])
    
    #print (error_neg + error_fi_neg, error_pos + error_fi_pos)

    return {
        "error_neg": error_neg + error_fi_neg,
        "error_pos": error_pos + error_fi_pos
    }
