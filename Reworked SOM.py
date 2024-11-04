# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:11:55 2024

@author: Sali
"""



#%%
#####
#Imports and Initialization

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import som_class
import json
import os

framerate1 = 1/50   #VM2
framerate2 = 1/100  #VM1
pixelsize = 14.7e-6

#%% inputs
# SOM params
alpha = 0.01
distance_threshold = 50
startradius = 100
endradius = 0.5
iterations = 55
epsilon = 3.5
# Tracing
min_length = 3

# set to True to save data to json
save = True

# Set directory/files of particle images and background (data folder requires calculated particle positions)
image_folder = 'C://Users/Lukas/Documents/GitHub/Make_BMP/Argon_3mu/VM1_AVI_231007_095631_15Pa_1mA/neg/'
#image_folder = 'VM1_AVI_231006_130201_90Pa_1mA/pos/'

particle_folder = image_folder[:-1] + '_positions/' #create folder for positions


#####
# Tracing particles over multipe images:

som = som_class.SOM(distance_threshold, alpha, startradius, endradius, iterations, epsilon)

available_images = len(os.listdir(particle_folder)) #trace over all images in folder
allmatches = np.array((),dtype=object)
match = np.array((),dtype=object)
original_coords = np.array((),dtype=object)
position_files = [os.path.join(particle_folder, img) for img in os.listdir(particle_folder) if img.endswith(".npy")]
position_files.sort()

images_to_match = available_images
for i in range(min(images_to_match,len(position_files)-1)):
    filename1 = position_files[i]
    filename2 = position_files[i+1]
    coords1, coords2 = som.read_in_coordinates(filename1, filename2)
    coords1, coords2 = som.match_particles(coords1, coords2)
    if i==0:
        match = np.array([coords1,None],dtype=object)
        match[1] = coords2
    else:
        match = np.array([coords2,None],dtype=object)
        match = np.delete(match,1)
    orig_coords = np.array([coords1,None],dtype=object)
    orig_coords = np.delete(orig_coords,1)
    allmatches = np.append(allmatches,match)
    original_coords = np.append(original_coords,orig_coords)
    print(f"finished number {i+1}")
print("now tracing")
allmatches = som.tracing(allmatches,original_coords)
starting_image = int(position_files[0].split('/')[10].split('.')[0].split('_')[1])
dataframe = som.convert_to_dataframe(allmatches,starting_image)
filtered_particles = som.dataframe_min_length_filter(dataframe,min_length)


#%%
#frame with most particle coordinates (for density calc.)

all_coords = np.array((), dtype=object)
particle_number =  np.array(())
images_to_match = available_images

for i in range(min(images_to_match,len(position_files)-1)):
    filename1 = position_files[i]
    filename2 = position_files[i+1]
    coords1, coords2 = som.read_in_coordinates(filename1, filename2)
    number_found = len(coords1)
    particle_number = np.append(particle_number, number_found)
    new_coords = np.array([coords1,None],dtype=object)
    new_coords = np.delete(new_coords, 1)
    all_coords = np.append(all_coords, new_coords)

imax = np.argmax(particle_number)
most_coords = all_coords[imax]
x_coords = most_coords[:,0]
y_coords = most_coords[:,1]

#%%
# velocity calculation 

# Define deviation and corridor parameters
max_deviation_y = 0.014  # For example, particles can deviate by 0.1 = 10% of x direction in y
y_corridor_min = 10  # Lower bound of the corridor in y-axis
y_corridor_max = 100   # Upper bound of the corridor in y-axis


# Parameters for particle selection
percentage = 1.  # Percentage of particles to consider (e.g., 0.5 means 50%) or not if second_half
range_selection = 'first_half'  # Choose between 'first_half' or 'second_half'

# Get the list of all particle IDs
all_particle_ids = filtered_particles['particle_id'].unique().astype(int)

# Determine the number of particles to select based on percentage
n_particles_to_select = int(len(all_particle_ids) * percentage)

# Sort particle IDs to make the selection predictable
sorted_particle_ids = np.sort(all_particle_ids)

# Select particles based on the specified range (first_half or second_half)
if range_selection == 'first_half':
    selected_particle_ids = sorted_particle_ids[:n_particles_to_select]
elif range_selection == 'second_half':
    selected_particle_ids = sorted_particle_ids[n_particles_to_select:]
else:
    raise ValueError("range_selection must be either 'first_half' or 'second_half'.")

eval_df = pd.DataFrame(columns=['avx', 'avy', 'avdxy', 'id', 'frame'])    
frame_calc = 3 #number of frames to calc. v
min_movement = 0.001 #minimum movement being captured
i = 0

if image_folder.split('/')[8][:3] == 'VM1':
    framerate = framerate2
elif image_folder.split('/')[8][:3] == 'VM2':
    framerate = framerate1

for pid in selected_particle_ids:
    pid_df = filtered_particles.loc[filtered_particles['particle_id']==pid]
    maxframes = len(pid_df['frame_number'])
    frame_slices = int((maxframes-maxframes%frame_calc)/frame_calc)
    
    for xf in range(frame_slices):
        slice_df = pid_df.sort_values('frame_number').iloc[(xf*frame_calc):((xf+1)*frame_calc)]
        dx = slice_df['x'].diff()
        dy = slice_df['y'].diff()
        dxy = np.sqrt(dx**2 + dy**2)*pixelsize / (framerate*np.abs(slice_df['frame_number'].diff()))
        
        # Calculate the average X and Y
        avg_x = slice_df['x'].mean()
        avg_y = slice_df['y'].mean()
       
        # Check Y deviation (ensuring it does not exceed allowed deviation)
        x_range = np.abs(dx[1:]).sum()  # Total movement in x direction
        y_deviation_allowed = max_deviation_y * x_range
        y_deviation_actual = np.abs(dy[1:]).sum()
       
        # Check if within the allowed y deviation
        if y_deviation_actual <= y_deviation_allowed:
            # Check if within the corridor limits
            if y_corridor_min <= avg_y <= y_corridor_max:
                if np.mean(dxy[1:]) > min_movement:
                    eval_df.loc[i, 'id'] = pid
                    eval_df.loc[i, 'avx'] = avg_x
                    eval_df.loc[i, 'avy'] = avg_y
                    eval_df.loc[i, 'avdxy'] = np.mean(dxy[1:])
                    eval_df.loc[i, 'frame'] = slice_df['frame_number'].iloc[-1]
                    i = i + 1
                else:
                    print(f"Particle {pid} skipped (not moving).")
            else:
                # Skip particle if it goes out of the y corridor
                print(f"Particle {pid} skipped (outside y corridor).")
        else:
            # Skip particle if it exceeds allowed deviation
            print(f"Particle {pid} skipped (exceeded y deviation).")
        
eval_df = eval_df.astype({'avx':float,'avy':float,'avdxy':float, 'id':int, 'frame':int})


## plots
title = 'alpha=%.4f, iter.=%d, eps.=%d,'%(alpha, iterations, epsilon)

# particle traces
trace_df = filtered_particles
plt.figure(dpi=300)
for ids in trace_df['particle_id']:
    trace_slice = trace_df.loc[filtered_particles['particle_id']==ids]
    trace_slice = trace_slice.sort_values('frame_number')
    plt.plot(trace_slice['x'], trace_slice['y'])
plt.xlim(0,1600)
#plt.ylim(0,600)
plt.title(title)
plt.suptitle(image_folder.split('_')[-2] + ' | '  +image_folder.split('_')[-1])
plt.show()

xyz_df = eval_df.sort_values('avx')
xyz_df = xyz_df.sort_values('frame')

plt.figure(dpi=300)
plt.plot(xyz_df['avx'], xyz_df['avdxy'], '.', markersize=1)
vel_av = np.average(xyz_df['avdxy'])
plt.ylim(0, vel_av*2)
plt.ylabel('Velocity [m/s]')
plt.xlabel('x [px]')
plt.title(title)
plt.suptitle(image_folder.split('_')[-2] + ' | '  +image_folder.split('_')[-1])
plt.show()

plt.figure(dpi=300)
plt.plot(xyz_df['avy'], xyz_df['avdxy'], '.', markersize=1)
vel_av = np.average(xyz_df['avdxy'])
plt.ylim(0, vel_av*2)
plt.ylabel('Velocity [m/s]')
plt.xlabel('y [px]')
plt.title(title)
plt.suptitle(image_folder.split('_')[-2] + ' | '  +image_folder.split('_')[-1])
plt.show()

plt.figure(dpi=300)
scatter = plt.scatter(xyz_df['avx'], xyz_df['avy'],c=xyz_df['avdxy'], s=2, cmap='inferno', vmin=0, vmax=np.mean(xyz_df['avdxy'])*2)
plt.colorbar(scatter)
plt.xlim(0,1600)
#plt.ylim(0,600)
plt.ylabel('y [px]')
plt.xlabel('x [px]')
plt.title(title)
plt.suptitle(image_folder.split('_')[-2] + ' | '  +image_folder.split('_')[-1])
plt.show()

#%%
### save to json
#raw/filtered particles

pressure = int(image_folder.split('_')[-2][-5:-2])

if save == True:
    folder_json = 'json_files'
    json_data = {
                'pressure':pressure,
                'current':image_folder.split('/')[8].split('_')[-1],
                'polarity':image_folder.split('/')[9],
                'alpha':alpha,
                'epsilon':epsilon,
                'iterations':iterations,
                'framerate':framerate,
                'pixelsize':pixelsize,
                'x':eval_df['avx'].tolist(),
                'y':eval_df['avy'].tolist(),
                'velocity':eval_df['avdxy'].tolist(),
                'pid':eval_df['id'].tolist(),
                'x_raw':filtered_particles['x'].tolist(),
                'y_raw':filtered_particles['y'].tolist(),
                'pid_raw':filtered_particles['particle_id'].tolist(),
                'frame_number_raw':filtered_particles['frame_number'].tolist(),
                'frame':eval_df['frame'].tolist(),
                'x_1frame':x_coords.tolist(),
                'y_1frame':y_coords.tolist()
                 }

    save_file = open(folder_json + '/' + image_folder.split('/')[8] + '_' + image_folder.split('/')[9] +'.json','w')

    json.dump(json_data, save_file)
    save_file.close() 

# x: average x position of each particle
# y: average y position of each particle
# raw: "not reduced" data before velocity calculation
# x/y_1frame: coordinates of particles during the frame with most particles
# pid: particle id


