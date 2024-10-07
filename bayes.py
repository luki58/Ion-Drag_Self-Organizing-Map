# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:52:57 2024

@author: Sali
"""

#%%
#####
#Imports and Initialization

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import som_class
import bayes_opt
import os

framerate1 = 1/50   #VM2
framerate2 = 1/100  #VM1
pixelsize = 14.7e-6

#%% inputs
# SOM parameters
#alpha      = 0.04
distance_threshold = 20
startradius = 100
endradius   = 0.5
iterations  = 40
epsilon     = 4

#bayes opt. inputs
opt_init_points = 5
opt_iterations = 50
pbounds = {'param_alpha':(0.001,0.2)} #bounds of parameter variation

# Set directory/files of particle images and background (data folder requires calculated particle positions)
image_folder = 'VM1_AVI_231006_130519_80Pa_1mA/neg/'
#image_folder = 'VM2_AVI_231005_113723_120pa_1mA/pos/'

particle_folder = image_folder[:-1] + '_positions/' #folder for positions

#%%
#####
# Tracing particles over multipe images:

def som_output(alpha):    

    som = som_class.SOM(distance_threshold, alpha, startradius, endradius, iterations, epsilon)
    available_images = len(os.listdir(particle_folder)) #trace over all images in folder
    allmatches = np.array((),dtype=object)
    original_coords = np.array((),dtype=object)
    position_files = [os.path.join(particle_folder, img) for img in os.listdir(particle_folder) if img.endswith(".npy")]
    position_files.sort()
    min_length = 5
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

    allmatches = som.tracing(allmatches,original_coords)
    starting_image = int(position_files[0].split('/')[2].split('.')[0].split('_')[1])
    dataframe = som.convert_to_dataframe(allmatches,starting_image)
    filtered_particles = som.dataframe_min_length_filter(dataframe,min_length)
    
    # calculation velocity and positions
    particle_ids = filtered_particles['particle_id'].unique().astype(int)
    eval_df = pd.DataFrame(columns=['avx', 'avy', 'avdxy', 'id', 'frame'])
    if image_folder.split('_')[0] == 'VM1':
        framerate = framerate2
    elif image_folder.split('_')[0] == 'VM2':
        framerate = framerate1
    frame_calc = 3 #number of frames used to calc. velocity
    i = 0
    for pid in particle_ids:
        
        pid_df = filtered_particles.loc[filtered_particles['particle_id']==pid]
        maxframes = len(pid_df['frame_number'])
        frame_slices = int((maxframes-maxframes%frame_calc)/frame_calc)
        
        for xf in range(frame_slices):
            slice_df = pid_df.sort_values('frame_number').iloc[(xf*frame_calc):((xf+1)*frame_calc)]
            dx = slice_df['x'].diff()
            dy = slice_df['y'].diff()
            dxy = np.sqrt(dx**2 + dy**2)*pixelsize / (framerate*np.abs(slice_df['frame_number'].diff())) #velocity
            
            eval_df.loc[i, 'id'] = pid
            eval_df.loc[i, 'avx'] = slice_df['x'].mean()
            eval_df.loc[i, 'avy'] = slice_df['y'].mean()
            eval_df.loc[i, 'avdxy'] = np.mean(dxy[1:])
            eval_df.loc[i, 'frame'] = slice_df['frame_number'].iloc[-1]
            i = i+1
        
    eval_df = eval_df.astype({'avx':float,'avy':float,'avdxy':float, 'id':int})
    
    return filtered_particles, eval_df

#%%
# bayes opt fct
def opt_condition(param_alpha):
    print('params:\n a = %.4f'%(param_alpha))
    f_p, e_df = som_output(param_alpha)
    print(e_df.shape)
    if e_df.shape[0] < 2: #bandaid solution if no particle matches found
        rel_error = 100
    else:
        v_error   = np.std(np.array(e_df['avdxy']))
        vel_mean  = np.mean(e_df['avdxy'])
        rel_error = v_error/vel_mean 
    print('rel error = %.4f' %rel_error)
    return 1/rel_error    

#%%
### optimization ###
optimizer = bayes_opt.BayesianOptimization(opt_condition, pbounds=pbounds)
optimizer.maximize(init_points=opt_init_points, n_iter=opt_iterations)

alpha_opt = optimizer.max['params']['param_alpha']
filtered_particles, eval_df = som_output(alpha_opt)
title = 'alpha=%.4f'%(alpha_opt)


#%%
##plots
suptitle = image_folder.split('_')[4] +  ' / ' + image_folder.split('/')[1] 

# particle traces
trace_df = filtered_particles

plt.figure(dpi=500)
for ids in trace_df['particle_id']:
    trace_slice = trace_df.loc[filtered_particles['particle_id']==ids]
    trace_slice = trace_slice.sort_values('frame_number')
    plt.plot(trace_slice['x'], trace_slice['y'])
plt.xlim(0,1600)
plt.ylim(0,600)
plt.title(title)
plt.suptitle(suptitle)
plt.show()

xyz_df = eval_df.sort_values('avx')
xyz_df = xyz_df.sort_values('frame')

#velocities
plt.figure(dpi=500)
plt.plot(xyz_df['avx'], xyz_df['avdxy'], '.', markersize=1)
vel_av = np.average(xyz_df['avdxy'])
plt.ylim(0, vel_av*2)
plt.ylabel('Velocity [m/s]')
plt.xlabel('x [px]')
plt.title(title)
plt.suptitle(suptitle)
plt.show()

plt.figure(dpi=500)
scatter = plt.scatter(xyz_df['avx'], xyz_df['avy'],c=xyz_df['avdxy'], s=2, cmap='inferno', vmin=0, vmax=np.mean(xyz_df['avdxy'])*2)
plt.colorbar(scatter)
plt.xlim(0,1600)
plt.ylim(0,600)
plt.ylabel('y [px]')
plt.xlabel('x [px]')
plt.title(title)
plt.suptitle(suptitle)
plt.show()



