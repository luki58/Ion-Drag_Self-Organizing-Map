# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:52:57 2024

@author: Sali
"""

#%%
#####
#Imports and Initialization

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import pandas as pd
import som_class
import json
import scipy
import bayes_opt


framerate = 1/50
pixelsize = 14.7e-6

#alpha      = 0.04
distance_threshold = 10
startradius = 100
endradius   = 0.5
#iterations  = 50
epsilon     = 4

save = 0

#distance_threshold = 10
#alpha = 0.012
#startradius = 100
#endradius = 1
#iterations = 10
#epsilon = 4


#####
#First gather positions in original images by using U-Net

import tensorflow as tf
import os
import skimage.io

unet = tf.keras.models.load_model("unet_mixedfloat16.h5", compile=False)


#%%
#Set directory/files of particle images and background

# #
# background_file = 'Background_VM2_AVI_231005_112452/frame_0000.bmp'
# image_folder = 'VM2_AVI_231005_113723_120pa_1mA/neg/'
# image_folder = 'VM2_AVI_231005_113723_120pa_1mA/pos/'
# image_folder = 'VM2_AVI_231005_114720_100pa_1mA/neg/'
# image_folder = 'VM2_AVI_231005_114720_100pa_1mA/pos/'

# #
# background_file = 'Background_VM2_AVI_231005_115847/frame_0000.bmp'
# image_folder = 'VM2_AVI_231005_120016_090pa_1mA/neg/'
# image_folder = 'VM2_AVI_231005_120016_090pa_1mA/pos/'
# image_folder = 'VM2_AVI_231005_120730_070pa_1mA/neg/'
# image_folder = 'VM2_AVI_231005_120730_070pa_1mA/pos/'
# image_folder = 'VM2_AVI_231005_121115_060pa_1mA/neg/'
# image_folder = 'VM2_AVI_231005_121115_060pa_1mA/pos/'

# #
# background_file = 'Background_VM1_AVI_240124_115747/frame_0002.bmp'
# image_folder = 'VM1_AVI_240124_121230_050pa_1mA/neg/'
# image_folder = 'VM1_AVI_240124_121230_050pa_1mA/pos/'
# image_folder = 'VM1_AVI_240124_133913_40pa_1mA/neg/'
# image_folder = 'VM1_AVI_240124_133913_40pa_1mA/pos/'

# #
background_file = 'Background_VM2_AVI_240124_133031/frame_0001.bmp'
image_folder = 'VM2_AVI_240124_140000_30pa_1mA/neg/'
# image_folder = 'VM2_AVI_240124_140000_30pa_1mA/pos/'

# #
# background_file = 'Background_VM2_AVI_240125_142119/frame_0002.bmp'
# image_folder = 'VM2_AVI_240125_142119_023pa_1mA/pos/'
# image_folder = 'VM2_AVI_240125_142119_23pa_1p5mA/neg/'
# image_folder = 'VM2_AVI_240125_142119_23pa_1p5mA/pos/'

# #
# background_file = 'Background_VM1_AVI_240125_142118/frame_0002.bmp'
# image_folder = 'VM1_AVI_240125_142118_18pa_1mA/neg/'
# image_folder = 'VM1_AVI_240125_142118_18pa_1mA/pos/'
# image_folder = 'VM1_AVI_240125_142118_18pa_1p5mA/neg/'
# image_folder = 'VM1_AVI_240125_142118_18pa_1p5mA/pos/'


#%%

background_data = tf.keras.utils.load_img(background_file, color_mode='grayscale', target_size=None)
background_data = np.expand_dims(background_data, axis=0)
background_data = np.expand_dims(background_data, axis=-1) / 255

image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".bmp")]
image_files.sort()


particle_folder = image_folder[:-1] + '_positions/' #create folder for positions
if not os.path.exists(particle_folder):
    os.makedirs(particle_folder)    
    
    for filename in image_files:
        image_tensor = tf.keras.utils.load_img(filename, color_mode='grayscale', target_size=None)
        image_tensor = np.expand_dims(image_tensor, axis=0)
        image_tensor = np.expand_dims(image_tensor, axis=-1) / 255
        image_tensor = image_tensor - (background_data)*0.95 #!!! background subtract
        unet_result = unet(image_tensor)
        particle_mask = unet_result[0, :, :, 0]>0.99
        particles = np.array(skimage.measure.regionprops(skimage.measure.label(particle_mask)))
        if len(particles) > 0:
            particles = np.array([c["Centroid"] for c in particles])
            particles[:, [0, 1]] = particles[:,[1, 0]]  # correcting, so particles[:,0] is x and particles[:,1] is y
            np.save(particle_folder + filename.split('/')[2].split('.')[0] + '.npy', particles)
    img = np.array(Image.open(image_files[0]))/255
    particles_to_show = np.load(particle_folder + image_files[0].split('/')[2].split('.')[0] + '.npy')


#%%
#####
# Tracing particles over multipe images:
    
def som_output(alpha, iterations, iopt):    

    som = som_class.SOM(distance_threshold, alpha, startradius, endradius, iterations, epsilon)
    
    available_images = len(os.listdir(particle_folder)) #trace over all images in folder
    allmatches = np.array((),dtype=object)
    original_coords = np.array((),dtype=object)
    position_files = [os.path.join(particle_folder, img) for img in os.listdir(particle_folder) if img.endswith(".npy")]
    position_files.sort()
    min_length = 5
    
    if iopt==1:
        images_to_match = np.min([available_images,20]) #!!!
    
    else:
        images_to_match = available_images
    
    for i in range(min(images_to_match,len(position_files)-1)):
        filename1 = position_files[i]
        filename2 = position_files[i+1]
        coords1, coords2 = som.read_in_coordinates(filename1, filename2)
        coords1, coords2 = som.match_particles(coords1, coords2)
        if i==0:
            match = np.array((coords1,coords2),dtype=object)
        else:
            match = np.array([coords2,None],dtype=object)
            match = np.delete(match,1)
        orig_coords = np.array([coords1,None],dtype=object)
        orig_coords = np.delete(orig_coords,1)
        allmatches = np.append(allmatches,match)
        original_coords = np.append(original_coords,orig_coords)
     #   print(f"finished number {i+1}")
    #print("now tracing")
    allmatches = som.tracing(allmatches,original_coords)
    starting_image = int(position_files[0].split('/')[2].split('.')[0].split('_')[1])
    dataframe = som.convert_to_dataframe(allmatches,starting_image)
    filtered_particles = som.dataframe_min_length_filter(dataframe,min_length)
    
    # calculation velocity and positions
    
    particle_ids = filtered_particles['particle_id'].unique().astype(int)
    
    eval_df = pd.DataFrame(columns=['avx', 'avy', 'avdxy', 'id', 'frame'])
    
    frame_calc = 3 #number of frames to calc. velocity #!!!
    i = 0
    valid_count = 1
    invalid_count = 1
    for pid in particle_ids:
        
        pid_df = filtered_particles.loc[filtered_particles['particle_id']==pid]
        maxframes = len(pid_df['frame_number'])
        frame_slices = int((maxframes-maxframes%frame_calc)/frame_calc)
                    
        y_diff = np.abs(pid_df['y'].iloc[0]-pid_df['y'].iloc[-1])
        x_diff = np.abs(pid_df['x'].iloc[0]-pid_df['x'].iloc[-1])
        
        if x_diff>y_diff and maxframes>6: #!!!
            valid_count += 1
        else:
            invalid_count += 1
        
        for xf in range(frame_slices):
            slice_df = pid_df.sort_values('frame_number').iloc[(xf*frame_calc):((xf+1)*frame_calc)]
            
            dx = slice_df['x'].diff()
            dy = slice_df['y'].diff()
            dxy = np.sqrt(dx**2 + dy**2)*pixelsize / (framerate*np.abs(slice_df['frame_number'].diff()))
            #dxy = np.abs(dx)*pixelsize / (50*np.abs(slice_df['frame_number'].diff())) # only vx
            
            eval_df.loc[i, 'id'] = pid
            eval_df.loc[i, 'avx'] = slice_df['x'].mean()
            eval_df.loc[i, 'avy'] = slice_df['y'].mean()
            eval_df.loc[i, 'avdxy'] = np.mean(dxy[1:])
            eval_df.loc[i, 'frame'] = slice_df['frame_number'].iloc[-1]
    
            i = i+1
        
    eval_df = eval_df.astype({'avx':float,'avy':float,'avdxy':float, 'id':int})
    count_ratio = valid_count/invalid_count
    
    return filtered_particles, eval_df,  count_ratio

#%%
#def opt_condition(param_a, param_i, param_e):
def opt_condition(param_alpha, param_iter):
    param_iter_int = int(param_iter)
    #param_e_int = int(param_e)
    iopt=1
    print('params:\n a = %.4f \n i = %d \n'%(param_alpha, param_iter_int))
    f_p, e_df, c_ratio = som_output(param_alpha, param_iter_int, iopt)
    
    if e_df.shape[0] == 0:
        rel_error = 100
    else:
        v_error   = scipy.stats.mstats.sem(np.array(e_df['avdxy']))
        vel_mean  = np.mean(e_df['avdxy'])
        rel_error = v_error/vel_mean
        
    print('rel error = %.4f' %rel_error)
    
    print('count: %d' %c_ratio)
    #return c_ratio
    return 1/rel_error    

#%%

## optimization #!!!
pbounds = {'param_alpha':(0.001,0.1), 'param_iter':(10,100)}
optimizer = bayes_opt.BayesianOptimization(opt_condition, pbounds=pbounds)

optimizer.maximize(init_points=5, n_iter=50)

alpha_opt = optimizer.max['params']['param_alpha']
iterations_opt = int(optimizer.max['params']['param_iter'])
#epsilon_opt = int(optimizer.max['params']['param_e'])
#endradius_opt = optimizer.max['params']['param_endradius']

iopt=0
#filtered_particles, eval_df , count_opt = som_output(alpha_opt, iterations_opt, epsilon_opt)
filtered_particles, eval_df , count_opt = som_output(alpha_opt, iterations_opt, iopt)

title = 'alpha=%.4f | iter.=%d '%(alpha_opt, iterations_opt)


suptitle = image_folder.split('_')[4] +  ' / ' + image_folder.split('/')[1] 

#%%
##plots

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
#som.plot_traces(filtered_particles)

xyz_df = eval_df.sort_values('avx')
#xyz_df['avdxy'] = xyz_df['avdxy']/np.max(xyz_df['avdxy'])
xyz_df = xyz_df.sort_values('frame')

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



#%%
### save to json
#raw/filtered particles
# if save != 0:
#     folder_json_raw = 'json_files_opt'
    
#     json_raw = filtered_particles.to_json()
#     json_raw = json.loads(json_raw)
    
#     save_file = open(folder_json_raw + '/' + image_folder.split('/')[0] + '_' + image_folder.split('/')[1] + '_filtered_particles' + '.json','w')
#     json.dump(json_raw, save_file)
#     save_file.close()
    
#     #calcs.
#     folder_json = 'json_files'
    
#     json_calc = eval_df.to_json()
#     json_calc = json.loads(json_calc)
    
#     save_file = open(folder_json + '/' + image_folder.split('/')[0] + '_' + image_folder.split('/')[1] + '_calcs' + '.json','w')
#     json.dump(json_calc, save_file)
#     save_file.close()

#%%
#todo

# good params? -> bayesian opt
# -- max condition ?!
# keep iterations constant?!


