# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:11:55 2024

@author: Sali
"""



    #%%
    #####
    #Imports and Initialization

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy

import pandas as pd
import som_class

distance_threshold = 10
alpha = 0.012
startradius = 100
endradius = 1
iterations = 10
epsilon = 4

som = som_class.SOM(distance_threshold, alpha, startradius, endradius, iterations, epsilon)


    #####
    #First gather positions in original images by using U-Net

import tensorflow as tf
import os
import skimage.io

unet = tf.keras.models.load_model("unet_mixedfloat16.h5", compile=False)


#%%
#Set directory/files of particle images and background #!!!


#background_file = 'Background_VM2_AVI_231005_112452/frame_0000.bmp'

#image_folder = 'VM2_AVI_231005_113723_120pa_1mA/neg/'
#image_folder = 'VM2_AVI_231005_113723_120pa_1mA/pos/'
#image_folder = 'VM2_AVI_231005_114720_100pa_1mA/neg/'
#image_folder = 'VM2_AVI_231005_114720_100pa_1mA/pos/'

background_file = 'Background_VM2_AVI_231005_115847/frame_0000.bmp'

#image_folder = 'VM2_AVI_231005_120016_090pa_1mA/neg/'
#image_folder = 'VM2_AVI_231005_120016_090pa_1mA/pos/'
#image_folder = 'VM2_AVI_231005_120730_070pa_1mA/neg/'
#image_folder = 'VM2_AVI_231005_120730_070pa_1mA/pos/'
#image_folder = 'VM2_AVI_231005_121115_060pa_1mA/neg/'
image_folder = 'VM2_AVI_231005_121115_060pa_1mA/pos/'

background_data = tf.keras.utils.load_img(background_file, color_mode='grayscale', target_size=None)
background_data = np.expand_dims(background_data, axis=0)
background_data = np.expand_dims(background_data, axis=-1) / 255

image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".bmp")]
image_files.sort()

particle_folder = image_folder[:-1] + '_positions/' #!!! create folder for positions
if not os.path.exists(particle_folder):
    os.makedirs(particle_folder)    



for filename in image_files:
    image_tensor = tf.keras.utils.load_img(filename, color_mode='grayscale', target_size=None)
    image_tensor = np.expand_dims(image_tensor, axis=0)
    image_tensor = np.expand_dims(image_tensor, axis=-1) / 255
    image_tensor = image_tensor - (background_data)*0.99 #!!! background subtract / 0.99
    unet_result = unet(image_tensor)
    particle_mask = unet_result[0, :, :, 0]>0.99
    particles = np.array(skimage.measure.regionprops(skimage.measure.label(particle_mask)))
    if len(particles) > 0:
        particles = np.array([c["Centroid"] for c in particles])
        particles[:, [0, 1]] = particles[:,[1, 0]]  # correcting, so particles[:,0] is x and particles[:,1] is y
        np.save(particle_folder + filename.split('/')[2].split('.')[0] + '.npy', particles) #!!! split 1->2
img = np.array(Image.open(image_files[0]))/255
particles_to_show = np.load(particle_folder + image_files[0].split('/')[2].split('.')[0] + '.npy') #!!! split 1->2
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap="gray")
plt.scatter(particles_to_show[:, 0], particles_to_show[:, 1], facecolors='None',edgecolors='blue')
plt.show()


#%%
# filename1 = particle_folder + image_files[0].split('/')[2].split('.')[0] + '.npy' #!!! split 1->2
# filename2 = particle_folder + image_files[1].split('/')[2].split('.')[0] + '.npy' #!!! split 1->2
# coords1, coords2 = som.read_in_coordinates(filename1, filename2) 
# coords1, coords2 = som.match_2048_images(coords1, coords2)


# plt.figure(figsize=(35, 35))
# plt.scatter(coords1[:,0],coords1[:,1],c='blue',s=12)
# plt.scatter(coords2[:,0],coords2[:,1],c='red',s=12)
# i = 0
# for particle in coords2:
#     if particle[2] in coords1[:,2]:
#         i+=1
#         coords1_id = np.where(coords1[:,2]==particle[2])[0][0]
#         coords2_id = np.where(coords2[:,2]==particle[2])[0][0]
#         plt.arrow(coords1[coords1_id][0],coords1[coords1_id][1],coords2[coords2_id][0]-coords1[coords1_id][0],coords2[coords2_id][1]-coords1[coords1_id][1], head_width=3)
# plt.gca().invert_yaxis()
# plt.show()
# print(str(i)+' particles of '+str(len(coords1))+' matched.')


    #%%
    #####
    #Tracing particles over multipe images:

allmatches = np.array((),dtype=object)
original_coords = np.array((),dtype=object)
#images_to_match = 5
images_to_match = len(os.listdir(particle_folder)) #!!! trace over all images in folder
position_files = [os.path.join(particle_folder, img) for img in os.listdir(particle_folder) if img.endswith(".npy")]
position_files.sort()
min_length = 5
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
    print(f"finished number {i+1}")
print("now tracing")
allmatches = som.tracing(allmatches,original_coords)
starting_image = int(position_files[0].split('/')[2].split('.')[0].split('_')[1]) #!!! split 1->2 /.split('_')[1])
dataframe = som.convert_to_dataframe(allmatches,starting_image)
filtered_particles = som.dataframe_min_length_filter(dataframe,min_length)
#som.plot_traces(filtered_particles)

### save to csv
folder_csv = 'csv_files_raw'

save_filtered = filtered_particles.sort_values(['frame_number', 'particle_id'])
save_filtered.to_csv(folder_csv + '/' + image_folder.split('/')[0] + '_' + image_folder.split('/')[1] +'_filtered_particles'+  '.csv')


