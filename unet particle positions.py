# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 03:02:09 2024

@author: Sali
"""


#%%
#####
#Imports and Initialization

from PIL import Image
import numpy as np
import tensorflow as tf
import os
import skimage.io

#####
#First gather positions in original images by using U-Net
unet = tf.keras.models.load_model("unet_mixedfloat16.h5", compile=False)


#%%
#Set directory/files of particle images and background

#
background_file = 'Background_VM1_AVI_231006_130018/frame_0000.bmp'
image_folder = 'VM1_AVI_231006_130201_90Pa_1mA/neg/'
# image_folder = 'VM1_AVI_231006_130201_90Pa_1mA/pos/'

#%%
background_data = tf.keras.utils.load_img(background_file, color_mode='grayscale', target_size=None)

### resolutions need to be divisible by 4 ###
if image_folder[:3]=='VM1':
    background_data = tf.keras.utils.load_img(background_file, color_mode='grayscale', target_size=(264,1600))
background_data = np.expand_dims(background_data, axis=0)
background_data = np.expand_dims(background_data, axis=-1) / 255

image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".bmp")]
image_files.sort()

particle_folder = image_folder[:-1] + '_positions/' #create folder for positions
if not os.path.exists(particle_folder):
    os.makedirs(particle_folder)    

for filename in image_files:
    image_tensor = tf.keras.utils.load_img(filename, color_mode='grayscale', target_size=None)
    if image_folder[:3]=='VM1':
        image_tensor = tf.keras.utils.load_img(filename, color_mode='grayscale', target_size=(264,1600))
    image_tensor = np.expand_dims(image_tensor, axis=0)
    image_tensor = np.expand_dims(image_tensor, axis=-1) / 255
    image_tensor = image_tensor - (background_data)*0.95
    unet_result = unet(image_tensor)
    particle_mask = unet_result[0, :, :, 0]>0.99
    particles = np.array(skimage.measure.regionprops(skimage.measure.label(particle_mask)))
    if len(particles) > 0:
        particles = np.array([c["Centroid"] for c in particles])
        particles[:, [0, 1]] = particles[:,[1, 0]]  # correcting, so particles[:,0] is x and particles[:,1] is y
        np.save(particle_folder + filename.split('/')[2].split('.')[0] + '.npy', particles)
img = np.array(Image.open(image_files[0]))/255
particles_to_show = np.load(particle_folder + image_files[0].split('/')[2].split('.')[0] + '.npy')


