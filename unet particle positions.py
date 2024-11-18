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
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import tensorflow as tf
import os
import skimage.io
from skimage.filters import gabor
import cv2

#####
#First gather positions in original images by using U-Net
unet = tf.keras.models.load_model("unet_mixedfloat16.h5", compile=False)


#%%
#Set directory/files of particle images and background
#
#background_file = 'C://Users/Lukas/Documents/GitHub/Make_BMP/VM2_AVI_240124_133031_Background/frame_0000.bmp'
image_folder = 'C://Users/Lukas/Documents/GitHub/Make_BMP/Neon_3mu/VM1_AVI_231005_120639_70pa_1p5mA/pos/'
#
# Variable to control how often to plot
plot_interval = 5  # Change this to 10 if you want to plot every 10th image
#
#%%
def plot_image_with_mask(image, mask, particles):
    """
    Function to plot an image with its corresponding mask and particle centroids.
    
    Parameters:
    image (numpy array): The original image
    mask (numpy array): The binary mask predicted by the U-Net model
    particles (numpy array): Array of particle positions/centroids
    
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot the original image with the mask overlay
    ax.imshow(image, cmap="gray")
    #ax.imshow(mask, cmap="jet", alpha=0.8)  # Overlay mask with transparency

    # Add red circles for each detected particle centroid
    for particle in particles:
        circle = Circle((particle[0], particle[1]), radius=10, edgecolor='red', facecolor='none', lw=.5)
        ax.add_patch(circle)

    ax.set_title("Original Image with Predicted Mask and Particle Centroids")
    ax.axis("off")

    plt.show()

def apply_gabor_filter(image, frequency):
    # Apply the Gabor filter (real part) to the image
    filtered_real, _ = gabor(image, frequency=frequency)
    
    # Normalize the result
    filtered_real = (filtered_real - filtered_real.min()) / (filtered_real.max() - filtered_real.min()) * 255
    return filtered_real.astype(np.uint8)

def scale_image_with_threshold(image, min_threshold, max_threshold):
    """
    Scales the image pixel values to the range 0 - 255 and applies the given min/max thresholds.
    
    Parameters:
    - image: Input image as a NumPy array.
    - min_threshold: Minimum threshold value.
    - max_threshold: Maximum threshold value.
    
    Returns:
    - Scaled image with values between 0 and 255.
    """
    # Clip the image values to the range defined by min_threshold and max_threshold
    clipped_image = np.clip(image, min_threshold, max_threshold)
    
    # Normalize the image to the range 0 - 1
    normalized_image = (clipped_image - min_threshold) / (max_threshold - min_threshold)
    
    # Scale the normalized image to the range 0 - 255
    scaled_image = (normalized_image * 255).astype(np.uint8)
    
    return scaled_image

def combined_enhancement(image, filter_kernel=5, noise_threshold=12):
    """
    Combine CLAHE, noise reduction, and sharpening for enhanced particle detection.
    """
    min_brightness_var = 11
    max_brightness_var = 14
    normalized_image = scale_image_with_threshold(image, min_threshold=min_brightness_var, max_threshold=max_brightness_var)
    
    # Apply a median filter to reduce noise (remove isolated pixels)
    denoised_image = cv2.medianBlur(normalized_image, ksize=filter_kernel)
    
    # Remove remaining noise by setting pixels below the noise threshold to zero
    denoised_image[denoised_image < noise_threshold] = 0
   
    return denoised_image

#%%
image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".bmp")]
image_files.sort()

particle_folder = image_folder[:-1] + '_positions/' #create folder for positions
if not os.path.exists(particle_folder):
    os.makedirs(particle_folder)    

for idx, filename in enumerate(image_files):
    image = np.array(Image.open(filename))
    if image_folder[51:54]=='VM1':
        target_size = (1600, 264)  # (width, height)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = image
    # Enhance the image before U-Net processing
    enhanced_image = combined_enhancement(resized_image)
    image_tensor = np.expand_dims(enhanced_image, axis=0)
    image_tensor = np.expand_dims(image_tensor, axis=-1)#/255
    unet_result = unet(image_tensor)
    particle_mask = unet_result[0, :, :, 0]>0.99
    particles = np.array(skimage.measure.regionprops(skimage.measure.label(particle_mask)))
    if len(particles) > 0:
        particles = np.array([c["Centroid"] for c in particles])
        particles[:, [0, 1]] = particles[:,[1, 0]]  # correcting, so particles[:,0] is x and particles[:,1] is y
        np.save(particle_folder + filename.split('/')[10].split('.')[0] + '.npy', particles)
    # Load and display image with mask and particles if condition is met
    if idx % plot_interval == 0:
        #img = np.array(Image.open(filename)) / 255
        particles_to_show = np.load(particle_folder + filename.split('/')[10].split('.')[0] + '.npy')

        # Plot the image, mask, and particles
        plot_image_with_mask(enhanced_image, particle_mask, particles_to_show)
        
print(image_folder)
#%%

