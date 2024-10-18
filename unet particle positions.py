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
background_file = 'C://Users/Lukas/Documents/GitHub/Make_BMP/VM1_AVI_240124_133913_Background/frame_0000.bmp'
image_folder = 'C://Users/Lukas/Documents/GitHub/Make_BMP/Neon/VM1_AVI_240124_133913_40pa_1mA/neg/'
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

def normalize_brightness(image, min_brightness, max_brightness):
    # Ensure the image is a numpy array
    image = np.array(image).astype(np.float32)

    # Normalize the image to [0, 1] range
    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Scale the image to [min_brightness, max_brightness] range
    normalized_image = image_normalized * (max_brightness - min_brightness) + min_brightness

    return np.clip(normalized_image, min_brightness, max_brightness).astype(np.uint8)

def enhance_contrast_clahe(image, filter_kernel):
    """
    Enhance the contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    This method improves local contrast and avoids over-amplifying noise.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(filter_kernel, filter_kernel))
    enhanced_image = clahe.apply(image)

    return enhanced_image   
 
def sharpen_image(image, filter_kernel):
    """
    Apply sharpening filter to enhance the edges and details of the image.
    """
    kernel = np.array([[0, -1, 0], 
                       [-1, 8,-1], 
                       [0, -1, 0]])  # Simple sharpening kernel
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image
    
def combined_enhancement(image, background, frequency=8.0, filter_kernel = 9):
    """
    Combine CLAHE and sharpening for enhanced particle detection.
    """
    normalized_image = normalize_brightness(image, min_brightness=0, max_brightness=255)
    normalized_background = normalize_brightness(background, min_brightness=0, max_brightness=255)
    normalized = normalized_image - (normalized_background*0.25)
    #clahe_image = enhance_contrast_clahe(normalized_image, filter_kernel)
    #gabor_image = apply_gabor_filter(normalized_image, frequency)
    #sharpened_image = sharpen_image(gabor_image, filter_kernel)
    
    return normalized

#%%
background = np.array(Image.open(background_file))

image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".bmp")]
image_files.sort()

particle_folder = image_folder[:-1] + '_positions/' #create folder for positions
if not os.path.exists(particle_folder):
    os.makedirs(particle_folder)    

for idx, filename in enumerate(image_files):
    image = np.array(Image.open(filename))
    if image_folder[47:50]=='VM1':
        target_size = (1600, 264)  # (width, height)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        resized_background = cv2.resize(np.array(background), (1600, 264), interpolation=cv2.INTER_LINEAR)
    # Enhance the image before U-Net processing
    enhanced_image = combined_enhancement(resized_image, resized_background)
    image_tensor = np.expand_dims(resized_image, axis=0)
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
#%%

