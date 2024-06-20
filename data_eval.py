# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:58:37 2024

@author: Sali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

load_filename = 'VM2_AVI_231005_114720_100pa_1mA_neg'
folder_csv = 'csv_files'

#%%

### load csv
eval_df = pd.read_csv(folder_csv+'/'+load_filename+'.csv')

xyz_df = eval_df.sort_values('avx')
xyz_df['avdxy'] = xyz_df['avdxy']/np.max(xyz_df['avdxy'])
xyz_df = xyz_df.sort_values('frame')



fwindow = 10
framecount = len(eval_df['frame'].unique().astype(int))
uframes = np.arange(0, framecount, fwindow)

for i in uframes:
     a = i+min(xyz_df['frame'])
     b = (i+fwindow)+min(xyz_df['frame'])
     
     plt.figure(dpi=500)
     plt.plot(xyz_df['avx'], xyz_df['avdxy'], '.', markersize=1)
     plt.xlim(0, 1600)
     plt.ylim(0, 1)
     plt.title('Frames %d-%d' %(a, b))
     plt.show()
     
     plt.figure(dpi=500)
     scatter = plt.scatter(xyz_df['avx'].loc[(a < xyz_df['frame']) & (xyz_df['frame'] < b)],
                           xyz_df['avy'].loc[(a < xyz_df['frame']) & (xyz_df['frame'] < b)],
                           c=xyz_df['avdxy'].loc[(a < xyz_df['frame']) & (xyz_df['frame'] < b)],
                           s=2, cmap='inferno', vmin=0, vmax=1)
     plt.colorbar(scatter)
     plt.xlim(0, 1600)
     plt.ylim(0, 600)
     plt.title('Frames %d-%d' %(a, b))
     plt.show()


plt.figure(dpi=500)
plt.plot(xyz_df['avx'], xyz_df['avdxy'], '.', markersize=1)
plt.show()

plt.figure(dpi=500)
scatter = plt.scatter(xyz_df['avx'], xyz_df['avy'],c=xyz_df['avdxy'], s=2, cmap='inferno', vmin=0, vmax=1)
plt.colorbar(scatter)
plt.xlim(0, 1600)
plt.ylim(0, 600)
plt.show()


#%% contour plots
# xi = np.linspace(0, 1600, 300)
# yi = np.linspace(0,600, 300)
# xgrid, ygrid = np.meshgrid(xi, yi)

# zgrid = scipy.interpolate.griddata((eval_df['avx'].values, eval_df['avy'].values), eval_df['avdxy'].values, (xgrid, ygrid))
# zgrid[np.isnan(zgrid)] = 0
# zgrid = zgrid/np.max(zgrid)

# zvmin = np.min(zgrid[zgrid>np.min(zgrid)])
# plt.figure(dpi=500)
# imshow = plt.imshow(zgrid, vmin=zvmin, vmax=1, cmap='inferno', interpolation='gaussian')
# plt.colorbar(imshow)
# plt.show()

# clevels = np.arange(0, 1, 0.1)
# plt.figure(dpi=500)
# contour = plt.contourf(xgrid, ygrid, zgrid, clevels, cmap='inferno')
# plt.show()






