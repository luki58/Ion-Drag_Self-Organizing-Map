# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:39:39 2024

@author: Sali
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

load_filename = 'VM2_AVI_231005_114720_100pa_1mA_neg_filtered_particles'
load_csv = 'csv_files_raw'
filtered_particles = pd.read_csv(load_csv+'/'+load_filename+'.csv')

#%%
### calculate differences of x/y pos. and average x/y values for each particle id

#grouped_df = filtered_particles.sort_values('particle_id')

particle_ids = filtered_particles['particle_id'].unique().astype(int)

eval_df = pd.DataFrame(columns=['avx', 'avy', 'avdxy', 'id', 'frame'])
row_df = pd.DataFrame(columns=['avx', 'avy', 'avdxy', 'id', 'frame'])

frame_calc = 5 #number of frames to calc. v
i = 0

for pid in particle_ids:
    
    pid_df = filtered_particles.loc[filtered_particles['particle_id']==pid]
    maxframes = len(pid_df['frame_number'])
    frame_slices = int((maxframes-maxframes%frame_calc)/frame_calc)
    #print('pid = %d' %pid)
    
    for xf in range(frame_slices):
        slice_df = pid_df.sort_values('frame_number').iloc[(xf*frame_calc):((xf+1)*frame_calc)]
        
        dx = slice_df['x'].diff()
        dy = slice_df['y'].diff()
        dxy = np.sqrt(dx**2 + dy**2)
        
        eval_df.loc[i, 'id'] = pid
        eval_df.loc[i, 'avx'] = slice_df['x'].mean()
        eval_df.loc[i, 'avy'] = slice_df['y'].mean()
        eval_df.loc[i, 'avdxy'] = np.mean(dxy[1:])
        eval_df.loc[i, 'frame'] = slice_df['frame_number'].iloc[-1]

        i = i+1
eval_df = eval_df.astype({'avx':float,'avy':float,'avdxy':float, 'id':int})

#%%
# save to csv and json

save_filename = 'VM2_AVI_231005_114720_100pa_1mA_neg'
folder_csv = 'csv_files'
folder_json = 'json_files'

eval_df = eval_df.sort_values(['frame', 'id'])
eval_df.to_csv(folder_csv+'/'+save_filename+'.csv')


json_file = {
            'index':eval_df.index.to_list(),
            'id':eval_df['id'].to_list(),
            'avx':eval_df['avx'].to_list(),
            'avy':eval_df['avy'].to_list(),
            'avdxy':eval_df['avdxy'].to_list(),
            'frame':eval_df['frame'].to_list(),
            }

save_file = open(folder_json+'/'+save_filename+'.json','w')
json.dump(json_file, save_file)
save_file.close()


#%%
### plots

xyz_df = eval_df.sort_values('avx')
xyz_df['avdxy'] = xyz_df['avdxy']/np.max(xyz_df['avdxy'])
xyz_df = xyz_df.sort_values('frame')

plt.figure(dpi=500)
plt.plot(xyz_df['avx'], xyz_df['avdxy'], '.', markersize=1)
plt.show()

plt.figure(dpi=500)
scatter = plt.scatter(xyz_df['avx'], xyz_df['avy'],c=xyz_df['avdxy'], s=2, cmap='inferno', vmin=0, vmax=1)
plt.colorbar(scatter)
plt.xlim(0,1600)
plt.ylim(0,600)
plt.show()

#%%
#todo

#pixel/fps values
#x / x+y compare
# pos/neg compare


