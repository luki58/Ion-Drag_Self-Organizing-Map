# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:39:39 2024

@author: Sali
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.stats as scistat

#120pa
#load_filename = 'VM2_AVI_231005_113723_120pa_1mA_neg_filtered_particles'
#load_filename = 'VM2_AVI_231005_113723_120pa_1mA_pos_filtered_particles'

#100pa
#load_filename = 'VM2_AVI_231005_114720_100pa_1mA_neg_filtered_particles'
#load_filename = 'VM2_AVI_231005_114720_100pa_1mA_pos_filtered_particles'

#90pa
#load_filename = 'VM2_AVI_231005_120016_090pa_1mA_neg_filtered_particles'
#load_filename = 'VM2_AVI_231005_120016_090pa_1mA_pos_filtered_particles'

#70pa
load_filename = 'VM2_AVI_231005_120730_070pa_1mA_neg_filtered_particles'
#load_filename = 'VM2_AVI_231005_120730_070pa_1mA_pos_filtered_particles'

#60pa
#load_filename = 'VM2_AVI_231005_121115_060pa_1mA_neg_filtered_particles'
#load_filename = 'VM2_AVI_231005_121115_060pa_1mA_pos_filtered_particles'


title = load_filename.split('_')[4] +  ' / ' + load_filename.split('_')[6] 

framerate = 1/50
pixelsize = 14.7e-6

load_folder = 'json_files_opt'

filtered_particles = pd.read_json(load_folder + '/' + load_filename + '.json')

#%%
#particle traces
i=0
trace_df = filtered_particles
for ids in trace_df['particle_id']:
    
    if i == 0:
        plt.figure(dpi=500)
    trace_slice = trace_df.loc[filtered_particles['particle_id']==ids]
    trace_slice = trace_slice.sort_values('frame_number')
    plt.plot(trace_slice['x'], trace_slice['y'])
    plt.xlim(0,1600)
    plt.ylim(0,600)
    plt.title(title)
    i+=1
    
    #if i==1000:
    #    plt.xlim(np.min(filtered_particles['x']), np.max(filtered_particles['x']))
    #    plt.ylim(np.min(filtered_particles['y']), np.max(filtered_particles['y']))
    #    plt.show()
    #    plt.clf()
    #    i=0


#%%
### calculate differences of x/y pos. and average x/y values for each particle id

#grouped_df = filtered_particles.sort_values('particle_id')

particle_ids = filtered_particles['particle_id'].unique().astype(int)

eval_df = pd.DataFrame(columns=['avx', 'avy', 'avdxy', 'id', 'frame'])

frame_calc = 3 #number of frames to calc. v
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
        dxy = 1000*np.sqrt(dx**2 + dy**2)*pixelsize / (framerate*np.abs(slice_df['frame_number'].diff())) #m/s -> mm/s
        #dxy = np.abs(dx)*pixelsize / (50*np.abs(slice_df['frame_number'].diff())) # only vx
        
        eval_df.loc[i, 'id'] = pid
        eval_df.loc[i, 'avx'] = slice_df['x'].mean()
        eval_df.loc[i, 'avy'] = slice_df['y'].mean()
        eval_df.loc[i, 'avdxy'] = np.mean(dxy[1:])
        eval_df.loc[i, 'frame'] = slice_df['frame_number'].iloc[-1]

        i = i+1
eval_df = eval_df.astype({'avx':float,'avy':float,'avdxy':float, 'id':int})

#%%
### plots

xyz_df = eval_df.sort_values('avx')
#xyz_df['avdxy'] = xyz_df['avdxy']/np.max(xyz_df['avdxy'])
xyz_df = xyz_df.sort_values('frame')

plt.figure(dpi=500)
plt.plot(xyz_df['avx'], xyz_df['avdxy'], '.', markersize=1)
vel_av = np.average(xyz_df['avdxy'])
plt.ylim(0, vel_av*2)
plt.ylabel('Velocity [mm/s]')
plt.xlabel('x [px]')
plt.title(title)
plt.show()

plt.figure(dpi=500)
scatter = plt.scatter(xyz_df['avx'], xyz_df['avy'],c=xyz_df['avdxy'], s=2, cmap='inferno', vmin=0, vmax=np.mean(xyz_df['avdxy'])*2)
plt.colorbar(scatter)
plt.xlim(0,1600)
plt.ylim(0,600)
plt.ylabel('y [px]')
plt.xlabel('x [px]')
plt.title(title)
plt.show()

vel_mean = np.mean(xyz_df['avdxy'])
vel_round = xyz_df['avdxy'].round(2)
counts = vel_round.value_counts()
v_error = scistat.mstats.sem(np.array(eval_df['avdxy']))

plt.figure(dpi=500)
plt.bar(counts.index, height=counts, width=5e-2)
plt.xlim(0, vel_mean*2)
plt.xlabel('v[mm/s]')
plt.ylabel('count')
plt.title(title)
plt.suptitle('mean=%.4f | error=%.6f [mm/s]'%(vel_mean,v_error))
plt.show()

most_particles_frame = filtered_particles['frame_number'].value_counts().index[0]
fp_slice = filtered_particles.loc[filtered_particles['frame_number']==most_particles_frame]

plt.figure(dpi=500)
plt.plot(fp_slice['x'],fp_slice['y'],'x')
plt.xlim(0,1600)
plt.ylim(0,600)
plt.show()

# folder_json_raw = 'coordinate_data'

# json_raw = fp_slice.to_json()
# json_raw = json.loads(json_raw)

# save_file = open(folder_json_raw + '/' + '_'.join(load_filename.split('_')[:-2]) + '_coords' +'.json','w')
# json.dump(json_raw, save_file)
# save_file.close()





