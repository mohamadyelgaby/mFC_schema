#!/usr/bin/env python
# coding: utf-8

# In[125]:


##Importing libraries
#%matplotlib notebook

from mpl_toolkits.mplot3d import axes3d    
from joblib import dump, load
import os, sys, pickle, time, re, csv
from collections import defaultdict#

import numpy as np
import pandas as pd

import scipy.stats as st
#import pycircstat as circ_st
import math

import matplotlib.pyplot as plt
import seaborn as sns
import collections, numpy

from itertools import groupby
#from pingouin import partial_corr
from collections import Counter
import random
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter1d
import warnings
import scipy as sp

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

from matplotlib.colors import LinearSegmentedColormap


# In[ ]:





# In[ ]:





# In[2]:


###Folders
Data_folder_P='/Taskspace_abstraction/Data/' ## if working in P
base_dropbox='C:/Users/moham/team_mouse Dropbox/Mohamady El-Gaby/'

#base_dropbox='D:/team_mouse Dropbox/Mohamady El-Gaby/'

Data_folder_dropbox=base_dropbox+'/Taskspace_abstraction/Data/' ##if working in C
Behaviour_output_folder = '/Taskspace_abstraction/Results/Behaviour/'
Ephys_output_folder = '/Taskspace_abstraction/Results/Ephys/'
Ephys_output_folder_dropbox = base_dropbox+'/Taskspace_abstraction/Results/Ephys/'
Intermediate_object_folder_dropbox = Data_folder_dropbox+'/Intermediate_objects/'

Intermediate_object_folder = Data_folder_dropbox+'/Intermediate_objects/'

base_ceph='Z:/mohamady_el-gaby/'
Data_folder_ceph='Z:/mohamady_el-gaby/Taskspace_abstraction_2/Data/'
Data_folder_ceph1='Z:/mohamady_el-gaby/Taskspace_abstraction/Data/'
Data_folder_ceph2='Z:/mohamady_el-gaby/Taskspace_abstraction_2/Data/'

Intermediate_object_folder_ceph = Data_folder_ceph1+'/Intermediate_objects/'

Data_folder=Intermediate_object_folder ###


Code_folder='/Taskspace_abstraction/Code/'

'Data is here: https://drive.google.com/drive/folders/1vJw8AVZmHQrUnvqkASUwAd4t549uKN6b '


# In[21]:


##Importing custom functions
module_path = os.path.abspath(os.path.join(Code_folder))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from mBaseFunctions import rec_dd, remove_empty, rank_repeat, dict_to_array, concatenate_complex2, smooth_circular,polar_plot_stateX2, indep_roll, bar_plotX, plot_scatter, non_repeat_ses_maker, noplot_timecourseBx,non_repeat_ses_maker_old


# In[4]:


'''

-normalised data 
-trial-by-trial pseudo ensemble? - i.e. take lowest number of trials across all recording days and then 
run UMAP on concatenated data
-alternative: take means and do UMAP on that


From Tang et al 2023
In each region, all spatially-tuned units that were active in both environments were pooled across animals 
to build a pseudo-population of neurons (n = 98 CA1 and 171 PFC cells, respectively). 
For each trajectory type, the minimal number of trials (or passes) across animals starting from the first 
trial in a given session was used as the trial number. For each population, single-trial linearized rate map 
of each cell was calculated and binned at a resolution of 1 cm. Bins with spikes from less than 5 neurons were
discarded. 
Uniform Manifold Approximation and Projection (UMAP)80,81 was then run on these n-dimensional data to extract 
low-dimensional neural manifolds (Figures 3J and 3G). 
The hyperparameters for UMAP were: ‘n_dims’ = 3, ‘metric’ = ‘cosine’, ‘n_neighbours’ = 50, and ‘min_dist’ = 0.6, 
similar to previous studies.86,87 To compare the neural manifolds between the novel and familiar environment, the 
UMAP transformation calculated for the N’ session was re-applied to the population activity of the F session, by
applying the fitted N’ UMAP transformation as the ‘template_file’ to the ‘run_umap’ function in the UMAP MATLAB
toolbox81 (Figures 3F and 3G, right).86


'''


# In[ ]:





# In[130]:


#UMAP on ABCD data
abstract_structure='ABCD'
num_states=len(abstract_structure)
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')
ephys_mean_z_all=[]
ephys_mean_z_pertask_all=[]

exclude_place=True

neuron_thr=10


ephys_mean_z_dic=rec_dd()
ephys_mean_z_pertask_dic=rec_dd()
ephys_mean_z_split_dic=rec_dd()

for mouse_recday in recording_days_:
    print(mouse_recday)
    try:
        awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy'                                         ,allow_pickle=True)
        awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy'                              ,allow_pickle=True)
        num_sessions=len(awake_sessions_behaviour)
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        found_ses=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            try:
                Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy',allow_pickle=True)
                found_ses.append(ses_ind)

            except:
                print('Files not found for session '+str(ses_ind))
                continue
        
        noplace_bool=~np.load(Intermediate_object_folder_dropbox+'Place_'+mouse_recday+'.npy',allow_pickle=True)
        
        found_ses_nonrepeat=np.intersect1d(found_ses,non_repeat_ses)
        
        if len(found_ses_nonrepeat)<6:
            print('Not enough tasks')
            continue
        
        len_trials_all=[]
        for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
            ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                           ,allow_pickle=True)
            len_trials_all.append(np.shape(ephys_)[1])

        min_len=min(len_trials_all)
        ephys_allses=[]
        ephys_allses_meanz=[]

        for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
            ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                           ,allow_pickle=True)
            
            
            
            if exclude_place==True:
                ephys_=ephys_[noplace_bool]

            ##set up for across task means
            #ephys_allses.append(ephys_[:,-min_len:])
            ephys_mean=np.nanmean(ephys_,axis=1)
            ephys_allses.append(ephys_mean)

            ###set up for within task means
            ephys_ses_means=np.nanmean(ephys_,axis=1)
            ephys_ses_meansz=st.zscore(ephys_ses_means,axis=1)
            ephys_allses_meanz.append(ephys_ses_meansz)

        ##across task means
        #ephys_allses=np.hstack((ephys_allses))
        #ephys_mean=np.nanmean(ephys_allses,axis=1) ###taking mean across sessions
        ephys_mean=np.nanmean(ephys_allses,axis=0) ###taking mean of means across sessions
        ephys_mean_z=st.zscore(ephys_mean,axis=1)        
        ephys_mean_z_all.append(ephys_mean_z)
        if len(ephys_)>neuron_thr:
            ephys_mean_z_dic[mouse_recday]=ephys_mean_z
        
        ### within task means
        ephys_allses_meanz=np.vstack((ephys_allses_meanz))
        ephys_mean_z_pertask_all.append(ephys_allses_meanz)
        
        #ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'\
        #                               ,allow_pickle=True)
        if len(ephys_)>neuron_thr:
            print('used for per day dictionary')
            ephys_mean_z_pertask_dic[mouse_recday]=ephys_allses_meanz
        
        
            ###test-train split means
            num_ses=len(ephys_allses)
            merged_ephys_all=np.zeros((num_ses,np.shape(ephys_allses)[1],np.shape(ephys_allses)[2]*2))
            merged_ephys_all[:]=np.nan
            for test_ses_ind in np.arange(num_ses):
                training_ses=np.setdiff1d(np.arange(num_ses),test_ses_ind)
                ephys_training_ses_mean=np.nanmean(np.asarray(ephys_allses)[training_ses],axis=0)
                ephys_test_ses_mean=ephys_allses[test_ses_ind]
                merged_ephys=np.hstack((ephys_training_ses_mean,ephys_test_ses_mean))
                merged_ephys_all[test_ses_ind]=merged_ephys
            merged_ephys_all[np.isnan(merged_ephys_all)]=0
            merged_ephys_all_z=st.zscore(merged_ephys_all,axis=0)

            for test_ses_ind in np.arange(num_ses): 
                ephys_mean_z_split_dic[test_ses_ind][mouse_recday]=merged_ephys_all_z[test_ses_ind]
                
        else:
            print('Not enough neurons')

    except Exception as e:
        print(e)
        print('Not used')
ephys_mean_z_all=np.vstack((ephys_mean_z_all))
ephys_mean_z_pertask_all=np.vstack((ephys_mean_z_pertask_all))

ephys_mean_z_pertask_all=ephys_mean_z_pertask_all[~np.isnan(np.mean(ephys_mean_z_pertask_all,axis=1))]


# In[131]:


num_neurons_all=0
for mouse_recday in ephys_mean_z_pertask_dic.keys():
    num_neurons_all+=len(ephys_mean_z_dic[mouse_recday])
    
print(len(ephys_mean_z_pertask_dic.keys()))
print(num_neurons_all)


# In[154]:


ephys_mean_z_pertask_dic.keys()


# In[ ]:





# In[72]:


##UMAP on control (shuffling states)
tt=time.time()

num_iterations=100
abstract_structure='ABCD'
num_states=len(abstract_structure)
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')
ephys_mean_z_shuff_all_alliterations=[]
ephys_mean_z_shuff_dic=rec_dd()
for iteration in np.arange(num_iterations):
    print(iteration)
    ephys_mean_z_shuff_all=[]
    for mouse_recday in recording_days_:
        #print(mouse_recday)
        #try:
        awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
        awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
        num_sessions=len(awake_sessions_behaviour)
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        found_ses=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            try:
                Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                found_ses.append(ses_ind)

            except:
                #print('Files not found for session '+str(ses_ind))
                continue


        found_ses_nonrepeat=np.intersect1d(found_ses,non_repeat_ses)

        if len(found_ses_nonrepeat)<6:
            #print('Not enough tasks')
            continue

        len_trials_all=[]
        for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
            ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                           ,allow_pickle=True)
            len_trials_all.append(np.shape(ephys_)[1])

        min_len=min(len_trials_all)
        ephys_allses=[]
        for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
            ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                           ,allow_pickle=True)

            ephys_shuff_=np.asarray(([np.vstack(([np.hstack((ephys_[neuron,trial].reshape(num_states,90)                              [random.sample(list(np.arange(num_states)), num_states)]))                for trial in np.arange(np.shape(ephys_)[1])]))                                          for neuron in np.arange(np.shape(ephys_)[0])]))

            #ephys_allses.append(ephys_shuff_[:,-min_len:])

            ephys_mean=np.nanmean(ephys_shuff_,axis=1)
            ephys_allses.append(ephys_mean)

        #ephys_allses=np.hstack((ephys_allses))
        ephys_mean=np.nanmean(ephys_allses,axis=0) ###taking mean across sessions
        ephys_mean_z=st.zscore(ephys_mean,axis=1)
        ephys_mean_z_shuff_all.append(ephys_mean_z)
        ephys_mean_z_shuff_dic[iteration][mouse_recday]=ephys_mean_z
        #except:
        #    print('Not used')
    ephys_mean_z_shuff_all=np.vstack((ephys_mean_z_shuff_all))
    ephys_mean_z_shuff_all_alliterations.append(ephys_mean_z_shuff_all)
print(time.time()-tt)


# In[34]:


np.shape(ephys_mean_z_shuff_all_alliterations)


# In[35]:



ses_ind=0 ##only used to calculate num neurons
ephys_mean_z_shuff_dic=rec_dd()
for iteration in np.arange(num_iterations):
    print(iteration)
    num_neurons_all=0
    for mouse_recday in ephys_mean_z_pertask_dic.keys():
        ephys_=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')
        num_neurons=len(ephys_)
        ephys_mean_z_shuff_dic[iteration][mouse_recday]=ephys_mean_z_shuff_all_alliterations[iteration]        [num_neurons_all:num_neurons_all+num_neurons]
        num_neurons_all+=num_neurons


# In[ ]:





# In[ ]:





# In[7]:


plt.plot(np.arange(360),np.nanmean(ephys_mean_z_all,axis=0))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


import matplotlib as mpl

def colorFader(c1_,c2_,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1_))
    c2=np.array(mpl.colors.to_rgb(c2_))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


# In[ ]:





# In[ ]:





# In[37]:


##UMAP embedding
reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
embedding_mean = reducer.fit_transform(ephys_mean_z_all.T)

c1_='yellow'
c2_='purple' 
n=len(embedding_mean)

colors_state=[]
for x in np.arange(n):
    colors_state.append(colorFader(c1_,c2_,x/n))

    
c1_='lightgreen'
c2_='black' 
colors_phase=[]
num_states=(n/90)
for state in np.arange(num_states):
    for x in np.arange(90):
        colors_phase.append(colorFader(c1_,c2_,x/90))


# In[10]:


###saving examples
'''
08042024_1321 - but no angles
08042024_1437 - with angles
'''
#np.save(Intermediate_object_folder+'Embedding_example_ABCD_08042024_1321.npy',embedding_mean)
#np.save(Intermediate_object_folder+'Embedding_example_ABCD_08042024_1437.npy',embedding_mean)
#angles_embedding=[160,80]
#np.save(Intermediate_object_folder+'Embedding_example_ABCD_angles_08042024_1437.npy',angles_embedding)


# In[38]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#embedding_mean=np.load(Intermediate_object_folder+'Embedding_example_ABCD_08042024_1321.npy')

x = embedding_mean[:,0]
y = embedding_mean[:,1]
z = embedding_mean[:,2]

ax.scatter(x, y, z, c=colors_state, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 60 ## rotate around z axis
ax.elev = 30 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)
plt.show()


# In[39]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = embedding_mean[:,0]
y = embedding_mean[:,1]
z = embedding_mean[:,2]

ax.scatter(x, y, z, c=colors_phase, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 30 ## rotate around z axis
ax.elev = 30 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)

plt.show()


# In[40]:


embedding_mean_pertask_perstate=np.asarray([embedding_mean[state_ind*90:(state_ind+1)*90]                                            for state_ind in range(int(num_states))])
state_ind_X,state_ind_Y=0,1
distances_all=[]
for state_ind_X in np.arange(int(num_states)):
    for state_ind_Y in np.arange(int(num_states)):
        if state_ind_X!=state_ind_Y:
            distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate[state_ind_X]                                        -embedding_mean_pertask_perstate[state_ind_Y],axis=1)
            distances_all.append(distance_XY_)
distance_mean=np.nanmean(distances_all)


# In[41]:


ephys_mean_z_dic[mouse_recday]
ephys_mean_z_pertask_dic[mouse_recday]


# In[132]:


##UMAP embedding

reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
distance_mean_abstract_perday_all=[]
for mouse_recday in ephys_mean_z_dic.keys():
    ephys_mean_z_pertask_day=ephys_mean_z_dic[mouse_recday]
    embedding_mean_pertask_perday = reducer.fit_transform(ephys_mean_z_pertask_day.T)
    
    embedding_mean_pertask_perstate_perday=np.asarray([embedding_mean_pertask_perday[state_ind*90:(state_ind+1)*90]                                            for state_ind in range(int(num_states))])
    distances_all_perday=[]
    for state_ind_X in np.arange(int(num_states)):
        for state_ind_Y in np.arange(int(num_states)):
            if state_ind_X!=state_ind_Y:
                distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate_perday[state_ind_X]                                            -embedding_mean_pertask_perstate_perday[state_ind_Y],axis=1)
                distances_all_perday.append(distance_XY_)
    distance_mean_perday=np.nanmean(distances_all_perday)
    distance_mean_abstract_perday_all.append(distance_mean_perday)
distance_mean_abstract_perday_all=np.vstack((distance_mean_abstract_perday_all))


# In[133]:


neuron_thr=5
neuron_number_bool=np.repeat(False,len(ephys_mean_z_dic.keys()))
for ind_day, mouse_recday in enumerate(list(ephys_mean_z_dic.keys())):
    print(mouse_recday)
    print(len(ephys_mean_z_dic[mouse_recday]))
    if len(ephys_mean_z_dic[mouse_recday])>neuron_thr:
        neuron_number_bool[ind_day]=True


# In[134]:


len(ephys_mean_z_dic[mouse_recday])


# In[135]:


neuron_number_bool


# In[ ]:





# In[ ]:





# In[140]:


##UMAP embedding
reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
embedding_mean_pertask = reducer.fit_transform(ephys_mean_z_pertask_all.T)

c1_='yellow'
c2_='purple' 
n=len(embedding_mean)

colors_state=[]
for x in np.arange(n):
    colors_state.append(colorFader(c1_,c2_,x/n))

    
c1_='lightgreen'
c2_='black' 
colors_phase=[]
num_states=(n/90)
for state in np.arange(num_states):
    for x in np.arange(90):
        colors_phase.append(colorFader(c1_,c2_,x/90))

        


# In[145]:


plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#embedding_mean=np.load(Intermediate_object_folder+'Embedding_example_ABCD_08042024_1321.npy')

x = embedding_mean_pertask[:,0]
y = embedding_mean_pertask[:,1]
z = embedding_mean_pertask[:,2]

ax.scatter(x, y, z, c=colors_state, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 120 ## rotate around z axis
ax.elev = 40 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)
plt.savefig(Ephys_output_folder_dropbox+'UMAP_withintask_state.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[146]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#embedding_mean=np.load(Intermediate_object_folder+'Embedding_example_ABCD_08042024_1321.npy')

x = embedding_mean_pertask[:,0]
y = embedding_mean_pertask[:,1]
z = embedding_mean_pertask[:,2]

ax.scatter(x, y, z, c=colors_phase, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 120 ## rotate around z axis
ax.elev = 40 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)
plt.savefig(Ephys_output_folder_dropbox+'UMAP_withintask_phase.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()



# In[56]:


###stupid way of making cmap - unstupidify it

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#embedding_mean=np.load(Intermediate_object_folder+'Embedding_example_ABCD_08042024_1321.npy')

x = embedding_mean_pertask[:,0]
y = embedding_mean_pertask[:,1]
z = embedding_mean_pertask[:,2]

ax.scatter(x, y, z, c=colors_phase, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 30 ## rotate around z axis
ax.elev = 40 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)


cmap_name = 'my_list'
cmapX=LinearSegmentedColormap.from_list(cmap_name,colors_phase[:90])

x = np.arange(0, np.pi, 0.1)
y = np.arange(0, 2 * np.pi, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.cos(X) * np.sin(Y) * 10

im = ax.imshow(Z, interpolation='nearest', origin='lower', cmap=cmapX)

#im = ax.scatter(x, y, z, c=cmapX, marker='o',s=100)

fig.colorbar(im, ax=ax)
plt.savefig(Ephys_output_folder_dropbox+'UMAP_phase_cmap.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[147]:


embedding_mean_pertask_perstate=np.asarray([embedding_mean_pertask[state_ind*90:(state_ind+1)*90]                                            for state_ind in range(int(num_states))])
distances_all=[]
for state_ind_X in np.arange(int(num_states)):
    for state_ind_Y in np.arange(int(num_states)):
        if state_ind_X!=state_ind_Y:
            distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate[state_ind_X]                                        -embedding_mean_pertask_perstate[state_ind_Y],axis=1)
            distances_all.append(distance_XY_)
distance_mean=np.nanmean(distances_all)


# In[148]:


distance_mean


# In[117]:


tt=time.time()

num_iterations=len(ephys_mean_z_shuff_all_alliterations)
distance_mean_shuff_all=np.zeros(num_iterations)
distance_mean_shuff_all[:]=np.nan
for iteration in range(num_iterations):
    print(iteration)
    
    ##UMAP embedding
    ephys_mean_z_shuff_iteration=ephys_mean_z_shuff_all_alliterations[iteration]
    reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
    embedding_mean_pertask_shuff = reducer.fit_transform(ephys_mean_z_shuff_iteration.T)
    
    embedding_mean_pertask_perstate_shuff=np.asarray([embedding_mean_pertask_shuff[state_ind*90:(state_ind+1)*90]                                            for state_ind in range(int(num_states))])
    
    distances_all_iteration=[]
    for state_ind_X in np.arange(int(num_states)):
        for state_ind_Y in np.arange(int(num_states)):
            if state_ind_X!=state_ind_Y:
                distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate_shuff[state_ind_X]                                            -embedding_mean_pertask_perstate_shuff[state_ind_Y],axis=1)
                distances_all_iteration.append(distance_XY_)
    distance_mean_iteration=np.nanmean(distances_all_iteration)
    distance_mean_shuff_all[iteration]=distance_mean_iteration
    
print(time.time()-tt)


# In[104]:


plt.hist(distance_mean_shuff_all,bins=40,color='grey')
plt.axvline(distance_mean,color='black')
plt.show()


# In[669]:


list(ephys_mean_z_dic.keys())


# In[149]:


##UMAP embedding - per task distances between states within and between goal-progress

reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
distance_mean_perday_all=[]
distance_mean_perday_crossphase_all=[]

distance_all_perday_all=[]
distance_all_perday_crossphase_all=[]
for mouse_recday in ephys_mean_z_dic.keys():
    print(mouse_recday)
    ephys_mean_z_pertask_day=ephys_mean_z_pertask_dic[mouse_recday]
    if len(ephys_mean_z_pertask_day)==0:
        print('not used')
        continue
    embedding_mean_pertask_perday = reducer.fit_transform(ephys_mean_z_pertask_day.T)
    
    embedding_mean_pertask_perstate_perday=np.asarray([embedding_mean_pertask_perday[state_ind*90:(state_ind+1)*90]                                            for state_ind in range(int(num_states))])
    distances_all_perday=[]
    distances_all_perday_crossphase=[]
    for state_ind_X in np.arange(int(num_states)):
        for state_ind_Y in np.arange(int(num_states)):
            if state_ind_X!=state_ind_Y:
                distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate_perday[state_ind_X]                                            -embedding_mean_pertask_perstate_perday[state_ind_Y],axis=1)
                
                
                distance_XY_crossphase_=np.linalg.norm(embedding_mean_pertask_perstate_perday[state_ind_X]                        -np.roll(embedding_mean_pertask_perstate_perday[state_ind_Y],45,axis=0),axis=1)

                
                distances_all_perday.append(distance_XY_)
                distances_all_perday_crossphase.append(distance_XY_crossphase_)
    distance_mean_perday=np.nanmean(distances_all_perday)
    distance_all_perday_all.append(distances_all_perday)
    distance_mean_perday_all.append(distance_mean_perday)
    
    distance_mean_perday_crossphase=np.nanmean(distances_all_perday_crossphase)
    distance_all_perday_crossphase_all.append(distances_all_perday_crossphase)
    distance_mean_perday_crossphase_all.append(distance_mean_perday_crossphase)
    
distance_mean_perday_all=np.vstack((distance_mean_perday_all))
distance_mean_perday_crossphase_all=np.vstack((distance_mean_perday_crossphase_all))

#distance_all_perday_all=np.vstack((distance_all_perday_all))
#distance_all_perday_crossphase_all=np.vstack((distance_all_perday_crossphase_all))


# In[150]:


distance_all_perday_all_timeline=np.nanmean(distance_all_perday_all,axis=1)
distance_all_perday_crossphase_all_timeline=np.nanmean(distance_all_perday_crossphase_all,axis=1)

diff_withinvscrossphase=distance_all_perday_crossphase_all_timeline-distance_all_perday_all_timeline


# In[151]:


noplot_timecourseBx(np.arange((np.shape(diff_withinvscrossphase)[1])),diff_withinvscrossphase.T, 'black')
plt.axhline(0,ls='dashed',color='black')
plt.show()


# In[108]:





# In[152]:


##UMAP embedding - shuffled data - per task distances between states within and between goal-progress
tt=time.time()

num_iterations_=5
distance_mean_shuff_all=np.zeros(len(list(ephys_mean_z_shuff_dic[0].keys())))
distance_mean_shuff_all[:]=np.nan



for day_ind, mouse_recday in enumerate(list(ephys_mean_z_shuff_dic[0].keys())):
    print(mouse_recday)
    distance_mean_iteration_day=[]
    for iteration in range(num_iterations_):
        print(iteration)


        ##UMAP embedding
        ephys_mean_z_shuff_iteration=ephys_mean_z_shuff_dic[iteration][mouse_recday]
        reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
        if len(ephys_mean_z_shuff_iteration)==0:
            continue
        embedding_mean_pertask_shuff = reducer.fit_transform(ephys_mean_z_shuff_iteration.T)

        embedding_mean_pertask_perstate_shuff=np.asarray([embedding_mean_pertask_shuff[state_ind*90:(state_ind+1)*90]                                                for state_ind in range(int(num_states))])

        distances_all_iteration=[]
        for state_ind_X in np.arange(int(num_states)):
            for state_ind_Y in np.arange(int(num_states)):
                if state_ind_X!=state_ind_Y:
                    distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate_shuff[state_ind_X]                                                -embedding_mean_pertask_perstate_shuff[state_ind_Y],axis=1)
                    distances_all_iteration.append(distance_XY_)
        distance_mean_iteration=np.nanmean(distances_all_iteration)
        distance_mean_iteration_day.append(distance_mean_iteration)
    distance_mean_shuff_all[day_ind]=np.nanmean(distance_mean_iteration_day)
    
print(time.time()-tt)


# In[122]:


len(ephys_mean_z_shuff_iteration)


# In[ ]:





# In[ ]:





# num_states=4
# ###making per state input
# ephys_mean_z_pertask_all_shuff_=ephys_mean_z_pertask_all.reshape(np.shape(ephys_mean_z_pertask_all)[0],\
#                                  int(num_states),int(np.shape(ephys_mean_z_pertask_all)[1]//num_states))
# 
# ###shuffles
# num_iterations=100
# distance_mean_shuff_all=np.zeros(num_iterations)
# distance_mean_shuff_all[:]=np.nan
# for iteration in range(num_iterations):
#     ephys_mean_z_pertask_all_shuff=np.asarray([np.hstack((ephys_mean_z_pertask_all_shuff_\
#                                                           [neuron_task,random.sample(list(np.arange(num_states)),\
#                                                                                      num_states)]))\
#     for neuron_task in np.arange(np.shape(ephys_mean_z_pertask_all)[0])])
#     
#     ##UMAP embedding
#     reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
#     embedding_mean_pertask_shuff = reducer.fit_transform(ephys_mean_z_pertask_all_shuff.T)
#     
#     embedding_mean_pertask_perstate_shuff=np.asarray([embedding_mean_pertask_shuff[state_ind*90:(state_ind+1)*90]\
#                                             for state_ind in range(int(num_states))])
#     distances_all_shuff=[]
#     for state_ind_X in np.arange(int(num_states)):
#         for state_ind_Y in np.arange(int(num_states)):
#             if state_ind_X!=state_ind_Y:
#                 distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate_shuff[state_ind_X]\
#                                             -embedding_mean_pertask_perstate_shuff[state_ind_Y],axis=1)
#                 distances_all_shuff.append(distance_XY_)
#     distance_mean_shuff=np.nanmean(distances_all_shuff)
#     distance_mean_shuff_all[iteration]=distance_mean_shuff
# 

# plt.hist(distance_mean_shuff_all,bins=40)
# plt.axvline(distance_mean,color='black')
# plt.show()

# In[ ]:





# In[153]:


##Within task between state - ABCD
distance_mean_shuff_all_=[]
#neuron_number_bool_=[]
for day_ind, mouse_recday in enumerate(list(ephys_mean_z_shuff_dic[0].keys())):
    if len(ephys_mean_z_dic[mouse_recday])>0:
        distance_mean_shuff_all_.append(distance_mean_shuff_all[day_ind])
#        neuron_number_bool_.append(neuron_number_bool[day_ind])
        
        
distance_mean_shuff_all_=np.asarray(distance_mean_shuff_all_)
#neuron_number_bool_=np.asarray(neuron_number_bool_)

distances_=np.column_stack((distance_mean_perday_crossphase_all.squeeze(),distance_mean_perday_all.squeeze(),                              distance_mean_shuff_all_))
#distances_=distances_[neuron_number_bool_==True]

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

bar_plotX(distances_.T,'none',0,6.2,'points','paired',0.025)

plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'UMAP_withintask_vs_shuffle_mean_statedistance.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(st.ttest_rel(distances_[:,0],distances_[:,1]))
print(st.ttest_rel(distances_[:,0],distances_[:,2]))
print(st.ttest_rel(distances_[:,1],distances_[:,2]))

from statsmodels.sandbox.stats.multicomp import multipletests
ttests_ps= [st.ttest_rel(distances_[:,0],distances_[:,1])[1], st.ttest_rel(distances_[:,0],distances_[:,2])[1],st.ttest_rel(distances_[:,1],distances_[:,2])[1]]

print(multipletests(ttests_ps,alpha=0.05,method='bonferroni'))


# In[68]:


np.shape(distance_mean_perday_crossphase_all)


# In[575]:


test_ses_ind=0
split_activity_trainingtest_phasemean_all=[]
split_activity_training_phasemean_all=[]

within_phase_distance_all=[]
between_phase_distance_all=[]
for mouse_recday in ephys_mean_z_split_dic[test_ses_ind].keys():
    split_activity=ephys_mean_z_split_dic[test_ses_ind][mouse_recday]
    split_activity_reshaped=split_activity.reshape(len(split_activity),2,len(split_activity.T)//2)

    split_activity_training_phasemean=np.nanmean(split_activity_reshaped[:,0].        reshape(len(split_activity_reshaped),4,np.shape(split_activity_reshaped)[2]//4),axis=1)

    split_activity_test_phasemean=np.nanmean(split_activity_reshaped[:,1].        reshape(len(split_activity_reshaped),4,np.shape(split_activity_reshaped)[2]//4),axis=1)

    split_activity_trainingtest_phasemean=np.hstack((split_activity_training_phasemean,split_activity_test_phasemean))
    ##removing neurons with nans (think about interpolating)
    split_activity_trainingtest_phasemean=    split_activity_trainingtest_phasemean[~np.isnan(np.mean(split_activity_trainingtest_phasemean,axis=1))]

    split_activity_training_phasemean=split_activity_training_phasemean    [~np.isnan(np.mean(split_activity_training_phasemean,axis=1))]
    
    split_activity_trainingtest_phasemean_all.append(split_activity_trainingtest_phasemean)
    
    split_activity_training_phasemean_all.append(split_activity_training_phasemean)
    
    ##UMAP per day
    reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
    embedding_mean_phase_trainingtest_day = reducer.fit_transform(split_activity_trainingtest_phasemean.T)
    
    ###distances
    distance_XY_withinphase_=np.linalg.norm(embedding_mean_phase_trainingtest_day[:90]                                                -embedding_mean_phase_trainingtest_day[90:],axis=1)
    
    distance_XY_between_phase_=np.linalg.norm(embedding_mean_phase_trainingtest_day[:90]                                                -np.roll(embedding_mean_phase_trainingtest_day[90:],22,axis=0),axis=1)
    
    within_phase_distance_all.append(np.nanmean(distance_XY_withinphase_))
    between_phase_distance_all.append(np.nanmean(distance_XY_between_phase_))
    
split_activity_trainingtest_phasemean_all=np.vstack((split_activity_trainingtest_phasemean_all))
split_activity_training_phasemean_all=np.vstack((split_activity_training_phasemean_all))

within_phase_distance_all=np.hstack((within_phase_distance_all))
between_phase_distance_all=np.hstack((between_phase_distance_all))


# In[579]:


np.shape(np.roll(embedding_mean_phase_trainingtest_day[90:],22,axis=0))


# In[577]:


np.nanmean(between_phase_distance_all)


# In[578]:


within_between_distance_all=np.vstack((within_phase_distance_all,between_phase_distance_all))
bar_plotX(within_between_distance_all,'none',0,40,'points','paired',0.025)


# In[ ]:





# In[582]:


reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
embedding_mean_phase_trainingtest = reducer.fit_transform(split_activity_trainingtest_phasemean_all.T)

c1_='lightgreen'
c2_='black' 
colors_phase=[]
for x in np.arange(90):
    colors_phase.append(colorFader(c1_,c2_,x/90))

c1_='lightblue'
c2_='black' 
colors_phase2=[]
for x in np.arange(90):
    colors_phase2.append(colorFader(c1_,c2_,x/90))



colors_phase_all=np.hstack((colors_phase,colors_phase2))

#embedding_mean=np.load(Intermediate_object_folder+'Embedding_example_ABCD_08042024_1321.npy')


# In[ ]:





# In[583]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = embedding_mean_phase_trainingtest[:,0]
y = embedding_mean_phase_trainingtest[:,1]
z = embedding_mean_phase_trainingtest[:,2]

ax.scatter(x, y, z, c=colors_phase_all, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 30 ## rotate around z axis
ax.elev = 60 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)
plt.savefig(Ephys_output_folder_dropbox+'UMAP_acrosstask_phase.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[ ]:





# In[16]:


'''num_bins=np.shape(ephys_mean_z_all)[1]
random_bins=random.sample(list(np.arange(num_bins)), num_bins)

ephys_mean_z_all_jumbled=ephys_mean_z_all[:,random_bins]

##UMAP embedding
reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
embedding_mean_jumbled = reducer.fit_transform(ephys_mean_z_all_jumbled.T)

embedding_mean=np.zeros((np.shape(embedding_mean_jumbled)[0],np.shape(embedding_mean_jumbled)[1]))
embedding_mean[:]=np.nan

for bin_ind in range(num_bins):
    embedding_mean[random_bins[bin_ind]]=embedding_mean_jumbled[bin_ind]
#embedding_mean=embedding_mean_jumbled'''


# In[ ]:





# In[ ]:





# In[17]:


##UMAP embedding - shuffled data
reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
embedding_mean_control = reducer.fit_transform(ephys_mean_z_shuff_all.T)

c1_='yellow'
c2_='purple' 
n=len(embedding_mean_control)

colors_state=[]
for x in np.arange(n):
    colors_state.append(colorFader(c1_,c2_,x/n))

    
c1_='lightgreen'
c2_='black' 
colors_phase=[]
num_states=(n/90)
for state in np.arange(num_states):
    for x in np.arange(90):
        colors_phase.append(colorFader(c1_,c2_,x/90))


# In[18]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = embedding_mean_control[:,0]
y = embedding_mean_control[:,1]
z = embedding_mean_control[:,2]

ax.scatter(x, y, z, c=colors_state, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#ax.azim = 130
#ax.elev = 150

ax.azim = 30 ## rotate around z axis
ax.elev = 30 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)

plt.show()


# In[19]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = embedding_mean_control[:,0]
y = embedding_mean_control[:,1]
z = embedding_mean_control[:,2]

ax.scatter(x, y, z, c=colors_phase, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 30 ## rotate around z axis
ax.elev = 90 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)

plt.show()


# In[ ]:





# In[82]:


embedding_mean_pertask_perstate=np.asarray([embedding_mean_control[state_ind*90:(state_ind+1)*90]                                            for state_ind in range(int(num_states))])
state_ind_X,state_ind_Y=0,1
distances_all=[]
for state_ind_X in np.arange(int(num_states)):
    for state_ind_Y in np.arange(int(num_states)):
        if state_ind_X!=state_ind_Y:
            distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate[state_ind_X]                                        -embedding_mean_pertask_perstate[state_ind_Y],axis=1)
            distances_all.append(distance_XY_)
distance_mean=np.nanmean(distances_all)


# In[83]:


distance_mean


# In[ ]:





# In[ ]:





# In[ ]:


#######ABCDE#########


# In[ ]:





# In[19]:


recording_days_


# In[22]:


for mouse_recday in recording_days_:
    print(non_repeat_ses_maker_old(mouse_recday))


# In[23]:


for mouse_recday in recording_days_:
    print(non_repeat_ses_maker(mouse_recday))


# In[ ]:





# In[4]:


#UMAP ABCDE
abstract_structure='ABCDE'
num_states=len(abstract_structure)
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDE_days.npy')

ephys_mean_z_ABCDE_pertask_dic=rec_dd()
ephys_mean_z_ABCDE_all=[]
ephys_mean_z_ABCDE_pertask_all=[]

for mouse_recday in recording_days_:
    
    print(mouse_recday)
    
    try:
        awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
        awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
        num_sessions=len(awake_sessions_behaviour)
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        found_ses=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            try:
                Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                found_ses.append(ses_ind)

            except:
                print('Files not found for session '+str(ses_ind))
                continue
        num_non_repeat_ses_found=len(found_ses)

        found_ses_nonrepeat=np.intersect1d(found_ses,non_repeat_ses)

        len_trials_all=[]
        for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
            ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                           ,allow_pickle=True)
            if np.shape(ephys_)[2]==450:
                len_trials_all.append(np.shape(ephys_)[1])

        min_len=min(len_trials_all)

        ephys_allses=[]
        ephys_allses_meanz=[]
        for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
            ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                           ,allow_pickle=True)

            if np.shape(ephys_)[2]==450:
                #ephys_allses.append(ephys_[:,-min_len:])
                
                ephys_mean=np.nanmean(ephys_,axis=1)
                ephys_allses.append(ephys_mean)
                
                ephys_ses_means=np.nanmean(ephys_,axis=1)
                ephys_ses_meansz=st.zscore(ephys_ses_means,axis=1)

                ephys_allses_meanz.append(ephys_ses_meansz)
                

        #ephys_allses=np.hstack((ephys_allses))
        ephys_mean=np.nanmean(ephys_allses,axis=0) ###taking mean across sessions
        ephys_mean_z=st.zscore(ephys_mean,axis=1)
        ephys_mean_z_ABCDE_all.append(ephys_mean_z)
        
        ephys_allses_meanz=np.vstack((ephys_allses_meanz))
        ephys_mean_z_ABCDE_pertask_all.append(ephys_allses_meanz)
        
        ephys_mean_z_ABCDE_pertask_dic[mouse_recday]=ephys_allses_meanz
    except:
        print('Not used')
ephys_mean_z_ABCDE_all=np.vstack((ephys_mean_z_ABCDE_all))
ephys_mean_z_ABCDE_pertask_all=np.vstack((ephys_mean_z_ABCDE_pertask_all))


# In[ ]:





# In[5]:


##UMAP on control (shuffling states) - ABCDE
tt=time.time()

num_iterations=100
abstract_structure='ABCDE'
num_states=len(abstract_structure)

ephys_mean_z_shuff_ABCDE_dic=rec_dd()
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDE_days.npy')
ephys_mean_z_shuff_ABCDE_all_alliterations=[]
for iteration in np.arange(num_iterations):
    print(iteration)
    ephys_mean_z_shuff_ABCDE_all=[]
    for mouse_recday in recording_days_:
        #print(mouse_recday)
        #try:
        awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
        awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
        num_sessions=len(awake_sessions_behaviour)
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        found_ses=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            try:
                Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                found_ses.append(ses_ind)

            except:
                #print('Files not found for session '+str(ses_ind))
                continue


        found_ses_nonrepeat=np.intersect1d(found_ses,non_repeat_ses)

        if len(found_ses_nonrepeat)<6:
            #print('Not enough tasks')
            continue

        len_trials_all=[]
        for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
            ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                           ,allow_pickle=True)
            len_trials_all.append(np.shape(ephys_)[1])

        min_len=min(len_trials_all)
        ephys_allses=[]
        for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
            ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                           ,allow_pickle=True)
            
            if np.shape(ephys_)[2]==450:

                ephys_shuff_=np.asarray(([np.vstack(([np.hstack((ephys_[neuron,trial].reshape(num_states,90)                                  [random.sample(list(np.arange(num_states)), num_states)]))                    for trial in np.arange(np.shape(ephys_)[1])]))                                              for neuron in np.arange(np.shape(ephys_)[0])]))

                #ephys_allses.append(ephys_shuff_[:,-min_len:])

                ephys_mean=np.nanmean(ephys_shuff_,axis=1)
                ephys_allses.append(ephys_mean)

        #ephys_allses=np.hstack((ephys_allses))
        ephys_mean=np.nanmean(ephys_allses,axis=0) ###taking mean across sessions
        ephys_mean_z=st.zscore(ephys_mean,axis=1)
        ephys_mean_z_shuff_ABCDE_all.append(ephys_mean_z)
        ephys_mean_z_shuff_ABCDE_dic[iteration][mouse_recday]=ephys_mean_z
        #except:
        #    print('Not used')
    ephys_mean_z_shuff_ABCDE_all=np.vstack((ephys_mean_z_shuff_ABCDE_all))
    ephys_mean_z_shuff_ABCDE_all_alliterations.append(ephys_mean_z_shuff_ABCDE_all)
print(time.time()-tt)


# In[159]:


num_neurons_all=0
ses_ind=0
for mouse_recday in ephys_mean_z_ABCDE_pertask_dic.keys():
    Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
    num_neurons_all+=len(Neuron_raw)
    
print(len(ephys_mean_z_ABCDE_pertask_dic.keys()))
print(num_neurons_all)


# In[158]:


np.shape(ephys_mean_z_ABCDE_pertask_dic[mouse_recday])


# In[ ]:





# In[7]:


##UMAP embedding
reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
embedding_ABCDE_mean = reducer.fit_transform(ephys_mean_z_ABCDE_all.T)

c1_='yellow'
c2_='purple' 
n=len(embedding_ABCDE_mean)

colors_state=[]
for x in np.arange(n):
    colors_state.append(colorFader(c1_,c2_,x/n))

    
c1_='lightgreen'
c2_='black' 
colors_phase=[]
num_states=(n/90)
for state in np.arange(num_states):
    for x in np.arange(90):
        colors_phase.append(colorFader(c1_,c2_,x/90))


# In[ ]:





# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = embedding_ABCDE_mean[:,0]
y = embedding_ABCDE_mean[:,1]
z = embedding_ABCDE_mean[:,2]

ax.scatter(x, y, z, c=colors_state, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 140 ## rotate around z axis
ax.elev = 30 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)

plt.show()


# In[ ]:





# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = embedding_ABCDE_mean[:,0]
y = embedding_ABCDE_mean[:,1]
z = embedding_ABCDE_mean[:,2]

ax.scatter(x, y, z, c=colors_phase, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.azim = 140 ## rotate around z axis
ax.elev = 30 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)
plt.show()


# In[ ]:


n


# In[10]:


##UMAP embedding
reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
embedding_ABCDE_all = reducer.fit_transform(ephys_mean_z_ABCDE_pertask_all.T)

c1_='yellow'
c2_='purple' 
n=len(embedding_ABCDE_all)

colors_state=[]
for x in np.arange(n):
    colors_state.append(colorFader(c1_,c2_,x/n))

    
c1_='lightgreen'
c2_='black' 
colors_phase=[]
num_states=(n/90)
for state in np.arange(num_states):
    for x in np.arange(90):
        colors_phase.append(colorFader(c1_,c2_,x/90))


# In[28]:


plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = embedding_ABCDE_all[:,0]
y = embedding_ABCDE_all[:,1]
z = embedding_ABCDE_all[:,2]

ax.scatter(x, y, z, c=colors_state, marker='o',s=100)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.azim = 60 ## rotate around z axis
ax.elev = 20 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)
plt.savefig(Ephys_output_folder_dropbox+'UMAP_withintask_ABCDE_state.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[29]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = embedding_ABCDE_all[:,0]
y = embedding_ABCDE_all[:,1]
z = embedding_ABCDE_all[:,2]

ax.scatter(x, y, z, c=colors_phase, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 60 ## rotate around z axis
ax.elev = 20 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)
plt.savefig(Ephys_output_folder_dropbox+'UMAP_withintask_ABCDE_phase.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[ ]:





# In[16]:


##UMAP embedding - per task distances between states within and between goal-progress

reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
distance_mean_perday_ABCDE_all=[]
distance_mean_perday_ABCDE_crossphase_all=[]
for mouse_recday in ephys_mean_z_ABCDE_pertask_dic.keys():
    ephys_mean_z_pertask_day=ephys_mean_z_ABCDE_pertask_dic[mouse_recday]
    embedding_mean_pertask_perday = reducer.fit_transform(ephys_mean_z_pertask_day.T)
    
    embedding_mean_pertask_perstate_perday=np.asarray([embedding_mean_pertask_perday[state_ind*90:(state_ind+1)*90]                                            for state_ind in range(int(num_states))])
    distances_all_perday=[]
    distances_all_perday_crossphase=[]
    for state_ind_X in np.arange(int(num_states)):
        for state_ind_Y in np.arange(int(num_states)):
            if state_ind_X!=state_ind_Y:
                distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate_perday[state_ind_X]                                            -embedding_mean_pertask_perstate_perday[state_ind_Y],axis=1)
                
                
                distance_XY_crossphase_=np.linalg.norm(embedding_mean_pertask_perstate_perday[state_ind_X]                        -np.roll(embedding_mean_pertask_perstate_perday[state_ind_Y],45,axis=0),axis=1)

                
                distances_all_perday.append(distance_XY_)
                distances_all_perday_crossphase.append(distance_XY_crossphase_)
    distance_mean_perday=np.nanmean(distances_all_perday)
    distance_mean_perday_ABCDE_all.append(distance_mean_perday)
    
    distance_mean_perday_crossphase=np.nanmean(distances_all_perday_crossphase)
    distance_mean_perday_ABCDE_crossphase_all.append(distance_mean_perday_crossphase)
    
distance_mean_perday_ABCDE_all=np.vstack((distance_mean_perday_ABCDE_all))
distance_mean_perday_ABCDE_crossphase_all=np.vstack((distance_mean_perday_ABCDE_crossphase_all))


# In[ ]:





# In[17]:


tt=time.time()

num_iterations_=5
distance_mean_shuff_ABCDE_all=np.zeros(len(list(ephys_mean_z_shuff_ABCDE_dic[0].keys())))
distance_mean_shuff_ABCDE_all[:]=np.nan

for day_ind, mouse_recday in enumerate(list(ephys_mean_z_shuff_ABCDE_dic[0].keys())):
    print(mouse_recday)
    distance_mean_iteration_day=[]
    for iteration in range(num_iterations_):
        print(iteration)


        ##UMAP embedding
        ephys_mean_z_shuff_iteration=ephys_mean_z_shuff_ABCDE_dic[iteration][mouse_recday]
        reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
        embedding_mean_pertask_shuff = reducer.fit_transform(ephys_mean_z_shuff_iteration.T)

        embedding_mean_pertask_perstate_shuff=np.asarray([embedding_mean_pertask_shuff[state_ind*90:(state_ind+1)*90]                                                for state_ind in range(int(num_states))])

        distances_all_iteration=[]
        for state_ind_X in np.arange(int(num_states)):
            for state_ind_Y in np.arange(int(num_states)):
                if state_ind_X!=state_ind_Y:
                    distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate_shuff[state_ind_X]                                                -embedding_mean_pertask_perstate_shuff[state_ind_Y],axis=1)
                    distances_all_iteration.append(distance_XY_)
        distance_mean_iteration=np.nanmean(distances_all_iteration)
        distance_mean_iteration_day.append(distance_mean_iteration)
    distance_mean_shuff_ABCDE_all[day_ind]=np.nanmean(distance_mean_iteration_day)
    
print(time.time()-tt)


# In[ ]:





# In[18]:


##Within task between state - ABCDE 
#distance_mean_shuff_all_=np.asarray(distance_mean_shuff_all_)
#neuron_number_bool_=np.asarray(neuron_number_bool_)
from statsmodels.sandbox.stats.multicomp import multipletests
distances_=np.column_stack((distance_mean_perday_ABCDE_crossphase_all.squeeze(),distance_mean_perday_ABCDE_all.squeeze(),                              distance_mean_shuff_ABCDE_all))
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
#distances_=distances_[neuron_number_bool_==True]
bar_plotX(distances_.T,'none',0,5.2,'points','paired',0.025)

plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'UMAP_withintask_vs_shuffle_ABCDE_mean_statedistance.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(st.ttest_rel(distances_[:,0],distances_[:,1]))
print(st.ttest_rel(distances_[:,0],distances_[:,2]))
print(st.ttest_rel(distances_[:,1],distances_[:,2]))


ttests_ps= [st.ttest_rel(distances_[:,0],distances_[:,1])[1], st.ttest_rel(distances_[:,0],distances_[:,2])[1],st.ttest_rel(distances_[:,1],distances_[:,2])[1]]

print(multipletests(ttests_ps,alpha=0.05,method='bonferroni'))


# In[94]:


ttests_ps


# In[ ]:





# In[ ]:





# In[27]:


'''
Aligning by anchors

1-for each cell find peak in anchoring GLM on all but the current task
2-find all visits to anchor point
3-trigger cell's activity on current task to anchor visit times
4-Average activity per cell per session
5-zscore across session means and recompute umap



'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[335]:


#UMAP on ABCD data  - aligned by anchor
abstract_structure='ABCD'
num_states=len(abstract_structure)
neuron_thr=10
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')
ephys_mean_z_Anchoraligned_all=[]

num_phases,num_nodes,num_lags,num_states=3,9,12,4
ephys_mean_z_aligned_dic=rec_dd()

ephys_mean_z_aligned_split_dic=rec_dd()

for mouse_recday in recording_days_:
    print(mouse_recday)
    #try:
    awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
    awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
    
    num_sessions=len(awake_sessions_behaviour)
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    found_ses=[]
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            found_ses.append(ses_ind)

        except:
            print('Files not found for session '+str(ses_ind))
            continue


    found_ses_nonrepeat=np.intersect1d(found_ses,non_repeat_ses)

    if len(found_ses_nonrepeat)<6:
        print('Not enough tasks')
        continue
    
    ###finding each neuron's anchor in each task
    coeffs=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_coeffs_all_'+mouse_recday+'.npy')
    indices_regressors=np.reshape(np.arange(np.shape(coeffs)[2]),(num_nodes,num_phases,num_lags))
    max_coeffs=np.argmax(coeffs,axis=2)
    anchors_neurons_ses=np.asarray([[np.hstack(((np.where(indices_regressors==max_coeffs[neuron,ses_ind_ind]))))
    for ses_ind_ind in range(len(max_coeffs.T))] for neuron in range(len(max_coeffs))])
    
    #print(np.sum(np.vstack((anchors_neurons_ses))[:,1]==0)/len(np.vstack((anchors_neurons_ses))[:,1]))

    ##finding times of anchor visits
    
    len_trials_all=[]
    for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
        ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                       ,allow_pickle=True)
        len_trials_all.append(np.shape(ephys_)[1])

    min_len=min(len_trials_all)
    ephys_allses=[]
    ephys_allses_meanz=[]

    for ses_ind_ind, ses_ind in enumerate(found_ses_nonrepeat):
        ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                       ,allow_pickle=True)
        location_=np.load(Intermediate_object_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy'                                                   ,allow_pickle=True)-1

        phases=np.tile(np.tile(np.repeat(np.arange(num_phases),90/num_phases),num_states),len(location_)).        reshape(np.shape(location_)[0],np.shape(location_)[1])

        alignment_times_ses=np.hstack(([np.argmax(np.nanmean(np.logical_and(location_==                                                                            anchors_neurons_ses[neuron,ses_ind_ind][0],                               phases==anchors_neurons_ses[neuron,ses_ind_ind][1]),axis=0))                for neuron in range(len(max_coeffs))]))

        ephys_mean=np.nanmean(ephys_,axis=1)
        ephys_mean_aligned=indep_roll(ephys_mean,-alignment_times_ses)

        ephys_allses.append(ephys_mean_aligned)

    #print(np.shape(ephys_allses))
    #ephys_allses=np.hstack((ephys_allses))
    ephys_mean=np.nanmean(ephys_allses,axis=0) ###taking mean of means across sessions
    ephys_mean_z=st.zscore(ephys_mean,axis=1)
    ephys_mean_z_Anchoraligned_all.append(ephys_mean_z)
    
    if len(ephys_)>neuron_thr:
        print('used for per day dictionary')
        ephys_mean_z_aligned_dic[mouse_recday]=ephys_mean_z
    
        num_ses=len(ephys_allses)
        merged_ephys_all=np.zeros((num_ses,np.shape(ephys_allses)[1],np.shape(ephys_allses)[2]*2))
        merged_ephys_all[:]=np.nan
        for test_ses_ind in np.arange(num_ses):
            training_ses=np.setdiff1d(np.arange(num_ses),test_ses_ind)
            ephys_training_ses_mean=np.nanmean(np.asarray(ephys_allses)[training_ses],axis=0)
            ephys_test_ses_mean=ephys_allses[test_ses_ind]
            merged_ephys=np.hstack((ephys_training_ses_mean,ephys_test_ses_mean))
            merged_ephys_all[test_ses_ind]=merged_ephys
        merged_ephys_all[np.isnan(merged_ephys_all)]=0
        merged_ephys_all_z=st.zscore(merged_ephys_all,axis=0)

        for test_ses_ind in np.arange(num_ses): 
            ephys_mean_z_aligned_split_dic[test_ses_ind][mouse_recday]=merged_ephys_all_z[test_ses_ind]


    #except:
    #    print('Not used')
ephys_mean_z_Anchoraligned_all=np.vstack((ephys_mean_z_Anchoraligned_all))


# In[350]:


np.shape(ephys_allses)


# In[ ]:





# In[ ]:





# In[29]:


##UMAP embedding
reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
embedding_mean_aligned = reducer.fit_transform(ephys_mean_z_Anchoraligned_all.T)

c1_='yellow'
c2_='purple' 
n=len(embedding_mean_aligned)
colors_state=[]
for x in np.arange(n):
    colors_state.append(colorFader(c1_,c2_,x/n))

    
c1_='lightgreen'
c2_='black' 
colors_phase=[]
num_states=(n/90)
for state in np.arange(num_states):
    for x in np.arange(90):
        colors_phase.append(colorFader(c1_,c2_,x/90))
        
c1_='Black'
c2_='grey' 
colors_lags=[]
for x in np.arange(n):
    colors_lags.append(colorFader(c1_,c2_,x/n))


# In[30]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#embedding_mean=np.load(Intermediate_object_folder+'Embedding_example_ABCD_08042024_1321.npy')

x = embedding_mean_aligned[:,0]
y = embedding_mean_aligned[:,1]
z = embedding_mean_aligned[:,2]

ax.scatter(x, y, z, c=colors_state, marker='o',s=100)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.azim = 60 ## rotate around z axis
ax.elev = 60 ##angle between eye and xy plane (i.e. 0 = looking from side, 90= looking from top)
plt.show()


# In[113]:


embedding_mean_aligned_perstate=np.asarray([embedding_mean_aligned[state_ind*90:(state_ind+1)*90]                                            for state_ind in range(int(num_states))])
distances_all_aligned=[]
for state_ind_X in np.arange(int(num_states)):
    for state_ind_Y in np.arange(int(num_states)):
        if state_ind_X!=state_ind_Y:
            distance_XY_=np.linalg.norm(embedding_mean_aligned_perstate[state_ind_X]                                        -embedding_mean_aligned_perstate[state_ind_Y],axis=1)
            distances_all_aligned.append(distance_XY_)
distance_mean_aligned=np.nanmean(distances_all_aligned)


# In[114]:


distance_mean_aligned


# In[ ]:





# In[385]:


##UMAP embedding - per day aligned

reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
distance_mean_aligned_perday_all=[]
distance_mean_aligned_perday_crossphase_all=[]
for mouse_recday in ephys_mean_z_dic.keys():
    ephys_mean_z_pertask_day=ephys_mean_z_aligned_dic[mouse_recday]
    embedding_mean_pertask_perday = reducer.fit_transform(ephys_mean_z_pertask_day.T)
    
    embedding_mean_pertask_perstate_perday=np.asarray([embedding_mean_pertask_perday[state_ind*90:(state_ind+1)*90]                                            for state_ind in range(int(num_states))])
    distances_all_perday=[]
    distances_all_perday_crossphase=[]
    for state_ind_X in np.arange(int(num_states)):
        for state_ind_Y in np.arange(int(num_states)):
            if state_ind_X!=state_ind_Y :
                distance_XY_=np.linalg.norm(embedding_mean_pertask_perstate_perday[state_ind_X]                                            -embedding_mean_pertask_perstate_perday[state_ind_Y],axis=1)
                
                distance_XY_crossphase_=np.linalg.norm(embedding_mean_pertask_perstate_perday[state_ind_X]                                            -np.roll(embedding_mean_pertask_perstate_perday[state_ind_Y],45,axis=0)                                                       ,axis=1)

                distances_all_perday.append(distance_XY_)
                distances_all_perday_crossphase.append(distance_XY_crossphase_)
    distance_mean_perday=np.nanmean(distances_all_perday)
    distance_mean_aligned_perday_all.append(distance_mean_perday)
    
    distance_mean_perday_crossphase=np.nanmean(distances_all_perday_crossphase)
    distance_mean_aligned_perday_crossphase_all.append(distance_mean_perday_crossphase)
    
distance_mean_aligned_perday_all=np.vstack((distance_mean_aligned_perday_all))
distance_mean_aligned_perday_crossphase_all=np.vstack((distance_mean_aligned_perday_crossphase_all))


# In[ ]:





# In[ ]:





# In[162]:


'''
balance across phases
run single phase
run in a way that allows direct comparison with A-aligned data

'''


# In[163]:


np.sum(np.vstack((anchors_neurons_ses))[:,1]==0)/len(np.vstack((anchors_neurons_ses))[:,1])


# In[183]:


distances_alltypes=np.hstack((distance_mean_perday_all,distance_mean_abstract_perday_all,                              distance_mean_aligned_perday_all))
distances_alltypes=np.column_stack((distance_mean_shuff_all,distances_alltypes))

distances_alltypes=distances_alltypes[neuron_number_bool==True]
bar_plotX(distances_alltypes.T,'none',0,6,'points','paired',0.025)


# In[647]:


len(distance_mean_shuff_all)


# In[ ]:





# In[ ]:





# In[165]:


##A-aligned vs anchor aligned
distances_alltypes=np.hstack((distance_mean_abstract_perday_all,                              distance_mean_aligned_perday_all))

distances_alltypes=distances_alltypes[neuron_number_bool==True]
bar_plotX(distances_alltypes.T,'none',0,4,'points','paired',0.025)
plt.show()

print(st.ttest_rel(distances_alltypes[:,0],distances_alltypes[:,1]))


# In[ ]:





# In[ ]:


'''
1) 5-1 split of tasks - align to A or anchor on the 5 and average after aligning
2) add the aligned test task (unaeveraged) to the umap (as an extension, so 720 bins instead of 360)
3) visualise
4) measure distance between test and training
5) repeat across all splits and average

'''


# In[ ]:





# In[375]:


##UMAP embedding - for anchor-aligned data - training vs test
test_ses_ind=1
restrict_to_zeroanchored=False
restrict_to_nonzeroanchored=False


distance_mean_aligned_perday_split_all_allsplits=[]
for test_ses_ind in np.arange(6):
    distance_mean_aligned_perday_split_all=[]
    for mouse_recday in ephys_mean_z_aligned_split_dic[test_ses_ind].keys():
        ephys_mean_z_pertask_day=ephys_mean_z_aligned_split_dic[test_ses_ind][mouse_recday]

        ###replacing nans with each neuron's minimum
        for neuron in np.arange(len(ephys_mean_z_pertask_day)):
            ephys_mean_z_pertask_day[neuron][np.isnan(ephys_mean_z_pertask_day[neuron])]                                      =np.nanmin(ephys_mean_z_pertask_day,axis=1)[neuron]

        ###finding anchors
        coeffs=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_coeffs_all_'+mouse_recday+'.npy')
        indices_regressors=np.reshape(np.arange(np.shape(coeffs)[2]),(num_nodes,num_phases,num_lags))
        max_coeffs=np.argmax(coeffs,axis=2)
        anchors_neurons_ses=np.asarray([[np.hstack(((np.where(indices_regressors==max_coeffs[neuron,ses_ind_ind]))))
        for ses_ind_ind in range(len(max_coeffs.T))] for neuron in range(len(max_coeffs))])[:,test_ses_ind]

        phase_anchor_neurons_ses=anchors_neurons_ses[:,1]
        if restrict_to_zeroanchored==True:
            ephys_mean_z_pertask_day=ephys_mean_z_pertask_day[phase_anchor_neurons_ses==0]
        elif restrict_to_nonzeroanchored==True:
            ephys_mean_z_pertask_day=ephys_mean_z_pertask_day[phase_anchor_neurons_ses!=0]
        
        reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
        embedding_mean_pertask_perday = reducer.fit_transform(ephys_mean_z_pertask_day.T)

        embedding_mean_pertask_perday_reshaped=embedding_mean_pertask_perday.            reshape(2,len(embedding_mean_pertask_perday)//2,len(embedding_mean_pertask_perday.T))

        distance_XY_=np.linalg.norm(embedding_mean_pertask_perday_reshaped[0]                                    -embedding_mean_pertask_perday_reshaped[1],axis=1)

        distance_mean_aligned_perday_split_all.append(distance_XY_)
    distance_mean_aligned_perday_split_all=np.vstack((distance_mean_aligned_perday_split_all))
    distance_mean_aligned_perday_split_all_allsplits.append(np.nanmean(distance_mean_aligned_perday_split_all,axis=1))


# In[370]:


np.shape(distance_XY_)


# In[ ]:





# In[345]:





# In[376]:


##UMAP embedding - for tone-aligned data - training vs test
test_ses_ind=1
restrict_to_zeroanchored=False
restrict_to_nonzeroanchored=False


distance_mean_perday_split_all_allsplits=[]
for test_ses_ind in np.arange(6):
    distance_mean_perday_split_all=[]
    for mouse_recday in ephys_mean_z_split_dic[test_ses_ind].keys():
        ephys_mean_z_pertask_day=ephys_mean_z_split_dic[test_ses_ind][mouse_recday]

        ###replacing nans with each neuron's minimum
        for neuron in np.arange(len(ephys_mean_z_pertask_day)):
            ephys_mean_z_pertask_day[neuron][np.isnan(ephys_mean_z_pertask_day[neuron])]                                      =np.nanmin(ephys_mean_z_pertask_day,axis=1)[neuron]


        ###finding anchors
        coeffs=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_coeffs_all_'+mouse_recday+'.npy')
        indices_regressors=np.reshape(np.arange(np.shape(coeffs)[2]),(num_nodes,num_phases,num_lags))
        max_coeffs=np.argmax(coeffs,axis=2)
        anchors_neurons_ses=np.asarray([[np.hstack(((np.where(indices_regressors==max_coeffs[neuron,ses_ind_ind]))))
        for ses_ind_ind in range(len(max_coeffs.T))] for neuron in range(len(max_coeffs))])[:,test_ses_ind]

        phase_anchor_neurons_ses=anchors_neurons_ses[:,1]
        if restrict_to_zeroanchored==True:
            ephys_mean_z_pertask_day=ephys_mean_z_pertask_day[phase_anchor_neurons_ses==0]
        elif restrict_to_nonzeroanchored==True:
            ephys_mean_z_pertask_day=ephys_mean_z_pertask_day[phase_anchor_neurons_ses!=0]

        reducer = umap.UMAP(n_components = 3, metric = 'cosine', n_neighbors = 50, min_dist = 0.6)
        embedding_mean_pertask_perday = reducer.fit_transform(ephys_mean_z_pertask_day.T)

        embedding_mean_pertask_perday_reshaped=embedding_mean_pertask_perday.            reshape(2,len(embedding_mean_pertask_perday)//2,len(embedding_mean_pertask_perday.T))

        distance_XY_=np.linalg.norm(embedding_mean_pertask_perday_reshaped[0]                                    -embedding_mean_pertask_perday_reshaped[1],axis=1)

        distance_mean_perday_split_all.append(distance_XY_)
    distance_mean_perday_split_all=np.vstack((distance_mean_perday_split_all))
    
    distance_mean_perday_split_all_allsplits.append(np.nanmean(distance_mean_perday_split_all,axis=1))


# In[401]:





# In[ ]:





# In[406]:


##A-aligned vs anchor aligned

recording_daysX=np.asarray(list(ephys_mean_z_aligned_split_dic[test_ses_ind].keys()))
recording_daysY=np.asarray(list(ephys_mean_z_split_dic[test_ses_ind].keys()))

rec_day_bool=np.isin(recording_daysX,recording_daysY)




distance_mean_aligned_perday_split_mean=np.nanmean(distance_mean_aligned_perday_split_all_allsplits,axis=0)
distance_mean_perday_split_mean=np.nanmean(distance_mean_perday_split_all_allsplits,axis=0)

distances_alltypes=np.vstack((distance_mean_aligned_perday_split_mean,distance_mean_perday_split_mean))

#distances_alltypes=distances_alltypes[neuron_number_bool==True]
bar_plotX(distances_alltypes,'none',0,35,'points','paired',0.025)
plt.show()

print(st.ttest_rel(distances_alltypes[:,0],distances_alltypes[:,1]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




