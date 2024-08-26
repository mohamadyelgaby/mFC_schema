#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:





# In[ ]:





# In[2]:


##Importing libraries
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
from pingouin import partial_corr
from collections import Counter
import random
from sklearn.linear_model import LogisticRegression
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter1d
import warnings
import statsmodels
from sklearn.preprocessing import MaxAbsScaler


# In[3]:


360/5


# In[4]:


##Importing custom functions
module_path = os.path.abspath(os.path.join(Code_folder))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from mBaseFunctions import rec_dd, remove_empty, rank_repeat, dict_to_array, concatenate_complex2, smooth_circular,polar_plot_stateX2, indep_roll, bar_plotX, plot_scatter, non_repeat_ses_maker, two_proportions_test, noplot_timecourseA,circular_sem


# In[ ]:





# In[5]:




def partition(alist, indices):
    return np.asarray([np.asarray(alist[i:j]) for i, j in zip(indices[:-1], indices[1:])])

def normalise(xx,num_bins=90,take_max=False):
    lenxx=len(xx)
    if lenxx<num_bins:
        xx=np.repeat(xx,10)/10
        lenxx=lenxx*10
    indices_polar=np.arange(lenxx)
    if take_max==True:
        normalized_xx=st.binned_statistic(indices_polar,xx, 'max', bins=num_bins)[0]
    else:
        normalized_xx=st.binned_statistic(indices_polar,xx, 'mean', bins=num_bins)[0]
    return(normalized_xx)

def raw_to_norm(raw_neuron,Trial_times_conc,num_states=4,return_mean=True,smoothing=True,                take_max=False,smoothing_sigma=10):
    raw_neuron_split=remove_empty(partition(list(raw_neuron),list(Trial_times_conc)))
    if len(raw_neuron_split)%num_states!=0:
        raw_neuron_split=raw_neuron_split[:len(raw_neuron_split)-len(raw_neuron_split)%num_states]
    
    if take_max==True:
        raw_neuron_split_norm=np.asarray([normalise(raw_neuron_split[ii],take_max=True)                                          for ii in np.arange(len(raw_neuron_split))])
    else:
        raw_neuron_split_norm=np.asarray([normalise(raw_neuron_split[ii]) for ii in np.arange(len(raw_neuron_split))])
    
    Actual_norm=(raw_neuron_split_norm.reshape(len(raw_neuron_split_norm)//num_states,                                               len(raw_neuron_split_norm[0])*num_states))
    
    if return_mean==True:
        Actual_norm_mean=np.nanmean(Actual_norm,axis=0)
        if smoothing==True:
            Actual_norm_smoothed=smooth_circular(Actual_norm_mean,sigma=smoothing_sigma)
            return(Actual_norm_smoothed)
        else:
            return(Actual_norm_mean)
    else:
        return(Actual_norm)
    
def remove_nan(x):
    x=x[~np.isnan(x)]
    return(x)

def unique_nosort(a):
    indexes = np.unique(a, return_index=True)[1]
    return(np.asarray([a[index] for index in sorted(indexes)]))

def one_hot_encode(x,length):
    array=np.zeros((len(x),length))
    for entry in np.arange(len(x)):
        if ~np.isnan(x[entry]):
            array[entry,int(x[entry])]=1
    return(array)


# #LOADING FILES - tracking, behaviour, Ephys raw, Ephys binned
# tt=time.time()
# 
# 
# try:
#     os.mkdir(Intermediate_object_folder)
# except FileExistsError:
#     pass
# 
# dictionaries_list=['day_type_dicX','session_dic_behaviour','session_dic','cluster_dic','Task_num_dic',\
#                   'Num_trials_dic2','speed_dic','Phases_raw_dic2','States_raw_dic','Times_from_reward_dic',\
#                   'tuning_singletrial_dic2','Tuned_dic']
# 
# ##,'binned_FR_dic'
# 
# for name in dictionaries_list:
#     try:
#         data_filename_memmap = os.path.join(Intermediate_object_folder, name)
#         data = load(data_filename_memmap)#, mmap_mode='r')
#         exec(name+'= data')
#     except Exception as e:
#         print(name)
#         print(e)
#         print('Not loaded')
# print(time.time()-tt)

# In[ ]:





# In[ ]:





# In[6]:


##Defining Task grid
from scipy.spatial import distance_matrix
from itertools import product
len_side=3
x=np.arange(len_side)
S=np.asarray(list(product(x, x)))
Task_matrix_blank=np.zeros((len_side,len_side))

A = [[-1, 0], [1, 0], [0, 1], [0, -1]]

###shortest distances 
from scipy.spatial import distance_matrix
from itertools import product
x=(0,1,2)
Task_grid=np.asarray(list(product(x, x)))

mapping_pyth={2:2,5:3,8:4}

distance_mat_raw=distance_matrix(Task_grid, Task_grid)
len_matrix=len(distance_mat_raw)
distance_mat=np.zeros((len_matrix,len_matrix))
for ii in range(len_matrix):
    for jj in range(len_matrix):
        if (distance_mat_raw[ii,jj]).is_integer()==False:
            hyp=int((distance_mat_raw[ii,jj])**2)
            distance_mat[ii,jj]=mapping_pyth[hyp]
        else:
            distance_mat[ii,jj]=distance_mat_raw[ii,jj]
mindistance_mat=distance_mat.astype(int)


abstract_structure='ABCD'


# In[7]:


mindistance_mat


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# for mouse_recday in recording_days_:
#     np.save(Intermediate_object_folder_dropbox+'session_behaviour_'+mouse_recday+'.npy',\
#             session_dic_behaviour['awake'][mouse_recday])
#     np.save(Intermediate_object_folder_dropbox+'session_'+mouse_recday+'.npy',\
#             session_dic['awake'][mouse_recday])
#     np.save(Intermediate_object_folder_dropbox+'Task_num_'+mouse_recday+'.npy',\
#            Task_num_dic[mouse_recday])
#     
#     np.save(Intermediate_object_folder_dropbox+'State_zmax_'+mouse_recday+'.npy',\
#             Tuned_dic['State_zmax'][mouse_recday])
#     
#     np.save(Intermediate_object_folder_dropbox+'Phase_tuned_'+mouse_recday+'.npy',\
#             Tuned_dic['Phase'][mouse_recday])
#     
#     np.save(Intermediate_object_folder_dropbox+'State_tuned_'+mouse_recday+'.npy',\
#             Tuned_dic['State_zmax_bool'][mouse_recday])
#     
#     np.save(Intermediate_object_folder_dropbox+'Num_trials_'+mouse_recday+'.npy',\
#             dict_to_array(Num_trials_dic2[mouse_recday]))
#     
#     
#     
#     
#     awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
#     num_sessions=len(awake_sessions_behaviour)
#     for ses_ind in np.arange(num_sessions):
#         
#         np.save(Intermediate_object_folder_dropbox+'Phases_raw2_'+mouse_recday+'_'+str(ses_ind)+'.npy',\
#            Phases_raw_dic2[mouse_recday][ses_ind])
#         
#         np.save(Intermediate_object_folder_dropbox+'States_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy',\
#            States_raw_dic[mouse_recday][ses_ind])
#         
#         np.save(Intermediate_object_folder_dropbox+'Times_from_reward_'+mouse_recday+'_'+str(ses_ind)+'.npy',\
#            Times_from_reward_dic[mouse_recday][ses_ind])
#         
#         np.save(Intermediate_object_folder_dropbox+'speed_'+mouse_recday+'_'+str(ses_ind)+'.npy',\
#                 speed_dic[mouse_recday][ses_ind])
#                 
#         np.save(Intermediate_object_folder_dropbox+'tuning_phase_boolean_max_'+mouse_recday+'_'+str(ses_ind)+'.npy',\
#                 tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][ses_ind])
#         
#         
#         
#         
# 

# In[ ]:





# In[ ]:


'''re-write below to fit with longer lags'''


# In[10]:


recording_days_[0]


# In[16]:


###generating lagged regressors
num_task_states=4
num_task_phases=3
num_nodes=9
num_spatial_locations=num_nodes


limited=False ##if true restricts lags to single trial, if false extends lags beyond this (see below)
num_repeats=2
if limited==True:
    num_lags=int(num_task_states*num_task_phases)
    addition=''
else:
    num_lags=int(num_task_states*num_task_phases*num_repeats)
    addition='_beyond'
remove_edges=True
rerun=True
#GLM_anchoring_prep_dic=rec_dd()

recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')

for mouse_recday in recording_days_:
    print(mouse_recday)
    
    if rerun==False:
    
        try:
            np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_regressors'+mouse_recday+'.npy',                   allow_pickle=True)
            print('Already run')
            continue
        except:
            print('Running')
    else:
        print("Running")
    
    
    awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
    awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')

    #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
    #np.arange(len(awake_sessions_behaviour))])

    num_sessions=len(awake_sessions_behaviour)

    
    sessions=np.load(Intermediate_object_folder_dropbox+'Task_num_'+mouse_recday+'.npy')
    num_refses=len(np.unique(sessions))
    num_comparisons=num_refses-1
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 


    Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)



    #phases_conc_all_=[]
    #states_conc_all_=[]
    #Location_raw_eq_all_=[]
    #Neuron_raw_all_=[]

    regressors_flat_allTasks=[]
    Location_allTasks=[]
    Neuron_allTasks=[]


    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):

        try:
            Neuron_raw=np.load(Intermediate_object_folder_dropbox+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            Location_raw=np.load(Intermediate_object_folder_dropbox+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            Location_norm=np.load(Intermediate_object_folder_dropbox+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy',                                 allow_pickle=True)
            XY_raw=np.load(Intermediate_object_folder_dropbox+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            speed_raw=np.load(Intermediate_object_folder_dropbox+'speed_'+mouse_recday+'_'+str(ses_ind)+'.npy')


            acceleration_raw_=np.diff(speed_raw)/0.025
            acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
            Trial_times=np.load(Intermediate_object_folder_dropbox+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')

        except:
            ('Trying ceph')
            try:
                Neuron_raw=np.load(Intermediate_object_folder_ceph+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Location_raw=np.load(Intermediate_object_folder_ceph+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Location_norm=np.load(Intermediate_object_folder_ceph+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy',                                     allow_pickle=True)
                XY_raw=np.load(Intermediate_object_folder_ceph+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                speed_raw=np.load(Intermediate_object_folder_ceph+'speed_'+mouse_recday+'_'+str(ses_ind)+'.npy')


                acceleration_raw_=np.diff(speed_raw)/0.025
                acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                Trial_times=np.load(Intermediate_object_folder_ceph+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            except:
                print('Files not found for session '+str(ses_ind))
                continue

        len_variables=[]
        for variable in [Neuron_raw,Location_raw,Location_norm,XY_raw,speed_raw,Trial_times]:
            len_variables.append(len(variable))
        
        if np.min(len_variables)==0:
            print('Some files on dropbox empty - trying ceph')
            try:
                Neuron_raw=np.load(Intermediate_object_folder_ceph+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Location_raw=np.load(Intermediate_object_folder_ceph+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Location_norm=np.load(Intermediate_object_folder_ceph+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy',                                     allow_pickle=True)
                XY_raw=np.load(Intermediate_object_folder_ceph+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                speed_raw=np.load(Intermediate_object_folder_ceph+'speed_'+mouse_recday+'_'+str(ses_ind)+'.npy')


                acceleration_raw_=np.diff(speed_raw)/0.025
                acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                Trial_times=np.load(Intermediate_object_folder_ceph+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')

            except:
                print('Atleast one incomplete  for session '+str(ses_ind))
                continue
             
            
        num_neurons=len(Neuron_raw)
        


        phases=np.load(Intermediate_object_folder_dropbox+'Phases_raw2_'+mouse_recday+'_'+str(ses_ind)+'.npy',                      allow_pickle=True)       
        phases_conc=concatenate_complex2(concatenate_complex2(phases))
        states=np.load(Intermediate_object_folder_dropbox+'States_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy',                      allow_pickle=True)
        states_conc=concatenate_complex2(concatenate_complex2(states))
        times=np.load(Intermediate_object_folder_dropbox+'Times_from_reward_'+mouse_recday+'_'+str(ses_ind)+'.npy',                      allow_pickle=True)
        times_conc=concatenate_complex2(concatenate_complex2(times))
        times_conc_eq=times_conc[:len(phases_conc)]
        speed_raw_eq=speed_raw[:len(phases_conc)]
        acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
        Location_raw_eq=Location_raw[:len(phases_conc)]
        Location_norm_conc=np.concatenate(Location_norm)

        Neuron_raw_eq=Neuron_raw[:,:len(phases_conc)]

        if remove_edges==True:
            Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges


        ###Using the model to calculate the regressors

        Task_phasetminus1=-1
        locationtminus1=-1
        Module_anchor_progress_dic=rec_dd()
        #for mouse_recday in day_type_dicX['combined_ABCDonly']:
        #    for session in np.arange(num_sessions):

        ##Importing occupancy and making phase/state 
        #nodes_=occupancy_dic[mouse_recday][session]
        nodes=(Location_raw_eq-1).astype(int)
        len_bins=len(nodes)


        #structure=(np.asarray([st.mode(Location_norm_conc.reshape(int(len(Location_norm_conc)/360),360)[:,ii])[0][0]+1\
        #            for ii in np.arange(4)*90])).astype(int)
        structure=Tasks[ses_ind]
        ### NOT 0 based indexed (so 1 is location 1)


        ##Task states
        
        states=states_conc


        ##Rewarded states
        rewarded_state0=S[structure[0]-1]
        rewarded_statet=rewarded_state0

        ##Task phase
        
        phases=phases_conc

        Task_phaset=0

        All_modules_primed=0

        ###conditions
        T=len(nodes)
        #num_trials_planning=5
        #planning_active=True
        #sweep_forward=True ###whether to sweep forward across attractor to find next goal when using planning
        multiple_bumps=True
        probabilistic_choice=False

        plot=False
        make_video=False

        ##plotting 
        if plot==True and make_video==False:
            plt.figure(figsize = (20,T/5))
            plotting_coords=np.asarray(list(product(np.arange(np.sqrt(T)), np.arange(np.sqrt(T)))))

        elif plot==True and make_video==True:
            fig = plt.figure()
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122, projection='polar')
            #fig, axs = plt.subplots(2)
            #fig2, ax2 = plt.subplots(subplot_kw=dict(projection='polar'))

        #plt.rcParams['animation.ffmpeg_path'] =\
        #r'C:\Users\moham\anaconda3\envs\DLC-GPU\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win64-v4.2.2'

        metadata = dict(title='Movie', artist='codinglikemad')
        #writer = PillowWriter(fps=5, metadata=metadata)
        #writer = FFMpegWriter(fps=5, metadata=metadata)

        ##Module to plot
        spatial_loc=6
        spatial_ind=spatial_loc-1
        phase_ind=2

        move_phase=0

        ##Relative task_states
        module_anchor_phases=np.zeros(num_spatial_locations)
        #module_anchor_initiated=np.zeros((num_spatial_locations,num_task_phases))
        module_anchor_progress=np.zeros((num_spatial_locations,num_task_phases,num_lags))
        reward_status_all=np.zeros(T)
        module_anchor_progress_all2=np.zeros((T,num_spatial_locations,num_task_phases,num_lags))

        trial_no=-1

        if probabilistic_choice==True:
            condition_choice='probabilistic'
        else:
            condition_choice='deterministic'

        #with writer.saving(fig, Modelling_output_folder_dropbox+video_name, 100):
        for t in range(T):

            location_indt=int(nodes[t])
            Task_phaset=int(phases[t])
            Task_state_next=int(states[t])
            rewarded_statet=S[structure[int(Task_state_next)]-1]


            reward_status=np.sum(location_indt+1 in structure)
            reward_status_all[t]=int(reward_status)

            if t>=0:
                if Task_phaset!=Task_phasetminus1:
                    move_phase=1
                else:
                    move_phase=0

                if location_indt!=locationtminus1 and location_indt in np.arange(num_nodes):
                    move_location=1
                else:
                    move_location=0



            ###Moving "activity bump" along spatially anchored modules
            for location_ind_ in np.arange(num_spatial_locations): ##looping over modules by location
                for Task_phase_ in np.arange(num_task_phases): ##looping over modules by phase
                    if move_phase==1: #module_anchor_initiated[location_ind_,Task_phase_]==1: 
                        ##i.e. has module been initiated and has phase changed? 

                        ##Spatial/phase input
                        if location_indt==location_ind_ and Task_phaset==Task_phase_:
                            if multiple_bumps==True or np.sum(module_anchor_progress[location_ind_,Task_phase_])==0:
                                module_anchor_progress[location_ind_,Task_phase_,0]=prev_module_activity=1
                            elif multiple_bumps==False and np.sum(module_anchor_progress[location_ind_,Task_phase_])>0:
                                prev_module_activity=module_anchor_progress[location_ind_,Task_phase_,0]
                        else:
                            prev_module_activity=module_anchor_progress[location_ind_,Task_phase_,0]

                        ##Moving bumps(s) when phase changes
                        module_anchor_progress[location_ind_,Task_phase_]=np.roll(module_anchor_progress[location_ind_,                                                                                                         Task_phase_],1)

                        ##adjusting activity based on whether currently active bump received spatial/phase input
                        if module_anchor_progress[location_ind_,Task_phase_,1]>0: ##changed
                            if location_indt==location_ind_ and Task_phaset==Task_phase_:
                                current_module_activity=1 #-ReLU(1-prev_module_activity+0.5)
                            else:
                                current_module_activity=0#prev_module_activity*0.5
                            module_anchor_progress[location_ind_,Task_phase_,1]=current_module_activity


                    if move_phase==0 and move_location==1:
                        ##Spatial/phase input
                        if location_indt==location_ind_ and Task_phaset==Task_phase_:
                            if multiple_bumps==True or np.sum(module_anchor_progress[location_ind_,Task_phase_])==0:
                                module_anchor_progress[location_ind_,Task_phase_,1]=prev_module_activity=1
                            elif multiple_bumps==False and np.sum(module_anchor_progress[location_ind_,Task_phase_])>0:
                                prev_module_activity=module_anchor_progress[location_ind_,Task_phase_,0]
                        else:
                            prev_module_activity=module_anchor_progress[location_ind_,Task_phase_,0]


                        ##adjusting activity based on whether currently active bump received spatial/phase input
                        #if module_anchor_progress[location_ind_,Task_phase_,1]>0: ##changed
                        #    if location_indt==location_ind_ and Task_phaset==Task_phase_:
                        #        current_module_activity=1 #-ReLU(1-prev_module_activity+0.5)
                        #    else:
                        #        current_module_activity=prev_module_activity*0.5
                        #    module_anchor_progress[location_ind_,Task_phase_,1]=current_module_activity


            Task_phasetminus1=Task_phaset
            locationtminus1=location_indt

            module_anchor_progress_all2[t]=module_anchor_progress

        regressors=np.roll(module_anchor_progress_all2,-1,axis=3) 
        ###rolled back as module_anchor_progress is lagged forward by 1
        regressors=np.asarray(regressors)
        #Module_anchor_progress_dic[mouse_recday][session]=module_anchor_progress_all2
        regressors_flat=np.reshape(regressors, (regressors.shape[0], np.prod(regressors.shape[1:])))


        regressors_flat_allTasks.append(regressors_flat)
        Location_allTasks.append(Location_raw_eq)
        Neuron_allTasks.append(Neuron_raw_eq.T)

    regressors_flat_allTasks=np.asarray(regressors_flat_allTasks)
    Location_allTasks=np.asarray(Location_allTasks)
    Neuron_allTasks=np.asarray(Neuron_allTasks)
    
    np.save(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_regressors'+addition+mouse_recday+'.npy',           regressors_flat_allTasks)
    
    np.save(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Location'+addition+mouse_recday+'.npy',           Location_allTasks)
    
    np.save(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Neuron'+addition+mouse_recday+'.npy',           Neuron_allTasks)
    
    #GLM_anchoring_prep_dic['regressors'][mouse_recday]=regressors_flat_allTasks
    #GLM_anchoring_prep_dic['Location'][mouse_recday]=Location_allTasks
    #GLM_anchoring_prep_dic['Neuron'][mouse_recday]=Neuron_allTasks


# In[15]:


module_anchor_progress


# In[12]:


module_anchor_progress


# In[ ]:





# In[13]:


module_anchor_progress_all2


# In[ ]:





# In[21]:


####Lagged regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import PoissonRegressor

tt=time.time()


num_states=4
num_phases=3
num_nodes=9
num_lags=12


limited=False ##if true restricts lags to single trial, if false extends lags beyond this (see below)
num_repeats=2
if limited==True:
    num_regressors=num_phases*num_nodes*num_lags
    addition=''
else:
    num_regressors=num_phases*num_nodes*num_lags*num_repeats
    addition='_beyond'




re_run=True
use_prefphase=True #
regularize=True
Poisson_regression=True

#alpha=0.01 ##0.01 used in first submission
if Poisson_regression==True:
    alpha=1
else:
    alpha=0.01 ##0.01 used in first submission
    

recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')

#GLM_anchoring_dic=rec_dd()
for mouse_recday in recording_days_:
    print(mouse_recday)
    if re_run==False:
        try:
            np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_coeffs_all_'+mouse_recday+'.npy',                   allow_pickle=True)
            print('Already run')
            continue
        except:
            print('Running')
        #if mouse_recday in list(GLM_anchoring_dic['coeffs_all'].keys()):
        #    print('Already run')
        #    continue

    try:
        
        awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+                                          mouse_recday+'.npy')
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
        num_neurons=len(Neuron_raw)

        regressors_flat_allTasks=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_regressors'                                         +addition+mouse_recday+'.npy',allow_pickle=True)
        Location_allTasks=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Location'                                         +addition+mouse_recday+'.npy',allow_pickle=True)
        Neuron_allTasks=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Neuron'                                         +addition+mouse_recday+'.npy',allow_pickle=True)


        num_non_repeat_ses_found=len(regressors_flat_allTasks)

        coeffs_all=np.zeros((num_neurons,num_non_repeat_ses_found,num_regressors))
        coeffs_all[:]=np.nan



        for ses_ind_ind_test in np.arange(num_non_repeat_ses_found):
            print(ses_ind_ind_test)
            ses_ind_actual=found_ses[ses_ind_ind_test]

            training_sessions=np.setdiff1d(np.arange(num_non_repeat_ses_found),ses_ind_ind_test)

            ##concatenating arrays
            regressors_flat_trainingTasks_=regressors_flat_allTasks[training_sessions]
            Location_trainingTasks_=Location_allTasks[training_sessions]
            Neuron_trainingTasks_=Neuron_allTasks[training_sessions]

            regressors_flat_trainingTasks=np.vstack((regressors_flat_trainingTasks_))
            Location_trainingTasks=np.hstack((Location_trainingTasks_))
            Neuron_trainingTasks=np.vstack((Neuron_trainingTasks_)).T

            ##phase
            #phase_peaks=np.load(Intermediate_object_folder_dropbox+'tuning_phase_boolean_max_'+mouse_recday+'_'\
            #                    +str(ses_ind_actual)+'.npy')

            phase_peaks=np.load(Intermediate_object_folder_dropbox+                                    'tuning_phase_boolean_max_'+mouse_recday+'.npy')[ses_ind_actual]
            pref_phase_neurons=np.argmax(phase_peaks,axis=1)
            phases=np.load(Intermediate_object_folder_dropbox+'Phases_raw2_'+                                                       mouse_recday+'_'+str(ses_ind_actual)+'.npy',allow_pickle=True)

            phase_ses_indices=np.asarray(found_ses)[training_sessions]
            phases_conc_=np.hstack((np.vstack([np.load(Intermediate_object_folder_dropbox+'Phases_raw2_'+                                                       mouse_recday+'_'+str(ses)+'.npy',allow_pickle=True)                                               for ses in phase_ses_indices])))
            phases_conc=concatenate_complex2(phases_conc_)

            ####Doing the regression
            for neuron in np.arange(num_neurons):
                ##independent variables
                regressors_nonan=regressors_flat_trainingTasks[~np.isnan(Location_trainingTasks)]
                regressors_flat=np.reshape(regressors_nonan, (regressors_nonan.shape[0],                                                              np.prod(regressors_nonan.shape[1:])))
                #times_conc_eq_nonan=times_conc_eq[~np.isnan(Location_raw_eq)]
                #speed_raw_eq_nonan=speed_raw_eq[~np.isnan(Location_raw_eq)]
                #phases_conc_nonan=phases_conc[~np.isnan(Location_raw_eq)]
                #acceleration_raw_eq_nonan=acceleration_raw_eq[~np.isnan(Location_raw_eq)]

                ##dependent variable
                Neuron_raw_eq_neuron=Neuron_trainingTasks[neuron]
                Neuron_raw_eq_neuron_nonan=Neuron_raw_eq_neuron[~np.isnan(Location_trainingTasks)]


                ##subsetting by phase
                pref_phase=pref_phase_neurons[neuron]
                phases_conc_nonan=phases_conc[~np.isnan(Location_trainingTasks)]

                regressors_flat_prefphase=regressors_flat[phases_conc_nonan==pref_phase]
                Neuron_raw_eq_neuron_nonan_prefphase=Neuron_raw_eq_neuron_nonan[phases_conc_nonan==pref_phase]

                ###regression
                if use_prefphase==True:
                    X = regressors_flat_prefphase
                    y = Neuron_raw_eq_neuron_nonan_prefphase
                else:
                    X = regressors_flat
                    y = Neuron_raw_eq_neuron_nonan


                if Poisson_regression==True:
                    reg = PoissonRegressor(alpha=alpha).fit(X, y)

                else:
                    if regularize==True and Poisson_regression==False:
                        reg = ElasticNet(alpha=alpha,positive=True).fit(X, y)
                    else:
                        reg = LinearRegression(positive=True).fit(X, y)


                coeffs_flat=reg.coef_
                coeffs_all[neuron,ses_ind_ind_test]=coeffs_flat

        #GLM_anchoring_dic['coeffs_all'][mouse_recday]=coeffs_all
        if Poisson_regression==True:
            np.save(Intermediate_object_folder_dropbox+'Poisson_GLM_anchoring_coeffs_all_'+addition+mouse_recday+                    '.npy',coeffs_all)
        else:
            np.save(Intermediate_object_folder_dropbox+'GLM_anchoring_coeffs_all_'+addition+mouse_recday+'.npy',                       coeffs_all)
    except:
        print('Files not found')
print(time.time()-tt)


# In[19]:


num_regressors


# In[ ]:





# In[40]:


coeffs_all[0,0]


# In[ ]:





# In[9]:


'''
here - need to run Tining_basic on the no tone days

'''


# In[ ]:





# In[10]:


'''
Predict sequence of actions for entire next trial from instantaneous mFC activity:
-Do this from a different set of neurons
-Do this for changes in behaviour (i.e. accounting for previous actions)


Method 1-
i) identify lagged profile of every neuron - GLM_dic
ii) at lag 1 from present - compare activity of all neurons that are significantly anchored to this lag 
(in the opposite direction) - significantly anchored (or top 3 peaks)

z-scored
or 
raw

The neuron(s) that win determine which location is visited (means taken if many neurons with same lag) at this lag

iii) repeat step ii for lags 2,3...etc up to 12
iv) this gives a (partial) policy per trial - repeat across all trials for the test session
v) repeat steps ii-iv for all test tasks
vi) report % accuracy
vii) repeat steps ii-vi for all 12 lags (this shows that its different neurons) - and show that this works with
different sets of neurons
viii) Now report %accuracy but only taking changes in policy - i.e. when animal visits a point in the task that it didnt
visit in the previous trial

Possible refinements: 

in step ii restrict possibile locations to those that are one-step from actual previous location 
only include instances where neurons representing all possible options are sampled 
'''


# In[ ]:





# In[58]:


###Calculating correlations between predicted and actual activity
tt=time.time()
close_to_anchor_bins_90=[0,1,2,11,10,9]
close_to_anchor_bins_30=[0,11]
re_run=True
use_prefphase=True ###if set to false correlations are calculated seperately for each phase and then averaged
use_mean=True ##use normalised, averaged activity for correlations - if true uses mean for each state in each task
###if false, uses trial by trial means for eahc state

Num_max=3 ##how many peaks should NOT be in the excluded regression columns for a neuron to be considered

##paramaters
num_bins=90
num_states=4
num_phases=3
num_nodes=9
num_lags=12

limited=False ##if true restricts lags to single trial, if false extends lags beyond this (see below)
num_repeats=2
if limited==True:
    num_regressors=num_phases*num_nodes*num_lags
    addition2=''
else:
    num_regressors=num_phases*num_nodes*num_lags*num_repeats
    addition2='_beyond'
    close_to_anchor_bins_90=np.arange(12)

smoothing_sigma=10

Poisson_regression=True
if Poisson_regression==True:
    addition='Poisson_'
else:
    addition=''

regressor_indices=np.arange(num_regressors)

if limited==True:
    regressor_indices_reshaped=np.reshape(regressor_indices,(num_phases*num_nodes,num_lags))
else:
    regressor_indices_reshaped=np.reshape(regressor_indices,(num_phases*num_nodes,num_lags*num_repeats))
    

zero_indices=regressor_indices_reshaped[:,0]
close_to_anchor_indices30=np.concatenate(regressor_indices_reshaped[:,close_to_anchor_bins_30])
close_to_anchor_indices90=np.concatenate(regressor_indices_reshaped[:,close_to_anchor_bins_90])

if limited==True:
    phase_norm_mean=np.tile(np.repeat(np.arange(num_phases),num_bins/num_phases),num_states)
    phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states,num_bins))
    
else:
    phase_norm_mean=np.tile(np.tile(np.repeat(np.arange(num_phases),num_bins/num_phases),num_states),num_repeats)
    phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states*num_repeats,num_bins))

#Entropy_thr_all=np.nanmean(np.hstack((concatenate_complex2(dict_to_array(Entropy_dic['Entropy_thr'])))))
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')
for mouse_recday in recording_days_:
    print(mouse_recday)
    
    #if re_run==False:
    #    if mouse_recday in list(GLM_anchoring_dic['Predicted_Actual_correlation'].keys()):
    #        print('Already run')
    #        continue
            
    if re_run==False:
        try:
            np.load(Intermediate_object_folder_dropbox+addition2+addition                    +'Predicted_Actual_correlation_'+mouse_recday+'.npy',allow_pickle=True)
            print('Already run')
            continue
        except:
            print('Running')
    #try:

    awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'                                      +mouse_recday+'.npy')
    awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
    sessions=np.load(Intermediate_object_folder_dropbox+'Task_num_'+mouse_recday+'.npy')
    Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy')

    num_sessions=len(awake_sessions_behaviour)
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    #regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]

    regressors_flat_allTasks=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_regressors'                                     +addition2+mouse_recday+'.npy',allow_pickle=True)

    num_non_repeat_ses_found=len(regressors_flat_allTasks)
    
    if mouse_recday=='me11_05122021_06122021':
        num_non_repeat_ses_found=6
        non_repeat_ses=non_repeat_ses[non_repeat_ses!=3] ### task in session 3 was almost identical to session 0 (mistake)

    found_ses=[]
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            found_ses.append(ses_ind)

        except:
            print('Files not found for session '+str(ses_ind))
            continue
    num_neurons=len(Neuron_raw) 
    #num_state_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)\
    #          for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T

    #entropy_tuned_withinneuron=Entropy_dic['Entropy_actual'][mouse_recday]<Entropy_dic['Entropy_thr'][mouse_recday]
    #entropy_tuned_global=Entropy_dic['Entropy_actual'][mouse_recday]<Entropy_thr_all
    #entropy_tuned_all=np.logical_and(entropy_tuned_withinneuron,entropy_tuned_global)

    state_zmax=np.load(Intermediate_object_folder_dropbox+'State_zmax_'+mouse_recday+'.npy',allow_pickle=True)

    corrs_all=np.zeros((num_neurons,num_non_repeat_ses_found))
    corrs_all_nozero=np.zeros((num_neurons,num_non_repeat_ses_found))
    corrs_all_nozero_strict=np.zeros((num_neurons,num_non_repeat_ses_found))

    corrs_all[:]=np.nan
    corrs_all_nozero[:]=np.nan
    corrs_all_nozero_strict[:]=np.nan

    for ses_ind_ind in np.arange(num_non_repeat_ses_found):
        ses_ind_actual=found_ses[ses_ind_ind]

        #regressors_ses=GLM_anchoring_prep_dic['regressors'][mouse_recday][ses_ind_ind]
        #location_ses=GLM_anchoring_prep_dic['Location'][mouse_recday][ses_ind_ind]
        #Actual_activity_ses=GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]

        regressors_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_regressors'                                     +addition2+mouse_recday+'.npy',allow_pickle=True)[ses_ind_ind]
        location_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Location'                                         +addition2+mouse_recday+'.npy',allow_pickle=True)[ses_ind_ind]
        Actual_activity_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Neuron'                                         +addition2+mouse_recday+'.npy',allow_pickle=True)[ses_ind_ind]

        #phase_peaks=np.load(Intermediate_object_folder_dropbox+'tuning_phase_boolean_max_'+mouse_recday+'_'+\
        #                            str(ses_ind_actual)+'.npy')
        phase_peaks=np.load(Intermediate_object_folder_dropbox+                                'tuning_phase_boolean_max_'+mouse_recday+'.npy')[ses_ind_actual]

        pref_phase_neurons=np.argmax(phase_peaks,axis=1)

        phases=np.load(Intermediate_object_folder_dropbox+'Phases_raw2_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy',                      allow_pickle=True)       
        phases_conc=concatenate_complex2(concatenate_complex2(phases))

        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
        Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25

        for neuron in np.arange(num_neurons):
            pref_phase=pref_phase_neurons[neuron]
            Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]
            #coeffs_ses_neuron=GLM_anchoring_dic['coeffs_all'][mouse_recday][neuron][ses_ind_ind]

            coeffs_ses_neuron_=np.load(Intermediate_object_folder_dropbox+addition+'GLM_anchoring_coeffs_all_'                                       +addition2+mouse_recday+'.npy',allow_pickle=True)[neuron,ses_ind_ind]
            coeffs_ses_neuron=np.copy(coeffs_ses_neuron_)
            #coeffs_ses_neuron[coeffs_ses_neuron_<0]=0

            ###maximum indices
            indices_sorted=np.flip(np.argsort(coeffs_ses_neuron))
            indices_sorted_nonan=indices_sorted[~np.isnan(coeffs_ses_neuron[indices_sorted])]
            topN_indices=indices_sorted_nonan[:Num_max]

            Predicted_activity_ses_neuron=np.sum(regressors_ses*coeffs_ses_neuron,axis=1)
            Predicted_activity_ses_neuron_scaled=Predicted_activity_ses_neuron*(            np.mean(Actual_activity_ses_neuron)/np.mean(Predicted_activity_ses_neuron))

            #num_state_peaks_neuronses=num_state_peaks_all[neuron,ses_ind_ind]
            #entropy_tuned_neuronses=entropy_tuned_all[neuron,ses_ind_ind]
            state_zmax_neuronses=state_zmax[neuron,ses_ind_ind]

            if np.isnan(np.nanmean(Predicted_activity_ses_neuron))==False and            np.nanmean(Predicted_activity_ses_neuron)>0:# and state_zmax_neuronses<0.05:#num_state_peaks_neuronses>0:

                ## prediction for all neurons/entire regression matrix
                if use_prefphase==False:
                    Predicted_Actual_correlation_all=[]
                    for phase_ind in np.arange(num_phases):
                        Predicted_Actual_correlation_=                        st.pearsonr(Actual_activity_ses_neuron[phases_conc==phase_ind],                        Predicted_activity_ses_neuron[phases_conc==phase_ind])[0]
                        Predicted_Actual_correlation_all.append(Predicted_Actual_correlation_)
                    Predicted_Actual_correlation=np.nanmean(Predicted_Actual_correlation_all)
                else:
                    if use_mean==False:
                        Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                                smoothing=False,return_mean=False)
                        Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron,Trial_times_conc,                                                           smoothing=False,return_mean=False)

                        Actual_norm_means=np.concatenate([[np.nanmean(Actual_norm[trial,num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                                          for trial in np.arange(len(Actual_norm))])
                        Predicted_norm_means=np.concatenate([[np.nanmean(Predicted_norm[trial,num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase])                            for ii in range(num_states)] for trial in np.arange(len(Predicted_norm))])
                        Predicted_Actual_correlation=st.pearsonr(Actual_norm_means,Predicted_norm_means)[0]
                    else:
                        Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                                                        smoothing=False)
                        Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron,Trial_times_conc,                                                           smoothing=False)

                        Actual_norm_means=np.asarray([np.nanmean(Actual_norm[num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])
                        Predicted_norm_means=np.asarray([np.nanmean(Predicted_norm[num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])
                        Predicted_Actual_correlation=st.pearsonr(Actual_norm_means,Predicted_norm_means)[0]

                corrs_all[neuron,ses_ind_ind]=Predicted_Actual_correlation

                ###prediction after removing 0 lagged betas
                coeffs_ses_neuron_copy1=np.copy(coeffs_ses_neuron)
                coeffs_ses_neuron_copy1[close_to_anchor_indices30]=np.nan
                Predicted_activity_ses_neuron_nozero=np.nansum(regressors_ses*coeffs_ses_neuron_copy1,axis=1)

                #if np.nanargmax(coeffs_ses_neuron) in zero_indices:
                if np.sum(np.isin(topN_indices,close_to_anchor_indices30))>0:
                    Predicted_Actual_correlation_nozero=np.nan
                else:
                    if use_prefphase==False:
                        Predicted_Actual_correlation_nozero_all=[]
                        for phase_ind in np.arange(num_phases):
                            Predicted_Actual_correlation_nozero_=                            st.pearsonr(Actual_activity_ses_neuron[phases_conc==phase_ind],                            Predicted_activity_ses_neuron_nozero[phases_conc==phase_ind])[0]
                            Predicted_Actual_correlation_nozero_all.append(Predicted_Actual_correlation_nozero_)
                        Predicted_Actual_correlation_nozero=np.nanmean(Predicted_Actual_correlation_nozero_all)

                    else:
                        if use_mean==False:
                            Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron_nozero,Trial_times_conc,                                                               smoothing=False,return_mean=False)
                            Predicted_norm_means=np.concatenate([[np.nanmean(Predicted_norm                                                                             [trial,num_bins*ii:num_bins*(ii+1)]                                [phase_norm_mean_states[ii]==pref_phase])                                for ii in range(num_states)] for trial in np.arange(len(Predicted_norm))])
                            Predicted_Actual_correlation_nozero=st.pearsonr(Actual_norm_means,Predicted_norm_means)[0]
                        else:
                            Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron_nozero,Trial_times_conc,                                                               smoothing=False)

                            Predicted_norm_means=np.asarray([np.nanmean(Predicted_norm[num_bins*ii:num_bins*(ii+1)]                                [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])

                            Predicted_Actual_correlation_nozero=st.pearsonr(Actual_norm_means,Predicted_norm_means)[0]

                corrs_all_nozero[neuron,ses_ind_ind]=Predicted_Actual_correlation_nozero

                ###prediction after removing 90 degrees either side of 0 lag
                coeffs_ses_neuron_copy2=np.copy(coeffs_ses_neuron)
                coeffs_ses_neuron_copy2[close_to_anchor_indices90]=np.nan
                Predicted_activity_ses_neuron_nozero_strict=np.nansum(regressors_ses*coeffs_ses_neuron_copy2,axis=1)


                #np.nanargmax(coeffs_ses_neuron) in close_to_anchor_indices:
                if np.sum(np.isin(topN_indices,close_to_anchor_indices90))>0: 
                    Predicted_Actual_correlation_nozero_strict=np.nan
                else:
                    if use_prefphase==False:
                        Predicted_Actual_correlation_nozero_strict_all=[]
                        for phase_ind in np.arange(num_phases):
                            Predicted_Actual_correlation_nozero_strict_=                            st.pearsonr(Actual_activity_ses_neuron[phases_conc==phase_ind],                            Predicted_activity_ses_neuron_nozero_strict[phases_conc==phase_ind])[0]
                            Predicted_Actual_correlation_nozero_strict_all.append(                            Predicted_Actual_correlation_nozero_strict_)
                        Predicted_Actual_correlation_nozero_strict=np.nanmean(                        Predicted_Actual_correlation_nozero_strict_all)
                    else:
                        if use_mean==False:
                            Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron_nozero_strict,Trial_times_conc,                                                               smoothing=False,return_mean=False)
                            Predicted_norm_means=np.concatenate([[np.nanmean(Predicted_norm                                                                             [trial,num_bins*ii:num_bins*(ii+1)]                                [phase_norm_mean_states[ii]==pref_phase])                                for ii in range(num_states)] for trial in np.arange(len(Predicted_norm))])
                            Predicted_Actual_correlation_nozero_strict=st.pearsonr(Actual_norm_means,                                                                                   Predicted_norm_means)[0]

                        else:
                            Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron_nozero_strict,Trial_times_conc,                                                               smoothing=False)
                            Predicted_norm_means=np.asarray([np.nanmean(Predicted_norm[num_bins*ii:num_bins*(ii+1)]                                [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])

                            Predicted_Actual_correlation_nozero_strict=st.pearsonr(Actual_norm_means,                                                                                   Predicted_norm_means)[0]

                corrs_all_nozero_strict[neuron,ses_ind_ind]=Predicted_Actual_correlation_nozero_strict

    corrs_mean=np.nanmean(corrs_all,axis=1)
    corrs_all_nozero_mean=np.nanmean(corrs_all_nozero,axis=1)
    corrs_all_nozero_strict_mean=np.nanmean(corrs_all_nozero_strict,axis=1)

    #GLM_anchoring_dic['Predicted_Actual_correlation'][mouse_recday]=corrs_all
    #GLM_anchoring_dic['Predicted_Actual_correlation_mean'][mouse_recday]=corrs_mean
    #GLM_anchoring_dic['Predicted_Actual_correlation_nonzero_mean'][mouse_recday]=corrs_all_nozero_mean
    #GLM_anchoring_dic['Predicted_Actual_correlation_nonzero_strict_mean'][mouse_recday]=corrs_all_nozero_strict_mean


    np.save(Intermediate_object_folder_dropbox+addition2+addition+'Predicted_Actual_correlation_'+mouse_recday+'.npy',               corrs_all)
    np.save(Intermediate_object_folder_dropbox+addition2+addition+            'Predicted_Actual_correlation_mean_'+mouse_recday+'.npy',corrs_mean)
    np.save(Intermediate_object_folder_dropbox+addition2+addition            +'Predicted_Actual_correlation_nonzero_mean_'+mouse_recday+'.npy',corrs_all_nozero_mean)
    np.save(Intermediate_object_folder_dropbox+addition2+addition            +'Predicted_Actual_correlation_nonzero_strict_mean_'+mouse_recday+'.npy',corrs_all_nozero_strict_mean)

    #except:
    #    print('Not analysed')
print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[59]:


Poisson_regression=True
if Poisson_regression==True:
    addition='Poisson_'
else:
    addition=''
    
limited=False ##if true restricts lags to single trial, if false extends lags beyond this (see below)
if limited==True:
    addition2=''
else:
    addition2='_beyond'

mouse_recdays_found=[]
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')
for mouse_recday in recording_days_:
    try:
        np.load(Intermediate_object_folder_dropbox+addition2+addition                +'Predicted_Actual_correlation_mean_'+mouse_recday+'.npy')
        mouse_recdays_found.append(mouse_recday)
    except:
        'Not found'
        
print(len(recording_days_))
print(len(mouse_recdays_found))

print(np.setdiff1d(recording_days_,mouse_recdays_found))
xx=np.hstack(([np.load(Intermediate_object_folder_dropbox+addition2+addition+'Predicted_Actual_correlation_mean_'+                       mouse_recday+'.npy')               for mouse_recday in mouse_recdays_found]))

print(len(xx))
len(remove_nan(xx))


# In[69]:


addition='Poisson_'
np.load(Intermediate_object_folder_dropbox+addition+name+'_'+mouse_recday+'.npy')


# In[86]:


state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'State_99'+mouse_recday+'.npy')                                     for mouse_recday in mouse_recdays_found]))    
state_tuning_=np.hstack(([np.load(Intermediate_object_folder_dropbox+'State_95'+mouse_recday+'.npy')                                 for mouse_recday in mouse_recdays_found]))


# In[88]:


np.sum(state_tuning_)


# In[ ]:





# In[ ]:





# In[60]:


Poisson_regression=True
if Poisson_regression==True:
    addition='Poisson_'
else:
    addition=''
    
limited=False ##if true restricts lags to single trial, if false extends lags beyond this (see below)
if limited==True:
    addition2=''
else:
    addition2='_beyond'


use_tuned=True
bins=50
use_strict=False ##if true uses p=0.01 threshold for state tuning
phase_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'Phase_'+mouse_recday+'.npy')                         for mouse_recday in mouse_recdays_found]))
    

if use_strict==True:
    state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'State_99'+mouse_recday+'.npy')                                     for mouse_recday in mouse_recdays_found]))    
else:
    state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'State_95'+mouse_recday+'.npy')                                     for mouse_recday in mouse_recdays_found]))


neurons_tuned=state_tuning
###i.e. phase/state tuned neurons that have had non-zero betas calculated for atleast half of the sessions

plt.rcParams["figure.figsize"] = (7,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True

for name in ['Predicted_Actual_correlation_mean','Predicted_Actual_correlation_nonzero_mean', 'Predicted_Actual_correlation_nonzero_strict_mean']:
    print(name)
    
    if use_tuned==True:
        
        corrs_allneurons=remove_nan(np.hstack(([np.load(Intermediate_object_folder_dropbox+addition2+addition+                                                        name+'_'+mouse_recday+'.npy') for mouse_recday in                                                mouse_recdays_found]))[neurons_tuned])
    else:

        
        corrs_allneurons=remove_nan(np.hstack(([np.load(Intermediate_object_folder_dropbox+addition2+addition+                                                        name+'_'+mouse_recday+'.npy') for mouse_recday in                                                mouse_recdays_found])))
        
    
        
    plt.hist(corrs_allneurons,bins=bins,color='grey')
    #plt.xlim(-1,1)
    plt.axvline(0,color='black',ls='dashed')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+addition+'GLM_analysis_'+name+'.svg',                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    #plt.boxplot(corrs_allneurons)
    #plt.axhline(0,ls='dashed',color='black')
    #plt.show()
    print(len(corrs_allneurons))
    print(st.ttest_1samp(corrs_allneurons,0))
    


# In[85]:


np.sum(state_tuning)


# In[ ]:





# In[ ]:





# In[93]:


###Per mouse analysis
GLM_anchoring_perday_dic=rec_dd()
Mice=np.load(Intermediate_object_folder_dropbox+'Mice.npy')
for subset in ['Predicted_Actual_correlation_mean','Predicted_Actual_correlation_nonzero_mean', 'Predicted_Actual_correlation_nonzero_strict_mean']:
    for mouse in Mice:
        mouse_recdays_bool=np.asarray([mouse in mouse_recday for mouse_recday in mouse_recdays_found])
        mouse_recdays_mouse=np.asarray(mouse_recdays_found)[mouse_recdays_bool]

        if len(mouse_recdays_mouse)==0:
            continue
        per_mouse_betas=[]
        for mouse_recday in mouse_recdays_mouse:
            #state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday]
            state_tuning=np.load(Intermediate_object_folder_dropbox+'State_zmax_bool_'+mouse_recday+'.npy')
            neurons_tuned=state_tuning
            betas_day=np.load(Intermediate_object_folder_dropbox+                                                        subset+'_'+mouse_recday+'.npy')[neurons_tuned]
            per_mouse_betas.append(betas_day)
        per_mouse_betas=np.hstack((per_mouse_betas))
        
        ttest_res=st.ttest_1samp(remove_nan(per_mouse_betas),0)
        GLM_anchoring_perday_dic['per_mouse_subsetted'][subset][mouse]=per_mouse_betas
        GLM_anchoring_perday_dic['per_mouse_subsetted_mean'][subset][mouse]=np.nanmean(per_mouse_betas)
        GLM_anchoring_perday_dic['per_mouse_subsetted_sem'][subset][mouse]=st.sem(per_mouse_betas,nan_policy='omit')
        GLM_anchoring_perday_dic['per_mouse_subsetted_ttest'][subset][mouse]=ttest_res


# In[ ]:





# In[94]:



for subset in ['Predicted_Actual_correlation_mean','Predicted_Actual_correlation_nonzero_mean', 'Predicted_Actual_correlation_nonzero_strict_mean']:
    per_mouse_betas_means=dict_to_array(GLM_anchoring_perday_dic['per_mouse_subsetted_mean'][subset])
    per_mouse_betas_sems=dict_to_array(GLM_anchoring_perday_dic['per_mouse_subsetted_sem'][subset])
    per_mouse_betas_ttest=dict_to_array(GLM_anchoring_perday_dic['per_mouse_subsetted_ttest'][subset])
    Mice=np.asarray(list(GLM_anchoring_perday_dic['per_mouse_subsetted_mean'][subset].keys()))
    
    plt.errorbar(per_mouse_betas_means,np.arange(len(per_mouse_betas_means)),xerr=per_mouse_betas_sems,ls='none',
            marker='o',color='grey')
    plt.axvline(0,ls='dashed',color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'GLM_analysis_permouse_'+subset+'.svg',                bbox_inches = 'tight', pad_inches = 0)
    
    plt.show()
    print(np.column_stack((Mice,per_mouse_betas_means)))
    print(per_mouse_betas_ttest)
    
    print(st.ttest_1samp(per_mouse_betas_means,0))
    
    per_mouse_betas_means_nonan=remove_nan(per_mouse_betas_means)
    num_positive=len(np.where(per_mouse_betas_means_nonan>0)[0])
    
    print(two_proportions_test(num_positive, len(per_mouse_betas_means_nonan),                               len(per_mouse_betas_means_nonan)*0.5, len(per_mouse_betas_means_nonan)))
    
    print(st.binom_test(x=num_positive, n=len(per_mouse_betas_means_nonan), p=0.5, alternative='greater'))


# In[ ]:





# In[67]:


mouse_recdays_found


# In[23]:


recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')


for mouse_recday in recording_days_:
    xx=np.load(Intermediate_object_folder_dropbox+'Place_policy_forward_'                                            +mouse_recday+'.npy',allow_pickle=True) 
    print(mouse_recday)
    print(np.sum(~xx))
    
print(len(recording_days_))


# In[92]:


recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')

for direction in ['forward','reverse']:
    Place_policy_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'Place_policy_'+direction+'_'                                            +mouse_recday+'.npy',                                           allow_pickle=True) for mouse_recday in mouse_recdays_found]))


    #neurons_tuned=np.logical_and(np.logical_and(state_tuning,phase_tuning),half_used)
    neurons_tuned=state_tuning
    ###i.e. phase/state tuned neurons that have had non-zero betas calculated for atleast half of the sessions

    neurons_not_policy=~Place_policy_tuning

    for name in ['Predicted_Actual_correlation_mean']:
        print(name)

        #corrs_allneurons=remove_nan(concatenate_complex2(dict_to_array\
        #                                             (GLM_anchoring_dic[name]))[neurons_not_policy])

        corrs_allneurons=remove_nan(np.hstack(([np.load(Intermediate_object_folder_dropbox+name+                                                        '_'+mouse_recday+'.npy') for mouse_recday in                                                    mouse_recdays_found]))[neurons_not_policy])




        plt.hist(corrs_allneurons,bins=bins,color='grey')
        #plt.xlim(-1,1)
        
        plt.axvline(0,color='black',ls='dashed')
        plt.tick_params(axis='both',  labelsize=20)
        plt.tick_params(width=2, length=6)
        plt.savefig(Ephys_output_folder_dropbox+'GLM_analysis_Policyremoved_'+direction+'.svg',                    bbox_inches = 'tight', pad_inches = 0)
        plt.show()

        #plt.boxplot(corrs_allneurons)
        #plt.axhline(0,ls='dashed',color='black')
        #plt.show()
        print(len(corrs_allneurons))
        print(st.ttest_1samp(corrs_allneurons,0))


# In[ ]:





# In[ ]:





# In[ ]:


###Testing individual examples

mouse_recday='ah03_12082021_13082021'
neuron_order_indx=0
##for mouse_recday in ['ah04_01122021_02122021', 'ah04_05122021_06122021',
#       'ah04_07122021_08122021', 'ah04_09122021_10122021','me11_01122021_02122021',
#       'me11_05122021_06122021', 'me11_07122021_08122021']:
#for neuron_order_indx in np.arange(10):

awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'session_behaviour_'+mouse_recday+'.npy')
awake_sessions=np.load(Intermediate_object_folder_dropbox+'session_'+mouse_recday+'.npy')


num_sessions=len(awake_sessions_behaviour)
non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
regressors_flat_allTasks=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_regressors'                                     +mouse_recday+'.npy',allow_pickle=True)
num_non_repeat_ses_found=len(regressors_flat_allTasks)

found_ses=[]
for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
    try:
        Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
        found_ses.append(ses_ind)

    except:
        print('Files not found for session '+str(ses_ind))
        continue
num_neurons=len(Neuron_raw)


#thresholds=GLM_anchoring_dic['significance_thresholds'][mouse_recday]
#Corrs_=GLM_anchoring_dic['Predicted_Actual_correlation_mean'][mouse_recday]
#Anchored_=np.where(Corrs_>thresholds)[0]
State_tuned=np.where(np.load(Intermediate_object_folder_dropbox+'State_tuned_'+mouse_recday+'.npy')==True)[0]
Phase_tuned=np.where(np.load(Intermediate_object_folder_dropbox+'Phase_tuned_'+mouse_recday+'.npy')==True)[0]
State_phase_tuned=np.intersect1d(State_tuned,Phase_tuned)
half_used=np.where(GLM_anchoring_dic['half_used_bool'][mouse_recday]==True)[0]

Predicted_Actual_correlation_mean=GLM_anchoring_dic['Predicted_Actual_correlation_mean'][mouse_recday]

sorted_indices_=np.flip(np.argsort(Predicted_Actual_correlation_mean))

sorted_indices=sorted_indices_[~np.isnan(Predicted_Actual_correlation_mean[sorted_indices_])]

sorted_indices=sorted_indices[np.in1d(sorted_indices,State_tuned)]
sorted_indices=sorted_indices[np.in1d(sorted_indices,Phase_tuned)]
#if len(sorted_indices)<(neuron_order_indx+1):
#    continue
neuron=sorted_indices[neuron_order_indx]

fontsize=10

mean_corr=Predicted_Actual_correlation_mean[neuron]

print(mouse_recday)
print(neuron)
print(mean_corr)


fig1, f1_axes = plt.subplots(figsize=(15, 7.5),ncols=len(non_repeat_ses), constrained_layout=True,                                subplot_kw={'projection': 'polar'})
fig2, f2_axes = plt.subplots(figsize=(15, 7.5),ncols=len(non_repeat_ses), constrained_layout=True,                                subplot_kw={'projection': 'polar'})


for ses_ind_ind in np.arange(num_non_repeat_ses_found):
    ax1=f1_axes[ses_ind_ind]
    ax2=f2_axes[ses_ind_ind]


    #print(ses_ind_ind)
    ses_ind_actual=found_ses[ses_ind_ind]

    #regressors_ses=GLM_anchoring_prep_dic['regressors'][mouse_recday][ses_ind_ind]
    #location_ses=GLM_anchoring_prep_dic['Location'][mouse_recday][ses_ind_ind]
    #Actual_activity_ses=GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]
    
    regressors_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_regressors'                                     +mouse_recday+'.npy',allow_pickle=True)[ses_ind_ind]
    location_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Location'                                     +mouse_recday+'.npy',allow_pickle=True)[ses_ind_ind]
    Actual_activity_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Neuron'                                     +mouse_recday+'.npy',allow_pickle=True)[ses_ind_ind]

    Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
    Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25

    Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]
    coeffs_ses_neuron=GLM_anchoring_dic['coeffs_all'][mouse_recday][neuron][ses_ind_ind]

    coeffs_ses_neuron_=GLM_anchoring_dic['coeffs_all'][mouse_recday][neuron][ses_ind_ind]
    coeffs_ses_neuron=np.copy(coeffs_ses_neuron_)
    coeffs_ses_neuron[coeffs_ses_neuron_<0]=0

    Predicted_activity_ses_neuron=np.sum(regressors_ses*coeffs_ses_neuron,axis=1)#[0]
    Predicted_activity_ses_neuron_scaled=Predicted_activity_ses_neuron*(    np.mean(Actual_activity_ses_neuron)/(np.mean(Predicted_activity_ses_neuron)+np.min(Predicted_activity_ses_neuron)))

    if np.isnan(np.nanmean(Predicted_activity_ses_neuron))==False:# and np.nanmean(Predicted_activity_ses_neuron)>0:
        ## prediction for all neurons/entire regression matrix
        Actual_norm_=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                                        smoothing=False, return_mean=False)
        Predicted_norm_=raw_to_norm(Predicted_activity_ses_neuron,Trial_times_conc,                                           smoothing=False, return_mean=False)
        Predicted_scaled_norm=raw_to_norm(Predicted_activity_ses_neuron_scaled,Trial_times_conc,                                           smoothing=False)

    else:
        print('Not calculated - betas go to zero')


    coeffs_reshaped=np.reshape(coeffs_ses_neuron, (num_phases*num_nodes,num_lags))

    #plt.matshow(coeffs_reshaped,vmin=0)
    #for n in np.arange(num_nodes):
    #    plt.axhline(3*n-0.5,color='white',ls='dashed')
    #plt.show()

    Actual_norm=np.nanmean(Actual_norm_,axis=0)
    Actual_norm_sem=st.sem(Actual_norm_,axis=0,nan_policy='omit')

    Predicted_norm=np.nanmean(Predicted_norm_,axis=0)
    Predicted_norm_sem=st.sem(Predicted_norm_,axis=0,nan_policy='omit')


    Actual_norm_smoothed=smooth_circular(Actual_norm)
    Actual_norm_sem_smoothed=smooth_circular(Actual_norm_sem)
    Predicted_norm_smoothed=smooth_circular(Predicted_norm)
    Actual_norm_sem_smoothed=smooth_circular(Actual_norm_sem)
    Predicted_norm_sem_smoothed=smooth_circular(Predicted_norm_sem)
    scaling_factor=(np.max(Actual_norm_smoothed)/np.max(Predicted_norm_smoothed))
    Predicted_scaled_norm_smoothed=Predicted_norm_smoothed*scaling_factor
    Predicted_norm_sem_smoothed=Predicted_norm_sem_smoothed*scaling_factor

    polar_plot_stateX2(Actual_norm_smoothed,Actual_norm_smoothed+Actual_norm_sem_smoothed,                       Actual_norm_smoothed-Actual_norm_sem_smoothed,labels='angles',color='blue',                          ax=ax1,repeated=False,fontsize=fontsize)
    polar_plot_stateX2(Predicted_scaled_norm_smoothed,Predicted_scaled_norm_smoothed+Predicted_norm_sem_smoothed                      ,Predicted_scaled_norm_smoothed-Predicted_norm_sem_smoothed,labels='angles',                              ax=ax2,repeated=False,fontsize=fontsize,color='red')
plt.tight_layout()
fig1.savefig(Ephys_output_folder_dropbox+'Example_cells/Taskmaps_'+mouse_recday+'_neuron_'+str(neuron)+            '.svg', bbox_inches = 'tight', pad_inches = 0)
plt.tight_layout()
fig2.savefig(Ephys_output_folder_dropbox+'Example_cells/PredictedTaskmaps_'+mouse_recday+'_neuron_'+str(neuron)+            '.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()

ses_ind_ind=1
coeffs_ses_neuron_=GLM_anchoring_dic['coeffs_all'][mouse_recday][neuron][ses_ind_ind]
coeffs_ses_neuron=np.copy(coeffs_ses_neuron_)
coeffs_ses_neuron[coeffs_ses_neuron_<0]=0
coeffs_reshaped=np.reshape(coeffs_ses_neuron, (num_phases*num_nodes,num_lags))
plt.matshow(coeffs_reshaped,vmin=0)
for n in np.arange(num_nodes):
    plt.axhline(3*n-0.5,color='white',ls='dashed')
plt.savefig(Ephys_output_folder_dropbox+'Example_cells/GLM_matrix_'+mouse_recday+'_neuron_'+str(neuron)+            '.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[24]:


num_locations=9


# In[45]:


###where are the anchors for each neuron - from GLM - crossvalidated

limited=False ##if true restricts lags to single trial, if false extends lags beyond this (see below)
if limited==True:
    addition2=''
else:
    addition2='_beyond'
    num_repeats=2
    
Poisson_regression=True
if Poisson_regression==True:
    addition='Poisson_'
else:
    addition=''

N=3
num_phases=3
num_nodes=num_locations=9
num_lags=12
Anchor_topN_GLM_crossval_dic=rec_dd()
day_type='combined_ABCDonly'
for mouse_recday in mouse_recdays_found:
    print(mouse_recday)
    #coeffs_all=GLM_anchoring_dic['coeffs_all'][mouse_recday]
    coeffs_all=np.load(Intermediate_object_folder_dropbox+addition+'GLM_anchoring_coeffs_all_'                                           +addition2+mouse_recday+'.npy',allow_pickle=True)
    num_ses_used=np.shape(coeffs_all)[1]
    num_neurons=len(coeffs_all)
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)
    
    found_ses=[]
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            found_ses.append(ses_ind)

        except:
            print('Files not found for session '+str(ses_ind))
            continue
    
     
    for ses_ind_ind, ses_ind in enumerate(non_repeat_ses): #range(num_ses_used):
        if ses_ind not in found_ses:
            continue
        anchor_lags_allneurons=np.zeros((num_neurons,N,3))
        anchor_lags_allneurons[:]=np.nan
        for neuron in range(num_neurons):
            coeffs_ses_neuron=coeffs_all[neuron,ses_ind_ind]
            
            if np.nanmean(coeffs_ses_neuron)==0:
                continue
            
            indices_sorted=np.flip(np.argsort(coeffs_ses_neuron))
            indices_sorted_nonan=indices_sorted[~np.isnan(coeffs_ses_neuron[indices_sorted])]
            topN_indices=indices_sorted_nonan[:N]

            
            if limited==True:
                coeffs_indices_reshaped=np.reshape(np.arange(num_phases*num_nodes*num_lags),                                                   (num_phases*num_nodes,num_lags))
            else:
                coeffs_indices_reshaped=np.reshape(np.arange(num_phases*num_nodes*num_lags*num_repeats),                                                   (num_phases*num_nodes,num_lags*num_repeats))
            
            
            topN_indices_reshaped=np.vstack(([np.hstack((np.where(coeffs_indices_reshaped==topN_indices[ii])))                                              for ii in range(len(topN_indices))]))

            lags=topN_indices_reshaped[:,1]
            anchor_phases=topN_indices_reshaped[:,0]%num_phases
            anchor_locations=topN_indices_reshaped[:,0]//num_phases
            anchor_lags=np.column_stack((anchor_phases,anchor_locations,lags))

            anchor_lags_allneurons[neuron]=anchor_lags

        anchor_lags_allneurons_stacked=np.vstack((anchor_lags_allneurons))

        for anchor_phase in np.arange(num_phases):
            for anchor_location in np.arange(num_nodes):
                anchored_indices=np.where(np.logical_and(anchor_lags_allneurons_stacked[:,0]==anchor_phase,                                        anchor_lags_allneurons_stacked[:,1]==anchor_location))[0]
                lags=anchor_lags_allneurons_stacked[anchored_indices,2]
                neurons=anchored_indices//N

                Anchor_topN_GLM_crossval_dic['Neurons_per_anchor'][mouse_recday][ses_ind_ind]                [anchor_phase][anchor_location]=np.column_stack((neurons,lags))
                
        Anchor_topN_GLM_crossval_dic['Anchors_per_neuron'][mouse_recday][ses_ind_ind]=anchor_lags_allneurons


# In[ ]:





# In[47]:


###where are the anchors for each neuron - from GLM
limited=False ##if true restricts lags to single trial, if false extends lags beyond this (see below)
if limited==True:
    addition2=''
else:
    addition2='_beyond'
    num_repeats=2
    
Poisson_regression=True
if Poisson_regression==True:
    addition='Poisson_'
else:
    addition=''
N=3
Anchor_topN_GLM_dic=rec_dd()
for mouse_recday in mouse_recdays_found:
    print(mouse_recday)
    coeffs_all=np.load(Intermediate_object_folder_dropbox+addition+'GLM_anchoring_coeffs_all_'                                           +addition2+mouse_recday+'.npy',allow_pickle=True)
    num_ses_used=np.shape(coeffs_all)[1]
    num_neurons=len(coeffs_all)
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)    
     
    anchor_lags_allneurons=np.zeros((num_neurons,N,3))
    anchor_lags_allneurons[:]=np.nan
    for neuron in range(num_neurons):
        coeffs_ses_neuron=np.nanmean(coeffs_all[neuron],axis=0)

        if np.nanmean(coeffs_ses_neuron)==0:
            continue

        indices_sorted=np.flip(np.argsort(coeffs_ses_neuron))
        indices_sorted_nonan=indices_sorted[~np.isnan(coeffs_ses_neuron[indices_sorted])]
        topN_indices=indices_sorted_nonan[:N]

        #coeffs_indices_reshaped=np.reshape(np.arange(num_phases*num_nodes*num_lags), (num_phases*num_nodes,num_lags))
        

        if limited==True:
            coeffs_indices_reshaped=np.reshape(np.arange(num_phases*num_nodes*num_lags),                                               (num_phases*num_nodes,num_lags))
        else:
            coeffs_indices_reshaped=np.reshape(np.arange(num_phases*num_nodes*num_lags*num_repeats),                                               (num_phases*num_nodes,num_lags*num_repeats))
        
        
        topN_indices_reshaped=np.vstack(([np.hstack((np.where(coeffs_indices_reshaped==topN_indices[ii])))                                          for ii in range(len(topN_indices))]))

        lags=topN_indices_reshaped[:,1]
        anchor_phases=topN_indices_reshaped[:,0]%num_phases
        anchor_locations=topN_indices_reshaped[:,0]//num_phases
        anchor_lags=np.column_stack((anchor_phases,anchor_locations,lags))

        anchor_lags_allneurons[neuron]=anchor_lags

    anchor_lags_allneurons_stacked=np.vstack((anchor_lags_allneurons))

    for anchor_phase in np.arange(num_phases):
        for anchor_location in np.arange(num_nodes):
            anchored_indices=np.where(np.logical_and(anchor_lags_allneurons_stacked[:,0]==anchor_phase,                                    anchor_lags_allneurons_stacked[:,1]==anchor_location))[0]
            lags=anchor_lags_allneurons_stacked[anchored_indices,2]
            neurons=anchored_indices//N

            Anchor_topN_GLM_dic['Neurons_per_anchor'][mouse_recday][anchor_phase][anchor_location]            =np.column_stack((neurons,lags))

    Anchor_topN_GLM_dic['Anchors_per_neuron'][mouse_recday]=anchor_lags_allneurons


# In[51]:


limited=False ##if true restricts lags to single trial, if false extends lags beyond this (see below)
if limited==True:
    addition2=''
else:
    addition2='_beyond'
    num_repeats=2
    
Poisson_regression=True
if Poisson_regression==True:
    addition='Poisson_'
else:
    addition=''

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams["figure.figsize"] = (8,6)

All_neuron_betas=[]
for mouse_recday in mouse_recdays_found:
    coeffs_all=np.load(Intermediate_object_folder_dropbox+addition+'GLM_anchoring_coeffs_all_'                                           +addition2+mouse_recday+'.npy',allow_pickle=True)
    mean_betas_neurons=np.nanmean(coeffs_all,axis=1)
    
    np.save(Intermediate_object_folder_dropbox+addition+addition2+'GLM_anchoring_coeffs_mean'                                           +mouse_recday+'.npy',mean_betas_neurons)
    
    All_neuron_betas.append(mean_betas_neurons)
    

All_neuron_betas=np.vstack((All_neuron_betas))
All_neuron_betas_mean=np.nanmean(All_neuron_betas,axis=0)
#coeffs_neuron_reshaped=All_neuron_betas_mean.reshape((num_locations*num_phases,num_lags))


if limited==True:
    coeffs_neuron_reshaped=All_neuron_betas_mean.reshape((num_locations*num_phases,num_lags))
else:
    coeffs_neuron_reshaped=All_neuron_betas_mean.reshape((num_locations*num_phases,num_lags*num_repeats))
plt.matshow(coeffs_neuron_reshaped,vmin=0)
for n in np.arange(num_nodes):
    plt.axhline(3*n-0.5,color='white',ls='dashed')
plt.show()
#print(coeffs_neuron_reshaped)

All_neuron_betas_clean=All_neuron_betas[[np.nanmean(All_neuron_betas[neuron])>0 for neuron in                                        range(len(All_neuron_betas))]]
max_bins=np.nanargmax(All_neuron_betas_clean,axis=1)
Peak_boolean=np.zeros((len(All_neuron_betas_clean),len(All_neuron_betas_clean.T)))
Peak_boolean_topN=np.zeros((len(All_neuron_betas_clean),len(All_neuron_betas_clean.T)))
for neuron in range(len(All_neuron_betas_clean)):
    topN_neuron=np.flip(np.argsort(All_neuron_betas_clean[neuron]))[:N]
    Peak_boolean[neuron,max_bins[neuron]]=1
    
    Peak_boolean_topN[neuron,topN_neuron]=1
    
Sum_peak_boolean=np.sum(Peak_boolean,axis=0)

if limited==True:
    Sum_peak_boolean_reshaped=Sum_peak_boolean.reshape((num_locations*num_phases,num_lags))
else:
    Sum_peak_boolean_reshaped=Sum_peak_boolean.reshape((num_locations*num_phases,num_lags*num_repeats))
    

plt.matshow(Sum_peak_boolean_reshaped,vmin=0)
for n in np.arange(num_nodes):
    plt.axhline(3*n-0.5,color='white',ls='dashed')
plt.show()
#print(Sum_peak_boolean_reshaped)

if limited==True:
    plt.bar(np.arange(num_lags),np.nansum(Sum_peak_boolean_reshaped,axis=0),color='black')
else:
    plt.bar(np.arange(num_lags*num_repeats),np.nansum(Sum_peak_boolean_reshaped,axis=0),color='black')
    
    plt.show()

Sum_peak_boolean=np.sum(Peak_boolean_topN,axis=0)

if limited==True:
    Sum_peak_boolean_reshaped=Sum_peak_boolean.reshape((num_locations*num_phases,num_lags))
else:
    Sum_peak_boolean_reshaped=Sum_peak_boolean.reshape((num_locations*num_phases,num_lags*num_repeats))



plt.matshow(Sum_peak_boolean_reshaped,vmin=0)
for n in np.arange(num_nodes):
    plt.axhline(3*n-0.5,color='white',ls='dashed')
plt.show()

if limited==True:
    plt.bar(np.arange(num_lags),np.nansum(Sum_peak_boolean_reshaped,axis=0),color='black')
else:
    plt.bar(np.arange(num_lags*num_repeats),np.nansum(Sum_peak_boolean_reshaped,axis=0),color='black')
    
plt.savefig(Ephys_output_folder_dropbox+'GLM_Anchor_analysis_lags.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
#print(Sum_peak_boolean_reshaped)

sum_across_anchors=np.nansum(Sum_peak_boolean_reshaped,axis=1)
sum_across_anchors_reshaped=sum_across_anchors.reshape(9,3)

plt.matshow(sum_across_anchors_reshaped.T,vmin=0,cmap='Reds')


# In[34]:


plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False


maze_hist=np.zeros((num_phases,3,3))
maze_hist[:]=np.nan

for phase_ in np.arange(num_phases):
    for location_ in np.arange(num_locations):
        maze_hist[phase_,Task_grid[location_,0],Task_grid[location_,1]]=sum_across_anchors_reshaped[location_,phase_]
for phase_ in np.arange(num_phases):
    plt.matshow(maze_hist[phase_],vmin=0,cmap='Blues')
    plt.colorbar(orientation='vertical',fraction=.1)
    plt.savefig(Ephys_output_folder_dropbox+'GLM_Anchor_analysis_locations_phase'+str(phase_)+'.svg' ,                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    
for ii in np.arange(len(maze_hist)):
    print(st.entropy(np.concatenate(maze_hist[ii]),base=len(np.concatenate(maze_hist[ii]))))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[69]:


num_anatomy_bins=4
num_channels_neuropixels=384
bin_size=num_channels_neuropixels/num_anatomy_bins
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')

mouse_recdays_found_anchoring=[]
for mouse_recday in recording_days_:
    print(mouse_recday)
    try:
        np.load(Intermediate_object_folder_dropbox+'_channel_num_neuron_'+mouse_recday+'.npy')
        mouse_recdays_found_anchoring.append(mouse_recday)
    except:
        print('Not found')


Anchoring_anatomy_dic=rec_dd()
for name in ['Predicted_Actual_correlation_mean','Predicted_Actual_correlation_nonzero_mean', 'Predicted_Actual_correlation_nonzero_strict_mean']:
    print(name)
    
    for mouse_recday in mouse_recdays_found_anchoring:

    
        state_tuning=np.load(Intermediate_object_folder_dropbox+'State_'+mouse_recday+'.npy',allow_pickle=True)
        corrs_allneurons=np.load(Intermediate_object_folder_dropbox+                                                        name+'_'+mouse_recday+'.npy')[state_tuning]

        channel_num_corrected=np.load(Intermediate_object_folder_dropbox+'_channel_num_neuron_'+mouse_recday+'.npy')
        anatomy_bin_neuron=((channel_num_corrected-1)//bin_size).astype(int)
        anatomy_bin_neuron_tuned=anatomy_bin_neuron[state_tuning]
        
        ##n.b. lower channel numbers=deeper channels

        measure_anat_bins=np.hstack(([np.nanmean(corrs_allneurons[anatomy_bin_neuron_tuned==anat_bin])            if len(corrs_allneurons[anatomy_bin_neuron_tuned==anat_bin])>0 else np.nan for anat_bin                                       in np.arange(num_anatomy_bins)]))

        Anchoring_anatomy_dic[name][mouse_recday]=measure_anat_bins

for mouse_recday in mouse_recdays_found_anchoring:
    lags_day=Anchor_topN_GLM_dic['Anchors_per_neuron'][mouse_recday][:,0][:,2]
    state_tuning=np.load(Intermediate_object_folder_dropbox+'State_'+mouse_recday+'.npy',allow_pickle=True)
    lags_day_tuned=lags_day[state_tuning]
    channel_num_corrected=np.load(Intermediate_object_folder_dropbox+'_channel_num_neuron_'+mouse_recday+'.npy')
    anatomy_bin_neuron=((channel_num_corrected-1)//bin_size).astype(int)
    anatomy_bin_neuron_tuned=anatomy_bin_neuron[state_tuning]

    forward_lag_bins=np.hstack(([np.nanmean(lags_day_tuned[anatomy_bin_neuron_tuned==anat_bin])        if len(lags_day_tuned[anatomy_bin_neuron_tuned==anat_bin])>0 else np.nan for anat_bin                                   in np.arange(num_anatomy_bins)]))
    
    circular_lag_bins=np.hstack(([np.rad2deg(st.circmean(np.deg2rad(remove_nan(lags_day_tuned[    anatomy_bin_neuron_tuned==anat_bin]*30))))    if len(lags_day_tuned[anatomy_bin_neuron_tuned==anat_bin])>0 else np.nan for anat_bin                                   in np.arange(num_anatomy_bins)]))

    Anchoring_anatomy_dic['forward_lags'][mouse_recday]=forward_lag_bins
    Anchoring_anatomy_dic['circular_lags'][mouse_recday]=circular_lag_bins


# In[ ]:





# In[68]:


anat_bin=0


# In[62]:


measure_anat_bins


# In[ ]:





# In[31]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
from scipy.stats import tukey_hsd

for measure in ['Predicted_Actual_correlation_mean','Predicted_Actual_correlation_nonzero_mean', 'Predicted_Actual_correlation_nonzero_strict_mean']:
    print(measure)

    Measure_prop_anat_mean=np.nanmean(dict_to_array(Anchoring_anatomy_dic[measure]),axis=0)
    Measure_prop_anat_sem=st.sem(dict_to_array(Anchoring_anatomy_dic[measure]),nan_policy='omit',axis=0)
    plt.rcParams["figure.figsize"] = (3,6)
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    
    plt.errorbar(y=-np.arange(len(Measure_prop_anat_mean.T)),x=np.flip(Measure_prop_anat_mean),                 xerr=np.flip(Measure_prop_anat_sem),                marker='o',markersize=10,color='black')
    plt.axvline(0,ls='dashed',color='black',linewidth=4)
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.xlim(-0.1,0.9)
    
    plt.savefig(Ephys_output_folder_dropbox+'DV_vs_proportion_'+measure+'.svg' , bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    Measure_prop_anat=dict_to_array(Anchoring_anatomy_dic[measure])


    stats=st.f_oneway(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
    print(stats)
    
    if stats[1]<0.05:
        res = tukey_hsd(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
        print(res)


# In[70]:


measure='forward_lags'

Measure_prop_anat_mean=np.nanmean(dict_to_array(Anchoring_anatomy_dic[measure]),axis=0)
Measure_prop_anat_sem=st.sem(dict_to_array(Anchoring_anatomy_dic[measure]),nan_policy='omit',axis=0)
plt.rcParams["figure.figsize"] = (3,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.errorbar(y=-np.arange(len(Measure_prop_anat_mean.T)),x=np.flip(Measure_prop_anat_mean)*30,                 xerr=np.flip(Measure_prop_anat_sem)*30,                marker='o',markersize=10,color='black')
plt.axvline(0,ls='dashed',color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.xlim(0,360)
plt.savefig(Ephys_output_folder_dropbox+'DV_vs_proportion_'+measure+'.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()

Measure_prop_anat=dict_to_array(Anchoring_anatomy_dic[measure])

stats=st.f_oneway(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
print(stats)

if stats[1]<0.05:
    res = tukey_hsd(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                  remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
    print(res)


# In[ ]:


def circular_sem(a):
    if len(np.shape(a))==2:
        sem_=np.rad2deg(np.hstack(([st.circvar(remove_nan(a[:,ii]))/np.sqrt(len(remove_nan(a[:,ii])))                               for ii in range(len(a.T))])))
    elif len(np.shape(a))==1:
        sem_=np.rad2deg(st.circvar(remove_nan(a))/np.sqrt(len(remove_nan(a))))
        
    return(sem_)


# In[112]:


measure='circular_lags'

Measure_prop_anat_mean=np.rad2deg(st.circmean(np.deg2rad(dict_to_array(Anchoring_anatomy_dic[measure]),                                                        ),axis=0,nan_policy='omit'))
Measure_prop_anat_sem=circular_sem(dict_to_array(Anchoring_anatomy_dic[measure]))
plt.rcParams["figure.figsize"] = (3,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.errorbar(y=-np.arange(len(Measure_prop_anat_mean.T)),x=np.flip(Measure_prop_anat_mean),                 xerr=np.flip(Measure_prop_anat_sem),                marker='o',markersize=10,color='black')
plt.axvline(0,ls='dashed',color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.xlim(0,380)
plt.savefig(Ephys_output_folder_dropbox+'DV_vs_proportion_'+measure+'.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()

Measure_prop_anat=dict_to_array(Anchoring_anatomy_dic[measure])

stats=st.f_oneway(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
print(stats)

if stats[1]<0.05:
    res = tukey_hsd(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                  remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
    print(res)


# In[ ]:





# In[102]:





# In[109]:


Measure_prop_anat_mean


# In[113]:


Measure_prop_anat_sem


# In[ ]:





# In[ ]:


###Policy


# In[ ]:





# In[ ]:





# In[ ]:


recording_days_


# In[115]:


mouse=mouse_recday.split('_',1)[0]
rec_day=mouse_recday.split('_',1)[1]
file = open(Intermediate_object_folder+'all_mice_exp.pickle','rb')
structure_probabilities_exp = pickle.load(file)
file.close()

###Baseline (one-step) transition probabilities
step_no=1
N_step_pr=structure_probabilities_exp[mouse][str(step_no)].values
N_step_pr[:,0]=(N_step_pr[:,0]).astype(int)
median_pr=np.median(N_step_pr[:,3])
High_pr_bool=N_step_pr[:,3]>median_pr
Low_pr_bool=~High_pr_bool


# In[ ]:


N_step_pr[:,0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


mouse_recdays_found=[]
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')
for mouse_recday in recording_days_:
    try:
        np.load(Intermediate_object_folder_dropbox+'Predicted_Actual_correlation_mean_'+mouse_recday+'.npy')
        mouse_recdays_found.append(mouse_recday)
    except:
        'Not found'
        
print(len(recording_days_))
print(len(mouse_recdays_found))

print(np.setdiff1d(recording_days_,mouse_recdays_found))
xx=np.hstack(([np.load(Intermediate_object_folder_dropbox+'Predicted_Actual_correlation_mean_'+mouse_recday+'.npy')               for mouse_recday in mouse_recdays_found]))

print(len(xx))
len(remove_nan(xx))


# In[79]:


###Predicting policy from neuronal activity\n",
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')
tt=time.time()
close_to_anchor_bins_90=[0,1,2,11,10,9]
close_to_anchor_bins_30=[0,11]

use_prefphase=True ###if set to false correlations are calculated seperately for each phase and then averaged\n",
use_mean=True ##use normalised, averaged activity for correlations - if true uses mean for each state in each task\n",
###if false, uses trial by trial means for each state\n",
##paramaters
num_bins=90
num_states=4
num_phases=3
num_nodes=9
num_lags=12
neuron_thr=10

use_proportion=True
proportion_location_thr=0.5
remove_repeated=True ###i.e. only look at changes in policy\n",
use_anchored=True
use_X_pr=False
prob_level='Low'
step_no=1

use_neuron_thr=False
classifier='logistic'
#shift=True ##null distribution
remake_dic=False

if remake_dic==True:
    Policy_prediction_dic=rec_dd()
for shift in [False]:#,False]:
    for mouse_recday in mouse_recdays_found:
        print(mouse_recday)

        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]


        ###Baseline (one-step) transition probabilities
        if use_X_pr==True:
            file = open(Intermediate_object_folder+'all_mice_exp.pickle','rb')
            structure_probabilities_exp = pickle.load(file)
            file.close()

            N_step_pr=structure_probabilities_exp[mouse][str(step_no)].values
            N_step_pr[:,0]=(N_step_pr[:,0]).astype(int)
            median_pr=np.median(N_step_pr[:,3])
            High_pr_bool=N_step_pr[:,3]>median_pr
            Low_pr_bool=~High_pr_bool

        awake_sessions_behaviour=np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
        awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
        sessions=np.load(Intermediate_object_folder_dropbox+'Task_num_'+mouse_recday+'.npy')
        Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy', allow_pickle=True)

        state_tuned_bool=np.load(Intermediate_object_folder_dropbox+'State_'+mouse_recday+'.npy',                                allow_pickle=True)

        anchoring_corr_all=np.load(Intermediate_object_folder_dropbox+'Predicted_Actual_correlation_'+mouse_recday+'.npy')
        anchoring_corr=np.nanmean(anchoring_corr_all,axis=1)

        anchoring_bool=anchoring_corr>0

        if use_anchored==True:
            neurons_used_bool=anchoring_bool
        else: 
            neurons_used_bool=state_tuned_bool

        num_sessions=len(awake_sessions_behaviour)
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        regressors_flat_allTasks=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_regressors'                                     +mouse_recday+'.npy',allow_pickle=True)

        found_ses=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            try:
                Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                found_ses.append(ses_ind)

            except:
                print('Files not found for session '+str(ses_ind))
                continue

        num_non_repeat_ses_found=len(regressors_flat_allTasks)


        state_zmax=np.load(Intermediate_object_folder_dropbox+'State_zmax_'+mouse_recday+'.npy')



        normalised_location_flat_all=[]
        normalised_activity_means_flatz_all=[]

        for ses_ind_ind in np.arange(num_non_repeat_ses_found):
            ses_ind_actual=found_ses[ses_ind_ind]

            normalised_activity=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
            normalised_location=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')

            regressors_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_regressors'                                     +mouse_recday+'.npy',allow_pickle=True)[ses_ind_ind]
            location_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Location'                                             +mouse_recday+'.npy',allow_pickle=True)[ses_ind_ind]
            Actual_activity_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_prep_dic_Neuron'                                             +mouse_recday+'.npy',allow_pickle=True)[ses_ind_ind]

            coeffs_ses=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_coeffs_all_'                                           +mouse_recday+'.npy',allow_pickle=True)[:,ses_ind_ind]

            coeffs_reshaped_all=np.asarray(([np.reshape(coeffs_ses[neuron], (num_phases*num_nodes,num_lags))                             for neuron in np.arange(len(coeffs_ses))]))


            ###Binning/flattening locations
            normalised_location_noedge=np.copy(normalised_location)
            normalised_location_noedge[normalised_location_noedge>9]=np.nan

            if use_X_pr==True:
                ###removing 
                normalised_location_noedge_flat=np.concatenate(normalised_location_noedge)
                starts=np.logical_and((normalised_location_noedge_flat[1:]!=normalised_location_noedge_flat[:-1]),                                      ~np.isnan(normalised_location_noedge_flat[1:]))
                ends=np.logical_and((normalised_location_noedge_flat[1:]!=normalised_location_noedge_flat[:-1]),                                    ~np.isnan(normalised_location_noedge_flat[:-1]))
                full_run_x=normalised_location_noedge_flat[np.where(starts==True)[0]+1]
                full_run_y=normalised_location_noedge_flat[np.where(ends==True)[0]]


                if prob_level=='High':
                    X_pr_bool=High_pr_bool
                elif prob_level=='Low':
                    X_pr_bool=Low_pr_bool


                X_pr_choices=np.concatenate([X_pr_bool[np.where(np.logical_and(N_step_pr[:,1]==full_run_x[ii],                                                                    N_step_pr[:,0]==full_run_y[ii]))[0]]                                               if len(np.where(np.logical_and(N_step_pr[:,1]==full_run_x[ii],                                                                    N_step_pr[:,0]==full_run_y[ii]))[0])>0
                                               else [np.nan]
                for ii in range(len(full_run_x))])

                starts_times=np.where(starts==True)[0][:-1]
                ends_times=np.where(ends==True)[0][1:]

                if len(ends_times)>len(starts_times):
                    ends_times=ends_times[:len(starts_times)]
                elif len(ends_times)<len(starts_times):
                    starts_times=starts_times[:len(end_times)]


                starts_times_X_pr=starts_times[X_pr_choices[:-1]==1]
                ends_times_X_pr=ends_times[X_pr_choices[:-1]==1]

                X_prob_transition_true=np.repeat(False,len(normalised_location_noedge_flat))
                for ii in range(len(starts_times_X_pr)):
                    X_prob_transition_true[starts_times_X_pr[ii]:ends_times_X_pr[ii]]=True 

                normalised_location_noedge_flat_Xpr=np.copy(normalised_location_noedge_flat)
                normalised_location_noedge_flat_Xpr[X_prob_transition_true==False]=np.nan

                normalised_location_noedge=np.reshape(normalised_location_noedge_flat_Xpr,                                                      (np.shape(normalised_location_noedge)[0],                                                       np.shape(normalised_location_noedge)[1]))


            location_perphase_actual=np.vstack(([np.hstack((st.mode(normalised_location_noedge[ii].reshape(num_lags,            len(normalised_location_noedge[ii])//num_lags),axis=1)[0]))            for ii in range(len(normalised_location_noedge))]))

            location_perphase_actual_proportions=np.vstack(([np.hstack((st.mode(normalised_location_noedge[ii].                                                                                reshape(num_lags,            len(normalised_location_noedge[ii])//num_lags),axis=1)[1]/(len(normalised_location_noedge[ii])//num_lags)))            for ii in range(len(normalised_location_noedge))]))

            location_repeated_prev_trial=np.vstack(([location_perphase_actual[ii]!=location_perphase_actual[ii-1]                    if ii>0
                    else np.repeat(True,len(location_perphase_actual[ii]))
                    for ii in np.arange(len(location_perphase_actual))]))
            location_repeated_prev_trial_flat=np.concatenate(location_repeated_prev_trial)


            normalised_location_flat=np.concatenate(location_perphase_actual)
            normalised_location_flat[:num_lags]=np.nan ##naning out first trial - clocks not set
            normalised_location_proportion_flat=np.concatenate(location_perphase_actual_proportions)

            if use_proportion==True:
                normalised_location_flat[normalised_location_proportion_flat<proportion_location_thr]=np.nan
                ###i.e. removing any locations where animal wasnt there for atleast threshold percent of the time in 
                ##a given bin

            if remove_repeated==True:
                normalised_location_flat[location_repeated_prev_trial_flat==False]=np.nan
                ##i.e. removing

            ###Binning/flattening activity
            normalised_activity_means=np.asarray([np.nanmean(np.reshape(normalised_activity[neuron],                    (np.shape(normalised_activity[neuron])[0],                    num_lags, np.shape(normalised_activity[neuron])[1]//num_lags)),axis=2)                                                  for neuron in np.arange(np.shape(normalised_activity)[0])])

            normalised_activity_means=normalised_activity_means[neurons_used_bool]  ##using neuron boolean

            normalised_activity_means_flat=np.asarray([np.hstack((normalised_activity_means[neuron]))                      for neuron in np.arange(len(normalised_activity_means))])

            normalised_activity_means_flatz=st.zscore(normalised_activity_means_flat,axis=1)

            #if shift==True:
            #    shifts=np.asarray([random.randint(1,num_lags) for ii in range(len(normalised_activity_means_flatz))])
            #    normalised_activity_means_flatz_copy=np.copy(normalised_activity_means_flatz)
            #    normalised_activity_means_flatz_shifted=indep_roll(normalised_activity_means_flatz_copy,shifts)
            #    normalised_activity_means_flatz_used=normalised_activity_means_flatz_shifted
            #else:
            #    normalised_activity_means_flatz_used=normalised_activity_means_flatz


            normalised_location_flat_all.append(normalised_location_flat)
            normalised_activity_means_flatz_all.append(normalised_activity_means_flatz)

        if shift==True:
            num_iterations=100
        else:
            num_iterations=1

        for iteration in np.arange(num_iterations):
            accuracy_test_all=[]
            predicted_all=[]
            actual_all=[]
            for lag in np.arange(num_lags):
                X_allsessions=[]
                y_allsessions=[]

                for ses_ind_ind in np.arange(num_non_repeat_ses_found):
                    normalised_location_flat=normalised_location_flat_all[ses_ind_ind]
                    normalised_activity_means_flatz=normalised_activity_means_flatz_all[ses_ind_ind]

                    if shift==True:
                        shifts=np.asarray([random.randint(1,num_lags) for ii in                                           range(len(normalised_activity_means_flatz))])
                        normalised_activity_means_flatz_copy=np.copy(normalised_activity_means_flatz)
                        normalised_activity_means_flatz_shifted=indep_roll(normalised_activity_means_flatz_copy,shifts)
                        normalised_activity_means_flatz_used=normalised_activity_means_flatz_shifted
                    else:
                        normalised_activity_means_flatz_used=normalised_activity_means_flatz



                    X_all=[]
                    y_all=[]


                    for start in np.arange(len(normalised_location_flat)-num_lags):
                        X=normalised_activity_means_flatz_used[:,start]
                        future_steps=normalised_location_flat[start:start+num_lags]
                        y=future_steps[lag]

                        if ~np.isnan(y)==True:

                            X_all.append(X)
                            y_all.append(y)

                    if len(X_all)<2:
                        print('Not enough valid trials')
                        continue
                    X_all=np.vstack((X_all))
                    y_all=np.vstack((y_all))


                    X_allsessions.append(X_all)
                    y_allsessions.append(y_all)

                X_allsessions=np.asarray(X_allsessions)
                y_allsessions=np.asarray(y_allsessions)


                All_ses=np.arange(len(X_allsessions))
                accuracy_test_lag=[]
                predicted_lag=[]
                actual_lag=[]
                for test_ses in All_ses:
                    train_ses=np.setdiff1d(All_ses,test_ses)


                    X_train=np.vstack((X_allsessions[[train_ses]].squeeze()))
                    y_train=np.vstack((y_allsessions[[train_ses]].squeeze()))

                    X_test=X_allsessions[test_ses]
                    y_test=np.concatenate(y_allsessions[test_ses])


                    if classifier=='logistic':
                        model = LogisticRegression(solver='saga',penalty='elasticnet', C=0.1, l1_ratio=0.5)
                    elif classifier=='svm':
                        model = make_pipeline(StandardScaler(), SVC())

                    model.fit(X_train, y_train)



                    predicted_test=model.predict(X_test)

                    accuracy_test=np.sum(predicted_test==y_test)/len(y_test)
                    accuracy_test_lag.append(accuracy_test)
                    
                    predicted_lag.append(predicted_test)
                    actual_lag.append(y_test)
                    
                    

                accuracy_test_all.append(accuracy_test_lag)
                predicted_all.append(predicted_lag)
                actual_all.append(actual_lag)
            if shift==True:
                Policy_prediction_dic['Shifted'][iteration][mouse_recday]=accuracy_test_all
            else:
                Policy_prediction_dic['Real'][mouse_recday]=accuracy_test_all
                Policy_prediction_dic['Real_actual_locations'][mouse_recday]=actual_all
                Policy_prediction_dic['Real_predicted_locations'][mouse_recday]=predicted_all


# In[104]:


confusion_matrix_all=[]
for lag in np.arange(num_lags):
    print(lag)
    actual_day_ALL=[]
    predicted_day_ALL=[]
    for mouse_recday in Policy_prediction_dic['Real_actual_locations'].keys():
        actual_day=Policy_prediction_dic['Real_actual_locations'][mouse_recday]
        predicted_day=Policy_prediction_dic['Real_predicted_locations'][mouse_recday]
        actual_day_flat=np.hstack((actual_day[lag]))
        predicted_day_flat=np.hstack((predicted_day[lag]))
        
        actual_day_ALL.append(actual_day_flat)
        predicted_day_ALL.append(predicted_day_flat)
    actual_day_ALL=np.hstack((actual_day_ALL))
    predicted_day_ALL=np.hstack((predicted_day_ALL))
    
    confusion_matrix=[[np.sum(np.logical_and(actual_day_ALL==locationX,predicted_day_ALL==locationY))    for locationX in np.arange(9)+1] for locationY in np.arange(9)+1]
    confusion_matrix=np.asarray(confusion_matrix)

    #plt.matshow(confusion_matrix)
    #plt.show()
    
    confusion_matrix=[[np.sum(np.logical_and(actual_day_ALL==locationX,predicted_day_ALL==locationY))/                       np.sum(predicted_day_ALL==locationY)    for locationX in np.arange(9)+1] for locationY in np.arange(9)+1]
    confusion_matrix=np.asarray(confusion_matrix)

    plt.matshow(confusion_matrix)
    plt.show()
    
    confusion_matrix_all.append(confusion_matrix)
        


# In[108]:


confusion_matrix_distal=np.nanmean(confusion_matrix_all[3:],axis=0)
plt.matshow(confusion_matrix_distal)
plt.show()


# In[109]:


mouse_recday='ab03_23112023_24112023'

Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
np.shape(Neuron_raw)


# In[74]:


shift=True

accuracy_test_all=[]
for lag in np.arange(num_lags):
    X_allsessions=[]
    y_allsessions=[]

    for ses_ind_ind in np.arange(num_non_repeat_ses_found):
        normalised_location_flat=normalised_location_flat_all[ses_ind_ind]
        normalised_activity_means_flatz=normalised_activity_means_flatz_all[ses_ind_ind]

        if shift==True:
            shifts=np.asarray([random.randint(3,num_lags*2) for ii in                               range(len(normalised_activity_means_flatz))])
            normalised_activity_means_flatz_copy=np.copy(normalised_activity_means_flatz)
            normalised_activity_means_flatz_shifted=indep_roll(normalised_activity_means_flatz_copy,shifts)
            normalised_activity_means_flatz_used=normalised_activity_means_flatz_shifted
        else:
            normalised_activity_means_flatz_used=normalised_activity_means_flatz



        X_all=[]
        y_all=[]


        for start in np.arange(len(normalised_location_flat)-num_lags):
            X=normalised_activity_means_flatz_used[:,start]
            future_steps=normalised_location_flat[start:start+num_lags]
            y=future_steps[lag]

            if ~np.isnan(y)==True:

                X_all.append(X)
                y_all.append(y)

        if len(X_all)<2:
            print('Not enough valid trials')
            continue
        X_all=np.vstack((X_all))
        y_all=np.vstack((y_all))


        X_allsessions.append(X_all)
        y_allsessions.append(y_all)

    X_allsessions=np.asarray(X_allsessions)
    y_allsessions=np.asarray(y_allsessions)


    All_ses=np.arange(len(X_allsessions))
    accuracy_test_lag=[]
    for test_ses in All_ses:
        train_ses=np.setdiff1d(All_ses,test_ses)


        X_train=np.vstack((X_allsessions[[train_ses]].squeeze()))
        y_train=np.vstack((y_allsessions[[train_ses]].squeeze()))

        X_test=X_allsessions[test_ses]
        y_test=np.concatenate(y_allsessions[test_ses])


        if classifier=='logistic':
            model = LogisticRegression(solver='saga',penalty='elasticnet', C=0.1, l1_ratio=0.5)
        elif classifier=='svm':
            model = make_pipeline(StandardScaler(), SVC())

        model.fit(X_train, y_train)



        predicted_test=model.predict(X_test)

        accuracy_test=np.sum(predicted_test==y_test)/len(y_test)
        accuracy_test_lag.append(accuracy_test)

    accuracy_test_all.append(accuracy_test_lag)


# In[ ]:





# In[75]:


np.nanmean(accuracy_test_all)


# In[ ]:





# In[ ]:





# In[49]:


shifts=np.asarray([random.randint(1,num_lags) for ii in                   range(len(normalised_activity_means_flatz))])
normalised_activity_means_flatz_copy=np.copy(normalised_activity_means_flatz)
normalised_activity_means_flatz_shifted=indep_roll(normalised_activity_means_flatz_copy,shifts)


# In[50]:


shifts


# In[53]:


np.shape(normalised_activity_means_flatz)


# In[ ]:





# In[28]:


policy_pred_day_all=[]
policy_pred_day_shuff_all=[]
num_iterations=100

for mouse_recday in mouse_recdays_found:
    print(mouse_recday)
    policy_pred_day_=np.asarray(Policy_prediction_dic['Real'][mouse_recday])
    
    
    
    noplot_timecourseA(np.arange(len(policy_pred_day_)),policy_pred_day_.T,color='black')
    plt.axhline(1/9,color='grey',ls='dashed')
    plt.show()
    
    policy_pred_day_all.append(np.nanmean(policy_pred_day_,axis=1))
    
    policy_pred_day_perm_=np.asarray([Policy_prediction_dic['Shifted'][ii][mouse_recday]                                for ii in range(num_iterations)])
    
    policy_pred_day_shuff_all.append(np.nanmean(np.nanmean(policy_pred_day_perm_,axis=0),axis=1))


# In[76]:


1/0.12


# In[46]:


policy_pred_day_all=np.asarray(policy_pred_day_all)
policy_pred_day_shuff_all=np.asarray(policy_pred_day_shuff_all)

noplot_timecourseA(np.arange(len(policy_pred_day_all.T)),policy_pred_day_all,color='black')
plt.axhline(1/9,color='grey',ls='dashed')
#plt.show()


noplot_timecourseA(np.arange(len(policy_pred_day_shuff_all.T)),policy_pred_day_shuff_all,color='grey')
plt.axhline(1/9,color='grey',ls='dashed')
plt.show()


# In[77]:


real_pred=np.nanmean(policy_pred_day_all,axis=1)
shuff_pred=np.nanmean(policy_pred_day_shuff_all,axis=1)

bar_plotX([real_pred,shuff_pred],'none',0,0.2,'points','paired',0.025)
plt.show()
print(st.wilcoxon(real_pred,shuff_pred))

real_pred=np.nanmean(policy_pred_day_all[:,1:],axis=1)
shuff_pred=np.nanmean(policy_pred_day_shuff_all[:,1:],axis=1)

bar_plotX([real_pred,shuff_pred],'none',0,0.2,'points','paired',0.025)
plt.show()
print(st.wilcoxon(real_pred,shuff_pred))

real_pred=np.nanmean(policy_pred_day_all[:,3:-3],axis=1)
shuff_pred=np.nanmean(policy_pred_day_shuff_all[:,3:-3],axis=1)

bar_plotX([real_pred,shuff_pred],'none',0,0.2,'points','paired',0.025)
plt.show()
print(st.wilcoxon(real_pred,shuff_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#####ABCDE#####


# In[ ]:





# In[7]:


###generating lagged regressors
num_phases,num_nodes,num_lags=3,9,12
remove_edges=True

GLM_anchoring_prep_ABCDE_dic=rec_dd()

recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDE_days.npy')

for mouse_recday in recording_days_:
    print(mouse_recday)
    awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
    awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')

    #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
    #np.arange(len(awake_sessions_behaviour))])

    num_sessions=len(awake_sessions_behaviour)

    
    sessions=np.load(Intermediate_object_folder_dropbox+'Task_num_'+mouse_recday+'.npy')
    num_refses=len(np.unique(sessions))
    num_comparisons=num_refses-1
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 


    Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)



    #phases_conc_all_=[]
    #states_conc_all_=[]
    #Location_raw_eq_all_=[]
    #Neuron_raw_all_=[]

    regressors_flat_allTasks=[]
    Location_allTasks=[]
    Neuron_allTasks=[]
    
    if mouse_recday in ['ah04_19122021_20122021','me11_15122021_16122021']:
        addition='_'
    else:
        addition=''


    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):

        try:
            Neuron_raw=np.load(Intermediate_object_folder_dropbox+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+                               addition+'.npy')
            Location_raw=np.load(Intermediate_object_folder_dropbox+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+                                 addition+'.npy')
            Location_norm=np.load(Intermediate_object_folder_dropbox+'Location_'+mouse_recday+'_'+str(ses_ind)+                                  addition+'.npy',allow_pickle=True)
            XY_raw=np.load(Intermediate_object_folder_dropbox+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+                           addition+'.npy')
            speed_raw=np.load(Intermediate_object_folder_dropbox+'speed_'+mouse_recday+'_'+str(ses_ind)+                              '.npy')


            acceleration_raw_=np.diff(speed_raw)/0.025
            acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
            Trial_times=np.load(Intermediate_object_folder_dropbox+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+                                addition+'.npy')


        except:
            ('Trying ceph')
            try:
                Neuron_raw=np.load(Intermediate_object_folder_ceph+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+                                   addition+'.npy')
                Location_raw=np.load(Intermediate_object_folder_ceph+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+                                     addition+'.npy')
                Location_norm=np.load(Intermediate_object_folder_ceph+'Location_'+mouse_recday+'_'+str(ses_ind)+                                      addition+'.npy',allow_pickle=True)
                XY_raw=np.load(Intermediate_object_folder_ceph+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+                               addition+'.npy')
                speed_raw=np.load(Intermediate_object_folder_ceph+'speed_'+mouse_recday+'_'+str(ses_ind)+                                  '.npy')


                acceleration_raw_=np.diff(speed_raw)/0.025
                acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                Trial_times=np.load(Intermediate_object_folder_ceph+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+                                    addition+'.npy')
            except:
                print('Files not found for session '+str(ses_ind))
                continue

        len_variables=[]
        for variable in [Neuron_raw,Location_raw,Location_norm,XY_raw,speed_raw,Trial_times]:
            len_variables.append(len(variable))
        
        if np.min(len_variables)==0:
            print('Some files on dropbox empty - trying ceph')
            try:
                Neuron_raw=np.load(Intermediate_object_folder_ceph+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+                                   addition+'.npy')
                Location_raw=np.load(Intermediate_object_folder_ceph+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+                                     addition+'.npy')
                Location_norm=np.load(Intermediate_object_folder_ceph+'Location_'+mouse_recday+'_'+str(ses_ind)+                                      addition+'.npy',allow_pickle=True)
                XY_raw=np.load(Intermediate_object_folder_ceph+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+                               addition+'.npy')
                speed_raw=np.load(Intermediate_object_folder_ceph+'speed_'+mouse_recday+'_'+str(ses_ind)+                                  '.npy')


                acceleration_raw_=np.diff(speed_raw)/0.025
                acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                Trial_times=np.load(Intermediate_object_folder_ceph+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+                                    addition+'.npy')

            except:
                print('Atleast one incomplete  for session '+str(ses_ind))
                continue
             
            
        num_neurons=len(Neuron_raw)
        


        phases=np.load(Intermediate_object_folder_dropbox+'Phases_raw2_'+mouse_recday+'_'+str(ses_ind)+                       '.npy',allow_pickle=True)       
        phases_conc=concatenate_complex2(concatenate_complex2(phases))
        states=np.load(Intermediate_object_folder_dropbox+'States_raw_'+mouse_recday+'_'+str(ses_ind)+                       '.npy',allow_pickle=True)
        states_conc=concatenate_complex2(concatenate_complex2(states))
        times=np.load(Intermediate_object_folder_dropbox+'Times_from_reward_'+mouse_recday+'_'+str(ses_ind)+                      '.npy',allow_pickle=True)
        times_conc=concatenate_complex2(concatenate_complex2(times))
        times_conc_eq=times_conc[:len(phases_conc)]
        speed_raw_eq=speed_raw[:len(phases_conc)]
        acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
        Location_raw_eq=Location_raw[:len(phases_conc)]
        Location_norm_conc=np.concatenate(Location_norm)

        Neuron_raw_eq=Neuron_raw[:,:len(phases_conc)]

        if remove_edges==True:
            Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges


        ###Using the model to calculate the regressors

        Task_phasetminus1=-1
        locationtminus1=-1
        Module_anchor_progress_dic=rec_dd()
        #for mouse_recday in day_type_dicX['combined_ABCDonly']:
        #    for session in np.arange(num_sessions):

        ##Importing occupancy and making phase/state 
        #nodes_=occupancy_dic[mouse_recday][session]
        nodes=(Location_raw_eq-1).astype(int)
        len_bins=len(nodes)


        #structure=(np.asarray([st.mode(Location_norm_conc.reshape(int(len(Location_norm_conc)/360),360)[:,ii])[0][0]+1\
        #            for ii in np.arange(4)*90])).astype(int)
        structure=Tasks[ses_ind]
        ### NOT 0 based indexed (so 1 is location 1)

        ##Spatial locations
        num_spatial_locations=num_nodes

        ##Task states
        num_task_states=len(Tasks[ses_ind])
        states=states_conc


        ##Rewarded states
        rewarded_state0=S[structure[0]-1]
        rewarded_statet=rewarded_state0

        ##Task phase
        num_task_phases=3
        phases=phases_conc

        Task_phaset=0

        All_modules_primed=0

        ###conditions
        T=len(nodes)
        #num_trials_planning=5
        #planning_active=True
        #sweep_forward=True ###whether to sweep forward across attractor to find next goal when using planning
        multiple_bumps=True
        probabilistic_choice=False

        plot=False
        make_video=False

        ##plotting 
        if plot==True and make_video==False:
            plt.figure(figsize = (20,T/5))
            plotting_coords=np.asarray(list(product(np.arange(np.sqrt(T)), np.arange(np.sqrt(T)))))

        elif plot==True and make_video==True:
            fig = plt.figure()
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122, projection='polar')
            #fig, axs = plt.subplots(2)
            #fig2, ax2 = plt.subplots(subplot_kw=dict(projection='polar'))

        #plt.rcParams['animation.ffmpeg_path'] =\
        #r'C:\Users\moham\anaconda3\envs\DLC-GPU\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win64-v4.2.2'

        metadata = dict(title='Movie', artist='codinglikemad')
        #writer = PillowWriter(fps=5, metadata=metadata)
        #writer = FFMpegWriter(fps=5, metadata=metadata)

        ##Module to plot
        spatial_loc=6
        spatial_ind=spatial_loc-1
        phase_ind=2

        move_phase=0

        ##Relative task_states
        module_anchor_phases=np.zeros(num_spatial_locations)
        #module_anchor_initiated=np.zeros((num_spatial_locations,num_task_phases))
        module_anchor_progress=np.zeros((num_spatial_locations,num_task_phases,int(num_task_states*num_task_phases)))
        reward_status_all=np.zeros(T)
        module_anchor_progress_all2=np.zeros((T,num_spatial_locations,num_task_phases,                                              int(num_task_states*num_task_phases)))

        trial_no=-1

        if probabilistic_choice==True:
            condition_choice='probabilistic'
        else:
            condition_choice='deterministic'

        #with writer.saving(fig, Modelling_output_folder_dropbox+video_name, 100):
        for t in range(T):

            location_indt=int(nodes[t])
            Task_phaset=int(phases[t])
            Task_state_next=int(states[t])
            rewarded_statet=S[structure[int(Task_state_next)]-1]


            reward_status=np.sum(location_indt+1 in structure)
            reward_status_all[t]=int(reward_status)

            if t>=0:
                if Task_phaset!=Task_phasetminus1:
                    move_phase=1
                else:
                    move_phase=0

                if location_indt!=locationtminus1 and location_indt in np.arange(num_nodes):
                    move_location=1
                else:
                    move_location=0



            ###Moving "activity bump" along spatially anchored modules
            for location_ind_ in np.arange(num_spatial_locations): ##looping over modules by location
                for Task_phase_ in np.arange(num_task_phases): ##looping over modules by phase
                    if move_phase==1: #module_anchor_initiated[location_ind_,Task_phase_]==1: 
                        ##i.e. has module been initiated and has phase changed? 

                        ##Spatial/phase input
                        if location_indt==location_ind_ and Task_phaset==Task_phase_:
                            if multiple_bumps==True or np.sum(module_anchor_progress[location_ind_,Task_phase_])==0:
                                module_anchor_progress[location_ind_,Task_phase_,0]=prev_module_activity=1
                            elif multiple_bumps==False and np.sum(module_anchor_progress[location_ind_,Task_phase_])>0:
                                prev_module_activity=module_anchor_progress[location_ind_,Task_phase_,0]
                        else:
                            prev_module_activity=module_anchor_progress[location_ind_,Task_phase_,0]

                        ##Moving bumps(s) when phase changes
                        module_anchor_progress[location_ind_,Task_phase_]=np.roll(module_anchor_progress[location_ind_,                                                                                                         Task_phase_],1)

                        ##adjusting activity based on whether currently active bump received spatial/phase input
                        if module_anchor_progress[location_ind_,Task_phase_,1]>0: ##changed
                            if location_indt==location_ind_ and Task_phaset==Task_phase_:
                                current_module_activity=1 #-ReLU(1-prev_module_activity+0.5)
                            else:
                                current_module_activity=0#prev_module_activity*0.5
                            module_anchor_progress[location_ind_,Task_phase_,1]=current_module_activity


                    if move_phase==0 and move_location==1:
                        ##Spatial/phase input
                        if location_indt==location_ind_ and Task_phaset==Task_phase_:
                            if multiple_bumps==True or np.sum(module_anchor_progress[location_ind_,Task_phase_])==0:
                                module_anchor_progress[location_ind_,Task_phase_,1]=prev_module_activity=1
                            elif multiple_bumps==False and np.sum(module_anchor_progress[location_ind_,Task_phase_])>0:
                                prev_module_activity=module_anchor_progress[location_ind_,Task_phase_,0]
                        else:
                            prev_module_activity=module_anchor_progress[location_ind_,Task_phase_,0]


                        ##adjusting activity based on whether currently active bump received spatial/phase input
                        #if module_anchor_progress[location_ind_,Task_phase_,1]>0: ##changed
                        #    if location_indt==location_ind_ and Task_phaset==Task_phase_:
                        #        current_module_activity=1 #-ReLU(1-prev_module_activity+0.5)
                        #    else:
                        #        current_module_activity=prev_module_activity*0.5
                        #    module_anchor_progress[location_ind_,Task_phase_,1]=current_module_activity


            Task_phasetminus1=Task_phaset
            locationtminus1=location_indt

            module_anchor_progress_all2[t]=module_anchor_progress

        regressors=np.roll(module_anchor_progress_all2,-1,axis=3) 
        ###rolled back as module_anchor_progress is lagged forward by 1
        regressors=np.asarray(regressors)
        #Module_anchor_progress_dic[mouse_recday][session]=module_anchor_progress_all2
        regressors_flat=np.reshape(regressors, (regressors.shape[0], np.prod(regressors.shape[1:])))
        
        print(np.prod(regressors.shape[1:]))


        regressors_flat_allTasks.append(regressors_flat)
        Location_allTasks.append(Location_raw_eq)
        Neuron_allTasks.append(Neuron_raw_eq.T)

    regressors_flat_allTasks=np.asarray(regressors_flat_allTasks)
    Location_allTasks=np.asarray(Location_allTasks)
    Neuron_allTasks=np.asarray(Neuron_allTasks)
    
    GLM_anchoring_prep_ABCDE_dic['regressors'][mouse_recday]=regressors_flat_allTasks
    GLM_anchoring_prep_ABCDE_dic['Location'][mouse_recday]=Location_allTasks
    GLM_anchoring_prep_ABCDE_dic['Neuron'][mouse_recday]=Neuron_allTasks


# In[ ]:





# In[ ]:





# In[41]:


####Lagged regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet

tt=time.time()


#num_states=4
num_phases=3
num_nodes=9
#num_lags=12

use_prefphase=True #
regularize=True
alpha=0.01

recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDE_days.npy')

#GLM_anchoring_ABCDE_dic=rec_dd()
for mouse_recday in recording_days_:
    print(mouse_recday)
    
    if mouse_recday in ['ah04_19122021_20122021','me11_15122021_16122021']:
        addition='_'
    else:
        addition=''
    
    awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
    awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
    
    Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)


    num_sessions=len(awake_sessions_behaviour)
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    
    found_ses=[]
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+addition+'.npy')
            found_ses.append(ses_ind)

        except:
            print('Files not found for session '+str(ses_ind))
            continue
    num_neurons=len(Neuron_raw)
    
    
    
    ###nonrepeat sessions
    regressors_flat_allTasks=GLM_anchoring_prep_ABCDE_dic['regressors'][mouse_recday]
    Location_allTasks=GLM_anchoring_prep_ABCDE_dic['Location'][mouse_recday]
    Neuron_allTasks=GLM_anchoring_prep_ABCDE_dic['Neuron'][mouse_recday]
    
    num_non_repeat_ses_found=len(regressors_flat_allTasks)
    
    Tasks_nonrepeat=Tasks[found_ses]
    
    len_tasks=np.asarray([len(Tasks[ii]) for ii in range(len(Tasks))])
    ABCD_sessions=np.intersect1d(found_ses,np.where(len_tasks==4)[0])
    ABCDE_sessions=np.intersect1d(found_ses,np.where(len_tasks==5)[0])
    
    
    ses_array=ABCDE_sessions
    abstract_structure='ABCDE'
    num_states=len(abstract_structure)
    num_lags=int(num_states*num_phases)
    
    
    
    if len(GLM_anchoring_ABCDE_dic[abstract_structure]['coeffs_all'][mouse_recday])==num_neurons:
        print('Already analysed')
        continue
    
    try:
        coeffs_all=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_'+abstract_structure+'_coeffs_all'                        +'_'+mouse_recday+'.npy')
        if len(coeffs_all)==num_neurons:
            print('Already analysed')
            continue
    except:
        print('Analysing')
        
    
    
    
    coeffs_all=np.zeros((num_neurons,len(ses_array),num_phases*num_nodes*num_lags))
    coeffs_all[:]=np.nan



    for ses_ind_ind_test in np.arange(len(ses_array)):
        print(ses_ind_ind_test)
        ses_ind_actual=ses_array[ses_ind_ind_test]

        ses_ind_abstractstructure_order=list(np.where(len_tasks==len(abstract_structure))[0]).index(ses_ind_actual)

        training_sessions=np.setdiff1d(np.arange(len(ses_array)),ses_ind_ind_test)
        training_sessions_actual=ses_array[training_sessions]
        training_sessions_nonrepeat=[list(found_ses).index(training_sessions_actual[ii])                                     for ii in range(len(training_sessions_actual))]

        ##concatenating arrays
        regressors_flat_trainingTasks_=regressors_flat_allTasks[training_sessions_nonrepeat]
        Location_trainingTasks_=Location_allTasks[training_sessions_nonrepeat]
        Neuron_trainingTasks_=Neuron_allTasks[training_sessions_nonrepeat]

        regressors_flat_trainingTasks=np.vstack((regressors_flat_trainingTasks_))
        Location_trainingTasks=np.hstack((Location_trainingTasks_))
        Neuron_trainingTasks=np.vstack((Neuron_trainingTasks_)).T

        ##phase
        phase_peaks=np.load(Intermediate_object_folder_dropbox+abstract_structure+                            '_tuning_phase_boolean_max_'+mouse_recday+'.npy')[ses_ind_abstractstructure_order]

        pref_phase_neurons=np.argmax(phase_peaks,axis=1)
        #phases=np.load(Intermediate_object_folder_dropbox+'Phases_raw2_'+\
        #                                           mouse_recday+'_'+str(ses_ind_actual)+'.npy',allow_pickle=True)

        phases_conc_=np.hstack((np.vstack([np.load(Intermediate_object_folder_dropbox+'Phases_raw2_'+                                                   mouse_recday+'_'+str(ses)+'.npy',allow_pickle=True)                                           for ses in training_sessions_actual])))

        phases_conc=concatenate_complex2(phases_conc_)

        ####Doing the regression
        for neuron in np.arange(num_neurons):
            ##independent variables
            regressors_nonan=regressors_flat_trainingTasks[~np.isnan(Location_trainingTasks)]
            regressors_flat=np.reshape(regressors_nonan, (regressors_nonan.shape[0], np.prod(regressors_nonan.shape[1:])))
            #times_conc_eq_nonan=times_conc_eq[~np.isnan(Location_raw_eq)]
            #speed_raw_eq_nonan=speed_raw_eq[~np.isnan(Location_raw_eq)]
            #phases_conc_nonan=phases_conc[~np.isnan(Location_raw_eq)]
            #acceleration_raw_eq_nonan=acceleration_raw_eq[~np.isnan(Location_raw_eq)]

            ##dependent variable
            Neuron_raw_eq_neuron=Neuron_trainingTasks[neuron]
            Neuron_raw_eq_neuron_nonan=Neuron_raw_eq_neuron[~np.isnan(Location_trainingTasks)]


            ##subsetting by phase
            pref_phase=pref_phase_neurons[neuron]
            phases_conc_nonan=phases_conc[~np.isnan(Location_trainingTasks)]

            regressors_flat_prefphase=regressors_flat[phases_conc_nonan==pref_phase]
            Neuron_raw_eq_neuron_nonan_prefphase=Neuron_raw_eq_neuron_nonan[phases_conc_nonan==pref_phase]

            ###regression
            if use_prefphase==True:
                X = regressors_flat_prefphase
                y = Neuron_raw_eq_neuron_nonan_prefphase
            else:
                X = regressors_flat
                y = Neuron_raw_eq_neuron_nonan



            if regularize==True:
                reg = ElasticNet(alpha=alpha,positive=True).fit(X, y)
            else:
                reg = LinearRegression(positive=True).fit(X, y)


            coeffs_flat=reg.coef_
            coeffs_all[neuron,ses_ind_ind_test]=coeffs_flat

    GLM_anchoring_ABCDE_dic[abstract_structure]['coeffs_all'][mouse_recday]=coeffs_all
    np.save(Intermediate_object_folder_dropbox+'GLM_anchoring_'+abstract_structure+'_coeffs_all'                        +'_'+mouse_recday+'.npy',coeffs_all)
print(time.time()-tt)


# In[12]:





# In[ ]:





# In[34]:


###Calculating correlations between predicted and actual activity
tt=time.time()


use_prefphase=True ###if set to false correlations are calculated seperately for each phase and then averaged
use_mean=True ##use normalised, averaged activity for correlations - if true uses mean for each state in each task
###if false, uses trial by trial means for eahc state

Num_max=1 ##how many peaks should NOT be in the excluded regression columns

##paramaters
num_bins=90
#num_states=4
num_phases=3
num_nodes=9
#num_lags=12
smoothing_sigma=10
trial_start=0 ##which trial to start from

close_to_anchor_bins_90=[0,1,2,14,13,12]
ABCD_bins=list(np.arange(12))
close_to_anchor_bins_30=[0] ##[0,1,14]

redo=True

#Entropy_thr_all=np.nanmean(np.hstack((concatenate_complex2(dict_to_array(Entropy_dic['Entropy_thr'])))))

for mouse_recday in recording_days_:
    print(mouse_recday)
    
    if mouse_recday in ['ah04_19122021_20122021','me11_15122021_16122021']:
        addition='_'
    else:
        addition=''

    
    awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
    awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
    sessions=np.load(Intermediate_object_folder_dropbox+'Task_num_'+mouse_recday+'.npy')
    Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)
    
    num_sessions=len(awake_sessions_behaviour)
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    regressors_flat_allTasks=GLM_anchoring_prep_ABCDE_dic['regressors'][mouse_recday]
    num_non_repeat_ses_found=len(regressors_flat_allTasks)

    found_ses=[]
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+addition+'.npy')
            found_ses.append(ses_ind)

        except:
            print('Files not found for session '+str(ses_ind))
            continue
    num_neurons=len(Neuron_raw) 
    #num_state_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)\
    #          for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T
    
    #entropy_tuned_withinneuron=Entropy_dic['Entropy_actual'][mouse_recday]<Entropy_dic['Entropy_thr'][mouse_recday]
    #entropy_tuned_global=Entropy_dic['Entropy_actual'][mouse_recday]<Entropy_thr_all
    #entropy_tuned_all=np.logical_and(entropy_tuned_withinneuron,entropy_tuned_global)
    
    state_zmax=np.load(Intermediate_object_folder_dropbox+abstract_structure+'_State_zmax_'+mouse_recday+'.npy')
    
    
    len_tasks=np.asarray([len(Tasks[ii]) for ii in range(len(Tasks))])
    ABCD_sessions=np.intersect1d(found_ses,np.where(len_tasks==4)[0])
    ABCDE_sessions=np.intersect1d(found_ses,np.where(len_tasks==5)[0])
    
    
    ses_array=ABCDE_sessions
    abstract_structure='ABCDE'
    
    
    if redo==False:
        if len(GLM_anchoring_ABCDE_dic[abstract_structure]['Predicted_Actual_correlation'][mouse_recday])==num_neurons:
            print('Already analysed')
            continue

        try:
            Predicted_Actual_correlation=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_'+abstract_structure+                               '_Predicted_Actual_correlation'+'_'+mouse_recday+'.npy')
            if len(Predicted_Actual_correlation)==num_neurons:
                print('Already analysed')
                continue
        except:
            print('Analysing')
    else:
        print('Analysing')
    
    #for abstract_structure, ses_array in {'ABCD':ABCD_sessions, 'AB':AB_sessions}.items():
    num_states=len(abstract_structure)
    num_lags=int(num_states*num_phases)
    #coeffs_all=np.zeros((num_neurons,len(ses_array),num_phases*num_nodes*num_lags))
    #coeffs_all[:]=np.nan

    regressor_indices=np.arange(num_phases*num_nodes*num_lags)
    regressor_indices_reshaped=np.reshape(regressor_indices,(num_phases*num_nodes,num_lags))
    zero_indices=regressor_indices_reshaped[:,0]
    close_to_anchor_indices30=np.concatenate(regressor_indices_reshaped[:,close_to_anchor_bins_30])
    ABCD_indices90=np.concatenate(regressor_indices_reshaped[:,ABCD_bins])


    phase_norm_mean=np.tile(np.repeat(np.arange(num_phases),num_bins/num_phases),num_states)
    phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states,num_bins))


    corrs_all=np.zeros((num_neurons,len(ses_array)))
    corrs_all_nozero=np.zeros((num_neurons,len(ses_array)))
    corrs_all_nozero_strict=np.zeros((num_neurons,len(ses_array)))

    corrs_all[:]=np.nan
    corrs_all_nozero[:]=np.nan
    corrs_all_nozero_strict[:]=np.nan

    for ses_ind_ind in np.arange(len(ses_array)):          
        ses_ind_actual=ses_array[ses_ind_ind]

        ses_ind_nonrepeat=list(found_ses).index(ses_ind_actual)
        ses_ind_abstractstructure_order=list(np.where(len_tasks==len(abstract_structure))[0]).index(ses_ind_actual)


        regressors_ses=GLM_anchoring_prep_ABCDE_dic['regressors'][mouse_recday][ses_ind_nonrepeat]
        location_ses=GLM_anchoring_prep_ABCDE_dic['Location'][mouse_recday][ses_ind_nonrepeat]
        Actual_activity_ses=GLM_anchoring_prep_ABCDE_dic['Neuron'][mouse_recday][ses_ind_nonrepeat]

        #phase_peaks=np.load(Intermediate_object_folder_dropbox+'tuning_phase_boolean_max_'+mouse_recday+'_'+\
        #                            str(ses_ind_actual)+'.npy')
        phase_peaks=np.load(Intermediate_object_folder_dropbox+abstract_structure+                                '_tuning_phase_boolean_max_'+mouse_recday+'.npy')[ses_ind_abstractstructure_order]

        pref_phase_neurons=np.argmax(phase_peaks,axis=1)

        phases=np.load(Intermediate_object_folder_dropbox+'Phases_raw2_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy',                      allow_pickle=True)       
        phases_conc=concatenate_complex2(concatenate_complex2(phases))

        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+addition+'.npy')

        if trial_start>0:
            Trial_times=Trial_times[trial_start:]

        if len(Trial_times)<5:
            print('Not enough trials in session'+str(ses_ind_actual))
            continue


        Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25

        for neuron in np.arange(num_neurons):
            pref_phase=pref_phase_neurons[neuron]
            Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]
            #coeffs_ses_neuron_=GLM_anchoring_ABCDE_dic[abstract_structure]['coeffs_all'][mouse_recday]\
            #[neuron][ses_ind_ind]
            
            coeffs_ses_neuron_=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_'+abstract_structure+                                       '_coeffs_all'+'_'+mouse_recday+'.npy',allow_pickle=True)[neuron][ses_ind_ind]
            coeffs_ses_neuron=np.copy(coeffs_ses_neuron_)
            #coeffs_ses_neuron[coeffs_ses_neuron_<0]=0

            ###maximum indices
            indices_sorted=np.flip(np.argsort(coeffs_ses_neuron))
            indices_sorted_nonan=indices_sorted[~np.isnan(coeffs_ses_neuron[indices_sorted])]
            topN_indices=indices_sorted_nonan[:Num_max]

            Predicted_activity_ses_neuron=np.sum(regressors_ses*coeffs_ses_neuron,axis=1)
            Predicted_activity_ses_neuron_scaled=Predicted_activity_ses_neuron*(            np.mean(Actual_activity_ses_neuron)/np.mean(Predicted_activity_ses_neuron))

            #num_state_peaks_neuronses=num_state_peaks_all[neuron,ses_ind_ind]
            #entropy_tuned_neuronses=entropy_tuned_all[neuron,ses_ind_ind]
            state_zmax_neuronses=state_zmax[neuron,ses_ind_ind]

            if np.isnan(np.nanmean(Predicted_activity_ses_neuron))==False and            np.nanmean(Predicted_activity_ses_neuron)>0:# and state_zmax_neuronses<0.05:#num_state_peaks_neuronses>0:

                ## prediction for all neurons/entire regression matrix
                if use_prefphase==False:
                    Predicted_Actual_correlation_all=[]
                    for phase_ind in np.arange(num_phases):
                        Predicted_Actual_correlation_=                        st.pearsonr(Actual_activity_ses_neuron[phases_conc==phase_ind],                        Predicted_activity_ses_neuron[phases_conc==phase_ind])[0]
                        Predicted_Actual_correlation_all.append(Predicted_Actual_correlation_)
                    Predicted_Actual_correlation=np.nanmean(Predicted_Actual_correlation_all)
                else:
                    if use_mean==False:
                        Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                                num_states=len(abstract_structure),smoothing=False,return_mean=False)
                        Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron,Trial_times_conc,                                                           num_states=len(abstract_structure),                                                   smoothing=False,return_mean=False)

                        Actual_norm_means=np.concatenate([[np.nanmean(Actual_norm[trial,num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                                          for trial in np.arange(len(Actual_norm))])
                        Predicted_norm_means=np.concatenate([[np.nanmean(Predicted_norm                                                                         [trial,num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase])                            for ii in range(num_states)] for trial in np.arange(len(Predicted_norm))])
                        Predicted_Actual_correlation=st.pearsonr(Actual_norm_means,Predicted_norm_means)[0]
                    else:
                        Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                                                        num_states=len(abstract_structure), smoothing=False)
                        Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron,Trial_times_conc,                                                           num_states=len(abstract_structure), smoothing=False)

                        Actual_norm_means=np.asarray([np.nanmean(Actual_norm[num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])
                        Predicted_norm_means=np.asarray([np.nanmean(Predicted_norm[num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])
                        Predicted_Actual_correlation=st.pearsonr(Actual_norm_means,Predicted_norm_means)[0]

                corrs_all[neuron,ses_ind_ind]=Predicted_Actual_correlation

                ###prediction after removing 0 lagged betas
                coeffs_ses_neuron_copy1=np.copy(coeffs_ses_neuron_)
                coeffs_ses_neuron_copy1[close_to_anchor_indices30]=np.nan
                Predicted_activity_ses_neuron_nozero=np.nansum(regressors_ses*coeffs_ses_neuron_copy1,axis=1)

                #if np.nanargmax(coeffs_ses_neuron) in zero_indices:
                if np.sum(np.isin(topN_indices,close_to_anchor_indices30))>0:
                    Predicted_Actual_correlation_nozero=np.nan
                else:
                    if use_prefphase==False:
                        Predicted_Actual_correlation_nozero_all=[]
                        for phase_ind in np.arange(num_phases):
                            Predicted_Actual_correlation_nozero_=                            st.pearsonr(Actual_activity_ses_neuron[phases_conc==phase_ind],                            Predicted_activity_ses_neuron_nozero[phases_conc==phase_ind])[0]
                            Predicted_Actual_correlation_nozero_all.append(Predicted_Actual_correlation_nozero_)
                        Predicted_Actual_correlation_nozero=np.nanmean(Predicted_Actual_correlation_nozero_all)

                    else:
                        if use_mean==False:
                            Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron_nozero,Trial_times_conc,                                                num_states=len(abstract_structure), smoothing=False,return_mean=False)
                            Predicted_norm_means=np.concatenate([[np.nanmean(Predicted_norm                                                                             [trial,num_bins*ii:num_bins*(ii+1)]                                [phase_norm_mean_states[ii]==pref_phase])                                for ii in range(num_states)] for trial in np.arange(len(Predicted_norm))])
                            Predicted_Actual_correlation_nozero=st.pearsonr(Actual_norm_means,Predicted_norm_means)[0]
                        else:
                            Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron_nozero,Trial_times_conc,                                 num_states=len(abstract_structure),smoothing=False)

                            Predicted_norm_means=np.asarray([np.nanmean(Predicted_norm[num_bins*ii:num_bins*(ii+1)]                                [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])

                            Predicted_Actual_correlation_nozero=st.pearsonr(Actual_norm_means,Predicted_norm_means)[0]

                corrs_all_nozero[neuron,ses_ind_ind]=Predicted_Actual_correlation_nozero

                ###prediction after removing 90 degrees either side of 0 lag
                coeffs_ses_neuron_copy2=np.copy(coeffs_ses_neuron_)
                coeffs_ses_neuron_copy2[ABCD_indices90]=np.nan
                Predicted_activity_ses_neuron_nozero_strict=np.nansum(regressors_ses*coeffs_ses_neuron_copy2,axis=1)


                #np.nanargmax(coeffs_ses_neuron) in close_to_anchor_indices:
                if np.sum(np.isin(topN_indices,ABCD_indices90))>0: 
                    Predicted_Actual_correlation_nozero_strict=np.nan
                else:
                    if use_prefphase==False:
                        Predicted_Actual_correlation_nozero_strict_all=[]
                        for phase_ind in np.arange(num_phases):
                            Predicted_Actual_correlation_nozero_strict_=                            st.pearsonr(Actual_activity_ses_neuron[phases_conc==phase_ind],                            Predicted_activity_ses_neuron_nozero_strict[phases_conc==phase_ind])[0]
                            Predicted_Actual_correlation_nozero_strict_all.append(                            Predicted_Actual_correlation_nozero_strict_)
                        Predicted_Actual_correlation_nozero_strict=np.nanmean(                        Predicted_Actual_correlation_nozero_strict_all)
                    else:
                        if use_mean==False:
                            Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron_nozero_strict,Trial_times_conc,                                                num_states=len(abstract_structure), smoothing=False,return_mean=False)
                            Predicted_norm_means=np.concatenate([[np.nanmean(Predicted_norm                                                                             [trial,num_bins*ii:num_bins*(ii+1)]                                [phase_norm_mean_states[ii]==pref_phase])                                for ii in range(num_states)] for trial in np.arange(len(Predicted_norm))])
                            Predicted_Actual_correlation_nozero_strict=st.pearsonr(Actual_norm_means,                                                                                   Predicted_norm_means)[0]

                        else:
                            Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron_nozero_strict,Trial_times_conc,                                                num_states=len(abstract_structure), smoothing=False)
                            Predicted_norm_means=np.asarray([np.nanmean(Predicted_norm[num_bins*ii:num_bins*(ii+1)]                                [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])

                            Predicted_Actual_correlation_nozero_strict=st.pearsonr(Actual_norm_means,                                                                                   Predicted_norm_means)[0]

                corrs_all_nozero_strict[neuron,ses_ind_ind]=Predicted_Actual_correlation_nozero_strict

    corrs_mean=np.nanmean(corrs_all,axis=1)
    corrs_all_nozero_mean=np.nanmean(corrs_all_nozero,axis=1)
    corrs_all_nozero_strict_mean=np.nanmean(corrs_all_nozero_strict,axis=1)

    GLM_anchoring_ABCDE_dic[abstract_structure]['Predicted_Actual_correlation'][mouse_recday]=corrs_all
    GLM_anchoring_ABCDE_dic[abstract_structure]['Predicted_Actual_correlation_mean'][mouse_recday]=corrs_mean
    GLM_anchoring_ABCDE_dic[abstract_structure]['Predicted_Actual_correlation_nonzero_mean'][mouse_recday]=    corrs_all_nozero_mean
    GLM_anchoring_ABCDE_dic[abstract_structure]['Predicted_Actual_correlation_nonzero_strict_mean'][mouse_recday]=    corrs_all_nozero_strict_mean
    
print(time.time()-tt)


# In[ ]:


'''
reload coefficients and re-run while only removing first bin

'''


# In[40]:


mouse_recday
abstract_structure


# In[39]:


coeffs_ses_neuron_=np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_'+abstract_structure+                                       '_coeffs_all'+'_'+mouse_recday+'.npy',allow_pickle=True)#[neuron][ses_ind_ind]

len(coeffs_ses_neuron_)


# In[21]:


recording_days_


# In[32]:



mouse_recdays_found_ABCDE=recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDE_days.npy')
#mouse_recdays_found_ABCDE=['ab03_23112023_24112023', 'ah07_23112023_24112023']
print(len(mouse_recdays_found_ABCDE))

use_tuned=True
bins=50
use_strict=False ##if true uses p=0.01 threshold for state tuning
#phase_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'Phase_'+mouse_recday+'.npy')\
#                         for mouse_recday in mouse_recdays_found_ABCDE]))
    
state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'ABCDE_State_'+mouse_recday+'.npy')                                 for mouse_recday in mouse_recdays_found_ABCDE]))

state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'ABCDE_State_zmax_bool_'+mouse_recday+'.npy')                                 for mouse_recday in mouse_recdays_found_ABCDE]))

if use_strict==True:
    state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'ABCDE_State_zmax_bool_strict_'+                                     mouse_recday+'.npy') for mouse_recday in mouse_recdays_found_ABCDE]))

#half_used=np.hstack(([GLM_anchoring_dic['half_used_bool'][mouse_recday] for mouse_recday in\
#                         mouse_recdays_found]))


#neurons_tuned=np.logical_and(np.logical_and(state_tuning,phase_tuning),half_used)
neurons_tuned=state_tuning
###i.e. phase/state tuned neurons that have had non-zero betas calculated for atleast half of the sessions


for name in ['Predicted_Actual_correlation_mean','Predicted_Actual_correlation_nonzero_mean', 'Predicted_Actual_correlation_nonzero_strict_mean']:
    print(name)
    
    if use_tuned==True:
        
        corrs_allneurons=remove_nan(np.hstack(([np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_ABCDE_'+                                                        name+'_'+mouse_recday+'.npy') for mouse_recday in                                                mouse_recdays_found_ABCDE]))[neurons_tuned])
    else:

        
        corrs_allneurons=remove_nan(np.hstack(([np.load(Intermediate_object_folder_dropbox+'GLM_anchoring_ABCDE_'+                                                        name+'_'+mouse_recday+'.npy') for mouse_recday in                                                mouse_recdays_found_ABCDE])))
        
    
        
    plt.hist(corrs_allneurons,bins=bins,color='grey')
    #plt.xlim(-1,1)
    plt.axvline(0,color='black',ls='dashed')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'GLM_analysis_ABCDE_'+name+'.svg',                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    #plt.boxplot(corrs_allneurons)
    #plt.axhline(0,ls='dashed',color='black')
    #plt.show()
    print(len(corrs_allneurons))
    print(st.ttest_1samp(corrs_allneurons,0))
    


# In[ ]:





# In[ ]:





# In[29]:


for measure in GLM_anchoring_dic.keys():
    for mouse_recday in GLM_anchoring_dic[measure].keys():
        if len(GLM_anchoring_dic[measure][mouse_recday])>0:
            np.save(Intermediate_object_folder_dropbox+'GLM_anchoring_'+measure+'_'+mouse_recday+'.npy',                    GLM_anchoring_dic[measure][mouse_recday])

GLM_anchoring_ABCDE_dic['ABCDE'].keys()
for abstract_structure in GLM_anchoring_ABCDE_dic.keys():
    for measure in GLM_anchoring_ABCDE_dic[abstract_structure].keys():
        for mouse_recday in GLM_anchoring_ABCDE_dic[abstract_structure][measure].keys():
            if len(GLM_anchoring_ABCDE_dic[abstract_structure][measure][mouse_recday])>0:
                np.save(Intermediate_object_folder_dropbox+'GLM_anchoring_'+abstract_structure+'_'+measure                        +'_'+mouse_recday+'.npy', GLM_anchoring_ABCDE_dic[abstract_structure][measure][mouse_recday])


# In[ ]:





# In[ ]:





# In[ ]:


#SAVING FILES 

try: 
    os.mkdir(Intermediate_object_folder) 
except FileExistsError: 
    pass

objects_dic={'GLM_anchoring_dic':GLM_anchoring_dic,'GLM_anchoring_ABCDE_dic':GLM_anchoring_ABCDE_dic}

for name, dicX in objects_dic.items(): 
    data=dicX 
    data_filename_memmap = os.path.join(Intermediate_object_folder, name) 
    dump(data, data_filename_memmap)


# In[ ]:





# In[ ]:




