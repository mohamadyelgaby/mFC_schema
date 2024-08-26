#!/usr/bin/env python
# coding: utf-8

# In[17]:


###Folders
Data_folder='/Taskspace_abstraction/Data/'
Data_folder_P='/Taskspace_abstraction/Data/' ## if working in P
base_dropbox='C:/Users/moham/team_mouse Dropbox/Mohamady El-Gaby/'

#base_dropbox='D:/team_mouse Dropbox/Mohamady El-Gaby/'

Data_folder_dropbox=base_dropbox+'/Taskspace_abstraction/Data/' ##if working in C
Behaviour_output_folder = '/Taskspace_abstraction/Results/Behaviour/'
Ephys_output_folder = '/Taskspace_abstraction/Results/Ephys/'
Ephys_output_folder_dropbox = base_dropbox+'/Taskspace_abstraction/Results/Ephys/'
Intermediate_object_folder_dropbox = Data_folder_dropbox+'/Intermediate_objects/'

Intermediate_object_folder = Data_folder_dropbox+'/Intermediate_objects/'

Code_folder='/Taskspace_abstraction/Code/'


# In[18]:


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
from sklearn.linear_model import LinearRegression
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter1d
import warnings


# In[19]:


##Importing custom functions
module_path = os.path.abspath(os.path.join(Code_folder))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from mBaseFunctions import rec_dd, remove_empty, indicesX2, create_binary, unique_adjacent, dict_to_array, flatten,remove_empty, concatenate_complex2,mean_complex2, std_complex2, rand_jitter, rand_jitterX, bar_plotX, remove_nan,remove_nanX, column_stack_clean, polar_plot_state, positive_angle, circular_angle, plot_grouped_error,timestamp_to_binary, fill_nans, fill_nansX, angle_to_state, binned_array, binned_arrayX, plot_scatter,two_proportions_test, rearrange_matrix, matrix_triangle, scramble, smooth_circular, plot_spatial_maps, state_to_phase,middle_value,max_bin_safe, random_rotation, non_cluster_indices, cumulativeDist_plot, cumulativeDist_plot_norm,angle_to_distance, rotate, angle_to_stateX, range_ratio_peaks, equalize_rowsX, cross_corr_fast, plot_dendrogram,rotate, rank_repeat, edge_node_fill, split_mode, concatenate_states, predict_task_map, predict_task_map_policy,number_of_repeats, find_direction, Edge_grid, polar_plot_stateX, polar_plot_stateX2,arrange_plot_statecells_persession, arrange_plot_statecells, arrange_plot_statecells_persessionX, Task_grid_plotting2,Task_grid_plotting, Task_grid, Task_grid2, Edge_grid_coord, Edge_grid_coord2, direction_dic_plotting, plot_spatial_mapsX,angle_to_distance, rank_repeat, number_of_repeats, noplot_timecourseA, noplot_timecourseAx, noplot_timecourseB,noplot_timecourseBx, two_proportions_test, noplot_scatter, number_of_repeats_ALL, non_repeat_ses_maker,non_repeat_ses_maker_old


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


def num_of_repeats2(MyList):
    my_dict = {i:list(MyList).count(i) for i in MyList}
    
    return(np.asarray([my_dict[element] for element in MyList]))
'''def non_repeat_ses_maker_(mouse_recday):
    sessions=Task_num_dic[mouse_recday]
    num_trials_day=dict_to_array(Num_trials_dic2[mouse_recday])
    non_repeat_ses=[]
    for session_unique in np.unique(sessions):
        sessions_ses=np.where(sessions==session_unique)[0]
        num_sessions=len(sessions_ses)
        if num_sessions==1:
            non_repeat_ses.append(sessions_ses[0])
        else:
            sessions_ses_trials=sessions_ses[np.where(num_trials_day[sessions_ses]!=0)]
            non_repeat_ses.append(sessions_ses_trials[0])
    non_repeat_ses=np.asarray(non_repeat_ses)
    return(non_repeat_ses)'''

###function to plot scatter plots (e.g. comparing assembly strength at correct vs incorrect dispensers)
def plot_scatter(x,y,name):
    plt.plot(x, y, 'o')
    z= [-10000, 0, 10000]
    plt.plot(z,z,'k--')

    xy=np.hstack((x,y))

    xmin=min(xy)-np.mean(xy)*0.1
    xmax=max(xy)+np.mean(xy)*0.1
    ymin=min(xy)-np.mean(xy)*0.1
    ymax=max(xy)+np.mean(xy)*0.1
    


    #plt.xlim(-0.2,0.2)
    #plt.ylim(-0.2,0.2)

    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    if name != 'none':
        plt.savefig(name)
    plt.show()

###function to plot scatter plots (e.g. comparing assembly strength at correct vs incorrect dispensers)
def noplot_scatter(x,y, color):
    plt.plot(x, y, 'o', color=color, alpha=0.7)
    z= [-10000, 0, 10000]
    plt.plot(z,z,'k--')

    xy=np.hstack((x,y))
    
    global xmin
    global xmax
    global ymin
    global ymax
    
    xmin=min(xy)-np.mean(xy)*0.1
    xmax=max(xy)+np.mean(xy)*0.1    
    ymin=min(xy)-np.mean(xy)*0.1
    ymax=max(xy)+np.mean(xy)*0.1

    


    #plt.xlim(-0.2,0.2)
    #plt.ylim(-0.2,0.2)

    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.gca().set_aspect('equal', adjustable='box')

def is_invertible(a):
     return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
    


# In[21]:


'''
FR_binned


'''


# In[22]:


#LOADING FILES - tracking, behaviour, Ephys raw, Ephys binned
tt=time.time()


try:
    os.mkdir(Intermediate_object_folder)
except FileExistsError:
    pass

dictionaries_list=['Num_trials_dic2','Xneuron_correlations','Xneuron_correlations2','Xneuron_phaseangle_dic','session_dic',                   'Xsession_correlations','Xsession_correlations2','edge_rate_matrices_dic',                   'cluster_dic','sampling_dic','recday_numbers_dic','day_type_dicX','Spatial_anchoring_dic',                  'Anchor_trial_dic','scores_dic','Task_num_dic','Sleepcorr_matrix_shifted_dic',                  'Tuned_dic','session_dic_behaviour','FR_shuff_dic','Sleepcorr_pairs_timebin_dic']

##,'binned_FR_dic'

for name in dictionaries_list:
    try:
        data_filename_memmap = os.path.join(Intermediate_object_folder, name)
        data = load(data_filename_memmap)#, mmap_mode='r')
        exec(name+'= data')
    except Exception as e:
        print(name)
        print(e)
        print('Not loaded')
print(time.time()-tt)


# In[23]:


##Importing meta Data
Mice_cohort_dic={'me03':2,'me04':2,'me05':2,'me06':2,'me08':3,'ah02':3,'ah03':3,'me10':4,'me11':4,'ah04':4,'ah05':4,                'ab03':6,'ah07':6} ##'me12':5,'me13':5,
Mice_recnode_dic={'me03':110,'me08':131,'ah02':129,'ah03':110}
Mouse_FPGAno={'me03':'109.0','me08':'121.0','ah02':'109.0','ah03':'109.0'}
Mice=np.asarray(list(Mice_cohort_dic.keys()))
Nonephys_mice=['me04','me05','me06','me12','me13']
Ephys_mice=np.setdiff1d(Mice,Nonephys_mice)

Cohort_ephys_type_dic={2:'Cambridge_neurotech',3:'Cambridge_neurotech',4:'Cambridge_neurotech',                      5:'Neuropixels',6:'Neuropixels'}

Mice_sleep=[mouse+'_sleep' for mouse in Ephys_mice]
Mice_withsleep=np.hstack((Mice,Mice_sleep))

Variable_dic=rec_dd()

for mouse in Mice_withsleep:
    cohort=Mice_cohort_dic[mouse[:4]]
    data_directory=Data_folder_dropbox+'/cohort'+str(cohort)+'/'#+mouse[:4]+'/'
    MetaDatafile_path=data_directory+'MetaData.xlsx - '+mouse+'.csv'
    with open(MetaDatafile_path, 'r') as f:
        MetaData = np.genfromtxt(f, delimiter=',',dtype=str, usecols=np.arange(0,18))
    MetaData_structure=MetaData[0]
    Include_mask=MetaData[1:,np.where(MetaData[0]=='Include')[0][0]]=='1' ##added 06/07/2021
    for indx, variable in enumerate(MetaData_structure):
        if variable != 'Weight_pre':
            Variable_dic[mouse][variable]=np.asarray(remove_empty(MetaData[1:,indx][Include_mask]))
        else:
            Variable_dic[mouse][variable]=MetaData[1:,indx][Include_mask]

structure_nums=(np.linspace(1,10,10)).astype(int)
states=np.asarray(['A','B','C','D'])
place_type_dic={1:2,2:1,3:2,4:1,5:0,6:1,7:2,8:1,9:2}

for mouse in Mice:
    Variables=Variable_dic[mouse]
    Structure_no_=Variables['Structure_no']
    Structure_no_[Structure_no_=='-']=-1


# In[24]:


day_type_dicX['combined_ABCDonly']


# In[ ]:





# In[25]:


Awake_sesind_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]
    rec_day1=rec_day.split('_',1)[0]
    rec_day2=rec_day.split('_',1)[1]
    Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
    Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

    mouse_recday1=mouse+'_'+rec_day1
    mouse_recday2=mouse+'_'+rec_day2
    
    Tasks=np.load(Intermediate_object_folder+'Task_data_'+mouse_recday+'.npy')
    Task_ind_it=0
    Task_ind_dup_all=np.zeros(len(Tasks))
    for Task_ind, Task in enumerate(Tasks):
        repeated_tasks=np.where((Tasks[:Task_ind] == (Task)).all(axis=1))[0]
        if len(repeated_tasks)==0:
            Task_ind_dup=Task_ind_it
            Task_ind_it+=1
        else:
            Task_ind_dup=Task_ind_dup_all[repeated_tasks[0]]
        Task_ind_dup_all[Task_ind]=Task_ind_dup
    
    awake_sessions=session_dic['awake'][mouse_recday]
    #first_day2_awakeses=np.asarray([np.logical_and(int(awake_sessions[ii][:2])<int(awake_sessions[ii-1][:2]),ii>0)\
    #        for ii in range(len(awake_sessions))])
    
    first_day2_ses_timestamp=Variable_dic[mouse]['Ephys'][np.where(Variable_dic[mouse]['Date']==Date2)[0]][0]
    first_day2_ses_ind=np.where(awake_sessions==first_day2_ses_timestamp)[0][0]
    first_day2_awakeses=np.repeat(False,len(awake_sessions))
    first_day2_awakeses[first_day2_ses_ind]=True

    Awake_sesind_dic[mouse_recday]=Task_ind_dup_all
    Awake_sesind_dic['first_day2_boolean'][mouse_recday]=first_day2_awakeses


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


###sleep stages 

sleep_stage_dic=rec_dd()


for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]
    rec_day1=rec_day.split('_',1)[0]
    rec_day2=rec_day.split('_',1)[1]
    Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
    Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

    mouse_recday1=mouse+'_'+rec_day1
    mouse_recday2=mouse+'_'+rec_day2
    
    awake_sessions=session_dic['awake'][mouse_recday]
    sleep_sessions=session_dic['sleep'][mouse_recday]
    all_sessions=session_dic['All'][mouse_recday]

    sleep_ses_inds=np.asarray([ses_ind for ses_ind, ses in enumerate(all_sessions) if ses in sleep_sessions])
    awake_ses_inds=np.asarray([ses_ind for ses_ind, ses in enumerate(all_sessions) if ses in awake_sessions])


    sleep_stages=np.asarray(np.repeat('preXXX',len(sleep_ses_inds)))

    for awake_ses_ind, all_ses_ind in enumerate(awake_ses_inds):
        stage='post'+str(int(Awake_sesind_dic[mouse_recday][awake_ses_ind]))
        sleep_stages[sleep_ses_inds>all_ses_ind]=stage
    
    ###Dealing with the first session of day 2 (or 3/4...etc) 
    first_day2_ses_timestamp=Variable_dic[mouse+'_sleep']['Ephys'][np.where(Variable_dic[mouse+'_sleep']['Date']                                                                            ==Date2)[0]][0]
    first_day2_ses_ind=np.where(sleep_sessions==first_day2_ses_timestamp)[0][0]
    first_day2_ses=np.repeat(False,len(sleep_sessions))
    first_day2_ses[first_day2_ses_ind]=True
    #first_day2_ses=np.asarray([np.logical_and(int(sleep_sessions[ii][:2])<int(sleep_sessions[ii-1][:2]),ii>0)\
    #        for ii in range(len(sleep_sessions))])
    #first_day2_ses_ind=np.where(first_day2_ses==True)[0][0]
    same_firstday2ses_bool_all=np.repeat(False,len(sleep_sessions))
    for sleep_ses_ind in np.arange(len(sleep_sessions)):
        same_firstday2ses_bool=np.logical_and(sleep_stages[sleep_ses_ind]==sleep_stages[first_day2_ses],                                     sleep_ses_ind==int(first_day2_ses_ind+1))[0]
        if same_firstday2ses_bool==True:
            first_day2_ses_ind+=1
        same_firstday2ses_bool_all[sleep_ses_ind]=same_firstday2ses_bool    
        
    sleep_stages[first_day2_ses]='preX2'
    sleep_stages[same_firstday2ses_bool_all]='preX2'

    sleep_stages[sleep_stages=='preXXX']='preX'
    sleep_stage_dic[mouse_recday]=np.asarray(sleep_stages)


# In[ ]:





# In[ ]:





# ###making session dics for combined days
# for mouse_recday in day_type_dicX['combined']:
#     print(mouse_recday)
#     
#     mouse=mouse_recday.split('_',1)[0]
#     rec_day=mouse_recday.split('_',1)[1]
#     rec_day1=rec_day.split('_',1)[0]
#     rec_day2=rec_day.split('_',1)[1]
#     Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
#     Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]
# 
#     mouse_recday1=mouse+'_'+rec_day1
#     mouse_recday2=mouse+'_'+rec_day2
# 
# 
#     single_days=[mouse_recday1,mouse_recday2]
#     
#     
#     for ses_type in session_dic.keys():
#         sessions_all=[]
#         sessions_behaviour_all=[]
#         for mouse_recday_single in single_days:
#             
#             sessions_=session_dic[ses_type][mouse_recday_single]
#             sessions_behaviour_=session_dic_behaviour[ses_type][mouse_recday_single]
#             
#             sessions_all.append(sessions_)
#             sessions_behaviour_all.append(sessions_behaviour_)
#         if len(session_dic[ses_type][mouse_recday])==0:
#             session_dic[ses_type][mouse_recday]=np.hstack((sessions_all))
#         if len(session_dic_behaviour[ses_type][mouse_recday])==0:
#             session_dic_behaviour[ses_type][mouse_recday]=np.hstack((sessions_behaviour_all))

# In[ ]:





# In[27]:


session_dic['sleep']['ah04_16122021']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


#sleep activity per stage

rescaling_factor=1 #1=no rescaling
Activity_sleep_stage_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    sleep_sessions=session_dic['sleep'][mouse_recday]
    All_sessions=session_dic['All'][mouse_recday]
    stages=np.unique(sleep_stage_dic[mouse_recday])
    print(mouse_recday)
    for stage_ind, stage in enumerate(stages):

        Activity_sleep_stage=[]
        for ses_ind, timestamp in enumerate(sleep_sessions):
            stage_ses=sleep_stage_dic[mouse_recday][ses_ind]
            if stage_ses==stage:
                All_session_ind=np.where(All_sessions==timestamp)[0][0]
                
                name='binned_FR_dic_'+mouse_recday+'_'+str(All_session_ind)
                data_filename_memmap = os.path.join(Intermediate_object_folder, name)
                try:
                    Activity=load(data_filename_memmap)
                except:
                    print('File not found for session: '+str(All_session_ind))
                    continue
                #Activity=binned_FR_dic[mouse_recday][All_session_ind]
                Activity_sleep_stage.append(Activity)
        Activity_sleep_stage=np.hstack(Activity_sleep_stage)
        
        
        
        len_activity_X=int((len(Activity_sleep_stage.T)//rescaling_factor)*rescaling_factor)
        Activity_sleep_stageX=binned_arrayX(Activity_sleep_stage[:,:len_activity_X],                                               int(len_activity_X/rescaling_factor),statistic='mean')
        
        Activity_sleep_stage_dic[mouse_recday][stage]=Activity_sleep_stageX


# In[29]:


np.shape(Activity_sleep_stage)


# In[ ]:





# In[ ]:





# In[32]:


###Boolean for firing rate
tt=time.time()
thr_ratio=0.2
FR_sleep_boolean_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    stages=np.unique(sleep_stage_dic[mouse_recday])
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    awake_sessions=session_dic['awake'][mouse_recday]
    All_sessions=session_dic['All'][mouse_recday]
    FR_means_all=np.zeros((len(awake_sessions),num_neurons))
    for indx, timestamp in enumerate(awake_sessions):
        ephys_indx=np.where(All_sessions==timestamp)[0][0]
        
        name='binned_FR_dic_'+mouse_recday+'_'+str(ephys_indx)
        data_filename_memmap = os.path.join(Intermediate_object_folder, name)
        try:
            Activity=load(data_filename_memmap)
        except:
            print('File not found for session: '+str(All_session_ind))
            continue
        FR_means=np.mean(Activity,axis=1)*40
        #FR_means=np.mean(binned_FR_dic[mouse_recday][ephys_indx],axis=1)*40
        FR_means_all[indx]=FR_means

    FR_sleep_means_all=np.zeros((len(stages),num_neurons))
    for stage_ind, stage in enumerate(stages):
        #print(stage)
        FR_sleep_means=np.mean(Activity_sleep_stage_dic[mouse_recday][stage],axis=1)*40
        FR_sleep_means_all[stage_ind]=FR_sleep_means

    FR_sleep_boolean=np.any(FR_sleep_means_all>np.mean(FR_means_all,axis=0)*thr_ratio, axis=0)
        
    FR_sleep_boolean_dic[mouse_recday]=FR_sleep_boolean
    
print(time.time()-tt)


# In[ ]:





# In[15]:


FR_sleep_boolean_dic['ab03_01092023_02092023']


# In[ ]:





# In[33]:


###Sleep corr matrix
Sleepcorr_matrix_dic=rec_dd()

for mouse_recday in day_type_dicX['combined_ABCDonly']:
    stages=np.unique(sleep_stage_dic[mouse_recday])
    for stage_ind, stage in enumerate(stages):
        Activity_sleep_stage=Activity_sleep_stage_dic[mouse_recday][stage]
        corr_matrix_stage=np.corrcoef(Activity_sleep_stage)

        Sleepcorr_matrix_dic[stage][mouse_recday]=corr_matrix_stage


# In[ ]:





# In[17]:


###Shifted correlations

tt=time.time()

shifts=np.linspace(-40,40,81) ## i.e. correlating with n bins in the future 
#Sleepcorr_matrix_shifted_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    stages=np.unique(sleep_stage_dic[mouse_recday])
    for stage_ind, stage in enumerate(stages):
        Activity_sleep_stage=Activity_sleep_stage_dic[mouse_recday][stage]
        num_neurons=len(Activity_sleep_stage)
        
        if len(Sleepcorr_matrix_shifted_dic[stage][shifts[-1]][mouse_recday])==num_neurons:
            print('Already Analyzed')
            continue
        
        for shift_ind,shift in enumerate(shifts):
            shift=int(shift)
            corr_shift=np.zeros((num_neurons,num_neurons))
            for neuron in range(num_neurons):
                if shift>=0:
                    corrx=np.corrcoef(np.roll(Activity_sleep_stage[neuron],shift)[shift:],                                      Activity_sleep_stage[:,shift:])[0,1:]
                else:
                    corrx=np.corrcoef(np.roll(Activity_sleep_stage[neuron],shift)[:shift],                                      Activity_sleep_stage[:,:shift])[0,1:]
                    
                corr_shift[neuron]=corrx
                
            Sleepcorr_matrix_shifted_dic[stage][shift][mouse_recday]=corr_shift
print(time.time()-tt)


# In[117]:


###Shifted correlations 
len_unit=10*60*40 ##10 mins
num_timebins=6
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    stages=np.unique(sleep_stage_dic[mouse_recday])
    for stage_ind, stage in enumerate(stages):
        
        #if 'pre' in stage:
        #    num_timebins=4
        #elif 'post' in stage:
        #    num_timebins=2
        
        Activity_sleep_stage=Activity_sleep_stage_dic[mouse_recday][stage]
        #print((np.shape(Activity_sleep_stage)[1]//40)/60)

        Activity_sleep_stage_divided=[Activity_sleep_stage[:,len_unit*(ii):len_unit*(ii+1)] for ii in range(num_timebins)]
        num_neurons=len(Activity_sleep_stage)

        if len(Sleepcorr_matrix_shifted_dic['divided_time'][0][stage][shifts[-1]][mouse_recday])==num_neurons:
            print('Already Analyzed')
            continue
            
        

        for time_bin in np.arange(num_timebins):
            if np.shape(Activity_sleep_stage_divided[time_bin])[1]==0:
                continue
            for shift_ind,shift in enumerate(shifts):
                shift=int(shift)
                corr_shift=np.zeros((num_neurons,num_neurons))
                for neuron in range(num_neurons):
                    if shift>=0:
                        corrx=np.corrcoef(np.roll(Activity_sleep_stage_divided[time_bin][neuron],shift)[shift:],                                          Activity_sleep_stage_divided[time_bin][:,shift:])[0,1:]
                    else:
                        corrx=np.corrcoef(np.roll(Activity_sleep_stage_divided[time_bin][neuron],shift)[:shift],                                          Activity_sleep_stage_divided[time_bin][:,:shift])[0,1:]

                    corr_shift[neuron]=corrx

                Sleepcorr_matrix_shifted_dic['divided_time'][time_bin][stage][shift][mouse_recday]=corr_shift


# In[ ]:


###fixing bin 40.0 to 40
for time_bin in np.arange(num_timebins):
    for stage in sleep_stages_unique:
        Sleepcorr_matrix_shifted_dic['divided_time'][time_bin][stage][40]=        Sleepcorr_matrix_shifted_dic['divided_time'][time_bin][stage][40.0]


# In[ ]:





# In[ ]:





# In[19]:


###MAKE THIS FASTER BY USING MATRIX MULTIPLICATON!


# In[45]:


###Rebinned correlations - bigger bins

tt=time.time()
bin_length_original=25
bin_length=250
factor=bin_length/bin_length_original

Sleepcorr_matrix_rebinned_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    stages=np.unique(sleep_stage_dic[mouse_recday])
    for stage_ind, stage in enumerate(stages):
        Activity_sleep_stage=Activity_sleep_stage_dic[mouse_recday][stage]
        num_neurons=len(Activity_sleep_stage)

        bins=np.arange(len(Activity_sleep_stage.T)//factor)*factor

        Activity_sleep_stage_rebinned=        np.vstack(([st.binned_statistic(np.arange(len(bins)*factor),Activity_sleep_stage[neuron,:int(len(bins)*factor)],                            bins=bins)[0] for neuron in np.arange(num_neurons)]))

        corrx=np.corrcoef(Activity_sleep_stage_rebinned)

        Sleepcorr_matrix_rebinned_dic[str(bin_length)][stage][mouse_recday]=corrx
print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[46]:


###Rebinned correlations - bigger bins - timecourse

tt=time.time()
bin_length_original=25
bin_length=250
factor=bin_length/bin_length_original

len_unit=int(10*60*40*(25/bin_length)) ##10 mins
num_timebins=6
Sleepcorr_matrix_rebinned_timebins_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    stages=np.unique(sleep_stage_dic[mouse_recday])
    for stage_ind, stage in enumerate(stages):
        Activity_sleep_stage=Activity_sleep_stage_dic[mouse_recday][stage]        
        num_neurons=len(Activity_sleep_stage)

        bins=np.arange(len(Activity_sleep_stage.T)//factor)*factor

        Activity_sleep_stage_rebinned=        np.vstack(([st.binned_statistic(np.arange(len(bins)*factor),Activity_sleep_stage[neuron,:int(len(bins)*factor)],                            bins=bins)[0] for neuron in np.arange(num_neurons)]))
        
        Activity_sleep_stage_divided=[Activity_sleep_stage_rebinned[:,len_unit*(ii):len_unit*(ii+1)]                                      for ii in range(num_timebins)]
        
        for time_bin in np.arange(num_timebins):
            if np.shape(Activity_sleep_stage_divided[time_bin])[1]==0:
                continue
            
            corrx=np.corrcoef(Activity_sleep_stage_divided[time_bin])

            Sleepcorr_matrix_rebinned_timebins_dic[str(bin_length)][time_bin][stage][mouse_recday]=corrx
print(time.time()-tt)


# In[ ]:





# In[47]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


##calculating pairwise phase differences
Phase_diff_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    awake_sessions=session_dic_behaviour['awake'][mouse_recday]
    
    for awake_session_ind, timestamp in enumerate(awake_sessions):
        print(awake_session_ind)

        percentiles_ses=FR_shuff_dic['percentiles_all'][awake_session_ind][mouse_recday]
        if len(percentiles_ses)>0:
            ses_max_phase=percentiles_ses_max=np.argmax(percentiles_ses,axis=1)%90
            ses_max_phase_diff_raw=np.subtract.outer(ses_max_phase,ses_max_phase)
            ses_max_phase_diff_circ=ses_max_phase_diff_raw*4
            ses_max_phase_diff_circ[ses_max_phase_diff_circ<0]=360+ses_max_phase_diff_circ[ses_max_phase_diff_circ<0]

            Phase_diff_dic[mouse_recday][awake_session_ind]=ses_max_phase_diff_circ


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:


###Angles and Sleep correlations for pairs across anchors
Sleepcorr_pairs_dic=rec_dd()
shifts=np.linspace(-40,40,81) ## i.e. correlating with n bins in the future

num_phases=3
num_locations=9


num_trials_thr=6


num_phase_place_diffs_=5
useplace_phase_diff=True
use_tuned=False
use_anchored=True
use_nonspatial=False
use_spatial=False
use_anchor_angles=False
use_FRstable=True
bin_length=250 ##for rebinned correlations
thr_lower=30
thr_upper=360-thr_lower
sleep_dicX=Sleepcorr_matrix_shifted_dic

#groups_=np.hstack((np.arange(num_phase_place_diffs_),'Non_coanchored'))
groups_=np.hstack((0,'Non_coanchored'))

for phase_place_diff in groups_:
        for mouse_recday in day_type_dicX['combined_ABCDonly']:
            #try:
            print(mouse_recday)

            num_sessions=len(session_dic['awake'][mouse_recday])
            non_repeat_ses=non_repeat_ses_maker(mouse_recday)
            #num_trials=dict_to_array(Num_trials_dic2[mouse_recday])
            
            num_trials=[]
            for ses_ind in non_repeat_ses:
                num_trials_ses=len(np.load(Intermediate_object_folder+'trialtimes_'+mouse_recday+'_'                                           +str(ses_ind)+'.npy'))
                num_trials.append(num_trials_ses)
            num_trials=np.asarray(num_trials)
            ses_trials=np.where(num_trials>=num_trials_thr)[0]
            non_repeat_ses=np.intersect1d(non_repeat_ses,ses_trials)

            len_ephys=[]
            for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
                try:
                    ephys_ = np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    location_mat_=np.load(Intermediate_object_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    len_ephys.append(len(ephys_))
                except:
                    len_ephys.append(0)
                    continue

            len_ephys=np.hstack((len_ephys))
            non_repeat_ses=non_repeat_ses[len_ephys>0]


            found_ses=[]
            for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
                try:
                    Neuron_raw=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    found_ses.append(ses_ind)

                except:
                    print('Files not found for session '+str(ses_ind))
                    continue

            found_ses=np.asarray(found_ses)


            #angle_units=Xneuron_correlations['combined_ABCDonly']['angle_units'][mouse_recday]
            #max_bins=Xneuron_correlations['combined_ABCDonly']['Max_bins'][mouse_recday]#[0]
            angles=Xneuron_correlations['combined_ABCDonly']['Angles'][mouse_recday]

            Anchor_lags=Anchor_trial_dic['Best_shift_time'][mouse_recday]    
            Anchor_lags_mean=np.rad2deg(st.circmean(np.deg2rad(Anchor_lags),axis=1,nan_policy='omit'))
            #angles_anchor=np.vstack(([positive_angle([Anchor_lags_mean[ii]-Anchor_lags_mean[jj]\
            #                                   for ii in range(len(Anchor_lags_mean))])\
            # for jj in range(len(Anchor_lags_mean))]))

            angles_anchor=[np.vstack(([[Anchor_lags[ii,ses]-Anchor_lags[jj,ses]                                               for ii in range(len(Anchor_lags))]             for jj in range(len(Anchor_lags))])) for ses in np.arange(len(Anchor_lags.T))]


            Best_anchor_all=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday]
            Spatial_maps_all=Spatial_anchoring_dic['Phase_shifted_maps'][mouse_recday][:,:,0] ##0 as in 0 lag maps
            FR_bool=FR_sleep_boolean_dic[mouse_recday]
            FR_included=np.where(FR_bool==True)[0]

            awake_sesinds=Awake_sesind_dic[mouse_recday]
            sleep_stages=sleep_stage_dic[mouse_recday]
            sleep_stages_unique=np.unique(sleep_stages)

            first_day2_boolean=Awake_sesind_dic['first_day2_boolean'][mouse_recday]

            neurons_tuned=np.where(Tuned_dic['State_zmax_bool'][mouse_recday]==True)[0]
            Anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]
            Anchored_neurons=np.where(Anchored_bool==True)[0]

            Anchor_lags_used=Anchor_trial_dic['Best_shift_time'][mouse_recday]
            Anchor_lags_used_mean=np.rad2deg(st.circmean(np.deg2rad(Anchor_lags_used),axis=1,nan_policy='omit'))
            non_spatial_neurons=np.where(np.logical_and(Anchor_lags_used_mean>thr_lower,Anchor_lags_used_mean<thr_upper)                                         ==True)[0]
            spatial_neurons=np.where(np.logical_or(Anchor_lags_used_mean<thr_lower,Anchor_lags_used_mean>thr_upper)                                         ==True)[0]


            for stage in sleep_stages_unique:
                if stage=='preX':
                    ses_ind=0
                elif 'post' in stage:
                    ses_ind_unique=int(stage.split('post',1)[1])
                    ses_ind=np.arange(num_sessions)[np.where(awake_sesinds==ses_ind_unique)[0][0]]
                elif stage=='preX2':
                    ses_ind=np.where(first_day2_boolean==True)[0][0]

                ses_ind_ind_=np.where(found_ses==ses_ind)[0]
                if len(ses_ind_ind_)==0:
                    continue
                ses_ind_ind=ses_ind_ind_[0]

                #phases=Xneuron_phaseangle_dic[ses_ind][mouse_recday] ##phase angle diff
                phases=Phase_diff_dic[mouse_recday][ses_ind] ##phase angle diff

                if len(phases)==0:
                    continue


                Spatial_corrs_ses=np.corrcoef(Spatial_maps_all[:,ses_ind])

                if len(Spatial_maps_all)<2:
                    print('less than 2 neurons')
                    continue

                All_anchored_pairsX=[]
                All_anchored_pairsY=[]
                All_anchored_pairs_angles=[]
                All_anchored_pairs_phases=[]
                All_anchored_pairs_Spatialcorrs=[]


                for phase in np.arange(num_phases):
                    if useplace_phase_diff==True and phase_place_diff!='Non_coanchored':
                        phase_place_diff=int(phase_place_diff)
                        phase_next=(phase+phase_place_diff)%3
                    else:
                        phase_next=phase
                    for location in np.arange(num_locations):


                        neurons_anchoredX=np.where(np.logical_and(Best_anchor_all[:,0]==phase,                                                                  Best_anchor_all[:,1]==location,                                                                 FR_bool==True))[0]

                        if phase_place_diff=='Non_coanchored':
                            neurons_anchoredY=np.where(np.logical_or(Best_anchor_all[:,0]!=phase,                                                                  Best_anchor_all[:,1]!=location,                                                                 FR_bool==True))[0]
                        else:
                            locations_next=np.where(mindistance_mat[location]==phase_place_diff)[0]

                            neurons_anchoredY=np.where(np.logical_and(Best_anchor_all[:,0]==phase_next                                                                  ,np.isin(Best_anchor_all[:,1],locations_next),                                                                 FR_bool==True))[0]
                        if use_tuned==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,neurons_tuned)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,neurons_tuned)
                        if use_anchored==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,Anchored_neurons)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,Anchored_neurons)
                        if use_nonspatial==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,non_spatial_neurons)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,non_spatial_neurons)
                        if use_spatial==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,spatial_neurons)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,spatial_neurons)

                        if use_FRstable==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,FR_included)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,FR_included)

                        All_anchored_pairsX.append(neurons_anchoredX)
                        All_anchored_pairsY.append(neurons_anchoredY)

                        if phase_place_diff!=0:
                            angle_mat_pairs_anchor_=angles_anchor[ses_ind_ind][neurons_anchoredX][:,neurons_anchoredY]
                            if use_anchor_angles==True:
                                angle_mat_pairs_=angles_anchor[ses_ind_ind][neurons_anchoredX][:,neurons_anchoredY]
                                angle_mat_pairs_=abs(angle_mat_pairs_)
                            else:
                                angle_mat_pairs_=angles[:,:,ses_ind][neurons_anchoredX][:,neurons_anchoredY]

                                ##aligning forward angles to anchor
                                #angle_mat_pairs_anchor_=angle_mat_pairs_anchor_.squeeze()
                                #angle_mat_pairs_=angle_mat_pairs_.squeeze()
                                if len(angle_mat_pairs_)>0:
                                    angle_mat_pairs_[angle_mat_pairs_anchor_<0]=                                    360-angle_mat_pairs_[angle_mat_pairs_anchor_<0]#

                                    #angle_mat_pairs_=[angle_mat_pairs_]

                            phase_mat_pairs_=phases[neurons_anchoredX][:,neurons_anchoredY]
                            Spatial_corrs_pairs_=Spatial_corrs_ses[neurons_anchoredX][:,neurons_anchoredY]
                        elif phase_place_diff==0:
                            angle_mat_pairs_anchor_=matrix_triangle(angles_anchor[ses_ind_ind][neurons_anchoredX]                                                                 [:,neurons_anchoredY],'lower')
                            if use_anchor_angles==True:
                                angle_mat_pairs_=matrix_triangle(angles_anchor[ses_ind_ind][neurons_anchoredX]                                                                 [:,neurons_anchoredY],'lower')
                                angle_mat_pairs_=abs(angle_mat_pairs_)
                            else:
                                angle_mat_pairs_=matrix_triangle(angles[:,:,ses_ind][neurons_anchoredX]                                                                 [:,neurons_anchoredY],'lower')

                                ##aligning forward angles to anchor
                                #angle_mat_pairs_anchor_=angle_mat_pairs_anchor_.squeeze()
                                #angle_mat_pairs_=angle_mat_pairs_.squeeze()
                                if len(angle_mat_pairs_)>0:
                                    angle_mat_pairs_[angle_mat_pairs_anchor_<0]=                                    360-angle_mat_pairs_[angle_mat_pairs_anchor_<0]

                                    #angle_mat_pairs_=[angle_mat_pairs_]


                            phase_mat_pairs_=matrix_triangle(phases[neurons_anchoredX][:,neurons_anchoredY],                                                             'lower')
                            Spatial_corrs_pairs_=matrix_triangle(Spatial_corrs_ses[neurons_anchoredX]                                                                 [:,neurons_anchoredY],'lower')



                        if len(angle_mat_pairs_)>0 and phase_place_diff!=0:
                            angle_mat_pairs=np.hstack((angle_mat_pairs_))
                            phase_mat_pairs=np.hstack((phase_mat_pairs_))
                            Spatial_corrs_pairs_=np.hstack((Spatial_corrs_pairs_))
                        else:
                            angle_mat_pairs=angle_mat_pairs_
                            phase_mat_pairs=phase_mat_pairs_
                            Spatial_corrs_pairs_=Spatial_corrs_pairs_

                        All_anchored_pairs_angles.append(angle_mat_pairs)
                        All_anchored_pairs_phases.append(phase_mat_pairs)
                        All_anchored_pairs_Spatialcorrs.append(Spatial_corrs_pairs_)

                All_anchored_pairs_angles_=All_anchored_pairs_angles
                All_anchored_pairs_angles=concatenate_complex2(All_anchored_pairs_angles)
                All_anchored_pairs_phases=concatenate_complex2(All_anchored_pairs_phases)
                All_anchored_pairs_Spatialcorrs=concatenate_complex2(All_anchored_pairs_Spatialcorrs)

                All_anchored_pairs_sleepcorrs_allshifts=np.zeros((len(shifts),len(All_anchored_pairs_angles)))
                All_anchored_pairs_sleepcorrs_allshifts[:]=np.nan
                for shift_ind,shift in enumerate(shifts):
                    shift=int(shift)
                    All_anchored_pairs_sleepcorrs=[]
                    for phase in np.arange(num_phases):
                        if useplace_phase_diff==True and phase_place_diff!='Non_coanchored':
                            phase_next=(phase+phase_place_diff)%3
                        else:
                            phase_next=phase
                        for location in np.arange(num_locations):

                            neurons_anchoredX=np.where(np.logical_and(Best_anchor_all[:,0]==phase,                                                                  Best_anchor_all[:,1]==location,                                                                 FR_bool==True))[0]

                            if phase_place_diff=='Non_coanchored':
                                neurons_anchoredY=np.where(np.logical_or(Best_anchor_all[:,0]!=phase,                                                                  Best_anchor_all[:,1]!=location,                                                                 FR_bool==True))[0]
                            else:
                                locations_next=np.where(mindistance_mat[location]==phase_place_diff)[0]
                                neurons_anchoredY=np.where(np.logical_and(Best_anchor_all[:,0]==phase_next                                                                  ,np.isin(Best_anchor_all[:,1],locations_next),                                                                 FR_bool==True))[0]


                            if use_tuned==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,neurons_tuned)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,neurons_tuned)
                            if use_anchored==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,Anchored_neurons)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,Anchored_neurons)
                            if use_nonspatial==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,non_spatial_neurons)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,non_spatial_neurons)
                            if use_spatial==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,spatial_neurons)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,spatial_neurons)

                            if use_FRstable==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,FR_included)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,FR_included)




                            sleepcorr_mat=sleep_dicX[stage][shift][mouse_recday]

                            if phase_place_diff!=0:
                                sleepcorr_mat_pairs_=sleepcorr_mat[neurons_anchoredX][:,neurons_anchoredY]

                            elif phase_place_diff==0:
                                sleepcorr_mat_pairs_=matrix_triangle(sleepcorr_mat[neurons_anchoredX]                                                                     [:,neurons_anchoredY],                                                                     'lower')


                            if len(sleepcorr_mat_pairs_)>0 and phase_place_diff!=0:
                                sleepcorr_mat_pairs=np.hstack((sleepcorr_mat_pairs_))
                            else:
                                sleepcorr_mat_pairs=sleepcorr_mat_pairs_

                            All_anchored_pairs_sleepcorrs.append(sleepcorr_mat_pairs)

                    All_anchored_pairs_sleepcorrs_=All_anchored_pairs_sleepcorrs
                    All_anchored_pairs_sleepcorrs=concatenate_complex2(All_anchored_pairs_sleepcorrs)
                    All_anchored_pairs_sleepcorrs_allshifts[shift_ind]=All_anchored_pairs_sleepcorrs



                All_anchored_pairs_sleepcorrs=[]
                for phase in np.arange(num_phases):
                    if useplace_phase_diff==True and phase_place_diff!='Non_coanchored':
                        phase_next=(phase+phase_place_diff)%3
                    else:
                        phase_next=phase
                    for location in np.arange(num_locations):

                        neurons_anchoredX=np.where(np.logical_and(Best_anchor_all[:,0]==phase,                                                              Best_anchor_all[:,1]==location,                                                             FR_bool==True))[0]


                        if phase_place_diff=='Non_coanchored':
                            neurons_anchoredY=np.where(np.logical_or(Best_anchor_all[:,0]!=phase,                                                              Best_anchor_all[:,1]!=location,                                                             FR_bool==True))[0]
                        else:
                            locations_next=np.where(mindistance_mat[location]==phase_place_diff)[0]

                            neurons_anchoredY=np.where(np.logical_and(Best_anchor_all[:,0]==phase_next                                                              ,np.isin(Best_anchor_all[:,1],locations_next),                                                             FR_bool==True))[0]
                        if use_tuned==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,neurons_tuned)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,neurons_tuned)
                        if use_anchored==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,Anchored_neurons)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,Anchored_neurons)
                        if use_nonspatial==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,non_spatial_neurons)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,non_spatial_neurons)
                        if use_spatial==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,spatial_neurons)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,spatial_neurons)

                        if use_FRstable==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,FR_included)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,FR_included)



                        sleepcorr_mat=Sleepcorr_matrix_rebinned_dic[str(bin_length)][stage][mouse_recday]

                        if phase_place_diff!=0:
                            sleepcorr_mat_pairs_=sleepcorr_mat[neurons_anchoredX][:,neurons_anchoredY]

                        elif phase_place_diff==0:
                            sleepcorr_mat_pairs_=matrix_triangle(sleepcorr_mat[neurons_anchoredX]                                                                 [:,neurons_anchoredY],'lower')

                        if len(sleepcorr_mat_pairs_)>0 and phase_place_diff!=0:
                            sleepcorr_mat_pairs=np.hstack((sleepcorr_mat_pairs_))
                        else:
                            sleepcorr_mat_pairs=sleepcorr_mat_pairs_

                        All_anchored_pairs_sleepcorrs.append(sleepcorr_mat_pairs)

                All_anchored_pairs_sleepcorrs=concatenate_complex2(All_anchored_pairs_sleepcorrs)
                All_anchored_pairs_sleepcorrs_rebinned=All_anchored_pairs_sleepcorrs



                Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs'][mouse_recday]=All_anchored_pairsX,                All_anchored_pairsY
                Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_angles'][stage][mouse_recday]=                All_anchored_pairs_angles
                Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_phases'][stage][mouse_recday]=                All_anchored_pairs_phases
                All_anchored_pairs_phases
                Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_Spatialcorrs'][stage][mouse_recday]=                All_anchored_pairs_Spatialcorrs

                Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs'][stage][mouse_recday]=                All_anchored_pairs_sleepcorrs_allshifts
                Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned'][stage][mouse_recday]=                All_anchored_pairs_sleepcorrs_rebinned
            #except:
            #    print('Not done')


# In[43]:


num_trials


# In[ ]:


'''
-why are forward and circular distances not giving exactly opposite betas? - sorted
-why does circular distance vs forward not just give a negative beta - sorted
-error bars - sorted (but check)
-statistics - ttest against 0


'''


# In[270]:


non_repeat_ses_maker('me08_12092021_13092021') 


# In[ ]:





# In[49]:


##Task space distance and spatial corr vs sleep corrs
corr_crossanchor_task_all=[]
corr_crossanchor_space_all=[]
coeffs_crossanchor_all=[]
sems_crossanchor_all=[]
only_far=False
co_regress=True

num_pairs_all=[]

if only_far==True:
    co_regress==False
stage_type='All'
if stage_type=='pre':
    stages=['preX','preX2']
elif stage_type=='post':
    stages=['post0', 'post1', 'post2', 'post3', 'post4', 'post5']
elif stage_type=='All':
    stages=sleep_stages_unique
for circular_angle in [True]:
    corr_crossanchor_task=[]
    corr_crossanchor_space=[]
    coeffs_crossanchor=[]
    sems_crossanchor=[]
    N_all=[]
    for phase_place_diff in groups_:
        if phase_place_diff!='Non_coanchored':
            phase_place_diff=int(phase_place_diff)
        print(phase_place_diff)
        corr_phase_place_diff=[]
        corr_spatial_corr=[]

        coeffs_phase_place_diff=[]
        sems_phase_place_diff=[]
        N_place_phase_diff=[]
        ##['post0', 'post1', 'post2', 'post3', 'post4', 'post5']:#sleep_stages_unique: ['preX','preX2']:
        for stage in stages:#['preX','preX2']:#['preX']:#,'preX2']:#
            print(stage)

            Anchored_pairs_sleepcorrs_rebinned_=[]
            Anchored_pairs_angles_=[]
            Anchored_pairs_spatialcorrs_=[]
            Anchored_pairs_phases_=[]
            for mouse_recday in day_type_dicX['combined_ABCDonly']:
                
                if phase_place_diff==0 and stage=='preX':
                    num_pairs_all.append(len(Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs'][mouse_recday]))
    

                
                print(mouse_recday)
                mouse=mouse_recday.split('_',1)[0]
                cohort=Mice_cohort_dic[mouse]
                Ephys_type=Cohort_ephys_type_dic[int(cohort)]

                #if Ephys_type=='Neuropixels':
                #    print('skipped')
                #    continue
                if len(Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned'][stage][mouse_recday])>0:
                    Anchored_pairs_sleepcorrs_rebinned_.append(                        Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned'][stage][mouse_recday])
                    Anchored_pairs_angles_.append(Sleepcorr_pairs_dic[phase_place_diff]                                                  ['Anchored_pairs_angles'][stage][mouse_recday])
                    Anchored_pairs_spatialcorrs_.append(Sleepcorr_pairs_dic[phase_place_diff]                                                        ['Anchored_pairs_Spatialcorrs'][stage][mouse_recday])

                    Anchored_pairs_phases_.append(Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_phases']                                                        [stage][mouse_recday])

            Anchored_pairs_sleepcorrs_rebinned_=np.hstack((Anchored_pairs_sleepcorrs_rebinned_))
            Anchored_pairs_angles_=np.hstack((Anchored_pairs_angles_))
            Anchored_pairs_spatialcorrs_=np.hstack((Anchored_pairs_spatialcorrs_))
            Anchored_pairs_phases_=np.hstack((Anchored_pairs_phases_))




            filter_boolean_nan=~np.isnan(Anchored_pairs_angles_)
            Anchored_pairs_angles_[filter_boolean_nan]=abs((Anchored_pairs_angles_[filter_boolean_nan]))
            ###taking absolute angle between neurons

            if only_far==True:
                filter_boolean_=np.logical_and(Anchored_pairs_spatialcorrs_<0.9,Anchored_pairs_angles_>=180)
            else:
                filter_boolean_=Anchored_pairs_spatialcorrs_<0.9

            if circular_angle==True:
                Anchored_pairs_angles_forward=np.copy(Anchored_pairs_angles_)
                Anchored_pairs_angles_[Anchored_pairs_angles_>=180]=360-Anchored_pairs_angles_                [Anchored_pairs_angles_>=180]


            else:
                Anchored_pairs_angles_forward=np.copy(Anchored_pairs_angles_)
                #filter_boolean=np.logical_and(filter_boolean_==True,\
                #                              np.logical_and(Anchored_pairs_angles_>0,Anchored_pairs_angles_<350))
            filter_boolean=np.logical_and(filter_boolean_==True,Anchored_pairs_angles_>0)
            filter_boolean=np.logical_and(filter_boolean,~np.isnan(Anchored_pairs_phases_))
            #filter_boolean=np.logical_and(filter_boolean_,filter_boolean_nan)

            Anchored_pairs_angles=Anchored_pairs_angles_[filter_boolean]
            Anchored_pairs_angles_forward=Anchored_pairs_angles_forward[filter_boolean]
            Anchored_pairs_spatialcorrs=Anchored_pairs_spatialcorrs_[filter_boolean]
            Anchored_pairs_phases=Anchored_pairs_phases_[filter_boolean]
            Anchored_pairs_sleepcorrs_rebinned=Anchored_pairs_sleepcorrs_rebinned_[filter_boolean]

            if phase_place_diff==0:
                #sns.regplot(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned)
                #plt.show()
                print(st.pearsonr(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned))

                #sns.regplot(Anchored_pairs_angles_forward,Anchored_pairs_sleepcorrs_rebinned)
                #plt.show()
                print(st.pearsonr(Anchored_pairs_angles_forward,Anchored_pairs_sleepcorrs_rebinned))

            corr_phase_place_diff.append(st.pearsonr(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned)[0])


            #sns.regplot(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned)
            #plt.show()
            #print(st.pearsonr(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned))
            corr_spatial_corr.append(st.pearsonr(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned)[0])


            ###regression

            if co_regress==True:
                X=np.column_stack((st.zscore(Anchored_pairs_angles,nan_policy='omit'),                                   st.zscore(Anchored_pairs_angles_forward,nan_policy='omit'),                                   st.zscore(Anchored_pairs_spatialcorrs,nan_policy='omit'),                          st.zscore(Anchored_pairs_phases,nan_policy='omit'),                                   np.repeat(1,len(Anchored_pairs_angles))))

            else:
                X=np.column_stack((st.zscore(Anchored_pairs_angles,nan_policy='omit'),                                   st.zscore(Anchored_pairs_spatialcorrs,nan_policy='omit'),                          st.zscore(Anchored_pairs_phases,nan_policy='omit'),                                   np.repeat(1,len(Anchored_pairs_angles))))


            y=Anchored_pairs_sleepcorrs_rebinned

            y=y[~np.isnan(X[:,2])]
            X=X[~np.isnan(X[:,2])]

            N=len(X)
            p=len(X.T)
            
            N_place_phase_diff.append(N)

            model=LinearRegression()
            reg=model.fit(X, y)
            coeffs=reg.coef_
            coeffs_phase_place_diff.append(coeffs)

            y_hat = model.predict(X)
            residuals = y - y_hat
            residual_sum_of_squares = residuals.T @ residuals
            sigma_squared_hat = residual_sum_of_squares / (N - p)

            var_beta_hat = np.linalg.inv(X.T @ X) * sigma_squared_hat

            std_errors=[]
            for indx in np.arange(p):
                std_err=var_beta_hat[indx, indx] ** 0.5
                std_errors.append(std_err)

            sems_phase_place_diff.append(std_errors)


        coeffs_phase_place_diff=np.vstack((coeffs_phase_place_diff))
        sems_phase_place_diff=np.vstack((sems_phase_place_diff))


        corr_crossanchor_task.append(corr_phase_place_diff)
        corr_crossanchor_space.append(corr_spatial_corr)
        coeffs_crossanchor.append(coeffs_phase_place_diff)
        sems_crossanchor.append(sems_phase_place_diff)
        
        N_all.append(N_place_phase_diff)
        
        
    corr_crossanchor_task_all.append(corr_crossanchor_task)
    corr_crossanchor_space_all.append(corr_crossanchor_space)
    coeffs_crossanchor_all.append(coeffs_crossanchor)
    sems_crossanchor_all.append(sems_crossanchor)
    
N_all=np.asarray(N_all).T


# In[33]:


phase_place_diff


# In[ ]:





# In[ ]:





# In[50]:


corr_all_task=corr_crossanchor_task_all[0]
bar_plotX(corr_all_task,'none',-0.2,0.13,'points','paired',0.025)#
plt.show()
print(st.wilcoxon(corr_all_task[0],np.nanmean(corr_all_task[1:],axis=0)))


# In[57]:


np.shape(coeffs_crossanchor_all)


# In[56]:


print('''Task Space - betas for circular and forward angles within-anchor 
      when co-regressed (regressing out phase and place) ''')
scaling_factor=1000
means_all=[]
sems_all=[]
phase_place_diff=0
for ind,angle_type in enumerate(['circular angle','forward angle']):
    print('______')
    print(angle_type)
    
    coeffs_crossanchor_=coeffs_crossanchor_all[0][phase_place_diff][:,ind]
    sems_crossanchor_=sems_crossanchor_all[0][phase_place_diff][:,ind]
    #coeffs_coanchored_task=np.asarray((coeffs_crossanchor))[0,:,0]
    #coeffs_crossanchor_task=np.vstack(([np.asarray(coeffs_crossanchor)[ii][:,0] for ii in\
    #                                    np.arange(len(coeffs_crossanchor)-1)+1]))
    coeffs_all_task=coeffs_crossanchor_

    #sems_coanchored_task=np.asarray((sems_crossanchor))[0,:,0]
    #sems_crossanchor_task=np.vstack(([np.asarray(sems_crossanchor)[ii][:,0]\
    #                                  for ii in np.arange(len(coeffs_crossanchor)-1)+1]))
    sems_all_task=sems_crossanchor_

    task_mean=np.mean(coeffs_all_task)
    task_sem_mean=np.sqrt(np.sum(sems_all_task**2)/len(sems_all_task.T)**2)
    
    
    
    means_all.append(task_mean)
    sems_all.append(task_sem_mean)
    
    plt.rcParams["figure.figsize"] = (2,5)
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False


    plt.bar(1,task_mean*scaling_factor,yerr=task_sem_mean*scaling_factor,color='black',ecolor='grey',capsize=3,width=0.4)
    #plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
    plt.axhline(0,color='black',ls='dashed',linewidth=4)

    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.xlim(0.7,1.3)
    plt.ylim(-0.006*scaling_factor,0.0035*scaling_factor)

    plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_coactivity_Coregression_bar_preplay_'+                angle_type+'_.svg',                    bbox_inches = 'tight', pad_inches = 0) 
    plt.show()
    
    
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False

    data=(coeffs_all_task*scaling_factor).T

    filtered_data = remove_nan(data)

    sns.violinplot(filtered_data, color='grey',alpha=0.5)
    #sns.stripplot(filtered_data,color='white',edgecolor='black',linewidth=1,alpha=0.5)
    plt.axhline(0,color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_coactivity_Coregression_bar_preplay_'+                angle_type+'_violin.svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1)
    plt.axhline(0,color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_coactivity_Coregression_bar_preplay_'+                angle_type+'_swarm.svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    
plt.rcParams["figure.figsize"] = (5,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
    
    
plt.bar(np.arange(len(means_all)),means_all,yerr=sems_all,color='black',ecolor='grey',capsize=3)
#plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
plt.axhline(0,color='black',ls='dashed',linewidth=4)

plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
#plt.xlim(-0.1,0.9)

plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_coactivity_Coregression_bar_preplay.svg',                bbox_inches = 'tight', pad_inches = 0) 
plt.show()

print(means_all)


###sem averaging taken from ehre: 
##https://stats.stackexchange.com/questions/21104/calculate-average-of-a-set-numbers-with-reported-standard-errors


# In[ ]:





# In[60]:


t_values_=np.asarray(means_all)/np.asarray(sems_all)
print(t_values_)

N_=np.nanmean(N_all[:,0])
print(N_)

std_all=np.asarray(sems_all)*np.sqrt(N_)


from scipy.stats import ttest_ind_from_stats

tstat, pvalue = ttest_ind_from_stats(means_all[0], std_all[0], N_, 0, 0, 2, equal_var=False,)

print(tstat)
print(pvalue)

tstat, pvalue = ttest_ind_from_stats(means_all[1], std_all[1], N_, 0, 0, 2, equal_var=False,)

print(tstat)
print(pvalue)


# In[ ]:





# In[ ]:





# In[61]:


print('''Task Space - betas for circular and forward angles within-anchor 
      when co-regressed (regressing out phase and place) ''')
scaling_factor=1000
means_all=[]
sems_all=[]
phase_place_diff=0

measures=['circular angle','forward angle','spatial_corr']
stage_types=['pre','post']

means_all_types=np.zeros((len(measures),len(stage_types)))
sems_all_types=np.zeros((len(measures),len(stage_types)))

means_all_types[:]=np.nan
sems_all_types[:]=np.nan

for measure_ind,measure in enumerate(measures):
    print('______')
    print(measure)
    
    
    for stage_type_ind, stage_type in enumerate(stage_types):
        ses_inds=np.where(np.asarray([stage_type in sleep_stages_unique[ii]                                      for ii in range(len(sleep_stages_unique))])==True)[0]
        coeffs_crossanchor_=coeffs_crossanchor_all[0][phase_place_diff][ses_inds,measure_ind]
        sems_crossanchor_=sems_crossanchor_all[0][phase_place_diff][ses_inds,measure_ind]
        coeffs_all_task=coeffs_crossanchor_
        sems_all_task=sems_crossanchor_

        task_mean=np.mean(coeffs_all_task)
        task_sem_mean=np.sqrt(np.sum(sems_all_task**2)/len(sems_all_task.T)**2)

        
        means_all_types[measure_ind][stage_type_ind]=task_mean
        sems_all_types[measure_ind][stage_type_ind]=task_sem_mean

    plt.rcParams["figure.figsize"] = (2,5)
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False


    plt.bar(np.arange(len(means_all_types[measure_ind])),means_all_types[measure_ind]*scaling_factor,            yerr=sems_all_types[measure_ind]*scaling_factor,color='black',ecolor='grey',capsize=3,width=1)
    #plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
    plt.axhline(0,color='black',ls='dashed',linewidth=4)

    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    #plt.xlim(-0.1,0.9)

    plt.savefig(Ephys_output_folder_dropbox+'Pair_'+measure+'_vs_coactivity_Coregression_bar_preplay.svg',                    bbox_inches = 'tight', pad_inches = 0) 
    plt.show()

    #print(means_all_types[measure_ind])
    
    ###ttest for pre vs post
    ###claculated using: https://www.graphpad.com/quickcalcs/ttest2/
    print(means_all_types[measure_ind])
    print(sems_all_types[measure_ind])
    N_pre=np.nanmean(N_all[:2,0])
    N_post=np.nanmean(N_all[2:,0])
    print(N_pre)
    print(N_post)
    dof= (N_pre - 1) + (N_post - 1)
    
    
    
    
    std_0=np.asarray(sems_all_types[measure_ind][0])*np.sqrt(N_pre)
    std_1=np.asarray(sems_all_types[measure_ind][1])*np.sqrt(N_post)
    
    
    tstat, pvalue = ttest_ind_from_stats(means_all_types[measure_ind][0], std_0, N_pre,                                         means_all_types[measure_ind][1], std_1, N_post,                                        equal_var=False)

    print(tstat)
    print(pvalue)
    print(dof)
    
    
    


    ###sem averaging taken from ehre: 
    ##https://stats.stackexchange.com/questions/21104/calculate-average-of-a-set-numbers-with-reported-standard-errors


# In[ ]:





# In[62]:


print('''Task Space - betas for circular angles within versus between-anchor 
      when co-regressed (regressing out forward angle, phase and place) ''')


ind=0
angle_type='circular angle'
print('______')
print(angle_type)

coeffs_crossanchor_=coeffs_crossanchor_all[ind]
sems_crossanchor_=sems_crossanchor_all[ind]
#coeffs_coanchored_task=np.asarray((coeffs_crossanchor))[0,:,0]
#coeffs_crossanchor_task=np.vstack(([np.asarray(coeffs_crossanchor)[ii][:,0] for ii in\
#                                    np.arange(len(coeffs_crossanchor)-1)+1]))
coeffs_all_task=np.vstack(([np.asarray(coeffs_crossanchor_)[ii][:,0] for ii in                                    np.arange(len(coeffs_crossanchor_))]))


#sems_coanchored_task=np.asarray((sems_crossanchor))[0,:,0]
#sems_crossanchor_task=np.vstack(([np.asarray(sems_crossanchor)[ii][:,0]\
#                                  for ii in np.arange(len(coeffs_crossanchor)-1)+1]))
sems_all_task=np.vstack(([np.asarray(sems_crossanchor_)[ii][:,0]                                  for ii in np.arange(len(coeffs_crossanchor_))]))

task_mean=np.mean(coeffs_all_task,axis=1)
task_sem_mean=np.sqrt(np.sum(sems_all_task**2,axis=1)/len(sems_all_task.T)**2)

plt.bar(np.arange(len(task_mean)),task_mean,yerr=task_sem_mean,color='black',ecolor='grey',capsize=3)
#plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
plt.axhline(0,color='black',ls='dashed')
#plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_coactivity_regression_bar_preplay.svg',\
#                bbox_inches = 'tight', pad_inches = 0) 
plt.show()



print('Within vs between')

coeffs_crossanchor_task_all_mean=np.nanmean(coeffs_all_task[-1])
sems_crossanchor_task_all=sems_all_task[-1]
sems_crossanchor_task_all_mean=np.sqrt(np.sum(sems_crossanchor_task_all**2)/len(sems_crossanchor_task_all)**2)

within_vs_between_means=np.asarray([task_mean[0],coeffs_crossanchor_task_all_mean])
within_vs_between_sems=np.asarray([task_sem_mean[0],sems_crossanchor_task_all_mean])

plt.rcParams["figure.figsize"] = (2,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.bar(np.arange(2),within_vs_between_means*scaling_factor,yerr=within_vs_between_sems*scaling_factor,        color='black',ecolor='grey',capsize=3,width=1)
#plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
#plt.axhline(0,color='black',ls='dashed')
#plt.ylim(-0.005,0.001)

plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
#plt.xlim(0,2.5)
plt.ylim(-0.006*scaling_factor,0.0005*scaling_factor)
plt.savefig(Ephys_output_folder_dropbox+angle_type+'_WithinvsBetween_coactivity_regression_bar.svg',                bbox_inches = 'tight', pad_inches = 0) 
plt.show()
#print(within_vs_between_means,np.asarray(within_vs_between_sems)*np.sqrt(N))

N_within=np.nanmean(N_all[:,0])
N_between=np.nanmean(N_all[:,-1])
std_0=np.asarray(within_vs_between_sems[0])*np.sqrt(N_within)
std_1=np.asarray(within_vs_between_sems[1])*np.sqrt(N_between)
dof= (N_within - 1) + (N_between - 1)

tstat, pvalue = ttest_ind_from_stats(within_vs_between_means[0], std_0, N_within,                                     within_vs_between_means[1], std_1, N_between,                                    equal_var=False, alternative='less')
print(N_within)
print(N_between)
print(tstat)
print(pvalue)
print(dof)

tstat, pvalue = ttest_ind_from_stats(within_vs_between_means[0], std_0, N_within,                                    0, 0, 2, equal_var=False)
print(tstat)
print(pvalue)

tstat, pvalue = ttest_ind_from_stats(within_vs_between_means[1], std_1, N_between,                                    0, 0, 2, equal_var=False)
print(tstat)
print(pvalue)

print('')

print('Per Session')
###Session-wise regressions

for ii,name in enumerate(['preX','preX2']):
    print(name)
    plt.bar(np.arange(len(coeffs_all_task)),coeffs_all_task[:,ii],yerr=sems_all_task[:,ii],           color='black',ecolor='grey',capsize=3)
    #plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
    plt.axhline(0,color='black',ls='dashed')
    plt.savefig(Ephys_output_folder_dropbox+name+'_Pair_distance_vs_coactivity_regression_bar.svg',                bbox_inches = 'tight', pad_inches = 0) 
    plt.show()

###sem averaging taken from here: 
##https://stats.stackexchange.com/questions/21104/calculate-average-of-a-set-numbers-with-reported-standard-errors


# In[ ]:





# In[63]:


##Task space distance and spatial corr vs sleep corrs
corr_crossanchor_task_all=[]
corr_crossanchor_space_all=[]
coeffs_crossanchor_all=[]
sems_crossanchor_all=[]

only_far=False
only_close=False

cross_corr_dic=rec_dd()
spatial_corr_thr=0.9
num_bins=8
bins_angles_=(np.arange(num_bins)+1)*(180//num_bins)
bins_angles_[-1]=bins_angles_[-1]+1
for circular_angle in [False]:#,False]:
    print('')
    print(circular_angle)
    corr_crossanchor_task=[]
    corr_crossanchor_space=[]
    coeffs_crossanchor=[]
    sems_crossanchor=[]
    N_all=[]
    
    binned_corrs_means_all=[]
    binned_corrs_sems_all=[]

    for phase_place_diff in np.arange(num_phase_place_diffs_):
        print(phase_place_diff)
        corr_phase_place_diff=[]
        corr_spatial_corr=[]

        coeffs_phase_place_diff=[]
        sems_phase_place_diff=[]
        N_place_phase_diff=[]
        
        Anchored_pairs_angles_all=[]
        Anchored_pairs_sleepcorrs_rebinned_all=[]

        for stage in sleep_stages_unique:# ['preX','preX2']:
            print(stage)
            try:

                Anchored_pairs_sleepcorrs_rebinned_=[]
                Anchored_pairs_angles_=[]
                Anchored_pairs_spatialcorrs_=[]
                Anchored_pairs_phases_=[]
                for mouse_recday in day_type_dicX['combined_ABCDonly']:
                    if len(Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned']                           [stage][mouse_recday])>0:
                        Anchored_pairs_sleepcorrs_rebinned_day_=np.copy(Sleepcorr_pairs_dic[phase_place_diff]                        ['Anchored_pairs_sleepcorrs_rebinned'][stage][mouse_recday])
                        Anchored_pairs_angles_day_=np.copy(Sleepcorr_pairs_dic[phase_place_diff]                                                      ['Anchored_pairs_angles'][stage][mouse_recday])
                        Anchored_pairs_spatialcorrs_day_=np.copy(Sleepcorr_pairs_dic[phase_place_diff]                                                            ['Anchored_pairs_Spatialcorrs'][stage][mouse_recday])
                        Anchored_pairs_phases_day_=np.copy(Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_phases']                                                            [stage][mouse_recday])                        
                        
                        
                        
                        
                        Anchored_pairs_sleepcorrs_rebinned_.append(Anchored_pairs_sleepcorrs_rebinned_day_)
                        Anchored_pairs_angles_.append(Anchored_pairs_angles_day_)
                        Anchored_pairs_spatialcorrs_.append(Anchored_pairs_spatialcorrs_day_)
                        Anchored_pairs_phases_.append(Anchored_pairs_phases_day_)
                        

                        ###per day analysis
                        filter_boolean_nan_day=~np.isnan(Anchored_pairs_angles_day_)
                        Anchored_pairs_angles_unfiltered_day=Anchored_pairs_angles_day_
                        Anchored_pairs_angles_day_[filter_boolean_nan_day]=                        abs((Anchored_pairs_angles_day_[filter_boolean_nan_day]))
                        
                        

                        if only_far==True:
                            filter_boolean_day_=np.logical_and(Anchored_pairs_spatialcorrs_day_<spatial_corr_thr                                                               ,Anchored_pairs_angles_day_>=180)
                        elif only_close==True:
                            filter_boolean_day_=np.logical_and(Anchored_pairs_spatialcorrs_day_<spatial_corr_thr                                                               ,Anchored_pairs_angles_day_<180)
                            
                        else:
                            filter_boolean_day_=Anchored_pairs_spatialcorrs_day_<spatial_corr_thr

                        if circular_angle==True:
                            Anchored_pairs_angles_day_[Anchored_pairs_angles_day_>=180]=                            360-Anchored_pairs_angles_day_[Anchored_pairs_angles_day_>=180]                        

                        #filter_boolean_day=np.logical_and(filter_boolean_day_==True,Anchored_pairs_angles_day_>0)
                        filter_boolean_day=filter_boolean_day_

                       
                        Anchored_pairs_angles_day=Anchored_pairs_angles_day_[filter_boolean_day]
                        Anchored_pairs_spatialcorrs_day=Anchored_pairs_spatialcorrs_day_[filter_boolean_day]
                        Anchored_pairs_phases_day=Anchored_pairs_phases_day_[filter_boolean_day]
                        Anchored_pairs_sleepcorrs_rebinned_day=Anchored_pairs_sleepcorrs_rebinned_day_[filter_boolean_day]
                        

                        if phase_place_diff==0:
                            
                            

                            Anchored_pairs_angles_bin=np.digitize(Anchored_pairs_angles_day,bins_angles_)

                            binned_corrs_means_day=np.asarray([np.nanmean(Anchored_pairs_sleepcorrs_rebinned_day                                                                      [Anchored_pairs_angles_bin==bin_])                                                           for bin_ in np.arange(num_bins)])
                            
                            cross_corr_dic['angles'][stage][mouse_recday]=Anchored_pairs_angles_day
                            cross_corr_dic['sleep_corrs'][stage][mouse_recday]=Anchored_pairs_sleepcorrs_rebinned_day
                            cross_corr_dic['binned_corrs_means'][mouse_recday][stage]=binned_corrs_means_day
            except:
                print('Not done')


# In[ ]:





# In[40]:


sleep_stages_unique


# In[64]:



angles_all_=np.hstack(([np.hstack((dict_to_array(cross_corr_dic['angles'][stage])))for stage in sleep_stages_unique]))
sleep_corr_all_=np.hstack(([np.hstack((dict_to_array(cross_corr_dic['sleep_corrs'][stage])))                 for stage in sleep_stages_unique ]))


# In[76]:





# In[ ]:





# In[65]:


num_bins=8
scaling_factor=360/num_bins
bins_=np.arange(num_bins+1)*scaling_factor
means_all=[]
sems_all=[]
for stage in sleep_stages_unique:
    print(stage)
    angles_all_stage_=np.hstack((dict_to_array(cross_corr_dic['angles'][stage])))
    sleep_corr_all_stage_=np.hstack((dict_to_array(cross_corr_dic['sleep_corrs'][stage])))


    means=st.binned_statistic(angles_all_stage_,sleep_corr_all_stage_,bins=bins_)[0]
    std=st.binned_statistic(angles_all_stage_,sleep_corr_all_stage_,'std',bins=bins_)[0]
    sem=std/np.sqrt(len(angles_all_stage_))
    #plt.scatter(angles_all_stage_,sleep_corr_all_stage_)
    plt.errorbar(x=np.arange(num_bins),y=means,             yerr=sem, marker='o',markersize=10,color='black')
    plt.show()
    
    xy=column_stack_clean(angles_all_stage_,sleep_corr_all_stage_)
    #print(st.pearsonr(xy[:,0],xy[:,1]))
    
    means_all.append(means)
    sems_all.append(sem)
    

    sleep_corr_all_stage_close=sleep_corr_all_stage_[angles_all_stage_<180]
    angles_all_stage_close=angles_all_stage_[angles_all_stage_<180]
    sleep_corr_all_stage_far=sleep_corr_all_stage_[angles_all_stage_>=180]
    angles_all_stage_far=angles_all_stage_[angles_all_stage_>=180]

    print(st.pearsonr(angles_all_stage_close,sleep_corr_all_stage_close))

    print(st.pearsonr(angles_all_stage_far,sleep_corr_all_stage_far))
    
means_all=np.vstack((means_all)).T
sems_all=np.vstack((sems_all)).T
means_mean=np.mean(means_all,axis=1)
sems_mean=np.sqrt(np.sum(sems_all**2,axis=1)/len(sems_all.T)**2)


# In[71]:


np.shape(angles_all_stage_)


# In[66]:


angles_all_stage_

sleep_corr_all_stage_close=sleep_corr_all_stage_[angles_all_stage_<180]
angles_all_stage_close=angles_all_stage_[angles_all_stage_<180]
sleep_corr_all_stage_far=sleep_corr_all_stage_[angles_all_stage_>=180]
angles_all_stage_far=angles_all_stage_[angles_all_stage_>=180]

print(st.pearsonr(angles_all_stage_close,sleep_corr_all_stage_close))

print(st.pearsonr(angles_all_stage_far,sleep_corr_all_stage_far))


# In[ ]:





# In[67]:


from statsmodels.stats.anova import AnovaRM
from scipy.stats import tukey_hsd



#means=st.binned_statistic(angles_all_,sleep_corr_all_,bins=bins_)[0]
#std=st.binned_statistic(angles_all_,sleep_corr_all_,'std',bins=bins_)[0]
#sem=std/np.sqrt(len(angles_all_))

plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.errorbar(x=np.arange(num_bins),y=means_mean,             yerr=sems_mean, marker='o',markersize=10,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.axvline(3.5,ls='dashed',color='black')

plt.savefig(Ephys_output_folder_dropbox+'Forward_distance_vs_corr_Allsleep.svg',            bbox_inches = 'tight', pad_inches = 0) 
plt.show()

bin_num=st.binned_statistic(angles_all_,sleep_corr_all_,bins=bins_)[2]
corrs_divided=[sleep_corr_all_[bin_num==bin_] for bin_ in np.arange(num_bins)+1]

stats=st.f_oneway(remove_nan(corrs_divided[0]),remove_nan(corrs_divided[1]),                      remove_nan(corrs_divided[2]),remove_nan(corrs_divided[3]),                        remove_nan(corrs_divided[4]),remove_nan(corrs_divided[5]),                      remove_nan(corrs_divided[6]),remove_nan(corrs_divided[7]))
print(stats)

#if stats[1]<0.05:
res = tukey_hsd(remove_nan(corrs_divided[0]),remove_nan(corrs_divided[1]),                      remove_nan(corrs_divided[2]),remove_nan(corrs_divided[3]),                        remove_nan(corrs_divided[4]),remove_nan(corrs_divided[5]),                      remove_nan(corrs_divided[6]),remove_nan(corrs_divided[7]))
print(res)
    
print(st.mannwhitneyu(remove_nan(corrs_divided[0]),remove_nan(corrs_divided[3])))


# In[80]:


sleep_corrs_allbins=[]
for bin_ind, bin_ in enumerate(bins_[:-1]):
    sleep_corrs_all_bin=[]
    for stage in sleep_stages_unique:
        angles_all_stage_=np.hstack((dict_to_array(cross_corr_dic['angles'][stage])))
        sleep_corr_all_stage_=np.hstack((dict_to_array(cross_corr_dic['sleep_corrs'][stage])))


        sleep_corrs_all_bin_stage=sleep_corr_all_stage_[np.logical_and(angles_all_stage_>=bins_[bin_ind],                                             angles_all_stage_<bins_[bin_ind+1])]
        sleep_corrs_all_bin.append(sleep_corrs_all_bin_stage)
        
    sleep_corrs_allbins.append(remove_nan(np.hstack((sleep_corrs_all_bin))))


# In[ ]:





# In[ ]:





# In[81]:


plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False

data=(coeffs_all_task*scaling_factor).T

filtered_data = sleep_corrs_allbins

sns.violinplot(filtered_data, color='grey',alpha=0.5)
#sns.stripplot(filtered_data,color='white',edgecolor='black',linewidth=1,alpha=0.5)
plt.axhline(0,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'Forward_distance_vs_corr_Allsleep_violin.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1)
plt.axhline(0,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'Forward_distance_vs_corr_Allsleep_swarm.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[84]:


num_bins=9
scaling_factor=360/num_bins
bins_=np.arange(num_bins+1)*scaling_factor
min_pairs_day=10
means_all_all=[]
means_all_all_sessions=[]
corr_close_far_all=[]
corr_close_far_all_sessions=[]
days_used_analysis_=[]
num_pairs_all=[]
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    means_all=[]
    sems_all=[]
    corr_close_far_day=[]
    num_pairs_day=[]
    for stage in sleep_stages_unique:
        #print(stage)
        angles_all_stage_=cross_corr_dic['angles'][stage][mouse_recday]
        sleep_corr_all_stage_=cross_corr_dic['sleep_corrs'][stage][mouse_recday]
        
        if len(angles_all_stage_)==0:
            continue
        print(mouse_recday)
        sleep_corr_all_stage_close=sleep_corr_all_stage_[angles_all_stage_<180]
        angles_all_stage_close=angles_all_stage_[angles_all_stage_<180]
        sleep_corr_all_stage_far=sleep_corr_all_stage_[angles_all_stage_>=180]
        angles_all_stage_far=angles_all_stage_[angles_all_stage_>=180]
        
        
        
        if len(angles_all_stage_close)>min_pairs_day:
            corr_close=st.pearsonr(angles_all_stage_close,sleep_corr_all_stage_close)[0]
            days_used_analysis_.append(mouse_recday)
        else:
            corr_close=np.nan
            
        if len(angles_all_stage_far)>min_pairs_day:
            corr_far=st.pearsonr(angles_all_stage_far,sleep_corr_all_stage_far)[0]
        else:
            corr_far=np.nan
            
        corr_close_far_day.append([corr_close,corr_far])
        
        
        num_pairs=len(angles_all_stage_)
        num_pairs_day.append(num_pairs)
        

        means=st.binned_statistic(angles_all_stage_,sleep_corr_all_stage_,bins=bins_)[0]
        std=st.binned_statistic(angles_all_stage_,sleep_corr_all_stage_,'std',bins=bins_)[0]
        sem=std/np.sqrt(len(angles_all_stage_))
        #plt.scatter(angles_all_stage_,sleep_corr_all_stage_)
        #plt.errorbar(x=np.arange(num_bins),y=means,\
        #         yerr=sem, marker='o',markersize=10,color='black')
        #plt.show()

        xy=column_stack_clean(angles_all_stage_,sleep_corr_all_stage_)
        #print(st.pearsonr(xy[:,0],xy[:,1]))

        means_all.append(means)
    if len(means_all)>0:
        means_all_day=np.nanmean(np.vstack((means_all)),axis=0)
        means_all_all.append(means_all_day)
        means_all_all_sessions.append(np.vstack((means_all)))
    
    
        corr_close_far_all.append(np.nanmean(np.vstack((corr_close_far_day)),axis=0))
        corr_close_far_all_sessions.append(np.vstack((corr_close_far_day)))
        num_pairs_all.append(np.min(num_pairs_day))
        
    else:
        print('Not used')
corr_close_far_all=np.vstack((corr_close_far_all))
corr_close_far_all_sessions=np.vstack((corr_close_far_all_sessions))
days_used_analysis_=np.unique(np.hstack((days_used_analysis_)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[85]:


#bar_plotX(corr_close_far_all.T,'none', -0.2, 0.1, 'nopoints', 'paired', 0.025)


plt.rcParams["figure.figsize"] = (2,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.bar(np.arange(2),np.nanmean(corr_close_far_all,axis=0),        yerr=st.sem(corr_close_far_all,axis=0,nan_policy='omit'),        color='black',ecolor='grey',capsize=3,width=1)


plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.ylim(-0.2, 0.15)
plt.savefig(Ephys_output_folder_dropbox+angle_type+'_closevsfar_correlation_bar_days.svg',                bbox_inches = 'tight', pad_inches = 0) 
plt.show()

xy=column_stack_clean(corr_close_far_all[:,0],corr_close_far_all[:,1])
print(len(xy))
print(st.ttest_rel(xy[:,0],xy[:,1]))
print(st.ttest_1samp(remove_nan(corr_close_far_all[:,0]),0))
print(st.ttest_1samp(remove_nan(corr_close_far_all[:,1]),0))


# In[ ]:





# In[89]:


plt.rcParams["figure.figsize"] = (2,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False


xy=column_stack_clean(corr_close_far_all_sessions[:,0],corr_close_far_all_sessions[:,1])

plt.bar(np.arange(2),np.nanmean(xy,axis=0),        yerr=st.sem(xy,axis=0,nan_policy='omit'),        color='black',ecolor='grey',capsize=3,width=1)


plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.ylim(-0.15, 0.15)
plt.savefig(Ephys_output_folder_dropbox+angle_type+'_closevsfar_correlation_bar_sessions.svg',                bbox_inches = 'tight', pad_inches = 0) 
plt.show()
print(len(xy))
print(st.ttest_rel(xy[:,0],xy[:,1]))
print(st.ttest_1samp(xy[:,0],0))
print(st.ttest_1samp(xy[:,1],0))

plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
noplot_scatter(xy[:,0],xy[:,1],'black')
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(axis='both',  labelsize=15)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+angle_type+'_closevsfar_correlation_scatter_sessions.svg',           bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(st.wilcoxon(xy[:,0],xy[:,1]))
print(len(xy))


# In[ ]:





# In[49]:


means_all_all=np.vstack((means_all_all))
means_all_mean=np.nanmean(means_all_all,axis=0)
means_all_sem=st.sem(means_all_all,axis=0,nan_policy='omit')

plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.errorbar(x=np.arange(num_bins),y=means_all_mean,             yerr=means_all_sem, marker='o',markersize=10,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'Forward_distance_vs_corr_Allsleep_days.svg',            bbox_inches = 'tight', pad_inches = 0) 
plt.show()


# In[ ]:





# In[50]:


means_all_all_sessions=np.vstack((means_all_all_sessions))
means_all_mean=np.nanmean(means_all_all,axis=0)
means_all_sem=st.sem(means_all_all_sessions,axis=0,nan_policy='omit')

plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.errorbar(x=np.arange(num_bins),y=means_all_mean,             yerr=means_all_sem, marker='o',markersize=10,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'Forward_distance_vs_corr_Allsleep_sessions.svg',            bbox_inches = 'tight', pad_inches = 0) 
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[318]:


###Angles and Sleep correlations for pairs across anchors - across sleep time epochs
#Sleepcorr_pairs_timebin_dic=rec_dd()
shifts=np.linspace(-40,40,81) ## i.e. correlating with n bins in the future

num_phases=3
num_locations=9


num_phase_place_diffs_=5
useplace_phase_diff=True
use_tuned=False
use_anchored=True
use_nonspatial=False
use_spatial=False
use_anchor_angles=False
use_FRstable=True
bin_length=250 ##for rebinned correlations
thr_lower=30
thr_upper=360-thr_lower
sleep_dicX=Sleepcorr_matrix_shifted_dic

rerun=True

num_trials_thr=6

for phase_place_diff in [0]:#np.arange(num_phase_place_diffs_):
    print(phase_place_diff)
    for time_bin in np.arange(num_timebins):
        print(time_bin)
        
        for mouse_recday in day_type_dicX['combined_ABCDonly']:
            try:
                print(mouse_recday)

                if rerun==False:
                    if len(Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]['Anchored_pairs'][mouse_recday])>0:
                        print('Already analysed')
                        continue



                num_sessions=len(session_dic['awake'][mouse_recday])
                non_repeat_ses=non_repeat_ses_maker(mouse_recday)

                ses_trials=np.where(num_trials>=num_trials_thr)[0]
                non_repeat_ses=np.intersect1d(non_repeat_ses,ses_trials)

                found_ses=[]
                for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
                    try:
                        Neuron_raw=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                        found_ses.append(ses_ind)

                    except:
                        print('Files not found for session '+str(ses_ind))
                        continue

                found_ses=np.asarray(found_ses)


                #angle_units=Xneuron_correlations['combined_ABCDonly']['angle_units'][mouse_recday]
                #max_bins=Xneuron_correlations['combined_ABCDonly']['Max_bins'][mouse_recday]#[0]
                angles=Xneuron_correlations['combined_ABCDonly']['Angles'][mouse_recday]

                Anchor_lags=Anchor_trial_dic['Best_shift_time'][mouse_recday]    
                Anchor_lags_mean=np.rad2deg(st.circmean(np.deg2rad(Anchor_lags),axis=1,nan_policy='omit'))
                #angles_anchor=np.vstack(([positive_angle([Anchor_lags_mean[ii]-Anchor_lags_mean[jj]\
                #                                   for ii in range(len(Anchor_lags_mean))])\
                # for jj in range(len(Anchor_lags_mean))]))

                angles_anchor=[np.vstack(([[Anchor_lags[ii,ses]-Anchor_lags[jj,ses]                                                   for ii in range(len(Anchor_lags))]                 for jj in range(len(Anchor_lags))])) for ses in np.arange(len(Anchor_lags.T))]


                Best_anchor_all=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday]
                Spatial_maps_all=Spatial_anchoring_dic['Phase_shifted_maps'][mouse_recday][:,:,0] ##0 as in 0 lag maps
                FR_bool=FR_sleep_boolean_dic[mouse_recday]
                FR_included=np.where(FR_bool==True)[0]

                awake_sesinds=Awake_sesind_dic[mouse_recday]
                sleep_stages=sleep_stage_dic[mouse_recday]
                sleep_stages_unique=np.unique(sleep_stages)

                first_day2_boolean=Awake_sesind_dic['first_day2_boolean'][mouse_recday]

                neurons_tuned=np.where(Tuned_dic['State_zmax_bool'][mouse_recday]==True)[0]
                Anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]
                Anchored_neurons=np.where(Anchored_bool==True)[0]

                Anchor_lags_used=Anchor_trial_dic['Best_shift_time'][mouse_recday]
                Anchor_lags_used_mean=np.rad2deg(st.circmean(np.deg2rad(Anchor_lags_used),axis=1,nan_policy='omit'))
                non_spatial_neurons=np.where(np.logical_and(Anchor_lags_used_mean>thr_lower,Anchor_lags_used_mean<thr_upper)                                             ==True)[0]
                spatial_neurons=np.where(np.logical_or(Anchor_lags_used_mean<thr_lower,Anchor_lags_used_mean>thr_upper)                                             ==True)[0]


                for stage in sleep_stages_unique:

                    if stage=='preX':
                        ses_ind=0
                    elif 'post' in stage:
                        ses_ind_unique=int(stage.split('post',1)[1])
                        ses_ind=np.arange(num_sessions)[np.where(awake_sesinds==ses_ind_unique)[0][0]]
                    elif stage=='preX2':
                        ses_ind=np.where(first_day2_boolean==True)[0][0]

                    ses_ind_ind_=np.where(found_ses==ses_ind)[0]
                    if len(ses_ind_ind_)==0:
                        continue
                    ses_ind_ind=ses_ind_ind_[0]

                    #phases=Xneuron_phaseangle_dic[ses_ind][mouse_recday] ##phase angle diff
                    phases=Phase_diff_dic[mouse_recday][ses_ind] ##phase angle diff

                    if len(phases)==0:
                        continue


                    Spatial_corrs_ses=np.corrcoef(Spatial_maps_all[:,ses_ind])

                    if len(Spatial_maps_all)<2:
                        print('less than 2 neurons')
                        continue

                    All_anchored_pairsX=[]
                    All_anchored_pairsY=[]
                    All_anchored_pairs_angles=[]
                    All_anchored_pairs_phases=[]
                    All_anchored_pairs_Spatialcorrs=[]


                    for phase in np.arange(num_phases):
                        if useplace_phase_diff==True:
                            phase_next=(phase+phase_place_diff)%3
                        else:
                            phase_next=phase
                        for location in np.arange(num_locations):
                            locations_next=np.where(mindistance_mat[location]==phase_place_diff)[0]

                            neurons_anchoredX=np.where(np.logical_and(Best_anchor_all[:,0]==phase,                                                                      Best_anchor_all[:,1]==location,                                                                     FR_bool==True))[0]

                            neurons_anchoredY=np.where(np.logical_and(Best_anchor_all[:,0]==phase_next                                                                      ,np.isin(Best_anchor_all[:,1],locations_next),                                                                     FR_bool==True))[0]
                            if use_tuned==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,neurons_tuned)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,neurons_tuned)
                            if use_anchored==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,Anchored_neurons)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,Anchored_neurons)
                            if use_nonspatial==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,non_spatial_neurons)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,non_spatial_neurons)
                            if use_spatial==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,spatial_neurons)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,spatial_neurons)

                            if use_FRstable==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,FR_included)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,FR_included)

                            All_anchored_pairsX.append(neurons_anchoredX)
                            All_anchored_pairsY.append(neurons_anchoredY)

                            '''if phase_place_diff>0:
                                if use_anchor_angles==True:
                                    angle_mat_pairs_=angles_anchor[ses_ind_ind][neurons_anchoredX][:,neurons_anchoredY]
                                else:
                                    angle_mat_pairs_=angles[:,:,ses_ind][neurons_anchoredX][:,neurons_anchoredY]

                                phase_mat_pairs_=phases[neurons_anchoredX][:,neurons_anchoredY]
                                Spatial_corrs_pairs_=Spatial_corrs_ses[neurons_anchoredX][:,neurons_anchoredY]
                            elif phase_place_diff==0:
                                if use_anchor_angles==True:
                                    angle_mat_pairs_=matrix_triangle(angles_anchor[ses_ind_ind][neurons_anchoredX]\
                                                                     [:,neurons_anchoredY],'lower')
                                else:
                                    angle_mat_pairs_=matrix_triangle(angles[:,:,ses_ind][neurons_anchoredX]\
                                                                     [:,neurons_anchoredY],'lower')

                                phase_mat_pairs_=matrix_triangle(phases[neurons_anchoredX][:,neurons_anchoredY],\
                                                                 'lower')
                                Spatial_corrs_pairs_=matrix_triangle(Spatial_corrs_ses[neurons_anchoredX]\
                                                                     [:,neurons_anchoredY],'lower')'''
                            
                            if phase_place_diff!=0:
                                angle_mat_pairs_anchor_=angles_anchor[ses_ind_ind][neurons_anchoredX][:,neurons_anchoredY]
                                if use_anchor_angles==True:
                                    angle_mat_pairs_=angles_anchor[ses_ind_ind][neurons_anchoredX][:,neurons_anchoredY]
                                    angle_mat_pairs_=abs(angle_mat_pairs_)
                                else:
                                    angle_mat_pairs_=angles[:,:,ses_ind][neurons_anchoredX][:,neurons_anchoredY]

                                    ##aligning forward angles to anchor
                                    #angle_mat_pairs_anchor_=angle_mat_pairs_anchor_.squeeze()
                                    #angle_mat_pairs_=angle_mat_pairs_.squeeze()
                                    if len(angle_mat_pairs_)>0:
                                        angle_mat_pairs_[angle_mat_pairs_anchor_<0]=                                        360-angle_mat_pairs_[angle_mat_pairs_anchor_<0]#

                                        #angle_mat_pairs_=[angle_mat_pairs_]

                                phase_mat_pairs_=phases[neurons_anchoredX][:,neurons_anchoredY]
                                Spatial_corrs_pairs_=Spatial_corrs_ses[neurons_anchoredX][:,neurons_anchoredY]
                            elif phase_place_diff==0:
                                angle_mat_pairs_anchor_=matrix_triangle(angles_anchor[ses_ind_ind][neurons_anchoredX]                                                                     [:,neurons_anchoredY],'lower')
                                if use_anchor_angles==True:
                                    angle_mat_pairs_=matrix_triangle(angles_anchor[ses_ind_ind][neurons_anchoredX]                                                                     [:,neurons_anchoredY],'lower')
                                    angle_mat_pairs_=abs(angle_mat_pairs_)
                                else:
                                    angle_mat_pairs_=matrix_triangle(angles[:,:,ses_ind][neurons_anchoredX]                                                                     [:,neurons_anchoredY],'lower')

                                    ##aligning forward angles to anchor
                                    #angle_mat_pairs_anchor_=angle_mat_pairs_anchor_.squeeze()
                                    #angle_mat_pairs_=angle_mat_pairs_.squeeze()
                                    if len(angle_mat_pairs_)>0:
                                        angle_mat_pairs_[angle_mat_pairs_anchor_<0]=                                        360-angle_mat_pairs_[angle_mat_pairs_anchor_<0]

                                        #angle_mat_pairs_=[angle_mat_pairs_]
                                        
                                    phase_mat_pairs_=matrix_triangle(phases[neurons_anchoredX][:,neurons_anchoredY],                                                                 'lower')
                                    Spatial_corrs_pairs_=matrix_triangle(Spatial_corrs_ses[neurons_anchoredX]                                                                     [:,neurons_anchoredY],'lower')



                            if len(angle_mat_pairs_)>0 and phase_place_diff>0:
                                angle_mat_pairs=np.hstack((angle_mat_pairs_))
                                phase_mat_pairs=np.hstack((phase_mat_pairs_))
                                Spatial_corrs_pairs_=np.hstack((Spatial_corrs_pairs_))
                            else:
                                angle_mat_pairs=angle_mat_pairs_
                                phase_mat_pairs=phase_mat_pairs_
                                Spatial_corrs_pairs_=Spatial_corrs_pairs_

                            All_anchored_pairs_angles.append(angle_mat_pairs)
                            All_anchored_pairs_phases.append(phase_mat_pairs)
                            All_anchored_pairs_Spatialcorrs.append(Spatial_corrs_pairs_)

                    All_anchored_pairs_angles_=All_anchored_pairs_angles
                    All_anchored_pairs_angles=concatenate_complex2(All_anchored_pairs_angles)
                    All_anchored_pairs_phases=concatenate_complex2(All_anchored_pairs_phases)
                    All_anchored_pairs_Spatialcorrs=concatenate_complex2(All_anchored_pairs_Spatialcorrs)

                    All_anchored_pairs_sleepcorrs_allshifts=np.zeros((len(shifts),len(All_anchored_pairs_angles)))
                    All_anchored_pairs_sleepcorrs_allshifts[:]=np.nan
                    for shift_ind,shift in enumerate(shifts):
                        shift=int(shift)
                        All_anchored_pairs_sleepcorrs=[]
                        for phase in np.arange(num_phases):
                            if useplace_phase_diff==True:
                                phase_next=(phase+phase_place_diff)%3
                            else:
                                phase_next=phase
                            for location in np.arange(num_locations):
                                locations_next=np.where(mindistance_mat[location]==phase_place_diff)[0]
                                neurons_anchoredX=np.where(np.logical_and(Best_anchor_all[:,0]==phase,                                                                      Best_anchor_all[:,1]==location,                                                                     FR_bool==True))[0]

                                neurons_anchoredY=np.where(np.logical_and(Best_anchor_all[:,0]==phase_next                                                                          ,np.isin(Best_anchor_all[:,1],locations_next),                                                                         FR_bool==True))[0]
                                if use_tuned==True:
                                    neurons_anchoredX=np.intersect1d(neurons_anchoredX,neurons_tuned)
                                    neurons_anchoredY=np.intersect1d(neurons_anchoredY,neurons_tuned)
                                if use_anchored==True:
                                    neurons_anchoredX=np.intersect1d(neurons_anchoredX,Anchored_neurons)
                                    neurons_anchoredY=np.intersect1d(neurons_anchoredY,Anchored_neurons)
                                if use_nonspatial==True:
                                    neurons_anchoredX=np.intersect1d(neurons_anchoredX,non_spatial_neurons)
                                    neurons_anchoredY=np.intersect1d(neurons_anchoredY,non_spatial_neurons)
                                if use_spatial==True:
                                    neurons_anchoredX=np.intersect1d(neurons_anchoredX,spatial_neurons)
                                    neurons_anchoredY=np.intersect1d(neurons_anchoredY,spatial_neurons)

                                if use_FRstable==True:
                                    neurons_anchoredX=np.intersect1d(neurons_anchoredX,FR_included)
                                    neurons_anchoredY=np.intersect1d(neurons_anchoredY,FR_included)




                                #sleepcorr_mat=sleep_dicX[stage][shift][mouse_recday]
                                sleepcorr_mat_shifted=sleep_dicX['divided_time'][time_bin][stage][shift][mouse_recday]

                                if len(sleepcorr_mat_shifted)==0:
                                    continue

                                if phase_place_diff>0:
                                    sleepcorr_mat_pairs_=sleepcorr_mat_shifted[neurons_anchoredX][:,neurons_anchoredY]

                                elif phase_place_diff==0:
                                    sleepcorr_mat_pairs_=matrix_triangle(sleepcorr_mat_shifted[neurons_anchoredX]                                                                         [:,neurons_anchoredY],'lower')


                                if len(sleepcorr_mat_pairs_)>0 and phase_place_diff>0:
                                    sleepcorr_mat_pairs=np.hstack((sleepcorr_mat_pairs_))
                                else:
                                    sleepcorr_mat_pairs=sleepcorr_mat_pairs_

                                All_anchored_pairs_sleepcorrs.append(sleepcorr_mat_pairs)

                        if len(All_anchored_pairs_sleepcorrs)==0:
                            continue

                        All_anchored_pairs_sleepcorrs_=All_anchored_pairs_sleepcorrs
                        All_anchored_pairs_sleepcorrs=concatenate_complex2(All_anchored_pairs_sleepcorrs)
                        All_anchored_pairs_sleepcorrs_allshifts[shift_ind]=All_anchored_pairs_sleepcorrs



                    sleepcorr_mat=Sleepcorr_matrix_rebinned_timebins_dic[str(bin_length)][time_bin][stage][mouse_recday]

                    if len(sleepcorr_mat)==0:
                        continue

                    All_anchored_pairs_sleepcorrs=[]
                    for phase in np.arange(num_phases):
                        if useplace_phase_diff==True:
                            phase_next=(phase+phase_place_diff)%3
                        else:
                            phase_next=phase
                        for location in np.arange(num_locations):
                            locations_next=np.where(mindistance_mat[location]==phase_place_diff)[0]
                            neurons_anchoredX=np.where(np.logical_and(Best_anchor_all[:,0]==phase,                                                                  Best_anchor_all[:,1]==location,                                                                 FR_bool==True))[0]

                            neurons_anchoredY=np.where(np.logical_and(Best_anchor_all[:,0]==phase_next                                                                      ,np.isin(Best_anchor_all[:,1],locations_next),                                                                     FR_bool==True))[0]
                            if use_tuned==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,neurons_tuned)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,neurons_tuned)
                            if use_anchored==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,Anchored_neurons)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,Anchored_neurons)
                            if use_nonspatial==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,non_spatial_neurons)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,non_spatial_neurons)
                            if use_spatial==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,spatial_neurons)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,spatial_neurons)

                            if use_FRstable==True:
                                neurons_anchoredX=np.intersect1d(neurons_anchoredX,FR_included)
                                neurons_anchoredY=np.intersect1d(neurons_anchoredY,FR_included)






                            if phase_place_diff>0:
                                sleepcorr_mat_pairs_=sleepcorr_mat[neurons_anchoredX][:,neurons_anchoredY]

                            elif phase_place_diff==0:
                                sleepcorr_mat_pairs_=matrix_triangle(sleepcorr_mat[neurons_anchoredX][:,neurons_anchoredY],                                                                     'lower')

                            if len(sleepcorr_mat_pairs_)>0 and phase_place_diff>0:
                                sleepcorr_mat_pairs=np.hstack((sleepcorr_mat_pairs_))
                            else:
                                sleepcorr_mat_pairs=sleepcorr_mat_pairs_

                            All_anchored_pairs_sleepcorrs.append(sleepcorr_mat_pairs)

                    All_anchored_pairs_sleepcorrs=concatenate_complex2(All_anchored_pairs_sleepcorrs)
                    All_anchored_pairs_sleepcorrs_rebinned=All_anchored_pairs_sleepcorrs



                    Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]['Anchored_pairs'][mouse_recday]=                    All_anchored_pairsX,All_anchored_pairsY
                    Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]['Anchored_pairs_angles'][stage][mouse_recday]=                    All_anchored_pairs_angles
                    Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]['Anchored_pairs_phases'][stage][mouse_recday]=                    All_anchored_pairs_phases
                    All_anchored_pairs_phases
                    Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]['Anchored_pairs_Spatialcorrs'][stage]                    [mouse_recday]=All_anchored_pairs_Spatialcorrs

                    Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]['Anchored_pairs_sleepcorrs'][stage]                    [mouse_recday]=All_anchored_pairs_sleepcorrs_allshifts
                    Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned'][stage]                    [mouse_recday]=All_anchored_pairs_sleepcorrs_rebinned
            except:
                print('Not done')


# In[ ]:





# In[ ]:





# In[314]:


num_pairs_all=[]
time_bin=2
phase_place_diff=0
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    num_pairs_all.append(len(Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]['Anchored_pairs'][mouse_recday]))
    
num_pairs_all=np.hstack((num_pairs_all))
np.sum(num_pairs_all>0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[456]:


##Task space distance and spatial corr vs sleep corrs
corr_crossanchor_task_all=[]
corr_crossanchor_space_all=[]
coeffs_crossanchor_all=[]
sems_crossanchor_all=[]
only_far=False
co_regress=True
#circular_angle=True
Regression_timebins_dic=rec_dd()
stage_type='All'
if stage_type=='pre':
    stages=['preX','preX2']
elif stage_type=='post':
    stages=['post0', 'post1', 'post2', 'post3', 'post4', 'post5']
elif stage_type=='All':
    stages=sleep_stages_unique
for circular_angle in [True]:
    corr_crossanchor_task=[]
    corr_crossanchor_space=[]
    coeffs_crossanchor=[]
    sems_crossanchor=[]
    N_all=[]
    #for phase_place_diff in np.arange(num_phase_place_diffs_):
    for time_bin in np.arange(num_timebins):
        
        phase_place_diff=0
        print(phase_place_diff)
        corr_phase_place_diff=[]
        corr_spatial_corr=[]

        coeffs_phase_place_diff=[]
        sems_phase_place_diff=[]
        N_timebin=[]
        ##['post0', 'post1', 'post2', 'post3', 'post4', 'post5']:#sleep_stages_unique: ['preX','preX2']:
        for stage in stages:#['preX','preX2']:#['preX']:#,'preX2']:#
            print(stage)

            Anchored_pairs_sleepcorrs_rebinned_=[]
            Anchored_pairs_angles_=[]
            Anchored_pairs_spatialcorrs_=[]
            Anchored_pairs_phases_=[]
            for mouse_recday in day_type_dicX['combined_ABCDonly']:
                print(mouse_recday)
                mouse=mouse_recday.split('_',1)[0]
                cohort=Mice_cohort_dic[mouse]
                Ephys_type=Cohort_ephys_type_dic[int(cohort)]

                #if Ephys_type=='Neuropixels':
                #    print('skipped')
                #    continue
                if len(Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned']                       [stage][mouse_recday])>0:

                    
                    Anchored_pairs_sleepcorrs_rebinned_day_=np.copy(Sleepcorr_pairs_timebin_dic[time_bin]                                                                    [phase_place_diff]                    ['Anchored_pairs_sleepcorrs_rebinned'][stage][mouse_recday])
                    Anchored_pairs_angles_day_=np.copy(Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]                                                  ['Anchored_pairs_angles'][stage][mouse_recday])
                    Anchored_pairs_spatialcorrs_day_=np.copy(Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]                                                        ['Anchored_pairs_Spatialcorrs'][stage][mouse_recday])
                    Anchored_pairs_phases_day_=np.copy(Sleepcorr_pairs_timebin_dic[time_bin][phase_place_diff]                                                       ['Anchored_pairs_phases'][stage][mouse_recday])                        


                    Anchored_pairs_sleepcorrs_rebinned_.append(Anchored_pairs_sleepcorrs_rebinned_day_)
                    Anchored_pairs_angles_.append(Anchored_pairs_angles_day_)
                    Anchored_pairs_spatialcorrs_.append(Anchored_pairs_spatialcorrs_day_)
                    Anchored_pairs_phases_.append(Anchored_pairs_phases_day_)
                    
                    Anchored_pairs_angles_day_=np.copy(Anchored_pairs_angles_day_)
                    
                    ###per day analysis
                    filter_boolean_nan_day=~np.isnan(Anchored_pairs_angles_day_)
                    Anchored_pairs_angles_unfiltered_day=np.copy(Anchored_pairs_angles_day_)
                    Anchored_pairs_angles_day_[filter_boolean_nan_day]=                    abs((Anchored_pairs_angles_day_[filter_boolean_nan_day]))



                    if only_far==True:
                        filter_boolean_day_=np.logical_and(Anchored_pairs_spatialcorrs_day_<0.9                                                           ,Anchored_pairs_angles_day_>=180)
                    else:
                        filter_boolean_day_=Anchored_pairs_spatialcorrs_day_<0.9

                    if circular_angle==True:
                        Anchored_pairs_angles_forward_day_=np.copy(Anchored_pairs_angles_day_)
                        Anchored_pairs_angles_day_[Anchored_pairs_angles_day_>=180]=                        360-Anchored_pairs_angles_day_[Anchored_pairs_angles_day_>=180]

                    #print(np.max(Anchored_pairs_angles_day))


                    filter_boolean_day=np.logical_and(filter_boolean_day_==True,Anchored_pairs_angles_day_>0)

                    Anchored_pairs_angles_day=Anchored_pairs_angles_day_[filter_boolean_day]
                    Anchored_pairs_angles_forward_day=Anchored_pairs_angles_forward_day_[filter_boolean_day]
                    Anchored_pairs_spatialcorrs_day=Anchored_pairs_spatialcorrs_day_[filter_boolean_day]
                    Anchored_pairs_phases_day=Anchored_pairs_phases_day_[filter_boolean_day]
                    Anchored_pairs_sleepcorrs_rebinned_day=Anchored_pairs_sleepcorrs_rebinned_day_[filter_boolean_day]
                    
                    ###regression
                    
                    circular_forward_identical_bool=np.sum(Anchored_pairs_angles_day==Anchored_pairs_angles_forward_day)                    ==len(Anchored_pairs_angles_day)

                    if co_regress==True:
                        X=np.column_stack((st.zscore(Anchored_pairs_angles_day,nan_policy='omit'),                                           st.zscore(Anchored_pairs_angles_forward_day,nan_policy='omit'),                                           st.zscore(Anchored_pairs_spatialcorrs_day,nan_policy='omit'),                                  st.zscore(Anchored_pairs_phases_day,nan_policy='omit'),                                           np.repeat(1,len(Anchored_pairs_angles_day))))

                    else:
                        X=np.column_stack((st.zscore(Anchored_pairs_angles_day,nan_policy='omit'),                                           st.zscore(Anchored_pairs_spatialcorrs_day,nan_policy='omit'),                                  st.zscore(Anchored_pairs_phases_day,nan_policy='omit'),                                           np.repeat(1,len(Anchored_pairs_angles_day))))
                    
                    if np.isnan(np.mean(X))==True:
                        coeffs=np.repeat(np.nan,np.shape(X)[1])
                        std_errors=np.repeat(np.nan,np.shape(X)[1])
                    
                    elif circular_forward_identical_bool==True or is_invertible(X.T @ X)==False:
                        coeffs=np.repeat(np.nan,np.shape(X)[1])
                        std_errors=np.repeat(np.nan,np.shape(X)[1])
                        
                    else:

                        y=Anchored_pairs_sleepcorrs_rebinned_day

                        y=y[~np.isnan(X[:,2])]
                        X=X[~np.isnan(X[:,2])]

                        y=y[np.sum(np.isnan(X),axis=1)==0]
                        X=X[np.sum(np.isnan(X),axis=1)==0]

                        X=X[~np.isnan(y)]
                        y=y[~np.isnan(y)]

                        if len(X)==0 or len(y)==0:
                            continue

                        N=len(X)
                        p=len(X.T)

                        N_place_phase_diff.append(N)

                        model=LinearRegression()
                        reg=model.fit(X, y)
                        coeffs=reg.coef_

                        y_hat = model.predict(X)
                        residuals = y - y_hat
                        residual_sum_of_squares = residuals.T @ residuals
                        sigma_squared_hat = residual_sum_of_squares / (N - p)

                        var_beta_hat = np.linalg.inv(X.T @ X) * sigma_squared_hat

                        std_errors=[]
                        for indx in np.arange(p):
                            std_err=var_beta_hat[indx, indx] ** 0.5
                            std_errors.append(std_err)
                    
                    Regression_timebins_dic['coeffs'][time_bin][stage][mouse_recday]=coeffs
                    Regression_timebins_dic['std_errors'][time_bin][stage][mouse_recday]=std_errors
                    
                    
            
            if len(Anchored_pairs_sleepcorrs_rebinned_)==0:
                continue
            Anchored_pairs_sleepcorrs_rebinned_=np.hstack((Anchored_pairs_sleepcorrs_rebinned_))
            Anchored_pairs_angles_=np.hstack((Anchored_pairs_angles_))
            Anchored_pairs_spatialcorrs_=np.hstack((Anchored_pairs_spatialcorrs_))
            Anchored_pairs_phases_=np.hstack((Anchored_pairs_phases_))




            filter_boolean_nan=~np.isnan(Anchored_pairs_angles_)
            Anchored_pairs_angles_[filter_boolean_nan]=abs((Anchored_pairs_angles_[filter_boolean_nan]))
            ###taking absolute angle between neurons

            if only_far==True:
                filter_boolean_=np.logical_and(Anchored_pairs_spatialcorrs_<0.9,Anchored_pairs_angles_>=180)
            else:
                filter_boolean_=Anchored_pairs_spatialcorrs_<0.9

            if circular_angle==True:
                Anchored_pairs_angles_forward=np.copy(Anchored_pairs_angles_)
                Anchored_pairs_angles_[Anchored_pairs_angles_>=180]=360-Anchored_pairs_angles_                [Anchored_pairs_angles_>=180]


            else:
                Anchored_pairs_angles_forward=np.copy(Anchored_pairs_angles_)
                #filter_boolean=np.logical_and(filter_boolean_==True,\
                #                              np.logical_and(Anchored_pairs_angles_>0,Anchored_pairs_angles_<350))
            filter_boolean=np.logical_and(filter_boolean_==True,Anchored_pairs_angles_>0)
            filter_boolean=np.logical_and(filter_boolean,~np.isnan(Anchored_pairs_phases_))
            #filter_boolean=np.logical_and(filter_boolean_,filter_boolean_nan)

            Anchored_pairs_angles=Anchored_pairs_angles_[filter_boolean]
            Anchored_pairs_angles_forward=Anchored_pairs_angles_forward[filter_boolean]
            Anchored_pairs_spatialcorrs=Anchored_pairs_spatialcorrs_[filter_boolean]
            Anchored_pairs_phases=Anchored_pairs_phases_[filter_boolean]
            Anchored_pairs_sleepcorrs_rebinned=Anchored_pairs_sleepcorrs_rebinned_[filter_boolean]

            ###regression

            if co_regress==True:
                X=np.column_stack((st.zscore(Anchored_pairs_angles,nan_policy='omit'),                                   st.zscore(Anchored_pairs_angles_forward,nan_policy='omit'),                                   st.zscore(Anchored_pairs_spatialcorrs,nan_policy='omit'),                          st.zscore(Anchored_pairs_phases,nan_policy='omit'),                                   np.repeat(1,len(Anchored_pairs_angles))))

            else:
                X=np.column_stack((st.zscore(Anchored_pairs_angles,nan_policy='omit'),                                   st.zscore(Anchored_pairs_spatialcorrs,nan_policy='omit'),                          st.zscore(Anchored_pairs_phases,nan_policy='omit'),                                   np.repeat(1,len(Anchored_pairs_angles))))


            y=Anchored_pairs_sleepcorrs_rebinned

            y=y[~np.isnan(X[:,2])]
            X=X[~np.isnan(X[:,2])]
            
            y=y[np.sum(np.isnan(X),axis=1)==0]
            X=X[np.sum(np.isnan(X),axis=1)==0]

            X=X[~np.isnan(y)]
            y=y[~np.isnan(y)]
            
            if len(X)==0 or len(y)==0:
                continue

            N=len(X)
            p=len(X.T)
            
            N_timebin.append(N)

            model=LinearRegression()
            reg=model.fit(X, y)
            coeffs=reg.coef_
            coeffs_phase_place_diff.append(coeffs)

            y_hat = model.predict(X)
            residuals = y - y_hat
            residual_sum_of_squares = residuals.T @ residuals
            sigma_squared_hat = residual_sum_of_squares / (N - p)
            
            try:
                var_beta_hat = np.linalg.inv(X.T @ X) * sigma_squared_hat
            except Exception as e:
                print(e)
                continue

            std_errors=[]
            for indx in np.arange(p):
                std_err=var_beta_hat[indx, indx] ** 0.5
                std_errors.append(std_err)

            sems_phase_place_diff.append(std_errors)


        coeffs_phase_place_diff=np.vstack((coeffs_phase_place_diff))
        sems_phase_place_diff=np.vstack((sems_phase_place_diff))


        #corr_crossanchor_task.append(corr_phase_place_diff)
        #corr_crossanchor_space.append(corr_spatial_corr)
        coeffs_crossanchor.append(coeffs_phase_place_diff)
        sems_crossanchor.append(sems_phase_place_diff)
        
        N_timebin=np.asarray(N_timebin).T
        
        N_all.append(N_timebin)
        
        
    #corr_crossanchor_task_all.append(corr_crossanchor_task)
    #corr_crossanchor_space_all.append(corr_crossanchor_space)
    coeffs_crossanchor_all.append(coeffs_crossanchor)
    sems_crossanchor_all.append(sems_crossanchor)
    
    
N_all=np.asarray(N_all).T


# In[ ]:





# In[457]:


filter_boolean_day_


# In[458]:


Anchored_pairs_angles_day_


# In[ ]:





# In[ ]:





# In[459]:


from statsmodels.stats.anova import AnovaRM
from scipy.stats import tukey_hsd


plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

time_bins_used=3

for coeff_ind, coeff_type in enumerate(['Circular_distance','Forward_distance','Spatial_corr']):
    coeffs_timebins_all_means=np.hstack(([np.nanmean(coeffs_crossanchor_all[0][time_bin][:,coeff_ind])                                          for time_bin in np.arange(num_timebins)]))
    coeffs_timebins_all_sems=np.hstack(([np.sqrt(np.sum(sems_crossanchor_all[0][time_bin][:,coeff_ind]**2)/                                                 len(sems_crossanchor_all[0][time_bin][:,coeff_ind])**2)                for time_bin in np.arange(num_timebins)]))


    plt.errorbar(x=np.arange(len(coeffs_timebins_all_means[:time_bins_used])),                 y=coeffs_timebins_all_means[:time_bins_used]*scaling_factor,                 yerr=coeffs_timebins_all_sems[:time_bins_used]*scaling_factor, marker='o',markersize=10,color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.axhline(0,ls='dashed',color='black',linewidth=4)
    #plt.ylim(-0.0042,0)

    plt.savefig(Ephys_output_folder_dropbox+coeff_type+'_beta_vs_timebins_Allsleep.svg',                bbox_inches = 'tight', pad_inches = 0) 
    plt.show()


# In[ ]:





# In[460]:


multipletests([0,0,0])[1]


# In[ ]:





# In[462]:


from statsmodels.sandbox.stats.multicomp import multipletests

time_bins_used=3
equal_var=False
if equal_var==True:
    addition=''
else:
    addition='(Welch’s T-test)'
N_coanchored=np.hstack(([N_all[ii][0] for ii in range(len(N_all))]))
for coeff_ind, coeff_type in enumerate(['Circular_distance','Forward_distance','Spatial_corr']):
    means_type=[]
    sem_type=[]
    for stage_type,color_ in {'pre':'black','post':'grey'}.items():
        stage_type_bool=np.asarray([stage_type in stages[ii] for ii in range(len(stages))])
        coeffs_timebins_all_means=np.hstack(([np.nanmean(                                            coeffs_crossanchor_all[0][time_bin][stage_type_bool][:,coeff_ind])                                              for time_bin in np.arange(time_bins_used)]))
        coeffs_timebins_all_sems=np.hstack(([np.sqrt(np.sum(sems_crossanchor_all[0][time_bin]                                                            [stage_type_bool][:,coeff_ind]**2)/                                                len(sems_crossanchor_all[0][time_bin][stage_type_bool][:,coeff_ind])**2)                    for time_bin in np.arange(time_bins_used)]))


        plt.errorbar(x=np.arange(len(coeffs_timebins_all_means[:time_bins_used])),                     y=coeffs_timebins_all_means[:time_bins_used]*scaling_factor,                     yerr=coeffs_timebins_all_sems[:time_bins_used]*scaling_factor, marker='o',markersize=10,                     color=color_)
        plt.tick_params(axis='both',  labelsize=20)
        plt.tick_params(width=2, length=6)
        plt.axhline(0,ls='dashed',color='black',linewidth=4)
        #plt.ylim(-0.0042,0)
        
        #print(coeffs_timebins_all_means)
        #print(coeffs_timebins_all_sems)
        #print(np.nanmean(np.vstack((N_all[:num_timebins_reduced]))[:,stage_type_bool],axis=1))
        #print('')
        
        
        means_type.append(coeffs_timebins_all_means)
        sem_type.append(coeffs_timebins_all_sems)
    
    means_type=np.vstack((means_type))
    sem_type=np.vstack((sem_type))
    tstats=[]
    pvalues=[]
    for bin_ in np.arange(time_bins_used):
        #print('')
        #print(means_type[:,bin_])
        #print(sem_type[:,bin_])
        N_pre=np.nanmean(N_coanchored[:2])
        N_post=np.nanmean(N_coanchored[2:])
        #print(N_pre)
        #print(N_post)
        dof= (N_pre - 1) + (N_post - 1)
        
        bin_




        std_0=np.asarray(sem_type[0,bin_])*np.sqrt(N_pre)
        std_1=np.asarray(sem_type[1,bin_])*np.sqrt(N_post)


        tstat, pvalue = ttest_ind_from_stats(means_type[0,bin_], std_0, N_pre,                                             means_type[1,bin_], std_1, N_post,                                            equal_var=False)

        #print(tstat)
        #print(pvalue)
        #print(dof)
        
        tstats.append(tstat)
        pvalues.append(pvalue)
    
    
    corrected_pvalues=multipletests(pvalues)[1]
    for bin_ in np.arange(time_bins_used):
        pvalue_corrected=corrected_pvalues[bin_]
        print(str(bin_*10)+'-'+str((bin_+1)*10)+' minutes post-sleep: N='+str(int(N_pre))+              ' (pre-task) N='+str(int(N_post))+' (post-task): t='+str(round(tstats[bin_],2))+              ', P='+str(round(pvalue_corrected,3))+', df='+str(int(dof))+'.')


    plt.savefig(Ephys_output_folder_dropbox+coeff_type+'_beta_vs_timebins_Allsleep.svg',                bbox_inches = 'tight', pad_inches = 0) 
    plt.show()
    
   


# In[287]:


means_type


# In[ ]:





# In[ ]:


###stats


print('''
        ##Circular distance##
        
        p=0.7867
        t = 0.2706
        df = 2774
        Group
        Mean
        SD
        SEM
        N
        Group One
        -0.0029636100
        0.0555803136
        0.0014566000
        1456            
        Group Two
        -0.0034400700
        0.0332472341
        0.0009151000
        1320    
        
        p=0.7994
        t = 0.2541
        df = 2773
        Group
        Mean
        SD
        SEM
        N
        Group One
        -0.0018334400
        0.0555123932
        0.0014548200
        1456            
        Group Two
        -0.0022790800
        0.0328162542
        0.0009035800
        1319       

        p=0.7790
        t = 0.2806
        df = 2673
        Group
        Mean
        SD
        SEM
        N
        Group One
        -0.0023294600
        0.0542337331
        0.0014213100
        1456            
        Group Two
        -0.0029463200
        0.0593481714
        0.0016998300
        1219      

        

        ##Forward distance##

        p=0.3878
        t = 0.8637
        df = 2672

        Group
        Mean
        SD
        SEM
        N
        Group One
        -0.0015468800
        0.0544687837
        0.0014274700
        1456            
        Group Two
        0.0003577500
        0.0594480670
        0.0017033900
        1218  



        p= 0.7546
        t = 0.3126
        df = 2773

        Group
        Mean
        SD
        SEM
        N
        Group One
        -0.0007733600
        0.0557581279
        0.0014612600
        1456            
        Group Two
        -0.0002231300
        0.0328514827
        0.0009045500
        1319 


        p=0.7288
        t = 0.3467
        df = 2773

        Group
        Mean
        SD
        SEM
        N
        Group One
        0.0001586100
        0.0558034410
        0.0014629500
        1455            
        Group Two
        0.0007710900
        0.0332944654
        0.0009164000
        1320     


        
        
        ##Spatial corr##
        
        
        p=0.0317
        t = 2.1492
        df = 2774
        
        Group
        Mean
        SD
        SEM
        N
        Group One
        0.00003595328700
        0.05557497157183
        0.00145646000000
        1456                
        Group Two
        0.00381997000000
        0.03326648992467
        0.00091563000000
        1320 
        

        p=0.2642
        t = 1.1167
        df = 2773
        Group
        Mean
        SD
        SEM
        N
        Group One
        0.0011560009400
        0.0555200246738
        0.0014550200000
        1456               
        Group Two
        0.0031149700000
        0.0328362291387
        0.0009041300000
        1319  


        p=0.6898
        t = 0.3992
        df = 2535
        Group
        Mean
        SD
        SEM
        N
        Group One
        0.0024628281500
        0.0496108443434
        0.0014215200000
        1218               
        Group Two
        0.0015725600000
        0.0615332913049
        0.0016942900000
        1319   
        
        
        
        
        ''')


# In[ ]:





# In[61]:


###Per day analyses


# In[ ]:





# In[175]:


##Per day analysis - all stages
coeffs_used=['Circular_distance','Forward_distance','Spatial_corr']
means_perday=np.zeros((len(coeffs_used),num_timebins, len(day_type_dicX['combined_ABCDonly'])))
means_perday[:]=np.nan
for coeff_ind, coeff_type in enumerate(['Circular_distance','Forward_distance','Spatial_corr']):
    for time_bin in np.arange(num_timebins):
        means_=np.hstack(([np.nanmean([Regression_timebins_dic['coeffs'][time_bin][stage][mouse_recday][coeff_ind]                     if len(Regression_timebins_dic['coeffs'][time_bin][stage][mouse_recday])>0
                    else np.nan for stage in stages])\
         for mouse_recday in day_type_dicX['combined_ABCDonly']]))
        
        means_perday[coeff_ind,time_bin]=means_


# In[ ]:





# In[176]:


np.shape(coeffs_timebins_all_means)


# In[177]:


for coeff_ind, coeff_type in enumerate(['Circular_distance','Forward_distance','Spatial_corr']):
    coeffs_timebins_all_means=np.nanmean(means_perday[coeff_ind],axis=1)
    coeffs_timebins_all_sems=st.sem(means_perday[coeff_ind],axis=1,nan_policy='omit')


    plt.errorbar(x=np.arange(len(coeffs_timebins_all_means[:time_bins_used])),                 y=coeffs_timebins_all_means[:time_bins_used]*scaling_factor,                 yerr=coeffs_timebins_all_sems[:time_bins_used]*scaling_factor, marker='o',markersize=10,color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.axhline(0,ls='dashed',color='black',linewidth=4)
    #plt.ylim(-0.0042,0)

    #plt.savefig(Ephys_output_folder_dropbox+coeff_type+'_beta_vs_timebins_Allsleep.svg',\
    #            bbox_inches = 'tight', pad_inches = 0) 
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[179]:


##Per day analysis
coeffs_used=['Circular_distance','Forward_distance','Spatial_corr']
means_perday=np.zeros((len(coeffs_used),num_timebins, len(day_type_dicX['combined_ABCDonly'])))
means_perday[:]=np.nan
for coeff_ind, coeff_type in enumerate(['Circular_distance','Forward_distance','Spatial_corr']):
    for time_bin in np.arange(num_timebins):
        means_=np.hstack(([np.nanmean([np.nanmean(Regression_timebins_dic['coeffs'][time_bin][stage][mouse_recday])                     if len(Regression_timebins_dic['coeffs'][time_bin][stage][mouse_recday])>0
                    else np.nan for stage in stages[stage_type_bool]])\
         for mouse_recday in day_type_dicX['combined_ABCDonly']]))
        
        means_perday[coeff_ind,time_bin]=means_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


####END HERE: mainly duplications below


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[489]:


##Task space distance and spatial corr vs sleep corrs - for only pairs >180 apart
corr_crossanchor_task_all=[]
corr_crossanchor_space_all=[]
coeffs_crossanchor_all=[]
sems_crossanchor_all=[]

only_far=True
for circular_angle in [True]:#,False]:
    print('')
    print(circular_angle)
    corr_crossanchor_task=[]
    corr_crossanchor_space=[]
    coeffs_crossanchor=[]
    sems_crossanchor=[]
    N_all=[]
    
    binned_corrs_means_all=[]
    binned_corrs_sems_all=[]

    for phase_place_diff in np.arange(num_phase_place_diffs_):
        print(phase_place_diff)
        corr_phase_place_diff=[]
        corr_spatial_corr=[]

        coeffs_phase_place_diff=[]
        sems_phase_place_diff=[]
        N_place_phase_diff=[]
        
        Anchored_pairs_angles_all=[]
        Anchored_pairs_sleepcorrs_rebinned_all=[]

        ##['post0', 'post1', 'post2', 'post3', 'post4', 'post5']:#sleep_stages_unique: ['preX','preX2']:
        for stage in sleep_stages_unique:# ['preX','preX2']:
            print(stage)
            try:

                Anchored_pairs_sleepcorrs_rebinned_=[]
                Anchored_pairs_angles_=[]
                Anchored_pairs_spatialcorrs_=[]
                Anchored_pairs_phases_=[]
                for mouse_recday in day_type_dicX['combined_ABCDonly']:
                    if len(Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned']                           [stage][mouse_recday])>0:
                        Anchored_pairs_sleepcorrs_rebinned_.append(                            Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned']                                                                   [stage][mouse_recday])
                        Anchored_pairs_angles_.append(Sleepcorr_pairs_dic[phase_place_diff]                                                      ['Anchored_pairs_angles'][stage][mouse_recday])
                        Anchored_pairs_spatialcorrs_.append(Sleepcorr_pairs_dic[phase_place_diff]                                                            ['Anchored_pairs_Spatialcorrs'][stage][mouse_recday])

                        Anchored_pairs_phases_.append(Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_phases']                                                            [stage][mouse_recday])

                Anchored_pairs_sleepcorrs_rebinned_=np.hstack((Anchored_pairs_sleepcorrs_rebinned_))
                Anchored_pairs_angles_=np.hstack((Anchored_pairs_angles_))
                Anchored_pairs_spatialcorrs_=np.hstack((Anchored_pairs_spatialcorrs_))
                Anchored_pairs_phases_=np.hstack((Anchored_pairs_phases_))




                filter_boolean_nan=~np.isnan(Anchored_pairs_angles_)
                Anchored_pairs_angles_unfiltered=Anchored_pairs_angles_
                Anchored_pairs_angles_[filter_boolean_nan]=abs((Anchored_pairs_angles_[filter_boolean_nan]))
                
                if only_far==True:
                    filter_boolean_=np.logical_and(Anchored_pairs_spatialcorrs_<0.9,Anchored_pairs_angles_>=180)
                else:
                    filter_boolean_=Anchored_pairs_spatialcorrs_<0.9
                    
                if circular_angle==True:
                    Anchored_pairs_angles_[Anchored_pairs_angles_>=180]=                    360-Anchored_pairs_angles_[Anchored_pairs_angles_>=180]

                filter_boolean=np.logical_and(filter_boolean_==True,Anchored_pairs_angles_>0)
                #else:
                #    filter_boolean=np.logical_and(filter_boolean_==True,\
                #                                  np.logical_and(Anchored_pairs_angles_>0,Anchored_pairs_angles_<350))

                Anchored_pairs_angles=Anchored_pairs_angles_[filter_boolean]
                Anchored_pairs_spatialcorrs=Anchored_pairs_spatialcorrs_[filter_boolean]
                Anchored_pairs_phases=Anchored_pairs_phases_[filter_boolean]
                Anchored_pairs_sleepcorrs_rebinned=Anchored_pairs_sleepcorrs_rebinned_[filter_boolean]
                
                if phase_place_diff==0:
                    #sns.regplot(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned)
                    #plt.show()
                    print(st.pearsonr(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned))
                    
                    bins_angles_=(np.arange(4)+1)*45
                    bins_angles_[-1]=bins_angles_[-1]+1

                    Anchored_pairs_angles_bin=np.digitize(Anchored_pairs_angles,bins_angles_)

                    binned_corrs_means=np.asarray([np.nanmean(Anchored_pairs_sleepcorrs_rebinned[Anchored_pairs_angles_bin==bin_])                                                   for bin_ in np.arange(4)])

                    binned_corrs_sems=np.asarray([st.sem(Anchored_pairs_sleepcorrs_rebinned[Anchored_pairs_angles_bin==bin_])                                                   for bin_ in np.arange(4)])
                    plt.errorbar(x=np.arange(4),y=binned_corrs_means,yerr=binned_corrs_sems)
                    
                    binned_corrs_means_all.append(binned_corrs_means)
                    binned_corrs_sems_all.append(binned_corrs_sems)
                    plt.show()
                
                Anchored_pairs_angles_all.append(Anchored_pairs_angles)
                Anchored_pairs_sleepcorrs_rebinned_all.append(Anchored_pairs_sleepcorrs_rebinned)
                
                
                corr_phase_place_diff.append(st.pearsonr(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned)[0])

                #sns.regplot(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned)
                #plt.show()
                #print(st.pearsonr(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned))
                corr_spatial_corr.append(st.pearsonr(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned)[0])


                ###regression
                X=np.column_stack((st.zscore(Anchored_pairs_angles,nan_policy='omit'),                                   st.zscore(Anchored_pairs_spatialcorrs,nan_policy='omit'),                                   st.zscore(Anchored_pairs_phases,nan_policy='omit'),                                   np.repeat(1,len(Anchored_pairs_angles))))
                y=Anchored_pairs_sleepcorrs_rebinned

                y=y[~np.isnan(X[:,2])]
                X=X[~np.isnan(X[:,2])]


                #reg=LinearRegression().fit(X, y)
                #coeffs=reg.coef_
                #coeffs_phase_place_diff.append(coeffs)

                N=len(X)
                p=len(X.T)
                
                N_place_phase_diff.append(N)


                model=LinearRegression()
                reg=model.fit(X, y)
                coeffs=reg.coef_
                coeffs_phase_place_diff.append(coeffs)

                y_hat = model.predict(X)
                residuals = y - y_hat
                residual_sum_of_squares = residuals.T @ residuals
                sigma_squared_hat = residual_sum_of_squares / (N - p)

                var_beta_hat = np.linalg.inv(X.T @ X) * sigma_squared_hat

                std_errors=[]
                for indx in np.arange(p):
                    std_err=var_beta_hat[indx, indx] ** 0.5
                    std_errors.append(std_err)

                sems_phase_place_diff.append(std_errors)


            except Exception as e:
                print(e)

        coeffs_phase_place_diff=np.vstack((coeffs_phase_place_diff))
        sems_phase_place_diff=np.vstack((sems_phase_place_diff))


        corr_crossanchor_task.append(corr_phase_place_diff)
        corr_crossanchor_space.append(corr_spatial_corr)
        coeffs_crossanchor.append(coeffs_phase_place_diff)
        sems_crossanchor.append(sems_phase_place_diff)
        N_all.append(N_place_phase_diff)
        
        
    corr_crossanchor_task_all.append(corr_crossanchor_task)
    corr_crossanchor_space_all.append(corr_crossanchor_space)
    coeffs_crossanchor_all.append(coeffs_crossanchor)
    sems_crossanchor_all.append(sems_crossanchor)

N_all=np.asarray(N_all).T


# In[492]:


Anchored_pairs_angles_


# In[493]:


Anchored_pairs_angles


# In[452]:


binned_corrs_means_all


# In[ ]:





# In[451]:


print('All')
binned_corrs_means_all=np.vstack((binned_corrs_means_all))
binned_corrs_sems_all=np.vstack((binned_corrs_sems_all))

binned_corrs_means_mean=np.mean(binned_corrs_means_all,axis=0)
binned_corrs_sems_mean=np.sqrt(np.sum(binned_corrs_sems_all**2,axis=0)/len(binned_corrs_sems_all)**2)

plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.errorbar(x=np.arange(4),y=binned_corrs_means_mean,yerr=binned_corrs_sems_mean, marker='o',markersize=10,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'Pre_distance_vs_corr_Allsleep.svg',            bbox_inches = 'tight', pad_inches = 0) 
plt.show()


    
    

print('Pre')
binned_corrs_means_pre=binned_corrs_means_all[-2:]
binned_corrs_sems_pre=binned_corrs_sems_all[-2:]

binned_corrs_means_mean_pre=np.mean(binned_corrs_means_pre[:6],axis=0)
binned_corrs_sems_mean_pre=np.sqrt(np.sum(binned_corrs_sems_pre**2,axis=0)/len(binned_corrs_sems_pre)**2)


plt.errorbar(x=np.arange(4),y=binned_corrs_means_mean_pre,yerr=binned_corrs_sems_mean_pre, marker='o',             markersize=10,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_corr_Presleep.svg',            bbox_inches = 'tight', pad_inches = 0) 
plt.show()

print('Post')
binned_corrs_means_post=binned_corrs_means_all[:-2]
binned_corrs_sems_post=binned_corrs_sems_all[:-2]

binned_corrs_means_mean_post=np.mean(binned_corrs_means_post[:6],axis=0)
binned_corrs_sems_mean_post=np.sqrt(np.sum(binned_corrs_sems_post**2,axis=0)/len(binned_corrs_sems_post)**2)


plt.errorbar(x=np.arange(4),y=binned_corrs_means_mean_post,yerr=binned_corrs_sems_mean_post, marker='o',             markersize=10,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_corr_Postsleep.svg',            bbox_inches = 'tight', pad_inches = 0) 
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[433]:


corr_all_task=corr_crossanchor_task_all[0]
bar_plotX(corr_all_task,'none',-0.2,0.13,'points','paired',0.025)#
plt.show()
print(st.wilcoxon(corr_all_task[0],np.nanmean(corr_all_task[1:],axis=0)))


# In[434]:


corr_all_space=corr_crossanchor_space_all[0]
bar_plotX(corr_all_space,'none',0,0.35,'points','paired',0.025)
plt.show()
print(st.wilcoxon(corr_all_space[0],np.nanmean(corr_all_space[1:],axis=0)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[416]:


print('Task Space')

within_vs_between_means_all=[]
within_vs_between_sems_all=[]
for ind,angle_type in enumerate(['circular angle']): ##enumerate(['circular angle','forward angle'])
    print('______')
    print(angle_type)
    
    coeffs_crossanchor_=coeffs_crossanchor_all[ind]
    sems_crossanchor_=sems_crossanchor_all[ind]
    #coeffs_coanchored_task=np.asarray((coeffs_crossanchor))[0,:,0]
    #coeffs_crossanchor_task=np.vstack(([np.asarray(coeffs_crossanchor)[ii][:,0] for ii in\
    #                                    np.arange(len(coeffs_crossanchor)-1)+1]))
    coeffs_all_task=np.vstack(([np.asarray(coeffs_crossanchor_)[ii][:,0] for ii in                                        np.arange(len(coeffs_crossanchor_))]))

    #sems_coanchored_task=np.asarray((sems_crossanchor))[0,:,0]
    #sems_crossanchor_task=np.vstack(([np.asarray(sems_crossanchor)[ii][:,0]\
    #                                  for ii in np.arange(len(coeffs_crossanchor)-1)+1]))
    sems_all_task=np.vstack(([np.asarray(sems_crossanchor_)[ii][:,0]                                      for ii in np.arange(len(coeffs_crossanchor_))]))

    task_mean=np.mean(coeffs_all_task,axis=1)
    task_sem_mean=np.sqrt(np.sum(sems_all_task**2,axis=1)/len(sems_all_task.T)**2)

    plt.bar(np.arange(len(task_mean)),task_mean,yerr=task_sem_mean,color='black',ecolor='grey',capsize=3)
    #plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
    plt.axhline(0,color='black',ls='dashed')
    #plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_coactivity_regression_bar_preplay.svg',\
    #                bbox_inches = 'tight', pad_inches = 0) 
    plt.show()
    
    
    
    print('Within vs between')

    coeffs_crossanchor_task_all_mean=np.nanmean(coeffs_all_task[1:])
    sems_crossanchor_task_all=np.concatenate(sems_all_task[1:])
    sems_crossanchor_task_all_mean=np.sqrt(np.sum(sems_crossanchor_task_all**2)/len(sems_crossanchor_task_all)**2)

    within_vs_between_means=[task_mean[0],coeffs_crossanchor_task_all_mean]
    within_vs_between_sems=[task_sem_mean[0],sems_crossanchor_task_all_mean]
    
    within_vs_between_means_all.append(within_vs_between_means)
    within_vs_between_sems_all.append(within_vs_between_sems)


    plt.bar(np.arange(2),within_vs_between_means,yerr=within_vs_between_sems,color='black',ecolor='grey',capsize=3,           width=0.8)
    #plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
    plt.axhline(0,color='black',ls='dashed')
    plt.ylim(-0.006,0.001)
    plt.savefig(Ephys_output_folder_dropbox+angle_type+'_WithinvsBetween_coactivity_regression_bar.svg',                    bbox_inches = 'tight', pad_inches = 0) 
    plt.show()
    print(within_vs_between_means,np.asarray(within_vs_between_sems)*np.sqrt(N))
    print('')
    
    print('Per Session')
    ###Session-wise regressions

    for ii,name in enumerate(['preX','preX2']):
        print(name)
        plt.bar(np.arange(len(coeffs_all_task)),coeffs_all_task[:,ii],yerr=sems_all_task[:,ii],               color='black',ecolor='grey',capsize=3)
        #plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
        plt.axhline(0,color='black',ls='dashed')
        plt.savefig(Ephys_output_folder_dropbox+name+'_Pair_distance_vs_coactivity_regression_bar.svg',                    bbox_inches = 'tight', pad_inches = 0) 
        plt.show()
        
within_vs_between_means_all=np.vstack((within_vs_between_means_all))
within_vs_between_sems_all=np.vstack((within_vs_between_sems_all))

###sem averaging taken from ehre: 
##https://stats.stackexchange.com/questions/21104/calculate-average-of-a-set-numbers-with-reported-standard-errors


# In[417]:


###ttest for within vs between
###claculated using: https://www.graphpad.com/quickcalcs/ttest2/
'''
t=-2.4113
P=0.0170

'''
###validating this below:
(within_vs_between_means_all[0][0]-within_vs_between_means_all[0][1])/np.sqrt(np.sum(within_vs_between_sems_all[0]**2)/                                                                              (len(within_vs_between_sems_all[0])/2)**2)


# In[ ]:





# In[ ]:





# In[445]:


within_means_all=within_vs_between_means_all[:,0]
within_sems_all=within_vs_between_sems_all[:,0]
scaling_factor=1000

plt.rcParams["figure.figsize"] = (2,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.bar(np.arange(1),within_means_all[0]*scaling_factor,yerr=within_sems_all[0]*scaling_factor        ,color='black',ecolor='grey',capsize=3,width=0.8)
#plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
plt.axhline(0,color='black',ls='dashed')
plt.ylim(-0.005*scaling_factor,0.001*scaling_factor)
plt.xlim(-0.8,0.8)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'CircularvsForward_coactivity_regression_bar.svg',                bbox_inches = 'tight', pad_inches = 0) 
plt.show()
print('')
print(within_means_all)
print(within_sems_all)


# In[449]:


t_values_=within_means_all[0]/within_sems_all[0]
print(t_values_)
N_=np.nanmean(N_all[:,0])

print(N_)
##from calculator here: https://www.socscistatistics.com/pvalues/tdistribution.aspx

'''

Circular
t=-3.2559946658709396
DOF=N-1=677
two-tailed=.00119.

'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###Duplicated below - why??


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[210]:


###Angles and Sleep correlations for pairs across anchors
Sleepcorr_pairs_dic2=rec_dd()
shifts=np.linspace(-40,40,81) ## i.e. correlating with n bins in the future

num_phases=3
num_locations=9


num_phase_place_diffs_=5
useplace_phase_diff=True
use_tuned=False
use_anchored=True
use_nonspatial=False
use_spatial=False
use_anchor_angles=False
use_FRstable=True
bin_length=250 ##for rebinned correlations
thr_lower=30
thr_upper=360-thr_lower
sleep_dicX=Sleepcorr_matrix_shifted_dic

for phase_place_diff in np.arange(num_phase_place_diffs_):
    for mouse_recday in day_type_dicX['combined_ABCDonly']:
        print(mouse_recday)
        
        
        num_sessions=len(session_dic['awake'][mouse_recday])
        non_repeat_ses=non_repeat_ses_maker(mouse_recday)
        
        found_ses=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            try:
                Neuron_raw=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                found_ses.append(ses_ind)

            except:
                print('Files not found for session '+str(ses_ind))
                continue
        
        found_ses=np.asarray(found_ses)
        
        
        #angle_units=Xneuron_correlations['combined_ABCDonly']['angle_units'][mouse_recday]
        #max_bins=Xneuron_correlations['combined_ABCDonly']['Max_bins'][mouse_recday]#[0]
        angles=Xneuron_correlations['combined_ABCDonly']['Angles'][mouse_recday]
        
        Anchor_lags=Anchor_trial_dic['Best_shift_time'][mouse_recday]    
        Anchor_lags_mean=np.rad2deg(st.circmean(np.deg2rad(Anchor_lags),axis=1,nan_policy='omit'))
        #angles_anchor=np.vstack(([positive_angle([Anchor_lags_mean[ii]-Anchor_lags_mean[jj]\
        #                                   for ii in range(len(Anchor_lags_mean))])\
        # for jj in range(len(Anchor_lags_mean))]))
        
        angles_anchor=[np.vstack(([[Anchor_lags[ii,ses]-Anchor_lags[jj,ses]                                           for ii in range(len(Anchor_lags))]         for jj in range(len(Anchor_lags))])) for ses in np.arange(len(Anchor_lags.T))]


        Best_anchor_all=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday]
        Spatial_maps_all=Spatial_anchoring_dic['Phase_shifted_maps'][mouse_recday][:,:,0] ##0 as in 0 lag maps
        FR_bool=FR_sleep_boolean_dic[mouse_recday]
        FR_included=np.where(FR_bool==True)[0]

        awake_sesinds=Awake_sesind_dic[mouse_recday]
        sleep_stages=sleep_stage_dic[mouse_recday]
        sleep_stages_unique=np.unique(sleep_stages)

        first_day2_boolean=Awake_sesind_dic['first_day2_boolean'][mouse_recday]
        
        neurons_tuned=np.where(Tuned_dic['State_zmax_bool'][mouse_recday]==True)[0]
        Anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]
        Anchored_neurons=np.where(Anchored_bool==True)[0]
        
        Anchor_lags_used=Anchor_trial_dic['Best_shift_time'][mouse_recday]
        Anchor_lags_used_mean=np.rad2deg(st.circmean(np.deg2rad(Anchor_lags_used),axis=1,nan_policy='omit'))
        non_spatial_neurons=np.where(np.logical_and(Anchor_lags_used_mean>thr_lower,Anchor_lags_used_mean<thr_upper)                                     ==True)[0]
        spatial_neurons=np.where(np.logical_or(Anchor_lags_used_mean<thr_lower,Anchor_lags_used_mean>thr_upper)                                     ==True)[0]


        for stage in sleep_stages_unique:
            if stage=='preX':
                ses_ind=0
            elif 'post' in stage:
                ses_ind_unique=int(stage.split('post',1)[1])
                ses_ind=np.arange(num_sessions)[np.where(awake_sesinds==ses_ind_unique)[0][0]]
            elif stage=='preX2':
                ses_ind=np.where(first_day2_boolean==True)[0][0]
            
            ses_ind_ind_=np.where(found_ses==ses_ind)[0]
            if len(ses_ind_ind_)==0:
                continue
            ses_ind_ind=ses_ind_ind_[0]

            #phases=Xneuron_phaseangle_dic[ses_ind][mouse_recday] ##phase angle diff
            phases=Phase_diff_dic[mouse_recday][ses_ind] ##phase angle diff
            
            if len(phases)==0:
                continue

            
            Spatial_corrs_ses=np.corrcoef(Spatial_maps_all[:,ses_ind])
            
            if len(Spatial_maps_all)<2:
                print('less than 2 neurons')
                continue
            
            All_anchored_pairsX=[]
            All_anchored_pairsY=[]
            All_anchored_pairs_angles=[]
            All_anchored_pairs_phases=[]
            All_anchored_pairs_Spatialcorrs=[]


            for phase in np.arange(num_phases):
                if useplace_phase_diff==True:
                    phase_next=(phase+phase_place_diff)%3
                else:
                    phase_next=phase
                for location in np.arange(num_locations):
                    locations_next=np.where(mindistance_mat[location]==phase_place_diff)[0]

                    neurons_anchoredX=np.where(np.logical_and(Best_anchor_all[:,0]==phase,                                                              Best_anchor_all[:,1]==location,                                                             FR_bool==True))[0]

                    neurons_anchoredY=np.where(np.logical_and(Best_anchor_all[:,0]==phase_next                                                              ,np.isin(Best_anchor_all[:,1],locations_next),                                                             FR_bool==True))[0]
                    if use_tuned==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,neurons_tuned)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,neurons_tuned)
                    if use_anchored==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,Anchored_neurons)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,Anchored_neurons)
                    if use_nonspatial==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,non_spatial_neurons)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,non_spatial_neurons)
                    if use_spatial==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,spatial_neurons)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,spatial_neurons)
                        
                    if use_FRstable==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,FR_included)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,FR_included)



                    All_anchored_pairsX.append(neurons_anchoredX)
                    All_anchored_pairsY.append(neurons_anchoredY)
                    
                    if phase_place_diff>0:
                        if use_anchor_angles==True:
                            angle_mat_pairs_=angles_anchor[ses_ind_ind][neurons_anchoredX][:,neurons_anchoredY]
                        else:
                            angle_mat_pairs_=angles[:,:,ses_ind][neurons_anchoredX][:,neurons_anchoredY]
                        
                        phase_mat_pairs_=phases[neurons_anchoredX][:,neurons_anchoredY]
                        Spatial_corrs_pairs_=Spatial_corrs_ses[neurons_anchoredX][:,neurons_anchoredY]
                    elif phase_place_diff==0:
                        if use_anchor_angles==True:
                            angle_mat_pairs_=matrix_triangle(angles_anchor[ses_ind_ind][neurons_anchoredX]                                                             [:,neurons_anchoredY],'lower')
                        else:
                            angle_mat_pairs_=matrix_triangle(angles[:,:,ses_ind][neurons_anchoredX][:,neurons_anchoredY],                                                             'lower')
                            
                        phase_mat_pairs_=matrix_triangle(phases[neurons_anchoredX][:,neurons_anchoredY],                                                         'lower')
                        Spatial_corrs_pairs_=matrix_triangle(Spatial_corrs_ses[neurons_anchoredX][:,neurons_anchoredY],                                                         'lower')
                        
                        
                        
                    if len(angle_mat_pairs_)>0 and phase_place_diff>0:
                        angle_mat_pairs=np.hstack((angle_mat_pairs_))
                        phase_mat_pairs=np.hstack((phase_mat_pairs_))
                        Spatial_corrs_pairs_=np.hstack((Spatial_corrs_pairs_))
                    else:
                        angle_mat_pairs=angle_mat_pairs_
                        phase_mat_pairs=phase_mat_pairs_
                        Spatial_corrs_pairs_=Spatial_corrs_pairs_

                    All_anchored_pairs_angles.append(angle_mat_pairs)
                    All_anchored_pairs_phases.append(phase_mat_pairs)
                    All_anchored_pairs_Spatialcorrs.append(Spatial_corrs_pairs_)
            
            All_anchored_pairs_angles_=All_anchored_pairs_angles
            All_anchored_pairs_angles=concatenate_complex2(All_anchored_pairs_angles)
            All_anchored_pairs_phases=concatenate_complex2(All_anchored_pairs_phases)
            All_anchored_pairs_Spatialcorrs=concatenate_complex2(All_anchored_pairs_Spatialcorrs)

            All_anchored_pairs_sleepcorrs_allshifts=np.zeros((len(shifts),len(All_anchored_pairs_angles)))
            All_anchored_pairs_sleepcorrs_allshifts[:]=np.nan
            for shift_ind,shift in enumerate(shifts):
                shift=int(shift)
                All_anchored_pairs_sleepcorrs=[]
                for phase in np.arange(num_phases):
                    if useplace_phase_diff==True:
                        phase_next=(phase+phase_place_diff)%3
                    else:
                        phase_next=phase
                    for location in np.arange(num_locations):
                        locations_next=np.where(mindistance_mat[location]==phase_place_diff)[0]
                        neurons_anchoredX=np.where(np.logical_and(Best_anchor_all[:,0]==phase,                                                              Best_anchor_all[:,1]==location,                                                             FR_bool==True))[0]

                        neurons_anchoredY=np.where(np.logical_and(Best_anchor_all[:,0]==phase_next                                                                  ,np.isin(Best_anchor_all[:,1],locations_next),                                                                 FR_bool==True))[0]
                        if use_tuned==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,neurons_tuned)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,neurons_tuned)
                        if use_anchored==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,Anchored_neurons)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,Anchored_neurons)
                        if use_nonspatial==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,non_spatial_neurons)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,non_spatial_neurons)
                        if use_spatial==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,spatial_neurons)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,spatial_neurons)
                            
                        if use_FRstable==True:
                            neurons_anchoredX=np.intersect1d(neurons_anchoredX,FR_included)
                            neurons_anchoredY=np.intersect1d(neurons_anchoredY,FR_included)




                        sleepcorr_mat=sleep_dicX[stage][shift][mouse_recday]
                        
                        if phase_place_diff>0:
                            sleepcorr_mat_pairs_=sleepcorr_mat[neurons_anchoredX][:,neurons_anchoredY]
                        
                        elif phase_place_diff==0:
                            sleepcorr_mat_pairs_=matrix_triangle(sleepcorr_mat[neurons_anchoredX][:,neurons_anchoredY],                                                                 'lower')


                        if len(sleepcorr_mat_pairs_)>0 and phase_place_diff>0:
                            sleepcorr_mat_pairs=np.hstack((sleepcorr_mat_pairs_))
                        else:
                            sleepcorr_mat_pairs=sleepcorr_mat_pairs_

                        All_anchored_pairs_sleepcorrs.append(sleepcorr_mat_pairs)

                All_anchored_pairs_sleepcorrs_=All_anchored_pairs_sleepcorrs
                All_anchored_pairs_sleepcorrs=concatenate_complex2(All_anchored_pairs_sleepcorrs)
                All_anchored_pairs_sleepcorrs_allshifts[shift_ind]=All_anchored_pairs_sleepcorrs
                
            
            
            All_anchored_pairs_sleepcorrs=[]
            for phase in np.arange(num_phases):
                if useplace_phase_diff==True:
                    phase_next=(phase+phase_place_diff)%3
                else:
                    phase_next=phase
                for location in np.arange(num_locations):
                    locations_next=np.where(mindistance_mat[location]==phase_place_diff)[0]
                    neurons_anchoredX=np.where(np.logical_and(Best_anchor_all[:,0]==phase,                                                          Best_anchor_all[:,1]==location,                                                         FR_bool==True))[0]

                    neurons_anchoredY=np.where(np.logical_and(Best_anchor_all[:,0]==phase_next                                                              ,np.isin(Best_anchor_all[:,1],locations_next),                                                             FR_bool==True))[0]
                    if use_tuned==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,neurons_tuned)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,neurons_tuned)
                    if use_anchored==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,Anchored_neurons)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,Anchored_neurons)
                    if use_nonspatial==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,non_spatial_neurons)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,non_spatial_neurons)
                    if use_spatial==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,spatial_neurons)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,spatial_neurons)
                        
                    if use_FRstable==True:
                        neurons_anchoredX=np.intersect1d(neurons_anchoredX,FR_included)
                        neurons_anchoredY=np.intersect1d(neurons_anchoredY,FR_included)



                    sleepcorr_mat=Sleepcorr_matrix_rebinned_dic[str(bin_length)][stage][mouse_recday]
                    
                    if phase_place_diff>0:
                        sleepcorr_mat_pairs_=sleepcorr_mat[neurons_anchoredX][:,neurons_anchoredY]

                    elif phase_place_diff==0:
                        sleepcorr_mat_pairs_=matrix_triangle(sleepcorr_mat[neurons_anchoredX][:,neurons_anchoredY],                                                             'lower')

                    if len(sleepcorr_mat_pairs_)>0 and phase_place_diff>0:
                        sleepcorr_mat_pairs=np.hstack((sleepcorr_mat_pairs_))
                    else:
                        sleepcorr_mat_pairs=sleepcorr_mat_pairs_

                    All_anchored_pairs_sleepcorrs.append(sleepcorr_mat_pairs)

            All_anchored_pairs_sleepcorrs=concatenate_complex2(All_anchored_pairs_sleepcorrs)
            All_anchored_pairs_sleepcorrs_rebinned=All_anchored_pairs_sleepcorrs
                
                
            
            Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs'][mouse_recday]=All_anchored_pairsX,            All_anchored_pairsY
            Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_angles'][stage][mouse_recday]=All_anchored_pairs_angles
            Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_phases'][stage][mouse_recday]=All_anchored_pairs_phases
            All_anchored_pairs_phases
            Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_Spatialcorrs'][stage][mouse_recday]=            All_anchored_pairs_Spatialcorrs
            
            Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_sleepcorrs'][stage][mouse_recday]=            All_anchored_pairs_sleepcorrs_allshifts
            Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned'][stage][mouse_recday]=            All_anchored_pairs_sleepcorrs_rebinned


# In[222]:


Sleepcorr_pairs_dic2[0]['Anchored_pairs_sleepcorrs_rebinned'][stage][mouse_recday]


# In[219]:


Sleepcorr_pairs_dic[0]['Anchored_pairs_sleepcorrs_rebinned'][stage][mouse_recday]


# In[541]:


All_anchored_pairs_angles
angle_mat_pairs


# In[211]:


##Task space distance and spatial corr vs sleep corrs
corr_crossanchor_task_all=[]
corr_crossanchor_space_all=[]
coeffs_crossanchor_all=[]
sems_crossanchor_all=[]
only_far=False
for circular_angle in [True]:
    print('')
    print(circular_angle)
    corr_crossanchor_task=[]
    corr_crossanchor_space=[]
    coeffs_crossanchor=[]
    sems_crossanchor=[]
    
    N_all=[]
    for phase_place_diff in np.arange(num_phase_place_diffs_):
        print(phase_place_diff)
        corr_phase_place_diff=[]
        corr_spatial_corr=[]

        coeffs_phase_place_diff=[]
        sems_phase_place_diff=[]
        N_place_phase_diff=[]

        ##['post0', 'post1', 'post2', 'post3', 'post4', 'post5']:#sleep_stages_unique: ['preX','preX2']:
        for stage in ['preX','preX2']:#['post0', 'post1', 'post2', 'post3', 'post4', 'post5']:#sleep_stages_unique:#['post0', 'post1', 'post2', 'post3', 'post4', 'post5']:
            print(stage)
            try:

                Anchored_pairs_sleepcorrs_rebinned_=[]
                Anchored_pairs_angles_=[]
                Anchored_pairs_spatialcorrs_=[]
                Anchored_pairs_phases_=[]
                for mouse_recday in day_type_dicX['combined_ABCDonly']:
                    if len(Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned']                           [stage][mouse_recday])>0:
                        Anchored_pairs_sleepcorrs_rebinned_.append(                            Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned']                                                                   [stage][mouse_recday])
                        Anchored_pairs_angles_.append(Sleepcorr_pairs_dic2[phase_place_diff]                                                      ['Anchored_pairs_angles'][stage][mouse_recday])
                        Anchored_pairs_spatialcorrs_.append(Sleepcorr_pairs_dic2[phase_place_diff]                                                            ['Anchored_pairs_Spatialcorrs'][stage][mouse_recday])

                        Anchored_pairs_phases_.append(Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_phases']                                                            [stage][mouse_recday])

                Anchored_pairs_sleepcorrs_rebinned_=np.hstack((Anchored_pairs_sleepcorrs_rebinned_))
                Anchored_pairs_angles_=np.hstack((Anchored_pairs_angles_))
                Anchored_pairs_spatialcorrs_=np.hstack((Anchored_pairs_spatialcorrs_))
                Anchored_pairs_phases_=np.hstack((Anchored_pairs_phases_))




                filter_boolean_nan=~np.isnan(Anchored_pairs_angles_)
                Anchored_pairs_angles_[filter_boolean_nan]=abs((Anchored_pairs_angles_[filter_boolean_nan]))
                
                if only_far==True:
                    filter_boolean_=np.logical_and(Anchored_pairs_spatialcorrs_<0.9,Anchored_pairs_angles_>=180)
                else:
                    filter_boolean_=Anchored_pairs_spatialcorrs_<0.9
                    
                if circular_angle==True:
                    Anchored_pairs_angles_[Anchored_pairs_angles_>=180]=                    360-Anchored_pairs_angles_[Anchored_pairs_angles_>=180]

                filter_boolean=np.logical_and(filter_boolean_==True,Anchored_pairs_angles_>0)
                #else:
                #    filter_boolean=np.logical_and(filter_boolean_==True,\
                #                                  np.logical_and(Anchored_pairs_angles_>0,Anchored_pairs_angles_<350))

                Anchored_pairs_angles=Anchored_pairs_angles_[filter_boolean]
                Anchored_pairs_spatialcorrs=Anchored_pairs_spatialcorrs_[filter_boolean]
                Anchored_pairs_phases=Anchored_pairs_phases_[filter_boolean]
                Anchored_pairs_sleepcorrs_rebinned=Anchored_pairs_sleepcorrs_rebinned_[filter_boolean]
                
                if phase_place_diff==0:
                    #sns.regplot(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned)
                    #plt.show()
                    print(st.pearsonr(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned))

                
                
                corr_phase_place_diff.append(st.pearsonr(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned)[0])

                #sns.regplot(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned)
                #plt.show()
                #print(st.pearsonr(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned))
                corr_spatial_corr.append(st.pearsonr(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned)[0])


                ###regression
                X=np.column_stack((st.zscore(Anchored_pairs_angles,nan_policy='omit'),                                   st.zscore(Anchored_pairs_spatialcorrs,nan_policy='omit'),                                   st.zscore(Anchored_pairs_phases,nan_policy='omit'),                                   np.repeat(1,len(Anchored_pairs_angles))))
                y=Anchored_pairs_sleepcorrs_rebinned

                y=y[~np.isnan(X[:,2])]
                X=X[~np.isnan(X[:,2])]


                #reg=LinearRegression().fit(X, y)
                #coeffs=reg.coef_
                #coeffs_phase_place_diff.append(coeffs)

                N=len(X)
                p=len(X.T)
                
                N_place_phase_diff.append(N)


                model=LinearRegression()
                reg=model.fit(X, y)
                coeffs=reg.coef_
                coeffs_phase_place_diff.append(coeffs)

                y_hat = model.predict(X)
                residuals = y - y_hat
                residual_sum_of_squares = residuals.T @ residuals
                sigma_squared_hat = residual_sum_of_squares / (N - p)

                var_beta_hat = np.linalg.inv(X.T @ X) * sigma_squared_hat

                std_errors=[]
                for indx in np.arange(p):
                    std_err=var_beta_hat[indx, indx] ** 0.5
                    std_errors.append(std_err)

                sems_phase_place_diff.append(std_errors)


            except Exception as e:
                print(e)

        coeffs_phase_place_diff=np.vstack((coeffs_phase_place_diff))
        sems_phase_place_diff=np.vstack((sems_phase_place_diff))


        corr_crossanchor_task.append(corr_phase_place_diff)
        corr_crossanchor_space.append(corr_spatial_corr)
        coeffs_crossanchor.append(coeffs_phase_place_diff)
        sems_crossanchor.append(sems_phase_place_diff)
        N_all.append(N_place_phase_diff)

        
        
    corr_crossanchor_task_all.append(corr_crossanchor_task)
    corr_crossanchor_space_all.append(corr_crossanchor_space)
    coeffs_crossanchor_all.append(coeffs_crossanchor)
    sems_crossanchor_all.append(sems_crossanchor)

N_all=np.asarray(N_all).T


# In[ ]:





# In[ ]:





# In[212]:


corr_all_task=corr_crossanchor_task_all[0]
bar_plotX(corr_all_task,'none',-0.2,0.13,'points','paired',0.025)#
plt.show()
print(st.wilcoxon(corr_all_task[0],np.nanmean(corr_all_task[1:],axis=0)))


# In[213]:


corr_all_space=corr_crossanchor_space_all[0]
bar_plotX(corr_all_space,'none',0,0.35,'points','paired',0.025)
plt.show()
print(st.wilcoxon(corr_all_space[0],np.nanmean(corr_all_space[1:],axis=0)))


# In[ ]:





# In[214]:


print('Task Space')

within_vs_between_means_all=[]
within_vs_between_sems_all=[]
ind=0
#for ind,angle_type in enumerate(['circular angle','forward angle']):


coeffs_crossanchor_=coeffs_crossanchor_all[ind]
sems_crossanchor_=sems_crossanchor_all[ind]
#coeffs_coanchored_task=np.asarray((coeffs_crossanchor))[0,:,0]
#coeffs_crossanchor_task=np.vstack(([np.asarray(coeffs_crossanchor)[ii][:,0] for ii in\
#                                    np.arange(len(coeffs_crossanchor)-1)+1]))
coeffs_all_task=np.vstack(([np.asarray(coeffs_crossanchor_)[ii][:,0] for ii in                                    np.arange(len(coeffs_crossanchor_))]))

#sems_coanchored_task=np.asarray((sems_crossanchor))[0,:,0]
#sems_crossanchor_task=np.vstack(([np.asarray(sems_crossanchor)[ii][:,0]\
#                                  for ii in np.arange(len(coeffs_crossanchor)-1)+1]))
sems_all_task=np.vstack(([np.asarray(sems_crossanchor_)[ii][:,0]                                  for ii in np.arange(len(coeffs_crossanchor_))]))

task_mean=np.mean(coeffs_all_task,axis=1)
task_sem_mean=np.sqrt(np.sum(sems_all_task**2,axis=1)/len(sems_all_task.T)**2)

plt.bar(np.arange(len(task_mean)),task_mean,yerr=task_sem_mean,color='black',ecolor='grey',capsize=3)
#plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
plt.axhline(0,color='black',ls='dashed')
#plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_coactivity_regression_bar_preplay.svg',\
#                bbox_inches = 'tight', pad_inches = 0) 
plt.show()


print('Within vs between')

coeffs_crossanchor_task_all_mean=np.nanmean(coeffs_all_task[1:])
sems_crossanchor_task_all=np.concatenate(sems_all_task[1:])
sems_crossanchor_task_all_mean=np.sqrt(np.sum(sems_crossanchor_task_all**2)/len(sems_crossanchor_task_all)**2)

within_vs_between_means=[task_mean[0],coeffs_crossanchor_task_all_mean]
within_vs_between_sems=[task_sem_mean[0],sems_crossanchor_task_all_mean]

within_vs_between_means_all.append(within_vs_between_means)
within_vs_between_sems_all.append(within_vs_between_sems)


plt.bar(np.arange(2),within_vs_between_means,yerr=within_vs_between_sems,color='black',ecolor='grey',capsize=3,       width=0.8)
#plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
plt.axhline(0,color='black',ls='dashed')
plt.ylim(-0.012,0.0025)
plt.savefig(Ephys_output_folder_dropbox+'WithinvsBetween_coactivity_regression_bar.svg',                bbox_inches = 'tight', pad_inches = 0) 
plt.show()
print('')
print(within_vs_between_means,np.asarray(within_vs_between_sems)*np.sqrt(N))

print('Per Session')
###Session-wise regressions

for ii,name in enumerate(['preX','preX2']):
    print(name)
    plt.bar(np.arange(len(coeffs_all_task)),coeffs_all_task[:,ii],yerr=sems_all_task[:,ii],           color='black',ecolor='grey',capsize=3)
    #plt.errorbar(x=np.arange(len(task_mean)),y=task_mean,yerr=task_sem_mean,barsabove=True)
    plt.axhline(0,color='black',ls='dashed')
    plt.savefig(Ephys_output_folder_dropbox+name+'_Pair_distance_vs_coactivity_regression_bar.svg',                bbox_inches = 'tight', pad_inches = 0) 
    plt.show()

within_vs_between_means_all=np.vstack((within_vs_between_means_all))
within_vs_between_sems_all=np.vstack((within_vs_between_sems_all))

###sem averaging taken from ehre: 
##https://stats.stackexchange.com/questions/21104/calculate-average-of-a-set-numbers-with-reported-standard-errors


# In[215]:


###ttest for within vs between
###claculated using: https://www.graphpad.com/quickcalcs/ttest2/
'''
t=-1.4767
P=0.1403

'''
###validating this below:
(within_vs_between_means[0]-within_vs_between_means[1])/np.sqrt(np.sum(np.asarray(within_vs_between_sems)**2)/                                                                              (len(within_vs_between_sems)/2)**2)


# In[204]:


N_mean=np.nanmean(N_all,axis=0)
print(int(N_mean[0]))
int(np.sum(N_mean[1:]))


# In[ ]:





# In[227]:


##Task space distance and spatial corr vs sleep corrs
corr_crossanchor_task_all=[]
corr_crossanchor_space_all=[]
coeffs_crossanchor_all=[]
sems_crossanchor_all=[]

only_far=True
for circular_angle in [True]:#,False]:
    print('')
    print(circular_angle)
    corr_crossanchor_task=[]
    corr_crossanchor_space=[]
    coeffs_crossanchor=[]
    sems_crossanchor=[]
    N_all=[]
    
    binned_corrs_means_all=[]
    binned_corrs_sems_all=[]

    for phase_place_diff in np.arange(num_phase_place_diffs_):
        print(phase_place_diff)
        corr_phase_place_diff=[]
        corr_spatial_corr=[]

        coeffs_phase_place_diff=[]
        sems_phase_place_diff=[]
        N_place_phase_diff=[]
        
        Anchored_pairs_angles_all=[]
        Anchored_pairs_sleepcorrs_rebinned_all=[]

        ##['post0', 'post1', 'post2', 'post3', 'post4', 'post5']:#sleep_stages_unique: ['preX','preX2']:
        for stage in sleep_stages_unique:# ['preX','preX2']:
            print(stage)
            try:

                Anchored_pairs_sleepcorrs_rebinned_=[]
                Anchored_pairs_angles_=[]
                Anchored_pairs_spatialcorrs_=[]
                Anchored_pairs_phases_=[]
                for mouse_recday in day_type_dicX['combined_ABCDonly']:
                    if len(Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned']                           [stage][mouse_recday])>0:
                        Anchored_pairs_sleepcorrs_rebinned_.append(                            Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_sleepcorrs_rebinned']                                                                   [stage][mouse_recday])
                        Anchored_pairs_angles_.append(Sleepcorr_pairs_dic2[phase_place_diff]                                                      ['Anchored_pairs_angles'][stage][mouse_recday])
                        Anchored_pairs_spatialcorrs_.append(Sleepcorr_pairs_dic2[phase_place_diff]                                                            ['Anchored_pairs_Spatialcorrs'][stage][mouse_recday])

                        Anchored_pairs_phases_.append(Sleepcorr_pairs_dic2[phase_place_diff]['Anchored_pairs_phases']                                                            [stage][mouse_recday])

                Anchored_pairs_sleepcorrs_rebinned_=np.hstack((Anchored_pairs_sleepcorrs_rebinned_))
                Anchored_pairs_angles_=np.hstack((Anchored_pairs_angles_))
                Anchored_pairs_spatialcorrs_=np.hstack((Anchored_pairs_spatialcorrs_))
                Anchored_pairs_phases_=np.hstack((Anchored_pairs_phases_))




                filter_boolean_nan=~np.isnan(Anchored_pairs_angles_)
                Anchored_pairs_angles_unfiltered=Anchored_pairs_angles_
                Anchored_pairs_angles_[filter_boolean_nan]=abs((Anchored_pairs_angles_[filter_boolean_nan]))
                
                if only_far==True:
                    filter_boolean_=np.logical_and(Anchored_pairs_spatialcorrs_<0.9,Anchored_pairs_angles_>=180)
                else:
                    filter_boolean_=Anchored_pairs_spatialcorrs_<0.9
                    
                if circular_angle==True:
                    Anchored_pairs_angles_[Anchored_pairs_angles_>=180]=                    360-Anchored_pairs_angles_[Anchored_pairs_angles_>=180]

                filter_boolean=np.logical_and(filter_boolean_==True,Anchored_pairs_angles_>0)
                #else:
                #    filter_boolean=np.logical_and(filter_boolean_==True,\
                #                                  np.logical_and(Anchored_pairs_angles_>0,Anchored_pairs_angles_<350))

                Anchored_pairs_angles=Anchored_pairs_angles_[filter_boolean]
                Anchored_pairs_spatialcorrs=Anchored_pairs_spatialcorrs_[filter_boolean]
                Anchored_pairs_phases=Anchored_pairs_phases_[filter_boolean]
                Anchored_pairs_sleepcorrs_rebinned=Anchored_pairs_sleepcorrs_rebinned_[filter_boolean]
                
                if phase_place_diff==0:
                    #sns.regplot(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned)
                    #plt.show()
                    print(st.pearsonr(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned))
                    
                    bins_angles_=(np.arange(4)+1)*45
                    bins_angles_[-1]=bins_angles_[-1]+1

                    Anchored_pairs_angles_bin=np.digitize(Anchored_pairs_angles,bins_angles_)

                    binned_corrs_means=np.asarray([np.nanmean(Anchored_pairs_sleepcorrs_rebinned[Anchored_pairs_angles_bin==bin_])                                                   for bin_ in np.arange(4)])

                    binned_corrs_sems=np.asarray([st.sem(Anchored_pairs_sleepcorrs_rebinned[Anchored_pairs_angles_bin==bin_])                                                   for bin_ in np.arange(4)])
                    plt.errorbar(x=np.arange(4),y=binned_corrs_means,yerr=binned_corrs_sems)
                    
                    binned_corrs_means_all.append(binned_corrs_means)
                    binned_corrs_sems_all.append(binned_corrs_sems)
                    plt.show()
                
                Anchored_pairs_angles_all.append(Anchored_pairs_angles)
                Anchored_pairs_sleepcorrs_rebinned_all.append(Anchored_pairs_sleepcorrs_rebinned)
                
                
                corr_phase_place_diff.append(st.pearsonr(Anchored_pairs_angles,Anchored_pairs_sleepcorrs_rebinned)[0])

                #sns.regplot(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned)
                #plt.show()
                #print(st.pearsonr(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned))
                corr_spatial_corr.append(st.pearsonr(Anchored_pairs_spatialcorrs,Anchored_pairs_sleepcorrs_rebinned)[0])


                ###regression
                X=np.column_stack((st.zscore(Anchored_pairs_angles,nan_policy='omit'),                                   st.zscore(Anchored_pairs_spatialcorrs,nan_policy='omit'),                                   st.zscore(Anchored_pairs_phases,nan_policy='omit'),                                   np.repeat(1,len(Anchored_pairs_angles))))
                y=Anchored_pairs_sleepcorrs_rebinned

                y=y[~np.isnan(X[:,2])]
                X=X[~np.isnan(X[:,2])]


                #reg=LinearRegression().fit(X, y)
                #coeffs=reg.coef_
                #coeffs_phase_place_diff.append(coeffs)

                N=len(X)
                p=len(X.T)
                
                N_place_phase_diff.append(N)


                model=LinearRegression()
                reg=model.fit(X, y)
                coeffs=reg.coef_
                coeffs_phase_place_diff.append(coeffs)

                y_hat = model.predict(X)
                residuals = y - y_hat
                residual_sum_of_squares = residuals.T @ residuals
                sigma_squared_hat = residual_sum_of_squares / (N - p)

                var_beta_hat = np.linalg.inv(X.T @ X) * sigma_squared_hat

                std_errors=[]
                for indx in np.arange(p):
                    std_err=var_beta_hat[indx, indx] ** 0.5
                    std_errors.append(std_err)

                sems_phase_place_diff.append(std_errors)


            except Exception as e:
                print(e)

        coeffs_phase_place_diff=np.vstack((coeffs_phase_place_diff))
        sems_phase_place_diff=np.vstack((sems_phase_place_diff))


        corr_crossanchor_task.append(corr_phase_place_diff)
        corr_crossanchor_space.append(corr_spatial_corr)
        coeffs_crossanchor.append(coeffs_phase_place_diff)
        sems_crossanchor.append(sems_phase_place_diff)
        N_all.append(N_place_phase_diff)
        
        
    corr_crossanchor_task_all.append(corr_crossanchor_task)
    corr_crossanchor_space_all.append(corr_crossanchor_space)
    coeffs_crossanchor_all.append(coeffs_crossanchor)
    sems_crossanchor_all.append(sems_crossanchor)

N_all=np.asarray(N_all).T


# In[228]:


print('All')
binned_corrs_means_all=np.vstack((binned_corrs_means_all))
binned_corrs_sems_all=np.vstack((binned_corrs_sems_all))

binned_corrs_means_mean=np.mean(binned_corrs_means_all,axis=0)
binned_corrs_sems_mean=np.sqrt(np.sum(binned_corrs_sems_all**2,axis=0)/len(binned_corrs_sems_all)**2)


plt.errorbar(x=np.arange(4),y=binned_corrs_means_mean,yerr=binned_corrs_sems_mean)
plt.savefig(Ephys_output_folder_dropbox+'Pre_distance_vs_corr_Allsleep.svg',            bbox_inches = 'tight', pad_inches = 0) 
plt.show()

print('Pre')
binned_corrs_means_pre=binned_corrs_means_all[-2:]
binned_corrs_sems_pre=binned_corrs_sems_all[-2:]

binned_corrs_means_mean_pre=np.mean(binned_corrs_means_pre[:6],axis=0)
binned_corrs_sems_mean_pre=np.sqrt(np.sum(binned_corrs_sems_pre**2,axis=0)/len(binned_corrs_sems_pre)**2)


plt.errorbar(x=np.arange(4),y=binned_corrs_means_mean_pre,yerr=binned_corrs_sems_mean_pre)
plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_corr_Presleep.svg',            bbox_inches = 'tight', pad_inches = 0) 
plt.show()

print('Post')
binned_corrs_means_post=binned_corrs_means_all[:-2]
binned_corrs_sems_post=binned_corrs_sems_all[:-2]

binned_corrs_means_mean_post=np.mean(binned_corrs_means_post[:6],axis=0)
binned_corrs_sems_mean_post=np.sqrt(np.sum(binned_corrs_sems_post**2,axis=0)/len(binned_corrs_sems_post)**2)


plt.errorbar(x=np.arange(4),y=binned_corrs_means_mean_post,yerr=binned_corrs_sems_mean_post)
plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_corr_Postsleep.svg',            bbox_inches = 'tight', pad_inches = 0) 
plt.show()


# In[ ]:





# In[198]:


'''
sems per session
averaging sems
ttest from sems
    -bar within vs between
    -ANOVA? for phase_place distance 

'''


# In[ ]:





# In[199]:


'''
cross-correlograms:

if circle - neurons in last quartile vs first quartile should have similar cross-correlation to first vs 2nd,\
2nd vs 3rd and 3rd vs 4th
-1st vs 3rd and 2nd vs 4th should be lowest

-no assumptions about directionality (but test)


'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
1-Identify co-anchored pairs of neurons for each condition (e.g. quadrant 4 and quadrant 1)
2-do cross-correlation (need spike times)


'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


########Replay analysis########


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


use_phase=True
minus_flipped=True
forward_distance=True
remove_zerotuned=True
use_coactivity=False


feature='state'
color='black'
num_states=4

centre_point=int((len(shifts)-1)/2)
thr_lower=10
thr_upper=360-thr_lower

num_phase_place_diffs=5
num_stages=6

beta_max_min_all=np.zeros((num_phase_place_diffs,len(day_type_dicX['combined_ABCDonly']),2))
beta_max_mindiff_all=np.zeros((num_phase_place_diffs,len(day_type_dicX['combined_ABCDonly'])))
betas_mean_all=np.zeros((num_phase_place_diffs,len(day_type_dicX['combined_ABCDonly']),len(shifts)))
coactivity_mean_all=np.zeros((num_phase_place_diffs,len(day_type_dicX['combined_ABCDonly']),len(shifts)))

beta_max_min_all[:]=np.nan
beta_max_mindiff_all[:]=np.nan
betas_mean_all[:]=np.nan
coactivity_mean_all[:]=np.nan

coactivity_mean_dic=rec_dd()

tuningangle_all=[]
corrs_all=[]
phaseangle_all=[]

tuningangle_all_Xanchor=[]
phaseangle_all_Xanchor=[]
corrs_all_Xanchor=[]

tuningangle_all_dic=rec_dd()
phaseangle_all_dic=rec_dd()
corrs_all_dic=rec_dd()

for phase_place_diff in np.arange(num_phase_place_diffs):
    exec('tuningangle_all'+str(phase_place_diff)+'=[]')
    exec('phaseangle_all'+str(phase_place_diff)+'=[]')
    exec('corrs_all'+str(phase_place_diff)+'=[]')

for mouse_recday_ind, mouse_recday in enumerate(day_type_dicX['combined_ABCDonly']):
    print(mouse_recday)
    try:
        for phase_place_diff in np.arange(num_phase_place_diffs):
            betas_all=[]

            coactivity_all=[]
            #for stage in ['preX','preX2']:
            #for stage_ind in np.arange(num_stages):
            for stage in stages:
                #stage='post'+str(stage_ind)

                #try:
                if phase_place_diff==0:
                    tuningangle_=Sleepcorr_pairs_dic['Anchored_pairs_angles'][stage][mouse_recday]
                    phaseangle_=Sleepcorr_pairs_dic['Anchored_pairs_phases'][stage][mouse_recday]
                    #phaseangle_=state_to_phase(tuningangle_,num_states)
                    corrs_=Sleepcorr_pairs_dic['Anchored_pairs_sleepcorrs'][stage][mouse_recday]

                else:
                    tuningangle_=Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_angles'][stage][mouse_recday]
                    phaseangle_=Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_phases'][stage][mouse_recday]
                    corrs_=Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs'][stage][mouse_recday]

                if len(tuningangle_)==0:
                    continue

                if forward_distance==False:
                    tuningangle_[tuningangle_>180]=360-tuningangle_[tuningangle_>180]
                    phaseangle_[phaseangle_>180]=360-phaseangle_[phaseangle_>180]
                    ##i.e. taking distance either side


                if phase_place_diff==0:
                    tuningangle_all.append(tuningangle_)
                    phaseangle_all.append(phaseangle_)
                    corrs_all.append(corrs_)
                else:
                    tuningangle_all_Xanchor.append(tuningangle_)
                    phaseangle_all_Xanchor.append(phaseangle_)
                    corrs_all_Xanchor.append(corrs_)

                exec('tuningangle_all'+str(phase_place_diff)+'.append(tuningangle_)')
                exec('phaseangle_all'+str(phase_place_diff)+'.append(phaseangle_)')
                exec('corrs_all'+str(phase_place_diff)+'.append(corrs_)')



                tuningangle_all_dic[phase_place_diff][stage][mouse_recday]=tuningangle_
                phaseangle_all_dic[phase_place_diff][stage][mouse_recday]=phaseangle_
                corrs_all_dic[phase_place_diff][stage][mouse_recday]=corrs_

                phase_nonans=~np.isnan(phaseangle_)

                if remove_zerotuned==True:
                    not_zero_boolean=np.logical_and(tuningangle_>thr_lower,tuningangle_<thr_upper)
                else:
                    not_zero_boolean=np.repeat(True,len(tuningangle_))



                if use_phase==True:
                    used_boolean=np.logical_and(not_zero_boolean,phase_nonans)
                    matx=np.vstack((np.repeat(1,len(tuningangle_)),phaseangle_,tuningangle_)).T[used_boolean] #
                    demeaned_1=matx[:,1]-np.mean(matx[:,1])
                    demeaned_2=matx[:,2]-np.mean(matx[:,2])
                    demeaned_matx=np.column_stack((matx[:,0],demeaned_1,demeaned_2))

                    phase_ind=1
                    state_ind=2
                else:
                    used_boolean=not_zero_boolean
                    matx=np.vstack((np.repeat(1,len(tuningangle_)),tuningangle_)).T[used_boolean]
                    demeaned_1=matx[:,1]-np.mean(matx[:,1])
                    demeaned_matx=np.column_stack((matx[:,0],demeaned_1))
                    state_ind=1


                y=np.asarray(corrs_).T[used_boolean]
                coactivity_all.append(np.asarray(corrs_).T)

                #demeaned_matx_=np.asarray([[np.repeat(demeaned_matx[ii][jj], len(y.T))\
                #                           for jj in range(len(demeaned_matx[ii]))] for ii in range(len(demeaned_matx))])

                y_minus_flipped=np.asarray([y[ii]-np.flip(y[ii]) for ii in range(len(y))])
                y_mean=np.asarray([(y[ii]+np.flip(y[ii]))/2 for ii in range(len(y))])

                X_coactivity=np.asarray([np.vstack(([np.repeat(demeaned_matx[ii][jj], len(y.T))                                                     for jj in range(len(demeaned_matx[ii]))],y_mean[ii]-np.mean(y_mean)))                                         for ii in range(len(demeaned_matx))])

                if len(demeaned_matx)<=1:
                    continue
                if use_coactivity==False:
                    X_used=demeaned_matx

                    if minus_flipped==True:
                        y_used=y_minus_flipped
                    else:
                        y_used=y


                    model = LinearRegression()
                    model.fit(X=X_used, y=y_used)

                    N=len(X_used)
                    p=len(X_used.T)

                    beta_hat = np.linalg.inv(X_used.T @ X_used) @ X_used.T @ y_used ##ordinary least squares
                    #print(beta_hat)

                    #print(model.coef_[:,1])

                    y_hat = model.predict(X_used)
                    residuals = y_used - y_hat
                    residual_sum_of_squares = residuals.T @ residuals
                    sigma_squared_hat = np.diagonal(residual_sum_of_squares) / (N - p)

                    var_beta_hat = np.asarray([np.linalg.inv(X_used.T @ X_used) * sigma_squared_hat[ii]                                               for ii in range(len(sigma_squared_hat))])


                else:
                    X_used=X_coactivity
                    y_used=y

                    beta_hat=np.zeros((np.shape(X_used)[1],len(y_used.T)))
                    for ii in range(len(y_used.T)):

                        X_used_lag=X_used[:,:,ii]
                        y_used_lag=y_used[:,ii]

                        model = LinearRegression()
                        model.fit(X=X_used_lag, y=y_used_lag)
                        beta_hat[:,ii]=model.coef_


                    #std_err=var_beta_hat[:,indx, indx] ** 0.5
                    #plt.errorbar((np.arange(len(y_minus_flipped.T))-centre_point)*25,beta_hat[indx],yerr=std_err,color=color)
                    #plt.axhline(0,color='black')
                    #plt.axvline(0,color='black')
                    #Replay_pairs_perday_dic[coherence_status]['betas'][sleep_stage][mouse_recday]=beta_hat[indx]
                    #Replay_pairs_perday_dic[coherence_status]['betas_sems'][sleep_stage][mouse_recday]=std_err

                if feature=='state':
                    indx=state_ind
                elif feature=='phase':
                    indx=phase_ind
                betas_all.append(beta_hat[indx])
                #except Exception as e:
                #    print(e)

            if len(betas_all)==0:
                continue
            betas_all=np.vstack((betas_all))
            betas_mean=np.mean(betas_all,axis=0)
            betas_sem=st.sem(betas_all,axis=0)
            plt.errorbar((np.arange(len(y_minus_flipped.T))-centre_point)*25,betas_mean,yerr=betas_sem,color=color)
            plt.axhline(0,color='black')
            plt.axvline(0,color='black')
            plt.show()

            beta_oneside=betas_mean[:centre_point]
            beta_max=np.max(beta_oneside)
            beta_min=np.min(beta_oneside)
            beta_max_min_all[phase_place_diff,mouse_recday_ind]=[beta_max,beta_min]
            beta_max_mindiff_all[phase_place_diff,mouse_recday_ind]=beta_max-beta_min

            betas_mean_all[phase_place_diff,mouse_recday_ind]=betas_mean

            coactivity_all=np.asarray((coactivity_all))

            coactivity_mean=np.mean(coactivity_all,axis=0)
            coactivity_mean_dic[phase_place_diff][mouse_recday]=coactivity_mean
            #coactivity_mean_all[phase_place_diff,mouse_recday_ind]=coactivity_mean
    except Exception as e:
        print(e)
        


# In[ ]:


phase_place_diff=1
np.nanmean(Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs'][stage][mouse_recday])


# In[ ]:


mouse_recday
np.arange(num_phase_place_diffs)
Sleepcorr_pairs_dic.keys()


# In[ ]:


use_phase=True
minus_flipped=False
forward_distance=False
remove_zerotuned=True
use_coactivity=False

phase_place_diff = 0
betas_all=[]
coactivity_all=[]
#for stage in ['preX','preX2']:
#for stage_ind in np.arange(num_stages):
for stage in stages:
    #stage='post'+str(stage_ind)

    #try:
    if phase_place_diff==0:
        tuningangle_=Sleepcorr_pairs_dic['Anchored_pairs_angles'][stage][mouse_recday]
        phaseangle_=Sleepcorr_pairs_dic['Anchored_pairs_phases'][stage][mouse_recday]
        #phaseangle_=state_to_phase(tuningangle_,num_states)
        corrs_=Sleepcorr_pairs_dic['Anchored_pairs_sleepcorrs'][stage][mouse_recday]

    else:
        tuningangle_=Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_angles'][stage][mouse_recday]
        phaseangle_=Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_phases'][stage][mouse_recday]
        corrs_=Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs'][stage][mouse_recday]

    if len(tuningangle_)==0:
        continue

    if forward_distance==False:
        tuningangle_[tuningangle_>180]=360-tuningangle_[tuningangle_>180]
        phaseangle_[phaseangle_>180]=360-phaseangle_[phaseangle_>180]
        ##i.e. taking distance either side


    if phase_place_diff==0:
        tuningangle_all.append(tuningangle_)
        phaseangle_all.append(phaseangle_)
        corrs_all.append(corrs_)
    else:
        tuningangle_all_Xanchor.append(tuningangle_)
        phaseangle_all_Xanchor.append(phaseangle_)
        corrs_all_Xanchor.append(corrs_)

    exec('tuningangle_all'+str(phase_place_diff)+'.append(tuningangle_)')
    exec('phaseangle_all'+str(phase_place_diff)+'.append(phaseangle_)')
    exec('corrs_all'+str(phase_place_diff)+'.append(corrs_)')



    tuningangle_all_dic[phase_place_diff][stage][mouse_recday]=tuningangle_
    phaseangle_all_dic[phase_place_diff][stage][mouse_recday]=phaseangle_
    corrs_all_dic[phase_place_diff][stage][mouse_recday]=corrs_

    phase_nonans=~np.isnan(phaseangle_)

    if remove_zerotuned==True:
        not_zero_boolean=np.logical_and(tuningangle_>thr_lower,tuningangle_<thr_upper)
    else:
        not_zero_boolean=np.repeat(True,len(tuningangle_))



    if use_phase==True:
        used_boolean=np.logical_and(not_zero_boolean,phase_nonans)
        matx=np.vstack((np.repeat(1,len(tuningangle_)),phaseangle_,tuningangle_)).T[used_boolean] #
        demeaned_1=matx[:,1]-np.mean(matx[:,1])
        demeaned_2=matx[:,2]-np.mean(matx[:,2])
        demeaned_matx=np.column_stack((matx[:,0],demeaned_1,demeaned_2))

        phase_ind=1
        state_ind=2
    else:
        used_boolean=not_zero_boolean
        matx=np.vstack((np.repeat(1,len(tuningangle_)),tuningangle_)).T[used_boolean]
        demeaned_1=matx[:,1]-np.mean(matx[:,1])
        demeaned_matx=np.column_stack((matx[:,0],demeaned_1))
        state_ind=1


    y=np.asarray(corrs_).T[used_boolean]
    coactivity_all.append(np.asarray(corrs_).T)

    y_minus_flipped=np.asarray([y[ii]-np.flip(y[ii]) for ii in range(len(y))])
    y_mean=np.asarray([(y[ii]+np.flip(y[ii]))/2 for ii in range(len(y))])

    X_coactivity=np.asarray([np.vstack(([np.repeat(demeaned_matx[ii][jj], len(y.T))                                         for jj in range(len(demeaned_matx[ii]))],y_mean[ii]-np.mean(y_mean)))                             for ii in range(len(demeaned_matx))])

    if len(demeaned_matx)<=1:
        continue
    if use_coactivity==False:
        X_used=demeaned_matx

        if minus_flipped==True:
            y_used=y_minus_flipped
        else:
            y_used=y


        model = LinearRegression()
        model.fit(X=X_used, y=y_used)

        N=len(X_used)
        p=len(X_used.T)

        beta_hat = np.linalg.inv(X_used.T @ X_used) @ X_used.T @ y_used ##ordinary least squares

        y_hat = model.predict(X_used)
        residuals = y_used - y_hat
        residual_sum_of_squares = residuals.T @ residuals
        sigma_squared_hat = np.diagonal(residual_sum_of_squares) / (N - p)

        var_beta_hat = np.asarray([np.linalg.inv(X_used.T @ X_used) * sigma_squared_hat[ii]                                   for ii in range(len(sigma_squared_hat))])


    else:
        X_used=X_coactivity
        y_used=y

        beta_hat=np.zeros((np.shape(X_used)[1],len(y_used.T)))
        for ii in range(len(y_used.T)):

            X_used_lag=X_used[:,:,ii]
            y_used_lag=y_used[:,ii]

            model = LinearRegression()
            model.fit(X=X_used_lag, y=y_used_lag)
            beta_hat[:,ii]=model.coef_


    if feature=='state':
        indx=state_ind
    elif feature=='phase':
        indx=phase_ind
    betas_all.append(beta_hat[indx])
    #except Exception as e:
    #    print(e)


betas_all=np.vstack((betas_all))
betas_mean=np.mean(betas_all,axis=0)
betas_sem=st.sem(betas_all,axis=0)
plt.errorbar((np.arange(len(y_minus_flipped.T))-centre_point)*25,betas_mean,yerr=betas_sem,color=color)
plt.axhline(0,color='black')
plt.axvline(0,color='black')
plt.show()

beta_oneside=betas_mean[:centre_point]
beta_max=np.max(beta_oneside)
beta_min=np.min(beta_oneside)
beta_max_min_all[phase_place_diff,mouse_recday_ind]=[beta_max,beta_min]
beta_max_mindiff_all[phase_place_diff,mouse_recday_ind]=beta_max-beta_min

betas_mean_all[phase_place_diff,mouse_recday_ind]=betas_mean

coactivity_all=np.asarray((coactivity_all))

coactivity_mean=np.mean(coactivity_all,axis=0)
coactivity_mean_dic[phase_place_diff][mouse_recday]=coactivity_mean
#coactivity_mean_all[phase_place_diff,mouse_recday_ind]=coactivity_mean


# In[ ]:


np.nanmean(abs(y_used))
y_used[:,45]


# In[ ]:


(X_used[:,1]).astype(int)


# In[ ]:


phase_place_diff


# In[ ]:


centre_point=40
#gap=1
color_order=['midnightblue','blue','dodgerblue','aqua','grey']

coactivity_alllags=[]
for phase_place_diff in np.arange(num_phase_place_diffs):
    coactivity_all_=np.vstack((dict_to_array(coactivity_mean_dic[phase_place_diff])))
    coactivity_mean_=np.mean(coactivity_all_,axis=0)
    coactivity_sem_=st.sem(coactivity_all_,axis=0)
    
    coactivity_alllags.append(np.nanmean(coactivity_all_,axis=1))

    
    plt.errorbar((np.arange(len(coactivity_mean_))-centre_point)*25,coactivity_mean_,                 yerr=coactivity_sem_,color=color_order[phase_place_diff])
coactivity_alllags=np.asarray(coactivity_alllags)
bar_plotX(coactivity_alllags.T,'none', 0, 0.005, 'nopoints', 'paired', 0.025)


# In[ ]:


coactivity_mean_dic[phase_place_diff]


# In[ ]:





# In[ ]:


bar_plotX(beta_max_mindiff_all,'none',0,20e-5,'points','paired',0.025)
plt.show()

xy=beta_max_mindiff_all[:,~np.isnan(beta_max_mindiff_all[0])]

plot_scatter(xy[0],np.mean(xy[1:],axis=0),'none')
plt.show()
st.wilcoxon(xy[0],np.mean(xy[1:],axis=0))


# In[ ]:





# In[ ]:


day_type_dicX['combined_ABCDonly']


# In[ ]:


###means across recording days
means_alldays=np.nanmean(betas_mean_all,axis=1)
sems_alldays=st.sem(betas_mean_all,axis=1,nan_policy='omit')
color_order=['midnightblue','blue','dodgerblue','aqua','grey']
for phase_place_diff in np.arange(num_phase_place_diffs):
    print(phase_place_diff)
    plt.errorbar((np.arange(len(y_minus_flipped.T))-centre_point)*25,means_alldays[phase_place_diff],                 yerr=sems_alldays[phase_place_diff],color=color_order[phase_place_diff])
    plt.axhline(0,color='black')
    plt.axvline(0,color='black')
    #plt.show()


# In[ ]:





# In[ ]:


#mean across all pairs and sleep sessions - within vs between
color_ind=0
color_order=['midnightblue','aqua']
feature='state'


for anchor_status,name_addition in {'co_Anchored':'','X_Anchor':'_Xanchor'}.items():
    exec('tuningangle_all_=tuningangle_all'+name_addition)
    exec('phaseangle_all_=phaseangle_all'+name_addition)
    exec('corrs_all_=corrs_all'+name_addition)


    tuningangle_=np.hstack((tuningangle_all_))
    phaseangle_=np.hstack((phaseangle_all_))
    corrs_=np.hstack((corrs_all_))
    
    if forward_distance==False:
        tuningangle_[tuningangle_>180]=360-tuningangle_[tuningangle_>180]
        phaseangle_[phaseangle_>180]=360-phaseangle_[phaseangle_>180]
        ##i.e. taking distance either side

    if remove_zerotuned==True:
        not_zero_boolean=np.logical_and(tuningangle_>thr_lower,tuningangle_<thr_upper)
    else:
        not_zero_boolean=np.repeat(True,len(tuningangle_))

    phase_nonans=~np.isnan(phaseangle_)

    used_boolean=not_zero_boolean
    used_boolean=np.logical_and(not_zero_boolean,phase_nonans)
    if remove_zerotuned==False:
        used_boolean=phase_nonans

    matx=np.vstack((np.repeat(1,len(tuningangle_)),tuningangle_)).T[used_boolean]
    demeaned_1=matx[:,1]-np.mean(matx[:,1])
    demeaned_matx=np.column_stack((matx[:,0],demeaned_1))

    state_ind=1

    if use_phase==True:
        matx=np.vstack((np.repeat(1,len(tuningangle_)),phaseangle_,tuningangle_)).T[used_boolean] #
        demeaned_1=matx[:,1]-np.nanmean(matx[:,1])
        demeaned_2=matx[:,2]-np.nanmean(matx[:,2])
        demeaned_matx=np.column_stack((matx[:,0],demeaned_1,demeaned_2))

        phase_ind=1
        state_ind=2


    X=demeaned_matx

    y=np.asarray(corrs_).T[used_boolean]

    y_minus_flipped=np.asarray([y[ii]-np.flip(y[ii]) for ii in range(len(y))])
    
    if minus_flipped==True:
        y_used=y_minus_flipped
    else:
        y_used=y
    
    model = LinearRegression()
    model.fit(X=X, y=y_used)

    N=len(X)
    p=len(X.T)

    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y_used ##ordinary least squares
    #print(beta_hat)

    #print(model.coef_[:,1])

    y_hat = model.predict(X)
    residuals = y_used - y_hat
    residual_sum_of_squares = residuals.T @ residuals
    sigma_squared_hat = np.diagonal(residual_sum_of_squares) / (N - p)

    var_beta_hat = np.asarray([np.linalg.inv(X.T @ X) * sigma_squared_hat[ii]                               for ii in range(len(sigma_squared_hat))])


    if feature=='state':
        indx=state_ind
    elif feature=='phase':
        indx=phase_ind
    
    
    std_err=var_beta_hat[:,indx, indx] ** 0.5
    plt.errorbar((np.arange(len(y_used.T))-centre_point)*25,beta_hat[indx],yerr=std_err,                 color=color_order[color_ind])
    plt.axhline(0,color='black')
    plt.axvline(0,color='black')
    color_ind=+1
    
    print(np.nanmean(beta_hat[indx,:centre_point]))
    #Replay_pairs_perday_dic[coherence_status]['betas'][sleep_stage][mouse_recday]=beta_hat[indx]
    #Replay_pairs_perday_dic[coherence_status]['betas_sems'][sleep_stage][mouse_recday]=std_err
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


#mean across all pairs and sleep sessions - within vs different place/phase distances
color_ind=0
color_order=['midnightblue','aqua']
feature='state'


color_order=['midnightblue','blue','dodgerblue','aqua','grey']


for phase_place_diff in np.arange(num_phase_place_diffs):
    #tuningangle_all_=np.hstack((dict_to_array(tuningangle_all_dic[phase_place_diff])))
    #phaseangle_all_=np.hstack((dict_to_array(phaseangle_all_dic[phase_place_diff])))    
    #corrs_all_=np.hstack(([corrs_all_dic[phase_place_diff][mouse_recday] \
    #                               for mouse_recday in corrs_all_dic[phase_place_diff].keys()]))
    #corrs_=corrs_all_
    
    exec('tuningangle_all_=tuningangle_all'+str(phase_place_diff))
    exec('phaseangle_all_=phaseangle_all'+str(phase_place_diff))
    exec('corrs_all_=corrs_all'+str(phase_place_diff))

    tuningangle_=np.hstack((tuningangle_all_))
    phaseangle_=np.hstack((phaseangle_all_))
    corrs_=np.hstack((corrs_all_))
    
    if forward_distance==False:
        tuningangle_[tuningangle_>180]=360-tuningangle_[tuningangle_>180]
        phaseangle_[phaseangle_>180]=360-phaseangle_[phaseangle_>180]
        ##i.e. taking distance either side

    if remove_zerotuned==True:
        not_zero_boolean=np.logical_and(tuningangle_>thr_lower,tuningangle_<thr_upper)
    else:
        not_zero_boolean=np.repeat(True,len(tuningangle_))

    phase_nonans=~np.isnan(phaseangle_)

    used_boolean=not_zero_boolean
    used_boolean=np.logical_and(not_zero_boolean,phase_nonans)
    if remove_zerotuned ==False:
        used_boolean=phase_nonans

    matx=np.vstack((np.repeat(1,len(tuningangle_)),tuningangle_)).T[used_boolean]
    demeaned_1=matx[:,1]-np.mean(matx[:,1])
    demeaned_matx=np.column_stack((matx[:,0],demeaned_1))

    state_ind=1

    if use_phase==True:
        matx=np.vstack((np.repeat(1,len(tuningangle_)),phaseangle_,tuningangle_)).T[used_boolean] #
        demeaned_1=matx[:,1]-np.nanmean(matx[:,1])
        demeaned_2=matx[:,2]-np.nanmean(matx[:,2])
        demeaned_matx=np.column_stack((matx[:,0],demeaned_1,demeaned_2))

        phase_ind=1
        state_ind=2


    X=demeaned_matx

    y=np.asarray(corrs_).T[used_boolean]

    y_minus_flipped=np.asarray([y[ii]-np.flip(y[ii]) for ii in range(len(y))])
    
    if minus_flipped==True:
        y_used=y_minus_flipped
    else:
        y_used=y
        
    model = LinearRegression()
    model.fit(X=X, y=y_used)

    N=len(X)
    p=len(X.T)

    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y_used ##ordinary least squares
    #print(beta_hat)

    #print(model.coef_[:,1])

    y_hat = model.predict(X)
    residuals = y_used - y_hat
    residual_sum_of_squares = residuals.T @ residuals
    sigma_squared_hat = np.diagonal(residual_sum_of_squares) / (N - p)

    var_beta_hat = np.asarray([np.linalg.inv(X.T @ X) * sigma_squared_hat[ii]                               for ii in range(len(sigma_squared_hat))])


    if feature=='state':
        indx=state_ind
    elif feature=='phase':
        indx=phase_ind
    
    
    std_err=var_beta_hat[:,indx, indx] ** 0.5
    plt.errorbar((np.arange(len(y_used.T))-centre_point)*25,beta_hat[indx],yerr=std_err,                 color=color_order[color_ind])
    plt.axhline(0,color='black')
    plt.axvline(0,color='black')
    color_ind+=1
    
    print(np.nanmean(beta_hat[indx,:centre_point]))
    #Replay_pairs_perday_dic[coherence_status]['betas'][sleep_stage][mouse_recday]=beta_hat[indx]
    #Replay_pairs_perday_dic[coherence_status]['betas_sems'][sleep_stage][mouse_recday]=std_err

plt.show()


# In[ ]:





# In[ ]:


#mean across all pairs - divided by sleep session - within vs different place/phase distances

color_order=['midnightblue','aqua']
feature='state'


color_order=['midnightblue','blue','dodgerblue','aqua','grey']

for stage in stages:
    print(stage)
    color_ind=0
    for phase_place_diff in np.arange(num_phase_place_diffs):
        tuningangle_all_=np.hstack((dict_to_array(tuningangle_all_dic[phase_place_diff][stage])))
        phaseangle_all_=np.hstack((dict_to_array(phaseangle_all_dic[phase_place_diff][stage])))    
        corrs_all_=np.hstack(([corrs_all_dic[phase_place_diff][stage][mouse_recday]                                        for mouse_recday in corrs_all_dic[phase_place_diff][stage].keys()]))

        #exec('tuningangle_all_=tuningangle_all'+str(phase_place_diff))
        #exec('phaseangle_all_=phaseangle_all'+str(phase_place_diff))
        #exec('corrs_all_=corrs_all'+str(phase_place_diff))

        tuningangle_=np.hstack((tuningangle_all_))
        phaseangle_=np.hstack((phaseangle_all_))
        #corrs_=np.hstack((corrs_all_))
        corrs_=corrs_all_

        if forward_distance==False:
            tuningangle_[tuningangle_>180]=360-tuningangle_[tuningangle_>180]
            phaseangle_[phaseangle_>180]=360-phaseangle_[phaseangle_>180]
            ##i.e. taking distance either side

        if remove_zerotuned==True:
            not_zero_boolean=np.logical_and(tuningangle_>thr_lower,tuningangle_<thr_upper)
        else:
            not_zero_boolean=np.repeat(True,len(tuningangle_))

        phase_nonans=~np.isnan(phaseangle_)

        used_boolean=not_zero_boolean
        used_boolean=np.logical_and(not_zero_boolean,phase_nonans)
        if remove_zerotuned ==False:
            used_boolean=phase_nonans

        matx=np.vstack((np.repeat(1,len(tuningangle_)),tuningangle_)).T[used_boolean]
        demeaned_1=matx[:,1]-np.mean(matx[:,1])
        demeaned_matx=np.column_stack((matx[:,0],demeaned_1))

        state_ind=1

        if use_phase==True:
            matx=np.vstack((np.repeat(1,len(tuningangle_)),phaseangle_,tuningangle_)).T[used_boolean] #
            demeaned_1=matx[:,1]-np.nanmean(matx[:,1])
            demeaned_2=matx[:,2]-np.nanmean(matx[:,2])
            demeaned_matx=np.column_stack((matx[:,0],demeaned_1,demeaned_2))

            phase_ind=1
            state_ind=2


        X=demeaned_matx

        y=np.asarray(corrs_).T[used_boolean]

        y_minus_flipped=np.asarray([y[ii]-np.flip(y[ii]) for ii in range(len(y))])

        if minus_flipped==True:
            y_used=y_minus_flipped
        else:
            y_used=y

        model = LinearRegression()
        model.fit(X=X, y=y_used)

        N=len(X)
        p=len(X.T)

        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y_used ##ordinary least squares
        #print(beta_hat)

        #print(model.coef_[:,1])

        y_hat = model.predict(X)
        residuals = y_used - y_hat
        residual_sum_of_squares = residuals.T @ residuals
        sigma_squared_hat = np.diagonal(residual_sum_of_squares) / (N - p)

        var_beta_hat = np.asarray([np.linalg.inv(X.T @ X) * sigma_squared_hat[ii]                                   for ii in range(len(sigma_squared_hat))])


        if feature=='state':
            indx=state_ind
        elif feature=='phase':
            indx=phase_ind


        std_err=var_beta_hat[:,indx, indx] ** 0.5
        plt.errorbar((np.arange(len(y_used.T))-centre_point)*25,beta_hat[indx],yerr=std_err,                     color=color_order[color_ind])
        plt.axhline(0,color='black')
        plt.axvline(0,color='black')
        color_ind+=1

        print(np.nanmean(beta_hat[indx,:centre_point]))
        #Replay_pairs_perday_dic[coherence_status]['betas'][sleep_stage][mouse_recday]=beta_hat[indx]
        #Replay_pairs_perday_dic[coherence_status]['betas_sems'][sleep_stage][mouse_recday]=std_err
    plt.savefig(Ephys_output_folder_dropbox+'Pair_distance_vs_coactivity_regression_session_'+stage+'.svg',                bbox_inches = 'tight', pad_inches = 0) 
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
findings:


-small/insignificant difference in coactivity between coanchored and non-anchored
-more coactvity for closer neurons if co-anchored comapred to non-co-anchored - ONLY for long sleep sessions!
-replay/preplay flips around
    -flips per day
    -last sleep session: net reverse replay - more for co-anchored pairs
    -preX2: forward preplay (preX seems mixed)

To do:
-longer bin coactivity (250ms) - done  - not better
-spatial replay
-basic checks (correlations)


spatial replay
every pair

'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


for phase_place_diff in np.arange(num_phase_place_diffs):
    betas_all=[]

    coactivity_all=[]
    #for stage in ['preX','preX2']:
    for stage_ind in np.arange(num_stages):
        stage='post'+str(stage_ind)


        if phase_place_diff==0:
            tuningangle_=Sleepcorr_pairs_dic['Anchored_pairs_angles'][stage][mouse_recday]
            phaseangle_=Sleepcorr_pairs_dic['Anchored_pairs_phases'][stage][mouse_recday]
            #phaseangle_=state_to_phase(tuningangle_,num_states)
            corrs_=Sleepcorr_pairs_dic['Anchored_pairs_sleepcorrs'][stage][mouse_recday]

        else:
            tuningangle_=Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_angles'][stage][mouse_recday]
            phaseangle_=Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_phases'][stage][mouse_recday]
            corrs_=Sleepcorr_pairs_dic[phase_place_diff]['Anchored_pairs_sleepcorrs'][stage][mouse_recday]


# In[ ]:





# In[ ]:





# In[ ]:





# In[68]:


#SAVING FILES 

try: 
    os.mkdir(Intermediate_object_folder) 
except FileExistsError: 
    pass

objects_dic={'Sleepcorr_matrix_shifted_dic':Sleepcorr_matrix_shifted_dic,'session_dic':session_dic,             'session_dic_behaviour':session_dic_behaviour, 'Sleepcorr_pairs_timebin_dic':Sleepcorr_pairs_timebin_dic}

for name, dicX in objects_dic.items(): 
    data=dicX 
    data_filename_memmap = os.path.join(Intermediate_object_folder, name) 
    dump(data, data_filename_memmap)


# In[ ]:





# In[ ]:





# In[ ]:




