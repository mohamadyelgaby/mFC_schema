#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


###Folders
#Data_folder='/Taskspace_abstraction/Data/'
Data_folder='C:/Users/moham/team_mouse Dropbox/Mohamady El-Gaby//Taskspace_abstraction/Data//Intermediate_objects/'

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


# In[2]:


###Defining directories

Data_folder='C:/Users/moham/team_mouse Dropbox/Mohamady El-Gaby//Taskspace_abstraction/Data//Intermediate_objects/'
Data_folder_P='P:/Taskspace_abstraction/Data/' ## if working in P
base_dropbox='C:/Users/moham/team_mouse Dropbox/Mohamady El-Gaby/'

#base_dropbox='D:/team_mouse Dropbox/Mohamady El-Gaby/'

Data_folder_dropbox=base_dropbox+'/Taskspace_abstraction/Data/' ##if working in C
Behaviour_output_folder = 'P:/Taskspace_abstraction/Results/Behaviour/'
Ephys_output_folder = 'P:/Taskspace_abstraction/Results/Ephys/'
Ephys_output_folder_dropbox = base_dropbox+'/Taskspace_abstraction/Results/Ephys/'
Intermediate_object_folder_dropbox = Data_folder_dropbox+'/Intermediate_objects/'


Code_folder='/Taskspace_abstraction/Code/'

base_ceph='Z:/mohamady_el-gaby/'
Data_folder_ceph='Z:/mohamady_el-gaby/Taskspace_abstraction_2/Data/'
Data_folder_ceph1='Z:/mohamady_el-gaby/Taskspace_abstraction/Data/'
Data_folder_ceph2='Z:/mohamady_el-gaby/Taskspace_abstraction_2/Data/'

Intermediate_object_folder_ceph = Data_folder_ceph1+'/Intermediate_objects/'

Intermediate_object_folder=Intermediate_object_folder_dropbox


# In[3]:


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
#from pingouin import partial_corr
from collections import Counter
import random
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter1d
import warnings
import scipy as sp


# In[4]:


numpy.__version__


# In[5]:


'''

-get phase vectors and compare within and between days
-shuffle and compare to shuffle
-what % of significant within-day cells are also significant across days 


'''


# In[6]:


##Importing custom functions
module_path = os.path.abspath(os.path.join(Code_folder))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from mBaseFunctions import rec_dd, remove_empty, indicesX2, create_binary, unique_adjacent, dict_to_array, flatten,remove_empty, concatenate_complex2,mean_complex2, std_complex2, rand_jitter, rand_jitterX, bar_plotX, remove_nan,remove_nanX, column_stack_clean, polar_plot_state, positive_angle, circular_angle, plot_grouped_error,timestamp_to_binary, fill_nans, fill_nansX, angle_to_state, binned_array, binned_arrayX, plot_scatter,two_proportions_test, rearrange_matrix, matrix_triangle, scramble, smooth_circular, plot_spatial_maps, state_to_phase,middle_value,max_bin_safe, random_rotation, non_cluster_indices, cumulativeDist_plot, cumulativeDist_plot_norm,angle_to_distance, rotate, angle_to_stateX, range_ratio_peaks, equalize_rowsX, cross_corr_fast, plot_dendrogram,rotate, rank_repeat, edge_node_fill, split_mode, concatenate_states, predict_task_map, predict_task_map_policy,number_of_repeats, find_direction, Edge_grid, polar_plot_stateX, polar_plot_stateX2,arrange_plot_statecells_persession, arrange_plot_statecells, arrange_plot_statecells_persessionX2, Task_grid_plotting2,Task_grid_plotting, Task_grid, Task_grid2, Edge_grid_coord, Edge_grid_coord2, direction_dic_plotting, plot_spatial_mapsX,angle_to_distance, rank_repeat, number_of_repeats, noplot_timecourseA, noplot_timecourseAx, noplot_timecourseB,noplot_timecourseBx, noplot_scatter, number_of_repeats_ALL, rearrange_for_ANOVAX, non_repeat_ses_maker


# In[7]:


def subset_complex(x,y,z,scaling):
    return[x[y[z[ii],0]//scaling:y[z[ii],-1]//scaling] for ii in range(len(z))]

def subset_complex2(x,y,z,scaling):
    return[x[:,y[z[ii],0]//scaling:y[z[ii],-1]//scaling] for ii in range(len(z))]

def num_of_repeats2(MyList):
    my_dict = {i:list(MyList).count(i) for i in MyList}
    
    return(np.asarray([my_dict[element] for element in MyList]))


###Defining functions
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
    
def indep_roll(arr, shifts, axis=1):
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted. 
    """
    arr = np.swapaxes(arr,axis,-1)
    all_idcs = np.ogrid[[slice(0,n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1] 
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result,-1,axis)
    return arr


def arrange_plot_statecells_persessionX2(mouse_recday,neuron,Data_folder,sessions_included=None                                       ,fignamex=False,sigma=10,                                       save=False,plot=False,figtype='.svg', Marker=False,                                       fields_booleanx=[],measure_type='mean', abstract_structures=[],                                      repeated=False,behaviour_oversampling_factor=3,behaviour_rate=1000,                                       tracking_oversampling_factor=50):

    awake_sessions=session_dic['awake'][mouse_recday]
    awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
    
    colors=np.repeat('blue',len(awake_sessions_behaviour))
    plot_boolean=np.repeat(False,len(awake_sessions_behaviour))
    plot_boolean[sessions_included]=True
    
    
    
    num_trials_day=np.load(Intermediate_object_folder+'Num_trials_'+mouse_recday+'.npy')

    fig= plt.figure(figsize=plt.figaspect(1)*4.5)
    fig.tight_layout()
    for awake_session_ind, timestamp in enumerate(awake_sessions_behaviour):
        structure_abstract=abstract_structures[awake_session_ind]
        
        if num_trials_day[awake_session_ind]<2:
            print('Not enough trials session'+str(awake_session_ind))
            continue
        if timestamp not in awake_sessions:
            print('Ephys not used for session'+str(awake_session_ind))
            continue
            
            
        try:
            norm_activity_all=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
        except:
            print('No file found session'+str(awake_session_ind))
            continue
        
        norm_activity_neuron=norm_activity_all[neuron]
        
        xxx=np.asarray(norm_activity_neuron).T
        standardized_FR=np.hstack([np.nanmean(xxx[ii],axis=0) for ii in range(len(xxx))])*        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        standardized_FR_sem=np.hstack([st.sem(xxx[ii],axis=0,nan_policy='omit') for ii in range(len(xxx))])*        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        standardized_FR_smoothed=smooth_circular(standardized_FR,sigma=sigma)            
        standardized_FR_sem_smoothed=smooth_circular(standardized_FR_sem,sigma=sigma)                    

        
        standardized_FR_smoothed_upper=standardized_FR_smoothed+standardized_FR_sem_smoothed
        standardized_FR_smoothed_lower=standardized_FR_smoothed-standardized_FR_sem_smoothed
       
        
        color=colors[awake_session_ind]
        
        ax = fig.add_subplot(1, len(awake_sessions_behaviour), awake_session_ind+1, projection='polar')
        if len(fields_booleanx)>0:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,                              ax,color=color, Marker=Marker,fields_booleanx=fields_booleanx[awake_session_ind],                             structure_abstract=structure_abstract,repeated=repeated)
        else:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,                              ax,color=color, Marker=False,structure_abstract=structure_abstract,repeated=repeated)
    plt.margins(0,0)
    #plt.tight_layout()
    if save==True:
        plt.savefig(fignamex+str(awake_session_ind)+figtype)
    if plot==True & plot_boolean[awake_session_ind]==True:
        plt.show()
    else:
        plt.close() 
        
def split_mode(xx,num_bins):
    xxx=np.array_split(xx,num_bins)
    return(np.asarray([st.mode(xxx[ii],nan_policy='omit')[0]                       if np.isnan(np.nanmean(xxx[ii]))==False                       else st.mode(xxx[ii])[0]
                       for ii in range(len(xxx))]))


# In[ ]:





# In[8]:


##Importing meta Data
Mice_cohort_dic={'me08':3,'ah03':3,'me10':4,'me11':4,'ah04':4,                'ab03':6,'ah07':6} ##'me12':5,'me13':5,
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


# In[ ]:





# In[9]:


#LOADING FILES - tracking, behaviour, Ephys raw, Ephys binned
tt=time.time()


try:
    os.mkdir(Intermediate_object_folder)
except FileExistsError:
    pass

dictionaries_list=['Num_trials_dic','Num_trials_dic2','ROI_accuracy_dic','day_type_sameTask_dic','session_dic',                  'cluster_dic','recday_numbers_dic','day_type_dicX','Confidence_rotations_dic','Task_num_dic',                   'scores_dic','times_dic','Combined_days_dic','session_dic_behaviour','GLM_dic','GLM_withinTask_dic',                  'GLM_dic2','tuning_singletrial_dic','tuning_singletrial_dic2','Tuned_dic','States_raw_dic',
                  'Phases_raw_dic','Phases_raw_dic2','Times_from_reward_dic','GLM_dic_policy','Tuned_dic2',\
                  'GLM_anchoring_ABCDE_dic']

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


# In[10]:


len(cluster_dic['good_clus']['ah07_21112023_22112023'])


# In[270]:


sum_good_postcuration=np.sum([len(cluster_dic['good_clus'][mouse_recday]) for mouse_recday in day_type_dicX['combined']])
sum_good_precuration=np.sum([np.sum(cluster_dic['label'][mouse_recday]) for mouse_recday in day_type_dicX['combined']])


# In[271]:


sum_good_postcuration/sum_good_precuration


# In[272]:


sum_good_postcuration_single=np.sum([len(cluster_dic['good_clus'][mouse_recday]) for mouse_recday in day_type_dicX['3_task']])
sum_good_precuration_single=np.sum([np.sum(cluster_dic['label'][mouse_recday]) for mouse_recday in day_type_dicX['3_task']])


# In[273]:


sum_good_postcuration_single/sum_good_precuration_single


# In[ ]:





# In[11]:


##Defining Task grid
from scipy.spatial import distance_matrix
from itertools import product
x=(0,1,2)
Task_grid=np.asarray(list(product(x, x)))
Task_grid_plotting=np.column_stack((Task_grid[:,1],Task_grid[:,0]))
Task_grid_plotting2=[]
for yy in np.arange(3):
    y=int(yy*2)
    for xx in np.arange(3):
        x=int(xx*2)    
        Task_grid_plotting2.append([x,y])
Task_grid_plotting2=np.asarray(Task_grid_plotting2)    
Task_grid2=np.column_stack((Task_grid_plotting2[:,1],Task_grid_plotting2[:,0]))

Edge_grid=np.asarray([[1,2],[2,3],[1,4],[2,5],[3,6],[4,5],[5,6],[4,7],[5,8],[6,9],[7,8],[8,9]]) ###
Edge_grid_=Edge_grid-1
Edge_grid_coord_x=[Task_grid[Edge_grid_[ii][0]][0]+Task_grid[Edge_grid_[ii][1]][0] for ii in range(len(Edge_grid_))]
Edge_grid_coord_y=rank_repeat(Edge_grid_coord_x)
Edge_grid_coord=np.column_stack((Edge_grid_coord_x,Edge_grid_coord_y))
Edge_grid_coord2=np.asarray([[0,1],[0,3],[1,0],[1,2],[1,4],[2,1],[2,3],[3,0],[3,2],[3,4],[4,1],[4,3]])

direction_dic={'N':[1,0],'S':[-1,0],'W':[0,1],'E':[0,-1]}
direction_dic_plotting={'N': [0, -1], 'S': [0, 1], 'W': [-1, 0], 'E': [1, 0]}

##Defining state-action pairs (for policy calculation)
node_one_step_coord=np.asarray([np.asarray(remove_empty([Task_grid[jj]-Task_grid[ii]                                                    if np.sum(abs(Task_grid[jj]-Task_grid[ii]))==1                                                    else [] for ii in range(len(Task_grid))]))                           for jj in range(len(Task_grid))])

node_states=np.asarray([str(ii+1)+'_'+list(direction_dic.keys())[list(direction_dic.values()).                           index(list(node_one_step_coord[ii][jj]))] for ii in range(len(node_one_step_coord))for jj in range(len(node_one_step_coord[ii]))])


edge_one_step_coord=np.asarray([np.asarray(remove_empty([Edge_grid_coord2[jj]-Task_grid2[ii]                                                    if np.sum(abs(Edge_grid_coord2[jj]-Task_grid2[ii]))==1                                                    else [] for ii in range(len(Task_grid2))]))                           for jj in range(len(Edge_grid_coord2))])

edge_states=np.asarray([str(ii+10)+'_'+list(direction_dic.keys())[list(direction_dic.values()).                           index(list(edge_one_step_coord[ii][jj]))] for ii in range(len(edge_one_step_coord))for jj in range(len(edge_one_step_coord[ii]))])

State_action_grid=np.concatenate((node_states,edge_states))

##Defining reverse state-action pairs
node_one_step_coord_reverse=np.asarray([np.asarray(remove_empty([Task_grid[ii]-Task_grid[jj]                                                    if np.sum(abs(Task_grid[ii]-Task_grid[jj]))==1                                                    else [] for ii in range(len(Task_grid))]))                           for jj in range(len(Task_grid))])

node_states_reverse=np.asarray([str(ii+1)+'_'+list(direction_dic.keys())[list(direction_dic.values()).                           index(list(node_one_step_coord_reverse[ii][jj]))]                                for ii in range(len(node_one_step_coord_reverse))for jj in range(len(node_one_step_coord_reverse[ii]))])


State_action_grid_reverse=np.concatenate((node_states_reverse,edge_states))


# In[ ]:





# In[12]:


State_action_grid


# In[ ]:





# In[13]:


#Maze measurements
maze_measurements_dic=rec_dd()
for maze_number in ['1','2']:
    maze_measurementspath=Intermediate_object_folder+'/Maze_measurements/Maze'+str(maze_number)+'_measurements.txt'
    df=pd.read_csv(maze_measurementspath, sep='\t', lineterminator='\r')
    df = df.replace(r'\n','', regex=True) 
    measurements=df.values
    maze_measurements_dic['num_pixels_mean'][maze_number]=np.mean(measurements[:6,1])
    maze_measurements_dic['length_cm'][maze_number]=measurements[6,1]
    maze_measurements_dic['width_cm'][maze_number]=measurements[7,1]
maze_measurements_dic['num_pixels_mean']['BIGMAZE']=350
maze_measurements_dic['length_cm']['BIGMAZE']=47
maze_measurements_dic['width_cm']['BIGMAZE']=47


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


###calculating speed
speed_dic=rec_dd()
smooth_SDs=5

for mouse_recday in day_type_dicX['All']:
    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]
    awake_sessions=session_dic_behaviour['awake'][mouse_recday]
    
    #frame_rate=sampling_dic['frame_rate'][mouse_recday]
    frame_rate=60
    
    print(mouse_recday)
    


    for ses_ind in np.arange(len(awake_sessions)):
        try:
            maze_number=Variable_dic[mouse]['Maze'][Variable_dic[mouse]['Ephys']==awake_sessions[ses_ind]][0]


            maze_length_pixels=maze_measurements_dic['num_pixels_mean'][maze_number]
            maze_length_cm=maze_measurements_dic['length_cm'][maze_number] ##cm
            pixels_per_cm=maze_length_pixels/maze_length_cm

            whl=np.load(Intermediate_object_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            z=np.shape(whl)[0]
            whl_XX=np.column_stack((np.roll(whl[:,0],1),np.roll(whl[:,1],1),whl))
            smoothSDs=2
            whl_XX[whl[:,0]<0]=np.nan
            whl_XX=sp.ndimage.filters.gaussian_filter1d(whl_XX, smoothSDs,axis=0)

            ##speed calculation
            speed=[]
            x1=whl_XX[:,0]
            y1=whl_XX[:,1]
            x2=whl_XX[:,2]
            y2=whl_XX[:,3]
            speedxx=(((x2-x1)**2+(y2-y1)**2)**0.5)
            speedx=speedxx/pixels_per_cm ##pixels to cm
            speed=speedx*frame_rate ##frames to seconds 

            speed_dic[mouse_recday][ses_ind]=speed    
        except Exception as e:
            print(ses_ind)
            print('speed_dic not made')
            print(e)
            


# In[15]:


speed_dic['ab03_01092023']


# In[ ]:





# In[16]:


'''
no XY files:
ah04_19122021_20122021 - all sessions - AB day
me11_15122021_16122021 - all sessions - AB day


me08_12092021_13092021 - ses2


'''


# In[17]:


Phases_raw_dic2.keys()


# In[18]:


###Making phase, state and time arrays
ii=0
jj=0
#num_states=4 - now defined below seperately for each session 
num_phases=5
num_phases2=3
Phases_raw_dic=rec_dd()
Phases_raw_dic2=rec_dd()
States_raw_dic=rec_dd()
Distances_from_reward_dic=rec_dd()

Times_from_reward_dic=rec_dd()
Times_from_start_dic=rec_dd()
for day_type in ['3_task','combined','3_task_all']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        #if mouse_recday in list(Phases_raw_dic2.keys()):
        #    print('Already analysed')
        #    continue
        
        
         
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]
        
        if mouse_recday in ['ah04_19122021_20122021','me11_15122021_16122021']:
            addition='_'
        else:
            addition=''
        
        for ses_ind in np.arange(len(awake_sessions)):
            print(ses_ind)
            
            
            try:
                Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+addition+'.npy')
                num_states=np.shape(Trial_times)[1]-1
                
                distances=speed_dic[mouse_recday][ses_ind]
                
                
                residual_cum=0
                time_cum=0
                
                distances_all=[]
                times_all=[]
                phases_all=[]
                states_all=[]
                times_from_start_all=[]
                for ii in range(len(Trial_times)):
                    distances_trial=[]
                    phases_trial=[]
                    states_trial=[]
                    times_trial=[]
                    times_from_start_trial=[]
                    for jj in range(num_states):
                        start=Trial_times[ii,jj]//25
                        end=Trial_times[ii,jj+1]//25
                        distances_trial_state=np.cumsum(distances[start:end])

                        
                        num_bins=(Trial_times[ii,jj+1]-Trial_times[ii,jj])//25
                        residual=((Trial_times[ii,jj+1]-Trial_times[ii,jj])%25)/25
                        residual_cum+=residual

                        phase_trial_state=np.hstack((np.repeat(np.arange(num_phases),num_bins//num_phases),                                                         np.repeat(int(num_phases-1),num_bins-                                                                   (num_bins//num_phases)*num_phases)))


                        if residual_cum<1:
                            phase_trial_state=phase_trial_state
                        else:
                            phase_trial_state=np.hstack((phase_trial_state,int(num_phases-1)))
                            residual_cum=residual_cum-1

                        states_trial_state=np.repeat(jj,len(phase_trial_state))

                        time_from_reward=np.arange(len(phase_trial_state))

                        time_from_start=time_from_reward+time_cum
                        time_cum+=len(time_from_start)

                        distances_trial.append(distances_trial_state)
                        phases_trial.append(phase_trial_state)
                        states_trial.append(states_trial_state)
                        times_trial.append(time_from_reward)
                        times_from_start_trial.append(time_from_start)

                    distances_all.append(distances_trial)
                    phases_all.append(phases_trial)
                    states_all.append(states_trial)
                    times_all.append(times_trial)
                    times_from_start_all.append(times_from_start_trial)

                Phases_raw_dic2[mouse_recday][ses_ind]=phases_all
                Phases_raw_dic[mouse_recday][ses_ind]=phases_all
                States_raw_dic[mouse_recday][ses_ind]=states_all
                Times_from_reward_dic[mouse_recday][ses_ind]=times_all
                Times_from_start_dic[mouse_recday][ses_ind]=times_from_start_all
                Distances_from_reward_dic[mouse_recday][ses_ind]=distances_all
            except Exception as e:
                print('Phase_raw_dic and Times_from_reward_dic not made')                
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


# In[ ]:





# In[12]:


'''
'me11_05122021' ses0 trial 21
state=1,2
cell 11, 21 = reward cell
cells 0,4,7,8 = still cells - 0 increases after animal ends trajectory (poking?)
2,3,12,13 = start of trajectory
5,9,14,23,29 = middle of trajectory
10,15,19?,0?

9=anticipates trajectory

'''
mouse_recday='ah04_22122021_23122021'
ses_ind=2
Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
speed_raw_=speed_dic[mouse_recday][ses_ind]
acceleration_raw_=np.diff(speed_raw_)/0.025


smooth_SDs=5

speed_raw=gaussian_filter1d(speed_raw_,smooth_SDs)
acceleration_raw=gaussian_filter1d(acceleration_raw_,smooth_SDs)

Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')



trial=21
state=0
smooth_SDs_neuron=3

for neuron in np.arange(30):
    print(neuron)
    activity_state_trial=Neuron_raw[neuron,Trial_times[trial,state]//40:Trial_times[trial,state+1]//40]
    activity_smoothed=gaussian_filter1d(activity_state_trial, smooth_SDs_neuron)
    speed_state_trial=speed_raw[Trial_times[trial,state]//40:Trial_times[trial,state+1]//40]
    acceleration_state_trial=acceleration_raw[Trial_times[trial,state]//40:Trial_times[trial,state+1]//40]


    plt.plot(speed_state_trial/np.mean(speed_state_trial))
    #plt.plot(activity_state_trial/np.mean(activity_state_trial))
    plt.plot(activity_smoothed/np.mean(activity_smoothed))
    plt.plot(acceleration_state_trial/np.mean(abs(acceleration_state_trial)))
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[280]:


###making arrays for split double days 
day_typeX='3_task_all'
day_type='combined_ABCDonly'
for day_type in ['combined_ABCDonly','combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        try:
            mouse=mouse_recday.split('_',1)[0]
            rec_day=mouse_recday.split('_',1)[1]
            rec_day1=rec_day.split('_',1)[0]
            rec_day2=rec_day.split('_',1)[1]
            Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
            Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

            mouse_recday1=mouse+'_'+rec_day1
            mouse_recday2=mouse+'_'+rec_day2

            num_neurons=len(cluster_dic['good_clus'][mouse_recday])

            awake_sessions=session_dic['awake'][mouse_recday]

            days=Combined_days_dic['awake'][mouse_recday]
            for mouse_recdayX in [mouse_recday1,mouse_recday2]:
                awake_sessions_day=session_dic['awake'][mouse_recdayX]
                if mouse_recdayX in day_type_dicX['3_task']:
                    continue
                awake_day_bool=np.hstack(([awake_sessions[ii] in awake_sessions_day for ii in range(len(awake_sessions))]))
                awake_ses_day=np.where(awake_day_bool==True)[0]


                for awake_session_ind_ind, awake_session_ind in enumerate(awake_ses_day): 
                    Neuron_raw=np.load(Intermediate_object_folder_dropbox+'Neuron_raw_'+mouse_recday+'_'+                                       str(awake_session_ind)+'.npy')
                    Location_raw=np.load(Intermediate_object_folder_dropbox+'Location_raw_'+mouse_recday+'_'+                                         str(awake_session_ind)+'.npy')
                    XY_raw=np.load(Intermediate_object_folder_dropbox+'XY_raw_'+mouse_recday+'_'                                   +str(awake_session_ind)+'.npy')
                    Trial_times=np.load(Intermediate_object_folder_dropbox+'trialtimes_'+mouse_recday+'_'                                        +str(awake_session_ind)+'.npy')
                    distances=np.load(Intermediate_object_folder_dropbox+'Distances_from_reward_'+                        mouse_recday+'_'+str(awake_session_ind)+'.npy',allow_pickle=True)

                    np.save(Intermediate_object_folder_dropbox+'Neuron_raw_'+mouse_recdayX+'_'+                                       str(awake_session_ind_ind)+'.npy',Neuron_raw)
                    np.save(Intermediate_object_folder_dropbox+'Location_raw_'+mouse_recdayX+'_'+                                         str(awake_session_ind_ind)+'.npy',Location_raw)
                    np.save(Intermediate_object_folder_dropbox+'XY_raw_'+mouse_recdayX+'_'                                   +str(awake_session_ind_ind)+'.npy',XY_raw)
                    np.save(Intermediate_object_folder_dropbox+'trialtimes_'+mouse_recdayX+'_'                                        +str(awake_session_ind_ind)+'.npy',Trial_times)
                    np.save(Intermediate_object_folder_dropbox+'Distances_from_reward_'+                        mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',distances)


        except Exception as e:
            print(e)


# In[194]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[281]:


###GLM - across tasks/states (using left out state/task combination as test data)

#GLM_dic2=rec_dd()
num_phases=5
num_nodes=9
num_locations=21
num_states=4
num_regressors=6 ##phase, place, time (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place

smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1
state_bins=np.arange(num_states+1)
redo=True
remove_edges=True

redo_list=['ah07_01092023_02092023','ah07_01092023','ah07_02092023']
specific_days=True

if remove_edges==True:
    num_locations=num_nodes
    location_bins=np.arange(num_nodes+1)+1
        

for day_type in ['3_task_all']:
    for mouse_recday in day_type_dicX[day_type]:

        print(mouse_recday)
        
        if specific_days==True and mouse_recday not in redo_list:
            continue
        
        if redo==False:
            if mouse_recday in GLM_dic2['coeffs_all'].keys():
                print('Already Analysed')
                continue

        try: 
            awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
            awake_sessions=session_dic['awake'][mouse_recday]

            #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
            #np.arange(len(awake_sessions_behaviour))])

            num_sessions=len(awake_sessions_behaviour)

            num_neurons=len(cluster_dic['good_clus'][mouse_recday])
            sessions=Task_num_dic[mouse_recday]
            num_refses=len(np.unique(sessions))
            num_comparisons=num_refses-1
            repeat_ses=np.where(rank_repeat(sessions)>0)[0]
            non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

            coeffs_all=np.zeros((num_neurons,len(non_repeat_ses)*num_states,num_regressors))
            coeffs_all[:]=np.nan
            
            
            
            for ses_ind_test_ind,ses_ind_test in enumerate(non_repeat_ses):
                print(ses_ind_test)
                training_sessions=np.setdiff1d(non_repeat_ses,ses_ind_test)
                
                ###Training
                phases_conc_all_=[]
                states_conc_all_=[]
                Location_raw_eq_all_=[]
                Neuron_raw_all_=[]
                for ses_ind_training_ind, ses_ind_training in enumerate(training_sessions):
                    try:
                        Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        speed_raw=speed_dic[mouse_recday][ses_ind_training]


                        acceleration_raw_=np.diff(speed_raw)/0.025
                        acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')

                    except:
                        print('Files not found for session '+str(ses_ind_training))
                        continue

                    phases=Phases_raw_dic[mouse_recday][ses_ind_training]
                    phases_conc=concatenate_complex2(concatenate_complex2(phases))
                    states=States_raw_dic[mouse_recday][ses_ind_training]
                    states_conc=concatenate_complex2(concatenate_complex2(states))
                    times=Times_from_reward_dic[mouse_recday][ses_ind_training]
                    times_conc=concatenate_complex2(concatenate_complex2(times))
                    #distances=Distances_from_reward_dic[mouse_recday][ses_ind_training]
                    distances=np.load(Intermediate_object_folder_dropbox+'Distances_from_reward_'+                    mouse_recday+'_'+str(ses_ind_training)+'.npy',allow_pickle=True)
                    distances_conc=concatenate_complex2(concatenate_complex2(distances))
                    speed_raw_eq=speed_raw[:len(phases_conc)]
                    acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
                    Location_raw_eq=Location_raw[:len(phases_conc)]

                    if remove_edges==True:
                        Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges

                    if len(phases_conc)>=len(speed_raw_eq):
                        print('Mismatch between speed and ephys - work around here but check')
                        phases_conc=phases_conc[:len(speed_raw_eq)]
                        states_conc=states_conc[:len(speed_raw_eq)]
                        times_conc=times_conc[:len(speed_raw_eq)]
                        distances_conc=distances_conc[:len(speed_raw_eq)]
                        Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]
                        Neuron_raw=Neuron_raw[:,:len(speed_raw_eq)]

                    speed_phases=st.binned_statistic(phases_conc, speed_raw_eq , bins=phase_bins)[0]
                    acceleration_phases=st.binned_statistic(phases_conc, acceleration_raw_eq , bins=phase_bins)[0]
                    
                    phases_conc_all_.append(phases_conc)
                    states_conc_all_.append(states_conc)
                    Location_raw_eq_all_.append(Location_raw_eq)
                    Neuron_raw_all_.append(Neuron_raw)
                    


                phases_conc_all_=np.hstack((phases_conc_all_))
                states_conc_all_=np.hstack((states_conc_all_))
                Location_raw_eq_all_=np.hstack((Location_raw_eq_all_))
                Neuron_raw_all_=np.hstack((Neuron_raw_all_))
                
                



                ###Test
                try:
                    Neuron_raw_test=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                    Location_raw_test=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                except:
                    print('Files not found for session '+str(ses_ind_test))
                    continue
                speed_raw_test=speed_dic[mouse_recday][ses_ind_test]
                acceleration_raw_test_=np.diff(speed_raw_test)/0.025
                acceleration_raw_test=np.hstack((acceleration_raw_test_[0],acceleration_raw_test_))

                phases_test=Phases_raw_dic[mouse_recday][ses_ind_test]
                phases_conc_test=concatenate_complex2(concatenate_complex2(phases_test))
                states_test=States_raw_dic[mouse_recday][ses_ind_test]
                states_conc_test=concatenate_complex2(concatenate_complex2(states_test))
                times_test=Times_from_reward_dic[mouse_recday][ses_ind_test]
                times_conc_test=concatenate_complex2(concatenate_complex2(times_test))
                #distances_test=Distances_from_reward_dic[mouse_recday][ses_ind_test]
                distances_test=np.load(Intermediate_object_folder_dropbox+'Distances_from_reward_'+                mouse_recday+'_'+str(ses_ind_test)+'.npy',allow_pickle=True)
                distances_conc_test=concatenate_complex2(concatenate_complex2(distances_test))

                speed_raw_eq_test=gaussian_filter1d(speed_raw_test[:len(phases_conc_test)],smooth_SDs)
                acceleration_raw_eq_test=gaussian_filter1d(acceleration_raw_test[:len(phases_conc_test)],smooth_SDs)
                Location_raw_eq_test=Location_raw_test[:len(phases_conc_test)]
                
                Neuron_raw_eq_test_all=Neuron_raw_test[:,:len(phases_conc_test)]
                
                if remove_edges==True:
                    Location_raw_eq_test[Location_raw_eq_test>num_nodes]=np.nan ### removing edges

                if len(phases_conc_test)>=len(speed_raw_eq_test):
                    print('Mismatch between speed and ephys - work around here but check')
                    phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
                    states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
                    times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
                    distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
                    Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]
                    Neuron_raw_eq_test_all=Neuron_raw_eq_test_all[:,:len(speed_raw_eq_test)]
                
                
                ###extracting test state  
                states_=np.arange(num_states)
                for test_state in np.arange(num_states):
                    
                    ind_ses_state=(ses_ind_test_ind*num_states)+test_state

                    ###completing training arrays
                    phases_conc_test_training=phases_conc_test[states_conc_test!=test_state]
                    Location_raw_eq_test_training=Location_raw_eq_test[states_conc_test!=test_state]
                    Neuron_raw_test_training=Neuron_raw_eq_test_all[:,states_conc_test!=test_state]

                    phases_conc_all=np.hstack((phases_conc_all_,phases_conc_test_training))
                    Location_raw_eq_all=np.hstack((Location_raw_eq_all_,Location_raw_eq_test_training))
                    Neuron_raw_all=np.hstack((Neuron_raw_all_,Neuron_raw_test_training))
                    
                    
                    Neuron_phases_all=np.zeros((num_neurons,num_phases))
                    #Neuron_states_all=np.zeros((num_neurons,num_states))
                    Neuron_locations_all=np.zeros((num_neurons,num_locations))
                    Neuron_phases_all[:]=np.nan
                    Neuron_locations_all[:]=np.nan
                    #Neuron_states_all[:]=np.nan
                    for neuron in np.arange(num_neurons):
                        Neuron_raw_eq_all=Neuron_raw_all[neuron,:len(phases_conc_all)]
                        Neuron_phases=st.binned_statistic(phases_conc_all, Neuron_raw_eq_all, bins=phase_bins)[0]
                        #Neuron_states=st.binned_statistic(states_conc_all, Neuron_raw_eq_all, bins=state_bins)[0]
                        Neuron_locations=st.binned_statistic(Location_raw_eq_all, Neuron_raw_eq_all, bins=location_bins)[0]
                        Neuron_phases_all[neuron]=Neuron_phases
                        #Neuron_states_all[neuron]=Neuron_states
                        Neuron_locations_all[neuron]=Neuron_locations
                    
                    
                    ###defining test arrays
                    Neuron_raw_test_test=Neuron_raw_eq_test_all[:,states_conc_test==test_state]
                    phases_conc_test_test=phases_conc_test[states_conc_test==test_state]
                    Location_raw_eq_test_test=Location_raw_eq_test[states_conc_test==test_state]
                    times_conc_test_test=times_conc_test[states_conc_test==test_state]
                    distances_conc_test_test=distances_conc_test[states_conc_test==test_state]
                    
                    speed_raw_eq_test_test=speed_raw_eq_test[states_conc_test==test_state]
                    acceleration_raw_eq_test_test=acceleration_raw_eq_test[states_conc_test==test_state]

                    Location_raw_eq_test_nonan=Location_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    speed_raw_eq_test_nonan=speed_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    acceleration_raw_eq_test_nonan=acceleration_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    times_conc_test_nonan=times_conc_test_test[~np.isnan(Location_raw_eq_test_test)]
                    distances_conc_test_nonan=distances_conc_test_test[~np.isnan(Location_raw_eq_test_test)]

                    coeffs_all_ses=np.zeros((num_neurons,num_regressors))
                    for neuron in np.arange(num_neurons):
                        Neuron_phases=Neuron_phases_all[neuron]
                        #Neuron_states=Neuron_states_all[neuron]
                        Neuron_locations=Neuron_locations_all[neuron]

                        Neuron_raw_eq_test=Neuron_raw_test_test[neuron]
                        Neuron_raw_eq_test_nonan=Neuron_raw_eq_test[~np.isnan(Location_raw_eq_test_test)]

                        FR_training_phases=Neuron_phases[phases_conc_test_test]
                        #FR_training_states=Neuron_states[states_conc_test_test]
                        FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test_test)]
                        #FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test_test)]
                        FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan-1).astype(int)]


                        ###regression
                        X = np.vstack((FR_training_phases_nonan,                                       FR_training_locations_nonan,                                       times_conc_test_nonan,                                       distances_conc_test_nonan,                                       speed_raw_eq_test_nonan,                                       acceleration_raw_eq_test_nonan)).T

                        X_clean=X[~np.isnan(X).any(axis=1)]
                        X_z=st.zscore(X_clean,axis=0)

                        y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
                        reg = LinearRegression().fit(X_z, y)
                        coeffs=reg.coef_
                        coeffs_all_ses[neuron]=coeffs
                        
                        

                        coeffs_all[neuron,ind_ses_state]=coeffs


            mean_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            p_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            mean_neuron_betas[:]=np.nan
            p_neuron_betas[:]=np.nan
            for neuron in np.arange(num_neurons):
                mean_phase_coeff=np.nanmean(coeffs_all[neuron,:,0])
                mean_place_coeff=np.nanmean(coeffs_all[neuron,:,1])

                p_phase_coeff=st.ttest_1samp(remove_nan(coeffs_all[neuron,:,0]),0)[1]
                p_place_coeff=st.ttest_1samp(remove_nan(coeffs_all[neuron,:,1]),0)[1]

                mean_neuron_betas[neuron]=mean_phase_coeff,mean_place_coeff
                p_neuron_betas[neuron]=p_phase_coeff,p_place_coeff

            GLM_dic2['coeffs_all'][mouse_recday]=coeffs_all
            GLM_dic2['mean_neuron_betas'][mouse_recday]=mean_neuron_betas
            GLM_dic2['p_neuron_betas'][mouse_recday]=p_neuron_betas


        except Exception as e:
            print('betas not calculated')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


# In[282]:


###GLM - shuffled - across tasks (using left out state/task combination as test data)
tt=time.time()
num_phases=5
num_nodes=9
num_locations=21
num_states=4
num_regressors=6 ##phase, place, time (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place
num_iterations=100
re_run=True
lag_min=30*40 ###1200 bins = 30 seconds
smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1
state_bins=np.arange(num_states+1)

remove_edges=True

if remove_edges==True:
    num_locations=num_nodes
    location_bins=np.arange(num_nodes+1)+1
        

for day_type in ['3_task','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        
        
        if mouse_recday not in ['ah07_01092023_02092023','ah07_01092023','ah07_02092023']:
            continue
        
        if re_run==False:
            if mouse_recday in GLM_dic2['percentile_neuron_betas'].keys():
                print('Already Analysed')
                continue

        try: 
            awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
            awake_sessions=session_dic['awake'][mouse_recday]

            #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
            #np.arange(len(awake_sessions_behaviour))])

            num_sessions=len(awake_sessions_behaviour)

            num_neurons=len(cluster_dic['good_clus'][mouse_recday])
            sessions=Task_num_dic[mouse_recday]
            num_refses=len(np.unique(sessions))
            num_comparisons=num_refses-1
            repeat_ses=np.where(rank_repeat(sessions)>0)[0]
            non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

            coeffs_all=np.zeros((num_neurons,len(non_repeat_ses)*num_states,num_iterations,num_regressors))
            coeffs_all[:]=np.nan
            
            
            
            for ses_ind_test_ind,ses_ind_test in enumerate(non_repeat_ses):
                print(ses_ind_test)
                

                training_sessions=np.setdiff1d(non_repeat_ses,ses_ind_test)
                
                ###Training
                phases_conc_all_=[]
                states_conc_all_=[]
                Location_raw_eq_all_=[]
                Neuron_raw_all_=[]
                for ses_ind_training_ind, ses_ind_training in enumerate(training_sessions):
                    try:
                        Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        speed_raw=speed_dic[mouse_recday][ses_ind_training]


                        acceleration_raw_=np.diff(speed_raw)/0.025
                        acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')

                    except:
                        print('Files not found (1) for session '+str(ses_ind_training))
                        continue

                    phases=Phases_raw_dic[mouse_recday][ses_ind_training]
                    phases_conc=concatenate_complex2(concatenate_complex2(phases))
                    states=States_raw_dic[mouse_recday][ses_ind_training]
                    states_conc=concatenate_complex2(concatenate_complex2(states))
                    times=Times_from_reward_dic[mouse_recday][ses_ind_training]
                    times_conc=concatenate_complex2(concatenate_complex2(times))
                    distances=Distances_from_reward_dic[mouse_recday][ses_ind_training]
                    distances_conc=concatenate_complex2(concatenate_complex2(distances))
                    speed_raw_eq=speed_raw[:len(phases_conc)]
                    acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
                    Location_raw_eq=Location_raw[:len(phases_conc)]

                    if remove_edges==True:
                        Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges

                    if len(phases_conc)>=len(speed_raw_eq):
                        print('Mismatch between speed and ephys - work around here but check')
                        phases_conc=phases_conc[:len(speed_raw_eq)]
                        states_conc=states_conc[:len(speed_raw_eq)]
                        times_conc=times_conc[:len(speed_raw_eq)]
                        distances_conc=distances_conc[:len(speed_raw_eq)]
                        Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]
                        Neuron_raw=Neuron_raw[:,:len(speed_raw_eq)]

                    speed_phases=st.binned_statistic(phases_conc, speed_raw_eq , bins=phase_bins)[0]
                    acceleration_phases=st.binned_statistic(phases_conc, acceleration_raw_eq , bins=phase_bins)[0]
                    
                    phases_conc_all_.append(phases_conc)
                    states_conc_all_.append(states_conc)
                    Location_raw_eq_all_.append(Location_raw_eq)
                    Neuron_raw_all_.append(Neuron_raw)
                    


                phases_conc_all_=np.hstack((phases_conc_all_))
                states_conc_all_=np.hstack((states_conc_all_))
                Location_raw_eq_all_=np.hstack((Location_raw_eq_all_))
                Neuron_raw_all_=np.hstack((Neuron_raw_all_))
                
                



                ###Test
                try:
                    Neuron_raw_test=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                    Location_raw_test=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                except:
                    print('Files not found (2) for session '+str(ses_ind_test))
                    continue
                speed_raw_test=speed_dic[mouse_recday][ses_ind_test]
                acceleration_raw_test_=np.diff(speed_raw_test)/0.025
                acceleration_raw_test=np.hstack((acceleration_raw_test_[0],acceleration_raw_test_))

                phases_test=Phases_raw_dic[mouse_recday][ses_ind_test]
                phases_conc_test=concatenate_complex2(concatenate_complex2(phases_test))
                states_test=States_raw_dic[mouse_recday][ses_ind_test]
                states_conc_test=concatenate_complex2(concatenate_complex2(states_test))
                times_test=Times_from_reward_dic[mouse_recday][ses_ind_test]
                times_conc_test=concatenate_complex2(concatenate_complex2(times_test))
                distances_test=Distances_from_reward_dic[mouse_recday][ses_ind_test]
                distances_conc_test=concatenate_complex2(concatenate_complex2(distances_test))

                speed_raw_eq_test=gaussian_filter1d(speed_raw_test[:len(phases_conc_test)],smooth_SDs)
                acceleration_raw_eq_test=gaussian_filter1d(acceleration_raw_test[:len(phases_conc_test)],smooth_SDs)
                Location_raw_eq_test=Location_raw_test[:len(phases_conc_test)]
                
                Neuron_raw_eq_test_all=Neuron_raw_test[:,:len(phases_conc_test)]
                
                if remove_edges==True:
                    Location_raw_eq_test[Location_raw_eq_test>num_nodes]=np.nan ### removing edges

                if len(phases_conc_test)>=len(speed_raw_eq_test):
                    print('Mismatch between speed and ephys - work around here but check')
                    phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
                    states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
                    times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
                    distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
                    Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]
                    Neuron_raw_eq_test_all=Neuron_raw_eq_test_all[:,:len(speed_raw_eq_test)]
                
                
                ###extracting test state  
                states_=np.arange(num_states)
                for test_state in np.arange(num_states):
                    
                    ind_ses_state=(ses_ind_test_ind*num_states)+test_state

                    ###completing training arrays
                    phases_conc_test_training=phases_conc_test[states_conc_test!=test_state]
                    Location_raw_eq_test_training=Location_raw_eq_test[states_conc_test!=test_state]
                    Neuron_raw_test_training=Neuron_raw_eq_test_all[:,states_conc_test!=test_state]

                    phases_conc_all=np.hstack((phases_conc_all_,phases_conc_test_training))
                    Location_raw_eq_all=np.hstack((Location_raw_eq_all_,Location_raw_eq_test_training))
                    Neuron_raw_all=np.hstack((Neuron_raw_all_,Neuron_raw_test_training))
                    
                    
                    Neuron_phases_all=np.zeros((num_neurons,num_phases))
                    #Neuron_states_all=np.zeros((num_neurons,num_states))
                    Neuron_locations_all=np.zeros((num_neurons,num_locations))
                    Neuron_phases_all[:]=np.nan
                    Neuron_locations_all[:]=np.nan
                    #Neuron_states_all[:]=np.nan
                    for neuron in np.arange(num_neurons):
                        Neuron_raw_eq_all=Neuron_raw_all[neuron,:len(phases_conc_all)]
                        Neuron_phases=st.binned_statistic(phases_conc_all, Neuron_raw_eq_all, bins=phase_bins)[0]
                        #Neuron_states=st.binned_statistic(states_conc_all, Neuron_raw_eq_all, bins=state_bins)[0]
                        Neuron_locations=st.binned_statistic(Location_raw_eq_all, Neuron_raw_eq_all, bins=location_bins)[0]
                        Neuron_phases_all[neuron]=Neuron_phases
                        #Neuron_states_all[neuron]=Neuron_states
                        Neuron_locations_all[neuron]=Neuron_locations
                    
                    
                    ###defining test arrays
                    Neuron_raw_test_test=Neuron_raw_eq_test_all[:,states_conc_test==test_state]
                    phases_conc_test_test=phases_conc_test[states_conc_test==test_state]
                    Location_raw_eq_test_test=Location_raw_eq_test[states_conc_test==test_state]
                    times_conc_test_test=times_conc_test[states_conc_test==test_state]
                    distances_conc_test_test=distances_conc_test[states_conc_test==test_state]
                    
                    speed_raw_eq_test_test=speed_raw_eq_test[states_conc_test==test_state]
                    acceleration_raw_eq_test_test=acceleration_raw_eq_test[states_conc_test==test_state]

                    Location_raw_eq_test_nonan=Location_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    speed_raw_eq_test_nonan=speed_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    acceleration_raw_eq_test_nonan=acceleration_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    times_conc_test_nonan=times_conc_test_test[~np.isnan(Location_raw_eq_test_test)]
                    distances_conc_test_nonan=distances_conc_test_test[~np.isnan(Location_raw_eq_test_test)]

                    for neuron in np.arange(num_neurons):
                        Neuron_phases=Neuron_phases_all[neuron]
                        #Neuron_states=Neuron_states_all[neuron]
                        Neuron_locations=Neuron_locations_all[neuron]

                        Neuron_raw_eq_test=Neuron_raw_test_test[neuron]
                        Neuron_raw_eq_test_nonan=Neuron_raw_eq_test[~np.isnan(Location_raw_eq_test_test)]

                        FR_training_phases=Neuron_phases[phases_conc_test_test]
                        #FR_training_states=Neuron_states[states_conc_test_test]
                        FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test_test)]
                        #FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test_test)]
                        FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan-1).astype(int)]


                        ###regression
                        X = np.vstack((FR_training_phases_nonan,                                       FR_training_locations_nonan,                                       times_conc_test_nonan,                                       distances_conc_test_nonan,                                       speed_raw_eq_test_nonan,                                       acceleration_raw_eq_test_nonan)).T

                        X_clean=X[~np.isnan(X).any(axis=1)]
                        X_z=st.zscore(X_clean,axis=0)

                        y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
                        reg = LinearRegression().fit(X_z, y)
                        coeffs=reg.coef_

                        if len(y)<lag_min:
                            continue
                        
                        ##Random shifts
                        max_roll=int(len(y)-lag_min)
                        min_roll=int(lag_min)
                        
                        if max_roll<min_roll:
                            continue
                        for iteration in range(num_iterations):
                            copy_y=np.copy(y)

                            shift=random.randrange(max_roll-min_roll)+min_roll
                            y_shifted=np.roll(copy_y,shift)

                            reg = LinearRegression().fit(X_z, y_shifted)
                            coeffs=reg.coef_
                            coeffs_all[neuron,ind_ses_state,iteration]=coeffs

            actual_mean_neuron_betas=GLM_dic2['mean_neuron_betas'][mouse_recday]
            
            thr_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            thr_neuron_betas[:]=np.nan

            percentile_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            percentile_neuron_betas[:]=np.nan
            for neuron in np.arange(num_neurons):
                dist_phase=np.asarray([np.nanmean(coeffs_all[neuron,:,ii,0]) for ii in range(num_iterations)])
                dist_place=np.asarray([np.nanmean(coeffs_all[neuron,:,ii,1]) for ii in range(num_iterations)])
                
                thr_phase_coeff=np.nanpercentile(dist_phase,95)
                thr_place_coeff=np.nanpercentile(dist_place,95)
                thr_neuron_betas[neuron]=thr_phase_coeff,thr_place_coeff
                
                percentile_phase_coeff=st.percentileofscore(dist_phase,actual_mean_neuron_betas[neuron,0])
                percentile_place_coeff=st.percentileofscore(dist_place,actual_mean_neuron_betas[neuron,1])
                
                percentile_neuron_betas[neuron]=percentile_phase_coeff,percentile_place_coeff

            GLM_dic2['thr_neuron_betas'][mouse_recday]=thr_neuron_betas            
            GLM_dic2['percentile_neuron_betas'][mouse_recday]=percentile_neuron_betas
            

        except Exception as e:
            print('betas not calculated')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


###GLM - across tasks/states (using left out task as test data)

GLM_dic3=rec_dd()
num_phases=5
num_nodes=9
num_locations=21
num_states=4
num_regressors=6 ##phase, place, time (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place

smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1
state_bins=np.arange(num_states+1)
redo=False
remove_edges=True

redo_list=['ab03_01092023']
specific_days=False

if remove_edges==True:
    num_locations=num_nodes
    location_bins=np.arange(num_nodes+1)+1
        

for day_type in ['3_task_all','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        
        if specific_days==True and mouse_recday not in redo_list:
            continue
        
        if redo==False:
            if mouse_recday in GLM_dic3['coeffs_all'].keys():
                print('Already Analysed')
                continue

        try: 
            awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
            awake_sessions=session_dic['awake'][mouse_recday]

            #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
            #np.arange(len(awake_sessions_behaviour))])

            num_sessions=len(awake_sessions_behaviour)

            num_neurons=len(cluster_dic['good_clus'][mouse_recday])
            sessions=Task_num_dic[mouse_recday]
            num_refses=len(np.unique(sessions))
            num_comparisons=num_refses-1
            repeat_ses=np.where(rank_repeat(sessions)>0)[0]
            non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

            coeffs_all=np.zeros((num_neurons,len(non_repeat_ses),num_regressors))
            coeffs_all[:]=np.nan
            
            
            
            for ses_ind_test_ind,ses_ind_test in enumerate(non_repeat_ses):
                print(ses_ind_test)
                training_sessions=np.setdiff1d(non_repeat_ses,ses_ind_test)
                
                ###Training
                phases_conc_all=[]
                states_conc_all=[]
                Location_raw_eq_all=[]
                Neuron_raw_all=[]
                for ses_ind_training_ind, ses_ind_training in enumerate(training_sessions):
                    try:
                        Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        speed_raw=speed_dic[mouse_recday][ses_ind_training]


                        acceleration_raw_=np.diff(speed_raw)/0.025
                        acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')

                    except:
                        print('Files not found for session '+str(ses_ind_training))
                        continue

                    phases=Phases_raw_dic[mouse_recday][ses_ind_training]
                    phases_conc=concatenate_complex2(concatenate_complex2(phases))
                    states=States_raw_dic[mouse_recday][ses_ind_training]
                    states_conc=concatenate_complex2(concatenate_complex2(states))
                    times=Times_from_reward_dic[mouse_recday][ses_ind_training]
                    times_conc=concatenate_complex2(concatenate_complex2(times))
                    #distances=Distances_from_reward_dic[mouse_recday][ses_ind_training]
                    distances=np.load(Intermediate_object_folder_dropbox+'Distances_from_reward_'+                    mouse_recday+'_'+str(ses_ind_training)+'.npy',allow_pickle=True)
                    distances_conc=concatenate_complex2(concatenate_complex2(distances))
                    speed_raw_eq=speed_raw[:len(phases_conc)]
                    acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
                    Location_raw_eq=Location_raw[:len(phases_conc)]

                    if remove_edges==True:
                        Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges

                    if len(phases_conc)>=len(speed_raw_eq):
                        print('Mismatch between speed and ephys - work around here but check')
                        phases_conc=phases_conc[:len(speed_raw_eq)]
                        states_conc=states_conc[:len(speed_raw_eq)]
                        times_conc=times_conc[:len(speed_raw_eq)]
                        distances_conc=distances_conc[:len(speed_raw_eq)]
                        Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]
                        Neuron_raw=Neuron_raw[:,:len(speed_raw_eq)]

                    speed_phases=st.binned_statistic(phases_conc, speed_raw_eq , bins=phase_bins)[0]
                    acceleration_phases=st.binned_statistic(phases_conc, acceleration_raw_eq , bins=phase_bins)[0]
                    
                    phases_conc_all.append(phases_conc)
                    states_conc_all.append(states_conc)
                    Location_raw_eq_all.append(Location_raw_eq)
                    Neuron_raw_all.append(Neuron_raw)
                    


                phases_conc_all=np.hstack((phases_conc_all))
                states_conc_all=np.hstack((states_conc_all))
                Location_raw_eq_all=np.hstack((Location_raw_eq_all))
                Neuron_raw_all=np.hstack((Neuron_raw_all))
                
                



                ###Test
                try:
                    Neuron_raw_test=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                    Location_raw_test=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                except:
                    print('Files not found for session '+str(ses_ind_test))
                    continue
                speed_raw_test=speed_dic[mouse_recday][ses_ind_test]
                acceleration_raw_test_=np.diff(speed_raw_test)/0.025
                acceleration_raw_test=np.hstack((acceleration_raw_test_[0],acceleration_raw_test_))

                phases_test=Phases_raw_dic[mouse_recday][ses_ind_test]
                phases_conc_test=concatenate_complex2(concatenate_complex2(phases_test))
                states_test=States_raw_dic[mouse_recday][ses_ind_test]
                states_conc_test=concatenate_complex2(concatenate_complex2(states_test))
                times_test=Times_from_reward_dic[mouse_recday][ses_ind_test]
                times_conc_test=concatenate_complex2(concatenate_complex2(times_test))
                #distances_test=Distances_from_reward_dic[mouse_recday][ses_ind_test]
                distances_test=np.load(Intermediate_object_folder_dropbox+'Distances_from_reward_'+                mouse_recday+'_'+str(ses_ind_test)+'.npy',allow_pickle=True)
                distances_conc_test=concatenate_complex2(concatenate_complex2(distances_test))

                speed_raw_eq_test=gaussian_filter1d(speed_raw_test[:len(phases_conc_test)],smooth_SDs)
                acceleration_raw_eq_test=gaussian_filter1d(acceleration_raw_test[:len(phases_conc_test)],smooth_SDs)
                Location_raw_eq_test=Location_raw_test[:len(phases_conc_test)]
                
                Neuron_raw_eq_test_all=Neuron_raw_test[:,:len(phases_conc_test)]
                
                if remove_edges==True:
                    Location_raw_eq_test[Location_raw_eq_test>num_nodes]=np.nan ### removing edges

                if len(phases_conc_test)>=len(speed_raw_eq_test):
                    print('Mismatch between speed and ephys - work around here but check')
                    phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
                    states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
                    times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
                    distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
                    Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]
                    Neuron_raw_eq_test_all=Neuron_raw_eq_test_all[:,:len(speed_raw_eq_test)]

                ###defining test arrays
                Neuron_raw_test_test=Neuron_raw_eq_test_all
                phases_conc_test_test=phases_conc_test
                Location_raw_eq_test_test=Location_raw_eq_test
                times_conc_test_test=times_conc_test
                distances_conc_test_test=distances_conc_test

                speed_raw_eq_test_test=speed_raw_eq_test
                acceleration_raw_eq_test_test=acceleration_raw_eq_test

                Location_raw_eq_test_nonan=Location_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                speed_raw_eq_test_nonan=speed_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                acceleration_raw_eq_test_nonan=acceleration_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                times_conc_test_nonan=times_conc_test_test[~np.isnan(Location_raw_eq_test_test)]
                distances_conc_test_nonan=distances_conc_test_test[~np.isnan(Location_raw_eq_test_test)]

                
                Neuron_phases_all=np.zeros((num_neurons,num_phases))
                #Neuron_states_all=np.zeros((num_neurons,num_states))
                Neuron_locations_all=np.zeros((num_neurons,num_locations))
                Neuron_phases_all[:]=np.nan
                Neuron_locations_all[:]=np.nan
                #Neuron_states_all[:]=np.nan
                for neuron in np.arange(num_neurons):
                    Neuron_raw_eq_all=Neuron_raw_all[neuron,:len(phases_conc_all)]
                    Neuron_phases=st.binned_statistic(phases_conc_all, Neuron_raw_eq_all, bins=phase_bins)[0]
                    #Neuron_states=st.binned_statistic(states_conc_all, Neuron_raw_eq_all, bins=state_bins)[0]
                    Neuron_locations=st.binned_statistic(Location_raw_eq_all, Neuron_raw_eq_all, bins=location_bins)[0]
                    Neuron_phases_all[neuron]=Neuron_phases
                    #Neuron_states_all[neuron]=Neuron_states
                    Neuron_locations_all[neuron]=Neuron_locations
                
                coeffs_all_ses=np.zeros((num_neurons,num_regressors))
                for neuron in np.arange(num_neurons):
                    Neuron_phases=Neuron_phases_all[neuron]
                    #Neuron_states=Neuron_states_all[neuron]
                    Neuron_locations=Neuron_locations_all[neuron]

                    Neuron_raw_eq_test=Neuron_raw_test_test[neuron]
                    Neuron_raw_eq_test_nonan=Neuron_raw_eq_test[~np.isnan(Location_raw_eq_test_test)]

                    FR_training_phases=Neuron_phases[phases_conc_test_test]
                    #FR_training_states=Neuron_states[states_conc_test_test]
                    FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test_test)]
                    #FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test_test)]
                    FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan-1).astype(int)]


                    ###regression
                    X = np.vstack((FR_training_phases_nonan,                                   FR_training_locations_nonan,                                   times_conc_test_nonan,                                   distances_conc_test_nonan,                                   speed_raw_eq_test_nonan,                                   acceleration_raw_eq_test_nonan)).T

                    X_clean=X[~np.isnan(X).any(axis=1)]
                    X_z=st.zscore(X_clean,axis=0)

                    y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
                    reg = LinearRegression().fit(X_z, y)
                    coeffs=reg.coef_
                    coeffs_all_ses[neuron]=coeffs



                    coeffs_all[neuron,ses_ind_test_ind]=coeffs


            mean_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            p_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            mean_neuron_betas[:]=np.nan
            p_neuron_betas[:]=np.nan
            for neuron in np.arange(num_neurons):
                mean_phase_coeff=np.nanmean(coeffs_all[neuron,:,0])
                mean_place_coeff=np.nanmean(coeffs_all[neuron,:,1])

                p_phase_coeff=st.ttest_1samp(remove_nan(coeffs_all[neuron,:,0]),0)[1]
                p_place_coeff=st.ttest_1samp(remove_nan(coeffs_all[neuron,:,1]),0)[1]

                mean_neuron_betas[neuron]=mean_phase_coeff,mean_place_coeff
                p_neuron_betas[neuron]=p_phase_coeff,p_place_coeff

            GLM_dic3['coeffs_all'][mouse_recday]=coeffs_all
            GLM_dic3['mean_neuron_betas'][mouse_recday]=mean_neuron_betas
            GLM_dic3['p_neuron_betas'][mouse_recday]=p_neuron_betas


        except Exception as e:
            print('betas not calculated')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
1-Are goal progress cells intact in AB task? GLM_dic2
2-Is goal progress tuning preserved across ABCD and AB? GLM_dic
3-Are state cells in AB task also goal progress cells? GLM_dic2
4-Are state cells in AB task truely AB periodic? - state cross-correlation of spatially tuned and
non-spatially tuned neurons

5-same as 1,3 and 4 above but for ABCDE?


'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


GLM_dic2.keys()


# In[ ]:


'''
mean beta, use mean shuffled and std shuffled to calculate percentile

'''


# In[23]:


###GLM - shuffled - across tasks (using left out task as test data)
tt=time.time()
num_phases=5
num_nodes=9
num_locations=21
num_states=4
num_regressors=6 ##phase, place, time (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place
num_iterations=100
re_run=False
lag_min=30*40 ###1200 bins = 30 seconds
smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1
state_bins=np.arange(num_states+1)

remove_edges=True

if remove_edges==True:
    num_locations=num_nodes
    location_bins=np.arange(num_nodes+1)+1
        

for day_type in ['3_task_all','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)

        
        if re_run==False:
            if mouse_recday in GLM_dic3['percentile_neuron_betas'].keys():
                print('Already Analysed')
                continue

        try: 
            awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
            awake_sessions=session_dic['awake'][mouse_recday]

            #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
            #np.arange(len(awake_sessions_behaviour))])

            num_sessions=len(awake_sessions_behaviour)

            num_neurons=len(cluster_dic['good_clus'][mouse_recday])
            sessions=Task_num_dic[mouse_recday]
            num_refses=len(np.unique(sessions))
            num_comparisons=num_refses-1
            repeat_ses=np.where(rank_repeat(sessions)>0)[0]
            non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

            coeffs_all=np.zeros((num_neurons,len(non_repeat_ses),num_iterations,num_regressors))
            coeffs_all[:]=np.nan
            
            
            
            for ses_ind_test_ind,ses_ind_test in enumerate(non_repeat_ses):
                print(ses_ind_test)
                

                training_sessions=np.setdiff1d(non_repeat_ses,ses_ind_test)
                
                ###Training
                phases_conc_all=[]
                states_conc_all=[]
                Location_raw_eq_all=[]
                Neuron_raw_all=[]
                for ses_ind_training_ind, ses_ind_training in enumerate(training_sessions):
                    try:
                        Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        speed_raw=speed_dic[mouse_recday][ses_ind_training]


                        acceleration_raw_=np.diff(speed_raw)/0.025
                        acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')

                    except:
                        print('Files not found (1) for session '+str(ses_ind_training))
                        continue

                    phases=Phases_raw_dic[mouse_recday][ses_ind_training]
                    phases_conc=concatenate_complex2(concatenate_complex2(phases))
                    states=States_raw_dic[mouse_recday][ses_ind_training]
                    states_conc=concatenate_complex2(concatenate_complex2(states))
                    times=Times_from_reward_dic[mouse_recday][ses_ind_training]
                    times_conc=concatenate_complex2(concatenate_complex2(times))
                    distances=Distances_from_reward_dic[mouse_recday][ses_ind_training]
                    distances_conc=concatenate_complex2(concatenate_complex2(distances))
                    speed_raw_eq=speed_raw[:len(phases_conc)]
                    acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
                    Location_raw_eq=Location_raw[:len(phases_conc)]

                    if remove_edges==True:
                        Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges

                    if len(phases_conc)>=len(speed_raw_eq):
                        print('Mismatch between speed and ephys - work around here but check')
                        phases_conc=phases_conc[:len(speed_raw_eq)]
                        states_conc=states_conc[:len(speed_raw_eq)]
                        times_conc=times_conc[:len(speed_raw_eq)]
                        distances_conc=distances_conc[:len(speed_raw_eq)]
                        Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]
                        Neuron_raw=Neuron_raw[:,:len(speed_raw_eq)]

                    speed_phases=st.binned_statistic(phases_conc, speed_raw_eq , bins=phase_bins)[0]
                    acceleration_phases=st.binned_statistic(phases_conc, acceleration_raw_eq , bins=phase_bins)[0]
                    
                    phases_conc_all.append(phases_conc)
                    states_conc_all.append(states_conc)
                    Location_raw_eq_all.append(Location_raw_eq)
                    Neuron_raw_all.append(Neuron_raw)
                    


                phases_conc_all=np.hstack((phases_conc_all))
                states_conc_all=np.hstack((states_conc_all))
                Location_raw_eq_all=np.hstack((Location_raw_eq_all))
                Neuron_raw_all=np.hstack((Neuron_raw_all))
                
                



                ###Test
                try:
                    Neuron_raw_test=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                    Location_raw_test=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                except:
                    print('Files not found (2) for session '+str(ses_ind_test))
                    continue
                speed_raw_test=speed_dic[mouse_recday][ses_ind_test]
                acceleration_raw_test_=np.diff(speed_raw_test)/0.025
                acceleration_raw_test=np.hstack((acceleration_raw_test_[0],acceleration_raw_test_))

                phases_test=Phases_raw_dic[mouse_recday][ses_ind_test]
                phases_conc_test=concatenate_complex2(concatenate_complex2(phases_test))
                states_test=States_raw_dic[mouse_recday][ses_ind_test]
                states_conc_test=concatenate_complex2(concatenate_complex2(states_test))
                times_test=Times_from_reward_dic[mouse_recday][ses_ind_test]
                times_conc_test=concatenate_complex2(concatenate_complex2(times_test))
                distances_test=Distances_from_reward_dic[mouse_recday][ses_ind_test]
                distances_conc_test=concatenate_complex2(concatenate_complex2(distances_test))

                speed_raw_eq_test=gaussian_filter1d(speed_raw_test[:len(phases_conc_test)],smooth_SDs)
                acceleration_raw_eq_test=gaussian_filter1d(acceleration_raw_test[:len(phases_conc_test)],smooth_SDs)
                Location_raw_eq_test=Location_raw_test[:len(phases_conc_test)]
                
                Neuron_raw_eq_test_all=Neuron_raw_test[:,:len(phases_conc_test)]
                
                if remove_edges==True:
                    Location_raw_eq_test[Location_raw_eq_test>num_nodes]=np.nan ### removing edges

                if len(phases_conc_test)>=len(speed_raw_eq_test):
                    print('Mismatch between speed and ephys - work around here but check')
                    phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
                    states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
                    times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
                    distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
                    Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]
                    Neuron_raw_eq_test_all=Neuron_raw_eq_test_all[:,:len(speed_raw_eq_test)]
                
                    
                Neuron_phases_all=np.zeros((num_neurons,num_phases))
                #Neuron_states_all=np.zeros((num_neurons,num_states))
                Neuron_locations_all=np.zeros((num_neurons,num_locations))
                Neuron_phases_all[:]=np.nan
                Neuron_locations_all[:]=np.nan
                #Neuron_states_all[:]=np.nan
                print(num_neurons)
                for neuron in np.arange(num_neurons):
                    Neuron_raw_eq_all=Neuron_raw_all[neuron,:len(phases_conc_all)]
                    Neuron_phases=st.binned_statistic(phases_conc_all, Neuron_raw_eq_all, bins=phase_bins)[0]
                    #Neuron_states=st.binned_statistic(states_conc_all, Neuron_raw_eq_all, bins=state_bins)[0]
                    Neuron_locations=st.binned_statistic(Location_raw_eq_all, Neuron_raw_eq_all, bins=location_bins)[0]
                    Neuron_phases_all[neuron]=Neuron_phases
                    #Neuron_states_all[neuron]=Neuron_states
                    Neuron_locations_all[neuron]=Neuron_locations
                    
                    
                    ###defining test arrays
                    Neuron_raw_test_test=Neuron_raw_eq_test_all
                    phases_conc_test_test=phases_conc_test
                    Location_raw_eq_test_test=Location_raw_eq_test
                    times_conc_test_test=times_conc_test
                    distances_conc_test_test=distances_conc_test
                    
                    speed_raw_eq_test_test=speed_raw_eq_test
                    acceleration_raw_eq_test_test=acceleration_raw_eq_test

                    Location_raw_eq_test_nonan=Location_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    speed_raw_eq_test_nonan=speed_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    acceleration_raw_eq_test_nonan=acceleration_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    times_conc_test_nonan=times_conc_test_test[~np.isnan(Location_raw_eq_test_test)]
                    distances_conc_test_nonan=distances_conc_test_test[~np.isnan(Location_raw_eq_test_test)]

                    #for neuron in np.arange(num_neurons):
                    #Neuron_phases=np.copy(Neuron_phases_all[neuron])
                    #Neuron_states=Neuron_states_all[neuron]
                    #Neuron_locations=Neuron_locations_all[neuron]

                    Neuron_raw_eq_test=Neuron_raw_test_test[neuron]
                    Neuron_raw_eq_test_nonan=Neuron_raw_eq_test[~np.isnan(Location_raw_eq_test_test)]

                    FR_training_phases=Neuron_phases[phases_conc_test_test]
                    #FR_training_states=Neuron_states[states_conc_test_test]
                    FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test_test)]
                    #FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test_test)]
                    FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan-1).astype(int)]


                    ###regression
                    X = np.vstack((FR_training_phases_nonan,                                   FR_training_locations_nonan,                                   times_conc_test_nonan,                                   distances_conc_test_nonan,                                   speed_raw_eq_test_nonan,                                   acceleration_raw_eq_test_nonan)).T

                    X_clean=X[~np.isnan(X).any(axis=1)]
                    X_z=st.zscore(X_clean,axis=0)

                    y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
                    reg = LinearRegression().fit(X_z, y)
                    coeffs=reg.coef_

                    if len(y)<lag_min:
                        continue

                    ##Random shifts
                    max_roll=int(len(y)-lag_min)
                    min_roll=int(lag_min)

                    if max_roll<min_roll:
                        continue
                    for iteration in range(num_iterations):
                        copy_y=np.copy(y)

                        shift=random.randrange(max_roll-min_roll)+min_roll
                        y_shifted=np.roll(copy_y,shift)

                        reg = LinearRegression().fit(X_z, y_shifted)
                        coeffs=reg.coef_
                        coeffs_all[neuron,ses_ind_test_ind,iteration]=coeffs

            actual_mean_neuron_betas=GLM_dic3['mean_neuron_betas'][mouse_recday]
            
            thr_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            thr_neuron_betas[:]=np.nan

            percentile_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            percentile_neuron_betas[:]=np.nan
            for neuron in np.arange(num_neurons):
                dist_phase=np.asarray([np.nanmean(coeffs_all[neuron,:,ii,0]) for ii in range(num_iterations)])
                dist_place=np.asarray([np.nanmean(coeffs_all[neuron,:,ii,1]) for ii in range(num_iterations)])
                
                thr_phase_coeff=np.nanpercentile(dist_phase,95)
                thr_place_coeff=np.nanpercentile(dist_place,95)
                thr_neuron_betas[neuron]=thr_phase_coeff,thr_place_coeff
                
                percentile_phase_coeff=st.percentileofscore(dist_phase,actual_mean_neuron_betas[neuron,0])
                percentile_place_coeff=st.percentileofscore(dist_place,actual_mean_neuron_betas[neuron,1])
                
                percentile_neuron_betas[neuron]=percentile_phase_coeff,percentile_place_coeff

            GLM_dic3['thr_neuron_betas'][mouse_recday]=thr_neuron_betas            
            GLM_dic3['percentile_neuron_betas'][mouse_recday]=percentile_neuron_betas
            

        except Exception as e:
            print('betas not calculated')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
print(time.time()-tt)


# In[463]:





# In[31]:


###Cell proportions using cross-task GLM
phase_bool_all=[]
#abstract_state_bool_all=[]
place_bool_all=[]
phase_place_bool_all=[]
for day_type in ['3_task','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:#list(GLM_dic['mean_neuron_betas'].keys()):
        print(mouse_recday)
        try:
            mean_neuron_betas=GLM_dic2['mean_neuron_betas'][mouse_recday]
            thr_neuron_beta=GLM_dic2['thr_neuron_betas'][mouse_recday]

            p_neuron_betas=GLM_dic2['p_neuron_betas'][mouse_recday]

            phase_bool=np.logical_and(mean_neuron_betas[:,0]>0,p_neuron_betas[:,0]<0.05)
            #state_bool=np.logical_and(mean_neuron_betas[:,1]>0,p_neuron_betas[:,1]<0.05)
            place_bool=np.logical_and(mean_neuron_betas[:,1]>0,p_neuron_betas[:,1]<0.05)
            #phase_place_bool=np.logical_and(mean_neuron_betas[:,2]>0,p_neuron_betas[:,2]<0.05)

            phase_bool=mean_neuron_betas[:,0]>thr_neuron_beta[:,0]
            #state_bool=mean_neuron_betas[:,1]>thr_neuron_beta[:,1]
            place_bool=mean_neuron_betas[:,1]>thr_neuron_beta[:,1]


            phase_and_place_bool=np.logical_and(phase_bool,place_bool)
            #phase_state_bool=np.logical_and(phase_bool,state_bool)


            print('Proportion Phase: ' + str(np.sum(phase_bool)/len(phase_bool)))
            #print('Proportion Abstract state: '+str(np.sum(state_bool)/len(state_bool)))
            print('Proportion Place: '+str(np.sum(place_bool)/len(place_bool)))
            #print('Proportion Phase/Place: '+str(np.sum(phase_place_bool)/len(phase_place_bool)))
            print('')
            print('Proportion Place of Phase: '+str(np.sum(phase_and_place_bool)/np.sum(phase_bool)))
            #print('Proportion Abstract state of Phase: '+str(np.sum(phase_state_bool)/np.sum(phase_bool)))
            print('')

            phase_bool_all.append(phase_bool)
            #abstract_state_bool_all.append(state_bool)
            place_bool_all.append(place_bool)
            #phase_place_bool_all.append(phase_place_bool)
        except Exception as e:
            print('Not calculated')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

phase_bool_all=np.concatenate(phase_bool_all)
#abstract_state_bool_all=np.concatenate(abstract_state_bool_all)
place_bool_all=np.concatenate(place_bool_all)
#phase_place_bool_all=np.concatenate(phase_place_bool_all)


# In[32]:


print('Phase cells: '+str(np.sum(phase_bool_all)/len(phase_bool_all)))
#print('Abstract state cells: '+str(np.sum(abstract_state_bool_all)/len(abstract_state_bool_all)))
print('Place cells: '+str(np.sum(place_bool_all)/len(place_bool_all)))
print('Phase/Place cells: '+str(np.sum(phase_place_bool_all)/len(phase_place_bool_all)))

#phase_abstract_state_bool=np.logical_and(phase_bool_all,abstract_state_bool_all)
#print('Abstract cells as proportion of Phase cells: '+str(np.sum(phase_abstract_state_bool)/np.sum(phase_bool_all)))

phase_place_bool=np.logical_and(phase_bool_all,place_bool_all)
print('Place cells as proportion of Phase cells: '+str(np.sum(phase_place_bool)/np.sum(phase_bool_all)))


# In[ ]:





# In[ ]:





# In[426]:


###Cell proportions using within-task GLM
phase_bool_all=[]
state_bool_all=[]
place_bool_all=[]
for day_type in ['3_task','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        try:
            print(mouse_recday)
            mean_neuron_betas=GLM_withinTask_dic['mean_neuron_betas'][mouse_recday]
            p_neuron_betas=GLM_withinTask_dic['p_neuron_betas'][mouse_recday]
            thr_neuron_beta=GLM_withinTask_dic['thr_neuron_betas'][mouse_recday]



            phase_bool=mean_neuron_betas[:,0]>thr_neuron_beta[:,0]
            state_bool=mean_neuron_betas[:,1]>thr_neuron_beta[:,1]
            place_bool=mean_neuron_betas[:,2]>thr_neuron_beta[:,2]

            #phase_bool=np.logical_and(mean_neuron_betas[:,0]>0,p_neuron_betas[:,0]<0.05)
            #state_bool=np.logical_and(mean_neuron_betas[:,1]>0,p_neuron_betas[:,1]<0.05)
            #place_bool=np.logical_and(mean_neuron_betas[:,2]>0,p_neuron_betas[:,2]<0.05)

            phase_place_bool=np.logical_and(phase_bool,place_bool)
            phase_state_bool=np.logical_and(phase_bool,state_bool)


            print('Proportion Phase: ' + str(np.sum(phase_bool)/len(phase_bool)))
            print('Proportion Abstract state: '+str(np.sum(state_bool)/len(state_bool)))
            print('Proportion Place: '+str(np.sum(place_bool)/len(place_bool)))
            print('')
            print('Proportion Place of Phase: '+str(np.sum(phase_place_bool)/np.sum(phase_bool)))
            print('Proportion Abstract state of Phase: '+str(np.sum(phase_state_bool)/np.sum(phase_bool)))
            print('')

            phase_bool_all.append(phase_bool)
            state_bool_all.append(state_bool)
            place_bool_all.append(place_bool)
        except Exception as e:
            #print(e)
            print('Not calculated')
            print('')
            

phase_bool_all=np.concatenate(phase_bool_all)
state_bool_all=np.concatenate(state_bool_all)
place_bool_all=np.concatenate(place_bool_all)


# In[26]:


print('phase cells: '+str(np.sum(phase_bool_all)/len(phase_bool_all)))
print('state cells: '+str(np.sum(state_bool_all)/len(state_bool_all)))
print('place cells: '+str(np.sum(place_bool_all)/len(place_bool_all)))

print('')

phase_state_bool=np.logical_and(phase_bool_all,state_bool_all)
print('state cells as proportion of phase cells: '+str(np.sum(phase_state_bool)/np.sum(phase_bool_all)))

phase_place_bool=np.logical_and(phase_bool_all,place_bool_all)
print('place cells as proportion of phase cells: '+str(np.sum(phase_place_bool)/np.sum(phase_bool_all)))

phase_state_unique=np.logical_and(phase_state_bool,~phase_place_bool)
phase_place_unique=np.logical_and(phase_place_bool,~phase_state_bool)
print('pure state cells as proportion of phase cells: '+str(np.sum(phase_state_unique)/np.sum(phase_bool_all)))
print('pure place cells as proportion of phase cells: '+str(np.sum(phase_place_unique)/np.sum(phase_bool_all)))

print('')

phase_place_or_state=np.logical_or(phase_place_bool,phase_state_bool)
print('place or state cells as proportion of phase cells: '+str(np.sum(phase_place_or_state)/np.sum(phase_bool_all)))


# In[ ]:





# In[283]:


###Tuning from single trials
tt=time.time()
#tuning_singletrial_dic2=rec_dd()
num_states=4
num_phases=3
for day_type in ['3_task', 'combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        #Importing Ephys
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])

        sessions=Task_num_dic[mouse_recday]
        num_refses=len(np.unique(sessions))
        num_comparisons=num_refses-1
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        num_nonrepeat_sessions=len(non_repeat_ses)
        
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]



        for session in np.arange(num_sessions):
            print(session)
            try:
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
                exec('ephys_ses_'+str(session)+'_=ephys_')

            except Exception as e:
                print(e)
                exec('ephys_ses_'+str(session)+'_=[]')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('Not calculated')


        
        tuning_z_matrix=np.zeros((num_sessions,num_neurons))
        tuning_p_matrix=np.zeros((num_sessions,num_neurons))
        tuning_z_matrix_allstates=np.zeros((num_sessions,num_neurons,num_states))
        tuning_p_matrix_allstates=np.zeros((num_sessions,num_neurons,num_states))
        tuning_z_matrix_allphases=np.zeros((num_sessions,num_neurons,num_phases))
        tuning_p_matrix_allphases=np.zeros((num_sessions,num_neurons,num_phases))
        for ses_ind, session in enumerate(np.arange(num_sessions)):
            exec('ephys_=ephys_ses_'+str(session)+'_')
            if len(ephys_)==0:
                continue
            for neuron in np.arange(num_neurons):
                

                if len(ephys_)==0:
                    tuning_z_matrix[ses_ind][neuron]=np.nan
                    tuning_p_matrix[ses_ind][neuron]=np.nan
                    continue

                ephys_neuron_unbinned_=ephys_[neuron]
                ephys_neuron_unbinned=np.asarray(np.split(ephys_neuron_unbinned_,num_states,axis=1))
                ephys_neuron=np.mean(np.split(ephys_neuron_unbinned,10,axis=2),axis=0)
                z_max=st.zscore(np.nanmax(ephys_neuron,axis=2),axis=0)
                z_max_prefstate=z_max[np.argmax(np.mean(z_max,axis=1))]
                tuning_z_matrix[ses_ind][neuron]=np.nanmean(z_max_prefstate)
                tuning_p_matrix[ses_ind][neuron]=st.ttest_1samp(remove_nan(z_max_prefstate),0)[1]

                tuning_z_matrix_allstates[ses_ind][neuron]=np.nanmean(z_max,axis=1)
                tuning_p_matrix_allstates[ses_ind][neuron]=np.asarray([st.ttest_1samp(remove_nan(z_max[ii]),0)[1]                                                            for ii in range(len(z_max))])


                ##Phase peaks
                ephys_neuron_3=np.asarray(np.split(ephys_neuron_unbinned,3,axis=2))
                max_phase=np.max(np.mean(np.mean(ephys_neuron_3,axis=1),axis=1),axis=1)
                z_max_phase=st.zscore(max_phase)

                #tuning_z_matrix_allphases[ses_ind][neuron]=np.nanmean(z_max_phase,axis=0)
                #tuning_p_matrix_allphases[ses_ind][neuron]=np.asarray([st.ttest_1samp(remove_nan(z_max_phase[:,ii]),0)\
                #                                                       [1] for ii in range(len(z_max_phase.T))])
                tuning_z_matrix_allphases[ses_ind][neuron]=z_max_phase
                
                ##replace ttests with permutation tests

        tuning_singletrial_dic2['tuning_z'][mouse_recday]=tuning_z_matrix
        tuning_singletrial_dic2['tuning_p'][mouse_recday]=tuning_p_matrix

        tuning_singletrial_dic2['tuning_z_allstates'][mouse_recday]=tuning_z_matrix_allstates
        tuning_singletrial_dic2['tuning_p_allstates'][mouse_recday]=tuning_p_matrix_allstates

        tuning_singletrial_dic2['tuning_z_allphases'][mouse_recday]=tuning_z_matrix_allphases
        #tuning_singletrial_dic2['tuning_p_allphases'][mouse_recday]=tuning_p_matrix_allphases

        

print(time.time()-tt)


# In[ ]:





# In[34]:


tuning_singletrial_dic2['tuning_z_allphases'].keys()


# In[ ]:





# In[52]:


mouse_recday='ah04_01122021_02122021'

print(mouse_recday)
#Importing Ephys
print('Importing Ephys')
num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
num_neurons=len(cluster_dic['good_clus'][mouse_recday])

sessions=Task_num_dic[mouse_recday]
num_refses=len(np.unique(sessions))
num_comparisons=num_refses-1
repeat_ses=np.where(rank_repeat(sessions)>0)[0]
non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
num_nonrepeat_sessions=len(non_repeat_ses)


for session in np.arange(num_sessions):
    print(session)
    try:
        ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
        exec('ephys_ses_'+str(session)+'_=ephys_')

    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('Not calculated')



tuning_z_matrix=np.zeros((num_sessions,num_neurons))
tuning_p_matrix=np.zeros((num_sessions,num_neurons))
tuning_z_matrix_allstates=np.zeros((num_sessions,num_neurons,num_states))
tuning_p_matrix_allstates=np.zeros((num_sessions,num_neurons,num_states))
tuning_z_matrix_allphases=np.zeros((num_sessions,num_neurons,num_phases))
tuning_p_matrix_allphases=np.zeros((num_sessions,num_neurons,num_phases))

ses_ind=session=0
neuron=45
exec('ephys_=ephys_ses_'+str(session)+'_')


ephys_neuron_unbinned_=ephys_[neuron]
ephys_neuron_unbinned=np.asarray(np.split(ephys_neuron_unbinned_,4,axis=1))


# In[53]:


plt.matshow(ephys_neuron_unbinned_)
for xx in [0,90,180,270]:
    plt.axvline(xx,color='white',ls='dashed')
print(tuning_singletrial_dic2['tuning_z_allphases'][mouse_recday][ses_ind][neuron])
tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][session][neuron]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[284]:


###Tuning booleans for states and phases
num_states=4 
num_phases=3 
 
p_thr=0.05 ###to account for occasional low num of trials 
for day_type in ['3_task','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday) 
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday]) 
        num_neurons=len(cluster_dic['good_clus'][mouse_recday]) 
        sessions=Task_num_dic[mouse_recday]
        num_refses=len(np.unique(sessions))
        num_comparisons=num_refses-1
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        num_nonrepeat_sessions=len(non_repeat_ses)

        peak_boolean_all_states=np.zeros((num_sessions,num_neurons,num_states)) 
        peak_boolean_all_phases=np.zeros((num_sessions,num_neurons,num_phases)) 
        peak_boolean_all_phases_max=np.zeros((num_sessions,num_neurons,num_phases)) 
        for ses_ind, session in enumerate(np.arange(num_sessions)): 
            print(ses_ind)
            
            z_ses_state=tuning_singletrial_dic2['tuning_z_allstates'][mouse_recday][ses_ind] 
            if len(z_ses_state)==0:
                print('No trials detected')
                continue
            
            p_ses_state=tuning_singletrial_dic2['tuning_p_allstates'][mouse_recday][ses_ind] 

            z_ses_phase=tuning_singletrial_dic2['tuning_z_allphases'][mouse_recday][ses_ind] 
            #p_ses_phase=tuning_singletrial_dic2['tuning_p_allphases'][mouse_recday][ses_ind] 


            for neuron in np.arange(num_neurons): 
                peak_boolean_all_states[ses_ind][neuron]=np.logical_and(z_ses_state[neuron]>0,                                                                        p_ses_state[neuron]<=p_thr) 
                #peak_boolean_all_phases[ses_ind][neuron]=np.logical_and(z_ses_phase[neuron]>0,\
                #                                                        p_ses_phase[neuron]<=p_thr) 
                peak_boolean_all_phases_max[ses_ind][neuron]=z_ses_phase[neuron]==np.max(z_ses_phase[neuron]) 

        tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday]=peak_boolean_all_states 
        #tuning_singletrial_dic2['tuning_phase_boolean'][mouse_recday]=peak_boolean_all_phases 
        tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday]=peak_boolean_all_phases_max 


# In[ ]:





# In[ ]:





# In[285]:


###making arrays for split double days 
day_typeX='3_task_all'
day_type='combined_ABCDonly'
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    try:
        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]
        rec_day1=rec_day.split('_',1)[0]
        rec_day2=rec_day.split('_',1)[1]
        Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
        Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

        mouse_recday1=mouse+'_'+rec_day1
        mouse_recday2=mouse+'_'+rec_day2

        num_neurons=len(cluster_dic['good_clus'][mouse_recday])

        for mouse_recdayX in [mouse_recday1,mouse_recday2]:
            if mouse_recdayX in day_type_dicX['3_task']:
                continue
                
            tuning_singletrial_dic2['tuning_state_boolean'][mouse_recdayX]=            tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday] 
            tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recdayX]=            tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday] 
            

            
    except Exception as e:
        print(e)

        


# In[ ]:





# In[ ]:





# In[286]:


###computing state tuning using per trial zscore
#Tuned_dic=rec_dd()
##paramaters
num_bins=90
num_states=4
num_phases=3
num_nodes=9
num_lags=12
smoothing_sigma=10
num_iterations=100
phase_norm_mean=np.tile(np.repeat(np.arange(num_phases),num_bins/num_phases),num_states)
phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states,num_bins))

for day_type in ['combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        
        print(mouse_recday)
        mouse=mouse_recday.split('_',1)[0]
        #if mouse not in ['ah07','ab03']:
        #    print('Already done')
        #    continue
        
        #if len(Tuned_dic['State_zmax'][mouse_recday])>0:
        #    print('Already done')
        #    continue
        try:
            awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
            awake_sessions=session_dic['awake'][mouse_recday]
            num_sessions=len(awake_sessions_behaviour)
            num_neurons=len(cluster_dic['good_clus'][mouse_recday])
            non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
            #regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]


            found_ses=[]
            for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
                try:
                    Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    found_ses.append(ses_ind)

                except:
                    print('Files not found for session '+str(ses_ind))
                    continue
            num_non_repeat_ses_found=len(found_ses)

            #num_state_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)\
            #          for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T

            zmax_all=np.zeros((num_neurons,num_non_repeat_ses_found))
            zmax_all[:]=np.nan

            zmax_all_strict=np.zeros((num_neurons,num_non_repeat_ses_found))
            zmax_all_strict[:]=np.nan

            corr_mean_max_all=np.zeros((num_neurons,num_non_repeat_ses_found,2))
            corr_mean_max_all[:]=np.nan
            try:
                for ses_ind_ind in np.arange(num_non_repeat_ses_found):
                    ses_ind_actual=found_ses[ses_ind_ind]

                    Actual_activity_ses_=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
                    Actual_activity_ses=Actual_activity_ses_.T
                    #GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]

                    phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][ses_ind_actual]
                    pref_phase_neurons=np.argmax(phase_peaks,axis=1)
                    phases=Phases_raw_dic2[mouse_recday][ses_ind_actual]
                    phases_conc=concatenate_complex2(concatenate_complex2(phases))

                    Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
                    Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25

                    for neuron in np.arange(num_neurons):
                        pref_phase=pref_phase_neurons[neuron]
                        Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]

                        Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                                smoothing=False,return_mean=False)

                        Actual_norm_means=np.vstack(([[np.nanmean(Actual_norm[trial,num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                                          for trial in np.arange(len(Actual_norm))]))
                        max_state=np.argmax(np.nanmean(Actual_norm_means,axis=0))
                        zactivity_prefstate=st.zscore(Actual_norm_means,axis=1)[:,max_state]
                        zmax_all[neuron,ses_ind_ind]=st.ttest_1samp(zactivity_prefstate,0)[1]

                        zmax_shifted=np.zeros(num_iterations)
                        zmax_shifted[:]=np.nan
                        for iteration in range(num_iterations):
                            shifts=np.random.randint(0,4,len(Actual_norm_means))
                            Actual_norm_means_shifted=indep_roll(Actual_norm_means,shifts)
                            max_state=np.argmax(np.nanmean(Actual_norm_means_shifted,axis=0))
                            zactivity_prefstate=st.zscore(Actual_norm_means_shifted,axis=1)[:,max_state]
                            zactivity_prefstate_mean=np.nanmean(zactivity_prefstate)
                            zmax_shifted[iteration]=zactivity_prefstate_mean
                        mean_zmax_shifted=np.nanmean(zmax_shifted)
                        zmax_all_strict[neuron,ses_ind_ind]=st.ttest_1samp(zactivity_prefstate,mean_zmax_shifted)[1]



                        Actual_norm_max=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                                smoothing=False,return_mean=False,take_max=True)

                        Actual_norm_max_means=np.vstack(([[np.nanmean(Actual_norm_max[trial,num_bins*ii:num_bins*(ii+1)]                            [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                                          for trial in np.arange(len(Actual_norm_max))]))

                        r_,p_=st.pearsonr(np.concatenate(Actual_norm_means),np.concatenate(Actual_norm_max_means))
                        corr_mean_max_all[neuron,ses_ind_ind]=[r_,p_]
            except:
                print('Session not analysed')
            Tuned_dic['State_zmax'][mouse_recday]=zmax_all
            Tuned_dic['State_zmax_strict'][mouse_recday]=zmax_all_strict
            Tuned_dic['corr_mean_max'][mouse_recday]=corr_mean_max_all
        except:
            print('Not found')


# In[100]:


np.shape(Tuned_dic['corr_mean_max'][mouse_recday])


# In[287]:


day_type='combined_ABCDonly'
for day_type in ['3_task','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        try:
            #num_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)\
            #                          for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean']\
            #                                                       [mouse_recday]))]))

            #state_bool=np.sum(num_peaks_all>0,axis=0)>(len(num_peaks_all)//2)

            State_zmax=Tuned_dic['State_zmax'][mouse_recday]

            state_bool_zmax=np.sum(State_zmax<0.05,axis=1)>(len(State_zmax.T)/3)
            state_bool_zmax_one=np.sum(State_zmax<0.05,axis=1)>1

            Tuned_dic['State_zmax_bool'][mouse_recday]=state_bool_zmax
            Tuned_dic['State_zmax_bool_one'][mouse_recday]=state_bool_zmax_one
            
            
            state_bool_zmax_strict=np.sum(State_zmax<0.01,axis=1)>(len(State_zmax.T)/3)
            state_bool_zmax_one_strict=np.sum(State_zmax<0.01,axis=1)>1
            
            Tuned_dic['State_zmax_bool_strict'][mouse_recday]=state_bool_zmax_strict
            Tuned_dic['State_zmax_bool_one_strict'][mouse_recday]=state_bool_zmax_one_strict
        except:
            print('Not found')


# In[288]:


day_type_dicX['3_task_all']


# In[289]:


GLM_dic2['percentile_neuron_betas'].keys()


# In[ ]:





# In[ ]:





# In[14]:


###Making spatial maps
Place_tuned_dic=rec_dd()
for day_type in ['3_task_all','combined_ABCDonly','combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        non_repeat_ses=non_repeat_ses_maker(mouse_recday)

        Neurons_raw_0=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_0.npy')

        FR_maps=np.zeros((len(Neurons_raw_0),len(non_repeat_ses),3,3))
        FR_maps[:]=np.nan
        for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
            try:
                Neurons_raw_=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Location_=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            except:
                print(str(ses_ind)+'Not found')
                continue

            Location_=Location_[:len(Neurons_raw_.T)]
            Neurons_raw_=Neurons_raw_[:,:len(Location_)]

            nodes=(Location_[np.logical_and(Location_<=9,~np.isnan(Location_))]-1).astype(int)

            for neuron in np.arange(len(Neurons_raw_)):
                neuron_=Neurons_raw_[neuron]
                neuron_nodes=neuron_[np.logical_and(Location_<=9,~np.isnan(Location_))]
                FR_vector=st.binned_statistic(nodes,neuron_nodes,bins=np.arange(10))[0]
                FR_map=np.zeros((3,3))
                FR_map[:]=np.nan

                for ind in np.arange(len(FR_vector)):
                    FR_map[Task_grid[ind,0],Task_grid[ind,1]]=FR_vector[ind]
                FR_maps[neuron,ses_ind_ind]=FR_map

        cross_ses_corr=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((FR_maps[neuron][ses_ind]))                                                            for ses_ind in np.arange(len(FR_maps[neuron]))]))))        for neuron in np.arange(len(FR_maps))]))

        min_crosscorr=np.min(cross_ses_corr,axis=1)
        corr_min_thr=min_crosscorr>0
        p_values=np.hstack(([st.ttest_1samp(cross_ses_corr[neuron],0)[1] for neuron in np.arange(len(cross_ses_corr))]))
        corr_p_bool=p_values<0.05

        Place_tuned_dic[mouse_recday]=corr_p_bool


# In[ ]:





# In[291]:


cross_ses_corr[0]


# In[ ]:





# In[ ]:





# In[15]:


###Making phase maps
Phase_tuned_dic=rec_dd()
num_phases=5 
for day_type in ['3_task_all','combined_ABCDonly','combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        non_repeat_ses=non_repeat_ses_maker(mouse_recday)

        Neurons_raw_0=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_0.npy')

        FR_vectors=np.zeros((len(Neurons_raw_0),len(non_repeat_ses),num_phases))
        FR_vectors[:]=np.nan
        for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
            try:
                Neurons_raw_=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                #Location_=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Phases_=np.hstack((np.hstack((Phases_raw_dic[mouse_recday][ses_ind]))))
            except:
                print(str(ses_ind)+' not analysed')
                continue
            Phases_=Phases_[:len(Neurons_raw_.T)]
            Neurons_raw_=Neurons_raw_[:,:len(Phases_)]


            for neuron in np.arange(len(Neurons_raw_)):
                neuron_=Neurons_raw_[neuron]
                FR_vector=st.binned_statistic(Phases_,neuron_,bins=np.arange(num_phases+1))[0]

                FR_vectors[neuron,ses_ind_ind]=FR_vector

        cross_ses_corr=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((FR_vectors[neuron][ses_ind]))                                                            for ses_ind in np.arange(len(FR_vectors[neuron]))]))))        for neuron in np.arange(len(FR_vectors))]))

        min_crosscorr=np.min(cross_ses_corr,axis=1)
        corr_min_thr=min_crosscorr>0
        p_values=np.hstack(([st.ttest_1samp(cross_ses_corr[neuron],0)[1] for neuron in np.arange(len(cross_ses_corr))]))
        corr_p_bool=p_values<0.05

        Phase_tuned_dic[mouse_recday]=corr_p_bool


# In[ ]:





# In[17]:


len(day_type_dicX['combined_ABCDonly'])


# In[145]:


np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)


# In[19]:


###Phase within vs between days
Phase_tuned_crossday_dic=rec_dd()
num_phases=5 
for day_type in ['combined_ABCDonly','combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        #try:
        
        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]
        rec_day1=rec_day.split('_',1)[0]
        rec_day2=rec_day.split('_',1)[1]
        Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
        Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

        mouse_recday1=mouse+'_'+rec_day1
        mouse_recday2=mouse+'_'+rec_day2
        
        Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)
        Tasks1=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday1+'.npy',allow_pickle=True)
        Tasks2=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday2+'.npy',allow_pickle=True)

        non_repeat_ses_all=non_repeat_ses_maker(mouse_recday)
        non_repeat_ses_day1=non_repeat_ses_maker(mouse_recday1)
        non_repeat_ses_day2=non_repeat_ses_maker(mouse_recday2)

        Neurons_raw_0=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_0.npy')

        for ses_ind_ind, ses_ind in enumerate(non_repeat_ses_all):
            try:
                Neurons_raw_=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Phases_=np.hstack((np.hstack((Phases_raw_dic[mouse_recday][ses_ind]))))
            except:
                non_repeat_ses_all=np.setdiff1d(non_repeat_ses_all,ses_ind)
                non_repeat_ses_day1=np.setdiff1d(non_repeat_ses_day1,ses_ind)
                non_repeat_ses_day2=np.setdiff1d(non_repeat_ses_day2,ses_ind-len(Tasks1))
                continue

        FR_vectors=np.zeros((len(Neurons_raw_0),len(non_repeat_ses_all),num_phases))
        FR_vectors[:]=np.nan
        for ses_ind_ind, ses_ind in enumerate(non_repeat_ses_all):
            try:
                Neurons_raw_=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                #Location_=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Phases_=np.hstack((np.hstack((Phases_raw_dic[mouse_recday][ses_ind]))))
            except:
                print(str(ses_ind)+' not analysed')
                continue
            Phases_=Phases_[:len(Neurons_raw_.T)]
            Neurons_raw_=Neurons_raw_[:,:len(Phases_)]


            for neuron in np.arange(len(Neurons_raw_)):
                neuron_=Neurons_raw_[neuron]
                FR_vector=st.binned_statistic(Phases_,neuron_,bins=np.arange(num_phases+1))[0]

                FR_vectors[neuron,ses_ind_ind]=FR_vector

                
        
        if len(non_repeat_ses_day1)>1 and len(non_repeat_ses_day2)>1:
            cross_ses_corr_within1=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((                    FR_vectors[neuron][ses_ind])) for ses_ind in np.arange(len(non_repeat_ses_day1))]))))            for neuron in np.arange(len(FR_vectors))]))


            cross_ses_corr_within2=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((                    FR_vectors[neuron][ses_ind])) for ses_ind in                    np.arange(len(non_repeat_ses_day2))+len(non_repeat_ses_day1)                        ])))) for neuron in np.arange(len(FR_vectors))]))

            mean_crosscorr_within1=np.nanmean(cross_ses_corr_within1,axis=1)
            mean_crosscorr_within2=np.nanmean(cross_ses_corr_within2,axis=1)

            mean_crosscorr_within=np.nanmean(np.vstack((mean_crosscorr_within1,mean_crosscorr_within2)),axis=0)
        
        elif len(non_repeat_ses_day2)==1:
            cross_ses_corr_within1=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((                    FR_vectors[neuron][ses_ind])) for ses_ind in np.arange(len(non_repeat_ses_day1))]))))            for neuron in np.arange(len(FR_vectors))]))
            mean_crosscorr_within=np.nanmean(cross_ses_corr_within1,axis=1)
            
        elif len(non_repeat_ses_day1)==1:
            cross_ses_corr_within2=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((                    FR_vectors[neuron][ses_ind])) for ses_ind in                    np.arange(len(non_repeat_ses_day2))+len(non_repeat_ses_day1)                        ])))) for neuron in np.arange(len(FR_vectors))]))
            mean_crosscorr_within=np.nanmean(cross_ses_corr_within2,axis=1)

        cross_ses_corr_across_days=[np.hstack(([[st.pearsonr(FR_vectors[neuron][ses_ind_day1],                                                             FR_vectors[neuron][ses_ind_day2])[0]            for ses_ind_day1 in np.arange(len(non_repeat_ses_day1))]            for ses_ind_day2 in np.arange(len(non_repeat_ses_day2))+len(non_repeat_ses_day1)]))            for neuron in np.arange(len(FR_vectors))]

        

        mean_crosscorr_across_days=np.nanmean(cross_ses_corr_across_days,axis=1)



        Phase_tuned_crossday_dic[mouse_recday]=np.vstack((mean_crosscorr_within,mean_crosscorr_across_days)).T
        #except Exception as e:
        #    print(e)


# In[20]:


within_between_day_phasecorr=np.vstack((dict_to_array(Phase_tuned_crossday_dic)))
Phase_tuned_crossday_dic[mouse_recday]

plot_scatter(within_between_day_phasecorr[:,0],within_between_day_phasecorr[:,1])
print(st.pearsonr(within_between_day_phasecorr[:,0],within_between_day_phasecorr[:,1]))

print(len(within_between_day_phasecorr))


# In[21]:


print(np.nanmean(within_between_day_phasecorr[:,0]))
print(st.sem(within_between_day_phasecorr[:,0]))
print(np.nanmean(within_between_day_phasecorr[:,1]))
print(st.sem(within_between_day_phasecorr[:,1]))
st.wilcoxon(within_between_day_phasecorr[:,0],within_between_day_phasecorr[:,1])


# In[18]:


'''
-which sessions are in what day?
-within day correlation
-across day correlation
-shuffles
-percentage of within day significant cells that are also significant across days

'''


# In[37]:


###Phase within vs between days - circular shifts

lag_min=30*40 ###1200 bins = 30 seconds
num_iterations=100

Phase_tuned_crossday_shifted_dic=rec_dd()
num_phases=5 
for day_type in ['combined_ABCDonly','combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        #try:
        
        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]
        rec_day1=rec_day.split('_',1)[0]
        rec_day2=rec_day.split('_',1)[1]
        Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
        Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

        mouse_recday1=mouse+'_'+rec_day1
        mouse_recday2=mouse+'_'+rec_day2


        Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)
        Tasks1=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday1+'.npy',allow_pickle=True)
        Tasks2=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday2+'.npy',allow_pickle=True)

        non_repeat_ses_all=non_repeat_ses_maker(mouse_recday)
        non_repeat_ses_day1=non_repeat_ses_maker(mouse_recday1)
        non_repeat_ses_day2=non_repeat_ses_maker(mouse_recday2)


        for ses_ind_ind, ses_ind in enumerate(non_repeat_ses_all):
            try:
                Neurons_raw_=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                #Location_=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Phases_=np.hstack((np.hstack((Phases_raw_dic[mouse_recday][ses_ind]))))
            except:
                non_repeat_ses_all=np.setdiff1d(non_repeat_ses_all,ses_ind)
                non_repeat_ses_day1=np.setdiff1d(non_repeat_ses_day1,ses_ind)
                non_repeat_ses_day2=np.setdiff1d(non_repeat_ses_day2,ses_ind-len(Tasks1))

                #print(str(ses_ind)+' not analysed')
                continue

        Neurons_raw_0=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_0.npy')

        mean_crosscorr_within_all=np.zeros((num_iterations,len(Neurons_raw_0)))
        mean_crosscorr_across_days_all=np.zeros((num_iterations,len(Neurons_raw_0)))
        mean_crosscorr_within_all[:]=np.nan
        mean_crosscorr_across_days_all[:]=np.nan
        for iteration in np.arange(num_iterations):
            FR_vectors=np.zeros((len(Neurons_raw_0),len(non_repeat_ses_all),num_phases))
            FR_vectors[:]=np.nan
            for ses_ind_ind, ses_ind in enumerate(non_repeat_ses_all):
                try:
                    Neurons_raw_=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    #Location_=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    Phases_=np.hstack((np.hstack((Phases_raw_dic[mouse_recday][ses_ind]))))
                except:
                    print(str(ses_ind)+' not analysed')
                    continue
                Phases_=Phases_[:len(Neurons_raw_.T)]
                Neurons_raw_=Neurons_raw_[:,:len(Phases_)]

                Phases_shifted=np.copy(Phases_)
                max_roll=int(len(Phases_shifted)-lag_min)
                min_roll=int(lag_min)

                if max_roll<min_roll:
                    continue
                #for iteration in range(num_iterations):
                Phases_copy=np.copy(Phases_)

                shift=random.randrange(max_roll-min_roll)+min_roll
                Phases_shifted=np.roll(Phases_copy,shift)


                for neuron in np.arange(len(Neurons_raw_)):
                    neuron_=Neurons_raw_[neuron]
                    FR_vector=st.binned_statistic(Phases_shifted,neuron_,bins=np.arange(num_phases+1))[0]

                    FR_vectors[neuron,ses_ind_ind]=FR_vector

            if len(non_repeat_ses_day1)>1 and len(non_repeat_ses_day2)>1:
                cross_ses_corr_within1=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((                        FR_vectors[neuron][ses_ind])) for ses_ind in np.arange(len(non_repeat_ses_day1))]))))                for neuron in np.arange(len(FR_vectors))]))


                cross_ses_corr_within2=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((                        FR_vectors[neuron][ses_ind])) for ses_ind in                        np.arange(len(non_repeat_ses_day2))+len(non_repeat_ses_day1)]))))                                                   for neuron in np.arange(len(FR_vectors))]))

                mean_crosscorr_within1=np.nanmean(cross_ses_corr_within1,axis=1)
                mean_crosscorr_within2=np.nanmean(cross_ses_corr_within2,axis=1)
                mean_crosscorr_within=np.nanmean(np.vstack((mean_crosscorr_within1,mean_crosscorr_within2)),axis=0)

            elif len(non_repeat_ses_day2)==1:
                cross_ses_corr_within1=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((                        FR_vectors[neuron][ses_ind])) for ses_ind in np.arange(len(non_repeat_ses_day1))]))))                for neuron in np.arange(len(FR_vectors))]))
                mean_crosscorr_within=np.nanmean(cross_ses_corr_within1,axis=1)

            elif len(non_repeat_ses_day1)==1:
                cross_ses_corr_within2=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((                        FR_vectors[neuron][ses_ind])) for ses_ind in                        np.arange(len(non_repeat_ses_day2))+len(non_repeat_ses_day1)                            ])))) for neuron in np.arange(len(FR_vectors))]))
                mean_crosscorr_within=np.nanmean(cross_ses_corr_within2,axis=1)

            cross_ses_corr_across_days=[np.hstack(([[st.pearsonr(FR_vectors[neuron][ses_ind_day1],                                                                 FR_vectors[neuron][ses_ind_day2])[0]                for ses_ind_day1 in np.arange(len(non_repeat_ses_day1))]                for ses_ind_day2 in np.arange(len(non_repeat_ses_day2))+len(non_repeat_ses_day1)]))                for neuron in np.arange(len(FR_vectors))]


            mean_crosscorr_across_days=np.nanmean(cross_ses_corr_across_days,axis=1)


            mean_crosscorr_within_all[iteration]=mean_crosscorr_within
            mean_crosscorr_across_days_all[iteration]=mean_crosscorr_across_days

            #print(np.nanmean(mean_crosscorr_within_all[0]))

        thr_within=np.percentile(mean_crosscorr_within_all,95,axis=0)
        thr_across=np.percentile(mean_crosscorr_across_days_all,95,axis=0)




        Phase_tuned_crossday_shifted_dic[mouse_recday]=thr_within,thr_across
        #except Exception as e:
        #    print(e)


# In[ ]:





# In[38]:


thr_within_between=np.vstack(([np.asarray(Phase_tuned_crossday_shifted_dic[mouse_recday]).Tfor mouse_recday in Phase_tuned_crossday_shifted_dic.keys()]))

significant_within=within_between_day_phasecorr[:,0]>thr_within_between[:,0]
significant_between=within_between_day_phasecorr[:,1]>thr_within_between[:,1]
significant_within_between=np.logical_and(significant_within,significant_between)


# In[40]:


prop_within_between=np.sum(significant_within_between)/np.sum(significant_within)
print(prop_within_between)
print(np.sum(significant_within))
print(two_proportions_test(np.sum(significant_within_between), np.sum(significant_within),                       int(np.sum(significant_within)*0.05), np.sum(significant_within)))


# In[ ]:





# In[307]:


###Plotting spatial (place) maps
mouse_recday='ab03_01092023_02092023'
non_repeat_ses=non_repeat_ses_maker(mouse_recday)

Neurons_raw_0=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_0.npy')

FR_maps=np.zeros((len(Neurons_raw_0),len(non_repeat_ses),3,3))
FR_maps[:]=np.nan
for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
    Neurons_raw_=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
    Location_=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')

    Location_=Location_[:len(Neurons_raw_.T)]
    Neurons_raw_=Neurons_raw_[:,:len(Location_)]

    nodes=(Location_[np.logical_and(Location_<=9,~np.isnan(Location_))]-1).astype(int)

    for neuron in np.arange(len(Neurons_raw_)):
        neuron_=Neurons_raw_[neuron]
        neuron_nodes=neuron_[np.logical_and(Location_<=9,~np.isnan(Location_))]
        FR_vector=st.binned_statistic(nodes,neuron_nodes,bins=np.arange(10))[0]
        FR_map=np.zeros((3,3))
        FR_map[:]=np.nan

        for ind in np.arange(len(FR_vector)):
            FR_map[Task_grid[ind,0],Task_grid[ind,1]]=FR_vector[ind]
        FR_maps[neuron,ses_ind_ind]=FR_map
        
Place_bool=Tuned_dic['Place'][mouse_recday]
Place_bool2=Place_tuned_dic[mouse_recday]

for neuron in np.arange(len(FR_maps)):
    print(neuron)
    print(Place_bool[neuron])
    print(Place_bool2[neuron])
    FR_maps_neuron=FR_maps[neuron]

    max_rate=np.nanmax(FR_maps_neuron)
    min_rate=np.nanmin(FR_maps_neuron)

    fig1, f1_axes = plt.subplots(figsize=(7.5, 7.5),ncols=len(FR_maps_neuron), nrows=1, constrained_layout=True)  
    for ses_ind in np.arange(len(FR_maps_neuron)):
        FR_map_neuron_ses=FR_maps_neuron[ses_ind]
        ax1=f1_axes[ses_ind]
        ax1.matshow(FR_map_neuron_ses, cmap='coolwarm')
        ax1.axis('off')
    plt.axis('off') 
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


'''
-use new state definition for figure 5 (3 and 6 if cbb)
-

'''


# In[260]:


Tuned_dic['State_zmax_bool_one'][mouse_recday]


# In[259]:


Tuned_dic['State_zmax_bool_one_strict'][mouse_recday]


# In[86]:


thr


# In[16]:


phase_bool_all=[]
state_bool_all=[]
place_bool_all=[]
phase_place_bool_all=[]

threshold_state_sessions=0

use_permuted=True ###for phase and place tuning
use_both=False ###for phase and place tuning - use both permutation and ttest
use_zscored=True ##for state tuning
lowest_thr=False

if use_both==True:
    use_permuted=False

Tuned_dic2=rec_dd()

GLM_dicX=GLM_dic2
thr='95'
for day_type in ['3_task_all','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:#list(GLM_dic['mean_neuron_betas'].keys()):
        print(mouse_recday)
        
        #if mouse_recday=='me10_20122021_21122021': ##fix
        #    continue
        
        try:
            mean_neuron_betas=GLM_dicX['mean_neuron_betas'][mouse_recday]
            p_neuron_betas=GLM_dicX['p_neuron_betas'][mouse_recday]
            thr_neuron_beta=GLM_dicX['thr_neuron_betas'][mouse_recday]
            percentile_neuron_beta=GLM_dicX['percentile_neuron_betas'][mouse_recday]
            
            mean_neuron_betas_withintask=GLM_withinTask_dic['mean_neuron_betas'][mouse_recday]
            thr_neuron_betas_withintask=GLM_withinTask_dic['thr_neuron_betas'][mouse_recday]
            
            
            
            phase_bool_ttest=np.logical_and(mean_neuron_betas[:,0]>0,p_neuron_betas[:,0]<0.05)
            place_bool_ttest=np.logical_and(mean_neuron_betas[:,1]>0,p_neuron_betas[:,1]<0.05)
            #phase_place_bool_ttest=np.logical_and(mean_neuron_betas[:,2]>0,p_neuron_betas[:,2]<0.05)
            
            
            
            
            phase_bool_permutation=mean_neuron_betas[:,0]>thr_neuron_beta[:,0]
            place_bool_permutation=mean_neuron_betas[:,1]>thr_neuron_beta[:,1]
            
            if use_permuted==True:
                phase_bool=phase_bool_permutation
                place_bool=place_bool_permutation
                #phase_place_bool=mean_neuron_betas[:,2]>thr_neuron_beta[:,2]
                try:
                    state_within_bool=mean_neuron_betas_withintask[:,1]>thr_neuron_betas_withintask[:,1]
                    Tuned_dic['State_withintask'][mouse_recday]=state_within_bool
                except:
                    print('within task state tuning not calculated')
            elif use_both==True:
                phase_bool=np.logical_and(phase_bool_permutation,phase_bool_ttest)
                place_bool=np.logical_and(place_bool_permutation,place_bool_ttest)
                
            else:
                phase_bool=phase_bool_ttest
                place_bool=place_bool_ttest
                #phase_place_bool=phase_place_bool_ttest
                
                

            num_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)                          for ses_ind in np.arange(len(tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday]))]))
            
            if use_zscored==True:
                if lowest_thr==True:
                    state_bool=Tuned_dic['State_zmax_bool_one'][mouse_recday]
                
                else:
                    state_bool=Tuned_dic['State_zmax_bool'][mouse_recday]
                
            else:
                if threshold_state_sessions=='half':
                    state_bool=np.sum(num_peaks_all>0,axis=0)>(len(num_peaks_all)//2)
                else:
                    state_bool=np.sum(num_peaks_all>0,axis=0)>threshold_state_sessions

            Tuned_dic['Phase'][mouse_recday]=phase_bool
            Tuned_dic['State'][mouse_recday]=state_bool
            Tuned_dic['Place'][mouse_recday]=place_bool
            #Tuned_dic['Phase_Place'][mouse_recday]=place_bool
            
            
            #Tuned_dic['Phase_ttest'][mouse_recday]=phase_bool_ttest
            #Tuned_dic['Place_ttest'][mouse_recday]=place_bool_ttest
            
            
            for thr in [95,99]:
                
                if thr==99:
                    phase_bool_ttest=np.logical_and(mean_neuron_betas[:,0]>0,p_neuron_betas[:,0]<0.01)
                    place_bool_ttest=np.logical_and(mean_neuron_betas[:,1]>0,p_neuron_betas[:,1]<0.01)
                    
                else:
                    phase_bool_ttest=np.logical_and(mean_neuron_betas[:,0]>0,p_neuron_betas[:,0]<0.05)
                    place_bool_ttest=np.logical_and(mean_neuron_betas[:,1]>0,p_neuron_betas[:,1]<0.05)
                
                if use_permuted==True:
                    phase_bool2=percentile_neuron_beta[:,0]>thr
                    place_bool2=percentile_neuron_beta[:,1]>thr
                    
                    
                        
                elif use_both==True:
                    phase_bool2=np.logical_and(percentile_neuron_beta[:,0]>thr, phase_bool_ttest)
                    place_bool2=np.logical_and(percentile_neuron_beta[:,1]>thr,place_bool_ttest)
                    
                else:
                    phase_bool2=phase_bool_ttest
                    place_bool2=place_bool_ttest
                    
                if thr ==99:
                    state_bool2=Tuned_dic['State_zmax_bool_strict'][mouse_recday]
                elif thr==95:
                    state_bool2=Tuned_dic['State_zmax_bool'][mouse_recday]

                Tuned_dic2['Phase'][str(thr)][mouse_recday]=phase_bool2
                Tuned_dic2['Place'][str(thr)][mouse_recday]=place_bool2
                Tuned_dic2['State'][str(thr)][mouse_recday]=state_bool2
            

            phase_and_place_bool=np.logical_and(phase_bool,place_bool)
            state_place_bool=np.logical_and(phase_bool,state_bool)


            print(np.sum(phase_bool)/len(phase_bool))
            print(np.sum(state_bool)/len(state_bool))
            print(np.sum(place_bool)/len(place_bool))
            #print(np.sum(phase_place_bool)/len(phase_place_bool))
            print('')
            #print(np.sum(phase_state_bool)/np.sum(phase_bool))
            print(np.sum(phase_and_place_bool)/np.sum(phase_bool))
            print('')

            phase_bool_all.append(phase_bool)
            state_bool_all.append(state_bool)
            place_bool_all.append(place_bool)
            #phase_place_bool_all.append(phase_place_bool)
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('Not calculated')
            print('')

phase_bool_all=np.concatenate(phase_bool_all)
state_bool_all=np.concatenate(state_bool_all)
place_bool_all=np.concatenate(place_bool_all)
#phase_place_bool_all=np.concatenate(phase_place_bool_all)


# In[62]:


len(day_type_dicX['combined_ABCDonly'])


# In[19]:


###Plotting neuron proportions across single day
use_both=False
for thr in [95,99]:
    print('')
    print(str(thr))
    percentile_neuron_bool_all=[]
    for mouse_recday in day_type_dicX['3_task_all']:
        
        mouse=mouse_recday.split('_',1)[0]
        cohort=Mice_cohort_dic[mouse]
        
        #if cohort!=6:
        #    continue
        
        phase_bool1=Tuned_dic2['Phase'][str(thr)][mouse_recday]
        
        if len(phase_bool1)==0:
            continue
        place_bool1=Tuned_dic2['Place'][str(thr)][mouse_recday]
        state_bool=Tuned_dic2['State'][str(thr)][mouse_recday]
        
        phase_bool2=Phase_tuned_dic[mouse_recday]
        place_bool2=Place_tuned_dic[mouse_recday]
        
        if use_both==True:
            phase_bool=np.logical_and(phase_bool1,phase_bool2)
            place_bool=np.logical_and(place_bool1,place_bool2)
            
            Tuned_dic2['Phase_strict'][str(thr)][mouse_recday]=phase_bool
            Tuned_dic2['Place_strict'][str(thr)][mouse_recday]=place_bool
            
            
        else:
            phase_bool=phase_bool1
            place_bool=place_bool1
            
        
            
        percentile_neuron_bool=np.column_stack((phase_bool,place_bool,state_bool))

        percentile_neuron_bool_all.append(percentile_neuron_bool)

    percentile_neuron_bool_all=np.vstack((percentile_neuron_bool_all))

    prop_phase=np.sum(percentile_neuron_bool_all[:,0])/len(percentile_neuron_bool_all[:,0])
    prop_place=np.sum(percentile_neuron_bool_all[:,1])/len(percentile_neuron_bool_all[:,1])
    prop_state=np.sum(percentile_neuron_bool_all[:,2])/len(percentile_neuron_bool_all[:,2])
    
    phase_state_bool=np.logical_and(percentile_neuron_bool_all[:,0],percentile_neuron_bool_all[:,2])
    phase_state_place_bool=np.logical_and(phase_state_bool,percentile_neuron_bool_all[:,1])
    prop_state_of_phase=np.sum(phase_state_bool)/np.sum(percentile_neuron_bool_all[:,0])
    prop_place_of_phasestate=np.sum(phase_state_place_bool)/np.sum(phase_state_bool)
    prop_phase_of_state=np.sum(phase_state_bool)/np.sum(percentile_neuron_bool_all[:,2])
    
    print('Number of neurons: '+str(len(percentile_neuron_bool_all)))
    print('Number of phase neurons: '+str(np.sum(percentile_neuron_bool_all[:,0])))
    print('Number of phase_state neurons: '+str(np.sum(phase_state_bool)))
    print('Number of state neurons: '+str(np.sum(percentile_neuron_bool_all[:,2])))
    
    print('Proportion phase: '+str(prop_phase))
    print(two_proportions_test(np.sum(percentile_neuron_bool_all[:,0]), len(percentile_neuron_bool_all[:,0]),                           int(len(percentile_neuron_bool_all[:,0])*0.05), len(percentile_neuron_bool_all[:,0])))
    print('Proportion place: '+str(prop_place))
    print(two_proportions_test(np.sum(percentile_neuron_bool_all[:,1]), len(percentile_neuron_bool_all[:,1]),                           int(len(percentile_neuron_bool_all[:,1])*0.05), len(percentile_neuron_bool_all[:,1])))
    print('Proportion state: '+str(prop_state))
    print(two_proportions_test(np.sum(percentile_neuron_bool_all[:,2]), len(percentile_neuron_bool_all[:,2]),                           int(len(percentile_neuron_bool_all[:,2])*0.05), len(percentile_neuron_bool_all[:,2])))
    
    print('Proportion state of phase: '+str(prop_state_of_phase))
    print(two_proportions_test(np.sum(phase_state_bool), np.sum(percentile_neuron_bool_all[:,0]),                           int(np.sum(percentile_neuron_bool_all[:,0]))*0.05, np.sum(percentile_neuron_bool_all[:,0])))
    print('Proportion place of phase-state: '+str(prop_place_of_phasestate))
    print(two_proportions_test(np.sum(phase_state_place_bool), np.sum(phase_state_bool),                           int(np.sum(phase_state_bool))*0.05, np.sum(phase_state_bool)))
    
    print('Proportion phase of state: '+str(prop_phase_of_state))
    print(two_proportions_test(np.sum(phase_state_bool), np.sum(percentile_neuron_bool_all[:,2]),                           int(np.sum(percentile_neuron_bool_all[:,2]))*0.05, np.sum(percentile_neuron_bool_all[:,2])))
    
    vals=[np.sum(percentile_neuron_bool_all[:,0]), len(percentile_neuron_bool_all)-np.sum(percentile_neuron_bool_all[:,0])]
    colors=['lightgreen','grey']
    plt.pie(vals, colors=colors)
    plt.savefig(Ephys_output_folder_dropbox+'All_phase_pie'+str(thr)+'.svg' , bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    vals=[np.sum(phase_state_bool), np.sum(percentile_neuron_bool_all[:,0])-np.sum(phase_state_bool)]
    colors=['lightblue','grey']
    plt.pie(vals, colors=colors)
    plt.savefig(Ephys_output_folder_dropbox+'Phase_state_pie'+str(thr)+'.svg' , bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    vals=[np.sum(phase_state_place_bool), np.sum(phase_state_bool)-np.sum(phase_state_place_bool)]
    colors=['orange','grey']
    plt.pie(vals, colors=colors)
    plt.savefig(Ephys_output_folder_dropbox+'State_place_pie'+str(thr)+'.svg' , bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print('_______')
    
    
    
    print('Number of state neurons: '+str(np.sum(percentile_neuron_bool_all[:,2])))
    vals=[np.sum(phase_state_bool), np.sum(percentile_neuron_bool_all[:,2])-np.sum(phase_state_bool)]
    colors=['lightblue','lightgreen']
    plt.pie(vals, colors=colors)
    plt.savefig(Ephys_output_folder_dropbox+'State_phase_pie'+str(thr)+'.svg' , bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print('_______')


# In[31]:


phase_bool1
Tuned_dic2['Phase'][str(thr)][mouse_recday]


# In[32]:


phase_bool,place_bool,state_bool


# In[ ]:





# In[295]:


###making arrays for split double days 
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    try:
        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]
        rec_day1=rec_day.split('_',1)[0]
        rec_day2=rec_day.split('_',1)[1]
        Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
        Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

        mouse_recday1=mouse+'_'+rec_day1
        mouse_recday2=mouse+'_'+rec_day2

        num_neurons=len(cluster_dic['good_clus'][mouse_recday])

        for mouse_recdayX in [mouse_recday1,mouse_recday2]:
            if mouse_recdayX in day_type_dicX['3_task']:
                continue
            
            
            for keyX in list(Tuned_dic.keys()):
                Tuned_dic[keyX][mouse_recdayX]=Tuned_dic[keyX][mouse_recday]
                
            for keyX in list(Tuned_dic2.keys()):
                for thr_X in Tuned_dic2[keyX].keys():
                    Tuned_dic2[keyX][thr_X][mouse_recdayX]=Tuned_dic2[keyX][thr_X][mouse_recday]
                
            for keyX in list(GLM_dic2.keys()):
                GLM_dic2[keyX][mouse_recdayX]=GLM_dic2[keyX][mouse_recday]
            
            for keyX in list(tuning_singletrial_dic.keys()):
                tuning_singletrial_dic[keyX][mouse_recdayX]=tuning_singletrial_dic[keyX][mouse_recday]
                

            
    except Exception as e:
        print(e)


# In[296]:


for day_type in ['3_task_all','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        Goal_progress_strict=Tuned_dic2['Phase_strict']['95'][mouse_recday]
        Place_strict=Tuned_dic2['Place_strict']['95'][mouse_recday]
        State_strict=Tuned_dic['State_zmax_bool_strict'][mouse_recday]
        
        
        
        Place=Tuned_dic2['Place']['95'][mouse_recday]
        Goal_progress=Tuned_dic2['Phase']['95'][mouse_recday]
        State=Tuned_dic['State_zmax_bool'][mouse_recday]
        
        for measure, arrayX in {'Place':Place,'Goal_progress':Goal_progress,'State':State,                'Place_strict':Place_strict,'Goal_progress_strict':Goal_progress_strict,                                'State_strict':State_strict}.items():
            np.save(Intermediate_object_folder_dropbox+measure+'_'+mouse_recday+'.npy',arrayX)
        
        
        


# In[297]:


mouse_recday='ab03_01092023_02092023'
np.sum(Tuned_dic2['Place']['95'][mouse_recday])


# In[298]:


xx=np.hstack(([Tuned_dic2['Place']['95'][mouse_recday]                            for mouse_recday in day_type_dicX['combined_ABCDonly']]))
np.sum(xx)/len(xx)


# In[ ]:





# In[299]:


print('phase cells: '+str(np.sum(phase_bool_all)/len(phase_bool_all)))
print('state cells: '+str(np.sum(state_bool_all)/len(state_bool_all)))
print('place cells: '+str(np.sum(place_bool_all)/len(place_bool_all)))

print('')

phase_state_bool=np.logical_and(phase_bool_all,state_bool_all)
print('state cells as proportion of phase cells: '+str(np.sum(phase_state_bool)/np.sum(phase_bool_all)))

phase_place_bool=np.logical_and(phase_bool_all,place_bool_all)
#print('place cells as proportion of phase cells: '+str(np.sum(phase_place_bool)/np.sum(phase_bool_all)))

phase_state_unique=np.logical_and(phase_state_bool,~phase_place_bool)
phase_place_unique=np.logical_and(phase_place_bool,~phase_state_bool)
#print('pure state cells as proportion of phase cells: '+str(np.sum(phase_state_unique)/np.sum(phase_bool_all)))
#print('pure place cells as proportion of phase cells: '+str(np.sum(phase_place_unique)/np.sum(phase_bool_all)))

print('')

#phase_place_or_state=np.logical_or(phase_place_bool,phase_state_bool)
#print('place or state cells as proportion of phase cells: '+str(np.sum(phase_place_or_state)/np.sum(phase_bool_all)))

phase_place_and_state=np.logical_and(phase_place_bool,phase_state_bool)
print('place cells as proportion of phase/state cells: '+str(np.sum(phase_place_and_state)/np.sum(phase_state_bool)))

print(len(phase_bool_all))


# In[ ]:





# In[300]:


###Plotting GLM results
import upsetplot
all_cells=np.arange(len(phase_bool_all))
phase_cells=np.where(phase_bool_all==True)[0]
state_cells=np.where(state_bool_all==True)[0]
place_cells=np.where(place_bool_all==True)[0]


non_phase_cells=np.where(phase_bool_all==False)[0]
non_state_cells=np.where(state_bool_all==False)[0]
non_place_cells=np.where(place_bool_all==False)[0]

nothing_bool_all=np.logical_and(np.logical_and(~phase_bool_all,~state_bool_all),~place_bool_all)
nothing_cells=np.where(nothing_bool_all==True)[0]

data=upsetplot.from_memberships([['Phase'],                                 ['Phase','State'],                                 ['Phase','Non_State'],                                 ['Phase','State','Non_Place'],                  ['Phase','State','Place'],[],['Non_Phase','State']],                 data=np.asarray([len(phase_cells),                                  len(np.intersect1d(state_cells,phase_cells)),                                  len(np.intersect1d(non_state_cells,phase_cells)),                                  len(np.intersect1d(np.intersect1d(state_cells,non_place_cells),phase_cells)),                                  len(np.intersect1d(np.intersect1d(state_cells,place_cells),phase_cells)),                                 len(nothing_cells),                                 len(np.intersect1d(state_cells,non_phase_cells))])/len(all_cells))
#print(data)

upsetplot.UpSet(data, sort_by='degree',sort_categories_by='cardinality')

upsetplot.plot(data)
plt.savefig(Ephys_output_folder_dropbox+'_UpsetPlot_cells.svg')
plt.show()


# In[301]:


from matplotlib import cm
fig, ax = plt.subplots()

size = 0.3
vals = np.array([[len(np.intersect1d(state_cells,phase_cells)),len(np.setdiff1d(phase_cells,state_cells))],                 [len(all_cells)-len(phase_cells),0]])

cmap = cmap = cm.get_cmap('tab20c', 20) #plt.colormaps["tab20c"]
outer_colors = cmap(np.arange(3)*4)
middle_colors = cmap([1, 2, 5])
inner_colors = cmap([3,8,9,10])

vals_outer=np.array([len(phase_cells),len(non_phase_cells)])
vals_middle=np.array([len(np.intersect1d(state_cells,phase_cells)),len(np.setdiff1d(phase_cells,state_cells)),                     len(non_phase_cells)])
vals_inner=np.array([len(np.intersect1d(np.intersect1d(state_cells,place_cells),phase_cells)),                     len(np.intersect1d(np.intersect1d(state_cells,non_place_cells),phase_cells)),                     len(np.setdiff1d(phase_cells,state_cells)),                     len(non_phase_cells)])

ax.pie(vals_outer, radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(vals_middle, radius=1-size, colors=middle_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(vals_inner, radius=1-(2*size), colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.set(aspect="equal")
plt.savefig(Ephys_output_folder_dropbox+'_PieChart_cells.svg')
plt.show()


# In[302]:


(1252*2)
day_type_dicX['3_taskonly']=np.setdiff1d(day_type_dicX['3_task'],                                         day_type_dicX['merged_only_3_task'])


# In[303]:


day_type_dicX['3_taskonly']=['ah04_26112021','ah04_30112021','me10_08122021', 'me10_16122021','me11_30112021']


# In[304]:


day_type_dicX['combined_ABCDonly']


# In[305]:


day_type_dicX['3_task_all']


# In[306]:


day_type_dicX.keys()#['combined_ABCDonly']


# In[307]:


1252+359+1252


# In[465]:


#phase_place_betas=np.vstack((remove_empty(dict_to_array(GLM_dic2['mean_neuron_betas']))))
day_type='combined_ABCDonly'
factor=40 ###because FR calculated in 25 ms bins 
phase_place_betas=np.vstack(([GLM_dic2['mean_neuron_betas'][mouse_recday] for mouse_recday                              in day_type_dicX[day_type]]))
phase_betas=phase_place_betas[:,0]
place_betas=phase_place_betas[:,1]
plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.hist(phase_betas*factor,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'Phase_betas.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(phase_betas))
print(st.ttest_1samp(phase_betas,0))
print('')

plt.hist(place_betas*factor,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'Place_betas.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(place_betas))
print(st.ttest_1samp(place_betas,0))
print('')

state_zvalues=np.hstack(([np.nanmean(Tuned_dic['State_zmax'][mouse_recday],axis=1)               for mouse_recday in list(Tuned_dic['State_zmax'].keys())                          if len(Tuned_dic['State_zmax'][mouse_recday])>0]))
plt.hist(state_zvalues,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed')
plt.savefig(Ephys_output_folder_dropbox+'State_zvalues.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(state_zvalues))
print(st.ttest_1samp(remove_nan(state_zvalues),0))
print('')


# In[ ]:





# In[466]:


day_type='combined_ABCDonly'
factor=40 ###because FR calculated in 25 ms bins 


print('Speed')
speed_betas=np.hstack(([np.nanmean(GLM_dic2['coeffs_all'][mouse_recday],axis=1)[:,4]for mouse_recday in day_type_dicX[day_type]]))

plt.hist(speed_betas*factor,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'Speed_betas.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(speed_betas))
print(st.ttest_1samp(speed_betas,0))
print('')

print('Acceleration')
acceleration_betas=np.hstack(([np.nanmean(GLM_dic2['coeffs_all'][mouse_recday],axis=1)[:,5]for mouse_recday in day_type_dicX[day_type]]))

plt.hist(acceleration_betas*factor,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'Acceleration_betas.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(acceleration_betas))
print(st.ttest_1samp(acceleration_betas,0))
print('')


# In[ ]:





# In[ ]:





# In[310]:


###making arrays for split double days 
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    try:
        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]
        rec_day1=rec_day.split('_',1)[0]
        rec_day2=rec_day.split('_',1)[1]
        Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
        Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

        mouse_recday1=mouse+'_'+rec_day1
        mouse_recday2=mouse+'_'+rec_day2

        num_neurons=len(cluster_dic['good_clus'][mouse_recday])

        for mouse_recdayX in [mouse_recday1,mouse_recday2]:
            if mouse_recdayX in day_type_dicX['3_task']:
                continue
            
            
            for keyX in list(Tuned_dic.keys()):
                Tuned_dic[keyX][mouse_recdayX]=Tuned_dic[keyX][mouse_recday]
                
            for keyX in list(Tuned_dic2.keys()):
                for thr_X in Tuned_dic2[keyX].keys():
                    Tuned_dic2[keyX][thr_X][mouse_recdayX]=Tuned_dic2[keyX][thr_X][mouse_recday]
                
            for keyX in list(GLM_dic2.keys()):
                GLM_dic2[keyX][mouse_recdayX]=GLM_dic2[keyX][mouse_recday]
            
            for keyX in list(tuning_singletrial_dic.keys()):
                tuning_singletrial_dic[keyX][mouse_recdayX]=tuning_singletrial_dic[keyX][mouse_recday]
                

            
    except Exception as e:
        print(e)


# In[ ]:





# In[ ]:





# In[311]:


all_cells=np.arange(len(phase_bool_all))
phase_cells=np.where(phase_bool_all==True)[0]
state_cells=np.where(state_bool_all==True)[0]
place_cells=np.where(place_bool_all==True)[0]


non_phase_cells=np.where(phase_bool_all==False)[0]
non_state_cells=np.where(state_bool_all==False)[0]
non_place_cells=np.where(place_bool_all==False)[0]


phase_state_cells=np.intersect1d(phase_cells,state_cells)
phase_state_place_cells=np.intersect1d(phase_state_cells,place_cells)
phase_state_nonplace_cells=np.intersect1d(phase_state_cells,non_place_cells)

cell_type_dic={'Phase':phase_cells,'State':state_cells,'Phase_state':phase_state_cells,               'Phase_state_place':phase_state_place_cells,                    'Phase_state_noplace':phase_state_nonplace_cells}

for name, arrayX in cell_type_dic.items():
    print(name)
    print(len(arrayX)/len(all_cells))
    print(two_proportions_test(len(arrayX), len(all_cells), len(all_cells)*0.05, len(all_cells)))
print('')


print('State cells as proportion of phase cells')
print(len(phase_cells))
print(len(phase_state_cells)/len(phase_cells))
print(two_proportions_test(len(phase_state_cells), len(phase_cells), len(phase_cells)*0.05, len(phase_cells)))

print('Place cells as proportion of phase-state cells')
print(len(phase_state_cells))
print(len(phase_state_place_cells)/len(phase_state_cells))
print(two_proportions_test(len(phase_state_place_cells), len(phase_state_cells),                           len(phase_state_cells)*0.05, len(phase_state_cells)))

print('Non place cells as proportion of phase-state cells')
print(len(phase_state_cells))
print(len(phase_state_nonplace_cells)/len(phase_state_cells))
print(two_proportions_test(len(phase_state_nonplace_cells), len(phase_state_cells),                           int(len(phase_state_cells)*0.05), len(phase_state_cells)))
 
print('Phase-state cells as proportion of state cells')
print(len(state_cells))
print(len(phase_state_cells)/len(state_cells))
print(two_proportions_test(len(phase_state_cells), len(state_cells),                           int(len(state_cells)*0.05), len(state_cells)))

vals=[len(phase_state_cells), len(state_cells)-len(phase_state_cells)]
colors=['blue','grey']
plt.pie(vals, colors=colors)
plt.savefig(Ephys_output_folder_dropbox+'state_phase_pie.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[312]:


print('Number of state cells')
print(len(state_cells))
#print(print(len(phase_state_cells)/len(state_cells)))
#print(two_proportions_test(len(phase_state_cells), len(state_cells),\
#                           int(len(state_cells)*0.05), len(state_cells)))


# In[404]:


Ephys_mice


# In[451]:


regions=['IL','PL','ACA','MOs','ORB','OLF','TTd','DP','STR','ccg','LSr']
shanks=['A','B','C','D','E','F'] ##for cambridge neurotech F 6 shank probe
Anatomy_channel_dic=rec_dd()
for mouse in Mice_cohort_dic.keys():
    cohort=Mice_cohort_dic[mouse]
    if cohort in [5,6]:
        Anatomy_channels_file_path=Intermediate_object_folder_dropbox+mouse+'_channelanatomy.csv'
        
        with open(Anatomy_channels_file_path, 'r') as f:
            Anatomy_channels = np.genfromtxt(f, delimiter=',',dtype=str, usecols=np.arange(0,7))
        Anatomy_channels_structure=Anatomy_channels[0]
        Anatomy_channels_structure_corrected=np.hstack((Anatomy_channels_structure[:2],                                                        np.repeat(Anatomy_channels_structure[2],3),        Anatomy_channels_structure[3:-1],np.repeat(Anatomy_channels_structure[-1],3)))
        for indx, variable_ in enumerate(Anatomy_channels_structure):
            variable=Anatomy_channels_structure_corrected[indx]
            Anatomy_channel_dic[mouse][variable]=Anatomy_channels[1:,indx]
    else:
        
        for shank in shanks:
            Anatomy_channels_file_path=Intermediate_object_folder_dropbox+mouse+'_'+shank+'.csv'

            with open(Anatomy_channels_file_path, 'r') as f:
                Anatomy_channels = np.genfromtxt(f, delimiter=',',dtype=str, usecols=np.arange(0,7))
            Anatomy_channels_structure=Anatomy_channels[0]
            Anatomy_channels_structure_corrected=np.hstack((Anatomy_channels_structure[:2],                                                            np.repeat(Anatomy_channels_structure[2],3),            Anatomy_channels_structure[3:-1],np.repeat(Anatomy_channels_structure[-1],3)))
            for indx, variable_ in enumerate(Anatomy_channels_structure):
                variable=Anatomy_channels_structure_corrected[indx]
                Anatomy_channel_dic[mouse][variable][shank]=Anatomy_channels[1:,indx]
        
for mouse in Mice_cohort_dic.keys():      
    cohort=Mice_cohort_dic[mouse]
    if cohort in [5,6]:
        acronym_=Anatomy_channel_dic[mouse]['acronym']
    else:
        acronym_=np.hstack((dict_to_array(Anatomy_channel_dic[mouse]['acronym'])))
    for region in regions:
        X_bool=np.hstack(([region in acronym_[channel_id] for channel_id in np.arange(len(acronym_))]))
        Anatomy_channel_dic[mouse][region+'_bool']=X_bool
    
    


# In[443]:


Anatomy_channel_dic['ah07']['acronym']


# In[ ]:





# In[452]:


Anatomy_neuron_dic=rec_dd()
for mouse_recday in day_type_dicX['3_task_all']:
    mouse=mouse_recday.split('_',1)[0]
    cohort=Mice_cohort_dic[mouse]

    
    good_clus=cluster_dic['good_clus'][mouse_recday]
    channel_num_=cluster_dic['channel_number'][mouse_recday]
    channel_num=channel_num_[:,1][np.isin(channel_num_[:,0], good_clus)]
    
    
    ###
    ####region bins
    bins_=[]
    for region_ind,region in enumerate(regions):
        channel_ids_region=np.where(Anatomy_channel_dic[mouse][region+'_bool']==True)[0]
        bins_.append(channel_ids_region)

    region_id_neuron=np.repeat(np.nan,len(channel_num))
    for ii in range(len(regions)):
        region_id_neuron[np.isin(channel_num,bins_[ii])]=ii
        
    num_neurons_region=[np.sum(region_id_neuron==region_ind) for region_ind in np.arange(len(regions))]

    Anatomy_neuron_dic['within_mFC'][mouse_recday]=num_neurons_region
    Anatomy_neuron_dic['outside_mFC'][mouse_recday]=len(channel_num)-np.sum(num_neurons_region)


# In[ ]:





# In[453]:


within_mFC_total=np.sum(dict_to_array(Anatomy_neuron_dic['within_mFC']),axis=0)
outside_mFC_total=np.sum(dict_to_array(Anatomy_neuron_dic['outside_mFC']))
within_outside_MFC_total=np.hstack((within_mFC_total,outside_mFC_total))


# In[458]:


percentages=(within_outside_MFC_total/np.sum(within_outside_MFC_total))*100
np.sum(percentages[:-1])
np.column_stack((regions,percentages[:-1]))


# In[456]:


regions


# In[455]:


percentages


# In[ ]:





# In[333]:


Tuning_anatomy_dic=rec_dd()
num_anatomy_bins=4
num_channels_neuropixels=384
bin_size=num_channels_neuropixels/num_anatomy_bins

anat_ratio_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    cohort=Mice_cohort_dic[mouse]
    
    if cohort not in [5,6]:
        continue
    good_clus=cluster_dic['good_clus'][mouse_recday]
    channel_num_=cluster_dic['channel_number'][mouse_recday]
    channel_num=channel_num_[:,1][np.isin(channel_num_[:,0], good_clus)]
    
    diff_to_max=num_channels_neuropixels-np.max(channel_num)
    
    if len(anat_ratio_dic[mouse])>0:
        if diff_to_max>anat_ratio_dic[mouse]:
            anat_ratio_dic[mouse]=[diff_to_max]
            
    else:
        anat_ratio_dic[mouse]=[diff_to_max]


for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    cohort=Mice_cohort_dic[mouse]
    
    if cohort not in [5,6]:
        continue
    #if mouse_recday=='ah07_01092023_02092023':
    #    continue
    print(mouse_recday)
    
    good_clus=cluster_dic['good_clus'][mouse_recday]
    channel_num_=cluster_dic['channel_number'][mouse_recday]
    channel_num=channel_num_[:,1][np.isin(channel_num_[:,0], good_clus)]
    
    diff_to_max=num_channels_neuropixels-np.max(channel_num)
    
    channel_num_corrected=channel_num+diff_to_max
    
    np.save(Intermediate_object_folder_dropbox+'_channel_num_neuron_'+mouse_recday+'.npy',channel_num_corrected)

    
    anatomy_bin_neuron=((channel_num_corrected-1)//bin_size).astype(int)
    
    
    
    ####region bins
    bins_=[]
    for region_ind,region in enumerate(regions):
        channel_ids_region=np.where(Anatomy_channel_dic[mouse][region+'_bool']==True)[0]
        bins_.append(channel_ids_region)

    region_id_neuron=np.repeat(np.nan,len(channel_num))
    for ii in range(len(regions)):
        region_id_neuron[np.isin(channel_num,bins_[ii])]=ii
    
    
    ##n.b. lower channel numbers=deeper channels

    phase_bool=Tuned_dic2['Phase_strict']['95'][mouse_recday]
    state_bool=Tuned_dic2['State']['95'][mouse_recday]
    place_bool=Tuned_dic2['Place_strict']['95'][mouse_recday]
    
    mean_phase_coeff=np.nanmean(GLM_dic2['coeffs_all'][mouse_recday][:,0],axis=1)
    mean_state_zmax=np.nanmean(Tuned_dic['State_zmax'][mouse_recday],axis=1)
    mean_place_coeff=np.nanmean(GLM_dic2['coeffs_all'][mouse_recday][:,1],axis=1)

    for measure,arrayX in {'Phase':phase_bool,'State':state_bool,'Place':place_bool}.items():
        measure_anat_bins=np.hstack(([np.sum(arrayX[anatomy_bin_neuron==anat_bin])/                                    len(arrayX[anatomy_bin_neuron==anat_bin])        if len(arrayX[anatomy_bin_neuron==anat_bin])>0 else np.nan for anat_bin in np.arange(num_anatomy_bins)]))
        
        
        measure_region_bins=np.hstack(([np.sum(arrayX[region_id_neuron==anat_bin])/                                    len(arrayX[region_id_neuron==anat_bin])        if len(arrayX[region_id_neuron==anat_bin])>0 else np.nan for anat_bin in np.arange(len(regions))]))

        Tuning_anatomy_dic['DV_bin'][measure][mouse_recday]=measure_anat_bins
        Tuning_anatomy_dic['region_id'][measure][mouse_recday]=measure_region_bins
        
    for measure,arrayX in {'Phase':mean_phase_coeff,'State':mean_state_zmax,'Place':mean_place_coeff}.items():
        measure_anat_bins=np.hstack(([np.nanmean(arrayX[anatomy_bin_neuron==anat_bin])/                                    len(arrayX[anatomy_bin_neuron==anat_bin])        if len(arrayX[anatomy_bin_neuron==anat_bin])>0 else np.nan for anat_bin in np.arange(num_anatomy_bins)]))
        
        
        measure_region_bins=np.hstack(([np.nanmean(arrayX[region_id_neuron==anat_bin])/                                    len(arrayX[region_id_neuron==anat_bin])        if len(arrayX[region_id_neuron==anat_bin])>0 else np.nan for anat_bin in np.arange(len(regions))]))

        Tuning_anatomy_dic['DV_bin'][measure+'_coeff'][mouse_recday]=measure_anat_bins
        Tuning_anatomy_dic['region_id'][measure+'_coeff'][mouse_recday]=measure_region_bins


# In[332]:





# In[ ]:





# In[316]:


#np.isin(channel_num_[:,0], good_clus)


# In[335]:


Tuning_anatomy_dic['region_id']['Phase_coeff']#[mouse_recday]


# In[318]:


Tuning_anatomy_dic['DV_bin'][measure]#[mouse_recday]


# In[323]:


np.where(Anatomy_channel_dic['ab03']['IL_bool']==True)[0]


# In[461]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
from scipy.stats import tukey_hsd

anatomy_type='region_id'
addition='_coeff'

for measure in ['Phase','State','Place']:
    print(measure)
    
    if measure =='State':
        factor=1
    else:
        factor=40

    Measure_prop_anat_mean=np.nanmean(dict_to_array(Tuning_anatomy_dic[anatomy_type][measure+addition]),axis=0)*factor
    Measure_prop_anat_sem=st.sem(dict_to_array(Tuning_anatomy_dic[anatomy_type][measure+addition])                                 ,nan_policy='omit',axis=0)*factor
    plt.rcParams["figure.figsize"] = (3,6)
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    
    if anatomy_type=='region_id':
        y=['M2','ACC','PrL','IrL']
    else:
        y=-np.arange(len(Measure_prop_anat_mean.T))
    plt.errorbar(y=y,x=np.flip(Measure_prop_anat_mean),                 xerr=np.flip(Measure_prop_anat_sem),                marker='o',markersize=10,color='black')
    plt.xlim(0,np.nanmax(Measure_prop_anat_mean)+np.nanmax(Measure_prop_anat_sem))
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    if anatomy_type=='region_id':
        plt.gca().invert_yaxis()
    plt.savefig(Ephys_output_folder_dropbox+'DV_vs_proportion_'+measure+'.svg' , bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    import statsmodels.api as sm


    Measure_prop_anat=dict_to_array(Tuning_anatomy_dic[anatomy_type][measure+addition])
    
    stats=st.f_oneway(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
    print(stats)
    
    if stats[1]<0.05:
        res = tukey_hsd(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
        print(res)


# In[345]:


remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3])


# values=np.hstack((Measure_prop_anat))
# bins_tiled=np.tile(np.arange(num_anatomy_bins), len(Measure_prop_anat))
# ids=np.repeat(np.arange(len(Measure_prop_anat)),num_anatomy_bins)
# 
# 
# bins_tiled=bins_tiled[~np.isnan(values)]
# ids=ids[~np.isnan(values)]
# values=values[~np.isnan(values)]
# 
# datax=pd.DataFrame({"id":ids, "depth_bin": bins_tiled, "value":values})
# 
# formula = 'value ~ C(depth_bin)'
# model = ols(formula, datax).fit()
# aov_table = anova_lm(model, typ=2)
# aov_table
# 
# aov_table = AnovaRM(datax, 'value', 'id', within=['depth_bin'], between=None,aggregate_func='mean')
# 
# st.f_oneway()

# In[227]:


xyz


# In[215]:


aov_results=aov_table.fit()
print(aov_results)

xy=column_stack_clean(bins_tiled, values)
print(st.linregress((xy[:,0], xy[:,1])))


# In[ ]:





# In[ ]:





# In[ ]:





# In[175]:





# In[ ]:


'''
1-chase up missing days
2-define neuropixels days (cohorts 5 and 6)
3-plot proportions of phase, state and place cells as a function of DV (8 bins)


'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[319]:


###defining sampling
frame_rate=60
tracking_oversampling_factor=50 ##oversampling of nodes_cut_dic
behaviour_oversampling_factor=3 ##oversampling of trialtimes_dic
behaviour_rate=1000


# In[331]:


###smoothed activity 
smoothed_activity_dic2=rec_dd()
Num_neurons_dic=rec_dd()
sigma=10
for day_type in ['3_task','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        
        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]
        All_sessions=session_dic_behaviour['All'][mouse_recday]
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]
        rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
        rec_day_structure_abstract=recday_numbers_dic['structure_abstract'][mouse_recday]
        structure_nums=np.unique(rec_day_structure_numbers)

        print(mouse_recday)
        #try:
            
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        num_trials_day=np.load(Intermediate_object_folder+'Num_trials_'+mouse_recday+'.npy')

        mean_activity_all=np.zeros((len(awake_sessions),num_neurons,360))
        sem_activity_all=np.zeros((len(awake_sessions),num_neurons,360))
        mean_activity_all[:]=np.nan
        sem_activity_all[:]=np.nan

        for awake_session_ind, timestamp in enumerate(awake_sessions):
            print(awake_session_ind)
            
            if num_trials_day[awake_session_ind]==0:
                print('no trials completed')
                continue
                
            try:
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
            except:
                print('Ephys file not found')
                continue

            for neuron in range(num_neurons):
                if len(ephys_)==0:
                    continue
                activity_neuron=ephys_[neuron]                *behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
                mean_activity=np.mean(activity_neuron,axis=0)
                mean_activity_smoothed=smooth_circular(mean_activity,sigma=sigma)

                sem_activity=st.sem(activity_neuron,axis=0)
                sem_activity_smoothed=smooth_circular(sem_activity,sigma=sigma)

                mean_activity_all[awake_session_ind,neuron]=mean_activity_smoothed
                sem_activity_all[awake_session_ind,neuron]=sem_activity_smoothed

        smoothed_activity_dic2['Mean'][mouse_recday]=mean_activity_all
        smoothed_activity_dic2['SEM'][mouse_recday]=sem_activity_all

        Num_neurons_dic[mouse_recday]=num_neurons

        if day_type=='combined_ABCDonly':
            rec_day1=rec_day.split('_',1)[0]
            rec_day2=rec_day.split('_',1)[1]
            Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
            Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

            mouse_recday1=mouse+'_'+rec_day1
            mouse_recday2=mouse+'_'+rec_day2

            days=[]
            for mouse_recdayX in [mouse_recday1,mouse_recday2]:
                num_awake_ses=len(session_dic_behaviour['awake'][mouse_recdayX])
                days.append(np.repeat(mouse_recdayX,num_awake_ses))
            days=np.hstack((days))
            for mouse_recdayX in [mouse_recday1,mouse_recday2]:
                if mouse_recdayX in day_type_dicX['3_task']:
                    continue
                mean_activity_day=mean_activity_all[days==mouse_recdayX]
                sem_activity_day=sem_activity_all[days==mouse_recdayX]

                smoothed_activity_dic2['Mean'][mouse_recdayX]=mean_activity_day
                smoothed_activity_dic2['SEM'][mouse_recdayX]=sem_activity_day

                Num_neurons_dic[mouse_recdayX]=num_neurons

                    
                    
        #except Exception as e:
        #    print(e)
            


# In[ ]:





# In[81]:


def arrange_plot_statecells_persessionX2(mouse_recday,neuron,Data_folder,sessions_included=None                                       ,fignamex=False,sigma=10,                                       save=False,plot=False,figtype='.svg', Marker=False,                                       fields_booleanx=[],measure_type='mean', abstract_structures=[],                                      repeated=False,behaviour_oversampling_factor=3,behaviour_rate=1000,                                       tracking_oversampling_factor=50):

    awake_sessions=session_dic['awake'][mouse_recday]
    awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
    
    colors=np.repeat('blue',len(awake_sessions_behaviour))
    plot_boolean=np.repeat(False,len(awake_sessions_behaviour))
    plot_boolean[sessions_included]=True
    
    
    
    

    fig= plt.figure(figsize=plt.figaspect(1)*4.5)
    fig.tight_layout()
    for awake_session_ind, timestamp in enumerate(awake_sessions_behaviour):
        structure_abstract=abstract_structures[awake_session_ind]
        
        num_trials_day=np.load(Intermediate_object_folder+'Num_trials_'+mouse_recday+'.npy')
        
        if num_trials_day[awake_session_ind]<2:
            print('Not enough trials session'+str(awake_session_ind))
            continue
        if timestamp not in awake_sessions:
            print('Ephys not used for session'+str(awake_session_ind))
            continue
            
            
        try:
            norm_activity_all=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
        except:
            print('No file found session'+str(awake_session_ind))
            continue
        
        norm_activity_neuron=norm_activity_all[neuron]
        
        xxx=np.asarray(norm_activity_neuron).T
        standardized_FR=np.hstack([np.nanmean(xxx[ii],axis=0) for ii in range(len(xxx))])*        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        standardized_FR_sem=np.hstack([st.sem(xxx[ii],axis=0,nan_policy='omit') for ii in range(len(xxx))])*        behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
        standardized_FR_smoothed=smooth_circular(standardized_FR,sigma=sigma)            
        standardized_FR_sem_smoothed=smooth_circular(standardized_FR_sem,sigma=sigma)                    

        
        standardized_FR_smoothed_upper=standardized_FR_smoothed+standardized_FR_sem_smoothed
        standardized_FR_smoothed_lower=standardized_FR_smoothed-standardized_FR_sem_smoothed
       
        
        color=colors[awake_session_ind]
        
        ax = fig.add_subplot(1, len(awake_sessions_behaviour), awake_session_ind+1, projection='polar')
        if len(fields_booleanx)>0:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,                              ax,color=color, Marker=Marker,fields_booleanx=fields_booleanx[awake_session_ind],                             structure_abstract=structure_abstract,repeated=repeated)
        else:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,                              ax,color=color, Marker=False,structure_abstract=structure_abstract,repeated=repeated)
    plt.margins(0,0)
    plt.tight_layout()
    if save==True:
        plt.savefig(fignamex+str(awake_session_ind)+figtype)
    if plot==True & plot_boolean[awake_session_ind]==True:
        plt.show()
    else:
        plt.close() 


# In[ ]:





# In[239]:


Intermediate_object_folder


# In[ ]:





# In[259]:


mouse_recday='ah07_01092023_02092023'
for awake_ses_ind in np.arange(7):
    binned_FR_ses=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(awake_ses_ind)+'.npy')

    print(np.shape(binned_FR_ses)[1]/40/60)
    
for All_session_ind in np.arange(18):
    name='binned_FR_dic_'+mouse_recday+'_'+str(All_session_ind)
    data_filename_memmap = os.path.join(Intermediate_object_folder, name)
    Activity=load(data_filename_memmap)
    #print(np.shape(Activity)[1]/40/60)


# In[254]:


ses_ind=5
xx=np.load(Intermediate_object_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')
xx[-1,-1]/1000/60

len(xx)


# In[255]:


Num_trials_dic2['ah07_01092023_02092023']


# In[260]:


##Examples - pre analysis
mouse_recday='ah07_01092023_02092023'

#for mouse_recday in day_type_dicX['3_task']:
print(mouse_recday)

all_neurons=np.arange(len(cluster_dic['good_clus'][mouse_recday]))
abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

for neuron in all_neurons[10:20]:
    print(neuron)

    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]

    All_sessions=session_dic['All'][mouse_recday]    
    awake_sessions=session_dic['awake'][mouse_recday]
    rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
    rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
    structure_nums=np.unique(rec_day_structure_numbers)

    fignamex=Ephys_output_folder+'/Example_cells/'+mouse_recday+'_neuron_id'+str(neuron)+'_task'

    arrange_plot_statecells_persessionX2(mouse_recday,neuron,                                          Data_folder=Intermediate_object_folder_dropbox,                                         abstract_structures=abstract_structures,                                        plot=True, save=False, fignamex=fignamex, figtype='.svg',Marker=False)


# In[20]:


###making arrays for split double days - cluster dic
##remove if working on Ephys_base
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]
    rec_day1=rec_day.split('_',1)[0]
    rec_day2=rec_day.split('_',1)[1]
    Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
    Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

    mouse_recday1=mouse+'_'+rec_day1
    mouse_recday2=mouse+'_'+rec_day2

    for mouse_recdayX in [mouse_recday1,mouse_recday2]:
        if mouse_recdayX in day_type_dicX['3_task']:
            continue        
        dic_keys=list(cluster_dic.keys())
        for keyX in dic_keys: 
            
            cluster_dic[keyX][mouse_recdayX]=            cluster_dic[keyX][mouse_recday]


# ###making arrays for split double days 
# for mouse_recday in day_type_dicX['combined_ABCDonly']:
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
#     days=Combined_days_dic['awake'][mouse_recday]
#     for mouse_recdayX in [mouse_recday1,mouse_recday2]:
#         print(mouse_recdayX)
#         if mouse_recdayX in day_type_dicX['3_task']:
#             continue
#         awake_ses_day=np.where(days==mouse_recdayX)[0]
#         for awake_session_ind_ind, awake_session_ind in enumerate(awake_ses_day): 
#             
#             try:
#             
#                 binned_FR_ses=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
#                 trial_times_state_simple_aligned=np.load(Intermediate_object_folder+'trialtimes_'+mouse_recday+'_'\
#                                                          +str(awake_session_ind)+'.npy')
#                 pokes_all_aligned=np.load(Intermediate_object_folder+'pokes_'+mouse_recday+'_'\
#                                           +str(awake_session_ind)+'.npy',allow_pickle=True)
#                 occupancy_flat_downsampled=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+\
#                                                    str(awake_session_ind)+'.npy')
#                 whl_day_ses_downsampled_aligned=np.load(Intermediate_object_folder+'XY_raw_'+mouse_recday+'_'\
#                                                         +str(awake_session_ind)+'.npy')
#                 scores_ses=np.load(Intermediate_object_folder+'Scores_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
#                 tasks_ses=np.load(Intermediate_object_folder+'Tasks_'+mouse_recday+'_'+str(awake_session_ind)+'.npy'\
#                                  ,allow_pickle=True)
#             except:
#                 print('Files not found for session '+str(awake_session_ind))
#                 continue
#             
#             
#             #######Saving
#             np.save(Intermediate_object_folder+'Neuron_raw_'+mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',\
#                         binned_FR_ses)
#             np.save(Intermediate_object_folder+'trialtimes_'+mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',\
#                          trial_times_state_simple_aligned)
#             np.save(Intermediate_object_folder+'pokes_'+mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',\
#                      pokes_all_aligned)
#             np.save(Intermediate_object_folder+'Location_raw_'+mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',\
#                      occupancy_flat_downsampled)
#             np.save(Intermediate_object_folder+'XY_raw_'+mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',\
#                      whl_day_ses_downsampled_aligned)            
#             np.save(Intermediate_object_folder+'Scores_'+mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',\
#                     scores_ses)
#             np.save(Intermediate_object_folder+'Tasks_'+mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',\
#                     tasks_ses)
#             
#             
#             try:
#                 ephys_array=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
#                 occupancy_equal=np.load(Intermediate_object_folder+'Location_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
#                 
#                 np.save(Intermediate_object_folder+'Neuron_'+mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',\
#                 ephys_array)
#                 np.save(Intermediate_object_folder+'Location_'+mouse_recdayX+'_'+str(awake_session_ind_ind)+'.npy',\
#                         occupancy_equal)
#             
#             except Exception as e:
#                 print(e)
#                 print('Normalised ephys Not saved')
#             

# In[ ]:





# In[35]:


mouse_recday='ah04_01122021_02122021'
Num_trials_dic2[mouse_recday]


# In[ ]:





# In[ ]:





# In[ ]:


##################
###AB and ABCDE###
##################


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


##Examples - pre analysis
import warnings; warnings.simplefilter('ignore')

mouse_recday='ah04_22122021_23122021'

#for mouse_recday in day_type_dicX['3_task']:
print(mouse_recday)

all_neurons=np.arange(len(cluster_dic['good_clus'][mouse_recday]))
abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

print(abstract_structures)
for neuron in np.arange(10):
    print(neuron)

    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]

    All_sessions=session_dic['All'][mouse_recday]    
    awake_sessions=session_dic['awake'][mouse_recday]
    rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
    rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
    structure_nums=np.unique(rec_day_structure_numbers)

    fignamex=Ephys_output_folder+'/Example_cells/'+mouse_recday+'_neuron_id'+str(neuron)+'_task'

    arrange_plot_statecells_persessionX2(mouse_recday,neuron,                                          Data_folder=Intermediate_object_folder_dropbox,                                         abstract_structures=abstract_structures,                                        plot=True, save=False, fignamex=fignamex, figtype='.svg',Marker=False)


# In[ ]:





# In[ ]:





# In[ ]:


'''
1-Is goal progress tuning preserved across ABCD and AB? GLM_dic - coded but fix missing days
2-Are state cells in AB task also goal progress cells? GLM_dic
3-Are state cells in AB task truely AB periodic? - state cross-correlation of spatially tuned and
non-spatially tuned neurons

4-same as above but for ABCDE?

potential extra qs (tbh not needed as 1 already identifies robust goal progress neurons)
5-Are goal progress cells intact in AB task? GLM_dic2  


'''


# In[ ]:





# In[ ]:





# In[139]:


Data_folder


# In[17]:


###Tuning from single trials ABCD AB
tt=time.time()
tuning_singletrial_ABCDAB_dic2=rec_dd()
#num_states=4
num_phases=3
for day_type in ['combined_ABCD_AB']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        
        #Importing Ephys
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])

        sessions=Task_num_dic[mouse_recday]
        num_refses=len(np.unique(sessions))
        num_comparisons=num_refses-1
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        num_nonrepeat_sessions=len(non_repeat_ses)
        
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]



        for session in np.arange(num_sessions):
            print(session)
            try:
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
                exec('ephys_ses_'+str(session)+'_=ephys_')

            except Exception as e:
                print(e)
                exec('ephys_ses_'+str(session)+'_=[]')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('Not calculated')

        ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
        AB_sessions=np.where(abstract_structures=='AB')[0]
        for abstract_structure, ses_array in {'ABCD':ABCD_sessions,'AB':AB_sessions}.items():
        
            num_states=len(abstract_structure)

            tuning_z_matrix=np.zeros((len(ses_array),num_neurons))
            tuning_p_matrix=np.zeros((len(ses_array),num_neurons))
            tuning_z_matrix_allstates=np.zeros((len(ses_array),num_neurons,num_states))
            tuning_p_matrix_allstates=np.zeros((len(ses_array),num_neurons,num_states))
            tuning_z_matrix_allphases=np.zeros((len(ses_array),num_neurons,num_phases))
            tuning_p_matrix_allphases=np.zeros((len(ses_array),num_neurons,num_phases))
            for ses_ind, session in enumerate(ses_array):
                exec('ephys_=ephys_ses_'+str(session)+'_')
                if len(ephys_)==0:
                    continue
                for neuron in np.arange(num_neurons):


                    if len(ephys_)==0:
                        tuning_z_matrix[ses_ind][neuron]=np.nan
                        tuning_p_matrix[ses_ind][neuron]=np.nan
                        continue

                    ephys_neuron_unbinned_=ephys_[neuron]
                    ephys_neuron_unbinned=np.asarray(np.split(ephys_neuron_unbinned_,num_states,axis=1))
                    ephys_neuron=np.mean(np.split(ephys_neuron_unbinned,10,axis=2),axis=0)
                    z_max=st.zscore(np.nanmax(ephys_neuron,axis=2),axis=0)
                    z_max_prefstate=z_max[np.argmax(np.mean(z_max,axis=1))]
                    tuning_z_matrix[ses_ind][neuron]=np.nanmean(z_max_prefstate)
                    tuning_p_matrix[ses_ind][neuron]=st.ttest_1samp(remove_nan(z_max_prefstate),0)[1]

                    tuning_z_matrix_allstates[ses_ind][neuron]=np.nanmean(z_max,axis=1)
                    tuning_p_matrix_allstates[ses_ind][neuron]=np.asarray([st.ttest_1samp(remove_nan(z_max[ii]),0)[1]                                                                for ii in range(len(z_max))])


                    ##Phase peaks
                    ephys_neuron_3=np.asarray(np.split(ephys_neuron_unbinned,3,axis=2))
                    max_phase=np.max(np.mean(np.mean(ephys_neuron_3,axis=1),axis=1),axis=1)
                    z_max_phase=st.zscore(max_phase)

                    #tuning_z_matrix_allphases[ses_ind][neuron]=np.nanmean(z_max_phase,axis=0)
                    #tuning_p_matrix_allphases[ses_ind][neuron]=np.asarray([st.ttest_1samp(remove_nan(
                    #z_max_phase[:,ii]),0)\
                    #                                                       [1] for ii in range(len(z_max_phase.T))])
                    tuning_z_matrix_allphases[ses_ind][neuron]=z_max_phase

                    ##replace ttests with permutation tests

            tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_z'][mouse_recday]=tuning_z_matrix
            tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_p'][mouse_recday]=tuning_p_matrix

            tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_z_allstates'][mouse_recday]=            tuning_z_matrix_allstates
            tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_p_allstates'][mouse_recday]=            tuning_p_matrix_allstates

            tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_z_allphases'][mouse_recday]=            tuning_z_matrix_allphases
            #tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_p_allphases'][mouse_recday]=\
            #tuning_p_matrix_allphases

        

print(time.time()-tt)


# In[18]:


###Tuning booleans for states and phases
#num_states=4 
num_phases=3 
 
p_thr=0.05 ###to account for occasional low num of trials 
for day_type in ['combined_ABCD_AB']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday) 
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday]) 
        num_neurons=len(cluster_dic['good_clus'][mouse_recday]) 
        sessions=Task_num_dic[mouse_recday]
        num_refses=len(np.unique(sessions))
        num_comparisons=num_refses-1
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        num_nonrepeat_sessions=len(non_repeat_ses)
        
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
        ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
        AB_sessions=np.where(abstract_structures=='AB')[0]
        
        for abstract_structure, ses_array in {'ABCD':ABCD_sessions,'AB':AB_sessions}.items():
        
            num_states=len(abstract_structure)

            peak_boolean_all_states=np.zeros((len(ses_array),num_neurons,num_states)) 
            peak_boolean_all_phases=np.zeros((len(ses_array),num_neurons,num_phases)) 
            peak_boolean_all_phases_max=np.zeros((len(ses_array),num_neurons,num_phases)) 
            for ses_ind, session in enumerate(ses_array): 
                print(ses_ind)

                z_ses_state=tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_z_allstates'][mouse_recday][ses_ind] 
                if len(z_ses_state)==0:
                    print('No trials detected')
                    continue

                p_ses_state=tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_p_allstates'][mouse_recday][ses_ind] 

                z_ses_phase=tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_z_allphases'][mouse_recday][ses_ind] 
                #p_ses_phase=tuning_singletrial_dic2['tuning_p_allphases'][mouse_recday][ses_ind] 


                for neuron in np.arange(num_neurons): 
                    peak_boolean_all_states[ses_ind][neuron]=np.logical_and(z_ses_state[neuron]>0,                                                                            p_ses_state[neuron]<=p_thr) 
                    #peak_boolean_all_phases[ses_ind][neuron]=np.logical_and(z_ses_phase[neuron]>0,\
                    #                                                        p_ses_phase[neuron]<=p_thr) 
                    peak_boolean_all_phases_max[ses_ind][neuron]=z_ses_phase[neuron]==np.max(z_ses_phase[neuron]) 

            tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_state_boolean'][mouse_recday]=            peak_boolean_all_states 
            #tuning_singletrial_dic2['tuning_phase_boolean'][mouse_recday]=peak_boolean_all_phases 
            tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_phase_boolean_max'][mouse_recday]=            peak_boolean_all_phases_max 


# In[ ]:





# In[36]:


###GLM - train on ABCD test on AB

#GLM_ABCDAB_dic=rec_dd()
num_phases=5
num_nodes=9
num_locations=21
#num_states=4 defined below
num_regressors=6 ##phase, place, time (from reward), distance (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place

smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1


abstract_structure_training='ABCD'
abstract_structure_test='AB'

remove_edges=True
num_iterations=100
lag_min=30*40 ###1200 bins = 30 seconds
smooth_SDs=5

if remove_edges==True:
    num_locations=num_nodes
    location_bins=np.arange(num_nodes+1)+1
        

day_type = 'combined_ABCD_AB'

for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)


    try:
        awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
        awake_sessions=session_dic['awake'][mouse_recday]

        #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
        #np.arange(len(awake_sessions_behaviour))])

        num_sessions=len(awake_sessions_behaviour)

        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        sessions=Task_num_dic[mouse_recday]
        num_refses=len(np.unique(sessions))
        num_comparisons=num_refses-1
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

        training_sessions=np.intersect1d(np.where(abstract_structures==abstract_structure_training)[0],                                   np.arange(len(non_repeat_ses)))      
        test_sessions=np.intersect1d(np.where(abstract_structures==abstract_structure_test)[0],                                     np.arange(len(non_repeat_ses)))

        coeffs_all=np.zeros((num_neurons,len(non_repeat_ses),len(non_repeat_ses),num_regressors))
        coeffs_all[:]=np.nan
        ###Training


        phases_conc_all=[]
        states_conc_all=[]
        times_conc_all=[]
        distances_conc_all=[]
        Location_raw_eq_all=[]
        Neuron_raw_eq_all=[]
        speed_raw_eq_all=[]
        acceleration_raw_eq_all=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            num_states_training=len(abstract_structure_training)
            state_bins_training=np.arange(num_states_training+1)
            try:
                Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                speed_raw=speed_dic[mouse_recday][ses_ind]
                acceleration_raw_=np.diff(speed_raw)/0.025
                acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')

            except:
                
                
                print('Trying Ceph')
                try:
                    Neuron_raw=np.load(Intermediate_object_folder_ceph+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    Location_raw=np.load(Intermediate_object_folder_ceph+'Location_raw_'+mouse_recday                                         +'_'+str(ses_ind)+'.npy')
                    XY_raw=np.load(Intermediate_object_folder_ceph+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    speed_raw=speed_dic[mouse_recday][ses_ind]
                    acceleration_raw_=np.diff(speed_raw)/0.025
                    acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                    Trial_times=np.load(Intermediate_object_folder_ceph+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                except:
                    print('Files not found for session '+str(ses_ind))
                    continue

            phases=Phases_raw_dic[mouse_recday][ses_ind]
            phases_conc=concatenate_complex2(concatenate_complex2(phases))
            states=States_raw_dic[mouse_recday][ses_ind]
            states_conc=concatenate_complex2(concatenate_complex2(states))
            times=Times_from_reward_dic[mouse_recday][ses_ind]
            times_conc=concatenate_complex2(concatenate_complex2(times))
            distances=Distances_from_reward_dic[mouse_recday][ses_ind]
            distances_conc=concatenate_complex2(concatenate_complex2(distances))
            speed_raw_eq=speed_raw[:len(phases_conc)]
            acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
            Location_raw_eq=Location_raw[:len(phases_conc)]

            if remove_edges==True:
                Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges

            if len(phases_conc)>=len(speed_raw_eq):
                print('Mismatch between speed and ephys - work around here but check')
                phases_conc=phases_conc[:len(speed_raw_eq)]
                states_conc=states_conc[:len(speed_raw_eq)]
                times_conc=times_conc[:len(speed_raw_eq)]
                distances_conc=distances_conc[:len(speed_raw_eq)]
                Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]

            phases_conc_all.append(phases_conc)
            states_conc_all.append(states_conc)
            times_conc_all.append(times_conc)
            distances_conc_all.append(distances_conc)
            Location_raw_eq_all.append(Location_raw_eq)
            speed_raw_eq_all.append(speed_raw_eq)
            acceleration_raw_eq_all.append(acceleration_raw_eq)
            Neuron_raw_eq_all.append(Neuron_raw[:,:len(speed_raw_eq)])

        phases_conc_all=np.asarray(phases_conc_all,dtype=object)
        states_conc_all=np.asarray(states_conc_all,dtype=object)
        times_conc_all=np.asarray(times_conc_all,dtype=object)
        distances_conc_all=np.asarray(distances_conc_all,dtype=object)
        Location_raw_eq_all=np.asarray(Location_raw_eq_all,dtype=object)
        speed_raw_eq_all=np.asarray(speed_raw_eq_all,dtype=object)
        acceleration_raw_eq_all=np.asarray(acceleration_raw_eq_all,dtype=object)


        phases_conc_training=np.hstack((phases_conc_all[training_sessions]))
        states_conc_training=np.hstack((states_conc_all[training_sessions]))
        times_conc_training=np.hstack((times_conc_all[training_sessions]))
        distances_conc_training=np.hstack((distances_conc_all[training_sessions]))
        Location_raw_eq_training=np.hstack((Location_raw_eq_all[training_sessions]))
        speed_raw_eq_training=np.hstack((speed_raw_eq_all[training_sessions]))
        acceleration_raw_eq_training=np.hstack((acceleration_raw_eq_all[training_sessions]))
        Neuron_raw_eq_training=np.hstack(([Neuron_raw_eq_all[ses_ind] for ses_ind in training_sessions]))





        speed_phases=st.binned_statistic(phases_conc_training, speed_raw_eq_training , bins=phase_bins)[0]
        acceleration_phases=st.binned_statistic(phases_conc_training, acceleration_raw_eq_training , bins=phase_bins)[0]

        Neuron_phases_training=np.zeros((num_neurons,num_phases))
        Neuron_states_training=np.zeros((num_neurons,num_states_training))
        Neuron_locations_training=np.zeros((num_neurons,num_locations))
        Neuron_phases_training[:]=np.nan
        Neuron_locations_training[:]=np.nan
        Neuron_states_training[:]=np.nan
        for neuron in np.arange(num_neurons):
            Neuron_raw_eq=Neuron_raw_eq_training[neuron]
            Neuron_phases=st.binned_statistic(phases_conc_training, Neuron_raw_eq , bins=phase_bins)[0]
            Neuron_states=st.binned_statistic(states_conc_training, Neuron_raw_eq , bins=state_bins_training)[0]
            Neuron_locations=st.binned_statistic(Location_raw_eq_training, Neuron_raw_eq , bins=location_bins)[0]
            #Neuron_states=st.binned_statistic(Location_raw_eq, Neuron_raw_eq , bins=location_bins)[0]
            Neuron_phases_training[neuron]=Neuron_phases
            Neuron_states_training[neuron]=Neuron_states
            Neuron_locations_training[neuron]=Neuron_locations



        ###Test

        phases_conc_test=np.hstack((phases_conc_all[test_sessions]))
        states_conc_test=np.hstack((states_conc_all[test_sessions]))
        times_conc_test=np.hstack((times_conc_all[test_sessions]))
        distances_conc_test=np.hstack((distances_conc_all[test_sessions]))
        Location_raw_eq_test=np.hstack((Location_raw_eq_all[test_sessions]))
        speed_raw_eq_test=np.hstack((speed_raw_eq_all[test_sessions]))
        acceleration_raw_eq_test=np.hstack((acceleration_raw_eq_all[test_sessions]))
        Neuron_raw_eq_test=np.hstack(([Neuron_raw_eq_all[ses_ind] for ses_ind in test_sessions]))


        if remove_edges==True:
            Location_raw_eq_test[Location_raw_eq_test>num_nodes]=np.nan ### removing edges

        if len(phases_conc_test)>=len(speed_raw_eq_test):
            print('Mismatch between speed and ephys - work around here but check')
            phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
            states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
            times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
            distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
            Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]




        Location_raw_eq_test_nonan=Location_raw_eq_test[~np.isnan(Location_raw_eq_test)]
        speed_raw_eq_test_nonan=speed_raw_eq_test[~np.isnan(Location_raw_eq_test)]
        acceleration_raw_eq_test_nonan=acceleration_raw_eq_test[~np.isnan(Location_raw_eq_test)]
        times_conc_test_nonan=times_conc_test[~np.isnan(Location_raw_eq_test)]
        distances_conc_test_nonan=distances_conc_test[~np.isnan(Location_raw_eq_test)]



        ###regression
        coeffs_all=np.zeros((num_neurons,num_regressors))
        coeffs_shuff_all=np.zeros((num_neurons,num_iterations,num_regressors))
        
        coeffs_all[:]=np.nan
        coeffs_shuff_all[:]=np.nan
        
        for neuron in np.arange(num_neurons):
            #print(neuron)
            Neuron_phases=Neuron_phases_training[neuron]
            Neuron_states=Neuron_states_training[neuron]
            Neuron_locations=Neuron_locations_training[neuron]

            Neuron_raw_eq_test_=Neuron_raw_eq_test[neuron,:len(phases_conc_test)]
            Neuron_raw_eq_test_nonan=Neuron_raw_eq_test_[~np.isnan(Location_raw_eq_test)]

            FR_training_phases=Neuron_phases[phases_conc_test]
            FR_training_states=Neuron_states[states_conc_test]
            FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test)]
            FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test)]
            FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan-1).astype(int)]


            ###regression
            X = np.vstack((FR_training_phases_nonan,                           FR_training_locations_nonan,                           times_conc_test_nonan,                           distances_conc_test_nonan,                           speed_raw_eq_test_nonan,                           acceleration_raw_eq_test_nonan)).T

            X_clean=X[~np.isnan(X).any(axis=1)]
            X_z=st.zscore(X_clean,axis=0)

            y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
            reg = LinearRegression().fit(X_z, y)
            coeffs=reg.coef_

            coeffs_all[neuron]=coeffs
            
            
            
            if len(y)<lag_min:
                continue
            
            
            
            
            ##Permutations
            max_roll=int(len(y)-lag_min)
            min_roll=int(lag_min)

            if max_roll<min_roll:
                continue
            for iteration in range(num_iterations):
                copy_y=np.copy(y)

                shift=random.randrange(max_roll-min_roll)+min_roll
                y_shifted=np.roll(copy_y,shift)

                reg = LinearRegression().fit(X_z, y_shifted)
                coeffs=reg.coef_
                coeffs_shuff_all[neuron,iteration]=coeffs


        thr_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
        thr_neuron_betas[:]=np.nan
        for neuron in np.arange(num_neurons):
            thr_phase_coeff=np.nanpercentile(np.asarray([np.nanmean(coeffs_shuff_all[neuron,ii,0])                                                         for ii in range(num_iterations)]),95)
            thr_place_coeff=np.nanpercentile(np.asarray([np.nanmean(coeffs_shuff_all[neuron,ii,1])                                                         for ii in range(num_iterations)]),95)
            thr_neuron_betas[neuron]=thr_phase_coeff,thr_place_coeff

            
        GLM_ABCDAB_dic['coeffs_all'][mouse_recday]=coeffs_all
        GLM_ABCDAB_dic['thr_neuron_betas'][mouse_recday]=thr_neuron_betas

        
        
    except Exception as e:
        print('betas not calculated')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


# In[37]:


coeffs_all_all=np.vstack((dict_to_array(GLM_ABCDAB_dic['coeffs_all'])))
phase_coeff_all=coeffs_all_all[:,0]
plt.hist(phase_coeff_all,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed')
#plt.savefig(Ephys_output_folder_dropbox+'Phase_betas.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(phase_coeff_all))
print(st.ttest_1samp(phase_coeff_all,0))
print('')

place_coeff_all=coeffs_all_all[:,1]
plt.hist(place_coeff_all,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed')
#plt.savefig(Ephys_output_folder_dropbox+'Phase_betas.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(place_coeff_all))
print(st.ttest_1samp(place_coeff_all,0))
print('')


# In[ ]:





# In[ ]:





# In[ ]:





# In[98]:


###computing state tuning using per trial zscore - ABCD AB days
Tuned_ABCDAB_dic=rec_dd()
##paramaters
num_bins=90
#num_states=4 replaced below by seperate value per session
num_phases=3
num_nodes=9
num_lags=12
smoothing_sigma=10
num_iterations=100

for day_type in ['combined_ABCD_AB']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
        awake_sessions=session_dic['awake'][mouse_recday]
        num_sessions=len(awake_sessions_behaviour)
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        #regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]
        
        
        if mouse_recday in ['ah04_19122021_20122021','me11_15122021_16122021']:
            addition='_'
        else:
            addition=''
        
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
        
        ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
        AB_sessions=np.where(abstract_structures=='AB')[0]


        found_ses=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            try:
                Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+addition+'.npy')
                found_ses.append(ses_ind)

            except:
                print('Files not found for session '+str(ses_ind))
                continue
        num_non_repeat_ses_found=len(found_ses)


        
        for abstract_structure, ses_array in {'ABCD':ABCD_sessions,'AB':AB_sessions}.items():
            zmax_all=np.zeros((num_neurons,len(ses_array)))
            zmax_all[:]=np.nan

            zmax_all_strict=np.zeros((num_neurons,len(ses_array)))
            zmax_all_strict[:]=np.nan

            corr_mean_max_all=np.zeros((num_neurons,len(ses_array),2))
            corr_mean_max_all[:]=np.nan

            for ses_ind_ind, ses_ind in enumerate(ses_array):
                ses_ind_actual=found_ses[ses_ind_ind]

                num_states=len(abstract_structures[ses_ind_ind])
                phase_norm_mean=np.tile(np.repeat(np.arange(num_phases),num_bins/num_phases),num_states)
                phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states,num_bins))


                Actual_activity_ses_=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_actual)+                                             addition+'.npy')
                Actual_activity_ses=Actual_activity_ses_.T
                #GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]

                phase_peaks=tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_phase_boolean_max']                [mouse_recday][ses_ind_ind]
                
                if len(phase_peaks)==0:
                    continue
                pref_phase_neurons=np.argmax(phase_peaks,axis=1)
                phases=Phases_raw_dic2[mouse_recday][ses_ind_actual]
                phases_conc=concatenate_complex2(concatenate_complex2(phases))

                Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+addition+                                    '.npy')
                Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25

                for neuron in np.arange(num_neurons):
                    pref_phase=pref_phase_neurons[neuron]
                    Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]

                    Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                            smoothing=False,return_mean=False)

                    Actual_norm_means=np.vstack(([[np.nanmean(Actual_norm[trial,num_bins*ii:num_bins*(ii+1)]                        [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                                      for trial in np.arange(len(Actual_norm))]))
                    max_state=np.argmax(np.nanmean(Actual_norm_means,axis=0))
                    zactivity_prefstate=st.zscore(Actual_norm_means,axis=1)[:,max_state]
                    zmax_all[neuron,ses_ind_ind]=st.ttest_1samp(zactivity_prefstate,0)[1]

                    zmax_shifted=np.zeros(num_iterations)
                    zmax_shifted[:]=np.nan
                    for iteration in range(num_iterations):
                        shifts=np.random.randint(0,num_states,len(Actual_norm_means))
                        Actual_norm_means_shifted=indep_roll(Actual_norm_means,shifts)
                        max_state=np.argmax(np.nanmean(Actual_norm_means_shifted,axis=0))
                        zactivity_prefstate=st.zscore(Actual_norm_means_shifted,axis=1)[:,max_state]
                        zactivity_prefstate_mean=np.nanmean(zactivity_prefstate)
                        zmax_shifted[iteration]=zactivity_prefstate_mean
                    mean_zmax_shifted=np.nanmean(zmax_shifted)
                    zmax_all_strict[neuron,ses_ind_ind]=st.ttest_1samp(zactivity_prefstate,mean_zmax_shifted)[1]



                    Actual_norm_max=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                            smoothing=False,return_mean=False,take_max=True)

                    Actual_norm_max_means=np.vstack(([[np.nanmean(Actual_norm_max[trial,num_bins*ii:num_bins*(ii+1)]                        [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                                      for trial in np.arange(len(Actual_norm_max))]))

                    r_,p_=st.pearsonr(np.concatenate(Actual_norm_means),np.concatenate(Actual_norm_max_means))
                    corr_mean_max_all[neuron,ses_ind_ind]=[r_,p_]

            Tuned_ABCDAB_dic[abstract_structure]['State_zmax'][mouse_recday]=zmax_all
            Tuned_ABCDAB_dic[abstract_structure]['State_zmax_strict'][mouse_recday]=zmax_all_strict
            Tuned_ABCDAB_dic[abstract_structure]['corr_mean_max'][mouse_recday]=corr_mean_max_all


# In[99]:


day_type='combined_ABCDonly'
for day_type in ['combined_ABCD_AB']:
    for abstract_structure in ['ABCD','AB']:
        for mouse_recday in day_type_dicX[day_type]:
            num_peaks_all=np.vstack(([np.sum(tuning_singletrial_ABCDAB_dic2[abstract_structure]                                             ['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)                                      for ses_ind in np.arange(len(tuning_singletrial_ABCDAB_dic2[abstract_structure]                                                                   ['tuning_state_boolean'][mouse_recday]))]))

            state_bool=np.sum(num_peaks_all>0,axis=0)>(len(num_peaks_all)//2)

            State_zmax=Tuned_ABCDAB_dic[abstract_structure]['State_zmax'][mouse_recday]

            state_bool_zmax=np.sum(State_zmax<0.05,axis=1)>(len(State_zmax.T)/3)
            state_bool_zmax_one=np.sum(State_zmax<0.05,axis=1)>1

            Tuned_ABCDAB_dic[abstract_structure]['State_zmax_bool'][mouse_recday]=state_bool_zmax
            Tuned_ABCDAB_dic[abstract_structure]['State_zmax_bool_one'][mouse_recday]=state_bool_zmax_one


# In[ ]:





# In[ ]:





# In[536]:




threshold_state_sessions=0

use_permuted=True ###for phase and place tuning
use_both=True ###for phase and place tuning - use both permutation and ttest
use_zscored=True ##for state tuning
lowest_thr=True

if use_both==True:
    use_permuted=False

for abstract_structure in ['ABCD','AB']:
    print('')
    print(abstract_structure)
    print('_____')
    phase_bool_all=[]
    state_bool_all=[]
    place_bool_all=[]
    phase_place_bool_all=[]
    for day_type in ['combined_ABCD_AB']:
        for mouse_recday in day_type_dicX[day_type]:#list(GLM_dic['mean_neuron_betas'].keys()):
            print(mouse_recday)
            try:
                mean_neuron_betas=GLM_ABCDAB_dic['coeffs_all'][mouse_recday]
                p_neuron_betas=GLM_ABCDAB_dic[abstract_structure]['p_neuron_betas'][mouse_recday]
                thr_neuron_beta=GLM_ABCDAB_dic['thr_neuron_betas'][mouse_recday]

                #mean_neuron_betas_withintask=GLM_withinTask_dic[abstract_structure]['mean_neuron_betas'][mouse_recday]
                #thr_neuron_betas_withintask=GLM_withinTask_dic[abstract_structure]['thr_neuron_betas'][mouse_recday]


                
                phase_bool_ttest=np.logical_and(mean_neuron_betas[:,0]>0,p_neuron_betas[:,0]<0.05)
                place_bool_ttest=np.logical_and(mean_neuron_betas[:,1]>0,p_neuron_betas[:,1]<0.05)
                #phase_place_bool_ttest=np.logical_and(mean_neuron_betas[:,2]>0,p_neuron_betas[:,2]<0.05)


                phase_bool_permutation=mean_neuron_betas[:,0]>thr_neuron_beta[:,0]
                place_bool_permutation=mean_neuron_betas[:,1]>thr_neuron_beta[:,1]

                if use_permuted==True:
                    phase_bool=phase_bool_permutation
                    place_bool=place_bool_permutation
                    #phase_place_bool=mean_neuron_betas[:,2]>thr_neuron_beta[:,2]
                    try:
                        state_within_bool=mean_neuron_betas_withintask[:,1]>thr_neuron_betas_withintask[:,1]
                        Tuned_dic['State_withintask'][mouse_recday]=state_within_bool
                    except:
                        print('within task state tuning not calculated')
                elif use_both==True:
                    phase_bool=np.logical_and(phase_bool_permutation,phase_bool_ttest)
                    place_bool=np.logical_and(place_bool_permutation,place_bool_ttest)

                else:
                    phase_bool=phase_bool_ttest
                    place_bool=place_bool_ttest
                    #phase_place_bool=phase_place_bool_ttest


                num_peaks_all=np.vstack(([np.sum(tuning_singletrial_ABCDAB_dic2[abstract_structure]                                                 ['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)                        for ses_ind in np.arange(len(tuning_singletrial_ABCDAB_dic2[abstract_structure]                                                     ['tuning_state_boolean'][mouse_recday]))]))

                if use_zscored==True:
                    if lowest_thr==True:
                        state_bool=Tuned_ABCDAB_dic[abstract_structure]['State_zmax_bool_one'][mouse_recday]

                    else:
                        state_bool=Tuned_ABCDAB_dic[abstract_structure]['State_zmax_bool'][mouse_recday]

                else:
                    if threshold_state_sessions=='half':
                        state_bool=np.sum(num_peaks_all>0,axis=0)>(len(num_peaks_all)//2)
                    else:
                        state_bool=np.sum(num_peaks_all>0,axis=0)>threshold_state_sessions

                Tuned_ABCDAB_dic[abstract_structure]['Phase'][mouse_recday]=phase_bool
                Tuned_ABCDAB_dic[abstract_structure]['State'][mouse_recday]=state_bool
                Tuned_ABCDAB_dic[abstract_structure]['Place'][mouse_recday]=place_bool
                #Tuned_dic['Phase_Place'][mouse_recday]=place_bool


                #Tuned_dic['Phase_ttest'][mouse_recday]=phase_bool_ttest
                #Tuned_dic['Place_ttest'][mouse_recday]=place_bool_ttest


                phase_and_place_bool=np.logical_and(phase_bool,place_bool)
                state_place_bool=np.logical_and(phase_bool,state_bool)


                print(np.sum(phase_bool)/len(phase_bool))
                print(np.sum(state_bool)/len(state_bool))
                print(np.sum(place_bool)/len(place_bool))
                #print(np.sum(phase_place_bool)/len(phase_place_bool))
                print('')
                #print(np.sum(phase_state_bool)/np.sum(phase_bool))
                #print(np.sum(phase_and_place_bool)/np.sum(phase_bool))
                print('')

                phase_bool_all.append(phase_bool)
                state_bool_all.append(state_bool)
                place_bool_all.append(place_bool)
                #phase_place_bool_all.append(phase_place_bool)
            except Exception as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('Not calculated')
                print('')
                
    exec('phase_bool_all_'+abstract_structure+'=phase_bool_all')
    exec('state_bool_all_'+abstract_structure+'=state_bool_all')
    exec('place_bool_all_'+abstract_structure+'=place_bool_all')
    exec('phase_place_bool_all_'+abstract_structure+'=phase_place_bool_all')
    


# In[ ]:





# In[102]:


abstract_structure='AB'
exec('phase_bool_all=phase_bool_all_'+abstract_structure)
exec('state_bool_all=state_bool_all_'+abstract_structure)
exec('place_bool_all=place_bool_all_'+abstract_structure)
exec('phase_place_bool_all=phase_place_bool_all_'+abstract_structure)

phase_bool_all=np.concatenate(phase_bool_all)
state_bool_all=np.concatenate(state_bool_all)
place_bool_all=np.concatenate(place_bool_all)
#phase_place_bool_all=np.concatenate(phase_place_bool_all)


print('phase cells: '+str(np.sum(phase_bool_all)/len(phase_bool_all)))
print('state cells: '+str(np.sum(state_bool_all)/len(state_bool_all)))
print('place cells: '+str(np.sum(place_bool_all)/len(place_bool_all)))

print('')

phase_state_bool=np.logical_and(phase_bool_all,state_bool_all)
print('state cells as proportion of phase cells: '+str(np.sum(phase_state_bool)/np.sum(phase_bool_all)))

phase_place_bool=np.logical_and(phase_bool_all,place_bool_all)
#print('place cells as proportion of phase cells: '+str(np.sum(phase_place_bool)/np.sum(phase_bool_all)))

phase_state_unique=np.logical_and(phase_state_bool,~phase_place_bool)
phase_place_unique=np.logical_and(phase_place_bool,~phase_state_bool)
#print('pure state cells as proportion of phase cells: '+str(np.sum(phase_state_unique)/np.sum(phase_bool_all)))
#print('pure place cells as proportion of phase cells: '+str(np.sum(phase_place_unique)/np.sum(phase_bool_all)))

print('')

#phase_place_or_state=np.logical_or(phase_place_bool,phase_state_bool)
#print('place or state cells as proportion of phase cells: '+str(np.sum(phase_place_or_state)/np.sum(phase_bool_all)))

phase_place_and_state=np.logical_and(phase_place_bool,phase_state_bool)
print('place cells as proportion of phase/state cells: '+str(np.sum(phase_place_and_state)/np.sum(phase_state_bool)))


state_phase_bool=np.logical_and(phase_bool_all,state_bool_all)
print('state cells as proportion of phase cells:: '+str(np.sum(state_phase_bool)/np.sum(state_bool_all)))


# In[103]:


###Plotting GLM results
import upsetplot
all_cells=np.arange(len(phase_bool_all))
phase_cells=np.where(phase_bool_all==True)[0]
state_cells=np.where(state_bool_all==True)[0]
place_cells=np.where(place_bool_all==True)[0]


non_phase_cells=np.where(phase_bool_all==False)[0]
non_state_cells=np.where(state_bool_all==False)[0]
non_place_cells=np.where(place_bool_all==False)[0]

nothing_bool_all=np.logical_and(np.logical_and(~phase_bool_all,~state_bool_all),~place_bool_all)
nothing_cells=np.where(nothing_bool_all==True)[0]

data=upsetplot.from_memberships([['Phase'],                                 ['Phase','State'],                                 ['Phase','Non_State'],                                 ['Phase','State','Non_Place'],                  ['Phase','State','Place'],[],['Non_Phase','State']],                 data=np.asarray([len(phase_cells),                                  len(np.intersect1d(state_cells,phase_cells)),                                  len(np.intersect1d(non_state_cells,phase_cells)),                                  len(np.intersect1d(np.intersect1d(state_cells,non_place_cells),phase_cells)),                                  len(np.intersect1d(np.intersect1d(state_cells,place_cells),phase_cells)),                                 len(nothing_cells),                                 len(np.intersect1d(state_cells,non_phase_cells))])/len(all_cells))
#print(data)

upsetplot.UpSet(data, sort_by='degree',sort_categories_by='cardinality')

upsetplot.plot(data)
#plt.savefig(Ephys_output_folder_dropbox+'_UpsetPlot_cells.svg')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


def equalize_rowsX(xxx):
    len_min=int(np.min([len(xxx[ii]) for ii in range(len(xxx))]))
    xx=np.asarray([xxx[ii][:len_min] for ii in range(len(xxx))])
    return(xx)

def cross_corr_fast(x,y):
    data_length = len(x)

    b = x
    a = np.zeros(data_length * 2)

    a[data_length//2:data_length//2+data_length] = y # This works for data_length being even

    # Do an array flipped convolution, which is a correlation.
    c = sp.signal.fftconvolve(b, a[::-1], mode='valid')
    return(c/100)

def cross_corr_plot(array,length_corr):

    cross_corr=cross_corr_fast(array[0],array[1])

    midpoint=int((len(cross_corr))/2)

    cross_corr_cut=cross_corr[midpoint-length_corr:midpoint+length_corr]
    cross_corr_cut_mid=(len(cross_corr_cut))/2
    #plt.plot(np.arange(len(cross_corr_cut))-cross_corr_cut_mid,cross_corr_cut)
    plt.bar(np.arange(len(cross_corr_cut))-cross_corr_cut_mid,cross_corr_cut)

    #plt.axvline(0,color='black',ls='dashed',alpha=0.2)
    plt.ylim(0,np.max(cross_corr))


# In[36]:


mouse_recday='ah04_22122021_23122021'
structure_abstract=Variable_dic[mouse]['Structure_abstract']
rec_day_structure_abstract=recday_numbers_dic['structure_abstract'][mouse_recday]
print(rec_day_structure_abstract)
ses_ind=0
#Neuron_norm=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')
#Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')

Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')
num_trials=len(Trial_times)
num_neurons=len(Neuron_raw)


structure_abstract_ses=rec_day_structure_abstract[ses_ind]


##Normalisation
num_bins_state=90
num_states=len(structure_abstract_ses)

Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25
Neurons_norm=np.zeros((num_neurons,num_trials,num_bins_state*num_states))
Neurons_norm[:]=np.nan
for neuron in np.arange(num_neurons):
    Neuron_raw_neuron=Neuron_raw[neuron]

    Neurons_norm[neuron]=raw_to_norm(Neuron_raw_neuron,Trial_times_conc,num_states=num_states,                                     return_mean=False, smoothing=False)
    
print(structure_abstract_ses)


# In[37]:


print(rec_day_structure_abstract)


# In[38]:


neuron=9
plt.matshow(Neurons_norm[neuron])
plt.show()

length_corr=30
neuron_conc=np.hstack((Neurons_norm[neuron]))

len_bin=36
neuron_conc_reshaped=np.reshape(neuron_conc,(len(neuron_conc)//len_bin,len_bin))
neuron_conc_binned=np.mean(neuron_conc_reshaped,axis=1)

cross_corr_plot([neuron_conc_binned,neuron_conc_binned],length_corr)

plt.show()


# In[35]:


neuron=9
plt.matshow(Neurons_norm[neuron])
plt.show()

length_corr=30
neuron_conc=np.hstack((Neurons_norm[neuron]))

len_bin=36
neuron_conc_reshaped=np.reshape(neuron_conc,(len(neuron_conc)//len_bin,len_bin))
neuron_conc_binned=np.mean(neuron_conc_reshaped,axis=1)

cross_corr_plot([neuron_conc_binned,neuron_conc_binned],length_corr)

plt.show()


# In[61]:


Tuned_dic['Phase']['ah03_18082021']


# In[53]:


np.shape(tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_phase_boolean_max']                [mouse_recday])#[ses_ind_ind]


# In[76]:


'''
Trajectory coding

-add policy to GLM (both egocentric and allocentric) 
-9 locations x 4 directions of travel

4 versions:

previous action - egocentric
previous action - allocentric
next action - egocentric
next action - allocentric

'''


# In[ ]:





# In[11]:


mouse_recday='ah04_01122021_02122021'
awake_session_ind=0
Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')

Location_raw


# In[ ]:





# In[96]:


##Calculating occupancy across states
tt=time.time()
policy_dic=rec_dd()
tracking_oversampling_factor=50 ##oversampling of nodes_cut_dic
behaviour_oversampling_factor=3 ##oversampling of trialtimes_dic

for day_type in ['3_task','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)

        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]

        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]
        All_sessions=session_dic_behaviour['All'][mouse_recday]


        rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
        rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]

        structure_numbers_day=recday_numbers_dic['structure_numbers_day'][mouse_recday]
        structure_abstract_day=recday_numbers_dic['structure_abstract'][mouse_recday]

        policy_map_raw_all=[]
        policy_map_raw_reverse_all=[]

        for awake_session_ind, timestamp in enumerate(awake_sessions):
            print(awake_session_ind)


            abstract_structure_type=structure_abstract_day[awake_session_ind]
            try:
                states=structure_abstract_day[awake_session_ind]
                num_states=len(states)

                nodes_edges=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')


                ##calculating policy (state_action_ind)
                nodes_edges_unique_=unique_adjacent(nodes_edges)
                repeats__=number_of_repeats(nodes_edges)

                ##cleaning spurious detections (less than 100 ms (4 bins))
                nodes_edges_clean=np.concatenate([np.repeat(nodes_edges_unique_[ii],repeats__[ii])                                                   if repeats__[ii]>4 else np.repeat(nodes_edges_unique_[ii-1],repeats__[ii])                                                   for ii in range(len(nodes_edges_unique_))])
                nodes_edges_unique=unique_adjacent(nodes_edges_clean)
                repeats_=number_of_repeats(nodes_edges_clean)

                next_nodes_edges=np.concatenate([np.repeat(nodes_edges_unique[ii+1],repeats_[ii])                                               if np.isnan(nodes_edges_unique[ii+1])==False                                               else np.repeat(nodes_edges_unique[ii+2],repeats_[ii])                                               for ii in range(len(nodes_edges_unique)-2)])

                state_action_ind=[]
                for ii in range(len(nodes_edges_unique)-2): ##i.e. misses out last node and last edge 
                    find_direction1=find_direction(nodes_edges_unique[ii],nodes_edges_unique[ii+1])
                    find_direction2=find_direction(nodes_edges_unique[ii],nodes_edges_unique[ii+2])
                    if np.isnan(nodes_edges_unique[ii+1])==False and find_direction1 in list(direction_dic.keys()):
                        state_action_=np.repeat(np.where(State_action_grid==                                 str(int(nodes_edges_unique[ii]))+'_'+str(find_direction1))[0],repeats_[ii])
                    elif np.isnan(nodes_edges_unique[ii+2])==False and find_direction2 in list(direction_dic.keys()):
                        state_action_=np.repeat(np.where(State_action_grid==str(int(nodes_edges_unique[ii]))+'_'+                                           str(find_direction2))[0],repeats_[ii])
                    else:
                        state_action_=np.repeat(np.nan,repeats_[ii])
                    state_action_ind.append(state_action_)
                state_action_ind=np.concatenate(state_action_ind)

                state_action_ind=np.hstack((state_action_ind,                                            np.repeat(np.nan,len(nodes_edges_clean)-len(state_action_ind))))
                
                state_action_reverse_ind=[]
                for ii in np.arange(len(nodes_edges_unique)-2)+2: ##i.e. misses out first node and first edge 
                    #print(ii)
                    find_direction1=find_direction(nodes_edges_unique[ii-1],nodes_edges_unique[ii])
                    find_direction2=find_direction(nodes_edges_unique[ii-2],nodes_edges_unique[ii])

                    if np.isnan(nodes_edges_unique[ii-1])==False and find_direction1 in list(direction_dic.keys()):
                        state_action_=np.repeat(np.where(State_action_grid_reverse==                                 str(int(nodes_edges_unique[ii]))+'_'+str(find_direction1))[0],repeats_[ii])
                    elif np.isnan(nodes_edges_unique[ii-2])==False and find_direction2 in list(direction_dic.keys()):
                        state_action_=np.repeat(np.where(State_action_grid_reverse==str(int(nodes_edges_unique[ii]))+'_'+                                           str(find_direction2))[0],repeats_[ii])
                    else:
                        state_action_=np.repeat(np.nan,repeats_[ii])


                    state_action_reverse_ind.append(state_action_)
                state_action_reverse_ind=np.concatenate(state_action_reverse_ind)

                state_action_reverse_ind=np.hstack((np.repeat(np.nan,                                            len(nodes_edges_clean)-len(state_action_reverse_ind)),                                                    state_action_reverse_ind))

                policy_map_raw_all.append(state_action_ind)
                policy_map_raw_reverse_all.append(state_action_reverse_ind)

                policy_dic['Forward'][mouse_recday][awake_session_ind]=state_action_ind
                policy_dic['Reverse'][mouse_recday][awake_session_ind]=state_action_reverse_ind

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(e)
                print('No maps made')




print(time.time()-tt)


# In[ ]:





# In[106]:


###GLM - Using Policy instead of place - across tasks/states (using left out state/task combination as test data)

#GLM_dic_policy=rec_dd()
num_phases=5
num_nodes=9
num_locations=21
num_states=4
num_regressors=6 ##phase, place, time (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place

specific_days=['me10_17122021_19122021']
specific_day_bool=False
smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1
state_bins=np.arange(num_states+1)

remove_edges=True

if remove_edges==True:
    num_locations=len(node_states)
    location_bins=np.arange(len(node_states)+1)

        
for direction in ['Forward','Reverse']:
    print(direction)
    for day_type in ['3_task','combined_ABCDonly']:
        for mouse_recday in day_type_dicX[day_type]:




            if specific_day_bool==True:
                if mouse_recday not in specific_days:
                    continue

            print(mouse_recday)

            if len(GLM_dic_policy[direction]['coeffs_all'][mouse_recday])>0:
                print('Already analysed')
                continue
            try: 
                awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
                awake_sessions=session_dic['awake'][mouse_recday]

                #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
                #np.arange(len(awake_sessions_behaviour))])

                num_sessions=len(awake_sessions_behaviour)

                num_neurons=len(cluster_dic['good_clus'][mouse_recday])
                sessions=Task_num_dic[mouse_recday]
                num_refses=len(np.unique(sessions))
                num_comparisons=num_refses-1
                repeat_ses=np.where(rank_repeat(sessions)>0)[0]
                non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

                coeffs_all=np.zeros((num_neurons,len(non_repeat_ses)*num_states,num_regressors))
                coeffs_all[:]=np.nan



                for ses_ind_test_ind,ses_ind_test in enumerate(non_repeat_ses):
                    print(ses_ind_test)
                    training_sessions=np.setdiff1d(non_repeat_ses,ses_ind_test)

                    ###Training
                    phases_conc_all_=[]
                    states_conc_all_=[]
                    Location_raw_eq_all_=[]
                    Neuron_raw_all_=[]
                    for ses_ind_training_ind, ses_ind_training in enumerate(training_sessions):
                        try:
                            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                            Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                            XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                            speed_raw=speed_dic[mouse_recday][ses_ind_training]


                            acceleration_raw_=np.diff(speed_raw)/0.025
                            acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                            Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')

                        except:
                            print('Files not found (training) for session '+str(ses_ind_training))
                            continue

                        Location_raw=Policy_raw=policy_dic[direction][mouse_recday][ses_ind_training]

                        phases=Phases_raw_dic[mouse_recday][ses_ind_training]
                        phases_conc=concatenate_complex2(concatenate_complex2(phases))
                        states=States_raw_dic[mouse_recday][ses_ind_training]
                        states_conc=concatenate_complex2(concatenate_complex2(states))
                        times=Times_from_reward_dic[mouse_recday][ses_ind_training]
                        times_conc=concatenate_complex2(concatenate_complex2(times))
                        distances=Distances_from_reward_dic[mouse_recday][ses_ind_training]
                        distances_conc=concatenate_complex2(concatenate_complex2(distances))
                        speed_raw_eq=speed_raw[:len(phases_conc)]
                        acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
                        Location_raw_eq=Location_raw[:len(phases_conc)]

                        if remove_edges==True:
                            Location_raw_eq[Location_raw_eq>len(node_states)-1]=np.nan ### removing edges

                        if len(phases_conc)>=len(speed_raw_eq):
                            print('Mismatch between speed and ephys - work around here but check')
                            phases_conc=phases_conc[:len(speed_raw_eq)]
                            states_conc=states_conc[:len(speed_raw_eq)]
                            times_conc=times_conc[:len(speed_raw_eq)]
                            distances_conc=distances_conc[:len(speed_raw_eq)]
                            Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]
                            Neuron_raw=Neuron_raw[:,:len(speed_raw_eq)]

                        speed_phases=st.binned_statistic(phases_conc, speed_raw_eq , bins=phase_bins)[0]
                        acceleration_phases=st.binned_statistic(phases_conc, acceleration_raw_eq , bins=phase_bins)[0]

                        phases_conc_all_.append(phases_conc)
                        states_conc_all_.append(states_conc)
                        Location_raw_eq_all_.append(Location_raw_eq)
                        Neuron_raw_all_.append(Neuron_raw)



                    phases_conc_all_=np.hstack((phases_conc_all_))
                    states_conc_all_=np.hstack((states_conc_all_))
                    Location_raw_eq_all_=np.hstack((Location_raw_eq_all_))
                    Neuron_raw_all_=np.hstack((Neuron_raw_all_))





                    ###Test
                    try:
                        Neuron_raw_test=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                        Location_raw_test=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                    except:
                        print('Files not found (test) for session '+str(ses_ind_test))
                        continue

                    Location_raw_test=Policy_raw_test=policy_dic[direction][mouse_recday][ses_ind_test]
                    speed_raw_test=speed_dic[mouse_recday][ses_ind_test]
                    acceleration_raw_test_=np.diff(speed_raw_test)/0.025
                    acceleration_raw_test=np.hstack((acceleration_raw_test_[0],acceleration_raw_test_))

                    phases_test=Phases_raw_dic[mouse_recday][ses_ind_test]
                    phases_conc_test=concatenate_complex2(concatenate_complex2(phases_test))
                    states_test=States_raw_dic[mouse_recday][ses_ind_test]
                    states_conc_test=concatenate_complex2(concatenate_complex2(states_test))
                    times_test=Times_from_reward_dic[mouse_recday][ses_ind_test]
                    times_conc_test=concatenate_complex2(concatenate_complex2(times_test))
                    distances_test=Distances_from_reward_dic[mouse_recday][ses_ind_test]
                    distances_conc_test=concatenate_complex2(concatenate_complex2(distances_test))

                    speed_raw_eq_test=gaussian_filter1d(speed_raw_test[:len(phases_conc_test)],smooth_SDs)
                    acceleration_raw_eq_test=gaussian_filter1d(acceleration_raw_test[:len(phases_conc_test)],smooth_SDs)
                    Location_raw_eq_test=Location_raw_test[:len(phases_conc_test)]

                    Neuron_raw_eq_test_all=Neuron_raw_test[:,:len(phases_conc_test)]

                    if remove_edges==True:
                        Location_raw_eq_test[Location_raw_eq_test>len(node_states)-1]=np.nan ### removing edges

                    if len(phases_conc_test)>=len(speed_raw_eq_test):
                        print('Mismatch between speed and ephys - work around here but check')
                        phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
                        states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
                        times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
                        distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
                        Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]
                        Neuron_raw_eq_test_all=Neuron_raw_eq_test_all[:,:len(speed_raw_eq_test)]


                    ###extracting test state  
                    states_=np.arange(num_states)
                    for test_state in np.arange(num_states):

                        ind_ses_state=(ses_ind_test_ind*num_states)+test_state

                        ###completing training arrays
                        phases_conc_test_training=phases_conc_test[states_conc_test!=test_state]
                        Location_raw_eq_test_training=Location_raw_eq_test[states_conc_test!=test_state]
                        Neuron_raw_test_training=Neuron_raw_eq_test_all[:,states_conc_test!=test_state]

                        phases_conc_all=np.hstack((phases_conc_all_,phases_conc_test_training))
                        Location_raw_eq_all=np.hstack((Location_raw_eq_all_,Location_raw_eq_test_training))
                        Neuron_raw_all=np.hstack((Neuron_raw_all_,Neuron_raw_test_training))


                        Neuron_phases_all=np.zeros((num_neurons,num_phases))
                        #Neuron_states_all=np.zeros((num_neurons,num_states))
                        Neuron_locations_all=np.zeros((num_neurons,num_locations))
                        Neuron_phases_all[:]=np.nan
                        Neuron_locations_all[:]=np.nan
                        #Neuron_states_all[:]=np.nan
                        for neuron in np.arange(num_neurons):
                            Neuron_raw_eq_all=Neuron_raw_all[neuron,:len(phases_conc_all)]
                            Neuron_phases=st.binned_statistic(phases_conc_all, Neuron_raw_eq_all, bins=phase_bins)[0]
                            #Neuron_states=st.binned_statistic(states_conc_all, Neuron_raw_eq_all, bins=state_bins)[0]
                            Neuron_locations=st.binned_statistic(Location_raw_eq_all, Neuron_raw_eq_all, bins=location_bins)[0]
                            Neuron_phases_all[neuron]=Neuron_phases
                            #Neuron_states_all[neuron]=Neuron_states
                            Neuron_locations_all[neuron]=Neuron_locations


                        ###defining test arrays
                        Neuron_raw_test_test=Neuron_raw_eq_test_all[:,states_conc_test==test_state]
                        phases_conc_test_test=phases_conc_test[states_conc_test==test_state]
                        Location_raw_eq_test_test=Location_raw_eq_test[states_conc_test==test_state]
                        times_conc_test_test=times_conc_test[states_conc_test==test_state]
                        distances_conc_test_test=distances_conc_test[states_conc_test==test_state]

                        speed_raw_eq_test_test=speed_raw_eq_test[states_conc_test==test_state]
                        acceleration_raw_eq_test_test=acceleration_raw_eq_test[states_conc_test==test_state]

                        Location_raw_eq_test_nonan=Location_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                        speed_raw_eq_test_nonan=speed_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                        acceleration_raw_eq_test_nonan=acceleration_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                        times_conc_test_nonan=times_conc_test_test[~np.isnan(Location_raw_eq_test_test)]
                        distances_conc_test_nonan=distances_conc_test_test[~np.isnan(Location_raw_eq_test_test)]

                        coeffs_all_ses=np.zeros((num_neurons,num_regressors))
                        for neuron in np.arange(num_neurons):
                            Neuron_phases=Neuron_phases_all[neuron]
                            #Neuron_states=Neuron_states_all[neuron]
                            Neuron_locations=Neuron_locations_all[neuron]

                            Neuron_raw_eq_test=Neuron_raw_test_test[neuron]
                            Neuron_raw_eq_test_nonan=Neuron_raw_eq_test[~np.isnan(Location_raw_eq_test_test)]

                            FR_training_phases=Neuron_phases[phases_conc_test_test]
                            #FR_training_states=Neuron_states[states_conc_test_test]
                            FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test_test)]
                            #FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test_test)]
                            FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan).astype(int)]


                            ###regression
                            X = np.vstack((FR_training_phases_nonan,                                           FR_training_locations_nonan,                                           times_conc_test_nonan,                                           distances_conc_test_nonan,                                           speed_raw_eq_test_nonan,                                           acceleration_raw_eq_test_nonan)).T

                            X_clean=X[~np.isnan(X).any(axis=1)]
                            X_z=st.zscore(X_clean,axis=0)

                            y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
                            reg = LinearRegression().fit(X_z, y)
                            coeffs=reg.coef_
                            coeffs_all_ses[neuron]=coeffs



                            coeffs_all[neuron,ind_ses_state]=coeffs


                mean_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
                p_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
                mean_neuron_betas[:]=np.nan
                p_neuron_betas[:]=np.nan
                for neuron in np.arange(num_neurons):
                    mean_phase_coeff=np.nanmean(coeffs_all[neuron,:,0])
                    mean_place_coeff=np.nanmean(coeffs_all[neuron,:,1])

                    p_phase_coeff=st.ttest_1samp(remove_nan(coeffs_all[neuron,:,0]),0)[1]
                    p_place_coeff=st.ttest_1samp(remove_nan(coeffs_all[neuron,:,1]),0)[1]

                    mean_neuron_betas[neuron]=mean_phase_coeff,mean_place_coeff
                    p_neuron_betas[neuron]=p_phase_coeff,p_place_coeff

                GLM_dic_policy[direction]['coeffs_all'][mouse_recday]=coeffs_all
                GLM_dic_policy[direction]['mean_neuron_betas'][mouse_recday]=mean_neuron_betas
                GLM_dic_policy[direction]['p_neuron_betas'][mouse_recday]=p_neuron_betas


            except Exception as e:
                print('betas not calculated')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


# In[ ]:





# In[108]:


direction='Forward'
GLM_dic_policy[direction]['mean_neuron_betas'][mouse_recday]


# In[109]:


direction='Reverse'
GLM_dic_policy[direction]['mean_neuron_betas'][mouse_recday]


# In[ ]:





# In[114]:


###GLM - shuffled - Policy - across tasks (using left out state/task combination as test data)
tt=time.time()
num_phases=5
num_nodes=9
num_locations=21
num_states=4
num_regressors=6 ##phase, place, time (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place
num_iterations=100

lag_min=30*40 ###1200 bins = 30 seconds
smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1
state_bins=np.arange(num_states+1)

remove_edges=True

if remove_edges==True:
    num_locations=len(node_states)
    location_bins=np.arange(len(node_states)+1)
        
for direction in ['Forward','Reverse']:
    
    print(direction)
    print('')
    for day_type in ['3_task','combined_ABCDonly']:
        for mouse_recday in day_type_dicX[day_type]:
            print(mouse_recday)

            if len(GLM_dic_policy[direction]['thr_neuron_betas'][mouse_recday])>0:
                print('Already analysed')
                continue

            try: 
                awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
                awake_sessions=session_dic['awake'][mouse_recday]

                #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
                #np.arange(len(awake_sessions_behaviour))])

                num_sessions=len(awake_sessions_behaviour)

                num_neurons=len(cluster_dic['good_clus'][mouse_recday])
                sessions=Task_num_dic[mouse_recday]
                num_refses=len(np.unique(sessions))
                num_comparisons=num_refses-1
                repeat_ses=np.where(rank_repeat(sessions)>0)[0]
                non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

                coeffs_all=np.zeros((num_neurons,len(non_repeat_ses)*num_states,num_iterations,num_regressors))
                coeffs_all[:]=np.nan



                for ses_ind_test_ind,ses_ind_test in enumerate(non_repeat_ses):
                    print(ses_ind_test)
                    training_sessions=np.setdiff1d(non_repeat_ses,ses_ind_test)

                    ###Training
                    phases_conc_all_=[]
                    states_conc_all_=[]
                    Location_raw_eq_all_=[]
                    Neuron_raw_all_=[]
                    for ses_ind_training_ind, ses_ind_training in enumerate(training_sessions):
                        try:
                            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                            Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                            XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                            speed_raw=speed_dic[mouse_recday][ses_ind_training]


                            acceleration_raw_=np.diff(speed_raw)/0.025
                            acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                            Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')

                        except:
                            print('Files not found (training) for session '+str(ses_ind_training))
                            continue

                        Location_raw=Policy_raw=policy_dic[direction][mouse_recday][ses_ind_training]
                        phases=Phases_raw_dic[mouse_recday][ses_ind_training]
                        phases_conc=concatenate_complex2(concatenate_complex2(phases))
                        states=States_raw_dic[mouse_recday][ses_ind_training]
                        states_conc=concatenate_complex2(concatenate_complex2(states))
                        times=Times_from_reward_dic[mouse_recday][ses_ind_training]
                        times_conc=concatenate_complex2(concatenate_complex2(times))
                        distances=Distances_from_reward_dic[mouse_recday][ses_ind_training]
                        distances_conc=concatenate_complex2(concatenate_complex2(distances))
                        speed_raw_eq=speed_raw[:len(phases_conc)]
                        acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
                        Location_raw_eq=Location_raw[:len(phases_conc)]

                        if remove_edges==True:
                            Location_raw_eq[Location_raw_eq>len(node_states)-1]=np.nan ### removing edges

                        if len(phases_conc)>=len(speed_raw_eq):
                            print('Mismatch between speed and ephys - work around here but check')
                            phases_conc=phases_conc[:len(speed_raw_eq)]
                            states_conc=states_conc[:len(speed_raw_eq)]
                            times_conc=times_conc[:len(speed_raw_eq)]
                            distances_conc=distances_conc[:len(speed_raw_eq)]
                            Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]
                            Neuron_raw=Neuron_raw[:,:len(speed_raw_eq)]

                        speed_phases=st.binned_statistic(phases_conc, speed_raw_eq , bins=phase_bins)[0]
                        acceleration_phases=st.binned_statistic(phases_conc, acceleration_raw_eq , bins=phase_bins)[0]

                        phases_conc_all_.append(phases_conc)
                        states_conc_all_.append(states_conc)
                        Location_raw_eq_all_.append(Location_raw_eq)
                        Neuron_raw_all_.append(Neuron_raw)



                    phases_conc_all_=np.hstack((phases_conc_all_))
                    states_conc_all_=np.hstack((states_conc_all_))
                    Location_raw_eq_all_=np.hstack((Location_raw_eq_all_))
                    Neuron_raw_all_=np.hstack((Neuron_raw_all_))





                    ###Test
                    try:
                        Neuron_raw_test=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                        Location_raw_test=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                    except:
                        print('Files not found (test) for session '+str(ses_ind_test))
                        continue

                    Location_raw_test=Policy_raw_test=policy_dic[direction][mouse_recday][ses_ind_test]
                    speed_raw_test=speed_dic[mouse_recday][ses_ind_test]
                    acceleration_raw_test_=np.diff(speed_raw_test)/0.025
                    acceleration_raw_test=np.hstack((acceleration_raw_test_[0],acceleration_raw_test_))

                    phases_test=Phases_raw_dic[mouse_recday][ses_ind_test]
                    phases_conc_test=concatenate_complex2(concatenate_complex2(phases_test))
                    states_test=States_raw_dic[mouse_recday][ses_ind_test]
                    states_conc_test=concatenate_complex2(concatenate_complex2(states_test))
                    times_test=Times_from_reward_dic[mouse_recday][ses_ind_test]
                    times_conc_test=concatenate_complex2(concatenate_complex2(times_test))
                    distances_test=Distances_from_reward_dic[mouse_recday][ses_ind_test]
                    distances_conc_test=concatenate_complex2(concatenate_complex2(distances_test))

                    speed_raw_eq_test=gaussian_filter1d(speed_raw_test[:len(phases_conc_test)],smooth_SDs)
                    acceleration_raw_eq_test=gaussian_filter1d(acceleration_raw_test[:len(phases_conc_test)],smooth_SDs)
                    Location_raw_eq_test=Location_raw_test[:len(phases_conc_test)]

                    Neuron_raw_eq_test_all=Neuron_raw_test[:,:len(phases_conc_test)]

                    if remove_edges==True:
                        Location_raw_eq_test[Location_raw_eq_test>len(node_states)-1]=np.nan ### removing edges

                    if len(phases_conc_test)>=len(speed_raw_eq_test):
                        print('Mismatch between speed and ephys - work around here but check')
                        phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
                        states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
                        times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
                        distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
                        Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]
                        Neuron_raw_eq_test_all=Neuron_raw_eq_test_all[:,:len(speed_raw_eq_test)]


                    ###extracting test state  
                    states_=np.arange(num_states)
                    for test_state in np.arange(num_states):

                        ind_ses_state=(ses_ind_test_ind*num_states)+test_state

                        ###completing training arrays
                        phases_conc_test_training=phases_conc_test[states_conc_test!=test_state]
                        Location_raw_eq_test_training=Location_raw_eq_test[states_conc_test!=test_state]
                        Neuron_raw_test_training=Neuron_raw_eq_test_all[:,states_conc_test!=test_state]

                        phases_conc_all=np.hstack((phases_conc_all_,phases_conc_test_training))
                        Location_raw_eq_all=np.hstack((Location_raw_eq_all_,Location_raw_eq_test_training))
                        Neuron_raw_all=np.hstack((Neuron_raw_all_,Neuron_raw_test_training))


                        Neuron_phases_all=np.zeros((num_neurons,num_phases))
                        #Neuron_states_all=np.zeros((num_neurons,num_states))
                        Neuron_locations_all=np.zeros((num_neurons,num_locations))
                        Neuron_phases_all[:]=np.nan
                        Neuron_locations_all[:]=np.nan
                        #Neuron_states_all[:]=np.nan
                        for neuron in np.arange(num_neurons):
                            Neuron_raw_eq_all=Neuron_raw_all[neuron,:len(phases_conc_all)]
                            Neuron_phases=st.binned_statistic(phases_conc_all, Neuron_raw_eq_all, bins=phase_bins)[0]
                            #Neuron_states=st.binned_statistic(states_conc_all, Neuron_raw_eq_all, bins=state_bins)[0]
                            Neuron_locations=st.binned_statistic(Location_raw_eq_all, Neuron_raw_eq_all, bins=location_bins)[0]
                            Neuron_phases_all[neuron]=Neuron_phases
                            #Neuron_states_all[neuron]=Neuron_states
                            Neuron_locations_all[neuron]=Neuron_locations


                        ###defining test arrays
                        Neuron_raw_test_test=Neuron_raw_eq_test_all[:,states_conc_test==test_state]
                        phases_conc_test_test=phases_conc_test[states_conc_test==test_state]
                        Location_raw_eq_test_test=Location_raw_eq_test[states_conc_test==test_state]
                        times_conc_test_test=times_conc_test[states_conc_test==test_state]
                        distances_conc_test_test=distances_conc_test[states_conc_test==test_state]

                        speed_raw_eq_test_test=speed_raw_eq_test[states_conc_test==test_state]
                        acceleration_raw_eq_test_test=acceleration_raw_eq_test[states_conc_test==test_state]

                        Location_raw_eq_test_nonan=Location_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                        speed_raw_eq_test_nonan=speed_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                        acceleration_raw_eq_test_nonan=acceleration_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                        times_conc_test_nonan=times_conc_test_test[~np.isnan(Location_raw_eq_test_test)]
                        distances_conc_test_nonan=distances_conc_test_test[~np.isnan(Location_raw_eq_test_test)]

                        for neuron in np.arange(num_neurons):
                            Neuron_phases=Neuron_phases_all[neuron]
                            #Neuron_states=Neuron_states_all[neuron]
                            Neuron_locations=Neuron_locations_all[neuron]

                            Neuron_raw_eq_test=Neuron_raw_test_test[neuron]
                            Neuron_raw_eq_test_nonan=Neuron_raw_eq_test[~np.isnan(Location_raw_eq_test_test)]

                            FR_training_phases=Neuron_phases[phases_conc_test_test]
                            #FR_training_states=Neuron_states[states_conc_test_test]
                            FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test_test)]
                            #FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test_test)]
                            FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan).astype(int)]


                            ###regression
                            X = np.vstack((FR_training_phases_nonan,                                           FR_training_locations_nonan,                                           times_conc_test_nonan,                                           distances_conc_test_nonan,                                           speed_raw_eq_test_nonan,                                           acceleration_raw_eq_test_nonan)).T

                            X_clean=X[~np.isnan(X).any(axis=1)]
                            X_z=st.zscore(X_clean,axis=0)

                            y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
                            reg = LinearRegression().fit(X_z, y)
                            coeffs=reg.coef_

                            if len(y)<lag_min:
                                continue

                            ##Random shifts
                            max_roll=int(len(y)-lag_min)
                            min_roll=int(lag_min)

                            if max_roll<min_roll:
                                continue
                            for iteration in range(num_iterations):
                                copy_y=np.copy(y)

                                shift=random.randrange(max_roll-min_roll)+min_roll
                                y_shifted=np.roll(copy_y,shift)

                                reg = LinearRegression().fit(X_z, y_shifted)
                                coeffs=reg.coef_
                                coeffs_all[neuron,ind_ses_state,iteration]=coeffs


                thr_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
                thr_neuron_betas[:]=np.nan
                for neuron in np.arange(num_neurons):
                    thr_phase_coeff=np.nanpercentile(np.asarray([np.nanmean(coeffs_all[neuron,:,ii,0])                                                                 for ii in range(num_iterations)]),95)
                    thr_place_coeff=np.nanpercentile(np.asarray([np.nanmean(coeffs_all[neuron,:,ii,1])                                                                 for ii in range(num_iterations)]),95)
                    thr_neuron_betas[neuron]=thr_phase_coeff,thr_place_coeff


                GLM_dic_policy[direction]['thr_neuron_betas'][mouse_recday]=thr_neuron_betas

            except Exception as e:
                print('betas not calculated')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
print(time.time()-tt)


# In[112]:


GLM_dic_policy[direction]['thr_neuron_betas']['me10_17122021_19122021']
direction='Forward'
GLM_dic_policy[direction]['thr_neuron_betas']


# In[ ]:


'''

1-do glm with defintiion of policy as future action
2-make other definitions (past action - simply reverse some of the functions above)
'''


# In[125]:


GLM_dic_policy['Forward']['thr_neuron_betas'][mouse_recday][:,1]


# In[126]:


GLM_dic_policy['Reverse']['thr_neuron_betas'][mouse_recday][:,1]


# In[129]:


for day_type in ['combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        
        print(mouse_recday)
        print(np.nanmean(GLM_dic_policy['Forward']['mean_neuron_betas'][mouse_recday][:,1]))
        print(np.nanmean(GLM_dic_policy['Reverse']['mean_neuron_betas'][mouse_recday][:,1]))


# In[148]:


len(day_type_dicX[day_type])


# In[ ]:





# In[160]:


for day_type in ['3_task','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        
        print(mouse_recday)
        try:
            
            for direction in ['Forward','Reverse']:
                place_policy_coeff=GLM_dic_policy[direction]['mean_neuron_betas'][mouse_recday][:,1]
                place_policy_thr=GLM_dic_policy[direction]['thr_neuron_betas'][mouse_recday][:,1]

                p_value=GLM_dic_policy[direction]['p_neuron_betas'][mouse_recday][:,1]


                Place_policy_tuned_bool=np.logical_and(place_policy_coeff>place_policy_thr,p_value<0.05)
                
                exec('Place_policy_tuned_bool_'+direction+'=Place_policy_tuned_bool')
            Place_policy_tuned_bool_all=np.logical_or(Place_policy_tuned_bool_Forward,Place_policy_tuned_bool_Reverse)
            Tuned_dic['Place_policy'][mouse_recday]=Place_policy_tuned_bool_all
            Tuned_dic['Place_policy_forward'][mouse_recday]=Place_policy_tuned_bool_Forward
            Tuned_dic['Place_policy_reverse'][mouse_recday]=Place_policy_tuned_bool_Reverse
        
        except Exception as e:
            print('betas not calculated')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


# In[161]:


Place_policy_all=np.hstack((remove_empty(dict_to_array(Tuned_dic['Place_policy']))))
np.sum(Place_policy_all)/len(Place_policy_all)


# In[162]:


Place_policy_reverse=np.hstack((remove_empty(dict_to_array(Tuned_dic['Place_policy_reverse']))))
np.sum(Place_policy_reverse)/len(Place_policy_reverse)


# In[163]:


len(Place_policy_reverse)#*0.17


# In[ ]:





# In[523]:


for coeff_ind in np.arange(2):
    X_betas=np.vstack((remove_empty(dict_to_array(GLM_dic_policy['mean_neuron_betas']))))[:,coeff_ind]
    thr_betas=np.vstack((remove_empty(dict_to_array(GLM_dic_policy['thr_neuron_betas']))))[:,coeff_ind]

    plt.hist(X_betas,bins=50)
    plt.show()
    
    plt.hist(thr_betas,bins=50)
    plt.show()

    X_tuned_bool=X_betas>thr_betas
    print(np.sum(X_tuned_bool)/len(X_tuned_bool))


# In[ ]:





# In[104]:





# In[95]:


Tuned_ABCDAB_dic['AB']['State']#.keys()


# In[89]:


np.shape(tuning_singletrial_ABCDAB_dic2['AB']['tuning_state_boolean']['ah04_19122021_20122021'])


# In[426]:


day_type_dicX['combined_ABCDE']
Ephys_output_folder


# In[149]:


##Examples - pre analysis
import warnings; warnings.simplefilter('ignore')

mouse_recday='ab03_21112023_22112023'

#for mouse_recday in day_type_dicX['3_task']:
print(mouse_recday)

all_neurons=np.arange(len(cluster_dic['good_clus'][mouse_recday]))
abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

print(abstract_structures)
for neuron in np.arange(8)+30:
    print(neuron)

    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]

    All_sessions=session_dic['All'][mouse_recday]    
    awake_sessions=session_dic['awake'][mouse_recday]
    rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
    rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
    structure_nums=np.unique(rec_day_structure_numbers)

    fignamex=Ephys_output_folder_dropbox+'/Example_cells/'+mouse_recday+'_neuron_id'+str(neuron)+'_task'

    arrange_plot_statecells_persessionX2(mouse_recday,neuron,                                          Data_folder=Intermediate_object_folder_dropbox,                                         abstract_structures=abstract_structures,                                        plot=True, save=True, fignamex=fignamex, figtype='.svg',Marker=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


###GLM - ABCDE - across tasks/states (using left out state/task combination as test data)

GLM_ABCDE_dic=rec_dd()
num_phases=5
num_nodes=9
num_locations=21
num_states=4
num_regressors=6 ##phase, place, time (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place

smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1
state_bins=np.arange(num_states+1)

remove_edges=True
use_ABCDE_only=True

if remove_edges==True:
    num_locations=num_nodes
    location_bins=np.arange(num_nodes+1)+1
        

for day_type in ['combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        

        try: 
            awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
            awake_sessions=session_dic['awake'][mouse_recday]
            
            abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
            
            ABCDE_ses=np.where(abstract_structures=='ABCDE')[0]



            num_sessions=len(awake_sessions_behaviour)

            num_neurons=len(cluster_dic['good_clus'][mouse_recday])
            sessions=Task_num_dic[mouse_recday]
            num_refses=len(np.unique(sessions))
            num_comparisons=num_refses-1
            repeat_ses=np.where(rank_repeat(sessions)>0)[0]
            non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
            if use_ABCDE_only==True:
                non_repeat_ses=np.intersect1d(non_repeat_ses,ABCDE_ses)
                num_states=5

            coeffs_all=np.zeros((num_neurons,len(non_repeat_ses)*num_states,num_regressors))
            coeffs_all[:]=np.nan
            
            
            
            for ses_ind_test_ind,ses_ind_test in enumerate(non_repeat_ses):
                print(ses_ind_test)
                num_states=len(abstract_structures[ses_ind_test])
                training_sessions=np.setdiff1d(non_repeat_ses,ses_ind_test)
                
                ###Training
                phases_conc_all_=[]
                states_conc_all_=[]
                Location_raw_eq_all_=[]
                Neuron_raw_all_=[]
                for ses_ind_training_ind, ses_ind_training in enumerate(training_sessions):
                    try:
                        Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        speed_raw=speed_dic[mouse_recday][ses_ind_training]


                        acceleration_raw_=np.diff(speed_raw)/0.025
                        acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')

                    except:
                        print('Files not found for session '+str(ses_ind_training))
                        continue

                    phases=Phases_raw_dic[mouse_recday][ses_ind_training]
                    phases_conc=concatenate_complex2(concatenate_complex2(phases))
                    states=States_raw_dic[mouse_recday][ses_ind_training]
                    states_conc=concatenate_complex2(concatenate_complex2(states))
                    times=Times_from_reward_dic[mouse_recday][ses_ind_training]
                    times_conc=concatenate_complex2(concatenate_complex2(times))
                    distances=Distances_from_reward_dic[mouse_recday][ses_ind_training]
                    distances_conc=concatenate_complex2(concatenate_complex2(distances))
                    speed_raw_eq=speed_raw[:len(phases_conc)]
                    acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
                    Location_raw_eq=Location_raw[:len(phases_conc)]

                    if remove_edges==True:
                        Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges

                    if len(phases_conc)>len(speed_raw_eq):
                        print('Mismatch between speed and ephys - work around here but check')
                        phases_conc=phases_conc[:len(speed_raw_eq)]
                        states_conc=states_conc[:len(speed_raw_eq)]
                        times_conc=times_conc[:len(speed_raw_eq)]
                        distances_conc=distances_conc[:len(speed_raw_eq)]
                        Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]
                        Neuron_raw=Neuron_raw[:,:len(speed_raw_eq)]

                    speed_phases=st.binned_statistic(phases_conc, speed_raw_eq , bins=phase_bins)[0]
                    acceleration_phases=st.binned_statistic(phases_conc, acceleration_raw_eq , bins=phase_bins)[0]
                    
                    phases_conc_all_.append(phases_conc)
                    states_conc_all_.append(states_conc)
                    Location_raw_eq_all_.append(Location_raw_eq)
                    Neuron_raw_all_.append(Neuron_raw)
                    


                phases_conc_all_=np.hstack((phases_conc_all_))
                states_conc_all_=np.hstack((states_conc_all_))
                Location_raw_eq_all_=np.hstack((Location_raw_eq_all_))
                Neuron_raw_all_=np.hstack((Neuron_raw_all_))
                
                



                ###Test
                try:
                    Neuron_raw_test=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                    Location_raw_test=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                except:
                    print('Files not found for session '+str(ses_ind_test))
                    continue
                speed_raw_test=speed_dic[mouse_recday][ses_ind_test]
                acceleration_raw_test_=np.diff(speed_raw_test)/0.025
                acceleration_raw_test=np.hstack((acceleration_raw_test_[0],acceleration_raw_test_))

                phases_test=Phases_raw_dic[mouse_recday][ses_ind_test]
                phases_conc_test=concatenate_complex2(concatenate_complex2(phases_test))
                states_test=States_raw_dic[mouse_recday][ses_ind_test]
                states_conc_test=concatenate_complex2(concatenate_complex2(states_test))
                times_test=Times_from_reward_dic[mouse_recday][ses_ind_test]
                times_conc_test=concatenate_complex2(concatenate_complex2(times_test))
                distances_test=Distances_from_reward_dic[mouse_recday][ses_ind_test]
                distances_conc_test=concatenate_complex2(concatenate_complex2(distances_test))

                speed_raw_eq_test=gaussian_filter1d(speed_raw_test[:len(phases_conc_test)],smooth_SDs)
                acceleration_raw_eq_test=gaussian_filter1d(acceleration_raw_test[:len(phases_conc_test)],smooth_SDs)
                Location_raw_eq_test=Location_raw_test[:len(phases_conc_test)]
                
                Neuron_raw_eq_test_all=Neuron_raw_test[:,:len(phases_conc_test)]
                
                if remove_edges==True:
                    Location_raw_eq_test[Location_raw_eq_test>num_nodes]=np.nan ### removing edges

                if len(phases_conc_test)>len(speed_raw_eq_test):
                    print('Mismatch between speed and ephys - work around here but check')
                    phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
                    states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
                    times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
                    distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
                    Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]
                    Neuron_raw_eq_test_all=Neuron_raw_eq_test_all[:,:len(speed_raw_eq_test)]
                
                
                ###extracting test state  
                states_=np.arange(num_states)
                for test_state in np.arange(num_states):
                    
                    ind_ses_state=(ses_ind_test_ind*num_states)+test_state

                    ###completing training arrays
                    phases_conc_test_training=phases_conc_test[states_conc_test!=test_state]
                    Location_raw_eq_test_training=Location_raw_eq_test[states_conc_test!=test_state]
                    Neuron_raw_test_training=Neuron_raw_eq_test_all[:,states_conc_test!=test_state]

                    phases_conc_all=np.hstack((phases_conc_all_,phases_conc_test_training))
                    Location_raw_eq_all=np.hstack((Location_raw_eq_all_,Location_raw_eq_test_training))
                    Neuron_raw_all=np.hstack((Neuron_raw_all_,Neuron_raw_test_training))
                    
                    
                    Neuron_phases_all=np.zeros((num_neurons,num_phases))
                    #Neuron_states_all=np.zeros((num_neurons,num_states))
                    Neuron_locations_all=np.zeros((num_neurons,num_locations))
                    Neuron_phases_all[:]=np.nan
                    Neuron_locations_all[:]=np.nan
                    #Neuron_states_all[:]=np.nan
                    for neuron in np.arange(num_neurons):
                        Neuron_raw_eq_all=Neuron_raw_all[neuron,:len(phases_conc_all)]
                        Neuron_phases=st.binned_statistic(phases_conc_all, Neuron_raw_eq_all, bins=phase_bins)[0]
                        #Neuron_states=st.binned_statistic(states_conc_all, Neuron_raw_eq_all, bins=state_bins)[0]
                        Neuron_locations=st.binned_statistic(Location_raw_eq_all, Neuron_raw_eq_all, bins=location_bins)[0]
                        Neuron_phases_all[neuron]=Neuron_phases
                        #Neuron_states_all[neuron]=Neuron_states
                        Neuron_locations_all[neuron]=Neuron_locations
                    
                    
                    ###defining test arrays
                    Neuron_raw_test_test=Neuron_raw_eq_test_all[:,states_conc_test==test_state]
                    phases_conc_test_test=phases_conc_test[states_conc_test==test_state]
                    Location_raw_eq_test_test=Location_raw_eq_test[states_conc_test==test_state]
                    times_conc_test_test=times_conc_test[states_conc_test==test_state]
                    distances_conc_test_test=distances_conc_test[states_conc_test==test_state]
                    
                    speed_raw_eq_test_test=speed_raw_eq_test[states_conc_test==test_state]
                    acceleration_raw_eq_test_test=acceleration_raw_eq_test[states_conc_test==test_state]

                    Location_raw_eq_test_nonan=Location_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    speed_raw_eq_test_nonan=speed_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    acceleration_raw_eq_test_nonan=acceleration_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    times_conc_test_nonan=times_conc_test_test[~np.isnan(Location_raw_eq_test_test)]
                    distances_conc_test_nonan=distances_conc_test_test[~np.isnan(Location_raw_eq_test_test)]

                    coeffs_all_ses=np.zeros((num_neurons,num_regressors))
                    for neuron in np.arange(num_neurons):
                        Neuron_phases=Neuron_phases_all[neuron]
                        #Neuron_states=Neuron_states_all[neuron]
                        Neuron_locations=Neuron_locations_all[neuron]

                        Neuron_raw_eq_test=Neuron_raw_test_test[neuron]
                        Neuron_raw_eq_test_nonan=Neuron_raw_eq_test[~np.isnan(Location_raw_eq_test_test)]

                        FR_training_phases=Neuron_phases[phases_conc_test_test]
                        #FR_training_states=Neuron_states[states_conc_test_test]
                        FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test_test)]
                        #FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test_test)]
                        FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan-1).astype(int)]


                        ###regression
                        X = np.vstack((FR_training_phases_nonan,                                       FR_training_locations_nonan,                                       times_conc_test_nonan,                                       distances_conc_test_nonan,                                       speed_raw_eq_test_nonan,                                       acceleration_raw_eq_test_nonan)).T

                        X_clean=X[~np.isnan(X).any(axis=1)]
                        X_z=st.zscore(X_clean,axis=0)

                        y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
                        reg = LinearRegression().fit(X_z, y)
                        coeffs=reg.coef_
                        coeffs_all_ses[neuron]=coeffs
                        
                        

                        coeffs_all[neuron,ind_ses_state]=coeffs


            mean_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            p_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            mean_neuron_betas[:]=np.nan
            p_neuron_betas[:]=np.nan
            for neuron in np.arange(num_neurons):
                mean_phase_coeff=np.nanmean(coeffs_all[neuron,:,0])
                mean_place_coeff=np.nanmean(coeffs_all[neuron,:,1])

                p_phase_coeff=st.ttest_1samp(remove_nan(coeffs_all[neuron,:,0]),0)[1]
                p_place_coeff=st.ttest_1samp(remove_nan(coeffs_all[neuron,:,1]),0)[1]

                mean_neuron_betas[neuron]=mean_phase_coeff,mean_place_coeff
                p_neuron_betas[neuron]=p_phase_coeff,p_place_coeff

            GLM_ABCDE_dic['coeffs_all'][mouse_recday]=coeffs_all
            GLM_ABCDE_dic['mean_neuron_betas'][mouse_recday]=mean_neuron_betas
            GLM_ABCDE_dic['p_neuron_betas'][mouse_recday]=p_neuron_betas


        except Exception as e:
            print('betas not calculated')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


# In[166]:


GLM_ABCDE_dic['p_neuron_betas']


# In[ ]:





# In[ ]:





# In[50]:


###GLM - shuffled - across tasks (using left out state/task combination as test data)
tt=time.time()
num_phases=5
num_nodes=9
num_locations=21
num_states=4
num_regressors=6 ##phase, place, time (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place
num_iterations=100

lag_min=30*40 ###1200 bins = 30 seconds
smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1
state_bins=np.arange(num_states+1)

remove_edges=True
use_ABCDE_only=True

if remove_edges==True:
    num_locations=num_nodes
    location_bins=np.arange(num_nodes+1)+1
        

for day_type in ['combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        
        #if mouse_recday not in ['me10_16122021','me08_06092021_09092021','me10_09122021_10122021',
        #                        'me11_05122021_06122021']:
        #    continue

        try: 
            awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
            awake_sessions=session_dic['awake'][mouse_recday]
            
            abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
            ABCDE_ses=np.where(abstract_structures=='ABCDE')[0]

            #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
            #np.arange(len(awake_sessions_behaviour))])

            num_sessions=len(awake_sessions_behaviour)

            num_neurons=len(cluster_dic['good_clus'][mouse_recday])
            sessions=Task_num_dic[mouse_recday]
            num_refses=len(np.unique(sessions))
            num_comparisons=num_refses-1
            repeat_ses=np.where(rank_repeat(sessions)>0)[0]
            non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
            
            if use_ABCDE_only==True:
                non_repeat_ses=np.intersect1d(non_repeat_ses,ABCDE_ses)
                num_states=5

            coeffs_all=np.zeros((num_neurons,len(non_repeat_ses)*num_states,num_iterations,num_regressors))
            coeffs_all[:]=np.nan
            
            
            
            for ses_ind_test_ind,ses_ind_test in enumerate(non_repeat_ses):
                print(ses_ind_test)
                num_states=len(abstract_structures[ses_ind_test])
                training_sessions=np.setdiff1d(non_repeat_ses,ses_ind_test)
                
                ###Training
                phases_conc_all_=[]
                states_conc_all_=[]
                Location_raw_eq_all_=[]
                Neuron_raw_all_=[]
                for ses_ind_training_ind, ses_ind_training in enumerate(training_sessions):
                    try:
                        Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')
                        speed_raw=speed_dic[mouse_recday][ses_ind_training]


                        acceleration_raw_=np.diff(speed_raw)/0.025
                        acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_training)+'.npy')

                    except:
                        print('Files not found (1) for session '+str(ses_ind_training))
                        continue

                    phases=Phases_raw_dic[mouse_recday][ses_ind_training]
                    phases_conc=concatenate_complex2(concatenate_complex2(phases))
                    states=States_raw_dic[mouse_recday][ses_ind_training]
                    states_conc=concatenate_complex2(concatenate_complex2(states))
                    times=Times_from_reward_dic[mouse_recday][ses_ind_training]
                    times_conc=concatenate_complex2(concatenate_complex2(times))
                    distances=Distances_from_reward_dic[mouse_recday][ses_ind_training]
                    distances_conc=concatenate_complex2(concatenate_complex2(distances))
                    speed_raw_eq=speed_raw[:len(phases_conc)]
                    acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
                    Location_raw_eq=Location_raw[:len(phases_conc)]

                    if remove_edges==True:
                        Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges

                    if len(phases_conc)>len(speed_raw_eq):
                        print('Mismatch between speed and ephys - work around here but check')
                        phases_conc=phases_conc[:len(speed_raw_eq)]
                        states_conc=states_conc[:len(speed_raw_eq)]
                        times_conc=times_conc[:len(speed_raw_eq)]
                        distances_conc=distances_conc[:len(speed_raw_eq)]
                        Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]
                        Neuron_raw=Neuron_raw[:,:len(speed_raw_eq)]

                    speed_phases=st.binned_statistic(phases_conc, speed_raw_eq , bins=phase_bins)[0]
                    acceleration_phases=st.binned_statistic(phases_conc, acceleration_raw_eq , bins=phase_bins)[0]
                    
                    phases_conc_all_.append(phases_conc)
                    states_conc_all_.append(states_conc)
                    Location_raw_eq_all_.append(Location_raw_eq)
                    Neuron_raw_all_.append(Neuron_raw)
                    


                phases_conc_all_=np.hstack((phases_conc_all_))
                states_conc_all_=np.hstack((states_conc_all_))
                Location_raw_eq_all_=np.hstack((Location_raw_eq_all_))
                Neuron_raw_all_=np.hstack((Neuron_raw_all_))
                
                



                ###Test
                try:
                    Neuron_raw_test=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                    Location_raw_test=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind_test)+'.npy')
                except:
                    print('Files not found (2) for session '+str(ses_ind_test))
                    continue
                speed_raw_test=speed_dic[mouse_recday][ses_ind_test]
                acceleration_raw_test_=np.diff(speed_raw_test)/0.025
                acceleration_raw_test=np.hstack((acceleration_raw_test_[0],acceleration_raw_test_))

                phases_test=Phases_raw_dic[mouse_recday][ses_ind_test]
                phases_conc_test=concatenate_complex2(concatenate_complex2(phases_test))
                states_test=States_raw_dic[mouse_recday][ses_ind_test]
                states_conc_test=concatenate_complex2(concatenate_complex2(states_test))
                times_test=Times_from_reward_dic[mouse_recday][ses_ind_test]
                times_conc_test=concatenate_complex2(concatenate_complex2(times_test))
                distances_test=Distances_from_reward_dic[mouse_recday][ses_ind_test]
                distances_conc_test=concatenate_complex2(concatenate_complex2(distances_test))

                speed_raw_eq_test=gaussian_filter1d(speed_raw_test[:len(phases_conc_test)],smooth_SDs)
                acceleration_raw_eq_test=gaussian_filter1d(acceleration_raw_test[:len(phases_conc_test)],smooth_SDs)
                Location_raw_eq_test=Location_raw_test[:len(phases_conc_test)]
                
                Neuron_raw_eq_test_all=Neuron_raw_test[:,:len(phases_conc_test)]
                
                if remove_edges==True:
                    Location_raw_eq_test[Location_raw_eq_test>num_nodes]=np.nan ### removing edges

                if len(phases_conc_test)>len(speed_raw_eq_test):
                    print('Mismatch between speed and ephys - work around here but check')
                    phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
                    states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
                    times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
                    distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
                    Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]
                    Neuron_raw_eq_test_all=Neuron_raw_eq_test_all[:,:len(speed_raw_eq_test)]
                
                
                ###extracting test state  
                states_=np.arange(num_states)
                for test_state in np.arange(num_states):
                    
                    ind_ses_state=(ses_ind_test_ind*num_states)+test_state

                    ###completing training arrays
                    phases_conc_test_training=phases_conc_test[states_conc_test!=test_state]
                    Location_raw_eq_test_training=Location_raw_eq_test[states_conc_test!=test_state]
                    Neuron_raw_test_training=Neuron_raw_eq_test_all[:,states_conc_test!=test_state]

                    phases_conc_all=np.hstack((phases_conc_all_,phases_conc_test_training))
                    Location_raw_eq_all=np.hstack((Location_raw_eq_all_,Location_raw_eq_test_training))
                    Neuron_raw_all=np.hstack((Neuron_raw_all_,Neuron_raw_test_training))
                    
                    
                    Neuron_phases_all=np.zeros((num_neurons,num_phases))
                    #Neuron_states_all=np.zeros((num_neurons,num_states))
                    Neuron_locations_all=np.zeros((num_neurons,num_locations))
                    Neuron_phases_all[:]=np.nan
                    Neuron_locations_all[:]=np.nan
                    #Neuron_states_all[:]=np.nan
                    for neuron in np.arange(num_neurons):
                        Neuron_raw_eq_all=Neuron_raw_all[neuron,:len(phases_conc_all)]
                        Neuron_phases=st.binned_statistic(phases_conc_all, Neuron_raw_eq_all, bins=phase_bins)[0]
                        #Neuron_states=st.binned_statistic(states_conc_all, Neuron_raw_eq_all, bins=state_bins)[0]
                        Neuron_locations=st.binned_statistic(Location_raw_eq_all, Neuron_raw_eq_all, bins=location_bins)[0]
                        Neuron_phases_all[neuron]=Neuron_phases
                        #Neuron_states_all[neuron]=Neuron_states
                        Neuron_locations_all[neuron]=Neuron_locations
                    
                    
                    ###defining test arrays
                    Neuron_raw_test_test=Neuron_raw_eq_test_all[:,states_conc_test==test_state]
                    phases_conc_test_test=phases_conc_test[states_conc_test==test_state]
                    Location_raw_eq_test_test=Location_raw_eq_test[states_conc_test==test_state]
                    times_conc_test_test=times_conc_test[states_conc_test==test_state]
                    distances_conc_test_test=distances_conc_test[states_conc_test==test_state]
                    
                    speed_raw_eq_test_test=speed_raw_eq_test[states_conc_test==test_state]
                    acceleration_raw_eq_test_test=acceleration_raw_eq_test[states_conc_test==test_state]

                    Location_raw_eq_test_nonan=Location_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    speed_raw_eq_test_nonan=speed_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    acceleration_raw_eq_test_nonan=acceleration_raw_eq_test_test[~np.isnan(Location_raw_eq_test_test)]
                    times_conc_test_nonan=times_conc_test_test[~np.isnan(Location_raw_eq_test_test)]
                    distances_conc_test_nonan=distances_conc_test_test[~np.isnan(Location_raw_eq_test_test)]

                    for neuron in np.arange(num_neurons):
                        Neuron_phases=Neuron_phases_all[neuron]
                        #Neuron_states=Neuron_states_all[neuron]
                        Neuron_locations=Neuron_locations_all[neuron]

                        Neuron_raw_eq_test=Neuron_raw_test_test[neuron]
                        Neuron_raw_eq_test_nonan=Neuron_raw_eq_test[~np.isnan(Location_raw_eq_test_test)]

                        FR_training_phases=Neuron_phases[phases_conc_test_test]
                        #FR_training_states=Neuron_states[states_conc_test_test]
                        FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test_test)]
                        #FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test_test)]
                        FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan-1).astype(int)]


                        ###regression
                        X = np.vstack((FR_training_phases_nonan,                                       FR_training_locations_nonan,                                       times_conc_test_nonan,                                       distances_conc_test_nonan,                                       speed_raw_eq_test_nonan,                                       acceleration_raw_eq_test_nonan)).T

                        X_clean=X[~np.isnan(X).any(axis=1)]
                        X_z=st.zscore(X_clean,axis=0)

                        y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
                        reg = LinearRegression().fit(X_z, y)
                        coeffs=reg.coef_

                        if len(y)<lag_min:
                            continue
                        
                        ##Random shifts
                        max_roll=int(len(y)-lag_min)
                        min_roll=int(lag_min)
                        
                        if max_roll<min_roll:
                            continue
                        for iteration in range(num_iterations):
                            copy_y=np.copy(y)

                            shift=random.randrange(max_roll-min_roll)+min_roll
                            y_shifted=np.roll(copy_y,shift)

                            reg = LinearRegression().fit(X_z, y_shifted)
                            coeffs=reg.coef_
                            coeffs_all[neuron,ind_ses_state,iteration]=coeffs


            thr_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
            thr_neuron_betas[:]=np.nan
            for neuron in np.arange(num_neurons):
                thr_phase_coeff=np.nanpercentile(np.asarray([np.nanmean(coeffs_all[neuron,:,ii,0])                                                             for ii in range(num_iterations)]),95)
                thr_place_coeff=np.nanpercentile(np.asarray([np.nanmean(coeffs_all[neuron,:,ii,1])                                                             for ii in range(num_iterations)]),95)
                thr_neuron_betas[neuron]=thr_phase_coeff,thr_place_coeff


            GLM_ABCDE_dic['thr_neuron_betas'][mouse_recday]=thr_neuron_betas

        except Exception as e:
            print('betas not calculated')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


for coeff_ind in np.arange(2):
    X_betas=np.vstack((dict_to_array(GLM_ABCDE_dic['mean_neuron_betas'])))[:,coeff_ind]
    thr_betas=np.vstack((dict_to_array(GLM_ABCDE_dic['thr_neuron_betas'])))[:,coeff_ind]

    plt.hist(X_betas,bins=50)
    plt.show()
    
    plt.hist(thr_betas,bins=50)
    plt.show()

    X_tuned_bool=X_betas>thr_betas
    print(np.sum(X_tuned_bool)/len(X_tuned_bool))


# In[ ]:





# In[ ]:





# In[51]:


###Tuning from single trials ABCD ABCDE
tt=time.time()
tuning_singletrial_ABCDE_dic=rec_dd()
#num_states=4
num_phases=3
for day_type in ['combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        
        #Importing Ephys
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])

        sessions=Task_num_dic[mouse_recday]
        num_refses=len(np.unique(sessions))
        num_comparisons=num_refses-1
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        num_nonrepeat_sessions=len(non_repeat_ses)
        
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]



        for session in np.arange(num_sessions):
            print(session)
            try:
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
                exec('ephys_ses_'+str(session)+'_=ephys_')

            except Exception as e:
                print(e)
                exec('ephys_ses_'+str(session)+'_=[]')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('Not calculated')

        ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
        ABCDE_sessions=np.where(abstract_structures=='ABCDE')[0]
        abstract_structure='ABCDE'
        ses_array=ABCDE_sessions        
        num_states=len(abstract_structure)

        tuning_z_matrix=np.zeros((len(ses_array),num_neurons))
        tuning_p_matrix=np.zeros((len(ses_array),num_neurons))
        tuning_z_matrix_allstates=np.zeros((len(ses_array),num_neurons,num_states))
        tuning_p_matrix_allstates=np.zeros((len(ses_array),num_neurons,num_states))
        tuning_z_matrix_allphases=np.zeros((len(ses_array),num_neurons,num_phases))
        tuning_p_matrix_allphases=np.zeros((len(ses_array),num_neurons,num_phases))
        for ses_ind, session in enumerate(ses_array):
            exec('ephys_=ephys_ses_'+str(session)+'_')
            if len(ephys_)==0:
                continue
            for neuron in np.arange(num_neurons):


                if len(ephys_)==0:
                    tuning_z_matrix[ses_ind][neuron]=np.nan
                    tuning_p_matrix[ses_ind][neuron]=np.nan
                    continue

                ephys_neuron_unbinned_=ephys_[neuron]
                ephys_neuron_unbinned=np.asarray(np.split(ephys_neuron_unbinned_,num_states,axis=1))
                ephys_neuron=np.mean(np.split(ephys_neuron_unbinned,10,axis=2),axis=0)
                z_max=st.zscore(np.nanmax(ephys_neuron,axis=2),axis=0)
                z_max_prefstate=z_max[np.argmax(np.mean(z_max,axis=1))]
                tuning_z_matrix[ses_ind][neuron]=np.nanmean(z_max_prefstate)
                tuning_p_matrix[ses_ind][neuron]=st.ttest_1samp(remove_nan(z_max_prefstate),0)[1]

                tuning_z_matrix_allstates[ses_ind][neuron]=np.nanmean(z_max,axis=1)
                tuning_p_matrix_allstates[ses_ind][neuron]=np.asarray([st.ttest_1samp(remove_nan(z_max[ii]),0)[1]                                                            for ii in range(len(z_max))])


                ##Phase peaks
                ephys_neuron_3=np.asarray(np.split(ephys_neuron_unbinned,3,axis=2))
                max_phase=np.max(np.mean(np.mean(ephys_neuron_3,axis=1),axis=1),axis=1)
                z_max_phase=st.zscore(max_phase)

                #tuning_z_matrix_allphases[ses_ind][neuron]=np.nanmean(z_max_phase,axis=0)
                #tuning_p_matrix_allphases[ses_ind][neuron]=np.asarray([st.ttest_1samp(remove_nan(
                #z_max_phase[:,ii]),0)\
                #                                                       [1] for ii in range(len(z_max_phase.T))])
                tuning_z_matrix_allphases[ses_ind][neuron]=z_max_phase

                ##replace ttests with permutation tests

        tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_z'][mouse_recday]=tuning_z_matrix
        tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_p'][mouse_recday]=tuning_p_matrix

        tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_z_allstates'][mouse_recday]=        tuning_z_matrix_allstates
        tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_p_allstates'][mouse_recday]=        tuning_p_matrix_allstates

        tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_z_allphases'][mouse_recday]=        tuning_z_matrix_allphases
        #tuning_singletrial_ABCDAB_dic2[abstract_structure]['tuning_p_allphases'][mouse_recday]=\
        #tuning_p_matrix_allphases

        

print(time.time()-tt)


# In[52]:


###Tuning booleans for states and phases
#num_states=4 
num_phases=3 
 
p_thr=0.05 ###to account for occasional low num of trials 
for day_type in ['combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday) 
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday]) 
        num_neurons=len(cluster_dic['good_clus'][mouse_recday]) 
        sessions=Task_num_dic[mouse_recday]
        num_refses=len(np.unique(sessions))
        num_comparisons=num_refses-1
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        num_nonrepeat_sessions=len(non_repeat_ses)
        
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
        ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
        ABCDE_sessions=np.where(abstract_structures=='ABCDE')[0]
        abstract_structure='ABCDE'
        ses_array=ABCDE_sessions
                
        num_states=len(abstract_structure)

        peak_boolean_all_states=np.zeros((len(ses_array),num_neurons,num_states)) 
        peak_boolean_all_phases=np.zeros((len(ses_array),num_neurons,num_phases)) 
        peak_boolean_all_phases_max=np.zeros((len(ses_array),num_neurons,num_phases)) 
        for ses_ind, session in enumerate(ses_array): 
            print(ses_ind)

            z_ses_state=tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_z_allstates'][mouse_recday][ses_ind] 
            if len(z_ses_state)==0:
                print('No trials detected')
                continue

            p_ses_state=tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_p_allstates'][mouse_recday][ses_ind] 

            z_ses_phase=tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_z_allphases'][mouse_recday][ses_ind] 
            #p_ses_phase=tuning_singletrial_dic2['tuning_p_allphases'][mouse_recday][ses_ind] 


            for neuron in np.arange(num_neurons): 
                peak_boolean_all_states[ses_ind][neuron]=np.logical_and(z_ses_state[neuron]>0,                                                                        p_ses_state[neuron]<=p_thr) 
                #peak_boolean_all_phases[ses_ind][neuron]=np.logical_and(z_ses_phase[neuron]>0,\
                #                                                        p_ses_phase[neuron]<=p_thr) 
                peak_boolean_all_phases_max[ses_ind][neuron]=z_ses_phase[neuron]==np.max(z_ses_phase[neuron]) 

        tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_state_boolean'][mouse_recday]=        peak_boolean_all_states 
        #tuning_singletrial_dic2['tuning_phase_boolean'][mouse_recday]=peak_boolean_all_phases 
        tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_phase_boolean_max'][mouse_recday]=        peak_boolean_all_phases_max 


# In[ ]:





# In[53]:


###computing state tuning using per trial zscore - ABCDE
Tuned_ABCDE_dic=rec_dd()
##paramaters
num_bins=90
#num_states=4 replaced below by seperate value per session
num_phases=3
num_nodes=9
num_lags=12
smoothing_sigma=10
num_iterations=100

for day_type in ['combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
        awake_sessions=session_dic['awake'][mouse_recday]
        num_sessions=len(awake_sessions_behaviour)
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        #regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]
        
        
        if mouse_recday in ['ah04_19122021_20122021','me11_15122021_16122021']:
            addition='_'
        else:
            addition=''
        
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
        
        ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
        ABCDE_sessions=np.where(abstract_structures=='ABCDE')[0]
        abstract_structure='ABCDE'
        ses_array=ABCDE_sessions
        
        


        found_ses=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            try:
                Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+addition+'.npy')
                found_ses.append(ses_ind)

            except:
                print('Files not found for session '+str(ses_ind))
                continue
        num_non_repeat_ses_found=len(found_ses)
        
        ses_array=np.intersect1d(ses_array,found_ses)


        
        zmax_all=np.zeros((num_neurons,len(ses_array)))
        zmax_all[:]=np.nan

        zmax_all_strict=np.zeros((num_neurons,len(ses_array)))
        zmax_all_strict[:]=np.nan

        corr_mean_max_all=np.zeros((num_neurons,len(ses_array),2))
        corr_mean_max_all[:]=np.nan

        for ses_ind_ind, ses_ind in enumerate(ses_array):
            ses_ind_actual=found_ses[ses_ind_ind]

            num_states=len(abstract_structures[ses_ind_ind])
            phase_norm_mean=np.tile(np.repeat(np.arange(num_phases),num_bins/num_phases),num_states)
            phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states,num_bins))


            Actual_activity_ses_=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind_actual)+                                         addition+'.npy')
            Actual_activity_ses=Actual_activity_ses_.T
            #GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]

            phase_peaks=tuning_singletrial_ABCDE_dic[abstract_structure]['tuning_phase_boolean_max']            [mouse_recday][ses_ind_ind]

            if len(phase_peaks)==0:
                continue
            pref_phase_neurons=np.argmax(phase_peaks,axis=1)
            phases=Phases_raw_dic2[mouse_recday][ses_ind_actual]
            phases_conc=concatenate_complex2(concatenate_complex2(phases))

            Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+addition+                                '.npy')
            Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25

            for neuron in np.arange(num_neurons):
                pref_phase=pref_phase_neurons[neuron]
                Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]

                Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                        num_states=len(abstract_structure),smoothing=False,return_mean=False)

                Actual_norm_means=np.vstack(([[np.nanmean(Actual_norm[trial,num_bins*ii:num_bins*(ii+1)]                    [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                                  for trial in np.arange(len(Actual_norm))]))
                max_state=np.argmax(np.nanmean(Actual_norm_means,axis=0))
                zactivity_prefstate=st.zscore(Actual_norm_means,axis=1)[:,max_state]
                zmax_all[neuron,ses_ind_ind]=st.ttest_1samp(zactivity_prefstate,0)[1]

                zmax_shifted=np.zeros(num_iterations)
                zmax_shifted[:]=np.nan
                for iteration in range(num_iterations):
                    shifts=np.random.randint(0,num_states,len(Actual_norm_means))
                    Actual_norm_means_shifted=indep_roll(Actual_norm_means,shifts)
                    max_state=np.argmax(np.nanmean(Actual_norm_means_shifted,axis=0))
                    zactivity_prefstate=st.zscore(Actual_norm_means_shifted,axis=1)[:,max_state]
                    zactivity_prefstate_mean=np.nanmean(zactivity_prefstate)
                    zmax_shifted[iteration]=zactivity_prefstate_mean
                mean_zmax_shifted=np.nanmean(zmax_shifted)

                zmax_all_strict[neuron,ses_ind_ind]=st.ttest_1samp(zactivity_prefstate,mean_zmax_shifted)[1]

                Actual_norm_max=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                        num_states=len(abstract_structure),smoothing=False,return_mean=False,take_max=True)

                Actual_norm_max_means=np.vstack(([[np.nanmean(Actual_norm_max[trial,num_bins*ii:num_bins*(ii+1)]                    [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                                  for trial in np.arange(len(Actual_norm_max))]))

                r_,p_=st.pearsonr(np.concatenate(Actual_norm_means),np.concatenate(Actual_norm_max_means))
                corr_mean_max_all[neuron,ses_ind_ind]=[r_,p_]

        Tuned_ABCDE_dic[abstract_structure]['State_zmax'][mouse_recday]=zmax_all
        Tuned_ABCDE_dic[abstract_structure]['State_zmax_strict'][mouse_recday]=zmax_all_strict
        Tuned_ABCDE_dic[abstract_structure]['corr_mean_max'][mouse_recday]=corr_mean_max_all


# In[54]:


day_type='combined_ABCDE'
abstract_structure='ABCDE'
for mouse_recday in day_type_dicX[day_type]:
    num_peaks_all=np.vstack(([np.sum(tuning_singletrial_ABCDE_dic[abstract_structure]                                     ['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)                              for ses_ind in np.arange(len(tuning_singletrial_ABCDE_dic[abstract_structure]                                                           ['tuning_state_boolean'][mouse_recday]))]))

    state_bool=np.sum(num_peaks_all>0,axis=0)>(len(num_peaks_all)//2)

    State_zmax=Tuned_ABCDE_dic[abstract_structure]['State_zmax'][mouse_recday]

    state_bool_zmax=np.sum(State_zmax<0.05,axis=1)>(len(State_zmax.T)/3)
    state_bool_zmax_one=np.sum(State_zmax<0.05,axis=1)>1

    Tuned_ABCDE_dic[abstract_structure]['State_zmax_bool'][mouse_recday]=state_bool_zmax
    Tuned_ABCDE_dic[abstract_structure]['State_zmax_bool_one'][mouse_recday]=state_bool_zmax_one


# In[ ]:





# In[55]:



###Making spatial maps
Place_tuned_ABCDE_dic=rec_dd()
for day_type in ['combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        non_repeat_ses=non_repeat_ses_maker(mouse_recday)

        Neurons_raw_0=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_0.npy')

        FR_maps=np.zeros((len(Neurons_raw_0),len(non_repeat_ses),3,3))
        FR_maps[:]=np.nan
        for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
            try:
                Neurons_raw_=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Location_=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            except:
                print(str(ses_ind)+'Not found')
                continue

            Location_=Location_[:len(Neurons_raw_.T)]
            Neurons_raw_=Neurons_raw_[:,:len(Location_)]

            nodes=(Location_[np.logical_and(Location_<=9,~np.isnan(Location_))]-1).astype(int)

            for neuron in np.arange(len(Neurons_raw_)):
                neuron_=Neurons_raw_[neuron]
                neuron_nodes=neuron_[np.logical_and(Location_<=9,~np.isnan(Location_))]
                FR_vector=st.binned_statistic(nodes,neuron_nodes,bins=np.arange(10))[0]
                FR_map=np.zeros((3,3))
                FR_map[:]=np.nan

                for ind in np.arange(len(FR_vector)):
                    FR_map[Task_grid[ind,0],Task_grid[ind,1]]=FR_vector[ind]
                FR_maps[neuron,ses_ind_ind]=FR_map

        cross_ses_corr=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((FR_maps[neuron][ses_ind]))                                                            for ses_ind in np.arange(len(FR_maps[neuron]))]))))        for neuron in np.arange(len(FR_maps))]))

        min_crosscorr=np.min(cross_ses_corr,axis=1)
        corr_min_thr=min_crosscorr>0
        p_values=np.hstack(([st.ttest_1samp(cross_ses_corr[neuron],0)[1] for neuron in np.arange(len(cross_ses_corr))]))
        corr_p_bool=p_values<0.05

        Place_tuned_ABCDE_dic[mouse_recday]=corr_p_bool
        
        
########






###Making phase maps
Phase_tuned_ABCDE_dic=rec_dd()
num_phases=5 
for mouse_recday in day_type_dicX['combined_ABCDE']:
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)

    Neurons_raw_0=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_0.npy')

    FR_vectors=np.zeros((len(Neurons_raw_0),len(non_repeat_ses),num_phases))
    FR_vectors[:]=np.nan
    for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
        Neurons_raw_=np.load(Intermediate_object_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
        #Location_=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
        Phases_=np.hstack((np.hstack((Phases_raw_dic[mouse_recday][ses_ind]))))

        Phases_=Phases_[:len(Neurons_raw_.T)]
        Neurons_raw_=Neurons_raw_[:,:len(Phases_)]


        for neuron in np.arange(len(Neurons_raw_)):
            neuron_=Neurons_raw_[neuron]
            FR_vector=st.binned_statistic(Phases_,neuron_,bins=np.arange(num_phases+1))[0]

            FR_vectors[neuron,ses_ind_ind]=FR_vector

    cross_ses_corr=np.vstack(([matrix_triangle(np.corrcoef(np.vstack(([np.hstack((FR_vectors[neuron][ses_ind]))                                                        for ses_ind in np.arange(len(FR_vectors[neuron]))]))))    for neuron in np.arange(len(FR_vectors))]))

    min_crosscorr=np.min(cross_ses_corr,axis=1)
    corr_min_thr=min_crosscorr>0
    p_values=np.hstack(([st.ttest_1samp(cross_ses_corr[neuron],0)[1] for neuron in np.arange(len(cross_ses_corr))]))
    corr_p_bool=p_values<0.05
    
    Phase_tuned_ABCDE_dic[mouse_recday]=corr_p_bool
    
    


# In[ ]:





# In[383]:


Phase_tuned_ABCDE_dic[mouse_recday]


# In[388]:


Place_tuned_ABCDE_dic[mouse_recday]


# In[387]:


Phase_tuned_ABCDE_dic[mouse_recday]


# In[ ]:





# In[60]:


###Summarising and Plotting tuning ABCDE 
threshold_state_sessions=0


use_permuted=True ###for phase and place tuning
use_both=True ###for phase and place tuning - use both permutation and ttest
use_zscored=True ##for state tuning
lowest_thr=False

if use_both==True:
    use_permuted=False

# for abstract_structure in ['ABCD','AB']:
abstract_structure='ABCDE'
print('')
print(abstract_structure)
print('_____')
phase_bool_all=[]
state_bool_all=[]
place_bool_all=[]
phase_place_bool_all=[]
for day_type in ['combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:#list(GLM_dic['mean_neuron_betas'].keys()):
        print(mouse_recday)
        try:
            mean_neuron_betas=GLM_ABCDE_dic['mean_neuron_betas'][mouse_recday]
            p_neuron_betas=GLM_ABCDE_dic['p_neuron_betas'][mouse_recday]
            thr_neuron_beta=GLM_ABCDE_dic['thr_neuron_betas'][mouse_recday]

            #mean_neuron_betas_withintask=GLM_withinTask_dic[abstract_structure]['mean_neuron_betas'][mouse_recday]
            #thr_neuron_betas_withintask=GLM_withinTask_dic[abstract_structure]['thr_neuron_betas'][mouse_recday]


            
            #phase_bool_ttest=np.logical_and(mean_neuron_betas[:,0]>0,p_neuron_betas[:,0]<0.05)
            #place_bool_ttest=np.logical_and(mean_neuron_betas[:,1]>0,p_neuron_betas[:,1]<0.05)
            #phase_place_bool_ttest=np.logical_and(mean_neuron_betas[:,2]>0,p_neuron_betas[:,2]<0.05)
            
            phase_bool2=Phase_tuned_ABCDE_dic[mouse_recday]
            place_bool2=Place_tuned_ABCDE_dic[mouse_recday]
            
            #phase_bool2=Phase_tuned_dic[mouse_recday]
            #place_bool2=Place_tuned_dic[mouse_recday]

            phase_bool_permutation=mean_neuron_betas[:,0]>thr_neuron_beta[:,0]
            place_bool_permutation=mean_neuron_betas[:,1]>thr_neuron_beta[:,1]

            if use_permuted==True:
                phase_bool=phase_bool_permutation
                place_bool=place_bool_permutation
                #phase_place_bool=mean_neuron_betas[:,2]>thr_neuron_beta[:,2]
                try:
                    state_within_bool=mean_neuron_betas_withintask[:,1]>thr_neuron_betas_withintask[:,1]
                    Tuned_dic['State_withintask'][mouse_recday]=state_within_bool
                except:
                    print('within task state tuning not calculated')
            elif use_both==True:
                phase_bool=np.logical_and(phase_bool_permutation,phase_bool2)
                place_bool=np.logical_and(place_bool_permutation,place_bool2)

            else:
                phase_bool=phase_bool_ttest
                place_bool=place_bool_ttest
                #phase_place_bool=phase_place_bool_ttest


            num_peaks_all=np.vstack(([np.sum(tuning_singletrial_ABCDE_dic[abstract_structure]                                             ['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)                    for ses_ind in np.arange(len(tuning_singletrial_ABCDE_dic[abstract_structure]                                                 ['tuning_state_boolean'][mouse_recday]))]))

            if use_zscored==True:
                if lowest_thr==True:
                    state_bool=Tuned_ABCDE_dic[abstract_structure]['State_zmax_bool_one'][mouse_recday]

                else:
                    state_bool=Tuned_ABCDE_dic[abstract_structure]['State_zmax_bool'][mouse_recday]

            else:
                if threshold_state_sessions=='half':
                    state_bool=np.sum(num_peaks_all>0,axis=0)>(len(num_peaks_all)//2)
                else:
                    state_bool=np.sum(num_peaks_all>0,axis=0)>threshold_state_sessions
                    
            if use_both==True:
                Tuned_ABCDE_dic[abstract_structure]['Phase_strict'][mouse_recday]=phase_bool
                Tuned_ABCDE_dic[abstract_structure]['State_strict'][mouse_recday]=                Tuned_ABCDE_dic[abstract_structure]['State_zmax_strict_bool'][mouse_recday]
                Tuned_ABCDE_dic[abstract_structure]['Place_strict'][mouse_recday]=place_bool
                
            
            else:
                Tuned_ABCDE_dic[abstract_structure]['Phase'][mouse_recday]=phase_bool
                Tuned_ABCDE_dic[abstract_structure]['State'][mouse_recday]=                Tuned_ABCDE_dic[abstract_structure]['State_zmax_bool'][mouse_recday]
                Tuned_ABCDE_dic[abstract_structure]['Place'][mouse_recday]=place_bool
                #Tuned_dic['Phase_Place'][mouse_recday]=place_bool


            #Tuned_dic['Phase_ttest'][mouse_recday]=phase_bool_ttest
            #Tuned_dic['Place_ttest'][mouse_recday]=place_bool_ttest


            phase_and_place_bool=np.logical_and(phase_bool,place_bool)
            state_place_bool=np.logical_and(phase_bool,state_bool)


            print(np.sum(phase_bool)/len(phase_bool))
            print(np.sum(state_bool)/len(state_bool))
            print(np.sum(place_bool)/len(place_bool))
            #print(np.sum(phase_place_bool)/len(phase_place_bool))
            print('')
            #print(np.sum(phase_state_bool)/np.sum(phase_bool))
            #print(np.sum(phase_and_place_bool)/np.sum(phase_bool))
            print('')

            phase_bool_all.append(phase_bool)
            state_bool_all.append(state_bool)
            place_bool_all.append(place_bool)
            #phase_place_bool_all.append(phase_place_bool)
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('Not calculated')
            print('')

exec('phase_bool_all_'+abstract_structure+'=phase_bool_all')
exec('state_bool_all_'+abstract_structure+'=state_bool_all')
exec('place_bool_all_'+abstract_structure+'=place_bool_all')
exec('phase_place_bool_all_'+abstract_structure+'=phase_place_bool_all')

    
    
#####


exec('phase_bool_all=phase_bool_all_'+abstract_structure)
exec('state_bool_all=state_bool_all_'+abstract_structure)
exec('place_bool_all=place_bool_all_'+abstract_structure)
exec('phase_place_bool_all=phase_place_bool_all_'+abstract_structure)

phase_bool_all=np.concatenate(phase_bool_all)
state_bool_all=np.concatenate(state_bool_all)
place_bool_all=np.concatenate(place_bool_all)
#phase_place_bool_all=np.concatenate(phase_place_bool_all)


print('phase cells: '+str(np.sum(phase_bool_all)/len(phase_bool_all)))
print('state cells: '+str(np.sum(state_bool_all)/len(state_bool_all)))
print('place cells: '+str(np.sum(place_bool_all)/len(place_bool_all)))

print('')

phase_state_bool=np.logical_and(phase_bool_all,state_bool_all)
print('state cells as proportion of phase cells: '+str(np.sum(phase_state_bool)/np.sum(phase_bool_all)))

phase_place_bool=np.logical_and(phase_bool_all,place_bool_all)
#print('place cells as proportion of phase cells: '+str(np.sum(phase_place_bool)/np.sum(phase_bool_all)))

phase_state_unique=np.logical_and(phase_state_bool,~phase_place_bool)
phase_place_unique=np.logical_and(phase_place_bool,~phase_state_bool)
#print('pure state cells as proportion of phase cells: '+str(np.sum(phase_state_unique)/np.sum(phase_bool_all)))
#print('pure place cells as proportion of phase cells: '+str(np.sum(phase_place_unique)/np.sum(phase_bool_all)))

print('')

#phase_place_or_state=np.logical_or(phase_place_bool,phase_state_bool)
#print('place or state cells as proportion of phase cells: '+str(np.sum(phase_place_or_state)/np.sum(phase_bool_all)))

phase_place_and_state=np.logical_and(phase_place_bool,phase_state_bool)
print('place cells as proportion of phase/state cells: '+str(np.sum(phase_place_and_state)/np.sum(phase_state_bool)))


state_phase_bool=np.logical_and(phase_bool_all,state_bool_all)
print('state cells as proportion of phase cells:: '+str(np.sum(state_phase_bool)/np.sum(state_bool_all)))



####


###Plotting GLM results
import upsetplot
all_cells=np.arange(len(phase_bool_all))
phase_cells=np.where(phase_bool_all==True)[0]
state_cells=np.where(state_bool_all==True)[0]
place_cells=np.where(place_bool_all==True)[0]


non_phase_cells=np.where(phase_bool_all==False)[0]
non_state_cells=np.where(state_bool_all==False)[0]
non_place_cells=np.where(place_bool_all==False)[0]

nothing_bool_all=np.logical_and(np.logical_and(~phase_bool_all,~state_bool_all),~place_bool_all)
nothing_cells=np.where(nothing_bool_all==True)[0]

data=upsetplot.from_memberships([['Phase'],                                 ['Phase','State'],                                 ['Phase','Non_State'],                                 ['Phase','State','Non_Place'],                  ['Phase','State','Place'],[],['Non_Phase','State']],                 data=np.asarray([len(phase_cells),                                  len(np.intersect1d(state_cells,phase_cells)),                                  len(np.intersect1d(non_state_cells,phase_cells)),                                  len(np.intersect1d(np.intersect1d(state_cells,non_place_cells),phase_cells)),                                  len(np.intersect1d(np.intersect1d(state_cells,place_cells),phase_cells)),                                 len(nothing_cells),                                 len(np.intersect1d(state_cells,non_phase_cells))])/len(all_cells))
#print(data)

upsetplot.UpSet(data, sort_by='degree',sort_categories_by='cardinality')

upsetplot.plot(data)
#plt.savefig(Ephys_output_folder_dropbox+'_UpsetPlot_cells.svg')
plt.show()


# In[61]:


prop_statephase_ABCDE=np.sum(state_phase_bool)/np.sum(state_bool_all)
vals=[np.sum(state_phase_bool), np.sum(state_bool_all)-np.sum(state_phase_bool)]

colors=['lightgreen','grey']
plt.pie(vals, colors=colors)
plt.savefig(Ephys_output_folder_dropbox+'All_phase_pie_ABCDE_ABCD.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()

print('Number of state neurons: '+str(np.sum(state_bool_all)))
print('Proportion state_phase: '+str(prop_statephase_ABCDE))

print(two_proportions_test(np.sum(state_phase_bool), np.sum(state_bool_all),                       int(np.sum(state_bool_all)*0.05), np.sum(state_bool_all)))


# In[ ]:


###here


# In[389]:


for day_type in ['combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        Goal_progress_strict=Tuned_ABCDE_dic['Phase_strict'][mouse_recday]
        Place_strict=Tuned_ABCDE_dic['Place_strict'][mouse_recday]
        State_strict=Tuned_ABCDE_dic['State_strict'][mouse_recday]
        
        
        
        Place=Tuned_ABCDE_dic['Place'][mouse_recday]
        Goal_progress=Tuned_ABCDE_dic['Phase'][mouse_recday]
        State=Tuned_ABCDE_dic['State'][mouse_recday]
        
        for measure, arrayX in {'Place':Place,'Goal_progress':Goal_progress,'State':State,                'Place_strict':Place_strict,'Goal_progress_strict':Goal_progress_strict,                                'State_strict':State_strict}.items():
            np.save(Intermediate_object_folder_dropbox+measure+'_'+mouse_recday+'.npy',arrayX)
        


# In[ ]:





# In[ ]:


'''
Train on ABCD test on ABCDE (probably reverse is better) - reuse ABCD-AB code
plot % phase
compare betas across ABCDE only vs ABCD-ABCDE

use ABCDE only to plot % phase of state


'''


# In[ ]:





# In[ ]:





# In[24]:


###GLM - train on ABCDE test on ABCD

GLM_ABCD_ABCDE_dic=rec_dd()
num_phases=5
num_nodes=9
num_locations=21
#num_states=4 defined below
num_regressors=6 ##phase, place, time (from reward), distance (from reward), speed, acceleration
num_regressors_interest=2 ##phase, place

smooth_SDs=5

phase_bins=np.arange(num_phases+1)
location_bins=np.arange(num_locations+1)+1


abstract_structure_training='ABCDE'
abstract_structure_test='ABCD'

remove_edges=True
num_iterations=100
lag_min=30*40 ###1200 bins = 30 seconds
smooth_SDs=5

if remove_edges==True:
    num_locations=num_nodes
    location_bins=np.arange(num_nodes+1)+1
        

day_type = 'combined_ABCDE'

for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)


    try:
        awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
        awake_sessions=session_dic['awake'][mouse_recday]

        #ephys_found_booleean=np.asarray([awake_sessions_behaviour[ii] in awake_sessions for ii in \
        #np.arange(len(awake_sessions_behaviour))])

        num_sessions=len(awake_sessions_behaviour)

        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        sessions=Task_num_dic[mouse_recday]
        num_refses=len(np.unique(sessions))
        num_comparisons=num_refses-1
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

        training_sessions=np.intersect1d(np.where(abstract_structures==abstract_structure_training)[0],                                   np.arange(len(non_repeat_ses)))      
        test_sessions=np.intersect1d(np.where(abstract_structures==abstract_structure_test)[0],                                     np.arange(len(non_repeat_ses)))
        
        if len(training_sessions)==0 or len(test_sessions)==0:
            print('Recording day doesnt have both abstract structure types')
            continue
        
        coeffs_all=np.zeros((num_neurons,len(non_repeat_ses),len(non_repeat_ses),num_regressors))
        coeffs_all[:]=np.nan
        ###Training


        phases_conc_all=[]
        states_conc_all=[]
        times_conc_all=[]
        distances_conc_all=[]
        Location_raw_eq_all=[]
        Neuron_raw_eq_all=[]
        speed_raw_eq_all=[]
        acceleration_raw_eq_all=[]
        for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
            num_states_training=len(abstract_structure_training)
            state_bins_training=np.arange(num_states_training+1)
            try:
                Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                speed_raw=speed_dic[mouse_recday][ses_ind]
                acceleration_raw_=np.diff(speed_raw)/0.025
                acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')

            except:
                
                
                print('Trying Ceph')
                try:
                    Neuron_raw=np.load(Intermediate_object_folder_ceph+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    Location_raw=np.load(Intermediate_object_folder_ceph+'Location_raw_'+mouse_recday                                         +'_'+str(ses_ind)+'.npy')
                    XY_raw=np.load(Intermediate_object_folder_ceph+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    speed_raw=speed_dic[mouse_recday][ses_ind]
                    acceleration_raw_=np.diff(speed_raw)/0.025
                    acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
                    Trial_times=np.load(Intermediate_object_folder_ceph+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                except:
                    print('Files not found for session '+str(ses_ind))
                    continue

            phases=Phases_raw_dic[mouse_recday][ses_ind]
            phases_conc=concatenate_complex2(concatenate_complex2(phases))
            states=States_raw_dic[mouse_recday][ses_ind]
            states_conc=concatenate_complex2(concatenate_complex2(states))
            times=Times_from_reward_dic[mouse_recday][ses_ind]
            times_conc=concatenate_complex2(concatenate_complex2(times))
            distances=Distances_from_reward_dic[mouse_recday][ses_ind]
            distances_conc=concatenate_complex2(concatenate_complex2(distances))
            speed_raw_eq=speed_raw[:len(phases_conc)]
            acceleration_raw_eq=acceleration_raw[:len(phases_conc)]
            Location_raw_eq=Location_raw[:len(phases_conc)]

            if remove_edges==True:
                Location_raw_eq[Location_raw_eq>num_nodes]=np.nan ### removing edges

            if len(phases_conc)>=len(speed_raw_eq):
                print('Mismatch between speed and ephys - work around here but check')
                phases_conc=phases_conc[:len(speed_raw_eq)]
                states_conc=states_conc[:len(speed_raw_eq)]
                times_conc=times_conc[:len(speed_raw_eq)]
                distances_conc=distances_conc[:len(speed_raw_eq)]
                Location_raw_eq=Location_raw_eq[:len(speed_raw_eq)]

            phases_conc_all.append(phases_conc)
            states_conc_all.append(states_conc)
            times_conc_all.append(times_conc)
            distances_conc_all.append(distances_conc)
            Location_raw_eq_all.append(Location_raw_eq)
            speed_raw_eq_all.append(speed_raw_eq)
            acceleration_raw_eq_all.append(acceleration_raw_eq)
            Neuron_raw_eq_all.append(Neuron_raw[:,:len(speed_raw_eq)])

        phases_conc_all=np.asarray(phases_conc_all,dtype=object)
        states_conc_all=np.asarray(states_conc_all,dtype=object)
        times_conc_all=np.asarray(times_conc_all,dtype=object)
        distances_conc_all=np.asarray(distances_conc_all,dtype=object)
        Location_raw_eq_all=np.asarray(Location_raw_eq_all,dtype=object)
        speed_raw_eq_all=np.asarray(speed_raw_eq_all,dtype=object)
        acceleration_raw_eq_all=np.asarray(acceleration_raw_eq_all,dtype=object)


        phases_conc_training=np.hstack((phases_conc_all[training_sessions]))
        states_conc_training=np.hstack((states_conc_all[training_sessions]))
        times_conc_training=np.hstack((times_conc_all[training_sessions]))
        distances_conc_training=np.hstack((distances_conc_all[training_sessions]))
        Location_raw_eq_training=np.hstack((Location_raw_eq_all[training_sessions]))
        speed_raw_eq_training=np.hstack((speed_raw_eq_all[training_sessions]))
        acceleration_raw_eq_training=np.hstack((acceleration_raw_eq_all[training_sessions]))
        Neuron_raw_eq_training=np.hstack(([Neuron_raw_eq_all[ses_ind] for ses_ind in training_sessions]))





        speed_phases=st.binned_statistic(phases_conc_training, speed_raw_eq_training , bins=phase_bins)[0]
        acceleration_phases=st.binned_statistic(phases_conc_training, acceleration_raw_eq_training , bins=phase_bins)[0]

        Neuron_phases_training=np.zeros((num_neurons,num_phases))
        Neuron_states_training=np.zeros((num_neurons,num_states_training))
        Neuron_locations_training=np.zeros((num_neurons,num_locations))
        Neuron_phases_training[:]=np.nan
        Neuron_locations_training[:]=np.nan
        Neuron_states_training[:]=np.nan
        for neuron in np.arange(num_neurons):
            Neuron_raw_eq=Neuron_raw_eq_training[neuron]
            Neuron_phases=st.binned_statistic(phases_conc_training, Neuron_raw_eq , bins=phase_bins)[0]
            Neuron_states=st.binned_statistic(states_conc_training, Neuron_raw_eq , bins=state_bins_training)[0]
            Neuron_locations=st.binned_statistic(Location_raw_eq_training, Neuron_raw_eq , bins=location_bins)[0]
            #Neuron_states=st.binned_statistic(Location_raw_eq, Neuron_raw_eq , bins=location_bins)[0]
            Neuron_phases_training[neuron]=Neuron_phases
            Neuron_states_training[neuron]=Neuron_states
            Neuron_locations_training[neuron]=Neuron_locations



        ###Test

        phases_conc_test=np.hstack((phases_conc_all[test_sessions]))
        states_conc_test=np.hstack((states_conc_all[test_sessions]))
        times_conc_test=np.hstack((times_conc_all[test_sessions]))
        distances_conc_test=np.hstack((distances_conc_all[test_sessions]))
        Location_raw_eq_test=np.hstack((Location_raw_eq_all[test_sessions]))
        speed_raw_eq_test=np.hstack((speed_raw_eq_all[test_sessions]))
        acceleration_raw_eq_test=np.hstack((acceleration_raw_eq_all[test_sessions]))
        Neuron_raw_eq_test=np.hstack(([Neuron_raw_eq_all[ses_ind] for ses_ind in test_sessions]))


        if remove_edges==True:
            Location_raw_eq_test[Location_raw_eq_test>num_nodes]=np.nan ### removing edges

        if len(phases_conc_test)>len(speed_raw_eq_test):
            print('Mismatch between speed and ephys - work around here but check')
            phases_conc_test=phases_conc_test[:len(speed_raw_eq_test)]
            states_conc_test=states_conc_test[:len(speed_raw_eq_test)]
            times_conc_test=times_conc_test[:len(speed_raw_eq_test)]
            distances_conc_test=distances_conc_test[:len(speed_raw_eq_test)]
            Location_raw_eq_test=Location_raw_eq_test[:len(speed_raw_eq_test)]




        Location_raw_eq_test_nonan=Location_raw_eq_test[~np.isnan(Location_raw_eq_test)]
        speed_raw_eq_test_nonan=speed_raw_eq_test[~np.isnan(Location_raw_eq_test)]
        acceleration_raw_eq_test_nonan=acceleration_raw_eq_test[~np.isnan(Location_raw_eq_test)]
        times_conc_test_nonan=times_conc_test[~np.isnan(Location_raw_eq_test)]
        distances_conc_test_nonan=distances_conc_test[~np.isnan(Location_raw_eq_test)]



        ###regression
        coeffs_all=np.zeros((num_neurons,num_regressors))
        coeffs_shuff_all=np.zeros((num_neurons,num_iterations,num_regressors))
        
        coeffs_all[:]=np.nan
        coeffs_shuff_all[:]=np.nan
        
        for neuron in np.arange(num_neurons):
            #print(neuron)
            Neuron_phases=Neuron_phases_training[neuron]
            Neuron_states=Neuron_states_training[neuron]
            Neuron_locations=Neuron_locations_training[neuron]

            Neuron_raw_eq_test_=Neuron_raw_eq_test[neuron,:len(phases_conc_test)]
            Neuron_raw_eq_test_nonan=Neuron_raw_eq_test_[~np.isnan(Location_raw_eq_test)]

            FR_training_phases=Neuron_phases[phases_conc_test]
            FR_training_states=Neuron_states[states_conc_test]
            FR_training_phases_nonan=FR_training_phases[~np.isnan(Location_raw_eq_test)]
            FR_training_states_nonan=FR_training_states[~np.isnan(Location_raw_eq_test)]
            FR_training_locations_nonan=Neuron_locations[(Location_raw_eq_test_nonan-1).astype(int)]


            ###regression
            X = np.vstack((FR_training_phases_nonan,                           FR_training_locations_nonan,                           times_conc_test_nonan,                           distances_conc_test_nonan,                           speed_raw_eq_test_nonan,                           acceleration_raw_eq_test_nonan)).T

            X_clean=X[~np.isnan(X).any(axis=1)]
            X_z=st.zscore(X_clean,axis=0)

            y=Neuron_raw_eq_test_nonan[~np.isnan(X).any(axis=1)]
            reg = LinearRegression().fit(X_z, y)
            coeffs=reg.coef_

            coeffs_all[neuron]=coeffs
            
            
            
            if len(y)<lag_min:
                continue
            
            
            
            
            ##Permutations
            max_roll=int(len(y)-lag_min)
            min_roll=int(lag_min)

            if max_roll<min_roll:
                continue
            for iteration in range(num_iterations):
                copy_y=np.copy(y)

                shift=random.randrange(max_roll-min_roll)+min_roll
                y_shifted=np.roll(copy_y,shift)

                reg = LinearRegression().fit(X_z, y_shifted)
                coeffs=reg.coef_
                coeffs_shuff_all[neuron,iteration]=coeffs


        thr_neuron_betas=np.zeros((num_neurons,num_regressors_interest))
        thr_neuron_betas[:]=np.nan
        for neuron in np.arange(num_neurons):
            thr_phase_coeff=np.nanpercentile(np.asarray([np.nanmean(coeffs_shuff_all[neuron,ii,0])                                                         for ii in range(num_iterations)]),95)
            thr_place_coeff=np.nanpercentile(np.asarray([np.nanmean(coeffs_shuff_all[neuron,ii,1])                                                         for ii in range(num_iterations)]),95)
            thr_neuron_betas[neuron]=thr_phase_coeff,thr_place_coeff

            
        GLM_ABCD_ABCDE_dic['coeffs_all'][mouse_recday]=coeffs_all
        GLM_ABCD_ABCDE_dic['thr_neuron_betas'][mouse_recday]=thr_neuron_betas

        
        
    except Exception as e:
        print('betas not calculated')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


# In[36]:


GLM_ABCD_ABCDE_dic['coeffs_all'].keys()


# In[25]:


mouse_recday='ab03_21112023_22112023'
phase_coeffs=np.hstack(([GLM_ABCD_ABCDE_dic['coeffs_all'][mouse_recday][:,0] for mouse_recday inGLM_ABCD_ABCDE_dic['coeffs_all'].keys()]))
phase_thrs=np.hstack(([GLM_ABCD_ABCDE_dic['thr_neuron_betas'][mouse_recday][:,0] for mouse_recday inGLM_ABCD_ABCDE_dic['coeffs_all'].keys()]))

phase_bool_ABCDE_ABCD=phase_coeffs>phase_thrs
prop_phase_ABCDE_ABCD=np.sum(phase_bool_ABCDE_ABCD)/len(phase_bool_ABCDE_ABCD)
print('Proportion phase: '+str(prop_phase_ABCDE_ABCD))

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.hist(phase_coeffs,bins=10,color='grey')
plt.axvline(0,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'Phase_betas_ABCDvsABCDE.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
st.ttest_1samp(phase_coeffs,0)


# In[26]:


vals=[np.sum(phase_bool_ABCDE_ABCD), len(phase_bool_ABCDE_ABCD)-np.sum(phase_bool_ABCDE_ABCD)]

colors=['lightgreen','grey']
plt.pie(vals, colors=colors)
plt.savefig(Ephys_output_folder_dropbox+'All_phase_pie_ABCDE_ABCD.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()

print('Number of neurons: '+str(len(phase_bool_ABCDE_ABCD)))
print('Proportion phase: '+str(prop_phase_ABCDE_ABCD))

print(two_proportions_test(np.sum(phase_bool_ABCDE_ABCD), len(phase_bool_ABCDE_ABCD),                       int(len(phase_bool_ABCDE_ABCD)*0.05), len(phase_bool_ABCDE_ABCD)))


# In[349]:


for measure in Tuned_dic.keys():
    for mouse_recday in Tuned_dic[measure].keys():
        np.save(Intermediate_object_folder_dropbox+measure+'_'+mouse_recday+'.npy',               Tuned_dic[measure][mouse_recday])


# In[362]:


np.sum(Tuned_dic2['State']['95']['ab03_01092023_02092023'])


# In[ ]:





# In[356]:


for measure in Tuned_dic2.keys():
    for percentile in Tuned_dic2[measure].keys():
        for mouse_recday in Tuned_dic2[measure][percentile].keys():
            np.save(Intermediate_object_folder_dropbox+measure+'_'+percentile+mouse_recday+'.npy',                   Tuned_dic2[measure][percentile][mouse_recday])


# In[365]:


mouse_recday='ab03_01092023_02092023'
xx=np.load(Intermediate_object_folder_dropbox+'State_95'+mouse_recday+'.npy')

np.sum(xx)


# In[ ]:





# In[27]:


###Saving dictionary outputs as npy files     

for measure in Tuned_dic.keys():
    for mouse_recday in Tuned_dic[measure].keys():
        np.save(Intermediate_object_folder_dropbox+measure+'_'+mouse_recday+'.npy',               Tuned_dic[measure][mouse_recday])
        
for day_type in ['3_task','combined']:
    for mouse_recday in day_type_dicX[day_type]:
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]
        for ses_ind in np.arange(len(awake_sessions)):
            
            for name, dictionaryX in {'Phases_raw2_':Phases_raw_dic2, 'Phases_raw_':Phases_raw_dic,                                     'States_raw_':States_raw_dic,'Times_from_reward_':Times_from_reward_dic,                                     'Distances_from_reward_':Distances_from_reward_dic,                                     'Times_from_start_':Times_from_start_dic}.items():
                np.save(Intermediate_object_folder_dropbox+name+mouse_recday+'_'+str(ses_ind)+'.npy',                          dictionaryX[mouse_recday][ses_ind])
                
                        
for measure in tuning_singletrial_dic2.keys():
    for mouse_recday in tuning_singletrial_dic2[measure].keys():
        np.save(Intermediate_object_folder_dropbox+measure+'_'+mouse_recday+'.npy',               tuning_singletrial_dic2[measure][mouse_recday])    
        
for mouse_recday in speed_dic.keys():
    for ses_ind in speed_dic[mouse_recday].keys():
        np.save(Intermediate_object_folder_dropbox+'speed_'+mouse_recday+'_'+str(ses_ind)+'.npy',               speed_dic[mouse_recday][ses_ind])
        
        


# In[350]:


for measure in tuning_singletrial_dic2.keys():
    for mouse_recday in tuning_singletrial_dic2[measure].keys():
        np.save(Intermediate_object_folder_dropbox+measure+'_'+mouse_recday+'.npy',               tuning_singletrial_dic2[measure][mouse_recday]) 


# In[41]:


tuning_singletrial_dic2.keys()


# In[42]:


num_neurons_all=0
for day_type in ['combined_ABCDonly','3_task']:
    
    for mouse_recday in day_type_dicX[day_type]:
        ses_ind=0
        num_neuronsX=len(cluster_dic['good_clus'][mouse_recday])
        num_neuronsY=len(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy'))
        print(mouse_recday)
        print(num_neuronsX)
        print(num_neuronsY)
        
        num_neurons_all+=num_neuronsX

print(num_neurons_all)


# In[43]:


Num_trials_dic['ABCD'].keys()


# In[44]:


###Number of neurons
total_N_neurons_=[]
for day_type in ['3_task','combined_ABCDonly']:
    print(day_type)
    num_neurons_=np.sum([Num_neurons_dic[mouse_recday] for mouse_recday in day_type_dicX[day_type]])
    num_days_=len(day_type_dicX[day_type])
    print(num_neurons_)
    print(num_days_)
    
    total_N_neurons_.append(num_neurons_)
print('Total neurons seperately sorted')
print(np.sum(total_N_neurons_))

print('Total neurons')
print(np.sum([Num_neurons_dic[mouse_recday] for mouse_recday in day_type_dicX['3_task_all']]))


# In[348]:


xx=np.hstack((day_type_dicX['3_task'],day_type_dicX['combined_ABCDonly']))

yy=np.hstack((list(GLM_dic2['mean_neuron_betas'].keys())))


# In[351]:


len(yy)


# In[ ]:





# In[ ]:


for abstract_structure in Tuned_ABCDAB_dic.keys():
    for measure in Tuned_ABCDAB_dic[abstract_structure].keys():
        for mouse_recday in Tuned_ABCDAB_dic[abstract_structure][measure].keys():
            np.save(Intermediate_object_folder_dropbox+abstract_structure+'_'+measure+'_'+mouse_recday+'.npy',                   Tuned_ABCDAB_dic[abstract_structure][measure][mouse_recday])
            

for abstract_structure in tuning_singletrial_ABCDAB_dic2.keys():
    for measure in tuning_singletrial_ABCDAB_dic2[abstract_structure].keys():
        for mouse_recday in tuning_singletrial_ABCDAB_dic2[abstract_structure][measure].keys():
            np.save(Intermediate_object_folder_dropbox+abstract_structure+'_'+measure+'_'+mouse_recday+'.npy',                   tuning_singletrial_ABCDAB_dic2[abstract_structure][measure][mouse_recday])


# In[45]:


for abstract_structure in Tuned_ABCDE_dic.keys():
    for measure in Tuned_ABCDE_dic[abstract_structure].keys():
        for mouse_recday in Tuned_ABCDE_dic[abstract_structure][measure].keys():
            np.save(Intermediate_object_folder_dropbox+abstract_structure+'_'+measure+'_'+mouse_recday+'.npy',                   Tuned_ABCDE_dic[abstract_structure][measure][mouse_recday])
            
        
for abstract_structure in tuning_singletrial_ABCDE_dic.keys():
    for measure in tuning_singletrial_ABCDE_dic[abstract_structure].keys():
        for mouse_recday in tuning_singletrial_ABCDE_dic[abstract_structure][measure].keys():
            np.save(Intermediate_object_folder_dropbox+abstract_structure+'_'+measure+'_'+mouse_recday+'.npy',                   tuning_singletrial_ABCDE_dic[abstract_structure][measure][mouse_recday])
            
np.save(Intermediate_object_folder_dropbox+'Mice.npy',Mice)


# In[ ]:





# In[166]:


tuning_singletrial_dic2['tuning_state_boolean'].keys()#[mouse_recday]


# In[ ]:





# In[352]:


#SAVING FILES 

try: 
    os.mkdir(Intermediate_object_folder) 
except FileExistsError: 
    pass

objects_dic={'tuning_singletrial_dic':tuning_singletrial_dic,'GLM_dic':GLM_dic,             'GLM_withinTask_dic':GLM_withinTask_dic,             'Tuned_dic':Tuned_dic,'cluster_dic':cluster_dic,            'tuning_singletrial_dic2':tuning_singletrial_dic2,'Phases_raw_dic':Phases_raw_dic,             'States_raw_dic':States_raw_dic,'Times_from_reward_dic':Times_from_reward_dic,            'speed_dic':speed_dic,'Phases_raw_dic2':Phases_raw_dic2,'GLM_dic2':GLM_dic2,'GLM_dic_policy':GLM_dic_policy,            'Tuned_dic2':Tuned_dic2,'smoothed_activity_dic2':smoothed_activity_dic2,'Num_neurons_dic':Num_neurons_dic}

for name, dicX in objects_dic.items(): 
    data=dicX 
    data_filename_memmap = os.path.join(Intermediate_object_folder, name) 
    dump(data, data_filename_memmap)


# In[ ]:





# In[451]:


Tuned_dic['Phase']['me10_17122021_19122021']


# In[ ]:





# In[377]:


Tuned_dic2['Place']['95']#['me10_19122021']


# In[ ]:





# In[ ]:




