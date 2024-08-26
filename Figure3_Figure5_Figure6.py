#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
What features are PFC neurons tuned to?

phase, state, place
'''


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
Data_folder=Intermediate_object_folder
Code_folder='/Taskspace_abstraction/Code/'


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
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter1d
import warnings
import statsmodels
from sklearn.preprocessing import MaxAbsScaler


# In[4]:


##Importing custom functions
module_path = os.path.abspath(os.path.join(Code_folder))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from mBaseFunctions import rec_dd, indicesX2, create_binary, unique_adjacent, dict_to_array, flatten, remove_empty,concatenate_complex2,mean_complex2, std_complex2, rand_jitter, rand_jitterX, bar_plotX, remove_nan,remove_nanX, column_stack_clean, polar_plot_state, positive_angle, circular_angle, plot_grouped_error,timestamp_to_binary, fill_nans, fill_nansX, angle_to_state, binned_array, binned_arrayX, plot_scatter,two_proportions_test, rearrange_matrix, matrix_triangle, scramble, smooth_circular, plot_spatial_maps, state_to_phase,middle_value,max_bin_safe, random_rotation, non_cluster_indices, cumulativeDist_plot, cumulativeDist_plot_norm,angle_to_distance, rotate, angle_to_stateX, range_ratio_peaks, equalize_rowsX, cross_corr_fast, plot_dendrogram,rotate, rank_repeat, edge_node_fill, split_mode, concatenate_states, predict_task_map, predict_task_map_policy,number_of_repeats, find_direction, Edge_grid, polar_plot_stateX, polar_plot_stateX2,arrange_plot_statecells_persession, arrange_plot_statecells, arrange_plot_statecells_persessionX2, Task_grid_plotting2,Task_grid_plotting, Task_grid, Task_grid2, Edge_grid_coord, Edge_grid_coord2, direction_dic_plotting, plot_spatial_mapsX,angle_to_distance, rank_repeat, number_of_repeats, noplot_timecourseA, noplot_timecourseAx, noplot_timecourseB,noplot_timecourseBx, two_proportions_test, noplot_scatter, number_of_repeats_ALL, non_repeat_ses_maker,non_repeat_ses_maker_old, circular_sem


# In[ ]:





# In[5]:


def polar_plot_stateX2(meanx,upperx,lowerx,ax,repeated,color='black',labels='states',plot_type='line',Marker=False,                      fields_booleanx=[], structure_abstract='ABCD',fontsize=35,set_max=False,max_val=1):
    rx = list(meanx)
    theta = list(range(len(rx)))
    thetax = [2 * np.pi * (x/len(rx)) for x in theta]
    r = rx + [rx[0]]
    theta = thetax + [thetax[0]]
    
    plt.rcParams['axes.linewidth'] = 4
    
    #ax=plt.subplot(111, projection='polar')
    
    if Marker==True:
        fields_booleanx=fields_booleanx*(np.max(upperx)+0.1*np.max(upperx))
        fields_boolean=list(fields_booleanx)+[list(fields_booleanx)[0]]

    upper=list(upperx)+[list(upperx)[0]]
    lower=list(lowerx)+[list(lowerx)[0]]
    
    if plot_type=='line':
        ax.plot(theta, r,color=color,linewidth=4)
        ax.fill_between(theta, upper, lower, alpha=0.2,color=color)
        if set_max==False:
            ax.set_rmax(np.max(upper)+0.01*np.max(upper))
        else:
            ax.set_rmax(max_val)
            
        if Marker==True:
            ax.plot(theta, fields_boolean,color='black',linestyle='None',marker='.')

    elif plot_type=='bar':
        ax.bar(theta,r,width=5/len(r),color=color)
    elif plot_type=='marker':
        ax.plot(theta, r,color=color)
        
    
    ax.grid(True)
    #ax.set_rorigin(-1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    if labels=='states':
        if structure_abstract=='ABCD':
            ax.set_xticklabels(['A', '', 'B', '', 'C', '', 'D', ''],fontsize=fontsize)
        elif structure_abstract=='AB' and repeated==False:
            ax.set_xticklabels(['A', '', '', '', 'B', '', '', ''],fontsize=fontsize)
        elif structure_abstract=='AB' and repeated==True:
            ax.set_xticklabels(['A', '', 'B', '', 'A', '', 'B', ''],fontsize=fontsize)
    elif labels == 'angles':
        ax.set_xticklabels(['0', '', '90', '', '180', '', '270', ''],fontsize=fontsize)
        
    plt.tick_params(axis='both',  labelsize=fontsize)
    plt.tick_params(width=2, length=6)

def arrange_plot_statecells_persessionX(mouse_recday,neuron,awake_sessions,standardized_FR_smoothed_all,                                        standardized_FR_sem_all,sessions_included=None                                       ,fignamex=False,sigma=10,                                       save=False,plot=False,figtype='.svg', Marker=False,                                       fields_booleanx=[],measure_type='mean', abstract_structures=[],                                      repeated=False):
    
    colors=np.repeat('blue',len(awake_sessions))
    plot_boolean=np.repeat(False,len(awake_sessions))
    plot_boolean[sessions_included]=True
    

    fig= plt.figure(figsize=plt.figaspect(1)*4.5)
    fig.tight_layout()
    for awake_session_ind, timestamp in enumerate(awake_sessions):
        structure_abstract=abstract_structures[awake_session_ind]
        standardized_FR_smoothed=standardized_FR_smoothed_all[awake_session_ind]
        standardized_FR_sem=standardized_FR_sem_all[awake_session_ind]
                    
        if len(standardized_FR_smoothed)==0:
            print('Empty: Possibly No trials completed')
            continue
        
        
        standardized_FR_smoothed_upper=standardized_FR_smoothed+standardized_FR_sem
        standardized_FR_smoothed_lower=standardized_FR_smoothed-standardized_FR_sem
       
        
        color=colors[awake_session_ind]
        
        ax = fig.add_subplot(1, len(awake_sessions), awake_session_ind+1, projection='polar')
        if len(fields_booleanx)>0:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,                              ax,color=color, Marker=Marker,fields_booleanx=fields_booleanx[awake_session_ind],                             structure_abstract=structure_abstract,repeated=repeated)
        else:
            polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,                              ax,color=color, Marker=False,structure_abstract=structure_abstract,repeated=repeated)
    plt.margins(0,0)
    plt.tight_layout()
    if save==True:
        plt.savefig(fignamex+str(awake_session_ind)+figtype, bbox_inches = 'tight', pad_inches = 0)
    if plot==True & plot_boolean[awake_session_ind]==True:
        plt.show()
    else:
        plt.close() 
        
def arrange_plot_statecells_persessionX2(mouse_recday,neuron,Data_folder,sessions_included=None                                       ,fignamex=False,sigma=10,                                       save=False,plot=False,figtype='.svg', Marker=False,                                       fields_booleanx=[],measure_type='mean', abstract_structures=[],                                      repeated=False,behaviour_oversampling_factor=3,behaviour_rate=1000,                                       tracking_oversampling_factor=50):

    awake_sessions_behaviour= np.load(Intermediate_object_folder_dropbox+'awake_session_behaviour_'+mouse_recday+'.npy')
    awake_sessions=np.load(Intermediate_object_folder_dropbox+'awake_session_'+mouse_recday+'.npy')
    
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
        plt.savefig(fignamex+str(awake_session_ind)+figtype, bbox_inches = 'tight', pad_inches = 0)
    if plot==True & plot_boolean[awake_session_ind]==True:
        plt.show()
    else:
        plt.close() 
        
def plot_spatial_mapsX(mouse_recday,neuron,sessions_used, plot_edge=True, per_state=False,save_fig=False,                       fignamex=None,figtype=None,sessions_custom=False):
    mouse=mouse_recday[:4]
    
    Num_trials_completed_=dict_to_array(Num_trials_dic2[mouse_recday])
    All_sessions=session_dic_behaviour['All'][mouse_recday]  
    if sessions_custom==False:
        awake_sessions=session_dic_behaviour['awake'][mouse_recday][Num_trials_completed_>0]
        awake_ses_inds=np.arange(awake_sessions)
    else:
        awake_ses_inds=sessions_used
    rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
    rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
    structure_nums=np.unique(rec_day_structure_numbers)

    print('')
    print('Mean Rate maps')
    ###ploting firing maps per state

    fig1, f1_axes = plt.subplots(figsize=(7.5, 7.5),ncols=len(awake_ses_inds), constrained_layout=True)

    node_rate_matrices=[]
    for awake_session_ind in awake_ses_inds:
        node_rate_matrices.append(node_rate_matrices_dic['All_states'][awake_session_ind][mouse_recday][neuron])

    #max_rate=np.nanmax(node_rate_matrices)
    #min_rate=np.nanmin(node_rate_matrices)
    for awake_ses_ind_ind, awake_session_ind in enumerate(awake_ses_inds):
        node_rate_mat=node_rate_matrices_dic['All_states'][awake_session_ind][mouse_recday][neuron]
        edge_rate_mat=edge_rate_matrices_dic['All_states'][awake_session_ind][mouse_recday][neuron]

        node_edge_mat=edge_node_fill(edge_rate_mat,node_rate_mat)
        
        if plot_edge==True:
            mat_used=node_edge_mat
            gridX=Task_grid_plotting2
        else:
            mat_used=node_rate_mat
            gridX=Task_grid_plotting
            
        arrow_length=0.2
        prop_scaling=0.3
        adjustment=0.25
            

        max_rate=np.nanmax(mat_used)
        min_rate=np.nanmin(mat_used)
        #exec('node_rate_matrix'+str(awake_session_ind)+'=node_rate_matrix')
        ax=f1_axes[awake_ses_ind_ind]

        structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]        [rec_day_session_numbers[awake_session_ind]]
        for state_port_ind, state_port in enumerate(states):
            node=structure[state_port_ind]-1
            ax.text(gridX[node,0]-adjustment, gridX[node,1]+adjustment,                    state_port.lower(), fontsize=22.5)
            
            
        ###Policy
        directions=Policy_dic['Mean'][mouse_recday][awake_session_ind]
        for node in np.arange(9):
            for dir_ind, (direction, coords_) in enumerate(direction_dic_plotting.items()):
                prop_=len(np.where(directions[node]==direction)[0])/len(directions[node])
                if prop_>0:
                    ax.arrow(gridX[node,0], gridX[node,1],                              coords_[0]*arrow_length,coords_[1]*arrow_length,width=prop_scaling*prop_/2,                              head_width=prop_scaling*prop_,color='white')
        ax.axis('off')
        ax.matshow(mat_used,vmin=min_rate, vmax=max_rate, cmap='coolwarm')
    plt.axis('off')   
    if save_fig==True:
        plt.savefig(fignamex+figtype, bbox_inches = 'tight', pad_inches = 0)  
    plt.show()


    if per_state==True:
        ###per state plot

        print('')
        print('per state Rate maps')
        fig2, f2_axes = plt.subplots(figsize=(7.5, 7.5),ncols=len(awake_ses_inds), nrows=len(states),                                     constrained_layout=True)   
        for awake_ses_ind_ind, awake_session_ind in enumerate(awake_ses_inds):   

            #print(awake_session_ind)
            for statename_ind, state in enumerate(states):
                #print(state)
                node_rate_matrix_state=node_rate_matrices_dic['Per_state'][awake_session_ind][mouse_recday][neuron]                [statename_ind]
                edge_rate_matrix_state=edge_rate_matrices_dic['Per_state'][awake_session_ind][mouse_recday][neuron]                [statename_ind]
                node_edge_mat_state=edge_node_fill(edge_rate_matrix_state,                                                       node_rate_matrix_state)
                
                if plot_edge==True:
                    mat_used=node_edge_mat_state
                    gridX=Task_grid_plotting2
                else:
                    mat_used=node_rate_matrix_state[state_port_ind]
                    gridX=Task_grid_plotting

                structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]                [rec_day_session_numbers[awake_session_ind]]
                
                
                ax=f2_axes[awake_ses_ind_ind,statename_ind]
                for state_port_ind, state_port in enumerate(states):
                    node=structure[state_port_ind]-1
                    ax.text(gridX[node,0]-0.25, gridX[node,1]+0.25,                            state_port.lower(), fontsize=22.5)

                ax.matshow(mat_used, cmap='coolwarm') #vmin=min_rate, vmax=max_rate
                ax.axis('off')
                #ax.savefig(str(neuron)+state+str(awake_session_ind)+'discmap.svg')
        plt.axis('off')
        if save_fig==True:
            plt.savefig(fignamex+'_perstate_'+figtype, bbox_inches = 'tight', pad_inches = 0)
            
def unique_adjacent(a):
    return(np.asarray([k for k,g in groupby(a)]))

def data_matrix(data, concatenate=False):
    data_mat=np.asarray([data[ii][:len(data[-1])] for ii in range (len(data))])
    if concatenate==True:
        data_mat=np.concatenate(np.hstack(data_mat))
    return(data_mat)

###counts num of repeats for each stretch of numbers
def rank_repeat2(a):
    num_repeats=number_of_repeats(a)
    arr=[]
    for n_ind, n in enumerate(unique_adjacent(a)):
        count=0
        indices=np.arange(num_repeats[n_ind])
        arr.append(indices)
    arr=np.concatenate(arr)
    arr=arr.astype(int)
    return(arr)

def continguous_field(array,num_bins,cont_thr=2):
    if len(array)==0:
        field=[np.nan]
    else:
        bool_xx=np.diff(array)<=cont_thr
        xx=0
        field=np.zeros(len(bool_xx))
        for ii in range(len(bool_xx)):
            if bool_xx[ii]==False:
                xx+=1
            field[ii]=xx
        field=np.hstack((0,field))
        if array[0]+(num_bins-1)-array[-1]<cont_thr:
            field[field==unique_adjacent(field)[-1]]=0

    return(field)


def most_common(aa):
    counts=list(Counter(aa).values())
    max_count=np.max(counts)
    return(np.asarray(list(Counter(aa).keys()))[counts==max_count],max_count)

def demean(x):
    return(x-np.nanmean(x))

from collections import Counter
from itertools import combinations

def most_common_pair(a_):
    a=np.copy(a_)
    d  = Counter()
    for sub in a:
        if len(a) < 2:
            continue
        #sub.sort()
        for comb in combinations(sub,2):
            d[comb] += 1

    return([d.most_common()[0][0][0],d.most_common()[0][0][1]], d.most_common()[0][1]/len(a))


def num_of_repeats2(MyList):
    my_dict = {i:list(MyList).count(i) for i in MyList}
    
    return(np.asarray([my_dict[element] for element in MyList]))


def fill_diagonal(source_array, diagonal):
    copy = source_array.copy()
    np.fill_diagonal(copy, diagonal)
    return copy

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
    
    
def _nanargmin(arr, axis=0):
    try:
        if len(np.shape(arr))==1:
            return np.nanargmin(arr)
        else:
            return np.nanargmin(arr, axis)
    except ValueError:
        return np.nan
    
def _nanargmax(arr, axis=0):
    try:
        if len(np.shape(arr))==1:
            return np.nanargmax(arr)
        else:
            return np.nanargmax(arr, axis)
    except ValueError:
        return np.nan
    
    
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

def matrix_triangle(a,direction='upper',return_indices=False):
    if direction=='upper':
        indices=np.triu_indices(len(a), k = 1)
    if direction=='lower':
        indices=np.tril_indices(len(a), k = -1)
    triangle=a[indices]
    if return_indices==True:
        return(triangle,indices)
    else:
        return(triangle)
    
from scipy.optimize import curve_fit
def func_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def circmedian(angs):
    pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
    pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
    pdists = np.abs(pdists).sum(1)
    return angs[np.argmin(pdists)]


# In[ ]:





# In[ ]:





# In[6]:


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


# In[ ]:





# In[7]:


#LOADING FILES - tracking, behaviour, Ephys raw, Ephys binned
tt=time.time()


try:
    os.mkdir(Intermediate_object_folder)
except FileExistsError:
    pass

dictionaries_list=['Num_trials_dic','Num_trials_dic2','ROI_accuracy_dic','FR_shuff_dic','FR_shuff_states_dic',                   'state_field_dic','num_field_dic','phase_field_dic','phase_middle_dic','state_middle_dic',                   'Xneuron_correlations','Xneuron_correlations2','day_type_sameTask_dic','session_dic',                   'Xsession_correlations','Xsession_correlations2','Policy_dic','Policy_dic2',                  'combined_recordingdays_dic','node_rate_matrices_dic','edge_rate_matrices_dic','cluster_dic',                  'sampling_dic','smoothed_activity_dic','recday_numbers_dic','day_type_dicX',                  'Confidence_rotations_dic','Xsession_correlations_predicted','structure_dic',                  'smoothed_activity_dic','Task_num_dic','Spatial_anchoring_dic',                  'Anchor_trial_dic','scores_dic','tuning_singletrial_dic','GLM_dic','GLM_withinTask_dic',                  'Combined_days_dic','Tuned_dic','session_dic_behaviour',                  'tuning_singletrial_dic2','Phases_raw_dic','States_raw_dic','Times_from_reward_dic','speed_dic',                  'Phases_raw_dic2','GLM_dic2','GLM_anchoring_prep_dic','GLM_anchoring_noregularisation_dic',                  'GLM_anchoring_regularised_dic','GLM_anchoring_regularised_high_dic',                   'GLM_anchoring_regularised_veryhigh_dic','Tuned_dic2','Num_neurons_dic','smoothed_activity_dic2',                  'module_dic','module_shuff_dic','Spatial_anchoring_dic_old','Anchor_trial_dic_old',                  'Anchor_trial_dic2_old']

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


# In[ ]:





# In[ ]:





# In[8]:


tt=time.time()
name='state_middle_dic'
data_filename_memmap = os.path.join(Intermediate_object_folder, name)
data = load(data_filename_memmap)#, mmap_mode='r')
exec(name+'= data')
print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


'''Also to import when needed
1-state_occupancy_dic (both occupancy and policy)
2-standardized_spike_events_dic - might need this
3-binned_FR_dic (maybe not here, will need it for sleep analysis)
'''


# In[11]:


'''
1-Basic tuning plots: task, spatial, policy 
2-Basic tuning quantification: spatial, number of peaks
3-Single cell generalization
4-Modules


'''


# In[ ]:





# In[12]:


####Basic Tuning####


# In[8]:


###defining sampling
frame_rate=60
tracking_oversampling_factor=50 ##oversampling of nodes_cut_dic
behaviour_oversampling_factor=3 ##oversampling of trialtimes_dic
behaviour_rate=1000


# In[145]:


###creating suffled activity per neuron
tt=time.time()
num_iterations=100

#FR_shuff_dic=rec_dd()
#FR_shuff_states_dic=rec_dd()
specific_days=True
specific_day_array=['ah07_01092023_02092023']
redo=True
#for day_type, day_type_array in day_type_dicX.items():   
day_type_array=day_type_dicX['All']
for mouse_recday in day_type_array:

    if specific_days==True:
        if mouse_recday not in specific_day_array:
            continue
    if redo==False:
        if len(FR_shuff_dic['percentiles_state'][0][mouse_recday])>0:
            print('ALready analysed')
            continue
    print(mouse_recday)
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    all_neurons=np.arange(num_neurons)
    neurons_used=all_neurons
    awake_sessions=session_dic_behaviour['awake'][mouse_recday]
    rec_day_structure_abstract=recday_numbers_dic['structure_abstract'][mouse_recday]

    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]
    All_sessions=session_dic_behaviour['All'][mouse_recday]
    awake_sessions=session_dic_behaviour['awake'][mouse_recday]
    #awake_session_ids=session_dic_behaviour['awake'][mouse_recday][1]

    if num_neurons==0:
        print('No neurons')
        continue
    for awake_session_ind, timestamp in enumerate(awake_sessions):
        print(awake_session_ind)
        try:

            if Num_trials_dic2[mouse_recday][awake_session_ind]==0:
                print('session'+str(awake_session_ind))
                print('no trials completed')
                continue

                
            abstract_structure_ses=rec_day_structure_abstract[awake_session_ind]
            
            num_states=len(abstract_structure_ses)
            num_bins=num_states*90
            sigma=40/num_states
            
            mean_shuffled=np.zeros((num_neurons,num_iterations,num_bins))
            mean_shuffled_state=np.zeros((num_neurons,num_iterations,num_bins))
            standardized_FR_allneurons=np.zeros((num_neurons,num_bins))

            if specific_days==True:
                array_length_to_test=0
            else:
                array_length_to_test=len(FR_shuff_dic[awake_session_ind][mouse_recday][int(num_neurons-1)])
            if redo==False:
                if array_length_to_test>0:
                    print('Already analyzed')
                    continue


            try:
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
            except:
                print('Ephys file not found')
                continue
            
            for iteration in range(num_iterations):
                ephys_copy_=np.copy(ephys_)

                ephys_shuff_=np.asarray([[scramble(ephys_copy_[neuron,trial]) for trial in                                          np.arange(len(ephys_copy_[neuron]))]                             for neuron in np.arange(len(ephys_copy_))])

                ephys_shuff_states_=np.asarray([np.vstack(([np.hstack((                                    ephys_copy_[neuron,trial].reshape(num_states,90)                                  [random.sample(list(np.arange(num_states)), num_states)]))                    for trial in np.arange(np.shape(ephys_copy_)[1])]))                                              for neuron in np.arange(len(ephys_copy_))])
                ephys_shuff_mean=np.nanmean(ephys_shuff_,axis=1)
                ephys_shuff_states_mean=np.nanmean(ephys_shuff_states_,axis=1)
                
                ephys_shuff_mean_smoothed_=np.vstack(([smooth_circular(ephys_shuff_mean[neuron],sigma=sigma)                                 for neuron in np.arange(len(ephys_shuff_mean))]))
                
                ephys_shuff_states_mean_smoothed_=np.vstack(([smooth_circular(ephys_shuff_states_mean[neuron],                                                                              sigma=sigma)                                 for neuron in np.arange(len(ephys_shuff_states_mean))]))

                mean_shuffled[:,iteration]=ephys_shuff_mean_smoothed_
                mean_shuffled_state[:,iteration]=ephys_shuff_states_mean_smoothed_


            
            
            for neuron in np.arange(len(ephys_copy_)):
                FR_shuff_dic[awake_session_ind][mouse_recday][neuron]=np.percentile(mean_shuffled[neuron],95,axis=0)

                FR_shuff_states_dic[awake_session_ind][mouse_recday][neuron]=                np.percentile(mean_shuffled_state[neuron],95,axis=0)
            
            

            ephys_mean_=np.nanmean(ephys_,axis=1)
            ephys_mean_smoothed_=np.vstack(([smooth_circular(ephys_mean_[neuron],sigma=sigma)                                 for neuron in np.arange(len(ephys_mean_))]))

            percentiles_all=np.asarray(([[st.percentileofscore(mean_shuffled[neuron,:,bin_],                                                               ephys_mean_smoothed_[neuron][bin_])                for bin_ in np.arange(len(ephys_mean_.T))]                for neuron in np.arange(len(ephys_mean_))]))

            percentiles_state=np.asarray(([[st.percentileofscore(mean_shuffled_state[neuron,:,bin_],                                                                 ephys_mean_smoothed_[neuron][bin_])                for bin_ in np.arange(len(ephys_mean_.T))]                for neuron in np.arange(len(ephys_mean_))]))

            FR_shuff_dic['percentiles_all'][awake_session_ind][mouse_recday]=percentiles_all
            FR_shuff_dic['percentiles_state'][awake_session_ind][mouse_recday]=percentiles_state
                
                

            
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('Not analyzed')
            
print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[146]:


###making arrays for split double days 
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]
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
        awake_ses_day=np.where(days==mouse_recdayX)[0]
        for awake_session_ind_ind, awake_session_ind in enumerate(awake_ses_day): 
            FR_shuff_dic[awake_session_ind_ind][mouse_recdayX]=FR_shuff_dic[awake_session_ind][mouse_recday]
            FR_shuff_states_dic[awake_session_ind_ind][mouse_recdayX]=FR_shuff_states_dic[awake_session_ind][mouse_recday]
            
            FR_shuff_dic['percentiles_all'][awake_session_ind_ind][mouse_recdayX]=            FR_shuff_dic['percentiles_all'][awake_session_ind][mouse_recday]
            
            FR_shuff_dic['percentiles_state'][awake_session_ind_ind][mouse_recdayX]=            FR_shuff_dic['percentiles_state'][awake_session_ind][mouse_recday]

        


# In[85]:


#list(Tuned_dic2['Phase']['95'].keys())
FR_shuff_dic['percentiles_state'][0].keys()


# In[175]:


from statsmodels.sandbox.stats.multicomp import multipletests
all_peaks_OR_all=[]
all_peaks_AND_all=[]
all_statepeaks_all=[]
all_peaks_median_all=[]
#for mouse_recday in FR_shuff_dic['percentiles_state'][0].keys():
recording_days=list(Tuned_dic2['Phase']['95'].keys())
#np.hstack((day_type_dicX['combined_ABCDonly'],day_type_dicX['3_task']))
num_neurons_all=[]
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    if mouse_recday not in FR_shuff_dic['percentiles_state'][0].keys():
        print('Not calculated')
        continue
    
    awake_sessions=session_dic_behaviour['awake'][mouse_recday]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    rec_day_structure_abstract=recday_numbers_dic['structure_abstract'][mouse_recday]

    all_peaks_median_ses=[]
    for awake_session_ind, awake_session in enumerate(non_repeat_ses):
        abstract_structure_ses=rec_day_structure_abstract[awake_session]
        num_states=len(abstract_structure_ses)
        
        if num_states!=4:
            continue
        
        Phase_bool=Tuned_dic2['Phase_strict']['95'][mouse_recday]
        State_bool=Tuned_dic2['State']['95'][mouse_recday]
        Place_bool=Tuned_dic2['Place']['95'][mouse_recday]
        
        Phase_state_bool=np.logical_and(Phase_bool,State_bool)
        
        Phase_non_place_bool=np.logical_and(Phase_bool,~Place_bool)
        Phase_non_state_bool=np.logical_and(Phase_bool,~State_bool)
        
        
        if len(FR_shuff_dic['percentiles_all'][awake_session][mouse_recday])==0:
            #print(mouse_recday)
            print('session '+str(awake_session)+'not found')
            continue
        
        percentiles_all_=FR_shuff_dic['percentiles_all'][awake_session][mouse_recday][Phase_bool]
        if len(percentiles_all_)==0:
            continue
        p_adjusted_allneurons = np.vstack(([multipletests((100-percentiles_all_[neuron])/100, method='bonferroni')[1]        for neuron in np.arange(len(percentiles_all_))]))
        p_adjusted_allneurons_reshaped=p_adjusted_allneurons.reshape(np.shape(p_adjusted_allneurons)[0],int(num_states),
                                                                     int(np.shape(p_adjusted_allneurons)[1]/num_states))
        all_peaks_OR_ses=np.asarray(([np.where(np.any(p_adjusted_allneurons_reshaped[neuron]<0.01,axis=0)==True)[0]                    for neuron in np.arange(len(p_adjusted_allneurons))]))

        all_peaks_AND_ses=np.asarray(([np.where(np.all(p_adjusted_allneurons_reshaped[neuron]<0.01,axis=0)==True)[0]                    for neuron in np.arange(len(p_adjusted_allneurons))]))

        all_statepeaks_ses=np.asarray([np.where(p_adjusted_allneurons[neuron]<0.01)[0]                    for neuron in np.arange(len(p_adjusted_allneurons))])
        
        circular_median=np.hstack(([np.rad2deg(circmedian(np.deg2rad(all_peaks_OR_ses[neuron]*4)))                         if len(all_peaks_OR_ses[neuron])>0 else np.nan for neuron in np.arange(len(all_peaks_OR_ses))]))
        
        all_peaks_OR_all.append(all_peaks_OR_ses)
        all_peaks_AND_all.append(all_peaks_AND_ses)
        all_statepeaks_all.append(all_statepeaks_ses)
        
        all_peaks_median_ses.append(circular_median)
        
        
    
    if len(all_peaks_median_ses)==0:
        print('Not used')
        continue
    all_peaks_median_ses=np.vstack((all_peaks_median_ses))
    all_peaks_median_meanall=np.hstack(([np.rad2deg(st.circmean(np.deg2rad(all_peaks_median_ses[:,neuron])))                                     for neuron in np.arange(len(all_peaks_median_ses.T))]))
    
    all_peaks_median_all.append(all_peaks_median_meanall)
    print(np.sum(Phase_bool))
    print(len(circular_median))
    num_neurons_all.append(len(circular_median))


# In[176]:


np.sum(num_neurons_all)


# In[168]:


xx=np.hstack(([Tuned_dic2['Phase_strict']['95'][mouse_recday]               for mouse_recday in day_type_dicX['combined_ABCDonly']]))
np.sum(xx)


# In[172]:


all_peaks_median_meanall


# In[69]:





# In[80]:


np.hstack(([np.rad2deg(circmedian(np.deg2rad(all_peaks_OR_ses[neuron]*4)))                         if len(all_peaks_OR_ses[neuron])>0 else np.nan for neuron in np.arange(len(all_peaks_OR_ses))]))


# In[78]:


xx=np.zeros(360)
xx[all_peaks_OR_ses[2]*4]=1

plt.plot(xx)


# In[ ]:





# In[148]:



all_peaks_OR_ses_conc=np.hstack((concatenate_complex2(all_peaks_OR_all)))
plt.hist(all_peaks_OR_ses_conc,bins=45,color='black',edgecolor='white', linewidth=0.3)
plt.savefig(Ephys_output_folder_dropbox+'phase_distribution_histogram.svg')
plt.show()

all_peaks_AND_ses_conc=np.hstack((concatenate_complex2(all_peaks_AND_all)))
plt.hist(all_peaks_AND_ses_conc,bins=45,color='black',edgecolor='white', linewidth=0.3)
plt.show()


all_statepeaks_ses_conc=np.hstack((concatenate_complex2(all_statepeaks_all)))
plt.hist(all_statepeaks_ses_conc,bins=180,color='black',edgecolor='white', linewidth=0.3)
plt.show()


# In[155]:


plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
all_peaks_OR_ses_conc=np.hstack((np.hstack((all_peaks_median_all))))/4
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.hist(all_peaks_OR_ses_conc,bins=45,color='black',edgecolor='white', linewidth=0.3)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'phase_distribution_perneuron_histogram.svg')
plt.show()
print(len(remove_nan(all_peaks_OR_ses_conc)))
from scipy import stats
print(stats.kstest(remove_nan(all_peaks_OR_ses_conc),stats.uniform.cdf))


# In[112]:


from scipy import stats
stats.kstest(remove_nan(all_peaks_OR_ses_conc),stats.uniform.cdf)


# In[177]:


len(all_peaks_OR_ses_conc)


# In[30]:


###updating 3_task_all entry of day_type_dicX
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]
    rec_day1=rec_day.split('_',1)[0]
    rec_day2=rec_day.split('_',1)[1]
    Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
    Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

    mouse_recday1=mouse+'_'+rec_day1
    mouse_recday2=mouse+'_'+rec_day2
    
    day_type_dicX['3_task_all']=np.hstack((day_type_dicX['3_task_all'],mouse_recday1,mouse_recday2))
    
day_type_dicX['3_task_all']=np.unique(day_type_dicX['3_task_all'])
    
for day_type in day_type_dicX.keys():
    recording_days_=np.save(Intermediate_object_folder_dropbox+day_type+'_days.npy',day_type_dicX[day_type])
    
#SAVING FILES 

try: 
    os.mkdir(Intermediate_object_folder) 
except FileExistsError: 
    pass

objects_dic={'day_type_dicX':day_type_dicX}

for name, dicX in objects_dic.items(): 
    data=dicX 
    data_filename_memmap = os.path.join(Intermediate_object_folder, name) 
    dump(data, data_filename_memmap)


# In[ ]:





# In[ ]:





# In[ ]:





# #Identifying state fields
# tt=time.time()
# state_field_dic=rec_dd()
# num_field_dic=rec_dd()
# phase_field_dic=rec_dd()
# phase_middle_dic=rec_dd()
# state_middle_dic=rec_dd()
# 
# 
# thr_field_seperation=2 ##this is how many bins(in degrees) need to be between two putative fields for them to be 
# #considered seperate fields (minimum is 2)
# 
# 
# for name, dicX in {'ALL':FR_shuff_dic, 'State':FR_shuff_states_dic}.items():
#     print(name)
#     for day_type, day_type_array in day_type_dicX.items():
#         if day_type not in ['combined_ABCDonly','3_task','3_task_all']:
#             continue
#         for mouse_recday in day_type_array:
#             print(mouse_recday)
#             num_neurons=Num_neurons_dic[mouse_recday]
#             awake_sessions=session_dic_behaviour['awake'][mouse_recday]
#             abstract_structure=recday_numbers_dic['structure_abstract'][mouse_recday]
#             for awake_session_ind, timestamp in enumerate(awake_sessions):
#                 try:
#                     print(awake_session_ind)
#                     if Num_trials_dic2[mouse_recday][awake_session_ind]==0:
#                         print('no trials completed')
#                         continue
# 
# 
#                     #if len(smoothed_activity_dic[mouse_recday][0][awake_session_ind])>0:
# 
#                     all_fields_allneurons=[]
#                     for neuron in range(num_neurons):
#                         thrs=dicX[awake_session_ind][mouse_recday][neuron]
#                         
#                         peak_=(smoothed_activity_dic2['Mean'][mouse_recday][awake_session_ind][neuron]>thrs).astype(int)
# 
# 
#                         peak_bins=np.where(peak_==1)[0]
# 
#                         diff_peaks_=np.diff(peak_bins)#[0]
#                         field_edges_=np.where(diff_peaks_>=thr_field_seperation)[0]+1
# 
#                         all_fields=[]
#                         if len(peak_bins)>0 and len(field_edges_)>0:
#                             if set([0,359]).issubset(set(peak_bins)):
#                                 first_bin=field_edges_[-1]
#                                 first_field=np.hstack((peak_bins[first_bin:],peak_bins[:field_edges_[0]]))
#                                 other_fields=[peak_bins[field_edges_[ii]:field_edges_[ii+1]] for ii\
#                                               in range(len(field_edges_)-1)]
# 
# 
#                             else:
#                                 first_field=peak_bins[:field_edges_[0]]
#                                 other_fields=[peak_bins[field_edges_[ii]:field_edges_[ii+1]] for ii\
#                                               in range(len(field_edges_)-1)]
#                                 other_fields.append(peak_bins[field_edges_[-1]:])
# 
# 
#                             all_fields.append(first_field)
#                             for xx in range(len(other_fields)):
#                                 all_fields.append(other_fields[xx])
#                             #all_fields_allneurons.append(all_fields)
#                         elif len(peak_bins)>0 and len(field_edges_)==0: ##i.e. only one field which doesnt span from D to A
#                             all_fields=[peak_bins]
# 
#                         all_state_middle=np.asarray([middle_value(all_fields[ii]) for ii in range(len(all_fields))])
#                         statesxx=[math.radians(all_state_middle[ii]) for ii in range(len(all_fields))]
#                         mean_state=math.degrees(circmean(statesxx))
# 
#                         all_phase_fields=np.asarray([state_to_phase(all_fields[ii],len(abstract_structure))\
#                                                      for ii in range(len(all_fields))], dtype="object")
#                         all_phase_middle=np.asarray([middle_value(all_phase_fields[ii]) for ii in range(len(all_fields))]\
#                                                    , dtype="object")
# 
#                         phasesxx=[math.radians(all_phase_middle[ii]) for ii in range(len(all_phase_middle))]
#                         mean_phase=math.degrees(circmean(phasesxx))
# 
# 
# 
#                         state_field_dic[name][awake_session_ind][mouse_recday][neuron]=all_fields
#                         num_field_dic[name][awake_session_ind][mouse_recday][neuron]=len(all_fields)
#                         state_middle_dic[name][awake_session_ind][mouse_recday][neuron]=all_state_middle
#                         state_middle_dic[name]['Mean_state'][awake_session_ind][mouse_recday][neuron]=mean_state
# 
#                         phase_field_dic[name][awake_session_ind][mouse_recday][neuron]=all_phase_fields
#                         phase_middle_dic[name][awake_session_ind][mouse_recday][neuron]=all_phase_middle
#                         phase_middle_dic[name]['Mean_phase'][awake_session_ind][mouse_recday][neuron]=mean_phase
# 
#                         #########
#                         state_field_dic[name][day_type][awake_session_ind][mouse_recday][neuron]=all_fields
#                         num_field_dic[name][day_type][awake_session_ind][mouse_recday][neuron]=len(all_fields)
#                         state_middle_dic[name][day_type][awake_session_ind][mouse_recday][neuron]=all_state_middle
#                         state_middle_dic[name]['Mean_state'][day_type][awake_session_ind][mouse_recday][neuron]=\
#                         mean_state
# 
#                         phase_field_dic[name][day_type][awake_session_ind][mouse_recday][neuron]=all_phase_fields
#                         phase_middle_dic[name][day_type][awake_session_ind][mouse_recday][neuron]=all_phase_middle
#                         phase_middle_dic[name]['Mean_phase'][day_type][awake_session_ind][mouse_recday][neuron]=\
#                         mean_phase
#                 except Exception as e: 
#                     print(e)
#                     exc_type, exc_obj, exc_tb = sys.exc_info()
#                     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#                     print(exc_type, fname, exc_tb.tb_lineno)
# 
# print(time.time()-tt)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
problems:
me10_15122021 - ses1 no ephys found, ses2 no trials 

'''


# In[ ]:





# ###making arrays for split double days
# day_typeX='3_task_all'
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
#     days=[]
#     for mouse_recdayX in [mouse_recday1,mouse_recday2]:
#         num_awake_ses=len(session_dic_behaviour['awake'][mouse_recdayX])
#         days.append(np.repeat(mouse_recdayX,num_awake_ses))
#     days=np.hstack((days))
#     for mouse_recdayX in [mouse_recday1,mouse_recday2]:
#         if mouse_recdayX in day_type_dicX['3_task']:
#             continue
#         awake_ses_day=np.where(days==mouse_recdayX)[0]
#         for awake_session_ind_ind, awake_session_ind in enumerate(awake_ses_day): 
#             for name, dicX in {'ALL':FR_shuff_dic, 'State':FR_shuff_states_dic}.items():
#                 state_field_dic[name][awake_session_ind_ind][mouse_recdayX]=\
#                 state_field_dic[name][awake_session_ind][mouse_recday]
#                 num_field_dic[name][awake_session_ind_ind][mouse_recdayX]=\
#                 num_field_dic[name][awake_session_ind][mouse_recday]
#                 state_middle_dic[name][awake_session_ind_ind][mouse_recdayX]=\
#                 state_middle_dic[name][awake_session_ind][mouse_recday]
#                 state_middle_dic[name]['Mean_state'][awake_session_ind_ind][mouse_recdayX]=\
#                 state_middle_dic[name]['Mean_state'][awake_session_ind][mouse_recday]
#                 phase_field_dic[name][awake_session_ind_ind][mouse_recdayX]=\
#                 phase_field_dic[name][awake_session_ind][mouse_recday]
#                 phase_middle_dic[name][awake_session_ind_ind][mouse_recdayX]=\
#                 phase_middle_dic[name][awake_session_ind][mouse_recday]
#                 phase_middle_dic[name]['Mean_phase'][awake_session_ind_ind][mouse_recdayX]=\
#                 phase_middle_dic[name]['Mean_phase'][awake_session_ind][mouse_recday]
# 
#                 #########
#                 state_field_dic[name][day_typeX][awake_session_ind_ind][mouse_recdayX]=\
#                 state_field_dic[name][awake_session_ind][mouse_recday]
#                 num_field_dic[name][day_typeX][awake_session_ind_ind][mouse_recdayX]=\
#                 num_field_dic[name][awake_session_ind][mouse_recday]
#                 state_middle_dic[name][day_typeX][awake_session_ind_ind][mouse_recdayX]=\
#                 state_middle_dic[name][awake_session_ind][mouse_recday]
#                 state_middle_dic[name]['Mean_state'][day_typeX][awake_session_ind_ind][mouse_recdayX]=\
#                 state_middle_dic[name]['Mean_state'][awake_session_ind][mouse_recday]
# 
#                 phase_field_dic[name][day_typeX][awake_session_ind_ind][mouse_recdayX]=\
#                 phase_field_dic[name][awake_session_ind][mouse_recday]
#                 phase_middle_dic[name][day_typeX][awake_session_ind_ind][mouse_recdayX]=\
#                 phase_middle_dic[name][awake_session_ind][mouse_recday]
#                 phase_middle_dic[name]['Mean_phase'][day_typeX][awake_session_ind_ind][mouse_recdayX]=\
#                 phase_middle_dic[name]['Mean_phase'][awake_session_ind][mouse_recday]
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ###Phase relationships between neurons 
# Xneuron_phaseangle_dic=rec_dd()
# for day_type in ['2_task','3_task','combined_ABCDonly']:
#     for mouse_recday in day_type_dicX[day_type]:
#         print(mouse_recday)
#         try:
#             num_neurons=len(cluster_dic['good_clus'][mouse_recday])
#             awake_sessions=session_dic_behaviour['awake'][mouse_recday]
#             for awake_session_ind, timestamp in enumerate(awake_sessions):
#                 phases_allneurons=dict_to_array(phase_middle_dic['ALL']['Mean_phase'][awake_session_ind][mouse_recday])
#                 if len(phases_allneurons)>0:
#                     phase_diff_matrix=np.zeros((num_neurons,num_neurons))
#                     for neuron in range(num_neurons):
#                         circ_anglex=circular_angle(phases_allneurons,\
#                                                    np.repeat(phases_allneurons[neuron],len(phases_allneurons)))
#                         pos_anglex=np.asarray([positive_angle([circ_anglex[ii]])[0] if np.isnan(circ_anglex[ii])==False\
#                                                else np.nan for ii in range(len(circ_anglex))])
#                         phase_diff_matrix[neuron]=pos_anglex
#     
#                     Xneuron_phaseangle_dic[awake_session_ind][mouse_recday]=phase_diff_matrix
# 
#         
#         except Exception as e:
#             print(e)
#             
# 

# In[ ]:





# In[ ]:





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
#     days=[]
#     for mouse_recdayX in [mouse_recday1,mouse_recday2]:
#         num_awake_ses=len(session_dic_behaviour['awake'][mouse_recdayX])
#         days.append(np.repeat(mouse_recdayX,num_awake_ses))
#     days=np.hstack((days))
#     for mouse_recdayX in [mouse_recday1,mouse_recday2]:
#         if mouse_recdayX in day_type_dicX['3_task']:
#             continue
#         awake_ses_day=np.where(days==mouse_recdayX)[0]
#         for awake_session_ind_ind, awake_session_ind in enumerate(awake_ses_day): 
#             Xneuron_phaseangle_dic[awake_session_ind_ind][mouse_recdayX]=\
#             Xneuron_phaseangle_dic[awake_session_ind][mouse_recday]
# 

# In[ ]:





# In[ ]:





# ##number of fields - 
# for day_type in ['2_task','3_task','combined_ABCDonly']:
#     for mouse_recday in day_type_dicX[day_type]:
#         print(mouse_recday)
#         num_neurons=len(cluster_dic['good_clus'][mouse_recday])
#         awake_sessions=session_dic_behaviour['awake'][mouse_recday]
#         for awake_session_ind, timestamp in enumerate(awake_sessions):
#             for neuron in range(num_neurons):
#                 state_fields_allx=state_field_dic['ALL'][awake_session_ind][mouse_recday][neuron]
#                 if len(state_fields_allx)>0:
#                     state_fields_all=np.hstack(state_fields_allx)
#                 else:
#                     state_fields_all=np.asarray([])
# 
# 
#                 state_fields_state=state_field_dic['State'][awake_session_ind][mouse_recday][neuron]
# 
#                 new_fields=[]
#                 for field in state_fields_state:
#                     new_field=np.intersect1d(field,state_fields_all)
#                     if len(new_field)>0:
#                         new_fields.append(new_field)
#                 new_fields=np.asarray(new_fields, dtype="object")
# 
# 
#                 state_field_dic['State_ALL'][awake_session_ind][mouse_recday][neuron]=new_fields
#                 num_field_dic['State_ALL'][awake_session_ind][mouse_recday][neuron]=len(new_fields)
# 
#                 for day_type, day_type_array in day_type_dicX.items():
#                     if mouse_recday in day_type_array:
#                         state_field_dic['State_ALL'][day_type][awake_session_ind][mouse_recday][neuron]=new_fields
#                         num_field_dic['State_ALL'][day_type][awake_session_ind][mouse_recday][neuron]=len(new_fields)

# ###making arrays for split double days 
# day_typeX='3_task_all'
# day_type='3_task'
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
#     days=[]
#     for mouse_recdayX in [mouse_recday1,mouse_recday2]:
#         num_awake_ses=len(session_dic_behaviour['awake'][mouse_recdayX])
#         days.append(np.repeat(mouse_recdayX,num_awake_ses))
#     days=np.hstack((days))
#     for mouse_recdayX in [mouse_recday1,mouse_recday2]:
#         if mouse_recdayX in day_type_dicX['3_task']:
#             continue
#         awake_ses_day=np.where(days==mouse_recdayX)[0]
#         for awake_session_ind_ind, awake_session_ind in enumerate(awake_ses_day): 
#             
#             state_field_dic['State_ALL'][awake_session_ind_ind][mouse_recdayX]=\
#             state_field_dic['State_ALL'][awake_session_ind][mouse_recday]
#             
#             num_field_dic['State_ALL'][awake_session_ind_ind][mouse_recdayX]=\
#             num_field_dic['State_ALL'][awake_session_ind][mouse_recday]
#             
# 
#             state_field_dic['State_ALL'][day_typeX][awake_session_ind_ind][mouse_recdayX]=\
#             state_field_dic['State_ALL'][awake_session_ind][mouse_recday]
#             num_field_dic['State_ALL'][day_typeX][awake_session_ind_ind][mouse_recdayX]=\
#             num_field_dic['State_ALL'][awake_session_ind][mouse_recday]
#                     
# 

# In[ ]:





# In[ ]:


##Phase


# ###phase tuning 3 task vs 2 task days
# 
# max_num_fields=4
# ses_ind=1
# print('session'+str(ses_ind))
# name='ALL'
# for day_type, day_type_array in day_type_dicX.items():
#     print(day_type)
#     num_fields=flatten(num_field_dic[name][day_type][0])
#     num_field_boolean=num_fields<=max_num_fields
# 
# 
# 
#     phase_tuning_=flatten(phase_middle_dic[name]['Mean_phase'][day_type][ses_ind])[num_field_boolean]
#     phase_tuning_radians=np.asarray([math.radians(phase_tuning_[ii]) for ii in range(len(phase_tuning_))])
#     exec('phase_tuning_'+day_type+'_ses'+str(ses_ind)+'=phase_tuning_'+day_type+'=phase_tuning_radians')
# 
#     angles_all=plt.hist(phase_tuning_,np.linspace(0,360,37))[0]
#     plt.close()
# 
#     polar_plot_stateX(angles_all,angles_all,angles_all,color='black',labels='angles',plot_type='bar')
#     plt.savefig(Ephys_output_folder+day_type+'_session'+str(ses_ind)+'phase_overrepresentation_polarplot.svg')
# 
# 
#     plt.show()
# 
# #print(circ_st.watson_williams(remove_nan(phase_tuning_2_task),remove_nan(phase_tuning_3_task))[0])
#     
# #print('')
# #print('session0 vs session2')
# #print(circ_st.watson_williams(remove_nan(phase_tuning_3_task_ses0),remove_nan(phase_tuning_3_task_ses2))[0])
# #print(circ_st.watson_williams(remove_nan(phase_tuning_2_task_ses0),remove_nan(phase_tuning_2_task_ses2))[0])
# #print('______')

# In[ ]:





# In[ ]:





# In[178]:


### phase tuning z-scored
sigma=10
activity_neuronz_mean_dic=rec_dd()
for day_type in ['3_task','combined_ABCDonly']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]

        for awake_ses_ind in np.arange(len(awake_sessions)):
            
            try:
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(awake_ses_ind)+'.npy')
            except:
                print('Ephys file not found')
                continue
                
            if len(ephys_)==0:
                print('No ephys')
                continue
            ephys_mean_=np.nanmean(ephys_,axis=1)
            ephys_mean_smoothed_=np.vstack(([smooth_circular(ephys_mean_[neuron],sigma=sigma)                                 for neuron in np.arange(len(ephys_mean_))]))
                
            try:
                activity_neuronz_mean_all=np.zeros((num_neurons,90))
                for neuron in range(num_neurons):
                    activity_neuron=(ephys_mean_smoothed_[neuron]).reshape(4,90)

                    activity_neuronz=st.zscore(activity_neuron,axis=1)
                    activity_neuronz_mean=np.mean(activity_neuronz,axis=0)
                    
                    if len(activity_neuronz_mean)!=90:
                        activity_neuronz_mean=np.repeat(np.nan,90)

                    activity_neuronz_mean_all[neuron]=activity_neuronz_mean


                activity_neuronz_mean_dic[awake_ses_ind][mouse_recday]=activity_neuronz_mean_all
            except Exception as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('Not made')
                continue


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[180]:


###making arrays for split double days 
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]
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
        awake_ses_day=np.where(days==mouse_recdayX)[0]
        for awake_session_ind_ind, awake_session_ind in enumerate(awake_ses_day): 
            
            activity_neuronz_mean_dic[awake_session_ind_ind][mouse_recdayX]=            activity_neuronz_mean_dic[awake_session_ind][mouse_recday]


# In[ ]:





# In[ ]:





# In[181]:


###phase tuning matrices

max_num_fields=4
name='ALL'
day_type='3_task_all' 
day_type_array=day_type_dicX[day_type]
#num_fields=flatten(num_field_dic[name][day_type][0])
#num_field_boolean=num_fields<=max_num_fields

remove_incomplete=True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
if day_type=='combined':
    num_ses=8
else:
    num_ses=4

fig, ax = plt.subplots(1,num_ses,figsize=plt.figaspect(1)*2)

fig.tight_layout()

exclude_list=[]
for mouse_recday in day_type_array:
    num_neurons=len(activity_neuronz_mean_dic[0][mouse_recday])
    for ses_ind, col in enumerate(ax):
        if len(activity_neuronz_mean_dic[ses_ind][mouse_recday])==0:
            arrayx=np.zeros((num_neurons,90))
            arrayx[:]=np.nan
            activity_neuronz_mean_dic[ses_ind][mouse_recday]=arrayx
        if np.isnan(np.mean(activity_neuronz_mean_dic[ses_ind][mouse_recday]))==True:
            exclude_list.append(mouse_recday)
exclude_list=np.unique(np.asarray(exclude_list))

day_type_array_updated=np.setdiff1d(day_type_array,exclude_list)

activity_neuronz_mean_ALL_allses=[]
for ses_ind, col in enumerate(ax):
    if remove_incomplete==True:
        day_type_array_used=day_type_array_updated
    else:
        day_type_array_used=day_type_array
        
    activity_neuronz_mean_ALL=np.vstack([activity_neuronz_mean_dic[ses_ind][mouse_recday]                                         for mouse_recday in day_type_array_used])
    
    activity_neuronz_mean_ALL_allses.append(activity_neuronz_mean_ALL)

    if ses_ind==0:
        sorted_ses0=np.argsort(np.argmax(activity_neuronz_mean_ALL,axis=1)) ## takes first max

    activity_neuronz_mean_ALL_sorted=activity_neuronz_mean_ALL[sorted_ses0]
    fig.suptitle(day_type)
    col.matshow(activity_neuronz_mean_ALL_sorted)
    col.axvline(30, color='white')
    col.axvline(60, color='white')

plt.margins(0,0)
plt.tight_layout()
plt.savefig(Ephys_output_folder_dropbox+'Phase_'+day_type+'_overrepresentation_aligned_plots.svg')


# In[182]:


len(day_type_array_updated)


# In[183]:


np.arange(len(activity_neuronz_mean_ALL_allses)-1)+1


# In[184]:


All_phase_corrs=np.vstack(([[st.pearsonr(activity_neuronz_mean_ALL_allses[0][neuron],                                         activity_neuronz_mean_ALL_allses[ses_ind][neuron])[0] for neuron in np.arange(len(activity_neuronz_mean_ALL_allses[0]))]
for ses_ind in [1,2]]))

mean_phase_corrs=np.nanmean(All_phase_corrs,axis=0)


# In[185]:



plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.hist(mean_phase_corrs,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed',linewidth=4)

plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'Phase_correlations.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(mean_phase_corrs))
print(st.ttest_1samp(mean_phase_corrs,0))
print('')


# In[ ]:





# In[ ]:





# In[15]:


### phase tuning z-scored
sigma=10
activity_neuronz_mean_ABCDE_dic=rec_dd()
for day_type in ['combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]
        
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

        for awake_ses_ind in np.arange(len(awake_sessions)):
            abstract_structure_ses=abstract_structures[awake_ses_ind]
            num_states=len(abstract_structure_ses)
            
            try:
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(awake_ses_ind)+'.npy')
            except:
                print('Ephys file not found')
                continue
                
            if len(ephys_)==0:
                print('No ephys')
                continue
            ephys_mean_=np.nanmean(ephys_,axis=1)
            ephys_mean_smoothed_=np.vstack(([smooth_circular(ephys_mean_[neuron],sigma=sigma)                                 for neuron in np.arange(len(ephys_mean_))]))
                
            try:
                activity_neuronz_mean_all=np.zeros((num_neurons,90))
                for neuron in range(num_neurons):
                    activity_neuron=(ephys_mean_smoothed_[neuron]).reshape(num_states,90)

                    activity_neuronz=st.zscore(activity_neuron,axis=1)
                    activity_neuronz_mean=np.mean(activity_neuronz,axis=0)
                    
                    if len(activity_neuronz_mean)!=90:
                        activity_neuronz_mean=np.repeat(np.nan,90)

                    activity_neuronz_mean_all[neuron]=activity_neuronz_mean


                activity_neuronz_mean_ABCDE_dic[abstract_structure_ses][mouse_recday][awake_ses_ind]=                activity_neuronz_mean_all
                
                activity_neuronz_mean_ABCDE_dic[mouse_recday][awake_ses_ind]=                activity_neuronz_mean_all
            except Exception as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('Not made')
                continue


# In[16]:


###phase tuning matrices

day_type_array=['ab03_21112023_22112023','ah07_21112023_22112023'] ##replace with days that have both ABCD and ABCDE
remove_incomplete=False

if day_type=='combined':
    num_ses=8
elif day_type=='combined_ABCDE':
    num_ses=8
else:
    num_ses=4

fig, ax = plt.subplots(1,num_ses,figsize=plt.figaspect(1)*2)

fig.tight_layout()

exclude_list=[]
for mouse_recday in day_type_array:
    num_neurons=len(activity_neuronz_mean_dic[0][mouse_recday])
    for ses_ind, col in enumerate(ax):
        if len(activity_neuronz_mean_dic[ses_ind][mouse_recday])==0:
            arrayx=np.zeros((num_neurons,90))
            arrayx[:]=np.nan
            activity_neuronz_mean_dic[ses_ind][mouse_recday]=arrayx
        if np.isnan(np.mean(activity_neuronz_mean_dic[ses_ind][mouse_recday]))==True:
            exclude_list.append(mouse_recday)
exclude_list=np.unique(np.asarray(exclude_list))

day_type_array_updated=np.setdiff1d(day_type_array,exclude_list)

activity_neuronz_mean_ALL_allses=[]
for ses_ind, col in enumerate(ax):
    if remove_incomplete==True:
        day_type_array_used=day_type_array_updated
    else:
        day_type_array_used=day_type_array
        
    activity_neuronz_mean_ALL=np.vstack([activity_neuronz_mean_ABCDE_dic[mouse_recday][ses_ind]                                         for mouse_recday in day_type_array_used])
    
    activity_neuronz_mean_ALL_allses.append(activity_neuronz_mean_ALL)

    if ses_ind==0:
        sorted_ses0=np.argsort(np.argmax(activity_neuronz_mean_ALL,axis=1)) ## takes first max

    activity_neuronz_mean_ALL_sorted=activity_neuronz_mean_ALL[sorted_ses0]
    fig.suptitle(day_type)
    col.matshow(activity_neuronz_mean_ALL_sorted)
    col.axvline(30, color='white')
    col.axvline(60, color='white')

plt.margins(0,0)
plt.tight_layout()
plt.savefig(Ephys_output_folder_dropbox+'Phase_'+day_type+'_overrepresentation_aligned_plots_ABCDvsABCDE.svg')


# In[ ]:





# In[ ]:





# In[17]:


day_type_array=['ab03_21112023_22112023','ah07_21112023_22112023'] 

ABCDE_tasks_firstday=np.hstack(([dict_to_array(activity_neuronz_mean_ABCDE_dic['ABCDE'][mouse_recday])                     for mouse_recday in day_type_array]))

ABCD_tasks_firstday=np.hstack(([dict_to_array(activity_neuronz_mean_ABCDE_dic['ABCD'][mouse_recday])                                for mouse_recday in day_type_array]))
cross_abstracttask_corr_all=[]
for task_ABCDE_ind in np.arange(len(ABCDE_tasks_firstday)):
    for task_ABCD_ind in np.arange(len(ABCD_tasks_firstday)):
        ABCDE_tasks_firstday[task_ABCDE_ind]
        cross_abstracttask_corr=np.hstack((st.pearsonr(ABCDE_tasks_firstday[task_ABCDE_ind,neuron],                                                       ABCD_tasks_firstday[task_ABCD_ind,neuron])[0]         for neuron in np.arange(len(ABCDE_tasks_firstday[0]))))
        
        cross_abstracttask_corr_all.append(cross_abstracttask_corr)
cross_abstracttask_corr_all=np.vstack((cross_abstracttask_corr_all))
cross_abstracttask_corr_mean=np.nanmean(cross_abstracttask_corr_all,axis=0)

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.hist(cross_abstracttask_corr_mean,bins=10,color='grey')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.axvline(0,color='black',ls='dashed')
plt.savefig(Ephys_output_folder_dropbox+'Phase_correlations_ABCDvsABCDE.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(st.ttest_1samp(cross_abstracttask_corr_mean,0))
print(len(cross_abstracttask_corr_mean))


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


###Space/Policy


# ###Policy correlations across sessions
# 
# Policy_corr_dic=rec_dd()
# 
# for day_type in ['2_task','3_task','combined_ABCDonly']:
#     for mouse_recday in day_type_dicX[day_type]:
#         print(mouse_recday)
#         Policy_mat=Policy_dic2[mouse_recday]
#         num_neurons=len(cluster_dic['good_clus'][mouse_recday])
# 
#         if mouse_recday in day_type_dicX['combined']:
#             Policy_mat=dict_to_array(Policy_mat)
# 
#         if len(np.shape(Policy_mat))==0 or np.shape(Policy_mat[0])==():
#             print('Empty')
#             continue
# 
#         Policy_corr=np.asarray([np.nanmean(matrix_triangle(np.corrcoef(Policy_mat[:,ii,:]),direction='lower'))\
#                                 for ii in range(num_neurons)])
# 
#         Policy_corr_dic[mouse_recday] = Policy_corr
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


#### --- EXAMPLES - PRE ANALYSIS --- ###


# def arrange_plot_statecells_persessionX2(mouse_recday,neuron,Data_folder,sessions_included=None\
#                                        ,fignamex=False,sigma=10,\
#                                        save=False,plot=False,figtype='.svg', Marker=False,\
#                                        fields_booleanx=[],measure_type='mean', abstract_structures=[],\
#                                       repeated=False,behaviour_oversampling_factor=3,behaviour_rate=1000,\
#                                        tracking_oversampling_factor=50):
# 
#     awake_sessions=session_dic_behaviour['awake'][mouse_recday]
#     awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
#     
#     colors=np.repeat('blue',len(awake_sessions_behaviour))
#     plot_boolean=np.repeat(False,len(awake_sessions_behaviour))
#     plot_boolean[sessions_included]=True
#     
#     
#     
#     
# 
#     fig= plt.figure(figsize=plt.figaspect(1)*4.5)
#     fig.tight_layout()
#     for awake_session_ind, timestamp in enumerate(awake_sessions_behaviour):
#         structure_abstract=abstract_structures[awake_session_ind]
#         
#         if Num_trials_dic2[mouse_recday][awake_session_ind]<2:
#             print('Not enough trials session'+str(awake_session_ind))
#             continue
#         if timestamp not in awake_sessions:
#             print('Ephys not used for session'+str(awake_session_ind))
#             continue
#             
#             
#         try:
#             norm_activity_all=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
#         except:
#             print('No file found session'+str(awake_session_ind))
#             continue
#         
#         norm_activity_neuron=norm_activity_all[neuron]
#         
#         xxx=np.asarray(norm_activity_neuron).T
#         standardized_FR=np.hstack([np.nanmean(xxx[ii],axis=0) for ii in range(len(xxx))])*\
#         behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
#         standardized_FR_sem=np.hstack([st.sem(xxx[ii],axis=0,nan_policy='omit') for ii in range(len(xxx))])*\
#         behaviour_oversampling_factor*behaviour_rate/tracking_oversampling_factor
#         standardized_FR_smoothed=smooth_circular(standardized_FR,sigma=sigma)            
#         standardized_FR_sem_smoothed=smooth_circular(standardized_FR_sem,sigma=sigma)                    
# 
#         
#         standardized_FR_smoothed_upper=standardized_FR_smoothed+standardized_FR_sem_smoothed
#         standardized_FR_smoothed_lower=standardized_FR_smoothed-standardized_FR_sem_smoothed
#        
#         
#         color=colors[awake_session_ind]
#         
#         ax = fig.add_subplot(1, len(awake_sessions_behaviour), awake_session_ind+1, projection='polar')
#         if len(fields_booleanx)>0:
#             polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,\
#                               ax,color=color, Marker=Marker,fields_booleanx=fields_booleanx[awake_session_ind],\
#                              structure_abstract=structure_abstract,repeated=repeated)
#         else:
#             polar_plot_stateX2(standardized_FR_smoothed,standardized_FR_smoothed_upper,standardized_FR_smoothed_lower,\
#                               ax,color=color, Marker=False,structure_abstract=structure_abstract,repeated=repeated)
#     plt.margins(0,0)
#     plt.tight_layout()
#     if save==True:
#         plt.savefig(fignamex+str(awake_session_ind)+figtype)
#     if plot==True & plot_boolean[awake_session_ind]==True:
#         plt.show()
#     else:
#         plt.close() 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
1-Occupancy_dic: conctenate across all trials (one big sequence for each task)
-clean (remove mistakes)
2-trigger activity on different locations: plot 5 trials either side
3-Find prefered location
4-add phase info

'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#### --- MODULES --- ###


# In[ ]:





# ##Correlations within neurons across sessions
# 
# tt=time.time()
# angle_units=2
# angle_values=np.linspace(0,int(360-angle_units),int(360/angle_units))
# 
# Xsession_correlations=rec_dd()
# 
# for day_type, day_type_array in day_type_dicX.items():
#     if day_type not in ['3_task','combined_ABCDonly']:
#         continue
#         
#     for mouse_recday in day_type_array:
#         try:
#             print(mouse_recday)
# 
#             num_neurons=len(cluster_dic['good_clus'][mouse_recday])
# 
#             if len(Xsession_correlations[day_type]['Max_bins'][mouse_recday])==num_neurons:
#                 print('Already analyzed')
#                 continue
# 
#             awake_sessions=session_dic_behaviour['awake'][mouse_recday]
#             correlations_all=np.zeros((num_neurons,len(awake_sessions),len(angle_values)))
#             max_all=np.zeros((num_neurons,len(awake_sessions)))
#             for neuron in range(num_neurons):
#                 smoothed_mean_neuron0=smoothed_activity_dic['Mean'][mouse_recday][neuron][0]
# 
#                 for sesY in (np.arange(int(len(awake_sessions)-1))+1).astype(int):
#                     if Num_trials_dic2[mouse_recday][sesY]==0:
#                         continue
#                     smoothed_mean_neuronY=smoothed_activity_dic['Mean'][mouse_recday][neuron][sesY]
#                     exec('smoothed_mean_neuron'+str(sesY)+'=smoothed_mean_neuronY')
#                     for rotation_angle_ind, rotation_angle in enumerate(angle_values):
#                         smoothed_mean_neuronY_rotated=np.roll(smoothed_mean_neuronY,int(rotation_angle))
# 
#                         correlation=st.pearsonr(smoothed_mean_neuron0,smoothed_mean_neuronY_rotated)[0]
#                         correlations_all[neuron,sesY-1,rotation_angle_ind]=correlation
# 
#                 ##adding ses2 vs ses3 comparison
#                 for rotation_angle_ind, rotation_angle in enumerate(angle_values):
#                     smoothed_mean_neuronY_rotated=np.roll(smoothed_mean_neuron2,int(rotation_angle))
# 
#                     correlation=st.pearsonr(smoothed_mean_neuron1,smoothed_mean_neuronY_rotated)[0]
#                     correlations_all[neuron,3,rotation_angle_ind]=correlation
# 
#                 max_bins_neuron=max_bin_safe(correlations_all[neuron],axisX=1)#np.argmax(correlations_all[neuron],axis=1)
#                 max_all[neuron]=max_bins_neuron
# 
# 
#                 Xsession_correlations[day_type]['Max_bins'][mouse_recday]=max_all
#                 Xsession_correlations[day_type]['Correlations'][mouse_recday]=correlations_all
#                 Xsession_correlations[day_type]['angle_units'][mouse_recday]=angle_units
#         except Exception as e:
#             print(e)
#             print('Not analysed')
#             
# print(time.time()-tt)

# In[ ]:





# In[195]:


###Calculating rotations using peaks
day_type_array=day_type_dicX['All']
sigma=10

Xsession_correlations_peaks=rec_dd()
Xneuron_correlations_peaks=rec_dd()
Xneuron_correlations_peaks2=rec_dd()

Neuron_peaks_dic=rec_dd()

for day_type, day_type_array in day_type_dicX.items():
    if day_type not in ['3_task','combined_ABCDonly','3_task_all']:
        continue
    for mouse_recday in day_type_array:

        print(mouse_recday)
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        all_neurons=np.arange(num_neurons)
        neurons_used=all_neurons
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]
        rec_day_structure_abstract=recday_numbers_dic['structure_abstract'][mouse_recday]

        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]
        All_sessions=session_dic_behaviour['All'][mouse_recday]
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]
        #awake_session_ids=session_dic_behaviour['awake'][mouse_recday][1]

        ephys_mean_smoothed_all=np.zeros((len(awake_sessions),num_neurons,360))
        ephys_mean_smoothed_all[:]=np.nan
        if num_neurons==0:
            print('No neurons')
            continue
        for awake_session_ind, timestamp in enumerate(awake_sessions):
            try:
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(awake_session_ind)+'.npy')
            except:
                print('Ephys file not found')
                continue
                
            if len(ephys_)==0:
                print('Empty ephys')
                continue
            ephys_mean_=np.nanmean(ephys_,axis=1)
            ephys_mean_smoothed_=np.vstack(([smooth_circular(ephys_mean_[neuron],sigma=sigma)                                     for neuron in np.arange(len(ephys_mean_))]))
            ephys_mean_smoothed_all[awake_session_ind]=ephys_mean_smoothed_


        #peaks_neurons=np.vstack(([np.argmax(ephys_mean_smoothed_all[:,neuron],axis=1) for neuron in np.arange(num_neurons)]))
        peaks_neurons=np.vstack(([[max_bin_safe(ephys_mean_smoothed_all[session,neuron])                                   for session in np.arange(len(awake_sessions))]                                  for neuron in np.arange(num_neurons)]))
        Xsession_diff=np.asarray([np.subtract.outer(peaks_neurons[neuron], peaks_neurons[neuron])                                  for neuron in np.arange(num_neurons)])

        Xsession_diff[Xsession_diff<0]=Xsession_diff[Xsession_diff<0]+360

        Xneuron_diff=np.asarray(([np.subtract.outer(peaks_neurons[:,session],peaks_neurons[:,session]).T               for session in np.arange(len(awake_sessions))]))

        Xneuron_diff[Xneuron_diff<0]=Xneuron_diff[Xneuron_diff<0]+360

        Xneuron_diff=Xneuron_diff.T


        Xneuron_diff2=np.asarray([[np.subtract.outer(peaks_neurons[neuronX], peaks_neurons[neuronY])                                  for neuronX in np.arange(num_neurons)] for neuronY in np.arange(num_neurons)])

        Xneuron_diff2[Xneuron_diff2<0]=Xneuron_diff2[Xneuron_diff2<0]+360


        Xsession_correlations_peaks[mouse_recday]=Xsession_diff
        Xneuron_correlations_peaks[mouse_recday]=Xneuron_diff
        Xneuron_correlations_peaks2[mouse_recday]=Xneuron_diff2
        Neuron_peaks_dic[mouse_recday]=peaks_neurons


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
start here:
-check angles make sense
-run generalisation and coherence analyses
-run clustering analysis

'''


# In[ ]:





# In[ ]:


##Correlations within neurons across sessions

tt=time.time()
angle_units=2
angle_values=np.linspace(0,int(360-angle_units),int(360/angle_units))

#Xsession_correlations=rec_dd()
specific_days=False
specific_day_array=['me10_14122021_15122021']

for day_type, day_type_array in day_type_dicX.items():
    if day_type not in ['3_task','combined_ABCDonly']:
        continue
    
    
    
    for mouse_recday in day_type_array:
        print(mouse_recday)
        
        if specific_days==True:
            if mouse_recday not in specific_day_array:
                continue
        try:
            num_neurons=len(cluster_dic['good_clus'][mouse_recday])

            if len(Xsession_correlations[day_type]['Angles'][mouse_recday])==num_neurons:
                print('Already analyzed')
                continue

            awake_sessions=session_dic_behaviour['awake'][mouse_recday]
            correlations_all=np.zeros((num_neurons,len(awake_sessions),len(awake_sessions),len(angle_values)))
            max_all=np.zeros((num_neurons,len(awake_sessions),len(awake_sessions)))
            angles_all=np.zeros((num_neurons,len(awake_sessions),len(awake_sessions)))
            max_all[:]=np.nan
            angles_all[:]=np.nan
            for neuron in range(num_neurons):
                for sesX in np.arange(len(awake_sessions)):
                    if neuron==0:
                        print(sesX)
                    smoothed_mean_neuronX=smoothed_activity_dic2['Mean'][mouse_recday][sesX][neuron]
                    if Num_trials_dic2[mouse_recday][sesX]==0:
                        if neuron==0:
                            print('No trials completed')
                        continue
                    if len(smoothed_mean_neuronX)==0 or np.isnan(np.nanmean(smoothed_mean_neuronX))==True:
                        if neuron==0:
                            print('No Ephys')
                        continue
                    for sesY in np.arange(len(awake_sessions)):
                        if sesX==sesY:
                            continue                        
                        if Num_trials_dic2[mouse_recday][sesY]==0:
                            continue
                        smoothed_mean_neuronY=smoothed_activity_dic2['Mean'][mouse_recday][sesY][neuron]
                        if len(smoothed_mean_neuronY)==0 or np.isnan(np.nanmean(smoothed_mean_neuronY))==True:
                            continue
                        for rotation_angle_ind, rotation_angle in enumerate(angle_values):
                            smoothed_mean_neuronY_rotated=np.roll(smoothed_mean_neuronY,int(rotation_angle))

                            correlation=st.pearsonr(smoothed_mean_neuronX,smoothed_mean_neuronY_rotated)[0]
                            correlations_all[neuron,sesX,sesY,rotation_angle_ind]=correlation

                        max_bins_neuron=max_bin_safe(correlations_all[neuron,sesX,sesY])#np.argmax(correlations_all[neuron],axis=1)
                        max_all[neuron,sesX,sesY]=max_bins_neuron
                        angles_all[neuron,sesX,sesY]=max_bins_neuron*angle_units


                #Xsession_correlations[day_type]['Max_bins'][mouse_recday]=max_all
                Xsession_correlations[day_type]['Correlations'][mouse_recday]=correlations_all
                #Xsession_correlations[day_type]['angle_units'][mouse_recday]=angle_units
                Xsession_correlations[day_type]['Angles'][mouse_recday]=angles_all
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('Not analysed')

print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:


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

        days=[]
        for mouse_recdayX in [mouse_recday1,mouse_recday2]:
            num_awake_ses=len(session_dic_behaviour['awake'][mouse_recdayX])
            days.append(np.repeat(mouse_recdayX,num_awake_ses))
        days=np.hstack((days))
        for mouse_recdayX in [mouse_recday1,mouse_recday2]:
            if mouse_recdayX in day_type_dicX['3_task']:
                continue
            awake_ses_day=np.where(days==mouse_recdayX)[0]

            Angles_all=np.zeros((num_neurons,len(awake_ses_day),len(awake_ses_day)))
            Correlations_all=np.zeros((num_neurons,len(awake_ses_day),len(awake_ses_day),180))

            for awake_session_ind_indX, awake_session_indX in enumerate(awake_ses_day): 
                for awake_session_ind_indY, awake_session_indY in enumerate(awake_ses_day): 
                    Angles_all[:,awake_session_ind_indX,awake_session_ind_indY]=                    Xsession_correlations[day_type]['Angles'][mouse_recday][:,awake_session_indX,awake_session_indY]
                    Correlations_all[:,awake_session_ind_indX,awake_session_ind_indY]=                    Xsession_correlations[day_type]['Correlations'][mouse_recday][:,awake_session_indX,awake_session_indY]


            Xsession_correlations[day_typeX]['Angles'][mouse_recdayX]=Angles_all
            Xsession_correlations[day_typeX]['Correlations'][mouse_recdayX]=Correlations_all
    except Exception as e:
        print(e)
            
for mouse_recday in day_type_dicX['3_task']:
    Xsession_correlations[day_typeX]['Angles'][mouse_recday]=    Xsession_correlations['3_task']['Angles'][mouse_recday]
    Xsession_correlations[day_typeX]['Correlations'][mouse_recday]=    Xsession_correlations['3_task']['Correlations'][mouse_recday]

    
    


# In[288]:


mouse_recday='ah04_06122021'


# In[271]:


##Examples - rotations
mouse_recday='ah04_06122021'

#for mouse_recday in day_type_dicX['3_task']:
print(mouse_recday)

all_neurons=np.arange(len(cluster_dic['good_clus'][mouse_recday]))
abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]


Xsession_correlations_day_=Xsession_correlations['3_task']['Angles'][mouse_recday]


for neuron in [53, 75 , 90 ,67, 52]:
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
    
    print(Xsession_correlations_day_[neuron])


# In[294]:


Xneuron_correlations_peaks[mouse_recday][90,67]


# In[293]:


Xneuron_correlations['3_task']['Angles'][mouse_recday][90,67]


# In[292]:


print(np.shape(Xsession_correlations_peaks[mouse_recday]))
print(np.shape(Xneuron_correlations_peaks[mouse_recday]))
print(np.shape(Xneuron_correlations_peaks2[mouse_recday]))


# In[274]:


Xsession_correlations_peaks[mouse_recday][90]


# In[ ]:





# In[ ]:


##Correlations between neuron pairs within sessions
tt=time.time()
angle_units=10
angle_values=np.linspace(0,int(360-angle_units),int(360/angle_units))

#Xneuron_correlations=rec_dd()
specific_days=False
specific_day_array=['me10_14122021_15122021']

for day_type, day_type_array in day_type_dicX.items():
    
    if day_type not in ['3_task','combined_ABCDonly']:
        continue
    for mouse_recday in day_type_array:
        print(mouse_recday)
        
        if specific_days==True:
            if mouse_recday not in specific_day_array:
                continue

        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]
        correlations_all=np.zeros((num_neurons,num_neurons,len(awake_sessions),36))
        max_all=np.zeros((num_neurons,num_neurons,len(awake_sessions)))
        angle_all=np.zeros((num_neurons,num_neurons,len(awake_sessions)))
        
        correlations_all[:]=np.nan
        max_all[:]=np.nan
        angle_all[:]=np.nan

        for ses_ind in np.arange(len(awake_sessions)):
            print(ses_ind)
            if Num_trials_dic2[mouse_recday][ses_ind]==0:
                print('Not analyzed')
                continue
            
            smoothed_mean_neuron0=smoothed_activity_dic2['Mean'][mouse_recday][ses_ind][0]
            if len(smoothed_mean_neuron0)==0 or np.isnan(np.nanmean(smoothed_mean_neuron0))==True:
                print('No Ephys')
                continue
                            
            if len(Xneuron_correlations[day_type]['Angles'][mouse_recday])==num_neurons            and np.mean(Xneuron_correlations[day_type]['Angles'][mouse_recday]!=0)            and np.isnan(np.nanmean(Xneuron_correlations[day_type]['Angles'][mouse_recday]))==False:
                print('Already Analyzed')
                continue

            for neuronX in range(num_neurons):
                smoothed_mean_neuronX=smoothed_activity_dic2['Mean'][mouse_recday][ses_ind][neuronX]
                for neuronY in range(num_neurons):
                    smoothed_mean_neuronY=smoothed_activity_dic2['Mean'][mouse_recday][ses_ind][neuronY]

                    for rotation_angle_ind, rotation_angle in enumerate(angle_values):
                        smoothed_mean_neuronY_rotated=np.roll(smoothed_mean_neuronY,int(rotation_angle))

                        correlation=st.pearsonr(smoothed_mean_neuronX,smoothed_mean_neuronY_rotated)[0]
                        correlations_all[neuronX,neuronY,ses_ind,rotation_angle_ind]=correlation

                    max_bins_neuron=max_bin_safe(correlations_all[neuronX,neuronY,ses_ind])
                    max_all[neuronX,neuronY,ses_ind]=max_bins_neuron
                    angle_all[neuronX,neuronY,ses_ind]=max_bins_neuron*angle_units

            
        Xneuron_correlations[day_type]['Correlations'][mouse_recday]=correlations_all
        Xneuron_correlations[day_type]['Angles'][mouse_recday]=angle_all

                
print(time.time()-tt)


# In[ ]:


Xneuron_correlations['3_task_all']['Angles']['ah04_06122021']
#Num_trials_dic2


# In[ ]:


#Xsession_correlations['3_task_all']['Angles']['ah04_06122021']


# In[ ]:


for day_type, day_type_array in day_type_dicX.items():
    
    if day_type not in ['3_task','combined_ABCDonly','3_task_all']:
        continue
    for mouse_recday in day_type_array:
        print(mouse_recday)
        
        if specific_days==True:
            if mouse_recday not in specific_day_array:
                continue

        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        awake_sessions=session_dic_behaviour['awake'][mouse_recday]
        correlations_all=np.zeros((num_neurons,num_neurons,len(awake_sessions),36))
        max_all=np.zeros((num_neurons,num_neurons,len(awake_sessions)))
        angle_all=np.zeros((num_neurons,num_neurons,len(awake_sessions)))


                            
        if len(Xneuron_correlations[day_type]['Angles'][mouse_recday])!=num_neurons        or np.mean(Xneuron_correlations[day_type]['Angles'][mouse_recday]==0)        or np.isnan(np.nanmean(Xneuron_correlations[day_type]['Angles'][mouse_recday]))==True:
            if len(Xneuron_correlations_old[day_type]['Angles'][mouse_recday])>0:
                print('Replacing')
                Xneuron_correlations[day_type]['Angles'][mouse_recday]=                Xneuron_correlations_old[day_type]['Angles'][mouse_recday]

                Xneuron_correlations[day_type]['Correlations'][mouse_recday]=                Xneuron_correlations_old[day_type]['Correlations'][mouse_recday]

            


# In[ ]:





# In[ ]:





# In[ ]:


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

        days=[]
        for mouse_recdayX in [mouse_recday1,mouse_recday2]:
            num_awake_ses=len(session_dic_behaviour['awake'][mouse_recdayX])
            days.append(np.repeat(mouse_recdayX,num_awake_ses))
        days=np.hstack((days))
        for mouse_recdayX in [mouse_recday1,mouse_recday2]:
            if mouse_recdayX in day_type_dicX['3_task']:
                continue
            awake_ses_day=np.where(days==mouse_recdayX)[0]

            Angles_all=np.zeros((num_neurons,num_neurons,len(awake_ses_day)))
            Correlations_all=np.zeros((num_neurons,num_neurons,len(awake_ses_day),36))

            for awake_session_ind_ind, awake_session_ind in enumerate(awake_ses_day): 
                Angles_all[:,:,awake_session_ind_ind]=                Xneuron_correlations[day_type]['Angles'][mouse_recday][:,:,awake_session_ind]

                Correlations_all[:,:,awake_session_ind_ind]=                Xneuron_correlations[day_type]['Correlations'][mouse_recday][:,:,awake_session_ind]


            Xneuron_correlations[day_typeX]['Angles'][mouse_recdayX]=Angles_all
            Xneuron_correlations[day_typeX]['Correlations'][mouse_recdayX]=Correlations_all

    except Exception as e:
        print(e)
            
for mouse_recday in day_type_dicX['3_task']:
    Xneuron_correlations[day_typeX]['Angles'][mouse_recday]=    Xneuron_correlations['3_task']['Angles'][mouse_recday]
    Xneuron_correlations[day_typeX]['Correlations'][mouse_recday]=    Xneuron_correlations['3_task']['Correlations'][mouse_recday]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[346]:


for day_type in ['3_task','combined_ABCDonly','3_task_all']:#,'combined_ABCD_AB']:
    print(day_type)
    day_type_array=day_type_dicX[day_type]
    
    if 'combined' in day_type:
        day_type_='combined'
    else:
        day_type_=day_type

    for mouse_recday in day_type_array:

        Xsession_angle_peaks=Xsession_correlations_peaks[mouse_recday]
        Xsession_angle=Xsession_correlations[day_type]['Angles'][mouse_recday]


# In[361]:


Xsession_angle_peaks[neuron,0]


# In[366]:


neuron=0
num_neurons=len(Xsession_angle_peaks)
diffs=np.vstack(([Xsession_angle_peaks[neuron,0,1:]-Xsession_angle[neuron,0,1:] for neuron in np.arange(num_neurons)]))
#diffs[diffs<0]=diffs[diffs<0]+360
diffs=abs(diffs)
diffs_bool_=np.logical_and(diffs>45,diffs<315)
non_discrepent_boolean=np.sum(diffs_bool_,axis=1)==0


# In[ ]:





# In[ ]:





# In[8]:


##what are the angles between each neuron's tuning across sessions? - Main defining cell

name='State_ALL'
max_num_fields=3
min_num_fields=1
num_fields_allowed=1

used_boolean_dic=rec_dd()
used_pairs_dic=rec_dd()
max_angle_dic=rec_dd()
spatial_sim_subset_dic=rec_dd()

exclude_spatial=False

thr_sim=0.5
thr_prop_corr=0.5

tuning_percentile='95'

lower=coherence_thr=1-math.cos(math.radians(45))
upper=1+math.cos(math.radians(45))#


use_peak=False
use_rotation_peak=False
#mean_rotation_confidence=mean_complex2(concatenate_complex2(dict_to_array(Confidence_rotations_dic['first_second_diff'])))
#thr_confidence=np.percentile(mean_rotation_confidence,50)


for day_type in ['3_task','combined_ABCDonly','3_task_all']:#,'combined_ABCD_AB']:
    print(day_type)
    day_type_array=day_type_dicX[day_type]
    
    if 'combined' in day_type:
        day_type_='combined'
    else:
        day_type_=day_type

    for mouse_recday in day_type_array:
        print(mouse_recday)
        try:

            awake_sessions=session_dic_behaviour['awake'][mouse_recday]
            #angle_units=Xsession_correlations[day_type]['angle_units'][mouse_recday]

            State_tuned_boolean=Tuned_dic2['State'][tuning_percentile][mouse_recday]
            #State_tuned_boolean=Tuned_dic['State_zmax_bool'][mouse_recday]
            Place_tuned_boolean=Tuned_dic2['Place'][tuning_percentile][mouse_recday]
            
            num_neurons=len(State_tuned_boolean)
            
            ##used boolean           
            if exclude_spatial==True:
                used_boolean=np.logical_and(State_tuned_boolean,~Place_tuned_boolean)
                used_boolean=~Place_tuned_boolean
            else:
                used_boolean=State_tuned_boolean ##using all neurons state tuned in half or more tasks

            if use_peak==True:
                Xneuron_angle=Xneuron_correlations_peaks[mouse_recday]
                
            else:
                Xneuron_angle=Xneuron_correlations[day_type]['Angles'][mouse_recday]
            
            if use_rotation_peak==True:
                Xsession_angle_peaks=Xsession_correlations_peaks[mouse_recday]
                Xsession_angle=Xsession_correlations[day_type]['Angles'][mouse_recday]
                diffs=np.vstack(([Xsession_angle_peaks[neuron,0,1:]-Xsession_angle[neuron,0,1:]                                  for neuron in np.arange(num_neurons)]))
                diffs=abs(diffs)
                diffs_bool_=np.logical_and(diffs>45,diffs<315)
                non_discrepent_boolean=np.sum(diffs_bool_,axis=1)==0
                
                
                used_boolean=np.logical_and(State_tuned_boolean,non_discrepent_boolean)

            
            Xneuron_angle_used=np.asarray(Xneuron_angle)[used_boolean,:,0][:,used_boolean]
            pair_indices=np.tril_indices(len(Xneuron_angle_used),k=-1)
            used_neurons=np.where(used_boolean==True)[0]
            used_pairs=np.vstack((used_neurons[pair_indices[0]],used_neurons[pair_indices[1]]))


            used_boolean_dic[mouse_recday]=used_boolean
            used_boolean_dic[day_type][mouse_recday]=used_boolean
            used_pairs_dic[mouse_recday]=used_pairs

            abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

            awake_sessions_ABCD=awake_sessions[abstract_structures=='ABCD']
            #if day_type=='combined':
            awake_sessions_AB=awake_sessions[abstract_structures=='AB']

            for ses_ind, ses in enumerate(awake_sessions):
                Xneuron_angle_ses=np.asarray(Xneuron_angle)[used_boolean,:,ses_ind][:,used_boolean]
                if day_type=='3_task':
                    pair_indices=np.tril_indices(len(Xneuron_angle_ses),k=-1)
                    max_angle_dic[ses_ind][mouse_recday]=Xneuron_angle_ses[pair_indices]

                elif 'combined' in day_type:
                    if ses in awake_sessions_ABCD:
                        max_angle_dic['combined']['ABCD'][ses_ind][mouse_recday]=Xneuron_angle_ses[pair_indices]

                    elif ses in awake_sessions_AB:
                        max_angle_dic['combined']['AB'][ses_ind][mouse_recday]=Xneuron_angle_ses[pair_indices]
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


day_typeX='3_task'
day_type='3_task_all'
for mouse_recday in day_type_dicX[day_typeX]:
    used_boolean_dic[day_type][mouse_recday]=used_boolean_dic[day_typeX][mouse_recday]


# In[ ]:





# In[ ]:





# In[ ]:





# In[158]:


'''
SUBSETTING GALORE

num_fields=dict_to_array(num_field_dic[name][day_type_][0][mouse_recday])
all_neurons_rotationbin=Xsession_correlations[day_type]['Max_bins'][mouse_recday]

##num field
#num_field_boolean=np.logical_and((num_fields>=min_num_fields),(num_fields<=max_num_fields))
#num_field_boolean=np.logical_or((num_fields==1),(num_fields==3))

##Tuned
Tuned_boolean=Tuned_dic['State'][mouse_recday]

###generalizing neurons
firstTask_rotation=sameTask_rotation=(all_neurons_rotationbin[:,0]*angle_units).astype(int)
firstTask_gen_boolean=np.logical_or(firstTask_rotation<45,firstTask_rotation>315)
secondTask_rotation=sameTask_rotation=(all_neurons_rotationbin[:,1]*angle_units).astype(int)
secondTask_gen_boolean=np.logical_or(secondTask_rotation<45,secondTask_rotation>315)
bothTasks_gen_boolean=np.logical_and(firstTask_gen_boolean,secondTask_gen_boolean)

##adjust for any number of tasks

##Stable neurons (stable across sessions of same task)
sameTask_rotation=(all_neurons_rotationbin[:,same_task_ses-1]*angle_units).astype(int)
sameTask_stable_boolean=np.logical_or(sameTask_rotation<45,sameTask_rotation>315)

##place cell booleans
#place_cell_booleanx=GLM_dic2["place_cells"][mouse_recday]
#place_type_cell_boolean=GLM_dic2["place_type_cells"][mouse_recday]
#place_cell_boolean=np.logical_or(place_cell_booleanx,place_type_cell_boolean)
#between_task_corr=dict_to_array(Spatial_correlation_dic['within_betweenTask'][mouse_recday])[:,1]
#between_state_corr=np.mean(dict_to_array(Spatial_correlation_dic['withinSession_betweenStates']\
#                                         [mouse_recday]),axis=1)

#prop_corr_statemap=Xsession_correlations_predicted['proportion_correct'][mouse_recday]
#policy_corr=Policy_corr_dic[mouse_recday]


if exclude_spatial==True:
    #place_cell_boolean=between_task_corr>thr_sim
    place_cell_boolean=policy_corr>thr_prop_corr
else:
    place_cell_boolean=sameTask_stable_boolean ##place holder for when not using spatial tuning filter


##confidence in rotations
#rotation_confidence=np.nanmean(Confidence_rotations_dic['first_second_diff'][mouse_recday],axis=1)
#confident_rotation_boolean=rotation_confidence>thr_confidence



##combined booleans
stable_numfield_boolean=[np.logical_and(sameTask_stable_boolean,num_field_boolean)][0]
#stable_numfield_nonplace_boolean=np.logical_and(stable_numfield_boolean,~place_cell_boolean)[0]
stable_numfield_nongen_boolean=np.logical_and(stable_numfield_boolean,~bothTasks_gen_boolean)[0]
numfield_nongen_boolean=np.logical_and(num_field_boolean,~bothTasks_gen_boolean)
numfield_nongen_nonplace_boolean=np.logical_and(numfield_nongen_boolean,~place_cell_boolean)
numfield_nongen_place_boolean=np.logical_and(numfield_nongen_boolean,place_cell_boolean)

numfield_nonplace_boolean=np.logical_and(num_field_boolean,~place_cell_boolean)
numfield_place_boolean=np.logical_and(num_field_boolean,place_cell_boolean)

#confidentrotation_nongen_boolean=np.logical_and(confident_rotation_boolean,~bothTasks_gen_boolean)
#confidentrotation_nongen_nonplace_boolean=np.logical_and(confidentrotation_nongen_boolean,~place_cell_boolean)
#confidentrotation_nonplace_boolean=np.logical_and(confident_rotation_boolean,~place_cell_boolean)


'''


# In[219]:


##Task_num_dic Num_trials_dic2
###making arrays for split double days 
day_typeX='3_task_all'
day_type='combined_ABCDonly'
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]
    rec_day1=rec_day.split('_',1)[0]
    rec_day2=rec_day.split('_',1)[1]
    Date1=rec_day1[-4:]+'-'+rec_day1[2:4]+'-'+rec_day1[:2]
    Date2=rec_day2[-4:]+'-'+rec_day2[2:4]+'-'+rec_day2[:2]

    mouse_recday1=mouse+'_'+rec_day1
    mouse_recday2=mouse+'_'+rec_day2

    num_neurons=len(cluster_dic['good_clus'][mouse_recday])

    days=[]
    for mouse_recdayX in [mouse_recday1,mouse_recday2]:
        num_awake_ses=len(session_dic_behaviour['awake'][mouse_recdayX])
        days.append(np.repeat(mouse_recdayX,num_awake_ses))
    days=np.hstack((days))
    for mouse_recdayX in [mouse_recday1,mouse_recday2]:
        if mouse_recdayX in day_type_dicX['3_task']:
            continue
        awake_ses_day=np.where(days==mouse_recdayX)[0]

        Task_num_dic[mouse_recdayX]=Task_num_dic[mouse_recday][awake_ses_day]

        for awake_session_ind_ind, awake_session_ind in enumerate(awake_ses_day): 
            Num_trials_dic2[mouse_recdayX][awake_session_ind_ind]=            Num_trials_dic2[mouse_recday][awake_session_ind]


# In[ ]:





# In[ ]:





# In[220]:


###Do single neurons generalize?

day_type='3_task_all'
ref_ses=0

coherence_thr=1-math.cos(math.radians(45)) #i.e. below 45 degrees either side (refine to first coherence bin?)

angles_all=[]
angles_X_all=[]
dual_prop_all=[]
single_prop_all=[]

single_prop_all_X=[]

num_neurons_all=[]
num_neurons_all_XX=[]
mouse_days_used_=[]
mouse_days_used_XX_=[]

use_peak=False

if day_type!='combined_ABCDonly':
    num_states=4
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    #try:
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    sessions=Task_num_dic[mouse_recday]
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

    X_all=np.where(sessions==sessions[ref_ses])[0]
    X_rep=np.intersect1d(X_all,repeat_ses)

    non_ref_ses=np.setdiff1d(non_repeat_ses,ref_ses)
    
    if use_peak==True:
        Xsession_correlations_day=Xsession_correlations_peaks[mouse_recday]
    else:
        Xsession_correlations_day=Xsession_correlations[day_type]['Angles'][mouse_recday]
    used_boolean=used_boolean_dic[day_type][mouse_recday]
    Xsession_correlations_day_used=Xsession_correlations_day[used_boolean]

    angles_day=Xsession_correlations_day_used[:,ref_ses,non_ref_ses]
    angles_day2=angles_day[:,:2]
    if np.shape(angles_day2)[1]==2:
        angles_all.append(angles_day2) ##only include first 2 comparison sessions (i.e. 3 sessions) 
    ##in some days animals get 4 tasks by accident


        distances_day2=1-np.cos(np.deg2rad(angles_day2))
        max_dist=np.max(distances_day2,axis=1)
        dual_prop=np.sum(max_dist<coherence_thr)/len(max_dist)
        single_prop=np.mean([np.sum(distances_day2[:,0]<coherence_thr)/len(distances_day2),                            np.sum(distances_day2[:,1]<coherence_thr)/len(distances_day2)])

        dual_prop_all.append(dual_prop)
        single_prop_all.append(single_prop)

    if len(X_rep)>0:
        mouse_days_used_XX_.append(mouse_recday)
        num_neurons_all_XX.append(np.sum(used_boolean))
        angles_X_day=Xsession_correlations_day_used[:,ref_ses,X_rep[0]]
        angles_X_all.append(angles_X_day)

        distances_day_X=1-np.cos(np.deg2rad(angles_X_day))
        if np.isnan(np.mean(distances_day_X))==False:
            single_prop_X=np.sum(distances_day_X<coherence_thr)/len(distances_day_X)
            single_prop_all_X.append(single_prop_X)

    num_neurons_all.append(np.sum(used_boolean))
    
    mouse_days_used_.append(mouse_recday)
    #except:
    #    print('not used')

print('Total number of neurons')
print(np.sum(num_neurons_all))

single_neuron_angles_all=np.vstack((angles_all))
single_neuron_angles_X_all=remove_nan(np.hstack((angles_X_all)))

###Plotting histograms
hist_all=[]
for session in range(2):
    print(session+1)
    angles_all_hist=np.histogram(single_neuron_angles_all[:,session],np.linspace(0,360,37))[0]
    
    hist_all.append(np.asarray(angles_all_hist))

    #plt.bar(np.arange(len(angles_all)),angles_all)
    #plt.show()
    print(angles_all_hist)
    print(np.sum(angles_all_hist))

    polar_plot_stateX(angles_all_hist,angles_all_hist,angles_all_hist,color='black',labels='angles',plot_type='bar')
    #plt.savefig(Ephys_output_folder+day_type+'_session0tosession'+str(session)+'_anglechange_polarplot.svg')
    plt.show()

hist_all=np.asarray(hist_all)

mean_hist=np.mean(hist_all,axis=0)
polar_plot_stateX(mean_hist,mean_hist,mean_hist,color='black',labels='angles',plot_type='bar')
plt.savefig(Ephys_output_folder_dropbox+'Mean_anglechange_polarplot.svg',           bbox_inches = 'tight', pad_inches = 0)
plt.show()

generalising_num=(np.sum(mean_hist[:5])+np.sum(mean_hist[-4:]))
generalissing_prop=generalising_num/np.sum(mean_hist)
print(generalissing_prop)
print(two_proportions_test(generalising_num, np.sum(mean_hist), np.sum(mean_hist)*(1/num_states),np.sum(mean_hist)))

###X vs X'
print(session+1)
angles_all_hist=np.histogram(single_neuron_angles_X_all,np.linspace(0,360,37))[0]
print(angles_all_hist)
print(np.sum(angles_all_hist))
polar_plot_stateX(angles_all_hist,angles_all_hist,angles_all_hist,color='black',labels='angles',plot_type='bar')
plt.savefig(Ephys_output_folder_dropbox+'Mean_anglechange_XvsX_polarplot.svg',           bbox_inches = 'tight', pad_inches = 0)
plt.show()

generalising_num=(np.sum(angles_all_hist[:5])+np.sum(angles_all_hist[-4:]))
generalissing_prop=generalising_num/np.sum(angles_all_hist)
print(generalissing_prop)
print(two_proportions_test(generalising_num, np.sum(angles_all_hist), np.sum(angles_all_hist)*(1/num_states),                           np.sum(angles_all_hist)))

single_prop_all=np.asarray(single_prop_all)
dual_prop_all=np.asarray(dual_prop_all)
single_prop_all_X=np.asarray(single_prop_all_X)

print('')
print('Per day stats')
print('Single prop - across tasks')
print(len(remove_nan(single_prop_all)))
print(np.nanmean(single_prop_all))
print(st.sem(single_prop_all,nan_policy='omit'))
print(st.ttest_1samp(remove_nan(single_prop_all),0.25))

print('Dual prop - across tasks')
print(np.nanmean(dual_prop_all))
print(st.sem(dual_prop_all,nan_policy='omit'))
print(st.ttest_1samp(remove_nan(dual_prop_all),1/16))
                                 
print('Single prop - within tasks')         
print(len(remove_nan(single_prop_all_X)))
print(np.nanmean(single_prop_all_X))
print(st.sem(single_prop_all_X,nan_policy='omit'))
print(st.ttest_1samp(remove_nan(single_prop_all_X),0.25))
print('Answer: Nope')


# In[ ]:





# In[ ]:





# In[221]:


##Do pairs of neurons remain coherent?

Coherence_dic=rec_dd()
coherence_thr=1-math.cos(math.radians(45)) #i.e. below 45 degrees either side (refine to first coherence bin?)
angle_changes=np.linspace(0,360,5)[:-1]

neurons_used_thr=10

day_type='3_task_all'
relative_angles_pairs_all=[]
relative_angles_pairs_X_all=[]

ref_ses=0 ##where angles between neurons are taken

excluded_num_sessions=[]
excluded_num_neurons=[]

included_XYZ_only=[]

neurons_used_all_=[]
use_peak=False
if day_type!='combined_ABCDE':
    num_states=4

for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    #try:
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    sessions=Task_num_dic[mouse_recday]
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)  


    X_all=np.where(sessions==sessions[ref_ses])[0]
    X_rep=np.intersect1d(X_all,repeat_ses)
    non_ref_ses=np.setdiff1d(non_repeat_ses,ref_ses)
    if use_peak==True:
        Xsession_correlations_day=Xsession_correlations_peaks[mouse_recday]
    else:
        Xsession_correlations_day=Xsession_correlations[day_type]['Angles'][mouse_recday]
    used_boolean=used_boolean_dic[day_type][mouse_recday]
    neurons_used=np.where(used_boolean==True)[0]
    Xsession_correlations_day_used=Xsession_correlations_day[used_boolean]

    angles_day=Xsession_correlations_day_used[:,ref_ses,non_ref_ses]
    angles_day2=angles_day[:,:2] ##only include first 2 comparison sessions (i.e. 3 sessions) 
    ##on some days animals get 4 tasks by accident



    if np.shape(angles_day2)[1]<2 or np.isnan(np.nanmean(angles_day2))==True:
        print('Less than 3 sessions - not included')
        excluded_num_sessions.append(mouse_recday)
        continue

    if len(neurons_used)<neurons_used_thr:
        print('Less than '+str(neurons_used_thr)+' usable neurons - not included')
        excluded_num_neurons.append(mouse_recday)
        continue
    
    num_trials_ses=[]
    for ses_ind in non_repeat_ses:
        location_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy',allow_pickle=True)
        num_trials_ses.append(len(location_))
    num_trials_ses=np.hstack((num_trials_ses))
    
    
        
    if np.sum(num_trials_ses>0)<3:
        print('Less than 3 tasks with completed trials')
        excluded_num_sessions.append(mouse_recday)
        continue

    relative_angles_pairs_day=[]
    for session in range(2):
        angles_ses=angles_day2[:,session]
        relative_angles_mat_ses=np.vstack(([positive_angle([circular_angle(angles_ses[neuronX],angles_ses[neuronY])                     for neuronX in np.arange(len(angles_ses))])                    for neuronY in np.arange(len(angles_ses))]))

        relative_angles_pairs_ses=matrix_triangle(relative_angles_mat_ses)
        relative_angles_pairs_day.append(relative_angles_pairs_ses)

    relative_angles_pairs_day=np.asarray(relative_angles_pairs_day).T
    relative_angles_pairs_all.append(relative_angles_pairs_day) 

    if len(X_rep)>0:
        Coherence_dic['Num_neurons_used_XYZX'][mouse_recday]=len(neurons_used)
        angles_X_day=Xsession_correlations_day_used[:,ref_ses,X_rep[0]]
        if np.isnan(np.nanmean(angles_X_day))==False:
            relative_angles_mat_X_day_=np.vstack(([positive_angle([circular_angle(angles_X_day[neuronX],                                                                                 angles_X_day[neuronY])                         for neuronX in np.arange(len(angles_X_day))])                        for neuronY in np.arange(len(angles_X_day))]))
            relative_angles_mat_X_day=matrix_triangle(relative_angles_mat_X_day_)
            relative_angles_pairs_X_all.append(relative_angles_mat_X_day)
            
    else:
        included_XYZ_only.append(mouse_recday)

    ###### coherence proportion

    pairXY=relative_angles_pairs_day[:,0]
    pairXZ=relative_angles_pairs_day[:,1]
    cosine_distXY=1-np.cos(np.deg2rad(pairXY))
    cosine_distXZ=1-np.cos(np.deg2rad(pairXZ))

    cosine_max=np.max((cosine_distXY,cosine_distXZ),axis=0)
    cosine_mean=np.mean((cosine_distXY,cosine_distXZ),axis=0)
    distance_used_pairs=cosine_max

    coherent_prop=len(np.where(distance_used_pairs<coherence_thr)[0])/len(distance_used_pairs)

    ##breakdown by tuning distance

    angles_between_neurons_used=Xneuron_correlations[day_type]['Angles'][mouse_recday]                                         [neurons_used][:,neurons_used][:,:,ref_ses]

    angle_all_all=matrix_triangle(angles_between_neurons_used)
    Tuning_dist=1-np.cos(np.deg2rad(angle_all_all))
    coherence_tuning_prop_all=np.zeros(len(angle_changes))
    tunings=angle_changes/2 ##because taking absolute angle/distance (i.e. maximum angle difference is 180 degrees)
    for tuning_ind,tuning in enumerate(tunings):
        lower_limit=1-math.cos(math.radians(tuning))
        upper_limit=1-math.cos(math.radians(tuning+45))

        coherence_tuning=distance_used_pairs[np.logical_and(Tuning_dist>=lower_limit,Tuning_dist<upper_limit)]
        if len(coherence_tuning)>0:
            coherence_tuning_prop=len(np.where(coherence_tuning<coherence_thr)[0])/len(coherence_tuning)
        else:
            coherence_tuning_prop=np.nan

        coherence_tuning_prop_all[tuning_ind]=coherence_tuning_prop

    Coherence_dic['coherent_prop'][mouse_recday]=coherent_prop
    Coherence_dic['coherent_tuning_prop'][mouse_recday]=coherence_tuning_prop_all
    Coherence_dic['Num_neurons_used'][mouse_recday]=len(neurons_used)

        
        
    #except Exception as e:
    #    print(e)
    #    exc_type, exc_obj, exc_tb = sys.exc_info()
    #    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #    print(exc_type, fname, exc_tb.tb_lineno)
        
relative_angles_pairs_all=np.vstack((relative_angles_pairs_all))
relative_angles_pairs_X_all=remove_nan(np.hstack((relative_angles_pairs_X_all)))

###Plotting histograms
hist_all=[]
for comparison in range(2):
    print(comparison+1)
    session=comparison+1
    relative_angles_all_hist=np.histogram(relative_angles_pairs_all[:,comparison],np.linspace(0,360,37))[0]
    
    hist_all.append(np.asarray(relative_angles_all_hist))

    #plt.bar(np.arange(len(angles_all)),angles_all)
    #plt.show()
    print(relative_angles_all_hist)
    print(np.sum(relative_angles_all_hist))

    polar_plot_stateX(relative_angles_all_hist,relative_angles_all_hist,relative_angles_all_hist,                      color='black',labels='angles',plot_type='bar')
    #plt.savefig(Ephys_output_folder+day_type+'_session0tosession'+str(session)+'_anglechange_polarplot.svg')
    plt.show()

print('')
print('Mean histogram')
hist_all=np.asarray(hist_all)

mean_hist=np.mean(hist_all,axis=0)
polar_plot_stateX(mean_hist,mean_hist,mean_hist,color='black',labels='angles',plot_type='bar')
plt.savefig(Ephys_output_folder_dropbox+'Mean_relative_anglechange_polarplot.svg',           bbox_inches = 'tight', pad_inches = 0)
plt.show()

generalising_num=(np.sum(mean_hist[:5])+np.sum(mean_hist[-4:]))
generalissing_prop=generalising_num/np.sum(mean_hist)
print(generalissing_prop)
print(two_proportions_test(generalising_num, np.sum(mean_hist), np.sum(mean_hist)*(1/num_states),                           np.sum(mean_hist)))


relative_angles_all_X_hist=np.histogram(relative_angles_pairs_X_all,np.linspace(0,360,37))[0]
print(relative_angles_all_X_hist)
print(np.sum(relative_angles_all_X_hist))

polar_plot_stateX(relative_angles_all_X_hist,relative_angles_all_X_hist,relative_angles_all_X_hist,                  color='black',labels='angles',plot_type='bar')
plt.savefig(Ephys_output_folder_dropbox+'Mean_relative_anglechange_XvsX_polarplot.svg',           bbox_inches = 'tight', pad_inches = 0)
plt.show()

generalising_num=(np.sum(relative_angles_all_X_hist[:5])+np.sum(relative_angles_all_X_hist[-4:]))
generalissing_prop=generalising_num/np.sum(relative_angles_all_X_hist)
print(generalissing_prop)
print(two_proportions_test(generalising_num, np.sum(relative_angles_all_X_hist),                           np.sum(relative_angles_all_X_hist)*(1/num_states),                           np.sum(relative_angles_all_X_hist)))

print('')
print('Per day stats')
print('Double coherent')
coherent_prop_all=dict_to_array(Coherence_dic['coherent_prop'])
print(len(remove_nan(coherent_prop_all)))
print(np.mean(coherent_prop_all))
print(st.sem(coherent_prop_all))
print(st.ttest_1samp(coherent_prop_all,1/16))

print('Answer: Yea a little bit')

print('Double coherent - XX')
coherent_prop_all=dict_to_array(Coherence_dic['Num_neurons_used_XYZX'])
print(len(remove_nan(coherent_prop_all)))
print(np.mean(coherent_prop_all))
print(st.sem(coherent_prop_all))
print(st.ttest_1samp(coherent_prop_all,1/16))


# In[222]:


##printing exclusions
print('Days excluded because less than 3 sessions: '+str(len(excluded_num_sessions)))
print(excluded_num_sessions)
print('Days excluded because less than '+str(neurons_used_thr)+' neurons: '+str(len(excluded_num_neurons)))
print(excluded_num_neurons)


# In[223]:


print(np.sum(dict_to_array(Coherence_dic['Num_neurons_used'])))
np.sum(dict_to_array(Coherence_dic['Num_neurons_used_XYZX']))


# In[224]:


generalising_num/np.sum(relative_angles_all_X_hist)


# In[209]:


num_states


# In[ ]:





# In[210]:


###correlate angle between in one session to another

xy=column_stack_clean(relative_angles_pairs_all[:,0],relative_angles_pairs_all[:,1])
relative_angle_ses1=xy[:,0]
relative_angle_ses2=xy[:,1]

#sns.kdeplot(relative_angle_ses1,relative_angle_ses2,fill=True)#
plt.scatter(rand_jitterX(relative_angle_ses1,0.01),rand_jitterX(relative_angle_ses2,0.01), color='black', s=5)
#plt.savefig(Ephys_output_folder+day_type+'_session1tosession2_anglechangecorrelation_kdeplot_noscatter.svg')
plt.show()

#sns.regplot(rand_jitterX(relative_angle_ses1,0.01),rand_jitterX(relative_angle_ses2,0.01))
#plt.savefig(Ephys_output_folder+day_type+'_session1tosession2_anglechangecorrelation_regplot.svg')
#plt.show()

print('Spearman rank correlation: '+str(st.spearmanr(relative_angle_ses1,relative_angle_ses2)))
print('Pearson rank correlation: '+str(st.pearsonr(relative_angle_ses1,relative_angle_ses2)))


# In[ ]:





# In[225]:


mouse_daysnotused_=[]
mice_used=[]
for mouse_recday in day_type_dicX['3_task_all']:
    print(mouse_recday)
    mouse=mouse_recday.split('_',1)[0]
    mice_used.append(mouse)
    if mouse_recday in Coherence_dic['coherent_tuning_prop'].keys():
        coh_tuning_=Coherence_dic['coherent_tuning_prop'][mouse_recday]
        neurons_used_=Coherence_dic['Num_neurons_used'][mouse_recday]
        print(neurons_used_)
        print(coh_tuning_)
    else:
        mouse_daysnotused_.append(mouse_recday)
        print('Not analysed')
        
mice_used=np.unique(np.hstack((mice_used)))


# In[212]:


print(len(Coherence_dic['coherent_tuning_prop'].keys()))
len(Coherence_dic['Num_neurons_used_XYZX'].keys())#[mouse_recday]
#included_XYZ_only


# In[213]:


len(day_type_dicX['3_task_all'])


# In[214]:


len(mouse_daysnotused_)


# In[215]:


mice_used


# In[216]:


'''
Account for all recording days

'''


# In[ ]:





# In[ ]:





# In[226]:


###Plotting coherence proportion as function of task distance
chance_coh=1/4**2
chance_coh2=(((1/4**2)*3+(1/3*1/4)))/4 ##i.e. taking into account non-uniformityies in single neuron rotations


figname=Ephys_output_folder_dropbox+'coherence_proportion_vs_tuningdistance.svg'
coh_tuning_all=dict_to_array(Coherence_dic['coherent_tuning_prop']).T
neurons_used_all=dict_to_array(Coherence_dic['Num_neurons_used'])
coh_tuning_all=coh_tuning_all

bar_plotX(coh_tuning_all,'none',0,0.14,'nopoints','paired',0.025)
plt.axhline(chance_coh,color='black',ls='dashed')
#plt.axhline(chance_coh2,color='black',ls='dashed')
#plt.axhline(np.mean(coherent_prop_shuff),color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=15)
plt.tick_params(width=2, length=6)
plt.savefig(figname)

plt.show()
print(np.nanmean(coh_tuning_all))
print(chance_coh)
print(len(coh_tuning_all.T))
#print(chance_coh2)


ttests_ps=np.asarray([st.ttest_1samp(remove_nan(coh_tuning_all[ii]),1/16)[1] for ii in range(len(coh_tuning_all))])
ttests_stat=np.asarray([st.ttest_1samp(remove_nan(coh_tuning_all[ii]),1/16)[0] for ii in range(len(coh_tuning_all))])
print(ttests_stat)
print(statsmodels.stats.multitest.multipletests(ttests_ps,alpha=0.05))


# In[434]:


2.86721336e-03


# In[228]:


plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False

data=coh_tuning_all.T
# Filter data using np.isnan
mask = ~np.isnan(data)
filtered_data = [d[m] for d, m in zip(data.T, mask.T)]

#sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1)
#plt.errorbar(np.arange(len(means)),means, yerr=sems, marker='o', fmt='.',color='black')

sns.violinplot(filtered_data, color='grey',alpha=0.5)
#sns.stripplot(filtered_data,color='white',edgecolor='black',linewidth=1,alpha=0.5)
plt.axhline(0,color='black')
plt.axhline(chance_coh,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'/Coherence_tuning_violin.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()

sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1)
plt.axhline(0,color='black')
plt.axhline(chance_coh,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'/Coherence_tuning_swarm.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[ ]:


####Clustering


# In[ ]:





# In[218]:


###Embedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

coherence_thr=1-math.cos(math.radians(45)) #i.e. below 45 degrees either side (refine to first coherence bin?)
angle_changes=np.linspace(0,360,5)[:-1]

#module_dic=rec_dd()
plot_=False
num_components=2
day_type='3_task_all'
use_peak=False
relative_angles_pairs_all=[]

ref_ses=0 ##where angles between neurons are taken
for mouse_recday in day_type_dicX[day_type]:
    #mouse_recday='ah03_18082021'
    print(mouse_recday)
    
    if mouse_recday not in Coherence_dic['coherent_tuning_prop'].keys():
        print('Day excluded from pairwise analysis so exluded here too')
        continue

    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    sessions=Task_num_dic[mouse_recday]
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)  

    non_ref_ses=np.setdiff1d(non_repeat_ses,ref_ses)
    
    if use_peak==True:
        Xsession_correlations_day=Xsession_correlations_peaks[mouse_recday]
    else:
        Xsession_correlations_day=Xsession_correlations[day_type]['Angles'][mouse_recday]
    used_boolean=used_boolean_dic[day_type][mouse_recday]
    neurons_used=np.where(used_boolean==True)[0]
    Xsession_correlations_day_used=Xsession_correlations_day[used_boolean]

    angles_day=Xsession_correlations_day_used[:,ref_ses,non_ref_ses]
    angles_dayX=angles_day[:,:2] ##only include first 2 comparison sessions (i.e. 3 sessions) 
    ##in some days animals get 4 tasks by accident
    
    #if np.shape(angles_dayX)[1]<2 or np.isnan(np.nanmean(angles_dayX))==True:
    #    print('Less than 3 sessions - not included')
    #    print('')
    #    continue


    #if len(neurons_used)<10:
    #    print('No usable neurons')
    #    print('')
    #    continue


    relative_angles_mat_day=[]
    for session in range(2):
        angles_ses=angles_dayX[:,session]
        relative_angles_mat_ses=np.vstack(([positive_angle([circular_angle(angles_ses[neuronX],angles_ses[neuronY])                     for neuronX in np.arange(len(angles_ses))])                    for neuronY in np.arange(len(angles_ses))]))
        relative_angles_mat_day.append(relative_angles_mat_ses)

    relative_angles_mat_day=np.asarray(relative_angles_mat_day)        

    distances_mat_day=1-np.cos(np.deg2rad(relative_angles_mat_day))
    distances_mat_day_max=np.max(distances_mat_day,axis=0)

    distance_matrix_used=distances_mat_day_max

    #for num_components in [1,2]:
    #MDS
    embedding = MDS(n_components=num_components,dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(distance_matrix_used)

    #tSNE
    if num_components==1:
        constant=20
    else:
        constant=0

    perp_=5*10**(2-num_components)+constant
    if perp_>=len(distance_matrix_used):
        print('Perplexity not less than number of samples')
        continue

    embedding_TSNE=TSNE(n_components=num_components, init='random',metric='precomputed',                        perplexity=perp_)
    X_transformed_TSNE = embedding_TSNE.fit_transform(distance_matrix_used)

    if plot_==True:
        if num_components==2:
            print(X_transformed.shape)
            print('MDS')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.scatter(X_transformed[:,0],X_transformed[:,1],s=5)
            #ax.set_aspect('equal', adjustable='box')
            plt.show()

            print('tSNE')
            plt.scatter(X_transformed_TSNE[:,0],X_transformed_TSNE[:,1],s=5)
            plt.show()
            print(embedding_TSNE.kl_divergence_)
            print('')
            print('')

    #module_dic[num_components]['distance_matrix_all'][mouse_recday]=distance_matrix_all
    module_dic[num_components]['distance_matrix_used'][mouse_recday]=distance_matrix_used
    module_dic[num_components]['MDS'][mouse_recday]=X_transformed
    module_dic[num_components]['tSNE'][mouse_recday]=X_transformed_TSNE
    module_dic[num_components]['tSNE_KL_divergence'][mouse_recday]=embedding_TSNE.kl_divergence_
    module_dic[num_components]['perplexity_used_tSNE'][mouse_recday]=perp_


# In[ ]:





# In[ ]:





# In[219]:


##Clustering 
import sklearn
from sklearn.metrics import silhouette_samples, silhouette_score
num_components=2
day_type='3_task_all'
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    
    if mouse_recday not in Coherence_dic['coherent_tuning_prop'].keys():
        print('Day excluded from pairwise analysis so exluded here too')
        continue
        
    X_transformed_=module_dic[num_components]['tSNE'][mouse_recday]
    
    if len(X_transformed_)==0:
        print('Not analysed')
        print('')
        continue

    clustering=sklearn.cluster.AgglomerativeClustering(distance_threshold=50,n_clusters=None)

    X_clustered = clustering.fit(X_transformed_)

    clusters = X_clustered.labels_
    cluster_labels=np.unique(clusters)    
    cluster_distances=np.hstack((0,X_clustered.distances_[-len(cluster_labels)+1:]))

    cluster_positions=np.zeros(len(clusters))
    for cluster in cluster_labels:
        X_cluster=X_transformed_[clusters==cluster]
        if cluster == -1:
            plt.scatter(X_cluster[:,0],X_cluster[:,1],s=20, marker='x', color='black')
        else:
            plt.scatter(X_cluster[:,0],X_cluster[:,1],s=10)

        cluster_positions[clusters==cluster]=cluster_distances[cluster]


    plt.savefig(Ephys_output_folder_dropbox+'/'+str(mouse_recday)+'_coherence_clusters.svg')
    plt.show()
    #print(module_dic[num_components]['tSNE_KL_divergence'][mouse_recday])


    plot_dendrogram(X_clustered, truncate_mode='lastp', p=len(cluster_labels))
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    if len(cluster_labels)>1 and len(clusters)>len(cluster_labels):
        silhouette_score_=silhouette_score(X_transformed_,clusters)
    else:
        silhouette_score_=np.nan

    print(silhouette_score_)
    module_dic[num_components]['clusters'][mouse_recday]=clusters
    module_dic[num_components]['cluster_positions'][mouse_recday]=cluster_positions
    module_dic[num_components]['silhouette_score'][mouse_recday]=silhouette_score_


# In[30]:


len(module_dic[num_components]['clusters'].keys())


# In[188]:


module_dic[num_components]['silhouette_score']


# In[ ]:





# mouse_recday='ah04_10122021'
# used_pairs=used_pairs_dic[mouse_recday]
# neurons_used=np.unique(used_pairs)
# X_transformed_tuning_all=np.zeros((4,len(neurons_used),2))
# X_transformed_tuning_TSNE_all=np.zeros((4,len(neurons_used),2))
# 
# 
# num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
# sessions=Task_num_dic[mouse_recday]
# repeat_ses=np.where(rank_repeat(sessions)>0)[0]
# non_repeat_ses=non_repeat_ses_maker(mouse_recday)  
# 
# non_ref_ses=np.setdiff1d(non_repeat_ses,ref_ses)
# 
# Xsession_correlations_day=Xsession_correlations[day_type]['Angles'][mouse_recday]
# used_boolean=used_boolean_dic[day_type][mouse_recday]
# neurons_used=np.where(used_boolean==True)[0]
# 
# 
# Xsession_correlations_day_used=Xsession_correlations_day[used_boolean]
# 
# angles_day=Xsession_correlations_day_used[:,ref_ses,non_ref_ses]
# angles_dayX=angles_day[:,:2] ##only include first 2 comparison sessions (i.e. 3 sessions) 
# ##in some days animals get 4 tasks by accident
# 
# relative_angles_mat_day=[]
# for session in range(2):
#     angles_ses=angles_dayX[:,session]
#     relative_angles_mat_ses=np.vstack(([positive_angle([circular_angle(angles_ses[neuronX],angles_ses[neuronY])\
#                  for neuronX in np.arange(len(angles_ses))])\
#                 for neuronY in np.arange(len(angles_ses))]))
#     relative_angles_mat_day.append(relative_angles_mat_ses)
# 
# relative_angles_mat_day=np.asarray(relative_angles_mat_day)        
# 
# distances_mat_day=1-np.cos(np.deg2rad(relative_angles_mat_day))
# distances_mat_day_max=np.max(distances_mat_day,axis=0)
# 
# distance_matrix_used=distances_mat_day_max
# 
# 
# for ses_ind in np.arange(4):
#     tuning_angle_mat=Xneuron_correlations['3_task_all']['Angles'][mouse_recday][:,:,ses_ind]
#     cosine_tuning_dist=1-np.asarray([[math.cos(math.radians(tuning_angle_mat[ii,jj]))\
#                                     for ii in range(len(tuning_angle_mat))] for jj in range(len(tuning_angle_mat))])
#     num_components=2
#     
#     distance_matrix_used=cosine_tuning_dist[neurons_used][:,neurons_used]
#     #MDS
#     embedding = MDS(n_components=num_components,dissimilarity='precomputed')
#     X_transformed_tuning = embedding.fit_transform(distance_matrix_used)
#     X_transformed_tuning_all[ses_ind]=X_transformed_tuning
# 
#     #tSNE
#     embedding_TSNE=TSNE(n_components=num_components, init='random',metric='precomputed',perplexity=5)
#     X_transformed_tuning_TSNE = embedding_TSNE.fit_transform(distance_matrix_used)
#     X_transformed_tuning_TSNE_all[ses_ind]=X_transformed_tuning_TSNE
#     
# tuning_dim=X_transformed_tuning_all[0]
# coherence_dim=module_dic[1]['tSNE'][mouse_recday]
# plt.scatter(np.repeat(1,len(coherence_dim)),coherence_dim,s=2)
# plt.show()
# plt.scatter(tuning_dim[:,0], tuning_dim[:,1],s=2)
# plt.show()
# fig = plt.figure(figsize=plt.figaspect(3)*1.5)
# ax = fig.add_subplot(projection='3d')
# 
# ax.scatter(tuning_dim[:,0], tuning_dim[:,1], coherence_dim,s=100)
# ax.view_init(0, 30)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#     
#    
#     
#     

# In[172]:


day_type_dicX['3_task_all']


# In[173]:


day_type_dicX['3_task']


# In[299]:


mouse_recday='ah04_10122021'
num_neurons=len(cluster_dic['good_clus'][mouse_recday])
num_neurons


# In[297]:


#Example modules
mouse_recday='me10_08122021'
num_neurons=len(cluster_dic['good_clus'][mouse_recday])
used_pairs=used_pairs_dic[mouse_recday]
neurons_used=np.unique(used_pairs)
num_components=2


### arranging by tuning

ses_seed=0
ses_seed2=3



state_tuning_all=Neuron_peaks_dic[mouse_recday]

state_tuning_used=state_tuning_all[neurons_used][:,0]

num_fields_all=np.hstack(([np.sum(unique_adjacent(FR_shuff_dic['percentiles_state'][0][mouse_recday][neuron]>95))                      for neuron in range(num_neurons)]))
num_fields_used=num_fields_all[neurons_used]

#positive_angle(circular_angle(state_tuning_used,state_tuning_used2)))

single_peak_neurons=np.where(num_fields_used==1)[0]
triple_peak_neurons=np.where(num_fields_used==3)[0]
if len(single_peak_neurons)==0:
    X_peak_neurons=triple_peak_neurons
else:
    X_peak_neurons=single_peak_neurons
    
seed_neuron=X_peak_neurons[-1]
#seed_neuron=7


tuning_angle_rel=Xneuron_correlations['3_task_all']['Angles'][mouse_recday][seed_neuron,:,ses_seed]
tuning_angle_rel_used=tuning_angle_rel[neurons_used]
state_tuning_used_seeded=(tuning_angle_rel_used+state_tuning_used[seed_neuron])%360

state_tuning_used_seeded=state_tuning_used

###plotting

tuning_dim_=X_transformed_tuning_all[ses_seed]
tuning_dim=rotate(tuning_dim_,origin=[0,0],degrees=0)
coherence_dim=module_dic[num_components]['clusters'][mouse_recday]
#coherence_dim=np.flip(module_dic[num_components]['cluster_positions'][mouse_recday])
print(coherence_dim)#=
#plt.scatter(np.repeat(1,len(coherence_dim)),coherence_dim,s=2)
#plt.show()
#plt.scatter(tuning_dim[:,0], tuning_dim[:,1],s=5,c=state_tuning_used_seeded, cmap='magma')
#plt.show()

from mpl_toolkits.mplot3d import Axes3D



fig= plt.figure(figsize=plt.figaspect(1)*4.5) #

#plt.rcParams["figure.figsize"] = (12,12)
#plt.rcParams['axes.linewidth'] = 4
#plt.rcParams['axes.spines.right'] = False
#plt.rcParams['axes.spines.top'] = False

#rotation_angles_plot=[0,-45,-90,45] ##automate!


modules_used=[1,4]

module_used_bool=np.hstack(([coherence_dim[ii] in modules_used for ii in range(len(coherence_dim))]))

neurons_used=neurons_used[module_used_bool]
state_tuning_used_seeded=state_tuning_used_seeded[module_used_bool]
coherence_dim=coherence_dim[module_used_bool]
for ses_ind in np.arange(4):
    ax = fig.add_subplot(1, 4, ses_ind+1, projection='3d')
    
    #fig.set_figheight(250)
    #fig.set_figwidth(50)

    
    tuning_angle_rel_=Xsession_correlations['3_task_all']['Angles'][mouse_recday][:,ses_seed,ses_ind]
    if ses_ind==0:
        tuning_angle_rel_=np.zeros(len(tuning_angle_rel_))
        
    tuning_angle_rel_used_=tuning_angle_rel_[neurons_used]

    state_tuning_used_seeded_=(tuning_angle_rel_used_+state_tuning_used_seeded)%360
    tuning_dim=np.column_stack((np.sin(np.deg2rad(state_tuning_used_seeded_)),                                 np.cos(np.deg2rad(state_tuning_used_seeded_))))

    
    ax.scatter(tuning_dim[:,0], tuning_dim[:,1], coherence_dim, c=state_tuning_used_seeded, cmap='magma', s=300)

    for module in modules_used:#np.unique(coherence_dim):
        ax.plot(np.sin(np.deg2rad(np.arange(360))), np.cos(np.deg2rad(np.arange(360))),                np.repeat(module,360), color='black', linewidth=4)

    ax.view_init(20, 30)
    ax.set_xlabel('Tuning1')
    ax.set_ylabel('Tuning2')
    ax.set_zlabel('Coherence')
    
plt.savefig(Ephys_output_folder_dropbox+'/Modules_'+mouse_recday+'.svg')

Modules=[neurons_used[coherence_dim==module] for module in np.unique(coherence_dim)]


# In[298]:


Modules


# In[296]:


state_tuning_used


# In[274]:


'''
me10_08122021 - 1, 4

me11_30112021 - 3, 10

'''


# In[276]:


##Examples - rotations
#mouse_recday='ah04_06122021'

#for mouse_recday in day_type_dicX['3_task']:
print(mouse_recday)

all_neurons=np.arange(len(cluster_dic['good_clus'][mouse_recday]))
abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]


Xsession_correlations_day_=Xsession_correlations['3_task_all']['Angles'][mouse_recday]


#module_num=6 ##3
for module_num in np.arange(len(Modules)):
    #print(module)
    for neuron in Modules[module_num]:
        print(module_num)
        print(neuron)

        mouse=mouse_recday.split('_',1)[0]
        rec_day=mouse_recday.split('_',1)[1]

        All_sessions=session_dic['All'][mouse_recday]    
        awake_sessions=session_dic['awake'][mouse_recday]
        rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
        rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
        structure_nums=np.unique(rec_day_structure_numbers)

        fignamex=Ephys_output_folder_dropbox+'/Example_cells/'+mouse_recday+'_neuron_id'+str(neuron)+'.svg'

        arrange_plot_statecells_persessionX2(mouse_recday,neuron,                                              Data_folder=Intermediate_object_folder_dropbox,                                             abstract_structures=abstract_structures,                                            plot=True, save=True, fignamex=fignamex, figtype='.svg',Marker=False)

        #plt.savefig(fignamex)

        print(Xsession_correlations_day_[neuron])


# In[ ]:





# In[ ]:





# In[ ]:


'''#X_transformed_tuning_all=np.zeros((4,len(neurons_used),2))
#X_transformed_tuning_TSNE_all=np.zeros((4,len(neurons_used),2))
#for ses_ind in np.arange(4):
#    tuning_angle_mat=Xneuron_correlations['3_task_all']['Angles'][mouse_recday][:,:,ses_ind]##

    cosine_tuning_dist=1-np.asarray([[math.cos(math.radians(tuning_angle_mat[ii,jj]))\
                                    for ii in range(len(tuning_angle_mat))] for jj in range(len(tuning_angle_mat))])
    
    
    distance_matrix_used=cosine_tuning_dist[neurons_used][:,neurons_used]
    #MDS
    embedding = MDS(n_components=num_components,dissimilarity='precomputed')
    X_transformed_tuning = embedding.fit_transform(distance_matrix_used)
    X_transformed_tuning_all[ses_ind]=X_transformed_tuning

    #tSNE
    embedding_TSNE=TSNE(n_components=num_components, init='random',metric='precomputed',perplexity=5)
    X_transformed_tuning_TSNE = embedding_TSNE.fit_transform(distance_matrix_used)
    X_transformed_tuning_TSNE_all[ses_ind]=X_transformed_tuning_TSNE
    
    
    
    
state_tuning_all=dict_to_array(state_middle_dic['State']['Mean_state']['3_task_all'][ses_seed][mouse_recday])
state_tuning_used=state_tuning_all[neurons_used]

state_tuning_all2=dict_to_array(state_middle_dic['State']['Mean_state']['3_task_all'][ses_seed2][mouse_recday])
state_tuning_used2=state_tuning_all[neurons_used]

#num_fields_all=dict_to_array(num_field_dic['ALL']['3_task'][0][mouse_recday])'''


# In[ ]:





# In[158]:


num_fields_all=np.hstack(([np.sum(unique_adjacent(FR_shuff_dic['percentiles_state'][0][mouse_recday][neuron]>95))                      for neuron in range(num_neurons)]))
num_fields_all


# In[455]:


#import matplotlib
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)


# In[220]:


###Permutations
tt=time.time()
angle_changes=np.linspace(0,360,5)[:-1]

#module_shuff_dic=rec_dd()
num_iterations=100

day_type='3_task_all'
num_components=2
redo=True

use_peak=False

if use_peak==True:
    sigma=15
else:
    sigma=7
#KL_mat=np.zeros((len(Ephys_mousedays_3task_clean),num_iterations))
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    if mouse_recday not in Coherence_dic['coherent_tuning_prop'].keys():
        print('Day excluded from pairwise analysis so exluded here too')
        continue
    if redo==False:
        if len(module_shuff_dic['distance_matrix_used'][mouse_recday][0])>0:
            print('Already analysed')
            continue

    try:
        for iteration in range(num_iterations):
            
            X_transformed_=module_dic[num_components]['tSNE'][mouse_recday]
            if len(X_transformed_)==0:
                if iteration==0:
                    print('Not analysed')
                    print('')
                continue
            
            #initial_mean_angles=dict_to_array(state_middle_dic['ALL']['Mean_state'][0][mouse_recday])
            ephys_=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            initial_mean_angles=np.argmax(np.nanmean(ephys_,axis=1),axis=1)
            
            

            used_pairs=used_pairs_dic[mouse_recday]
            neurons_used=np.unique(used_pairs)

            X_angles=initial_mean_angles
            X_angles[np.isnan(X_angles)]=0

            #rand_rotation=angle_changes[np.random.randint(0,3,size=len(neurons_used))]
            #noise=np.random.normal(0,10,len(neurons_used))

            ##randomly rotating neurons with jitter
            Y_angles=random_rotation(len(X_angles),angle_changes,sigma=5) ##default sigma=10 degrees
            Z_angles=random_rotation(len(X_angles),angle_changes,sigma=5)

            XY_angle_change_neuron=positive_angle(circular_angle(Y_angles,X_angles))
            XZ_angle_change_neuron=positive_angle(circular_angle(Z_angles,X_angles))

            pairXY=np.asarray([[positive_angle([circular_angle(XY_angle_change_neuron[ii],XY_angle_change_neuron[jj])])[0]                                for ii in range(len(X_angles))] for jj in range(len(X_angles))])



            pairXZ=np.asarray([[positive_angle([circular_angle(XZ_angle_change_neuron[ii],XZ_angle_change_neuron[jj])])[0]                                for ii in range(len(X_angles))] for jj in range(len(X_angles))])



            cosine_distXY=1-np.asarray([[math.cos(math.radians(pairXY[ii,jj]))                                        for ii in range(len(pairXY))] for jj in range(len(pairXY))])
            cosine_distXZ=1-np.asarray([[math.cos(math.radians(pairXZ[ii,jj]))                                        for ii in range(len(pairXZ))] for jj in range(len(pairXZ))])

            cosine_max=np.max((cosine_distXY,cosine_distXZ),axis=0)
            cosine_mean=np.mean((cosine_distXY,cosine_distXZ),axis=0)
            distance_matrix_all=cosine_max
            distance_matrix_used=distance_matrix_all[neurons_used][:,neurons_used]

            distance_used_pairs=matrix_triangle(distance_matrix_used,direction='lower')


            embedding = MDS(n_components=num_components,dissimilarity='precomputed')
            X_transformed = embedding.fit_transform(distance_matrix_used)
            
            perp_=module_dic[num_components]['perplexity_used_tSNE'][mouse_recday]
            embedding_TSNE=TSNE(n_components=num_components, init='random',metric='precomputed',perplexity=perp_)
            X_transformed_TSNE = embedding_TSNE.fit_transform(distance_matrix_used)
            #KL_mat[mouse_recday_ind][iteration]=embedding_TSNE.kl_divergence_

            if iteration==-1:
                print(mouse_recday)
                print(X_transformed.shape)
                print('MDS')
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.scatter(X_transformed[:,0],X_transformed[:,1],s=5)
                #ax.set_aspect('equal', adjustable='box')
                plt.show()

                #rand_jitterX(arr, X)

                print('tSNE')

                plt.scatter(X_transformed_TSNE[:,0],X_transformed_TSNE[:,1],s=5)
                plt.show()
                print(embedding_TSNE.kl_divergence_)

                print('')
                print('')



            module_shuff_dic['tSNE'][mouse_recday][iteration]=X_transformed_TSNE
            module_shuff_dic['tSNE_KL_divergence'][mouse_recday][iteration]=embedding_TSNE.kl_divergence_

            module_shuff_dic['distance_matrix_all'][mouse_recday][iteration]=distance_matrix_all
            module_shuff_dic['distance_matrix_used'][mouse_recday][iteration]=distance_matrix_used
            
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

print(time.time()-tt)


# In[182]:


module_shuff_dic['tSNE'].keys()


# In[ ]:


###Here


# In[ ]:





# In[ ]:





# In[319]:


angle_changes=np.linspace(0,360,5)[:-1]
bins=np.linspace(0,360,37)
angles_=random_rotation(1000,angle_changes,sigma=15)
angles_hist_=np.histogram(angles_,bins=bins)[0]
polar_plot_stateX(angles_hist_,angles_hist_,angles_hist_,color='black',labels='angles',plot_type='bar')


# In[221]:


##clustering- representative shuffles
day_type='3_task_all'
for mouse_recday_ind, mouse_recday in enumerate(day_type_dicX[day_type]):
    try:
        print(mouse_recday)
        
        if mouse_recday not in Coherence_dic['coherent_tuning_prop'].keys():
            print('Day excluded from pairwise analysis so exluded here too')
            continue
        KL_divegrence_shuff=dict_to_array(module_shuff_dic['tSNE_KL_divergence'][mouse_recday])
        mean_divergence=np.mean(KL_divegrence_shuff)

        closest_ind=np.argmin(np.abs(KL_divegrence_shuff-mean_divergence))
        X_transformed_=module_shuff_dic['tSNE'][mouse_recday][closest_ind]

        clustering=sklearn.cluster.AgglomerativeClustering(distance_threshold=50,n_clusters=None)

        X_clustered = clustering.fit(X_transformed_)

        clusters = X_clustered.labels_
        cluster_labels=np.unique(clusters)

        neurons_used=module_dic['neurons_used'][mouse_recday]

        for cluster in cluster_labels:
            X_cluster=X_transformed_[clusters==cluster]
            if cluster == -1:
                plt.scatter(X_cluster[:,0],X_cluster[:,1],s=20, marker='x', color='black')
            else:
                plt.scatter(X_cluster[:,0],X_cluster[:,1],s=10)


        plt.show()
        #print(module_shuff_dic['tSNE_KL_divergence'][mouse_recday][closest_ind])

        if len(cluster_labels)>1 and len(clusters)>len(cluster_labels):
            silhouette_score_=silhouette_score(X_transformed_,clusters)
        else:
            silhouette_score_=np.nan
        module_shuff_dic['silhouette_score'][mouse_recday]=silhouette_score_

        print(silhouette_score_)
    except Exception as e:
        print(e)
        


# In[222]:


###Calculating silhouette score for shuffled data

day_type='3_task_all'
for mouse_recday_ind, mouse_recday in enumerate(day_type_dicX[day_type]):
    print(mouse_recday)
    if mouse_recday not in Coherence_dic['coherent_tuning_prop'].keys():
        
        print('Day excluded from pairwise analysis so exluded here too')
        if isinstance(module_shuff_dic['silhouette_score_mean'][mouse_recday],float)==True:
            del(module_shuff_dic['silhouette_score_mean'][mouse_recday])
        continue
    try:
        silhouette_scores=np.zeros(num_iterations)
        
        

        X_transformed_=module_dic[num_components]['tSNE'][mouse_recday]
        if len(X_transformed_)==0:
            print('Not analysed')

            continue
        for iteration in range(num_iterations):



            X_transformed_=module_shuff_dic['tSNE'][mouse_recday][iteration]

            #eps_=70+500/len(X_transformed)
            #clustering = DBSCAN(eps=eps_, min_samples=2)
            #clustering=sklearn.cluster.AgglomerativeClustering(n_clusters=5)
            clustering=sklearn.cluster.AgglomerativeClustering(distance_threshold=50,n_clusters=None)

            X_clustered = clustering.fit(X_transformed_)

            clusters = X_clustered.labels_
            cluster_labels=np.unique(clusters)



            plt.show()
            #print(module_shuff_dic['tSNE_KL_divergence'][mouse_recday][closest_ind])

            if len(cluster_labels)>1 and len(clusters)>len(cluster_labels):
                silhouette_score_=silhouette_score(X_transformed_,clusters)
            else:
                silhouette_score_=np.nan

            silhouette_scores[iteration]=silhouette_score_

        module_shuff_dic['silhouette_score_mean'][mouse_recday]=np.nanmean(silhouette_scores)
        module_shuff_dic['silhouette_score_95percentile'][mouse_recday]=np.percentile(remove_nan(silhouette_scores),95)
    except Exception as e:
        print(e)


# In[223]:


module_shuff_dic['silhouette_score_95percentile'][mouse_recday]


# mouse_days_=np.copy(list(module_shuff_dic['silhouette_score_mean'].keys()))
# for mouse_recday in mouse_days_:
#     if mouse_recday not in module_dic[num_components]['silhouette_score'].keys():
#         del(module_shuff_dic['silhouette_score_mean'][mouse_recday])

# In[224]:


num_components=2
SS_real=np.hstack(([module_dic[num_components]['silhouette_score'][mouse_recday]for mouse_recday in module_dic[num_components]['silhouette_score'].keys()]))
SS_permuted=np.hstack(([module_shuff_dic['silhouette_score_mean'][mouse_recday]for mouse_recday in module_dic[num_components]['silhouette_score'].keys()]))

#real_bool=np.asarray([isinstance(SS_real[ii],float) for ii in range(len(SS_real))])
#perm_bool=np.asarray([isinstance(SS_permuted[ii],float) for ii in range(len(SS_permuted))])

#overall_bool=np.logical_and(real_bool,perm_bool)

#SS_real=(SS_real[overall_bool]).squeeze()
#SS_permuted=(SS_permuted[overall_bool]).squeeze()

SS_permuted=np.hstack((SS_permuted))

bar_plotX([SS_real,SS_permuted],'none',0,0.85,'points','paired',0.025)
plt.savefig(Ephys_output_folder_dropbox+'/SS_clustering_realvspermuted.svg')
plt.show()


xy=column_stack_clean(SS_permuted,SS_real)
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
noplot_scatter(xy[:,0],xy[:,1],'black')
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(axis='both',  labelsize=15)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'/SS_clustering_realvspermuted_scatter.svg')
plt.show()
print(st.wilcoxon(xy[:,0],xy[:,1]))
print(len(xy))


# In[190]:


overall_bool


# In[187]:


SS_real


# In[189]:


'''

WilcoxonResult(statistic=206.0, pvalue=0.027497459872392938)
37
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


##KL divergence (tSNE embedding vs real) - i.e. how well tSNE has performed
num_components=2
KL_real=dict_to_array(module_dic[num_components]['tSNE_KL_divergence'])
KL_permuted=np.asarray([np.mean(dict_to_array(module_shuff_dic['tSNE_KL_divergence'][mouse_recday]))                        for mouse_recday in list(module_dic[num_components]['tSNE_KL_divergence'].keys())])

bar_plotX([KL_real,KL_permuted],'none',0,0.7,'points','paired',0.025)
plt.show()
noplot_scatter(KL_real,KL_permuted,'black')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
print(st.ttest_rel(KL_real,KL_permuted))


# In[ ]:





# In[ ]:


'''
1-for each neuron find DV coordinate - only cohort 6
2-group neurons per day for each of the 4 DV bins
3-calculate proportion generalising per group per day and average this per comparison(X vs Y and X vs Z)
4-plot averages with Ns=days

'''


# In[229]:


regions=['IL','PL','ACA','MOs']

Anatomy_channel_dic=rec_dd()
for mouse in ['ab03','ah07']:
    Anatomy_channels_file_path=Intermediate_object_folder_dropbox+mouse+'_channelanatomy.csv'
    with open(Anatomy_channels_file_path, 'r') as f:
        Anatomy_channels = np.genfromtxt(f, delimiter=',',dtype=str, usecols=np.arange(0,7))
    Anatomy_channels_structure=Anatomy_channels[0]
    Anatomy_channels_structure_corrected=np.hstack((Anatomy_channels_structure[:2],                                                    np.repeat(Anatomy_channels_structure[2],3),    Anatomy_channels_structure[3:-1],np.repeat(Anatomy_channels_structure[-1],3)))
    for indx, variable_ in enumerate(Anatomy_channels_structure):
        variable=Anatomy_channels_structure_corrected[indx]
        Anatomy_channel_dic[mouse][variable]=Anatomy_channels[1:,indx]

for mouse in ['ab03','ah07']:        
    acronym_=Anatomy_channel_dic[mouse]['acronym']
    for region in regions:
        X_bool=np.hstack(([region in acronym_[channel_id] for channel_id in np.arange(len(acronym_))]))
        Anatomy_channel_dic[mouse][region+'_bool']=X_bool
    
    


# In[ ]:





# In[ ]:





# In[230]:


###Generalisation/coherence along DV axis - neuropixels only

Tuning_anatomy_dic=rec_dd()
num_anatomy_bins=4
num_channels_neuropixels=384
bin_size=num_channels_neuropixels/num_anatomy_bins

anat_ratio_dic=rec_dd()
for mouse_recday in day_type_dicX['3_task_all']:
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
Anat_bin_dic=rec_dd()
for mouse_recday in day_type_dicX['3_task_all']:
    mouse=mouse_recday.split('_',1)[0]
    cohort=Mice_cohort_dic[mouse]
    
    if cohort not in [5,6]:
        continue
        
    print(mouse_recday)
    
    good_clus=cluster_dic['good_clus'][mouse_recday]
    channel_num_=cluster_dic['channel_number'][mouse_recday]
    channel_num=channel_num_[:,1][np.isin(channel_num_[:,0], good_clus)]
    
    diff_to_max=num_channels_neuropixels-np.max(channel_num)
    
    channel_num_corrected=channel_num+diff_to_max
    
    #np.save(Intermediate_object_folder_dropbox+'_channel_num_neuron_'+mouse_recday+'.npy',channel_num_corrected)

    
    anatomy_bin_neuron=((channel_num_corrected-1)//bin_size).astype(int)
    #anatomy_bin_neuron[anatomy_bin_neuron>(num_anatomy_bins-1)]=num_anatomy_bins-1
    
    
    
    ####region bins
    bins_=[]
    for region_ind,region in enumerate(regions):
        channel_ids_region=np.where(Anatomy_channel_dic[mouse][region+'_bool']==True)[0]
        bins_.append(channel_ids_region)

    region_id_neuron=np.repeat(np.nan,len(channel_num))
    for ii in range(len(regions)):
        region_id_neuron[np.isin(channel_num,bins_[ii])]=ii
    
    
    Anat_bin_dic['DV_bin'][mouse_recday]=anatomy_bin_neuron
    Anat_bin_dic['region_id'][mouse_recday]=region_id_neuron
    


# In[ ]:





# In[231]:


###Do single neurons generalize? - laminar profile

day_type='3_task_all'
anatomy_type='region_id'
ref_ses=0

coherence_thr=1-math.cos(math.radians(45)) #i.e. below 45 degrees either side (refine to first coherence bin?)
num_neurons_all_allbins=[]

single_prop_all_allbins=[]
for bin_ in np.arange(4):
    angles_all=[]
    angles_X_all=[]
    dual_prop_all=[]
    single_prop_all=[]

    single_prop_all_X=[]

    num_neurons_all=[]
    num_neurons_all_XX=[]
    mouse_days_used_=[]
    mouse_days_used_XX_=[]

    use_peak=False

    if day_type!='combined_ABCDonly':
        num_states=4
    for mouse_recday in day_type_dicX[day_type]:
        
        mouse=mouse_recday.split('_',1)[0]
        cohort=Mice_cohort_dic[mouse]

        if cohort not in [5,6]:
            continue
        
        #print(mouse_recday)
        #try:
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
        sessions=Task_num_dic[mouse_recday]
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 

        X_all=np.where(sessions==sessions[ref_ses])[0]
        X_rep=np.intersect1d(X_all,repeat_ses)

        non_ref_ses=np.setdiff1d(non_repeat_ses,ref_ses)

        if use_peak==True:
            Xsession_correlations_day=Xsession_correlations_peaks[mouse_recday]
        else:
            Xsession_correlations_day=Xsession_correlations[day_type]['Angles'][mouse_recday]
        used_boolean_=used_boolean_dic[day_type][mouse_recday]
        bin_used_boolean=Anat_bin_dic[anatomy_type][mouse_recday]==bin_
        used_boolean=np.logical_and(used_boolean_,bin_used_boolean)
        
        Xsession_correlations_day_used=Xsession_correlations_day[used_boolean]

        angles_day=Xsession_correlations_day_used[:,ref_ses,non_ref_ses]
        angles_day2=angles_day[:,:2]
        if np.shape(angles_day2)[1]==2:
            angles_all.append(angles_day2) ##only include first 2 comparison sessions (i.e. 3 sessions) 
        ##in some days animals get 4 tasks by accident


            distances_day2=1-np.cos(np.deg2rad(angles_day2))
            max_dist=np.max(distances_day2,axis=1)
            dual_prop=np.sum(max_dist<coherence_thr)/len(max_dist)
            single_prop=np.mean([np.sum(distances_day2[:,0]<coherence_thr)/len(distances_day2),                                np.sum(distances_day2[:,1]<coherence_thr)/len(distances_day2)])

            dual_prop_all.append(dual_prop)
            single_prop_all.append(single_prop)

        if len(X_rep)>0:
            mouse_days_used_XX_.append(mouse_recday)
            num_neurons_all_XX.append(np.sum(used_boolean))
            angles_X_day=Xsession_correlations_day_used[:,ref_ses,X_rep[0]]
            angles_X_all.append(angles_X_day)

            distances_day_X=1-np.cos(np.deg2rad(angles_X_day))
            if np.isnan(np.mean(distances_day_X))==False:
                single_prop_X=np.sum(distances_day_X<coherence_thr)/len(distances_day_X)
                single_prop_all_X.append(single_prop_X)

        num_neurons_all.append(np.sum(used_boolean))

        mouse_days_used_.append(mouse_recday)
        #except:
        #    print('not used')

    print('Total number of neurons')
    print(np.sum(num_neurons_all))
    
    num_neurons_all_allbins.append(np.sum(num_neurons_all))

    single_neuron_angles_all=np.vstack((angles_all))
    single_neuron_angles_X_all=remove_nan(np.hstack((angles_X_all)))

    ###Plotting histograms
    hist_all=[]
    for session in range(2):
        #print(session+1)
        angles_all_hist=np.histogram(single_neuron_angles_all[:,session],np.linspace(0,360,37))[0]

        hist_all.append(np.asarray(angles_all_hist))

        #plt.bar(np.arange(len(angles_all)),angles_all)
        #plt.show()
        #print(angles_all_hist)
        #print(np.sum(angles_all_hist))

        #polar_plot_stateX(angles_all_hist,angles_all_hist,angles_all_hist,color='black',labels='angles',plot_type='bar')
        #plt.savefig(Ephys_output_folder+day_type+'_session0tosession'+str(session)+'_anglechange_polarplot.svg')
        #plt.show()

    hist_all=np.asarray(hist_all)

    mean_hist=np.mean(hist_all,axis=0)
    #polar_plot_stateX(mean_hist,mean_hist,mean_hist,color='black',labels='angles',plot_type='bar')
    #plt.savefig(Ephys_output_folder_dropbox+'Mean_anglechange_polarplot.svg',\
    #           bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    generalising_num=(np.sum(mean_hist[:5])+np.sum(mean_hist[-4:]))
    generalissing_prop=generalising_num/np.sum(mean_hist)
    #print(generalissing_prop)
    #print(two_proportions_test(generalising_num, np.sum(mean_hist), np.sum(mean_hist)*(1/num_states),np.sum(mean_hist)))

    ###X vs X'
    #print(session+1)
    angles_all_hist=np.histogram(single_neuron_angles_X_all,np.linspace(0,360,37))[0]
    #print(angles_all_hist)
    #print(np.sum(angles_all_hist))
    #polar_plot_stateX(angles_all_hist,angles_all_hist,angles_all_hist,color='black',labels='angles',plot_type='bar')
    #plt.savefig(Ephys_output_folder_dropbox+'Mean_anglechange_XvsX_polarplot.svg',\
    #           bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    generalising_num=(np.sum(angles_all_hist[:5])+np.sum(angles_all_hist[-4:]))
    generalissing_prop=generalising_num/np.sum(angles_all_hist)


    single_prop_all=np.asarray(single_prop_all)
    dual_prop_all=np.asarray(dual_prop_all)
    single_prop_all_X=np.asarray(single_prop_all_X)


    
    single_prop_all_allbins.append(single_prop_all)
    
single_prop_all_allbins=np.vstack((single_prop_all_allbins))


# In[ ]:





# In[ ]:





# In[232]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
from scipy.stats import tukey_hsd
#for measure in ['Phase','State','Place']:
#print(measure)
anatomy_type='region_id'
Measure_prop_anat_mean=np.nanmean(single_prop_all_allbins.T,axis=0)
Measure_prop_anat_sem=st.sem(single_prop_all_allbins.T,nan_policy='omit',axis=0)
plt.rcParams["figure.figsize"] = (3,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

if anatomy_type=='region_id':
    y=['M2','ACC','PrL','IrL']
else:
    y=-np.arange(len(Measure_prop_anat_mean.T))

plt.errorbar(y=y,x=np.flip(Measure_prop_anat_mean*100),             xerr=np.flip(Measure_prop_anat_sem*100),            marker='o',markersize=10,color='black')
plt.xlim(0,40)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.axvline(25,color='black',ls='dashed')
if anatomy_type=='region_id':
    plt.gca().invert_yaxis()

plt.savefig(Ephys_output_folder_dropbox+'DV_vs_proportion_generalization.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()

import statsmodels.api as sm

stats=st.f_oneway(remove_nan(single_prop_all_allbins[:,0]),remove_nan(single_prop_all_allbins[:,1]),                  remove_nan(single_prop_all_allbins[:,2]),remove_nan(single_prop_all_allbins[:,3]))
print(stats)

if stats[1]<0.05:
    res = tukey_hsd(remove_nan(single_prop_all_allbins[:,0]),remove_nan(single_prop_all_allbins[:,1]),                  remove_nan(single_prop_all_allbins[:,2]),remove_nan(single_prop_all_allbins[:,3]))
    print(res)


# In[ ]:





# In[ ]:





# In[ ]:





# In[233]:


##Do pairs of neurons remain coherent?

Coherence_laminar_dic=rec_dd()
coherence_thr=1-math.cos(math.radians(45)) #i.e. below 45 degrees either side (refine to first coherence bin?)
angle_changes=np.linspace(0,360,5)[:-1]

neurons_used_thr=10

day_type='3_task_all'
anatomy_type='region_id'

num_pairs_allbins=[]

use_peak=False
if day_type!='combined_ABCDE':
    num_states=4
for bin_ in np.arange(4):
    print(regions[bin_])
    for mouse_recday in day_type_dicX[day_type]:
        mouse=mouse_recday.split('_',1)[0]
        cohort=Mice_cohort_dic[mouse]

        if cohort not in [5,6]:
            continue
        print(mouse_recday)
        #try:
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
        sessions=Task_num_dic[mouse_recday]
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday)  


        X_all=np.where(sessions==sessions[ref_ses])[0]
        X_rep=np.intersect1d(X_all,repeat_ses)
        non_ref_ses=np.setdiff1d(non_repeat_ses,ref_ses)
        if use_peak==True:
            Xsession_correlations_day=Xsession_correlations_peaks[mouse_recday]
        else:
            Xsession_correlations_day=Xsession_correlations[day_type]['Angles'][mouse_recday]
        
        used_boolean_=used_boolean_dic[day_type][mouse_recday]
        bin_used_boolean=Anat_bin_dic[anatomy_type][mouse_recday]==bin_
        used_boolean=np.logical_and(used_boolean_,bin_used_boolean)
        
        neurons_used=np.where(used_boolean==True)[0]
        Xsession_correlations_day_used=Xsession_correlations_day[used_boolean]
        


        angles_day=Xsession_correlations_day_used[:,ref_ses,non_ref_ses]
        angles_day2=angles_day[:,:2] ##only include first 2 comparison sessions (i.e. 3 sessions) 
        ##on some days animals get 4 tasks by accident



        if np.shape(angles_day2)[1]<2 or np.isnan(np.nanmean(angles_day2))==True:
            print('Less than 3 sessions - not included')
            Coherence_laminar_dic[bin_]['coherent_prop'][mouse_recday]=np.nan
            Coherence_laminar_dic[bin_]['coherent_prop_dual'][mouse_recday]=np.nan
            continue

        if len(neurons_used)<neurons_used_thr:
            print('Less than '+str(neurons_used_thr)+' usable neurons - not included')
            Coherence_laminar_dic[bin_]['coherent_prop'][mouse_recday]=np.nan
            Coherence_laminar_dic[bin_]['coherent_prop_dual'][mouse_recday]=np.nan
            continue

        num_trials_ses=[]
        for ses_ind in non_repeat_ses:
            location_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy',allow_pickle=True)
            num_trials_ses.append(len(location_))
        num_trials_ses=np.hstack((num_trials_ses))



        if np.sum(num_trials_ses>0)<3:
            print('Less than 3 tasks with completed trials')
            excluded_num_sessions.append(mouse_recday)
            Coherence_laminar_dic[bin_]['coherent_prop'][mouse_recday]=np.nan
            Coherence_laminar_dic[bin_]['coherent_prop_dual'][mouse_recday]=np.nan
            continue

        relative_angles_pairs_day=[]
        for session in range(2):
            angles_ses=angles_day2[:,session]
            relative_angles_mat_ses=np.vstack(([positive_angle([circular_angle(angles_ses[neuronX],angles_ses[neuronY])                         for neuronX in np.arange(len(angles_ses))])                        for neuronY in np.arange(len(angles_ses))]))

            relative_angles_pairs_ses=matrix_triangle(relative_angles_mat_ses)
            relative_angles_pairs_day.append(relative_angles_pairs_ses)

        relative_angles_pairs_day=np.asarray(relative_angles_pairs_day).T

        ###### coherence proportion

        pairXY=relative_angles_pairs_day[:,0]
        pairXZ=relative_angles_pairs_day[:,1]
        cosine_distXY=1-np.cos(np.deg2rad(pairXY))
        cosine_distXZ=1-np.cos(np.deg2rad(pairXZ))

        cosine_max=np.max((cosine_distXY,cosine_distXZ),axis=0)
        cosine_mean=np.mean((cosine_distXY,cosine_distXZ),axis=0)
        distance_used_pairs=cosine_max
        
        coherent_prop_XY=len(np.where(cosine_distXY<coherence_thr)[0])/len(distance_used_pairs)
        coherent_prop_XZ=len(np.where(cosine_distXZ<coherence_thr)[0])/len(distance_used_pairs)
        coherent_prop=np.nanmean([coherent_prop_XY,coherent_prop_XZ])
        
        coherent_prop_dual=len(np.where(distance_used_pairs<coherence_thr)[0])/len(distance_used_pairs)
        
        print('Num_pairs ='+str(len(distance_used_pairs)))
        num_pairs_allbins.append(len(distance_used_pairs))

        ##breakdown by tuning distance

        angles_between_neurons_used=Xneuron_correlations[day_type]['Angles'][mouse_recday]                                             [neurons_used][:,neurons_used][:,:,ref_ses]

        angle_all_all=matrix_triangle(angles_between_neurons_used)
        Tuning_dist=1-np.cos(np.deg2rad(angle_all_all))
        coherence_tuning_prop_all=np.zeros(len(angle_changes))
        tunings=angle_changes/2 ##because taking absolute angle/distance (i.e. maximum angle difference is 180 degrees)
        for tuning_ind,tuning in enumerate(tunings):
            lower_limit=1-math.cos(math.radians(tuning))
            upper_limit=1-math.cos(math.radians(tuning+45))

            coherence_tuning=distance_used_pairs[np.logical_and(Tuning_dist>=lower_limit,Tuning_dist<upper_limit)]
            if len(coherence_tuning)>0:
                coherence_tuning_prop=len(np.where(coherence_tuning<coherence_thr)[0])/len(coherence_tuning)
            else:
                coherence_tuning_prop=np.nan

            coherence_tuning_prop_all[tuning_ind]=coherence_tuning_prop
        
        Coherence_laminar_dic[bin_]['coherent_prop'][mouse_recday]=coherent_prop
        Coherence_laminar_dic[bin_]['coherent_prop_dual'][mouse_recday]=coherent_prop_dual
        Coherence_laminar_dic[bin_]['coherent_tuning_prop'][mouse_recday]=coherence_tuning_prop_all
        Coherence_laminar_dic[bin_]['Num_neurons_used'][mouse_recday]=len(neurons_used)


# In[234]:


neurons_used_thr


# In[235]:


anatomy_type='region_id'


coherent_prop_allbins=np.vstack(([dict_to_array(Coherence_laminar_dic[bin_]['coherent_prop_dual'])                                  for bin_ in np.arange(4)]))

Measure_prop_anat_mean=np.nanmean(coherent_prop_allbins.T,axis=0)
Measure_prop_anat_sem=st.sem(coherent_prop_allbins.T,nan_policy='omit',axis=0)
plt.rcParams["figure.figsize"] = (3,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
if anatomy_type=='region_id':
    y=['M2','ACC','PrL','IrL']
else:
    y=-np.arange(len(Measure_prop_anat_mean.T))

plt.errorbar(y=y,x=np.flip(Measure_prop_anat_mean*100),             xerr=np.flip(Measure_prop_anat_sem*100),            marker='o',markersize=10,color='black')
plt.xlim(0,10)
#plt.ylim(-3,0.1)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.axvline((1/16)*100,color='black',ls='dashed')
if anatomy_type=='region_id':
    plt.gca().invert_yaxis()
plt.savefig(Ephys_output_folder_dropbox+'DV_vs_proportion_coherent.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()

import statsmodels.api as sm

stats=st.f_oneway(remove_nan(coherent_prop_allbins[:,0]),remove_nan(coherent_prop_allbins[:,1]),                  remove_nan(coherent_prop_allbins[:,2]),remove_nan(coherent_prop_allbins[:,3]))
print(stats)

if stats[1]<0.05:
    res = tukey_hsd(remove_nan(coherent_prop_allbins[:,0]),remove_nan(coherent_prop_allbins[:,1]),                  remove_nan(coherent_prop_allbins[:,2]),remove_nan(coherent_prop_allbins[:,3]))
    print(res)

np.shape(coherent_prop_allbins)


# In[ ]:





# In[236]:


##Do pairs of neurons remain coherent? - Dual ABCD days

Coherence_dic_dualday=rec_dd()
coherence_thr=1-math.cos(math.radians(45)) #i.e. below 45 degrees either side (refine to first coherence bin?)
angle_changes=np.linspace(0,360,5)[:-1]

day_type='combined_ABCDonly'
relative_angles_pairs_all=[]

ref_ses=0 ##where angles between neurons are taken

max_comparisons=5
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    try:
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
        sessions=Task_num_dic[mouse_recday]
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday)  

        non_ref_ses=np.setdiff1d(non_repeat_ses,ref_ses)
        num_comparisons=len(non_ref_ses)-1

        Xsession_correlations_day=Xsession_correlations[day_type]['Angles'][mouse_recday]
        used_boolean=used_boolean_dic[day_type][mouse_recday]
        neurons_used=np.where(used_boolean==True)[0]
        Xsession_correlations_day_used=Xsession_correlations_day[used_boolean]

        angles_day=Xsession_correlations_day_used[:,ref_ses,non_ref_ses] ##ses 0 vs all non-repeat sessions
        angles_dayX=angles_day[:,:max_comparisons] ##only include first 5 comparison sessions (i.e. 6 sessions) 
        ##on some days animals get more than 6 tasks by accident

        if np.shape(angles_dayX)[1]<2:
            print('Less than 3 sessions - not included')
            continue

        if len(neurons_used)==0:
            print('No usable neurons')
            continue

        relative_angles_pairs_day=[]
        for comparison in range(max_comparisons):
            if np.shape(angles_dayX)[1]<(comparison+1) or np.isnan(np.nanmean(angles_dayX[:,comparison]))==True:
                relative_angles_mat_ses=np.zeros((len(angles_dayX[:,0]),len(angles_dayX[:,0])))
                relative_angles_mat_ses[:]=np.nan
            else:
                angles_ses=angles_dayX[:,comparison]
                relative_angles_mat_ses=np.vstack(([positive_angle([circular_angle(angles_ses[neuronX],angles_ses[neuronY])                             for neuronX in np.arange(len(angles_ses))])                            for neuronY in np.arange(len(angles_ses))]))

            relative_angles_pairs_ses,pair_indices_=matrix_triangle(relative_angles_mat_ses,return_indices=True)
            pair_indices_day=np.column_stack((neurons_used[pair_indices_[0]],neurons_used[pair_indices_[1]]))

            relative_angles_pairs_day.append(relative_angles_pairs_ses)

        relative_angles_pairs_day=np.asarray(relative_angles_pairs_day).T
        relative_angles_pairs_all.append(relative_angles_pairs_day) 

        Coherence_dic_dualday['pair_indices'][mouse_recday]=pair_indices_day



        ###### coherence proportion

        pairXY=relative_angles_pairs_day[:,0]
        pairXZ=relative_angles_pairs_day[:,1]
        cosine_distXY=1-np.cos(np.deg2rad(pairXY))
        cosine_distXZ=1-np.cos(np.deg2rad(pairXZ))

        cosine_max=np.max((cosine_distXY,cosine_distXZ),axis=0)
        cosine_mean=np.mean((cosine_distXY,cosine_distXZ),axis=0)
        distance_used_pairs=cosine_max

        coherent_prop=len(np.where(distance_used_pairs<coherence_thr)[0])/len(distance_used_pairs)

        ##breakdown by tuning distance

        angles_between_neurons_used=Xneuron_correlations[day_type]['Angles'][mouse_recday]                                             [neurons_used][:,neurons_used][:,:,ref_ses]
        angle_all_all=angles_between_neurons_used[pair_indices_] ##i.e. using same indices as derived above for coherence

        Tuning_dist=1-np.cos(np.deg2rad(angle_all_all))
        coherence_tuning_prop_all=np.zeros(len(angle_changes))
        tunings=angle_changes/2 ##because taking absolute angle/distance (i.e. maximum angle difference is 180 degrees)
        for tuning_ind,tuning in enumerate(tunings):
            lower_limit=1-math.cos(math.radians(tuning))
            upper_limit=1-math.cos(math.radians(tuning+45))

            coherence_tuning=distance_used_pairs[np.logical_and(Tuning_dist>=lower_limit,Tuning_dist<upper_limit)]
            coherence_tuning_prop=len(np.where(coherence_tuning<coherence_thr)[0])/len(coherence_tuning)
            coherence_tuning_prop_all[tuning_ind]=coherence_tuning_prop

        Coherence_dic_dualday['coherent_prop'][mouse_recday]=coherent_prop
        Coherence_dic_dualday['coherent_tuning_prop'][mouse_recday]=coherence_tuning_prop_all
        Coherence_dic_dualday['Tuning_dist'][mouse_recday]=Tuning_dist


        relative_distances_pairs_day=1-np.cos(np.deg2rad(relative_angles_pairs_day))
        tasks_coherent=np.sum(relative_distances_pairs_day<coherence_thr,axis=1)
        num_comparisons=np.sum(np.isnan(relative_distances_pairs_day)==False,axis=1)[0]
        coherent_prop_all=np.asarray([np.sum(tasks_coherent>ii)/len(tasks_coherent) for ii in range(num_comparisons)])

        Coherence_dic_dualday['tasks_coherent'][mouse_recday]=tasks_coherent


        xdata=np.arange(len(coherent_prop_all))
        ydata=coherent_prop_all
        popt, pcov = curve_fit(func_decay, xdata, ydata)

        xdata_new=np.arange(100)
        y_pred=func_decay(xdata_new, *popt)

        X=asymptote=y_pred[-1]
        n=num_neurons_used=len(neurons_used)
        N=n/(X*(n-1)+1) ##estimated number of modules
        #N=1/X
        Coherence_dic_dualday['Number_modules'][mouse_recday]=N

        coherent_prop_all_nonan=np.hstack((coherent_prop_all,np.repeat(np.nan,max_comparisons-len(coherent_prop_all))))
        Coherence_dic_dualday['coherent_prop_all'][mouse_recday]=coherent_prop_all_nonan

        
        
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
relative_angles_pairs_all=np.vstack((relative_angles_pairs_all))


###Plotting histograms
hist_all=[]
for comparison in range(5):
    print('0 vs '+str(comparison+1))
    session=comparison+1
    relative_angles_all_hist=np.histogram(relative_angles_pairs_all[:,comparison],np.linspace(0,360,37))[0]
    
    hist_all.append(np.asarray(relative_angles_all_hist))

    #plt.bar(np.arange(len(angles_all)),angles_all)
    #plt.show()
    print(relative_angles_all_hist)
    print(np.sum(relative_angles_all_hist))

    polar_plot_stateX(relative_angles_all_hist,relative_angles_all_hist,relative_angles_all_hist,                      color='black',labels='angles',plot_type='bar')
    #plt.savefig(Ephys_output_folder+day_type+'_session0tosession'+str(session)+'_anglechange_polarplot.svg')
    plt.show()

print('')
print('Mean histogram')
hist_all=np.asarray(hist_all)

mean_hist=np.mean(hist_all,axis=0)
polar_plot_stateX(mean_hist,mean_hist,mean_hist,color='black',labels='angles',plot_type='bar')
#plt.savefig(Ephys_output_folder+day_type+'_session0tosession'+str(session)+'_anglechange_polarplot.svg')
plt.show()

print('Answer: Yea a little bit')


# In[ ]:


xdata


# In[ ]:


ydata


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[237]:


n=5
chance_all=[]
for numsessions_coh in np.arange(n)+1:
    chance=((1/4)**numsessions_coh)*((3/4)**(n-numsessions_coh))*    (math.factorial(n)/(math.factorial(n-numsessions_coh)*math.factorial(numsessions_coh)))
    ##(p**k)*((1-p)**(n-k))*(n!/((n-k)!k!)) ... n=num comparisons, k=num comparisons coherent
    chance_all.append(chance)
chance_all=np.asarray(chance_all)

coherent_prop_allcomps=dict_to_array(Coherence_dic_dualday['coherent_prop_all'])
print(np.nanmean(coherent_prop_allcomps,axis=0))
print(chance_all)
bar_plotX(coherent_prop_allcomps.T,'none',0,1,'nopoints','unpaired',0.025)
plt.show()

plt.plot(np.arange(len(coherent_prop_allcomps.T)),coherent_prop_allcomps.T,alpha=0.1,color='grey')
plt.errorbar(np.arange(len(coherent_prop_allcomps.T)),np.nanmean(coherent_prop_allcomps.T,axis=1),            yerr=st.sem(coherent_prop_allcomps.T,nan_policy='omit',axis=1))
plt.plot(np.arange(len(coherent_prop_allcomps.T)),chance_all,color='black',ls='dashed')
plt.show()


# In[238]:


Coherence_dic_dualday['Number_modules']

##obviously cant have negative number of modules, need to reconsider function type


# In[239]:


'''
fit logarithmic function and find where it plateus

X=proportion coherent from plateau
n=num_neurons_used
N=n/(X*(n-1)+1) ##number of modules
'''


# In[ ]:





# In[ ]:





# In[240]:


###Anatomical distances


# In[241]:


'''Need to modify to exclude neuropixels as no AP extent'''


# In[242]:


Mice_cohort_dic


# In[243]:


mouse=mouse_recday.split('_',1)[0]
cohort=Mice_cohort_dic[mouse]
ephys_type=Cohort_ephys_type_dic[cohort]

Cohort_ephys_type_dic


# In[244]:


for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    cohort=Mice_cohort_dic[mouse]
    ephys_type=Cohort_ephys_type_dic[cohort]
    
    if ephys_type!='Cambridge_neurotech':
        continue
    
    
    
    good_clus=cluster_dic['good_clus'][mouse_recday]
    channel_numbers=cluster_dic['channel_number'][mouse_recday]
    channel_number_good=np.asarray([channel_numbers[i,1] for i in range(len(channel_numbers)) if channel_numbers[i,0]                                    in good_clus])
    
    cluster_dic['channel_number_good'][mouse_recday]=channel_number_good


# In[245]:


###pairwise anatomical distances

channel_group_dic={'0':np.arange(10),'1':np.arange(11)+10,'2':np.arange(11)+21,'3':np.arange(11)+32,                   '4':np.arange(11)+43,'5':np.arange(10)+54}
channel_group_array=np.concatenate([np.repeat(int(ii),len(channel_group_dic[str(ii)])) for ii in channel_group_dic.keys()])
#21-31,32-42,43-53,54-63

AP_position_all=channel_group_array*200
DV_position_all=[]
for key,channel_ids in channel_group_dic.items():
    DV_position=np.arange(len(channel_ids))*12.5
    DV_position_all.append(DV_position)
    
DV_position_all=np.hstack((DV_position_all))

'''
12.5 um between each channel on a shank along vertical direction
200um between centres of shanks
channels arranged in a staggered configuration with two rows seperated by ~20 microns
https://www.cambridgeneurotech.com/neural-probes - look at F series, 6 shank probe
'''

anatomical_distance_matrix_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    cohort=Mice_cohort_dic[mouse]
    ephys_type=Cohort_ephys_type_dic[cohort]
    
    if ephys_type!='Cambridge_neurotech':
        continue
    channel_number_good=cluster_dic['channel_number_good'][mouse_recday]
    AP_distance_matrix=[[abs(AP_position_all[ii]-AP_position_all[jj])                                           for ii in channel_number_good] for jj in channel_number_good]
    
    DV_distance_matrix=[[abs(DV_position_all[ii]-DV_position_all[jj])                                           for ii in channel_number_good] for jj in channel_number_good]
    
    anatomical_distance_matrix_dic['AP_distance'][mouse_recday]=np.asarray(AP_distance_matrix)
    anatomical_distance_matrix_dic['AP_position'][mouse_recday]=AP_position_all[channel_number_good]
    
    anatomical_distance_matrix_dic['DV_distance'][mouse_recday]=np.asarray(DV_distance_matrix)
    anatomical_distance_matrix_dic['DV_position'][mouse_recday]=DV_position_all[channel_number_good]
    


# In[ ]:





# In[ ]:





# In[246]:


Distance_coherence_dic=rec_dd()
use_distant=False ##when true only uses pairs of neurons more than 45 degrees from each other

thr_incoh=2
thr_coh=5
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mouse=mouse_recday.split('_',1)[0]
    cohort=Mice_cohort_dic[mouse]
    ephys_type=Cohort_ephys_type_dic[cohort]
    
    if ephys_type!='Cambridge_neurotech':
        continue
    print(mouse_recday)
    try:
        num_comparisons=len(remove_nan(Coherence_dic_dualday['coherent_prop_all'][mouse_recday]))
        if num_comparisons<5:
            print('Less than 6 sessions - not included')
            continue

        pair_indices_=Coherence_dic_dualday['pair_indices'][mouse_recday]
        anat_dist_mat_AP=anatomical_distance_matrix_dic['AP_distance'][mouse_recday]
        anat_distances_AP_=anat_dist_mat_AP[pair_indices_[:,0],pair_indices_[:,1]]

        anat_dist_mat_DV=anatomical_distance_matrix_dic['DV_distance'][mouse_recday]
        anat_distances_DV_=anat_dist_mat_DV[pair_indices_[:,0],pair_indices_[:,1]]

        anat_distances_=np.sqrt(anat_distances_AP_**2+anat_distances_DV_**2)

        tasks_coherent_=Coherence_dic_dualday['tasks_coherent'][mouse_recday]
        tuning_dist_=Coherence_dic_dualday['Tuning_dist'][mouse_recday]

        distant_pairs=tuning_dist_>coherence_thr

        if use_distant==True:
            anat_dist_coh=anat_distances_[np.logical_and(tasks_coherent_==thr_coh, distant_pairs==True)]
            anat_dist_incoh=anat_distances_[np.logical_and(tasks_coherent_<thr_incoh, distant_pairs==True)]

            anat_dist_graded_coh=[np.mean(anat_distances_[np.logical_and(tasks_coherent_==ii, distant_pairs==True)])                                  for ii in np.arange(6)]

            pair_indices_used=pair_indices_[distant_pairs==True]
        else:
            anat_dist_coh=anat_distances_[tasks_coherent_==thr_coh]
            anat_dist_incoh=anat_distances_[tasks_coherent_<thr_incoh]

            anat_dist_graded_coh=[np.mean(anat_distances_[tasks_coherent_==ii])                                  for ii in np.arange(6)]

            pair_indices_used=pair_indices_


        anat_dist_coh_mean=np.mean(anat_dist_coh)
        anat_dist_incoh_mean=np.mean(anat_dist_incoh)

        len_coh,len_incoh=len(anat_dist_coh),len(anat_dist_incoh)

        num_neurons_used=len(np.unique(pair_indices_used))

        if len(anat_dist_coh)<2:
            anat_dist_coh_mean=np.nan
            len_coh=0
            num_neurons_used=0

        if len(anat_dist_incoh)<2:
            anat_dist_incoh_mean=np.nan
            len_incoh=0
            num_neurons_used=0


        Distance_coherence_dic['anat_dist_coh'][mouse_recday]=anat_dist_coh
        Distance_coherence_dic['anat_dist_incoh'][mouse_recday]=anat_dist_incoh
        Distance_coherence_dic['anat_dist_coh_incoh_mean'][mouse_recday]=anat_dist_coh_mean,anat_dist_incoh_mean
        Distance_coherence_dic['anat_dist_graded_coh'][mouse_recday]=anat_dist_graded_coh
        Distance_coherence_dic['anat_dist_graded_coh'][mouse_recday]=anat_dist_graded_coh
        Distance_coherence_dic['num_neurons_used'][mouse_recday]=num_neurons_used
    except:
        print('Not done')


# In[ ]:





# In[247]:


np.sum(dict_to_array(Distance_coherence_dic['num_neurons_used']))


# In[ ]:





# In[248]:


for mouse in Mice:
    anat_dist_coh_incoh_mean_mouse=[]
    for mouse_recday in Distance_coherence_dic['anat_dist_coh'].keys():
        mousex=mouse_recday[:4]
        anat_dist_coh_incoh_mean_=Distance_coherence_dic['anat_dist_coh_incoh_mean'][mouse_recday]
        if mousex==mouse and len(anat_dist_coh_incoh_mean_)>0:
            anat_dist_coh_incoh_mean_mouse.append(anat_dist_coh_incoh_mean_)
    
    
    if len(anat_dist_coh_incoh_mean_mouse)>0:
        anat_dist_coh_incoh_mean_mouse_mean=np.nanmean(np.vstack((anat_dist_coh_incoh_mean_mouse)),axis=0)
        Distance_coherence_dic['anat_dist_coh_incoh_mean_permouse'][mouse]=anat_dist_coh_incoh_mean_mouse_mean


# In[249]:


anat_dist_coh_all=np.hstack((dict_to_array(Distance_coherence_dic['anat_dist_coh'])))
anat_dist_incoh_all=np.hstack((dict_to_array(Distance_coherence_dic['anat_dist_incoh'])))

anat_dist_mean_coh_incoh=dict_to_array(Distance_coherence_dic['anat_dist_coh_incoh_mean'])

print(np.column_stack((list(Distance_coherence_dic['anat_dist_coh_incoh_mean'].keys()),anat_dist_mean_coh_incoh)))


# In[ ]:





# In[250]:


print('per mouse')
xy=dict_to_array(Distance_coherence_dic['anat_dist_coh_incoh_mean_permouse'])
xy=column_stack_clean(xy[:,0],xy[:,1])
bar_plotX(xy.T,'none',0,320,'points','paired',0.025)
plt.savefig(Ephys_output_folder_dropbox+'/Anatomical_distance_coherence_mice.svg')
plt.show()
print(st.ttest_rel(xy[:,0],xy[:,1]))
print(st.wilcoxon(xy[:,0],xy[:,1]))

print('')
print('per day')
xy=column_stack_clean(anat_dist_mean_coh_incoh[:,0],anat_dist_mean_coh_incoh[:,1])
bar_plotX(xy.T,'none',0,500,'points','paired',0.025)
plt.savefig(Ephys_output_folder_dropbox+'/Anatomical_distance_coherence_days.svg')
plt.show()
print(st.ttest_rel(xy[:,0],xy[:,1]))
print(st.wilcoxon(xy[:,0],xy[:,1]))
print('')

print('per pair')
bar_plotX(np.asarray([anat_dist_coh_all,anat_dist_incoh_all]).T,'none',0,310,'nopoints','unpaired',0.025)
plt.tick_params(axis='both',  labelsize=15)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'/Anatomical_distance_coherence_pairs.svg')
plt.show()
print(st.ttest_ind(anat_dist_coh_all,anat_dist_incoh_all))
print(st.mannwhitneyu(anat_dist_coh_all,anat_dist_incoh_all))
print(len(anat_dist_coh_all),len(anat_dist_incoh_all))

print(len(anat_dist_coh_all)+len(anat_dist_incoh_all))


# In[260]:


filtered_data[0]


# In[263]:


plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False

filtered_data=remove_nan(anat_dist_coh_all),remove_nan(anat_dist_incoh_all)

#sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1)
#plt.errorbar(np.arange(len(means)),means, yerr=sems, marker='o', fmt='.',color='black')

sns.violinplot(filtered_data, color='grey',alpha=0.5)
#sns.stripplot(filtered_data,color='white',edgecolor='black',linewidth=1,alpha=0.5)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'/Coherence_distance_violin.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[133]:



plt.hist(anat_dist_incoh_all)
plt.show()
plt.hist(anat_dist_coh_all)
plt.show()


# In[ ]:





# In[134]:


anat_dist_coh_incoh_mean_=dict_to_array(Distance_coherence_dic['anat_dist_graded_coh'])
bar_plotX(anat_dist_coh_incoh_mean_.T,'none',0,500,'points','paired',0.025)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


####modify below


# ###given coherence in n/4 session transitions, how likely is it that pairs are coherent in the 5th transition
# ses_transitions=np.arange(len(coherent_matrix))
# non_zero_boolean=angle_to_distance(Xneuron_tuningangle_used_pairs_ABCDonly_all)>0 #>1
# 
# coherent_num_all=np.zeros((len(coherent_matrix),len(coherent_matrix)))
# for ii in ses_transitions:
#     target=coherent_matrix[ii][non_zero_boolean]
#     training=coherent_matrix[np.setdiff1d(ses_transitions,ii)][:,non_zero_boolean]
# 
#     for num_coherent in range(len(coherent_num_all)):
#         coherent_boolean=np.sum(training,axis=0)==num_coherent
#         coherent_num_all[num_coherent][ii]=np.sum(target[coherent_boolean])/len(target[coherent_boolean])
#         
# mean_prop_=np.mean(coherent_num_all,axis=1)
# plt.plot(np.arange(len(mean_prop_)),mean_prop_,color='black')
# plt.savefig(Ephys_output_folder+'/proportion_coherent_trainingvstest_6tasks_increasewithprop.svg')
# plt.show()
# print(mean_prop_)
# 
# mean_prop_day_all=[]
# for mouse_recday in day_type_dicX['combined_ABCDonly']:
#     Xneuron_tuningangle_used_pairs_ABCDonly_day=Xneuron_tuningangle_used_pairs_ABCDonly_dic[mouse_recday]
#     non_zero_boolean_day=angle_to_distance(Xneuron_tuningangle_used_pairs_ABCDonly_day)>0
# 
#     angle_of_angles_ABCDonly_day=dict_to_array(Xneuron_correlations_used_pairs_ABCDonly_dic['angle_of_angles']\
#     [mouse_recday])
# 
#     incoherence_ABCDonly_day=np.vstack(([angle_to_distance(angle_of_angles_ABCDonly_day[ii])\
#     for ii in range(len(angle_of_angles_ABCDonly_day))]))
#     
#     
#     coherent_matrix_day=incoherence_ABCDonly_day<coherence_thr
#     num_coherent_day=np.sum(coherent_matrix_day,axis=0)
#     num_coherent_nonzero_day=num_coherent_day[non_zero_boolean_day]
# 
#     ses_transitions=np.arange(len(coherent_matrix_day))
# 
#     coherent_num_day=np.zeros((len(coherent_matrix),len(coherent_matrix)))
#     for ii in ses_transitions:
#         target=coherent_matrix_day[ii][non_zero_boolean_day]
#         training=coherent_matrix_day[np.setdiff1d(ses_transitions,ii)][:,non_zero_boolean_day]
# 
#         for num_coherent in range(len(coherent_num_all)):
#             coherent_boolean=np.sum(training,axis=0)==num_coherent
#             coherent_num_day[num_coherent][ii]=np.sum(target[coherent_boolean])/len(target[coherent_boolean])
#     
#     mean_prop_=np.mean(coherent_num_day,axis=1)
#     plt.plot(np.arange(len(mean_prop_)),mean_prop_)
#     #plt.savefig(Ephys_output_folder+'/proportion_coherent_trainingvstest_6tasks_increasewithprop.svg')
#     mean_prop_day_all.append(mean_prop_)
# mean_prop_day_all=np.asarray(mean_prop_day_all)
# plt.show()
# print(np.column_stack((day_type_dicX['combined_ABCDonly'],mean_prop_day_all)))

# In[ ]:





# In[ ]:





# In[ ]:


################################


# In[ ]:


###SPATIAL ANCHORING ANALYSIS###


# In[ ]:


################################


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


####Individual neuron spatial anchoring


# In[ ]:





# In[28]:


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


# In[ ]:





# In[62]:


###generating lagged regressors
num_phases,num_nodes,num_lags=3,9,12
remove_edges=True

#GLM_anchoring_prep_dic=rec_dd()

for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
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


    Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy')



    #phases_conc_all_=[]
    #states_conc_all_=[]
    #Location_raw_eq_all_=[]
    #Neuron_raw_all_=[]

    regressors_flat_allTasks=[]
    Location_allTasks=[]
    Neuron_allTasks=[]


    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):

        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            Location_raw=np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            Location_norm=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            XY_raw=np.load(Data_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            speed_raw=speed_dic[mouse_recday][ses_ind]


            acceleration_raw_=np.diff(speed_raw)/0.025
            acceleration_raw=np.hstack((acceleration_raw_[0],acceleration_raw_))
            Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')

        except:
            print('Files not found for session '+str(ses_ind))
            continue

        phases=Phases_raw_dic2[mouse_recday][ses_ind]
        phases_conc=concatenate_complex2(concatenate_complex2(phases))
        states=States_raw_dic[mouse_recday][ses_ind]
        states_conc=concatenate_complex2(concatenate_complex2(states))
        times=Times_from_reward_dic[mouse_recday][ses_ind]
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
        num_task_states=4
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


        regressors_flat_allTasks.append(regressors_flat)
        Location_allTasks.append(Location_raw_eq)
        Neuron_allTasks.append(Neuron_raw_eq.T)

    regressors_flat_allTasks=np.asarray(regressors_flat_allTasks)
    Location_allTasks=np.asarray(Location_allTasks)
    Neuron_allTasks=np.asarray(Neuron_allTasks)
    
    GLM_anchoring_prep_dic['regressors'][mouse_recday]=regressors_flat_allTasks
    GLM_anchoring_prep_dic['Location'][mouse_recday]=Location_allTasks
    GLM_anchoring_prep_dic['Neuron'][mouse_recday]=Neuron_allTasks


# In[ ]:


np.shape(GLM_anchoring_prep_dic['Neuron']['ah03_12082021_13082021'][0])


# In[ ]:


threshold_state_sessions=0
mouse_recday='me11_05122021_06122021'
num_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)              for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T


if threshold_state_sessions=='half':
    state_bool=np.sum(num_peaks_all>0,axis=0)>(len(num_peaks_all)//2)
else:
    state_bool=np.sum(num_peaks_all>0,axis=0)>threshold_state_sessions


# In[ ]:


num_peaks_all


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


####Lagged regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet

tt=time.time()


num_states=4
num_phases=3
num_nodes=9
num_lags=12

use_prefphase=True #
regularize=True
alpha=0.01

#GLM_anchoring_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
    awake_sessions=session_dic['awake'][mouse_recday]
    num_sessions=len(awake_sessions_behaviour)

    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    
    found_ses=[]
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            found_ses.append(ses_ind)

        except:
            print('Files not found for session '+str(ses_ind))
            continue


    regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]
    Location_allTasks=GLM_anchoring_prep_dic['Location'][mouse_recday]
    Neuron_allTasks=GLM_anchoring_prep_dic['Neuron'][mouse_recday]
    
    num_non_repeat_ses_found=len(regressors_flat_allTasks)

    coeffs_all=np.zeros((num_neurons,num_non_repeat_ses_found,num_phases*num_nodes*num_lags))
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
        phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][ses_ind_actual]
        pref_phase_neurons=np.argmax(phase_peaks,axis=1)
        phases=Phases_raw_dic2[mouse_recday][ses_ind_actual]
        
        phase_ses_indices=np.asarray(found_ses)[training_sessions]
        phases_conc_=np.hstack((np.vstack([Phases_raw_dic2[mouse_recday][ses] for ses in phase_ses_indices])))
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
            
    GLM_anchoring_dic['coeffs_all'][mouse_recday]=coeffs_all
print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


GLM_anchoring_dic=GLM_anchoring_regularised_high_dic
##paramaters
num_bins=90
num_states=4
num_phases=3
num_nodes=num_locations=9
num_lags=12
smoothing_sigma=10
num_iterations=100
#thr_anchored_GLM=np.nanpercentile(np.hstack((dict_to_array(GLM_anchoring_dic['Predicted_Actual_correlation_mean']))),50)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Entropy_dic=rec_dd()
num_iterations=100

##paramaters
num_bins=90
num_states=4
num_phases=3
num_nodes=9
num_lags=12
smoothing_sigma=10
phase_norm_mean=np.tile(np.repeat(np.arange(num_phases),num_bins/num_phases),num_states)
phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states,num_bins))

for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
    awake_sessions=session_dic['awake'][mouse_recday]
    num_sessions=len(awake_sessions_behaviour)
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]
    num_non_repeat_ses_found=len(regressors_flat_allTasks)

    found_ses=[]
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            found_ses.append(ses_ind)

        except:
            print('Files not found for session '+str(ses_ind))
            continue

    num_state_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)              for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T

    Entropy_actual_all=np.zeros((num_neurons,num_non_repeat_ses_found))
    Entropy_thr_all=np.zeros((num_neurons,num_non_repeat_ses_found))

    Entropy_actual_all[:]=np.nan
    Entropy_thr_all[:]=np.nan

    for ses_ind_ind in np.arange(num_non_repeat_ses_found):
        ses_ind_actual=found_ses[ses_ind_ind]

        regressors_ses=GLM_anchoring_prep_dic['regressors'][mouse_recday][ses_ind_ind]
        location_ses=GLM_anchoring_prep_dic['Location'][mouse_recday][ses_ind_ind]
        Actual_activity_ses=GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]

        phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][ses_ind_actual]
        pref_phase_neurons=np.argmax(phase_peaks,axis=1)
        phases=Phases_raw_dic2[mouse_recday][ses_ind_actual]
        phases_conc=concatenate_complex2(concatenate_complex2(phases))

        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
        Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25

        for neuron in np.arange(num_neurons):
            pref_phase=pref_phase_neurons[neuron]
            Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]

            Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                    smoothing=False,return_mean=False)

            Actual_norm_means=np.vstack(([[np.nanmean(Actual_norm[trial,num_bins*ii:num_bins*(ii+1)]                [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                              for trial in np.arange(len(Actual_norm))]))
            Actual_norm_means_mean=np.nanmean(Actual_norm_means,axis=0)
            Entropy_actual=st.entropy(Actual_norm_means_mean)



            Entropy_shifted=np.zeros(num_iterations)
            Entropy_shifted[:]=np.nan
            for iteration in range(num_iterations):
                shifts=np.random.randint(0,4,len(Actual_norm_means))
                Actual_norm_means_shifted=indep_roll(Actual_norm_means,shifts)
                Actual_norm_means_shifted_mean=np.nanmean(Actual_norm_means_shifted,axis=0)
                shift_entropy=st.entropy(Actual_norm_means_shifted_mean)
                Entropy_shifted[iteration]=shift_entropy
            Entropy_thr=np.nanpercentile(Entropy_shifted,5)

            Entropy_actual_all[neuron,ses_ind_ind]=Entropy_actual
            Entropy_thr_all[neuron,ses_ind_ind]=Entropy_thr

    Entropy_dic['Entropy_actual'][mouse_recday]=Entropy_actual_all
    Entropy_dic['Entropy_thr'][mouse_recday]=Entropy_thr_all


# In[ ]:





# In[ ]:


pref_phase=pref_phase_neurons[neuron]
Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]
Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                    smoothing=False,return_mean=False)
Actual_norm_means=np.vstack(([[np.nanmean(Actual_norm[trial,num_bins*ii:num_bins*(ii+1)]    [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                  for trial in np.arange(len(Actual_norm))]))

zmax_shifted=np.zeros(num_iterations)
zmax_shifted[:]=np.nan
for iteration in range(num_iterations):
    shifts=np.random.randint(0,4,len(Actual_norm_means))
    Actual_norm_means_shifted=indep_roll(Actual_norm_means,shifts)
    max_state=np.argmax(np.nanmean(Actual_norm_means_shifted,axis=0))
    zactivity_prefstate=st.zscore(Actual_norm_means_shifted,axis=1)[:,max_state]
    zactivity_prefstate_mean=np.mean(zactivity_prefstate)
    zmax_shifted[iteration]=zactivity_prefstate_mean
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###computing state tuning using per trial zscore
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
    awake_sessions=session_dic['awake'][mouse_recday]
    num_sessions=len(awake_sessions_behaviour)
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]
    num_non_repeat_ses_found=len(regressors_flat_allTasks)

    found_ses=[]
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            found_ses.append(ses_ind)

        except:
            print('Files not found for session '+str(ses_ind))
            continue

    num_state_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)              for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T

    zmax_all=np.zeros((num_neurons,num_non_repeat_ses_found))
    zmax_all[:]=np.nan
    
    zmax_all_strict=np.zeros((num_neurons,num_non_repeat_ses_found))
    zmax_all_strict[:]=np.nan
    
    corr_mean_max_all=np.zeros((num_neurons,num_non_repeat_ses_found,2))
    corr_mean_max_all[:]=np.nan

    for ses_ind_ind in np.arange(num_non_repeat_ses_found):
        ses_ind_actual=found_ses[ses_ind_ind]

        regressors_ses=GLM_anchoring_prep_dic['regressors'][mouse_recday][ses_ind_ind]
        location_ses=GLM_anchoring_prep_dic['Location'][mouse_recday][ses_ind_ind]
        Actual_activity_ses=GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]

        phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][ses_ind_actual]
        pref_phase_neurons=np.argmax(phase_peaks,axis=1)
        phases=Phases_raw_dic2[mouse_recday][ses_ind_actual]
        phases_conc=concatenate_complex2(concatenate_complex2(phases))

        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
        Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25

        for neuron in np.arange(num_neurons):
            pref_phase=pref_phase_neurons[neuron]
            Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]

            Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                    smoothing=False,return_mean=False)

            Actual_norm_means=np.vstack(([[np.nanmean(Actual_norm[trial,num_bins*ii:num_bins*(ii+1)]                [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                              for trial in np.arange(len(Actual_norm))]))
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
            
            
            
            Actual_norm_max=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                    smoothing=False,return_mean=False,take_max=True)

            Actual_norm_max_means=np.vstack(([[np.nanmean(Actual_norm_max[trial,num_bins*ii:num_bins*(ii+1)]                [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]                                              for trial in np.arange(len(Actual_norm_max))]))
            
            r_,p_=st.pearsonr(np.concatenate(Actual_norm_means),np.concatenate(Actual_norm_max_means))
            corr_mean_max_all[neuron,ses_ind_ind]=[r_,p_]

    Tuned_dic['State_zmax'][mouse_recday]=zmax_all
    Tuned_dic['State_zmax_strict'][mouse_recday]=zmax_all_strict
    Tuned_dic['corr_mean_max'][mouse_recday]=corr_mean_max_all


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Entropy_thr_all=np.nanmean(np.hstack((concatenate_complex2(dict_to_array(Entropy_dic['Entropy_thr'])))))
day_type='combined_ABCDonly'
for mouse_recday in day_type_dicX[day_type]:
    num_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)                              for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))]))

    state_bool=np.sum(num_peaks_all>0,axis=0)>(len(num_peaks_all)//2)
    
    Entropy_actual_all=Entropy_dic['Entropy_actual'][mouse_recday]
    Entropy_thr_=Entropy_dic['Entropy_thr'][mouse_recday]
    
    state_bool_entropy_withinneuron=np.sum(Entropy_actual_all<Entropy_thr_,axis=1)>(len(Entropy_actual_all.T)/2)
    state_bool_entropy_global=np.sum(Entropy_actual_all<Entropy_thr_all,axis=1)>(len(Entropy_actual_all.T)/2)
    state_bool_entropy=np.sum(np.logical_and(Entropy_actual_all<Entropy_thr_,Entropy_actual_all<Entropy_thr_all,)                                           ,axis=1)>1
    state_bool_entropy_withinneuron=np.sum(Entropy_actual_all<Entropy_thr_,axis=1)>1
    state_bool_entropy_global=np.sum(Entropy_actual_all<Entropy_thr_all,axis=1)>1
    state_bool_entropy=np.logical_and(state_bool_entropy_global,state_bool_entropy_withinneuron)
    
    State_zmax=Tuned_dic['State_zmax'][mouse_recday]
    
    state_bool_zmax=np.sum(State_zmax<0.05,axis=1)>(len(State_zmax.T)/3)
    
    
    Tuned_dic['State_strict'][mouse_recday]=state_bool
    Tuned_dic['State_entropy'][mouse_recday]=state_bool_entropy_withinneuron
    Tuned_dic['State_zmax_bool'][mouse_recday]=state_bool_zmax


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
1-concatenate all sessions and just calculate a regression matrix per neuron
2-cross-validate - 5 tasks to calculate betas then either:
    i-correlate with left out task
    ii-find if top n peaks correspond
3-check:
-regressors correct from model
-concatenation works properly
-reshaping works properly


'''


# In[ ]:


np.shape(Tuned_dic['State_zmax'][mouse_recday])


# In[ ]:


day_type_dicX['combined_ABCDonly'][10:]


# In[ ]:


###Calculating correlations between predicted and actual activity
tt=time.time()
close_to_anchor_bins_90=[0,1,2,11,10,9]
close_to_anchor_bins_30=[0,11]

use_prefphase=True ###if set to false correlations are calculated seperately for each phase and then averaged
use_mean=True ##use normalised, averaged activity for correlations - if true uses mean for each state in each task
###if false, uses trial by trial means for eahc state

Num_max=3 ##how many peaks should NOT be in the excluded regression columns

##paramaters
num_bins=90
num_states=4
num_phases=3
num_nodes=9
num_lags=12
smoothing_sigma=10

regressor_indices=np.arange(num_phases*num_nodes*num_lags)
regressor_indices_reshaped=np.reshape(regressor_indices,(num_phases*num_nodes,num_lags))
zero_indices=regressor_indices_reshaped[:,0]
close_to_anchor_indices30=np.concatenate(regressor_indices_reshaped[:,close_to_anchor_bins_30])
close_to_anchor_indices90=np.concatenate(regressor_indices_reshaped[:,close_to_anchor_bins_90])


phase_norm_mean=np.tile(np.repeat(np.arange(num_phases),num_bins/num_phases),num_states)
phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states,num_bins))

Entropy_thr_all=np.nanmean(np.hstack((concatenate_complex2(dict_to_array(Entropy_dic['Entropy_thr'])))))

for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)

    awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
    awake_sessions=session_dic['awake'][mouse_recday]
    num_sessions=len(awake_sessions_behaviour)
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]
    num_non_repeat_ses_found=len(regressors_flat_allTasks)

    found_ses=[]
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
        try:
            Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            found_ses.append(ses_ind)

        except:
            print('Files not found for session '+str(ses_ind))
            continue
            
    num_state_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)              for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T
    
    entropy_tuned_withinneuron=Entropy_dic['Entropy_actual'][mouse_recday]<Entropy_dic['Entropy_thr'][mouse_recday]
    entropy_tuned_global=Entropy_dic['Entropy_actual'][mouse_recday]<Entropy_thr_all
    entropy_tuned_all=np.logical_and(entropy_tuned_withinneuron,entropy_tuned_global)
    
    state_zmax=Tuned_dic['State_zmax'][mouse_recday]

    corrs_all=np.zeros((num_neurons,num_non_repeat_ses_found))
    corrs_all_nozero=np.zeros((num_neurons,num_non_repeat_ses_found))
    corrs_all_nozero_strict=np.zeros((num_neurons,num_non_repeat_ses_found))

    corrs_all[:]=np.nan
    corrs_all_nozero[:]=np.nan
    corrs_all_nozero_strict[:]=np.nan

    for ses_ind_ind in np.arange(num_non_repeat_ses_found):
        ses_ind_actual=found_ses[ses_ind_ind]

        regressors_ses=GLM_anchoring_prep_dic['regressors'][mouse_recday][ses_ind_ind]
        location_ses=GLM_anchoring_prep_dic['Location'][mouse_recday][ses_ind_ind]
        Actual_activity_ses=GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]

        phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][ses_ind_actual]
        pref_phase_neurons=np.argmax(phase_peaks,axis=1)
        phases=Phases_raw_dic2[mouse_recday][ses_ind_actual]
        phases_conc=concatenate_complex2(concatenate_complex2(phases))
        
        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
        Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25

        for neuron in np.arange(num_neurons):
            pref_phase=pref_phase_neurons[neuron]
            Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]
            coeffs_ses_neuron=GLM_anchoring_dic['coeffs_all'][mouse_recday][neuron][ses_ind_ind]
            
            coeffs_ses_neuron_=GLM_anchoring_dic['coeffs_all'][mouse_recday][neuron][ses_ind_ind]
            coeffs_ses_neuron=np.copy(coeffs_ses_neuron_)
            #coeffs_ses_neuron[coeffs_ses_neuron_<0]=0

            ###maximum indices
            indices_sorted=np.flip(np.argsort(coeffs_ses_neuron))
            indices_sorted_nonan=indices_sorted[~np.isnan(coeffs_ses_neuron[indices_sorted])]
            topN_indices=indices_sorted_nonan[:Num_max]

            Predicted_activity_ses_neuron=np.sum(regressors_ses*coeffs_ses_neuron,axis=1)
            Predicted_activity_ses_neuron_scaled=Predicted_activity_ses_neuron*(            np.mean(Actual_activity_ses_neuron)/np.mean(Predicted_activity_ses_neuron))
            
            num_state_peaks_neuronses=num_state_peaks_all[neuron,ses_ind_ind]
            entropy_tuned_neuronses=entropy_tuned_all[neuron,ses_ind_ind]
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

    GLM_anchoring_dic['Predicted_Actual_correlation'][mouse_recday]=corrs_all
    GLM_anchoring_dic['Predicted_Actual_correlation_mean'][mouse_recday]=corrs_mean
    GLM_anchoring_dic['Predicted_Actual_correlation_nonzero_mean'][mouse_recday]=corrs_all_nozero_mean
    GLM_anchoring_dic['Predicted_Actual_correlation_nonzero_strict_mean'][mouse_recday]=corrs_all_nozero_strict_mean
    
print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


for mouse_recday in day_type_dicX['combined_ABCDonly']:
    mean_regressors=np.nanmean(GLM_anchoring_dic['coeffs_all'][mouse_recday],axis=2)#.keys()
    predicted_actual_corr=GLM_anchoring_dic['Predicted_Actual_correlation'][mouse_recday]

    prop_regressors_nonzero=np.asarray([len(np.where((mean_regressors[neuron])>0)[0])/len(mean_regressors[neuron])                            for neuron in np.arange(len(mean_regressors))])
    half_used_bool=prop_regressors_nonzero>0.5
    
    prop_predictions_nonnan=np.asarray([len(np.where(~np.isnan(predicted_actual_corr[neuron]))[0])                                           /len(predicted_actual_corr[neuron])                            for neuron in np.arange(len(predicted_actual_corr))])
    half_used_bool=prop_predictions_nonnan>0.5
    
    GLM_anchoring_dic['half_used_bool'][mouse_recday]=half_used_bool


# In[ ]:





# In[ ]:


use_tuned=True
bins=50
phase_tuning=np.hstack(([Tuned_dic['Phase'][mouse_recday] for mouse_recday in                         GLM_anchoring_dic['Predicted_Actual_correlation_mean'].keys()]))
state_tuning=np.hstack(([Tuned_dic['State_zmax_bool'][mouse_recday] for mouse_recday in                         GLM_anchoring_dic['Predicted_Actual_correlation_mean'].keys()]))
half_used=np.hstack(([GLM_anchoring_dic['half_used_bool'][mouse_recday] for mouse_recday in                         GLM_anchoring_dic['Predicted_Actual_correlation_mean'].keys()]))


#neurons_tuned=np.logical_and(np.logical_and(state_tuning,phase_tuning),half_used)
neurons_tuned=state_tuning
###i.e. phase/state tuned neurons that have had non-zero betas calculated for atleast half of the sessions



for subset in ['Predicted_Actual_correlation_mean','Predicted_Actual_correlation_nonzero_mean', 'Predicted_Actual_correlation_nonzero_strict_mean']:
    print(subset)
    
    if use_tuned==True:
        corrs_allneurons=remove_nan(concatenate_complex2(dict_to_array                                                     (GLM_anchoring_dic[subset]))[neurons_tuned])
    else:
        corrs_allneurons=remove_nan(concatenate_complex2(dict_to_array                                                     (GLM_anchoring_dic[subset])))
        
    
        
    plt.hist(corrs_allneurons,bins=bins,color='grey')
    #plt.xlim(-1,1)
    plt.axvline(0,color='black',ls='dashed')
    plt.savefig(Ephys_output_folder_dropbox+'GLM_analysis_'+subset+'.svg',                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    #plt.boxplot(corrs_allneurons)
    #plt.axhline(0,ls='dashed',color='black')
    #plt.show()
    #print(len(corrs_allneurons))
    print(st.ttest_1samp(corrs_allneurons,0))


# In[ ]:





# In[ ]:


###Per mouse analysis
for subset in ['Predicted_Actual_correlation_mean','Predicted_Actual_correlation_nonzero_mean', 'Predicted_Actual_correlation_nonzero_strict_mean']:
    for mouse in Mice:
        mouse_recdays_bool=np.asarray([mouse in day_type_dicX['combined_ABCDonly'][ii]                             for ii in range(len(day_type_dicX['combined_ABCDonly']))])
        mouse_recdays_mouse=day_type_dicX['combined_ABCDonly'][mouse_recdays_bool]

        if len(mouse_recdays_mouse)==0:
            continue
        per_mouse_betas=[]
        for mouse_recday in mouse_recdays_mouse:
            state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday]
            neurons_tuned=state_tuning
            per_mouse_betas.append(GLM_anchoring_dic[subset][mouse_recday][neurons_tuned])
        per_mouse_betas=np.hstack((per_mouse_betas))
        
        ttest_res=st.ttest_1samp(remove_nan(per_mouse_betas),0)
        GLM_anchoring_dic['per_mouse_subsetted'][subset][mouse]=per_mouse_betas
        GLM_anchoring_dic['per_mouse_subsetted_mean'][subset][mouse]=np.nanmean(per_mouse_betas)
        GLM_anchoring_dic['per_mouse_subsetted_sem'][subset][mouse]=st.sem(per_mouse_betas,nan_policy='omit')
        GLM_anchoring_dic['per_mouse_subsetted_ttest'][subset][mouse]=ttest_res


# In[ ]:





# In[ ]:


for subset in ['Predicted_Actual_correlation_mean','Predicted_Actual_correlation_nonzero_mean', 'Predicted_Actual_correlation_nonzero_strict_mean']:
    per_mouse_betas_means=dict_to_array(GLM_anchoring_dic['per_mouse_subsetted_mean'][subset])
    per_mouse_betas_sems=dict_to_array(GLM_anchoring_dic['per_mouse_subsetted_sem'][subset])
    per_mouse_betas_ttest=dict_to_array(GLM_anchoring_dic['per_mouse_subsetted_ttest'][subset])
    Mice=np.asarray(list(GLM_anchoring_dic['per_mouse_subsetted_mean'][subset].keys()))
    
    plt.errorbar(per_mouse_betas_means,np.arange(len(per_mouse_betas_means)),xerr=per_mouse_betas_sems,ls='none',
            marker='o',color='grey')
    plt.axvline(0,ls='dashed',color='black')
    plt.savefig(Ephys_output_folder_dropbox+'GLM_analysis_permouse_'+subset+'.svg',                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    plt.show()
    print(np.column_stack((Mice,per_mouse_betas_means)))
    print(per_mouse_betas_ttest)
    
    print(st.ttest_1samp(per_mouse_betas_means,0))
    
    per_mouse_betas_means_nonan=remove_nan(per_mouse_betas_means)
    num_positive=len(np.where(per_mouse_betas_means_nonan>0)[0])
    
    print(two_proportions_test(num_positive, len(per_mouse_betas_means_nonan),                               len(per_mouse_betas_means_nonan)*0.5, len(per_mouse_betas_means_nonan)))
    
    print(st.binom_test(x=num_positive, n=len(per_mouse_betas_means_nonan), p=0.5, alternative='greater'))
    


# In[ ]:


(0.5**4)*3


# In[ ]:


num_positive=5
print(st.binom_test(x=num_positive, n=len(per_mouse_betas_means_nonan), p=0.5, alternative='greater'))


# In[ ]:





# In[ ]:





# In[ ]:


thr_anchored_GLM=np.nanpercentile(np.hstack((dict_to_array(GLM_anchoring_dic['Predicted_Actual_correlation_mean']))),50)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ###Calculating significance thresholds per neuron
# tt=time.time()
# close_to_anchor_bins=[0,1,2,11,10,9]
# 
# use_prefphase=True ###if set to false correlations are calculated seperately for each phase and then averaged
# use_mean=True ##use normalised, averaged activity for correlations
# 
# Num_max=1 ##how many peaks should NOT be in the excluded regression columns
# 
# ##paramaters
# num_bins=90
# num_states=4
# num_phases=3
# num_nodes=9
# num_lags=12
# smoothing_sigma=10
# num_iterations=100
# 
# regressor_indices=np.arange(num_phases*num_nodes*num_lags)
# regressor_indices_reshaped=np.reshape(regressor_indices,(num_phases*num_nodes,num_lags))
# zero_indices=regressor_indices_reshaped[:,0]
# close_to_anchor_indices=np.concatenate(regressor_indices_reshaped[:,close_to_anchor_bins])
# 
# 
# phase_norm_mean=np.tile(np.repeat(np.arange(num_phases),num_bins/num_phases),num_states)
# phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states,num_bins))
# 
# 
# for mouse_recday in day_type_dicX['combined_ABCDonly']:
#     print(mouse_recday)
# 
#     awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
#     awake_sessions=session_dic['awake'][mouse_recday]
#     num_sessions=len(awake_sessions_behaviour)
#     num_neurons=len(cluster_dic['good_clus'][mouse_recday])
#     non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
#     regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]
#     num_non_repeat_ses_found=len(regressors_flat_allTasks)
# 
#     found_ses=[]
#     for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
#         try:
#             Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
#             found_ses.append(ses_ind)
# 
#         except:
#             print('Files not found for session '+str(ses_ind))
#             continue
# 
#     corrs_all=np.zeros((num_neurons,num_non_repeat_ses_found,num_iterations))
# 
#     corrs_all[:]=np.nan
# 
# 
#     for ses_ind_ind in np.arange(num_non_repeat_ses_found):
#         ses_ind_actual=found_ses[ses_ind_ind]
# 
#         regressors_ses=GLM_anchoring_prep_dic['regressors'][mouse_recday][ses_ind_ind]
#         location_ses=GLM_anchoring_prep_dic['Location'][mouse_recday][ses_ind_ind]
#         Actual_activity_ses=GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]
# 
#         phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][ses_ind_actual]
#         pref_phase_neurons=np.argmax(phase_peaks,axis=1)
#         phases=Phases_raw_dic2[mouse_recday][ses_ind_actual]
#         phases_conc=concatenate_complex2(concatenate_complex2(phases))
#         
#         Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
#         Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25
# 
#         for neuron in np.arange(num_neurons):
#             pref_phase=pref_phase_neurons[neuron]
#             Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]
#             coeffs_ses_neuron_=GLM_anchoring_dic['coeffs_all'][mouse_recday][neuron][ses_ind_ind]
#             coeffs_ses_neuron=np.copy(coeffs_ses_neuron_)
#             coeffs_ses_neuron[coeffs_ses_neuron_<0]=0
#             
#             if np.isnan(np.nanmean(coeffs_ses_neuron))==False and\
#             np.nanmean(coeffs_ses_neuron)>0:
#             
#                 ###Circular shift in both directions - (discretised to retain phase tuning)
#                 for iteration in np.arange(num_iterations):
#                     coeffs_reshaped=np.reshape(coeffs_ses_neuron, (num_phases*num_nodes,num_lags))
#                     rand_shift_rows=np.random.randint(0,num_nodes)*num_phases
#                     rand_shift_columns=np.random.randint(0,num_states)*num_phases
#                     coeffs_reshaped_rolled_vertical=np.roll(np.copy(coeffs_reshaped),rand_shift_rows,axis=0)
#                     coeffs_reshaped_rolled_vertical_horizontal=np.roll(np.copy(coeffs_reshaped_rolled_vertical),\
#                                                                        rand_shift_columns,axis=1)
# 
#                     coeffs_ses_neuron_shifted=np.concatenate(coeffs_reshaped_rolled_vertical_horizontal)
# 
# 
#                     Predicted_activity_ses_neuron=np.sum(regressors_ses*coeffs_ses_neuron_shifted,axis=1)
#                     Predicted_activity_ses_neuron_scaled=Predicted_activity_ses_neuron*(\
#                     np.mean(Actual_activity_ses_neuron)/np.mean(Predicted_activity_ses_neuron))
# 
#                     ## prediction for all neurons/entire regression matrix
#                     if use_prefphase==False:
#                         Predicted_Actual_correlation_all=[]
#                         for phase_ind in np.arange(num_phases):
#                             Predicted_Actual_correlation_=\
#                             st.pearsonr(Actual_activity_ses_neuron[phases_conc==phase_ind],\
#                             Predicted_activity_ses_neuron[phases_conc==phase_ind])[0]
#                             Predicted_Actual_correlation_all.append(Predicted_Actual_correlation_)
#                         Predicted_Actual_correlation=np.nanmean(Predicted_Actual_correlation_all)
#                     else:
#                         if use_mean==False:
#                             Predicted_Actual_correlation=\
#                             st.pearsonr(Actual_activity_ses_neuron[phases_conc==pref_phase],\
#                             Predicted_activity_ses_neuron[phases_conc==pref_phase])[0]
#                         else:
#                             Actual_norm=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,\
#                                                             smoothing=False)
#                             Predicted_norm=raw_to_norm(Predicted_activity_ses_neuron,Trial_times_conc,\
#                                                                smoothing=False)
# 
#                             Actual_norm_means=np.asarray([np.nanmean(Actual_norm[num_bins*ii:num_bins*(ii+1)]\
#                                 [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])
#                             Predicted_norm_means=np.asarray([np.nanmean(Predicted_norm[num_bins*ii:num_bins*(ii+1)]\
#                                 [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)])
#                             Predicted_Actual_correlation=st.pearsonr(Actual_norm_means,Predicted_norm_means)[0]
# 
#                     corrs_all[neuron,ses_ind_ind,iteration]=Predicted_Actual_correlation
# 
#     thresholds=np.nanpercentile(np.nanmean(corrs_all,axis=1),95,axis=1)
#     GLM_anchoring_dic['significance_thresholds'][mouse_recday]=thresholds
# print(time.time()-tt)

# In[ ]:





# In[ ]:





# In[ ]:


day_type_dicX['combined_ABCDonly']


# In[ ]:





# In[ ]:


###Testing individual examples

mouse_recday='me08_10092021_11092021'
neuron_order_indx=0
##for mouse_recday in ['ah04_01122021_02122021', 'ah04_05122021_06122021',
#       'ah04_07122021_08122021', 'ah04_09122021_10122021','me11_01122021_02122021',
#       'me11_05122021_06122021', 'me11_07122021_08122021']:
#for neuron_order_indx in np.arange(10):
awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
awake_sessions=session_dic['awake'][mouse_recday]
num_sessions=len(awake_sessions_behaviour)
num_neurons=len(cluster_dic['good_clus'][mouse_recday])
non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
regressors_flat_allTasks=GLM_anchoring_prep_dic['regressors'][mouse_recday]
num_non_repeat_ses_found=len(regressors_flat_allTasks)

found_ses=[]
for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
    try:
        Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
        found_ses.append(ses_ind)

    except:
        print('Files not found for session '+str(ses_ind))
        continue

#thresholds=GLM_anchoring_dic['significance_thresholds'][mouse_recday]
#Corrs_=GLM_anchoring_dic['Predicted_Actual_correlation_mean'][mouse_recday]
#Anchored_=np.where(Corrs_>thresholds)[0]
State_tuned=np.where(Tuned_dic['State_zmax_bool'][mouse_recday]==True)[0]
Phase_tuned=np.where(Tuned_dic['Phase'][mouse_recday]==True)[0]
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
neuron=25

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

    regressors_ses=GLM_anchoring_prep_dic['regressors'][mouse_recday][ses_ind_ind]
    location_ses=GLM_anchoring_prep_dic['Location'][mouse_recday][ses_ind_ind]
    Actual_activity_ses=GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]

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


len(sorted_indices)


# In[ ]:


neuron_order_indx-1
len(sorted_indices)<(neuron_order_indx-1)


# In[ ]:


num_state_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)              for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T
num_state_peaks_all[neuron]


# In[ ]:


print(Tuned_dic['State_zmax_bool'][mouse_recday][neuron])
print(Tuned_dic['State_zmax'][mouse_recday][neuron]<0.05)


# In[ ]:




print(Entropy_dic['Entropy_actual'][mouse_recday][neuron])
print(Entropy_dic['Entropy_thr'][mouse_recday][neuron])

plt.hist(np.concatenate(Entropy_dic['Entropy_actual'][mouse_recday]),bins=50)
plt.axvline(np.nanmean(Entropy_dic['Entropy_thr'][mouse_recday]),color='black')
plt.show()
print(np.nanmean(Entropy_dic['Entropy_thr'][mouse_recday]))


# In[ ]:


'''

1-Poor state tuning

ah04_01122021_02122021
neuron 73 - basically no state tuning 
neuron 51 - inconsistent phase tuning

ah04_05122021_06122021
neuron 34 - only one session with prediction - noisey state tuning
neuron 59 - wrong phase pref

17 - phase tuning changes
40 - low firing and only 3 sessions with predictions
78 - only one session and low firing


'''


# In[ ]:





# In[ ]:


###Testing individual examples

mouse_recday='ah04_01122021_02122021'
print(mouse_recday)

awake_sessions_behaviour=session_dic_behaviour['awake'][mouse_recday]
awake_sessions=session_dic['awake'][mouse_recday]
num_sessions=len(awake_sessions_behaviour)
num_neurons=len(cluster_dic['good_clus'][mouse_recday])
non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
num_non_repeat_ses_found=len(regressors_flat_allTasks)

found_ses=[]
for ses_ind_ind,ses_ind in enumerate(non_repeat_ses):
    try:
        Neuron_raw=np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
        found_ses.append(ses_ind)

    except:
        print('Files not found for session '+str(ses_ind))
        continue

State_tuned=np.where(Tuned_dic['State_zmax_bool'][mouse_recday]==True)[0]

Non_state_tuned=np.where(Tuned_dic['State_zmax_bool'][mouse_recday]==False)[0]


fontsize=10

print(State_tuned)


# In[ ]:


for neuron in State_tuned:
    print(neuron)


    fig1, f1_axes = plt.subplots(figsize=(15, 7.5),ncols=len(non_repeat_ses), constrained_layout=True,                                    subplot_kw={'projection': 'polar'})


    for ses_ind_ind in np.arange(num_non_repeat_ses_found):
        ax1=f1_axes[ses_ind_ind]


        #print(ses_ind_ind)
        ses_ind_actual=found_ses[ses_ind_ind]
        Actual_activity_ses=GLM_anchoring_prep_dic['Neuron'][mouse_recday][ses_ind_ind]
        Trial_times=np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind_actual)+'.npy')
        Trial_times_conc=np.hstack((np.concatenate(Trial_times[:,:-1]),Trial_times[-1,-1]))//25


        ## prediction for all neurons/entire regression matrix
        Actual_activity_ses_neuron=Actual_activity_ses[:,neuron]
        Actual_norm_=raw_to_norm(Actual_activity_ses_neuron,Trial_times_conc,                                        smoothing=False, return_mean=False)
        Actual_norm=np.nanmean(Actual_norm_,axis=0)
        Actual_norm_sem=st.sem(Actual_norm_,axis=0,nan_policy='omit')

        Actual_norm_smoothed=smooth_circular(Actual_norm)
        Actual_norm_sem_smoothed=smooth_circular(Actual_norm_sem)

        Actual_norm_sem_smoothed=smooth_circular(Actual_norm_sem)

        polar_plot_stateX2(Actual_norm_smoothed,Actual_norm_smoothed+Actual_norm_sem_smoothed,                           Actual_norm_smoothed-Actual_norm_sem_smoothed,labels='angles',color='blue',                              ax=ax1,repeated=False,fontsize=fontsize)

    plt.tight_layout()
    fig1.savefig(Ephys_output_folder_dropbox+'Example_cells/Taskmaps_'+mouse_recday+'_neuron_'+str(neuron)+                '.svg', bbox_inches = 'tight', pad_inches = 0)

    plt.show()
    
    print(Entropy_dic['Entropy_actual'][mouse_recday][neuron])
    print(Entropy_dic['Entropy_thr'][mouse_recday][neuron])


# In[ ]:


'''
ah04_05122021_06122021

low firing
4,5,9,11,16

false negative?
35,54,73

me11_05122021_06122021
42
'''


# In[ ]:





# In[ ]:


GLM_anchoring_dic['coeffs_all'].keys()


# In[ ]:


mouse_recday


# In[ ]:





# In[ ]:


###where are the anchors for each neuron - from GLM

N=3
Anchor_topN_GLM_crossval_dic=rec_dd()
day_type='combined_ABCDonly'
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    coeffs_all=GLM_anchoring_dic['coeffs_all'][mouse_recday]
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

            coeffs_indices_reshaped=np.reshape(np.arange(num_phases*num_nodes*num_lags), (num_phases*num_nodes,num_lags))
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


###where are the anchors for each neuron - from GLM

N=3
Anchor_topN_GLM_dic=rec_dd()
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    coeffs_all=GLM_anchoring_dic['coeffs_all'][mouse_recday]
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

        coeffs_indices_reshaped=np.reshape(np.arange(num_phases*num_nodes*num_lags), (num_phases*num_nodes,num_lags))
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


# In[ ]:





# In[ ]:





# In[ ]:


for mouse_recday in day_type_dicX[day_type]:
    mean_betas_neurons=np.nanmean(GLM_anchoring_dic['coeffs_all'][mouse_recday],axis=1)
    GLM_anchoring_dic['coeffs_mean'][mouse_recday]=mean_betas_neurons

All_neuron_betas=np.vstack((dict_to_array(GLM_anchoring_dic['coeffs_mean'])))
All_neuron_betas_mean=np.nanmean(All_neuron_betas,axis=0)
coeffs_neuron_reshaped=All_neuron_betas_mean.reshape((num_locations*num_phases,num_lags))
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

Sum_peak_boolean_reshaped=Sum_peak_boolean.reshape((num_locations*num_phases,num_lags))
plt.matshow(Sum_peak_boolean_reshaped,vmin=0)
for n in np.arange(num_nodes):
    plt.axhline(3*n-0.5,color='white',ls='dashed')
plt.show()
#print(Sum_peak_boolean_reshaped)

plt.bar(np.arange(num_lags),np.nansum(Sum_peak_boolean_reshaped,axis=0),color='black')
plt.show()

Sum_peak_boolean=np.sum(Peak_boolean_topN,axis=0)

Sum_peak_boolean_reshaped=Sum_peak_boolean.reshape((num_locations*num_phases,num_lags))
plt.matshow(Sum_peak_boolean_reshaped,vmin=0)
for n in np.arange(num_nodes):
    plt.axhline(3*n-0.5,color='white',ls='dashed')
plt.show()

plt.bar(np.arange(num_lags),np.nansum(Sum_peak_boolean_reshaped,axis=0),color='black')
plt.savefig(Ephys_output_folder_dropbox+'GLM_Anchor_analysis_lags.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
#print(Sum_peak_boolean_reshaped)

sum_across_anchors=np.nansum(Sum_peak_boolean_reshaped,axis=1)
sum_across_anchors_reshaped=sum_across_anchors.reshape(9,3)

plt.matshow(sum_across_anchors_reshaped.T,vmin=0,cmap='Reds')


# In[ ]:


Task_grid

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
    


# In[ ]:


for ii in np.arange(len(maze_hist)):
    print(st.entropy(np.concatenate(maze_hist[ii]),base=len(np.concatenate(maze_hist[ii]))))


# In[ ]:


sum_across_anchors_reshaped[:,0]


# In[258]:


tt=time.time()
occupancy_phase_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday]) 
    ##Importing Occupancy
    #print('Importing Occupancy')
    #name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
    #data_filename_memmap = os.path.join(Intermediate_object_folder, name)
    #occupancy_ = load(data_filename_memmap)#, mmap_mode='r')
    for session in np.arange(num_sessions):
        try:
            #occupancy_mat=data_matrix(occupancy_[session])
            #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
            
            location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
            if len(location_mat_)==0:
                continue
            occupancy_mat=np.reshape(location_mat_,(4,len(location_mat_),len(location_mat_.T)//4))
            occupancy_conc=np.concatenate(location_mat_)
            
            phase_mat=np.zeros(np.shape(occupancy_mat))
            phase_mat[:,:,30:60]=1
            phase_mat[:,:,60:90]=2
            phase_conc=np.concatenate(np.hstack(phase_mat))
        except Exception as e:
            print(session)
            print(e)
        occupancy_phase_dic[mouse_recday][session]=np.column_stack((phase_conc,occupancy_conc))
    

print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


###Is spatial anchoring conserved across tasks?

tt=time.time()

#Spatial_anchoring_dic=rec_dd()

thr_visit=2
num_phases=3

num_locations=9
num_locations_withedges=21
#num_lags=12
#len_phase=int(360/num_lags)
len_phase=30
num_iterations=100

use_prefphase=True
smoothing_sigma=10
lag_min=90 ##for circular shifts
skip_analysed=True
run_perm=False
for day_type in ['combined_ABCDonly','combined_ABCDE']:
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

        if skip_analysed==True:
            if len(Spatial_anchoring_dic['rotation_angle_mat'][mouse_recday])==num_neurons:
                print('Already analyzed')
                continue
        #Importing Ephys
        print('Importing Ephys')
        num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
        ses_not_found=[]
        for session in np.arange(num_sessions):
            #name='standardized_spike_events_dic_'+mouse_recday+'_'+str(session)
            #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
            try:
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
                exec('ephys_ses_'+str(session)+'_=ephys_')
            except:
                print('Ephys not found')
                ses_not_found.append(session)
                
            if day_type=='combined_ABCDE' and abstract_structures[session]=='ABCD':
                ses_not_found.append(session)

        ##Importing Occupancy
        #print('Importing Occupancy')
        #name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
        #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
        #occupancy_ = load(data_filename_memmap)#, mmap_mode='r')
        
        if day_type=='combined_ABCDonly':
            num_states=4
        elif day_type=='combined_ABCDE':
            num_states=5
        num_lags=num_states*num_phases
        num_bins_total=num_lags*len_phase
        
        ##Calculating rotations
        print('Calculating rotations')
        rotation_dist_mat=np.zeros((num_neurons,num_phases,num_locations,num_sessions,num_sessions))
        rotation_angle_mat=np.zeros((num_neurons,num_phases,num_locations,num_sessions,num_sessions))
        mean_rotation_dist_all=np.zeros((num_neurons,num_phases,num_locations))
        num_passes_all=np.zeros((num_sessions,num_phases,num_locations))
        best_node_phase=np.zeros((num_neurons,2))

        mean_FRs_all=np.zeros((num_neurons,num_sessions,num_lags,num_locations_withedges))
        mean_FRs_shuff_all=np.zeros((num_neurons,num_sessions,num_iterations,num_locations_withedges))

        rotation_dist_mat[:]=np.nan
        rotation_angle_mat[:]=np.nan
        mean_rotation_dist_all[:]=np.nan
        num_passes_all[:]=np.nan
        best_node_phase[:]=np.nan
        mean_FRs_all[:]=np.nan
        mean_FRs_shuff_all[:]=np.nan

        for neuron in np.arange(num_neurons):
            print(neuron)
            for session in np.arange(num_sessions):
                if session in ses_not_found:
                    continue
                exec('ephys_=ephys_ses_'+str(session)+'_')
                #occupancy_mat=data_matrix(occupancy_[session])

                location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
                if len(location_mat_)==0:
                    continue
                occupancy_mat=np.reshape(location_mat_,(num_states,len(location_mat_),len(location_mat_.T)//num_states))


                ephys_neuron_=ephys_[neuron]
                #neuron_mat=data_matrix(ephys_neuron_,concatenate=False)
                neuron_mat=ephys_neuron_

                phase_mat=np.zeros(np.shape(occupancy_mat))
                phase_mat[:,:,30:60]=1
                phase_mat[:,:,60:90]=2
                phase_conc=np.concatenate(np.hstack(phase_mat))
                
                if day_type=='combined_ABCDonly':
                    phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][session]
                elif day_type=='combined_ABCDE':
                    num_ABCD_ses=np.sum(abstract_structures=='ABCD')
                    phase_peaks=np.load(Intermediate_object_folder_dropbox+'ABCDE_tuning_phase_boolean_max_'+                                        mouse_recday+'.npy')[session-num_ABCD_ses]
                pref_phase_neurons=np.argmax(phase_peaks,axis=1)
                pref_phase=pref_phase_neurons[neuron]

                mean_FRs_neuronsession=np.zeros((num_lags,num_locations_withedges))
                mean_FRs_neuronsession[:]=np.nan
                mean_FRs_shuff_session=np.zeros((num_iterations,num_locations_withedges))
                mean_FRs_shuff_session[:]=np.nan

                if len(neuron_mat)==0 or len(occupancy_mat)==0:
                    if neuron==0 and session==0:
                        print('No data for session'+str(session))

                else:
                    #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
                    occupancy_conc=np.concatenate(location_mat_)
                    #neuron_conc=np.concatenate(np.hstack(neuron_mat))
                    neuron_conc=np.concatenate(neuron_mat)

                    min_len=np.min([len(occupancy_conc),len(neuron_conc)])
                    neuron_conc=neuron_conc[:min_len]
                    occupancy_conc=occupancy_conc[:min_len]
                    phase_conc=phase_conc[:min_len]

                    ##Shifting
                    for shift in np.arange(num_lags):
                        occupancy_shifted=np.roll(occupancy_conc,len_phase*shift) 
                        ###i.e. if peak in 4th bin (bin index 3) then anchor is 4 bins behind peak of activity 
                        occupancy_shifted_nonan=occupancy_shifted[~np.isnan(occupancy_shifted)]
                        neuron_conc_nonan=neuron_conc[~np.isnan(occupancy_shifted)]

                        occupancy_shifted_nonan_prefphase=occupancy_shifted[np.logical_and(~np.isnan(occupancy_shifted),                                                                               phase_conc==pref_phase)]
                        neuron_conc_nonan_pref_phase=neuron_conc[np.logical_and(~np.isnan(occupancy_shifted),                                                                                phase_conc==pref_phase)]

                        if use_prefphase==True:
                            mean_FRs=st.binned_statistic(occupancy_shifted_nonan_prefphase,                                            neuron_conc_nonan_pref_phase,bins=np.arange(22)+1,statistic='mean')[0]
                        else:
                            mean_FRs=st.binned_statistic(occupancy_shifted_nonan, neuron_conc_nonan,bins=                                                     np.arange(num_locations_withedges+1)+1,statistic='mean')[0]

                        mean_FRs_neuronsession[shift]=mean_FRs

                    ###Permutations (random circular shift)
                    if run_perm==True:
                        occupancy_conc_shuff=np.split(np.copy(occupancy_conc),len(occupancy_conc)/90)
                        max_roll=int(len(occupancy_conc)-lag_min)
                        min_roll=int(lag_min)

                        for iteration in np.arange(num_iterations):
                            shift_rand=random.randrange(max_roll-min_roll)+min_roll
                            occupancy_conc_shuff=np.hstack((occupancy_conc_shuff))
                            occupancy_conc_shuff=np.roll(occupancy_conc_shuff,shift_rand)
                            occupancy_conc_shuff_nonan=occupancy_conc_shuff[~np.isnan(occupancy_conc_shuff)]
                            neuron_conc_nonan=neuron_conc[~np.isnan(occupancy_conc_shuff)]

                            occupancy_shuff_nonan_prefphase=occupancy_conc_shuff[np.logical_and                                                                                 (~np.isnan(occupancy_conc_shuff),                                                                                               phase_conc==pref_phase)]
                            neuron_conc_nonan_pref_phase=neuron_conc[np.logical_and(~np.isnan(occupancy_conc_shuff),                                                                                    phase_conc==pref_phase)]

                            if use_prefphase==True:
                                mean_FRs_shuff=st.binned_statistic(occupancy_shuff_nonan_prefphase,                                                                   neuron_conc_nonan_pref_phase,                                                    bins=np.arange(num_locations_withedges+1)+1,statistic='mean')[0]
                            else:
                                mean_FRs_shuff=st.binned_statistic(occupancy_conc_shuff_nonan, neuron_conc_nonan,                                                    bins=np.arange(num_locations_withedges+1)+1,statistic='mean')[0]


                            mean_FRs_shuff_session[iteration]=mean_FRs_shuff


                mean_FRs_all[neuron][session]=mean_FRs_neuronsession
                if run_perm==True:
                    mean_FRs_shuff_all[neuron][session]=mean_FRs_shuff_session



            for phase_ in np.arange(num_phases):
                for location_ in np.arange(num_locations)+1:
                    #print(location_)
                    mean_smooth_all=np.zeros((num_sessions,num_bins_total))
                    for session in np.arange(num_sessions):
                        exec('ephys_=ephys_ses_'+str(session)+'_')
                        #occupancy_mat=data_matrix(occupancy_[session])
                        try:
                            location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
                        except:
                            print('Location file not found')
                            continue
                        if len(location_mat_)==0:
                            continue
                        occupancy_mat=np.reshape(location_mat_,                                                 (num_states,len(location_mat_),len(location_mat_.T)//num_states))
                        #occupancy_conc=np.concatenate(location_mat_)

                        ephys_neuron_=ephys_[neuron]
                        neuron_mat=data_matrix(ephys_neuron_,concatenate=False)

                        if len(neuron_mat)==0 or len(occupancy_mat)==0:
                            if neuron==0 and location_==1 and phase_==0:
                                print('No data for session'+str(session))
                            mean_smooth_all[session]=np.repeat(np.nan,num_bins_total)
                            num_passes_all[session,phase_,int(location_-1)]=0
                            continue


                        #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
                        occupancy_conc=np.concatenate(location_mat_)
                        phase_mat=np.zeros(np.shape(occupancy_mat))
                        phase_mat[:,:,30:60]=1
                        phase_mat[:,:,60:90]=2
                        phase_conc=np.concatenate(np.hstack(phase_mat))


                        #neuron_conc=np.concatenate(np.hstack(neuron_mat))
                        neuron_conc=np.concatenate(neuron_mat)



                        timestamps_=np.where((np.logical_and(occupancy_conc==location_, phase_conc==phase_)))[0]
                        #timestamps_=np.where(occupancy_conc==location_)[0]
                        long_stays=np.where(rank_repeat2(occupancy_conc)>thr_visit-1)[0]
                        timestamps=np.intersect1d(timestamps_,long_stays-(thr_visit+1))

                        if len(timestamps)>0:

                            timestamps_start=timestamps[(np.hstack((1,np.diff(timestamps)>thr_visit))).astype(bool)]
                            timestamps_end=timestamps[(np.hstack((np.diff(timestamps)>thr_visit,1))).astype(bool)]
                            time_stamps_used=timestamps_start

                            aligned_activity=np.asarray([neuron_conc[ii:ii+num_bins_total]                                                         if len(neuron_conc[ii:])>=num_bins_total else                                                         np.repeat(np.nan,num_bins_total) for ii in time_stamps_used])


                            mean_=np.nanmean(aligned_activity,axis=0)
                            sem_=st.sem(aligned_activity,axis=0,nan_policy='omit')
                            mean_smooth=smooth_circular(mean_,sigma=smoothing_sigma)
                            sem_smooth=smooth_circular(sem_,sigma=smoothing_sigma)

                            if np.nanmean(mean_smooth)==0 or np.isnan(np.nanmean(mean_smooth))==True:
                                mean_smooth=np.repeat(np.nan,num_bins_total)

                            mean_smooth_all[session]=mean_smooth

                            num_passes_all[session,phase_,int(location_-1)]=len(time_stamps_used)

                        else:
                            #print('Not visited')
                            mean_smooth_all[session]=np.repeat(np.nan,num_bins_total)
                            num_passes_all[session,phase_,int(location_-1)]=0

                    for indX in np.arange(num_sessions):
                        for indY in np.arange(num_sessions):
                            if indX!=indY:
                                if np.logical_and(~np.isnan(np.nanmean(mean_smooth_all[indX])),                                                                            ~np.isnan(np.nanmean(mean_smooth_all[indY]))):
                                    shifted_corrs=np.asarray([st.pearsonr(mean_smooth_all[indX],                                                                          np.roll(mean_smooth_all[indY],int(n*10)))[0]                                                              for n in np.arange(num_bins_total/10) ])
                                    rotation=np.argmax(shifted_corrs)*10
                                    rotation_dist=1-math.cos(np.deg2rad(rotation))
                                else:
                                    rotation=np.nan
                                    rotation_dist=np.nan



                            else:
                                rotation=0
                                rotation_dist=0

                            rotation_dist_mat[neuron,phase_,int(location_-1),indX,indY]=rotation_dist
                            rotation_angle_mat[neuron,phase_,int(location_-1),indX,indY]=rotation

                    mean_rotation_dist=np.nanmean(matrix_triangle(rotation_dist_mat[neuron,phase_,int(location_-1)],                                                                  direction='lower'))
                    mean_rotation_dist_all[neuron,phase_,int(location_-1)]=mean_rotation_dist

                    if np.min(num_passes_all[:,phase_,int(location_-1)])<=thr_visit:
                        mean_rotation_dist_all[neuron,phase_,int(location_-1)]=np.nan



            best_anchor_=np.where(mean_rotation_dist_all[neuron]==np.nanmin(mean_rotation_dist_all[neuron]))
            if len(best_anchor_[0])==1:
                best_anchor_phase=best_anchor_[0][0]
                best_anchor_node=best_anchor_[1][0]+1
            elif len(best_anchor_[0])>1:
                print('Multiple best anchors')
            elif len(best_anchor_[0])==0:
                best_anchor_node=np.nan
                best_anchor_phase=np.nan


            #print(best_anchor_node,best_anchor_phase)
            best_node_phase[neuron]=np.asarray([best_anchor_node,best_anchor_phase])


        Spatial_anchoring_dic['rotation_angle_mat'][mouse_recday]=rotation_angle_mat
        Spatial_anchoring_dic['rotation_dist_mat'][mouse_recday]=rotation_dist_mat
        Spatial_anchoring_dic['mean_rotation_dist_all'][mouse_recday]=mean_rotation_dist_all
        Spatial_anchoring_dic['num_passes_all'][mouse_recday]=num_passes_all
        Spatial_anchoring_dic['best_node_phase'][mouse_recday]=best_node_phase

        #Spatial_anchoring_dic['Phase_shifted_maps'][mouse_recday]=mean_FRs_all

        if run_perm==True:
            Spatial_anchoring_dic['Shuffled_maps'][mouse_recday]=mean_FRs_shuff_all

print(time.time()-tt)


# In[24]:


(num_states,len(location_mat_),len(location_mat_.T)//num_states)


# In[26]:


np.shape(location_mat_)


# In[22]:


num_states


# In[30]:


mouse_recday='ah07_21112023_22112023'
location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
np.shape(location_mat_)
#                location_mat_=location_mat_.squeeze()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[259]:


###Running just the lagged spatial map analysis

thr_visit=2
num_phases=3

num_locations=9
num_locations_withedges=21
num_lags=12
len_phase=int(360/num_lags)
num_iterations=100

use_prefphase=True

lag_min=90 ##for circular shifts

for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])

    #Importing Ephys
    print('Importing Ephys')
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    ses_not_found=[]
    for session in np.arange(num_sessions):
        #name='standardized_spike_events_dic_'+mouse_recday+'_'+str(session)
        #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
        try:
            #ephys_ = load(data_filename_memmap)#, mmap_mode='r')
            ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
            exec('ephys_ses_'+str(session)+'_=ephys_')
        except:
            print('Ephys not found')
            ses_not_found.append(session)

    ##Importing Occupancy
    #print('Importing Occupancy')
    #name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
    #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
    #occupancy_ = load(data_filename_memmap)#, mmap_mode='r')

    ##Calculating rotations
    print('Calculating rotations')
    rotation_dist_mat=np.zeros((num_neurons,num_phases,num_locations,num_sessions,num_sessions))
    rotation_angle_mat=np.zeros((num_neurons,num_phases,num_locations,num_sessions,num_sessions))
    mean_rotation_dist_all=np.zeros((num_neurons,num_phases,num_locations))
    num_passes_all=np.zeros((num_sessions,num_phases,num_locations))
    best_node_phase=np.zeros((num_neurons,2))

    mean_FRs_all=np.zeros((num_neurons,num_sessions,num_lags,num_locations_withedges))
    mean_FRs_shuff_all=np.zeros((num_neurons,num_sessions,num_iterations,num_locations_withedges))
    
    mean_FRs_all[:]=np.nan
    mean_FRs_shuff_all[:]=np.nan
    for neuron in np.arange(num_neurons):
        print(neuron)
        for session in np.arange(num_sessions):
            if session in ses_not_found:
                print('not found x')
                continue
            exec('ephys_=ephys_ses_'+str(session)+'_')
            #occupancy_mat=data_matrix(occupancy_[session])
            location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
            if len(location_mat_)==0:
                continue
            occupancy_mat=np.reshape(location_mat_,(4,len(location_mat_),len(location_mat_.T)//4))
            occupancy_conc=np.concatenate(location_mat_)
            ephys_neuron_=ephys_[neuron]
            #neuron_mat=data_matrix(ephys_neuron_,concatenate=False)
            neuron_mat=ephys_neuron_

            phase_mat=np.zeros(np.shape(occupancy_mat))
            phase_mat[:,:,30:60]=1
            phase_mat[:,:,60:90]=2
            phase_conc=np.concatenate(np.hstack(phase_mat))

            phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][session]
            pref_phase_neurons=np.argmax(phase_peaks,axis=1)
            pref_phase=pref_phase_neurons[neuron]

            mean_FRs_neuronsession=np.zeros((num_lags,num_locations_withedges))
            mean_FRs_neuronsession[:]=np.nan
            mean_FRs_shuff_session=np.zeros((num_iterations,num_locations_withedges))
            mean_FRs_shuff_session[:]=np.nan

            if len(neuron_mat)==0 or len(occupancy_mat)==0:
                if neuron==0 and session==0:
                    print('No data for session'+str(session))

            else:
                #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
                #neuron_conc=np.concatenate(np.hstack(neuron_mat))
                neuron_conc=np.concatenate(neuron_mat)

                min_len=np.min([len(occupancy_conc),len(neuron_conc)])
                neuron_conc=neuron_conc[:min_len]
                occupancy_conc=occupancy_conc[:min_len]
                phase_conc=phase_conc[:min_len]

                ##Shifting
                for shift in np.arange(num_lags):
                    occupancy_shifted=np.roll(occupancy_conc,len_phase*shift) 
                    ###i.e. if peak in 4th bin (bin index 3) then anchor is 4 bins behind peak of activity 
                    occupancy_shifted_nonan=occupancy_shifted[~np.isnan(occupancy_shifted)]
                    neuron_conc_nonan=neuron_conc[~np.isnan(occupancy_shifted)]

                    occupancy_shifted_nonan_prefphase=occupancy_shifted[np.logical_and(~np.isnan(occupancy_shifted),                                                                           phase_conc==pref_phase)]
                    neuron_conc_nonan_pref_phase=neuron_conc[np.logical_and(~np.isnan(occupancy_shifted),                                                                            phase_conc==pref_phase)]

                    if use_prefphase==True:
                        mean_FRs=st.binned_statistic(occupancy_shifted_nonan_prefphase,                                                     neuron_conc_nonan_pref_phase,bins=np.arange(22)+1,statistic='mean')[0]
                    else:
                        mean_FRs=st.binned_statistic(occupancy_shifted_nonan, neuron_conc_nonan,bins=                                                 np.arange(num_locations_withedges+1)+1,statistic='mean')[0]

                    mean_FRs_neuronsession[shift]=mean_FRs

            mean_FRs_all[neuron][session]=mean_FRs_neuronsession

    Spatial_anchoring_dic['Phase_shifted_maps'][mouse_recday]=mean_FRs_all


# In[33]:


##Examples - pre analysis
mouse_recday='ah04_05122021_06122021'

#for mouse_recday in day_type_dicX['3_task']:
print(mouse_recday)

all_neurons=np.arange(len(cluster_dic['good_clus'][mouse_recday]))
abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

for neuron in [43]:
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





# In[26]:


###Making Spatial lagged matrices (for plotting)
num_lags=12
num_locations=num_nodes=9
num_locations_withedges=21
num_lags=12
day_type='combined_ABCDonly'
for mouse_recday in day_type_dicX[day_type]:
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    
    node_rate_matrices=np.empty((num_neurons,len(non_repeat_ses),num_lags,3,3))
    node_rate_matrices[:]=np.nan

    edge_rate_matrices=np.empty((num_neurons,len(non_repeat_ses),num_lags,5,5))
    edge_rate_matrices[:]=np.nan

    node_edge_rate_matrices=np.empty((num_neurons,len(non_repeat_ses),num_lags,5,5))
    node_edge_rate_matrices[:]=np.nan
    for neuron in np.arange(num_neurons):
        maps_neuron=Spatial_anchoring_dic['Phase_shifted_maps'][mouse_recday][neuron]
        
        node_rate_mat=np.zeros((3,3))
        node_rate_mat[:]=np.nan
        
        edge_rate_mat=np.zeros((5,5))
        edge_rate_mat[:]=np.nan


        

        for awake_session_ind_ind, awake_session_ind in enumerate(non_repeat_ses):
            for lag in np.arange(num_lags):
                for node_indx,mat_indx in enumerate(Task_grid):
                    node_rate_matrices[neuron,awake_session_ind_ind,lag,mat_indx[0],mat_indx[1]]=                    node_rate_mat[mat_indx[0],mat_indx[1]]=                    maps_neuron[awake_session_ind,lag,node_indx]

                for edge_indx,mat_indx in enumerate(Edge_grid_coord2):
                    edge_rate_matrices[neuron,awake_session_ind_ind,lag,mat_indx[0],mat_indx[1]]=                    edge_rate_mat[mat_indx[0],mat_indx[1]]=                    maps_neuron[awake_session_ind,lag,num_nodes:][edge_indx]

                node_edge_mat=edge_node_fill(edge_rate_mat,node_rate_mat)

                node_edge_rate_matrices[neuron,awake_session_ind_ind,lag]=node_edge_mat
    
    Spatial_anchoring_dic['Phase_shifted_node_edge_matrices'][mouse_recday]=node_edge_rate_matrices
    Spatial_anchoring_dic['Phase_shifted_node_matrices'][mouse_recday]=node_rate_matrices


# In[ ]:





# In[ ]:





# In[10]:


###Task-space Shifted spatial correlations
Phase_spatial_corr_dic=rec_dd()
run_shuffle=True
num_locations=9
num_locations_withedges=21
num_lags=12
len_phase=int(360/num_lags)
if run_shuffle==True:
    num_iterations=100
else:
    num_iterations=0
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    
    print(mouse_recday)
    sessions=Task_num_dic[mouse_recday]
    num_refses=len(np.unique(sessions))
    num_comparisons=num_refses-1
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)  

    #num_state_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)\
    #          for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T
    
    state_tuned=Tuned_dic2['State_onethirdtasks']['95'][mouse_recday]
    
    Phase_shifted_maps=Spatial_anchoring_dic['Phase_shifted_maps'][mouse_recday]
    
    len_lower_triangle=len(matrix_triangle(np.corrcoef(Phase_shifted_maps[0,non_repeat_ses,0]),direction='lower'))
    corrs_all=np.zeros((num_neurons,num_lags,len_lower_triangle)) 
    corrs_all_shuff=np.zeros((num_neurons,num_iterations,num_lags,len_lower_triangle))
    corrs_all_shuff_mean=np.zeros((num_neurons,num_iterations))
    corrs_crossval_all=np.zeros((num_neurons,num_lags,len(non_repeat_ses))) 
    
    corrs_all[:]=np.nan
    corrs_all_shuff[:]=np.nan
    corrs_all_shuff_mean[:]=np.nan
    corrs_crossval_all[:]=np.nan
    for neuron in np.arange(num_neurons):
        Phase_shifted_maps_clean=Phase_shifted_maps[neuron,non_repeat_ses,:,:num_locations]
        ###Note: removing edges from comparison as shifts cause systematic errors
        
        #state_peaks_neuron_bool=num_state_peaks_all[neuron]>0
        #Phase_shifted_maps_clean[state_peaks_neuron_bool==False]=np.nan
        
        corrs_neuron=np.vstack(([matrix_triangle(pd.DataFrame.to_numpy((pd.DataFrame(Phase_shifted_maps_clean[:,ii].T))                        .corr()),direction='lower') for ii in range(num_lags)]))
        
        corrs_all[neuron]=corrs_neuron
        
        ###leave one out approach
        corrs_neuron_crossval=np.vstack(([[pd.DataFrame(np.nanmean(Phase_shifted_maps_clean[np.setdiff1d                                (np.arange(len(non_repeat_ses)),session),lag],axis=0))[0]                         .corr(pd.DataFrame(Phase_shifted_maps_clean[session,lag])[0]) for lag in                         np.arange(np.shape(Phase_shifted_maps_clean)[1])] for session in np.arange(len(non_repeat_ses))]))
        
        corrs_crossval_all[neuron]=corrs_neuron_crossval.T
        
        ###logic of shuffle, randomly shuffling spatial maps across lags within a task, so that across task comparison
        ##is between maps at different lags (e.g. lag 6 in task 1 against lag 3 in task 2 and lag 0 in task 3...etc)
        ##then take mean correlation per iteration, then take 95th percentile of the means across iterations
        ##so one threshold per neuron
        for iteration in np.arange(num_iterations):
            Phase_shifted_maps_clean_copy=np.copy(Phase_shifted_maps_clean)
            [np.random.shuffle(Phase_shifted_maps_clean_copy[ii]) for ii in range(len(Phase_shifted_maps_clean_copy))]
            corrs_neuron_shuff=np.vstack(([matrix_triangle(pd.DataFrame.to_numpy((            pd.DataFrame(Phase_shifted_maps_clean_copy[:,ii].T)).corr()),direction='lower') for ii in range(num_lags)]))
        
            corrs_all_shuff[neuron][iteration]=corrs_neuron_shuff
            corrs_all_shuff_mean[neuron][iteration]=np.nanmean(corrs_neuron_shuff)
            
    
    mean_corrs=np.nanmean(corrs_all,axis=2)
    std_corrs=np.nanstd(corrs_all,axis=2)
    max_corr_bin=_nanargmax(mean_corrs,axis=1)
    
    mean_corrs_crossval=np.nanmean(corrs_crossval_all,axis=2)
    max_corr_bin_crossval=_nanargmax(mean_corrs_crossval,axis=1)
    
    Phase_spatial_corr_dic['corrs_all'][mouse_recday]=corrs_all
    Phase_spatial_corr_dic['Max_corr_bin'][mouse_recday]=max_corr_bin
    
    if run_shuffle==True:
        
        thr_corrs=np.percentile(corrs_all_shuff_mean,95,axis=1)
        Phase_spatial_corr_dic['Threshold'][mouse_recday]=thr_corrs
        Phase_spatial_corr_dic['corrs_all_shuff'][mouse_recday]=corrs_all_shuff

    Phase_spatial_corr_dic['corrs_crossval_all'][mouse_recday]=corrs_crossval_all
    Phase_spatial_corr_dic['Max_corr_bin_crossval'][mouse_recday]=max_corr_bin_crossval


# In[ ]:





# In[ ]:





# In[11]:


###Task-space Shifted spatial correlations - cross-validated single correlation value per neuron

close_to_anchor_bins30=[0,11]
close_to_anchor_bins90=[0,1,2,11,10,9]

use_tuned_only=False ##this is done when plotting below
include_bridges=False

for mouse_recday in day_type_dicX['combined_ABCDonly']:
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    
    print(mouse_recday)
    sessions=Task_num_dic[mouse_recday]
    num_refses=len(np.unique(sessions))
    num_comparisons=num_refses-1
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)  
    
    Phase_boolean=Tuned_dic['Phase'][mouse_recday]
    State_boolean=Tuned_dic['State'][mouse_recday]
    
    used_boolean=np.logical_and(Phase_boolean,State_boolean)
    
    #num_state_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)\
    #          for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T
    
    Phase_shifted_maps_=Spatial_anchoring_dic['Phase_shifted_maps'][mouse_recday]
    mean_bestlag_all=np.zeros((num_neurons)) 
    mean_bestlag_nonspatial=np.zeros((num_neurons))
    mean_bestlag_nonspatial_strict=np.zeros((num_neurons))
    max_lags_all=np.zeros((num_neurons,len(non_repeat_ses)))
    mean_bestlag_all[:]=np.nan
    mean_bestlag_nonspatial[:]=np.nan
    mean_bestlag_nonspatial_strict[:]=np.nan
    max_lags_all[:]=np.nan

    
    for neuron in np.arange(num_neurons):
        if include_bridges==True:
            Phase_shifted_map=Phase_shifted_maps_[neuron,non_repeat_ses]#,:,:num_locations]
        else:
            Phase_shifted_map=Phase_shifted_maps_[neuron,non_repeat_ses,:,:num_locations]
        ###Note: removing edges from comparison as shifts cause systematic errors
        
        #state_peaks_neuron_bool=num_state_peaks_all[neuron]>0
        #Phase_shifted_map[state_peaks_neuron_bool==False]=np.nan

        max_lags=[_nanargmax([np.nanmean(matrix_triangle(            pd.DataFrame.to_numpy(pd.DataFrame(Phase_shifted_map[                np.setdiff1d(np.arange(len(non_repeat_ses)),session),lag]).T.corr()))) for                    lag in np.arange(np.shape(Phase_shifted_map)[1])])                        for session in np.arange(len(non_repeat_ses))]

        max_corrs=[np.nanmax([np.nanmean(matrix_triangle(            pd.DataFrame.to_numpy(pd.DataFrame(Phase_shifted_map[                np.setdiff1d(np.arange(len(non_repeat_ses)),session),lag]).T.corr()))) for                    lag in np.arange(np.shape(Phase_shifted_map)[1])])                        for session in np.arange(len(non_repeat_ses))]

        mean_bestlag=np.nanmean([np.nanmean(pd.DataFrame(Phase_shifted_map[:,max_lags[session]]).T.corr()             [session][np.setdiff1d(np.arange(len(non_repeat_ses)),session)])                                 if np.isnan(max_lags[session])==False else np.nan
                                 for session in np.arange(len(non_repeat_ses))])
        
        
        
        mean_best_nonspatial=np.nanmean([np.nanmean(pd.DataFrame(Phase_shifted_map[:,max_lags[session]]).T.corr()                 [session][np.setdiff1d(np.arange(len(non_repeat_ses)),session)])                     if max_lags[session] not in close_to_anchor_bins30 and np.isnan(max_lags[session])==False                                         else np.nan for session in np.arange(len(non_repeat_ses))])
        
        mean_best_nonspatial_strict=np.nanmean([np.nanmean(pd.DataFrame(Phase_shifted_map[:,max_lags[session]]).T.corr()                 [session][np.setdiff1d(np.arange(len(non_repeat_ses)),session)])                     if max_lags[session] not in close_to_anchor_bins90 and np.isnan(max_lags[session])==False 
                                                else np.nan for session in np.arange(len(non_repeat_ses))])
        
        if use_tuned_only==True:
            if used_boolean[neuron]==False:
                mean_bestlag=mean_best_nonspatial=mean_best_nonspatial_strict=np.nan
        
        ##removing spatial neurons
        ##overwrites above as otherwise contaminating nonspatial with imperfections of spatial neurons
        if st.mode(max_lags,keepdims=True)[0][0] in close_to_anchor_bins30 and st.mode(max_lags,keepdims=True)[1][0]>1:
            mean_best_nonspatial=np.nan
            
        if st.mode(max_lags,keepdims=True)[0][0] in close_to_anchor_bins90 and st.mode(max_lags,keepdims=True)[1][0]>1:
            mean_best_nonspatial_strict=np.nan
            
        
        mean_bestlag_all[neuron]=mean_bestlag
        mean_bestlag_nonspatial[neuron]=mean_best_nonspatial
        mean_bestlag_nonspatial_strict[neuron]=mean_best_nonspatial_strict
        
        max_lags_all[neuron]=max_lags
    
    Phase_spatial_corr_dic['mean_bestlag_corr_crossval'][mouse_recday]=mean_bestlag_all
    Phase_spatial_corr_dic['mean_bestlag_corr_nonzerolag_crossval'][mouse_recday]=mean_bestlag_nonspatial
    Phase_spatial_corr_dic['mean_bestlag_corr_nonzerolag_strict_crossval'][mouse_recday]=mean_bestlag_nonspatial_strict
    Phase_spatial_corr_dic['max_lags_all'][mouse_recday]=max_lags_all


# In[ ]:





# In[ ]:





# In[262]:


use_tuned=True

used_recdays_=np.asarray(list(Phase_spatial_corr_dic['mean_bestlag_corr_crossval'].keys()))
non_neuropixels_bool=np.asarray([Cohort_ephys_type_dic[Mice_cohort_dic[used_recdays_[ii][:4]]]!='Neuropixels'                      for ii in range(len(used_recdays_))])
#used_recdays_=used_recdays_[non_neuropixels_bool]

phase_tuning=np.hstack(([Tuned_dic['Phase'][mouse_recday] for mouse_recday in used_recdays_]))
#state_tuning=np.hstack(([Tuned_dic['State_zmax_bool'][mouse_recday] for mouse_recday in\
#                         Phase_spatial_corr_dic['mean_bestlag_corr_crossval'].keys()]))

state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'State_'+mouse_recday+'.npy')                                 for mouse_recday in used_recdays_]))

state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'State_zmax_bool_'+mouse_recday+'.npy')        for mouse_recday in used_recdays_]))

place_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'Place_'+mouse_recday+'.npy')        for mouse_recday in used_recdays_]))


#state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'State_zmax_bool_one_strict_'+mouse_recday+'.npy')\
#            for mouse_recday in Phase_spatial_corr_dic['mean_bestlag_corr_crossval'].keys()]))

neurons_tuned=state_tuning

plt.rcParams["figure.figsize"] = (7,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
for subset in ['mean_bestlag_corr_crossval','mean_bestlag_corr_nonzerolag_crossval',               'mean_bestlag_corr_nonzerolag_strict_crossval']:
    print(subset)
    if use_tuned==True:
        #mean_bestlag_corr_crossval=remove_nan(np.hstack((dict_to_array(Phase_spatial_corr_dic[subset])))[neurons_tuned])
        mean_bestlag_corr_crossval=        remove_nan(np.hstack(([Phase_spatial_corr_dic[subset][mouse_recday] for mouse_recday in used_recdays_]))                   [neurons_tuned])
    else:
        mean_bestlag_corr_crossval=        remove_nan(np.hstack(([Phase_spatial_corr_dic[subset][mouse_recday] for mouse_recday in used_recdays_])))
        
    plt.hist(mean_bestlag_corr_crossval,bins=50,color='grey')
    plt.axvline(0,color='black',ls='dashed')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'SpatialLag_analysis_'+subset+'.svg' , bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print(len(mean_bestlag_corr_crossval))
    print(st.ttest_1samp(mean_bestlag_corr_crossval,0))
    print('')


# In[ ]:





# In[263]:


###Per mouse analysis
for subset in ['mean_bestlag_corr_crossval','mean_bestlag_corr_nonzerolag_crossval',               'mean_bestlag_corr_nonzerolag_strict_crossval']:
    for mouse in Mice:
        mouse_recdays_bool=np.asarray([mouse in day_type_dicX['combined_ABCDonly'][ii]                             for ii in range(len(day_type_dicX['combined_ABCDonly']))])
        mouse_recdays_mouse=day_type_dicX['combined_ABCDonly'][mouse_recdays_bool]

        if len(mouse_recdays_mouse)==0:
            continue
        per_mouse_betas=[]
        for mouse_recday in mouse_recdays_mouse:
            state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday]
            neurons_tuned=state_tuning
            per_mouse_betas.append(Phase_spatial_corr_dic[subset][mouse_recday][neurons_tuned])
        per_mouse_betas=np.hstack((per_mouse_betas))
        
        ttest_res=st.ttest_1samp(remove_nan(per_mouse_betas),0)
        Phase_spatial_corr_dic['per_mouse_subsetted'][subset][mouse]=per_mouse_betas
        Phase_spatial_corr_dic['per_mouse_subsetted_mean'][subset][mouse]=np.nanmean(per_mouse_betas)
        Phase_spatial_corr_dic['per_mouse_subsetted_sem'][subset][mouse]=st.sem(per_mouse_betas,nan_policy='omit')
        Phase_spatial_corr_dic['per_mouse_subsetted_ttest'][subset][mouse]=ttest_res


# In[264]:


np.asarray(list(Phase_spatial_corr_dic['per_mouse_subsetted_mean']['mean_bestlag_corr_crossval'].keys()))


# In[265]:


for subset in ['mean_bestlag_corr_crossval','mean_bestlag_corr_nonzerolag_crossval',               'mean_bestlag_corr_nonzerolag_strict_crossval']:
    per_mouse_betas_means=dict_to_array(Phase_spatial_corr_dic['per_mouse_subsetted_mean'][subset])
    per_mouse_betas_sems=dict_to_array(Phase_spatial_corr_dic['per_mouse_subsetted_sem'][subset])
    per_mouse_betas_ttest=dict_to_array(Phase_spatial_corr_dic['per_mouse_subsetted_ttest'][subset])
    Mice=np.asarray(list(Phase_spatial_corr_dic['per_mouse_subsetted_mean'][subset].keys()))
    
    plt.errorbar(per_mouse_betas_means,np.arange(len(per_mouse_betas_means)),xerr=per_mouse_betas_sems,ls='none',
            marker='o',color='grey')
    plt.axvline(0,ls='dashed',color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'SpatialLag_analysis_permouse_'+subset+'.svg',                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print(np.column_stack((Mice,per_mouse_betas_means)))
    print(per_mouse_betas_ttest)
    
    print(st.ttest_1samp(remove_nan(per_mouse_betas_means),0))
    
    per_mouse_betas_means_nonan=remove_nan(per_mouse_betas_means)
    num_positive=len(np.where(per_mouse_betas_means_nonan>0)[0])
    
    print(st.binom_test(x=num_positive, n=len(per_mouse_betas_means_nonan), p=0.5, alternative='greater'))


# In[ ]:





# In[41]:


day_type_dicX['combined_ABCDonly']


# In[23]:


##Plotting best spatial alignment
'''
Examples:
me11_01122021_02122021: 20,22,29,47

ah04_01122021_02122021: 11,22,36,72,90,115

ah04_07122021_08122021: 5,9

me10_09122021_10122021: 27

'''

mouse_recday='ab03_01092023_02092023'
#for mouse_recday in day_type_dicX['combined_ABCDonly']:

num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
num_neurons=len(cluster_dic['good_clus'][mouse_recday])

print(mouse_recday)
sessions=Task_num_dic[mouse_recday]
num_refses=len(np.unique(sessions))
num_comparisons=num_refses-1
repeat_ses=np.where(rank_repeat(sessions)>0)[0]
non_repeat_ses=non_repeat_ses_maker(mouse_recday)  
thr_visit=2
test_comp=0
fontsize=10
plot_maps=True
plot_edge=False 

mean_corrs=np.nanmean(Phase_spatial_corr_dic['corrs_crossval_all'][mouse_recday],axis=2)
sem_corrs=st.sem(Phase_spatial_corr_dic['corrs_crossval_all'][mouse_recday],axis=2,nan_policy='omit')

#field_peak_bins_=Phase_spatial_corr_dic['field_peak_bins'][mouse_recday]
thresholds_=Phase_spatial_corr_dic['Threshold'][mouse_recday]

reference_tasks=Spatial_anchoring_dic['Best_reference_task'][mouse_recday]

for neuron in [28]:
    print(mouse_recday)
    print('neuron'+str(neuron))
    max_value_all=[]
    #print(Tuned_day_[:,neuron])

    #### preferred phase
    phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][0]
    pref_phase_neurons=np.argmax(phase_peaks,axis=1)
    pref_phase=pref_phase_neurons[neuron]

    print(pref_phase)



    print('Lagged Spatial maps')

    states=['A','B','C','D']

    plt.rcParams["figure.figsize"] = (20,5)
    plt.errorbar(-np.arange(12),mean_corrs[neuron],yerr=sem_corrs[neuron])
    plt.axhline(thresholds_[neuron],ls='dashed',color='black')
    #plt.scatter(-field_peak_bins_[neuron],np.repeat(np.max(mean_corrs[neuron])+0.05,len(field_peak_bins_[neuron])),\
    #           marker='*')
    plt.tight_layout()
    plt.tick_params(axis='both',  labelsize=40)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'Example_cells/Lagged_Spatial_correlations_'+                mouse_recday+'_neuron_'+str(neuron)+'.svg', bbox_inches = 'tight', pad_inches = 0)

    plt.show()

    if plot_maps==True:
        node_edge_rate_matrices=Spatial_anchoring_dic['Phase_shifted_node_edge_matrices'][mouse_recday][neuron]
        node_rate_matrices=Spatial_anchoring_dic['Phase_shifted_node_matrices'][mouse_recday][neuron]

        if plot_edge==True:
            matrices_plotted=node_edge_rate_matrices
            gridX=Task_grid_plotting2
        else:
            matrices_plotted=node_rate_matrices
            gridX=Task_grid_plotting

        mouse=mouse_recday[:4]
        rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
        rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
        structure_nums=np.unique(rec_day_structure_numbers)

        structures_all=[]
        for awake_ses_ind_ind, awake_session_ind in enumerate(non_repeat_ses):   
            structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]                [rec_day_session_numbers[awake_session_ind]]

            structures_all.append(structure)
            
            max_value_ses=[]

            fig1, f1_axes = plt.subplots(figsize=(20, 2),ncols=num_lags, constrained_layout=True)
            for lag_ind, lag in enumerate(np.flip(np.arange(num_lags))):

                node_edge_mat_state=matrices_plotted[awake_ses_ind_ind,lag]
                mat_used=node_edge_mat_state
                
                max_FR=int(np.nanmax(mat_used*40))
                if max_FR==0:
                    max_FR=round(np.nanmax(mat_used*40),1)
                max_value_ses.append(max_FR)

                
                ax=f1_axes[lag_ind]
                ax.matshow(mat_used, cmap='coolwarm') #vmin=min_rate, vmax=max_rate
                for state_port_ind, state_port in enumerate(states):
                    node=structure[state_port_ind]-1
                    ax.text(gridX[node,0]-0.25, gridX[node,1]+0.25,                            state_port.lower(), fontsize=22.5)
                    
                ax.text(gridX[-1,0], gridX[0,1]-0.5,                            max_FR, fontsize=15)

                
                
                ax.axis('off')

                #ax.savefig(str(neuron)+state+str(awake_session_ind)+'discmap.svg')
            max_value_all.append(max_value_ses)
            plt.savefig(Ephys_output_folder_dropbox+'Example_cells/Lagged_Spatial_maps_'+                    mouse_recday+'_neuron_'+str(neuron)+'_session'+str(awake_ses_ind_ind)+'.svg',                        bbox_inches = 'tight', pad_inches = 0)
        plt.axis('off')

        plt.show()
    print(structures_all)

plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


# In[24]:


max_value_all#[0][0]/40


# In[68]:


mat=np.zeros((3,3))
plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False

plt.matshow(mat,cmap='coolwarm')
plt.colorbar()
plt.savefig(Ephys_output_folder_dropbox+'coolwarm_colorbar.svg')


# In[ ]:





# In[43]:


##Where are the spatial peaks of the anchors?
All_days_used=np.hstack((day_type_dicX['combined_ABCDonly'],day_type_dicX['3_task']))
day_type_ses_dic={'combined_ABCDonly':[0,1,2,4,5,6],'3_task':[0,1,2]}
num_locations=21
for mouse_recday in day_type_dicX['combined_ABCDonly']:

    print(mouse_recday)
    
    if mouse_recday in day_type_dicX['combined_ABCDonly']:
        day_type='combined_ABCDonly'
    elif mouse_recday in day_type_dicX['3_task']:
        day_type='3_task'
    print(mouse_recday)
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])

    Phase_shifted_maps_=Spatial_anchoring_dic['Phase_shifted_maps'][mouse_recday]
    field_peak_bins_=Phase_spatial_corr_dic['field_peak_bins'][mouse_recday]
    shuff_maps=Spatial_anchoring_dic['Shuffled_maps'][mouse_recday]
    
    sign_peak_all=[]
    sign_peak_all_repeated=[]
    for neuron in np.arange(num_neurons):
        sign_peak_neuron=[]
        sign_peak_neuron_repeated=[]
        for phase_anchor in field_peak_bins_[neuron]:
            sign_peak_anchor=[]
            for ses in np.arange(num_sessions):
                anchor_fields=Phase_shifted_maps_[neuron,ses,:]
                thr_ses=np.percentile(shuff_maps[neuron,ses],95,axis=0)
                sign_peak=np.where(anchor_fields[phase_anchor]>thr_ses)[0]
                sign_peak_anchor.append(sign_peak)
            
            num_peaks_spatial_all_=list(np.concatenate(sign_peak_anchor))
            most_common_peaks=most_common(num_peaks_spatial_all_)[0]
            
            my_dict = {i:num_peaks_spatial_all_.count(i) for i in num_peaks_spatial_all_}
            repeated_peaks=np.asarray(list(my_dict.keys()))[np.asarray(list(my_dict.values()))>2]
            
            
            sign_peak_neuron.append(most_common_peaks)
            sign_peak_neuron_repeated.append(repeated_peaks)
            
        sign_peak_all.append(sign_peak_neuron)
        sign_peak_all_repeated.append(sign_peak_neuron_repeated)
    Spatial_anchoring_dic['anchor_spatial_location'][mouse_recday]=sign_peak_all
    Spatial_anchoring_dic['anchor_num_spatial_peaks'][mouse_recday]=    np.asarray([np.asarray([len(sign_peak_all[ii][jj]) for jj in np.arange(len(sign_peak_all[ii]))])                for ii in range(num_neurons)])
    
    Spatial_anchoring_dic['anchor_spatial_location_repeated'][mouse_recday]=sign_peak_all_repeated
    Spatial_anchoring_dic['anchor_num_spatial_peaks_repeated'][mouse_recday]=    np.asarray([np.asarray([len(sign_peak_all_repeated[ii][jj]) for jj in np.arange(len(sign_peak_all_repeated[ii]))])                for ii in range(num_neurons)])
    


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





# In[44]:


###threshold crossings

use_ttest=True
use_crossval=True
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    print(mouse_recday)
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    all_corrs=Phase_spatial_corr_dic['corrs_all'][mouse_recday]
    all_corrs_crossval=Phase_spatial_corr_dic['corrs_crossval_all'][mouse_recday]
    mean_corrs=np.nanmean(all_corrs,axis=2)
    mean_corrs_crossval=np.nanmean(all_corrs_crossval,axis=2)
    thresholds_=Phase_spatial_corr_dic['Threshold'][mouse_recday]
    
    
    all_ps=np.asarray([[st.ttest_1samp(all_corrs[neuron][ii],thresholds_[neuron])[1]                        for ii in range(len(all_corrs[neuron]))] for neuron in np.arange(num_neurons)])
    
    all_ps_crossval=np.asarray([[st.ttest_1samp(all_corrs_crossval[neuron][ii],thresholds_[neuron])[1]                        for ii in range(len(all_corrs_crossval[neuron]))] for neuron in np.arange(num_neurons)])
    
    all_ps_crossval_corrected=    np.asarray([[statsmodels.stats.multitest.multipletests(                        st.ttest_1samp(all_corrs_crossval[neuron][ii],0)[1],alpha=0.05,                                                          method='bonferroni')[1][0]
                        for ii in range(len(all_corrs_crossval[neuron]))] for neuron in np.arange(num_neurons)])
    

    
    if use_ttest==True:
        if use_crossval==True:
            thr_crossings=np.asarray([np.where(np.logical_and(all_ps_crossval_corrected[neuron]<0.05,                                                          mean_corrs_crossval[neuron]>0))[0]                                  for neuron in range(num_neurons)])
        else:
            thr_crossings=np.asarray([np.where(np.logical_and(all_ps[ii]<0.05,mean_corrs[ii]>thresholds_[ii]))[0]                                  for ii in range(num_neurons)])
    else:
        thr_crossings=np.asarray([np.where(mean_corrs[ii]>thresholds_[ii])[0] for ii in range(len(mean_corrs))])
        
    fields=np.asarray([continguous_field(thr_crossings[ii],12,cont_thr=1) for ii in range(len(thr_crossings))])
    ##i.e. counting a contiguous set of crossings as one "field"
    
    field_peak_bins_all=[]
    for neuron in np.arange(num_neurons):
        mean_corrs_neuron=mean_corrs[neuron]
        thr_crossings_neuron=thr_crossings[neuron]
        
        if len(thr_crossings_neuron)==0:
            field_peak_bins_all.append(np.asarray([]))
            continue
            
        fields_neuron=fields[neuron]
        
        max_corrs_fields=[np.max(mean_corrs_neuron[thr_crossings_neuron][fields_neuron==ii])                          for ii in np.unique(fields_neuron)]
        field_peak_bins=np.asarray([np.where(mean_corrs_neuron==max_corrs_fields[ii])[0][0]                                    for ii in range(len(max_corrs_fields))])
        field_peak_bins_all.append(field_peak_bins)
    field_peak_bins_all=np.asarray(field_peak_bins_all)
    ##peak bin in each field

    Phase_spatial_corr_dic['Threshold_Crossings'][mouse_recday]=thr_crossings
    Phase_spatial_corr_dic['Threshold_Crossings_fields'][mouse_recday]=fields
    Phase_spatial_corr_dic['field_peak_bins'][mouse_recday]=field_peak_bins_all

    
    mean_corrs_shuff=np.nanmean(Phase_spatial_corr_dic['corrs_all_shuff'][mouse_recday],axis=3)
    all_corrs_shuff=Phase_spatial_corr_dic['corrs_all_shuff'][mouse_recday]
    for iteration in np.arange(100):
        mean_corrs_shuff_iteration=mean_corrs_shuff[:,iteration]
        #thr_crossings_shuff=np.asarray([np.where(mean_corrs_shuff_iteration[ii]>thresholds_[ii])[0]\
        #                                for ii in range(len(mean_corrs_shuff_iteration))])
        
        
        all_corrs_shuff_iteration=all_corrs_shuff[:,iteration]
        all_ps_iteration=np.asarray([[st.ttest_1samp(all_corrs_shuff_iteration[neuron][ii],thresholds_[neuron])[1]                            for ii in range(len(all_corrs_shuff_iteration[neuron]))] for neuron in np.arange(num_neurons)])
        
        if use_ttest==True:
            thr_crossings_shuff=np.asarray([np.where(np.logical_and(all_ps_iteration[ii]<0.05,                                                                    mean_corrs_shuff_iteration[ii]>thresholds_[ii]))[0]                                            for ii in range(num_neurons)])
        else:
            thr_crossings_shuff=np.asarray([np.where(mean_corrs_shuff_iteration[ii]>thresholds_[ii])[0]                                            for ii in range(num_neurons)])
            
        Phase_spatial_corr_dic['Threshold_Crossings_shuff'][iteration][mouse_recday]=thr_crossings_shuff
        
Spatial_anchoring_dic['field_peak_bins']=Phase_spatial_corr_dic['field_peak_bins']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


Phase_spatial_corr_dic['Max_corr_bin_crossval'].keys()


# In[46]:


Phase_spatial_corr_dic['Threshold_Crossings']['me11_05122021_06122021']


# In[ ]:





# In[47]:


Edge_grid


# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:


bins=np.arange(13)

##Max peaks
if use_crossval==True:
    max_peak=np.concatenate(dict_to_array(Phase_spatial_corr_dic['Max_corr_bin_crossval']))
else:
    max_peak=np.concatenate(dict_to_array(Phase_spatial_corr_dic['Max_corr_bin']))
    


##All peaks
xx_perneuron=xx_perneuron_real=concatenate_complex2(dict_to_array(Phase_spatial_corr_dic['Threshold_Crossings']))
num_anchors=np.asarray([len(xx_perneuron[ii]) for ii in range(len(xx_perneuron))])
xx=xx_real=np.hstack((xx_perneuron))




plt.hist(xx,bins=bins,color='black',edgecolor='white')
plt.savefig(Ephys_output_folder_dropbox+'/proportion_anchor_taskbins.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(np.histogram(xx,bins=bins)[0])
print(np.sum(np.histogram(xx,bins=bins)[0][0]))
print(np.sum(np.histogram(xx,bins=bins)[0][1:]))

anchored=anchored_real=(np.asarray([len(xx_perneuron[ii])>0                                    for ii in np.arange(len(xx_perneuron))])).astype(int)
non_zero_anchored=non_zero_anchored_real=(np.asarray([np.logical_and(max_peak[ii] in xx_perneuron[ii],max_peak[ii]!=0)            for ii in np.arange(len(xx_perneuron))])).astype(int)

#non_zero_anchored=non_zero_anchored_real=(np.asarray([np.logical_and(len(xx_perneuron[ii])>0,0 not in xx_perneuron[ii])\
#            for ii in np.arange(len(xx_perneuron))])).astype(int)


prop_nonzero_real=np.sum(non_zero_anchored)/len(non_zero_anchored)
prop_nonzero_anchored_real=np.sum(non_zero_anchored)/np.sum(anchored)

print('Proportion anchored of all neurons = '+str(len(np.where(xx!=0)[0])/len(xx)))
print('Proportion non zero anchored of all neurons = '+str(prop_nonzero_real))
print('Proportion non zero anchored of all anchored neurons = '+str(prop_nonzero_anchored_real))
print('')

xx_shuff_all=[]
for iteration in np.arange(100):
    xx_perneuron_shuff=concatenate_complex2(dict_to_array(Phase_spatial_corr_dic['Threshold_Crossings_shuff'][iteration]))
    num_anchors=np.asarray([len(xx_perneuron[ii]) for ii in range(len(xx_perneuron_shuff))])
    xx_shuff=np.hstack((xx_perneuron_shuff))
    xx_shuff_all.append(xx_shuff)
#xx_shuff_all=np.concatenate(xx_shuff_all)

xx_shuff_prop=np.mean([np.histogram(xx_shuff_all[ii])[0] for ii in range(len(xx_shuff_all))],axis=0)
#plt.bar(np.arange(len(xx_shuff_prop))+0.5,xx_shuff_prop,alpha=0.5,color='grey')
plt.show()


print('number of anchoring threshold crossings')
print(np.sum(np.histogram(num_anchors,bins=bins)[0][0]))
print(np.sum(np.histogram(num_anchors,bins=bins)[0][1:]))
plt.hist(num_anchors,bins=bins, color='black',edgecolor='white')
plt.savefig(Ephys_output_folder_dropbox+'/Number_anchor_crossings.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(bins)

print('number of continguous anchor peaks')
fields=np.asarray([continguous_field(xx_perneuron[ii],12,cont_thr=1) for ii in range(len(xx_perneuron))])
num_fields=np.asarray([np.nanmax(fields[ii])+1 for ii in range(len(fields))])
num_fields[np.isnan(num_fields)]=0

plt.hist(num_fields,bins=bins, color='black',edgecolor='white')
plt.savefig(Ephys_output_folder_dropbox+'/Number_anchor_peaks.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()

#####

print('Number of spatial peaks')
#anchor_num_spatial_peaks_=np.concatenate(np.hstack((dict_to_array(Spatial_anchoring_dic['anchor_num_spatial_peaks']))))
#plt.hist(anchor_num_spatial_peaks_,bins=np.arange(20)+1, color='black',edgecolor='white')
#plt.savefig(Ephys_output_folder_dropbox+'/Number_spatial_peaks.svg', bbox_inches = 'tight', pad_inches = 0)
#plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


###Null distributions

##All peaks

bins=np.arange(13)
hist_all=[]
prop_all=[]
for iteration in np.arange(100):
    xx_perneuron=concatenate_complex2(dict_to_array(Phase_spatial_corr_dic['Threshold_Crossings_shuff'][iteration]))
    num_anchors=np.asarray([len(xx_perneuron[ii]) for ii in range(len(xx_perneuron))])
    xx=np.hstack((xx_perneuron))
    #print(len(np.where(xx==0)[0])/len(xx))
    #plt.hist(xx,bins=bins)
    #plt.show()
    hist_all.append(np.histogram(xx,bins=bins)[0])
    
    non_zero_anchored=(np.asarray([np.logical_and(len(xx_perneuron[ii])>0,np.mean(xx_perneuron[ii])>0)            for ii in np.arange(len(xx_perneuron))])).astype(int)
    
    non_zero_anchored=(np.asarray([np.logical_and(len(xx_perneuron[ii])>0,0 not in xx_perneuron[ii])            for ii in np.arange(len(xx_perneuron))])).astype(int)
    prop_all.append(np.sum(non_zero_anchored)/len(non_zero_anchored))
    
    
hist_mean=np.mean(np.asarray(hist_all),axis=0)

print(np.histogram(xx_real,bins=bins)[0])
print(hist_mean)


prop_nonzero_shuff=np.mean(prop_all)
print(prop_nonzero_shuff)


xx_shuff_all=[]
for iteration in np.arange(100):
    xx_perneuron_shuff=concatenate_complex2(dict_to_array(Phase_spatial_corr_dic['Threshold_Crossings_shuff'][iteration]))
    num_anchors=np.asarray([len(xx_perneuron[ii]) for ii in range(len(xx_perneuron_shuff))])
    xx_shuff=np.hstack((xx_perneuron_shuff))
    xx_shuff_all.append(xx_shuff)
#xx_shuff_all=np.concatenate(xx_shuff_all)

xx_shuff_prop=np.mean([np.histogram(xx_shuff_all[ii])[0] for ii in range(len(xx_shuff_all))],axis=0)
#plt.bar(np.arange(len(xx_shuff_prop))+0.5,xx_shuff_prop,alpha=0.5,color='grey')
plt.show()


print('number of anchoring threshold crossings')
bins=np.arange(13)
print(np.sum(np.histogram(num_anchors,bins=bins)[0][0]))
print(np.sum(np.histogram(num_anchors,bins=bins)[0][1:]))
plt.hist(num_anchors,bins=bins)
plt.show()
print(bins)

print('number of continguous anchor peaks')
fields=np.asarray([continguous_field(xx_perneuron[ii],12,cont_thr=1) for ii in range(len(xx_perneuron))])
num_fields=np.asarray([np.nanmax(fields[ii])+1 for ii in range(len(fields))])
num_fields[np.isnan(num_fields)]=0

plt.hist(num_fields,bins=np.arange(6))


# In[ ]:





# In[ ]:





# In[50]:


'''
-clarify proportions of neurons vs shuffle - done
-anchoring generalization analysis with multi-anchor
-add drop-out to analyses
-Additional predictions
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


##Example anchors
mouse_recday = 'me11_01122021_02122021'
num_neurons=len(cluster_dic['good_clus'][mouse_recday])
mean_corrs=np.nanmean(Phase_spatial_corr_dic['corrs_crossval_all'][mouse_recday],axis=2)
std_corrs=np.nanstd(Phase_spatial_corr_dic['corrs_crossval_all'][mouse_recday],axis=2)

field_peak_bins_=Phase_spatial_corr_dic['field_peak_bins'][mouse_recday]
thresholds_=Phase_spatial_corr_dic['Threshold'][mouse_recday]

for neuron in range(num_neurons):
    print(neuron)
    plt.plot(-np.arange(12),mean_corrs[neuron])
    plt.axhline(thresholds_[neuron],ls='dashed',color='black')
    plt.scatter(-field_peak_bins_[neuron],np.repeat(np.max(mean_corrs[neuron])+0.05,len(field_peak_bins_[neuron])),               marker='*')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


### Sanity checks -GLM vs shifted correlations


# In[53]:


GLMvsAnchoring_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    spatial_neurons=np.where(Tuned_dic['Place'][mouse_recday]==True)[0]
    nonspatial_neurons=np.where(Tuned_dic['Place'][mouse_recday]==False)[0]
    
    max_peak=Phase_spatial_corr_dic['Max_corr_bin_crossval'][mouse_recday]
    
    Threshold_crossings=Phase_spatial_corr_dic['Threshold_Crossings'][mouse_recday]
    zero_bool=np.zeros(num_neurons)
    num_crossings_arr=np.zeros(num_neurons)
    zero_bool[:]=np.nan
    num_crossings_arr[:]=np.nan
    for neuron in np.arange(num_neurons):
        threshold_crossings_neuron=Threshold_crossings[neuron]
        zero_bool[neuron]=0 in threshold_crossings_neuron
        num_crossings_arr[neuron]=len(threshold_crossings_neuron)

    GLMvsAnchoring_dic['zero_bool'][mouse_recday]=zero_bool
    GLMvsAnchoring_dic['Num_crossings'][mouse_recday]=num_crossings_arr
    GLMvsAnchoring_dic['GLM_spatial'][mouse_recday]=(Tuned_dic['Place'][mouse_recday]).astype(int)
    GLMvsAnchoring_dic['Max_bin'][mouse_recday]=max_peak==0
    

    


# In[54]:


GLM_spatial_all=np.hstack((dict_to_array(GLMvsAnchoring_dic['GLM_spatial'])))
zero_bool_all=np.hstack((dict_to_array(GLMvsAnchoring_dic['zero_bool'])))
Num_crossings_all=np.hstack((dict_to_array(GLMvsAnchoring_dic['Num_crossings'])))
Max_peak_all=np.hstack((dict_to_array(GLMvsAnchoring_dic['Max_bin'])))


print('Proportion of neurons with one peak at zero lag')
spatial_zero_prop=np.sum(zero_bool_all[GLM_spatial_all==1])/len(np.where(GLM_spatial_all==1)[0])
non_spatial_zero_prop=np.sum(zero_bool_all[GLM_spatial_all==0])/len(np.where(GLM_spatial_all==0)[0])
print(spatial_zero_prop)
print(non_spatial_zero_prop)
print('')

print('Proportion of neurons with main peak at zero lag ')
spatial_zero_prop=np.sum(Max_peak_all[GLM_spatial_all==1])/len(np.where(GLM_spatial_all==1)[0])
non_spatial_zero_prop=np.sum(Max_peak_all[GLM_spatial_all==0])/len(np.where(GLM_spatial_all==0)[0])
print(spatial_zero_prop)
print(non_spatial_zero_prop)
print('')

#print('Mean number of crossings when anchored')
#spatial_numcrossings=Num_crossings_all[np.logical_and(GLM_spatial_all==1,Num_crossings_all>0)]
#non_spatial_numcrossings=Num_crossings_all[np.logical_and(GLM_spatial_all==0, Num_crossings_all>0)]
#print(np.mean(spatial_numcrossings))
#print(np.mean(non_spatial_numcrossings))
#print('')

print('Proportion of neurons with peak at zero lag when there is only one significant peak')
spatial_zero_onecrossing_prop=np.sum(zero_bool_all[np.logical_and(GLM_spatial_all==1,Num_crossings_all==1)])/len(np.where(np.logical_and(GLM_spatial_all==1,Num_crossings_all==1))[0])
non_spatial_zero_onecrossing_prop=np.sum(zero_bool_all[np.logical_and(GLM_spatial_all==0,Num_crossings_all==1)])/len(np.where(np.logical_and(GLM_spatial_all==0,Num_crossings_all==1))[0])
print(spatial_zero_onecrossing_prop)
print(non_spatial_zero_onecrossing_prop)


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





# In[9]:


Spatial_tuning_overlystrict_dic=rec_dd()


# In[ ]:





# In[116]:


###spatial correlations

Spatial_tuning_overlystrict_dic=rec_dd()
day_type='combined_ABCDonly'

for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    num_neurons=len(cluster_dic['good_clus'][mouse_recday]) 
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    sessions=Task_num_dic[mouse_recday]
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)  ###this defines the sessions used

    corrs_stat=np.zeros((num_neurons,2))
    for neuron in np.arange(num_neurons):
        #print(neuron)

        spatial_arrays=np.zeros((len(non_repeat_ses),21))
        node_arrays=np.zeros((len(non_repeat_ses),9))
        
        for awake_session_ind, awake_session in enumerate(non_repeat_ses):
            if neuron==0:
                print(awake_session_ind)
            node_rate_mat=node_rate_matrices_dic['All_states'][awake_session][mouse_recday][neuron]
            edge_rate_mat=edge_rate_matrices_dic['All_states'][awake_session][mouse_recday][neuron]
            
            if len(edge_rate_mat)==0:
                if neuron==0:
                    print('Not analysed')
                continue

            node_edge_mat=edge_node_fill(edge_rate_mat,node_rate_mat)
            node_edge_array=np.hstack(node_edge_mat)
            node_edge_array=node_edge_array[~np.isnan(node_edge_array)]
            spatial_arrays[awake_session_ind]=node_edge_array

            node_arrays[awake_session_ind]=np.hstack(node_rate_mat)


        corr_mat=np.corrcoef(spatial_arrays)
        corrs=corr_mat[np.triu_indices(len(non_repeat_ses), k = 1)]
        corrs_p=st.ttest_1samp(corrs,0)[1]

        corrs_stat[neuron]=np.mean(corrs),corrs_p
        

    spatial_bool=np.logical_and(corrs_stat[:,0]>0,corrs_stat[:,1]<0.05)

    Spatial_tuning_overlystrict_dic[mouse_recday]=spatial_bool
        
    
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


non_repeat_ses


# In[89]:


num_refses


# In[ ]:





# In[45]:


###Anchoring in left out session - single anchor 

'''
The logic here is as follows:
-train-test splits
-for each possible test session find the best anchor by searching each possible reference session and finding the
reference session that gives the anchor with the lowest mean distance across all training sessions (i.e. sessions that
arent reference or test sessions)
-this gives a single anchor per training-test split per neuron which is calculated exclusively from training sessions - 
hence no double-dipping

to see that this isnt biased - set shuffle_angles to True - can see uniform distribution of angles
'''


thr_tuning_sessions=0
coh_thr=1-math.cos(np.deg2rad(45))
angles_spatialanchor_num_allrefs=[]

thr_anchored=0.5

day_type='combined_ABCDonly'

#filter_untuned=False
remove_spatial=False
spatial_tuning_GLM=False
spatial_tuning_strict=False
keep_singlepeaked=False
remove_multipeaked=False
remove_nonpeaked=False
phase_random=False
shuffle_angles=False

delete_old=False


for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    
    if day_type=='combined_ABCDonly':
        num_states=4
        
    elif day_type=='combined_ABCDE':
        num_states=5
    
    coh_thr1=(360/num_states)//2
    coh_thr2=360-coh_thr1
    
    num_bins=num_states*90
    angle_correction=360/num_bins ### converts bin value into angles (used when number of states is not 4)
    num_trials=dict_to_array(Num_trials_dic2[mouse_recday])
    trials_completed_ses=np.where(num_trials>2)[0]
    
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    num_neurons=len(cluster_dic['good_clus'][mouse_recday]) 
    
    print(num_sessions)
    abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
    sessions=Task_num_dic[mouse_recday]
    
    
    if day_type=='combined_ABCDE':
        
        sessions=sessions[abstract_structures=='ABCDE']
    
    #num_refses=len(np.unique(sessions))
    
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)  ###this defines the sessions used
    ###only the first session from each task is used
    
    
    if day_type=='combined_ABCDE':
        ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
        num_ABCD_ses=len(ABCD_sessions)
        non_repeat_ses=np.setdiff1d(non_repeat_ses,ABCD_sessions)-len(ABCD_sessions)
    
    non_repeat_ses=np.intersect1d(non_repeat_ses,trials_completed_ses)
    
    num_comparisons=len(non_repeat_ses)-1
    num_refses=len(non_repeat_ses)
    
    #dists_=np.copy(Spatial_anchoring_dic['rotation_dist_mat'][mouse_recday]) 
    angles_=np.copy(Spatial_anchoring_dic['rotation_angle_mat'][mouse_recday])*angle_correction
    
    
    ###cross-validating against lagged spatial map analysis - check
    #anchor_spatial_bin=Spatial_anchoring_dic['anchor_spatial_location'][mouse_recday] 
    #Anchor_spatialbin=np.asarray([anchor_spatial_bin[ii][0][0] if len(anchor_spatial_bin[ii])>0 else np.nan 
    #    for ii in range(len(anchor_spatial_bin))]) 
    #anchor_first=np.asarray([anchor_spatial_bin[ii][0][0] if len(anchor_spatial_bin[ii])>0 else np.nan 
    #    for ii in range(len(anchor_spatial_bin))]) 
    #anchor_first[anchor_first>8]=np.nan 
    #anchor_task_bin_=Phase_spatial_corr_dic['field_peak_bins'][mouse_recday] 
    #anchor_task_bin=np.asarray([anchor_task_bin_[ii][0] if len(anchor_task_bin_[ii])>0 
    #                            else 0 for ii in range(len(anchor_task_bin_))]) 
    
    dists_all=np.zeros((num_refses,num_refses,num_neurons)) 
    angles_all=np.zeros((num_refses,num_refses,num_neurons)) 
    Best_anchor_all=np.zeros((num_refses,num_refses,num_neurons,2)) 
    Mins_all=np.zeros((num_refses,num_refses,num_neurons)) 
    
    dists_all[:]=np.nan
    angles_all[:]=np.nan
    Best_anchor_all[:]=np.nan
    Mins_all[:]=np.nan

    if shuffle_angles == True:        
        angles_shuff=np.copy(angles_)*angle_correction
        angles_shuff=(angles_shuff+numpy.random.random_integers(1, num_states, np.shape(angles_shuff))*90)%num_bins
        ###i.e. adding random number thats a multiple of 90 degrees - to keep phase tuning 
        anglesX=angles_shuff
        
        
        
        dists_shuff=1-np.cos(np.deg2rad(angles_shuff))
        distsX=dists_shuff
    else:
        anglesX=angles_
        distsX=1-np.cos(np.deg2rad(angles_))

    for ses_reference_ind, ses_reference in enumerate(non_repeat_ses):   
        non_repeat_ses_noref=np.sort(np.setdiff1d(non_repeat_ses,ses_reference)) 
        ###sessions used that arent the reference session

        ###cross-validating against lagged spatial map analysis - check
        #tuning_phase_boolean_max_all=tuning_singletrial_dic['tuning_phase_boolean_max'][mouse_recday][ses_reference_ind] 
        #max_phase=np.argmax(tuning_phase_boolean_max_all,axis=1) 
        #phase_first=(max_phase+((anchor_task_bin)%3))%3 

        dists_day=distsX[:,:,:,ses_reference,non_repeat_ses_noref] 
        angles_day=anglesX[:,:,:,ses_reference,non_repeat_ses_noref] 
        num_neurons=len(cluster_dic['good_clus'][mouse_recday]) 
        
        ###i.e. for each test-train split will get one best anchor 
        for Test_comp in np.arange(num_comparisons): 
            Test_ind=np.setdiff1d(np.arange(num_refses),ses_reference_ind)[Test_comp] ##used for indexing arrays
            
            Training_comp=np.sort(np.setdiff1d(np.arange(num_comparisons),Test_comp)) 
            Training=dists_day[:,:,:,Training_comp] 
            Test=dists_day[:,:,:,Test_comp] 
            Test_angles=angles_day[:,:,:,Test_comp] 

            mean_Training=np.nanmean(Training,axis=3)
            ##mean distance for the training sessions in this test-train split
            
            max_Training=np.nanmax(Training,axis=3)
            ##max distance for the training sessions in this test-train split
            
            XXX_training=mean_Training

            Best_anchor_=np.asarray([[np.where(x == np.nanmin(x))[0][0],np.where(x == np.nanmin(x))[1][0]] 
                        if ~np.isnan(np.nanmean(x)) else [np.nan,np.nan] for neuron,x in enumerate(XXX_training)])
            
            Best_anchor_randphase=np.copy(Best_anchor_)
            Best_anchor_randphase[:,0]=np.asarray(Best_anchor_[:,0]+random.choices(np.arange(2)+1,k=len(Best_anchor_)))%3
            
            if phase_random==True:
                Best_anchor=Best_anchor_randphase
            else:
                Best_anchor=Best_anchor_
                
            
            Mins=np.asarray([np.nanmin(x) for neuron,x in enumerate(XXX_training)])
            ###smallest mean distance (i.e. distance belonging to best anchor) 
            ##for the training sessions in this test-train split 

            
            ##this is where the distances/angles for the test session relative to the reference session is calculated
            dists=np.asarray([Test[neuron][int(Best_anchor[neuron][0]),int(Best_anchor[neuron][1])] 
                              if ~np.isnan(np.nanmean(x)) else np.nan for neuron,x in enumerate(XXX_training)]) 
            angles=np.asarray([Test_angles[neuron][int(Best_anchor[neuron][0]),int(Best_anchor[neuron][1])] 
                               if ~np.isnan(np.nanmean(x)) else np.nan for neuron,x in enumerate(XXX_training)]) 

            
            dists_all[ses_reference_ind,Test_ind]=dists 
            angles_all[ses_reference_ind,Test_ind]=angles 
            Best_anchor_all[ses_reference_ind,Test_ind]=Best_anchor 
            Mins_all[ses_reference_ind,Test_ind]=Mins
            ## for each neuron, a row is a reference session and a column is a training-test split (indexed
            ##by the test session), note that the number reported is the mean distance for best anchor over only 
            ##the training sessions (see Mins)
            ##so e.g. row 0 column 1 is the mean distance over the training sessions training-test split where test is 
            #session 1 (second session) and reference session is session 0 (first session)


    if np.isnan(np.nanmean(dists_all))==True:
        continue
    min_ses=np.nanargmin(Mins_all,axis=0)
    ### this is the reference session that has the best training distances for each test-train split and each neuron
    
    dists_test=np.asarray([[dists_all[min_ses[ses,neuron],ses,neuron]                        for ses in range(len(dists_all[:,:,neuron].T))]                       for neuron in range(num_neurons)]).T

    angles_test=np.asarray([[angles_all[min_ses[ses,neuron],ses,neuron]                            for ses in range(len(dists_all[:,:,neuron].T))]                           for neuron in range(num_neurons)]).T
    
    Best_anchor_test=np.asarray([[Best_anchor_all[min_ses[ses,neuron],ses,neuron]                            for ses in range(len(dists_all[:,:,neuron].T))]                           for neuron in range(num_neurons)]).T
    
    ##tuned sessions
    if day_type=='combined_ABCDonly':
        sum_statepeaks=np.sum(tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday],axis=2).T
    elif day_type=='combined_ABCDE':
        sum_statepeaks=np.sum(np.load(Intermediate_object_folder_dropbox+                                      'ABCDE_tuning_state_boolean_'+mouse_recday+'.npy'),axis=2).T
    sum_statepeaks=sum_statepeaks[:,non_repeat_ses]
    state_bool_=sum_statepeaks>0
    state_bool_
    
    ##finding most common  anchor for each neuron - Note dont use this when needing cross-validation
    
    
    #most_common_anchor=\
    #np.vstack(([most_common_pair(Best_anchor_test[:,:,ii].T)[0] for ii in range(num_neurons)]))
    
    #freq_most_common_anchor=\
    #np.hstack(([most_common_pair(Best_anchor_test[:,:,ii].T)[1] for ii in range(num_neurons)]))
    #Anchored_bool1=freq_most_common_anchor>=thr_anchored
    #Anchored_bool_always=freq_most_common_anchor==1
    
    #Min_dist=np.vstack(([[Mins_all[min_ses[ii,neuron],ii,neuron] for ii in range(len(min_ses[:,neuron]))]\
    #        for neuron in np.arange(num_neurons)]))
    #max_mindist=np.max(Min_dist,axis=1)
    
    #Anchored_bool2=max_mindist<coh_thr
    
    #Anchored_bool=np.logical_and(Anchored_bool1,Anchored_bool2)
    #Anchored_bool_strict=np.logical_and(Anchored_bool_always,Anchored_bool2)
    
    
    most_common_anchor=    np.vstack(([most_common_pair(Best_anchor_test[:,:,neuron].T)[0] for neuron in range(num_neurons)]))

    most_common_anchor_bool=np.vstack(([[np.all(Best_anchor_test[:,ses,neuron]==most_common_anchor[neuron])                 for ses in range(len(min_ses[:,neuron]))] for neuron in np.arange(num_neurons)]))

    freq_most_common_anchor=    np.hstack(([most_common_pair(Best_anchor_test[:,:,ii].T)[1] for ii in range(num_neurons)]))
    Anchored_bool_half=freq_most_common_anchor>=thr_anchored
    Anchored_bool_always=freq_most_common_anchor==1

    Min_dist=np.vstack(([[Mins_all[min_ses[ses,neuron],ses,neuron] for ses in range(len(min_ses[:,neuron]))]                         for neuron in np.arange(num_neurons)]))
    max_mindist=np.nanmax(Min_dist,axis=1)

    max_mindist_most_common_anchor=np.asarray([np.nanmax(Min_dist[neuron][most_common_anchor_bool[neuron]]) if     len(Min_dist[neuron][most_common_anchor_bool[neuron]])>0 else np.nan for neuron in range(num_neurons)])

    Anchored_bool_coherent_most_common_anchor=max_mindist_most_common_anchor<coh_thr

    Anchored_bool_coherent=max_mindist<coh_thr

    Anchored_bool=np.logical_and(Anchored_bool_half,Anchored_bool_coherent_most_common_anchor)
    Anchored_bool_strict=np.logical_and(Anchored_bool_always,Anchored_bool_coherent)
    
    
    #####
    most_common_anchor_tuned=np.vstack(([most_common_pair(Best_anchor_test[:,state_bool_[neuron],neuron].T)[0]                          if len(Best_anchor_test[:,state_bool_[neuron],neuron].T)>1 else [np.nan,np.nan]                          for neuron in range(num_neurons)]))

    most_common_anchor_bool_tuned_allses=np.vstack(([[np.all(Best_anchor_test[:,ses,neuron]                                                      ==most_common_anchor_tuned[neuron])                 for ses in range(len(min_ses[:,neuron]))] if np.isnan(most_common_anchor_tuned[neuron,0])==False                                              else np.repeat(False, len(min_ses[:,neuron]))
                                              for neuron in np.arange(num_neurons)]))

    most_common_anchor_bool_tuned=[[np.all(Best_anchor_test[:,ses,neuron]==most_common_anchor_tuned[neuron])                 for ses in np.where(state_bool_[neuron]==True)[0]] if np.isnan(most_common_anchor_tuned[neuron,0])==False                                              else [] for neuron in np.arange(num_neurons)]

    most_common_anchor_bool_tuned_ses=[np.where(state_bool_[neuron]==True)[0] if                                        np.isnan(most_common_anchor_tuned[neuron,0])==False else []                                        for neuron in np.arange(num_neurons)]

    most_common_anchor_bool_tuned_ses_common=[np.where(np.logical_and(state_bool_[neuron]==True,                                                                most_common_anchor_bool_tuned_allses[neuron]==True))[0] if                                        np.isnan(most_common_anchor_tuned[neuron,0])==False else []                                        for neuron in np.arange(num_neurons)]

    Min_dist_tuned=[[Mins_all[min_ses[ses,neuron],ses,neuron] for ses in                      most_common_anchor_bool_tuned_ses_common[neuron]] if                     len(most_common_anchor_bool_tuned_ses_common[neuron])>1 else [] for neuron in np.arange(num_neurons)]

    max_mindist_tuned=np.asarray([np.max(Min_dist_tuned[neuron]) if len(Min_dist_tuned[neuron])>1 else np.nan                      for neuron in np.arange(num_neurons)])
    
    proportion_mostcommon_tuned=np.asarray([np.sum(most_common_anchor_bool_tuned[neuron])                                      /len(most_common_anchor_bool_tuned[neuron])
    if len(most_common_anchor_bool_tuned[neuron])>1 else np.nan for neuron in range(num_neurons)])

    Anchored_bool_tuned_half=proportion_mostcommon_tuned>thr_anchored
    Anchored_bool_tuned_always=proportion_mostcommon_tuned==1
    Anchored_bool_coherent_tuned=max_mindist_tuned<coh_thr
    
    Anchored_bool_tuned=np.logical_and(Anchored_bool_tuned_half,Anchored_bool_coherent_tuned)
    Anchored_bool_tuned_strict=np.logical_and(Anchored_bool_tuned_always,Anchored_bool_coherent_tuned)
    
    ###cross-validated anchored booleans   
    most_common_anchor_crossval=    np.vstack(([np.vstack(([most_common_pair                            (Best_anchor_test[:,[ses_ind!=ind for ses_ind in range(len(min_ses))],ii].T)[0]                            for ii in range(num_neurons)])) for ind in range(len(min_ses))]))

    freq_most_common_anchor_crossval=    np.vstack(([np.hstack(([most_common_pair                            (Best_anchor_test[:,[ses_ind!=ind for ses_ind in range(len(min_ses))],ii].T)[1]                            for ii in range(num_neurons)])) for ind in range(len(min_ses))]))

    max_mindist_cross_val=np.vstack(([np.max(Min_dist[:,[ses_ind!=ind for ses_ind in range(len(min_ses))]],axis=1)            for ind in range(len(min_ses))]))
    Anchored_bool_coherent_crossval=max_mindist_cross_val<coh_thr    

    Anchored_bool_half_crossval=freq_most_common_anchor_crossval>=thr_anchored
    Anchored_bool_always_crossval=freq_most_common_anchor_crossval==1

    Anchored_bool_crossval=np.logical_and(Anchored_bool_half_crossval,Anchored_bool_coherent_crossval)
    Anchored_bool_strict_crossval=np.logical_and(Anchored_bool_always_crossval,Anchored_bool_coherent_crossval)

    ##Neuron by neuron coherence 
    coh_bool=dists_test<coh_thr 
    coh_bool_sum=np.sum(coh_bool,axis=0) 
    
    ###Subsetting by used neurons for mean plots 
    used_pairs=used_pairs_dic[mouse_recday] ###see main defining cell (neurons with 1-3 peaks)
    neurons_used_pairwise=np.unique(used_pairs) 
    anchor_max=Phase_spatial_corr_dic['Max_corr_bin'][mouse_recday] 
    nonspatial_neurons1=np.where(anchor_max!=0)[0] ###neurons where max correlation is at zero lag from occupancy
    Spatial_anchoring_fields=Spatial_anchoring_dic['field_peak_bins'][mouse_recday] 
    zero_anchor_boolean=np.asarray([0 in Spatial_anchoring_fields[ii] for ii in range(len(Spatial_anchoring_fields))]) 
    nonspatial_neurons2=np.where(zero_anchor_boolean==False)[0] ###neurons with any signioficant peak at
    ###zero lag (even if not maximum)
    

    
    if spatial_tuning_GLM==True:
        nonspatial_neurons=np.where(Tuned_dic['Place'][mouse_recday]==False)[0]
    
    if spatial_tuning_strict==True:
        nonspatial_neurons=np.intersect1d(nonspatial_neurons,nonspatial_neurons3) 
        nonspatial_neurons3=np.where(Spatial_tuning_overlystrict_dic[mouse_recday]==False)[0]
        nonspatial_neurons=np.intersect1d(nonspatial_neurons1,nonspatial_neurons2) 
    ##i.e. neurons with no significant spatial correlation at zero lag
    num_anchor_peaks=np.asarray([len(Spatial_anchoring_fields[ii])                                 for ii in range(len(Spatial_anchoring_fields))])
    single_peaked_neurons=np.where(num_anchor_peaks==1)[0]
    multi_peaked_neurons=np.where(num_anchor_peaks>1)[0]
    non_peaked_neurons=np.where(num_anchor_peaks==0)[0]
    
    ###this keeps only neurons tuned to state - as defined in Tuned_dic in Tuning_basic.ipynb 

    if day_type=='combined_ABCDonly':
        neurons_tuned=np.where(Tuned_dic['State_zmax_bool'][mouse_recday]==True)[0] ##state neurons
        
    elif day_type=='combined_ABCDE':
        neurons_tuned_bool=np.load(Intermediate_object_folder_dropbox+'ABCDE_State_zmax_bool_'+mouse_recday+'.npy')
        neurons_tuned=np.where(neurons_tuned_bool==True)[0]
        
    
    
    if len(neurons_tuned)==0:
        if delete_old==True:
            del(Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best'][mouse_recday])
            del(Spatial_anchoring_dic['best_node_phase_used'][mouse_recday])
            del(Spatial_anchoring_dic['most_common_anchor'][mouse_recday])
            del(Spatial_anchoring_dic['most_common_anchor_crossval'][mouse_recday])
            del(Spatial_anchoring_dic['Anchored_bool'][mouse_recday])
            del(Spatial_anchoring_dic['Anchored_bool_strict'][mouse_recday])
            del(Spatial_anchoring_dic['Anchored_bool_crossval'][mouse_recday])
            del(Spatial_anchoring_dic['Anchored_bool_strict_crossval'][mouse_recday])   
            del(Spatial_anchoring_dic['Angles_all'][mouse_recday])
            del(Spatial_anchoring_dic['Dists_all'][mouse_recday])
            del(Spatial_anchoring_dic['Coherent_proportion'][mouse_recday])
            del(Spatial_anchoring_dic['Coherent_proportion'][day_type][mouse_recday])
            del(Spatial_anchoring_dic['Neuron_coherence'][mouse_recday])
            del(Spatial_anchoring_dic['Best_reference_task'][mouse_recday])
            del(Spatial_anchoring_dic['Neuron_tuned'][mouse_recday])
            del(Spatial_anchoring_dic['Neuron_used_histogram'][mouse_recday])
        print('Not used - no tuned neurons')
        continue

    neurons_used=neurons_tuned ##ignoring number of peaks and just saying is the neuron tuned in atleast one task
    if remove_spatial==True: 
        neurons_used=np.intersect1d(neurons_used,nonspatial_neurons) 
    if keep_singlepeaked==True:
        neurons_used=np.intersect1d(neurons_used,single_peaked_neurons) 
    if remove_multipeaked==True:
        neurons_used=np.setdiff1d(neurons_used,multi_peaked_neurons)
    if remove_nonpeaked==True:
        neurons_used=np.setdiff1d(neurons_used,non_peaked_neurons)
    
        

    dists_used=dists_test[:,neurons_used] 
    angles_used=angles_test[:,neurons_used]
    
    ###making histograms by averaging across all training-test splits
    angles_spatialanchor_num_all=[] 
    angles_spatialanchor_cohprop_all=[] 
    for ii in np.arange(num_comparisons): 
        angles_spatialanchor=remove_nan(angles_used[ii]) 
        if len(angles_spatialanchor)>0: 
            coh_prop=len(np.where(np.logical_or(angles_spatialanchor<coh_thr1 ,angles_spatialanchor>coh_thr2))[0])            /len(angles_spatialanchor) 
            angles_spatialanchor_num=np.histogram(angles_spatialanchor,np.linspace(0,360,37))[0] 
        else: 
            coh_prop=np.nan 
            angles_spatialanchor_num=np.repeat(np.nan,36) 

        angles_spatialanchor_cohprop_all.append(coh_prop) 
        angles_spatialanchor_num_all.append(angles_spatialanchor_num) 
    angles_spatialanchor_num_mean=np.nanmean(np.asarray(angles_spatialanchor_num_all),axis=0) 
    angles_spatialanchor_cohprop_mean=np.nanmean(angles_spatialanchor_cohprop_all) 

    polar_plot_stateX(angles_spatialanchor_num_mean,angles_spatialanchor_num_mean, 
                      angles_spatialanchor_num_mean,color='black',labels='angles',plot_type='bar') 
    plt.show() 
    
    
    
    Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best'][mouse_recday]=angles_spatialanchor_num_mean 
    Spatial_anchoring_dic['best_node_phase_used'][mouse_recday]=Best_anchor_test
    Spatial_anchoring_dic['most_common_anchor'][mouse_recday]=most_common_anchor
    Spatial_anchoring_dic['most_common_anchor_bool'][mouse_recday]=most_common_anchor_bool
    Spatial_anchoring_dic['most_common_anchor_crossval'][mouse_recday]=most_common_anchor_crossval
    Spatial_anchoring_dic['Anchored_bool'][mouse_recday]=Anchored_bool
    Spatial_anchoring_dic['Anchored_bool_strict'][mouse_recday]=Anchored_bool_strict
    Spatial_anchoring_dic['Anchored_bool_crossval'][mouse_recday]=Anchored_bool_crossval
    Spatial_anchoring_dic['Anchored_bool_strict_crossval'][mouse_recday]=Anchored_bool_strict_crossval 
    Spatial_anchoring_dic['Anchored_bool_tuned'][mouse_recday]=Anchored_bool_tuned
    Spatial_anchoring_dic['Anchored_bool_tuned_strict'][mouse_recday]=Anchored_bool_tuned_strict   
    Spatial_anchoring_dic['Angles_all'][mouse_recday]=angles_test 
    Spatial_anchoring_dic['Dists_all'][mouse_recday]=dists_test 
    Spatial_anchoring_dic['Coherent_proportion'][mouse_recday]=angles_spatialanchor_cohprop_mean 
    Spatial_anchoring_dic['Neuron_coherence'][mouse_recday]=(coh_bool).astype(int) 
    Spatial_anchoring_dic['Best_reference_task'][mouse_recday]=min_ses
    Spatial_anchoring_dic['Neuron_tuned'][mouse_recday]=neurons_tuned
    Spatial_anchoring_dic['Neuron_used_histogram'][mouse_recday]=neurons_used 


# In[ ]:


ah07_21112023_22112023


# In[ ]:





# In[46]:


print('') 
print('__________________') 
print('All recording days') 
#angles_spatialanchor_num=np.nansum(remove_empty(dict_to_array(Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best']))\
#                                   ,axis=0) 


angles_spatialanchor_num=np.nansum(np.vstack(([Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best'][mouse_recday]            for mouse_recday in day_type_dicX[day_type]
                                    if len(Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best'][mouse_recday])>0
                                               ])),axis=0)
print(angles_spatialanchor_num) 

polar_plot_stateX(angles_spatialanchor_num,angles_spatialanchor_num, 
                  angles_spatialanchor_num,color='black',labels='angles',plot_type='bar') 
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'/Spatial_anchoring_histogram.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[47]:


total_coh=[]
total_neurons=[]
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    if len(Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best'][mouse_recday])==0:
        print('Not used')
        continue
        
    total_coh.append(Spatial_anchoring_dic['Coherent_proportion'][mouse_recday]*    len(Spatial_anchoring_dic['Neuron_used_histogram'][mouse_recday]))
    total_neurons.append(len(Spatial_anchoring_dic['Neuron_used_histogram'][mouse_recday]))
    
print('Total coherent proportion: '+str(np.sum(total_coh)/np.sum(total_neurons)))


# In[48]:


print(two_proportions_test(np.sum(total_coh), np.sum(total_neurons),                           np.sum(total_neurons)*(1/num_states), np.sum(total_neurons)))
print(np.sum(total_coh), np.sum(total_neurons))


# In[49]:


##Proportion of anchored neurons (same anchor in half the tasks or more and aligned)
anchored_bool_all=np.hstack(dict_to_array(Spatial_anchoring_dic['Anchored_bool']))
len(np.where(anchored_bool_all==True)[0])/len(anchored_bool_all)


# In[55]:





# In[ ]:





# In[56]:


###Simple anchor analysis 
apply_minimum_sesnumber=True
num_ses_thr=3 ##minimum number of comparisons for anchor to be considered
num_phases=3
num_nodes=9
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    
    
    
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    num_neurons=len(cluster_dic['good_clus'][mouse_recday]) 
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)
    dists_=np.copy(Spatial_anchoring_dic['rotation_dist_mat'][mouse_recday])
    #num_trials=dict_to_array(Num_trials_dic2[mouse_recday])
    num_trials=np.load(Intermediate_object_folder+'Num_trials_'+mouse_recday+'.npy')
    
    trials_completed_ses=np.where(num_trials>2)[0]
    non_repeat_ses=np.intersect1d(non_repeat_ses,trials_completed_ses)
    
    best_anchor=np.zeros((num_neurons,2))
    best_anchor_refses=np.zeros(num_neurons)
    best_anchor_mindist=np.zeros(num_neurons)
    
    best_anchor[:]=np.nan
    best_anchor_refses[:]=np.nan
    best_anchor_mindist[:]=np.nan
    for neuron in np.arange(num_neurons):
        dists_neuron=dists_[neuron]
        dists_neuron_nonrepeat=dists_neuron[:,:,non_repeat_ses][:,:,:,non_repeat_ses]
        dists_neuron_nonrepeat_copy=np.copy(dists_neuron_nonrepeat)
        
        if apply_minimum_sesnumber==False:
            min_ref_ses=np.vstack(([[np.nanargmin(np.nanmean(fill_diagonal(dists_neuron_nonrepeat_copy                                                                           [phase_,location_],np.nan),axis=0))                                      for phase_ in np.arange(num_phases)] for location_ in np.arange(num_nodes)])).T

            min_value=np.vstack(([[np.nanmin(np.nanmean(fill_diagonal(dists_neuron_nonrepeat_copy                                                                      [phase_,location_],np.nan),axis=0))                                      for phase_ in np.arange(num_phases)] for location_ in np.arange(num_nodes)])).T
        else:
            min_ref_ses=np.zeros((num_phases,num_locations))
            min_value=np.zeros((num_phases,num_locations))
            min_ref_ses[:]=np.nan
            min_value[:]=np.nan

            for phase_ in np.arange(num_phases):
                for location_ in np.arange(num_nodes):
                    refses_passing_thr=np.count_nonzero(~np.isnan                                                        (fill_diagonal(dists_neuron_nonrepeat_copy[phase_,location_]                                                                       ,np.nan)),axis=0)>=num_ses_thr
                    means=np.nanmean(fill_diagonal(dists_neuron_nonrepeat_copy[phase_,location_],np.nan),axis=0)
                    means[refses_passing_thr==False]=np.nan
                    if np.isnan(np.nanmean(means))==False:
                        min_ref_ses_anchor=np.nanargmin(means)
                        min_value_anchor=np.nanmin(means)
                    else:
                        min_ref_ses_anchor=np.nan
                        min_value_anchor=np.nan

                    min_ref_ses[phase_,location_]=min_ref_ses_anchor
                    min_value[phase_,location_]=min_value_anchor

        overall_min_dist_value=np.nanmin(min_value)
        pref_phase=np.where(min_value==overall_min_dist_value)[0][0]
        pref_location=np.where(min_value==overall_min_dist_value)[1][0]
        pref_ref_ses=min_ref_ses[pref_phase,pref_location]
        
        best_anchor[neuron]=pref_phase,pref_location
        best_anchor_refses[neuron]=pref_ref_ses
        best_anchor_mindist[neuron]=overall_min_dist_value
        
    Spatial_anchoring_dic['Best_anchor_all'][mouse_recday]=best_anchor
    Spatial_anchoring_dic['Best_anchor_all_refses'][mouse_recday]=best_anchor_refses
    Spatial_anchoring_dic['Best_anchor_all_mindist'][mouse_recday]=best_anchor_mindist


# In[ ]:





# In[ ]:





# In[57]:


for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    print(len(Spatial_anchoring_dic['Neuron_used_histogram'][mouse_recday]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


###Cross-validated single anchor correlation at best anchor
thr_visit=2
num_bins_state=90
#num_states=4
for mouse_recday in day_type_dicX[day_type]:

    print(mouse_recday)
    
    if day_type=='combined_ABCDonly':
        num_states=4
        
    elif day_type=='combined_ABCDE':
        num_states=5
    
    num_bins=num_states*90
    
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    sessions=Task_num_dic[mouse_recday]
    num_refses=len(np.unique(sessions))
    num_comparisons=num_refses-1
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
    
    num_trials=dict_to_array(Num_trials_dic2[mouse_recday])
    
    trials_completed_ses=np.where(num_trials>2)[0]
    non_repeat_ses=np.intersect1d(non_repeat_ses,trials_completed_ses)
    
    abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
    if day_type=='combined_ABCDE':
        ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
        num_ABCD_ses=len(ABCD_sessions)
        non_repeat_ses=np.setdiff1d(non_repeat_ses,ABCD_sessions)-len(ABCD_sessions)
    
    state_corrs_allneurons=np.zeros(num_neurons)
    state_corrs_allneurons[:]=np.nan
    
    for session in np.arange(num_sessions):
        try:
            #name='standardized_spike_events_dic_'+mouse_recday+'_'+str(session)
            #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
            #ephys_ = load(data_filename_memmap)#, mmap_mode='r')
            ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
            exec('ephys_ses_'+str(session)+'_=ephys_')
        except:
            print('Files not found')
            continue


    #print('Importing Occupancy')
    #name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
    #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
    #occupancy_ = load(data_filename_memmap)#, mmap_mode='r')
    
    if day_type=='combined_ABCDonly':
        phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][0]
        
    elif day_type=='combined_ABCDE':
        phase_peaks=np.load(Intermediate_object_folder_dropbox+'ABCDE_tuning_phase_boolean_max_'+mouse_recday+'.npy')[0]
    pref_phase_neurons=np.argmax(phase_peaks,axis=1)
    
    #num_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind],axis=1)\
    #          for ses_ind in np.arange(len(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday]))])).T

    for neuron in np.arange(num_neurons):
        pref_phase=pref_phase_neurons[neuron]
        mean_smooth_all=np.zeros((len(non_repeat_ses),2,num_bins))
        mean_smooth_all[:]=np.nan
        for ses_ind, session in enumerate(non_repeat_ses):  
            try:
                
                #if num_peaks_all[neuron,ses_ind]==0:
                #    continue
                
                best_anchor_phase_node=Spatial_anchoring_dic['best_node_phase_used'][mouse_recday][:,:,neuron][:,ses_ind]
                ref_task=Spatial_anchoring_dic['Best_reference_task'][mouse_recday][ses_ind,neuron]
                #best_anchor_phase_node=Spatial_anchoring_dic['most_common_anchor'][mouse_recday][neuron]
                #best_anchor_phase_node=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday][neuron]

                phase_=int(best_anchor_phase_node[0])
                location=int(best_anchor_phase_node[1]+1) ##because 0 based indexing used here but 1 based in locations

                for ind_no, ses_indX in enumerate([ses_ind,ref_task]):
                    session=non_repeat_ses[ses_indX]
                    #occupancy_mat=data_matrix(occupancy_[session])
                    #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
                    
                    location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
                    if len(location_mat_)==0:
                        continue
                    occupancy_mat=np.reshape(location_mat_,(num_states,len(location_mat_),                                                            len(location_mat_.T)//num_states))
                    occupancy_conc=np.concatenate(location_mat_)

                    phase_mat=np.zeros(np.shape(occupancy_mat))
                    phase_mat[:,:,30:60]=1
                    phase_mat[:,:,60:90]=2
                    phase_conc=np.concatenate(np.hstack(phase_mat))
                    
                    exec('ephys_=ephys_ses_'+str(session)+'_')
                    ephys_neuron_=ephys_[neuron]
                    #neuron_mat=data_matrix(ephys_neuron_,concatenate=False)
                    neuron_mat=ephys_neuron_
                    neuron_conc=np.concatenate(neuron_mat)

                    tone_aligned_activity=neuron_mat


                    mean_=np.mean(tone_aligned_activity,axis=0)
                    sem_=st.sem(tone_aligned_activity,axis=0)
                    mean_smooth=smooth_circular(mean_)


                    timestamps_=np.where((np.logical_and(occupancy_conc==location, phase_conc==phase_)))[0]
                    long_stays=np.where(rank_repeat2(occupancy_conc)>thr_visit)[0]
                    timestamps=np.intersect1d(timestamps_,long_stays-(thr_visit+1))
                    if len(timestamps)>0:

                        timestamps_start=timestamps[(np.hstack((1,np.diff(timestamps)>thr_visit))).astype(bool)]
                        timestamps_end=timestamps[(np.hstack((np.diff(timestamps)>thr_visit,1))).astype(bool)]
                        aligned_activity=np.asarray([neuron_conc[ii:ii+num_bins]                                                     if len(neuron_conc[ii:ii+num_bins])==num_bins                                                     else np.repeat(np.nan,num_bins) for ii in timestamps_start])

                        mean_=np.nanmean(aligned_activity,axis=0)
                        sem_=st.sem(aligned_activity,axis=0,nan_policy='omit')
                        mean_smooth=smooth_circular(mean_)
                        sem_smooth=smooth_circular(sem_)
                        if len(timestamps_start)==1:
                            sem_smooth=np.repeat(0,num_bins)

                        if np.nanmean(mean_smooth)==0 or np.isnan(np.nanmean(mean_smooth))==True:
                            mean_smooth=np.repeat(np.nan,num_bins)

                    else:
                        mean_smooth=np.repeat(np.nan,num_bins)
                        sem_smooth=np.repeat(np.nan,num_bins)



                    mean_smooth_all[ses_ind,ind_no]=mean_smooth
            except Exception as e:
                if neuron==0:
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print('session not analysed')
                continue

        phase_norm_mean=np.tile(np.repeat(np.arange(num_phases),num_bins_state/num_phases),num_states)
        phase_norm_mean_states=np.reshape(phase_norm_mean,(num_states,num_bins_state))
        corr_all=[]
        for ses_ind in np.arange(len(mean_smooth_all)):
            mean_smooth_all_means=[np.asarray([np.nanmean(mean_smooth_all[ses_ind,jj,num_bins_state*ii:                                                                          num_bins_state*(ii+1)]            [phase_norm_mean_states[ii]==pref_phase]) for ii in range(num_states)]) for jj in np.arange(2)]
            
            if np.isnan(np.mean(mean_smooth_all_means))==False: 
                corr_ses=st.pearsonr(mean_smooth_all_means[0],mean_smooth_all_means[1])[0]
            else:
                corr_ses=np.nan
            corr_all.append(corr_ses)

        corr_mean=np.nanmean(corr_all)
        
        state_corrs_allneurons[neuron]=corr_mean
    Spatial_anchoring_dic['Cross_val_corr'][mouse_recday]=state_corrs_allneurons


# In[59]:


day_type


# In[288]:


###plotting frequency of anchors
num_phases=3
num_locations=9
most_common_anchor_all=[]
bool_all=[]
for mouse_recday in day_type_dicX[day_type]:
    try:
        most_common_anchor=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday]   
        Anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]
        
        most_common_anchor_anchored=most_common_anchor[Anchored_bool]
        most_common_anchor_all.append(most_common_anchor_anchored)
    except:
        print(mouse_recday)
        print('Not used')
        
most_common_anchor_all=np.vstack(most_common_anchor_all)

xedges = np.hstack((np.arange(num_phases),num_phases))
yedges = np.hstack((np.arange(num_locations),num_locations))

phase_loc_hist=np.histogram2d(most_common_anchor_all[:,0],most_common_anchor_all[:,1],bins=[xedges, yedges])[0]
plt.matshow(phase_loc_hist,vmin=0)
plt.savefig(Ephys_output_folder_dropbox+'/Anchor_distribution.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(phase_loc_hist)
print(len(most_common_anchor_all))


# In[289]:


len(bool__)


# In[ ]:





# In[290]:


plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False

from scipy import stats
maze_hist=np.zeros((num_phases,3,3))
maze_hist[:]=np.nan

for phase_ in np.arange(num_phases):
    for location_ in np.arange(num_locations):
        maze_hist[phase_,Task_grid[location_,0],Task_grid[location_,1]]=phase_loc_hist[phase_,location_]
for phase_ in np.arange(num_phases):
    plt.matshow(maze_hist[phase_],vmin=0,cmap='Blues')
    plt.colorbar(orientation='vertical',fraction=.1)
    plt.savefig(Ephys_output_folder_dropbox+'Single_Anchor_analysis_locations_phase'+str(phase_)+'.svg' ,                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    print(stats.kstest(remove_nan(phase_loc_hist[phase_]),stats.uniform.cdf))
    
    print(st.entropy(phase_loc_hist[phase_],base=len(phase_loc_hist[phase_])))


# In[182]:


from scipy import stats
print(stats.kstest(remove_nan(phase_loc_hist[phase_]),stats.uniform.cdf))


# In[181]:


phase_loc_hist[phase_]


# In[94]:


###checking spatial tuning
mouse_recday='ah04_05122021_06122021'
spatial_tuning_strict=True
used_pairs=used_pairs_dic[mouse_recday] ###see main defining cell (neurons with 1-3 peaks)
neurons_used_pairwise=np.unique(used_pairs) 
anchor_max=Phase_spatial_corr_dic['Max_corr_bin'][mouse_recday] 
nonspatial_neurons1=np.where(anchor_max!=0)[0] ###neurons where max correlation is at zero lag from occupancy
Spatial_anchoring_fields=Spatial_anchoring_dic['field_peak_bins'][mouse_recday] 
zero_anchor_boolean=np.asarray([0 in Spatial_anchoring_fields[ii] for ii in range(len(Spatial_anchoring_fields))]) 
nonspatial_neurons2=np.where(zero_anchor_boolean==False)[0] ###neurons with any signioficant peak at
###zero lag (even if not maximum)
nonspatial_neurons3=np.where(Spatial_tuning_overlystrict_dic[mouse_recday]==False)[0]
nonspatial_neurons=np.intersect1d(nonspatial_neurons1,nonspatial_neurons2) 
if spatial_tuning_strict==True:
    nonspatial_neurons=np.intersect1d(nonspatial_neurons,nonspatial_neurons3) 

spatial_neurons=np.where(anchor_max==0)[0]


# In[ ]:





# In[95]:


##Plotting spatial maps
num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
sessions=Task_num_dic[mouse_recday]
repeat_ses=np.where(rank_repeat(sessions)>0)[0]
non_repeat_ses=non_repeat_ses_maker(mouse_recday)  ###this defines the sessions used

mouse_recday_individual1=mouse_recday[:13]
mouse_recday_individual2=mouse_recday[:4]+mouse_recday[13:]

All_sessions=session_dic['All'][mouse_recday]    
awake_sessions=session_dic['awake'][mouse_recday]
rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
structure_nums=np.unique(rec_day_structure_numbers)
abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]


for neuron in nonspatial_neurons:
    print(neuron)
    
    
    fignamex=Ephys_output_folder+'/Example_cells/'+mouse_recday+'_neuron_id'+str(neuron)+'_task'

    arrange_plot_statecells_persessionX2(mouse_recday,neuron,                                          Data_folder=Intermediate_object_folder_dropbox,                                         abstract_structures=abstract_structures,                                        plot=True, save=False, fignamex=fignamex, figtype='.svg',Marker=False)
    plt.show()
    
    spatial_arrays=np.zeros((len(non_repeat_ses),21))
    node_arrays=np.zeros((len(non_repeat_ses),9))
    for awake_session_ind, awake_session in enumerate(non_repeat_ses):
        node_rate_mat=node_rate_matrices_dic['All_states'][awake_session][mouse_recday][neuron]
        edge_rate_mat=edge_rate_matrices_dic['All_states'][awake_session][mouse_recday][neuron]

        node_edge_mat=edge_node_fill(edge_rate_mat,node_rate_mat)
        node_edge_array=np.hstack(node_edge_mat)
        node_edge_array=node_edge_array[~np.isnan(node_edge_array)]
        spatial_arrays[awake_session_ind]=node_edge_array
        
        node_arrays[awake_session_ind]=np.hstack(node_rate_mat)

    corr_mat=np.corrcoef(spatial_arrays)
    corrs=corr_mat[np.triu_indices(len(non_repeat_ses), k = 1)]
    print(np.mean(corrs))
    print(st.sem(corrs))
    print(st.ttest_1samp(corrs,0)[1])

    plot_spatial_mapsX(mouse_recday,neuron,non_repeat_ses,sessions_custom=True)
    #plot_spatial_mapsX(mouse_recday_individual2,neuron)
    
    print('')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


##Plotting state and spatial maps
num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
sessions=Task_num_dic[mouse_recday]
repeat_ses=np.where(rank_repeat(sessions)>0)[0]
non_repeat_ses=non_repeat_ses_maker(mouse_recday)  ###this defines the sessions used

mouse_recday_individual1=mouse_recday[:13]
mouse_recday_individual2=mouse_recday[:4]+mouse_recday[13:]

spatial_neurons=np.where(Tuned_dic['Place'][mouse_recday]==True)[0]
nonspatial_neurons=np.where(Tuned_dic['Place'][mouse_recday]==False)[0]
statewithin_neurons=np.where(Tuned_dic['State_withintask'][mouse_recday]==True)[0]
#spatial_neurons_ttest=np.where(Tuned_dic['Place_ttest'][mouse_recday]==True)[0]
#spatial_neurons_new=np.setdiff1d(spatial_neurons,spatial_neurons_ttest)

All_sessions=session_dic['All'][mouse_recday]    
awake_sessions=session_dic['awake'][mouse_recday]
rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
structure_nums=np.unique(rec_day_structure_numbers)
abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

Threshold_crossings=Phase_spatial_corr_dic['Threshold_Crossings'][mouse_recday]


remove_edges=True

for neuron in nonspatial_neurons:
    print(neuron)
    
    GLM_beta_neuron=GLM_dic2['mean_neuron_betas'][mouse_recday][neuron][1]
    
    fignamex=Ephys_output_folder+'/Example_cells/'+mouse_recday+'_neuron_id'+str(neuron)+'_task'

    #arrange_plot_statecells_persessionX2(mouse_recday,neuron, \
    #                                     Data_folder=Intermediate_object_folder_dropbox,\
    #                                     abstract_structures=abstract_structures,\
    #                                    plot=True, save=False, fignamex=fignamex, figtype='.svg',Marker=False)
    #plt.show()
    
    spatial_arrays=np.zeros((len(non_repeat_ses),21))
    node_arrays=np.zeros((len(non_repeat_ses),9))
    for awake_session_ind, awake_session in enumerate(non_repeat_ses):
        node_rate_mat=node_rate_matrices_dic['All_states'][awake_session][mouse_recday][neuron]
        edge_rate_mat=edge_rate_matrices_dic['All_states'][awake_session][mouse_recday][neuron]

        node_edge_mat=edge_node_fill(edge_rate_mat,node_rate_mat)
        node_edge_array=np.hstack(node_edge_mat)
        node_edge_array=node_edge_array[~np.isnan(node_edge_array)]
        spatial_arrays[awake_session_ind]=node_edge_array
        
        node_arrays[awake_session_ind]=np.hstack(node_rate_mat)

    
    
    if remove_edges==True:
        corr_mat=np.corrcoef(node_arrays)
    else:
        corr_mat=np.corrcoef(spatial_arrays)
    corrs=corr_mat[np.triu_indices(len(non_repeat_ses), k = 1)]
    

    plot_spatial_mapsX(mouse_recday,neuron,non_repeat_ses,sessions_custom=True)
    plt.show()
    #plot_spatial_mapsX(mouse_recday_individual2,neuron)
    print(np.mean(corrs))
    #print(st.sem(corrs))
    #print(st.ttest_1samp(corrs,0)[1])
    
    print(GLM_beta_neuron)
    print(Threshold_crossings[neuron])
    
    print('')


# In[ ]:





# In[ ]:


print('')
print('Lagged Spatial maps')

states=['A','B','C','D']
gridX=Task_grid_plotting2

#neuron=0
node_edge_rate_matrices=Spatial_anchoring_dic['Phase_shifted_node_edge_matrices'][mouse_recday][neuron]

mouse=mouse_recday[:4]
rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
structure_nums=np.unique(rec_day_structure_numbers)


for awake_ses_ind_ind, awake_session_ind in enumerate(non_repeat_ses):   

    fig1, f1_axes = plt.subplots(figsize=(20, 2),ncols=num_lags, constrained_layout=True)
    for lag_ind, lag in enumerate(np.arange(num_lags)):

        node_edge_mat_state=node_edge_rate_matrices[awake_ses_ind_ind,lag_ind]
        mat_used=node_edge_mat_state
        structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]        [rec_day_session_numbers[awake_session_ind]]

        ax=f1_axes[lag_ind]
        for state_port_ind, state_port in enumerate(states):
            node=structure[state_port_ind]-1
            ax.text(gridX[node,0]-0.25, gridX[node,1]+0.25,                    state_port.lower(), fontsize=22.5)

        ax.matshow(mat_used, cmap='coolwarm') #vmin=min_rate, vmax=max_rate
        ax.axis('off')
        #ax.savefig(str(neuron)+state+str(awake_session_ind)+'discmap.svg')
plt.axis('off')
plt.show()


# In[ ]:


print('')
print('Lagged Spatial maps')
mouse_recday='me11_01122021_02122021'
neuron=6
node_edge_rate_matrices=Spatial_anchoring_dic['Phase_shifted_node_edge_matrices'][mouse_recday][neuron]

states=['A','B','C','D']
gridX=Task_grid_plotting2

mouse=mouse_recday[:4]
rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
structure_nums=np.unique(rec_day_structure_numbers)

fig2, f2_axes = plt.subplots(figsize=(20, 10),ncols=num_lags, nrows=len(non_repeat_ses),                             constrained_layout=True)   
for awake_ses_ind_ind, awake_session_ind in enumerate(non_repeat_ses):   

    #print(awake_session_ind)
    for lag_ind, lag in enumerate(np.arange(num_lags)):

        node_edge_mat_state=node_edge_rate_matrices[awake_ses_ind_ind,lag_ind]
        mat_used=node_edge_mat_state
        structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]        [rec_day_session_numbers[awake_session_ind]]

        ax=f2_axes[awake_ses_ind_ind,lag_ind]
        for state_port_ind, state_port in enumerate(states):
            node=structure[state_port_ind]-1
            ax.text(gridX[node,0]-0.25, gridX[node,1]+0.25,                    state_port.lower(), fontsize=22.5)

        ax.matshow(mat_used, cmap='coolwarm') #vmin=min_rate, vmax=max_rate
        ax.axis('off')
        #ax.savefig(str(neuron)+state+str(awake_session_ind)+'discmap.svg')
plt.axis('off')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[67]:


###Trial by trial anchoring analysis

day_type='combined_ABCDonly'
#Anchor_trial_dic=rec_dd()

num_trials_thr=5

shifts=np.arange(11)-5 ##this is used for shifting across trials
num_shifts=len(shifts)
use_mean=False
use_timestamps=True

skip_analysed=False
regression=False
use_individualsession_anchor=False
thr_visit=2

for use_individualsession_anchor in [True,False]:
    if use_individualsession_anchor==True:
        name_addition='_cross_val'
    else:
        name_addition=''
    for mouse_recday in day_type_dicX[day_type]:
        print(mouse_recday)
        
        if day_type=='combined_ABCDonly':
            num_states=4
        
        elif day_type=='combined_ABCDE':
            num_states=5

        num_bins=num_states*90
        angle_correction=360/num_bins
        
        #try:

        num_neurons=len(cluster_dic['good_clus'][mouse_recday])
        abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]

        if skip_analysed==True:
            if len(Anchor_trial_dic['phase_location'][mouse_recday])>0:
                if np.shape(Anchor_trial_dic['phase_location'][mouse_recday])[1]==num_neurons:
                    print('Already analysed')
                    continue

        #Importing Ephys

        num_sessions=len(session_dic_behaviour['awake'][mouse_recday])

        ##defining sessions to use
        sessions=Task_num_dic[mouse_recday]
        num_refses=len(np.unique(sessions))
        num_comparisons=num_refses-1
        repeat_ses=np.where(rank_repeat(sessions)>0)[0]
        non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
        
        #num_trials=dict_to_array(Num_trials_dic2[mouse_recday])
        num_trials=np.load(Intermediate_object_folder+'Num_trials_'+mouse_recday+'.npy')
        
        trials_completed_ses=np.where(num_trials>2)[0]
        non_repeat_ses=np.intersect1d(non_repeat_ses,trials_completed_ses)
        
        
        if day_type=='combined_ABCDE':
            ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
            num_ABCD_ses=len(ABCD_sessions)
            non_repeat_ses=np.setdiff1d(non_repeat_ses,ABCD_sessions)-len(ABCD_sessions)

        num_nonrepeat_sessions=len(non_repeat_ses)


        ##Importing Occupancy
        #print('Importing Occupancy')
        #name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
        #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
        #occupancy_ = load(data_filename_memmap)#, mmap_mode='r')


        corr_all_all=[]
        corr_all_trials_all=[]
        corr_all_trials_half2_all=[]
        auto_corr_all_trials_all=[]
        innerp_all_trials_all=[]
        Beta_all=[]
        Beta_uncorrected_all=[]

        best_shift_all=np.zeros((num_neurons,num_nonrepeat_sessions))
        best_shift_trials_all=np.zeros((num_neurons,num_nonrepeat_sessions))
        best_shift_trials_half2_all=np.zeros((num_neurons,num_nonrepeat_sessions))
        phase_location_all=np.zeros((num_nonrepeat_sessions,num_neurons,2))

        best_shift_all[:]=np.nan
        best_shift_trials_all[:]=np.nan
        best_shift_trials_half2_all[:]=np.nan

        print('Importing Ephys')
        for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
            print(ses_ind)
            #####
            try:
                #name='standardized_spike_events_dic_'+mouse_recday+'_'+str(ses_ind)
                #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
                #ephys_ = load(data_filename_memmap)#, mmap_mode='r')
                ephys_ = np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            except:
                corr_all_all.append([])
                corr_all_trials_all.append([])
                corr_all_trials_half2_all.append([])
                auto_corr_all_trials_all.append([])
                innerp_all_trials_all.append([])
                Beta_all.append([])
                Beta_uncorrected_all.append([])
                print('No Ephys')
                continue

            #if len(occupancy_)==0:
            #    continue

            location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
            if len(location_mat_)==0:
                print('No entries in location file')
                continue
            occupancy_mat=np.reshape(location_mat_,(num_states,len(location_mat_),len(location_mat_.T)//num_states))
            occupancy_conc=np.concatenate(location_mat_)
            #occupancy_mat=data_matrix(occupancy_[ses_ind])
            #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))

            phase_mat=np.zeros(np.shape(occupancy_mat))
            phase_mat[:,:,30:60]=1
            phase_mat[:,:,60:90]=2

            #phase_mat=np.zeros(np.shape(occupancy_mat))
            #phase_mat[:,:,45:90]=1
            #phase_mat[:,:,60:90]=2
            phase_conc=np.concatenate(np.hstack(phase_mat))
            occupancy_mat_=location_mat_
            phase_mat_=np.hstack(phase_mat)

            ephys_neuron_=neuron_mat=ephys_[0]
            #neuron_mat=data_matrix(ephys_neuron_,concatenate=False)

            if len(neuron_mat)==0 or np.shape(occupancy_mat)[1]<num_trials_thr:
                corr_all_all.append([])
                corr_all_trials_all.append([])
                corr_all_trials_half2_all.append([])
                auto_corr_all_trials_all.append([])
                innerp_all_trials_all.append([])
                Beta_all.append([])
                Beta_uncorrected_all.append([])
                print('not enough trials')
                continue

            if np.shape(occupancy_mat)[1]<num_trials_thr:
                print('not enough trials')
                continue



            tone_aligned_activity=neuron_mat
            min_trials=int(np.min([len(occupancy_mat_),len(tone_aligned_activity)]))

            corr_all_ses=np.zeros((num_neurons,num_bins))
            corr_all_trials_ses=np.zeros((num_neurons,num_shifts))
            auto_corr_all_trials_ses=np.zeros((num_neurons,num_shifts))
            innerp_all_trials_ses=np.zeros((num_neurons,num_shifts))
            beta_neuron_means_ses=np.zeros((num_neurons,num_shifts))
            beta_neuron_means_uncorrected_ses=np.zeros((num_neurons,num_shifts))

            corr_all_ses[:]=np.nan
            corr_all_trials_ses[:]=np.nan
            auto_corr_all_trials_ses[:]=np.nan
            innerp_all_trials_ses[:]=np.nan
            beta_neuron_means_ses[:]=np.nan
            beta_neuron_means_uncorrected_ses[:]=np.nan
            
            if len(Spatial_anchoring_dic['best_node_phase_used'][mouse_recday])==0:
                print('Not used')
                continue

            for neuron in np.arange(num_neurons):

                if use_individualsession_anchor==True:
                    anchors=(Spatial_anchoring_dic['best_node_phase_used'][mouse_recday][:,:,neuron]).astype(int)
                    phase_=anchors[0,ses_ind_ind]
                    location_=anchors[1,ses_ind_ind]
                    location=location_+1
                    name_addition='_cross_val'
                else:
                    #anchor=(Spatial_anchoring_dic['best_node_phase'][mouse_recday][neuron]).astype(int)
                    anchor=(Spatial_anchoring_dic['Best_anchor_all'][mouse_recday][neuron]).astype(int)
                    location_=anchor[1]
                    phase_=anchor[0]
                    location=int(location_+1)
                    name_addition=''

                phase_location_all[ses_ind_ind,neuron]=phase_,location_

                ephys_neuron_=ephys_[neuron]
                neuron_mat=ephys_neuron_
                neuron_conc=np.concatenate(neuron_mat)

                ###tone aligned activity
                tone_aligned_activity=neuron_mat
                anchor_mat=(np.logical_and(phase_mat_==phase_,occupancy_mat_==location)).astype(int)
                min_trials=int(np.min([len(anchor_mat),len(tone_aligned_activity)]))

                tone_aligned_activity_matched=tone_aligned_activity[:min_trials]
                anchor_mat_matched=anchor_mat[:min_trials]
                corr_all=np.zeros(num_bins)
                for shift in range(num_bins):
                    tone_aligned_activity_shifted=np.roll(tone_aligned_activity_matched,-shift)#,axis=1)

                    if use_mean==False:
                        corr_mat=np.corrcoef(anchor_mat_matched,tone_aligned_activity_shifted)
                        cross_corr_mat=corr_mat[min_trials:,:min_trials]
                        corr_all[shift]=np.nanmean(np.diagonal(cross_corr_mat))
                    elif use_mean==True:
                        mean_neuron=np.mean(tone_aligned_activity_shifted,axis=0)
                        mean_anchor=np.mean(anchor_mat_matched,axis=0)
                        corr_all[shift]=st.pearsonr(mean_anchor,mean_neuron)[0]

                best_shift_corr=np.argmax(corr_all)

                #######
                if use_timestamps==True:
                    timestamps_=np.where((np.logical_and(occupancy_conc==location, phase_conc==phase_)))[0]
                    long_stays=np.where(rank_repeat2(occupancy_conc)>thr_visit)[0]
                    timestamps=np.intersect1d(timestamps_,long_stays-(thr_visit+1))
                    if len(timestamps)>0:

                        timestamps_start=timestamps[(np.hstack((1,np.diff(timestamps)>thr_visit))).astype(bool)]
                        timestamps_end=timestamps[(np.hstack((np.diff(timestamps)>thr_visit,1))).astype(bool)]
                        aligned_activity=np.asarray([neuron_conc[ii:ii+num_bins]                                                     if len(neuron_conc[ii:ii+num_bins])==num_bins                                                     else np.repeat(np.nan,num_bins) for ii in timestamps_start])
                        mean_=np.nanmean(aligned_activity,axis=0)
                        sem_=st.sem(aligned_activity,axis=0,nan_policy='omit')
                        mean_smooth=smooth_circular(mean_)

                        best_shift=np.argmax(mean_smooth)

                    else:
                        best_shift=np.nan

                else:
                    best_shift=best_shift_corr


                tone_aligned_activity_shifted_best=np.roll(tone_aligned_activity_matched,-best_shift_corr)#,axis=1)
                #corr_mat=np.corrcoef(anchor_mat_matched,tone_aligned_activity_shifted_best)
                #cross_corr_mat=corr_mat[min_trials:,:min_trials]

                bins_used=np.sum(anchor_mat_matched.T,axis=1)>0 ###i.e. regions in task space where anchor is visited 

                corr_all_trials=np.zeros(num_shifts)
                autocorr_all_trials=np.zeros(num_shifts)
                innerp_all_trials=np.zeros(num_shifts)
                beta_neuron_means=np.zeros(num_shifts)
                beta_neuron_means_uncorrected=np.zeros(num_shifts)

                corr_all_trials[:]=np.nan
                autocorr_all_trials[:]=np.nan
                innerp_all_trials[:]=np.nan
                beta_neuron_means[:]=np.nan
                beta_neuron_means_uncorrected[:]=np.nan

                anchor_matchedT=anchor_mat_matched.T[bins_used,:-1]
                tone_aligned_activity_usedT=tone_aligned_activity_shifted_best.T[bins_used,:-1]
                ##removing first and last trials because they have extraneous bins from 
                ##the shifting of tone_aligned_activity

                #for shift in range(len(anchor_mat_matched)):
                for shift_ind, shift in enumerate(shifts):
                    ###shift is always the shift of ephys relative to behaviour (i.e.-1 means ephys 
                    ##moved one bin back)
                    if shift<0:
                        tone_aligned_activity_usedT_=tone_aligned_activity_usedT[:,:shift]
                        anchor_matchedT_=anchor_matchedT[:,:shift]
                        tone_aligned_activity_shiftedT_=tone_aligned_activity_usedT[:,-shift:]
                        anchor_matched_shiftedT_=anchor_matchedT[:,-shift:]

                    elif shift>0:
                        tone_aligned_activity_usedT_=tone_aligned_activity_usedT[:,shift:]
                        anchor_matchedT_=anchor_matchedT[:,shift:]
                        tone_aligned_activity_shiftedT_=tone_aligned_activity_usedT[:,:-shift]
                        anchor_matched_shiftedT_=anchor_matchedT[:,:-shift]

                    elif shift==0:
                        anchor_matchedT_=anchor_matchedT
                        tone_aligned_activity_shiftedT_=tone_aligned_activity_usedT
                        anchor_matched_shiftedT_=anchor_matchedT
                        tone_aligned_activity_usedT_=tone_aligned_activity_usedT

                    ###for behaviour-neuron
                    xx=anchor_matchedT_
                    co_xx1=tone_aligned_activity_usedT_
                    yy=tone_aligned_activity_shiftedT_


                    ###for neuron-behaviour
                    #xx=tone_aligned_activity_usedT_
                    #co_xx1=anchor_matchedT_
                    #yy=anchor_matched_shiftedT_

                    corr_mat=np.corrcoef(xx,yy)
                    len_bins=len(xx)
                    cross_corr_mat=corr_mat[len_bins:,:len_bins]

                    #tone_aligned_activity_shiftedT=np.roll(tone_aligned_activity_shifted_best.T[bins_used],
                    ##-shift,axis=1)
                    #corr_mat=np.corrcoef(anchor_matchedT,tone_aligned_activity_shiftedT)
                    #len_bins=len(anchor_matchedT)
                    #cross_corr_mat=corr_mat[len_bins:,:len_bins]

                    corr_all_trials[shift_ind]=np.nanmean(np.diagonal(cross_corr_mat))
                    innerp_all_trials[shift_ind]=np.nanmean(np.diagonal(xx@yy.T))


                    #anchor_matched_shiftedT=np.roll(anchor_matchedT,-shift,axis=1)

                    corr_mat_=np.corrcoef(xx,co_xx1)
                    auto_corr_mat=corr_mat_[len_bins:,:len_bins]
                    autocorr_all_trials[shift_ind]=np.nanmean(np.diagonal(auto_corr_mat))
                    
                    if regression==True:
                        ###regression
                        beta_neuron_all=np.zeros(len(xx))
                        beta_neuron_all[:]=np.nan
                        beta_neuron_all_uncorrected=np.zeros(len(xx))
                        beta_neuron_all_uncorrected[:]=np.nan
                        for indx in np.arange(len(xx)):

                            ##corrected for current trial anchor visit
                            if np.std(xx[indx])==0 or shift==0 or np.max(xx[indx]-co_xx1[indx])==0                            or np.std(co_xx1[indx])==0:
                                beta_neuron_all[indx]=np.nan                        
                            else:
                                X=np.column_stack((np.ones(len(xx[indx])),demean(co_xx1[indx]),                                                   demean(xx[indx])))
                                y=yy[indx]
                                if np.linalg.det((X.T @ X))==0:
                                    beta_neuron_all[indx]=np.nan
                                else:
                                    beta_all=np.linalg.inv(X.T @ X) @ X.T @ y
                                    beta_neuron_all[indx]=beta_all[-1]

                            ##uncorrected
                            if np.std(xx[indx])==0:
                                beta_neuron_all_uncorrected[indx]=np.nan
                            else:
                                X_=np.column_stack((np.ones(len(xx[indx])),                                                   demean(xx[indx])))
                                y_=yy[indx]

                                if np.linalg.det((X_.T @ X_))==0:
                                    beta_neuron_all_uncorrected[indx]=np.nan
                                else:
                                    beta_all_=np.linalg.inv(X_.T @ X_) @ X_.T @ y_
                                    beta_neuron_all_uncorrected[indx]=beta_all_[-1]

                        beta_neuron_means[shift_ind]=np.nanmean(beta_neuron_all)
                        beta_neuron_means_uncorrected[shift_ind]=np.nanmean(beta_neuron_all_uncorrected)
                best_shift_trials=np.argmax(corr_all_trials)



                ###making session arrays
                corr_all_ses[neuron]=corr_all
                corr_all_trials_ses[neuron]=corr_all_trials
                auto_corr_all_trials_ses[neuron]=autocorr_all_trials
                innerp_all_trials_ses[neuron]=innerp_all_trials
                if regression==True:
                    beta_neuron_means_ses[neuron]=beta_neuron_means
                    beta_neuron_means_uncorrected_ses[neuron]=beta_neuron_means_uncorrected

                
                best_shift_all[neuron,ses_ind_ind]=best_shift
                best_shift_trials_all[neuron,ses_ind_ind]=best_shift_trials


            ###making day arrays    
            corr_all_all.append(corr_all_ses)
            corr_all_trials_all.append(corr_all_trials_ses)
            auto_corr_all_trials_all.append(auto_corr_all_trials_ses)
            innerp_all_trials_all.append(innerp_all_trials_ses)
            Beta_all.append(beta_neuron_means_ses)
            Beta_uncorrected_all.append(beta_neuron_means_uncorrected_ses)

        Anchor_trial_dic['phase_location'+name_addition][mouse_recday]=np.asarray(phase_location_all)
        Anchor_trial_dic['Corr_time'+name_addition][mouse_recday]=np.asarray(corr_all_all)
        Anchor_trial_dic['Corr_trials'+name_addition][mouse_recday]=np.asarray(corr_all_trials_all)
        Anchor_trial_dic['Auto_Corr_trials'+name_addition][mouse_recday]=np.asarray(auto_corr_all_trials_all)
        Anchor_trial_dic['Best_shift_time'+name_addition][mouse_recday]=np.asarray(best_shift_all)
        Anchor_trial_dic['Best_shift_trials'+name_addition][mouse_recday]=np.asarray(best_shift_trials_all)
        Anchor_trial_dic['Inner_product_trials'+name_addition][mouse_recday]=np.asarray(innerp_all_trials_all)
        if regression==True:
            Anchor_trial_dic['Beta_anchor'+name_addition][mouse_recday]=np.asarray(Beta_all)
            Anchor_trial_dic['Beta_anchor_uncorrected'+name_addition][mouse_recday]=np.asarray(Beta_uncorrected_all)
        #Anchor_trial_dic['Beta_anchor_neuronbehaviour'][mouse_recday]=np.asarray(Beta_all)
        #Anchor_trial_dic['Beta_anchor_neuronbehaviour_uncorrected'][mouse_recday]=np.asarray(Beta_uncorrected_all)
        #except:
        #    print('Not done')


# In[288]:


len(Spatial_anchoring_dic['best_node_phase_used'][mouse_recday])


# In[194]:


Num_trials_dic2['me08_12092021_13092021']


# In[26]:


non_repeat_ses_maker('me08_12092021_13092021') 


# In[ ]:





# In[ ]:





# In[68]:


####One best shift time per neuron
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    #try:

    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    most_common_angle_all=np.zeros(num_neurons)
    most_common_angle_all[:]=np.nan
    
    mean_angle_all=np.zeros(num_neurons)
    mean_angle_all[:]=np.nan
    for neuron in np.arange(num_neurons):
        most_common_anchor_bool=Spatial_anchoring_dic['most_common_anchor_bool'][mouse_recday][neuron]
        if len(most_common_anchor_bool)==0:
            print('Not used')
            continue
        Best_shift_time=Anchor_trial_dic['Best_shift_time'][mouse_recday][neuron]
        Best_shift_time_disc=(Best_shift_time[most_common_anchor_bool]//30)*30
        
        mean_angle=np.rad2deg(st.circmean(np.deg2rad(remove_nan(Best_shift_time[most_common_anchor_bool]))))
        
        if len(remove_nan(Best_shift_time_disc))==0:
            most_common_angle=np.nan
        else:  
            most_common_angle=st.mode(remove_nan(Best_shift_time_disc))[0][0]
      
        mean_angle_all[neuron]=mean_angle
        most_common_angle_all[neuron]=most_common_angle

    Anchor_trial_dic['Best_shift_time_mostcommon'][mouse_recday]=most_common_angle_all
    Anchor_trial_dic['Best_shift_time_mean'][mouse_recday]=mean_angle_all
    Anchor_trial_dic['Best_shift_time_mostcommon_all'][mouse_recday]=Best_shift_time
    
    #except:
    #    print('Not used')
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[152]:


All_lags


# In[70]:


###Single anchoring analysis - subsetted by lag from anchor 

condition='non-zero-strict' #'non-zero','non-zero-strict','>N-1states'

if day_type=='combined_ABCDonly':
    num_states=4
if day_type=='combined_ABCDE':
    num_states=5
num_bins=num_states*90

day_type='combined_ABCDonly'
    
coh_thr1=(360/num_states)//2
coh_thr2=360-coh_thr1


if condition=='non-zero':
    min_thr=(360/num_states)//3
    max_thr=360-min_thr
elif condition=='non-zero-strict':
    min_thr=(360/num_states)
    max_thr=360-min_thr
elif confition=='>N-1states':
    min_thr=(360/num_states)*(num_states-1)
    max_thr=360

for mouse_recday in day_type_dicX[day_type]:


    print(mouse_recday)
    #try:
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    num_neurons=len(cluster_dic['good_clus'][mouse_recday]) 
    abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
    print(num_sessions)

    sessions=Task_num_dic[mouse_recday]
    
     
    if day_type=='combined_ABCDE':
        
        sessions=sessions[abstract_structures=='ABCDE']
    
    #num_refses=len(np.unique(sessions))
    #num_comparisons=num_refses-1
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)  ###this defines the sessions used
    ###only the first session from each task is used
    
    #num_trials=dict_to_array(Num_trials_dic2[mouse_recday])
    num_trials=np.load(Intermediate_object_folder+'Num_trials_'+mouse_recday+'.npy')
    
    trials_completed_ses=np.where(num_trials>2)[0]
    non_repeat_ses=np.intersect1d(non_repeat_ses,trials_completed_ses)

    
    
    if day_type=='combined_ABCDE':
        ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
        num_ABCD_ses=len(ABCD_sessions)
        non_repeat_ses=np.setdiff1d(non_repeat_ses,ABCD_sessions)-len(ABCD_sessions)
    
    num_refses=len(non_repeat_ses)
    num_comparisons=num_refses-1
   
    dists_test=Spatial_anchoring_dic['Dists_all'][mouse_recday]
    angles_test=Spatial_anchoring_dic['Angles_all'][mouse_recday]


    #neurons_tuned=Spatial_anchoring_dic['Neuron_tuned'][mouse_recday]
    
    if day_type=='combined_ABCDonly':
        phase_tuning=Tuned_dic['Phase'][mouse_recday] 
        state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday]
        #state_tuning=Tuned_dic2['State']['95'][mouse_recday]
    elif day_type=='combined_ABCDE':
        phase_tuning=np.load(Intermediate_object_folder_dropbox+'ABCDE_Phase_'+mouse_recday+'.npy')
        state_tuning=np.load(Intermediate_object_folder_dropbox+'ABCDE_State_zmax_bool_'+mouse_recday+'.npy')
        
    #neurons_tuned=np.where(np.logical_and(state_tuning,phase_tuning)==True)[0]
    neurons_tuned=np.where(state_tuning==True)[0]
    neurons_usedx=Spatial_anchoring_dic['Neuron_used_histogram'][mouse_recday]

    Anchor_lags=Anchor_trial_dic['Best_shift_time'][mouse_recday]    
    Anchor_lags_mean=np.rad2deg(st.circmean(np.deg2rad(Anchor_lags),axis=1,nan_policy='omit'))

    non_zero_anchored=np.where(np.logical_and(Anchor_lags_mean>min_thr,Anchor_lags_mean<max_thr)==True)[0]
    neurons_used=np.intersect1d(neurons_tuned,non_zero_anchored) ##remove zero lag neurons
    #neurons_used=non_zero_anchored

    if len(neurons_used)==0:
        print('Not used')
        continue
    
    if day_type=='combined_ABCDonly':
        tuning_state_bool_day=tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday]
        
        
    elif day_type=='combined_ABCDE':
        tuning_state_bool_day=np.load(Intermediate_object_folder_dropbox+                                      'ABCDE_tuning_state_boolean_'+mouse_recday+'.npy')
        
    num_peaks_all=np.vstack(([np.sum(tuning_state_bool_day[ses_ind],axis=1)                              for ses_ind in np.arange(len(tuning_state_bool_day))])).T

    dists_used=dists_test[:,neurons_used] 
    angles_used=angles_test[:,neurons_used]
    num_peaks_used=num_peaks_all[neurons_used]

    ###making histograms by averaging across all training-test splits
    angles_spatialanchor_num_all=[] 
    angles_spatialanchor_cohprop_all=[] 
    for ses_ind in np.arange(num_refses): 
        angles_used[ses_ind,num_peaks_used[:,ses_ind]==0]=np.nan
        angles_spatialanchor=remove_nan(angles_used[ses_ind]) 
        if len(angles_spatialanchor)>0: 
            coh_prop=len(np.where(np.logical_or(angles_spatialanchor<coh_thr1 ,angles_spatialanchor>coh_thr2))[0])            /len(angles_spatialanchor) 
            angles_spatialanchor_num=np.histogram(angles_spatialanchor,np.linspace(0,num_bins,37))[0] 
        else: 
            coh_prop=np.nan 
            angles_spatialanchor_num=np.repeat(np.nan,36) 

        angles_spatialanchor_cohprop_all.append(coh_prop) 
        angles_spatialanchor_num_all.append(angles_spatialanchor_num) 
    angles_spatialanchor_num_mean=np.nanmean(np.asarray(angles_spatialanchor_num_all),axis=0) 
    angles_spatialanchor_cohprop_mean=np.nanmean(angles_spatialanchor_cohprop_all) 

    polar_plot_stateX(angles_spatialanchor_num_mean,angles_spatialanchor_num_mean, 
                      angles_spatialanchor_num_mean,color='black',labels='angles',plot_type='bar') 
    plt.show() 

    Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best_nonzero'][mouse_recday]=angles_spatialanchor_num_mean 
    Spatial_anchoring_dic['Coherent_proportion_nonzero'][mouse_recday]=angles_spatialanchor_cohprop_mean 
    Spatial_anchoring_dic['Neuron_used_histogram_nonzero'][mouse_recday]=neurons_used
    #except:
    #print('Not used')
###after looping over mouse_recdays

#angles_spatialanchor_num=np.nansum(dict_to_array(Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best_nonzero'])\
#                                   ,axis=0) 


# In[125]:


max_thr


# In[126]:


###49 neurons missing from before in non-zero lag


# In[ ]:





# In[71]:


angles_spatialanchor_num=np.nansum(np.vstack(([Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best_nonzero']                                               [mouse_recday]            for mouse_recday in day_type_dicX[day_type]
        if len(Spatial_anchoring_dic['MeanAngles_spatial_Anchor_best_nonzero'][mouse_recday])>0])),axis=0)
print(angles_spatialanchor_num) 

polar_plot_stateX(angles_spatialanchor_num,angles_spatialanchor_num, 
                  angles_spatialanchor_num,color='black',labels='angles',plot_type='bar') 
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'/Spatial_anchoring_histogram_nonspatial.svg', bbox_inches = 'tight',            pad_inches = 0)
plt.show()


# In[ ]:





# In[279]:


total_coh=[]
total_neurons=[]
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    try:

        total_coh.append(Spatial_anchoring_dic['Coherent_proportion_nonzero'][mouse_recday]*        len(Spatial_anchoring_dic['Neuron_used_histogram_nonzero'][mouse_recday]))
        total_neurons.append(len(Spatial_anchoring_dic['Neuron_used_histogram_nonzero'][mouse_recday]))
    except:
        print('Not used')
print('Total coherent proportion: '+str(np.nansum(total_coh)/np.nansum(total_neurons)))


# In[280]:


print(two_proportions_test(np.nansum(total_coh), np.nansum(total_neurons),                           np.nansum(total_neurons)*(1/num_states), np.nansum(total_neurons)))
print(np.nansum(total_coh), np.nansum(total_neurons))


# In[ ]:





# In[ ]:





# In[281]:


mean_bestlag_corr_crossval_nonspatial=[]
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    try:
        Neuron_used_histogram_nonzero=Spatial_anchoring_dic['Neuron_used_histogram_nonzero'][mouse_recday]
        Neurons_tuned=Tuned_dic['State_zmax_bool'][mouse_recday]
        mean_bestlag_corr_crossval_nonspatial.append(Spatial_anchoring_dic['Cross_val_corr'][mouse_recday]                                                     [Neuron_used_histogram_nonzero])
    except:
        print('Not used')


# In[ ]:





# In[ ]:





# In[ ]:





# In[283]:


use_tuned=True ###does nothing because already subsetted by phase

if day_type=='combined_ABCDonly':
    #phase_tuning=np.hstack(([Tuned_dic['Phase'][mouse_recday] for mouse_recday in\
    #                         day_type_dicX[day_type]]))
    #state_tuning=np.hstack(([Tuned_dic['State_zmax_bool'][mouse_recday] for mouse_recday in\
    #                         day_type_dicX[day_type]]))
    state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'State_zmax_bool_'+mouse_recday+'.npy')        for mouse_recday in day_type_dicX[day_type]]))
elif day_type=='combined_ABCDE':
    state_tuning=np.hstack(([np.load(Intermediate_object_folder_dropbox+'ABCDE_State_zmax_bool_'+mouse_recday+'.npy')                                 for mouse_recday in day_type_dicX[day_type]]))
neurons_tuned=state_tuning
###i.e. phase/state tuned neurons

print('All neurons')

plt.rcParams["figure.figsize"] = (7,5)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True

#mean_bestlag_corr_crossval=remove_nan(np.hstack((dict_to_array(Spatial_anchoring_dic['Cross_val_corr'])))\
#                                          [neurons_tuned])
mean_bestlag_corr_crossval=remove_nan(np.hstack(([Spatial_anchoring_dic['Cross_val_corr'][mouse_recday]                                                  for mouse_recday in day_type_dicX[day_type]]))[neurons_tuned])
    
plt.hist(mean_bestlag_corr_crossval,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'SingleAnchor_analysis.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(mean_bestlag_corr_crossval))
print(st.ttest_1samp(mean_bestlag_corr_crossval,0))
print('')

print('Non spatial neurons')

mean_bestlag_corr_crossval=remove_nan(np.hstack((mean_bestlag_corr_crossval_nonspatial)))

plt.hist(mean_bestlag_corr_crossval,bins=50,color='grey')
plt.axvline(0,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'SingleAnchor_analysis_nonzero.svg' , bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(len(mean_bestlag_corr_crossval))
print(st.ttest_1samp(mean_bestlag_corr_crossval,0))
print('')


# In[ ]:





# In[ ]:





# In[284]:


###Per mouse analysis

for subset in ['All','non_spatial']:
    for mouse in Mice:
        mouse_recdays_bool=np.asarray([mouse in day_type_dicX['combined_ABCDonly'][ii]                             for ii in range(len(day_type_dicX['combined_ABCDonly']))])
        mouse_recdays_mouse=day_type_dicX['combined_ABCDonly'][mouse_recdays_bool]

        if len(mouse_recdays_mouse)==0:
            continue
        per_mouse_betas=[]
        for mouse_recday in mouse_recdays_mouse:
            print(mouse_recday)
            try:
                state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday]
                Neuron_used_histogram_nonzero=Spatial_anchoring_dic['Neuron_used_histogram_nonzero'][mouse_recday]
                if subset=='non_spatial':
                    neurons_tuned=np.intersect1d(Neuron_used_histogram_nonzero,np.where(state_tuning==True)[0])
                else:
                    neurons_tuned=np.where(state_tuning==True)[0]
                per_mouse_betas.append(Spatial_anchoring_dic['Cross_val_corr'][mouse_recday][neurons_tuned])
            except:
                print('Not used')
        per_mouse_betas=np.hstack((per_mouse_betas))


        ttest_res=st.ttest_1samp(remove_nan(per_mouse_betas),0)
        Spatial_anchoring_dic['per_mouse_subsetted'][subset][mouse]=per_mouse_betas
        Spatial_anchoring_dic['per_mouse_subsetted_mean'][subset][mouse]=np.nanmean(per_mouse_betas)
        Spatial_anchoring_dic['per_mouse_subsetted_sem'][subset][mouse]=st.sem(per_mouse_betas,nan_policy='omit')
        Spatial_anchoring_dic['per_mouse_subsetted_ttest'][subset][mouse]=ttest_res
        


# In[285]:


subset='All'
np.asarray(list(Spatial_anchoring_dic['per_mouse_subsetted_mean'][subset].keys()))


# In[286]:


for subset in ['All','non_spatial']:
    per_mouse_betas_means=dict_to_array(Spatial_anchoring_dic['per_mouse_subsetted_mean'][subset])
    per_mouse_betas_sems=dict_to_array(Spatial_anchoring_dic['per_mouse_subsetted_sem'][subset])
    per_mouse_betas_ttest=dict_to_array(Spatial_anchoring_dic['per_mouse_subsetted_ttest'][subset])
    Mice=np.asarray(list(Spatial_anchoring_dic['per_mouse_subsetted_mean'][subset].keys()))

    plt.errorbar(per_mouse_betas_means,np.arange(len(per_mouse_betas_means)),xerr=per_mouse_betas_sems,ls='none',
            marker='o',color='grey')
    plt.axvline(0,ls='dashed',color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'SingleAnchor_analysis_permouse_'+subset+'.svg',                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print(np.column_stack((Mice,per_mouse_betas_means)))
    print(per_mouse_betas_ttest)

    print(st.ttest_1samp(per_mouse_betas_means,0))
    
    per_mouse_betas_means_nonan=remove_nan(per_mouse_betas_means)
    num_positive=len(np.where(per_mouse_betas_means_nonan>0)[0])
    print(st.binom_test(x=num_positive, n=len(per_mouse_betas_means_nonan), p=0.5, alternative='greater'))


# In[ ]:





# In[132]:


###Lags from anchor - single anchor analysis


# In[291]:


###Plotting lags from anchor
from scipy import stats
for angle_type in ['Best_shift_time_mostcommon','Best_shift_time_mean','Best_shift_time_all']:
    if angle_type=='Best_shift_time_all':
        All_lags=np.hstack(([np.hstack((Anchor_trial_dic['Best_shift_time'][mouse_recday]))                     for mouse_recday in day_type_dicX[day_type]]))
    else:
        All_lags=np.hstack(([Anchor_trial_dic[angle_type][mouse_recday] for mouse_recday     in day_type_dicX[day_type]]))
        
        
        bool_=np.asarray(np.hstack(([Spatial_anchoring_dic['Anchored_bool'][mouse_recday] for mouse_recday                                in day_type_dicX[day_type]])))
        bool__=bool_==True
        
        All_lags=All_lags[bool__]
        All_lags=remove_nan(All_lags)
        
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams["figure.figsize"] = (8,6)
    plt.hist(All_lags,bins=12,color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'Single_Anchor_'+angle_type+'analysis_lags.svg',                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print(len(All_lags))
    print(stats.kstest(remove_nan(All_lags),stats.uniform.cdf))


# In[134]:


##here yad


# In[135]:


num_anatomy_bins=4
num_channels_neuropixels=384
bin_size=num_channels_neuropixels/num_anatomy_bins
recording_days_=np.load(Intermediate_object_folder_dropbox+'combined_ABCDonly_days.npy')

mouse_recdays_found_anchoring=[]
for mouse_recday in recording_days_:
    #print(mouse_recday)
    try:
        np.load(Intermediate_object_folder_dropbox+'_channel_num_neuron_'+mouse_recday+'.npy')
        mouse_recdays_found_anchoring.append(mouse_recday)
    except:
        continue#print('Not found')


Anchoring_anatomy_dic=rec_dd()
for mouse_recday in mouse_recdays_found_anchoring:
    mouse=mouse_recday.split('_',1)[0]
    cohort=Mice_cohort_dic[mouse]
    if cohort not in [6]:
        continue
    lags_day=Anchor_trial_dic['Best_shift_time_mean'][mouse_recday]
    state_tuning=np.load(Intermediate_object_folder_dropbox+'State_'+mouse_recday+'.npy',allow_pickle=True)
    Anchoring=Spatial_anchoring_dic['Anchored_bool'][mouse_recday] 
    lags_day_anchored=lags_day[Anchoring]
    channel_num_corrected=np.load(Intermediate_object_folder_dropbox+'_channel_num_neuron_'+mouse_recday+'.npy')
    
    
    good_clus=cluster_dic['good_clus'][mouse_recday]
    channel_num_=cluster_dic['channel_number'][mouse_recday]
    channel_num=channel_num_[:,1][np.isin(channel_num_[:,0], good_clus)]
    
    anatomy_bin_neuron=((channel_num_corrected-1)//bin_size).astype(int)
    anatomy_bin_neuron_anchored=anatomy_bin_neuron[Anchoring]
    
    
    ####region bins
    bins_=[]
    for region_ind,region in enumerate(regions):
        channel_ids_region=np.where(Anatomy_channel_dic[mouse][region+'_bool']==True)[0]
        bins_.append(channel_ids_region)

    region_id_neuron=np.repeat(np.nan,len(channel_num))
    for ii in range(len(regions)):
        region_id_neuron[np.isin(channel_num,bins_[ii])]=ii
    region_id_neuron_anchored=region_id_neuron[Anchoring]
    
    
    
    for anatomy_type,arrayX in {'DV_bin':anatomy_bin_neuron,'region_id':region_id_neuron}.items():
        
        arrayX_anchored=arrayX[Anchoring]

        forward_lag_bins=np.hstack(([np.nanmean(lags_day_anchored[arrayX_anchored==anat_bin])            if len(lags_day_anchored[arrayX_anchored==anat_bin])>0 else np.nan for anat_bin                                       in np.arange(num_anatomy_bins)]))

        circular_lag_bins=np.hstack(([np.rad2deg(st.circmean(np.deg2rad(remove_nan(lags_day_anchored[        arrayX_anchored==anat_bin]))))        if len(lags_day_anchored[arrayX_anchored==anat_bin])>0 else np.nan for anat_bin                                       in np.arange(num_anatomy_bins)]))

        Anchoring_anatomy_dic[anatomy_type]['forward_lags'][mouse_recday]=forward_lag_bins
        Anchoring_anatomy_dic[anatomy_type]['circular_lags'][mouse_recday]=circular_lag_bins

        Anchoring_anatomy_dic[anatomy_type]['anatomy_bin'][mouse_recday]=arrayX_anchored
        Anchoring_anatomy_dic[anatomy_type]['All_lags'][mouse_recday]=lags_day_anchored


        Neuron_used_histogram_nonzero=Spatial_anchoring_dic['Neuron_used_histogram_nonzero'][mouse_recday]
        non_zero_bool=np.repeat(False,len(channel_num))
        non_zero_bool[Neuron_used_histogram_nonzero]=True
        corr_day=Spatial_anchoring_dic['Cross_val_corr'][mouse_recday]
        corr_day_nonzero=Spatial_anchoring_dic['Cross_val_corr'][mouse_recday][non_zero_bool]

        #arrayX_anchored_nonzero=arrayX[np.logical_and(Anchoring,non_zero_bool)]
        arrayX_nonzero=arrayX[non_zero_bool]

        corr_bins=np.hstack(([np.nanmean(corr_day[arrayX==anat_bin])            if len(corr_day[arrayX==anat_bin])>0 else np.nan for anat_bin                                           in np.arange(num_anatomy_bins)]))
        corr_bins_nonzero=np.hstack(([np.nanmean(corr_day_nonzero[arrayX_nonzero==anat_bin])            if len(corr_day_nonzero[arrayX_nonzero==anat_bin])>0 else np.nan for anat_bin                                           in np.arange(num_anatomy_bins)]))

        Anchoring_anatomy_dic[anatomy_type]['corr_bins'][mouse_recday]=corr_bins
        Anchoring_anatomy_dic[anatomy_type]['corr_bins_nonzero'][mouse_recday]=corr_bins_nonzero


# In[ ]:





# In[ ]:





# In[136]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
from scipy.stats import tukey_hsd

measure='corr_bins'
anatomy_type='region_id'

for measure in ['corr_bins','corr_bins_nonzero']:
    Measure_prop_anat_mean=np.nanmean(dict_to_array(Anchoring_anatomy_dic[anatomy_type][measure]),axis=0)
    Measure_prop_anat_sem=st.sem(dict_to_array(Anchoring_anatomy_dic[anatomy_type][measure]),nan_policy='omit',axis=0)
    plt.rcParams["figure.figsize"] = (3,6)
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    
    if anatomy_type=='region_id':
        y=['M2','ACC','PrL','IrL']
    else:
        y=-np.arange(len(Measure_prop_anat_mean.T))


    plt.errorbar(y=y,x=np.flip(Measure_prop_anat_mean),                 xerr=np.flip(Measure_prop_anat_sem),                marker='o',markersize=10,color='black')
    plt.axvline(0,ls='dashed',color='black',linewidth=4)
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.xlim(-0.1,0.4)
    if anatomy_type=='region_id':
        plt.gca().invert_yaxis()
    plt.savefig(Ephys_output_folder_dropbox+'DV_vs_proportion_'+measure+'.svg' , bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    

    Measure_prop_anat=dict_to_array(Anchoring_anatomy_dic[anatomy_type][measure])


    stats=st.f_oneway(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
    print(stats)

    if stats[1]<0.05:
        res = tukey_hsd(remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
        print(res)


# In[137]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
from scipy.stats import tukey_hsd
for measure in ['forward_lags','circular_lags']:
    
    if measure=='forward_lags':
        Measure_prop_anat_mean=np.nanmean(dict_to_array(Anchoring_anatomy_dic[anatomy_type][measure]),axis=0)
        Measure_prop_anat_sem=np.asarray(st.sem(dict_to_array(Anchoring_anatomy_dic[anatomy_type][measure]),nan_policy='omit',axis=0))
    elif measure=='circular_lags':
        Measure_prop_anat_mean=np.rad2deg(st.circmean(np.deg2rad(dict_to_array(Anchoring_anatomy_dic[anatomy_type]                                                                               [measure]),                                                        ),axis=0,nan_policy='omit'))
        Measure_prop_anat_sem=circular_sem(dict_to_array(Anchoring_anatomy_dic[anatomy_type][measure]))
    plt.rcParams["figure.figsize"] = (3,6)
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.errorbar(y=-np.arange(len(Measure_prop_anat_mean.T)),x=np.flip(Measure_prop_anat_mean),                     xerr=np.flip(Measure_prop_anat_sem),                    marker='o',markersize=10,color='black')
    plt.axvline(0,ls='dashed',color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.xlim(0,360)
    plt.savefig(Ephys_output_folder_dropbox+'DV_vs_proportion_'+measure+'.svg' , bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    Measure_prop_anat=dict_to_array(Anchoring_anatomy_dic[anatomy_type][measure])

    stats_=st.f_oneway(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                          remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
    print(stats_)

    if stats_[1]<0.05:
        res = tukey_hsd(remove_nan(Measure_prop_anat[:,0]),remove_nan(Measure_prop_anat[:,1]),                      remove_nan(Measure_prop_anat[:,2]),remove_nan(Measure_prop_anat[:,3]))
        print(res)


# In[138]:


from scipy import stats
anatomy_bin_neuron_anchored_all=np.hstack((dict_to_array(Anchoring_anatomy_dic[anatomy_type]['anatomy_bin'])))
lags_day_anchored_all=np.hstack((dict_to_array(Anchoring_anatomy_dic[anatomy_type]['All_lags'])))
len_all_neurons=[]
for bin_ in np.arange(4):
    All_lags=lags_day_anchored_all[anatomy_bin_neuron_anchored_all==bin_]
    
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams["figure.figsize"] = (8,6)
    plt.hist(All_lags,bins=12,color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'Single_Anchor_ANAtomybin_'+str(bin_)+'analysis_lags.svg',                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print(len(All_lags))
    print(stats.kstest(remove_nan(All_lags),stats.uniform.cdf))
    
    len_all_neurons.append(len(All_lags))
    
    
print(np.sum(len_all_neurons))

    


# In[ ]:





# In[ ]:





# In[ ]:


###Relationship between tuning and number of anchor crossings (assuming single anchor)
Anchor_tuning_dic=rec_dd()

day_type='combined_ABCDonly'
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)
    num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    sessions=Task_num_dic[mouse_recday]
    num_refses=len(np.unique(sessions))
    num_comparisons=num_refses-1
    repeat_ses=np.where(rank_repeat(sessions)>0)[0]
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)  ###this defines the sessions used
    ###only the first session from each task is used
    
    Anchor_tuning=np.zeros((num_neurons,len(non_repeat_ses),7))
    corr_crossing_tuning_all=np.zeros((num_neurons))
    corr_crossing_tuning_p_all=np.zeros((num_neurons))
    Tuning_crossing_score=np.zeros((num_neurons))
    
    Anchor_tuning[:]=np.nan
    corr_crossing_tuning_all[:]=np.nan
    corr_crossing_tuning_p_all[:]=np.nan
    Tuning_crossing_score[:]=np.nan
    for neuron in np.arange(num_neurons):
        if len(Spatial_anchoring_dic['best_node_phase_used'][mouse_recday])==0:
            if neuron==0:
                print('Not Analyzed')
            continue
        #anchors=(Spatial_anchoring_dic['best_node_phase_used'][mouse_recday][:,:,neuron]).astype(int)

        for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
            phase_occ=occupancy_phase_dic[mouse_recday][ses_ind]
            
            num_trials=int(len(phase_occ)/360)
            tuning_ses=tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind_ind][neuron]
            num_peaks_sign=np.sum(tuning_ses)
            num_peaks_all=num_field_dic['ALL'][day_type][ses_ind][mouse_recday][neuron]
            num_peaks_state=num_field_dic['State'][day_type][ses_ind][mouse_recday][neuron]
            
            if isinstance(num_peaks_all,int)==False:
                continue
            #tuned=int(num_peaks>0)

            #print(anchors)
            #bincounts_locations_=np.bincount(anchors[1])
            anchor_neuron=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday][neuron]

            #anchor_location=anchors[1,ses_ind_ind]
            #anchor_phase=anchors[0,ses_ind_ind]
            
            anchor_phase=int(anchor_neuron[0])
            anchor_location=int(anchor_neuron[1]+1)

            phase_occ_bool=(np.logical_and(phase_occ[:,0]==anchor_phase,phase_occ[:,1]==anchor_location)).astype(int)

            crossings_=np.diff(phase_occ_bool)
            crossings=np.where(crossings_==1)[0]
            if len(crossings)>0:
                crossing_gaps=np.hstack([360,np.diff(crossings)])
                crossings_spaced=crossings[crossing_gaps>30]
                num_crossings=len(crossings_spaced)

                trial_crossings=(crossings_spaced/360).astype(int)
                ind_trial_crossings=np.unique(trial_crossings)
                num_trial_crossings=number_of_repeats(trial_crossings)
                mean_nonzero_trial_crossings=np.mean(num_trial_crossings)
                indnum_trial_crossings=np.column_stack((ind_trial_crossings,num_trial_crossings))
                
                mean_trial_crossings=num_crossings/num_trials
            else:
                num_crossings=0
                mean_nonzero_trial_crossings=np.nan
                mean_trial_crossings=np.nan
                indnum_trial_crossings=[]
            #else:
            #    num_crossings=np.nan
            #    mean_trial_crossings=np.nan
            #    indnum_trial_crossings=[]
                
            
            
            
            Anchor_tuning[neuron,ses_ind_ind]=np.asarray([num_peaks_sign,num_peaks_all,num_peaks_state,                                                      num_crossings,mean_trial_crossings,num_trials,                                                          mean_nonzero_trial_crossings])
        
        num_crossings_untuned=np.nanmean(Anchor_tuning[neuron][:,3][Anchor_tuning[neuron][:,0]==0])
        num_crossings_tuned=np.nanmean(Anchor_tuning[neuron][:,3][Anchor_tuning[neuron][:,0]>0])
        
        crossings_score=(num_crossings_tuned-num_crossings_untuned)/(num_crossings_tuned+num_crossings_untuned)
        
        Tuning_crossing_score[neuron]=crossings_score
        #corr_crossing_tuning=st.pearsonr(Anchor_tuning[neuron][:,0],\
        #                                 Anchor_tuning[neuron][:,1])[0]
        
        if len(np.where(~np.isnan(Anchor_tuning[neuron][:,1]))[0])>=3        and len(np.where(~np.isnan(Anchor_tuning[neuron][:,4]))[0])>=3: 
            ##3 is minimum number of non nan entries for partial corr
            data = {'num_peaks_sign': Anchor_tuning[neuron][:,0],
                    'num_peaks_all': Anchor_tuning[neuron][:,1],
                    'num_peaks_state': Anchor_tuning[neuron][:,2],
                    'num_crossings': Anchor_tuning[neuron][:,3],
                    'mean_nonzero_trial_crossings': Anchor_tuning[neuron][:,6],
                    'num_trials':  Anchor_tuning[neuron][:,5]}

            df = pd.DataFrame(data, columns = ['num_peaks_all', 'num_crossings','mean_nonzero_trial_crossings'                                               ,'num_trials'])
            #,'spatial_simX'])

            corr_stats_=partial_corr(df, 'num_peaks_all', 'mean_nonzero_trial_crossings', covar=['num_trials'])
            corr_crossing_tuning=corr_stats_['r'][0]
            corr_crossing_tuning_p=corr_stats_['p-val'][0]

            corr_crossing_tuning_all[neuron]=corr_crossing_tuning
            corr_crossing_tuning_p_all[neuron]=corr_crossing_tuning_p
        else:
            corr_crossing_tuning_all[neuron]=np.nan
            corr_crossing_tuning_p_all[neuron]=np.nan
            

    Anchor_tuning_dic[mouse_recday]=Anchor_tuning
    Anchor_tuning_dic['num_peaks_all'][mouse_recday]=Anchor_tuning[:,:,1]
    Anchor_tuning_dic['num_peaks_state'][mouse_recday]=Anchor_tuning[:,:,2]
    Anchor_tuning_dic['correlation'][mouse_recday]=corr_crossing_tuning_all
    Anchor_tuning_dic['correlation_p'][mouse_recday]=corr_crossing_tuning_p_all
    Anchor_tuning_dic['Tuning_crossing_score'][mouse_recday]=Tuning_crossing_score


# In[ ]:


phase_occ_bool=(np.logical_and(phase_occ[:,0]==anchor_phase,phase_occ[:,1]==anchor_location)).astype(int)

crossings_=np.diff(phase_occ_bool)
crossings=np.where(crossings_==1)[0]
if len(crossings)>0:
    crossing_gaps=np.hstack([360,np.diff(crossings)])
    crossings_spaced=crossings[crossing_gaps>30]
    num_crossings=len(crossings_spaced)

    trial_crossings=(crossings_spaced/360).astype(int)
    ind_trial_crossings=np.unique(trial_crossings)
    num_trial_crossings=number_of_repeats(trial_crossings)
    mean_trial_crossings=np.mean(num_trial_crossings)
    indnum_trial_crossings=np.column_stack((ind_trial_crossings,num_trial_crossings))


# In[ ]:


crossings_spaced


# In[ ]:


trial_crossings


# In[ ]:


ind_trial_crossings


# In[ ]:





# In[ ]:





# In[ ]:


print('___All neurons___')
print('Tuning crossing score')
Tuning_crossing_score_all=np.hstack(((dict_to_array(Anchor_tuning_dic['Tuning_crossing_score']))))
print(np.nanmean(Tuning_crossing_score_all))
print(st.ttest_1samp(remove_nan(Tuning_crossing_score_all),0))

print('Tuning crossing')
corrs_all=np.hstack((dict_to_array(Anchor_tuning_dic['correlation'])))
print(np.nanmean(corrs_all))
print(st.ttest_1samp(remove_nan(corrs_all),0))

print('')
print('___Non spatial neurons___')
anchor_max=np.hstack((dict_to_array(Phase_spatial_corr_dic['Max_corr_bin'])))

min_thr=30
max_thr=360-min_thr
anchored_bool=concatenate_complex2(dict_to_array(Spatial_anchoring_dic['Anchored_bool']))#[mouse_recday]
Anchor_lags=concatenate_complex2(dict_to_array(Anchor_trial_dic['Best_shift_time']))
Anchor_lags_mean=np.asarray([np.rad2deg(st.circmean(np.deg2rad(Anchor_lags[ii]),nan_policy='omit'))                             for ii in range(len(Anchor_lags))])
nonzero_coherent_neurons=np.intersect1d(np.where(Coh_day_sum/len(Coh_day_)>0.75)[0],Neuron_used_day_)
anchored_neurons=np.intersect1d(np.where(anchored_bool==True)[0],Neuron_used_day_)
nonspatial_neurons=np.where(np.logical_and(Anchor_lags_mean>min_thr,Anchor_lags_mean<max_thr)==True)[0]
nonspatial_neurons_anchored=np.intersect1d(nonspatial_neurons,anchored_neurons)

print('Tuning crossing score')
Tuning_crossing_score_nonspatial=Tuning_crossing_score_all[nonspatial_neurons_anchored]
print(np.nanmean(Tuning_crossing_score_nonspatial))
print(st.ttest_1samp(remove_nan(Tuning_crossing_score_nonspatial),0))

corrs_nonspatial=corrs_all[nonspatial_neurons_anchored]
print(np.nanmean(corrs_nonspatial))
print(st.ttest_1samp(remove_nan(corrs_nonspatial),0))


# In[ ]:


Spatial_anchoring_dic['Anchored_bool'].keys()


# In[ ]:


###Anchor visits vs tuning
min_thr=30
max_thr=360-min_thr
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    try:
        conc_anchor_tuning=np.vstack((Anchor_tuning_dic[mouse_recday])) ##i.e. neuron-sessions
        anchor_max=Phase_spatial_corr_dic['Max_corr_bin'][mouse_recday]

        Neuron_used_day_=Spatial_anchoring_dic['Neuron_used_histogram'][mouse_recday]
        Coh_day_=Spatial_anchoring_dic['Neuron_coherence'][mouse_recday]
        Tuned_day_=Spatial_anchoring_dic['Neuron_tuned'][mouse_recday]
        Coh_day_sum=np.sum(Coh_day_,axis=0)

        anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]

        phase_tuning=Tuned_dic['Phase'][mouse_recday] 
        state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday]
        #neurons_tuned=np.where(np.logical_and(state_tuning,phase_tuning)==True)[0]
        neurons_tuned=np.where(state_tuning==True)

        Anchor_lags=Anchor_trial_dic['Best_shift_time'][mouse_recday]    
        Anchor_lags_mean=np.rad2deg(st.circmean(np.deg2rad(Anchor_lags),axis=1,nan_policy='omit'))




        if len(Coh_day_sum)==0:
            continue
        anchored_neurons=np.intersect1d(np.where(anchored_bool==True)[0],Neuron_used_day_)
        nonspatial_neurons=np.where(np.logical_and(Anchor_lags_mean>min_thr,Anchor_lags_mean<max_thr)==True)[0]
        spatial_neurons=np.where(np.logical_or(Anchor_lags_mean<min_thr,Anchor_lags_mean>max_thr)==True)[0]

        spatial_neurons_anchored=np.intersect1d(spatial_neurons,anchored_neurons)
        nonspatial_neurons_anchored=np.intersect1d(nonspatial_neurons,anchored_neurons)

        conc_anchor_tuning_nonspatial_perneuron=Anchor_tuning_dic[mouse_recday][nonspatial_neurons_anchored]
        num_crossings=conc_anchor_tuning_nonspatial_perneuron[:,:,3]
        num_trials=conc_anchor_tuning_nonspatial_perneuron[:,:,5]
        tuning_state=Tuned_dic['State_zmax'][mouse_recday][nonspatial_neurons_anchored]<0.05


        #conc_anchor_tuning_nonspatial=np.vstack((Anchor_tuning_dic[mouse_recday][nonspatial_neurons]))
        if len(nonspatial_neurons_anchored)>0:
            conc_anchor_tuning_nonspatial=np.vstack((Anchor_tuning_dic[mouse_recday][nonspatial_neurons_anchored]))  
            num_crossings_tuned=np.asarray([np.mean(num_crossings[neuron][tuning_state[neuron]==True])                                    for neuron in np.arange(len(num_crossings))])
            num_crossings_untuned=np.asarray([np.mean(num_crossings[neuron][tuning_state[neuron]==False])                                            for neuron in np.arange(len(num_crossings))])

            num_crossings_tuned_pertrial=np.asarray([np.mean(num_crossings[neuron][tuning_state[neuron]==True]/                                                   num_trials[neuron][tuning_state[neuron]==True])                                            for neuron in np.arange(len(num_crossings))])
            num_crossings_untuned_pertrial=np.asarray([np.mean(num_crossings[neuron][tuning_state[neuron]==False]/                                                     num_trials[neuron][tuning_state[neuron]==False])                                            for neuron in np.arange(len(num_crossings))])
        else:
            conc_anchor_tuning_nonspatial=np.repeat(np.nan,7)
            num_crossings_tuned=np.nan
            num_crossings_untuned=np.nan
            num_crossings_tuned_pertrial=np.nan
            num_crossings_untuned_pertrial=np.nan

        if len(spatial_neurons_anchored)>0:
            conc_anchor_tuning_spatial=np.vstack((Anchor_tuning_dic[mouse_recday][spatial_neurons_anchored]))
        else:
            conc_anchor_tuning_spatial=np.repeat(np.nan,7)

        Anchor_tuning_dic['All_sessions_concatenated'][mouse_recday]=conc_anchor_tuning
        Anchor_tuning_dic['All_sessions_concatenated_nonspatial'][mouse_recday]=conc_anchor_tuning_nonspatial
        Anchor_tuning_dic['All_sessions_concatenated_spatial'][mouse_recday]=conc_anchor_tuning_spatial

        Anchor_tuning_dic['Perneuron_nonspatial_totalcrossings'][mouse_recday]=np.column_stack((num_crossings_tuned,                                                                                                num_crossings_untuned))
        Anchor_tuning_dic['Perneuron_nonspatial_pertrialcrossings'][mouse_recday]=np.column_stack((            num_crossings_tuned_pertrial,num_crossings_untuned_pertrial))
    except Exception as e:
        print(mouse_recday)
        print(e)

Anchor_tuning_all=np.vstack((dict_to_array(Anchor_tuning_dic['All_sessions_concatenated_nonspatial'])))

peakno_used='num_peaks_all'
index_peakno=1

data = {'num_peaks_all': Anchor_tuning_all[:,1],
                    'num_peaks_state': Anchor_tuning_all[:,2],
                    'num_crossings': Anchor_tuning_all[:,3],
                    'mean_trial_crossings': Anchor_tuning_all[:,4],
                    'num_trials':  Anchor_tuning_all[:,5]}

df = pd.DataFrame(data, columns = ['num_peaks_all', 'num_peaks_state', 'num_crossings','mean_trial_crossings','num_trials'])
#,'spatial_simX'])

corr_stats_=partial_corr(df, peakno_used, 'mean_trial_crossings', covar=['num_trials'])
corr_crossing_tuning=corr_stats_['r'][0]
corr_crossing_tuning_p=corr_stats_['p-val'][0]

sns.regplot(Anchor_tuning_all[:,4],Anchor_tuning_all[:,index_peakno],color='black',           scatter_kws={'alpha':0.3})
plt.savefig(Ephys_output_folder_dropbox+'/Anchorvisit_numpeak_correlation.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(corr_stats_)

corr_stats_=partial_corr(df, peakno_used, 'num_crossings', covar=['num_trials'])
corr_crossing_tuning=corr_stats_['r'][0]
corr_crossing_tuning_p=corr_stats_['p-val'][0]


sns.regplot(Anchor_tuning_all[:,3],Anchor_tuning_all[:,index_peakno])
plt.show()
print(corr_stats_)


# In[ ]:


'''
1-neuron-sessions instead of neurons
2-no effect when averaging per trial - because fewer trials
3-state tuning boolean isnt the same as above (use zmax instead)

'''


# In[ ]:


conc_anchor_tuning_nonspatial_perneuron=Anchor_tuning_dic[mouse_recday][nonspatial_neurons_anchored]
num_crossings=conc_anchor_tuning_nonspatial_perneuron[:,:,3]
num_trials=conc_anchor_tuning_nonspatial_perneuron[:,:,5]
tuning_state=Tuned_dic['State_zmax'][mouse_recday][nonspatial_neurons_anchored]<0.05


# In[ ]:


neuron=0
num_crossings_tuned=np.asarray([np.mean(num_crossings[neuron][tuning_state[neuron]==True])                                for neuron in np.arange(len(num_crossings))])
num_crossings_untuned=np.asarray([np.mean(num_crossings[neuron][tuning_state[neuron]==False])                                for neuron in np.arange(len(num_crossings))])

num_crossings_tuned_pertrial=np.asarray([np.mean(num_crossings[neuron][tuning_state[neuron]==True]/                                       num_trials[neuron][tuning_state[neuron]==True])                                for neuron in np.arange(len(num_crossings))])
num_crossings_untuned_pertrial=np.asarray([np.mean(num_crossings[neuron][tuning_state[neuron]==False]/                                         num_trials[neuron][tuning_state[neuron]==False])                                for neuron in np.arange(len(num_crossings))])


# In[ ]:


neuron=0
num_crossings[neuron][tuning_state[neuron]==True]


# In[ ]:


num_crossings[neuron]


# In[ ]:


tuning_state[neuron]


# In[ ]:


mouse_recday


# In[ ]:


###number of anchor visits for tuned and untuned neuron-session combinations 
##(i.e.) each n is a neuron-session combination
##num_peaks_sign,num_peaks_all,num_peaks_state,num_crossings,mean_trial_crossings,num_trials,mean_nonzero_trial_crossings
###index 2==number of state peaks (i.e. subtracting mean for phase) 

Anchor_tuning_all=np.vstack((dict_to_array(Anchor_tuning_dic['All_sessions_concatenated_nonspatial'])))

min_num_trials=0
num_trials=Anchor_tuning_all[:,5]
Anchor_tuning_subset=Anchor_tuning_all[num_trials>min_num_trials]

print('Number of anchor vists vs tuning')
crossings_untuned=Anchor_tuning_subset[:,3][Anchor_tuning_subset[:,2]==0] ##indx3=number of crossings 
#indx2=number of state peaks
crossings_tuned=Anchor_tuning_subset[:,3][Anchor_tuning_subset[:,2]>0]

bar_plotX([crossings_untuned,crossings_tuned],'none',0,35,'nopoints','unpaired',0.025)
plt.savefig(Ephys_output_folder_dropbox+'/Anchorvisit_tuning_bar.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(st.mannwhitneyu(remove_nan(crossings_untuned),remove_nan(crossings_tuned)))

print('')
print('')
print('Number of anchor vists vs tuning')
crossings_untuned=Anchor_tuning_subset[:,3][Anchor_tuning_subset[:,2]==0]/Anchor_tuning_subset[:,5][Anchor_tuning_subset[:,2]==0]
crossings_tuned=Anchor_tuning_subset[:,3][Anchor_tuning_subset[:,2]>0]/Anchor_tuning_subset[:,5][Anchor_tuning_subset[:,2]>0]

bar_plotX([crossings_untuned,crossings_tuned],'none',0,1.5,'nopoints','unpaired',0.025)
plt.show()
print(st.mannwhitneyu(remove_nan(crossings_untuned),remove_nan(crossings_tuned)))

print('Note: this need not be significant according to the model...as 1 visit vs 3 vists will both give significance ')


# In[ ]:


Perneuron_nonspatial_totalcrossings_all=np.vstack((dict_to_array(Anchor_tuning_dic                                                                 ['Perneuron_nonspatial_totalcrossings'])))    
Perneuron_nonspatial_pertrialcrossings_all=np.vstack((dict_to_array(Anchor_tuning_dic['Perneuron_nonspatial_pertrialcrossings'])))

bar_plotX(Perneuron_nonspatial_totalcrossings_all.T,'none',0,40,'nopoints','unpaired',0.025)
plt.show()
xy=column_stack_clean(Perneuron_nonspatial_totalcrossings_all[:,0],Perneuron_nonspatial_totalcrossings_all[:,1])
print(st.wilcoxon(xy[:,0],xy[:,1]))

bar_plotX(Perneuron_nonspatial_pertrialcrossings_all.T,'none',0,1.5,'nopoints','unpaired',0.025)
plt.show()
xy=column_stack_clean(Perneuron_nonspatial_pertrialcrossings_all[:,0],Perneuron_nonspatial_pertrialcrossings_all[:,1])
print(st.wilcoxon(xy[:,0],xy[:,1]))


# In[ ]:


np.nanmean(Anchor_tuning_subset[:,5][Anchor_tuning_subset[:,2]>0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


GLMvsAnchoring_dic2=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    spatial_neurons=np.where(Tuned_dic['Place'][mouse_recday]==True)[0]
    nonspatial_neurons=np.where(Tuned_dic['Place'][mouse_recday]==False)[0]
    max_peak=Phase_spatial_corr_dic['Max_corr_bin_crossval'][mouse_recday]
    
    Threshold_crossings=Phase_spatial_corr_dic['Threshold_Crossings'][mouse_recday]
    zero_bool=np.zeros(num_neurons)
    num_crossings_arr=np.zeros(num_neurons)
    zero_bool[:]=np.nan
    num_crossings_arr[:]=np.nan
    for neuron in np.arange(num_neurons):
        threshold_crossings_neuron=Threshold_crossings[neuron]
        zero_bool[neuron]=0 in threshold_crossings_neuron
        num_crossings_arr[neuron]=len(threshold_crossings_neuron)
        
    anchor_lag_=Anchor_trial_dic['Anchor_lags_mean'][mouse_recday]
    anchor_dist_=1-np.cos(np.deg2rad(anchor_lag_))

    GLMvsAnchoring_dic2['zero_bool'][mouse_recday]=zero_bool
    GLMvsAnchoring_dic2['Num_crossings'][mouse_recday]=num_crossings_arr
    GLMvsAnchoring_dic2['GLM_spatial'][mouse_recday]=(Tuned_dic['Place'][mouse_recday]).astype(int)
    GLMvsAnchoring_dic2['zero_lag_spatialcorr'][mouse_recday]=max_peak==0
    GLMvsAnchoring_dic2['zero_lag_singleanchor'][mouse_recday]=anchor_dist_<coherence_thr
    
    GLMvsAnchoring_dic2['Anchored_bool'][mouse_recday]=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]


# In[ ]:





# In[ ]:





# In[ ]:





# In[96]:


use_anchored=True

Anchored_neurons=np.hstack((dict_to_array(GLMvsAnchoring_dic2['Anchored_bool'])))
GLM_spatial_all=np.hstack((dict_to_array(GLMvsAnchoring_dic2['GLM_spatial'])))
zero_bool_all=np.hstack((dict_to_array(GLMvsAnchoring_dic2['zero_bool'])))
Num_crossings_all=np.hstack((dict_to_array(GLMvsAnchoring_dic2['Num_crossings'])))
zero_lag_spatialcorr=np.hstack((dict_to_array(GLMvsAnchoring_dic2['zero_lag_spatialcorr']))) 
###is max lagged spatial correlation at 0?
zero_lag_singleanchor=np.hstack((dict_to_array(GLMvsAnchoring_dic2['zero_lag_singleanchor'])))
##is the lag from single anchor less than coherence threshold

if use_anchored==True:
    GLM_spatial_all=GLM_spatial_all[Anchored_neurons]
    zero_bool_all=zero_bool_all[Anchored_neurons]
    Num_crossings_all=Num_crossings_all[Anchored_neurons]
    zero_lag_spatialcorr=zero_lag_spatialcorr[Anchored_neurons]
    zero_lag_singleanchor=zero_lag_singleanchor[Anchored_neurons]
    


print('Proportion of neurons at zero lag from single position spatial anchor')
spatial_zero_prop=np.sum(zero_lag_singleanchor[GLM_spatial_all==1])/len(np.where(GLM_spatial_all==1)[0])
non_spatial_zero_prop=np.sum(zero_lag_singleanchor[GLM_spatial_all==0])/len(np.where(GLM_spatial_all==0)[0])
print(spatial_zero_prop)
print(non_spatial_zero_prop)
print('')

print('Proportion of neurons with one peak at zero lag')
spatial_zero_prop=np.sum(zero_bool_all[GLM_spatial_all==1])/len(np.where(GLM_spatial_all==1)[0])
non_spatial_zero_prop=np.sum(zero_bool_all[GLM_spatial_all==0])/len(np.where(GLM_spatial_all==0)[0])
print(spatial_zero_prop)
print(non_spatial_zero_prop)
print('')

print('Proportion of neurons with main peak at zero lag ')
spatial_zero_prop=np.sum(zero_lag_spatialcorr[GLM_spatial_all==1])/len(np.where(GLM_spatial_all==1)[0])
non_spatial_zero_prop=np.sum(zero_lag_spatialcorr[GLM_spatial_all==0])/len(np.where(GLM_spatial_all==0)[0])
print(spatial_zero_prop)
print(non_spatial_zero_prop)
print('')

#print('Mean number of crossings when anchored')
#spatial_numcrossings=Num_crossings_all[np.logical_and(GLM_spatial_all==1,Num_crossings_all>0)]
#non_spatial_numcrossings=Num_crossings_all[np.logical_and(GLM_spatial_all==0, Num_crossings_all>0)]
#print(np.mean(spatial_numcrossings))
#print(np.mean(non_spatial_numcrossings))
#print('')

print('Proportion of neurons with peak at zero lag when there is only one significant peak')
spatial_zero_onecrossing_prop=np.sum(zero_bool_all[np.logical_and(GLM_spatial_all==1,Num_crossings_all==1)])/len(np.where(np.logical_and(GLM_spatial_all==1,Num_crossings_all==1))[0])
non_spatial_zero_onecrossing_prop=np.sum(zero_bool_all[np.logical_and(GLM_spatial_all==0,Num_crossings_all==1)])/len(np.where(np.logical_and(GLM_spatial_all==0,Num_crossings_all==1))[0])
print(spatial_zero_onecrossing_prop)
print(non_spatial_zero_onecrossing_prop)


print('')
print('___________')
print('Comparing Analyses')
print('')
print('Proportion of neurons that have main spatial correlation peak at zero lag when:')
anchored_zero_prop=np.sum(zero_lag_spatialcorr[zero_lag_singleanchor==1])/len(np.where(zero_lag_singleanchor==1)[0])
nonanchored_zero_prop=np.sum(zero_lag_spatialcorr[zero_lag_singleanchor==0])/len(np.where(zero_lag_singleanchor==0)[0])
print('zero anchored - single anchor')
print(anchored_zero_prop)
print('nonzero anchored - single anchor')
print(nonanchored_zero_prop)
print('')

print('Proportion of cells with zero lag from single position anchor when:')
anchored_zero_prop=np.sum(zero_lag_singleanchor[zero_lag_spatialcorr==1])/len(np.where(zero_lag_spatialcorr==1)[0])
nonanchored_zero_prop=np.sum(zero_lag_singleanchor[zero_lag_spatialcorr==0])/len(np.where(zero_lag_spatialcorr==0)[0])
print('zero lag - spatial correlation')
print(anchored_zero_prop)
print('nonzero lag - spatial correlation')
print(nonanchored_zero_prop)
print('')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###Saving tuning and anchoring booleans
for mouse_recday in day_type_dicX[day_type]:
    print(mouse_recday)


    phase_bool=Tuned_dic['Phase'][mouse_recday]
    state_bool=Tuned_dic['State'][mouse_recday]
    place_bool=Tuned_dic['Place'][mouse_recday]
    Anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]
    
    anchor_lag_=Anchor_trial_dic['Anchor_lags_mean'][mouse_recday]
    anchor_dist_=1-np.cos(np.deg2rad(anchor_lag_))
    
    #anchor_lag_=Anchor_trial_dic['Anchor_lags'][mouse_recday]
    #anchor_dist_=np.asarray([np.mean(1-np.cos(np.deg2rad(anchor_lag_[ii]))) for ii in np.arange(len(anchor_lag_))])
    
    Phase_state_place_anchoring=np.vstack((phase_bool,state_bool,place_bool,Anchored_bool,anchor_dist_)).T
    
    Anchor_trial_dic['Phase_state_place_anchored'][mouse_recday]=Phase_state_place_anchoring
    
    np.save(Intermediate_object_folder+'Phase_state_place_anchored_'+mouse_recday+'.npy',Phase_state_place_anchoring)
    
    
    ###lagged spatial correlation analysis
    num_neurons=len(cluster_dic['good_clus'][mouse_recday])
    mean_corrs=np.nanmean(Phase_spatial_corr_dic['corrs_all'][mouse_recday],axis=2)
    thresholds_=Phase_spatial_corr_dic['Threshold'][mouse_recday]

    np.save(Intermediate_object_folder+'Anchor_lag_'+mouse_recday+'.npy',mean_corrs)
    np.save(Intermediate_object_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy',thresholds_)


# In[ ]:


Phase_state_place_anchored_all=np.vstack((dict_to_array(Anchor_trial_dic['Phase_state_place_anchored'])))
spatial_cell_dist=Phase_state_place_anchored_all[Phase_state_place_anchored_all[:,2]==1,4]
nonspatial_cell_dist=Phase_state_place_anchored_all[Phase_state_place_anchored_all[:,2]==0,4]


bar_plotX(np.asarray([spatial_cell_dist,nonspatial_cell_dist]).T,'none',0,1,'nopoints','unpaired',0.025)
print(st.ttest_ind(remove_nan(spatial_cell_dist),remove_nan(nonspatial_cell_dist)))


# In[ ]:





# In[ ]:





# In[ ]:


'''

1-check lags against matrices
2-cross-reference place glm against zero lag shifted spatial maps
3-cross-reference place glm against lag in anchor lag analysis

'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


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


'''
Note: CAN IMPORVE BELOW- by fixing when neuron has two peaks for same anchor

'''


# In[ ]:





# In[ ]:





# In[ ]:


##Can we predict the current task from activity of neurons anchored to different targets?

Predicted_task_dic=rec_dd()
day_type=='combined_ABCDonly'

thr_upper=330
thr_lower=30

num_phases=3
num_nodes=9
num_states=4

num_trials_tested=5 ##how many trials back to coregress out (to control for autocorrelation in behaviour)
num_trials_neural=0 ##how many trials back to take neuronal activity from (to look at attractor properties)

thr_anchored_GLM=np.nanpercentile(np.hstack((dict_to_array(GLM_anchoring_dic['Predicted_Actual_correlation_mean']))),50)
use_GLM=True ## automatically cross-validated

###applies whether you use GLM or single anchor
use_anchored_only=True

##only used for single anchor
use_mean=True
use_strict_anchored=False
use_cross_validated=False

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for activity_time in ['bump_time']: ##,'decision_time','random_time',90,180,270
        print('____')
        print(str(activity_time))
        print('')
        

        for mouse_recday in day_type_dicX[day_type]:
            #try:
            print(mouse_recday)

            mouse=mouse_recday.split('_',1)[0]
            rec_day=mouse_recday.split('_',1)[1]
            num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
            num_neurons=len(cluster_dic['good_clus'][mouse_recday])

            ##Importing Occupancy
            name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
            data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
            occupancy_ = load(data_filename_memmap)#, mmap_mode='r')

            ##Tasks
            Tasks=np.load(Intermediate_object_folder+'Task_data_'+mouse_recday+'.npy')
            task_numbers_recday=Task_num_dic[mouse_recday]

            ##what are the reference tasks (defined seperately for each neuron and test session)? 
            ref_tasks=Spatial_anchoring_dic['Best_reference_task'][mouse_recday]


            ##defining sessions to use
            sessions=Task_num_dic[mouse_recday]
            num_refses=len(np.unique(sessions))
            num_comparisons=num_refses-1
            repeat_ses=np.where(rank_repeat(sessions)>0)[0]
            non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
            num_nonrepeat_sessions=len(non_repeat_ses)

            ##Is the cell anchored? 
            Anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]
            Anchored_neurons=np.where(Anchored_bool==True)[0]

            ###what is the lag between the neuron's firing and the anchor? 
            Best_shift_time_=Anchor_trial_dic['Best_shift_time'][mouse_recday]
            Best_shift_time_=Anchor_trial_dic['Best_shift_time_mostcommon'][mouse_recday]
            

            ###defining array for regression betas
            norm_FR=np.zeros((num_nonrepeat_sessions,num_nodes))
            mean_dist_locations=np.zeros((num_nonrepeat_sessions,num_nodes))
            mean_angle_locations=np.zeros((num_nonrepeat_sessions,num_nodes))
            predicted_task_all=np.zeros((num_nonrepeat_sessions,4))
            Actual_task_all=np.zeros((num_nonrepeat_sessions,4))



            norm_FR[:]=np.nan
            mean_dist_locations[:]=np.nan
            mean_angle_locations[:]=np.nan
            predicted_task_all[:]=np.nan
            Actual_task_all[:]=np.nan



            ###looping over all sessions (change to used sessions?)
            for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
                print(ses_ind)
                try:
                    
                    if use_cross_validated==True:
                        if use_strict_anchored==True:
                            Anchored_bool=Spatial_anchoring_dic['Anchored_bool_strict_crossval'][mouse_recday][ses_ind_ind]
                        else:
                            Anchored_bool=Spatial_anchoring_dic['Anchored_bool_crossval'][mouse_recday][ses_ind_ind]

                        Anchored_neurons=np.where(Anchored_bool==True)[0]
                        
                        if use_GLM==True:
                            GLM_corr_day=GLM_anchoring_dic['Predicted_Actual_correlation_mean'][mouse_recday]
                            Anchored_neurons=np.where(GLM_corr_day>thr_anchored_GLM)[0]


                    ##What is the task?
                    Task=Tasks[ses_ind]

                    ##How does the animal perform?
                    performance_=dict_to_array(scores_dic[mouse][task_numbers_recday[ses_ind]]['ALL'])
                    performance=performance_[0]

                    ##Are the neurons tuned?
                    Neurons_tuned_ses=np.where(Spatial_anchoring_dic['Neuron_tuned'][mouse_recday][ses_ind_ind]==1)[0]

                    ###defining ephys, occupancy and phase matrices for this session
                    ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')

                    if len(ephys_[0])==0 or len(occupancy_)==0:
                        continue

                    #occupancy_mat=data_matrix(occupancy_[ses_ind])
                    #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
                    
                    location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    if len(location_mat_)==0:
                        continue
                    occupancy_mat=np.reshape(location_mat_,(4,len(location_mat_),len(location_mat_.T)//4))
                    occupancy_conc=np.concatenate(location_mat_)

                    phase_mat=np.zeros(np.shape(occupancy_mat))
                    phase_mat[:,:,30:60]=1
                    phase_mat[:,:,60:90]=2
                    phase_conc=np.concatenate(np.hstack(phase_mat))

                    occupancy_mat_=np.hstack(occupancy_mat)
                    phase_mat_=np.hstack(phase_mat)


                    ##What is the anchor and the lag between neuron and its anchor in reference task
                    
                    if use_GLM==False:
                        best_shift_times_ses=np.zeros(num_neurons)
                        phase_locations_neurons=np.zeros((num_neurons,2))
                        best_shift_times_ses[:]=np.nan
                        phase_locations_neurons[:]=np.nan

                        for neuron in np.arange(num_neurons):
                            ses_ind_ref_task=ref_tasks[ses_ind_ind,neuron]
                            #ses_ind_ref_task=non_repeat_ses[ref_task]
                            #best_shift_times_ses[neuron]=Best_shift_time_[neuron,ses_ind_ref_task]
                            best_shift_times_ses[neuron]=Best_shift_time_[neuron]

                            #phase_neuron_,location_neuron_=((Spatial_anchoring_dic['best_node_phase_used'][mouse_recday]\
                            #                  [:,ses_ind_ref_task,neuron]).astype(int))

                            #location_neuron_,phase_neuron_=(Spatial_anchoring_dic['best_node_phase'][mouse_recday][neuron]).\
                            #astype(int)

                            phase_neuron_,location_neuron_=                            (Spatial_anchoring_dic['Best_anchor_all'][mouse_recday][neuron]).astype(int)
                            location_neuron=int(location_neuron_+1)

                            phase_locations_neurons[neuron]=np.asarray([phase_neuron_,location_neuron])


                            ##sanity check - should never print anything
                            if ses_ind_ref_task==ses_ind_ind:
                                print('Reference task is the same as test task: something went wrong with indexing')
                        if len(phase_locations_neurons)==0:
                            continue

                        ###defining all non-spatial neurons (i.e. neurons with greater than threshold lag from their anchor)
                        nonzero_anchored_neurons=np.where(np.logical_and(best_shift_times_ses>thr_lower,                                                  best_shift_times_ses<thr_upper))[0]                    

                    ###looping over all location-phase conjunctions to find neurons anchored to them 
                    phase_=0
                    for location_ in np.arange(num_nodes):
                        location=location_+1 ## location arrays are not using zero-based indexing 

                        ##defining place/phase visits
                        placephase_mat=(np.logical_and(phase_mat_==phase_,occupancy_mat_==location)).astype(int)
                        placephase_conc=(np.logical_and(phase_conc==phase_,occupancy_conc==location)).astype(int)
                        visits=np.where(placephase_conc==1)[0]
                        if len(visits)==0:
                            continue

                        ##find neurons anchored to this location/phase with non-zero distance
                        
                        if use_GLM==True:
                            Neurons_per_anchor=Anchor_topN_GLM_dic['Neurons_per_anchor'][mouse_recday]                            [ses_ind_ind][phase_][location_]

                            if len(Neurons_per_anchor)>0:
                                neurons_anchorednext=(Neurons_per_anchor[:,0]).astype(int)
                                best_shift_times_=Neurons_per_anchor[:,1]*(360/num_lags)

                                neurons_anchorednext_nonzero=(neurons_anchorednext                                [np.logical_and(best_shift_times_>thr_lower,                                                best_shift_times_<thr_upper)]).astype(int)

                                best_shift_times_nonzero=(best_shift_times_                                [np.logical_and(best_shift_times_>thr_lower,                                                best_shift_times_<thr_upper)]).astype(int)
                                
                            else:
                                continue


                        else:
                            neurons_anchorednext=np.where(np.logical_and(phase_locations_neurons[:,0]==phase_,                                                                         phase_locations_neurons[:,1]==location))[0]####
                            neurons_anchorednext_nonzero=np.intersect1d(neurons_anchorednext,nonzero_anchored_neurons)

                            neurons_anchorednext_nonzero_tuned=np.intersect1d(neurons_anchorednext_nonzero,                                                                              Neurons_tuned_ses)
                            neurons_anchorednext_nonzero_anchored=np.intersect1d(Anchored_neurons,                                                                                 neurons_anchorednext_nonzero)
                            neurons_anchorednext_nonzero_anchored_tuned=np.intersect1d(                                                                            neurons_anchorednext_nonzero_anchored,                                                                                       Neurons_tuned_ses)
                        if use_anchored_only==True:
                            if use_GLM==True:
                                neurons_anchorednext_nonzero_anchored_=np.intersect1d(Anchored_neurons,                                                                                     neurons_anchorednext_nonzero)
                                neurons_used=neurons_anchorednext_nonzero[[neurons_anchorednext_nonzero[ii]                                                    in neurons_anchorednext_nonzero_anchored_                                                  for ii in range(len(neurons_anchorednext_nonzero))]]
                                best_shift_times=best_shift_times_nonzero[[neurons_anchorednext_nonzero[ii]                                                                            in neurons_anchorednext_nonzero_anchored_                                                                        for ii in range(len(neurons_anchorednext_nonzero))]]
                            else:
                                neurons_used=neurons_anchorednext_nonzero_anchored
                        else:
                            neurons_used=neurons_anchorednext_nonzero
                            best_shift_times=best_shift_times_nonzero


                        ##Below we're getting mean activity of neurons at different times before anchor visit
                        ##first defining arrays

                        if len(neurons_used)==0:
                            ##i.e. no neurons anchored to this location/phase
                            continue


                        ##mean first trial activity of all neurons
                        mean_allneurons=np.mean(np.mean(ephys_[:,0],axis=1)/np.mean(np.mean(ephys_,axis=1),axis=1))
                        ##normalised for each neuron by mean activity across all trials

                        ###defining primary independent variable (neurons activity at defined time)
                        ###defining time to take neuron's activity (for primary independent variable)
                        
                        if use_GLM==False:
                            best_shift_times=best_shift_times_ses[neurons_used]####

                        best_shift_times_A=np.zeros(len(neurons_used))
                        best_shift_times_A[:]=np.nan
                        for neuron_ind, neuron in enumerate(neurons_used):
                            ephys_neuron_=ephys_[neuron]
                            #neuron_conc=data_matrix(ephys_neuron_,concatenate=True)      
                            neuron_conc=np.hstack((ephys_neuron_))

                            ###tone aligned activity
                            tone_aligned_activity=ephys_neuron_
                            A_location=Task[0]
                            anchor_mat=(np.logical_and(phase_mat_==phase_,occupancy_mat_==A_location)).astype(int)
                            min_trials=len(anchor_mat)
                            corr_all=np.zeros(360)
                            for shift in np.arange(360):
                                tone_aligned_activity_shifted=np.roll(tone_aligned_activity,-shift)#,axis=1)

                                if use_mean==False:
                                    corr_mat=np.corrcoef(anchor_mat,tone_aligned_activity_shifted)
                                    cross_corr_mat=corr_mat[min_trials:,:min_trials]
                                    corr_all[shift]=np.nanmean(np.diagonal(cross_corr_mat))
                                elif use_mean==True:
                                    mean_neuron=np.mean(tone_aligned_activity_shifted,axis=0)
                                    mean_anchor=np.mean(anchor_mat,axis=0)
                                    corr_all[shift]=st.pearsonr(mean_anchor,mean_neuron)[0]


                            best_shiftA=np.argmax(corr_all)
                            best_shift_times_A[neuron_ind]=best_shiftA

                        angle_diff_anchorvsA=(best_shift_times_A-best_shift_times)%360
                        dist_diff_anchorvsA=np.asarray([1-math.cos(np.deg2rad(angle_diff_anchorvsA[ii]))                                                        for ii in range(len(angle_diff_anchorvsA))])
                        meandist_diff_anchorvsA=np.nanmean(dist_diff_anchorvsA)
                        mean_dist_locations[ses_ind_ind,location_]=meandist_diff_anchorvsA

                        angles = np.deg2rad(angle_diff_anchorvsA)
                        circmean_ = st.circmean(angles)
                        meanangle_diff_anchorvsA= np.rad2deg(circmean_)
                        mean_angle_locations[ses_ind_ind,location_]=meanangle_diff_anchorvsA


                    mean_angles=mean_angle_locations[ses_ind_ind]
                    locations_used_neural=np.where(np.isnan(mean_angles)==False)[0]+1
                    locations_used_intask=np.intersect1d(locations_used_neural,Task)

                    location_angles=np.asarray([mean_angles[int(locations_used_intask[ii]-1)]                                                for ii in range(len(locations_used_intask))])
                    Task_order_angles=np.asarray([int(((location_angles[ii]+45)/360)*4)                                                  for ii in range(len(location_angles))])%4

                    predicted_task=np.repeat(np.nan,len(Task))
                    for ii in range(len(location_angles)):
                        predicted_task[Task_order_angles[ii]]=locations_used_intask[ii]

                    predicted_task_all[ses_ind_ind]=predicted_task
                    Actual_task_all[ses_ind_ind]=Task
                
                except Exception as e:
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print('Not analysed')

            Predicted_task_dic['Predicted_task'][mouse_recday]=predicted_task_all
            Predicted_task_dic['Actual_task'][mouse_recday]=Actual_task_all


# In[ ]:





# In[ ]:


Predicted_tasks_all=np.vstack(dict_to_array(Predicted_task_dic['Predicted_task']))
Actual_tasks_all=np.vstack(dict_to_array(Predicted_task_dic['Actual_task']))

Predicted_tasks_conc=np.hstack((Predicted_tasks_all))
Actual_tasks_conc=np.hstack((Actual_tasks_all))
xy=column_stack_clean(Predicted_tasks_conc,Actual_tasks_conc)

accuracy=len(np.where(xy[:,0]==xy[:,1])[0])/len(xy)
print('Accuracy: '+str(accuracy))

confusion_matrix=np.zeros((4,4))

for Task_ind, Task_ in enumerate(Actual_tasks_all):
    pred_task_=Predicted_tasks_all[Task_ind]
    predicted_indices=np.asarray([[ii,np.where(Task_==pred_task_[ii])[0][0]] for ii in np.arange(len(Task_))                                 if len(np.where(Task_==pred_task_[ii])[0])>0])

    for ii in range(len(predicted_indices)):
        confusion_matrix[predicted_indices[ii,0],predicted_indices[ii,1]]+=1

confusion_matrix_norm=np.vstack(([confusion_matrix[ii]/np.sum(confusion_matrix,axis=1)[ii]                                  for ii in range(len(confusion_matrix))]))

plt.matshow(confusion_matrix_norm)
plt.savefig(Ephys_output_folder_dropbox+'/Task_prediction.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(confusion_matrix_norm)

print('rows=actual; columns=predicted')


# In[ ]:





# In[ ]:


###Sanity check of chance level
x=np.asarray([0,np.nan,2,np.nan])
num_iterations=100
accuracy_bool=np.zeros((num_iterations,len(remove_nan(x))))
accuracy_bool[:]=np.nan
for iteration in range(num_iterations):
    x_copy=np.copy(x)
    np.random.shuffle(x_copy)
    accuracy_bool[iteration]=np.asarray([x_copy[ii]==ii for ii in range(len(x)) if np.isnan(x[ii])==False ])
    
chance_level=len(np.where(np.hstack((accuracy_bool))==1)[0])/len(np.hstack((accuracy_bool)))
print('Chance level: '+str(chance_level))


# In[ ]:





# In[ ]:





# In[ ]:


###Testing regression params

reg_x1= np.repeat(np.asarray([0,1,1,0,0,0]),10)
reg_x2= np.repeat(np.asarray([0,1,0,1,0,1]),10)
reg_x3= np.repeat(np.asarray([0,1,1,1,0,0]),10)

anti_corr_coreg=np.repeat(np.asarray([1,1,0,0,1,1]),10)
corr_coreg=np.repeat(np.asarray([0,0,1,1,0,0]),10)
y= np.repeat(np.asarray([0,0,1,1,0,1]),10)

x1=reg_x3
x1=x1-np.mean(x1)
x2=corr_coreg

X=np.column_stack((x1,x2))
clf = LogisticRegression(fit_intercept=True,solver='liblinear',penalty='l1').fit(X, y)
beta_all=clf.coef_[0]

print(st.pearsonr(x1,y))
print(beta_all)


# In[ ]:





# In[ ]:





# In[ ]:


'''
1-using correct betas (regularised as this is what generalises) - yup
2-check beta anchors and lags against single anchoring analysis -yup
3-why anchored less good at predicting task?
4-check individual examples


'''


# In[ ]:


Intermediate_object_folder_dropbox


# In[ ]:





# In[153]:


## Zero-shot!
##Can we predict zero-shot from activity of neurons anchored to location a? - regression
##note this cell also has the simple FR analysis 

Regression_anchors_zeroshot_dic=rec_dd()
day_type=='combined_ABCDonly'

thr_upper=360
thr_lower=0

num_phases=3
num_nodes=9
num_states=4

num_trials_tested=5 ##how many trials back to coregress out (to control for autocorrelation in behaviour)
num_trials_neural=0 ##how many trials back to take neuronal activity from (to look at attractor properties)
use_GLM=False
use_anchored_only=True

use_strict_anchored=False
match_states_performance=False
lock_to_state=False
use_cross_validated=False

#thr_anchored_GLM=np.nanpercentile(np.hstack((dict_to_array(GLM_anchoring_dic['Predicted_Actual_correlation_mean']))),50)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for state_ind in np.arange(num_states):
        for activity_time in ['bump_time','decision_time','random_time',90,180,270]:
            print('____')
            print(str(activity_time))
            print('')

            X_zeroshot=[]
            y_zeroshot=[]
            mean_allneurons_all=[]
            for mouse_recday in day_type_dicX[day_type]:
                #try:
                print(mouse_recday)

                mouse=mouse_recday.split('_',1)[0]
                rec_day=mouse_recday.split('_',1)[1]



                #Importing Ephys

                num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
                num_neurons=len(cluster_dic['good_clus'][mouse_recday])

               
                ##Importing Occupancy
                #print('Importing Occupancy')
                #try:
                #name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
                #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
                #occupancy_ = load(data_filename_memmap)#, mmap_mode='r')
                #if len(occupancy_)==0:
                #    print('Not found in intermediate object folder - trying temp folder')
                #    temp_folder='C:\Users\moham\OneDrive\Documents\Manuscripts\El-Gaby2023\temp\'
                #    data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
                #    occupancy_ = load(data_filename_memmap)#, mmap_mode='r')
                
                ##Tasks
                Tasks=np.load(Intermediate_object_folder+'Task_data_'+mouse_recday+'.npy')
                task_numbers_recday=Task_num_dic[mouse_recday]

                ##defining sessions to use
                sessions=Task_num_dic[mouse_recday]
                num_refses=len(np.unique(sessions))
                num_comparisons=num_refses-1
                repeat_ses=np.where(rank_repeat(sessions)>0)[0]
                non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
                num_trials=dict_to_array(Num_trials_dic2[mouse_recday])
    
                trials_completed_ses=np.where(num_trials>2)[0]
                non_repeat_ses=np.intersect1d(non_repeat_ses,trials_completed_ses)
                num_nonrepeat_sessions=len(non_repeat_ses)

                ##Is the cell anchored? 
                Anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]
                Anchored_neurons=np.where(Anchored_bool==True)[0]
                
                ###which neurons are state tuned?
                num_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic2['tuning_state_boolean']                                                 [mouse_recday][ses_ind],axis=1)                    for ses_ind in np.arange(len(tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday]))])).T

                ###what is the lag between the neuron's firing and the anchor? 
                Best_shift_time_=Anchor_trial_dic['Best_shift_time'][mouse_recday]
                #Best_shift_time_=Anchor_trial_dic['Best_shift_time_mostcommon'][mouse_recday]

                ###defining array for session_wise FRs and zero-shot performance
                norm_FR_performance=np.zeros((num_nonrepeat_sessions,2))
                norm_FR_performance[:]=np.nan

                neurons_used_all=[]
                FR_time_all=[]
                performance_all=[]
                Trial_length_all=[]

                ###looping over used sessions 
                for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
                    print(ses_ind)
                    neurons_tuned_ses=np.where(num_peaks_all[:,ses_ind_ind]>0)[0]
                    
                    if use_cross_validated==True:
                        if use_strict_anchored==True:
                            Anchored_bool=Spatial_anchoring_dic['Anchored_bool_strict_crossval'][mouse_recday][ses_ind_ind]
                            Anchored_neurons=np.where(Anchored_bool==True)[0]
                        else:
                            Anchored_bool=Spatial_anchoring_dic['Anchored_bool_crossval'][mouse_recday][ses_ind_ind]
                            Anchored_neurons=np.where(Anchored_bool==True)[0]
                    
                    if use_GLM==True:
                        GLM_corr_day=GLM_anchoring_dic['Predicted_Actual_correlation_mean'][mouse_recday]
                        Anchored_neurons=np.where(GLM_corr_day>thr_anchored_GLM)[0]

                    
                    ##What is the task?
                    Task=Tasks[ses_ind]

                    ##How does the animal perform?
                    #performance_=dict_to_array(scores_dic[mouse][task_numbers_recday[ses_ind]]['ALL'])
                    #performance=performance_[0]
                    
                    try:
                        performance=np.load(Data_folder+'Scores_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    except:
                        print('Performance not scored')
                        continue
                    ###Length of trial
                    Trial_times=np.load(Intermediate_object_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    Trial_length=Trial_times[0,-1]

                    ##what is the anchor?
                    #phase_locations_neurons_=((Spatial_anchoring_dic['best_node_phase_used'][mouse_recday]\
                    #                      [:,ses_ind_ind,:]).astype(int)).T
                    
                    #phase_locations_neurons_=Spatial_anchoring_dic['most_common_anchor'][mouse_recday]
                    phase_locations_neurons_=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday]#[neuron]
                    phase_locations_neurons=(np.column_stack((phase_locations_neurons_[:,0],                                                             phase_locations_neurons_[:,1]+1))).astype(int)
                    if len(phase_locations_neurons)==0:
                        continue
                    phase_locations_neurons_unique=np.unique(phase_locations_neurons,axis=0)


                    ###defining all non-spatial neurons (i.e. neurons with greater than threshold lag from their anchor)
                    nonzero_anchored_neurons=np.where(np.logical_and(Best_shift_time_[:,ses_ind_ind]>thr_lower,                                                  Best_shift_time_[:,ses_ind_ind]<thr_upper))[0]
                    
                    #nonzero_anchored_neurons=np.where(np.logical_and(Best_shift_time_>thr_lower,\
                    #                              Best_shift_time_<thr_upper))[0]

                    ###defining ephys, occupancy and phase matrices for this session
                    try:
                        ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    except:
                        print('No Ephys')
                        continue
                    if len(ephys_)==0 :
                        print('No Ephys')
                        continue
                    
                    #occupancy_mat=data_matrix(occupancy_[ses_ind])
                    #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
                    
                    location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    if len(location_mat_)==0:
                        continue
                    occupancy_mat=np.reshape(location_mat_,(4,len(location_mat_),len(location_mat_.T)//4))
                    occupancy_conc=np.concatenate(location_mat_)
                    
                    phase_mat=np.zeros(np.shape(occupancy_mat))
                    phase_mat[:,:,30:60]=1
                    phase_mat[:,:,60:90]=2
                    phase_conc=np.concatenate(np.hstack(phase_mat))

                    occupancy_mat_=np.hstack(occupancy_mat)
                    phase_mat_=np.hstack(phase_mat)

                    ###only using location a at phase=0
                    phase_=0
                    location=Task[state_ind] ##0 for A
                    location_=int(location-1)
                    times_=np.asarray([359])


                    ##find neurons anchored to this location/phase with non-zero distance
                    neurons_anchorednext=np.where(np.logical_and(phase_locations_neurons[:,0]==phase_,                                                                 phase_locations_neurons[:,1]==location))[0]
                    neurons_anchorednext_nonzero=np.intersect1d(neurons_anchorednext,nonzero_anchored_neurons)
                    neurons_anchorednext_nonzero_anchored=np.intersect1d(Anchored_neurons,                                                                         neurons_anchorednext_nonzero)
                    
                    if use_GLM==True:
                        Neurons_per_anchor=Anchor_topN_GLM_crossval_dic['Neurons_per_anchor'][mouse_recday]                        [ses_ind_ind][phase_][location_]

                        if len(Neurons_per_anchor)>0:
                            neurons_anchorednext=(Neurons_per_anchor[:,0]).astype(int)
                            best_shift_times_=Neurons_per_anchor[:,1]*(360/num_lags)

                            neurons_anchorednext_nonzero=(neurons_anchorednext                            [np.logical_and(best_shift_times_>thr_lower,                                            best_shift_times_<thr_upper)]).astype(int)

                            best_shift_times_nonzero=(best_shift_times_                            [np.logical_and(best_shift_times_>thr_lower,                                            best_shift_times_<thr_upper)]).astype(int)

                            best_shift_times=best_shift_times_nonzero

                            if use_anchored_only==True:
                                neurons_anchorednext_nonzero_anchored_=np.intersect1d(Anchored_neurons,                                                                                     neurons_anchorednext_nonzero)
                                neurons_anchorednext_nonzero_anchored=neurons_anchorednext_nonzero[                                                                                    [neurons_anchorednext_nonzero[ii]                                                    in neurons_anchorednext_nonzero_anchored_                                                  for ii in range(len(neurons_anchorednext_nonzero))]]
                                best_shift_times=best_shift_times_nonzero[[neurons_anchorednext_nonzero[ii]                                                                            in neurons_anchorednext_nonzero_anchored_                                                                for ii in range(len(neurons_anchorednext_nonzero))]]

                        else:
                            continue


                    else:
                        neurons_anchorednext=np.where(np.logical_and(phase_locations_neurons[:,0]==phase_,                                                                     phase_locations_neurons[:,1]==location))[0]####
                        neurons_anchorednext_nonzero=np.intersect1d(neurons_anchorednext,nonzero_anchored_neurons)

                        neurons_anchorednext_nonzero_tuned=np.intersect1d(neurons_anchorednext_nonzero,                                                                          neurons_tuned_ses)
                        neurons_anchorednext_nonzero_anchored=np.intersect1d(Anchored_neurons,                                                                             neurons_anchorednext_nonzero)
                        neurons_anchorednext_nonzero_anchored_tuned=np.intersect1d(                                                                        neurons_anchorednext_nonzero_anchored,                                                                                   neurons_tuned_ses)


                    if use_anchored_only==True:
                        neurons_used=neurons_anchorednext_nonzero_anchored
                    else:
                        neurons_used=neurons_anchorednext_nonzero
                    
                    
                    neurons_used=np.intersect1d(neurons_used,neurons_tuned_ses)

                    neurons_used_all.append(neurons_used)


                    ##Below we're getting  mean activity of neurons at different times before anchor visit
                    ##first defining arrays
                    mean_activity_bump_neurons=np.zeros(len(neurons_used))
                    mean_activity_bump_neurons[:]=np.nan
                    activity_time_neurons=np.zeros(len(neurons_used))
                    activity_time_neurons[:]=np.nan
                    
                    if len(neurons_used)==0:
                        ##i.e. no neurons anchored to this location/phase
                        continue

                    ##whats the lag for each neuron to its anchor? 
                    if use_GLM==False:
                        best_shift_times=Best_shift_time_[neurons_used,ses_ind_ind]
                        #best_shift_times=Best_shift_time_[neurons_used]

                    ##mean first trial activity of all neurons
                    mean_allneurons=np.mean(np.mean(ephys_[:,0],axis=1)/np.mean(np.mean(ephys_,axis=1),axis=1))
                    ##normalised for each neuron by mean activity across all trials

                    ###defining primary independent variable (neurons activity at defined time)
                    ###defining time to take neuron's activity (for primary independent variable)
                    for neuron_ind, neuron in enumerate(neurons_used):
                        ephys_neuron_=ephys_[neuron]
                        #neuron_conc=data_matrix(ephys_neuron_,concatenate=True)      
                        neuron_conc=np.hstack((ephys_neuron_))
                        
                        
                        


                        ##defining activity times
                        if activity_time == 'bump_time':
                            gap=(num_trials_neural+1*360)-best_shift_times[neuron_ind]
                            ##i.e. times when neurons should be active on ring attractor
                        elif activity_time == 'decision_time':
                            gap=thr_lower
                            ##i.e. time when animal is about to visit anchor
                        elif activity_time == 'random_time':
                            gap=random.randint(0,359)
                        elif isinstance(activity_time,int)==True:
                            gap=((360-best_shift_times[neuron_ind])+int(activity_time))%360
                            ##times 90 degrees shifted from bump time
                        neuron_bump_time_start_=times_-gap ##definitely looking at activity BEFORE anchor visit
                        neuron_bump_time_start=(neuron_bump_time_start_).astype(int)
                        
                        if lock_to_state==True:
                            neuron_bump_time_start=neuron_bump_time_start-neuron_bump_time_start%90
                            range_bump=90
                        else:
                            range_bump=30

                        ##defining normalised mean activity at selected times (ranging from time to 30 degrees 
                        ##later)
                        mean_activity_bump=np.asarray([np.mean(neuron_conc[neuron_bump_time_start[ii]:                                                                           neuron_bump_time_start[ii]+range_bump])                                                       for ii in range(len(neuron_bump_time_start))])
                        mean_activity_bump[np.isnan(neuron_bump_time_start_)]=np.nan

                        times_int=(times_).astype(int)
                        mean_activity_trial=np.asarray([np.mean(neuron_conc[times_int[ii]:times_int[ii]+360])                                                        for ii in range(len(times_int))])

                        mean_activity_trial[mean_activity_trial==0]=np.nan ##to avoid dividing by zero
                        mean_activity_bump_neurons[neuron_ind]=(mean_activity_bump/mean_activity_trial).squeeze()
                        ##i.e. firing rate as proportion of each neuron's mean firing on a given trial
                        activity_time_neurons[neuron_ind]=neuron_bump_time_start.squeeze()


                    meanofmeans_activity_bump_neurons=                    np.nanmean(mean_activity_bump_neurons) ##3) INDPENDENT VARIABLE
                    ##mean relative firing rate across ALL neurons anchored to a given place/phase
                    ##i.e. collapsing neuron dimension and just keeping visit dimension

                    if len(performance)==0:
                        print('Not analysed')
                        continue


                    X_zeroshot.append(meanofmeans_activity_bump_neurons)
                    
                    

                    performance_first2trials=np.hstack((performance[:2]))

                    if match_states_performance==True:
                        performance_zeroshot=performance_first2trials[int(num_states-1)+state_ind]
                    else:
                        performance_zeroshot=performance_first2trials[int(num_states-1)]


                    y_zeroshot.append(performance_zeroshot)
                    mean_allneurons_all.append(mean_allneurons)
                    
                    FR_time_ses=np.column_stack((mean_activity_bump_neurons,activity_time_neurons))
                    FR_time_all.append(FR_time_ses)
                    performance_all.append(performance_zeroshot)
                    Trial_length_all.append(Trial_length)



                    norm_FR_performance[ses_ind_ind]=np.hstack((meanofmeans_activity_bump_neurons,performance_zeroshot))
                
                
                
                Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['norm_FR_performance'][mouse_recday]=                norm_FR_performance

                Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['neurons_used'][mouse_recday]=                neurons_used_all
                
                Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['FR_time'][mouse_recday]=                FR_time_all
                
                Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['performance'][mouse_recday]=                performance_all
                
                Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['Trial_length'][mouse_recday]=                Trial_length_all

                #except Exception as e:
                #    print(e)
                #    print('Not analysed')

            X_zeroshot=np.hstack((X_zeroshot))
            y_zeroshot=np.hstack((y_zeroshot))
            mean_allneurons_all=np.hstack((mean_allneurons_all))

            X_zeroshot=X_zeroshot[~np.isnan(y_zeroshot)]
            mean_allneurons_all=mean_allneurons_all[~np.isnan(y_zeroshot)]
            y_zeroshot=y_zeroshot[~np.isnan(y_zeroshot)]


            mean_allneurons_all=mean_allneurons_all[~np.isnan(X_zeroshot)]
            y_zeroshot=y_zeroshot[~np.isnan(X_zeroshot)]
            X_zeroshot=X_zeroshot[~np.isnan(X_zeroshot)]

            #Xy_zeroshot_clean=column_stack_clean(X_zeroshot,y_zeroshot)
            #X_zeroshot=Xy_zeroshot_clean[:,0]
            #y_zeroshot=Xy_zeroshot_clean[:,1]

            X_zeroshot_demeaned=X_zeroshot-np.nanmean(X_zeroshot)
            mean_allneurons_all_demeaned=mean_allneurons_all-np.mean(mean_allneurons_all)
            X=np.column_stack((X_zeroshot_demeaned,mean_allneurons_all_demeaned))
            #X_zeroshot_demeaned=X_zeroshot_demeaned.reshape(-1,1)

            clf_zeroshot = LogisticRegression(solver='liblinear',penalty='l1').fit(X, y_zeroshot)
            beta_all=clf_zeroshot.coef_[0]

            Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['All_betas_zeroshot']=beta_all
            Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['norm_FR_zeroshot_all']=X_zeroshot
            Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['mean_first_trial']=mean_allneurons_all
            Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['performance_zeroshot_all']=y_zeroshot


# In[151]:


np.shape(Best_shift_time_)


# In[152]:


non_repeat_ses


# In[ ]:





# In[154]:


state_ind=0
activity_time='bump_time'

FR_time_day_zeroshot_all=[]
FR_time_day_nozeroshot_all=[]
for mouse_recday in day_type_dicX[day_type]:
    FR_time_day=np.asarray(Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['FR_time'][mouse_recday])
    performance_zeroshot_day=np.asarray(Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]                                        ['performance'][mouse_recday])

    FR_time_day_zeroshot=FR_time_day[performance_zeroshot_day==1]
    FR_time_day_nozeroshot=FR_time_day[performance_zeroshot_day==0]
    
    if len(FR_time_day_zeroshot)>0:
         FR_time_day_zeroshot_all.append(np.vstack((FR_time_day_zeroshot)))
        
    if len(FR_time_day_nozeroshot)>0:
        FR_time_day_nozeroshot_all.append(np.vstack((FR_time_day_nozeroshot)))
        
FR_time_day_zeroshot_all=np.vstack((FR_time_day_zeroshot_all))
FR_time_day_nozeroshot_all=np.vstack((FR_time_day_nozeroshot_all))


# In[155]:


plt.scatter(FR_time_day_zeroshot_all[:,1],FR_time_day_zeroshot_all[:,0])
plt.scatter(FR_time_day_nozeroshot_all[:,1],FR_time_day_nozeroshot_all[:,0])


# In[ ]:





# In[156]:


###only plot this when excluding not

bins=[0,90,180,270,360]
xy=column_stack_clean(FR_time_day_zeroshot_all[:,0],FR_time_day_zeroshot_all[:,1])
mean_zeroshot=st.binned_statistic(xy[:,1],xy[:,0],bins=bins)[0]
std_zerosot=st.binned_statistic(xy[:,1],xy[:,0],statistic='std',bins=bins)[0]
count_zeroshot=st.binned_statistic(xy[:,1],xy[:,0],statistic='count',bins=bins)[0]
sem_zeroshot=std_zerosot/np.sqrt(count_zeroshot)

xy=column_stack_clean(FR_time_day_nozeroshot_all[:,0],FR_time_day_nozeroshot_all[:,1])
mean_nozeroshot=st.binned_statistic(xy[:,1],xy[:,0],bins=bins)[0]
std_nozerosot=st.binned_statistic(xy[:,1],xy[:,0],statistic='std',bins=bins)[0]
count_nozeroshot=st.binned_statistic(xy[:,1],xy[:,0],statistic='count',bins=bins)[0]
sem_nozeroshot=std_nozerosot/np.sqrt(count_nozeroshot)

plt.errorbar(np.arange(len(mean_zeroshot)),mean_zeroshot, yerr=sem_zeroshot,color='black')
plt.errorbar(np.arange(len(mean_nozeroshot)),mean_nozeroshot, yerr=sem_nozeroshot,color='red')
plt.savefig(Ephys_output_folder_dropbox+'/Zero_shot_FR_timecourse.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(mean_zeroshot)
print(mean_nozeroshot)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[157]:


##Zero-shot activity per mouse
Mice=[]
for mouse_recday in day_type_dicX[day_type]:
    mouse=mouse_recday.split('_',1)[0]
    Mice.append(mouse)
Mice=np.unique(np.asarray(Mice))

for state_ind in np.arange(num_states):
    for activity_time in ['bump_time','decision_time','random_time',90,180,270]:
        for mouse in Mice:
            try:
                norm_FR_performance_mouse=[]
                for mouse_recday in day_type_dicX[day_type]:
                    if mouse in mouse_recday:
                        norm_FR_performance=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]                        ['norm_FR_performance'][mouse_recday]
                        norm_FR_performance_mouse.append(norm_FR_performance)
                norm_FR_performance_mouse=np.vstack((norm_FR_performance_mouse))



                norm_FR_performance_mouse_zeroshot=norm_FR_performance_mouse[norm_FR_performance_mouse[:,1]==1,0]
                norm_FR_performance_mouse_nozeroshot=norm_FR_performance_mouse[norm_FR_performance_mouse[:,1]==0,0]

                norm_FR_performance_mouse_zeroshot_mean=np.nanmean(norm_FR_performance_mouse_zeroshot)
                norm_FR_performance_mouse_nozeroshot_mean=np.nanmean(norm_FR_performance_mouse_nozeroshot)


                Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['norm_FR_performance_permouse'][mouse]=                norm_FR_performance_mouse

                Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['norm_FR_performance_permouse_mean'][mouse]=                norm_FR_performance_mouse_zeroshot_mean,norm_FR_performance_mouse_nozeroshot_mean

            except:
                print(mouse)
                print('Not analysed')
                continue


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[158]:


###plotting zero shot prediction results
state_ind=0
for activity_time in ['bump_time','decision_time','random_time',90,180,270]:    
    print(activity_time)
    beta_all_=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['All_betas_zeroshot']
    X_zeroshot_=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['norm_FR_zeroshot_all']
    Meanactivity_zeroshot_=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['mean_first_trial']
    y_zeroshot_=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['performance_zeroshot_all']

    Activity_zeroshot=X_zeroshot_[y_zeroshot_==1]
    Activity_nozeroshot=X_zeroshot_[y_zeroshot_==0]

    bar_plotX([Activity_zeroshot,Activity_nozeroshot],'none',0,20,'nopoints','unpaired',0.1)
    plt.savefig(Ephys_output_folder_dropbox+'/'+str(activity_time)+'_Anchored_neurons_Zero_shot_FRnorm.svg',                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print(st.ttest_ind(Activity_zeroshot,Activity_nozeroshot))

Mean_allneuron_activity_zeroshot=Meanactivity_zeroshot_[y_zeroshot_==1]
Mean_allneuron_activity_nozeroshot=Meanactivity_zeroshot_[y_zeroshot_==0]

print('Total activity')
bar_plotX([Mean_allneuron_activity_zeroshot,Mean_allneuron_activity_nozeroshot],'none',0,2,'nopoints','unpaired',0.1)
plt.savefig(Ephys_output_folder_dropbox+'/'+'_ALLneurons_Zero_shot_FRnorm.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(st.ttest_ind(Mean_allneuron_activity_zeroshot,Mean_allneuron_activity_nozeroshot))
print(beta_all_)


# In[159]:


###Trial length control

state_ind=0
activity_time='bump_time'
performance=concatenate_complex2(dict_to_array(Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['performance']))

trial_lengths=concatenate_complex2(dict_to_array(Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]                                   ['Trial_length']))//1000

trial_lengths_zeroshot=trial_lengths[performance==1]
trial_lengths_nozeroshot=trial_lengths[performance==0]

bar_plotX([trial_lengths_zeroshot,trial_lengths_nozeroshot],'none',0,100,'nopoints','unpaired',0.025)
plt.show()
print(st.mannwhitneyu(trial_lengths_zeroshot,trial_lengths_nozeroshot))
print(st.ttest_ind(trial_lengths_zeroshot,trial_lengths_nozeroshot))


# In[160]:


state_ind=0
activity_time = 'bump_time'
Activity_zeroshot_permouse=dict_to_array(Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['norm_FR_performance_permouse_mean'])
bar_plotX(Activity_zeroshot_permouse.T,'none',0,20,'points','paired',0.1)
plt.savefig(Ephys_output_folder_dropbox+'/'+str(activity_time)+'_permouse_Anchored_neurons_Zero_shot_FRnorm.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()

xy=column_stack_clean(Activity_zeroshot_permouse[:,0],Activity_zeroshot_permouse[:,1])
print(st.ttest_rel(xy[:,0],xy[:,1]))

mice_used_zeroshot=list(Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['norm_FR_performance_permouse_mean'].keys())
print(np.column_stack((mice_used_zeroshot,Activity_zeroshot_permouse)))


# In[ ]:





# In[ ]:





# In[161]:


##Control: Activity of neurons anchored to different states

Control_Activity_zeroshot_all=[]
Control_Activity_nozeroshot_all=[]
for state_ind in np.arange(3)+1:    
    activity_time = 'bump_time'
    print(activity_time)
    beta_all_=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['All_betas_zeroshot']
    X_zeroshot_=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['norm_FR_zeroshot_all']
    Meanactivity_zeroshot_=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['mean_first_trial']
    y_zeroshot_=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['performance_zeroshot_all']

    Activity_zeroshot=X_zeroshot_[y_zeroshot_==1]
    Activity_nozeroshot=X_zeroshot_[y_zeroshot_==0]
    
    Control_Activity_zeroshot_all.append(Activity_zeroshot)
    Control_Activity_nozeroshot_all.append(Activity_nozeroshot)

    bar_plotX([Activity_zeroshot,Activity_nozeroshot],'none',0,10,'nopoints','unpaired',0.1)
    plt.show()
    print(st.ttest_ind(Activity_zeroshot,Activity_nozeroshot))

Control_Activity_zeroshot_all=np.hstack((Control_Activity_zeroshot_all))
Control_Activity_nozeroshot_all=np.hstack((Control_Activity_nozeroshot_all))

bar_plotX([Control_Activity_zeroshot_all,Control_Activity_nozeroshot_all],'none',0,10,'points','unpaired',0.1)
plt.savefig(Ephys_output_folder_dropbox+'/'+str(activity_time)+'_Control_neurons_Zero_shot_FRnorm.svg',                bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(st.ttest_ind(Control_Activity_zeroshot_all,Control_Activity_nozeroshot_all))


# In[ ]:





# In[ ]:





# In[ ]:





# In[25]:


file = open(Intermediate_object_folder+'all_mice_exp.pickle','rb')
structure_probabilities_exp = pickle.load(file)
file.close()
step_no=1
N_step_pr=structure_probabilities_exp[mouse][str(step_no)].values
N_step_pr[:,0]=(N_step_pr[:,0]).astype(int)
np.median(N_step_pr[:,3])


# In[ ]:





# In[97]:


##Can we predict the next step from activity of neurons anchored to different targets? - regression
##note this cell also has the simple FR analysis 

Regression_anchors_dic=rec_dd()
day_type='combined_ABCDonly'

step_no=1

num_phases=3
num_nodes=9
num_states=4

num_trials_tested=10 ##how many trials back to regress out (to control for autocorrelation in behaviour)
num_visits_min=20 ##cutoff for how many visits needed before calculating betas

use_kernel=False

use_GLM=False
use_anchored_only=True
use_strict_anchored=False
remove_zero_phase=False
use_tuned_only=False
use_all_neurons=False ##if true this uses all neurons in their bump time, not just ones anchored to the current
##placephase - i.e. this is a control 

use_crossval=False
use_distal=False


scale=False
z_score=False

use_high_pr=False
use_low_pr=False

use_perv0=False
use_perv1=False

remove_neuropixels=False
remove_Cambridge_neurotech=False

if day_type=='combined_ABCDonly':
    num_states=4
    state_degrees=360/num_states
    activity_time_options=['bump_time','decision_time','random_time',int(state_degrees),                           int(state_degrees*2),int(state_degrees*3)]

elif day_type=='combined_ABCDE':
    num_states=5
    state_degrees=360/num_states
    activity_time_options=['bump_time','decision_time','random_time',int(state_degrees),int(state_degrees*2),                           int(state_degrees*3),int(state_degrees*4)]

num_bins=num_states*90
angle_correction=360/num_bins
if use_distal==True:
    thr_lower=(360/num_states)
else:   
    thr_lower=(360/num_states)//3
thr_upper=num_bins-thr_lower


calculate_pr_vsFR=False



#thr_anchored_GLM=np.nanpercentile(np.hstack((dict_to_array(GLM_anchoring_dic['Predicted_Actual_correlation_mean']))),50)
#thr_anchored_GLM=0
if use_strict_anchored==True:
    thr_anchored=np.nanpercentile(np.hstack((dict_to_array(Spatial_anchoring_dic['Cross_val_corr']))),50)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for num_trials_neural in [0]: #[0,1,5,10]:
        ##how many trials back to take neuronal activity from (to look at attractor properties)
        for activity_time in activity_time_options:
            print('____')
            print(str(activity_time))
            print('')


            for mouse_recday in day_type_dicX[day_type]:
                print(mouse_recday)

                mouse=mouse_recday.split('_',1)[0]
                rec_day=mouse_recday.split('_',1)[1]
                cohort=Mice_cohort_dic[mouse]
                

                ephys_type=Cohort_ephys_type_dic[cohort]
                
                abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
                
                if remove_neuropixels==True:
                    if ephys_type=='Neuropixels':
                        continue
                if remove_Cambridge_neurotech==True:
                    if ephys_type=='Cambridge_neurotech':
                        continue

                
                ###Baseline (one-step) transition probabilities
                if (use_high_pr==True or use_low_pr==True) and int(cohort) not in [5,6]:
                    step_no=1
                    N_step_pr=structure_probabilities_exp[mouse][str(step_no)].values
                    N_step_pr[:,0]=(N_step_pr[:,0]).astype(int)
                    median_pr=np.median(N_step_pr[:,3])
                    High_pr_bool=N_step_pr[:,3]>median_pr
                    Low_pr_bool=~High_pr_bool

                #Importing Ephys
                print('Importing Ephys')
                num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
                num_neurons=len(cluster_dic['good_clus'][mouse_recday])


                ##Importing Occupancy
                #print('Importing Occupancy')
                #name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
                #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
                #occupancy_ = load(data_filename_memmap)#, mmap_mode='r')

                ##Tasks
                Tasks=np.load(Intermediate_object_folder+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)
                task_numbers_recday=Task_num_dic[mouse_recday]

                ##defining sessions to use
                sessions=Task_num_dic[mouse_recday]
                
                if day_type=='combined_ABCDE':
                    sessions=sessions[abstract_structures=='ABCDE']

                num_refses=len(np.unique(sessions))
                num_comparisons=num_refses-1
                repeat_ses=np.where(rank_repeat(sessions)>0)[0]
                non_repeat_ses=non_repeat_ses_maker(mouse_recday)  ###this defines the sessions used
                ###only the first session from each task is used
                
                num_trials_ses=[]
                for ses_ind in non_repeat_ses:
                    Trial_times=np.load(Intermediate_object_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    num_trials_ses.append(len(Trial_times))
                
                num_trials_ses=np.hstack((num_trials_ses))
    
                trials_completed_ses=np.where(num_trials_ses>2)[0]
                non_repeat_ses=np.intersect1d(non_repeat_ses,trials_completed_ses)
            


                if day_type=='combined_ABCDE':
                    ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
                    num_ABCD_ses=len(ABCD_sessions)
                    non_repeat_ses=np.setdiff1d(non_repeat_ses,ABCD_sessions)-len(ABCD_sessions)
                    
                    #if num_ABCD_ses>0:
                    #    non_repeat_ses=non_repeat_ses[4:]
                        
                num_nonrepeat_sessions=len(non_repeat_ses)
        
                ##Is the cell anchored? 
                Anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]
                Anchored_neurons=np.where(Anchored_bool==True)[0]

                #360
                
                ###which neurons are phase/state tuned?
                
                if day_type=='combined_ABCDonly':
                    tuning_state_bool_day=tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday]


                elif day_type=='combined_ABCDE':
                    tuning_state_bool_day=np.load(Intermediate_object_folder_dropbox+                                                  'ABCDE_tuning_state_boolean_'+mouse_recday+'.npy')

                num_peaks_all=np.vstack(([np.sum(tuning_state_bool_day[ses_ind],axis=1)                                          for ses_ind in np.arange(len(tuning_state_bool_day))])).T
                #num_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic2['tuning_state_boolean']\
                #                                 [mouse_recday][ses_ind],axis=1)\
                #    for ses_ind in np.arange(len(tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday]))])).T
                
                #phase_tuning=Tuned_dic['Phase'][mouse_recday] 
                #state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday] 
                
                if day_type=='combined_ABCDonly':
                    phase_tuning=Tuned_dic['Phase'][mouse_recday] 
                    state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday]
                elif day_type=='combined_ABCDE':
                    phase_tuning=np.load(Intermediate_object_folder_dropbox+'ABCDE_Phase_'+mouse_recday+'.npy')
                    state_tuning=np.load(Intermediate_object_folder_dropbox+'ABCDE_State_zmax_bool_'+mouse_recday+'.npy')
                
                if use_GLM==True:
                    half_used=GLM_anchoring_dic['half_used_bool'][mouse_recday] 
                    neurons_tuned=np.logical_and(np.logical_and(state_tuning,phase_tuning),half_used)
                else:
                    neurons_tuned=np.logical_and(state_tuning,phase_tuning)
                    

                ###defining array for regression betas
                
                if use_kernel==True:
                    num_trials_tested_arraysize=1
                else:
                    num_trials_tested_arraysize=num_trials_tested
                    
                all_betas_allanchors=np.zeros((num_nonrepeat_sessions,num_phases,num_nodes,                                               int(num_trials_tested_arraysize+1)))
                neuron_betas_allanchors=np.zeros((num_nonrepeat_sessions,num_phases,num_nodes))
                norm_FR_nonrepeat=np.zeros((num_nonrepeat_sessions,num_phases,num_nodes,2))
                norm_FR_allcombinations=np.zeros((num_nonrepeat_sessions,num_phases,num_nodes,4))
                mean_activity_ses_neurons=np.zeros((num_nonrepeat_sessions,num_neurons))
                num_anchorvisitsnorm_neurons=np.zeros((num_nonrepeat_sessions,num_neurons))


                all_betas_allanchors[:]=np.nan
                neuron_betas_allanchors[:]=np.nan
                norm_FR_nonrepeat[:]=np.nan
                norm_FR_allcombinations[:]=np.nan
                mean_activity_ses_neurons[:]=np.nan
                num_anchorvisitsnorm_neurons[:]=np.nan
                
                xy_11_all=[]
                xy_01_all=[]

                ###looping over all sessions (change to used sessions?)
                for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
                    
                    #neurons_tuned_ses=np.where(num_peaks_all[:,ses_ind_ind]>0)[0]
                    neurons_tuned_ses=np.where(neurons_tuned==True)[0]
                    
                    ##Is the neuron anchored
                    if use_strict_anchored==True:
                        Anchored_neurons_1=np.where(Spatial_anchoring_dic['Cross_val_corr'][mouse_recday]>thr_anchored)[0]
                        Anchored_bool=Spatial_anchoring_dic['Anchored_bool_strict_crossval'][mouse_recday][ses_ind_ind]
                        Anchored_neurons_2=np.where(Anchored_bool==True)[0]
                        #Anchored_neurons=np.intersect1d(Anchored_neurons_1,Anchored_neurons_2)
                        Anchored_neurons=Anchored_neurons_2
                    else:
                        Anchored_neurons_1=np.where(Spatial_anchoring_dic['Cross_val_corr'][mouse_recday]>0)[0]
                        Anchored_bool=Spatial_anchoring_dic['Anchored_bool_crossval'][mouse_recday][ses_ind_ind]
                        Anchored_neurons_2=np.where(Anchored_bool==True)[0]
                        #Anchored_neurons=np.intersect1d(Anchored_neurons_1,Anchored_neurons_2)
                        Anchored_neurons=Anchored_neurons_2
                        
                    if use_GLM==True:
                        GLM_corr_day=GLM_anchoring_dic['Predicted_Actual_correlation_mean'][mouse_recday]
                        Anchored_neurons=np.where(GLM_corr_day>thr_anchored_GLM)[0]

                    ##What is the task?
                    Task=Tasks[ses_ind]

                    ##How does the animal perform?
                    #performance_=dict_to_array(scores_dic[mouse][task_numbers_recday[ses_ind]]['ALL'])
                    #if len(scores_dic[mouse][task_numbers_recday[ses_ind]]['ALL'].keys())>1:
                    #performance=performance_[0]
                    #else:
                    #    performance=performance_

                    ##what is the anchor?
                    if use_crossval==True:
                        try:
                            phase_locations_neurons_=((Spatial_anchoring_dic['best_node_phase_used'][mouse_recday]                                              [:,ses_ind_ind,:]).astype(int)).T
                        except:
                            print('spatial anchors not found')
                            continue
                        ###what is the lag between the neuron's firing and the anchor? 
                        Best_shift_time_=Anchor_trial_dic['Best_shift_time_cross_val'][mouse_recday]

                    else:
                        #phase_locations_neurons_=Spatial_anchoring_dic['most_common_anchor'][mouse_recday]
                        phase_locations_neurons_=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday]
                        
                        ###what is the lag between the neuron's firing and the anchor? 
                        Best_shift_time_=Anchor_trial_dic['Best_shift_time'][mouse_recday]
                    
                    phase_locations_neurons=(np.column_stack((phase_locations_neurons_[:,0],                                                             phase_locations_neurons_[:,1]+1))).astype(int)
                    if len(phase_locations_neurons)==0:
                        continue
                    phase_locations_neurons_unique=np.unique(phase_locations_neurons,axis=0)


                    ###defining all non-spatial neurons (i.e. neurons with greater than threshold lag from their anchor)
                    nonzero_anchored_neurons=np.where(np.logical_and(Best_shift_time_[:,ses_ind_ind]>thr_lower,                                                  Best_shift_time_[:,ses_ind_ind]<thr_upper))[0]


                    ###defining ephys, occupancy and phase matrices for this session
                    try:
                        ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    except:
                        print('No Ephys')
                        continue
                    if len(ephys_)==0 :
                        print('No Ephys')
                        continue

                    if len(ephys_[0])==0:
                        continue

                    #occupancy_mat=data_matrix(occupancy_[ses_ind])
                    #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
                    
                    location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    occupancy_mat=np.reshape(location_mat_,(num_states,len(location_mat_),                                                            len(location_mat_.T)//num_states))
                    occupancy_conc=np.concatenate(location_mat_)

                    phase_mat=np.zeros(np.shape(occupancy_mat))
                    phase_mat[:,:,30:60]=1
                    phase_mat[:,:,60:90]=2
                    phase_conc=np.concatenate(np.hstack(phase_mat))

                    occupancy_mat_=location_mat_
                    phase_mat_=np.hstack(phase_mat)

                    num_trials=len(occupancy_mat_)


                    neurons_used_day=[]
                    ###looping over all location-phase conjunctions to find neurons anchored to them and do the regression
                    for phase_ in np.arange(num_phases):
                        if phase_==0 and remove_zero_phase==True:
                            continue
                        for location_ in np.arange(num_nodes):
                            location=location_+1 ## location arrays are not using zero-based indexing 

                            #elif zero_shot==True:
                            #visits_start=visits_start_[visits_start_<360+31] ###only including first trial
                            #visits_prev_end=visits_prev_end[visits_prev_end<360+31] ###only including first trial

                            ##defining place/phase visits
                            placephase_mat=(np.logical_and(phase_mat_==phase_,occupancy_mat_==location)).astype(int)
                            placephase_conc=(np.logical_and(phase_conc==phase_,occupancy_conc==location)).astype(int)
                            visits=np.where(placephase_conc==1)[0]
                            if len(visits)==0:
                                continue

                            ###where are the location/phase conjunctions one step away from the anchor?
                            ##This defines decision times
                            prev_locations_=np.where(mindistance_mat[location_]==1)[0]
                            prev_locations=prev_locations_+1
                            
                            if int(cohort) in [5,6]:
                                trans_mat_mean=np.load(Intermediate_object_folder_dropbox+'/'+                                mouse+'_Exploration_Transition_matrix.npy')
                                probs=[]
                                for portX_ in np.arange(9):
                                    for portY_ in np.where(mindistance_mat[portX_]==1)[0]:
                                        probs.append(trans_mat_mean[step_no-1][portX_,portY_])

                                probs_median=np.median(probs)

                                if use_high_pr==True:
                                    prev_loc_bool=np.hstack(([trans_mat_mean[step_no-1][prev_location-1,location-1]                                                              >=probs_median for prev_location in prev_locations]))
                                elif use_low_pr==True:
                                    prev_loc_bool=np.hstack(([trans_mat_mean[step_no-1][prev_location-1,location-1]                                                              <probs_median for prev_location in prev_locations]))
                            else:
                                if use_high_pr==True:
                                    prev_loc_bool=np.array([High_pr_bool[np.logical_and(N_step_pr[:,0]==                                                    prev_locations[ii], N_step_pr[:,1]==location)][0]                                                          for ii in range(len(prev_locations))])

                                elif use_low_pr==True:
                                    prev_loc_bool=np.array([Low_pr_bool[np.logical_and(N_step_pr[:,0]==                                                    prev_locations[ii], N_step_pr[:,1]==location)][0]                                                          for ii in range(len(prev_locations))])
                            if use_high_pr==True or use_low_pr==True:
                                prev_locations=prev_locations[prev_loc_bool]

                            if len(prev_locations)==0:
                                continue
                            
                            prev_phase_=(phase_-1)%3
                            location_conc_prev_=np.max(np.asarray([occupancy_conc==prev_locations[ii] for ii                                                          in range(len(prev_locations))]),axis=0)
                            location_conc_prev=location_conc_prev_>0 ##i.e. when visiting location one step from 
                            ##anchor location
                            placephase_conc_prev=(np.logical_and(phase_conc==prev_phase_                                                                 ,location_conc_prev==True)).astype(int)
                            #i.e. when visiting location one step from anchor location AND phase one step 
                            ##from anchor phase 
                            visits_prev=np.where(placephase_conc_prev==1)[0]
                            if len(visits_prev)==0:
                                continue
                            visits_start_=visits[np.hstack((num_bins,np.diff(visits)))>30] ##only the start of a visits 
                            ##in a given phase
                            visits_prev_end=visits_prev[np.hstack((np.diff(visits_prev),num_bins))>30] 
                            ##only the end of a visit in a given phase


                            visits_start=visits_start_[visits_start_>=num_bins] ###removing first trial
                            visits_prev_end=visits_prev_end[visits_prev_end>num_bins] ###removing first trial
                            visits_prev_end_nodes=occupancy_conc[visits_prev_end] 
                            ##which nodes are visited at decision points
                            
                            if len(visits_prev_end)<num_visits_min:
                                continue

                            ##where does the animal actually go after the decision points i.e when one step away
                            ##from anchor (visits_prev_end) - it can go to anchor or not
                            visited_node_start=np.zeros((len(visits_prev_end),2))
                            visited_node_start[:]=np.nan
                            for visit_prev_ind, visit_prev in enumerate(visits_prev_end):
                                next_90_occ=occupancy_conc[visit_prev+1:visit_prev+91]
                                next_90_phase=phase_conc[visit_prev+1:visit_prev+91]

                                next_node_=next_90_occ[np.logical_and(next_90_occ<10,next_90_occ!=                                                                     visits_prev_end_nodes[visit_prev_ind])]
                                ##only taking nodes (not edges) and only nodes that arent the same as the previous nodes
                                ##(because could have moved phases but stayed in same node)

                                ###excluding trials where animal stayed in same location across two phases
                                ###because then cant define decision point
                                if len(next_90_occ)==0 or len(next_node_)==0: 
                                    continue
                                next_node=next_node_[0] ###the very first node visited after decision point
                                next_node_phase_start_=np.where(np.logical_and(next_90_phase==phase_,                                                                               next_90_occ==next_node))[0]
                                ##when is next node visited at the next phase

                                ##whats the first bin where the next node is visited in the next phase  
                                if len(next_node_phase_start_)==0 or                                mindistance_mat[int(visits_prev_end_nodes[visit_prev_ind]-1),int(next_node-1)]!=1:
                                    ##note: the second condition is to deal with erroneous tracking where animal position
                                    ##jumps more than one node between bins
                                    next_node=np.nan
                                    next_node_phase_start=np.nan
                                else:
                                    next_node_phase_start=next_node_phase_start_[0]

                                visited_node_start[visit_prev_ind]=np.asarray([next_node,visit_prev+next_node_phase_start])

                            ###we now have visited_node_start which has both the visited node and its timestamp following
                            #each decision point (decision point being when animal was one place and one phase 
                            ##away from anchor)


                            ##Defining dependent variable (location/phase visits)                            
                            location_visited_bool=(visited_node_start[:,0]==location).astype(int) ## 1) DEPENDENT VARIABLE
                            
                            
                            ##i.e. when did animal visit the anchor after each decision point
                            ##the length of this array is all the times where animal was one location and one phase away
                            ##from anchor
                            times_=visited_node_start[:,1] ##times for dependent variable
                            nodes_=visited_node_start[:,0] ##nodes visited for dependent variable

                            ### Did animal visit the anchor location on N previous trials?
                            ## we regress this out to remove any effects of autocorrelation in behaviour 
                            trial_lag_booleans_all=np.zeros((num_trials_tested,len(times_)))
                            trial_lag_booleans_all[:]=np.nan
                            for trial_lag_ in np.arange(num_trials_tested):
                                trial_lag=trial_lag_+1
                                range_tested=[(num_bins*trial_lag)-30,                                              (num_bins*trial_lag)+30]
                                ##what range of time lags should we use to look for previous trial visits 
                                ##using num_trials_neural here when predicting behaviour using neural activity from 
                                ##M trials back, this means now the coregressors (animal's previous choices) for the 
                                ##behaviour are lagged by exactly N+M trials back (with tolerance of +/- 30 degrees))


                                ## for each bin in times_ (i.e. each timestamp following each decision point) what (if 
                                ##any) is the bin where animal visited the same location/phase N+M trials back
                                ##Note: for trials less then N+M you will effectively never have visited the same 
                                ##anchor at this trial lag and so the co-regressors for this trial lag will always be 
                                ##zero 
                                visit_at_lag=[np.where(np.logical_and((times_<(times_[ii]-range_tested[0])),                                                                      (times_>(times_[ii]-range_tested[1]))))[0]                                for ii in range(len(times_))]

                                ###did animal visit the anchor location M+N trials before the current visit time?
                                trial_lag_boolean=np.asarray([np.sum([nodes_[visit_at_lag[ii][jj]]==location                                                                      for jj in range(len(visit_at_lag[ii]))])                                            if len(visit_at_lag[ii]>0) else 0 for ii in range(len(nodes_))])
                                trial_lag_boolean[trial_lag_boolean>0]=1
                                trial_lag_booleans_all[trial_lag_]=trial_lag_boolean ## 2) CO-REGRESSORS

                                ###Now you have trial_lag_booleans_all which tells you when you visited place/phase 
                                ##anchor exactly N+M trials in the past for each N (and a fixed M) - note M=0 for the 
                                ##main analysis



                            ##find neurons anchored to this location/phase with non-zero distance
                            
                            if use_GLM==True:
                                Neurons_per_anchor=Anchor_topN_GLM_crossval_dic['Neurons_per_anchor'][mouse_recday]                                [ses_ind_ind][phase_][location_]
                                
                                #Neurons_per_anchor=Anchor_topN_GLM_dic['Neurons_per_anchor'][mouse_recday]\
                                #[phase_][location_]

                                if len(Neurons_per_anchor)>0:
                                    neurons_anchorednext=(Neurons_per_anchor[:,0]).astype(int)
                                    best_shift_times_=Neurons_per_anchor[:,1]*(num_bins/num_lags)

                                    neurons_anchorednext_nonzero=(neurons_anchorednext                                    [np.logical_and(best_shift_times_>thr_lower,                                                    best_shift_times_<thr_upper)]).astype(int)

                                    best_shift_times_nonzero=(best_shift_times_                                    [np.logical_and(best_shift_times_>thr_lower,                                                    best_shift_times_<thr_upper)]).astype(int)

                                    best_shift_times=best_shift_times_nonzero

                                    if use_anchored_only==True:
                                        neurons_anchorednext_nonzero_anchored_=np.intersect1d(Anchored_neurons,                                                                                             neurons_anchorednext_nonzero)
                                        neurons_anchorednext_nonzero_anchored=neurons_anchorednext_nonzero[                                                                                    [neurons_anchorednext_nonzero[ii]                                                            in neurons_anchorednext_nonzero_anchored_                                                          for ii in range(len(neurons_anchorednext_nonzero))]]
                                        best_shift_times=best_shift_times_nonzero[[neurons_anchorednext_nonzero[ii]                                                                         in neurons_anchorednext_nonzero_anchored_                                                            for ii in range(len(neurons_anchorednext_nonzero))]]

                                else:
                                    continue


                            else:
                                neurons_anchorednext=np.where(np.logical_and(phase_locations_neurons[:,0]==phase_,                                                                        phase_locations_neurons[:,1]==location))[0]####
                                neurons_anchorednext_nonzero=np.intersect1d(neurons_anchorednext,nonzero_anchored_neurons)

                                neurons_anchorednext_nonzero_tuned=np.intersect1d(neurons_anchorednext_nonzero,                                                                                  neurons_tuned_ses)
                                neurons_anchorednext_nonzero_anchored=np.intersect1d(Anchored_neurons,                                                                                     neurons_anchorednext_nonzero)
                                neurons_anchorednext_nonzero_anchored_tuned=np.intersect1d(                                                                                neurons_anchorednext_nonzero_anchored,                                                                                           neurons_tuned_ses)
                            ##i.e. neurons that have the same anchor for half the tasks or more 

                            if use_anchored_only==True:
                                neurons_used=neurons_anchorednext_nonzero_anchored
                            else:
                                neurons_used=neurons_anchorednext_nonzero
                            
                            if use_tuned_only==True:
                                neurons_used=np.intersect1d(neurons_used,neurons_tuned_ses)
                            
                            if use_all_neurons==True:
                                neurons_used=np.arange(num_neurons)
                            ##Below we're getting  mean activity of neurons at different times before anchor visit
                            ##first defining arrays
                            mean_activity_bump_neurons=np.zeros((len(neurons_used),len(location_visited_bool)))

                            mean_activity_bump_neurons[:]=np.nan


                            if len(neurons_used)==0:
                                ##i.e. no neurons anchored to this location/phase
                                all_betas_allanchors[ses_ind_ind,phase_,location_]=np.repeat(np.nan,                                                                                num_trials_tested_arraysize+1)
                                neuron_betas_allanchors[ses_ind_ind,phase_,location_]=np.nan
                                continue

                            ##whats the lag for each neuron to its anchor? 
                            if use_GLM==False:
                                best_shift_times=Best_shift_time_[neurons_used,ses_ind_ind]
                                #best_shift_times=Best_shift_time_[neurons_used]

                            ##mean first trial activity of all neurons
                            mean_allneurons=np.mean(np.mean(ephys_[:,0],axis=1)/np.mean(np.mean(ephys_,axis=1),axis=1))
                            ##normalised for each neuron by mean activity across all trials
                            
                            

                            ###defining primary independent variable (neurons activity at defined time)
                            ###defining time to take neuron's activity (for primary independent variable)
                            neurons_used_day.append(neurons_used)
                            for neuron_ind, neuron in enumerate(neurons_used):
                                ephys_neuron_=ephys_[neuron]
                                neuron_conc=np.hstack((ephys_neuron_))

                                ##defining activity times
                                if activity_time == 'bump_time':
                                    gap=num_bins-best_shift_times[neuron_ind]
                                    ##i.e. times when neurons should be active on ring attractor
                                elif activity_time == 'decision_time':
                                    gap=thr_lower
                                    ##i.e. time when animal is about to visit anchor
                                elif activity_time == 'random_time':
                                    gap=random.randint(0,num_bins-1)
                                elif isinstance(activity_time,int)==True:
                                    gap=((num_bins-best_shift_times[neuron_ind])+int(activity_time))%num_bins
                                    ##times 90 degrees shifted from bump time
                                neuron_bump_time_start_=times_-gap ##definitely looking at activity BEFORE anchor visit                               
                                neuron_bump_time_start=(neuron_bump_time_start_).astype(int)


                                ##defining normalised mean activity at selected times (ranging from time to 30 degrees 
                                ##later)
                                mean_activity_bump=np.asarray([np.mean(neuron_conc[neuron_bump_time_start[ii]:                                                                                   neuron_bump_time_start[ii]+30])                                                               for ii in range(len(neuron_bump_time_start))])
                                mean_activity_bump[np.isnan(neuron_bump_time_start_)]=np.nan


                                times_int=(times_).astype(int)
                                mean_activity_trial=np.asarray([np.mean(neuron_conc                                                                        [times_int[ii]:times_int[ii]+num_bins])                                                                for ii in range(len(times_int))])

                                #mean_activity_trial=remove_nan(mean_activity_trial)
                                mean_activity_trial[mean_activity_trial==0]=np.nan ##to avoid dividing by zero
                                mean_activity_bump_neurons[neuron_ind]=mean_activity_bump/mean_activity_trial
                                ##i.e. firing rate as proportion of each neuron's mean firing on a given trial

                                mean_activity_ses=np.nanmean(neuron_conc)
                                mean_activity_ses_neurons[ses_ind_ind,neuron]=mean_activity_ses
                                num_anchorvisitsnorm_neurons[ses_ind_ind,neuron]=len(times_)/num_trials


                            meanofmeans_activity_bump_neurons=                            np.nanmean(mean_activity_bump_neurons,axis=0) ##3) INDPENDENT VARIABLE
                            ##mean relative firing rate across ALL neurons anchored to a given place/phase
                            ##i.e. collapsing neuron dimension and just keeping visit dimension


                            ## Final inputs to regression
                            X_=np.column_stack((meanofmeans_activity_bump_neurons,trial_lag_booleans_all.T))
                            y_=location_visited_bool


                            X=X_[~np.isnan(meanofmeans_activity_bump_neurons)]
                            #X=np.vstack(([X[:,ii]-np.mean(X[:,ii]) for ii in range(len(X.T))])).T
                            X=np.column_stack((X[:,0]-np.mean(X[:,0]),X[:,1:])) ##de-meaning neuronal activity
                            y=y_[~np.isnan(meanofmeans_activity_bump_neurons)]
                            
                            

                            ###lagging by num_trials_neural
                            times__=times_[~np.isnan(meanofmeans_activity_bump_neurons)]##updated times removing nans
                            if len(times__)==0:
                                continue
                            times_shifted_=times__+num_bins*num_trials_neural
                            indices_trial_lagged_=[[ii,np.where(np.logical_and(times_shifted_[ii]<                                                                                         times__+30,times_shifted_[ii]>                                                                                         times__-30))[0][0]]                                        for ii in range(len(times_shifted_))                                        if len(np.where(np.logical_and(times_shifted_[ii]<times__+30,                                                                       times_shifted_[ii]>times__-30))[0])>0]
                            if len(indices_trial_lagged_)==0:
                                continue
                            indices_trial_lagged=np.vstack((indices_trial_lagged_))

                            X=np.column_stack((X[indices_trial_lagged[:,0],0],X[indices_trial_lagged[:,1],1:]))
                            y=y[indices_trial_lagged[:,1]]
                            
                            prev0_bool=X[:,2]==0
                            prev1_bool=X[:,2]==1
                            
                            if use_perv0==True:
                                X=X[prev0_bool]
                                y=y[prev0_bool]
                            elif use_perv1==True:
                                X=X[prev1_bool]
                                y=y[prev1_bool]
                                
                            if len(X)==0:
                                continue
                            
                            if use_kernel==True:
                                behaviour_coefficients=np.load(Intermediate_object_folder_dropbox+                                                               '_previous_chocies_coefficients_ABCD.npy')
                                xdata=np.arange(len(behaviour_coefficients))
                                ydata=behaviour_coefficients
                                popt, pcov = curve_fit(func_decay, xdata, ydata)

                                xdata_new=np.arange(num_trials_tested)
                                y_pred=func_decay(xdata_new, *popt)
                                X_beh=np.mean(X[:,1:]*y_pred,axis=1)
                                X=np.column_stack((X[:,0],X_beh))
                            
                            if scale==True:
                                transformer = MaxAbsScaler().fit(X)
                                X=transformer.transform(X)

                            if z_score==True:
                                X=st.zscore(X,axis=0)
                                X=X[~np.isnan(np.mean(X,axis=1))]
                                y=y[~np.isnan(np.mean(X,axis=1))]

                            ##doing the regression
                            if np.sum(abs(np.diff(y)))==0 or len(X)==0: ### i.e. always visited (or always didnt visit) 
                                ##place/phase from all decision points
                                beta_all=np.repeat(np.nan,num_trials_tested_arraysize+1)
                                beta_neurons=np.nan
                            else:
                                clf = LogisticRegression(solver='saga',penalty=None).fit(X, y) 
                                ##,max_iter=10000
                                beta_all=clf.coef_[0]
                                beta_neurons=clf.coef_[0][0]

                            all_betas_allanchors[ses_ind_ind,phase_,location_]=beta_all
                            neuron_betas_allanchors[ses_ind_ind,phase_,location_]=beta_neurons
                            
                            
                            if use_perv1!=True and use_perv0!=True:
                                
                                ###simple analysis to check for policy effect - do anchored neurons fire more in trial 
                                ##before animal goes to anchor even when controlling for previous choice
                                X=X_[~np.isnan(meanofmeans_activity_bump_neurons)]
                                X=np.column_stack((X[indices_trial_lagged[:,0],0],X[indices_trial_lagged[:,1],1:]))
                                non_repeat_visits=X[:,1]!=y
                                X_nonrepeat=X[non_repeat_visits,0]
                                y_nonrepeat=y[non_repeat_visits]


                                mean_rates_01=np.mean(X_nonrepeat[y_nonrepeat==1])
                                mean_rates_10=np.mean(X_nonrepeat[y_nonrepeat==0])

                                repeat_visits=X[:,1]==y
                                X_repeat=X[repeat_visits,0]
                                y_repeat=y[repeat_visits]
                                mean_rates_11=np.mean(X_repeat[y_repeat==1])
                                mean_rates_00=np.mean(X_repeat[y_repeat==0])

                                norm_FR_nonrepeat[ses_ind_ind,phase_,location_]=mean_rates_01,mean_rates_10
                                norm_FR_allcombinations[ses_ind_ind,phase_,location_]=                                mean_rates_00,mean_rates_01,mean_rates_10,mean_rates_11


                                prev_nodes_=occupancy_conc[visits_prev_end]
                                prev_nodes=prev_nodes_[~np.isnan(meanofmeans_activity_bump_neurons)]
                                ##updated pre nodes removing nans

                                if (use_high_pr==True or use_low_pr==True) and calculate_pr_vsFR==True:
                                    baseline_pr=np.array([N_step_pr[np.logical_and(N_step_pr[:,0]==prev_nodes[ii],                                    N_step_pr[:,1]==location),3][0] for ii in range(len(prev_nodes))])

                                    bool_11=np.logical_and(y==1,repeat_visits.astype(int)==1)
                                    bool_01=np.logical_and(y==1,repeat_visits.astype(int)==0)
                                    baseline_pr_11=baseline_pr[bool_11]
                                    baseline_pr_01=baseline_pr[bool_01]
                                    FR_11=X[bool_11,0]
                                    FR_01=X[bool_01,0]

                                    xy_11=np.column_stack((baseline_pr_11,FR_11))
                                    xy_01=np.column_stack((baseline_pr_01,FR_01))

                                    xy_11_all.append(xy_11)
                                    xy_01_all.append(xy_01)
                            
                            
                if len(xy_11_all)>0:
                    xy_11_all=np.vstack((xy_11_all))
                if len(xy_01_all)>0:
                    xy_01_all=np.vstack((xy_01_all))
                
                Regression_anchors_dic[str(activity_time)][num_trials_neural]['All_betas'][mouse_recday]=                all_betas_allanchors
                Regression_anchors_dic[str(activity_time)][num_trials_neural]['neuron_betas'][mouse_recday]=                neuron_betas_allanchors
                
                if use_perv1!=True and use_perv0!=True:
                    Regression_anchors_dic[str(activity_time)][num_trials_neural]['norm_FR_nonrepeat'][mouse_recday]=                    norm_FR_nonrepeat
                    Regression_anchors_dic[str(activity_time)][num_trials_neural]['norm_FR_allcombinations']                    [mouse_recday]=norm_FR_allcombinations

                    Regression_anchors_dic[str(activity_time)][num_trials_neural]['pr_vs_norm_FR_11'][mouse_recday]=                    xy_11_all
                    Regression_anchors_dic[str(activity_time)][num_trials_neural]['pr_vs_norm_FR_01'][mouse_recday]=                    xy_01_all

                
                if len(neurons_used_day)>0:
                    num_neurons_used_day=len(np.unique(np.hstack((neurons_used_day))))
                else:
                    num_neurons_used_day=0
                    
                Regression_anchors_dic[str(activity_time)][num_trials_neural]['num_neurons_used'][mouse_recday]=                num_neurons_used_day

                if activity_time=='bump_time':
                    Regression_anchors_dic[num_trials_neural]['mean_activity'][mouse_recday]=mean_activity_ses_neurons
                    Regression_anchors_dic[num_trials_neural]['num_anchor_visits'][mouse_recday]=                    num_anchorvisitsnorm_neurons


# In[23]:


non_repeat_ses


# In[ ]:





# In[98]:


num_days=0
num_neurons_all=0
activity_time='bump_time'
for mouse_recday in day_type_dicX[day_type]:
    regcoefs_=Regression_anchors_dic[num_trials_neural]['mean_activity'][mouse_recday]
    num_neurons= Regression_anchors_dic[str(activity_time)][num_trials_neural]['num_neurons_used'][mouse_recday]
    if len(regcoefs_)>0:
        if np.isnan(np.nanmean(regcoefs_))==False:
            num_days+=1
            num_neurons_all+=num_neurons
print(num_days)
print(num_neurons_all)


# In[ ]:





# In[100]:


###Betas for neurons and previous trials

activity_time='bump_time'
num_trials_neural=0

mean_per_session_all=[]
for lag in np.arange(num_trials_tested+1):
    mean_per_session_lag=[]
    for mouse_recday in Regression_anchors_dic[str(activity_time)][num_trials_neural]['All_betas'].keys():
        All_betas_day=Regression_anchors_dic[str(activity_time)][num_trials_neural]['All_betas'][mouse_recday]
        mean_per_session=np.nanmean(np.nanmean(All_betas_day[:,:,:,lag],axis=2),axis=1)
        mean_per_session_lag.append(mean_per_session)
    mean_per_session_all.append(np.hstack((mean_per_session_lag)))
    
    
    print(st.ttest_1samp(remove_nan(np.hstack((mean_per_session_lag))),0))

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
bar_plotX(mean_per_session_all,'none',0,0.9,'nopoints','unpaired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'/NeuronvsPrevBehaviour_Behaviour_regression.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

behaviour_coefficients=mean_complex2(mean_per_session_all)[1:]
if day_type=='combined_ABCDonly' and len(behaviour_coefficients)==10:
    np.save(Intermediate_object_folder_dropbox+'_previous_chocies_coefficients_ABCD.npy',behaviour_coefficients)


# In[74]:



plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False

data=np.asarray(mean_per_session_all).T
# Filter data using np.isnan
mask = ~np.isnan(data)
filtered_data = [d[m] for d, m in zip(data.T, mask.T)]

#sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1)
#plt.errorbar(np.arange(len(means)),means, yerr=sems, marker='o', fmt='.',color='black')

sns.violinplot(filtered_data, color='grey',alpha=0.5)
#sns.stripplot(filtered_data,color='white',edgecolor='black',linewidth=1,alpha=0.5)
plt.axhline(0,ls='dashed',color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'/Neuron_Behaviour_regression_prevchoices_violin.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

fig, ax = plt.subplots(figsize=(10,4))
sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1, ax=ax)
plt.axhline(0,ls='dashed',color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'/Neuron_Behaviour_regression_prevchoices_swarm.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[34]:


####plotting kernel
behaviour_coefficients=np.load(Intermediate_object_folder_dropbox+'_previous_chocies_coefficients_ABCD.npy')


xdata=np.arange(len(behaviour_coefficients))
ydata=behaviour_coefficients
popt, pcov = curve_fit(func_decay, xdata, ydata)

xdata_new=np.arange(10)
y_pred=func_decay(xdata_new, *popt)

plt.plot(y_pred)


# In[ ]:


'''
-find each cell's anchor in 5 tasks
-based on neuron activity, predict where rewards are in left out task
    -need to consider: tuned vs untuned, where the cell is relative to anchor, where cell is relative to A
    -these three should allow predicting what the task is (e.g. 5-4-9-2) when using the early phase anchored cells
-predict what the policy is (as above but now using all the cells)
-the above isnt proving that the cells tell you the policy - still need to show that we can predict changes in
next trial's policy 

'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


###Firing rates 
all_days_used=np.asarray(list(Regression_anchors_dic[num_trials_neural]['mean_activity'].keys()))
specific_days=np.intersect1d(day_type_dicX[day_type],all_days_used)



activity_time='bump_time'
num_trials_neural=0
#unconcatenated_FRs_=dict_to_array(Regression_anchors_dic[str(activity_time)][num_trials_neural]['norm_FR_allcombinations'])
unconcatenated_FRs_=np.asarray([Regression_anchors_dic[str(activity_time)][num_trials_neural]                                ['norm_FR_allcombinations'][mouse_recday]for mouse_recday in specific_days])

print('Anchors as ns')

concatenated_FRs_=[np.vstack((np.hstack((unconcatenated_FRs_[ii])))) for ii in np.arange(len(unconcatenated_FRs_))]
concatenated_FRs_all=np.vstack((concatenated_FRs_)).T

bar_plotX(concatenated_FRs_all,'none',0,2.5,'nopoints','unpaired',0.025)
plt.show()

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
bar_plotX(concatenated_FRs_all[:2],'none',0,2.5,'nopoints','paired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'/Rate_changedpolicy_0start_peranchor.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

bar_plotX(concatenated_FRs_all[2:],'none',0,2.5,'nopoints','paired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'/Rate_changedpolicy_1start_peranchor.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

concatenated_FRs_start0_clean=column_stack_clean(concatenated_FRs_all[0],concatenated_FRs_all[1]).T
concatenated_FRs_start1_clean=column_stack_clean(concatenated_FRs_all[2],concatenated_FRs_all[3]).T
#print(st.wilcoxon(concatenated_FRs_start0_clean[0],concatenated_FRs_start0_clean[1]))
#print(st.wilcoxon(concatenated_FRs_start1_clean[0],concatenated_FRs_start1_clean[1]))

wilc0_peranchor=st.wilcoxon(concatenated_FRs_start0_clean[0],concatenated_FRs_start0_clean[1])
wilc1_peranchor=st.wilcoxon(concatenated_FRs_start1_clean[0],concatenated_FRs_start1_clean[1])

N1=len(concatenated_FRs_start0_clean[0])
N2=len(concatenated_FRs_start1_clean[0])
min_N_peranchor=np.min([N1,N2])

print('')
print('Sessions as ns')
concatenated_FRs_mean=np.vstack(([np.nanmean(np.nanmean(unconcatenated_FRs_[ii],axis=1),axis=1)                                  for ii in range(len(unconcatenated_FRs_))])).T


#plt.rcParams["figure.figsize"] = (3,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
bar_plotX(concatenated_FRs_mean,'none',0,3.0,'nopoints','paired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'/Rate_changedpolicy.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

#plt.rcParams["figure.figsize"] = (3,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
bar_plotX(concatenated_FRs_mean[:2],'none',0,3.0,'nopoints','paired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'/Rate_changedpolicy_0start.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

#plt.rcParams["figure.figsize"] = (3,6)

bar_plotX(concatenated_FRs_mean[2:],'none',0,3.0,'nopoints','paired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Ephys_output_folder_dropbox+'/Rate_changedpolicy_1start.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

concatenated_FRs_mean_start0_clean=column_stack_clean(concatenated_FRs_mean[0],concatenated_FRs_mean[1]).T
concatenated_FRs_mean_start1_clean=column_stack_clean(concatenated_FRs_mean[2],concatenated_FRs_mean[3]).T
wilc0=st.wilcoxon(concatenated_FRs_mean_start0_clean[0],concatenated_FRs_mean_start0_clean[1])
wilc1=st.wilcoxon(concatenated_FRs_mean_start1_clean[0],concatenated_FRs_mean_start1_clean[1])
#print(wilc0)
#print(wilc1)
N1=len(concatenated_FRs_mean_start0_clean[0])
N2=len(concatenated_FRs_mean_start1_clean[0])
min_N=np.min([N1,N2])



plot_scatter(concatenated_FRs_mean_start0_clean[0],concatenated_FRs_mean_start0_clean[1],'none')
plot_scatter(concatenated_FRs_mean_start1_clean[0],concatenated_FRs_mean_start1_clean[1],'none')




concatenated_FRs_mean_day=np.vstack(([np.nanmean(np.nanmean(np.nanmean(unconcatenated_FRs_[ii],axis=1),axis=1),axis=0)            for ii in range(len(unconcatenated_FRs_))])).T
print('')
print('Days as ns')
bar_plotX(concatenated_FRs_mean_day,'none',0,3.5,'points','paired',0.025)
plt.savefig(Ephys_output_folder_dropbox+'/Rate_changedpolicy_daysNs.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

concatenated_FRs_day_start0_clean=column_stack_clean(concatenated_FRs_mean_day[0],concatenated_FRs_mean_day[1]).T
concatenated_FRs_day_start1_clean=column_stack_clean(concatenated_FRs_mean_day[2],concatenated_FRs_mean_day[3]).T
print(st.wilcoxon(concatenated_FRs_day_start0_clean[0],concatenated_FRs_day_start0_clean[1]))
print(st.wilcoxon(concatenated_FRs_day_start1_clean[0],concatenated_FRs_day_start1_clean[1]))


# In[ ]:





# In[89]:



for group_, array in {'0start':concatenated_FRs_mean[:2],'1start':concatenated_FRs_mean[2:]}.items():


    data=array[1]-array[0]
    # Filter data using np.isnan
        
    array=array[:,data<(np.nanmean(data)+np.nanstd(data)*5)]
    
    
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = True
    xy=column_stack_clean(array[0],array[1])
    noplot_scatter(xy[:,0],xy[:,1],color='black')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'/Rate_changedpolicy_'+group_+'_scatter.svg',            bbox_inches = 'tight', pad_inches = 0)
    plt.show()


# In[90]:


for group_, array in {'0start':concatenated_FRs_all[:2],'1start':concatenated_FRs_all[2:]}.items():


    data=array[1]-array[0]
    # Filter data using np.isnan
        
    array=array[:,data<(np.nanmean(data)+np.nanstd(data)*5)]
    
    
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = True
    xy=column_stack_clean(array[0],array[1])
    noplot_scatter(xy[:,0],xy[:,1],color='black')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Ephys_output_folder_dropbox+'/Rate_changedpolicy_'+group_+'_all_scatter.svg',            bbox_inches = 'tight', pad_inches = 0)
    plt.show()


# In[ ]:





# In[ ]:





# In[91]:


from statsmodels.stats.anova import AnovaRM
  
# Create the data

sessions=np.tile(np.arange(len(concatenated_FRs_mean.T)),len(concatenated_FRs_mean))
Past=np.repeat([0,1], len(concatenated_FRs_mean.T)*2)
Future=np.tile(np.repeat([0,1], len(concatenated_FRs_mean.T)),2)
FR_norm=np.hstack((concatenated_FRs_mean))
dataframe = pd.DataFrame({'sessions': sessions,
                          'Past': Past,
                          'Future': Future,\
                         'FR_norm':FR_norm})

import pingouin as pg

# Compute the 2-way repeated measures ANOVA. This will return a dataframe.
pg.rm_anova(dv='FR_norm', within=['Past','Future'], subject='sessions', data=dataframe)

# Optional post-hoc tests
#pg.pairwise_ttests(dv='FR_norm', within=['Past','Future'], subject='sessions', data=dataframe)

anova_result=dataframe.rm_anova(dv='FR_norm', within=['Past','Future'], subject='sessions')

print(anova_result)


# In[92]:


##Printing results for text
print('Anchor not visited in trial N: n='+str(len(concatenated_FRs_mean_start0_clean[0]))+' tasks, statistic='      +str(int(wilc0[0]))+', P='+str(round(wilc0[1],3))+'.')
print('Anchor visited in trial N: n='+str(len(concatenated_FRs_mean_start1_clean[0]))+' tasks, statistic='      +str(int(wilc1[0]))+', P='+str(round(wilc1[1],3))+'.')

print('In addition, an ANOVA on all data (N='+str(min_N)+' tasks) showed')
for var_ind, var in enumerate(['Past','Future','Past * Future']):
    F=anova_result['F'][anova_result['Source']==var][var_ind]
    P=anova_result['p-GG-corr'][anova_result['Source']==var][var_ind]
    df1=anova_result['ddof1'][anova_result['Source']==var][var_ind]
    df2=anova_result['ddof2'][anova_result['Source']==var][var_ind]
    
    if P<0.05:
        word='a'
    elif P>0.05 and P<0.1:
        word='a trend towards a'
    else:
        word='no'
        
    if '*' in var:
        var=var.replace('*', 'x')
        main_=' '
        addition=' interaction'
    else:
        main_=' main effect of '
        addition=''
        
    print(word+main_+var+addition+': F='+str(round(F,2))+', P='+str(round(P,3))+', df1='+str(df1)+', df2='+str(df2)+', ')


# In[93]:


##using anchors as ns
anchors=np.tile(np.arange(len(concatenated_FRs_all.T)),len(concatenated_FRs_all))
Past=np.repeat([0,1], len(concatenated_FRs_all.T)*2)
Future=np.tile(np.repeat([0,1], len(concatenated_FRs_all.T)),2)
FR_norm=np.hstack((concatenated_FRs_all))
dataframe = pd.DataFrame({'anchors': anchors,
                          'Past': Past,
                          'Future': Future,\
                         'FR_norm':FR_norm})

import pingouin as pg

# Compute the 2-way repeated measures ANOVA. This will return a dataframe.
pg.rm_anova(dv='FR_norm', within=['Past','Future'], subject='anchors', data=dataframe)

# Optional post-hoc tests
#pg.pairwise_ttests(dv='FR_norm', within=['Past','Future'], subject='sessions', data=dataframe)

dataframe.rm_anova(dv='FR_norm', within=['Past','Future'], subject='anchors')

anova_result=dataframe.rm_anova(dv='FR_norm', within=['Past','Future'], subject='anchors')

print(anova_result)


# In[ ]:





# In[94]:


##Printing results for text
print('Anchor not visited in trial N: n='+str(len(concatenated_FRs_start0_clean[0]))+' anchors, statistic='      +str(int(wilc0_peranchor[0]))+', P='+str(round(wilc0_peranchor[1],3))+'.')
print('Anchor visited in trial N: n='+str(len(concatenated_FRs_start1_clean[0]))+' anchors, statistic='      +str(int(wilc1_peranchor[0]))+', P='+str(round(wilc1_peranchor[1],3))+'.')

if round(wilc1_peranchor[1],3)==0:
    print(wilc1_peranchor[1])
          
if round(wilc0_peranchor[1],3)==0:
    print(wilc0_peranchor[1])


print('In addition, an ANOVA on all data (N='+str(min_N_peranchor)+' anchors) showed')
for var_ind, var in enumerate(['Past','Future','Past * Future']):
    F=anova_result['F'][anova_result['Source']==var][var_ind]
    P=anova_result['p-GG-corr'][anova_result['Source']==var][var_ind]
    df1=anova_result['ddof1'][anova_result['Source']==var][var_ind]
    df2=anova_result['ddof2'][anova_result['Source']==var][var_ind]
    
    if P<0.05:
        word='a'
    elif P>0.05 and P<0.1:
        word='a trend towards a'
    else:
        word='no'
        
    if '*' in var:
        var=var.replace('*', 'x')
        main_=' '
        addition=' interaction'
    else:
        main_=' main effect of '
        addition=''
        
    print(word+main_+var+addition+': F='+str(round(F,2))+', P='+str(round(P,3))+', df1='+str(df1)+', df2='+str(df2)+', ')


# In[95]:


all_days_used=np.asarray(list(Regression_anchors_dic[num_trials_neural]['mean_activity'].keys()))
specific_days=np.intersect1d(day_type_dicX[day_type],all_days_used)

all_betas_allconditions=[]
mean_betas_allconditions=[]
num_trials_neural=0
for activity_time in activity_time_options:
    #print(str(activity_time))
    #all_betas_=dict_to_array(Regression_anchors_dic[str(activity_time)][num_trials_neural]['neuron_betas'])
    all_betas_=np.asarray([Regression_anchors_dic[str(activity_time)][num_trials_neural]['neuron_betas'][mouse_recday]    for mouse_recday in specific_days])
    all_betas=remove_nan(concatenate_complex2(np.concatenate(concatenate_complex2(all_betas_))))

    all_betas_allconditions.append(all_betas)
    #plt.hist(all_betas,bins=np.linspace(-3,3,60))
    #plt.axvline(0,color='black',ls='dashed')
    #plt.show()
    #print(st.ttest_1samp(all_betas,0))
    
    mean_betas=np.nanmean(np.vstack(((np.vstack((all_betas_)).T))).T,axis=1)
    tt_test_=st.ttest_1samp(remove_nan(mean_betas),0)
    activity_time__=np.copy(activity_time)

    if isinstance(activity_time, int)==True:
        activity_time_str=str(activity_time)+' degree shifted time'
    else:
        activity_time_str=activity_time

    activity_time_nounderscore=activity_time_str.replace('_', ' ')
    print('"'+str(activity_time_nounderscore)+'": N='+str(len(remove_nan(mean_betas)))+' tasks, statistic='+          str(round(tt_test_[0],2))+', P='+str(round(tt_test_[1],3))+', df='+str(tt_test_.df)+'; ')
    if round(tt_test_[1],3)==0:
        print(tt_test_[1])
     #'''“bump time”: N=126 sessions, statistic=2.68, P=0.008, df=125'''
    mean_betas_allconditions.append(mean_betas)

#all_betas_allconditions=np.asarray(all_betas_allconditions)
#bar_plotX(all_betas_allconditions.T,'none',-0.3,0.3,'nopoints','unpaired',0.025)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
                
mean_betas_allconditions=np.asarray(mean_betas_allconditions)
bar_plotX(mean_betas_allconditions,'none',-0.4,0.4,'nopoints','unpaired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'/Neuron_Behaviour_regression.svg', bbox_inches = 'tight', pad_inches = 0)


# In[96]:


means=np.nanmean(mean_betas_allconditions,axis=1)
sems=st.sem(mean_betas_allconditions,axis=1)

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False

data=mean_betas_allconditions.T
# Filter data using np.isnan
mask = ~np.isnan(data)
filtered_data = [d[m] for d, m in zip(data.T, mask.T)]

#sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1)
#plt.errorbar(np.arange(len(means)),means, yerr=sems, marker='o', fmt='.',color='black')

sns.violinplot(filtered_data, color='grey',alpha=0.5)
#sns.stripplot(filtered_data,color='white',edgecolor='black',linewidth=1,alpha=0.5)
plt.axhline(0,ls='dashed',color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'/Neuron_Behaviour_regression_violin.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()

sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1)
plt.axhline(0,ls='dashed',color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'/Neuron_Behaviour_regression_swarm.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:


all_days_used=np.asarray(list(Regression_anchors_dic[num_trials_neural]['mean_activity'].keys()))
specific_days=np.intersect1d(day_type_dicX[day_type],all_days_used)

all_betas_allconditions=[]
mean_betas_allconditions=[]
num_trials_neural=0
for activity_time in activity_time_options:
    #print(str(activity_time))
    #all_betas_=dict_to_array(Regression_anchors_dic[str(activity_time)][num_trials_neural]['neuron_betas'])
    all_betas_=np.asarray([Regression_anchors_dic[str(activity_time)][num_trials_neural]['neuron_betas'][mouse_recday]    for mouse_recday in specific_days])
    all_betas=remove_nan(concatenate_complex2(np.concatenate(concatenate_complex2(all_betas_))))

    all_betas_allconditions.append(all_betas)
    #plt.hist(all_betas,bins=np.linspace(-3,3,60))
    #plt.axvline(0,color='black',ls='dashed')
    #plt.show()
    #print(st.ttest_1samp(all_betas,0))
    
    tt_test_=st.ttest_1samp(remove_nan(all_betas),0)
    activity_time__=np.copy(activity_time)

    if isinstance(activity_time, int)==True:
        activity_time_str=str(activity_time)+' degree shifted time'
    else:
        activity_time_str=activity_time

    activity_time_nounderscore=activity_time_str.replace('_', ' ')
    
    
    print('"'+str(activity_time_nounderscore)+'": N='+str(len(remove_nan(all_betas)))+' anchorxtasks, statistic='+          str(round(tt_test_[0],2))+', P='+str(round(tt_test_[1],3))+', df='+str(tt_test_.df)+'; ')
    if round(tt_test_[1],3)==0:
        print(tt_test_[1])
     #'''“bump time”: N=126 sessions, statistic=2.68, P=0.008, df=125'''
    mean_betas_allconditions.append(mean_betas)


plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
                
all_betas_allconditions=np.asarray(all_betas_allconditions)
bar_plotX(all_betas_allconditions,'none',-0.4,0.4,'nopoints','unpaired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'/Neuron_Behaviour_regression_peranchor.svg',            bbox_inches = 'tight', pad_inches = 0)


# In[197]:


plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False

data=all_betas_allconditions.T
# Filter data using np.isnan
mask = ~np.isnan(data)
filtered_data = [d[m] for d, m in zip(data.T, mask.T)]

#sns.swarmplot(filtered_data, color='white',edgecolor='black',linewidth=1)
#plt.errorbar(np.arange(len(means)),means, yerr=sems, marker='o', fmt='.',color='black')

sns.violinplot(filtered_data, color='grey',alpha=0.5)
#sns.stripplot(filtered_data,color='white',edgecolor='black',linewidth=1,alpha=0.5)
plt.axhline(0,ls='dashed',color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(Ephys_output_folder_dropbox+'/Neuron_Behaviour_regression_violin_all.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[59]:





# In[ ]:





# In[127]:


np.shape(all_data)


# In[425]:


'''
prev0
"bump time": N=126 tasks, statistic=1.84, P=0.068, df=125; 
"decision time": N=126 tasks, statistic=-2.1, P=0.038, df=125; 
"random time": N=126 tasks, statistic=-1.28, P=0.203, df=125; 
"90 degree shifted time": N=126 tasks, statistic=-1.03, P=0.307, df=125; 
"180 degree shifted time": N=126 tasks, statistic=-1.04, P=0.3, df=125; 
"270 degree shifted time": N=126 tasks, statistic=-2.14, P=0.034, df=125; 


prev1
"bump time": N=121 tasks, statistic=1.54, P=0.126, df=120; 
"decision time": N=121 tasks, statistic=-1.1, P=0.275, df=120; 
"random time": N=121 tasks, statistic=-0.72, P=0.47, df=120; 
"90 degree shifted time": N=121 tasks, statistic=-0.54, P=0.593, df=120; 
"180 degree shifted time": N=121 tasks, statistic=2.2, P=0.03, df=120; 
"270 degree shifted time": N=121 tasks, statistic=0.13, P=0.9, df=120; 

'''


# In[417]:


round(1.34567,2)


# In[ ]:





# In[76]:


mean_betas_allconditions_perday=[]
for activity_time in ['bump_time','decision_time','random_time',90,180,270]:
    #print(str(activity_time))
    mean_betas_perday=[]
    
    for mouse_recday in day_type_dicX[day_type]:
        try:
            all_betas_=Regression_anchors_dic[str(activity_time)][num_trials_neural]['neuron_betas'][mouse_recday]
            mean_betas=np.nanmean(all_betas_)
            mean_betas_perday.append(mean_betas)
        except:
            print(mouse_recday)
            print('Not used')
    mean_betas_allconditions_perday.append(mean_betas_perday)

mean_betas_allconditions_perday=np.asarray(mean_betas_allconditions_perday).T
bar_plotX(mean_betas_allconditions_perday.T,'none',-1,1,'points','paired',0.025)
                
plt.savefig(Ephys_output_folder_dropbox+'/Neuron_Behaviour_regression_perday.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()
#mean_betas_allconditions_perday


# In[79]:


means=np.nanmean(mean_betas_allconditions_perday,axis=0)
sems=st.sem(mean_betas_allconditions_perday,axis=0)

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.boxplot(mean_betas_allconditions_perday)
plt.errorbar(np.arange(len(means))+1,means, yerr=sems, marker='.', fmt='.',color='black')

plt.axhline(0,ls='dashed',color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.show()


# In[81]:


np.shape(mean_betas_allconditions)


# In[ ]:





# In[ ]:





# In[550]:


##Can we predict the next step from activity of neurons anchored to different targets? - regression
##note this cell also has the simple FR analysis 

Regression_anchors_anatomy_dic=rec_dd()
day_type='combined_ABCDonly'

step_no=1

num_phases=3
num_nodes=9
num_states=4

num_trials_tested=10 ##how many trials back to regress out (to control for autocorrelation in behaviour)
num_visits_min=20 ##cutoff for how many visits needed before calculating betas

use_GLM=False
use_anchored_only=True
use_strict_anchored=False
remove_zero_phase=False
use_tuned_only=False

use_crossval=False
use_distal=False

scale=False
z_score=False

use_high_pr=False
use_low_pr=False

remove_neuropixels=False
remove_Cambridge_neurotech=True

if day_type=='combined_ABCDonly':
    num_states=4
    state_degrees=360/num_states
    activity_time_options=['bump_time']

elif day_type=='combined_ABCDE':
    num_states=5
    state_degrees=360/num_states
    activity_time_options=['bump_time','decision_time','random_time',int(state_degrees),int(state_degrees*2),                           int(state_degrees*3),int(state_degrees*4)]

num_bins=num_states*90
angle_correction=360/num_bins
if use_distal==True:
    thr_lower=(360/num_states)
else:   
    thr_lower=(360/num_states)//3
thr_upper=num_bins-thr_lower


calculate_pr_vsFR=False



#thr_anchored_GLM=np.nanpercentile(np.hstack((dict_to_array(GLM_anchoring_dic['Predicted_Actual_correlation_mean']))),50)
#thr_anchored_GLM=0
thr_anchored=np.nanpercentile(np.hstack((dict_to_array(Spatial_anchoring_dic['Cross_val_corr']))),50)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    num_trials_neural=0
    for anatomy_bin in np.arange(4): #[0,1,5,10]:
        ##how many trials back to take neuronal activity from (to look at attractor properties)
        for activity_time in activity_time_options:
            print('____')
            print(str(activity_time))
            print('')


            for mouse_recday in day_type_dicX[day_type]:
                print(mouse_recday)

                mouse=mouse_recday.split('_',1)[0]
                rec_day=mouse_recday.split('_',1)[1]
                cohort=Mice_cohort_dic[mouse]
                

                ephys_type=Cohort_ephys_type_dic[cohort]
                
                abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
                
                if remove_neuropixels==True:
                    if ephys_type=='Neuropixels':
                        continue
                if remove_Cambridge_neurotech==True:
                    if ephys_type=='Cambridge_neurotech':
                        continue
                
                anatomy_bin_neurons=Anat_bin_dic[mouse_recday]
                anatomy_bin_bool=anatomy_bin_neurons==anatomy_bin
                
                ###Baseline (one-step) transition probabilities
                if (use_high_pr==True or use_low_pr==True) and int(cohort) not in [5,6]:
                    step_no=1
                    N_step_pr=structure_probabilities_exp[mouse][str(step_no)].values
                    N_step_pr[:,0]=(N_step_pr[:,0]).astype(int)
                    median_pr=np.median(N_step_pr[:,3])
                    High_pr_bool=N_step_pr[:,3]>median_pr
                    Low_pr_bool=~High_pr_bool

                #Importing Ephys
                print('Importing Ephys')
                num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
                num_neurons=len(cluster_dic['good_clus'][mouse_recday])


                ##Importing Occupancy
                #print('Importing Occupancy')
                #name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
                #data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
                #occupancy_ = load(data_filename_memmap)#, mmap_mode='r')

                ##Tasks
                Tasks=np.load(Intermediate_object_folder+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)
                task_numbers_recday=Task_num_dic[mouse_recday]

                ##defining sessions to use
                sessions=Task_num_dic[mouse_recday]
                
                if day_type=='combined_ABCDE':
                    sessions=sessions[abstract_structures=='ABCDE']

                num_refses=len(np.unique(sessions))
                num_comparisons=num_refses-1
                repeat_ses=np.where(rank_repeat(sessions)>0)[0]
                non_repeat_ses=non_repeat_ses_maker(mouse_recday)  ###this defines the sessions used
                ###only the first session from each task is used
                
                num_trials_ses=dict_to_array(Num_trials_dic2[mouse_recday])
    
                trials_completed_ses=np.where(num_trials_ses>2)[0]
                non_repeat_ses=np.intersect1d(non_repeat_ses,trials_completed_ses)
            


                if day_type=='combined_ABCDE':
                    ABCD_sessions=np.where(abstract_structures=='ABCD')[0]
                    num_ABCD_ses=len(ABCD_sessions)
                    non_repeat_ses=np.setdiff1d(non_repeat_ses,ABCD_sessions)-len(ABCD_sessions)
                    
                    #if num_ABCD_ses>0:
                    #    non_repeat_ses=non_repeat_ses[4:]
                        
                num_nonrepeat_sessions=len(non_repeat_ses)
        
                ##Is the cell anchored? 
                Anchored_bool=Spatial_anchoring_dic['Anchored_bool'][mouse_recday]
                Anchored_neurons=np.where(Anchored_bool==True)[0]

                #360
                
                ###which neurons are phase/state tuned?
                
                if day_type=='combined_ABCDonly':
                    tuning_state_bool_day=tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday]


                elif day_type=='combined_ABCDE':
                    tuning_state_bool_day=np.load(Intermediate_object_folder_dropbox+                                                  'ABCDE_tuning_state_boolean_'+mouse_recday+'.npy')

                num_peaks_all=np.vstack(([np.sum(tuning_state_bool_day[ses_ind],axis=1)                                          for ses_ind in np.arange(len(tuning_state_bool_day))])).T
                #num_peaks_all=np.vstack(([np.sum(tuning_singletrial_dic2['tuning_state_boolean']\
                #                                 [mouse_recday][ses_ind],axis=1)\
                #    for ses_ind in np.arange(len(tuning_singletrial_dic2['tuning_state_boolean'][mouse_recday]))])).T
                
                #phase_tuning=Tuned_dic['Phase'][mouse_recday] 
                #state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday] 
                
                if day_type=='combined_ABCDonly':
                    phase_tuning=Tuned_dic['Phase'][mouse_recday] 
                    state_tuning=Tuned_dic['State_zmax_bool'][mouse_recday]
                elif day_type=='combined_ABCDE':
                    phase_tuning=np.load(Intermediate_object_folder_dropbox+'ABCDE_Phase_'+mouse_recday+'.npy')
                    state_tuning=np.load(Intermediate_object_folder_dropbox+'ABCDE_State_zmax_bool_'+mouse_recday+'.npy')
                
                if use_GLM==True:
                    half_used=GLM_anchoring_dic['half_used_bool'][mouse_recday] 
                    neurons_tuned=np.logical_and(np.logical_and(state_tuning,phase_tuning),half_used)
                else:
                    neurons_tuned=np.logical_and(state_tuning,phase_tuning)
                    

                ###defining array for regression betas
                all_betas_allanchors=np.zeros((num_nonrepeat_sessions,num_phases,num_nodes,int(num_trials_tested+1)))
                neuron_betas_allanchors=np.zeros((num_nonrepeat_sessions,num_phases,num_nodes))
                norm_FR_nonrepeat=np.zeros((num_nonrepeat_sessions,num_phases,num_nodes,2))
                norm_FR_allcombinations=np.zeros((num_nonrepeat_sessions,num_phases,num_nodes,4))
                mean_activity_ses_neurons=np.zeros((num_nonrepeat_sessions,num_neurons))
                num_anchorvisitsnorm_neurons=np.zeros((num_nonrepeat_sessions,num_neurons))


                all_betas_allanchors[:]=np.nan
                neuron_betas_allanchors[:]=np.nan
                norm_FR_nonrepeat[:]=np.nan
                norm_FR_allcombinations[:]=np.nan
                mean_activity_ses_neurons[:]=np.nan
                num_anchorvisitsnorm_neurons[:]=np.nan
                
                xy_11_all=[]
                xy_01_all=[]

                ###looping over all sessions (change to used sessions?)
                for ses_ind_ind, ses_ind in enumerate(non_repeat_ses):
                    
                    #neurons_tuned_ses=np.where(num_peaks_all[:,ses_ind_ind]>0)[0]
                    neurons_tuned_ses=np.where(neurons_tuned==True)[0]
                    
                    ##Is the neuron anchored
                    if use_strict_anchored==True:
                        Anchored_neurons_1=np.where(Spatial_anchoring_dic['Cross_val_corr'][mouse_recday]>thr_anchored)[0]
                        Anchored_bool=Spatial_anchoring_dic['Anchored_bool_strict_crossval'][mouse_recday][ses_ind_ind]
                        Anchored_neurons_2=np.where(Anchored_bool==True)[0]
                        #Anchored_neurons=np.intersect1d(Anchored_neurons_1,Anchored_neurons_2)
                        Anchored_neurons=Anchored_neurons_2
                    else:
                        Anchored_neurons_1=np.where(Spatial_anchoring_dic['Cross_val_corr'][mouse_recday]>0)[0]
                        Anchored_bool=Spatial_anchoring_dic['Anchored_bool_crossval'][mouse_recday][ses_ind_ind]
                        Anchored_neurons_2=np.where(Anchored_bool==True)[0]
                        #Anchored_neurons=np.intersect1d(Anchored_neurons_1,Anchored_neurons_2)
                        Anchored_neurons=Anchored_neurons_2
                        
                    if use_GLM==True:
                        GLM_corr_day=GLM_anchoring_dic['Predicted_Actual_correlation_mean'][mouse_recday]
                        Anchored_neurons=np.where(GLM_corr_day>thr_anchored_GLM)[0]

                    ##What is the task?
                    Task=Tasks[ses_ind]

                    ##How does the animal perform?
                    #performance_=dict_to_array(scores_dic[mouse][task_numbers_recday[ses_ind]]['ALL'])
                    #if len(scores_dic[mouse][task_numbers_recday[ses_ind]]['ALL'].keys())>1:
                    #performance=performance_[0]
                    #else:
                    #    performance=performance_

                    ##what is the anchor?
                    if use_crossval==True:
                        try:
                            phase_locations_neurons_=((Spatial_anchoring_dic['best_node_phase_used'][mouse_recday]                                              [:,ses_ind_ind,:]).astype(int)).T
                        except:
                            print('spatial anchors not found')
                            continue
                        ###what is the lag between the neuron's firing and the anchor? 
                        Best_shift_time_=Anchor_trial_dic['Best_shift_time_cross_val'][mouse_recday]

                    else:
                        #phase_locations_neurons_=Spatial_anchoring_dic['most_common_anchor'][mouse_recday]
                        phase_locations_neurons_=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday]
                        
                        ###what is the lag between the neuron's firing and the anchor? 
                        Best_shift_time_=Anchor_trial_dic['Best_shift_time'][mouse_recday]
                    
                    phase_locations_neurons=(np.column_stack((phase_locations_neurons_[:,0],                                                             phase_locations_neurons_[:,1]+1))).astype(int)
                    if len(phase_locations_neurons)==0:
                        continue
                    phase_locations_neurons_unique=np.unique(phase_locations_neurons,axis=0)


                    ###defining all non-spatial neurons (i.e. neurons with greater than threshold lag from their anchor)
                    nonzero_anchored_neurons=np.where(np.logical_and(Best_shift_time_[:,ses_ind_ind]>thr_lower,                                                  Best_shift_time_[:,ses_ind_ind]<thr_upper))[0]


                    ###defining ephys, occupancy and phase matrices for this session
                    try:
                        ephys_=np.load(Intermediate_object_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    except:
                        print('No Ephys')
                        continue
                    if len(ephys_)==0 :
                        print('No Ephys')
                        continue

                    if len(ephys_[0])==0:
                        continue

                    #occupancy_mat=data_matrix(occupancy_[ses_ind])
                    #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
                    
                    location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
                    occupancy_mat=np.reshape(location_mat_,(num_states,len(location_mat_),                                                            len(location_mat_.T)//num_states))
                    occupancy_conc=np.concatenate(location_mat_)

                    phase_mat=np.zeros(np.shape(occupancy_mat))
                    phase_mat[:,:,30:60]=1
                    phase_mat[:,:,60:90]=2
                    phase_conc=np.concatenate(np.hstack(phase_mat))

                    occupancy_mat_=location_mat_
                    phase_mat_=np.hstack(phase_mat)

                    num_trials=len(occupancy_mat_)



                    ###looping over all location-phase conjunctions to find neurons anchored to them and do the regression
                    for phase_ in np.arange(num_phases):
                        if phase_==0 and remove_zero_phase==True:
                            continue
                        for location_ in np.arange(num_nodes):
                            location=location_+1 ## location arrays are not using zero-based indexing 

                            #elif zero_shot==True:
                            #visits_start=visits_start_[visits_start_<360+31] ###only including first trial
                            #visits_prev_end=visits_prev_end[visits_prev_end<360+31] ###only including first trial

                            ##defining place/phase visits
                            placephase_mat=(np.logical_and(phase_mat_==phase_,occupancy_mat_==location)).astype(int)
                            placephase_conc=(np.logical_and(phase_conc==phase_,occupancy_conc==location)).astype(int)
                            visits=np.where(placephase_conc==1)[0]
                            if len(visits)==0:
                                continue

                            ###where are the location/phase conjunctions one step away from the anchor?
                            ##This defines decision times
                            prev_locations_=np.where(mindistance_mat[location_]==1)[0]
                            prev_locations=prev_locations_+1
                            
                            if int(cohort) in [5,6]:
                                trans_mat_mean=np.load(Intermediate_object_folder_dropbox+'/'+                                mouse+'_Exploration_Transition_matrix.npy')
                                probs=[]
                                for portX_ in np.arange(9):
                                    for portY_ in np.where(mindistance_mat[portX_]==1)[0]:
                                        probs.append(trans_mat_mean[step_no-1][portX_,portY_])

                                probs_median=np.median(probs)

                                if use_high_pr==True:
                                    prev_loc_bool=np.hstack(([trans_mat_mean[step_no-1][prev_location-1,location-1]                                                              >=probs_median for prev_location in prev_locations]))
                                elif use_low_pr==True:
                                    prev_loc_bool=np.hstack(([trans_mat_mean[step_no-1][prev_location-1,location-1]                                                              <probs_median for prev_location in prev_locations]))
                            else:
                                if use_high_pr==True:
                                    prev_loc_bool=np.array([High_pr_bool[np.logical_and(N_step_pr[:,0]==                                                    prev_locations[ii], N_step_pr[:,1]==location)][0]                                                          for ii in range(len(prev_locations))])

                                elif use_low_pr==True:
                                    prev_loc_bool=np.array([Low_pr_bool[np.logical_and(N_step_pr[:,0]==                                                    prev_locations[ii], N_step_pr[:,1]==location)][0]                                                          for ii in range(len(prev_locations))])
                            if use_high_pr==True or use_low_pr==True:
                                prev_locations=prev_locations[prev_loc_bool]

                            if len(prev_locations)==0:
                                continue
                            
                            prev_phase_=(phase_-1)%3
                            location_conc_prev_=np.max(np.asarray([occupancy_conc==prev_locations[ii] for ii                                                          in range(len(prev_locations))]),axis=0)
                            location_conc_prev=location_conc_prev_>0 ##i.e. when visiting location one step from 
                            ##anchor location
                            placephase_conc_prev=(np.logical_and(phase_conc==prev_phase_                                                                 ,location_conc_prev==True)).astype(int)
                            #i.e. when visiting location one step from anchor location AND phase one step from
                            #anchor phase 
                            visits_prev=np.where(placephase_conc_prev==1)[0]
                            if len(visits_prev)==0:
                                continue
                            visits_start_=visits[np.hstack((num_bins,np.diff(visits)))>30] ##only the start of a visits 
                            ##in a given phase
                            visits_prev_end=visits_prev[np.hstack((np.diff(visits_prev),num_bins))>30] 
                            ##only the end of a visit in a given phase


                            visits_start=visits_start_[visits_start_>=num_bins] ###removing first trial
                            visits_prev_end=visits_prev_end[visits_prev_end>num_bins] ###removing first trial
                            visits_prev_end_nodes=occupancy_conc[visits_prev_end] 
                            ##which nodes are visited at decision points
                            
                            if len(visits_prev_end)<num_visits_min:
                                continue

                            ##where does the animal actually go after the decision points i.e when one step away
                            ##from anchor (visits_prev_end) - it can go to anchor or not
                            visited_node_start=np.zeros((len(visits_prev_end),2))
                            visited_node_start[:]=np.nan
                            for visit_prev_ind, visit_prev in enumerate(visits_prev_end):
                                next_90_occ=occupancy_conc[visit_prev+1:visit_prev+91]
                                next_90_phase=phase_conc[visit_prev+1:visit_prev+91]

                                next_node_=next_90_occ[np.logical_and(next_90_occ<10,next_90_occ!=                                                                     visits_prev_end_nodes[visit_prev_ind])]
                                ##only taking nodes (not edges) and only nodes that arent the same as the previous nodes
                                ##(because could have moved phases but stayed in same node)

                                ###excluding trials where animal stayed in same location across two phases
                                ###because then cant define decision point
                                if len(next_90_occ)==0 or len(next_node_)==0: 
                                    continue
                                next_node=next_node_[0] ###the very first node visited after decision point
                                next_node_phase_start_=np.where(np.logical_and(next_90_phase==phase_,                                                                               next_90_occ==next_node))[0]
                                ##when is next node visited at the next phase

                                ##whats the first bin where the next node is visited in the next phase  
                                if len(next_node_phase_start_)==0 or                                mindistance_mat[int(visits_prev_end_nodes[visit_prev_ind]-1),int(next_node-1)]!=1:
                                    ##note: the second condition is to deal with erroneous tracking where animal position
                                    ##jumps more than one node between bins
                                    next_node=np.nan
                                    next_node_phase_start=np.nan
                                else:
                                    next_node_phase_start=next_node_phase_start_[0]

                                visited_node_start[visit_prev_ind]=np.asarray([next_node,visit_prev+next_node_phase_start])

                            ###we now have visited_node_start which has both the visited node and its timestamp following
                            #each decision point (decision point being when animal was one place and one phase 
                            ##away from anchor)


                            ##Defining dependent variable (location/phase visits)
                            location_visited_bool=(visited_node_start[:,0]==location).astype(int) ## 1) DEPENDENT VARIABLE
                            ##i.e. when did animal visit the anchor after each decision point
                            ##the length of this array is all the times where animal was one location and one phase away
                            ##from anchor
                            times_=visited_node_start[:,1] ##times for dependent variable
                            nodes_=visited_node_start[:,0] ##nodes visited for dependent variable

                            ### Did animal visit the anchor location on N previous trials?
                            ## we regress this out to remove any effects of autocorrelation in behaviour 
                            trial_lag_booleans_all=np.zeros((num_trials_tested,len(times_)))
                            trial_lag_booleans_all[:]=np.nan
                            for trial_lag_ in np.arange(num_trials_tested):
                                trial_lag=trial_lag_+1
                                range_tested=[(num_bins*trial_lag)-30,                                              (num_bins*trial_lag)+30]
                                ##what range of time lags should we use to look for previous trial visits 
                                ##using num_trials_neural here when predicting behaviour using neural activity from 
                                ##M trials back, this means now the coregressors (animal's previous choices) for the 
                                ##behaviour are lagged by exactly N+M trials back (with tolerance of +/- 30 degrees))


                                ## for each bin in times_ (i.e. each timestamp following each decision point) what (if 
                                ##any) is the bin where animal visited the same location/phase N+M trials back
                                ##Note: for trials less then N+M you will effectively never have visited the same 
                                ##anchor at this trial lag and so the co-regressors for this trial lag will always be 
                                ##zero 
                                visit_at_lag=[np.where(np.logical_and((times_<(times_[ii]-range_tested[0])),                                                                      (times_>(times_[ii]-range_tested[1]))))[0]                                for ii in range(len(times_))]

                                ###did animal visit the anchor location M+N trials before the current visit time?
                                trial_lag_boolean=np.asarray([np.sum([nodes_[visit_at_lag[ii][jj]]==location                                                                      for jj in range(len(visit_at_lag[ii]))])                                            if len(visit_at_lag[ii]>0) else 0 for ii in range(len(nodes_))])
                                trial_lag_boolean[trial_lag_boolean>0]=1
                                trial_lag_booleans_all[trial_lag_]=trial_lag_boolean ## 2) CO-REGRESSORS

                                ###Now you have trial_lag_booleans_all which tells you when you visited place/phase 
                                ##anchor exactly N+M trials in the past for each N (and a fixed M) - note M=0 for the 
                                ##main analysis



                            ##find neurons anchored to this location/phase with non-zero distance
                            
                            if use_GLM==True:
                                Neurons_per_anchor=Anchor_topN_GLM_crossval_dic['Neurons_per_anchor'][mouse_recday]                                [ses_ind_ind][phase_][location_]
                                
                                #Neurons_per_anchor=Anchor_topN_GLM_dic['Neurons_per_anchor'][mouse_recday]\
                                #[phase_][location_]

                                if len(Neurons_per_anchor)>0:
                                    neurons_anchorednext=(Neurons_per_anchor[:,0]).astype(int)
                                    best_shift_times_=Neurons_per_anchor[:,1]*(num_bins/num_lags)

                                    neurons_anchorednext_nonzero=(neurons_anchorednext                                    [np.logical_and(best_shift_times_>thr_lower,                                                    best_shift_times_<thr_upper)]).astype(int)

                                    best_shift_times_nonzero=(best_shift_times_                                    [np.logical_and(best_shift_times_>thr_lower,                                                    best_shift_times_<thr_upper)]).astype(int)

                                    best_shift_times=best_shift_times_nonzero

                                    if use_anchored_only==True:
                                        neurons_anchorednext_nonzero_anchored_=np.intersect1d(Anchored_neurons,                                                                                             neurons_anchorednext_nonzero)
                                        neurons_anchorednext_nonzero_anchored=neurons_anchorednext_nonzero[                                                                                    [neurons_anchorednext_nonzero[ii]                                                            in neurons_anchorednext_nonzero_anchored_                                                          for ii in range(len(neurons_anchorednext_nonzero))]]
                                        best_shift_times=best_shift_times_nonzero[[neurons_anchorednext_nonzero[ii]                                                                         in neurons_anchorednext_nonzero_anchored_                                                            for ii in range(len(neurons_anchorednext_nonzero))]]

                                else:
                                    continue


                            else:
                                neurons_anchorednext=np.where(np.logical_and(phase_locations_neurons[:,0]==phase_,                                                                        phase_locations_neurons[:,1]==location))[0]####
                                neurons_anchorednext_nonzero=np.intersect1d(neurons_anchorednext,nonzero_anchored_neurons)

                                neurons_anchorednext_nonzero_tuned=np.intersect1d(neurons_anchorednext_nonzero,                                                                                  neurons_tuned_ses)
                                neurons_anchorednext_nonzero_anchored=np.intersect1d(Anchored_neurons,                                                                                     neurons_anchorednext_nonzero)
                                neurons_anchorednext_nonzero_anchored_tuned=np.intersect1d(                                                                                neurons_anchorednext_nonzero_anchored,                                                                                           neurons_tuned_ses)
                            ##i.e. neurons that have the same anchor for half the tasks or more 

                            if use_anchored_only==True:
                                neurons_used=neurons_anchorednext_nonzero_anchored
                            else:
                                neurons_used=neurons_anchorednext_nonzero
                            
                            if use_tuned_only==True:
                                neurons_used=np.intersect1d(neurons_used,neurons_tuned_ses)
                            
                            ###subsetting by anatomical bin
                            neurons_used=np.intersect1d(neurons_used,np.where(anatomy_bin_bool==True)[0])
                            
                            
                            ##Below we're getting  mean activity of neurons at different times before anchor visit
                            ##first defining arrays
                            mean_activity_bump_neurons=np.zeros((len(neurons_used),len(location_visited_bool)))

                            mean_activity_bump_neurons[:]=np.nan


                            if len(neurons_used)==0:
                                ##i.e. no neurons anchored to this location/phase
                                all_betas_allanchors[ses_ind_ind,phase_,location_]=np.repeat(np.nan,num_trials_tested+1)
                                neuron_betas_allanchors[ses_ind_ind,phase_,location_]=np.nan
                                continue

                            ##whats the lag for each neuron to its anchor? 
                            if use_GLM==False:
                                best_shift_times=Best_shift_time_[neurons_used,ses_ind_ind]
                                #best_shift_times=Best_shift_time_[neurons_used]

                            ##mean first trial activity of all neurons
                            mean_allneurons=np.mean(np.mean(ephys_[:,0],axis=1)/np.mean(np.mean(ephys_,axis=1),axis=1))
                            ##normalised for each neuron by mean activity across all trials

                            ###defining primary independent variable (neurons activity at defined time)
                            ###defining time to take neuron's activity (for primary independent variable)
                            for neuron_ind, neuron in enumerate(neurons_used):
                                ephys_neuron_=ephys_[neuron]
                                neuron_conc=np.hstack((ephys_neuron_))

                                ##defining activity times
                                if activity_time == 'bump_time':
                                    gap=num_bins-best_shift_times[neuron_ind]
                                    ##i.e. times when neurons should be active on ring attractor
                                elif activity_time == 'decision_time':
                                    gap=thr_lower
                                    ##i.e. time when animal is about to visit anchor
                                elif activity_time == 'random_time':
                                    gap=random.randint(0,num_bins-1)
                                elif isinstance(activity_time,int)==True:
                                    gap=((num_bins-best_shift_times[neuron_ind])+int(activity_time))%num_bins
                                    ##times 90 degrees shifted from bump time
                                neuron_bump_time_start_=times_-gap ##definitely looking at activity BEFORE anchor visit                               
                                neuron_bump_time_start=(neuron_bump_time_start_).astype(int)


                                ##defining normalised mean activity at selected times (ranging from time to 30 degrees 
                                ##later)
                                mean_activity_bump=np.asarray([np.mean(neuron_conc[neuron_bump_time_start[ii]:                                                                                   neuron_bump_time_start[ii]+30])                                                               for ii in range(len(neuron_bump_time_start))])
                                mean_activity_bump[np.isnan(neuron_bump_time_start_)]=np.nan


                                times_int=(times_).astype(int)
                                mean_activity_trial=np.asarray([np.mean(neuron_conc[times_int[ii]:times_int[ii]+                                                                                    num_bins])                                                                for ii in range(len(times_int))])

                                #mean_activity_trial=remove_nan(mean_activity_trial)
                                mean_activity_trial[mean_activity_trial==0]=np.nan ##to avoid dividing by zero
                                mean_activity_bump_neurons[neuron_ind]=mean_activity_bump/mean_activity_trial
                                ##i.e. firing rate as proportion of each neuron's mean firing on a given trial

                                mean_activity_ses=np.nanmean(neuron_conc)
                                mean_activity_ses_neurons[ses_ind_ind,neuron]=mean_activity_ses
                                num_anchorvisitsnorm_neurons[ses_ind_ind,neuron]=len(times_)/num_trials


                            meanofmeans_activity_bump_neurons=                            np.nanmean(mean_activity_bump_neurons,axis=0) ##3) INDPENDENT VARIABLE
                            ##mean relative firing rate across ALL neurons anchored to a given place/phase
                            ##i.e. collapsing neuron dimension and just keeping visit dimension


                            ## Final inputs to regression
                            X_=np.column_stack((meanofmeans_activity_bump_neurons,trial_lag_booleans_all.T))
                            y_=location_visited_bool


                            X=X_[~np.isnan(meanofmeans_activity_bump_neurons)]
                            #X=np.vstack(([X[:,ii]-np.mean(X[:,ii]) for ii in range(len(X.T))])).T
                            X=np.column_stack((X[:,0]-np.mean(X[:,0]),X[:,1:])) ##de-meaning neuronal activity
                            y=y_[~np.isnan(meanofmeans_activity_bump_neurons)]

                            ###lagging by num_trials_neural
                            times__=times_[~np.isnan(meanofmeans_activity_bump_neurons)]##updated times removing nans
                            if len(times__)==0:
                                continue
                            times_shifted_=times__+num_bins*num_trials_neural
                            indices_trial_lagged_=[[ii,np.where(np.logical_and(times_shifted_[ii]<                                                                                         times__+30,times_shifted_[ii]>                                                                                         times__-30))[0][0]]                                        for ii in range(len(times_shifted_))                                        if len(np.where(np.logical_and(times_shifted_[ii]<times__+30,                                                                       times_shifted_[ii]>times__-30))[0])>0]
                            if len(indices_trial_lagged_)==0:
                                continue
                            indices_trial_lagged=np.vstack((indices_trial_lagged_))

                            X=np.column_stack((X[indices_trial_lagged[:,0],0],X[indices_trial_lagged[:,1],1:]))
                            y=y[indices_trial_lagged[:,1]]
                            
                            if scale==True:
                                transformer = MaxAbsScaler().fit(X)
                                X=transformer.transform(X)

                            if z_score==True:
                                X=st.zscore(X,axis=0)
                                X=X[~np.isnan(np.mean(X,axis=1))]
                                y=y[~np.isnan(np.mean(X,axis=1))]

                            ##doing the regression
                            if np.sum(abs(np.diff(y)))==0 or len(X)==0: ### i.e. always visited (or always didnt visit) 
                                ##place/phase from all decision points
                                beta_all=np.repeat(np.nan,num_trials_tested+1)
                                beta_neurons=np.nan
                            else:
                                clf = LogisticRegression(solver='saga',penalty=None).fit(X, y) 
                                ##,max_iter=10000
                                beta_all=clf.coef_[0]
                                beta_neurons=clf.coef_[0][0]

                            all_betas_allanchors[ses_ind_ind,phase_,location_]=beta_all
                            neuron_betas_allanchors[ses_ind_ind,phase_,location_]=beta_neurons

                            ###simple analysis to check for policy effect - do anchored neurons fire more in trial 
                            ##before animal goes to anchor even when controlling for previous choice
                            X=X_[~np.isnan(meanofmeans_activity_bump_neurons)]
                            X=np.column_stack((X[indices_trial_lagged[:,0],0],X[indices_trial_lagged[:,1],1:]))
                            non_repeat_visits=X[:,1]!=y
                            X_nonrepeat=X[non_repeat_visits,0]
                            y_nonrepeat=y[non_repeat_visits]
                            mean_rates_01=np.mean(X_nonrepeat[y_nonrepeat==1])
                            mean_rates_10=np.mean(X_nonrepeat[y_nonrepeat==0])

                            repeat_visits=X[:,1]==y
                            X_repeat=X[repeat_visits,0]
                            y_repeat=y[repeat_visits]
                            mean_rates_11=np.mean(X_repeat[y_repeat==1])
                            mean_rates_00=np.mean(X_repeat[y_repeat==0])

                            norm_FR_nonrepeat[ses_ind_ind,phase_,location_]=mean_rates_01,mean_rates_10
                            norm_FR_allcombinations[ses_ind_ind,phase_,location_]=                            mean_rates_00,mean_rates_01,mean_rates_10,mean_rates_11
                            
                            
                            prev_nodes_=occupancy_conc[visits_prev_end]
                            prev_nodes=prev_nodes_[~np.isnan(meanofmeans_activity_bump_neurons)]
                            ##updated pre nodes removing nans

                
                Regression_anchors_anatomy_dic[anatomy_bin][str(activity_time)]['All_betas']                [mouse_recday]=all_betas_allanchors
                Regression_anchors_anatomy_dic[anatomy_bin][str(activity_time)]['neuron_betas']                [mouse_recday]=neuron_betas_allanchors
                Regression_anchors_anatomy_dic[anatomy_bin][str(activity_time)]['norm_FR_nonrepeat']                [mouse_recday]=norm_FR_nonrepeat
                Regression_anchors_anatomy_dic[anatomy_bin][str(activity_time)]                ['norm_FR_allcombinations'][mouse_recday]=norm_FR_allcombinations
 
                if activity_time=='bump_time':
                    Regression_anchors_anatomy_dic[anatomy_bin]['mean_activity'][mouse_recday]=                    mean_activity_ses_neurons
                    Regression_anchors_anatomy_dic[anatomy_bin]['num_anchor_visits'][mouse_recday]=                    num_anchorvisitsnorm_neurons


# In[548]:


Regression_anchors_anatomy_dic[0]['bump_time'][0]['neuron_betas']['ab03_05092023_06092023']


# In[552]:


activity_time='bump_time'
all_days_used=np.asarray(list(Regression_anchors_anatomy_dic[anatomy_bin][str(activity_time)][num_trials_neural].keys()))
specific_days=np.intersect1d(day_type_dicX[day_type],all_days_used)
mean_betas_allconditions=[]

for anatomy_bin in np.arange(4):
    all_betas_=np.asarray([Regression_anchors_anatomy_dic[anatomy_bin][str(activity_time)][num_trials_neural]                           ['neuron_betas'][mouse_recday]    for mouse_recday in specific_days])
    all_betas=remove_nan(concatenate_complex2(np.concatenate(concatenate_complex2(all_betas_))))

    all_betas_allconditions.append(all_betas)
    #plt.hist(all_betas,bins=np.linspace(-3,3,60))
    #plt.axvline(0,color='black',ls='dashed')
    #plt.show()
    #print(st.ttest_1samp(all_betas,0))

    mean_betas=np.nanmean(np.vstack(((np.vstack((all_betas_)).T))).T,axis=1)
    tt_test_=st.ttest_1samp(remove_nan(mean_betas),0)
    activity_time__=np.copy(activity_time)

    if isinstance(activity_time, int)==True:
        activity_time_str=str(activity_time)+' degree shifted time'
    else:
        activity_time_str=activity_time

    activity_time_nounderscore=activity_time_str.replace('_', ' ')
    print('"'+str(activity_time_nounderscore)+'": N='+str(len(remove_nan(mean_betas)))+' sessions, statistic='+          str(round(tt_test_[0],2))+', P='+str(round(tt_test_[1],3))+', df='+str(tt_test_.df)+'; ')
    
    mean_betas_allconditions.append(mean_betas)

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
                
mean_betas_allconditions=np.asarray(mean_betas_allconditions)
bar_plotX(mean_betas_allconditions,'none',-0.5,0.5,'nopoints','unpaired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Ephys_output_folder_dropbox+'/Neuron_Behaviour_regression_anatomybins.svg',            bbox_inches = 'tight', pad_inches = 0)


# In[545]:


all_betas_


# In[ ]:


########Examples Anchoring###########


# In[ ]:





# In[ ]:





# In[ ]:


day_type_dicX['combined_ABCDonly']


# In[ ]:


mouse_recday='ah04_01122021_02122021'
print(mouse_recday)
#Importing Ephys
print('Importing Ephys')
num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
num_neurons=len(cluster_dic['good_clus'][mouse_recday])

for session in np.arange(num_sessions):
    try:
        name='standardized_spike_events_dic_'+mouse_recday+'_'+str(session)
        data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
        ephys_ = load(data_filename_memmap)#, mmap_mode='r')
        exec('ephys_ses_'+str(session)+'_=ephys_')
    except:
        ('Ephys not found')
#print('Importing Occupancy')
#name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
#data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
#occupancy_ = load(data_filename_memmap)#, mmap_mode='r')
    
sessions=Task_num_dic[mouse_recday]
num_refses=len(np.unique(sessions))
num_comparisons=num_refses-1
repeat_ses=np.where(rank_repeat(sessions)>0)[0]
non_repeat_ses=non_repeat_ses_maker(mouse_recday)  

Neuron_used_day_=Spatial_anchoring_dic['Neuron_used_histogram'][mouse_recday]
Coh_day_=Spatial_anchoring_dic['Neuron_coherence'][mouse_recday]
Tuned_day_=Spatial_anchoring_dic['Neuron_tuned'][mouse_recday]
Coh_day_sum=np.sum(Coh_day_,axis=0)
Tuned_day_sum=np.sum(Tuned_day_,axis=0)
nonzero_coherent_neurons=np.intersect1d(np.where(Coh_day_sum/Tuned_day_sum>0.75)[0],Neuron_used_day_)
nonzero_coherent_neurons=np.intersect1d(np.where(Coh_day_sum/len(Coh_day_)>0.75)[0],Neuron_used_day_)

anchored_neurons=np.where(Spatial_anchoring_dic['Anchored_bool'][mouse_recday]==True)[0]
anchored_neurons_strict=np.where(Spatial_anchoring_dic['Anchored_bool_strict'][mouse_recday]==True)[0]

anchors_=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday][anchored_neurons]
shifts=Anchor_trial_dic['Best_shift_time'][mouse_recday][anchored_neurons]
arrayx=np.column_stack((anchored_neurons,anchors_[:,0],anchors_[:,1]+1,shifts))
np.savetxt(Ephys_output_folder_dropbox+'_example_anchored_neurons_'+mouse_recday,arrayx)


# In[ ]:


Ephys_output_folder_dropbox


# In[ ]:


anchors_=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday][anchored_neurons]
shifts=Anchor_trial_dic['Best_shift_time'][mouse_recday][anchored_neurons]
arrayx=np.column_stack((anchored_neurons,anchors_[:,0],anchors_[:,1]+1,shifts))
np.savetxt(Ephys_output_folder_dropbox+'_example_anchored_neurons_'+mouse_recday,arrayx)


# In[ ]:





# In[ ]:





# In[ ]:


##Plotting best spatial alignment
'''
Examples:
me11_01122021_02122021: 20,22,29,47

ah04_01122021_02122021: 11,22,36,72,90,115

ah04_07122021_08122021: 5,9

me10_09122021_10122021: 27

'''
thr_visit=2
test_comp=0
fontsize=10
plot_edge=True 

mean_corrs=np.nanmean(Phase_spatial_corr_dic['corrs_crossval_all'][mouse_recday],axis=2)
sem_corrs=st.sem(Phase_spatial_corr_dic['corrs_crossval_all'][mouse_recday],axis=2,nan_policy='omit')

#field_peak_bins_=Phase_spatial_corr_dic['field_peak_bins'][mouse_recday]
thresholds_=Phase_spatial_corr_dic['Threshold'][mouse_recday]

reference_tasks=Spatial_anchoring_dic['Best_reference_task'][mouse_recday]

for neuron in [17]:#,115]:
    print('neuron'+str(neuron))
    #print(Tuned_day_[:,neuron])
    
    #### preferred phase
    phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][0]
    pref_phase_neurons=np.argmax(phase_peaks,axis=1)
    pref_phase=pref_phase_neurons[neuron]
    
    #best_anchor_phase_node=Spatial_anchoring_dic['best_node_phase_used'][mouse_recday][:,:,neuron][:,test_comp]
    best_anchor_phase_node=Spatial_anchoring_dic['most_common_anchor'][mouse_recday][neuron]
    best_anchor_phase_node=Spatial_anchoring_dic['Best_anchor_all'][mouse_recday][neuron]

    phase_=int(best_anchor_phase_node[0])
    location=int(best_anchor_phase_node[1]+1) ##because 0 based indexing used here but 1 based in locations
    
    
    #phase_=2
    #location=7

    
    print(location)
    print(phase_)
    print('')
    print(pref_phase)
    
    #session=0
    mean_smooth_all=np.zeros((len(non_repeat_ses),360))
    mean_smooth_all[:]=np.nan
    
    fig1, f1_axes = plt.subplots(figsize=(15, 7.5),ncols=len(non_repeat_ses), constrained_layout=True,                                subplot_kw={'projection': 'polar'})
    fig2, f2_axes = plt.subplots(figsize=(15, 7.5),ncols=len(non_repeat_ses), constrained_layout=True,                                subplot_kw={'projection': 'polar'})
    ref_tasks_all=[]
    for ses_ind, session in enumerate(non_repeat_ses):
        
        ref_task=reference_tasks[ses_ind,neuron]
        ref_tasks_all.append(ref_task)
        
        ax1=f1_axes[ses_ind]
        ax2=f2_axes[ses_ind]
        exec('ephys_=ephys_ses_'+str(session)+'_')
        

        #occupancy_mat=data_matrix(occupancy_[session])
        #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
        
        location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
        if len(location_mat_)==0:
            continue
        occupancy_mat=np.reshape(location_mat_,(4,len(location_mat_),len(location_mat_.T)//4))
        occupancy_conc=np.concatenate(location_mat_)

        phase_mat=np.zeros(np.shape(occupancy_mat))
        phase_mat[:,:,30:60]=1
        phase_mat[:,:,60:90]=2

        #phase_mat=np.zeros(np.shape(occupancy_mat))
        #phase_mat[:,:,45:90]=1
        #phase_mat[:,:,60:90]=2
        phase_conc=np.concatenate(np.hstack(phase_mat))

        ephys_neuron_=ephys_[neuron]
        neuron_mat=data_matrix(ephys_neuron_,concatenate=False)
        neuron_conc=np.concatenate(np.hstack(neuron_mat))


        ###tone aligned activity
        tone_aligned_activity=np.hstack(neuron_mat)
        #plt.matshow(tone_aligned_activity)
        #plt.show()

        mean_=np.mean(tone_aligned_activity,axis=0)
        sem_=st.sem(tone_aligned_activity,axis=0)
        mean_smooth=smooth_circular(mean_)
        sem_smooth=smooth_circular(sem_)
        polar_plot_stateX2(mean_smooth,mean_smooth+sem_smooth,mean_smooth-sem_smooth,labels='angles',color='blue',                          ax=ax1,repeated=False,fontsize=fontsize)
        #plt.savefig(Ephys_output_folder_dropbox+'Example_cells/Taskmaps_'+mouse_recday+'_neuron_'+str(neuron)+\
        #                '_ses_'+str(session)+'.svg', bbox_inches = 'tight', pad_inches = 0)
        #plt.show()

        #for location in np.arange(9)+1:
        #print('')
        

        timestamps_=np.where((np.logical_and(occupancy_conc==location, phase_conc==phase_)))[0]
        #timestamps_=np.where(occupancy_conc==location)[0]
        long_stays=np.where(rank_repeat2(occupancy_conc)>thr_visit)[0]
        timestamps=np.intersect1d(timestamps_,long_stays-(thr_visit+1))
        if len(timestamps)>0:
        
            timestamps_start=timestamps[(np.hstack((1,np.diff(timestamps)>thr_visit))).astype(bool)]
            timestamps_end=timestamps[(np.hstack((np.diff(timestamps)>thr_visit,1))).astype(bool)]
            aligned_activity=np.asarray([neuron_conc[ii:ii+360] if len(neuron_conc[ii:ii+360])==360                                         else np.repeat(np.nan,360) for ii in timestamps_start])
            
            #print('session='+str(ses_ind)+'visits='+str(len(timestamps_start)))

            #plt.matshow(aligned_activity)
            #plt.show()

            mean_=np.nanmean(aligned_activity,axis=0)
            sem_=st.sem(aligned_activity,axis=0,nan_policy='omit')
            mean_smooth=smooth_circular(mean_)
            sem_smooth=smooth_circular(sem_)
            if len(timestamps_start)==1:
                sem_smooth=np.repeat(0,360)
                

            #plt.tight_layout()
            #plt.savefig(Ephys_output_folder_dropbox+'Example_cells/AnchoredTaskmaps_'+mouse_recday+
            #'_neuron_'+str(neuron)+\
            #            '_ses_'+str(session)+'.svg', bbox_inches = 'tight', pad_inches = 0)
            #plt.show()
            
            if np.nanmean(mean_smooth)==0 or np.isnan(np.nanmean(mean_smooth))==True:
                mean_smooth=np.repeat(np.nan,360)

        else:
            mean_smooth=np.repeat(np.nan,360)
            sem_smooth=np.repeat(np.nan,360)
            
        
        
        mean_smooth_all[ses_ind]=mean_smooth
        polar_plot_stateX2(mean_smooth,mean_smooth+sem_smooth,mean_smooth-sem_smooth,labels='angles',                          ax=ax2,repeated=False,fontsize=fontsize)
            #print('Not visited')
    plt.tight_layout()
    fig1.savefig(Ephys_output_folder_dropbox+'Example_cells/Taskmaps_'+mouse_recday+'_neuron_'+str(neuron)+                '.svg', bbox_inches = 'tight', pad_inches = 0)
    plt.tight_layout()
    fig2.savefig(Ephys_output_folder_dropbox+'Example_cells/AnchoredTaskmaps_'+mouse_recday+'_neuron_'+str(neuron)+                '.svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    

    best_shift_time=Anchor_trial_dic['Best_shift_time'][mouse_recday][neuron]
    print(best_shift_time)
    #print(ref_tasks_all)
    
    print('')

    
    print('Lagged Spatial maps')

    states=['A','B','C','D']
    
    plt.rcParams["figure.figsize"] = (20,5)
    plt.errorbar(-np.arange(12),mean_corrs[neuron],yerr=sem_corrs[neuron])
    plt.axhline(thresholds_[neuron],ls='dashed',color='black')
    #plt.scatter(-field_peak_bins_[neuron],np.repeat(np.max(mean_corrs[neuron])+0.05,len(field_peak_bins_[neuron])),\
    #           marker='*')
    plt.tight_layout()
    plt.savefig(Ephys_output_folder_dropbox+'Example_cells/Lagged_Spatial_correlations_'+                mouse_recday+'_neuron_'+str(neuron)+'.svg', bbox_inches = 'tight', pad_inches = 0)
    
    plt.show()
    
    node_edge_rate_matrices=Spatial_anchoring_dic['Phase_shifted_node_edge_matrices'][mouse_recday][neuron]
    node_rate_matrices=Spatial_anchoring_dic['Phase_shifted_node_matrices'][mouse_recday][neuron]
    
    if plot_edge==True:
        matrices_plotted=node_edge_rate_matrices
        gridX=Task_grid_plotting2
    else:
        matrices_plotted=node_rate_matrices
        gridX=Task_grid_plotting

    mouse=mouse_recday[:4]
    rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
    rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
    structure_nums=np.unique(rec_day_structure_numbers)

    structures_all=[]
    for awake_ses_ind_ind, awake_session_ind in enumerate(non_repeat_ses):   
        structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]            [rec_day_session_numbers[awake_session_ind]]
        
        structures_all.append(structure)

        fig1, f1_axes = plt.subplots(figsize=(20, 2),ncols=num_lags, constrained_layout=True)
        for lag_ind, lag in enumerate(np.flip(np.arange(num_lags))):

            node_edge_mat_state=matrices_plotted[awake_ses_ind_ind,lag]
            mat_used=node_edge_mat_state
            

            ax=f1_axes[lag_ind]
            for state_port_ind, state_port in enumerate(states):
                node=structure[state_port_ind]-1
                ax.text(gridX[node,0]-0.25, gridX[node,1]+0.25,                        state_port.lower(), fontsize=22.5)

            ax.matshow(mat_used, cmap='coolwarm') #vmin=min_rate, vmax=max_rate
            ax.axis('off')
            
            #ax.savefig(str(neuron)+state+str(awake_session_ind)+'discmap.svg')
        plt.savefig(Ephys_output_folder_dropbox+'Example_cells/Lagged_Spatial_maps_'+                mouse_recday+'_neuron_'+str(neuron)+'_session'+str(awake_ses_ind_ind)+'.svg',                    bbox_inches = 'tight', pad_inches = 0)
    plt.axis('off')
    
    plt.show()
    print(structures_all)

plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


# In[ ]:


for ses_ind_ind in np.arange(len(non_repeat_ses)):


    anchors=Anchor_topN_GLM_dic['Anchors_per_neuron'][mouse_recday][ses_ind_ind][neuron]
    coeffs_neuron=GLM_anchoring_dic['coeffs_all'][mouse_recday][neuron][ses_ind_ind]
    coeffs_neuron_reshaped=coeffs_neuron.reshape((num_locations*num_phases,num_lags))
    plt.matshow(coeffs_neuron_reshaped)
    for n in np.arange(num_nodes):
        plt.axhline(3*n-0.5,color='white',ls='dashed')
    plt.show()
    print(anchors)


# In[ ]:


ses_ind_ind=1
phase_=0
location_=3
Anchor_topN_GLM_dic['Neurons_per_anchor'][mouse_recday]                        [ses_ind_ind][phase_][location_]


# In[ ]:


plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


# In[ ]:





# In[ ]:


'''
Matched neurons:

ah04_01122021_02122021: 72, 115



'''


# In[ ]:





# In[ ]:


phase_peaks=tuning_singletrial_dic2['tuning_phase_boolean_max'][mouse_recday][0]
pref_phase_neurons=np.argmax(phase_peaks,axis=1)
pref_phase=pref_phase_neurons[neuron]


# In[ ]:


phase_peaks[neuron]


# In[ ]:





# In[ ]:


print(mouse_recday)
num_neurons=len(cluster_dic['good_clus'][mouse_recday])

#Importing Ephys

num_sessions=len(session_dic_behaviour['awake'][mouse_recday])

##defining sessions to use
sessions=Task_num_dic[mouse_recday]
num_refses=len(np.unique(sessions))
num_comparisons=num_refses-1
repeat_ses=np.where(rank_repeat(sessions)>0)[0]
non_repeat_ses=non_repeat_ses_maker(mouse_recday) 
num_nonrepeat_sessions=len(non_repeat_ses)




##Importing Occupancy
#print('Importing Occupancy')
#name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
#data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
#occupancy_ = load(data_filename_memmap)#, mmap_mode='r')


corr_all_all=[]
corr_all_trials_all=[]
corr_all_trials_half2_all=[]
auto_corr_all_trials_all=[]
innerp_all_trials_all=[]
Beta_all=[]
Beta_uncorrected_all=[]

best_shift_all=np.zeros((num_neurons,num_nonrepeat_sessions))
best_shift_trials_all=np.zeros((num_neurons,num_nonrepeat_sessions))
best_shift_trials_half2_all=np.zeros((num_neurons,num_nonrepeat_sessions))
phase_location_all=np.zeros((num_nonrepeat_sessions,num_neurons,2))

best_shift_all[:]=np.nan
best_shift_trials_all[:]=np.nan
best_shift_trials_half2_all[:]=np.nan

print('Importing Ephys')


ses_ind_ind=0
ses_ind=non_repeat_ses[ses_ind_ind]

print(ses_ind)
#####
name='standardized_spike_events_dic_'+mouse_recday+'_'+str(ses_ind)
data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
ephys_ = load(data_filename_memmap)#, mmap_mode='r')


#occupancy_mat=data_matrix(occupancy_[ses_ind])
#occupancy_conc=np.concatenate(np.hstack(occupancy_mat))

location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
if len(location_mat_)==0:
    continue
occupancy_mat=np.reshape(location_mat_,(4,len(location_mat_),len(location_mat_.T)//4))
occupancy_conc=np.concatenate(location_mat_)

phase_mat=np.zeros(np.shape(occupancy_mat))
phase_mat[:,:,30:60]=1
phase_mat[:,:,60:90]=2

#phase_mat=np.zeros(np.shape(occupancy_mat))
#phase_mat[:,:,45:90]=1
#phase_mat[:,:,60:90]=2
phase_conc=np.concatenate(np.hstack(phase_mat))
occupancy_mat_=np.hstack(occupancy_mat)
phase_mat_=np.hstack(phase_mat)

ephys_neuron_=ephys_[0]
neuron_mat=data_matrix(ephys_neuron_,concatenate=False)



tone_aligned_activity=np.hstack(neuron_mat)
min_trials=int(np.min([len(occupancy_mat_),len(tone_aligned_activity)]))

corr_all_ses=np.zeros((num_neurons,360))
corr_all_trials_ses=np.zeros((num_neurons,num_shifts))
auto_corr_all_trials_ses=np.zeros((num_neurons,num_shifts))
innerp_all_trials_ses=np.zeros((num_neurons,num_shifts))
beta_neuron_means_ses=np.zeros((num_neurons,num_shifts))
beta_neuron_means_uncorrected_ses=np.zeros((num_neurons,num_shifts))

corr_all_ses[:]=np.nan
corr_all_trials_ses[:]=np.nan
auto_corr_all_trials_ses[:]=np.nan
innerp_all_trials_ses[:]=np.nan
beta_neuron_means_ses[:]=np.nan
beta_neuron_means_uncorrected_ses[:]=np.nan

########

neuron=29


if use_individualsession_anchor==True:
    anchors=(Spatial_anchoring_dic['best_node_phase_used'][mouse_recday][:,:,neuron]).astype(int)
    phase_=anchors[0,ses_ind_ind]
    location_=anchors[1,ses_ind_ind]
    location=location_+1
else:
    #anchor=(Spatial_anchoring_dic['best_node_phase'][mouse_recday][neuron]).astype(int)
    anchor=(Spatial_anchoring_dic['most_common_anchor'][mouse_recday][neuron]).astype(int)
    location_=anchor[1]
    phase_=anchor[0]
    location=int(location_+1)

phase_location_all[ses_ind_ind,neuron]=phase_,location_

ephys_neuron_=ephys_[neuron]
neuron_mat=data_matrix(ephys_neuron_,concatenate=False)
neuron_conc=np.concatenate(np.hstack(neuron_mat))

###tone aligned activity
tone_aligned_activity=np.hstack(neuron_mat)
anchor_mat=(np.logical_and(phase_mat_==phase_,occupancy_mat_==location)).astype(int)
min_trials=int(np.min([len(anchor_mat),len(tone_aligned_activity)]))

tone_aligned_activity_matched=tone_aligned_activity[:min_trials]
anchor_mat_matched=anchor_mat[:min_trials]
corr_all=np.zeros(360)
corr_all[:]=np.nan

###taking only trials with anchor visits that last above the duration threshold
anchor_mat_refined=np.zeros((len(anchor_mat_matched),360))
indices_refined=[np.where(np.logical_and(rank_repeat2(anchor_mat_matched[ii])>thr_visit,anchor_mat_matched[ii]==1))[0]for ii in range(len(anchor_mat_matched))]
for ii in range(len(anchor_mat_matched)):
    anchor_mat_refined[ii,indices_refined[ii]]=1

mean_anchor_refined=np.mean(anchor_mat_refined[np.sum(anchor_mat_refined,axis=1)>0],axis=0)
mean_neuron_refined=np.mean(tone_aligned_activity_matched[np.sum(anchor_mat_refined,axis=1)>0],axis=0)



for shift in range(360):
    tone_aligned_activity_shifted=np.roll(tone_aligned_activity_matched,-shift)#,axis=1)

    if use_mean==False:
        corr_mat=np.corrcoef(anchor_mat_matched,tone_aligned_activity_shifted)
        cross_corr_mat=corr_mat[min_trials:,:min_trials]
        corr_all[shift]=np.nanmean(np.diagonal(cross_corr_mat))
    elif use_mean==True:
        #mean_neuron=np.mean(tone_aligned_activity_shifted,axis=0)
        #mean_anchor=np.mean(anchor_mat_matched,axis=0)
        #corr_all[shift]=st.pearsonr(mean_anchor,mean_neuron)[0]
        
        mean_neuron_shifted=np.roll(mean_neuron_refined,-shift)
        corr_all[shift]=st.pearsonr(mean_anchor_refined,mean_neuron_shifted)[0]

best_shift=np.argmax(corr_all)


# In[ ]:





# In[ ]:


timestamps_start
plt.matshow(aligned_activity)


# In[ ]:





# In[ ]:


np.argmax(mean_)


# In[ ]:


plt.matshow(anchor_mat_matched)
plt.show()
plt.matshow(tone_aligned_activity_matched)
plt.show()
#print(corr_all)
print(best_shift)

plt.matshow(anchor_mat_refined[np.sum(anchor_mat_refined,axis=1)>0])
plt.show()
plt.matshow(tone_aligned_activity_matched[np.sum(anchor_mat_refined,axis=1)>0])
plt.show()
#print(corr_all)
print(best_shift)

plt.plot(mean_neuron_refined)


# In[ ]:


xx=anchor_mat_refined[np.sum(anchor_mat_refined,axis=1)>0]

[np.where(xx[ii]>0)[0]%360 for ii in range(len(xx))]


# In[ ]:


occupancy_conc


# In[ ]:


'''
1-ephys
2-occupancy
3-anchor place/phase
2-

'''


# In[ ]:


timestamps_start


# In[ ]:





# In[ ]:





# In[ ]:


##Plotting anchor visits vs number of peaks
###'ah04_01122021_02122021' 36, 90 
###'ah04_05122021_06122021' 37, 103
###'ah04_09122021_10122021' 47


print(mouse_recday)
#Importing Ephys
print('Importing Ephys')
num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
num_neurons=len(cluster_dic['good_clus'][mouse_recday])

for session in np.arange(num_sessions):
    name='standardized_spike_events_dic_'+mouse_recday+'_'+str(session)
    data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
    ephys_ = load(data_filename_memmap)#, mmap_mode='r')
    exec('ephys_ses_'+str(session)+'_=ephys_')
    
##Importing Occupancy
#print('Importing Occupancy')
#name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
#data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
#occupancy_ = load(data_filename_memmap)#, mmap_mode='r')
    
sessions=Task_num_dic[mouse_recday]
num_refses=len(np.unique(sessions))
num_comparisons=num_refses-1
repeat_ses=np.where(rank_repeat(sessions)>0)[0]
non_repeat_ses=non_repeat_ses_maker(mouse_recday)  


##################
neuron=47
anchors=(Spatial_anchoring_dic['best_node_phase_used'][mouse_recday][:,:,neuron]).astype(int)

print(anchors)

print(Anchor_tuning_dic[mouse_recday][neuron])

test_comp=0
fontsize=10
print('neuron'+str(neuron))

best_anchor_phase_node=Spatial_anchoring_dic['best_node_phase_used'][mouse_recday][:,:,neuron][:,test_comp]

phase_=best_anchor_phase_node[0]
location_=best_anchor_phase_node[1]
location=location_+1 ##because 0 based indexing used here but 1 based in locations


print(location)
print(phase_)

#session=0
mean_smooth_all=np.zeros((num_sessions,360))


for ses_ind, session in enumerate(non_repeat_ses):
    print(session)
    exec('ephys_=ephys_ses_'+str(session)+'_')
    #occupancy_mat=data_matrix(occupancy_[session])
    #occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
    
    location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
    if len(location_mat_)==0:
        continue
    occupancy_mat=np.reshape(location_mat_,(4,len(location_mat_),len(location_mat_.T)//4))
    occupancy_conc=np.concatenate(location_mat_)

    phase_mat=np.zeros(np.shape(occupancy_mat))
    phase_mat[:,:,30:60]=1
    phase_mat[:,:,60:90]=2

    #phase_mat=np.zeros(np.shape(occupancy_mat))
    #phase_mat[:,:,45:90]=1
    #phase_mat[:,:,60:90]=2
    phase_conc=np.concatenate(np.hstack(phase_mat))



    ephys_neuron_=ephys_[neuron]
    neuron_mat=data_matrix(ephys_neuron_,concatenate=False)
    neuron_conc=np.concatenate(np.hstack(neuron_mat))


    ###tone aligned activity
    
    occupancy_mat_=np.hstack(occupancy_mat)
    phase_mat_=np.hstack(phase_mat)
    anchor_mat=(np.logical_and(phase_mat_==phase_,occupancy_mat_==location)).astype(int)
    plt.matshow(anchor_mat)
    plt.savefig(Ephys_output_folder_dropbox+'Example_cells/AnchorMat_'+mouse_recday+'_neuron_'+str(neuron)+                    '_ses_'+str(session)+'.svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    tone_aligned_activity=np.hstack(neuron_mat)
    plt.matshow(tone_aligned_activity)
    plt.savefig(Ephys_output_folder_dropbox+'Example_cells/ActivityMat_'+mouse_recday+'_neuron_'+str(neuron)+                    '_ses_'+str(session)+'.svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    min_trials=int(np.min([len(anchor_mat),len(tone_aligned_activity)]))
    
    tone_aligned_activity_matched=tone_aligned_activity[:min_trials]
    anchor_mat_matched=anchor_mat[:min_trials]
    corr_all=np.zeros(360)
    for shift in range(360):
        tone_aligned_activity_shifted=np.roll(tone_aligned_activity_matched,-shift)
        corr_mat=np.corrcoef(anchor_mat_matched,tone_aligned_activity_shifted)
        cross_corr_mat=corr_mat[min_trials:,:min_trials]
        corr_all[shift]=np.nanmean(np.diagonal(cross_corr_mat))

    best_shift=np.argmax(corr_all)

    tone_aligned_activity_shifted_best=np.roll(tone_aligned_activity_matched,best_shift)
    corr_mat=np.corrcoef(anchor_mat_matched,tone_aligned_activity_shifted_best)
    cross_corr_mat=corr_mat[min_trials:,:min_trials]
    #plt.matshow(cross_corr_mat)
    #plt.show()
    plt.plot(corr_all)
    plt.show()
    
    corr_all_time=np.zeros(len(anchor_mat_matched))
    for shift in range(len(anchor_mat_matched)):
        tone_aligned_activity_shiftedT=np.roll(tone_aligned_activity_shifted_best.T,-shift)
        corr_mat=np.corrcoef(anchor_mat_matched.T,tone_aligned_activity_shiftedT)
        cross_corr_mat=corr_mat[360:,:360]
        corr_all_time[shift]=np.nanmean(np.diagonal(cross_corr_mat))

    best_shift=np.argmax(corr_all_time)
    plt.plot(corr_all_time)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


##Plotting example trial-by-trial anchor visits and neural activity
## 'ah04_05122021_06122021' neuron27 ses0, neuron26 session5 (note, neuron 29 in session5 active earlier in phase 
##- reward locations are unknown on first trial and so makes sense that neuron fires earier than its "supposed" to)

### 'ah04_01122021_02122021' neuron107 session0; neuron51 session2
### 'me11_05122021_06122021' neuron9 and neuron 17 session0
state_ind=0
mouse_recday='me11_05122021_06122021'
activity_time='bump_time'
norm_FR_performance=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['norm_FR_performance'][mouse_recday]
neurons_used_all=Regression_anchors_zeroshot_dic[state_ind][str(activity_time)]['neurons_used'][mouse_recday]

Best_shift_time_=Anchor_trial_dic['Best_shift_time'][mouse_recday]
            

print(mouse_recday)
#Importing Ephys
print('Importing Ephys')
num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
num_neurons=len(cluster_dic['good_clus'][mouse_recday])

for ses_ind in np.arange(num_sessions):
    name='standardized_spike_events_dic_'+mouse_recday+'_'+str(ses_ind)
    data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
    ephys_ = load(data_filename_memmap)#, mmap_mode='r')
    exec('ephys_ses_'+str(ses_ind)+'_=ephys_')
    

    
sessions=Task_num_dic[mouse_recday]
num_refses=len(np.unique(sessions))
num_comparisons=num_refses-1
repeat_ses=np.where(rank_repeat(sessions)>0)[0]
non_repeat_ses=non_repeat_ses_maker(mouse_recday)  

##################


ses_ind_zero_shot=np.where(norm_FR_performance[:,1]==1)[0]
ses_ind_nozero_shot=np.where(norm_FR_performance[:,1]==0)[0]

dicX={'zero-shot':ses_ind_zero_shot, 'nonzero-shot':ses_ind_nozero_shot}

bump_time_array_dic=rec_dd()
for condition, ses_ind_array in dicX.items():
    print(condition)
    
    mean_FR=[]
    
    for ses_ind_ind in ses_ind_array:
        ses_ind=non_repeat_ses[ses_ind_ind]
        print('session'+str(ses_ind))
        #print(norm_FR_performance[ses_ind_ind])
        exec('ephys_=ephys_ses_'+str(ses_ind)+'_')
        neurons_used=neurons_used_all[ses_ind_ind]
        best_shift_times=Best_shift_time_[neurons_used,ses_ind]
        
        
        bump_time_array=np.zeros((len(neurons_used),2))
        
        for neuron_ind, neuron in enumerate(neurons_used):
            print('neuron'+str(neuron))
            best_shift_neuron=Anchor_trial_dic['Best_shift_time'][mouse_recday][neuron][ses_ind]
            print(best_shift_neuron)

            ephys_neuron_=ephys_[neuron]
            neuron_mat=data_matrix(ephys_neuron_,concatenate=False)
            neuron_conc=np.concatenate(np.hstack(neuron_mat))
            
            
            

            times_=np.asarray([359])
            gap=(num_trials_neural+1*360)-best_shift_times[neuron_ind]
            ##i.e. times when neurons should be active on ring attractor
            neuron_bump_time_start_=times_-gap ##definitely looking at activity BEFORE anchor visit
            neuron_bump_time_start=(neuron_bump_time_start_).astype(int)

            ##defining normalised mean activity at selected times (ranging from time to 30 degrees 
            ##later)
            

            tone_aligned_activity=np.hstack(neuron_mat)[0]
            tone_aligned_activity_norm=tone_aligned_activity/np.mean(tone_aligned_activity)
            plt.plot(tone_aligned_activity_norm)
            plt.fill_between(np.arange(30)+neuron_bump_time_start,np.max(tone_aligned_activity_norm)                             ,color='black',alpha=0.25)
            
            mean_activity_bump_norm=np.asarray([np.mean(tone_aligned_activity_norm[neuron_bump_time_start[ii]:                                                               neuron_bump_time_start[ii]+30])                                           for ii in range(len(neuron_bump_time_start))])
            #mean_activity_bump[np.isnan(neuron_bump_time_start_)]=np.nan
            #mean_activity_bump_norm=mean_activity_bump/np.mean(tone_aligned_activity)
            mean_FR.append(mean_activity_bump)
            
            bump_time_array[neuron_ind]=[int(neuron_bump_time_start),mean_activity_bump_norm]
            
            plt.savefig(Ephys_output_folder_dropbox+'Example_cells/'+condition+'_Activity_bumptime_'+mouse_recday+                        '_neuron_'+str(neuron)+'_ses_'+str(ses_ind)+'.svg', bbox_inches = 'tight', pad_inches = 0)
            plt.show()
    
        
        bump_time_array_dic[mouse_recday][condition][ses_ind_ind]=bump_time_array   
            



            
    print('Mean FR: '+str(np.nanmean(mean_FR)))
    print('')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
plot:
-activity of each neuron at its shift time on same plot for each task
-do this across tasks and compare size of *bump*
-for video: interpolate bump between timepoints

'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


##Plotting example trial-by-trial anchor visits and neural activity
###'ah04_09122021_10122021' order_max=1 (phase 2, location3) - neuron 2,8,84
##'me11_01122021_02122021' order_max=3 (phase2, location5 - neurons 0,14)
##'ah03_18082021_19082021' order_max=2, (phase 2, location 5 - neuron 5)

order_max=3
mouse_recday='me11_01122021_02122021'
activity_time = 'bump_time'
num_trials_neural=0
print(mouse_recday)
#Importing Ephys
print('Importing Ephys')
num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
num_neurons=len(cluster_dic['good_clus'][mouse_recday])

for ses_ind in np.arange(num_sessions):
    name='standardized_spike_events_dic_'+mouse_recday+'_'+str(ses_ind)
    data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
    ephys_ = load(data_filename_memmap)#, mmap_mode='r')
    exec('ephys_ses_'+str(ses_ind)+'_=ephys_')
    
##Importing Occupancy
#print('Importing Occupancy')
#name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
#data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
#occupancy_ = load(data_filename_memmap)#, mmap_mode='r')
    
sessions=Task_num_dic[mouse_recday]
num_refses=len(np.unique(sessions))
num_comparisons=num_refses-1
repeat_ses=np.where(rank_repeat(sessions)>0)[0]
non_repeat_ses=non_repeat_ses_maker(mouse_recday)  

thr_upper=330
thr_lower=30

##################

betas_day=Regression_anchors_dic[str(activity_time)][num_trials_neural]['neuron_betas'][mouse_recday]

beta_day_flat=remove_nan(np.concatenate((np.concatenate(betas_day))))
beta_day_flat_sorted=np.flip(np.sort(beta_day_flat))

max_coords=np.where(betas_day==beta_day_flat_sorted[order_max])

beta_max=betas_day[max_coords[0][0],max_coords[1][0],max_coords[2][0]]
print('Beta max')
print(beta_max)


ses_ind_ind=max_coords[0][0]
ses_ind=non_repeat_ses[ses_ind_ind]
phase_,location_=max_coords[1][0],max_coords[2][0]
location=location_+1


##what is the anchor?
phase_locations_neurons_=((Spatial_anchoring_dic['best_node_phase_used'][mouse_recday]                      [:,ses_ind_ind,:]).astype(int)).T
phase_locations_neurons=(np.column_stack((phase_locations_neurons_[:,0],                                         phase_locations_neurons_[:,1]+1))).astype(int)
neurons_anchored=np.where(np.logical_and(phase_locations_neurons[:,0]==phase_,                        phase_locations_neurons[:,1]==location))[0]

fontsize=10
print('ses_ind')
print(ses_ind)
print('ses_ind_ind')
print(ses_ind_ind)
print('Phase')
print(phase_)
print('location')
print(location)
#print(Anchor_tuning_dic[mouse_recday][neuron])


exec('ephys_=ephys_ses_'+str(ses_ind)+'_')
#occupancy_mat=data_matrix(occupancy_[ses_ind])
#occupancy_conc=np.concatenate(np.hstack(occupancy_mat))

location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
if len(location_mat_)==0:
    continue
occupancy_mat=np.reshape(location_mat_,(4,len(location_mat_),len(location_mat_.T)//4))
occupancy_conc=np.concatenate(location_mat_)

phase_mat=np.zeros(np.shape(occupancy_mat))
phase_mat[:,:,30:60]=1
phase_mat[:,:,60:90]=2
phase_conc=np.concatenate(np.hstack(phase_mat))




###tone aligned activity

occupancy_mat_=np.hstack(occupancy_mat)
phase_mat_=np.hstack(phase_mat)
anchor_mat=(np.logical_and(phase_mat_==phase_,occupancy_mat_==location)).astype(int)
plt.matshow(anchor_mat)
plt.savefig(Ephys_output_folder_dropbox+'Example_cells/AnchorMat_'+mouse_recday+'_neuron_'+str(neuron)+                '_ses_'+str(ses_ind)+'.svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()


for neuron in neurons_anchored:
    print('neuron'+str(neuron))
    best_shift_neuron=Anchor_trial_dic['Best_shift_time'][mouse_recday][neuron][ses_ind]
    print(best_shift_neuron)
    
    if best_shift_neuron<thr_lower or best_shift_neuron>thr_upper:
        print('Neuron outside of threshold range for regression analysis - not used to calculate beta')
    ephys_neuron_=ephys_[neuron]
    neuron_mat=data_matrix(ephys_neuron_,concatenate=False)
    neuron_conc=np.concatenate(np.hstack(neuron_mat))

    tone_aligned_activity=np.hstack(neuron_mat)
    #tone_aligned_activity=np.vstack((np.zeros(len(tone_aligned_activity.T)),tone_aligned_activity))
    plt.matshow(tone_aligned_activity)
    
    anchor_timestamps=[np.where(anchor_mat[ii]>0)[0] for ii in range(len(anchor_mat))]
    plt.eventplot(anchor_timestamps,color='yellow',linewidths=4)

    
    plt.savefig(Ephys_output_folder_dropbox+'Example_cells/ActivityMat_'+mouse_recday+'_neuron_'+str(neuron)+                    '_ses_'+str(ses_ind)+'.svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


neurons_anchored


# In[ ]:


'''
1-find group of anchored neurons with high beta
2-go through steps for logistic regression till you get the final arrays
3-plot the logistic regression
4-calculate the value

5-repeat 1-4 for several high and low beta values
6-repeat 1-4 but shifting back several trials

'''


# In[ ]:


np.shape(Anchor_trial_dic['Best_shift_time'][mouse_recday])


# In[ ]:


###testing logistic regression on example above
'''note: some variables defined in the cell above'''

print(mouse_recday)

num_trials_tested=5 ##how many trials back to coregress out (to control for autocorrelation in behaviour)
num_trials_neural=0 ##how many trials back to take neuronal activity from (to look at attractor properties)
activity_time = 'bump_time'

#Importing Ephys
print('Importing Ephys')

num_sessions=len(session_dic_behaviour['awake'][mouse_recday])
num_neurons=len(cluster_dic['good_clus'][mouse_recday])

for session in np.arange(num_sessions):
    name='standardized_spike_events_dic_'+mouse_recday+'_'+str(session)
    data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
    ephys_ = load(data_filename_memmap)#, mmap_mode='r')
    exec('ephys_ses_'+str(session)+'_=ephys_')

##Importing Occupancy
#print('Importing Occupancy')
#name='state_occupancy_dic_occupancy_normalized_'+mouse_recday
#data_filename_memmap = os.path.join(Intermediate_object_folder_dropbox, name)
#occupancy_ = load(data_filename_memmap)#, mmap_mode='r')

###what is the lag between the neuron's firing and the anchor? 
Best_shift_time_=Anchor_trial_dic['Best_shift_time'][mouse_recday]

###defining all non-spatial neurons (i.e. neurons with greater than threshold lag from their anchor)
nonzero_anchored_neurons=np.where(np.logical_and(Best_shift_time_[:,ses_ind]>thr_lower,                              Best_shift_time_[:,ses_ind]<thr_upper))[0]

###defining ephys, occupancy and phase matrices for this session
exec('ephys_=ephys_ses_'+str(ses_ind)+'_')
#occupancy_mat=data_matrix(occupancy_[ses_ind])
#occupancy_conc=np.concatenate(np.hstack(occupancy_mat))
location_mat_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(ses_ind)+'.npy')
if len(location_mat_)==0:
    continue
occupancy_mat=np.reshape(location_mat_,(4,len(location_mat_),len(location_mat_.T)//4))
occupancy_conc=np.concatenate(location_mat_)

phase_mat=np.zeros(np.shape(occupancy_mat))
phase_mat[:,:,30:60]=1
phase_mat[:,:,60:90]=2
phase_conc=np.concatenate(np.hstack(phase_mat))

occupancy_mat_=np.hstack(occupancy_mat)
phase_mat_=np.hstack(phase_mat)

###looping over all location-phase conjunctions to find neurons anchored to them and do the regression
location=location_+1 ## location arrays are not using zero-based indexing 

##defining place/phase visits
placephase_mat=(np.logical_and(phase_mat_==phase_,occupancy_mat_==location)).astype(int)
placephase_conc=(np.logical_and(phase_conc==phase_,occupancy_conc==location)).astype(int)
visits=np.where(placephase_conc==1)[0]

###where are the location/phase conjunctions one step away from the anchor?
##This defines decision times
prev_locations_=np.where(mindistance_mat[location_]==1)[0]
prev_locations=prev_locations_+1
prev_phase_=(phase_-1)%3
location_conc_prev_=np.max(np.asarray([occupancy_conc==prev_locations[ii] for ii                              in range(len(prev_locations))]),axis=0)
location_conc_prev=location_conc_prev_>0 ##i.e. when visiting location one step from 
##anchor location
placephase_conc_prev=(np.logical_and(phase_conc==prev_phase_                                     ,location_conc_prev==True)).astype(int)
#i.e. when visiting location one step from anchor location AND phase one step from anchor phase 
visits_prev=np.where(placephase_conc_prev==1)[0]
visits_start=visits[np.hstack((360,np.diff(visits)))>30] ##only the start of a visits 
##in a given phase
visits_start=visits_start[visits_start>360] ###removing first trial
visits_prev_end=visits_prev[np.hstack((np.diff(visits_prev),360))>30] 
##only the end of a visit in a given phase
visits_prev_end=visits_prev_end[visits_prev_end>360] ###removing first trial
visits_prev_end_nodes=occupancy_conc[visits_prev_end] 
##which nodes are visited at decision points

##where does the animal actually go after the decision points i.e when one step away
##from anchor (visits_prev_end) - it can go to anchor or not
visited_node_start=np.zeros((len(visits_prev_end),2))
visited_node_start[:]=np.nan
for visit_prev_ind, visit_prev in enumerate(visits_prev_end):
    next_90_occ=occupancy_conc[visit_prev+1:visit_prev+91]
    next_90_phase=phase_conc[visit_prev+1:visit_prev+91]

    next_node_=next_90_occ[np.logical_and(next_90_occ<10,next_90_occ!=                                         visits_prev_end_nodes[visit_prev_ind])]
    ##only taking nodes (not edges) and only nodes that arent the same as the previous nodes
    ##(because could have moved phases but stayed in same node)

    ###excluding trials where animal stayed in same location across two phases
    ###because then cant define decision point
    if len(next_90_occ)==0 or len(next_node_)==0: 
        continue
    next_node=next_node_[0] ###the very first node visited after decision point
    next_node_phase_start_=np.where(np.logical_and(next_90_phase==phase_,                                                   next_90_occ==next_node))[0]
    ##when is next node (cont)...visited at the next phase

    ##whats the first bin where the next node is visited in the next phase  
    if len(next_node_phase_start_)==0 or    mindistance_mat[int(visits_prev_end_nodes[visit_prev_ind]-1),int(next_node-1)]!=1:
        ##note: the second condition is to deal with erroneous tracking where animal position
        ##jumps more than one node between bins
        next_node=np.nan
        next_node_phase_start=np.nan
    else:
        next_node_phase_start=next_node_phase_start_[0]

    visited_node_start[visit_prev_ind]=np.asarray([next_node,visit_prev+next_node_phase_start])

###we now have visited_node_start which has both the visited node and its timestamp following
#each
##decision point (decision point being when animal was one place and one phase away from anchor)


##Defining dependent variable (location/phase visits)
location_visited_bool=(visited_node_start[:,0]==location).astype(int) ## 1) DEPENDENT VARIABLE
##i.e. when did animal visit the anchor after each decision point
##the length of this array is all the times where animal was one location and one phase away
##from anchoe
times_=visited_node_start[:,1] ##times for dependent variable
nodes_=visited_node_start[:,0] ##nodes visited for dependent variable

### Did animal visit the anchor location on N previous trials?
## we regress this out to remove any effects of autocorrelation in behaviour 
trial_lag_booleans_all=np.zeros((num_trials_tested,len(times_)))
trial_lag_booleans_all[:]=np.nan
for trial_lag_ in np.arange(num_trials_tested):
    trial_lag=trial_lag_+1
    range_tested=[360*(num_trials_neural+trial_lag)-30,360*(num_trials_neural+trial_lag)+30]
    ##what range of time lags should we use to look for previous trial visits 
    ##using num_trials_neural here when predicting behaviour using neural activity from M trials
    ##back, this means now the coregressors (animal's previous choices) for the behaviour are
    ##lagged by exactly N+M trials back (with tolerance of +/- 30 degrees))


    ## for each bin in times_ (i.e. each timestamp following each decision point) what (if any)
    ## is the bin where animal visited the same location/phase N+M trials back
    ##Note: for trials less then N+M you will effectively never have visited the same anchor 
    ##at this trial lag and so the co-regressors for this trial lag will always be zero 
    visit_at_lag=[np.where(np.logical_and((times_<(times_[ii]-range_tested[0])),                                          (times_>(times_[ii]-range_tested[1]))))[0]    for ii in range(len(times_))]

    ###did animal visit the anchor location M+N trials before the current visit time?
    trial_lag_boolean=np.asarray([np.sum([nodes_[visit_at_lag[ii][jj]]==location                                          for jj in range(len(visit_at_lag[ii]))])                if len(visit_at_lag[ii]>0) else 0 for ii in range(len(nodes_))])
    trial_lag_boolean[trial_lag_boolean>0]=1
    trial_lag_booleans_all[trial_lag_]=trial_lag_boolean ## 2) CO-REGRESSORS

    ###Now you have trial_lag_booleans_all which tells you when you visited place/phase anchor
    ##exactly N+M trials in the past for each N (and a fixed M) - note M=0 for the main analysis


##find neurons anchored to this location/phase with non-zero distance
neurons_anchorednext=np.where(np.logical_and(phase_locations_neurons[:,0]==phase_,                                             phase_locations_neurons[:,1]==location))[0]
neurons_anchorednext_nonzero=np.intersect1d(neurons_anchorednext,nonzero_anchored_neurons)

##Below were getting  mean activityof neurons at different times before anchor visit
##first defining arrays
mean_activity_bump_neurons=np.zeros((len(neurons_anchorednext_nonzero),len(times_)))
mean_activity_bump_neurons[:]=np.nan


##whats the lag for each neuron to its anchor? 
best_shift_times=Best_shift_time_[neurons_anchorednext_nonzero,ses_ind]

###defining primary independent variable (neurons activity at defined time)
###defining time to take neuron's activity (for primary independent variable)
for neuron_ind, neuron in enumerate(neurons_anchorednext_nonzero):
    ephys_neuron_=ephys_[neuron]
    neuron_conc=data_matrix(ephys_neuron_,concatenate=True)      

    ##defining activity times
    if activity_time == 'bump_time':
        gap=(num_trials_neural+1*360)-best_shift_times[neuron_ind]
        ##i.e. times when neurons should be active on ring attractor
    elif activity_time == 'decision_time':
        gap=thr_lower
        ##i.e. time when animal is about to visit anchor
    elif activity_time == 'random_time':
        gap=random.randint(0,359)
    elif isinstance(activity_time,int)==True:
        gap=((360-best_shift_times[neuron_ind])+int(activity_time))%360
        ##times 90 degrees shifted from bump time
    neuron_bump_time_start_=times_-gap
    neuron_bump_time_start=(neuron_bump_time_start_).astype(int)

    ##defining normalised mean activity at selected times (ranging from time to 30 degrees later)
    mean_activity_bump=np.asarray([np.mean(neuron_conc[neuron_bump_time_start[ii]:                                                       neuron_bump_time_start[ii]+30])                                   for ii in range(len(neuron_bump_time_start))])
    mean_activity_bump[np.isnan(neuron_bump_time_start_)]=np.nan

    times_int=(times_).astype(int)
    mean_activity_trial=np.asarray([np.mean(neuron_conc[times_int[ii]:times_int[ii]+360])                                    for ii in range(len(times_int))])

    mean_activity_trial[mean_activity_trial==0]=np.nan ##to avoid dividing by zero
    mean_activity_bump_neurons[neuron_ind]=mean_activity_bump/mean_activity_trial
    ##i.e. firing rate as proportion of each neuron's mean firing on a given trial


meanofmeans_activity_bump_neurons=np.nanmean(mean_activity_bump_neurons,axis=0) ##3) INDPENDENT VARIABLE
##mean relative firing rate of ALL neurons anchored to a given place/phase

X_=np.column_stack((meanofmeans_activity_bump_neurons,trial_lag_booleans_all.T))
y_=location_visited_bool
X=X_[~np.isnan(meanofmeans_activity_bump_neurons)]
#X=np.vstack(([X[:,ii]-np.mean(X[:,ii]) for ii in range(len(X.T))])).T
X=np.column_stack((X[:,0]-np.mean(X[:,0]),X[:,1:])) ##de-meaning neuronal activity
y=y_[~np.isnan(meanofmeans_activity_bump_neurons)]

if np.sum(abs(np.diff(y)))==0: ### i.e. always visited (or always didnt visit) 
    ##place/phase from all decision points
    beta_all=np.repeat(np.nan,num_trials_tested+1)
    beta_neurons=np.nan
else:
    clf = LogisticRegression(solver='liblinear',penalty='l1').fit(X, y)
    beta_all=clf.coef_[0]
    beta_neurons=clf.coef_[0][0]

    
print(phase_)
print(location)
print(neurons_anchorednext_nonzero)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


print('Beta values')
print(beta_all)

intercept=clf.intercept_[0]

print('num features')
print(clf.n_features_in_)
print('Intercept')
print(intercept)

print('Probability of visit at mean activity and not visited previously - i.e. at intercept')
print(math.exp(intercept)/(1+math.exp(intercept)))


# In[ ]:


y


# In[ ]:


plt.scatter(X[:,0],y)


# In[ ]:


plt.scatter(X[:,1],y)


# In[ ]:





# In[ ]:


num_trial_back=1
non_repeat_visits=X[:,num_trial_back]!=y

plt.scatter(X[non_repeat_visits,0],y[non_repeat_visits])


# In[ ]:





# In[ ]:


'''
example 1:
ah04_01122021_02122021
anchor=1,5
session=4
neurons 32 and 101 (212 and 210 degrees away from anchor)


'''


# In[ ]:





# In[ ]:


'''

General checks:

-new change in initial histogram (fewer non zero neurons)
-why peak at zero disappears completely but analysis below still works (assuming the disappearance means
that new lags to anchor are different)

-regression analysis:
bugs in code
logic of analysis 
    - when visits are and when animal is one step away
    - all filters
    - previous trial visit calculation (using visits to the to be visited node and correct times)
    - consider changing condition to visits only at multiples of 360 - other visits could be misleading
    - any biases due to anchor selection or bump time calculation



'''


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
each visit -each time animal one step away from anchor


1-did animal visit location 360 degree back?
2- "" 720
3- "" 1080
...etc to 5 trials
-neural activity - mean neural activity of all neurons anchored at non-zero lag to anchor 

e.g. for one visit:

independent variables: 1 0 0 1 0 0.67 
dependent variable: 1
'''


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





# In[ ]:


xy_clean_all=[]
for mouse_recday in day_type_dicX[day_type]:
    Accuracy_all_day=np.nansum(Accuracy_anchors_dic['Accuracy_mat'][mouse_recday],axis=0)    /np.nansum(Accuracy_anchors_dic['Visits_mat'][mouse_recday],axis=0)

    Accuracy_chance_day=np.nanmean(Accuracy_anchors_dic['Accuracy_chance_mean_mat'][mouse_recday],axis=0)

    plt.matshow(Accuracy_all_day,cmap='coolwarm')
    plt.show()
    #xy_clean=column_stack_clean(np.concatenate(Accuracy_all_day),np.concatenate(Accuracy_chance_day))
    #bar_plotX(xy_clean.T,'none',0,1,'points','paired',0.025)
    #plt.show()
    #print(st.ttest_rel(xy_clean[:,0],xy_clean[:,1]))
    
    accuracy_mat_day=Accuracy_anchors_dic['Accuracy_mat'][mouse_recday]
    visits_mat_day=Accuracy_anchors_dic['Visits_mat'][mouse_recday]
    chance_mat_day=Accuracy_anchors_dic['Accuracy_chance_mean_mat'][mouse_recday]

    accuracy_per_ses=np.asarray([np.nansum(accuracy_mat_day[ses_ind])/np.nansum(visits_mat_day[ses_ind])                for ses_ind in np.arange(len(accuracy_mat_day))])
    chance_per_ses=np.asarray([np.nansum(visits_mat_day[ses_ind]*                                         chance_mat_day[ses_ind])/np.nansum(visits_mat_day[ses_ind])                               for ses_ind in np.arange(len(accuracy_mat_day))])
    xy_clean=column_stack_clean(accuracy_per_ses,chance_per_ses)
    bar_plotX(xy_clean.T,'none',0,1,'points','paired',0.025)
    plt.show()
    print(st.ttest_rel(xy_clean[:,0],xy_clean[:,1]))
    
    xy_clean_all.append(xy_clean)
    
    
    ###accuracy ratio
    
    #accuracy_ratio=np.asarray([np.concatenate(np.nanmean(Accuracy_anchors_dic['Accuracy_ratio_mean_mat'][mouse_recday]\
    #                                                     ,axis=0))])
    #bar_plotX(accuracy_ratio,'none',0,2,'points','paired',0.25)
    #plt.axhline(1,ls='dashed',color='black')
    #print(st.ttest_1samp(remove_nan(accuracy_ratio[0]),1))
xy_clean_all=np.vstack((xy_clean_all))
bar_plotX(xy_clean_all.T,'none',0,1,'points','paired',0.025)
plt.savefig(Ephys_output_folder_dropbox+'/Prediction_accuracy_neurontoanchorvisit_nonrepeated_meanused.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()

noplot_scatter(xy_clean_all[:,0],xy_clean_all[:,1],'black')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(Ephys_output_folder_dropbox+'/Prediction_accuracy_neurontoanchorvisit_nonrepeated_meanused_scatter.svg',            bbox_inches = 'tight', pad_inches = 0)
plt.show()
print(st.ttest_rel(xy_clean_all[:,0],xy_clean_all[:,1]))


# In[ ]:





# In[ ]:


'''
Checks:
- neurons
- anchors
- trials
- times before anchor visit
- booleans

'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




'''
1-accuracy mean
2-accuracy_chance
3-accuracy timecourse (accuracy boolean)
4-timepoints for timecourse (task time)
5-neurons in each group
6-relative time gap of each neuron to each anchor
7-actual_next_relativeactivity_ratio
8-actual_next_relativeactivity_ratio_mean


another boolean for anchored neurons?
'''


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





# In[ ]:





# In[ ]:


###Is number of peaks correlated with number of anchors?
num_peaks_all=np.vstack((dict_to_array(Anchor_tuning_dic['num_peaks'])))
num_peaks_mean_all=np.nanmean(num_peaks_all,axis=1)
num_peaks_mean_all_nozero=np.asarray([np.mean(num_peaks_all[neuron][num_peaks_all[neuron]>0])                               for neuron in np.arange(len(num_peaks_all))])

field_peak_bins_all_=np.hstack((dict_to_array(Phase_spatial_corr_dic['field_peak_bins'])))
num_anchors_all=np.asarray([len(field_peak_bins_all_[neuron]) for neuron in range(len(field_peak_bins_all_))])

sns.regplot(num_anchors_all,num_peaks_mean_all)
plt.show()
print(st.pearsonr(num_anchors_all,num_peaks_mean_all))

sns.regplot(num_anchors_all,num_peaks_mean_all_nozero)
plt.show()
xy=column_stack_clean(num_anchors_all,num_peaks_mean_all_nozero)
print(st.pearsonr(xy[:,0],xy[:,1]))


bar_plotX([num_peaks_mean_all[num_anchors_all==1],num_peaks_mean_all[num_anchors_all==2]],         'none',0,1,'nopoints','unpaired',0.025)
plt.show()

bar_plotX([num_peaks_mean_all_nozero[num_anchors_all==1],num_peaks_mean_all_nozero[num_anchors_all==2]],         'none',0,2,'nopoints','unpaired',0.025)
plt.show()


print('No')


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


num_fields_all=[]
for ses_ind in [0,1,2,4,5,6]:
    num_fields_all.append(flatten(num_field_dic['State']['combined'][ses_ind]))
    
num_fields_all=np.vstack((num_fields_all))

num_fields_sign_all=[]
for ses_ind_ind, ses_ind in enumerate([0,1,2,4,5,6]):
    num_fields_sign_ses=[]
    for mouse_recday in day_type_dicX['combined_ABCDonly']:
        num_fields_sign_ses.append(np.sum(tuning_singletrial_dic['tuning_state_boolean'][mouse_recday][ses_ind_ind],axis=1))
    num_fields_sign_ses=np.hstack((num_fields_sign_ses))
    num_fields_sign_all.append(num_fields_sign_ses)
num_fields_sign_all=np.vstack((num_fields_sign_all))

###Tuning using mean number of peaks
print('Tuning using mean number of peaks')
max_numfield=np.max(num_fields_all,axis=0)
min_numfield=np.min(num_fields_all,axis=0)

tuned_boolean=min_numfield<=3

print('Proportion drops off for at least one task')
print(len(np.where(max_numfield[tuned_boolean]>=4)[0])/len(max_numfield[tuned_boolean]))
print('')

print('Proportion drops off for each task')
prop_drop_ses_all=[]
for ses_ind in np.arange(len(num_fields_all)):
    num_field_ses=num_fields_all[ses_ind][tuned_boolean]
    prop_drop_ses=len(np.where(num_field_ses>=4)[0])/len(num_field_ses)
    #print(prop_drop_ses)
    prop_drop_ses_all.append(prop_drop_ses)
    
print(np.mean(prop_drop_ses_all))


###Tuning using significant peaks
print('')
print('_____________')
print('Tuning using significant peaks')
max_numfield=np.max(num_fields_sign_all,axis=0)
min_numfield=np.min(num_fields_sign_all,axis=0)

tuned_boolean=max_numfield>0

print('Proportion drops off for at least one task')
print(len(np.where(min_numfield[tuned_boolean]==0)[0])/len(max_numfield[tuned_boolean]))
print('')

print('Proportion drops off for each task')
prop_drop_ses_all=[]
for ses_ind in np.arange(len(num_fields_all)):
    num_field_ses=num_fields_sign_all[ses_ind][tuned_boolean]
    prop_drop_ses=len(np.where(num_field_ses==0)[0])/len(num_field_ses)
    #print(prop_drop_ses)
    prop_drop_ses_all.append(prop_drop_ses)
    
print(np.mean(prop_drop_ses_all))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###Testing predictions of one vs multiple anchors 

'''
Prediction of spatial anchoring:

1-Only one anchor - distal spatial cells less likely to be coherent than proximal ones - 
should converge to chance with enough transitions
2-Multiple anchors - never converges to chance - ideally no relationship between distance in task space and coherence
if anchors uniformly distributed - but can have intermediate regime
'''

policy_corr_distrib=np.concatenate(dict_to_array(Policy_corr_dic))


upper_thr=np.nanpercentile(policy_corr_distrib,75)
lower_thr=np.nanpercentile(policy_corr_distrib,25)
day_type='3_task' #combined_ABCDonly

if day_type=='combined_ABCDonly':
    day_type_='combined'
else:
    day_type_=day_type

Xneuron_correlations_spatial_spatial_ABCDonly_all=[]
Xneuron_correlations_nonspatial_spatial_ABCDonly_all=[]
Xneuron_tuningangle_spatial_spatial_ABCDonly_all=[]
Xneuron_tuningangle_nonspatial_spatial_ABCDonly_all=[]
for mouse_recday in day_type_dicX[day_type]:

    policy_corr=Policy_corr_dic[mouse_recday]
    np.shape(Xneuron_tuningangle_used_pairs_ABCDonly_dic[mouse_recday])

    used_pairs_=used_pairs_dic[mouse_recday]

    used_pairs_spatial_spatial=used_pairs_[:,np.logical_and(policy_corr[used_pairs_[0]]>upper_thr,                                                            policy_corr[used_pairs_[1]]>upper_thr)]
    used_pairs_nonspatial_spatial=used_pairs_[:,np.logical_or(np.logical_and(policy_corr[used_pairs_[0]]<lower_thr,                                                               policy_corr[used_pairs_[1]]>upper_thr),                                                              np.logical_and(policy_corr[used_pairs_[0]]>upper_thr,                                                               policy_corr[used_pairs_[1]]<lower_thr))]

    Xneuron_correlations_=Xneuron_correlations2[day_type_]['Max_bins'][mouse_recday]
    Xneuron_correlations_spatial_spatial=np.asarray([Xneuron_correlations_[used_pairs_spatial_spatial[0,ii]]                                                     [used_pairs_spatial_spatial[1,ii]] for ii in                                                range(len(used_pairs_spatial_spatial.T))])

    Xneuron_correlations_nonspatial_spatial=np.asarray([Xneuron_correlations_[used_pairs_nonspatial_spatial[0,ii]]                                                     [used_pairs_nonspatial_spatial[1,ii]] for ii in                                                range(len(used_pairs_nonspatial_spatial.T))])

    
    angle_units2=Xneuron_correlations[day_type_]['angle_units'][mouse_recday]
    Xneuron_tuning_angle_=Xneuron_correlations[day_type_]['Max_bins'][mouse_recday][:,:,0]*angle_units2

    Xneuron_tuningangle_spatial_spatial=np.asarray([Xneuron_tuning_angle_[used_pairs_spatial_spatial[0,ii]]                                                     [used_pairs_spatial_spatial[1,ii]]                                                     for ii in range(len(used_pairs_spatial_spatial.T))])

    Xneuron_tuningangle_nonspatial_spatial=np.asarray([Xneuron_tuning_angle_[used_pairs_nonspatial_spatial[0,ii]]                                                     [used_pairs_nonspatial_spatial[1,ii]]                                                     for ii in range(len(used_pairs_nonspatial_spatial.T))])
    
    Xneuron_correlations_used_pairs_ABCDonly_dic['spatial_spatial'][mouse_recday]=Xneuron_correlations_spatial_spatial
    Xneuron_tuningangle_used_pairs_ABCDonly_dic['spatial_spatial'][mouse_recday]=Xneuron_tuningangle_spatial_spatial
    Xneuron_correlations_used_pairs_ABCDonly_dic['nonspatial_spatial'][mouse_recday]=    Xneuron_correlations_nonspatial_spatial
    Xneuron_tuningangle_used_pairs_ABCDonly_dic['nonspatial_spatial'][mouse_recday]=    Xneuron_tuningangle_nonspatial_spatial
    
    Xneuron_correlations_spatial_spatial_ABCDonly_all.append(Xneuron_correlations_spatial_spatial)
    Xneuron_correlations_nonspatial_spatial_ABCDonly_all.append(Xneuron_correlations_nonspatial_spatial)
    Xneuron_tuningangle_spatial_spatial_ABCDonly_all.append(Xneuron_tuningangle_spatial_spatial)
    Xneuron_tuningangle_nonspatial_spatial_ABCDonly_all.append(Xneuron_tuningangle_nonspatial_spatial)
                                                 
                                                 
Xneuron_correlations_spatial_spatial_ABCDonly_all=np.vstack(Xneuron_correlations_spatial_spatial_ABCDonly_all)
Xneuron_correlations_nonspatial_spatial_ABCDonly_all=np.vstack((Xneuron_correlations_nonspatial_spatial_ABCDonly_all))
Xneuron_tuningangle_spatial_spatial_ABCDonly_all=np.hstack(Xneuron_tuningangle_spatial_spatial_ABCDonly_all)
Xneuron_tuningangle_nonspatial_spatial_ABCDonly_all=np.hstack((Xneuron_tuningangle_nonspatial_spatial_ABCDonly_all))


for name, array in {'spatial_spatial':Xneuron_correlations_spatial_spatial_ABCDonly_all,                    'nonspatial_spatial':Xneuron_correlations_nonspatial_spatial_ABCDonly_all}.items():
    angle_of_angles_all_=[]
    abstract_structure_='ABCD'
    same_ses_1=day_type_sameTask_dic['combined_ABCDonly'][1]
    same_ses_2=day_type_sameTask_dic['combined_ABCDonly2'][1]

    for ses_ind in np.arange(int(np.shape(array)[1]-1)):
        ses1=0
        ses2=int(ses_ind+1)


        if ses2!=same_ses_1 and ses2!=same_ses_2:
            angle_of_angles_X=array[:,ses1,ses2]
            angles_all_X=np.histogram(angle_of_angles_X,np.linspace(0,360,37))[0]
            angle_of_angles_all_.append(angle_of_angles_X)

            for mouse_recday in day_type_dicX['combined_ABCDonly']:
                angle_of_angles_X_day=Xneuron_correlations_used_pairs_ABCDonly_dic[name][mouse_recday][:,ses1,ses2]
                Xneuron_correlations_used_pairs_ABCDonly_dic['angle_of_angles_'+name][mouse_recday][ses_ind]=                angle_of_angles_X_day
                
    Xneuron_correlations_used_pairs_ABCDonly_dic['angle_of_angles_'+name]['All']=angle_of_angles_all_


# In[ ]:





# In[ ]:


###Plotting

color_map={'spatial_spatial':'blue','nonspatial_spatial':'green'}
ls_map={'Proximal':'solid','Distal':'dotted'}

for name,tuning_angle_ in {'spatial_spatial':Xneuron_tuningangle_spatial_spatial_ABCDonly_all,             'nonspatial_spatial':Xneuron_tuningangle_nonspatial_spatial_ABCDonly_all}.items():
    coh_=np.asarray(Xneuron_correlations_used_pairs_ABCDonly_dic['angle_of_angles_'+name]['All'])
    tuning_distance=angle_to_distance(tuning_angle_)
    
    prox_boolean=np.logical_or(tuning_angle_>270,tuning_angle_<90)
    dist_boolean=np.logical_and(tuning_angle_<270,tuning_angle_>90)
   
    ##proportion coherent per session
    
    for name_proxdist,coh_proxdist_boolean in {'Proximal':prox_boolean,'Distal':dist_boolean}.items():
        
        coh_proxdist=coh_[:,coh_proxdist_boolean]
        coh_proxdist=np.vstack(([angle_to_distance(coh_proxdist[ii]) for ii in range(len(coh_proxdist))]))
        tuning_angle_proxdist=tuning_angle_[coh_proxdist_boolean]
        Xneuron_correlations_used_pairs_ABCDonly_dic['angle_of_angles_'+name][name_proxdist]=coh_proxdist
    
        
        prop_coh_sesno_all=[]
        prop_coh_sesno_nozero_all=[]
        chance_sesno_all=[]
        coherent_matrix=coh_proxdist<coherence_thr

        for ii in range(len(coherent_matrix)):
            coherent_array_=coherent_matrix[:ii+1]
            sum_coherent=np.sum(coherent_array_,axis=0)
            sum_coherent_nozero=sum_coherent[angle_to_distance(tuning_angle_proxdist)>0]
            prop_coh_sesno=np.sum(sum_coherent>ii)/len(sum_coherent)
            prop_coh_sesno_all.append(prop_coh_sesno)

            prop_coh_sesno_nozero=np.sum(sum_coherent_nozero>ii)/len(sum_coherent_nozero)
            prop_coh_sesno_nozero_all.append(prop_coh_sesno_nozero)
        x=np.arange(len(prop_coh_sesno_all))+1
        #plt.plot(x,prop_coh_sesno_all,color='black')
        #plt.plot(np.arange(len(prop_coh_sesno_all)),chance_coh_all,color='grey',ls='dashed')
        #plt.show()
        plt.plot(x,prop_coh_sesno_nozero_all,color=color_map[name], ls=ls_map[name_proxdist])


        y=1/4**(x)
        plt.plot(x,y,color='grey',ls='dashed')
        #plt.savefig(Ephys_output_folder+'/proportion_coherent_serial_6tasks.svg')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


##proportions of pairs with place and nonplace cells
policy_corr_distrib=np.concatenate(dict_to_array(Policy_corr_dic))


upper_thr=np.nanpercentile(policy_corr_distrib,50)
lower_thr=np.nanpercentile(policy_corr_distrib,50)
for mouse_recday in day_type_dicX['combined_ABCDonly']:

    used_pairs=used_pairs_dic[mouse_recday]
    #between_task_spatial_sim=dict_to_array(Spatial_correlation_dic['within_betweenTask'][mouse_recday])[:,1]
    
    policy_corr=Policy_corr_dic[mouse_recday]
    place_cell_boolean=policy_corr>thr_prop_corr
    between_task_corr=policy_corr
        
    angle_of_angles_day=dict_to_array(Xneuron_correlations_used_pairs_ABCDonly_dic['angle_of_angles'][mouse_recday])

    between_task_corr_usedpairs=np.vstack((between_task_corr[used_pairs[0]],between_task_corr[used_pairs[1]]))

    spatial_sim_diff=abs(between_task_corr_usedpairs[0]-between_task_corr_usedpairs[1])
    spatial_sim_mean=np.mean(between_task_corr_usedpairs,axis=0)
    
    incoherence_day=np.vstack(([angle_to_distance(angle_of_angles_day[ii])    for ii in range(len(angle_of_angles_day))]))
    proportion_coherent_day=np.sum(incoherence_day<coherence_thr,axis=0)
    
    id_place=between_task_corr_usedpairs>upper_thr
    id_nonplace=between_task_corr_usedpairs<lower_thr
    
    num_place=np.sum(id_place,axis=0)
    num_nonplace=np.sum(id_nonplace,axis=0)
    place_nonplace_boolean=np.logical_and(num_place==1,num_nonplace==1)
    place_place_boolean=num_place==2
    nonplace_nonplace_boolean=num_nonplace==2
    #num_place_coherent=num_place[proportion_coherent_day==5]
    
    for name, array in {'proportion_coherent':proportion_coherent_day, 'number_placecells':num_place,     'place_nonplace_boolean':place_nonplace_boolean, 'place_place_boolean':place_place_boolean,     'nonplace_nonplace_boolean':nonplace_nonplace_boolean, 'spatial_sim_diff':spatial_sim_diff,     'spatial_sim_mean':spatial_sim_mean, 'placecells':id_place, 'nonplacecells':id_nonplace}.items():
        Xneuron_correlations_used_pairs_ABCDonly_dic[name][mouse_recday]=array


# In[ ]:


prop_place_cohlevels_all=[]
spatial_sim_diff_cohlevels_dic=rec_dd()
spatial_sim_mean_cohlevels_dic=rec_dd()
prop_place_cohlevels_dic=rec_dd()
for mouse_recday in day_type_dicX['combined_ABCDonly']:
    proportion_coherent_=Xneuron_correlations_used_pairs_ABCDonly_dic['proportion_coherent'][mouse_recday]
    num_place_nonplace_=Xneuron_correlations_used_pairs_ABCDonly_dic['place_nonplace_boolean'][mouse_recday]
    spatial_sim_diff_=Xneuron_correlations_used_pairs_ABCDonly_dic['spatial_sim_diff'][mouse_recday]
    spatial_sim_mean_=Xneuron_correlations_used_pairs_ABCDonly_dic['spatial_sim_mean'][mouse_recday]
    prop_coh_uniq=np.unique(proportion_coherent_)
    
    prop_place_cohlevels_day_=[len(np.where(num_place_nonplace_[proportion_coherent_==prop_coh_uniq[ii]]==1)[0])/     len(num_place_nonplace_[proportion_coherent_==prop_coh_uniq[ii]]) for ii in prop_coh_uniq]
    
    
    prop_place_cohlevels_all.append(prop_place_cohlevels_day_)
    
    for ii in prop_coh_uniq:
        spatial_sim_diff_cohlevels_dic[ii][mouse_recday]=        spatial_sim_diff_[np.logical_and((proportion_coherent_==ii),num_place_nonplace_)]
        
        spatial_sim_mean_cohlevels_dic[ii][mouse_recday]=        spatial_sim_mean_[np.logical_and((proportion_coherent_==ii),num_place_nonplace_)]


prop_place_cohlevels_all=np.vstack((prop_place_cohlevels_all))


# In[ ]:


##plot of proportion of pairs that have atleast one spatial cell
bar_plotX(prop_place_cohlevels_all.T,'none',0,0.7,'points','paired',0.025)
plt.savefig(Ephys_output_folder+'/Proportion_spatialnonspatial_vscoherence.svg')
prop_place_cohlevels_all


# In[ ]:


print('Spatial similarity difference')
spatial_sim_diff_cohlevels_=np.asarray([np.hstack(dict_to_array(spatial_sim_diff_cohlevels_dic[ii])) for ii in range(len(list(spatial_sim_diff_cohlevels_dic.keys())))])
np.shape(spatial_sim_diff_cohlevels_)


bar_plotX(spatial_sim_diff_cohlevels_,'none',0,0.5,'nopoints','unpaired',0.025)
plt.savefig(Ephys_output_folder+'/Spatialsim_difference_vscoherence.svg')

print('Spatial similarity mean')
spatial_sim_mean_cohlevels_=np.asarray([np.hstack(dict_to_array(spatial_sim_mean_cohlevels_dic[ii])) for ii in range(len(list(spatial_sim_mean_cohlevels_dic.keys())))])
np.shape(spatial_sim_mean_cohlevels_)


bar_plotX(spatial_sim_mean_cohlevels_,'none',0,0.6,'nopoints','unpaired',0.025)
plt.savefig(Ephys_output_folder+'/Spatialsim_mean_vscoherence.svg')


# In[ ]:


###prediction error in state rotations based on spatial maps

used_sessions=np.asarray([0,1,2,4,5,6]) ##other two are repititions
mouse_recday='me11_01122021_02122021'

thr_spatialsim_upper=0.5
thr_spatialsim_lower=0.5


##Incoherence
angle_of_angles_day=dict_to_array(Xneuron_correlations_used_pairs_ABCDonly_dic['angle_of_angles'][mouse_recday])
incoherence_day=np.vstack(([angle_to_distance(angle_of_angles_day[ii])for ii in range(len(angle_of_angles_day))]))


##actual rotations
Max_bins=Xsession_correlations['combined_ABCDonly']['Max_bins'][mouse_recday]
angle_units=Xsession_correlations['combined_ABCDonly']['angle_units'][mouse_recday]
actual_angle_change=np.asarray(Max_bins*angle_units)[:,(used_sessions-1).astype(int)][:,1:]

##predicted rotations
#predicted_state_=Predicted_state_dic[mouse_recday].T[used_sessions]
predicted_state_=Xsession_correlations_predicted['Max_bins'][mouse_recday][:,0].T[used_sessions]
predicted_state_change=(predicted_state_[1:]-predicted_state_[0]).T
predicted_angle_change=(predicted_state_change*90)%360
angle_error=(predicted_angle_change-actual_angle_change)%360

### neuron booleans
used_pairs=used_pairs_dic[mouse_recday]
num_place_nonplace_=Xneuron_correlations_used_pairs_ABCDonly_dic['place_nonplace_boolean'][mouse_recday]
#between_task_corr=dict_to_array(Spatial_correlation_dic['within_betweenTask'][mouse_recday])[:,1]
between_task_corr=Policy_corr_dic[mouse_recday]
place_cells=np.where(between_task_corr>thr_spatialsim_upper)[0]
non_place_cells=np.where(between_task_corr<thr_spatialsim_lower)[0]
proportion_coherent_=Xneuron_correlations_used_pairs_ABCDonly_dic['proportion_coherent'][mouse_recday]
non_zero_distance=Xneuron_tuningangle_used_pairs_ABCDonly_all

angle_change_used=actual_angle_change[used_pairs]
predicted_angle_change_used=predicted_angle_change[used_pairs]

angle_error_used=angle_error[used_pairs]
angle_error_used_pair12=(predicted_angle_change_used[0]-angle_change_used[1])%360
angle_error_used_pair21=(predicted_angle_change_used[1]-angle_change_used[0])%360
angle_error_used_Xpair=np.asarray([angle_error_used_pair12,angle_error_used_pair21])

angle_diff=np.deg2rad((angle_change_used[0]-angle_change_used[1])%360)
coherence_mat_=angle_diff<coherence_thr ## i.e. which pairs are coherent

##identifying place cells based on predictability of state from place
mean_error=st.circmean(np.deg2rad(angle_error),axis=1)
num_lowerror_predictions=np.sum(np.deg2rad(angle_error)<coherence_thr,axis=1)
spatially_predictable_cells=np.where(num_lowerror_predictions>2)[0]


##place cell boolean used
place_boolean_used=place_cells #spatially_predictable_cells
nonplace_boolean_used=non_place_cells #np.logical_not(spatially_predictable_cells)#

###pairs with one place cell
place_nonplace_boolean1=np.asarray([np.logical_and((used_pairs[0][ii] in place_boolean_used),                          (used_pairs[1][ii] in nonplace_boolean_used)) for ii in range(len(used_pairs[0]))])
place_nonplace_boolean2=np.asarray([np.logical_and((used_pairs[1][ii] in place_boolean_used),                          (used_pairs[0][ii] in nonplace_boolean_used)) for ii in range(len(used_pairs[0]))])
place_nonplace_boolean12=np.logical_or(place_nonplace_boolean1,place_nonplace_boolean2)

booleanx=np.zeros(len(place_nonplace_boolean1))
booleanx[place_nonplace_boolean1]=1
booleanx[place_nonplace_boolean2]=2
pairid_spatial_boolean=booleanx[booleanx>0]-1 ## boolean to identify which of the pair is the spatial one 
##amongst spatial non-spatial pairs
pairid_nonspatial_boolean=(~(pairid_spatial_boolean).astype(bool)).astype(int) ##as above but for non spatial cell

place_nonplace_coherent_boolean=np.logical_and(place_nonplace_boolean12,proportion_coherent_>=4)

#place_nonplace_coherent_boolean=proportion_coherent_>=4

angle_error_placenonplace=angle_error_used[:,place_nonplace_boolean12]
angle_error_used_Xpair_place=angle_error_used_Xpair[:,place_nonplace_boolean12]
proportion_coherent_placenonplace=proportion_coherent_[place_nonplace_boolean12]
angle_error_placenonplace_cohprop=angle_error_used[:,place_nonplace_coherent_boolean]
angle_change_placenonplace_cohprop=angle_change_used[:,place_nonplace_coherent_boolean]


coherence_mat_placenonplace=coherence_mat_[place_nonplace_boolean12]


# In[ ]:


###given coherence in n/4 session transitions, how accurate is the prediction of the non-spatial cell's rotation
###from its own place cells in the 5th transition

Prediction_pairs_dic=rec_dd()

ses_transitions=np.arange(len(coherent_matrix))
#non_zero_boolean=angle_to_distance(Xneuron_tuningangle_used_pairs_ABCDonly_all)>0 #>1

boolean_used_=pairid_nonspatial_boolean

for cell_used, boolean_used_ in {'Spatial_cell':pairid_spatial_boolean,                                 'Nonspatial_cell':pairid_nonspatial_boolean}.items():
    for error_status, error_array in {'same_cell':angle_error_placenonplace,                                      'cross_pair':angle_error_used_Xpair_place}.items():
    
        correctprediction_placenonplace=np.zeros((len(coherent_num_all),len(coherent_num_all)))
        for ses_trans in ses_transitions:
            error_=error_array[:,:,ses_trans]
            error=np.asarray([error_[int(boolean_used_[ii])][ii] for ii in range(len(boolean_used_))])
            target=np.deg2rad(error)<coherence_thr
            training=coherence_mat_placenonplace[:,np.setdiff1d(ses_transitions,ses_trans)]#[:,non_zero_boolean]


            for num_coherent in range(len(coherent_num_all)):
                coherent_boolean=np.sum(training,axis=1)==num_coherent
                correctprediction_placenonplace[num_coherent][ses_trans]=                np.sum(target[coherent_boolean])/len(target[coherent_boolean]) 
                ##of the coherent pairs what proportion give you an accurate prediction of the spatial cell's rotation 
                ##(from its own spatial tuning)

        Prediction_pairs_dic[error_status][cell_used]=correctprediction_placenonplace


# In[ ]:


Spatial_cell_same_cell=np.nanmean(Prediction_pairs_dic['same_cell']['Spatial_cell'],axis=1)
Nonspatial_cell_same_cell=np.nanmean(Prediction_pairs_dic['same_cell']['Nonspatial_cell'],axis=1)
Spatial_cell_cross_pair=np.nanmean(Prediction_pairs_dic['cross_pair']['Spatial_cell'],axis=1)
Nonspatial_cell_cross_pair=np.nanmean(Prediction_pairs_dic['cross_pair']['Nonspatial_cell'],axis=1)

for cell_used, boolean_used_ in {'Spatial_cell':pairid_spatial_boolean,                                 'Nonspatial_cell':pairid_nonspatial_boolean}.items():
    for error_status, error_array in {'same_cell':angle_error_placenonplace,                                      'cross_pair':angle_error_used_Xpair_place}.items():
        prediction_pairs_mean_=np.nanmean(Prediction_pairs_dic[error_status][cell_used],axis=1)
        plt.plot(prediction_pairs_mean_)
plt.savefig(Ephys_output_folder+'/Prediction_space2state_vscoherence.svg')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#### --- EXAMPLES - POST ANALYSIS --- ###


# In[ ]:





# In[ ]:


##Examples - post analysis
mouse_recday='me03_14122020'
day_type='2_task'

all_neurons=np.arange(len(cluster_dic['good_clus'][mouse_recday]))

abstract_structures=recday_numbers_dic['structure_abstract'][mouse_recday]
prop_corr_all=Xsession_correlations_predicted['proportion_correct'][mouse_recday]
policy_corr=Policy_corr_dic[mouse_recday]

place_cells=np.where(policy_corr>0.5)[0]

all_neurons_rotationbin=Xsession_correlations[day_type]['Max_bins'][mouse_recday]
angle_units=dict_to_array(Xsession_correlations[day_type]['angle_units'])[0]
rotations=all_neurons_rotationbin*angle_units

plot_edge=True

Num_trials_completed_=dict_to_array(Num_trials_dic2[mouse_recday])
X_cells=[54,56,65]
#X_cells=place_cells
print(X_cells)
for neuron in X_cells:
    print(neuron)
    
    mouse=mouse_recday.split('_',1)[0]
    rec_day=mouse_recday.split('_',1)[1]

    All_sessions=session_dic_behaviour['All'][mouse_recday]    
    awake_sessions=session_dic_behaviour['awake'][mouse_recday][Num_trials_completed_>0]
    rec_day_structure_numbers=recday_numbers_dic['structure_numbers'][mouse_recday]
    rec_day_session_numbers=recday_numbers_dic['session_numbers'][mouse_recday]
    structure_nums=np.unique(rec_day_structure_numbers)
    
    standardized_FR_smoothed_all=smoothed_activity_dic['Mean'][mouse_recday][neuron]
    standardized_FR_sem_all=smoothed_activity_dic['SEM'][mouse_recday][neuron]
        

    fignamex=Ephys_output_folder+'/Example_cells/'+mouse_recday+'_neuron_id'+str(neuron)+'_task'
    
    arrange_plot_statecells_persessionX(mouse_recday,neuron,awake_sessions,standardized_FR_smoothed_all,                                        standardized_FR_sem_all, abstract_structures=abstract_structures,                                        plot=True, fignamex=fignamex, figtype='.svg',Marker=False,)
    
    fignamex=Ephys_output_folder+'/Example_cells/'+mouse_recday+'_neuron_id'+str(neuron)+'_session_discmap'
    
    plot_spatial_mapsX(mouse_recday,neuron, per_state=True,save_fig=True,                       fignamex=fignamex,figtype='.svg')
    
    
    plt.axis('off')
    #plt.savefig(Ephys_output_folder+'Example_cells/'+\
    #            mouse_recday+'_'+str(neuron)+'session_discmap.svg',bbox_inches = 'tight', pad_inches = 0)   
    plt.show()
    for awake_session_ind, timestamp in enumerate(awake_sessions):
        structure=structure_dic[mouse]['ABCD'][rec_day_structure_numbers[awake_session_ind]]        [rec_day_session_numbers[awake_session_ind]]
        print(structure)
        
        
    correct_pred_prop=Xsession_correlations_predicted['3_task']['proportion_correct'][mouse_recday][neuron]    
    print(correct_pred_prop)
    print('rotations:')
    print(rotations[neuron])


# In[ ]:


mouse_recday='me03_14122020'
neuron=65
plot_spatial_mapsX(mouse_recday,neuron, per_state=True)


# In[ ]:


node_rate_matrices_dic['Per_state'][0]['me03_14122020'][65][3]


# In[ ]:


name='state_occupancy_dic_occupancy_raw_'+mouse_recday
data_filename_memmap = os.path.join(Intermediate_object_folder, name)
data = load(data_filename_memmap)#, mmap_mode='r')
exec(name+'= data')


# In[ ]:


occ=np.concatenate(data[0][3])
len(np.where(occ==8)[0])/60


# In[ ]:


standardized_spike_events_dic[0]


# In[ ]:


##Saving output - 1) Checking size
name='Spatial_anchoring_dic'
data=Spatial_anchoring_dic
print(str(sys.getsizeof(data)))


# In[ ]:


##Saving output - 2) Saving
data_filename_memmap = os.path.join(Intermediate_object_folder, name)
dump(data, data_filename_memmap)


# In[ ]:


day_type_dicX


# In[ ]:


for mouse_recday in ['ah04_06122021','ah04_10122021', 'me11_08122021','me10_08122021']:
    print(mouse_recday)
    awake_sessions=session_dic_behaviour['awake'][mouse_recday]
    for ses_ind in np.arange(len(awake_sessions)):
        print(ses_ind)
        name='standardized_spike_events_dic_'+mouse_recday+'_'+str(ses_ind)
        data=standardized_spike_events_dic[ses_ind][mouse_recday]
        data_filename_memmap = os.path.join(Intermediate_object_folder, name)
        dump(data, data_filename_memmap)


# In[ ]:


GLM_anchoring_regularised_high_dic


# In[ ]:


'''
FR_shuff_dic
FR_shuff_states_dic

'''


# In[14]:


#SAVING FILES 

try: 
    os.mkdir(Intermediate_object_folder) 
except FileExistsError: 
    pass

objects_dic={'Anchor_trial_dic':Anchor_trial_dic,'Spatial_anchoring_dic':Spatial_anchoring_dic,            'Task_num_dic':Task_num_dic,'Num_trials_dic2':Num_trials_dic2,'Xsession_correlations':Xsession_correlations,            'Xneuron_correlations':Xneuron_correlations,'module_dic':module_dic,'module_shuff_dic':module_shuff_dic,             'Phase_spatial_corr_dic':Phase_spatial_corr_dic,            'GLM_anchoring_prep_dic':GLM_anchoring_prep_dic,'GLM_anchoring_regularised_high_dic':GLM_anchoring_dic,            'Tuned_dic':Tuned_dic,'state_middle_dic':state_middle_dic,'Regression_anchors_dic':Regression_anchors_dic}

for name, dicX in objects_dic.items(): 
    data=dicX 
    data_filename_memmap = os.path.join(Intermediate_object_folder, name) 
    dump(data, data_filename_memmap)


# In[ ]:


Xneuron_correlations, Xsession_correlations


# In[ ]:


#Spatial_anchoring_dic


# In[ ]:





# In[9]:


Anchor_trial_dic['Best_shift_time'].keys()#


# In[ ]:





# In[ ]:




