#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as st
import scipy as scp
import matplotlib.pyplot as plt
import matplotlib as ply
import os
import collections, numpy
#from scipy.interpolate import spline
from collections import defaultdict

import pickle
from datetime import datetime, date
from collections import namedtuple
import sys
from itertools import groupby
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

import os
from joblib import dump, load
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
from scipy import stats
import pandas as pd
import time

import pandas as pd
import seaborn as sns

from itertools import product
import csv

import pingouin as pg
from statsmodels.stats.anova import AnovaRM

import sklearn


# In[2]:


print(np.__version__)
print(pd.__version__)
print(scp.__version__)
print(sklearn.__version__)
print(ply.__version__)
print(sns.__version__)


# In[3]:


###Defining directories

Data_folder='/Taskspace_abstraction/Data/'
Data_folder_P='P:/Taskspace_abstraction/Data/' ## if working in P
base_dropbox='C:/Users/moham/team_mouse Dropbox/Mohamady El-Gaby/'

#base_dropbox='D:/team_mouse Dropbox/Mohamady El-Gaby/'

Data_folder_dropbox=base_dropbox+'/Taskspace_abstraction/Data/' ##if working in C
Behaviour_output_folder = 'P:/Taskspace_abstraction/Results/Behaviour/'
Ephys_output_folder = 'P:/Taskspace_abstraction/Results/Ephys/'
Ephys_output_folder_dropbox = base_dropbox+'/Taskspace_abstraction/Results/Ephys/'
Behaviour_output_folder_dropbox = base_dropbox+'/Taskspace_abstraction/Results/Behaviour/'
Intermediate_object_folder_dropbox = Data_folder_dropbox+'/Intermediate_objects/'

Intermediate_object_folder = Data_folder_dropbox+'/Intermediate_objects/'

Code_folder='/Taskspace_abstraction/Code/'

base_ceph='Z:/mohamady_el-gaby/'
Data_folder_ceph='Z:/mohamady_el-gaby/Taskspace_abstraction_2/Data/'
Data_folder_ceph1='Z:/mohamady_el-gaby/Taskspace_abstraction/Data/'
Data_folder_ceph2='Z:/mohamady_el-gaby/Taskspace_abstraction_2/Data/'

Intermediate_object_folder_ceph = Data_folder_ceph1+'/Intermediate_objects/'


# In[ ]:





# In[4]:


def rec_dd():
    return defaultdict(rec_dd)
def indicesX2(z,thr):
    indicesx=[]
    for xx in range(0,len(z)):
        diffx=(z[xx,1]-z[xx,0]+1)
        if diffx>thr:
            indicesx.append(np.linspace                            (z[xx,0],z[xx,1],diffx))
    indices=np.asarray(indicesx)
    return(indices)

def create_binary(a):
    ax=np.copy(a)
    a_values=np.unique(ax)

    try:
        len(a_values)==2
    except ValueError:
        print ("Not a valid Sync file")
    else:
        ax[a==np.max(a)]=1
        ax[a==np.min(a)]=0
    return(ax)
def unique_adjacent(a):
    return(np.asarray([k for k,g in groupby(a)]))

def flatten(d):    
    res = []  # Result list
    for key, val in d.items():
        res.extend(dict_to_array(val))
    return (res)

def remove_empty(xx):

    yy= [x for x in xx if len(x) > 0]
    return(yy)

def num_routes(start,end):
    num_steps=int(distance_mat[start][end])
    node=[start]
    nodes=[]
    for num_step in range(num_steps):
        next_nodes=[]
        for ii in node:
            next_nodes.append(np.where(distance_mat[ii]==1)[0])
        node=np.concatenate(next_nodes)
        nodes.append(node)
    prob_chance=len(np.where(node==end)[0])/len(node)
    return(prob_chance,len(node),nodes)

def mean_complex2(x):
    means=[]
    for i in x:
        meanx=np.nanmean(i)
        means.append(meanx)
    means=np.asarray(means)
    return(means)
def noplot_timecourseBx(x,y,color):
    ymean=mean_complex2(y)
    yerr=[st.sem(i,nan_policy='omit') for i in y]
    plt.errorbar(x,y=ymean, color=color, marker='o')
    plt.fill_between(x, ymean-yerr, ymean+yerr, color=color,alpha=0.5)

##convert nested dict into array
def dict_to_array(d):
    dictlist=[]
    for key, value in d.items():
        dictlist.append(value)
    return(np.asarray(dictlist))

def flatten(d):    
    res = []  # Result list
    for key, val in d.items():
        res.extend(dict_to_array(val))
    return (np.asarray(res))
def rand_jitter(arr):
    stdev = .05*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def rand_jitterX(arr, X):
    stdev = X*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev
def bar_plotX(y,name,ymin,ymax,points,pairing,jitt):
    leny=len(y)
    plt.figure(figsize=(leny*(3/2),6))
    
    if ymin =='auto':
        ymin=np.min(np.concatenate(y))
    if ymax =='auto':
        ymax=np.max(np.concatenate(y))
    
    ##bars
    y_mean=((np.zeros(len(y))))
    y_sem=((np.zeros(len(y))))
    for ii in range(0, len(y)):
        ymeanx=np.nanmean(y[ii])
        y_mean[ii]=ymeanx
        ysemx=st.sem(y[ii], nan_policy='omit')
        y_sem[ii]=ysemx
   
    
    xxx=np.linspace(0.15, 0.2+(0.2*(leny-1)), leny)

    xlocations = np.array(range(len(xxx)))
    width=0.2
    plt.bar(xxx, y_mean, width, yerr=y_sem, alpha=1, 
           error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), align='center')
    
    if points != 'points' and ymin == 'auto':
        ymin=np.min(y_mean-y_sem) #-np.max(y_sem)
        ymax=np.max(y_mean+y_sem) #+np.max(y_sem)
    
    #if ymin>0:
    #    ymin=0
    plt.ylim(ymin-(0.05*(ymax-ymin)),ymax+(0.05*(ymax-ymin)))
    plt.xlim(0,np.max(xxx)+0.15)
    
    

    ###points and lines
    if points == 'points':
        yyALL=[]
        for ii in range(0, len(y)):
            yy=np.column_stack((y[ii],np.repeat(xxx[ii],len(y[ii]))))
            yyALL.append(yy)

        xy=np.vstack((yyALL))
        jittered=rand_jitterX(xy[:,1],jitt)

        if pairing == 'paired':
            for ii in range(0, leny):
                x1=np.split(jittered,len(y))[ii]
                if ii == 0:
                    x1_all=x1
                else:
                    x1_all=np.column_stack((x1_all,x1))

            for jj in range(0,np.shape(y)[1]):
                yyyy=np.asarray(y)[:,jj]
                plt.plot(x1_all[jj],yyyy, color='gray')
        plt.plot(jittered,xy[:,0],'o',markersize=7,color='white',markeredgecolor='black')
    
    if name != 'none':
        plt.savefig(name)
    
    #plt.show()
    
def remove_nan(x):
    x=x[~np.isnan(x)]
    return(x)

def remove_nanX(x):
    xx=[]
    for ii in x:
        if len(np.shape(ii)) != 0:
            xx.append(ii)
        else:
            if np.isnan(ii) == False:
                xx.append(ii)
    xx=np.asarray(xx)
    return(xx)

def rearrange_for_ANOVA(xxx):
    x=np.repeat(np.arange(len(xxx)),np.shape(xxx)[1])
    y=(np.tile(np.arange(np.shape(xxx)[1]),len(xxx))).astype(int)
    z=np.concatenate(xxx)
    
    return(np.column_stack(((x).astype(int),(y).astype(int),z)))



def rearrange_for_ANOVAX(dicX):
    all_xxx=[]
    for label,array in dicX.items():
        xxx=array
        indx_all=[]
        for ii in np.arange(len(xxx)):
            indx=np.repeat(ii,len(xxx[ii]))
            indx_all.append(indx)

        all_xxx.extend(np.column_stack((np.repeat(label,len(np.concatenate(xxx))),                                        np.concatenate(indx_all),np.concatenate(xxx))))
    return(np.asarray(all_xxx))

def column_stack_clean(x,y):
    xy=np.column_stack((x,y))
    xy=xy[~np.isnan(xy).any(axis=1)]
    xy=xy[~np.isinf(xy).any(axis=1)]
    x=xy[:,0]
    y=xy[:,1]
    xy_new=np.column_stack((x,y))
    return(xy_new)

###function to plot scatter plots (e.g. comparing assembly strength at correct vs incorrect dispensers)
def plot_scatter(x,y,name='none'):
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
    
def number_of_repeats_ALL(array):
    unique_rows=np.unique(array,axis=0)
    return(np.asarray([sum((array == unique_rows[ii]).all(1)) for ii in range(len(unique_rows))]))

def bin_arrayX(array,factor):
    bins=np.arange(len(array.T)//factor+1)*factor
    array_binned=np.vstack(([st.binned_statistic(np.arange(len(array.T)),array[ii],                                bins=bins,statistic=np.nanmean)[0] for ii in np.arange(len(array))]))
    return(array_binned)

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


# In[5]:


##Importing custom functions
module_path = os.path.abspath(os.path.join(Code_folder))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from mBaseFunctions import rec_dd, remove_empty, rank_repeat, dict_to_array, concatenate_complex2, smooth_circular,polar_plot_stateX2, indep_roll, bar_plotX, plot_scatter, non_repeat_ses_maker, two_proportions_test, noplot_timecourseA


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


##Importing meta Data
Mice_cohort_dic={'me03':2,'me04':2,'me05':2,'me06':2,'me08':3,'ah02':3,'ah03':3,'me10':4,'me11':4,'ah04':4,'ah05':4,                'ab03':6,'ah07':6} 
Mice_recnode_dic={'me03':110,'me08':131,'ah02':129,'ah03':110}
Mouse_FPGAno={'me03':'109.0','me08':'121.0','ah02':'109.0','ah03':'109.0'}
Mice=np.asarray(list(Mice_cohort_dic.keys()))
Nonephys_mice=['me04','me05','me06','ah02','ah05']
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


##To do
##remake metaData files
##loop over structure numbers and create dictionary that divides times up by structure
##use both structure number and structure_abstract for this as first structure of AB also labelled 1
##calcukate number of trials to criteria


# In[ ]:





# In[ ]:





# In[9]:


#LOADING FILES
tt=time.time()

try:
    os.mkdir(Intermediate_object_folder)
except FileExistsError:
    pass


objects_list=['trialtimes_state_dic', 'scores_dic', 'nodes_cut_dic','structure_dic','times_dic',              'trialtimes_stateAB_dic', 'scoresAB_dic', 'nodes_cutAB_dic','ROI_accuracy_dic',             'Nodes_trials_dic','Session_structure_dic','Num_trials_dic','day_type_dicX']
             #'trialtimes_stateABCAD_dic', 'scoresABCAD_dic','nodes_cutABCAD_dic']
              
for name in objects_list:
    data_filename_memmap = os.path.join(Intermediate_object_folder, name)
    data = load(data_filename_memmap)#, mmap_mode='r')
    exec(name+'= data')

print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[173]:


#Importing Behavioural Data

times_dic=rec_dd()
structure_dic=rec_dd()
for mouse in Mice:
    #mouse='me10'
    cohort=Mice_cohort_dic[mouse]
    data_directory='P:/Taskspace_abstraction/Data/cohort'+str(cohort)+'/'+mouse[:4]+'/'
    
    Variables=Variable_dic[mouse]
    
    
    
    num_sessions=len(Variables['Date'])

    for session in range(num_sessions):
        try:
            Behaviourfile_path=data_directory+'/Behaviour/'+str(mouse)+'-'+str(Variables['Date'][session])+'-'+            str(Variables['Behaviour'][session])+'.txt'
            #file_path='me04-2020-11-12-145650.txt'
            with open(Behaviourfile_path, 'r') as f:
                print('Importing data file: '+os.path.split(Behaviourfile_path)[1])
                all_lines = [line.strip() for line in f.readlines() if line.strip()]
        

            int_subject_IDs=True
            # Extract and store session information.

            file_name = os.path.split(Behaviourfile_path)[1]

            Event = namedtuple('Event', ['time','name'])


            info_lines = [line[2:] for line in all_lines if line[0]=='I']

            experiment_name = next(line for line in info_lines if 'Experiment name' in line).split(' : ')[1]
            task_name       = next(line for line in info_lines if 'Task name'       in line).split(' : ')[1]
            subject_ID_string    = next(line for line in info_lines if 'Subject ID'      in line).split(' : ')[1]
            datetime_string      = next(line for line in info_lines if 'Start date'      in line).split(' : ')[1]

            #ephys_path  = '/media/behrenslab/My Book/Ephys_Reversal_Learning/neurons'+'/'+ subject_ID_string
            #ephys_path  = '/C/Data/Abstraction1/'+ subject_ID_string
            if int_subject_IDs: # Convert subject ID string to integer.
                subject_ID = int(''.join([i for i in subject_ID_string if i.isdigit()]))
            else:
                subject_ID = subject_ID_string

            datetime = datetime.strptime(datetime_string, '%Y/%m/%d %H:%M:%S')
            datetime_string = datetime.strftime('%Y-%m-%d %H:%M:%S')

            # Extract and store session data.

            state_IDs = eval(next(line for line in all_lines if line[0]=='S')[2:])
            event_IDs = eval(next(line for line in all_lines if line[0]=='E')[2:])
            
            variable_lines = [line[2:] for line in all_lines if line[0]=='V']
            
            Structure_abstract=Variables['Structure_abstract'][session]
            
            if Structure_abstract not in ['ABCD','AB','ABCDA2']:
                continue
            
            structurexx = next(line for line in variable_lines if 'active_poke' in line).split(' active_poke ')[1]
            if 'ot' in structurexx:
                structurex=structurexx[:8]+']'
            else:
                structurex=structurexx
            
            Structure_no=int(Variables['Structure_no'][session])
            
            
            #if len(times_dic[mouse][Structure_abstract][Structure_no][session])==0:
            if Structure_abstract in ['ABCD','AB']:
                structure=np.asarray((structurex[1:-1]).split(',')).astype(int)
            else:
                structure=structurex

            #data_lines = [line[2:] for line in all_lines if line[0]=='D']

            ID2name = {v: k for k, v in {**state_IDs, **event_IDs}.items()}

            data_lines = [line[2:].split(' ') for line in all_lines if line[0]=='D']

            events = [Event(int(dl[0]), ID2name[int(dl[1])]) for dl in data_lines]

            times = {event_name: np.array([ev.time for ev in events if ev.name == event_name])  
                          for event_name in ID2name.values()}

            times_dic[mouse][Structure_abstract][Structure_no][session]=times
            structure_dic[mouse][Structure_abstract][Structure_no][session]=structure

            #print_lines = [line[2:].split(' ',1) for line in all_lines if line[0]=='P'] 
        except FileNotFoundError:
            print("File: "+Behaviourfile_path+" Not Found")
            


# In[ ]:





# In[ ]:





# In[11]:


times_dic['me10']['ABCD'][35].keys()#[201]['rsync']


# In[ ]:


Variables['Structure_no'][172]


# In[ ]:





# In[ ]:





# In[12]:


##Number of trials (+sessions per structure dic)
Num_trials_dic=rec_dd()
Session_structure_dic=rec_dd()
for mouse in Mice:
    Variables=Variable_dic[mouse]
    #Structure_no=(Variables['Structure_no']).astype(int)
    structure_no_=np.copy(Variables['Structure_no'])
    structure_no_[structure_no_=='-']=-1
    Structure_no=(structure_no_).astype(int)
    Structure_abstract=Variables['Structure_abstract']
    ABCD_structure_no=Structure_no[Structure_abstract=='ABCD']
    
    for structure_num in np.unique(ABCD_structure_no):
        dicX=times_dic[mouse]['ABCD'][structure_num]
        structure_sessions=dict_to_array(dicX)
        num_trials_all=np.zeros(len(structure_sessions))
        
        for indx, session_struc in enumerate(structure_sessions):
            if len(session_struc['A_on']) > 0:
                if 'A_on_first' in session_struc.keys():
                    num_trials_indx=int(len(session_struc['A_on'][1:])+len(session_struc['A_on_first']))
                else:
                    num_trials_indx=int(len(session_struc['A_on'][1:]))

                num_trials_all[indx]=num_trials_indx
        Num_trials_dic[mouse][structure_num]=num_trials_all
        Session_structure_dic[mouse][structure_num]=np.asarray([*dicX])#.keys()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


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



corners=[0,2,6,8]
cardinals=[1,3,5,7]
middle=[4]

opposites={'0':8,'2':6,'1':7,'3':5,'8':0,'6':2,'7':1,'5':3}


##Number of shortest distances - refine analytically!
numshortest_path_mat=np.zeros((9,9))
for node_start in np.arange(9):
    for node_end in np.arange(9):
        node_start_end=node_start,node_end
        if node_start==node_end:
            num_shortest_paths=np.nan
        elif node_start in corners and node_end in corners: 
            if opposites[str(node_start)]==node_end:
                num_shortest_paths=6
            else:
                num_shortest_paths=1
        elif (node_start in corners and node_end in cardinals) or (node_start in cardinals and node_end in corners):
            if distance_mat[node_start,node_end]==1:
                num_shortest_paths=1
            else:
                num_shortest_paths=3
        elif (node_start in corners and node_end in middle) or (node_start in middle and node_end in corners):
                num_shortest_paths=2
                
                
        elif node_start in cardinals and node_end in cardinals:
            if opposites[str(node_start)]==node_end:
                num_shortest_paths=1
            else:
                num_shortest_paths=2
        elif (node_start in cardinals and node_end in middle) or (node_start in middle and node_end in cardinals):
                num_shortest_paths=1
                
        numshortest_path_mat[node_start,node_end]=num_shortest_paths    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


##defining chance probabilities of all transitions 
prob_mat=[[0 for x in range(9)] for y in range(9)]
for ii in range(9):
    for jj in range(9):
        if ii==jj:
            prob_mat[ii][jj]=0
        else:
            prob_mat[ii][jj]=num_routes(ii,jj)[0]
            
prob_mat=np.asarray(prob_mat)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


###Defining dictionaries
state_ind_dic={'A':0,'B':1,'C':2,'D':3}
state_nextstate_dic={'A':'B','B':'C','C':'D','D':'A'}


A_trial_indx=['A_on','B_on']
B_trial_indx=['B_on','C_on']
C_trial_indx=['C_on','D_on']
D_trial_indx=['D_on','A_on']
All_trials_indx=np.vstack((A_trial_indx,B_trial_indx,C_trial_indx,D_trial_indx))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
-Identify exploration sessions for each mouse
-import ROIs for those sessions
-find unique adjacent
-remove bridges
-find unique adjacent again
-make transition matrix for 1,2,3 and 4 steps

'''


# In[15]:


num_steps_transition=4
num_nodes=9
#use_interp=False
#mouse='ab03'

for mouse in Mice:
    print(mouse)
    cohort=Mice_cohort_dic[mouse[:4]]
    if int(cohort) in [3,4]:
        use_interp=True
    else:
        use_interp=False
    
    Behaviour_base=Data_folder_P+'cohort'+str(cohort)+'/'+mouse[:4]+'/Behaviour/'

    exploration_boolean=Variable_dic[mouse]['Structure_abstract']=='exploration'
    
    non_exploration_boolean=~exploration_boolean
    first_nonexploration=np.where(non_exploration_boolean==True)[0][0]
    exploration_boolean[first_nonexploration:]=False
    if np.sum(exploration_boolean)==0:
        print('No Pre-task exploration sessions')
        continue
    
    
    Dates=Variable_dic[mouse]['Date']
    Tracking_timestamps=Variable_dic[mouse]['Tracking']
    exploration_dates=Dates[exploration_boolean]
    exploration_timestamps=Tracking_timestamps[exploration_boolean]

    trans_mat_all=np.zeros((len(exploration_timestamps),num_steps_transition,num_nodes,num_nodes))
    for ses_ind in np.arange(len(exploration_timestamps)):
        date_=exploration_dates[ses_ind]
        timestamp_=exploration_timestamps[ses_ind]
        ROIfile_path=Behaviour_base+'ROIs_'+str(mouse)+'_'+str(date_)+'-'+                            str(timestamp_)+'.csv' ##change this!
        
        try:
            with open(ROIfile_path, 'r') as f:
                print('Importing data file: '+os.path.split(ROIfile_path)[1])
                data = np.genfromtxt(f, delimiter=',',dtype=str)
        except:
            print(os.path.split(ROIfile_path)[1]+' not found')
            continue
        #backHead = data[1:, 4]
        #betweenShoulders = data[1:, 5]
        #Majority = data[1:, 9]

        if use_interp==True:
            body_part_name,body_part_namex='0','0' ###uses betweenShoulders

        else:
            body_part_name='majority'#'betweenShoulders' ###choice of what point on animal to track
            body_part_namex='Majority'
        body_part_name_used =         np.asarray([body_part_name,body_part_namex])[[body_part_name in data[0],                                                      body_part_namex in data[0]]][0]

        body_part=(data[1:,data[0]==body_part_name_used]).squeeze()
        
        if len(body_part)/60/60 < 10:
            print('session shorter than 10 mins')
            continue

        ###nodes
        nodes=np.hstack(([int(body_part[ii][5:]) for ii in range(len(body_part)) if 'node' in body_part[ii] ]))
        nodes_unique=unique_adjacent(nodes)

        ##Transition matrices

        for num_steps in np.arange(num_steps_transition)+1:
            df = pd.DataFrame(nodes_unique)
            df['shift'] = df[0].shift(-num_steps)
            df['count'] = 1
            trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
            trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values

            trans_mat_all[ses_ind,num_steps-1]=trans_mat
    trans_mat_mean=np.nanmean(trans_mat_all,axis=0)

    np.save(Intermediate_object_folder_dropbox+'/'+mouse+'_Exploration_Transition_matrix.npy',trans_mat_mean)


# In[59]:


len(body_part)/60/60


# In[13]:


plt.matshow(trans_mat_mean[0])


# In[ ]:





# In[ ]:





# In[36]:


data_directory


# In[ ]:





# In[13]:


##Importing files and calculating scores
tt=time.time()

#current_cohort=4

#nodes_cut_dic=rec_dd()
#trialtimes_state_dic=rec_dd()
#scores_dic=rec_dd()
re_run=True
tracking_oversampling_factor=50 ##oversampling of nodes_cut_dic
behaviour_oversampling_factor=3 ##oversampling of trialtimes_dic

state_ind_dic={'A':0,'B':1,'C':2,'D':3}
state_nextstate_dic={'A':'B','B':'C','C':'D','D':'A'}


A_trial_indx=['A_on','B_on']
B_trial_indx=['B_on','C_on']
C_trial_indx=['C_on','D_on']
D_trial_indx=['D_on','A_on']
All_trials_indx=np.vstack((A_trial_indx,B_trial_indx,C_trial_indx,D_trial_indx))
mouse_recdays_todo=[]
abstract_structure_type='ABCD'
for mouse in ['ah07']:
    print(mouse)
    cohort=Mice_cohort_dic[mouse[:4]]
    
    ephys_type=Cohort_ephys_type_dic[int(cohort)]

    if int(cohort) in [3,4]:
        use_interp=True
    else:
        use_interp=False

    data_directory='P:/Taskspace_abstraction/Data/cohort'+str(cohort)+'/'+mouse[:4]+'/'
    Behaviour_base=Data_folder_P+'cohort'+str(cohort)+'/'+mouse[:4]+'/Behaviour/'

    Variables=Variable_dic[mouse]
    #Structure_no=(Variables['Structure_no']).astype(int)
    structure_no_=np.copy(Variables['Structure_no'])
    structure_no_[structure_no_=='-']=-1
    Structure_no=(structure_no_).astype(int)
    Structure_abstract=Variables['Structure_abstract']
    Structure_abstract=Variables['Structure_abstract']
    ABCD_structure_no=Structure_no[Structure_abstract==abstract_structure_type]



    for structure_num in np.unique(ABCD_structure_no):
        print(structure_num)

        for structure_ses, (session, content) in enumerate(times_dic[mouse][abstract_structure_type]                                                           [structure_num].items()):
            
            if re_run==False:
                if len(scores_dic[mouse][structure_num]['ALL'][session])>0:
                    print('Already analysed')
                    continue
                

            session_Date=Variables['Date'][session]
            session_timestamp=Variables['Behaviour'][session]
            rec_day=session_Date[8:10]+session_Date[5:7]+session_Date[:4]
            mouse_recday=mouse+'_'+rec_day
            ses_day=Variables['Behaviour'][Variable_dic[mouse]['Date']==session_Date]
            ses_ind=np.where(ses_day==session_timestamp)[0][0]

            ROI_accuracy_all=dict_to_array(ROI_accuracy_dic['accuracy'][mouse_recday])

            #if 0==0:
                
            #if mouse_recday not in mouse_recdays_todo:
            #    continue

            ##Importing pinstate and ROI files##

            try:
                pinstatefile_path=data_directory+'/Behaviour/'+str(mouse)+'_pinstate_'+str(Variables['Date']                                                                                           [session])+'-'+                str(Variables['Tracking'][session])+'.csv'

                #file_path='me04-2020-11-12-145650.txt'
                with open(pinstatefile_path, 'r') as f:
                    print('Importing data file: '+os.path.split(pinstatefile_path)[1])
                    ttl_tracking = np.asarray([line.strip() for line in f.readlines() if line.strip()]).astype(int)
                ttl_binary=create_binary(ttl_tracking)

                #pinstate_dic[mouse][session]=ttl_binary

            except FileNotFoundError:
                print("File: "+pinstatefile_path+" Not Found")



            try:
                #ROIfile_path=data_directory+'/Behaviour/ROIs_'+str(mouse)+'_'+str(Variables['Date'][session])+'-'+\
                #str(Variables['Tracking'][session])+'.csv' ##change this!

                if use_interp==False:
                    ROIfile_path=Behaviour_base+'ROIs_'+str(mouse)+'_'+str(Variables['Date'][session])+'-'+                    str(Variables['Tracking'][session])+'.csv' ##change this!
                elif use_interp==True:
                    if len(ROI_accuracy_all)>ses_ind+1:
                        print('Tracking accuracy: '+str(ROI_accuracy_all[ses_ind])+'%')
                    else:
                        print('Tracking accuracy: Not calculated')

                    ROIfile_path=Behaviour_base+'rescaled_ROIs'+str(mouse)+'_'+rec_day+'_'+str(ses_ind)+'_interp.csv'

                #file_path='me04-2020-11-12-145650.txt'
                with open(ROIfile_path, 'r') as f:
                    print('Importing data file: '+os.path.split(ROIfile_path)[1])
                    data = np.genfromtxt(f, delimiter=',',dtype=str)
                #backHead = data[1:, 4]
                #betweenShoulders = data[1:, 5]
                #Majority = data[1:, 9]

                if use_interp==True:
                    body_part_name,body_part_namex='0','0' ###uses betweenShoulders

                else:
                    body_part_name='majority'#'betweenShoulders' ###choice of what point on animal to track
                    body_part_namex='Majority'
                body_part_name_used =                 np.asarray([body_part_name,body_part_namex])[[body_part_name in data[0],                                                              body_part_namex in data[0]]][0]

                body_part=(data[1:,data[0]==body_part_name_used]).squeeze()



            except FileNotFoundError:
                print("File: "+ROIfile_path+" Not Found")
                continue




            ###Matching ROI and behaviour timestamps###

            structure=structure_dic[mouse]['ABCD'][structure_num][session]
            times=times_dic[mouse]['ABCD'][structure_num][session]
            #body_part=ROI_dic[mouse][session]['body_part']
            #ttl_binary=pinstate_dic[mouse][session]
            
            if len(times['A_on'])==0:
                print('Behaviour not found')
                continue



            if len(body_part)>0:
                print(len(ttl_binary)/len(body_part))
                if (len(ttl_binary)/len(body_part))==2:
                    body_part=np.repeat(body_part,2)

                elif (len(ttl_binary)/len(body_part))>1.99 and (len(ttl_binary)/len(body_part))!=2:
                    body_part=np.repeat(body_part,2)[:len(ttl_binary)]

                nodes=[np.nan if body_part[ii][:4]!='node' else int(body_part[ii][-1])                       for ii in range(len(body_part))]
                edges=[np.nan if body_part[ii][:4]!='edge' else [int(body_part[ii][-3]),int(body_part[ii][-1])]                       for ii in range(len(body_part)) ]
                
                times_oversampled=rec_dd()
                for index,item in times.items():
                    times_oversampled[index]=item*behaviour_oversampling_factor

                if 'A_on_first' in times_oversampled.keys():
                    times_oversampled['A_on']=np.hstack((times_oversampled['A_on_first'],                                                         times_oversampled['A_on']))

                ##Aligning behaviour and tracking
                if use_interp==False:


                    if ephys_type!='Neuropixels':
                        diff_behTrack=len(times['rsync'])-len(np.where(unique_adjacent(ttl_binary)==1)[0])
                        if diff_behTrack<0:
                            print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)+                                  ' '+session_Date+'-'+str(session_timestamp)+                                  ': more rsync pulses in pinstate than in behaviour files - SESSION NOT USED')
                            print(diff_behTrack)
                            continue
                        num_sync_missed=times_oversampled['rsync'][diff_behTrack]
                    else:
                        diff_behTrack=0
                        num_sync_missed=0
                    
                    if len(np.where(ttl_binary==1)[0])==0:
                        print('No sync pulses detected by camera!')
                        continue
                    ttl_first=np.where(ttl_binary==1)[0][diff_behTrack]
                    nodes_cut=nodes[ttl_first:]
                    nodes_cut_oversampledx=np.repeat(nodes_cut,tracking_oversampling_factor)
                    nodes_cut_oversampled=np.hstack((np.repeat(np.nan,num_sync_missed),nodes_cut_oversampledx))               

                    ##checks
                    if diff_behTrack>0:
                        print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)+                              ' '+session_Date+'-'+str(session_timestamp)+                              ': less rsync pulses in pinstate than in behaviour files')
                        print('Realigned ROI timestamps')


                    if len(ttl_binary)-len(body_part)!=0:
                        print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)                              +' '+session_Date+'-'+str(session_timestamp)+                              ': mismatch between length of pinstate file and ROI file')
                        if len(ttl_binary)-len(body_part)==1:
                            ttl_binary=ttl_binary[:-1]
                            print('But only one off so cut end of ttl_binary')
                        else:
                            continue

                    tracking_length_mins=(len(ttl_binary))/(60*60)
                    behaviour_length_mins=times['rsync'][-1]/(1000*60)
                    metaData_length_mins=int(Variable_dic[mouse]['Session_time'][session])

                    if np.abs(tracking_length_mins-metaData_length_mins)>10 or np.abs                    (behaviour_length_mins-metaData_length_mins)>10:
                        print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)+' '+                              session_Date+'-'+str(session_timestamp)+                              ': session length does not match metaData file')
                        #continue
                        print('Error Ignored')

                    numdigits_times_oversampled=len(str(times_oversampled['rsync'][-1]))
                    numdigits_nodes_cut_oversampled=len(str(len(nodes_cut_oversampled)))

                    if (numdigits_nodes_cut_oversampled-numdigits_times_oversampled)>1:
                        print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)+' '+session_Date                              +'-'+str(session_timestamp)+                              ': Oversampling causes mismatch between tracking and behaviour')
                        continue

                else:
                    nodes_cut=nodes
                    nodes_cut_oversampledx=np.repeat(nodes_cut,50)
                    nodes_cut_oversampled=nodes_cut_oversampledx 




                print(mouse+' structure'+str(structure_num)+' session'+str(session)+' '+session_Date                          +'-'+str(session_timestamp)+                          ': Passed all checks')
                ##ADD EXTRA CHECK THAT SYNC PULSES AND TTL PULSES RECEIVED BY CAMERA ARE AN EXACT MATCH

                #times_oversampled_dic[mouse][structure_num][session]=times_oversampled
                nodes_cut_dic[mouse][structure_num][session]=nodes_cut



                ## Scoring ##
                All_scores=[]
                All_scoresx=[]
                All_entropies=np.zeros(4)
                All_entropies[:]=np.nan
                for indx, state in enumerate(states):

                    ##Trail times per state
                    X=state
                    Y=state_nextstate_dic[X]

                    current_port=structure[indx]
                    next_port=np.roll(structure,-1)[indx]
                    if len(times_oversampled[All_trials_indx[indx,0]])>0:
                        if state != 'D':
                            if len(times_oversampled[All_trials_indx[indx,0]])==                            len(times_oversampled[All_trials_indx[indx,1]]):
                                trialtimes_=                                np.column_stack((times_oversampled[All_trials_indx[indx,0]],                                                 times_oversampled[All_trials_indx[indx,1]]))
                            else:

                                trialtimes_=                                np.column_stack((times_oversampled[All_trials_indx[indx,0]][:-1],                                                 times_oversampled[All_trials_indx[indx,1]]))
                        else:
                            trialtimes_=                            np.column_stack((times_oversampled['D_on'][:len(times_oversampled['A_on'])-1],                                             times_oversampled['A_on'][1:len(times_oversampled['A_on'])]))

                        if len(trialtimes_)>1:
                            print(session)
                            trialtimes_state_dic[mouse][structure_num][session][state]=trialtimes_


                            ###scoring
                            state_timesx=indicesX2(trialtimes_,0)
                            state_times =[(state_timesx[ii]).astype(int) for ii in range(len(state_timesx))]

                            if len(state_times)>0: 
                                if state_times[-1][-1]>len(nodes_cut_oversampled):
                                    print(mouse+' structure'+str(structure_num)+' session'+str(session)+                                          ' '+session_Date+'-'+str(session_timestamp)+                                          ' Behaviour longer than tracking')
                                    state_times =[(state_timesx[ii]).astype(int)                                                  for ii in range(len(state_timesx))                                                 if state_timesx[ii][-1]<len(nodes_cut_oversampled)]

                            portX=structure[state_ind_dic[X]]
                            portY=structure[state_ind_dic[Y]]
                            ###remove first instances of start node from node_times_trial


                            if len(nodes_cut_oversampled)>0 and len(state_times)>0 and                             np.isnan(np.nanmean(nodes_cut_oversampled))==False:

                                all_trials=np.asarray([(remove_nan(nodes_cut_oversampled[state_times[ii]]))                                                       .astype(int) for ii in range(len(state_times))])
                                
                                #all_trials=remove_empty(all_trials)

                                dists=np.asarray([len(unique_adjacent(trial[trial>=0])[1:]) for trial in all_trials])

                                ###removing tracking errors

                                error_mask_first=np.asarray([unique_adjacent(trial[trial>=0])[0]!=current_port                                                             if len(trial)>0 else False for trial in all_trials])
                                error_mask_next=np.asarray([unique_adjacent(trial[trial>=0])[-1]!=next_port                                                       if len(trial)>0 else False for trial in all_trials])

                                error_mask=np.logical_or(error_mask_first,error_mask_next)

                                dists=dists.astype(float)
                                dists[error_mask]=np.nan

                                mindist=mindistance_mat[portX-1,portY-1]
                                correct_boolean=dists==mindist
                                scores=dists==mindist

                                scores=(scores.astype(int)).astype(float)
                                scores[error_mask]=np.nan

                                exec(X+'_scores=scores')
                                exec(X+'_dists=dists')
                                exec(X+'_mindist=mindist')

                                ###path entropy
                                num_shortest_paths=numshortest_path_mat[int(portX-1),int(portY-1)]

                                if num_shortest_paths>1:
                                    paths=np.asarray([unique_adjacent(trial[trial>=0])[1:] for trial in all_trials])

                                    paths_correctx=paths[correct_boolean]
                                    if len(paths_correctx)>0:
                                        paths_correct=np.vstack(paths_correctx).tolist()
                                        num_times_pathx=number_of_repeats_ALL(paths_correct)
                                        num_times_path=np.hstack((np.repeat(0,                                                                            int(num_shortest_paths-len(num_times_pathx)))                                                                  ,num_times_pathx))
                                        prob_path=num_times_path/np.sum(num_times_path)
                                        transition_entropy=st.entropy(prob_path,base=num_shortest_paths)
                                else:
                                    transition_entropy=np.nan


                                All_scores.append(scores)
                                scores_dic[mouse][structure_num][X][session]=                                len(np.where(scores==True)[0])/len(scores)

                                scores_dic['entropy'][mouse][structure_num][X][session]=transition_entropy
                                All_entropies[indx]=transition_entropy

                                ##scoring non-task transitions (e.g. CA,BA,DB,DC...etc)
                                trial_nodes=[unique_adjacent(trial[trial>=0])[1:] for trial in all_trials]
                                for state_, indx_ in state_ind_dic.items():
                                    port_state=structure[indx_]
                                    distancesX_state=[np.where(trial_nodes[ii]==port_state)[0]+1                                                       for ii in range(len(trial_nodes))]
                                    mindistX_state=mindistance_mat[portX-1,port_state-1]
                                    X_state_score=(np.asarray([mindistX_state in distancesX_state[ii]                                                                        for ii in range(len(trial_nodes))])).                                    astype(float)
                                    X_state_score[error_mask]=np.nan
                                    exec('X'+str(state_)+'_score=X_state_score')

                                    X_state_dists=(np.asarray([distancesX_state[ii][0] if len(distancesX_state[ii])>0                                                              else np.nan for ii in range(len(distancesX_state))]))                                    .astype(float)
                                    X_state_dists[error_mask]=np.nan
                                    exec('X'+str(state_)+'_dists=X_state_dists')

                                    exec('X_state_mindist=X'+str(state_)+'_mindist=mindistX_state')


                            else:
                                All_scores=[]
                                for state_, indx_ in state_ind_dic.items():
                                    exec('X'+str(state_)+'_score=[]')
                                    exec('X'+str(state_)+'_dists=[]')
                                    exec('X'+str(state_)+'_mindist=[]')

                                exec(X+'_scores=[]')
                                exec(X+'_dists=[]')
                                exec(X+'_mindist=np.nan')

                            for state_, indx_ in state_ind_dic.items():
                                exec('scores_dic[mouse][structure_num][X+state_][session]=X'+str(state_)+'_score')
                                exec('scores_dic["dists"][mouse][structure_num][X+state_][session]=X'+str(state_)+                                     '_dists')
                                exec('scores_dic["mindist"][mouse][structure_num][X+state_][session]=X'+str(state_)+                                     '_mindist')
                        else:
                            trialtimes_state_dic[mouse][structure_num][session][state]=[]
                            print('Not enough trials')
                            continue
                scores_dic['entropy'][mouse][structure_num]['ALL'][session]=All_entropies

                if len(trialtimes_)>1:
                    if len(All_scores)>0 and len(A_scores)>=len(D_scores):        
                        All_scores=np.concatenate(All_scores)
                        Overall_score=len(np.where(All_scores==True)[0])/len(All_scores)

                        stacked_scores=np.column_stack((A_scores[:len(D_scores)],B_scores[:len(D_scores)],                                                        C_scores[:len(D_scores)],D_scores))

                        stacked_dists=np.column_stack((A_dists[:len(D_dists)],B_dists[:len(D_dists)],                                                        C_dists[:len(D_dists)],D_dists))

                        all_mindist=np.asarray([A_mindist,B_mindist,C_mindist,D_mindist])

                else:
                    Overall_score=np.nan
                    num_trials=Num_trials_dic[mouse][structure_num][structure_ses]#[session]
                    stacked_scores=stacked_dists=np.asarray([[np.nan for x in range(len(All_trials_indx))]                                               for y in range(int(num_trials))])
                    all_mindist=[]


                scores_dic[mouse][structure_num]['Mean'][session]=Overall_score 
                scores_dic[mouse][structure_num]['ALL'][session]=stacked_scores
                scores_dic['dists'][mouse][structure_num]['ALL'][session]=stacked_dists
                scores_dic['mindist'][mouse][structure_num]['ALL'][session]=all_mindist


            else:
                print(mouse+' structure'+str(structure_num)+' session'+str(session)+' '+session_Date+'-'+                      str(session_timestamp)+' ROI file not found')

        #else:
        #    print(mouse+' structure'+str(structure_num)+' session'+str(session)+' '+session_Date+'-'+\
        #              str(session_timestamp)+' Already Analyzed!')
print(time.time()-tt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[163]:


#Nodes_trials_dic=rec_dd()

tracking_oversampling_factor=50 ##oversampling of nodes_cut_dic
behaviour_oversampling_factor=3 ##oversampling of trialtimes_dic

state_ind_dic={'A':0,'B':1,'C':2,'D':3}
state_nextstate_dic={'A':'B','B':'C','C':'D','D':'A'}


A_trial_indx=['A_on','B_on']
B_trial_indx=['B_on','C_on']
C_trial_indx=['C_on','D_on']
D_trial_indx=['D_on','A_on']
All_trials_indx=np.vstack((A_trial_indx,B_trial_indx,C_trial_indx,D_trial_indx))
re_run=True
abstract_structure_type='ABCD'
#mouse_recdays_todo=['me03_14122020']
for mouse in ['ab03','ah07']:
    print(mouse)
    cohort=Mice_cohort_dic[mouse[:4]]
    
    ephys_type=Cohort_ephys_type_dic[int(cohort)]
     

    if int(cohort) in [3,4]:
        use_interp=True
    else:
        use_interp=False

    data_directory='P:/Taskspace_abstraction/Data/cohort'+str(cohort)+'/'+mouse[:4]+'/'
    Behaviour_base=Data_folder_P+'cohort'+str(cohort)+'/'+mouse[:4]+'/Behaviour/'

    Variables=Variable_dic[mouse]
    structure_no_=np.copy(Variables['Structure_no'])
    structure_no_[structure_no_=='-']=-1
    Structure_no=(structure_no_).astype(int)
    Structure_abstract=Variables['Structure_abstract']
    ABCD_structure_no=Structure_no[Structure_abstract==abstract_structure_type]



    for structure_num in np.unique(ABCD_structure_no):
        if structure_num>40:
            continue
        print(structure_num)
        
        num_trials_=Num_trials_dic[mouse][structure_num]
        num_trials_reached=0
        for structure_ses, (session, content) in enumerate(times_dic[mouse][abstract_structure_type]                                                           [structure_num].items()):
            
            
            if num_trials_reached>30:
                continue
            num_trials_reached+=num_trials_[structure_ses]
            
            session_Date=Variables['Date'][session]
            session_timestamp=Variables['Behaviour'][session]
            rec_day=session_Date[8:10]+session_Date[5:7]+session_Date[:4]
            mouse_recday=mouse+'_'+rec_day
            ses_day=Variables['Behaviour'][Variable_dic[mouse]['Date']==session_Date]
            ses_ind=np.where(ses_day==session_timestamp)[0][0]

            ROI_accuracy_all=dict_to_array(ROI_accuracy_dic['accuracy'][mouse_recday])

            #if mouse_recday not in mouse_recdays_todo:
            #    continue


            ##Importing pinstate and ROI files##

            try:
                pinstatefile_path=data_directory+'/Behaviour/'+str(mouse)+'_pinstate_'+str(Variables['Date']                                                                                           [session])+'-'+                str(Variables['Tracking'][session])+'.csv'

                #file_path='me04-2020-11-12-145650.txt'
                with open(pinstatefile_path, 'r') as f:
                    print('Importing data file: '+os.path.split(pinstatefile_path)[1])
                    ttl_tracking = np.asarray([line.strip() for line in f.readlines() if line.strip()]).astype(int)
                ttl_binary=create_binary(ttl_tracking)

                #pinstate_dic[mouse][session]=ttl_binary

            except FileNotFoundError:
                print("File: "+pinstatefile_path+" Not Found")



            try:
                #ROIfile_path=data_directory+'/Behaviour/ROIs_'+str(mouse)+'_'+str(Variables['Date'][session])+'-'+\
                #str(Variables['Tracking'][session])+'.csv' ##change this!

                if use_interp==False:
                    ROIfile_path=Behaviour_base+'ROIs_'+str(mouse)+'_'+str(Variables['Date'][session])+'-'+                    str(Variables['Tracking'][session])+'.csv' ##change this!
                elif use_interp==True:
                    if len(ROI_accuracy_all)>ses_ind+1:
                        print('Tracking accuracy: '+str(ROI_accuracy_all[ses_ind])+'%')
                    else:
                        print('Tracking accuracy: Not calculated')

                    ROIfile_path=Behaviour_base+'rescaled_ROIs'+str(mouse)+'_'+rec_day+'_'+str(ses_ind)+'_interp.csv'

                #file_path='me04-2020-11-12-145650.txt'
                with open(ROIfile_path, 'r') as f:
                    print('Importing data file: '+os.path.split(ROIfile_path)[1])
                    data = np.genfromtxt(f, delimiter=',',dtype=str)
                #backHead = data[1:, 4]
                #betweenShoulders = data[1:, 5]
                #Majority = data[1:, 9]
                
                if len(np.shape(data))==0:
                    continue

                if use_interp==True:
                    body_part_name,body_part_namex='0','0' ###uses betweenShoulders

                else:
                    body_part_name='majority'#'betweenShoulders' ###choice of what point on animal to track
                    body_part_namex='Majority'
                body_part_name_used =                 np.asarray([body_part_name,body_part_namex])[[body_part_name in data[0],                                                              body_part_namex in data[0]]][0]

                body_part=(data[1:,data[0]==body_part_name_used]).squeeze()



            except FileNotFoundError:
                print("File: "+ROIfile_path+" Not Found")
                continue




            ###Matching ROI and behaviour timestamps###

            structure=structure_dic[mouse]['ABCD'][structure_num][session]
            times=times_dic[mouse]['ABCD'][structure_num][session]
            #body_part=ROI_dic[mouse][session]['body_part']
            #ttl_binary=pinstate_dic[mouse][session]
            
            if len(times['A_on'])==0:
                print('No Behaviour')
                continue



            if len(body_part)>0:
                print(len(ttl_binary)/len(body_part))
                if (len(ttl_binary)/len(body_part))==2:
                    body_part=np.repeat(body_part,2)

                elif (len(ttl_binary)/len(body_part))>1.99 and (len(ttl_binary)/len(body_part))!=2:
                    body_part=np.repeat(body_part,2)[:len(ttl_binary)]

                nodes=[np.nan if body_part[ii][:4]!='node' else int(body_part[ii][-1])                       for ii in range(len(body_part))]
                edges=[np.nan if body_part[ii][:4]!='edge' else [int(body_part[ii][-3]),int(body_part[ii][-1])]                       for ii in range(len(body_part)) ]

                times_oversampled=rec_dd()
                for index,item in times.items():
                    times_oversampled[index]=item*behaviour_oversampling_factor

                if 'A_on_first' in times_oversampled.keys():
                    times_oversampled['A_on']=np.hstack((times_oversampled['A_on_first'],                                                         times_oversampled['A_on']))

                ##Aligning behaviour and tracking
                if use_interp==False:


                    #diff_behTrack=len(times['rsync'])-len(np.where(unique_adjacent(ttl_binary)==1)[0])
                    #if diff_behTrack<0:
                    #    print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)+\
                    #          ' '+session_Date+'-'+str(session_timestamp)+\
                    #          ': more rsync pulses in pinstate than in behaviour files - SESSION NOT USED')
                    #    continue
                        
                    
                    if ephys_type!='Neuropixels':
                        diff_behTrack=len(times['rsync'])-len(np.where(unique_adjacent(ttl_binary)==1)[0])
                        if diff_behTrack<0:
                            print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)+                                  ' '+session_Date+'-'+str(session_timestamp)+                                  ': more rsync pulses in pinstate than in behaviour files - SESSION NOT USED')
                            print(diff_behTrack)
                            continue
                        num_sync_missed=times_oversampled['rsync'][diff_behTrack]
                    else:
                        diff_behTrack=0
                        num_sync_missed=0
                    
                    num_sync_missed=times_oversampled['rsync'][diff_behTrack]

                    ttl_first=np.where(ttl_binary==1)[0][diff_behTrack]
                    nodes_cut=nodes[ttl_first:]
                    nodes_cut_oversampledx=np.repeat(nodes_cut,tracking_oversampling_factor)
                    nodes_cut_oversampled=np.hstack((np.repeat(np.nan,num_sync_missed),nodes_cut_oversampledx))               

                    ##checks
                    if diff_behTrack>0:
                        print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)+                              ' '+session_Date+'-'+str(session_timestamp)+                              ': less rsync pulses in pinstate than in behaviour files')
                        print('Realigned ROI timestamps')
                        
                    


                    if len(ttl_binary)-len(body_part)!=0:
                        print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)                              +' '+session_Date+'-'+str(session_timestamp)+                              ': mismatch between length of pinstate file and ROI file')
                        if len(ttl_binary)-len(body_part)==1:
                            ttl_binary=ttl_binary[:-1]
                            print('But only one off so cut end of ttl_binary')
                        else:
                            continue
                            
                            
                    

                    tracking_length_mins=(len(ttl_binary))/(60*60)
                    behaviour_length_mins=times['rsync'][-1]/(1000*60)
                    metaData_length_mins=int(Variable_dic[mouse]['Session_time'][session])

                    if np.abs(tracking_length_mins-metaData_length_mins)>10 or np.abs                    (behaviour_length_mins-metaData_length_mins)>10:
                        print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)+' '+                              session_Date+'-'+str(session_timestamp)+                              ': session length does not match metaData file')
                        #continue
                        print('Error Ignored')

                    numdigits_times_oversampled=len(str(times_oversampled['rsync'][-1]))
                    numdigits_nodes_cut_oversampled=len(str(len(nodes_cut_oversampled)))

                    if (numdigits_nodes_cut_oversampled-numdigits_times_oversampled)>1:
                        print('ERROR: '+mouse+' structure'+str(structure_num)+' session'+str(session)+' '+session_Date                              +'-'+str(session_timestamp)+                              ': Oversampling causes mismatch between tracking and behaviour')
                        continue

                else:
                    nodes_cut=nodes
                    nodes_cut_oversampledx=np.repeat(nodes_cut,50)
                    nodes_cut_oversampled=nodes_cut_oversampledx 




                print(mouse+' structure'+str(structure_num)+' session'+str(session)+' '+session_Date                          +'-'+str(session_timestamp)+                          ': Passed all checks')
                ##ADD EXTRA CHECK THAT SYNC PULSES AND TTL PULSES RECEIVED BY CAMERA ARE AN EXACT MATCH

                #times_oversampled_dic[mouse][structure_num][session]=times_oversampled
                nodes_cut_dic[mouse][structure_num][session]=nodes_cut



                ## Scoring ##
                All_trials_allstates=[]
                error_mask_all=[]
                for indx, state in enumerate(states):

                    ##Trail times per state
                    X=state
                    Y=state_nextstate_dic[X]

                    current_port=structure[indx]
                    next_port=np.roll(structure,-1)[indx]
                    if len(times_oversampled[All_trials_indx[indx,0]])>0:
                        if state != 'D':
                            if len(times_oversampled[All_trials_indx[indx,0]])==                            len(times_oversampled[All_trials_indx[indx,1]]):
                                trialtimes_=                                np.column_stack((times_oversampled[All_trials_indx[indx,0]],                                                 times_oversampled[All_trials_indx[indx,1]]))
                            else:

                                trialtimes_=                                np.column_stack((times_oversampled[All_trials_indx[indx,0]][:-1],                                                 times_oversampled[All_trials_indx[indx,1]]))
                        else:
                            trialtimes_=                            np.column_stack((times_oversampled['D_on'][:len(times_oversampled['A_on'])-1],                                             times_oversampled['A_on'][1:len(times_oversampled['A_on'])]))

                        if len(trialtimes_)>1:
                            print(session)
                            trialtimes_state_dic[mouse][structure_num][session][state]=trialtimes_


                            ###scoring
                            state_timesx=indicesX2(trialtimes_,0)
                            state_times =[(state_timesx[ii]).astype(int) for ii in range(len(state_timesx))]

                            if len(state_times)>0: 
                                if state_times[-1][-1]>len(nodes_cut_oversampled):
                                    print(mouse+' structure'+str(structure_num)+' session'+str(session)+                                          ' '+session_Date+'-'+str(session_timestamp)+                                          ' Behaviour longer than tracking')
                                    state_times =[(state_timesx[ii]).astype(int)                                                  for ii in range(len(state_timesx))                                                 if state_timesx[ii][-1]<len(nodes_cut_oversampled)]

                            portX=structure[state_ind_dic[X]]
                            portY=structure[state_ind_dic[Y]]
                            ###remove first instances of start node from node_times_trial


                            if len(nodes_cut_oversampled)>0 and len(state_times)>0 and                             np.isnan(np.nanmean(nodes_cut_oversampled))==False:

                                all_trials=np.asarray([(remove_nan(nodes_cut_oversampled[state_times[ii]]))                                                       .astype(int) for ii in range(len(state_times))])
                                
                                #all_trials=remove_empty(all_trials)


                                ###removing tracking errors
                                #error_mask_first=np.asarray([unique_adjacent(trial[trial>=0])[0]!=current_port\
                                #                       for trial in all_trials])
                                #error_mask_next=np.asarray([unique_adjacent(trial[trial>=0])[-1]!=next_port\
                                #                       for trial in all_trials])
                                
                                error_mask_first=np.asarray([unique_adjacent(trial[trial>=0])[0]!=current_port                                                             if len(trial)>0 else False for trial in all_trials])
                                error_mask_next=np.asarray([unique_adjacent(trial[trial>=0])[-1]!=next_port                                                       if len(trial)>0 else False for trial in all_trials])
                                error_mask=np.logical_or(error_mask_first,error_mask_next)

                                #all_trials=all_trials[~error_mask]

                                all_trials_unique_nodes=np.asarray([unique_adjacent(trial[trial>=0])[:-1]                                                                    for trial in all_trials])

                                All_trials_allstates.append(all_trials_unique_nodes)
                                error_mask_all.append(error_mask)
                Nodes_trials_dic[mouse]['Nodes'][structure_num][structure_ses]=All_trials_allstates
                Nodes_trials_dic[mouse]['Tracking_errors'][structure_num][structure_ses]=error_mask_all


# In[ ]:





# In[ ]:





# In[ ]:





# In[164]:


for mouse in Mice:
    print(mouse)
    cohort=Mice_cohort_dic[mouse[:4]]
    Variables=Variable_dic[mouse]
    #Structure_no=(Variables['Structure_no']).astype(int)
    structure_no_=np.copy(Variables['Structure_no'])
    structure_no_[structure_no_=='-']=-1
    Structure_no=(structure_no_).astype(int)
    Structure_abstract=Variables['Structure_abstract']
    ABCD_structure_no=Structure_no[Structure_abstract==abstract_structure_type]

    for structure_num in np.unique(ABCD_structure_no):
        
        if structure_num>40:
            continue

        sessions_used=list(Nodes_trials_dic[mouse]['Nodes'][structure_num].keys())

        Nodes_pertrial_all=[]
        Nodes_pertrial_perstate_all=[]
        for structure_ses in sessions_used:
            Nodes_=Nodes_trials_dic[mouse]['Nodes'][structure_num][structure_ses]
            if len(Nodes_)<4:
                continue
            #Nodes_trials_dic[mouse]['Tracking_errors'][structure_num][structure_ses]
            if structure_ses==sessions_used[0]:
                Nodes_firsttrial=[Nodes_[state][0] for state in range(len(Nodes_))]
                Nodes_firsttrial_=np.copy(Nodes_firsttrial)
                num_outofturn_first_states=np.asarray([np.sum(np.isin(Nodes_firsttrial_[-ii-1],                                                                      structure[:len(structure)-ii]))-1                             for ii in range(len(structure))])
                num_outofturn_first=np.sum(num_outofturn_first_states)
                Prop_outofturn_rewardvisits_firsttrial=num_outofturn_first/                (len(np.hstack((Nodes_firsttrial_)))-len(structure))
                
                if Prop_outofturn_rewardvisits_firsttrial<0:
                    Prop_outofturn_rewardvisits_firsttrial=np.nan


            Nodes_pertrial=list([np.hstack(([Nodes_[state][trial] for state in range(len(Nodes_))]))                            for trial in np.arange(len(Nodes_[-1]))])
            Nodes_pertrial_perstate=list([[Nodes_[state][trial] for state in range(len(Nodes_))]                            for trial in np.arange(len(Nodes_[-1]))])
            
            if len(np.shape(Nodes_pertrial))>1:
                Nodes_pertrial=Nodes_pertrial,[]
        

            Nodes_pertrial_all.append(Nodes_pertrial)
            Nodes_pertrial_perstate_all.append(Nodes_pertrial_perstate)
        if len(Nodes_pertrial_all)==0:
            continue
        Nodes_pertrial_all=remove_empty(np.hstack((Nodes_pertrial_all)))
        
        

        Prop_outofturn_rewardvisits=np.asarray([(np.sum(np.isin(Nodes_pertrial_all[trial], structure))-len(structure))        /(len(Nodes_pertrial_all[trial])-len(structure))        for trial in np.arange(len(Nodes_pertrial_all))])
        
        Prop_outofturn_rewardvisits[Prop_outofturn_rewardvisits<0]=np.nan

        
        
        Nodes_trials_dic['Nodes_pertrial_all'][mouse][structure_num]=Nodes_pertrial_all
        Nodes_trials_dic['Nodes_pertrial_perstate_all'][mouse][structure_num]=Nodes_pertrial_perstate_all
        Nodes_trials_dic['Prop_outofturn_rewardvisits_firsttrial'][mouse][structure_num]=        Prop_outofturn_rewardvisits_firsttrial
        Nodes_trials_dic['Prop_outofturn_rewardvisits'][mouse][structure_num]=Prop_outofturn_rewardvisits


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
on trial 1:
count out of turn Ds in D (1 is optimal)
cont out of turn Cs in D and C (1 is optimal)
...etc

on all other trials -
count out of turn visits in all trials - just all visits minus 1

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





# In[16]:


file = open(Intermediate_object_folder+"full_session_summary_1back.pickle",'rb')
structure_probabilities = pickle.load(file)
file.close()

file = open(Intermediate_object_folder+"ab_full_session_summary_1back.pickle",'rb')
structure_probabilitiesAB = pickle.load(file)
file.close()

#file = open(Intermediate_object_folder+"exp_session_summary_1back.pickle",'rb')
#structure_probabilities_exp = pickle.load(file)
#file.close()



file = open(Intermediate_object_folder+'all_mice_exp.pickle','rb')
structure_probabilities_exp = pickle.load(file)
file.close()


# In[ ]:





# In[ ]:





# In[23]:


##calculating behavioural biases - exploration
behbias_mat_dic=rec_dd()

for mouse in Mice:
    print(mouse)
    cohort=Mice_cohort_dic[mouse]
    if cohort<=2:
        num_tasks_mouse=10
    else:
        num_tasks_mouse=40
    
    Variables=Variable_dic[mouse]
    #Structure_no=(Variables['Structure_no']).astype(int)
    structure_no_=np.copy(Variables['Structure_no'])
    structure_no_[structure_no_=='-']=-1
    Structure_no=(structure_no_).astype(int)
    Structure_abstract=Variables['Structure_abstract']
    ABCD_structure_no=Structure_no[Structure_abstract=='ABCD']


    for structure_num in np.arange(num_tasks_mouse)+1:
        Prob_mat=np.zeros((4,4))
        Prob_mat[:]=np.nan
        structurex=dict_to_array(structure_dic[mouse]['ABCD'][structure_num])
        if len(structurex)>0:
            structure=structurex[0]
            
            if len(structure)<4:
                structure=structurex[-1]
            if len(structure)<4:
                continue
            for X_ind, (X, X_) in enumerate(state_nextstate_dic.items()):
                for Y_ind, (Y, Y_) in enumerate(state_nextstate_dic.items()):
                    if X!=Y:
                        portX=structure[state_ind_dic[X]]
                        portY=structure[state_ind_dic[Y]]

                        min_dist=mindistance_mat[portX-1,portY-1]
                        
                        if cohort<5:
                            structure_probdicX=structure_probabilities_exp[mouse][str(min_dist)]
                            portXs=(structure_probdicX['past_nodes']).values
                            portYs=(structure_probdicX['next_state']).values
                            probs=(structure_probdicX['prob']).values
                            mask1=[portXs==str(portX)]
                            mask2=[portYs==portY]
                            prob=probs[(np.asarray(mask1) & np.asarray(mask2)).squeeze()][0]
                        elif cohort>=5:
                            trans_mat_mean=np.load(Intermediate_object_folder_dropbox+'/'+                                                   mouse+'_Exploration_Transition_matrix.npy')
                            prob=trans_mat_mean[min_dist-1][portX-1,portY-1]
                            
                        Prob_mat[X_ind,Y_ind]=prob


        behbias_mat_dic[mouse][structure_num]=Prob_mat


# In[39]:


len(structure)


# In[ ]:





# In[19]:


###Making per recording_day/session dictionary and npy arrays
abstract_structure_type='ABCD'
for mouse in Mice:
    
    #mouse='me08'
    print(mouse)
    cohort=Mice_cohort_dic[mouse[:4]]

    if int(cohort)>2:
        use_interp=True
    else:
        use_interp=False

    data_directory='/Taskspace_abstraction/Data/cohort'+str(cohort)+'/'+mouse[:4]+'/'
    Behaviour_base=Data_folder_P+'cohort'+str(cohort)+'/'+mouse[:4]+'/Behaviour/'

    Variables=Variable_dic[mouse]
    structure_no_=np.copy(Variables['Structure_no'])
    structure_no_[structure_no_=='-']=-1
    Structure_no=(structure_no_).astype(int)
    Structure_abstract=Variables['Structure_abstract']
    ABCD_structure_no=Structure_no[Structure_abstract==abstract_structure_type]



    for structure_num in np.unique(ABCD_structure_no):
        print(structure_num)

        for structure_ses, (session, content) in enumerate(times_dic[mouse][abstract_structure_type]                                                           [structure_num].items()):
            session_Date=Variables['Date'][session]
            session_timestamp=Variables['Behaviour'][session]
            rec_day=session_Date[8:10]+session_Date[5:7]+session_Date[:4]
            mouse_recday=mouse+'_'+rec_day
            ses_day=Variables['Behaviour'][Variable_dic[mouse]['Date']==session_Date]
            ses_ind=np.where(ses_day==session_timestamp)[0][0]
            
            scores_ses=scores_dic[mouse][structure_num]['ALL'][session]
            #structure_ses=structure_dic[mouse]['ABCD'][task_num+1][session]
            structure_ses=structure_dic[mouse]['ABCD'][structure_num][session]
            
            if len(scores_ses)==0:
                scores_ses=[]
                
            scores_dic[mouse_recday][ses_ind]['ALL']=scores_ses
            structure_dic[mouse_recday][ses_ind]=structure_ses
            
            #np.save(Intermediate_object_folder+'Scores_'+mouse_recday+'_'+str(ses_ind)+'.npy',\
            #        scores_ses)
            ##this is done seperately in the tidying data notebook   


# ###cleaning up (removing old entries that may need updating due to re-loading metadata)
# structure_num_ses_dicX=rec_dd()
# for mouse in Mice:
#     print(mouse)
#     cohort=Mice_cohort_dic[mouse[:4]]
#     
#     
#     data_directory='/Taskspace_abstraction/Data/cohort'+str(cohort)+'/'+mouse[:4]+'/'
# 
#     Variables=Variable_dic[mouse]
#     Structure_no=(Variables['Structure_no']).astype(int)
#     Structure_abstract=Variables['Structure_abstract']
#     ABCD_structure_no=Structure_no[Structure_abstract==abstract_structure_type]
# 
#     for structure_num in np.unique(ABCD_structure_no):
#         print(structure_num)
#         
#         structure_num_sessions=[]
#         for structure_ses, (session, content) in enumerate(times_dic[mouse][abstract_structure_type]\
#                                                            [structure_num].items()):
#             structure_num_sessions.append(session)
#         
#         
#         for dicX in [trialtimes_state_dic,nodes_cut_dic]:
#             ses_actual_all=list(dicX[mouse][structure_num].keys())
#             for ses_actual in ses_actual_all:
#                 if ses_actual not in structure_num_sessions:
#                     print('cleaned')
#                     #del(dicX[mouse][structure_num][ses_actual])
#         
#         keys_scores=scores_dic[mouse][structure_num].keys()
#         for key_ in keys_scores: 
#             ses_actual_all=list(scores_dic[mouse][structure_num][key_].keys())
#             
#             for ses_actual in ses_actual_all:
#                 if ses_actual not in structure_num_sessions:
#                     print('cleaned')
#                     #del(scores_dic[mouse][structure_num][key_][ses_actual])
#                     #del(scores_dic['dists'][mouse][structure_num][key_][ses_actual])
#                     #del(scores_dic['mindist'][mouse][structure_num][key_][ses_actual])
#         
# 

# In[ ]:


#structure_dic.keys()


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
corner to opposite corner=6
corner to non opposite corner =1
corner to cardinal 1 = 1
corner to cardinal non 1 = 3
corner to middle = 2

cardinal to opposite cardinal = 1
cardinal to non-opposite cardinal = 2
cardinal to middle=1


'''


# In[ ]:





# In[184]:


half_way_implanted_mice=['me03']
All_task_mice=np.setdiff1d(Ephys_mice,half_way_implanted_mice)


# In[430]:


mean_entropy_dic=rec_dd()
for mouse in Mice:
    mean_entropy_all=np.zeros(40)
    mean_entropy_all[:]=np.nan
    for Task in np.arange(40)+1:
        entropy_all=dict_to_array(scores_dic['entropy'][mouse][Task]['ALL'])
        if len(entropy_all)>0:
            mean_entropy=np.nanmean(entropy_all)
        else:
            mean_entropy=np.nan
        mean_entropy_all[Task-1]=mean_entropy
    mean_entropy_dic['mean_entropy'][mouse]=mean_entropy_all


# In[ ]:





# In[431]:


All_task_mice


# In[ ]:





# In[ ]:





# In[432]:


mean_entropy=dict_to_array(mean_entropy_dic['mean_entropy'])
mouse_mean_entropy=np.nanmean(mean_entropy,axis=1)
mouse_sem_entropy=st.sem(mean_entropy,axis=1,nan_policy='omit')[0]
task_mean_entropy=np.nanmean(mean_entropy,axis=0)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

bar_plotX([mouse_mean_entropy,mouse_mean_entropy],'none',0,1.2,'points','unpaired',0.025)
plt.axhline(1,ls='dashed',color='black',linewidth=3)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'mean_entropy_bar.svg')
plt.show()
print(st.ttest_1samp(mouse_mean_entropy,1))
plt.errorbar(x=np.arange(len(task_mean_entropy)),y=task_mean_entropy, yerr=mouse_sem_entropy)

plt.show()


# In[427]:


np.shape(mouse_mean_entropy)


# In[ ]:





# In[ ]:





# In[187]:


for mouse in All_task_mice:
    task_performance_allsessions=[]
    mean_entropy_allsessions=[]

    task_performance_session1=np.zeros(40)
    mean_entropy_session1=np.zeros(40)

    task_performance_session1[:]=np.nan
    mean_entropy_session1[:]=np.nan


    for Task in np.arange(40)+1:
        task_performance=dict_to_array(scores_dic[mouse][Task]['Mean'])
        task_entropies=dict_to_array(scores_dic['entropy'][mouse][Task]['ALL'])
        if len(task_entropies)>0:
            task_entropies_means=np.nanmean(task_entropies,axis=1)

            task_performance_allsessions.append(task_performance)
            mean_entropy_allsessions.append(task_entropies_means)

            task_performance_session1[Task-1]=task_performance[0]
            mean_entropy_session1[Task-1]=task_entropies_means[0]

    mean_entropy_dic['performance_session'][mouse]=task_performance_allsessions
    mean_entropy_dic['entropy_session'][mouse]=mean_entropy_allsessions

    mean_entropy_dic['performance_firstsession'][mouse]=task_performance_session1
    mean_entropy_dic['entropy_firstsession'][mouse]=mean_entropy_session1


# In[ ]:





# In[188]:


for measure in ['entropy_firstsession','performance_firstsession']:
    _first_ses=dict_to_array(mean_entropy_dic[measure])

    mouse_firstses_=np.nanmean(_first_ses,axis=0)
    mouse_sem_firstses_=st.sem(_first_ses,axis=0,nan_policy='omit')[0]
    plt.errorbar(x=np.arange(len(mouse_firstses_)),y=mouse_firstses_, yerr=mouse_sem_firstses_)
    plt.show()


# In[ ]:





# In[190]:


for mouse in All_task_mice:
    print(mouse)
    all_perf=np.hstack((mean_entropy_dic['performance_session'][mouse]))
    all_entropy=np.hstack((mean_entropy_dic['entropy_session'][mouse]))

    xy=column_stack_clean(all_perf,all_entropy)
    #sns.regplot(xy[:,0],xy[:,1])
    plt.show()
    print(st.pearsonr(xy[:,0],xy[:,1]))


# In[193]:


all_perf=np.hstack((np.hstack((dict_to_array(mean_entropy_dic['performance_session'])))))
all_entropy=np.hstack((np.hstack((dict_to_array(mean_entropy_dic['entropy_session'])))))

xy=column_stack_clean(all_perf,all_entropy)
#sns.regplot(pd.DataFrame(xy))
st.pearsonr(xy[:,0],xy[:,1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


##exclusions - IGNORED

'''
me05: 9 triggered first A by foot 
me08: 37 task was a shifted version of another task on same day
ah02: 23 had to check during trialwhether node was delivering reliably
'''

exclusions_dic={'me03':[],'me04':[],'me05':[9],'me06':[],'me08':[37],'ah02':[23],'ah03':[19,24,32], 'me10':[10,13,28,37],                'me11':[],'ah04':[16,22,31,33,34],'ah05':[24,37,38,39,40],'ab03':[],'ah07':[]}


# In[ ]:





# In[19]:


Mice


# In[ ]:





# In[91]:


###Calculating zero-shot performance
zero_shot_dic=rec_dd()
num_trials_thr=2
Other_exclusions=False
transition_array=np.asarray(['DA','CA','BA','DB','DC'])
for mouse in Ephys_mice_3task:
    print(mouse)
    cohort=Mice_cohort_dic[mouse]
    if cohort<=2:
        num_tasks_mouse=10
    else:
        num_tasks_mouse=40
        
    excluded_tasks=exclusions_dic[mouse]
        
    zero_shot_array=np.zeros((num_tasks_mouse,len(transition_array)))
    zero_shot_array[:]=np.nan
    
    mindist_array=np.zeros((num_tasks_mouse,len(transition_array)))
    mindist_array[:]=np.nan
    
    chance_array=np.zeros((num_tasks_mouse,len(transition_array)))
    chance_array[:]=np.nan
    
    behbias_array=np.zeros((num_tasks_mouse,len(transition_array)))
    behbias_array[:]=np.nan
    
    for task_num in np.arange(num_tasks_mouse):
        
        if Other_exclusions==True and task_num in excluded_tasks:
            print('Not calculated - in exclusion list')
            continue
            
        
        
        print(task_num+1)
        for transition_ind, transition in enumerate(transition_array):

            sessions_all=np.sort(list(scores_dic[mouse][task_num+1]['ALL'].keys()))
            Num_trials_allses=Num_trials_dic[mouse][task_num+1]
            
            num_Num_trials_allses=len(Num_trials_allses)

            if len(sessions_all)==1:
                first_session=sessions_all[0]
                scores_all=scores_dic[mouse][task_num+1][transition][first_session]
            elif len(sessions_all)>1:
                trials_done_bool=np.zeros((len(sessions_all),2))
                scores_all=[]
                for session_ind_ind, session_ind in enumerate(sessions_all):
                    if session_ind_ind+1>num_Num_trials_allses:
                        print('Note: mismatch between num sessions in Num_trials_dic and scores_dic')
                        continue
                    
                    scores_session=scores_dic[mouse][task_num+1][transition][session_ind]
                    
                    Num_trials=Num_trials_allses[session_ind_ind]
                    if isinstance(Num_trials,float)==False and                    isinstance(Num_trials,int)==False:
                        print('Files not found')
                        continue
                    if Num_trials>1:
                        trials_done_bool[session_ind_ind,0]=1
                    
                    if len(scores_session)>0:
                        trials_done_bool[session_ind_ind,1]=1
                        scores_all.append(scores_session)
                        
                if len(scores_all)>0 and len(np.where(trials_done_bool[:,0]==1)[0])>0:
                    scores_all=np.hstack((scores_all))
                else:
                    print('Not calculated - no trials completed')
                    continue
                
                first_session_withtrials=np.where(trials_done_bool[:,0]==1)[0][0]
                if trials_done_bool[first_session_withtrials,1]==0:
                    zero_shot_array[task_num,transition_ind]=np.nan
                    print('Not calculated - first session not imported - remport!')
            elif len(sessions_all)==0:
                scores_all=[]

            if len(scores_all)>num_trials_thr:
                score_first_trial=scores_all[0]
                zero_shot_array[task_num,transition_ind]=score_first_trial
                
                min_dist_task=dict_to_array(scores_dic['mindist'][mouse][task_num+1][transition])[0]
                mindist_array[task_num,transition_ind]=min_dist_task
            else:
                print('Not calculated - number of trials too low')
                continue
            
            if len(structure_dic[mouse]['ABCD'][task_num+1])==0:
                print('Not calculated - no Structures_dic entry')
                continue
            structure_task=dict_to_array(structure_dic[mouse]['ABCD'][task_num+1])[0]
            
            if len(structure_task)<4:
                structure_task=dict_to_array(structure_dic[mouse]['ABCD'][task_num+1])[-1]
            
            if len(structure_task)<4:
                continue
                
                
            state_chance_mat=np.zeros((4,4))
            state_chance_mat[:]=np.nan
            for state_x in np.arange(4):
                for state_y in np.arange(4):
                    state_chance_mat[state_x,state_y]=prob_mat[structure_task[state_x]-1,structure_task[state_y]-1]

            state_indx=state_ind_dic[transition[0]]
            state_indy=state_ind_dic[transition[1]]
            chance_transition=state_chance_mat[state_indx,state_indy]
            chance_array[task_num,transition_ind]=chance_transition
            
            behbias_mat=behbias_mat_dic[mouse][task_num+1]
            behbias_transition=behbias_mat[state_indx,state_indy]
            behbias_array[task_num,transition_ind]=behbias_transition

            


    

    early_tasks=np.nanmean(zero_shot_array[:num_tasks_mouse//2],axis=0)
    late_tasks=np.nanmean(zero_shot_array[num_tasks_mouse//2:],axis=0)
    
    early_tasks_mindist=np.nanmean(mindist_array[:num_tasks_mouse//2],axis=0)
    late_tasks_mindist=np.nanmean(mindist_array[num_tasks_mouse//2:],axis=0)
    
    early_tasks_chance_array=np.nanmean(chance_array[:num_tasks_mouse//2],axis=0)
    late_tasks_chance_array=np.nanmean(chance_array[num_tasks_mouse//2:],axis=0)
    
    early_tasks_behbias_array=np.nanmean(behbias_array[:num_tasks_mouse//2],axis=0)
    late_tasks_behbias_array=np.nanmean(behbias_array[num_tasks_mouse//2:],axis=0)
    
    zero_shot_dic[mouse]=zero_shot_array
    zero_shot_dic['min_dist'][mouse]=mindist_array
    zero_shot_dic['chance_array'][mouse]=chance_array
    zero_shot_dic['chance_array'][mouse]=behbias_array
    
    zero_shot_dic['Early_tasks'][mouse]=early_tasks
    zero_shot_dic['Late_tasks'][mouse]=late_tasks
    
    zero_shot_dic['min_dist']['Early_tasks'][mouse]=early_tasks_mindist
    zero_shot_dic['min_dist']['Late_tasks'][mouse]=late_tasks_mindist
    
    zero_shot_dic['chance_array']['Early_tasks'][mouse]=early_tasks_chance_array
    zero_shot_dic['chance_array']['Late_tasks'][mouse]=late_tasks_chance_array
    
    zero_shot_dic['behbias_array']['Early_tasks'][mouse]=early_tasks_behbias_array
    zero_shot_dic['behbias_array']['Late_tasks'][mouse]=late_tasks_behbias_array


# In[ ]:





# In[67]:


'''
Missing sessions

me03 - accounted for: 8 - tried first session once and no trials so re-did transition on next day

me05 - accounted for: 3,4 no trials first session - FIXED

me08 - accounted for: 9 (first session - only one trial) - FIXED 34 (file not found) -  15,16 (no complete trials)

me10 - accounted for: 2 (FIXED?) 37 (only one trial)
     
me11 - accounted for: 3 - FIXED!

ah04 - accounted for: 31 (only two full trials) 

ah05 - accounted for: 37,38,39,40 (no or very few trials)
     
     - not accounted for: 32


'''


# In[68]:


Ephys_mice_3task=Ephys_mice[Ephys_mice!='me03']


# In[124]:





# In[ ]:





# In[92]:


for task_group in ['Early_tasks','Late_tasks']:
    print('')
    print(task_group)
    zero_shots_=dict_to_array(zero_shot_dic[task_group])
    min_dist_=dict_to_array(zero_shot_dic['min_dist'][task_group])
    DA_CA_BA=zero_shots_[:,:3]
    DA_DB_DC=zero_shots_[:,[0,-2,-1]]
    
    DA_CA_BA_min_dist=min_dist_[:,:3]
    DA_DB_DC_min_dist=min_dist_[:,[0,-2,-1]]

    print('DA CA BA')
    
    bar_plotX(DA_CA_BA.T,'none',0,0.8,'points','paired',0.025)
    plt.savefig(Behaviour_output_folder_dropbox+task_group+'_DA_CA_BA_bar.svg')
    plt.show()
    plt.rcParams["figure.figsize"] = (6,6)
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.top'] = True

    noplot_scatter(np.nanmean(DA_CA_BA[:,1:],axis=1),DA_CA_BA[:,0],color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(Behaviour_output_folder_dropbox+task_group+'_DA_CA_BA_scatter.svg')
    plt.show()
    print(st.wilcoxon(np.nanmean(DA_CA_BA[:,1:],axis=1),DA_CA_BA[:,0]))
    print(st.f_oneway(DA_CA_BA[:,0],DA_CA_BA[:,1],DA_CA_BA[:,2]))

    print('DA DB DC')
    zero_shots_=dict_to_array(zero_shot_dic['Late_tasks'])
    bar_plotX(DA_DB_DC.T,'none',0,0.8,'points','paired',0.025)
    plt.savefig(Behaviour_output_folder_dropbox+task_group+'_DA_DB_DC_bar.svg',               bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    plt.rcParams["figure.figsize"] = (8,6)
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.top'] = True

    noplot_scatter(np.nanmean(DA_DB_DC[:,1:],axis=1),DA_DB_DC[:,0],color='black')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(Behaviour_output_folder_dropbox+task_group+'_DA_DB_DC_scatter.svg',               bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print(st.wilcoxon(np.nanmean(DA_DB_DC[:,1:],axis=1),DA_DB_DC[:,0]))
    print(st.f_oneway(DA_DB_DC[:,0],DA_DB_DC[:,1],DA_DB_DC[:,2]))
    
    print('_____')


# In[95]:


zero_shot_differential=DA_CA_BA[:,0]-np.nanmean(DA_CA_BA[:,1:],axis=1)
print(np.mean(zero_shot_differential))
np.std(zero_shot_differential)


# In[142]:


anchoring=[0.06,0.12,0.06,0.06,0.12,0.09,0.08]

print(np.mean(anchoring))
print(np.std(anchoring))


# In[143]:


(0.08-(0.08*0.8))/0.02


# In[ ]:





# In[135]:


trial_inds1=np.arange(10)*2
trial_inds2=(np.arange(10)*2)+1

Late_zs1=np.hstack(([np.nanmean(zero_shot_dic[mouse][:,0][20:40][trial_inds1]) for mouse in Ephys_mice_3task]))
Late_zs2=np.hstack(([np.nanmean(zero_shot_dic[mouse][:,0][20:40][trial_inds2]) for mouse in Ephys_mice_3task]))


# In[136]:


Late_zs_diff=Late_zs1-Late_zs2
print(np.nanmean(Late_zs_diff))
print(np.std(Late_zs_diff))


# In[137]:


Late_zs1


# In[96]:


print(np.mean(DA_CA_BA[:,0]))
print(np.std(DA_CA_BA[:,0]))


# In[140]:


print(np.mean(DA_CA_BA[:,0])*0.2)
(0.36-(0.36*0.8))/0.04


# In[84]:


(np.nanmean(DA_CA_BA[:,0])*0.8)-np.nanmean(DA_CA_BA[:,1:])


# In[244]:


DAvsCA=(DA_CA_BA[:,0]-DA_CA_BA[:,1])/(DA_CA_BA[:,0]+DA_CA_BA[:,1])
CAvsBA=(DA_CA_BA[:,1]-DA_CA_BA[:,2])/(DA_CA_BA[:,1]+DA_CA_BA[:,2])
DAvsBA=(DA_CA_BA[:,0]-DA_CA_BA[:,2])/(DA_CA_BA[:,0]+DA_CA_BA[:,2])

DAvsDB=(DA_DB_DC[:,0]-DA_DB_DC[:,1])/(DA_DB_DC[:,0]+DA_DB_DC[:,1])
DBvsDC=(DA_DB_DC[:,1]-DA_DB_DC[:,2])/(DA_DB_DC[:,1]+DA_DB_DC[:,2])
DAvsDC=(DA_DB_DC[:,0]-DA_DB_DC[:,2])/(DA_DB_DC[:,0]+DA_DB_DC[:,2])

diff=(DA_CA_BA[:,0]-np.nanmean(DA_CA_BA[:,1:],axis=1))/np.nanmean(DA_CA_BA[:,1:],axis=1)
###i.e. which mice express the zero shot

#bool_used=np.repeat(True,len(DAvsBA))
bool_used=diff>0


bool_used[:4]=False



DAvsCA=DAvsCA[bool_used]
CAvsBA=CAvsBA[bool_used]
DAvsBA=DAvsBA[bool_used]

Mice_used=np.asarray(list(zero_shot_dic[task_group].keys()))[bool_used]

###i.e. of the mice that express the zero shot, do they show a generic novelty preference

bar_plotX([DAvsCA,CAvsBA],'none',-0.2,0.4,'nopoints','paired',0.025)
plt.show()
plot_scatter(DAvsCA,CAvsBA)
plt.show()
print(st.ttest_1samp(DAvsCA,0))
print(st.ttest_1samp(CAvsBA,0))
print(st.wilcoxon(DAvsCA,CAvsBA))

print(Mice)
print(Mice_used)
print(np.subtract(DAvsCA,CAvsBA))


# In[ ]:





# In[247]:


###Regression to tease out effect of novelty vs zeroshot 
from sklearn.linear_model import LinearRegression

novelty=np.hstack((np.repeat(0,len(DAvsCA)),np.repeat(0,len(CAvsBA)),np.repeat(1,len(DAvsBA)),                  ))
zeroshot=np.hstack((np.repeat(1,len(DAvsCA)),np.repeat(0,len(CAvsBA)),np.repeat(1,len(DAvsBA))                   ))
diffs=np.hstack((DAvsCA,CAvsBA,DAvsBA))

###Regression
X=np.vstack((zeroshot,novelty)).T
y=diffs
reg=LinearRegression(positive=True).fit(X, y)
coeff_real=reg.coef_

###shuffles
num_iterations=1000
coeffs_shuff=np.zeros((num_iterations,2))
coeffs_shuff[:]=np.nan

coeffs_shuff_diff=np.zeros(num_iterations)
coeffs_shuff_diff[:]=np.nan


for iteration in np.arange(num_iterations):
    y_shuff=np.copy(y)
    np.random.shuffle(y_shuff)
    X=np.vstack((zeroshot,novelty)).T
    reg=LinearRegression().fit(X, y_shuff)
    coeffs_shuff[iteration]=reg.coef_
    coeffs_shuff_diff[iteration]=reg.coef_[0]-reg.coef_[1]




# In[253]:


np.shape(coeffs_shuff)


# In[254]:


for coeff_ind in np.arange(2):
    plt.hist(coeffs_shuff[:,coeff_ind],bins=50,color='grey')
    plt.axvline(np.percentile(coeffs_shuff[:,coeff_ind],95),ls='dashed',color='black')
    plt.arrow(coeff_real[coeff_ind],5,0,-4, width=0.007,head_width=0.02,head_length=1,color='blue')
    
    plt.show()
    print(st.percentileofscore(coeffs_shuff[:,coeff_ind],coeff_real[coeff_ind]))
    
plt.hist(coeffs_shuff_diff,bins=50,color='grey')
plt.axvline(np.percentile(coeffs_shuff_diff,95),ls='dashed',color='black')
plt.arrow(coeff_real[0]-coeff_real[1],5,0,-4, width=0.007,head_width=0.02,head_length=1,color='blue')
plt.show()
print(st.percentileofscore(coeffs_shuff_diff,coeff_real[0]-coeff_real[1]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[410]:


##controls

for control in ['min_dist','chance_array','behbias_array']:
    print('')
    print('________')
    print('________')
    print(control)
    for task_group in ['Early_tasks','Late_tasks']:
        print('')
        print(task_group)
        zero_shots_=dict_to_array(zero_shot_dic[task_group])
        control_=dict_to_array(zero_shot_dic[control][task_group])
        DA_CA_BA=control_[:,:3]
        DA_DB_DC=control_[:,[0,-2,-1]]

        DA_CA_BA=control_[:,:3]
        DA_DB_DC=control_[:,[0,-2,-1]]

        print('DA CA BA')
        bar_plotX(DA_CA_BA.T,'none',0,np.nanmax(DA_CA_BA),'points','paired',0.025)
        plt.savefig(Behaviour_output_folder_dropbox+task_group+'_'+control+'_bar.svg')
        plt.show()
        
        plt.rcParams["figure.figsize"] = (8,6)
        plt.rcParams['axes.linewidth'] = 4
        plt.rcParams['axes.spines.right'] = True
        plt.rcParams['axes.spines.top'] = True
        
        noplot_scatter(np.nanmean(DA_CA_BA[:,1:],axis=1),DA_CA_BA[:,0],color='black')
        plt.tick_params(axis='both',  labelsize=20)
        plt.tick_params(width=2, length=6)
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(Behaviour_output_folder_dropbox+task_group+'_'+control+'_scatter.svg',                   bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        print(st.wilcoxon(np.nanmean(DA_CA_BA[:,1:],axis=1),DA_CA_BA[:,0]))
        print(st.f_oneway(DA_CA_BA[:,0],DA_CA_BA[:,1],DA_CA_BA[:,2]))

        print('DA DB DC')
        zero_shots_=dict_to_array(zero_shot_dic['Late_tasks'])
        bar_plotX(DA_DB_DC.T,'none',0,np.nanmax(DA_DB_DC),'points','paired',0.025)
        plt.show()
        #noplot_scatter(np.nanmean(DA_DB_DC[:,1:],axis=1),DA_DB_DC[:,0],color='black')
        #plt.tick_params(axis='both',  labelsize=20)
        #plt.tick_params(width=2, length=6)
        #plt.gca().set_aspect('equal', adjustable='box')
        
        #plt.show()
        #print(st.wilcoxon(np.nanmean(DA_DB_DC[:,1:],axis=1),DA_DB_DC[:,0]))
        #print(st.f_oneway(DA_DB_DC[:,0],DA_DB_DC[:,1],DA_DB_DC[:,2]))

        print('_____')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[98]:


###Initial performance calculation
performance_dic=rec_dd()
reldist_dic=rec_dd()
num_trials_first=20
num_states=len(states)
for mouse in Mice:
    print(mouse)
    cohort=Mice_cohort_dic[mouse]
    if cohort<=2:
        num_tasks_mouse=10
    else:
        num_tasks_mouse=40

    excluded_tasks=exclusions_dic[mouse]

    performance_array=np.zeros((num_tasks_mouse,num_trials_first,num_states))
    performance_array[:]=np.nan
    
    relative_distance_array=np.zeros((num_tasks_mouse,num_trials_first))
    relative_distance_array[:]=np.nan
    for task_num in np.arange(num_tasks_mouse):
        print(task_num+1)
        try:
            sessions=np.sort(list(scores_dic[mouse][task_num+1]['ALL'].keys()))
            Num_trials_allses=Num_trials_dic[mouse][task_num+1]
            num_Num_trials_allses=len(Num_trials_allses)

            scores_all=[]
            for session_ind, session in enumerate(sessions):
                if session_ind+1>num_Num_trials_allses:
                    print('Note: mismatch between num sessions in Num_trials_dic and scores_dic')
                    continue
                scores_session=scores_dic[mouse][task_num+1]['ALL'][session]
                if len(scores_session)>0:
                    scores_all.append(scores_session)

            scores_all=np.vstack((scores_all))
            scores_N=scores_all[:num_trials_first]
            if len(scores_N)<num_trials_first:
                added_array=np.zeros((num_trials_first-len(scores_N),4))
                added_array[:]=np.nan
                scores_N=np.vstack((scores_N,added_array))
            performance_array[task_num]=scores_N
            
            
            
            
            ###distances
            dists_task_=dict_to_array(scores_dic['dists'][mouse][task_num+1]['ALL'])
            if len(dists_task_)==1:
                dists_task=np.vstack((dists_task_))
            else:
                if len(dists_task_[1])==0:
                    dists_task=dists_task_[0]
                else:
                    dists_task=np.vstack((dists_task_))
            
            min_dists_task=dict_to_array(scores_dic['mindist'][mouse][task_num+1]['ALL'])[0]
            
            min_dists_task_sum=np.sum(min_dists_task)
            if min_dists_task_sum==0:
                min_dists_task_sum=np.nan
            relative_dists_task=(np.sum(dists_task,axis=1)/min_dists_task_sum)[:num_trials_first]
            if len(relative_dists_task)<num_trials_first:
                added_array=np.zeros((num_trials_first-len(relative_dists_task)))
                added_array[:]=np.nan
                relative_dists_task=np.concatenate((relative_dists_task,added_array))
            
            relative_distance_array[task_num]=relative_dists_task
            
        except Exception as e:
            print('Not calculated - non-existent')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    early_tasks=np.nanmean(performance_array[:num_tasks_mouse//2],axis=0)
    late_tasks=np.nanmean(performance_array[num_tasks_mouse//2:],axis=0)
    
    early_tasks_reldists=np.nanmean(relative_distance_array[:num_tasks_mouse//2],axis=0)
    late_tasks_reldists=np.nanmean(relative_distance_array[num_tasks_mouse//2:],axis=0)

    performance_dic[mouse]=performance_array
    performance_dic['Early_tasks'][mouse]=early_tasks
    performance_dic['Late_tasks'][mouse]=late_tasks
    
    reldist_dic[mouse]=relative_distance_array
    reldist_dic['Early_tasks'][mouse]=early_tasks_reldists
    reldist_dic['Late_tasks'][mouse]=late_tasks_reldists


# In[105]:


np.shape(dists_task)
trial_inds1=np.arange(10)*2
trial_inds2=(np.arange(10)*2)+1


# In[107]:


trial_inds2


# In[132]:


trial_inds1=np.arange(10)*2
trial_inds2=(np.arange(10)*2)+1


Late_perf=[np.nanmean(performance_dic['Late_tasks'][mouse]) for mouse in Ephys_mice_3task]
print(np.nanmean(Late_perf))
print(np.std(Late_perf))


# In[133]:


trial_inds1=np.arange(10)*2
trial_inds2=(np.arange(10)*2)+1

Late_perf1=np.hstack(([np.nanmean(performance_dic['Late_tasks'][mouse][trial_inds1]) for mouse in Ephys_mice_3task]))
Late_perf2=np.hstack(([np.nanmean(performance_dic['Late_tasks'][mouse][trial_inds2]) for mouse in Ephys_mice_3task]))


Late_perf_diff=Late_perf1-Late_perf2
print(np.nanmean(Late_perf_diff))
print(np.std(Late_perf_diff))


# In[117]:


0.42*0.2



# In[209]:


###Initial Performance calculation - full trial
num_trials_first=20
performance_dic2=rec_dd()
for mouse in Mice:
    print(mouse)
    try:
        cohort=Mice_cohort_dic[mouse]
        if cohort<=2:
            num_tasks_mouse=10
        else:
            num_tasks_mouse=40
            
        if cohort==5:
            continue

        excluded_tasks=exclusions_dic[mouse]

        performance_array_trial=[]
        performance_array_trial_end=[]
        for structure_num in np.arange(num_tasks_mouse)+1:
            if len(remove_empty(dict_to_array(scores_dic['dists'][mouse][structure_num]['ALL'])))==0:
                print('Missing entry for task'+str(structure_num))
                continue
            dists_all_structurenum=np.vstack((remove_empty(dict_to_array(scores_dic['dists'][mouse]                                                                         [structure_num]['ALL']))))

            min_dists_structurenum=dict_to_array(scores_dic['mindist'][mouse][structure_num]['ALL'])[0]
            Actual_trial_distance=np.sum(dists_all_structurenum,axis=1)
            min_trial_distance=np.sum(min_dists_structurenum)

            if min_trial_distance==0:
                print('Error in calculating min distance for structure num'+str(structure_num))
                Norm_trial_distance=np.repeat(np.nan,num_trials_first)
            else:
                Norm_trial_distance=Actual_trial_distance/min_trial_distance

            if len(Norm_trial_distance)<num_trials_first:
                Norm_trial_distance=np.hstack((Norm_trial_distance,                                                   np.repeat(np.nan,num_trials_first-len(Norm_trial_distance))))
            performance_array_trial.append(Norm_trial_distance[:num_trials_first])

            performance_array_trial_end.append(Norm_trial_distance[-num_trials_first:])

        performance_dic2['Norm_trial_distance'][mouse]=np.vstack((performance_array_trial))
        performance_dic2['Norm_trial_distance_early'][mouse]=np.vstack((performance_array_trial))[:num_tasks_mouse//2]
        performance_dic2['Norm_trial_distance_late'][mouse]=np.vstack((performance_array_trial))[num_tasks_mouse//2:]

        performance_dic2['Norm_trial_distance_end'][mouse]=np.vstack((performance_array_trial_end))
        performance_dic2['Norm_trial_distance_early_end'][mouse]=np.vstack((performance_array_trial_end))        [:num_tasks_mouse//2]
        performance_dic2['Norm_trial_distance_late_end'][mouse]=np.vstack((performance_array_trial_end))        [num_tasks_mouse//2:]

    except Exception as e:
        print(e)


# In[ ]:





# In[ ]:


print(np.mean(zero_shot_differential))
np.std(zero_shot_differential)


# In[ ]:





# In[333]:


Norm_trial_distance_early_allmice=np.vstack(([np.nanmean(performance_dic2['Norm_trial_distance_early'][mouse],axis=0)for mouse in performance_dic2['Norm_trial_distance_early'].keys()]))

Norm_trial_distance_late_allmice=np.vstack(([np.nanmean(performance_dic2['Norm_trial_distance_late'][mouse],axis=0)for mouse in performance_dic2['Norm_trial_distance_late'].keys()]))

plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.errorbar(np.arange(num_trials_first),np.nanmean(Norm_trial_distance_early_allmice,axis=0),             st.sem(Norm_trial_distance_early_allmice,axis=0,nan_policy='omit'),color='darkcyan',linewidth=3)
plt.errorbar(np.arange(num_trials_first),np.nanmean(Norm_trial_distance_late_allmice,axis=0),             st.sem(Norm_trial_distance_late_allmice,axis=0,nan_policy='omit'),color='firebrick', linewidth=3)

chance_wholetrial=np.nanmean(Norm_trial_distance_early_allmice,axis=0)[0]

plt.plot(Norm_trial_distance_early_allmice.T,color='cyan',alpha=0.2)
plt.plot(Norm_trial_distance_late_allmice.T,color='firebrick',alpha=0.2)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.axhline(chance_wholetrial,ls='dashed',color='grey')
plt.axhline(1,ls='dashed',color='black')
plt.ylim(0,8)

plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_timeline_distances_wholetrial.svg')
plt.show()


# In[375]:


Distance=np.hstack((np.hstack((Norm_trial_distance_early_allmice)),np.hstack((Norm_trial_distance_late_allmice))))
Mouse=np.tile(np.repeat(np.arange(len(Mice)),np.shape(late_task_timeline)[1]),2)
Trial=np.tile(np.tile(np.arange(num_trials_first),np.shape(late_task_timeline)[0]),2)
Task=np.repeat(np.arange(2),len(np.hstack((late_task_timeline))))


dataframe = pd.DataFrame({'Mouse': Mouse,
                          'Trial': Trial,
                          'Task': Task,\
                         'Distance':Distance})

import pingouin as pg

# Compute the 2-way repeated measures ANOVA. This will return a dataframe.
pg.rm_anova(dv='Distance', within=['Trial','Task'], subject='Mouse', data=dataframe)

# Optional post-hoc tests
#pg.pairwise_ttests(dv='Distance', within=['Trial','Task'], subject='Mouse', data=dataframe)

#dataframe.rm_anova(dv='Distance', within=['Trial','Task'], subject='Mouse')


# In[374]:


np.shape(Norm_trial_distance_early_allmice)


# In[ ]:





# In[372]:


##Proportion perfect trials first 20 trials

prop_perfect_early=np.vstack(([[np.sum(remove_nan(performance_dic2['Norm_trial_distance_early'][mouse][:,trial_no])==1)/len(remove_nan(performance_dic2['Norm_trial_distance_early'][mouse][:,trial_no])) for trial_no in np.arange(len(performance_dic2['Norm_trial_distance_early'][mouse].T))]for mouse in performance_dic2['Norm_trial_distance_early'].keys()]))

prop_perfect_late=np.vstack(([[np.sum(remove_nan(performance_dic2['Norm_trial_distance_late'][mouse][:,trial_no])==1)/len(remove_nan(performance_dic2['Norm_trial_distance_late'][mouse][:,trial_no])) for trial_no in np.arange(len(performance_dic2['Norm_trial_distance_late'][mouse].T))]for mouse in performance_dic2['Norm_trial_distance_late'].keys()]))


behavioural_biases=np.hstack(([np.nanmean([np.nanmean(np.hstack((np.diag(behbias_mat_dic[mouse][int(task_num)],1),                                                          behbias_mat_dic[mouse][int(task_num)][-1,0])))for task_num in behbias_mat_dic[mouse].keys()]) for mouse in Mice]))

behavioural_biases_wholetrial=np.hstack(([np.nanmean([np.product(np.hstack((np.diag(behbias_mat_dic[mouse][int(task_num)],1),                                                          behbias_mat_dic[mouse][int(task_num)][-1,0])))for task_num in behbias_mat_dic[mouse].keys()]) for mouse in Mice]))

plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.errorbar(np.arange(num_trials_first),np.nanmean(prop_perfect_early,axis=0),             st.sem(prop_perfect_early,axis=0,nan_policy='omit'),color='darkcyan',linewidth=3)
plt.errorbar(np.arange(num_trials_first),np.nanmean(prop_perfect_late,axis=0),             st.sem(prop_perfect_late,axis=0,nan_policy='omit'),color='firebrick', linewidth=3)

plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
chance_wholetrial_prop=np.nanmean(behavioural_biases_wholetrial)
plt.axhline(chance_wholetrial_prop,ls='dashed',color='grey')
plt.savefig(Behaviour_output_folder_dropbox+task_group+'_prop_perfect_earlylate_line.svg')
plt.show()


# In[377]:


Prop_perfect=np.hstack((np.hstack((prop_perfect_early)),np.hstack((prop_perfect_late))))
Mouse=np.tile(np.repeat(np.arange(len(Mice)),np.shape(late_task_timeline)[1]),2)
Trial=np.tile(np.tile(np.arange(num_trials_first),np.shape(late_task_timeline)[0]),2)
Task=np.repeat(np.arange(2),len(np.hstack((late_task_timeline))))


dataframe = pd.DataFrame({'Mouse': Mouse,
                          'Trial': Trial,
                          'Task': Task,\
                         'Prop_perfect':Prop_perfect})

import pingouin as pg

# Compute the 2-way repeated measures ANOVA. This will return a dataframe.
pg.rm_anova(dv='Prop_perfect', within=['Trial','Task'], subject='Mouse', data=dataframe)

# Optional post-hoc tests
#pg.pairwise_ttests(dv='Distance', within=['Trial','Task'], subject='Mouse', data=dataframe)

#dataframe.rm_anova(dv='Distance', within=['Trial','Task'], subject='Mouse')


# In[ ]:





# In[388]:


mean_early=np.nanmean(prop_perfect_early,axis=1)
mean_late=np.nanmean(prop_perfect_late,axis=1)

xyz=np.column_stack((mean_early,mean_late,behavioural_biases_wholetrial))

#bar_plotX(xyz.T,'none', 0, 0.2, 'points', 'paired', 0.025)

plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True

noplot_scatter(mean_early,behavioural_biases_wholetrial,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig(Behaviour_output_folder_dropbox+task_group+'_prop_perfect_early_scatter.svg')
plt.show()
print(st.wilcoxon(mean_early,behavioural_biases_wholetrial))


noplot_scatter(mean_late,behavioural_biases_wholetrial,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig(Behaviour_output_folder_dropbox+task_group+'_prop_perfect_late_scatter.svg')
plt.show()
print(st.wilcoxon(mean_late,behavioural_biases_wholetrial))

noplot_scatter(mean_early,mean_late,color='black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig(Behaviour_output_folder_dropbox+task_group+'_prop_perfect_latevsearly_scatter.svg')
plt.show()
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
bar_plotX([mean_late,mean_early],'none',0, 0.2, 'points', 'paired', 0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.axhline(np.nanmean(behavioural_biases_wholetrial),color='black',ls='dashed')
plt.savefig(Behaviour_output_folder_dropbox+task_group+'_prop_perfect_latevsearly_bar.svg')
plt.show()
print(st.wilcoxon(mean_late,mean_early))
print(st.ttest_1samp(mean_early,np.nanmean(behavioural_biases_wholetrial)))
print(st.ttest_1samp(mean_late,np.nanmean(behavioural_biases_wholetrial)))


# In[366]:


behavioural_biases_wholetrial


# In[435]:


np.nanmean(behavioural_biases_wholetrial)


# In[ ]:





# In[ ]:


'''
consistent across mice?
replicates to cohorts 5 and 6?

'''


# In[ ]:





# In[213]:


###distribution of relative lengths

Norm_trial_distance_early_allmice_all=np.concatenate(np.vstack(([performance_dic2['Norm_trial_distance_early'][mouse]for mouse in performance_dic2['Norm_trial_distance_early'].keys()])))

Norm_trial_distance_late_allmice_all=np.concatenate(np.vstack(([performance_dic2['Norm_trial_distance_late'][mouse]for mouse in performance_dic2['Norm_trial_distance_late'].keys()])))

plt.hist(Norm_trial_distance_early_allmice_all,bins=25)
plt.show()

plt.hist(Norm_trial_distance_late_allmice_all,bins=25)
plt.show()


# In[214]:


Norm_trial_distance_early_allmice=np.vstack(([np.nanmean(performance_dic2['Norm_trial_distance_early_end'][mouse],axis=0)for mouse in performance_dic2['Norm_trial_distance_early_end'].keys()]))

Norm_trial_distance_late_allmice=np.vstack(([np.nanmean(performance_dic2['Norm_trial_distance_late_end'][mouse],axis=0)for mouse in performance_dic2['Norm_trial_distance_late_end'].keys()]))


plt.errorbar(np.arange(num_trials_first),np.nanmean(Norm_trial_distance_early_allmice,axis=0),             st.sem(Norm_trial_distance_early_allmice,axis=0,nan_policy='omit'),color='grey')
plt.errorbar(np.arange(num_trials_first),np.nanmean(Norm_trial_distance_late_allmice,axis=0),             st.sem(Norm_trial_distance_late_allmice,axis=0,nan_policy='omit'),color='black')
plt.show()


# In[215]:


###distribution of relative lengths

Norm_trial_distance_early_allmice_all_end=np.concatenate(np.vstack(([performance_dic2['Norm_trial_distance_early_end']    [mouse] for mouse in performance_dic2['Norm_trial_distance_early_end'].keys()])))

Norm_trial_distance_late_allmice_all_end=np.concatenate(np.vstack(([performance_dic2['Norm_trial_distance_late_end']    [mouse] for mouse in performance_dic2['Norm_trial_distance_late_end'].keys()])))

plt.hist(Norm_trial_distance_early_allmice_all_end,bins=25)
plt.show()

plt.hist(Norm_trial_distance_late_allmice_all_end,bins=25)
plt.show()


# In[ ]:





# In[216]:


##Proportion perfect trials last 20 trials
proportion_perfect_early_end=np.sum((remove_nan(Norm_trial_distance_early_allmice_all_end)==1))/len(remove_nan(Norm_trial_distance_late_allmice_all))

proportion_perfect_late_end=np.sum((remove_nan(Norm_trial_distance_late_allmice_all_end)==1))/len(remove_nan(Norm_trial_distance_late_allmice_all))

print(proportion_perfect_late_end)
print(proportion_perfect_early_end)


# In[393]:


rec_days_


# In[394]:


Tasknum_used_dic


# In[ ]:





# In[430]:





# In[ ]:





# In[442]:





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
Import location per ABCDE day
Import trial times
unique adjacent
compare to minimum length
plot

'''


# In[ ]:





# In[31]:


Ephys_mice

['ah02','ah05']


# In[ ]:





# In[37]:


###How correlated are the tasks?
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
from scipy.stats import tukey_hsd

num_nodes=9
num_states=4
use_sequence=True
use_shifted=True
remove_first2=False
abstract_structure='ABCD'

unrecorded_ephys_mice=['ah02','ah05']

for task_group in ['All','Early','Late','3_task','Neural_days','Neural_cohort','Neural_cohort_early']:
    print(task_group)
    Corrs_tasks_all=[]
    Corrs_tasks_perday_all=[]
    distances_mean_combined_all_all=[]
    distances_mean_combined_mouse_all=[]
    for mouse in Mice:
        cohort=Mice_cohort_dic[mouse]
        if cohort<=2:
            #continue
            num_tasks_mouse=10
        else:
            num_tasks_mouse=40
        multi_hot_array_all=[]
        
        if task_group=='All':
            tasks_used=np.arange(num_tasks_mouse)+1
            if remove_first2==True:
                tasks_used=tasks_used[2:]
        elif task_group=='Early':
            tasks_used=np.arange(num_tasks_mouse//2)+1
        elif task_group=='Late':
            tasks_used=np.arange(num_tasks_mouse//2)+num_tasks_mouse//2+1
        elif task_group=='3_task':
            tasks_used=np.arange(30)+11
            if cohort<=2:
                continue
        elif task_group=='Neural_days':
            tasks_used=np.arange(20)+21
            if cohort<=2 or mouse in unrecorded_ephys_mice:
                continue
                
        elif task_group=='Neural_cohort':
            tasks_used=np.arange(num_tasks_mouse)+1
            if cohort<=2 or mouse in unrecorded_ephys_mice:
                continue
                
        elif task_group=='Neural_cohort_early':
            tasks_used=np.arange(20)
            if cohort<=2 or mouse in unrecorded_ephys_mice:
                continue
                
        distances_mean_combined_all=[]
        for structure_num in tasks_used:

            if len(remove_empty(dict_to_array(structure_dic[mouse]['ABCD'][structure_num])))==0:
                continue
            structure=remove_empty(dict_to_array(structure_dic[mouse]['ABCD'][structure_num]))[0]
            if len(structure)==len(abstract_structure):
                
                if use_sequence==True:
                    multi_hot_array=np.zeros(num_nodes*num_states)

                    for state_ind in np.arange(num_states):
                        multi_hot_array[(structure[state_ind]-1).astype(int)+(state_ind*num_nodes)]=1
                else:
                    multi_hot_array=np.zeros(num_nodes)
                    multi_hot_array[(structure-1).astype(int)]=1

            multi_hot_array_all.append(multi_hot_array)
            
            distances_mean_combined=[]
            structure_=structure-1
            for task_distance in np.arange(3)+1:
                exec('distances_'+str(task_distance)+'=distances_=np.hstack(([distance_mat[structure_[ii],                np.roll(structure_,-task_distance)[ii]]for ii in range(len(structure_))]))')

                exec('distances_mean_'+str(task_distance)+'=distances_mean_=np.nanmean(distances_)')
                distances_mean_combined.append(distances_mean_)
            distances_mean_combined=np.hstack((distances_mean_combined))
            distances_mean_combined_all.append(distances_mean_combined)
            
        distances_mean_combined_all=np.vstack((distances_mean_combined_all))
        distances_mean_combined_mouse=np.nanmean(distances_mean_combined_all,axis=0)
        
        distances_mean_combined_all_all.append(distances_mean_combined_all)
        distances_mean_combined_mouse_all.append(distances_mean_combined_mouse)

        multi_hot_array_all=np.vstack((multi_hot_array_all))
        
        
        
        if use_shifted==True:
            corrs_all=[]
            for task_indX in np.arange(len(multi_hot_array_all)):
                for task_indY in np.arange(len(multi_hot_array_all)):
                    if task_indX==task_indY:
                        continue
                    for indx_roll in np.arange(num_states):
                        corr=st.pearsonr(multi_hot_array_all[task_indX],                                    np.roll(multi_hot_array_all[task_indY],num_nodes*indx_roll))[0]
                        corrs_all.append(corr)

            corrs_all=np.hstack((corrs_all))
            corrs_mean=np.nanmean(corrs_all)
            
        else:
            corrs_all=matrix_triangle(np.corrcoef(multi_hot_array_all))
            corrs_mean=np.nanmean(corrs_all)
        
        
        Corrs_tasks_all.append(corrs_mean)
        
        if cohort>2 and task_group=='3_task':
            day_corrs=[]
            for day in np.arange(10):
                
                multi_hot_array_day=multi_hot_array_all[3*day:3*(day+1)]
                if len(multi_hot_array_day)<2:
                    continue
                day_corr=matrix_triangle(np.corrcoef(multi_hot_array_day))
                day_corr_mean=np.nanmean(day_corr)
                day_corrs.append(day_corr_mean)

            Corrs_tasks_perday_all.append(np.nanmean(day_corrs))
    
    plt.rcParams["figure.figsize"] = (3,6)
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    
    bar_plotX([Corrs_tasks_all], 'none', -0.05, 0.05, 'points', 'paired', 0.025)
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Behaviour_output_folder_dropbox+'Task_correlation_'+task_group+'.svg')
    plt.show()
    print(len(Corrs_tasks_all))
    print(np.nanmean(Corrs_tasks_all))
    print(st.ttest_1samp(Corrs_tasks_all,0))
    
    if task_group=='3_task':
        bar_plotX([Corrs_tasks_perday_all], 'none', -0.05, 0.05, 'points', 'paired', 0.025)
        plt.show()
        print(np.nanmean(Corrs_tasks_perday_all))
        print(st.ttest_1samp(Corrs_tasks_perday_all,0))
    
    distances_mean_combined_all_all=np.vstack((distances_mean_combined_all_all))
    distances_mean_combined_mouse_all=np.vstack((distances_mean_combined_mouse_all))
    
    bar_plotX(distances_mean_combined_mouse_all.T,'none',0,4,'points','paired',0.025)
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(Behaviour_output_folder_dropbox+'Taskspace_physicalspace_correlation_'+task_group+'.svg')
    plt.show()
    
    stats=st.f_oneway(remove_nan(distances_mean_combined_mouse_all[:,0]),                      remove_nan(distances_mean_combined_mouse_all[:,1]),                      remove_nan(distances_mean_combined_mouse_all[:,2]))
    print(stats)
    


# In[37]:


multi_hot_array
num_states=4
multi_hot_array=np.zeros(num_nodes*num_states)

for state_ind in np.arange(num_states):
    multi_hot_array[(structure[state_ind]-1).astype(int)+(state_ind*num_nodes)]=1


# In[63]:


np.roll(multi_hot_array_all[task_indY],num_nodes*indx_roll)


# In[64]:


len(multi_hot_array_all[task_indY])


# In[43]:


multi_hot_array_all=np.vstack((multi_hot_array_all))
        all_corr=matrix_triangle(np.corrcoef(multi_hot_array_all))


# In[59]:


corrs_all=[]
for task_indX in np.arange(len(multi_hot_array_all)):
    for task_indY in np.arange(len(multi_hot_array_all)):
        if task_indX==task_indY:
            continue
        for indx_roll in np.arange(num_states):
            corr=st.pearsonr(multi_hot_array_all[task_indX],                        np.roll(multi_hot_array_all[task_indY],num_nodes*indx_roll))[0]
            corrs_all.append(corr)
                
corrs_all=np.hstack((corrs_all))
corrs_mean=np.nanmean(corrs_all)


# In[58]:


corrs_all


# In[ ]:





# In[228]:


x=np.hstack((np.tile(np.arange(3)+1,len(distances_mean_combined_all_all))             .reshape(len(distances_mean_combined_all_all),3)))
y=np.hstack((distances_mean_combined_all_all))

plt.scatter(x,y)
st.pearsonr(x,y)


# In[229]:


bar_plotX(distances_mean_combined_mouse_all.T,'none',0,4,'points','paired',0.025)


# In[401]:





# In[ ]:





# In[402]:



mean_firstN_all=np.zeros((len(Mice),40))
mean_firstN_all[:]=np.nan

mean_earlylate_all=np.zeros((len(Mice),2))
mean_earlylate_all[:]=np.nan
for mouse_ind, mouse in enumerate(Mice):
    mean_firstN_=np.nanmean(np.nanmean(performance_dic[mouse],axis=1),axis=1)
    mean_firstN=np.hstack((mean_firstN_,np.repeat(np.nan,40-len(mean_firstN_))))
    mean_firstN_all[mouse_ind]=mean_firstN
    
    early_tasks_mean=np.nanmean(performance_dic['Early_tasks'][mouse])
    late_tasks_mean=np.nanmean(performance_dic['Late_tasks'][mouse])
    mean_earlylate_all[mouse_ind]=np.asarray([early_tasks_mean,late_tasks_mean])
    
num_bins_perf=4
factor_perf=int(40/num_bins_perf)
binned_perf=np.zeros((len(Mice),num_bins_perf))
binned_perf[:]=np.nan
for bin_ in np.arange(num_bins_perf):
    mean_per_mouse=np.nanmean(mean_firstN_all[:,bin_*factor_perf:(bin_+1)*factor_perf],axis=1)
    binned_perf[:,bin_]=mean_per_mouse
    
bar_plotX(binned_perf.T,'none',0,0.7,'points','paired',0.025)
plt.show()

bar_plotX(mean_earlylate_all.T,'none',0,0.7,'points','paired',0.025)
plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_bar.svg')
plt.show()

plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True

noplot_scatter(mean_earlylate_all[:,0],mean_earlylate_all[:,1],color='black')
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_scatter.svg')
plt.show()
print(st.wilcoxon(mean_earlylate_all[:,0],mean_earlylate_all[:,1]))

early_task_timeline=np.asarray([np.nanmean(np.nanmean(performance_dic[mouse][:20],axis=2),axis=0) for mouse in Mice])
late_task_timeline=np.asarray([np.nanmean(np.nanmean(performance_dic[mouse][20:],axis=2),axis=0) for mouse in Mice])

plt.errorbar(np.arange(num_trials_first),np.nanmean(early_task_timeline,axis=0),st.sem(early_task_timeline,axis=0,                                                                                       nan_policy='omit'),color='grey')
plt.errorbar(np.arange(num_trials_first),np.nanmean(late_task_timeline,axis=0),st.sem(late_task_timeline,axis=0,                                                                                     nan_policy='omit'),color='black')
plt.show()


# In[417]:


rel_dist_first_=np.asarray([reldist_dic[mouse][:,0] for mouse in Mice])
rel_dist_first=np.vstack(([np.hstack((rel_dist_first_[ii],np.repeat(np.nan,int(40-len(rel_dist_first_[ii]))))) for ii in range(len(rel_dist_first_))]))



factor=5
bins=np.arange(len(rel_dist_first.T)//factor+1)*factor
rel_dist_first_binned=np.vstack(([st.binned_statistic(np.arange(len(rel_dist_first.T)),rel_dist_first[ii],                            bins=bins,statistic=np.nanmean)[0] for ii in np.arange(len(rel_dist_first))]))
plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.errorbar(np.arange(len(rel_dist_first_binned.T)),np.nanmean(rel_dist_first_binned,axis=0),             st.sem(rel_dist_first_binned,axis=0,nan_policy='omit'),color='black',linewidth=3)
plt.plot(rel_dist_first_binned.T,color='grey',alpha=0.3)
chance=np.nanmean(rel_dist_first_binned,axis=0)[0]
plt.axhline(chance,ls='dashed',color='grey')
plt.axhline(1,ls='dashed',color='grey')
plt.ylim(0,15)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Behaviour_output_folder_dropbox+'Firsttrial_timeline_distances.svg')
plt.show()

plt.errorbar(np.arange(len(rel_dist_first.T)),np.nanmean(rel_dist_first,axis=0),st.sem(rel_dist_first,axis=0,                                                                                      nan_policy='omit'),color='black')

plt.axhline(chance,ls='dashed',color='grey')
plt.axhline(1,ls='dashed',color='grey')
plt.ylim(0,10)
plt.show()


# In[ ]:





# In[420]:


rel_dist_first_binned


# In[ ]:





# In[424]:


###note - stats only on mice that spanned full 40 tasks
rel_dist_first_complete=rel_dist_first_binned[4:]
Distance=np.hstack((rel_dist_first_complete))
Mouse=np.repeat(np.arange(len(rel_dist_first_complete)),np.shape(rel_dist_first_complete)[1])
Task=np.tile(np.arange(np.shape(rel_dist_first_complete)[1]),np.shape(rel_dist_first_complete)[0])


dataframe = pd.DataFrame({'Mouse': Mouse,
                          'Task': Task,
                         'Distance':Distance})

for key_ in ['Mouse','Task','Distance']:
    dataframe[[key_]] = dataframe[[key_]].astype(int)



# Compute the 1-way repeated measures ANOVA. This will return a dataframe.
pg.rm_anova(dv='Distance', within=['Task'], subject='Mouse', data=dataframe)

# Optional post-hoc tests
#pg.pairwise_ttests(dv='Distance', within=['Task'], subject='Mouse', data=dataframe)

#dataframe.rm_anova(dv='Distance', within=['Task'], subject='Mouse')


# In[ ]:





# In[ ]:





# In[ ]:





# In[403]:


mean_firstN_all=np.zeros((len(Mice),40))
mean_firstN_all[:]=np.nan

mean_earlylate_all=np.zeros((len(Mice),2))
mean_earlylate_all[:]=np.nan
for mouse_ind, mouse in enumerate(Mice):
    mean_firstN_=np.nanmean(reldist_dic[mouse],axis=1)
    mean_firstN=np.hstack((mean_firstN_,np.repeat(np.nan,40-len(mean_firstN_))))
    mean_firstN_all[mouse_ind]=mean_firstN
    
    early_tasks_mean=np.nanmean(reldist_dic['Early_tasks'][mouse])
    late_tasks_mean=np.nanmean(reldist_dic['Late_tasks'][mouse])
    mean_earlylate_all[mouse_ind]=np.asarray([early_tasks_mean,late_tasks_mean])
    
num_bins_perf=4
factor_perf=int(40/num_bins_perf)
binned_perf=np.zeros((len(Mice),num_bins_perf))
binned_perf[:]=np.nan
for bin_ in np.arange(num_bins_perf):
    mean_per_mouse=np.nanmean(mean_firstN_all[:,bin_*factor_perf:(bin_+1)*factor_perf],axis=1)
    binned_perf[:,bin_]=mean_per_mouse
    
bar_plotX(binned_perf.T,'none',0,7,'points','paired',0.025)
plt.axhline(chance,ls='dashed',color='grey')
plt.axhline(1,ls='dashed',color='grey')
#plt.savefig(Behaviour_output_folder_dropbox+'last_20_reldist.svg')
plt.show()

bar_plotX(mean_earlylate_all.T,'none',0,7,'points','paired',0.025)
plt.axhline(chance,ls='dashed',color='grey')
plt.axhline(1,ls='dashed',color='grey')

plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_reldist_bar.svg')
plt.show()

plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True

noplot_scatter(mean_earlylate_all[:,0],mean_earlylate_all[:,1],color='black')
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_reldist_scatter.svg')

plt.show()
print(st.wilcoxon(mean_earlylate_all[:,0],mean_earlylate_all[:,1]))

early_task_timeline=np.asarray([np.nanmean(reldist_dic[mouse][:5],axis=0) for mouse in Mice])
late_task_timeline=np.asarray([np.nanmean(reldist_dic[mouse][5:],axis=0) for mouse in Mice])

plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.errorbar(np.arange(num_trials_first),np.nanmean(early_task_timeline,axis=0),st.sem(early_task_timeline,axis=0,                                                                nan_policy='omit'),color='darkcyan',linewidth=3)
plt.errorbar(np.arange(num_trials_first),np.nanmean(late_task_timeline,axis=0),st.sem(late_task_timeline,axis=0,                                                                nan_policy='omit'),color='firebrick',linewidth=3)

plt.plot(early_task_timeline.T,color='cyan',alpha=0.2)
plt.plot(late_task_timeline.T,color='firebrick',alpha=0.2)

plt.axhline(chance,ls='dashed',color='grey')
plt.axhline(1,ls='dashed',color='grey')
plt.ylim(0,10)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_timeline_distances.svg')
plt.show()


# In[ ]:





# In[415]:


Distance=np.hstack((np.hstack((early_task_timeline)),np.hstack((late_task_timeline))))
Mouse=np.tile(np.repeat(np.arange(len(Mice)),np.shape(late_task_timeline)[1]),2)
Trial=np.tile(np.tile(np.arange(num_trials_first),np.shape(late_task_timeline)[0]),2)
Task=np.repeat(np.arange(2),len(np.hstack((late_task_timeline))))


dataframe = pd.DataFrame({'Mouse': Mouse,
                          'Trial': Trial,
                          'Task': Task,\
                         'Distance':Distance})

import pingouin as pg

# Compute the 2-way repeated measures ANOVA. This will return a dataframe.
pg.rm_anova(dv='Distance', within=['Trial','Task'], subject='Mouse', data=dataframe)

# Optional post-hoc tests
#pg.pairwise_ttests(dv='Distance', within=['Trial','Task'], subject='Mouse', data=dataframe)

#dataframe.rm_anova(dv='Distance', within=['Trial','Task'], subject='Mouse')


# In[ ]:





# In[ ]:





# In[ ]:


'''
animals perform well on very first trial of late tasks

is this purely because of zeroshot (DA)? or also because other transitions are above chance? - 
-look at distances of other transitions
if latter , is it because less likely to return to re-visit previously rewarded locations-
-quantify all visits to previous locations

'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[237]:


median_firstN_all=np.zeros((len(Mice),40))
median_firstN_all[:]=np.nan

median_earlylate_all=np.zeros((len(Mice),2))
median_earlylate_all[:]=np.nan
for mouse_ind, mouse in enumerate(Mice):
    median_firstN_=np.nanmedian(reldist_dic[mouse],axis=1)
    median_firstN=np.hstack((median_firstN_,np.repeat(np.nan,40-len(median_firstN_))))
    median_firstN_all[mouse_ind]=median_firstN
    
    early_tasks_median=np.nanmedian(reldist_dic['Early_tasks'][mouse])
    late_tasks_median=np.nanmedian(reldist_dic['Late_tasks'][mouse])
    median_earlylate_all[mouse_ind]=np.asarray([early_tasks_median,late_tasks_median])
    
num_bins_perf=4
factor_perf=int(40/num_bins_perf)
binned_perf=np.zeros((len(Mice),num_bins_perf))
binned_perf[:]=np.nan
for bin_ in np.arange(num_bins_perf):
    median_per_mouse=np.nanmedian(median_firstN_all[:,bin_*factor_perf:(bin_+1)*factor_perf],axis=1)
    binned_perf[:,bin_]=median_per_mouse
    
bar_plotX(binned_perf.T,'none',0,5,'points','paired',0.025)
plt.savefig(Behaviour_output_folder_dropbox+'last_20_reldist_median.svg')
plt.show()

bar_plotX(median_earlylate_all.T,'none',0,5,'points','paired',0.025)
#plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_reldist_median_bar.svg')
plt.show()
noplot_scatter(median_earlylate_all[:,0],median_earlylate_all[:,1],color='black')
plt.gca().set_aspect('equal', adjustable='box')
#plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_reldist_median_scatter.svg')
plt.show()
print(st.wilcoxon(median_earlylate_all[:,0],median_earlylate_all[:,1]))


# In[238]:


min_firstN_all=np.zeros((len(Mice),40))
min_firstN_all[:]=np.nan

min_earlylate_all=np.zeros((len(Mice),2))
min_earlylate_all[:]=np.nan
for mouse_ind, mouse in enumerate(Mice):
    min_firstN_=np.nanmin(reldist_dic[mouse],axis=1)
    min_firstN=np.hstack((min_firstN_,np.repeat(np.nan,40-len(min_firstN_))))
    min_firstN_all[mouse_ind]=min_firstN
    
    early_tasks_min=np.nanmin(reldist_dic['Early_tasks'][mouse])
    late_tasks_min=np.nanmin(reldist_dic['Late_tasks'][mouse])
    min_earlylate_all[mouse_ind]=np.asarray([early_tasks_min,late_tasks_min])
    
num_bins_perf=4
factor_perf=int(40/num_bins_perf)
binned_perf=np.zeros((len(Mice),num_bins_perf))
binned_perf[:]=np.nan
for bin_ in np.arange(num_bins_perf):
    min_per_mouse=np.nanmin(min_firstN_all[:,bin_*factor_perf:(bin_+1)*factor_perf],axis=1)
    binned_perf[:,bin_]=min_per_mouse
    
bar_plotX(binned_perf.T,'none',0,5,'points','paired',0.025)
plt.show()

bar_plotX(min_earlylate_all.T,'none',0,5,'points','paired',0.025)
plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_reldist_min_bar.svg')
plt.show()
noplot_scatter(min_earlylate_all[:,0],min_earlylate_all[:,1],color='black')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_reldist_min_scatter.svg')
plt.show()
print(st.wilcoxon(min_earlylate_all[:,0],min_earlylate_all[:,1]))


# In[240]:


Prop_firsttrial_all=np.vstack(([np.hstack((np.asarray(dict_to_array(Nodes_trials_dic                                                                    ['Prop_outofturn_rewardvisits_firsttrial']                                                         [mouse])[:40]),                np.repeat(np.nan,                int(40-len(dict_to_array(Nodes_trials_dic['Prop_outofturn_rewardvisits_firsttrial'][mouse])[:40])))))                for mouse in Mice]))

plt.errorbar(np.arange(40),np.nanmean(Prop_firsttrial_all,axis=0),             yerr=st.sem(Prop_firsttrial_all,axis=0,nan_policy='omit'))
plt.show()
Prop_firsttrial_all_binned=bin_arrayX(Prop_firsttrial_all,5)

plt.errorbar(np.arange(len(Prop_firsttrial_all_binned.T)),np.nanmean(Prop_firsttrial_all_binned,axis=0),             st.sem(Prop_firsttrial_all_binned,axis=0,nan_policy='omit'),color='black')
#plt.ylim(0,10)
#plt.savefig(Behaviour_output_folder_dropbox+'EarlyvsLate_timeline_distances.svg')
plt.savefig(Behaviour_output_folder_dropbox+'Firsttrial_outofturn_props.svg')
plt.show()


# In[ ]:





# In[ ]:


Prop_firsttrial_all_binned


# In[ ]:


np.shape(rel_dist_first_complete)


# In[241]:


###note - stats only on mice that spanned full 40 tasks
rel_dist_first_complete=Prop_firsttrial_all_binned[4:,:-1]
Distance=np.hstack((rel_dist_first_complete))
Mouse=np.repeat(np.arange(len(rel_dist_first_complete)),np.shape(rel_dist_first_complete)[1])
Task=np.tile(np.arange(np.shape(rel_dist_first_complete)[1]),np.shape(rel_dist_first_complete)[0])


dataframe = pd.DataFrame({'Mouse': Mouse,
                          'Task': Task,
                         'Distance':Distance})

for key_ in ['Mouse','Task','Distance']:
    dataframe[[key_]] = dataframe[[key_]].astype(int)



# Compute the 1-way repeated measures ANOVA. This will return a dataframe.
pg.rm_anova(dv='Distance', within=['Task'], subject='Mouse', data=dataframe)

# Optional post-hoc tests
#pg.pairwise_ttests(dv='Distance', within=['Task'], subject='Mouse', data=dataframe)

dataframe.rm_anova(dv='Distance', within=['Task'], subject='Mouse')


# In[ ]:





# In[319]:


###Final performance calculation
performance_last_dic=rec_dd()
reldist_last_dic=rec_dd()
num_trials_last=20
num_states=len(states)
for mouse in Mice:
    print(mouse)
    cohort=Mice_cohort_dic[mouse]
    if cohort<=2:
        num_tasks_mouse=10
    else:
        num_tasks_mouse=40

    excluded_tasks=exclusions_dic[mouse]

    performance_array=np.zeros((num_tasks_mouse,num_trials_last,num_states))
    performance_array[:]=np.nan
    
    relative_distance_array=np.zeros((num_tasks_mouse,num_trials_last))
    relative_distance_array[:]=np.nan
    for task_num in np.arange(num_tasks_mouse):
        print(task_num+1)
        try:
            sessions=np.sort(list(scores_dic[mouse][task_num+1]['ALL'].keys()))
            Num_trials_allses=Num_trials_dic[mouse][task_num+1]
            num_Num_trials_allses=len(Num_trials_allses)

            scores_all=[]
            for session_ind, session in enumerate(sessions):
                if session_ind+1>num_Num_trials_allses:
                    print('Note: mismatch between num sessions in Num_trials_dic and scores_dic')
                    continue
                scores_session=scores_dic[mouse][task_num+1]['ALL'][session]
                if len(scores_session)>0:
                    scores_all.append(scores_session)

            scores_all=np.vstack((scores_all))
            scores_N=scores_all[-num_trials_last:]
            if len(scores_N)<num_trials_last:
                added_array=np.zeros((num_trials_last-len(scores_N),4))
                added_array[:]=np.nan
                scores_N=np.vstack((added_array,scores_N))
            performance_array[task_num]=scores_N
            
            
            
            
            ###distances
            dists_task_=dict_to_array(scores_dic['dists'][mouse][task_num+1]['ALL'])
            if len(dists_task_)==1:
                dists_task=np.vstack((dists_task_))
            else:
                if len(dists_task_[1])==0:
                    dists_task=dists_task_[0]
                else:
                    dists_task=np.vstack((dists_task_))
            
            min_dists_task=dict_to_array(scores_dic['mindist'][mouse][task_num+1]['ALL'])[0]
            
            min_dists_task_sum=np.sum(min_dists_task)
            if min_dists_task_sum==0:
                min_dists_task_sum=np.nan
            relative_dists_task=(np.sum(dists_task,axis=1)/min_dists_task_sum)[-num_trials_last:]
            if len(relative_dists_task)<num_trials_last:
                added_array=np.zeros((num_trials_last-len(relative_dists_task)))
                added_array[:]=np.nan
                relative_dists_task=np.concatenate((added_array,relative_dists_task))
            
            relative_distance_array[task_num]=relative_dists_task
            
        except Exception as e:
            print('Not calculated - non-existent')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    early_tasks=np.nanmean(performance_array[:num_tasks_mouse//2],axis=0)
    late_tasks=np.nanmean(performance_array[num_tasks_mouse//2:],axis=0)
    
    early_tasks_reldists=np.nanmean(relative_distance_array[:num_tasks_mouse//2],axis=0)
    late_tasks_reldists=np.nanmean(relative_distance_array[num_tasks_mouse//2:],axis=0)

    performance_last_dic[mouse]=performance_array
    performance_last_dic['Early_tasks'][mouse]=early_tasks
    performance_last_dic['Late_tasks'][mouse]=late_tasks
    
    reldist_last_dic[mouse]=relative_distance_array
    reldist_last_dic['Early_tasks'][mouse]=early_tasks_reldists
    reldist_last_dic['Late_tasks'][mouse]=late_tasks_reldists


# In[414]:


mean_last_trials=np.hstack(([np.nanmean(reldist_last_dic[mouse][:10]) for mouse in Mice]))

plt.rcParams["figure.figsize"] = (4,8)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

bar_plotX([mean_last_trials],'none', 0, 7, 'points', 'paired', 0.025)


rel_dist_first_=np.asarray([reldist_dic[mouse][:,0] for mouse in Mice])
rel_dist_first=np.vstack(([np.hstack((rel_dist_first_[ii],np.repeat(np.nan,int(40-len(rel_dist_first_[ii]))))) for ii in range(len(rel_dist_first_))]))
rel_dist_first_binned=np.vstack(([st.binned_statistic(np.arange(len(rel_dist_first.T)),rel_dist_first[ii],                            bins=bins,statistic=np.nanmean)[0] for ii in np.arange(len(rel_dist_first))]))
chance=np.nanmean(rel_dist_first_binned,axis=0)[0]

plt.axhline(chance,color='grey',ls='dashed')
plt.axhline(1,color='black',ls='dashed')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Last20trials.svg')
plt.show()
print(st.ttest_1samp(mean_last_trials,chance))
print(chance)


# In[ ]:





# In[ ]:





# In[260]:


###baseline probability vs re;ative distance (i.e. effect of baseline biases on mistakes)
prob_corr_incorr_dic=rec_dd()
corr_dist_pr_dic=rec_dd()
max_steps=4

'''
on a given trial - 
1-take all 1,2 3 and 4 step transitons taken (loop over each seperately)
2-calculate baseline probability that this transition wouldve been taken in pre-task exploration
3-compare this to mean relative path distance for the entire trial - i.e. a seperate correlation per number of steps



'''
for step_no in np.arange(4)+1:
    for mouse in Mice:
        cohort=Mice_cohort_dic[mouse]
        tasks_=list(structure_dic[mouse]['ABCD'].keys())[:40]
        prob_corr_incorr_all=np.zeros((len(tasks_),2))
        prob_corr_incorr_all[:]=np.nan
        
        corr_dist_pr_all=np.zeros((len(tasks_)))
        corr_dist_pr_all[:]=np.nan
        
        xy_corr_dist_pr_all=[]
        for task_ind,task_num in enumerate(tasks_):

            if cohort<3 and task_num>10:
                continue

            structure=dict_to_array(structure_dic[mouse]['ABCD'][task_num])[0]
            
            if cohort<=4:
                N_step_pr=structure_probabilities_exp[mouse][str(step_no)].values
                N_step_pr[:,0]=(N_step_pr[:,0]).astype(int)
            elif cohort>=5:
                trans_mat_mean=np.load(Intermediate_object_folder_dropbox+'/'+                                                   mouse+'_Exploration_Transition_matrix.npy')
                #prob=trans_mat_mean[min_dist-1][portX-1,portY-1]
                #N_step_pr=rec_dd()
                #for min_dist in np.arange(4):
                N_step_pr=[]
                for portX in np.arange(num_nodes):
                    for portY in np.arange(num_nodes):
                        N_step_pr.append([portX+1,portY+1,trans_mat_mean[step_no-1][portX,portY],                                          trans_mat_mean[step_no-1][portX,portY]])
                N_step_pr=np.vstack((N_step_pr))
                #N_step_pr[str(min_dist+1)]=probs_all
                    
                    
            if len(Nodes_trials_dic['Nodes_pertrial_perstate_all'][mouse][task_num])==0:
                continue
            Nodes_pertrial_perstate_task=np.vstack((Nodes_trials_dic['Nodes_pertrial_perstate_all'][mouse][task_num]))
            Nodes_pertrial_task=np.asarray(Nodes_trials_dic['Nodes_pertrial_all'][mouse][task_num])

            num_trials=len(Nodes_pertrial_perstate_task)
            prob_exp_actual_state=np.zeros((num_trials,num_states))
            prob_exp_actual_state[:]=np.nan
            
            prob_exp_actual=np.zeros((num_trials))
            prob_exp_actual[:]=np.nan
            for trial in np.arange(num_trials):
                
                if trial> len(Nodes_pertrial_task)-1:
                    continue
                
                if step_no==1:
                    transitions_=np.column_stack((Nodes_pertrial_task[trial][:-1],
                    Nodes_pertrial_task[trial][1:]))
                else:
                    transitions_=np.column_stack((Nodes_pertrial_task[trial][:-(step_no)],
                    Nodes_pertrial_task[trial][step_no:]))
                    
                mean_pr=np.nanmean([N_step_pr[np.logical_and(N_step_pr[:,0]==transitions_[ii,0],                                               N_step_pr[:,1]==transitions_[ii,1]),3][0] if                      len(N_step_pr[np.logical_and(N_step_pr[:,0]==transitions_[ii,0],                                               N_step_pr[:,1]==transitions_[ii,1]),3])>0 else np.nan
                     for ii in range(len(transitions_))])
                
                prob_exp_actual[trial]=mean_pr
                
                for state in np.arange(len(states)):
                    if len(Nodes_pertrial_perstate_task[trial][state])==0:
                        continue
                        
                    if step_no==1:
                        transitions_=np.column_stack((Nodes_pertrial_perstate_task[trial][state],np.hstack((                        Nodes_pertrial_perstate_task[trial][state][1:],structure[(state+1)%4]))))
                    else:
                        transitions_=np.column_stack((Nodes_pertrial_perstate_task[trial][state][:-(step_no-1)],                                                      np.hstack((                        Nodes_pertrial_perstate_task[trial][state][1:],structure[(state+1)%4]))[step_no-1:]))
                        
                    mean_pr=np.nanmean([N_step_pr[np.logical_and(N_step_pr[:,0]==transitions_[ii,0],                                               N_step_pr[:,1]==transitions_[ii,1]),3][0] if                      len(N_step_pr[np.logical_and(N_step_pr[:,0]==transitions_[ii,0],                                               N_step_pr[:,1]==transitions_[ii,1]),3])>0 else np.nan
                     for ii in range(len(transitions_))])

                    prob_exp_actual_state[trial,state]=mean_pr

            Scores=performance_dic[mouse][task_num-1]
            prob_exp_actual_trunc=prob_exp_actual_state[:len(Scores)]
            Scores=Scores[:len(prob_exp_actual_trunc)]
            Scores_conc=np.concatenate(Scores)
            prob_exp_actual_conc=np.concatenate(prob_exp_actual_trunc)
            prob_exp_actual_corr=np.nanmean(prob_exp_actual_conc[Scores_conc==1])
            prob_exp_actual_incorr=np.nanmean(prob_exp_actual_conc[Scores_conc==0])
            
            
            dists_task=reldist_dic[mouse][task_num-1]
            prob_exp_actual=prob_exp_actual[:len(dists_task)]
            dists_task=dists_task[:len(prob_exp_actual)]
            xy=column_stack_clean(prob_exp_actual,dists_task)
            if len(xy)>2:
                corr_dist_pr=st.pearsonr(xy[:,0],xy[:,1])[0]
            else:
                corr_dist_pr=np.nan
                

            prob_corr_incorr_all[task_ind]=prob_exp_actual_corr,prob_exp_actual_incorr
            corr_dist_pr_all[task_ind]=corr_dist_pr
            xy_corr_dist_pr_all.append(xy)
        
        prob_corr_incorr_dic[step_no][mouse]=prob_corr_incorr_all
        corr_dist_pr_dic[step_no][mouse]=corr_dist_pr_all
        corr_dist_pr_dic['xy'][step_no][mouse]=xy_corr_dist_pr_all


# In[ ]:





# In[262]:


all_trials_allstepno=[]
for step_no in np.arange(4)+1:
    all_trials=np.vstack(([np.vstack((dict_to_array(corr_dist_pr_dic['xy'][step_no])[ii]))                           for ii in range(len(Mice))]))
    all_trials_allstepno.append(all_trials)
    xy=column_stack_clean(all_trials[:,0],all_trials[:,1])
    #sns.regplot(xy[:,0],xy[:,1])
    plt.show()
    print(st.pearsonr(xy[:,0],xy[:,1]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[390]:


corr_dist_pr_N_mean_all=[]
for step_no in np.arange(4)+1:
    print(step_no)
    corr_dist_pr_N=dict_to_array(corr_dist_pr_dic[step_no])#[mouse]
    corr_dist_pr_N_mean=np.asarray([[np.nanmean(corr_dist_pr_N[ii]) for ii in range(len(corr_dist_pr_N))]])
    corr_dist_pr_N_mean_all.append(corr_dist_pr_N_mean)
    bar_plotX(corr_dist_pr_N_mean,'none',-0.2,0.5,'points','paired',0.025)
    plt.show()

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False



corr_dist_pr_N_mean_all_mean_=np.nanmean(np.vstack((corr_dist_pr_N_mean_all)),axis=0)
corr_dist_pr_N_mean_all_mean=[corr_dist_pr_N_mean_all_mean_,corr_dist_pr_N_mean_all_mean_]
bar_plotX(corr_dist_pr_N_mean_all_mean,'none',-0.1,0.2,'points','unpaired',0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Distance_vs_baseline_pr.svg')
plt.show()
st.ttest_1samp(corr_dist_pr_N_mean_all_mean_,0)


# In[467]:


del(structure_dic['me05']['ABCD'][40])


# In[468]:


Num_trials_all_dic=rec_dd()
Num_trials_mean_dic=rec_dd()
for mouse in Mice:
    tasks_=list(structure_dic[mouse]['ABCD'].keys())
    if len(tasks_)>40:
        tasks_=tasks_[:40]
    num_trials_all=np.zeros(len(tasks_))
    num_trials_all[:]=np.nan
    num_trials_mean_all=np.zeros(len(tasks_))
    num_trials_mean_all[:]=np.nan
    for task_ind,task_num in enumerate(tasks_):
        num_trials=np.sum(Num_trials_dic[mouse][task_num])
        num_trials_mean=np.mean(Num_trials_dic[mouse][task_num])
        num_trials_all[task_ind]=num_trials
        num_trials_mean_all[task_ind]=num_trials_mean
    Num_trials_all_dic[mouse]=num_trials_all
    Num_trials_mean_dic[mouse]=num_trials_mean_all


# In[ ]:





# In[469]:


Num_trials_all_=dict_to_array(Num_trials_all_dic)
first_10_tasks=[Num_trials_all_[ii][:10] for ii in range(len(Num_trials_all_))]
print('Number of trials per task - first 10')
print(np.nanmean(first_10_tasks))
print(st.sem(np.hstack((first_10_tasks))))


tasks_11_onwards=[Num_trials_all_[ii][10:] for ii in np.arange(len(Num_trials_all_)-4)+4]
print('Number of trials per task - 11 onwards (3 task days)')
print(np.nanmean(tasks_11_onwards))
print(st.sem(np.hstack((tasks_11_onwards))))


# In[446]:


Num_trials_mean_=dict_to_array(Num_trials_mean_dic)
first_10_tasks=[Num_trials_mean_[ii][:10] for ii in range(len(Num_trials_mean_))]
print('Number of trials per task - first 10')
print(np.nanmean(first_10_tasks))
print(st.sem(np.hstack((first_10_tasks))))


tasks_11_onwards=[Num_trials_mean_[ii][10:] for ii in np.arange(len(Num_trials_mean_)-4)+4]
print('Number of trials per task - 11 onwards (3 task days)')
print(np.nanmean(tasks_11_onwards))
print(st.sem(np.hstack((tasks_11_onwards))))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[268]:


for step_no in np.arange(4)+1:
    print(step_no)
    prob_corr_incorr_all_=dict_to_array(prob_corr_incorr_dic[step_no])
    corr_incorr_pr=np.vstack(([np.nanmean(prob_corr_incorr_all_[ii],axis=0) for ii in range(len(prob_corr_incorr_all_))]))

    bar_plotX(corr_incorr_pr.T,'none',0,0.5,'points','paired',0.025)
    plt.show()
    plot_scatter(corr_incorr_pr[:,0],corr_incorr_pr[:,1])
    plt.show()
    print(st.wilcoxon(corr_incorr_pr[:,0],corr_incorr_pr[:,1]))


# In[ ]:





# In[ ]:





# In[484]:


###Initial performance calculation - tone no tone
performance_notone_dic=rec_dd()
reldist_notone_dic=rec_dd()
num_trials_first=20
abstract_structure='ABCD'
num_states=len(abstract_structure)
for mouse in Mice:
    
    cohort=Mice_cohort_dic[mouse]
    if cohort!=4 or mouse=='ah05':
        continue
    print(mouse)
    
    abstract_structure_=Variable_dic[mouse]['Structure_abstract']
    structure_no_=Variable_dic[mouse]['Structure_no']
    structure_=Variable_dic[mouse]['Structure']
    omitted_tone_bool=np.hstack((['ot' in structure_[ii] for ii in range(len(structure_))]))
    ABCD_bool=abstract_structure_=='ABCD'
    no_tone_structures=np.unique((structure_no_[np.logical_and(omitted_tone_bool,ABCD_bool)]).astype(int))
    
    
    #if cohort<=2:
    #    num_tasks_mouse=10
    #else:
    #    num_tasks_mouse=40

    #excluded_tasks=exclusions_dic[mouse]
    num_tasks_mouse=len(no_tone_structures)

    performance_array=np.zeros((num_tasks_mouse,num_trials_first,num_states))
    performance_array[:]=np.nan
    
    relative_distance_array=np.zeros((num_tasks_mouse,num_trials_first,num_states))
    relative_distance_array[:]=np.nan
    
    performance_array_tone=[]
    performance_array_notone=[]
    relative_distance_array_tone=[]
    relative_distance_array_notone=[]
    for task_num_ind, task_num in enumerate(no_tone_structures):
        print(task_num)
        try:
            scores__=dict_to_array(scores_dic[mouse][task_num]['ALL'])
            full_sessions_scores=np.hstack(([ii for ii in range(len(scores__)) if len(scores__[ii])>0]))
            num_full_sessions_scores=len(full_sessions_scores)
            sessions=np.sort(np.hstack((list(scores_dic[mouse][task_num]['ALL'].keys())))[list(full_sessions_scores)])
            Num_trials_allses=Num_trials_dic[abstract_structure][mouse][task_num]
            num_Num_trials_allses=len(Num_trials_allses)

            scores_all=[]
            tone_trial_bool_all=[]
            for session_ind, session in enumerate(sessions):
                if session_ind+1>num_Num_trials_allses:
                    print('Note: mismatch between num sessions in Num_trials_dic and scores_dic')
                    continue
                scores_session=scores_dic[mouse][task_num]['ALL'][session]
                if len(scores_session)>0:
                    scores_all.append(scores_session)
                else:
                    continue
                    
                    
                ###finding tone and no tone trials
                struc_=times_dic[mouse][abstract_structure][task_num]
                session_struc=struc_[session]

                if 'A_on_first' in session_struc.keys():
                    trial_starts=np.hstack((session_struc['A_on_first'],session_struc['A_on']))
                else:
                    trial_starts=session_struc['A_on']

                tone_times=session_struc['tone']

                tone_times_=((tone_times-10) +9) // 10 *10
                trial_starts_=(trial_starts+9) // 10 *10
                trial_starts_minus10_=trial_starts_-10
                trial_starts_plus10_=trial_starts_+10
                tone_trial_bool_=np.in1d(trial_starts_,tone_times_)
                tone_trial_minus10_bool=np.in1d(trial_starts_minus10_,tone_times_)
                tone_trial_plus10_bool=np.in1d(trial_starts_plus10_,tone_times_)
                tone_trial_bool=np.logical_or(np.logical_or(tone_trial_bool_,tone_trial_minus10_bool),                                              tone_trial_plus10_bool)
                
                tone_trial_bool_all.append(tone_trial_bool[:len(scores_session)])

                if np.sum(tone_trial_bool)!=len(tone_times_):
                    print('Not all tone times found')

                print(np.max(trial_starts[tone_trial_bool]-tone_times))
                print(np.min(trial_starts[tone_trial_bool]-tone_times))

            scores_all=np.vstack((scores_all))
            tone_trial_bool_all=np.hstack((tone_trial_bool_all))
            
            scores_tone=scores_all[tone_trial_bool_all==True]
            scores_no_tone=scores_all[tone_trial_bool_all==False]
            
            scores_N=scores_all[:num_trials_first]
            if len(scores_N)<num_trials_first:
                added_array=np.zeros((num_trials_first-len(scores_N),num_states))
                added_array[:]=np.nan
                scores_N=np.vstack((scores_N,added_array))
            performance_array[task_num_ind]=scores_N
            
            performance_array_tone.append(scores_tone)
            performance_array_notone.append(scores_no_tone)
            
            
            
            
            ###distances
            dists_task_=dict_to_array(scores_dic['dists'][mouse][task_num]['ALL'])
            if len(dists_task_)==1:
                dists_task=np.vstack((dists_task_))
            else:
                if len(dists_task_[1])==0:
                    dists_task=dists_task_[0]
                else:
                    dists_task=np.vstack((remove_empty(dists_task_)))
            
            min_dists_task=dict_to_array(scores_dic['mindist'][mouse][task_num]['ALL'])[0]
            
            min_dists_task_sum=np.sum(min_dists_task)
            if min_dists_task_sum==0:
                min_dists_task_sum=np.nan
            #relative_dists_task_all=np.sum(dists_task,axis=1)/min_dists_task_sum
            relative_dists_task_all=dists_task/min_dists_task
            relative_dists_task=relative_dists_task_all[:num_trials_first]
            if len(relative_dists_task)<num_trials_first:
                added_array=np.zeros((num_trials_first-len(relative_dists_task),num_states))
                added_array[:]=np.nan
                relative_dists_task=np.vstack((relative_dists_task,added_array))
            
            relative_distance_array[task_num_ind]=relative_dists_task
            
            relative_dists_task_tone=relative_dists_task_all[tone_trial_bool_all==True]
            relative_dists_task_no_tone=relative_dists_task_all[tone_trial_bool_all==False]
            
            relative_distance_array_tone.append(relative_dists_task_tone)
            relative_distance_array_notone.append(relative_dists_task_no_tone)
            
            
            
        except Exception as e:
            print('Not calculated')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    performance_notone_dic[mouse]=performance_array
    reldist_notone_dic[mouse]=relative_distance_array
    
    
    performance_notone_dic['tone'][mouse]=performance_array_tone
    reldist_notone_dic['tone'][mouse]=relative_distance_array_tone
    
    performance_notone_dic['no_tone'][mouse]=performance_array_notone
    reldist_notone_dic['no_tone'][mouse]=relative_distance_array_notone




# In[20]:


performance_notone_dic['tone']['me10']


# In[ ]:





# In[15]:


###All performance (excluding first trial)
means_perf_allmice_tone=np.hstack(([[np.nanmean(performance_notone_dic['tone'][mouse][session][1:])             for session in range(len(performance_notone_dic['tone'][mouse]))] for mouse in performance_notone_dic['tone'].keys()]))

means_perf_allmice_notone=np.hstack(([[np.nanmean(performance_notone_dic['no_tone'][mouse][session][1:])             for session in range(len(performance_notone_dic['no_tone'][mouse]))] for mouse in performance_notone_dic['no_tone'].keys()]))

plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

bar_plotX([means_perf_allmice_tone,means_perf_allmice_notone], 'none', 0, 1, 'points', 'paired', 0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_All_proportioncorrect.svg')
plt.show()
xy=column_stack_clean(means_perf_allmice_tone,means_perf_allmice_notone)
print(st.wilcoxon(xy[:,0],xy[:,1]))


plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
xy=column_stack_clean(means_perf_allmice_tone,means_perf_allmice_notone)
noplot_scatter(xy[:,0],xy[:,1],'black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_All_proportioncorrect_scatter.svg')
plt.show()
xy=column_stack_clean(means_perf_allmice_tone,means_perf_allmice_notone)
print(st.wilcoxon(xy[:,0],xy[:,1]))

means_dist_allmice_tone=np.hstack(([[np.nanmean(reldist_notone_dic['tone'][mouse][session][1:])             for session in range(len(reldist_notone_dic['tone'][mouse]))] for mouse in reldist_notone_dic['tone'].keys()]))

means_dist_allmice_notone=np.hstack(([[np.nanmean(reldist_notone_dic['no_tone'][mouse][session][1:])             for session in range(len(reldist_notone_dic['no_tone'][mouse]))] for mouse in reldist_notone_dic['no_tone'].keys()]))

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
bar_plotX([means_dist_allmice_tone,means_dist_allmice_notone], 'none', 0, 7, 'points', 'paired', 0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_All_relativedistance.svg')

plt.show()
xy=column_stack_clean(means_dist_allmice_tone,means_dist_allmice_notone)
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
xy=column_stack_clean(means_dist_allmice_tone,means_dist_allmice_notone)
noplot_scatter(xy[:,0],xy[:,1],'black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_All_relativedistance_scatter.svg')
plt.show()
xy=column_stack_clean(means_dist_allmice_tone,means_dist_allmice_notone)
print(st.wilcoxon(xy[:,0],xy[:,1]))


# In[ ]:





# In[17]:


###All performance (excluding first trial)
means_perf_allmice_tone=np.hstack(([[np.nanmean(performance_notone_dic['tone'][mouse][session][1:])             for session in range(len(performance_notone_dic['tone'][mouse]))] for mouse in performance_notone_dic['tone'].keys()]))

means_perf_allmice_notone=np.hstack(([[np.nanmean(performance_notone_dic['no_tone'][mouse][session][1:])             for session in range(len(performance_notone_dic['no_tone'][mouse]))] for mouse in performance_notone_dic['no_tone'].keys()]))

plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

bar_plotX([means_perf_allmice_tone,means_perf_allmice_notone], 'none', 0, 1, 'points', 'paired', 0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_All_proportioncorrect.svg')
plt.show()
xy=column_stack_clean(means_perf_allmice_tone,means_perf_allmice_notone)
print(st.wilcoxon(xy[:,0],xy[:,1]))


plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
xy=column_stack_clean(means_perf_allmice_tone,means_perf_allmice_notone)
noplot_scatter(xy[:,0],xy[:,1],'black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_All_proportioncorrect_scatter.svg')
plt.show()
xy=column_stack_clean(means_perf_allmice_tone,means_perf_allmice_notone)
print(len(xy))
print(st.wilcoxon(xy[:,0],xy[:,1]))

means_dist_allmice_tone=np.hstack(([[np.nanmean(reldist_notone_dic['tone'][mouse][session][1:])             for session in range(len(reldist_notone_dic['tone'][mouse]))] for mouse in reldist_notone_dic['tone'].keys()]))

means_dist_allmice_notone=np.hstack(([[np.nanmean(reldist_notone_dic['no_tone'][mouse][session][1:])             for session in range(len(reldist_notone_dic['no_tone'][mouse]))] for mouse in reldist_notone_dic['no_tone'].keys()]))

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
bar_plotX([means_dist_allmice_tone,means_dist_allmice_notone], 'none', 0, 7, 'points', 'paired', 0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_All_relativedistance.svg')

plt.show()
xy=column_stack_clean(means_dist_allmice_tone,means_dist_allmice_notone)
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
xy=column_stack_clean(means_dist_allmice_tone,means_dist_allmice_notone)
noplot_scatter(xy[:,0],xy[:,1],'black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_All_relativedistance_scatter.svg')
plt.show()
xy=column_stack_clean(means_dist_allmice_tone,means_dist_allmice_notone)
print(len(xy))
print(st.wilcoxon(xy[:,0],xy[:,1]))


# In[483]:


###DA performance (excluding first trial)
means_perf_allmice_tone=np.hstack(([[np.nanmean(performance_notone_dic['tone'][mouse][session][1:,3])             for session in range(len(performance_notone_dic['tone'][mouse]))] for mouse in performance_notone_dic['tone'].keys()]))

means_perf_allmice_notone=np.hstack(([[np.nanmean(performance_notone_dic['no_tone'][mouse][session][1:,3])             for session in range(len(performance_notone_dic['no_tone'][mouse]))] for mouse in performance_notone_dic['no_tone'].keys()]))
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
bar_plotX([means_perf_allmice_tone,means_perf_allmice_notone], 'none', 0, 1, 'points', 'paired', 0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_DA_proportioncorrect.svg')
plt.show()
xy=column_stack_clean(means_perf_allmice_tone,means_perf_allmice_notone)


plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
xy=column_stack_clean(means_perf_allmice_tone,means_perf_allmice_notone)
noplot_scatter(xy[:,0],xy[:,1],'black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_DA_proportioncorrect_scatter.svg')
plt.show()
xy=column_stack_clean(means_perf_allmice_tone,means_perf_allmice_notone)
print(len(xy))
print(st.wilcoxon(xy[:,0],xy[:,1]))

means_dist_allmice_tone=np.hstack(([[np.nanmean(reldist_notone_dic['tone'][mouse][session][1:,3])             for session in range(len(reldist_notone_dic['tone'][mouse]))] for mouse in reldist_notone_dic['tone'].keys()]))

means_dist_allmice_notone=np.hstack(([[np.nanmean(reldist_notone_dic['no_tone'][mouse][session][1:,3])             for session in range(len(reldist_notone_dic['no_tone'][mouse]))] for mouse in reldist_notone_dic['no_tone'].keys()]))
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
bar_plotX([means_dist_allmice_tone,means_dist_allmice_notone], 'none', 0, 25, 'points', 'paired', 0.025)
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_DA_relativedistance.svg')
plt.show()

plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
xy=column_stack_clean(means_dist_allmice_tone,means_dist_allmice_notone)
noplot_scatter(xy[:,0],xy[:,1],'black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'Tone_notone_DA_relativedistance_scatter.svg')
plt.show()
xy=column_stack_clean(means_dist_allmice_tone,means_dist_allmice_notone)
print(len(xy))
print(st.wilcoxon(xy[:,0],xy[:,1]))


# In[ ]:





# In[ ]:





# ###fixing scores_dic indices
# for mouse in Mice:
#     print('')
#     print(mouse)
#     tasks_all=np.hstack((list(times_dic[mouse][abstract_structure].keys())))
#     for task_num_ind, task_num in enumerate(tasks_all):
#         times_task_=times_dic[mouse][abstract_structure][task_num]
#         times_task_array=dict_to_array(times_task_)
#         if len(times_task_array)==0:
#             print('No times_dic for task'+str(task_num))
#             continue
#         non_zero_bool=np.hstack(([len(times_task_array[ii])>0 for ii in range(len(times_task_array))]))
#         sessions=np.sort(np.asarray(list(times_task_.keys()))[non_zero_bool])
#         
#         sessions_zero=np.sort(np.asarray(list(times_task_.keys()))[~non_zero_bool])
#         for session_zero in sessions_zero:
#             del(times_dic[mouse][abstract_structure][task_num][session_zero])
#         
#         dict_all_=scores_dic[mouse][task_num]['ALL']
#         scores_array_=dict_to_array(dict_all_)
#         
#         if len(list(dict_all_.keys()))==0:
#             print('Nothing in Scores dic for task'+str(task_num))
#             continue
#         session_keys_old=np.hstack((list(dict_all_.keys())))
#         non_zero_bool=np.hstack(([len(scores_array_[ii])>0 for ii in np.arange(len(scores_array_))]))
# 
#         session_keys_old_non_zero=np.sort(session_keys_old[non_zero_bool])
#         
#         session_keys_old_sorted=np.sort(session_keys_old)
#         
#         if len(session_keys_old_sorted)!=len(sessions):
#             print(len(sessions)-len(session_keys_old_non_zero)) ### positive means more sessions in times_dic
#             print('NOT CHANGED: Mismatch between sessions in Scores_dic and times_dic for task'+str(task_num))
#             continue
#         if len(session_keys_old_sorted)>0:
#             for session_ind, session in enumerate(sessions):
#                 old_session=session_keys_old_sorted[session_ind]
#                 
#                 for name in list(scores_dic[mouse][task_num].keys()):
#                     scores_dic[mouse][task_num][name][session]=\
#                     scores_dic[mouse][task_num][name][old_session]
#                     
#                 for name2 in list(scores_dic.keys()):
#                     scores_dic[name2][mouse][task_num][session]=\
#                     scores_dic[name2][mouse][task_num][old_session]
#                     
# 
#         non_overlap_sessions=np.setdiff1d(session_keys_old,sessions)
#         for session_notused in non_overlap_sessions:
#             for name in list(scores_dic[mouse][task_num].keys()):
#                 try:
#                     del(scores_dic[mouse][task_num][name][session_notused])
#                 except:
#                     x=1
#             for name2 in list(scores_dic.keys()):
#                 try:
#                     del(scores_dic[name2][mouse][task_num]['ALL'][session_notused])
#                 except:
#                     x=1
#                 
#             
#             
#             
#             

# In[ ]:





# In[ ]:





# In[477]:


###ABCDE performance
Tasknum_used_dic=rec_dd()
for mouse_ in ['ab03','ah07']:
    total_task_num_used_start=0
    for mouse_recday in rec_days_:
        
        mouse=mouse_recday.split('_',1)[0]
        
        if mouse!=mouse_:
            continue
        print(mouse_recday)
        
            
        Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)
        non_repeat_ses=non_repeat_ses_maker(mouse_recday)
        
        num_states=np.hstack(([len(Task)for Task  in Tasks]))
        non_repeat_ses_ABCDE=np.intersect1d(non_repeat_ses,np.where(num_states==5)[0])
        
        total_task_num_used=total_task_num_used_start+np.arange(len(non_repeat_ses_ABCDE))
        total_task_num_used_start=np.max(total_task_num_used)+1
        
        Tasknum_used_dic[mouse_recday]=total_task_num_used


day_type='combined_ABCDE'
rec_days_=day_type_dicX[day_type]
num_nodes=9

Scores_ABCDE_dic=rec_dd()
behbias_mat_ABCDE_dic=rec_dd()
for mouse_recday in rec_days_:
    print(mouse_recday)
    mouse=mouse_recday.split('_',1)[0]
    Tasks=np.load(Intermediate_object_folder_dropbox+'Task_data_'+mouse_recday+'.npy',allow_pickle=True)
    non_repeat_ses=non_repeat_ses_maker(mouse_recday)
    num_states=np.hstack(([len(Task)for Task  in Tasks]))
    non_repeat_ses_ABCDE=np.intersect1d(non_repeat_ses,np.where(num_states==5)[0])
    for ses_ind_ind,ses_ind in enumerate(non_repeat_ses_ABCDE):
        Task=Tasks[ses_ind]
        
        task_num_used=Tasknum_used_dic[mouse_recday][ses_ind_ind]

        num_states=len(Task)


        trial_times=np.load(Intermediate_object_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')//25
        Location=np.load(Intermediate_object_folder+'Location_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
        Nodes=np.copy(Location)
        Nodes[Nodes>num_nodes]=np.nan

        minimum_distances=[mindistance_mat[Task[state_ind]-1,np.roll(Task,-1)[state_ind]-1]        for state_ind in np.arange(num_states)]
        
        paths=np.vstack(([[unique_adjacent(remove_nan(Nodes[trial_times            [trial_ind,state_ind]:trial_times[trial_ind,state_ind+1]]))            for state_ind in np.arange(num_states)] for trial_ind in np.arange(len(trial_times))]))

        path_lengths=np.vstack(([[len(unique_adjacent(remove_nan(Nodes[trial_times            [trial_ind,state_ind]:trial_times[trial_ind,state_ind+1]]))[1:])            for state_ind in np.arange(num_states)] for trial_ind in np.arange(len(trial_times))]))
        relative_distances=(path_lengths/minimum_distances)
        mean_relative_distances=np.nanmean(relative_distances,axis=1)
        
        mean_relative_distances_20=np.hstack((mean_relative_distances[:20],                                                  np.repeat(np.nan,20-len(mean_relative_distances[:20]))))
        
        meanofmeans_relative_distances_20=np.nanmean(mean_relative_distances_20)
        Prop_correct=np.sum(np.concatenate(relative_distances[:20])==1)/len(np.concatenate(relative_distances[:20]))
        
        
        
        Scores_ABCDE_dic['Mean'][mouse][task_num_used]=mean_relative_distances_20
        Scores_ABCDE_dic['Mean_of_means'][mouse][task_num_used]=meanofmeans_relative_distances_20
        Scores_ABCDE_dic['Prop_correct'][mouse][task_num_used]=Prop_correct
        
        ###Chance levels
        Probs=[]
        for state_ind in np.arange(num_states):
            portX=Task[state_ind]
            portY=np.roll(Task,-1)[state_ind]
            min_dist=mindistance_mat[portX-1,portY-1]
            trans_mat_mean=np.load(Intermediate_object_folder_dropbox+'/'+                                   mouse+'_Exploration_Transition_matrix.npy')
            prob=trans_mat_mean[min_dist-1][portX-1,portY-1]
            Probs.append(prob)
            
        Probs=np.hstack(Probs)

        Scores_ABCDE_dic['beh_bias'][mouse][task_num_used]=np.nanmean(Probs)


# In[476]:





# In[ ]:





# In[444]:


total_scores_ABCDE=flatten(Scores_ABCDE_dic['Mean_of_means'])

rel_dist_first_=np.asarray([reldist_dic[mouse][:,0] for mouse in Mice])
factor=5
bins=np.arange(len(rel_dist_first.T)//factor+1)*factor
rel_dist_first=np.vstack(([np.hstack((rel_dist_first_[ii],np.repeat(np.nan,int(40-len(rel_dist_first_[ii]))))) for ii in range(len(rel_dist_first_))]))
rel_dist_first_binned=np.vstack(([st.binned_statistic(np.arange(len(rel_dist_first.T)),rel_dist_first[ii],                            bins=bins,statistic=np.nanmean)[0] for ii in np.arange(len(rel_dist_first))]))
chance=np.nanmean(rel_dist_first_binned,axis=0)[0]


plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

bar_plotX(np.asarray([total_scores_ABCDE]),'none', 0, 9, 'points', 'unpaired',0.025)
plt.axhline(chance,ls='dashed',color='grey')
plt.axhline(1,ls='dashed',color='grey')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)

plt.savefig(Behaviour_output_folder_dropbox+'ABCDE_relativedistance.svg')
plt.show()
print(len(total_scores_ABCDE))
print(st.ttest_1samp(total_scores_ABCDE,chance))


# In[485]:


Prop_correct=flatten(Scores_ABCDE_dic['Prop_correct'])
Chance_prop_correct=flatten(Scores_ABCDE_dic['beh_bias'])

plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
xy=column_stack_clean(Prop_correct,Chance_prop_correct)
noplot_scatter(xy[:,0],xy[:,1],'black')
plt.tick_params(axis='both',  labelsize=20)
plt.tick_params(width=2, length=6)
plt.savefig(Behaviour_output_folder_dropbox+'ABCDE_propcorrect.svg')
plt.show()
print(len(xy))
print(st.wilcoxon(xy[:,0],xy[:,1]))


# In[481]:





# In[604]:


Mice


# In[ ]:





# In[31]:


###Example trajectories

mouse_recday='me11_01122021_02122021'
ses_ind=2

xy=np.load(Intermediate_object_folder+'XY_raw_'+mouse_recday+'_'+str(ses_ind)+'.npy')
trial_times=np.load(Intermediate_object_folder+'trialtimes_'+mouse_recday+'_'+str(ses_ind)+'.npy')//25
Tasks=np.load(Intermediate_object_folder+'Task_data_'+mouse_recday+'.npy')
Tasks


# In[ ]:





# In[32]:


print(Tasks[ses_ind])

thr=10
xy_clean=xy#[xy[:,0]>1]
x_change=np.hstack((0,abs(np.diff(xy[:,0]))))
y_change=np.hstack((0,abs(np.diff(xy[:,1]))))
xy_clean=xy_clean[np.logical_and(x_change<np.mean(x_change)+np.std(x_change)*thr,                                 y_change<np.mean(y_change)+np.std(y_change)*thr)]





xy[np.logical_or(x_change>np.mean(x_change)+np.std(x_change)*thr,                                 y_change>np.mean(y_change)+np.std(y_change)*thr)]=np.nan

xy[xy[:,0]<=100]=[np.nan,np.nan]
xy_clean[xy_clean[:,0]<=100]=[np.nan,np.nan]
xy[xy[:,0]>=1100]=[np.nan,np.nan]
xy_clean[xy_clean[:,0]>=1100]=[np.nan,np.nan]


plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False

colors=['midnightblue','purple','mediumvioletred','palevioletred']
num_trials=5
start_trial=9
fig, axs = plt.subplots(1, num_trials, figsize=(15, 3), sharey=True)
for trial_ind, trial_no in enumerate(np.arange(num_trials)+start_trial):
    print(trial_no)
    for state in np.arange(4):
        traj_trial=xy[trial_times[trial_no,state]:trial_times[trial_no,state+1]]
        axs[trial_ind].plot(traj_trial[:,0],traj_trial[:,1],color=colors[state],linewidth=3)
    axs[trial_ind].plot(xy_clean[:,0],xy_clean[:,1],color='grey',alpha=0.3)
    

    
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.savefig(Behaviour_output_folder_dropbox+'Exampletrajectories'+mouse_recday+'_ses'+str(ses_ind)+'trial_'+            str(start_trial)+'_'+str(start_trial+num_trials-1)+'.png', bbox_inches = 'tight', pad_inches = 0)
plt.show()


# In[ ]:





# In[ ]:





# In[442]:


'''
used in Figure 1:
ah03_18082021_19082021 - ses6 - trials 26-30 - task: 8312

ah04_05122021_06122021 - ses0 - trials 22-26 - task: 5189

me11_01122021_02122021 - ses2 - trials 9-13 - task: 4627



Others:
ah03_18082021_19082021 - ses2 - trials 30-34 - task: 6125


me08_10092021_11092021 - ses1 - trials 11-15 - task: 9641

ah04_05122021_06122021 - ses4 - trials 21-25 - task: 9387

-

'''


# In[ ]:




