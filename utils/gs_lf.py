#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from argparse import Namespace
# import yaml
import os
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
print(os.getcwd())
# import torch
# from fast_transformers.masking import LengthMask as LM
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
import numpy as np


# In[113]:


def pom_get_dataIds_by_tasks(df, task_indices):
    #returns datapoints associated with a specific task 
    # y = np.array(y)
    task_indices = np.array(task_indices)
    selected_df   = df[df.y.apply(lambda arr: any(arr[i] == 1 for i in task_indices))]

    return selected_df


# In[178]:


def pom_label_by_index(ys,desired_indices,TASKS):
    labelss =[]
    # print("ys",ys)

    for label in ys:
        labels  = np.where(label)[0]
        # print("labelssss",labels)
        mask = np.isin(labels, desired_indices)
        # print(mask)
        result = labels[mask]
        # print(result)
        labelss.append(TASKS[result[0]])
        
    return labelss


# In[176]:


def pom_subset_by_tasks(df,tasks,TASKS):
    tasks_idx =pom_task_index_by_label(tasks,TASKS)
    subset_df = pom_get_dataIds_by_tasks(df, tasks_idx)
    # print("task_idx",tasks_idx)
    # subset_df = df(ids)
    subset_dataset=pom_keep_only_label(subset_df,tasks_idx)
    return subset_dataset


# In[116]:


def pom_onehot_to_array(row):
    return np.array(row)


# In[117]:


def pom_task_index_by_label(labels,TASKS):
    #returns index associated with each label in TASKS array
    # print(labels)
    # print(TASKS)
    common_labels = np.intersect1d(TASKS, labels)
    indices = np.where(np.isin(TASKS, common_labels))[0]
    result = [TASKS[i] for i in indices]
    sorted_indices = np.argsort([labels.index(task) for task in result])
    return indices[sorted_indices]


# In[187]:


def pom_keep_only_label(subset_dataset, task_indices):
    zero_mask = subset_dataset.copy()  # Create a copy of the input DataFrame
    zero_mask['y'] = zero_mask['y'].apply(lambda arr: [1 if idx in task_indices and arr[idx] else 0 for idx in range(len(arr))])
    return zero_mask

