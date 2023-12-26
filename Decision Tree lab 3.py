#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import DataFrame 


# In[2]:



import os
for dirname, _, filenames in os.walk(r'A:\MTECH(Data Science)\DataSet\Machin learing Lab\3\lab3.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


datasetValue = pd.read_csv(r'A:\MTECH(Data Science)\DataSet\Machin learing Lab\3\lab3.csv')


# In[8]:


#data set shown
datasetValue


# In[9]:


#Optional
datasetValue.info()


# In[10]:


#Optional
datasetValue.shape


# In[11]:


a = np.array(datasetValue)
print(a)


# In[16]:


#Calculating Entropy of Whole Data-set 

#Function to calculate final Entropy 

def entropy(probs):  
    import math
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

#Function to calculate Probabilities of positive and negative examples 

def entropy_of_list(a_list):
    from collections import Counter
    cnt = Counter(x for x in a_list) #Count the positive and negative ex
    num_instances = len(a_list)
    
    #Calculate the probabilities that we required for our entropy formula 
    probs = [x / num_instances for x in cnt.values()] 
    
    #Calling entropy function for final entropy 
    return entropy(probs)

total_entropy = entropy_of_list(datasetValue['Temperature'])
print("\n Total Entropy of  Data Set:",total_entropy)


# In[19]:


#Calculate Information Gain for each Attribute 
#Defining Information Gain Function 

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    print("Information Gain Calculation of ",split_attribute_name)
    print("target_attribute_name",target_attribute_name)

    #Grouping features of Current Attribute
    data_1 = pd.DataFrame(df)
    #print(data_1)
    df_split = data_1.groupby(split_attribute_name)
    #print(df_split)
    for name,group in df_split:
        print("Name: ",name)
        print("Group: ",group)
        nobs = len(df.index) * 1.0
       # print("NOBS",nobs)

    #Calculating Entropy of the Attribute and probability part of formula 
        df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
        df_agg_ent.columns = ['Entropy', 'PropObservations']
        print("df_agg_ent",df_agg_ent)

    # Calculate Information Gain
        avg_info = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
        old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - avg_info

#print('Info-gain for Outlook is :'+str(information_gain(data, 'Outlook', 'PlayTennis')),"\n")


# In[22]:


#Defining ID3  Algorithm Function

def id3(df, target_attribute_name, attribute_names, default_class=None):

    #Counting Total number of yes and no classes (Positive and negative Ex)
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])
    if len(cnt) == 1:
        return next(iter(cnt))
        # Return None for Empty Data Set 
    elif df.empty or (not attribute_names):
            return default_class
    else:
        default_class = max(cnt.keys())

    print("attribute_names:",attribute_names)
    gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names] 
    
    #Separating the maximum information gain attribute after calculating the information gain 
    index_of_max = gainz.index(max(gainz)) #Index of Best Attribute 
    best_attr = attribute_names[index_of_max] #choosing best attribute 

    #The tree is initially an empty dictionary
    tree = {best_attr:{}} # Initiate the tree with best attribute as a node 
    remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        
    for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,target_attribute_name,remaining_attribute_names,default_class)
            tree[best_attr][attr_val] = subtree
    return tree


# In[23]:


# Get Predictor Names (all but 'class')

attribute_names = list(datasetValue.columns)
print("List of Attributes:", attribute_names) 
attribute_names.remove('PlayTennis')

#Remove the class attribute 
print("Predicting Attributes:", attribute_names)


# In[24]:


# Run Algorithm (Calling ID3 function)

from pprint import pprint
tree = id3(datasetValue,'PlayTennis',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
pprint(tree)
attribute = next(iter(tree))
print("Best Attribute :\n",attribute)
print("Tree Keys:\n",tree[attribute].keys())


# In[ ]:




