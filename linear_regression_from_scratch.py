#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


teams=pd.read_csv("/teams.csv")


# In[7]:


teams


# In[12]:


X=teams[["athletes","prev_medals"]].copy()
Y=teams[["medals"]].copy()


# In[9]:


X


# In[13]:


y


# In[14]:


X['intercept']=1


# In[16]:


X=X[["intercept","athletes","prev_medals"]]


# In[17]:


X


# In[18]:


X_T=X.T


# In[19]:


X_T


# In[20]:


B=np.linalg.inv(X_T @ X)@ X_T @ Y


# In[21]:


B


# In[22]:


B.index=X.columns


# In[24]:


predictions=X@B


# In[25]:


predictions


# In[26]:


SSR=((Y-predictions)**2).sum()


# In[27]:


SST=((Y-Y.mean())**2).sum()


# In[28]:


SSR


# In[29]:


SST


# In[31]:


R2=1-(SSR/SST)


# In[32]:


R2


# In[ ]:




