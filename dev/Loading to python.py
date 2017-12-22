
# coding: utf-8

# # Loading into Python

# In[16]:


import os, csv
import numpy as np
import pandas as pd


# In[2]:


dev = os.path.abspath(os.path.dirname('__file__'))
data = os.path.abspath(os.path.relpath('../data/', dev))


# ## CSV

# In[3]:


get_ipython().run_cell_magic('bash', '', '\nls ../data/csv')


# In[4]:


get_ipython().run_cell_magic('bash', '', '\nhead ../data/csv/car.data')


# In[13]:


csv_data = []
with open(os.path.join(data, 'csv/car.data'), 'r') as file:
    csv_reader = csv.reader(file)
    for l in csv_reader:
        csv_data.append(l)


# In[14]:


for e in csv_data[:5]: print(e)


# In[46]:


len(csv_data)


# ## Numpy

# In[51]:


get_ipython().run_cell_magic('bash', '', '\nhead -n 50 ../data/tgz/ERA.arff')


# In[45]:


np_data = np.genfromtxt(os.path.join(data, 'tgz/ERA.arff'), delimiter=',', skip_header=49)
np_data


# In[59]:


np_data.shape


# ## Pandas

# In[20]:


get_ipython().run_cell_magic('bash', '', '\nhead ../data/zip/bank.csv')


# In[56]:


pd_data = pd.read_csv(os.path.join(data, 'zip/bank.csv'), delimiter=";")


# In[57]:


pd_data.head()


# In[60]:


pd_data.info


# In[61]:


pd_data.shape


# In[66]:


pd_data.describe()


# In[67]:


pd_data.info()


# In[ ]:




