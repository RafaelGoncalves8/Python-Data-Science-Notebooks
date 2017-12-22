
# coding: utf-8

# # Importing Data

# In[6]:


import os
import urllib.request


# In[7]:


os.sys.path.append('../src')
import import_data


# ## Download csv data

# In[11]:


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
data_path = os.path.abspath(os.path.relpath('../data'))
csv_dir = os.path.join(data_path, 'csv/')
csv_path = os.path.join(csv_dir, 'car.data')

if not os.path.isdir(csv_dir):
    os.makedirs(csv_dir)

urllib.request.urlretrieve(data_url, csv_path)


# In[12]:


get_ipython().run_cell_magic('bash', '', '\nls ../data/csv')


# ## Download tgz data

# In[13]:


data_url = 'https://sourceforge.net/projects/weka/files/datasets/regression-datasets/datasets-arie_ben_david.tar.gz/download?use_mirror=ufpr'

import_data.fetch_tgz_data(data_url, 'datasets-arie_ben_david.tar.gz', 'tgz/')


# In[14]:


get_ipython().run_cell_magic('bash', '', '\nls ../data/tgz')


# ## Download zip data

# In[15]:


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'

import_data.fetch_zip_data(data_url, 'bank.zip', 'zip/')


# In[16]:


get_ipython().run_cell_magic('bash', '', '\nls ../data/zip')

