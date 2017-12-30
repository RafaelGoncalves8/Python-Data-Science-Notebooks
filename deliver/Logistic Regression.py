
# coding: utf-8

# In[2]:


get_ipython().magic('matplotlib inline')

import random
import numpy as np
import matplotlib.pyplot as plt


# Logistic regression is a supervised learning (has labeled data as input) algorithm for classification. The algorithm is fed with samples - with multiple features and it's corresponding classes (labels).

# - $M$ - Number of samples
# - $N$ - Number of features
# - $\boldsymbol{\theta}$ - Vector of minimizing parameters
# - $\mathbf{X}$ - Matrix of features (columns) per samples (rows)
# - $\mathbf{y}$ - Vector of corresponding labels

# In[33]:


M = 100
M_half = 50
N = 2

X1 = np.array([[random.gauss(2,1.25), random.gauss(3,1.25)] for i in range(M_half)])
y1 = np.array((M_half)*[1])

X2 = np.array([[random.gauss(0,1.25), random.gauss(-1,1.25)] for i in range(M_half)])
y2 = np.array((M_half)*[0])

X = np.vstack([X1, X2])
y = np.hstack([y1.T, y2.T])

print("    x1    |     x2    |     y    ")
for i in range(M):
    print("%9.4f | %9.4f | %9.4f" % (X[i,0], X[i,1], y[i]))


# In[34]:


plt.plot(X[:M_half,0], X[:M_half,1], 'x'), plt.xlabel('x0'), plt.ylabel('x1')
plt.plot(X[M_half:,0], X[M_half:,1], 'o'), plt.xlabel('x0'), plt.ylabel('x1')
plt.show()


# In[ ]:




