
# coding: utf-8

# In[1]:


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

# In[2]:


M = 100
M_half = 50
N = 2

X1 = np.array([[random.gauss(2,1.25), random.gauss(3,1.25)] for i in range(M_half)])
y1 = np.array((M_half)*[1])

X2 = np.array([[random.gauss(0,1.25), random.gauss(-1,1.25)] for i in range(M_half)])
y2 = np.array((M_half)*[0])

X = np.vstack([X1, X2])
y = np.hstack([y1.T, y2.T])

y = y.reshape([M,1])

print("    x1    |     x2    |     y    ")
for i in range(M):
    print("%9.4f | %9.4f | %9.4f" % (X[i,0], X[i,1], y[i,0]))


# In[3]:


plt.plot(X[:M_half,0], X[:M_half,1], 'X'), plt.xlabel('x0'), plt.ylabel('x1')
plt.plot(X[M_half:,0], X[M_half:,1], 'o'), plt.xlabel('x0'), plt.ylabel('x1')
plt.show()


# # Feature scaling (rescale and mean normalization)

# One way of feature scaling is to keep the average of each feature as a predefined number (e.g. 0) and scale the values between the range of values. We can define two vectors mu and delta for keeping these numbers:
# 
# $$\mu_n = \frac{1}{M}\sum\limits_{m = 0}^M x_{mn}$$  
# $$\delta_n = max(\mathbf{x_n}) - min(\mathbf{x_n})$$  
# $$x_{mn} \leftarrow \frac{x_{mn} - \mu_{n}}{\delta_n}$$  

# In[4]:


mu = np.array([np.average(X[:,0]), np.average(X[:,1])])
delta = np.array([(np.max(X[:,0]) - np.min(X[:,0])), (np.max(X[:,1]) - np.min(X[:,1]))])

X = (X - np.ones([M,1])*mu)*(np.ones([M,1])*(1/delta))


# In[5]:


plt.plot(X[:M_half,0], X[:M_half,1], 'X'), plt.xlabel('x0'), plt.ylabel('x1')
plt.plot(X[M_half:,0], X[M_half:,1], 'o'), plt.xlabel('x0'), plt.ylabel('x1')
plt.show()


# # Hyphotesis and Cost function

# The objective of the logistic regression is to find a *hyphotesis function*  $\mathbf{h_\theta} = \mathbf{h_\theta (X)}$ - or the parameters of this function $\boldsymbol{\theta}$ - such as the *cost function* $v = v(\mathbf{X}, \mathbf{y}, \boldsymbol{\theta})$ - diference between predictions and labels - is minimized.

# In linear regression, the most general hypothesis was:
# 
# $$\mathbf{h_\theta} = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_Nx_N$$
# 
# If we define an matrix $\mathbf{X'}$ such as the first column (first feature) is always 1, then we can write:
# 
# $$\mathbf{h_\theta} = \sum \limits_{n = 0}^N \theta_nx'_n$$  
# 
# or in vectorized notation:
# 
# $$\mathbf{h_\theta} = \mathbf{X'\theta}$$

# In the case of logistic regression, it is best that the values range between 0 and 1, representing the probability of the selected point being marked as pertencent of the class (y = 1). We can use the logistic function (sigmoid) for the hypothesis:
# 
# ![Logistic Curve](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)
# 
# $$\mathbf{g}(\mathbf{z}) = \frac{1}{1+e^{-\mathbf{z}}} = \frac{e^{\mathbf{z}}}{e^{\mathbf{z}}+1}$$

# So we can define the hypothesis as:
# 
# $$\mathbf{h_\theta} = \mathbf{g}(\mathbf{X'}\boldsymbol{\theta}) = \frac{1}{1+e^{-\mathbf{X'}\boldsymbol{\theta}}}$$

# One possibility of cost function for a hypothis ranging between 0 and 1 is to use the *Cross-entropy cost* formula:  
# 
# $$v = -\frac{1}{M}\sum \limits_{m = 1}^{M}[y_m log(h_\theta(\mathbf{x_m})) + (1-y_m)log(1 - h_\theta(\mathbf{x_m}))]$$  
# 
# Or in the vectorized notation (thus less computacionaly demanding):  
# 
# $$v = -\frac{1}{M}[\mathbf{y} log(\mathbf{h_\theta}) + (\mathbf{1_m}-\mathbf{y})log(\mathbf{1_m} - \mathbf{h_\theta})]$$ 

# In[6]:


X_prime = np.hstack((np.ones((M,1)), X)); X_prime


# In[7]:


theta = np.zeros([N+1,1])


# In[8]:


g = lambda z: 1/(1 + np.exp(-z))


# In[9]:


h = lambda X, theta: g(np.dot(X, theta))


# In[10]:


def v (X, y, theta):
    return -(sum(y*np.log(h(X, theta)) + (1 - y)*np.log(1 - h(X, theta)))/(M))[0]


# In[11]:


v(X_prime, y, theta)


# In[12]:


def gradient_step(X, y, theta, alpha=1):
    return theta - alpha*(np.dot(X.T, (h(X,theta)-y)))/M


# In[13]:


axis_x1 = np.linspace(-50, 150, 100)/100
axis_x2 = np.linspace(-50, 150, 100)/100
axis_X = np.vstack([axis_x1, axis_x2])


# In[14]:


for i in range(50):
    print("iter = ", i)
    print("theta = %.4f, %.4f, %.4f" % (theta[0,0], theta[1,0], theta[2,0]))
    print("cost function = %.4f\n" % v(X_prime,y,theta))
    
    plt.plot(X[:M_half,0], X[:M_half,1], 'X'), plt.xlabel('x0'), plt.ylabel('x1')
    plt.plot(X[M_half:,0], X[M_half:,1], 'o'), plt.xlabel('x0'), plt.ylabel('x1')
    plt.plot(axis_x1, (-1*axis_x1*(theta[1] - 1) - theta[0] - 0.5)/(theta[2] - 1) )

    
    plt.show()
    
    theta = gradient_step(X_prime, y, theta, 10)


# In[15]:


i += 1

print("iter = ", i)
print("theta = %.4f, %.4f, %.4f" % (theta[0,0], theta[1,0], theta[2,0]))
print("cost function = %.4f\n" % v(X_prime,y,theta))

plt.plot(X[:M_half,0], X[:M_half,1], 'X'), plt.xlabel('x0'), plt.ylabel('x1')
plt.plot(X[M_half:,0], X[M_half:,1], 'o'), plt.xlabel('x0'), plt.ylabel('x1')
plt.plot(axis_x1, (-1*axis_x1*(theta[1] - 1) - theta[0] - 0.5)/(theta[2] - 1) )


plt.show()


# In[ ]:




