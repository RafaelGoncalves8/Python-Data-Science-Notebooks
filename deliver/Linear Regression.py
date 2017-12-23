
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')

import random
import numpy as np
import scipy
import matplotlib.pyplot as plt


# Linear regression is an algorithm for predicting values based on another features. It is a supervised learning algorithm - i.e. is fed with samples and it's corresponding labels ("correct values").

# # Variables and Constants

# - $M$ - Number of samples
# - $N$ - Number of features
# - $\boldsymbol{\theta}$ - Vector of minimizing parameters
# - $\mathbf{X}$ - Matrix of features (columns) per samples (rows)
# - $\mathbf{y}$ - Vector of corresponding labels

# In[2]:


M = 50
N = 1

# Random X and Y (y = ax + b w/ a~10, b~4)
x = np.array([[max(i,1)*random.gauss(1,0.10)] for i in range(M)]);
y = np.array([[(max(i,1)*random.gauss(4,1)+(10))] for i in range(M)]);

print("    x     |     y    ")
for i in range(M):
    print("%9.4f | %9.4f" % (x[i], y[i]))


# In[3]:


plt.plot(x, y, '.'), plt.xlabel('x'), plt.ylabel('y')
plt.show()


# # Normalizing

# In[4]:


min_val = [np.min(x), np.min(y)]
interval = [(np.max(x) - np.min(x)), (np.max(y) - np.min(y))]

x = (x - min_val[0]*np.ones(x.shape))/interval[0]
y = (y - min_val[1]*np.ones(y.shape))/interval[1]


# In[5]:


plt.plot(x, y, '.'), plt.xlabel('x'), plt.ylabel('y')
plt.show()


# # Hyphotesis and Cost function

# The objective of the linear regression is to find a *hyphotesis function*  $\mathbf{h_\theta} = \mathbf{h_\theta (X)}$ - or the parameters of this function $\boldsymbol{\theta}$ - such as the *cost function* $J = J(\boldsymbol{\theta})$ - diference between predictions and labels - is minimized. 

# One possibility for cost function is to use the *Mean Squared Error* (MSE) formula:  
# 
# $$J = \frac{1}{2M}\sum\limits_{m = 1}^{M}(h_m - y_m)^2$$  
# 
# Or in the vectorized notation (thus less computacionaly demanding):  
# 
# $$J = \frac{1}{2M}(\mathbf{h_\theta} - \mathbf{y})^T(\mathbf{h_\theta} - \mathbf{y})$$

# For the hyphotesis, in the case of linear regression, the most general formula is:
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

# In[6]:


x_prime = np.hstack((np.ones((M,1)), x)); x_prime


# In[7]:


theta = np.zeros([N+1,1])


# In[8]:


h = lambda X, theta: np.dot(X, theta)


# In[9]:


def J (X, y, theta):
    return np.sum((h(X,theta)-y)**2)/(2*M)


# # Gradient descent

# $$\boldsymbol{\theta} = \boldsymbol{\theta} - \frac{\partial{J}}{\partial \boldsymbol{\theta}}$$
# 
# $$\frac{\partial J}{\partial \theta_n} = (h_\theta(\mathbf{x_m}) - y_m)x_{mn}$$
# 
# $$\theta_n \leftarrow \theta_j - \frac{\alpha}{M} \sum \limits _{m=1}^M(h_\theta(\mathbf{x_m}) - y_m)x_{mn}$$
# 
# $$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \frac{\alpha}{M}\mathbf{X' ^T(h_\theta - y)}$$

# In[10]:


def gradient_step(X, y, theta, alpha=1):
    return theta - alpha*(np.dot(X.T, (h(X,theta)-y)))/M


# In[11]:


axis_x = np.linspace(-50, 150, 100)/100
axis_theta_0 = np.linspace(-100, 100, 100)/100
axis_theta_1 = np.linspace(-100, 100, 100)/100

X, Y = np.meshgrid(axis_theta_0, axis_theta_1, sparse=False)
Z = np.zeros((100,100))

for i, e in enumerate(X[0]):
    for j, f in enumerate(Y[:,0]):
        Z[i,j] = J(x_prime, y, np.array([[e],[f]]))
        #print(e,f,Z[i,j])


# In[12]:


for i in range(50):
    print("iter = ", i)
    print("theta = %.4f, %.4f" % (theta[0,0], theta[1,0]))
    print("cost function = %.4f\n" % J(x_prime,y,theta))
    
    fig, axis = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(15,5)
    axis[0].plot(x, y, '.'), axis[0].set_xlabel('x'), axis[0].set_ylabel('y'), axis[0].set_xlim([-0.1,1]),axis[0].set_ylim([-0.1,1])
    axis[0].plot(axis_x, theta[0] +axis_x*theta[1] )                      # regression
    
    axis[1].contour(X, Y, Z, 250), axis[1].set_xlabel('theta 1'), axis[1].set_ylabel('theta 0')
    #axis[1].set_xlim([-1,1]),axis[1].set_ylim([-1,1])
    axis[1].plot(theta[1,0], theta[0,0], 'ro')
    
    plt.show()
    
    theta = gradient_step(x_prime, y, theta, 1)


# In[13]:


i += 1

print("iter = ", i)
print("theta = %.4f, %.4f" % (theta[0,0], theta[1,0]))
print("cost function = %.4f\n" % J(x_prime,y,theta))

fig, axis = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(15,5)
axis[0].plot(x, y, '.'), axis[0].set_xlabel('x'), axis[0].set_ylabel('y'), axis[0].set_xlim([-0.1,1]),axis[0].set_ylim([-0.1,1])
axis[0].plot(axis_x, theta[0] +axis_x*theta[1] )                      # regression

axis[1].contour(X, Y, Z, 250), axis[1].set_xlabel('theta 1'), axis[1].set_ylabel('theta 0')
#axis[1].set_xlim([-1,1]),axis[1].set_ylim([-1,1])
axis[1].plot(theta[1,0], theta[0,0], 'ro')

plt.show()


# # Denormalize

# In[ ]:




