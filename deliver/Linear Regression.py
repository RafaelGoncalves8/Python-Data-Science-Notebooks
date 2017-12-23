
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')

import random
import numpy as np
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

# Random X and Y
x = np.array([[(i+1)*random.gauss(3,1)] for i in range(M)]);
y = np.array([((i+1)*np.random.random(1)+2*i) for i in range(M)]);
theta = np.zeros([N+1,1])

print("    x     |     y    ")
for i in range(M):
    print("%9.4f | %9.4f" % (x[i], y[i]))


# In[3]:


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
# $$\mathbf{h_\theta} = \sum\limits_{n = 0}^N \theta_nx'_n$$  
# 
# or in vectorized notation:
# 
# $$\mathbf{h_\theta} = \mathbf{X'\theta}$$

# In[4]:


x_prime = np.hstack((np.ones((M,1)), x))


# In[5]:


def hyp (x_prime):
    global theta
    return np.dot(x_prime, theta)


# In[6]:


def cost (h):
    global y
    return np.sum((h-y)**2)/(2*M); J # initial error


# In[7]:


h = hyp(x_prime)
J = cost(h); J


# # Gradient descent

# In[8]:


x_axis = np.linspace(-10, 250)


# In[15]:


def gradient_descent(X, y, alpha=0.0001, n_iter=15):
    m = X.size
    i = 0
    global theta
    
    h = hyp (X)
    print("iter = ", i)
    print("theta = %.4f, %.4f" % (theta[0,0], theta[1,0]))
    print("cost function = %.4f\n" % cost(h))
       
    plt.plot(x, y, '.'), plt.xlabel('x'), plt.ylabel('y')
    plt.plot(x_axis, theta[0] +x_axis*theta[1] )
    plt.show()
    
    for _ in range(n_iter):
        i += 1
        h = hyp (X)
        theta = theta - alpha*(np.dot(X.T,(h-y)))/m
        
        print("iter = ", i)
        print("theta = %.4f, %.4f" % (theta[0,0], theta[1,0]))
        print("cost function = %.4f\n" % cost(h))
        
        plt.plot(x, y, '.'), plt.xlabel('x'), plt.ylabel('y')
        plt.plot(x_axis, theta[0] +x_axis*theta[1] )
        plt.show()
        
    return theta


# In[10]:


theta = gradient_descent(x_prime, y)


# In[14]:


# data
plt.plot(x, y, '.'), plt.xlabel('x'), plt.ylabel('y')

# linear regression
plt.plot(x_axis, theta[0] +x_axis*theta[1] )

plt.show()
h = hyp(x_prime)
J = cost(h)

print("Cost function = %.4f" %J)


# In[ ]:


cost


# In[21]:


x_axis

plt.plot(x_axis, map(cost, hyp(x_axis)))


# In[ ]:




