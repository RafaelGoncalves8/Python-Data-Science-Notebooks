
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

# Random X and Y
x = np.array([[(i+1)*random.gauss(3,1)] for i in range(M)]);
y = np.array([((2*i+1)*np.random.random(1)+(20+5*i)) for i in range(M)]);
theta = np.zeros([N+1,1])

print("    x     |     y    ")
for i in range(M):
    print("%9.4f | %9.4f" % (x[i], y[i]))


# In[3]:


plt.plot(x, y, '.'), plt.xlabel('x'), plt.ylabel('y')
plt.show()


# # Normalizing

# In[4]:


x = (x - np.average(x)*np.ones(x.shape))/(np.max(x) - np.min(x))
y = (y - np.average(y)*np.ones(y.shape))/(np.max(y) - np.min(y))


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

# In[5]:


x_prime = np.hstack((np.ones((M,1)), x)); x_prime


# In[6]:


def h (x_prime):
    global theta
    return np.dot(x_prime, theta)


# In[7]:


def J_h (h):
    global y
    return np.sum((h-y)**2)/(2*M); J # initial error


# In[8]:


def J_theta(theta):
    global x_prime, y
    h = np.dot(x_prime, theta)
    return np.sum((h-y)**2)/(2*M);


# In[9]:


J_theta(theta)


# # Gradient descent

# $$\boldsymbol{\theta} = \boldsymbol{\theta} - \frac{\partial{J}}{\partial \boldsymbol{\theta}}$$
# 
# $$\frac{\partial J}{\partial \theta_n} = (h_\theta(\mathbf{x_m}) - y_m)x_{mn}$$
# 
# $$\theta_n \leftarrow \theta_j - \frac{\alpha}{M} \sum \limits _{m=1}^M(h_\theta(\mathbf{x_m}) - y_m)x_{mn}$$
# 
# $$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \frac{\alpha}{M}\mathbf{X' ^T(h_\theta - y)}$$

# In[15]:


x_axis = np.linspace(-10, 100)
theta_0 = np.linspace(-3000,3000,100)/1000
theta_1 = np.linspace(-1500,1500,100)/1000

X, Y = np.meshgrid(theta_0, theta_1, sparse=False)

Z = np.zeros((100,100))

for i, e in enumerate(X[0]):
    for j, f in enumerate(Y[:,0]):
        Z[i,j] = J_theta(np.array([[e],[f]]))
        print(e,f,Z[i,j])


# In[11]:


def gradient_descent(X, y, alpha=3):
    try:
        M = X.size[0]
    except:
        M = X.size
        
    global theta
    
    #h = h(X)
    theta = theta - alpha*(np.dot(X.T,(h(X)-y)))/M
    return theta


# In[12]:



for i in range(20):
    
    H = h(x_prime)
    
    print("iter = ", i)
    print("theta = %.4f, %.4f" % (theta[0,0], theta[1,0]))
    print("cost function = %.4f\n" % J_h(H))
    
    fig, axis = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(15,5)
    axis[0].plot(x, y, '.'), axis[0].set_xlabel('x'), axis[0].set_ylabel('y'), axis[0].set_xlim([-0.1,1]),axis[0].set_ylim([-0.1,1])
    axis[0].plot(x_axis, theta[0] +x_axis*theta[1] )                      # regression
    
    axis[1].contour(X, Y, Z, 40), axis[1].set_xlabel('theta 0'), axis[1].set_ylabel('theta 1')
    axis[1].set_xlim([-4,4]),axis[1].set_ylim([-1,1])
    axis[1].plot(theta[0,0], theta[1,0], 'ro')
    
    plt.show()
    
    theta = gradient_descent(x_prime, y)


# In[13]:


i += 1

print("iter = ", i)
print("theta = %.4f, %.4f" % (theta[0,0], theta[1,0]))
#print("cost function = %.4f\n" % (h))

fig, axis = plt.subplots(nrows=1, ncols=2)

axis[0].plot(x, y, '.'), axis[0].set_xlabel('x'), axis[0].set_ylabel('y') # data
axis[0].set_xlim([-0.2,1.0]),axis[0].set_ylim([-0.1,1.0])
axis[0].plot(x_axis, theta[0] +x_axis*theta[1] )                      # regression


axis[1].contour(X, Y, Z), axis[1].set_xlabel('theta 0'), axis[1].set_ylabel('theta 1')
axis[1].plot(theta[0,0], theta[1,0], 'ro')

plt.show()


# In[ ]:





# In[ ]:




