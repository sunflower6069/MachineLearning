
# coding: utf-8


#  Logistic PCA problem
# 
# We are given a matrix $X \in \mathbb{R}^{n \times d}$ where we think of this as $n$ examples in $d$ dimensions. We wish to find matrices $A \in \mathbb{R}^{n \times k}$ and $B \in \mathbb{R}^{k \times d}$ where the rows of $B$ have $\ell_2$ norm equal to $1$. That is $\|B[i,]\|_2^2 = 1$ for all $i \in [k]$ such that $\|AB - X\|_F^2$ is minimized. For simplicity, let $\mathcal{B} = \{B \mid \text{$\|B[i,]\|_2^2 = 1$ for all $i \in [k]$}\}$.
# 
# ### Problem 1.
# Find an equation for $A$ and $B$ in terms of $U_k$, $S_k$, and $V_k$. Be mindful of dimensions and transposes.
# 
# 
# ### Problem 2.
# Suppose that for our data, instead of having a data matrix $X$ that can be anything, our data matrix is binary. That is $X \in \{0,1\}^{n \times d}$. Above, we solved
# $$
# \arg \min_{A, B \in \mathcal{B}} \sum_{i,j} (A[i,] \times B[,j] - X[i,j])^2
# $$
# Thus, we can simply see that the standard PCA is a matrix factorization problem with the squared error loss. Now, find an analogous optimization that finds the best matrices $A$ and $B$ that minimizes the logistic loss. That is to say $1(A[i,] \times B[,j] > 0) \approx X[i,j]$ where the error between $A[i,] \times B[,j]$ and $X[i,j]$ is measured with respect to the logistic loss.
# 

# ### Answer to problem 1.
# $A$ = $U_k$$S_k$
# 
# $B$ = $V_k^T$

# ### Answer to problem 2.
# The analogous optimization that finds the best matrices $A$ and $B$ that minimizes the logistic loss is:
# $$
# \arg \min_{A,B \in \mathcal{B}} \sum_{i,j} \log(1+\exp(A[i,] \times B[,j])) - X[i,j] A[i,] \times B[,j]
# $$

# In[2]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


# In[3]:

from scipy import optimize
import pandas as pd


# The alternating minimization algorithm that solves the above minimization problem is:
# Let $A^{(t)}$, $B^{(t)}$ denote the values at the t'th iteration, and let $A^{(0)}$ be the initial random choice. 
# 
# We propose the iterative updates $A^{(t)} = \arg \min_{A} L(A,B^{(t-1)})$, and $B^{(t)} = \arg \min_{B \in \mathcal{B}} L(A^{(t-1)},B)$, where L is the logistic loss function.
# 
# Specifically, 
# 
# For $i$ = 1...n, $A[i,]^{(t)} = \arg \min_{A[i,]} \sum_{j} \log(1+\exp(-X[i,j]^*A[i,]B[,j]^{(t-1)}))$
# 
# For $j$ = 1...d, $B[,j]^{(t)} = \arg \min_{B[,j]} \sum_{i} \log(1+\exp(-X[i,j]^*A[i,]^{(t-1)}B[,j]))$
# 
# where $X[i,j]^* = 2 X[i,j] - 1$.
# 
# reference:  
# http://machinelearning.wustl.edu/mlpapers/paper_files/nips02-AA27.pdf  
# 
# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.fmin_cg.html#scipy.optimize.fmin_cg
# 
# http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/

# In[4]:

def finished(Aprev,A,Bprev,B,threshold):
    np.linalg.norm(Aprev-A)+np.linalg.norm(Bprev-B) < threshold


# In[5]:

def phi(t):
    # logistic function, returns 1./(1+exp(-t))
    idx = t>0
    out = np.zeros(t.size, dtype=np.float)
    out[idx] = 1./(1.+np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t/(exp_t+1.)
    return out


# In[6]:

def loss(w, X, y, alpha):
    # logistic loss function, with a regularization term incorporated to avoid divergence. Returns Sum{-log(phi(t))}
    t = y*(X.dot(w))
    idx = t>0
    out = np.zeros(t.size, dtype=np.float)
    out[idx] = np.log(1.+np.exp(-t[idx]))
    out[~idx] = -t[~idx] + np.log(1.+np.exp(t[~idx]))
    return np.sum(out) + 0.5*alpha*w.dot(w)


# In[7]:

def grad(w, X, y, alpha):
    # gradient of logistic loss
    t = y*(X.dot(w))
    z = y*(phi(t)-1.)
    out = X.T.dot(z) + alpha*w
    return out


# In[8]:

def PCA(M, k):
    U, s, VT = np.linalg.svd(M, full_matrices=False)
    UkSk = U[:,:k].dot(np.diag(s[:k]))
    VkT = VT[:k,:]
    return UkSk, VkT


# In[9]:

def updatelogpcaA(A,B,M):

    X = B.T
    alpha = 1.0

    for i in range(M.shape[0]):
        y = 2.*M[i,:]-1.
        y = y.T
        args = (X, y, alpha)
        w0 = A[i,:].T
        result = optimize.fmin_cg(loss, w0, fprime=grad, args=args,disp=0)
        A[i,:] = result.T
    
    return A


# In[10]:

def updatelogpcaB(A,B,M):
    
    X = A
    alpha = 1.0
    
    for j in range(M.shape[1]):
        y = 2.*M[:,j]-1.
        args = (X, y, alpha)
        w0 = B[:,j]
        result = optimize.fmin_cg(loss, w0, fprime=grad, args=args,disp=0)
        B[:,j] = result
    
    # normalizing each row of B
    for i in range(B.shape[0]):
        B[i,:] = B[i,:]/np.linalg.norm(B[i,:])
    
    return B


# In[11]:

## EDIT 11/20
def altminlogpca(X,k,threshold=10^(-6),maxiter=10):
    n,d=X.shape
    #A=Ainitializer
    #B=Binitializer
    A, B = PCA(X,k)
    Aprev=np.ones((n,k))
    Bprev=np.ones((k,d))
    itercount=0
    while (not finished(Aprev,A,Bprev,B,threshold)) and (itercount<maxiter):
        itercount+=1
        Aprev=A
        Bprev=B
        A=updatelogpcaA(A,B,X)
        B=updatelogpcaB(A,B,X)
    return A,B


# ### Problem 4.
# Instead of solving the problem using alternating minimization, solve it using gradient descent. 
# 

# In[12]:

def finished(Aprev,A,Bprev,B,threshold):
    np.linalg.norm(Aprev-A)+np.linalg.norm(Bprev-B) < threshold


# In[13]:

def Agradlogpca(A,B,M):
    Agrad = np.zeros_like(A)
    X = B.T
    alpha = 1.0
    for i in range(A.shape[0]):
        w = A[i,:].T
        y = 2.*M[i,:] - 1.
        y = y.T
        Agrad[i,:] = grad(w, X, y, alpha)
    return Agrad


# In[14]:

def Bgradlogpca(A,B,M):
    Bgrad = np.zeros_like(B)
    X = A
    alpha = 1.0
    for j in range(B.shape[1]):
        w = B[:,j]
        y = 2.*M[:,j] - 1.
        Bgrad[:,j] = grad(w, X, y, alpha)
    return Bgrad


# In[15]:

def projectell2B(B):
    for i in range(B.shape[0]):
        B[i,:] = B[i,:]/np.linalg.norm(B[i,:])
    return B


# In[16]:

def gradlogpca(X,k,stepsizeA=0.1,stepsizeB=0.1,threshold=10^(-6),maxiter=10):
    n,d=X.shape
    A, B = PCA(X,k)
    Aprev=np.ones((n,k))
    Bprev=np.ones((k,d))
    itercount=0
    while (not finished(Aprev,A,Bprev,B,threshold)) and (itercount<maxiter):
        itercount+=1
        Aprev=A
        Bprev=B
        A=A-stepsizeA*Agradlogpca(A,B,X)
        ## to the projected gradient descent step below
        B=projectell2B(B-stepsizeB*Bgradlogpca(A,B,X)) ##you should be able to reuse the Agradlogpca code
    return A,B


# ### Problem 5.
# Run functions on the given data. Adding a ridge regression penalty can help improve the convergence of the algorithms as well as the statistical performance.

# In[17]:

n=1000
d=50 ##EDIT: 11/21 changed d
k=4
Astar=np.random.randn(n,k)
Bstar=np.random.randn(k,d)/np.sqrt(k)

## EDIT 11/20/2016
Xnoisy=np.random.rand(n,d)<(1/(1+np.exp(-Astar.dot(Bstar))))
Xclean=(np.sign(Astar.dot(Bstar))+1.)/2.

A,B = altminlogpca(Xnoisy,k)
Ag,Bg = gradlogpca(Xnoisy,k)
Apca,Bpca = PCA(Xnoisy,k) ##use your previous PCA code here

Ac,Bc = altminlogpca(Xclean,k)
Acg,Bcg = gradlogpca(Xclean,k)
Acpca,Bcpca = PCA(Xclean,k) ##use your previous PCA code here


# 
# Compare which solutions do better for both `Xnoisy` and `Xclean` with respect to the logistic loss, the squared error loss, and the following loss
# 
# $$
# L(X,AB) = \| X - \phi(AB) \|_F^2
# $$
# where $\phi$ operates pointwise and
# $$
# \phi(t) = \frac{1}{1+\exp(-t)}
# $$
# 
# That is compare $A \times B$ to $Xnoisy$, $Ag \times Bg$ to $Xnoisy$, and $Apca \times Bpca$ to $Xnoisy$. Then compare
# $Ac \times Bc$ to $Xclean$, $Acg \times Bcg$ to $Xclean$, and $Acpca \times Bcpca$ to $Xclean$ with respect to each of the three different losses above.

# In[18]:

def logistic_loss(A,B,M):
    X = A
    loss = 0.
    for j in range(M.shape[1]):
        w = B[:,j]
        y = 2.*M[:,j] - 1.
        t = y*X.dot(w)
        idx = t>0
        out = np.zeros(t.size, dtype=np.float)
        out[idx] = np.log(1.+np.exp(-t[idx]))
        out[~idx] = -t[~idx] + np.log(1.+np.exp(t[~idx]))
        loss += np.sum(out)
    return loss


# In[19]:

def squared_loss(A,B,M):
    sign = (A.dot(B)>0)*1.0
    out = sign - M
    return np.linalg.norm(out)**2


# In[20]:

def new_loss(A,B,M):
    X = A
    loss = 0.
    for j in range(M.shape[1]):
        w = B[:,j]
        t = X.dot(w)
        out = M[:,j] - phi(t)
        loss += np.linalg.norm(out)**2
    return loss


# In[21]:

def report(A,B,Ag,Bg,Apca,Bpca,M):
    dic = {}
    dic['AltMin_logPCA'] = [logistic_loss(A,B,M),squared_loss(A,B,M),new_loss(A,B,M)]
    dic['Grad_logPCA'] = [logistic_loss(Ag,Bg,M),squared_loss(Ag,Bg,M),new_loss(Ag,Bg,M)]
    dic['PCA'] = [logistic_loss(Apca,Bpca,M),squared_loss(Apca,Bpca,M),new_loss(Apca,Bpca,M)]
    table = pd.DataFrame(dic, index=['logistic_loss','squared_loss','new_loss'])
    return table


# In[22]:

print "Report for Problem 5:\n"
print "Xnoisy"
print report(A,B,Ag,Bg,Apca,Bpca,Xnoisy)
print "\n"
print "Xclean"
print report(Ac,Bc,Acg,Bcg,Acpca,Bcpca,Xclean) 

