
# coding: utf-8

# LSH Problem

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')
import csv

with open("tuple_list.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(tuple_list)
    

# # Part A

# In[5]:

temp = pd.read_csv("booleanVecs.csv",sep=',',header=None)
temp.shape


# In[6]:

boolVecs = temp.as_matrix()
print "total number of ones in all 100000 vectors:", np.sum(boolVecs)
assert np.sum(boolVecs[0,:]) == 51 #vector 0 has 51 ones
assert np.sum(boolVecs[-1,:]) == 45 #vector 99999 has 51 ones


# # Part B

# In[7]:

def compute_S(v1, v2):
    return np.mean(v1 == v2)


# In[8]:

# test and debug
v0 = boolVecs[0,:]
for i in range(1,6):
    print "S( vec 0, vec",i,") = ", compute_S(v0, boolVecs[i,:])


# # Part C

# In[9]:

#analyze the distribution of similarities
v0 = boolVecs[0,:]
s0 = []
for i in range(boolVecs.shape[0]):
    s0.append(compute_S(v0,boolVecs[i,:]))


# In[10]:

plt.hist(s0,50)
plt.show()


# I observed most of the similarities are distributed in the range of S < 0.7. Therefore, we would like to have LSH parameters that minimizes the false positives for S < 0.7 and maximizes true positives for S > 0.95.

# In[11]:

# compute the probability of positive outcomes
def prob_gr(s,g,r):
    return 1-(1-s**g)**r


# In[1]:

s0 = 0.95 # similarity of two vectors
s1 = 0.70
# find the best parameters for LSH
p0max = 0
p1min = 1
gbest = 0
rbest = 0
for g in range(1,30):
    for r in range(1,20):
        p0 = prob_gr(s0,g,r)
        p1 = prob_gr(s1,g,r)
        if p0>p0max and p1<p1min:
            p0max = p0
            p1min = p1
            gbest = g
            rbest = r
print "best g = ", gbest, ", best r = ",rbest

g = gbest
r = rbest
print "estimated probablity of true positives at S=0.95 :", prob_gr(s0,g,r)
print "estimated probablity of false positives at S = 0.7 :", prob_gr(s1,g,r)


# In[13]:

ss = np.arange(0,1,.01)
pr = [prob_gr(s,g,r) for s in ss]
plt.plot(ss,pr)
plt.xlabel('similarity')
plt.ylabel('prob of some collision')


# This plot shows that with the current set of parameters, we are able to minimize the probability of collisions for S < 0.7, and maximizing for S > 0.95.

# In[14]:

#hashnum = np.random.choice(range(boolVecs.shape[1]),size=g*r,replace=True)
hashnum = []
for i in range(r): # create pool of hash numbers
    hashnum.extend(np.random.choice(range(boolVecs.shape[1]),size=g,replace=False))
hashes = boolVecs[:,hashnum]


# In[15]:

# group together "g" hash functions from a pool of hashed values ("hs") , and repeat this process "r" times
def group_hashes(hs, g, r, v):
    return [tuple(hs[v,(i*g):((i+1)*g)]) for i in range(r)]


# In[16]:

def empty_lsh_table(r):
    return [{} for i in range(r)]
lshT = empty_lsh_table(r)


# In[17]:

def add_hash(ht, h, i):
    if h in ht:
        ht[h].append(i)
    else:
        ht[h] = [i]


# In[18]:

def add_hashes(lshT, hs, i, g, r):
    group = group_hashes(hs, g, r, i)
    for j in range(r):
        add_hash(lshT[j], group[j], i)


# Perform LSH by hashing each vector and group them into dictionaries according to their hashed values. Repeat this process r times to generate r hash tables (lshT).

# In[19]:

get_ipython().run_cell_magic('time', '', 'lshT = empty_lsh_table(r)\nfor i in range(boolVecs.shape[0]):\n    add_hashes(lshT, hashes, i, g, r)')


# In[20]:

def lsh_lookup(lshT, i, g, r):
    group = group_hashes(hashes, g, r, i)
    s = set()
    for j in range(r):
        s = s.union(set(lshT[j][group[j]]))
    return s


# For each vector, find its pairing vectors that have similarity greater or equal to 0.95. Store the results in "pair_list".

# In[21]:

get_ipython().run_cell_magic('time', '', 'pair_list = []\nfor i in range(boolVecs.shape[0]):\n    s = lsh_lookup(lshT, i, g, r)\n    for j in s:\n        if j > i and compute_S(boolVecs[i,:], boolVecs[j,:]) >= 0.95:\n            pair_list.append(tuple([i,j]))')


# In[22]:

print len(pair_list)
print len(set(pair_list))


# # The following cell will save your list of pairs.

# In[23]:

import csv

with open(netid + ".csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(pair_list)

