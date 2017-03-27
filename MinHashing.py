
# coding: utf-8

#  MinHash

import numpy as np 
import pandas as pd
import os
import sys
get_ipython().magic('matplotlib inline')
import hashlib

def hash_word(hashnum, bytes, word):
    hash_object = hashlib.md5(str(hashnum) + word)
    hex_hash = hash_object.hexdigest()
    return int(hex_hash[0:bytes],16)

# The arguments to hash_word are:
#   hashnum : the number of the hash function, so that you can generate many hash functions
#   bytes : how many bytes you want the hash to be.  It takes values between 0 and 256^bytes - 1
#   word : the object (string or vector) to be hashed

# here are some examples
print hash_word(1,1,'x')
print hash_word(2,1,'x')
print hash_word(3,1,'x')
print hash_word(1,2,'x')
print hash_word(1,3,'x')
print hash_word(1,4,'x')


# 

# In[3]:

# find an n that is closest to n0, so that there exist sets A and B of size n for which J(A,B) = J.
def find_n (n0, J):
    epsilon = 1.e-4
    delta = 0
    for delta in range(n0):
        n = n0 + delta
        intersect = 2.0*n*J/(1.0+J)
        if abs(intersect - int(intersect)) < epsilon:
            return n
        else:
            n = n0 - delta
            intersect = 2.0*n*J/(1.0+J)
            if abs(intersect - int(intersect)) < epsilon:
                return n
    print "Uh-Oh, couldn't find such an n."
    return 0


# #test and debug
# for n0 in [10,100,1000]:
#     for J in [0.1,0.5,0.9]:
#         print "for n0=%d, J=%.1f, find n=%d" %(n0, J, find_n(n0,J))

# In[4]:

# Create sets A and B that each have n elements so that J(A,B)=J
def create_sets (n, J):
    intersect = int(2*n*J/(1+J))
    A = np.array(range(n))
    B = np.random.choice(A,size=intersect,replace=False)
    B = np.append(B,np.array(range(n,2*n-intersect)))
    return A, B


# #test and debug
# A, B = create_sets(11,0.1)
# print A, B
# total = len(A)+len(B)
# union = len(set(A).union(set(B)))
# jaccard = (total-union+0.0)/union
# print jaccard

# In[5]:

# for 1000 different hash functions, compute b-byte MinHashes of A and B. Return the count of MinHashes that are equal
def min_hash(hashnum, bytes, bag):
    return min([hash_word(hashnum,bytes,str(w)) for w in bag])

def nequal(nhashes, bytes, bag1, bag2):
    count = 0
    for hashnum in range(nhashes):
        mh1 = min_hash(hashnum, bytes, bag1)
        mh2 = min_hash(hashnum, bytes, bag2)
        count += (mh1==mh2)
    return count


# #test and debug
# count = nequal(1000, 2, A, B)
# print(count)

# In[6]:

# Report fractions of these MinHashes that were equal
def test_MinHash(n0, J, b, nhashes):
    n = find_n(n0, J)
    A, B = create_sets (n, J)
    count = nequal(nhashes, b, A, B)
    return (count+0.0)/nhashes


def report(b):
    nhashes = 1000 # try 1000 different hash functions
    dic = {}
    for n0 in [10,100,1000]:
        dic[n0] = []
        for J in [0.1,0.5,0.9]:
            dic[n0].append(test_MinHash(n0, J, b, nhashes))
    table = pd.DataFrame(dic, index=[0.1,0.5,0.9])
    return table


# #test and debug
# print test_MinHash(10,0.1,2,1000)
# print test_MinHash(1000,0.1,2,1000)

# In[7]:

print "Report for Problem 1"
for b in range(2,6):
    print "\nb=%d:" %b
    print report(b)

