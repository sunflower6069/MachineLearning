
# coding: utf-8

# # Using Personal PageRank to find clusters
# 
# In[2]:

import networkx as nx

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


# In[3]:

# Load the test graph
testg = nx.read_adjlist("demoGraph.txt")


# ## A. Write conductance
# 
# Define a function `conductance` that takes as input a graph and a set of its vertices,
# and returns the conductance of that set.  The set should be represented as a list.  Put it in the following cell.
# It should be callable in the following tests.

# In[4]:

def boundary(graph, subset):
    boundary = 0
    for u in subset:
        nb = graph.neighbors(u)
        for v in nb:
            if v not in subset:
                boundary += 1
    return boundary


# In[5]:

def volume(graph, subset):
    degs = np.array(graph.degree(subset).values())
    return np.sum(degs)


# In[6]:

def conductance(graph, subset):
    nodes = graph.nodes()
    volS = volume(graph, subset)
    return (boundary(graph,subset)+0.0)/min(volS, 2*len(graph.edges()) - volS)


# In[7]:

# The conductance of a single vertex is always one.  So, make sure the following gives 1.0
conductance(testg,['2'])


# In[8]:

# The conductance of every vertex but one is always 1.0.  Check that on this example.
nodes = testg.nodes()
allbut = nodes[1:]
conductance(testg,allbut)


# In[9]:

# Compute the conductance of the set of the first 15 vertices.  It should be 0.16. 
# Note that they are named '1' through '15'.
first15 = [str(i) for i in range(1,16)]
conductance(testg,first15)


# ## B. Write sweep
# 
# `Sweep` should take as input a graph and a vector `v` whose length is equal to the number of vertices in the graph.  The ith entry of v, `v[i]` is the value it assigns to the ith node in the list produced by g.nodes().
# 
# The algorithm should produce a new vector by dividing each entry of v by the degree of the corresponding node.  It should then consider sets $S_k$, where $S_k$ is the set of the k vertices for which the resulting vector is largest.  `Sweep` should compute the conductance of each such set, and return the set nodes of this form of the least conductance.
# 
# You should compute the conductances quickly, and not (unless you can't find a better way) do it by computing each conductance afresh.  The reason you can do this is that $S_k$ and $S_{k+1}$ only differ in one vertex.  So, the volume of $S_{k+1}$ is larger than the volume of $S_k$ by the degree of the $k+1$ vertex in this order.  Similarly, the set of edges that are on the boundary changes very little between $S_{k+1}$ and $S_k$.  If you can keep track of which they are, you can make this computation many times faster than it would be if you computed each conductance afresh.
# 
# In[10]:

def sweep(graph, vec):
    nodes = graph.nodes()
    nnodes = len(nodes)
    degs = np.array([graph.degree(i) for i in nodes])
    idx = np.argsort(vec/degs)[-1::-1] # return node indices with v in descending order
    dic = {} # dic[i] = nodes in S_k
    for k in range(1,nnodes+1):
        dic[k] = [nodes[i] for i in idx[:k]]
    
    # compute conductance for each S_k and return the set with least conductance
    vol = degs[idx[0]]
    bound = vol
    volV = 2*len(graph.edges())
    mincond = 1.
    mink = 1
    for k in range(1,nnodes-1):
        new = nodes[idx[k]]
        vol += degs[idx[k]]
        nb = graph.neighbors(new)
        for v in nb:
            if v not in dic[k]:
                bound += 1
            else:
                bound -= 1
        cond = float(bound)/min(vol, volV-vol)
        if cond < mincond:
            mincond = cond
            mink = k+1
    
    return dic[mink]


# In[11]:

# To test sweep, let's check its results on the vector 
# in which the value of each node is its name, converted to a number
s = sweep(testg,np.array([float(i)*testg.degree(i) for i in nodes]))


# In[12]:

print len(s)
print conductance(testg,s)


# In[13]:

# you should have just found a set of 144 vertices of conductance 0.016

# Let's draw the graph so you can see this set.
# the nodes in s will be in blue.  The others will be red.
# The blue ones should correspond to 4 clusters in the picture.

plt.figure()
pos=nx.spectral_layout(testg)
nx.draw_networkx_edges(testg,pos)
nx.draw_networkx_nodes(testg,pos,nodelist=s, node_size = 30, node_color='blue')

# compute all nodes not in s
others = [i for i in nodes if (i not in s)]
nx.draw_networkx_nodes(testg,pos,nodelist=others, node_size = 30, node_color='red')
plt.axis('off')
plt.show()


# ## C. Write Personal PageRank
# 
# Write a routine called `ppr` that takes 4 inputs:
# * the graph, g
# * a vertex, v.  This should be the name of the node.
# * alpha - the teleport parameter, and
# * epsilon - the accuracy parameter
# It should compute an epsilon approximate personal pagerank vector from the vertex
# with teleport probability alpha.
# 
# In[14]:

def ppr(g,v,alpha,epsilon):
    nodes = g.nodes()
    v_ind = nodes.index(v)
    ev = np.zeros(len(nodes),dtype=np.float)
    ev[v_ind] = 1.

    A = nx.adjacency_matrix(g)
    degs = np.array([g.degree(i) for i in nodes])
    Dinv = np.diag(1.0/degs)
    W = A.T.dot(Dinv) # walk matrix
    
    delta = 1. - alpha
    Pvold = ev
    term = alpha*W.dot(Pvold) 
    Pv = Pvold + term
    maxiters = 100
    t = 1
    while (t < maxiters and np.any(np.absolute(Pvold - Pv) > epsilon*degs)):
        Pvold = Pv
        term = alpha*W.dot(term)
        Pv = Pv + term
        t += 1

    if t == maxiters:
        print "Calculation hasn't converged."
    else:
        print "Personal PageRank calculation successfully terminated in %d iterations" %t
        Pv = Pv/np.sum(Pv)
    
    return Pv


# In[15]:

# To test this, compute on our test graph as follows:
p = ppr(testg, '1', 0.9, 0.01)

# and, check the conductance of the set provided by a sweep
s = sweep(testg, p)
conductance(testg, s)


# The following cell will load in a protein-protein interaction graph.
# I have cleaned it up a bit by giving you the largest connected component, and by removing removing all degree 1 vertices.
# 
# Compute the Personal PageRank vector from many different seeds (I suggest random seeds)
# with alpha = 0.9 and epsilon = 0.001, and then sweep to find a set of low conductance, until you find a set of conductance at most 0.1.  Print out the conductance of s, the number of vertices in s, its volume, and the size of its boundary.
# 
# 

# In[16]:

import hashlib
hash_object = hashlib.md5(netid)
hex_hash = hash_object.hexdigest()
seed = int(hex_hash[0:4],16)
print "seed : ", seed
np.random.seed(int(hex_hash[0:4],16))


# In[17]:

g = nx.read_adjlist("Bg_Arabidopsis_thaliana-core.edges")


# In[18]:

nodes = g.nodes()
alpha = 0.9
epsilon = 0.001
maxiters = 100
for i in range(maxiters):
    v = nodes[np.random.randint(0,len(nodes))]
    print "iter %d: use vertex %s as reference" %(i, v)
    p = ppr(g,v,alpha,epsilon)
    s = sweep(g,p)
    if conductance(g,s) < 0.1:
        print "The conductance of s is %f." %conductance(g,s)
        print "The number of vertices in s is %d." %len(s)
        print "The volume of s is %d." %volume(g,s)
        print "The size of boundary of s is %d." %boundary(g,s)
        break
        
if i == maxiters-1:
    print "Haven't found a set of conductance less than 0.1 yet."


# In[19]:

# Let's draw the graph so you can see this set.
# the nodes in s will be in blue.  The others will be red.

plt.figure()
pos=nx.spectral_layout(g)
nx.draw_networkx_edges(g,pos)
nx.draw_networkx_nodes(g,pos,nodelist=s, node_size = 30, node_color='blue')

# compute all nodes not in s
others = [i for i in nodes if (i not in s)]
nx.draw_networkx_nodes(g,pos,nodelist=others, node_size = 30, node_color='red')
plt.axis('off')
plt.show()


# ## E. Run the following cell to save the set you found

# In[20]:

file = open(netid + ".txt", 'w')
for i in s:
    file.write("%s\n" % i)
file.close()

