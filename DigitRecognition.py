
# coding: utf-8

# Digits Problem

# # Start up Spark.

DataBricks = False


if not(DataBricks):

    import findspark
    import os

    findspark.init()
    import pyspark
    sc = pyspark.SparkContext()


# In[4]:

if not(DataBricks):
    get_ipython().magic('matplotlib inline')
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage
else:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage


# # Load in the Digits Data.
# The following cell should load the data files.
# Note that this is only a small fragment of the MNIST data set.
# We are using a small data set so that the notebooks do not take too long to run.
# 
# If you are using DataBricks, the following code will load the images from the Web.  If you are not using DataBricks, you could use that code, too.
# Or, you could download the files and use the `fromfile` methods.
# 
# Feel free to change the following cell to load the data in the way you prefer.

# In[5]:

if not DataBricks:
    train_labels = np.genfromtxt("http://cs-www.cs.yale.edu/homes/spielman/262/train_labels.txt",delimiter=',')
    train_imgs = np.genfromtxt("http://cs-www.cs.yale.edu/homes/spielman/262/train_imgs.txt",delimiter=',')    
    train_imgs.shape=(1000,28,28) 
    
    test_labels = np.genfromtxt("http://cs-www.cs.yale.edu/homes/spielman/262/test_labels.txt",delimiter=',')
    test_imgs = np.genfromtxt("http://cs-www.cs.yale.edu/homes/spielman/262/test_imgs.txt",delimiter=',')
    test_imgs.shape=(1000,28,28)    
else:
    train_labels = np.fromfile("train_labels.txt",sep = ",")
    train_imgs = np.fromfile("train_imgs.txt",sep = ",")
    train_imgs.shape=(1000,28,28)

    test_labels = np.fromfile("test_labels.txt",sep = ",")
    test_imgs = np.fromfile("test_imgs.txt",sep = ",")
    test_imgs.shape=(1000,28,28)


# # A. Create an RDD out of the training images and their labels.
# Call the resulting RDD `Train`.

# In[6]:

Train = sc.parallelize(zip(train_labels,train_imgs))


# In[7]:

# You will want to use this many times, so make it persistent.
Train.persist()


# # B.  A function  that finds the closest training image
# Write a function that takes as input an image (28-by-28 matrix) and the RDD of training images, and returns the label of the training image that is closest to it.
# That is, if `train_img[i]` is closest, it should return `train_label[i]`.
# Your function should *only* use Spark operations on the RDD you created in the previous cell.  It is possible to do this with one map and one reduce.
# 
# Call the function `closest` so that you can run the checkpoint below.
# 
# Your code should definitely *not* have a loop.

# In[8]:

def closest(img, Train):
    Fnorm = Train.map(lambda x : (x[0],np.sum((x[1]-img)**2,axis=(0,1)))) #compute the Frobenius norm for each matrix
    minFnorm = Fnorm.reduce(lambda x, y: x if x[1]<y[1] else y)
    return minFnorm[0]


# ## Checkpoint.
# The labels of the training images closest to train_img[i] for i in {0,1,2} are 7, 2, and 1.  Run the next cell to check that you get this answer.

# In[9]:

print closest(test_imgs[0],Train)
print closest(test_imgs[1],Train)
print closest(test_imgs[2],Train)


# # C. How accurate is nearest neighbor?

# In[18]:

get_ipython().run_cell_magic('time', '', 'lab_assigned = []\nfor img in test_imgs:\n    lab_assigned.append(closest(img,Train))\nlab_assigned = np.array(lab_assigned)\nerr_rate = np.mean(lab_assigned != test_labels)\nprint err_rate')


# # Creating more training data

# In[19]:

# the original
img = test_imgs[11]

fig = plt.figure()
plt.imshow(img,cmap=plt.cm.gray)
plt.title("The Original")
if DataBricks:
    display(fig)


# In[20]:

fig = plt.figure()
plt.imshow(ndimage.rotate(img,10),cmap=plt.cm.gray)
plt.title("Rotated Right")
if DataBricks:
    display(fig)


# In[22]:

print img.shape
print ndimage.rotate(img,10).shape


# In[23]:

fig = plt.figure()
plt.imshow(ndimage.rotate(img,-10),cmap=plt.cm.gray)
plt.title("Rotated Left")
if DataBricks:
    display(fig)


# In[24]:

print img.shape
print ndimage.rotate(img,-10).shape


# In[25]:

fig = plt.figure()
plt.imshow(ndimage.gaussian_filter(img,1),cmap=plt.cm.gray)
plt.title("Blurred")
if DataBricks:
    display(fig)


# In[26]:

print img.shape
print ndimage.gaussian_filter(img,1).shape


# In[27]:

fig = plt.figure()
plt.imshow(ndimage.zoom(img,(1,0.8)),cmap=plt.cm.gray)
plt.title("Shrunk Horizontally")
if DataBricks:
    display(fig)


# In[28]:

print img.shape
print ndimage.zoom(img,(1,0.8)).shape


# In[29]:

fig = plt.figure()
plt.imshow(ndimage.zoom(img,(0.8,1)),cmap=plt.cm.gray)
plt.title("Shrunk Vertically")
if DataBricks:
    display(fig)


# In[30]:

print img.shape
print ndimage.zoom(img,(0.8,1)).shape


# ## Warning:
# Some of these operations change the size of the image.
# You will need to crop or expand each back to 28-by-28
# so that you can compare them to training images.
# Do so symmetrically.

# In[48]:

def resize(img, shape):
    height = img.shape[0]
    width = img.shape[1]
    newimg = img.copy()
    if height < shape[0]:
        pad = np.zeros(((shape[0]-height)/2,shape[1]))
        newimg = np.vstack((pad,newimg,pad))
    elif height > shape[0]:
        h2 = int((height-shape[0])/2)
        newimg = newimg[h2:-h2,:]
    if width < shape[1]:
        pad = np.zeros((shape[0],(shape[1]-width)/2))
        newimg = np.hstack((pad,newimg,pad))
    elif width > shape[1]:
        w2 = int((width-shape[1])/2)
        newimg = newimg[:,w2:-w2]
    return newimg


# #debug
# print img.shape
# newimg = resize(ndimage.rotate(img,-10),img.shape)
# print newimg.shape
# plt.imshow(newimg,cmap=plt.cm.gray)

# # Apply these using Spark.
# Create a Spark command that applies all of these operations to the training images, while preserving the labels.  The result should be a larger set of images (6000 in total) with labels.
# This can be done with one line of Spark, once you create the appropriate functions to transform the images.
# 
# Call the resulting RDD `TrainMore`

# In[70]:

def rot_right(img):
    return resize(ndimage.rotate(img,10), img.shape)


# In[71]:

def rot_left(img):
    return resize(ndimage.rotate(img,-10), img.shape)


# In[72]:

def blur(img):
    return ndimage.gaussian_filter(img,1)


# In[73]:

def shrk_h(img):
    return resize(ndimage.zoom(img,(1,0.8)), img.shape)


# In[74]:

def shrk_v(img):
    return resize(ndimage.zoom(img,(0.8,1)), img.shape)


# In[75]:

def apply_all(tup):
    ls = [tup]
    ls.append((tup[0],rot_right(tup[1])))
    ls.append((tup[0],rot_left(tup[1])))
    ls.append((tup[0],blur(tup[1])))
    ls.append((tup[0],shrk_h(tup[1])))
    ls.append((tup[0],shrk_v(tup[1])))
    return ls


# In[83]:

TrainMore = Train.flatMap(apply_all)


# ## Checkpoint: Make sure `TrainMore` has 6000 entries.

# In[84]:

TrainMore.persist()
TrainMore.count()


# # Compute the error rate with the new images.
# Now, use all 6000 training images (the original 1000 and the new 5000 you created).  Compute the error rate on the test data using these.  It should be significantly lower.
# 
# This might take a while.

# In[85]:

get_ipython().run_cell_magic('time', '', 'lab_assigned = []\nfor img in test_imgs:\n    lab_assigned.append(closest(img,TrainMore))\nlab_assigned = np.array(lab_assigned)\nerr_rate = np.mean(lab_assigned != test_labels)\nprint err_rate')

