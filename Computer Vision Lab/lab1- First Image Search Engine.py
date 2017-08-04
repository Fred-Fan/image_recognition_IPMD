# import the necessary packages
import numpy as np
import cv2
#from pyimagesearch.rgbhistogram import RGBHistogram
import argparse
import pickle
import glob
import cv2


class RGBHistogram:
    def __init__(self, bins):
        # store the number of bins the histogram will use
        self.bins = bins

    def describe(self, image):
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist,hist)

        # return out 3D histogram as a flattened array
        return hist.flatten()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', "--dataset", help = "Path to the directory that contains the images to be indexed", required = True)
ap.add_argument("-i", "--index",  help = "Path to where the computed index will be stored", required = True)
args = ap.parse_args()


index = {}

# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
desc = RGBHistogram([8, 8, 8])


# In[8]:

for imagePath in glob.glob("dataset/lab1/*.jpeg"):
    # extract our unique image ID (i.e. the filename)
    k = imagePath[imagePath.rfind("\\")+1:imagePath.rfind("j")-1]
    print(k)
 
    # load the image, describe it using our RGB histogram
    # descriptor, and update the index
    image = cv2.imread(imagePath)
    features = desc.describe(image)
    index[k] = features


# In[9]:

pickle.dump(index, open('index.dmp', 'wb'))


# In[10]:

# import the necessary packages
import numpy as np
class Searcher:
    def __init__(self, index):
        # store our index of images
        self.index = index
 
    def search(self, queryFeatures):
        # initialize our dictionary of results
        results = {}
 
        # loop over the index
        for (k, features) in self.index.items():
            # compute the chi-squared distance between the features
            # in our index and our query features -- using the
            # chi-squared distance which is normally used in the
            # computer vision field to compare histograms
            d = self.chi2_distance(features, queryFeatures)
 
            # now that we have the distance between the two feature
            # vectors, we can udpate the results dictionary -- the
            # key is the current image ID in the index and the
            # value is the distance we just computed, representing
            # how 'similar' the image in the index is to our query
            results[k] = d
 
        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])
 
        # return our results
        return results
 
    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
 
        # return the chi-squared distance
        return d


# In[11]:

# import the necessary packages
#from pyimagesearch.searcher import Searcher
import numpy as np
import argparse
import pickle
import cv2
 
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory that contains the images we just indexed")
#ap.add_argument("-i", "--index", required = True, help = "Path to where we stored our index")
#args = vars(ap.parse_args())
 
# load the index and initialize our searcher
index = pickle.loads(open('index.dmp','rb').read())
searcher = Searcher(index)


# In[12]:

# loop over images in the index -- we will use each one as
# a query image
#for (query, queryFeatures) in index.items():
#    # perform the search using the current query
#    results = searcher.search(queryFeatures)
 
    


# In[15]:

index['20']


# In[16]:

# load the query image and display it
#path = args["dataset"] + "/%s" % (query)

query = '20'
path = 'dataset\\'+query+'.jpeg'
results = searcher.search(index[query])
queryImage = cv2.imread(path)
cv2.imshow("Query", queryImage)
print("query: %s" % (query))


# In[19]:

results[1]


# In[21]:


    # initialize the two montages to display our results --
    # we have a total of 25 images in the index, but let's only
    # display the top 10 results; 5 images per montage, with
    # images that are 400x166 pixels
montageA = np.zeros((87 * 5, 204, 3), dtype = "uint8")
montageB = np.zeros((87 * 5, 204, 3), dtype = "uint8")
 
    # loop over the top ten results
for j in range(0, 10):
        # grab the result (we are using row-major order) and
        # load the result image
    (score, imageName) = results[j]
        #path = args["dataset"] + "/%s" % (imageName)
    path = 'dataset\\'+imageName+'.jpeg'
    result = cv2.imread(path)
    print("\t%d. %s : %.3f" % (j + 1, imageName, score))
 
        # check to see if the first montage should be used
    if j < 5:
        montageA[j * 87:(j + 1) * 87, :] = result
 
        # otherwise, the second montage should be used
    else:
        montageB[(j - 5) * 87:((j - 5) + 1) * 87, :] = result
 
    # show the results
cv2.imshow("Results 1-5", montageA)
cv2.imshow("Results 6-10", montageB)
cv2.waitKey(0)


# In[ ]:




# In[ ]:



