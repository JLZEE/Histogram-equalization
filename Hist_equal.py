# Histogram-equalization use python
# In this program, we do histogram equalization use python and see the operating speed

import numpy as np
import time
import pickle

# First calculate the histogram of the image
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins

# Second calculate the CDF of histogram
def scan(x):
    CDF = np.zeros(256, np.uint32)
    for v in xrange(0,256):
        try:
            CDF[v] = CDF[v-1] + x[v]
        except:
            CDF[v] = x[v]
    return CDF

# Finally calculate the equalization matrix
def equl(img,cdf):
    
    flatimg = img.flat                                              #flat the image
    he_img = np.zeros(len(flatimg), np.uint8)                       #create new equalized image
    
    for v in xrange(1,len(flatimg)):
        he_img[v] = round(255*(cdf[flatimg[v]] - min(cdf))/(len(flatimg) - min(cdf)))
    return he_img

# read the image
img=np.fromfile('orimg.bin',dtype=np.uint8)

start = time.time()             # start counting time
hist_py = hist(img)             # get the histogram of image
scan_py = scan(hist_py)         # get the cdf of image
he_img = equl(img,scan_py)      # use the cdf to process the original image
time = time.time()-start        # calculate the processing time

he_img.tofile("heimg.bin")      # save the image
print 'histogram: ', hist_py,'\n', 'scan: ', scan_py, '\n', 'Operating time', time
