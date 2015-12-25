#!/usr/bin/env python

"""
Basic 2d histogram equalization.
"""

import time

import pyopencl as cl
import pyopencl.array
import numpy as np

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

# Compute histogram in Python:
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins
	
def scan(x):
    CDF = np.zeros(256, np.uint32)
    for v in xrange(0,256):
        try:
            CDF[v] = CDF[v-1] + x[v]
        except:
            CDF[v] = x[v]
    return CDF

def equl(img,cdf):
    
    #flat the image
    flatimg = img.flat
    
    #create new equalized image
    he_img = np.zeros(len(flatimg), np.uint8)
    for v in xrange(0,len(flatimg)):
        he_img[v] = 255*(cdf[flatimg[v]] - min(cdf))/(len(flatimg) - min(cdf))

    return he_img
# Create input image containing 8-bit pixels; the image contains N = R*C bytes;
#P = 1024
#R = P
#C = P
#N = R*C
#img = np.random.randint(0, 255, N).astype(np.uint8).reshape(R, C)
img = np.fromfile('orimg.bin',dtype=np.uint8)
fimg = img.flat
N = len(fimg)
groupNum = (N-1)/256 + 1
Nfix = groupNum*256
imgfix = np.zeros((Nfix),dtype=np.uint8)

for i in xrange(0,N):
	imgfix[i] = fimg[i]
bin = np.zeros((256),dtype=np.uint16)

# Kernel of Step 1 Histogram
func1 = cl.Program(ctx, """
__kernel void func1(__global unsigned char *img, __global unsigned int *bins,
                   const unsigned int N) {
    unsigned int i = get_global_id(0);
    unsigned int k;
    __local unsigned int bins_loc[256];

    for (k=0; k<256; k++)
        bins_loc[k] = 0;
    if (i<N)
        atomic_add(&(bins_loc[img[i]]), 1);				// use simple stomic add in each bin
    barrier(CLK_LOCAL_MEM_FENCE);
    for (k=0; k<256; k++)
        atomic_add(&bins[k], bins_loc[k]);
}
""").build().func1

func1.set_scalar_arg_dtypes([None, None, np.uint32])

# Kernel of Step 2 CDF
func2 = cl.Program(ctx, """
__kernel void func2(__global unsigned int *hist, __global unsigned int *cdf) {
    unsigned int i = get_global_id(0);
    unsigned int k;		unsigned int binsize = 256;
    //unsigned int bins_loc[256];

    for (k=0; k<256; k++)
        cdf[k] = hist[k];
	
	for (int stride=1; stride<=binsize/2; stride*=2){
		int index = (i+1)*stride*2 - 1;
		if (index<binsize)
			cdf[index] += cdf[index-stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	for (int stride=binsize/4; stride>=1; stride/=2){
		barrier(CLK_LOCAL_MEM_FENCE);
		int index = (i+1)*stride*2 - 1;
		if((index+stride)<binsize)
			cdf[index+stride] += cdf[index];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//for (k=0; k<256; k++)
	//	cdf[k] = bins_loc[k];

}
""").build().func2

func2.set_scalar_arg_dtypes([None, None])

# Kernel of Step 3 Equlization
func3 = cl.Program(ctx, """
__kernel void func3(__global unsigned int *scanop, __global unsigned char *heimg,
					__global unsigned char *img, const unsigned int N) {
    unsigned int globalId = get_global_id(0);
	unsigned int localId = get_local_id(0);		unsigned int groupId = get_group_id(0);
    unsigned int k;		unsigned int binsize = 256;
    volatile unsigned __local char img_loc[256];

	for (k=0; k<256; k++){
		img_loc[k] = img[groupId*256 + k];
		barrier(CLK_LOCAL_MEM_FENCE);
		img_loc[k] = 255*(scanop[img_loc[k]]-scanop[0])/(N-scanop[0]);
		barrier(CLK_LOCAL_MEM_FENCE);
		heimg[k + groupId*256] = img_loc[k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

}
""").build().func3

func3.set_scalar_arg_dtypes([None, None, None, np.uint32])


# Time Python function:
start = time.time()
h_py = hist(img)
scan_py = scan(h_py)
heimg_py = equl(img,scan_py)
print 'python time: ',time.time()-start, '\n'

# Time OpenCL function:
img_gpu = cl.array.to_device(queue, img)
imgfix_gpu = cl.array.to_device(queue, imgfix)
bin_gpu = cl.array.zeros(queue, 256, np.uint32)
cdf_gpu = cl.array.zeros(queue, 256, np.uint32)
heimgfix_gpu = cl.array.zeros(queue, Nfix, np.uint8)

start = time.time()
func1(queue, (N,), (1,), img_gpu.data, bin_gpu.data, N)				# Step 1, histogram
print 'OpenCL step1 time: ',time.time()-start, '\n'

h_op =  bin_gpu.get()
hist = h_op
hist_gpu = cl.array.to_device(queue, hist)

start = time.time()
func2(queue, hist.shape , None, hist_gpu.data, cdf_gpu.data)		# Step 2, CDF of histogram
print 'OpenCL step2 time: ',time.time()-start, '\n'

cdf_op =  cdf_gpu.get()
scanop = cdf_op
scanop_gpu = cl.array.to_device(queue, scanop)

start = time.time()
func3(queue, (Nfix,1) , (256,1), scanop_gpu.data, heimgfix_gpu.data, imgfix_gpu.data, N)	# Step 3, Equlizate the image
print 'OpenCL step3 time: ',time.time()-start, '\n'

heimgfix_op =  heimgfix_gpu.get()
heimg_op = np.zeros((N),dtype=np.uint8)
for i in xrange(0,N):
	heimg_op[i] = heimgfix_op[i]

print heimg_py,'	',heimg_op


heimg_op.tofile("heimg_op.bin")
heimg_py.tofile("heimg_py.bin")
