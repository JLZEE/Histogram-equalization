"""
Basic 2d JEPG
"""

import time

import pyopencl as cl
import pyopencl.array
import numpy as np
import math

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

# Get the image
img=np.fromfile('orimg.bin',dtype=np.uint8)
img_size = np.fromfile('orimg_size.bin',dtype=np.uint16)
R = img_size[1]
C = img_size[0]
pi = 3.14159265
img = img.reshape(R, C)

Rfix = ((R-1)/8+1)*8
Cfix = ((C-1)/8+1)*8
imgfix = np.zeros([Rfix,Cfix],np.uint8)     # Change the image size 
imgDCT = np.zeros([Rfix,Cfix],np.int)       # Create a matrix to save DCT of the image
for i in xrange(0,R):
    for j in xrange(0,C):
        imgfix[i,j] = img[i,j]
		
Rgroup = Rfix/8
Cgroup = Cfix/8

Qmatrix = np.fromfile('Qmatrix.bin',dtype=np.uint16)
Qmatrix = Qmatrix.reshape(8,8)
Q = np.transpose(Qmatrix)

imgQti = np.zeros([Rfix,Cfix],np.int)
# DCT transform
##start = time.time()

##for i in xrange(0,Rgroup):
##    for j in xrange(0,Cgroup):
##        for u in xrange(0,8):
##            for v in xrange(0,8):
##                if u==0:
##                    alpha_u = 0.3536
##                else:
##                    alpha_u = 0.5
##                if v==0:
##                    alpha_v = 0.3536
##                else:
 ##                   alpha_v = 0.5
##                inner_sum = 0
 ##               for y in xrange(0,8):
 ##                   for x in xrange(0,8):
##                        inner_sum += imgfix[i*8+y,j*8+x]*math.cos((2*y+1)*u*pi/16)*math.cos((2*x+1)*v*pi/16)
##                imgDCT[i*8+u,j*8+v] = alpha_u*alpha_v*inner_sum

# imgDCT.tofile("imgDCT.bin")
# Quantilization       
##for i in xrange(0,Rgroup):
 ##   for j in xrange(0,Cgroup):
 ##       for u in xrange(0,8):
 ##           for v in xrange(0,8):
 ##               imgQti[i*8+u,j*8+v] = round(imgDCT[i*8+u,j*8+v]/Q[u,v])

# imgQti.tofile("imgQti.bin")
# I-quantilization
imgIQ = np.zeros([Rfix,Cfix],np.int)

##for i in xrange(0,Rgroup):
##    for j in xrange(0,Cgroup):
##        for u in xrange(0,8):
##            for v in xrange(0,8):
##                imgIQ[i*8+u,j*8+v] = imgQti[i*8+u,j*8+v]*Q[u,v]

# imgIQ.tofile("imgIQ.bin")
# I-DCT
imgIDCT = np.zeros([Rfix,Cfix],np.uint8)
##for i in xrange(0,Rgroup):
##    for j in xrange(0,Cgroup):
##        for u in xrange(0,8):
##            for v in xrange(0,8):
##                inner_sum = 0
##                for y in xrange(0,8):
##                    for x in xrange(0,8):
##                        if y==0:
##                            alpha_y = 0.3536
##                        else:
##                            alpha_y = 0.5
##                        if x==0:
##                            alpha_x = 0.3536
##                        else:
##                            alpha_x = 0.5
##                        inner_sum +=  alpha_x*alpha_y*imgIQ[i*8+y,j*8+x]*math.cos((2*u+1)*y*pi/16)*math.cos((2*v+1)*x*pi/16)
##                imgIDCT[i*8+u,j*8+v] = round(inner_sum)
				
##print 'Python time: ',time.time()-start, '\n'
##imgIDCTrec_py = np.zeros([R,C],np.uint8)
##for i in xrange(0,R):
##    for j in xrange(0,C):
##        imgIDCTrec_py[i,j] = imgIDCT[i,j]
# Write the image operated by Python
##imgIDCTrec_py.tofile("imgIDCT_py.bin")

# Now start OpenCL!
# Kernel 
func1 = cl.Program(ctx, """
__kernel __attribute__((reqd_work_group_size(8,8,1)))
void func1(__global unsigned char *wbopimg, __global unsigned char *qtimat, __global unsigned char *jgopimg,
                   const unsigned int Row, const unsigned int Col) {
	unsigned int globalIdy = get_global_id(0);	unsigned int globalIdx = get_global_id(1);
	unsigned int localIdy = get_local_id(0);	unsigned int localIdx = get_local_id(1);
	unsigned int groupIdy = get_group_id(0);	unsigned int groupIdx = get_group_id(1);
	__local float img[8*8];
	__local float DCT[8*8];
	__local float Qtmat[8*8];
	__local int IQmat[8*8];
	__local float IDCT[8*8];
	__local float Qvmat[8*8];
	
	float alpha_u;		float alpha_v;
	float alpha_y;		float alpha_x;
	
	for (int i=0; i<8; i++){
		for (int j=0; j<8; j++){
			img[i*8+j] = 0;
			DCT[i*8+j] = 0;
			Qtmat[i*8+j] = 0;
			IQmat[i*8+j] = 0;
			IDCT[i*8+j] = 0;
			Qvmat[i*8+j] = 0;
		}
	}
	for (int i=0; i<8; i++){
		for (int j=0; j<8; j++){	
			Qvmat[i*8+j] = qtimat[i*8+j];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// copy the image
	int idx_y = groupIdy * 8 + localIdy;
	int idx_x = groupIdx * 8 + localIdx;
	img[localIdy*8+localIdx] = wbopimg[idx_y*Col+idx_x];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	

	// DCT
	float inner_sum1 = 0.0;
	for (int i=0; i<8; i++){
		for (int j=0; j<8; j++){
			inner_sum1 += img[i*8+j]*cos((2*(float)(i)+1)*(float)(localIdy)*3.14159265/16)*cos((2*(float)(j)+1)*(float)(localIdx)*3.14159265/16);
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	if (localIdy==0)
		alpha_y = 0.3536;
	else
		alpha_y = 0.5;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localIdx==0)
		alpha_x = 0.3536;
	else
		alpha_x = 0.5;
	barrier(CLK_LOCAL_MEM_FENCE);
	DCT[localIdy*8+localIdx] = alpha_y * alpha_x * inner_sum1;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Quantilization
	Qtmat[localIdy*8+localIdx] = DCT[localIdy*8+localIdx]/Qvmat[localIdy*8+localIdx];
	barrier(CLK_LOCAL_MEM_FENCE);

	// I-quantilization
	IQmat[localIdy*8+localIdx] = Qtmat[localIdy*8+localIdx]*Qvmat[localIdy*8+localIdx];
	barrier(CLK_LOCAL_MEM_FENCE);	
	
	// I-DCT
	float inner_sum2 = 0.0;
	for (int u=0; u<8; u++){
		for (int v=0; v<8; v++){
			if (u==0)
				alpha_u = 0.3536;
			else
				alpha_u = 0.5;
			barrier(CLK_LOCAL_MEM_FENCE);
			if (v==0)
				alpha_v = 0.3536;
			else
				alpha_v = 0.5;
			barrier(CLK_LOCAL_MEM_FENCE);
			inner_sum2 += alpha_u*alpha_v*(float)(IQmat[u*8+v])*cos((2*(float)(localIdy)+1)*(float)(u)*3.14159265/16)*cos((2*(float)(localIdx)+1)*(float)(v)*3.14159265/16);
		}
	}
	IDCT[localIdy*8+localIdx] = inner_sum2;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// rewrite the image
	idx_y = groupIdy * 8 + localIdy;
	idx_x = groupIdx * 8 + localIdx;
	jgopimg[idx_y*Col+idx_x] = IDCT[localIdy*8+localIdx];
	barrier(CLK_LOCAL_MEM_FENCE);
}
""").build().func1

func1.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32])				   

wbopimg_gpu = cl.array.to_device(queue, imgfix)
qtimat_gpu = cl.array.to_device(queue, Q)
jgopimg_gpu = cl.array.zeros(queue, imgIDCT.shape, np.uint8)
start = time.time()
func1(queue, imgfix.shape, (8,8), wbopimg_gpu.data, qtimat_gpu.data, jgopimg_gpu.data, Rfix, Cfix)	
print 'opencl time: ',time.time()-start, '\n'
jgopimg = np.zeros([Rfix,Cfix],np.uint8)
jgopimg = jgopimg_gpu.get()
jgopimgunfix = np.zeros([R,C],np.uint8)
for x in xrange(0,R):
	for y in xrange(0,C):
		jgopimgunfix[x,y] = jgopimg[x,y]
		
jgopimgunfix.tofile("jpeg_op.bin")

print 'Orimg: ', img
print 'Op_img: ', jgopimg
