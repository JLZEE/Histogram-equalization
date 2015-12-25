"""
Basic 2d Sobel operator
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
# Get the image
img=np.fromfile('orimgS.bin',dtype=np.uint8)
img_size = np.fromfile('orimg_size.bin',dtype=np.uint16)
R = img_size[1]
C = img_size[0]
img = img.reshape(R, C)

tile_size = 16
Rfix = R
Cfix = C
# Set up a matrix for operating convolution
wb_img = np.zeros([R+2,C+2],np.uint8)
wbop_img = np.zeros([Rfix+2,Cfix+2],np.uint8)
sbop_img = np.zeros([Rfix+2,Cfix+2],np.uint8)
for i in xrange(1,R+1):
    for j in xrange(1,C+1):
        wb_img[i,j] = img[i-1,j-1]
	wbop_img[i,j] = img[i-1,j-1]
# Set up Gx and Gy
Gx = np.zeros([R+2,C+2],np.int)
Gy = np.zeros([R+2,C+2],np.int)
Gx_op = np.zeros([Rfix+2,Cfix+2],np.int)
Gy_op = np.zeros([Rfix+2,Cfix+2],np.int)

Soble_x = np.zeros([3,3],np.int)
Soble_x[0,0] = -1
Soble_x[0,2] = 1
Soble_x[1,0] = -2
Soble_x[1,2] = 2
Soble_x[2,0] = -1
Soble_x[2,2] = 1
Soble_y = np.zeros([3,3],np.int)
Soble_y[0,0] = -1
Soble_y[0,1] = -2
Soble_y[0,2] = -1
Soble_y[2,0] = 1
Soble_y[2,1] = 2
Soble_y[2,2] = 1
# Kernel 
func1 = cl.Program(ctx, """
__kernel void func1(__global unsigned char *wbopimg,__global unsigned char *sbopimg, __global int *Gxop,__global int *Gyop,
                   __global int *sbox, __global int *sboy, const unsigned int Row, const unsigned int Col) {
    unsigned int Idy = get_global_id(0);	unsigned int Idx = get_global_id(1);
    unsigned int idx;
	unsigned int V11; unsigned int V12; unsigned int V13; 
	unsigned int V21; unsigned int V22; unsigned int V23;
	unsigned int V31; unsigned int V32; unsigned int V33;

    if((Idx>0)&&(Idx<=Col)&&(Idy>0)&&(Idy<=Row)){
		V11 = (Idx-1)+(Idy-1)*(Col+2); V12 = (Idx)+(Idy-1)*(Col+2); V13 = (Idx+1)+(Idy-1)*(Col+2);
		V21 = (Idx-1)+(Idy)*(Col+2);   V22 = (Idx)+(Idy)*(Col+2);   V23 = (Idx+1)+(Idy)*(Col+2);
		V31 = (Idx-1)+(Idy+1)*(Col+2); V32 = (Idx)+(Idy+1)*(Col+2); V33 = (Idx+1)+(Idy+1)*(Col+2);
		Gxop[V22] = wbopimg[V11]*sbox[0]+wbopimg[V13]*sbox[2]+wbopimg[V21]*sbox[3]+wbopimg[V23]*sbox[5]+wbopimg[V31]*sbox[6]+wbopimg[V33]*sbox[8];
		Gyop[V22] = wbopimg[V11]*sboy[0]+wbopimg[V12]*sboy[1]+wbopimg[V13]*sboy[2]+wbopimg[V31]*sboy[6]+wbopimg[V32]*sboy[7]+wbopimg[V33]*sboy[8];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if((Idx>0)&&(Idx<=Col)&&(Idy>0)&&(Idy<=Row)){
		idx = (Idx)+(Idy)*(Col+2);
		if (Gxop[idx]*Gxop[idx]+Gyop[idx]*Gyop[idx]>40000)
			sbopimg[idx] = wbopimg[idx];
		else
			sbopimg[idx] = 255;
	}
}
""").build().func1

func1.set_scalar_arg_dtypes([None, None, None, None, None, None, np.uint32, np.uint32])

func2 = cl.Program(ctx, """
__kernel void func2(__global unsigned char *wbopimg,__global unsigned char *sbopimg, __global int *Gxop,__global int *Gyop,
                   const unsigned int Row, const unsigned int Col) {
    unsigned int Idy = get_global_id(0);	unsigned int Idx = get_global_id(1);
    unsigned int idx;
	unsigned int V11; unsigned int V12; unsigned int V13; 
	unsigned int V21; unsigned int V22; unsigned int V23;
	unsigned int V31; unsigned int V32; unsigned int V33;

    if((Idx>0)&&(Idx<=Col)&&(Idy>0)&&(Idy<=Row)){
		V11 = (Idx-1)+(Idy-1)*(Col+2); V12 = (Idx)+(Idy-1)*(Col+2); V13 = (Idx+1)+(Idy-1)*(Col+2);
		V21 = (Idx-1)+(Idy)*(Col+2);   V22 = (Idx)+(Idy)*(Col+2);   V23 = (Idx+1)+(Idy)*(Col+2);
		V31 = (Idx-1)+(Idy+1)*(Col+2); V32 = (Idx)+(Idy+1)*(Col+2); V33 = (Idx+1)+(Idy+1)*(Col+2);
		Gxop[V22] = (wbopimg[V13]+2*wbopimg[V23]+wbopimg[V33])-(wbopimg[V11]+2*wbopimg[V21]+wbopimg[V31]);
		Gyop[V22] = (wbopimg[V11]+2*wbopimg[V12]+wbopimg[V13])-(wbopimg[V31]+2*wbopimg[V32]+wbopimg[V33]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if((Idx>0)&&(Idx<=Col)&&(Idy>0)&&(Idy<=Row)){
		idx = (Idx)+(Idy)*(Col+2);
		if (Gxop[idx]*Gxop[idx]+Gyop[idx]*Gyop[idx]>40000)
			sbopimg[idx] = wbopimg[idx];
		else
			sbopimg[idx] = 255;
	}
}
""").build().func2

func2.set_scalar_arg_dtypes([None, None, None, None, np.uint32, np.uint32])

func3 = cl.Program(ctx, """
__kernel void func3(__global unsigned char *wbopimg,__global unsigned char *sbopimg, __global int *Gxop,__global int *Gyop,
                   const unsigned int Row, const unsigned int Col) {
    unsigned int Idy = get_global_id(0);	unsigned int Idx = get_global_id(1);
	__local int img_block[3*3];
    unsigned int idx;

    if((Idx>0)&&(Idx<=Col)&&(Idy>0)&&(Idy<=Row)){
		img_block[0] = wbopimg[(Idx-1)+(Idy-1)*(Col+2)]; 	img_block[1] = wbopimg[(Idx)+(Idy-1)*(Col+2)]; 
		img_block[2] = wbopimg[(Idx+1)+(Idy-1)*(Col+2)];	img_block[3] = wbopimg[(Idx-1)+(Idy)*(Col+2)];   
		img_block[4] = wbopimg[(Idx)+(Idy)*(Col+2)];   		img_block[5] = wbopimg[(Idx+1)+(Idy)*(Col+2)];
		img_block[6] = wbopimg[(Idx-1)+(Idy+1)*(Col+2)]; 	img_block[7] = wbopimg[(Idx)+(Idy+1)*(Col+2)]; 
		img_block[8] = wbopimg[(Idx+1)+(Idy+1)*(Col+2)];
		
		Gxop[(Idx)+(Idy)*(Col+2)] = (img_block[2]+2*img_block[5]+img_block[8])-(img_block[0]+2*img_block[3]+img_block[6]);
		Gyop[(Idx)+(Idy)*(Col+2)] = (img_block[0]+2*img_block[1]+img_block[2])-(img_block[6]+2*img_block[7]+img_block[8]);
		
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if((Idx>0)&&(Idx<=Col)&&(Idy>0)&&(Idy<=Row)){
		idx = (Idx)+(Idy)*(Col+2);
		if ((Gxop[idx]*Gxop[idx]+Gyop[idx]*Gyop[idx])>22500)
			sbopimg[idx] = wbopimg[idx];
		else
			sbopimg[idx] = 255;
	}
}
""").build().func3

func3.set_scalar_arg_dtypes([None, None, None, None, np.uint32, np.uint32])


# Start python
start = time.time()
for x in xrange(1,R+1):
    for y in xrange(1,C+1):
        Gx[x,y] =(wb_img[x+1,y-1]+2*wb_img[x+1,y]+wb_img[x+1,y+1])-(wb_img[x-1,y-1]+2*wb_img[x-1,y]+wb_img[x-1,y+1])
        Gy[x,y] =(wb_img[x-1,y-1]+2*wb_img[x,y-1]+wb_img[x+1,y-1])-(wb_img[x-1,y+1]+2*wb_img[x,y+1]+wb_img[x+1,y+1])
sobel_img = np.zeros([R,C],np.uint8)
for x in xrange(0,R):
    for y in xrange(0,C):
        if Gx[x+1,y+1]*Gx[x+1,y+1] + Gy[x+1,y+1]*Gy[x+1,y+1] > 40000:
            sobel_img[x,y] = img[x,y]
        else:
            sobel_img[x,y] = 255
print 'python1 time: ',time.time()-start, '		\n'
sobel_img.tofile("sobelimg_py.bin")

# Set up input for kernel1
wbopimg_gpu = cl.array.to_device(queue, wbop_img)
soblex_gpu = cl.array.to_device(queue, Soble_x)
sobley_gpu = cl.array.to_device(queue, Soble_y)
sbopimg_gpu = cl.array.zeros(queue, sbop_img.shape, np.uint8)
Gxop_gpu = cl.array.zeros(queue, Gx_op.shape, np.int)
Gyop_gpu = cl.array.zeros(queue, Gy_op.shape, np.int)
start = time.time()
func1(queue, wbop_img.shape, None, wbopimg_gpu.data, sbopimg_gpu.data, Gxop_gpu.data, Gxop_gpu.data, soblex_gpu.data, sobley_gpu.data, Rfix, Cfix)	
print 'opencl1 time: ',time.time()-start, '		\n'

# Set up input for kernel2
wbopimg_gpu = cl.array.to_device(queue, wbop_img)
sbopimg_gpu = cl.array.zeros(queue, sbop_img.shape, np.uint8)
Gxop_gpu = cl.array.zeros(queue, Gx_op.shape, np.int)
Gyop_gpu = cl.array.zeros(queue, Gy_op.shape, np.int)
start = time.time()
func2(queue, wbop_img.shape, None, wbopimg_gpu.data, sbopimg_gpu.data, Gxop_gpu.data, Gxop_gpu.data, Rfix, Cfix)	
print 'opencl2 time: ',time.time()-start, '		\n'

# Set up input for kernel3
wbopimg_gpu = cl.array.to_device(queue, wbop_img)
sbopimg_gpu = cl.array.zeros(queue, sbop_img.shape, np.uint8)
Gxop_gpu = cl.array.zeros(queue, Gx_op.shape, np.int)
Gyop_gpu = cl.array.zeros(queue, Gy_op.shape, np.int)
start = time.time()
func3(queue, wbop_img.shape, None, wbopimg_gpu.data, sbopimg_gpu.data, Gxop_gpu.data, Gxop_gpu.data, Rfix, Cfix)	
print 'opencl3 time: ',time.time()-start, '		\n'

sbop_img = sbopimg_gpu.get()
sbop_imgunfix = np.zeros([R,C],np.uint8)
for x in xrange(0,R):
	for y in xrange(0,C):
		sbop_imgunfix[x,y] = sbop_img[x+1,y+1]
		
sbop_imgunfix.tofile("sobelimg_op.bin")
print sbop_imgunfix,'	',sobel_img
