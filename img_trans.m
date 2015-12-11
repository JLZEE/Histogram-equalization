% this program could be used to transform image from .jpg to .bin, which could be handled easily
% the image and image size are saved seperately

clc;
clear all;

SARimage = 'mapserv7.jpg';
SARgray = rgb2gray(im2double(imread(SARimage)));
[SARgrayH,SARgrayW] = size(SARgray);
SARgray = round(SARgray.*256);
SARgray = uint8(SARgray);

fip = fopen('E:\orimgS.bin','wb');
fwrite(fip, SARgray, 'uint8');
fclose(fip);
fip = fopen('E:\orimg_size.bin','wb');
fwrite(fip, [SARgrayH,SARgrayW], 'uint16');
fclose(fip);

fid1=fopen('E:\sobelimg_py.bin','rb');
[SBimg1,COUNT1]=fread(fid1, [SARgrayH,SARgrayW] ,'*uint8');
fid2=fopen('E:\sobelimg_op.bin','rb');
[SBimg2,COUNT2]=fread(fid2,  [SARgrayH,SARgrayW]  ,'*uint8');
 
 subplot(1,3,1); imshow(SARgray); title('Original image'); hold on;
 subplot(1,3,2); imshow(SBimg1); title('Python image'); hold on;
 subplot(1,3,3); imshow(SBimg2); title('OpenCL image');
 
