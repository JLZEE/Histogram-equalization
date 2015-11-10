% Here I use a SAR image for example, you could use your images.

SARimage = 'mapserv1.jpg';
SARgray = rgb2gray(im2double(imread(SARimage)));

%Use filter to denoise
%SARgray = medfilt2(imnoise(SARgray,'salt & pepper'));

subplot(1,2,1); imshow(SARgray); hold on;
[SARgrayH,SARgrayW] = size(SARgray);

sobelX = [-1,0,1;-2,0,2;-1,0,1];
sobelY = [1,2,1;0,0,0;-1,-2,-1];

Gx = conv2(SARgray,sobelX);
Gy = conv2(SARgray,sobelY);

G1 = zeros(SARgrayH+2,SARgrayW+2);              % Either G1 and G2 could detect the edge of image.
for i = 1:SARgrayH+2
    for j = 1:SARgrayW+2
        G1(i,j) = (Gx(i,j)^2+Gy(i,j)^2)^0.5;
    end
end
G2 = abs(Gy)+abs(Gx);

G1sobel = ones(SARgrayH,SARgrayW);
G2sobel = ones(SARgrayH,SARgrayW);

for i = 1:SARgrayH
    for j = 1:SARgrayW
        if G1(i+1,j+1)>0.8                      % You could change index "0.8", it depends on your image.
            G1sobel(i,j) = SARgray(i,j);
        end
        if G2(i+1,j+1)>0.8
            G2sobel(i,j) = SARgray(i,j);
        end
    end
end
subplot(1,2,2); imshow(G2sobel);                % Either G1sobel and G2sobel is the final result of edge detection
