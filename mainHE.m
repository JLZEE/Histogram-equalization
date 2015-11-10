% Here I use a SARimage to do the histogram equalization. You can change it to your own images.

SARimage = 'mapserv16.jpg';
SARgray = rgb2gray(im2double(imread(SARimage)));
[SARgrayH,SARgrayW] = size(SARgray);
subplot(2,2,1); imshow(SARgray); hold on;       % Original image
subplot(2,2,2); hist(SARgray); hold on;         % Histogram of original image

SARhistomat = hist(SARgray,256);
SARhistogram = sum(SARhistomat')';
cdf = SARhistogram;                             % Calculate the cdf of histogram
for i = 2:256
    cdf(i) = cdf(i)+cdf(i-1);
end

TN = SARgrayH*SARgrayW;
HEimage = ones(SARgrayH,SARgrayW);              % Change every pixel based on the value of cdf
for i = 1:SARgrayH
    for j = 1:SARgrayW
        HEindex = round(SARgray(i,j)*256);
        HEimage(i,j) = round((cdf(HEindex)-min(cdf))/(TN-min(cdf))*255);
        HEimage(i,j) = HEimage(i,j)/256;
    end
end

subplot(2,2,3); imshow(HEimage); hold on;       % Processed image
subplot(2,2,4); hist(HEimage);                  % Histogram of processed image
