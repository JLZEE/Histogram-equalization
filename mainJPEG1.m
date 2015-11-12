% This program does not include Huffman coding
% This program is used to simulate DFT trasform of an image and then compress image by quantization.

SARimage = 'mapserv15.jpg';
SARgray = rgb2gray(im2double(imread(SARimage)));

[High,Wide] = size(SARgray);
subplot(1,3,1); imshow(SARgray); hold on; title('Original image');
SARgray = SARgray*256;

m=[16 11 10 16 24 40 51 61;
   12 12 14 19 26 58 60 55;
   14 13 16 24 40 57 69 56;
   14 17 22 29 51 87 80 62;
   18 22 37 56 68 109 103 77;
   24 35 55 64 81 104 113 92;
   49 64 78 87 103 121 120 101;
   72 92 95 98 112 100 103 99];

blockH = floor((High-1)/8)+1;
blockW = floor((Wide-1)/8)+1;

SARdct = ones(High,Wide);
for i = 1:blockH
    for j = 1:blockW
        blockmat = zeros(8);
        for p = 1:8
            for q = 1:8
                if (((i-1)*8+p)<=High)&&(((j-1)*8+q)<=Wide)
                    blockmat(p,q) =  SARgray((i-1)*8+p,(j-1)*8+q);
                end             
            end
        end
        dctmat = dct2(blockmat);
        for p = 1:8
            for q = 1:8
                if (((i-1)*8+p)<=High)&&(((j-1)*8+q)<=Wide)
                    SARdct((i-1)*8+p,(j-1)*8+q) = round(dctmat(p,q)/m(p,q));
                end               
            end
        end
    end
end

subplot(1,3,2); imshow(SARdct); hold on; title('DCT & Quantization');

SARinv = ones(High,Wide);
for i = 1:blockH
    for j = 1:blockW
        blockmat = zeros(8);
        for p = 1:8
            for q = 1:8
                if (((i-1)*8+p)<=High)&&(((j-1)*8+q)<=Wide)
                    blockmat(p,q) =  SARdct((i-1)*8+p,(j-1)*8+q)*m(p,q);
                end                
            end
        end
        idctmat = idct2(blockmat);
        for p = 1:8
            for q = 1:8
                if (((i-1)*8+p)<=High)&&(((j-1)*8+q)<=Wide)
                    SARinv((i-1)*8+p,(j-1)*8+q) = idctmat(p,q);
                end              
            end
        end
    end
end

SARinv = SARinv/256;
subplot(1,3,3); imshow(SARinv); title('Compressed image');
