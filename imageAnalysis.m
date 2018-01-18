% Fourier analysis of image 
% wavelet function has used to analysis the Image  
% svd technique has implemented to analys the Image  
% Coded by Abol Basher 
% Code is written at 26 October 2017
% Updated at 9 November 2017
%%
%Explain when you would use CIC filtering and when you would polyphase filtering for decimation
% CIC filter doesn't have Multiplyer where polyphase filter has multiplyer
% existing of Multylyer is increase the cost of the filter. So we choose 
%CIC Filter over PolyPhase Filter 

%Explain the difference between hard filtering, soft filtering and
%thresholding in the Fourier domain?
% The reason for doing the filtering in the frequency domain is generally because
% it is computationally faster to perform two 2D Fourier transforms
% and a filter multiply than to perform a convolution in the image (spatial) domain.
% Image processing procedures are usually carried out in the spatial domain where
% the images are acquired and presented/utilized. The linear nature of 
% Fourier transform allows only those operations that are linear to be mapped into the
% frequency domain. In contrast, nonlinear operations and manipulations
% cannot be realized directly in the frequency domain
% source:IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 10, NO. 5, MAY 2001
%(Thresholding Implemented in the Frequency Domain)
% Hard thresholding can be described as the usual process of setting to zero the
% elements whose absolute values are lower than the threshold.
% Soft thresholding is an extension of hard thresholding, first
% setting to zero the elements whose absolute values are lower
% than the threshold, and then shrinking the nonzero
% coefficientstowards 0.
% source: IACSIT International Journal of Engineering and Technology Vol.1,No.5,December,2009
% ISSN: 1793-8236 
%Performance Evaluation of Different Thresholding Methods in Time Adaptive Wavelet Based Speech Enhancement

%wavelet point of view:http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xlghtmlnode93.html

% what is the effect of mother wavelet for analysis of data or image ?
% The discrete wavelet transform with specific mother functions is 
% widely adopted to extract features to
% perform clustering of images in a given database.
% Wavelet analysis is widely used to explore an image database for its usefulness to detect spatial
% scales and clusters in images [5]. Moreover, wavelet analysis is able to preserve and display
% hierarchical information while allowing for pattern decomposition [6]. Features computed from
% the wavelet decomposed images are used for texture categorization to perform visual searching.
% source:International Journal of Computer Science, Engineering and Applications 
%(IJCSEA) Vol.1, No.5, October 2011 
%THE EFFECT OF MOTHER WAVELET AND SUBBAND CHOICE ON CLUSTERING OF IMAGES DATABASE
% Salim Lahmiri1
%

%Explain What is the singular value? 
% There are two types of singular values, one in the context of elliptic integrals,
% and the other in linear algebra.
% For a square matrix A, the square roots of the eigenvalues of A^(H)A, 
% where A^(H) is the conjugate transpose, are called singular values 
% (Marcus and Minc 1992, p. 69).
% The so-called singular value decomposition of a complex matrix A is given by
%  A=UDV^(H)
% where U and V are unitary matrices and D is a diagonal matrix whose elements
% are the singular values of A 

% For elliptic integrals, a elliptic modulus k_r such that
%  (K^'(k_r))/(K(k_r))=sqrt(r), 	
% where K(k) is a complete elliptic integral of the first kind, 
%and K^'(k_r)=K(sqrt(1-k_r^2)). The elliptic lambda function lambda^*(r)
%gives the value of k_r. Abel (quoted in Whittaker and Watson 1990, p. 525)
%proved that if r is an integer, or more generally whenever
%  (K^'(k))/(K(k))=(a+bsqrt(n))/(c+dsqrt(n)), 	
% where a, b, c, d, and n are integers, then the elliptic modulus k is 
%the root of an algebraic equation with integer coefficients.
%source:http://mathworld.wolfram.com/SingularValue.html
%%
clc
clf


%%
%################################################%
% fourier analysis of image 
%################################################%
originalImageFile=load(fullfile('C:\Users\abshe\OneDrive\MATLAB\Professor Niek\lab2\labImages.mat'));

imageNosieData=originalImageFile.imageNoise;
imageOriginalData=originalImageFile.imageOriginal;
% figure(1);
% imshow(imageOriginalData)
% figure(2);
% imshow(imageNosieData);
%%
originalImageFile.spectr=fft2(imageNosieData); %Calculate 2-D Fourier spectrum
originalImageFile.fourier = fftshift(originalImageFile.spectr);

%%
originalImageFile.magnitude = mat2gray(log(abs(originalImageFile.fourier)+1)); %amplitude
originalImageFile.phase = angle(originalImageFile.fourier);

figure('Name', '2-D signal spectrum amplitude');
imshow(originalImageFile.magnitude, []); % Display the spectrum
figure('Name', '2-D signal spectrum phase');
imshow(originalImageFile.phase, []); % Display the spectrum


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Hard Window Filter                                            %%%
%%%                                                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Applying Hard Window filting to the Fourier Co-efficient  
% we have move the energy to the edge of the image. 
% making dummy image to move the energy from center to the edge
rows=size(originalImageFile.magnitude,1);% size of image row 
columns=size(originalImageFile.magnitude,2);% size of image column 
P=zeros(length(rows));% size of dummy image row 
Q=zeros(length(columns));% size of dummy image column
[p,q]=meshgrid(P,Q);% 2-D grid with the vector size p and q 
distanceImage = sqrt(p.^2 + q.^2); % dummy image distance 
% It is an arbitrary formula to mask the original image 
a=1;
b=1;
c=0.001;
d=0.0;
e=0.1;
Z= a ./ (b + c*distanceImage) .* exp(-d*distanceImage ) .^ e;

figure('Name', '2-D signal spectrum amplitude');
imshow(originalImageFile.magnitude.*Z); % Display the spectrum


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Image Reconstruction                                          %%%
%%%                                                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
softFourier=originalImageFile.fourier.*Z;% mask image spectrum
imshow(softFourier,[])
softFourierShift=fftshift(softFourier); % Shift the zero frequency location 
%from (0,0)to the center of the display 
hFilteredRconstrtImage=ifft2(softFourierShift);  % Taking the Inverse FFT to convert the
%high pass filtered image back to Spatial domain.
figure;
imagesc(real(hFilteredRconstrtImage)),colormap('gray')% display the image

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   soft Window Filtering                                         %%%
%%%                                                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Creating the masking image 
rows=size(originalImageFile.magnitude,1); % size of mask row 
columns=size(originalImageFile.magnitude,2);% size of mask column 
X = linspace(-columns/2, columns/2, columns); % axis setting  
Y = linspace(-rows/2, rows/2, rows);% axis setting 
[p,q]=meshgrid(X,Y); % 2-D coordinate matrix with vector size of X and Y
distanceImage = double(sqrt(p.^2 + q.^2)); % distance of image 

% mask image formation using ideal low pass filtering  
softFourier=originalImageFile.fourier.*distanceImage;
%imshow(softFourier,[])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   soft image reconstruction                                     %%%
%%%                                                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
softFourierShift=fftshift(softFourier);% Shift the zero frequency location 
%from (0,0)to the center of the display 
sFilteredRconstrtImage=ifft2(softFourierShift);%Taking the Inverse FFT to convert the
%high pass filtered image back to Spatial domain.
figure;
imagesc(real(sFilteredRconstrtImage)),colormap('gray')% display the image
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Threshold implementation in Fourier Coefficient               %%%
%%%                                                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

thresholdimage=graythresh(originalImageFile.magnitude);% using graythresh 
% function converting the gray image to graythreshhold image 
%imshow(thrld); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   From threshold image to Original image reconstruction         %%%
%%%                                                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

threshFourierShift=fftshift(originalImageFile.fourier.*thresholdimage);
% Shift the zero frequency location 
%from (0,0)to the center of the display 
originalImageRconst=ifft2(threshFourierShift);%Taking the Inverse FFT to convert the
%high pass filtered image back to Spatial domain.
figure(2);
imagesc(originalImageRconst),colormap('gray')% display the image
axis off

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Image compression using Singular Value Decomposition Method   %%%
%%%                                                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% decomposing the image using singular value decomposition
[U,S,V]=svd(imageOriginalData);

% displaying the singular value 
singularValue=diag(S);
disp(singularValue) 

%%
% Using different number of singular values (diagonal of S) to compress and
% reconstruct the image
dispEr = []; % empty matrix to store error value 
numSVals = []; % empty matrix to store Singular Value 
for N=0:1:20
    % store the singular values in a temporary variable named singularValue
    singularValue = double(diag(S));

    % discard the diagonal values not required for compression
     singularValue(N+1:end,:)=0;
     singularValue(:,N+1:end)=0;

    % Construct an Image using the selected singular values
    svdReconstructedImage=U*singularValue(1)*V';

    % display and compute error
    figure;
    buffer = sprintf('Image output using %d singular values', N);
    imshow(uint8(svdReconstructedImage));
    title(buffer);
    
%     error=sum(sum((imageOriginalData-svdReconstructedImage).^2));
%     % store vals for display
%     dispEr = [dispEr; error];
%     numSVals = [numSVals; N];
end

% dislay the error graph
% figure; 
% title('Error in compression');
% plot(numSVals, dispEr);
% grid on
% xlabel('Number of Singular Values used');
% ylabel('Error between compress and original image');


%% To compress image  50% the number of singular value used 

% extract singular values
%singularValue = diag(S);
%display(singularValue) % showing the singulare value 
%indices=0.5*singvals;
% find out where to truncate the U, S, V matrices
indices = find(singularValue <= 0.5 * singularValue(1));
%size(indices)
% reduce SVD matrices
U_red = U(:,indices);
S_red = S(indices,indices);
V_red = V(:,indices);

% construct low-rank approximation of original image
imageOriginalData_red = U_red * S_red * V_red';
figure(1);
imshow(imageOriginalData_red)
% print results to command window
r = num2str(length(indices));
m = num2str(length(singularValue));
disp(['Number of Singular used:',r,' of ',m,' singular values']);
%% 
%approximation for half of singular value and using that data 
%we construct the image 

[U,S,V] = svd(imageOriginalData);

% extract singular values
singularValue1 = diag(S);
%indices=0.5*singvals;
% find out where to truncate the U, S, V matrices
indices = find(singularValue1 <= 0.002055 * singularValue1(1));
%size(indices)
% reduce SVD matrices
U_red = U(:,indices);
S_red = S(indices,indices);
V_red = V(:,indices);

% construct low-rank approximation of Lena
imageOriginalData_red = U_red * S_red * V_red';
figure(1);
imshow(imageOriginalData_red)
% print results to command window
r = num2str(length(indices));
m = num2str(length(singularValue1));
disp(['Number of Singular used:',r,' of ',m,' singular values']);

%%
%End
%Image analysis ends here 
% 2D Wavelet Analysis will be explained in DOCX file attached with this 
% folder in a seperate file. 
