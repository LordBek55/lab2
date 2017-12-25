originalImagefile=load(fullfile('D:\lab2\labImages.mat')); %load image
originalImage=originalImagefile.imageOriginal;
noisyImage=originalImagefile.imageNoise;
%%
originalImagefile.spectr=fft2(noisyImage); %Calculate 2-D Fourier spectrum
originalImagefile.fourier = fftshift(noisyImage.spectr);
originalImagefile.magnitude = mat2gray(log(abs(noisyImage.fourier)+1)); %amplitude
originalImagefile.phase = angle (noisyImage.fourier);
figure('Name', '2-D signal spectrum emplitude');
imshow(noisyImage.magnitude, []); % Display the spectrum
figure('Name', '2-D signal spectrum phase');
imshow(noisyImage.phase, []); % Display the spectrum

%%
% noisyImage.spectr=fft2(noisyImage.ns); %Calculate 2-D Fourier spectrum
% noisyImage.fourier = fftshift(noisyImage.spectr);
% noisyImage.magnitude = mat2gray(log(abs(noisyImage.fourier)+1)); %amplitude
% noisyImage.phase = angle (noisyImage.fourier);
% figure('Name', '2-D signal spectrum emplitude');
% imshow(noisyImage.magnitude, []); % Display the spectrum
% figure('Name', '2-D signal spectrum phase');
% % imshow(noisyImage.phase, []); % Display the spectrum

%%
rows=size(noisyImage.ns,1);
columns=size(noisyImage.ns,2);
X = linspace(-columns/2, columns/2, columns);
Y = linspace(-rows/2, rows/2, rows);
[x, y] = meshgrid(X, Y);
distanceImage = sqrt(x.^2 + y.^2);
a=1;
b=1;
c=1;
d=0.5;
e=0.1;
A = a * (b + c*distanceImage) .* exp(-d*distanceImage ) .^ e;
imshow(A)
figure('Name', '2-D signal spectrum amplitude');
imshow(originalImage.magnitude.*A, []); % Display the spectrum