%This Program is about to polyphase, CIC and Halfband filter techniques.  
% last updated by Abol Basher 
% last updated date 12-11-2017

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Audio signal loading                                          %%%
%%%   Preprocessing of Audio                                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% 
[fileName,path]=uigetfile('*.*','select the signal file');
fileName=strcat(path,fileName);% For character array inputs, strcat removes
[originalAudioSignal,originalSamplingFrequency]=audioread(fileName);
properDimensionalization=originalAudioSignal(1:end-3);
%sound(originalAudioSignal,originalSamplingFrequency)
%figure(1);
%plot(originalAudioSignal)
%%
%Polyphas filtering using systme object for Decimation
%setting the decimation factors
decimationFactor = 8;
%designing filter for decimation
decimationFilterCoeff = designMultirateFIR(decimationFactor,1);
%creating a FIR Decimator system object from the DSP toolbox
decimationFilter = dsp.FIRDecimator(decimationFactor ,decimationFilterCoeff);
%polyphase(decimationFilter)
decimatedSignal=decimationFilter(originalAudioSignal(1:end-3));
%sound(decimatedSignal,2756);
%%
% amplitude response analysis 
freqz(decimationFilter)
%%
cost(decimationFilter);
%==========================================================
% How many Multiplier is required to for Decimation ? 
%Answer:
%Number of Multiplier per Input Sample 
%                  NumCoefficients: 168
%                         NumStates: 184
%     MultiplicationsPerInputSample: 21
%           AdditionsPerInputSample: 21
%===========================================================
%%
% Polyphase filtering using system object for interpolation 
%designing filter for interpolation
%setting the interpolation factors
interpolationFactor = 8; % 8 factor interpolation to reconstruct the signal 
interpolationFilterCoeff = designMultirateFIR(1,interpolationFactor); % filter coefficient 
interpolationFilter = dsp.FIRInterpolator(interpolationFactor,interpolationFilterCoeff);
%%
%amplitude response analysis
freqz(interploationFilter)
%%
% desing interpolation filter
% upsample 
% upsampling the signal using polyphase filter 
interpolatedSignal= interpolationFilter(decimatedSignal(1:end-1));
%sound(interpolatedSignal,22050);

%cost(interpolationFilter)
%=========================================================
% How many Multiplier is required to for Interpolation ? 
%Answer:
%Number of Multiplier per Input Sample 

%                   NumCoefficients: 169
%                         NumStates: 23
%     MultiplicationsPerInputSample: 169
%           AdditionsPerInputSample: 161

%=========================================================

%%
% Comparision between the Original Signal and Interpolated Signal 

figure(2);
subplot(3,1,1)
plot(originalAudioSignal)
subplot(3,1,2)
plot(decimatedSignal)
subplot(3,1,3) 
plot(interpolatedSignal)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Audio Signal filtering using CIC filter                       %%%
%%%   Preprocessing of Audio to transmit                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% CIC 
%=============================
% CIC Filter Design Parameters
%=============================
% CIC decimation using 8 factor
CICDecimator = dsp.CICDecimator('DecimationFactor',8,...
    'NumSections', 8);
 DecimatedSignal=CICDecimator(originalAudioSignal(1:end-3));
%%
%[h,w] = freqz(CICDecimator);
% freqz(CICDecimator)
%
%%
cost(CICDecimator)
%                   NumCoefficients: 0
%                         NumStates: 16
%     MultiplicationsPerInputSample: 0
%           AdditionsPerInputSample: 9
%%
%decimatedSignal=step(CICDecim,p);
fs = 22050;     % Sampling frequency of input of compensation decimator
fPass = 5.4e3;   % Passband frequency
fStop = 6.6e3; % Stopband frequency
% designing CIC compensator filter 
CICCompensatedDecimator = dsp.CICCompensationDecimator(CICDecimator, ...
     'DecimationFactor',2,'PassbandFrequency',fPass, ...
    'StopbandFrequency',fStop,'SampleRate',fs);

% compensatedSignal=CICCompDecim(decimatedSignal);
% Cost(CICCompensatedDecimator)
%                   NumCoefficients: 63
%                         NumStates: 62
%     MultiplicationsPerInputSample: 31.5000
%           AdditionsPerInputSample: 31
compensatedSignal=CICCompensatedDecimator(DecimatedSignal(1:end-1));

% filter cascading using dsp tool
filtCascadedDecimator = dsp.FilterCascade(CICDecimator,CICCompensatedDecimator);% cascaded filter system object
%generateFilteringCode(filtCasc, 'stepDecimator');
%w=stepDecimator(originalAudioSignal);
%%
%Amplitude Response Analysis 
freqz(filtCascadedDecimator)
%%
 %cascadedDecimatedSignal=filtCasc(originalAudioSignal(1:end-3)); % filtering using cascaded filter 

% figure(1);
% subplot(2,1,1)
% plot(x);
% subplot(2,1,2)
% plot(DecimatedSignal,'r');

%%
% Upsampling the downsample signal 
% designing the CIC 8 factor interpolation filter 
CICInterpolator = dsp.CICInterpolator(8);% interpolator factor setup
interpolatedSignal=CICInterpolator(compensatedSignal); % cic interpolator
% CIC compensated Interpolater design as like interpolatro 
fs = 22050;     % Sampling frequency of input of compensation decimator
fPass = 5.4e3;   % Passband frequency
fStop = 6.6e3; % Stopband frequency

CICCompensatedInterpolator = dsp.CICCompensationInterpolator('InterpolationFactor',2,...
'PassbandFrequency',fPass,'StopbandFrequency',fStop,'SampleRate',fs);
%upsampling the signal using generated interpolated filter property 
filtCascadedInterpolator = dsp.FilterCascade(CICInterpolator,CICCompensatedInterpolator);% cascaded filter system object
%%
% Response Analysis 
freqz(filtCascadedInterpolator)
%%
compensatedInterpolatedSignal=CICCompensatedInterpolator(interpolatedSignal); % compensated inerpolated output  

%% output analysis plot 
 figure(1);
 subplot(6,1,1);plot(originalAudioSignal);
 subplot(6,1,2);plot(properDimensionalization);
 subplot(6,1,3);plot(DecimatedSignal);
 subplot(6,1,4);plot(compensatedSignal);
 subplot(6,1,5);plot(interpolatedSignal);
 subplot(6,1,6);plot(compensatedInterpolatedSignal);
 %%
 sound(compensatedInterpolatedSignal,22050)% reconstructed Signal using CIC filter system object 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Halfband Decimator design                                     %%%
%%%   Preprocessing of Audio using Halfbanddecimator                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%FIR halfbanddecimator designing
%
halfBandDecimatorFilter = dsp.FIRHalfbandDecimator(...
    'Specification','Transition width and stopband attenuation',...
    'TransitionWidth',2000,'StopbandAttenuation',60,'SampleRate',22050);

halfBandfiltCascDecimator = dsp.FilterCascade(halfBandDecimatorFilter,...
halfBandDecimatorFilter,halfBandDecimatorFilter);
%%
% Response Analysis 
freqz(halfBandfiltCascDecimator)
%%
cost(halfBandfiltCascDecimator)
%                   NumCoefficients: 63
%                         NumStates: 114
%     MultiplicationsPerInputSample: 18.3750
%           AdditionsPerInputSample: 17.5000

filterhalfbanddecimatedSignal= halfBandfiltCascDecimator(originalAudioSignal(1:end-3));
% u=hfirhalfbanddecim(p);
% u1=hfirhalfbanddecim(u);
% u2=hfirhalfbanddecim(u1);
halfBandDecimatedSignal=halfBandDecimatorFilter(filterhalfbanddecimatedSignal(1:end-1));
%%
% FIR half band interpolator designing 
Fs = 22050;% sampling frequency 
filterspec = 'Filter order and transition width';
Order = 100;% 100 worder filter 
TW =2e3;% transition width 
halfBandInterpolatorFilter = dsp.FIRHalfbandInterpolator(...
                                               'Specification',filterspec,...
                                               'FilterOrder',Order,...
                                               'TransitionWidth',TW,...
                                                'SampleRate',Fs);
 % creating cascaded filter 
halfBandfiltCascadedInterpolator = dsp.FilterCascade(halfBandInterpolatorFilter,...
halfBandInterpolatorFilter,halfBandInterpolatorFilter,halfBandInterpolatorFilter);
%%
% Response Analysis 
freqz(halfBandfiltCascadedInterpolator)
%%
% cost(halfBandfiltCascadedInterpolator)
%                    NumCoefficients: 200
%                         NumStates: 200
%     MultiplicationsPerInputSample: 750
%           AdditionsPerInputSample: 735
 halfBandInterpolatedSignal=halfBandfiltCascadedInterpolator(halfBandDecimatedSignal);
%  v=firhalfbandinterp(u3);
%  v1=firhalfbandinterp(u3);
%  v2=firhalfbandinterp(u3);
%  v3=firhalfbandinterp(u3);
%%
sound(halfBandInterpolatedSignal,22050)
%% End
%Audio processing code have been concluded here 
