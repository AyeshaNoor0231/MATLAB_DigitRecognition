%Digit Recognition Using CNN
%Step 1: Load & Read dataset
%Step 2: Split Data
%Step 3: Create a network
%Step 4: Mention training options
%Step 5: Train the network
%Step 6: Testing
%%

%load dataset
datapath = "E:\DigitDataset";

%Read data from folder creating datastore
dataimages = imageDatastore(datapath, "IncludeSubfolders",true, "LabelSource","foldernames");
%%
%Split data into train & test
numTrain = 750; %750 images from each folder
[Train, Test] = splitEachLabel(dataimages, numTrain, "randomized"); %randomly pick images
%%
%Creating a CNN
layers = [
   imageInputLayer([28 28 1], 'Name','Input')
   
   convolution2dLayer(3,8,'Padding','same','Name','Conv_1')
   batchNormalizationLayer('Name','BN_1')
   reluLayer('Name','Relu_1')
   maxPooling2dLayer(2, 'Stride', 2, 'Name','MaxPool_1')

   convolution2dLayer(3,16,'Padding','same','Name','Conv_2')
   batchNormalizationLayer('Name','BN_2')
   reluLayer('Name','Relu_2')
   maxPooling2dLayer(2, 'Stride', 2, 'Name','MaxPool_2')

   convolution2dLayer(3,32,'Padding','same','Name','Conv_3')
   batchNormalizationLayer('Name','BN_3')
   reluLayer('Name','Relu_3')
   maxPooling2dLayer(2, 'Stride', 2, 'Name','MaxPool_3')

   convolution2dLayer(3,64,'Padding','same','Name','Conv_4')
   batchNormalizationLayer('Name','BN_4')
   reluLayer('Name','Relu_4')
   maxPooling2dLayer(2, 'Stride', 2, 'Name','MaxPool_4')

   convolution2dLayer(3,128,'Padding','same','Name','Conv_5')
   batchNormalizationLayer('Name','BN_5')
   reluLayer('Name','Relu_5')

   fullyConnectedLayer(10,'Name','FC')
   softmaxLayer('Name','SoftMax');
   classificationLayer('Name','Output'); 
];
lgraph = layerGraph(layers);
plot(lgraph);
%%
%training options
options = trainingOptions("sgdm",'InitialLearnRate',0.01, ...
    'MaxEpochs',4, Shuffle='every-epoch',...
    Plots = 'training-progress',...
    Verbose = true, ValidationData=Test,ValidationFrequency=40);
%%
%train network
net = trainNetwork(Train,layers,options);
%%
%Accuracy 
YPred = classify(net,Test);
YValid = Test.Labels;
accuracy = sum(YPred == YValid)/numel (YValid)
%%
%Testing
[filename,pathname] = uigetfile('*.*', 'Select Input grayscale image');
filewithpath = strcat(pathname, filename);
I = imread(filewithpath);
figure
imshow(I)

%Classification using trained network
label = classify(net,I);
title(['Digit is: ' char(label)])