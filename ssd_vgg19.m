%% 一、训练VGG19模型识别焊缝缺陷
% 这是主函数，运行这个函数训练网络。
%% 数据导入
clear;clc;
DatasetPath = 'traindata4'; %数据集路径
imds = imageDatastore(DatasetPath, 'IncludeSubfolders',true, 'LabelSource','foldernames');

% %% 展示数据集(此部分可删除，不影响)
% figure('Name','数据集部分图像展示','NumberTitle','off');
% numImages = numel(imds.Files);                 %数据数量
% perm = randperm(numImages,15);   %从数据集随机选取15张图像
% for ii = 1:15                   %展示数据集中部分图像
%     subplot(3,5,ii);
%     imshow(imds.Files{perm(ii)}); %调用数据并展示
%     title(ii);
% end

%% 划分训练集与验证集
[imgTrain,imgValid] = splitEachLabel(imds,0.8,'randomize'); %按比例拆分 ImageDatastore 标签,即80%训练集20%验证集
% 调整图像大小以匹配网络输入层。
imdsTrain = augmentedImageDatastore([300 300 3],imgTrain);
imdsValid = augmentedImageDatastore([300 300 3],imgValid);
%% VGG19 网络
%注：可以考虑将预训练参数与神经网络做成子函数进行调用

% 预训练参数
params = load("D:\焊缝缺陷识别分类\params_2023_12_25__17_36_44.mat");

%%
%'WeightL2Factor'          - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   weights. The default is 1.
%       'BiasL2Factor'            - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   biases. The default is 0.
%%
% 神经网络
layers = [
    imageInputLayer([300 300 3],"Name","imageinput")
    convolution2dLayer([3 3],64,"Name","conv1_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv1_1.Bias,"Weights",params.conv1_1.Weights)
    reluLayer("Name","relu1_1")
    convolution2dLayer([3 3],64,"Name","conv1_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv1_2.Bias,"Weights",params.conv1_2.Weights)
    reluLayer("Name","relu1_2")
    maxPooling2dLayer([2 2],"Name","pool1","Padding","same")
    convolution2dLayer([3 3],128,"Name","conv2_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv2_1.Bias,"Weights",params.conv2_1.Weights)
    reluLayer("Name","relu2_1")
    convolution2dLayer([3 3],128,"Name","conv2_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv2_2.Bias,"Weights",params.conv2_2.Weights)
    reluLayer("Name","relu2_2")
    maxPooling2dLayer([2 2],"Name","pool2","Padding",[1 1 1 1],"Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv3_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv3_1.Bias,"Weights",params.conv3_1.Weights)
    reluLayer("Name","relu3_1")
    convolution2dLayer([3 3],256,"Name","conv3_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv3_2.Bias,"Weights",params.conv3_2.Weights)
    reluLayer("Name","relu3_2")
    convolution2dLayer([3 3],256,"Name","conv3_3","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv3_3.Bias,"Weights",params.conv3_3.Weights)
    reluLayer("Name","relu3_3")
    convolution2dLayer([3 3],256,"Name","conv3_4","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv3_4.Bias,"Weights",params.conv3_4.Weights)
    reluLayer("Name","relu3_4")
    maxPooling2dLayer([2 2],"Name","pool3","Padding",[1 1 1 1],"Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv4_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv4_1.Bias,"Weights",params.conv4_1.Weights)
    reluLayer("Name","relu4_1")
    convolution2dLayer([3 3],512,"Name","conv4_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv4_2.Bias,"Weights",params.conv4_2.Weights)
    reluLayer("Name","relu4_2")
    convolution2dLayer([3 3],512,"Name","conv4_3","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv4_3.Bias,"Weights",params.conv4_3.Weights)
    reluLayer("Name","relu4_3")
    convolution2dLayer([3 3],512,"Name","conv4_4","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv4_4.Bias,"Weights",params.conv4_4.Weights)
    reluLayer("Name","relu4_4")
    maxPooling2dLayer([2 2],"Name","pool4","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv5_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv5_1.Bias,"Weights",params.conv5_1.Weights)
    reluLayer("Name","relu5_1")
    convolution2dLayer([3 3],512,"Name","conv5_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv5_2.Bias,"Weights",params.conv5_2.Weights)
    reluLayer("Name","relu5_2")
    convolution2dLayer([3 3],512,"Name","conv5_3","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv5_3.Bias,"Weights",params.conv5_3.Weights)
    reluLayer("Name","relu5_3")
    convolution2dLayer([3 3],512,"Name","conv5_4","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv5_4.Bias,"Weights",params.conv5_4.Weights)
    reluLayer("Name","relu5_4")
    maxPooling2dLayer([2 2],"Name","pool5","Stride",[2 2])
    convolution2dLayer([3 3],1024,"Name","conv6","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu6")
    maxPooling2dLayer([2 2],"Name","pool6","Padding","same")
    convolution2dLayer([3 3],1024,"Name","conv7","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu7")
    maxPooling2dLayer([2 2],"Name","pool7","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv8_1","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu8_1")
    convolution2dLayer([3 3],512,"Name","conv8_2","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu8_2")
    maxPooling2dLayer([2 2],"Name","pool8","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv9_1","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu9_1")
    convolution2dLayer([3 3],256,"Name","conv9_2","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu9_2")
    maxPooling2dLayer([2 2],"Name","pool9","Padding",[1 1 1 1],"Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv10_1","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu10_1")
    convolution2dLayer([3 3],256,"Name","conv10_2","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu10_2")
    maxPooling2dLayer([2 2],"Name","pool10","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv11_1","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu11_1")
    convolution2dLayer([3 3],256,"Name","conv11_2","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu11_2")
    fullyConnectedLayer(4096,"Name","fc12","WeightL2Factor",0)
    reluLayer("Name","relu12")
    fullyConnectedLayer(4096,"Name","fc13","WeightL2Factor",0)
    reluLayer("Name","relu13")
    fullyConnectedLayer(3,"Name","fc14","WeightL2Factor",0)
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
%% 配置训练参数
%  options = trainingOptions('sgdm', ...
%     'MiniBatchSize',64, ...%若内存不足，改为1，如果还不行换电脑吧%一般为2的指数幂
%     'InitialLearnRate',0.001, ...%0.0001
%     'ValidationFrequency',50,...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsValid,...
%     'ExecutionEnvironment','cpu',...%'auto'
%     'Verbose',false, ...
%     'Plots','training-progress');
 
 options = trainingOptions('sgdm', ...
    'MiniBatchSize',64, ...%若内存不足，改为1，如果还不行换电脑吧%5
    'InitialLearnRate',0.001, ... 
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValid,...
     'ValidationFrequency',5,...
    'ExecutionEnvironment','cpu',...%'auto'
    'Verbose',false, ...
    'Plots','training-progress');

%% 训练神经网络
 [net,traininfo]= trainNetwork(imdsTrain,layers,options);

%% 保存训练好参数
save('模型参数.mat','net');%保存训练好的神经网络到本地，文件名为模型参数.mat
save('trainedInfo.mat','traininfo');%保存训练信息