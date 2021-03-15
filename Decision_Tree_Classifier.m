

%% Load in Data
close all; clear all; clc;
[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[test_images, test_labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

images = im2double(images);
[m,n,k] = size(images);

for i = 1:k
    rawData(:,i) = reshape(images(:,:,i), m*n,1);
end 

test_images = im2double(test_images);
[m,n,k] = size(test_images);

for i = 1:k
    testData(:,i) = reshape(test_images(:,:,i), m*n,1);
end 

%% PCA Projection
clc;
[m,n] = size(rawData);
mn = mean(rawData, 2);
X = rawData - repmat(mn, 1, n);
A = X/sqrt(n-1);

[U,S,V] = svd(A,'econ');

projection_training = U(:, 1:154)'*X;
projection_training = projection_training./max(S(:));

[m, n] = size(testData);
test_avg = testData - repmat(mn, 1, n);

projection_test = U(:, 1:154)'*test_avg;
projection_test = projection_test./max(S(:));

%% All 10 digits, if you only want 2, you have to change the code slightly
% xtrain = projection_training(:, labels == 1| labels == 2) gives you 1 and
% 2 matrix

xtrain = projection_training;
label = labels';

proj_test = projection_test;
true_label = test_labels;

%% Increase number of max splits for higher accuracy.
Md1 =fitctree(xtrain',label, 'MaxNumSplits', 20);
view(Md1,'Mode','graph');
% classError = kfoldLoss(Md1) 

%% First way of calculating error Tree
clc;
approx_labels = predict(Md1, proj_test');

testNum = size(approx_labels,1);
err = abs(approx_labels - true_label);
err = err > 0;
errNum = sum(err);
sucRate = 1 - errNum/testNum

confusionchart(true_label, approx_labels);
title("Classification Chart of 10-digit Decision Tree Classifier");

%% 2nd way of calculating error for Tree

% tree = fitctree(xtrain', label,'CrossVal', 'on');
% classErrorTree = kfoldLoss(tree);
% TreeError10 = 1 - classErrorTree


