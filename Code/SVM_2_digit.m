
%% Load in Data
clear all; close all; clc;

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

%% The two digits being chosen for SVM - Just change the two digits to what u want.

% Test Data and labels
proj_test = projection_test(:, test_labels == 1 | test_labels == 0);
true_label = test_labels(test_labels == 1 | test_labels == 0, :);

% Training Data and label
label = labels(labels == 1 | labels == 0, :);
label = label';
xtrain = projection_training(:, labels == 1 | labels == 0);

%% 

% SVM classifier with training data, labels and test set
Mdl = fitcsvm(xtrain',label);

%% 

testlabels = predict(Mdl,proj_test');

testNum = size(testlabels,1);
err = abs(testlabels - true_label);
err = err > 0;
errNum = sum(err);
sucRate = 1 - errNum/testNum

