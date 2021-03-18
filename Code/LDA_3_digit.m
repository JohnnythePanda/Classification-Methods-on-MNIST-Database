%% Loading in Data
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

%% The three digits chosen for LDA

%Training Data
% xtrain = projection_training(:, labels == 1 | labels == 7| labels == 2);
xtrain = projection_training;
% label = labels(labels == 1 | labels == 7 | labels == 2, :);
label = labels';

% Test Data
% proj_test = projection_test(:, test_labels == 1 | test_labels == 7 | test_labels == 2);
% true_label = test_labels(test_labels == 1 | test_labels == 7 | test_labels == 2, :);

proj_test = projection_test;
true_label = test_labels;


%% LDA with 3 digits
clc;

Md1 = fitcdiscr(xtrain', label, 'discrimType', 'linear');
approx_label = predict(Md1, proj_test');


testNum = size(approx_label, 1);
err = abs(approx_label - true_label);
errTrue = err > 0;
errNum = sum(errTrue);
sucRate = 1 - errNum/testNum

confusionchart(true_label, approx_label);
title("Classification Chart of 10-digit LDA Classifier");

