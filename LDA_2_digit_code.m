
%% Load images
clear all; close all; clc; 
[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');

images = im2double(images);
[m,n,k] = size(images);

imshow(images(:,:,1));

for i = 1:k
    rawData(:,i) = reshape(images(:,:,i), m*n,1);
end 

[test_images, test_labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

test_images = im2double(test_images);
[m,n,k] = size(test_images);

for i = 1:k
    testData(:,i) = reshape(test_images(:,:,i), m*n,1);
end 


%% Principal Component Analysis
close all;
[m,n] = size(rawData);
mn = mean(rawData, 2);
X = rawData - repmat(mn, 1, n);
A = X/sqrt(n-1);

[U,S,V] = svd(A,'econ');

figure(1)
plot((diag(S).^2)/(sum(diag(S).^2))*100, 'o');
set(gca, 'Fontsize', 12);
ylabel("Energy (%)"); title("Energy of Principal Components");
xlabel("Principal Component");


diagonal = (diag(S).^2)/(sum(diag(S).^2))*100;
sum_Sing = 0;
index = 0;

for i = 1:784
    sum_Sing = sum_Sing + diagonal(i);
    index = index + 1;
    if (sum_Sing > 90.00)
        break
    end 
end

% 154
projection = U(:, 1:154)'*X;

figure(2)
proj = U'*X;

for i = 0:9
   scatter3(proj(1,labels==i), proj(2,labels==i), proj(3,labels==i));
    hold on
end 

set(gca, 'Fontsize', 12);
xlabel("PCA Mode 1"); ylabel("PCA Mode 2"); zlabel("PCA Mode 3");
title("Projection onto 3 Principal Components");
legend("0","1", "2", "3", "4", "5", "6", "7", "8", "9");

%%
% figure(3)
% rank154 = U(:, 1:154)*S(1:154,1:154)*V(:,1:154)';
% image154 = rank154(:,1);
% image154 = image154*sqrt(n-1) + repmat(mn, 1, 1);
% image154 = reshape(image154, [28,28]);
% 
% imshow(image154);
% set(gca,'Fontsize',8)
% title("Rank-154 Approximation of 5");

%% Separating the images into their group
one_matrix = rawData(:, labels == 3);
two_matrix = rawData(:, labels == 5);



%% Constructing LDA classifier

feature = 154;
[U,S,V,threshold,w,sortOne,sortTwo] = dc_trainer(one_matrix,two_matrix,feature);


%% Using LDA Classifier

filterData = [];
filterLabel = [];

for i = 1:60000
    if (labels(i) == 3)
        filterData = [filterData rawData(:,i)];
        filterLabel = [filterLabel 0];
    end 
    if(labels(i) == 5)
        filterData = [filterData rawData(:,i)];
        filterLabel = [filterLabel 1];
    end 
end 

TestNum = size(filterData,2);
TestMat = U'*(filterData); % PCA projection
pval = w'*TestMat;

ResVec = (pval>threshold);

err = abs(ResVec - filterLabel);
errNum = sum(err);
sucRate = 1 - errNum/TestNum



