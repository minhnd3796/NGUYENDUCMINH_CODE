% setup MatConvNet
run ../../matconvnet-1.0-beta25/matlab/vl_setupnn;

% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('../../pretrained_models/imagenet-resnet-101-dag.mat')) ;
net.mode = 'test' ;

% load and preprocess an image
im = imread('../test_images/tank1.jpg') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

% run the CNN
net.eval({'data', im_}) ;

% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

% show the classification results
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;