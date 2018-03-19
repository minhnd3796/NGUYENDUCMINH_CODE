% setup MatConvNet
run ../../matconvnet-1.0-beta25/matlab/vl_setupnn;

% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('../../pretrained_models/imagenet-resnet-101-dag.mat')) ;
net.mode = 'test' ;
net.conserveMemory = 0;

% load and preprocess an image
im = imread('../test_images/tank1.jpg') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2), 'AntiAliasing', false) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
% im__ = single(ones(224,224,3));

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
conv1 = vl_nnconv(single(im_), single(net.params(1).value), [], 'pad', [3 3 3 3], 'stride', 2);