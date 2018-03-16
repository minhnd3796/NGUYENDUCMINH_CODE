% setup MatConvNet
run  matlab/vl_setupnn

% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('/media/minhnd/Computer 2/TRINHVANDUY_Thesis/pretrained_models/imagenet-resnet-101-dag.mat')) ;
net.mode = 'test' ;

% load and preprocess an image
im = imread('/media/minhnd/Computer 2/TRINHVANDUY_Thesis/TRINHVANDUY_CODE/test_images/tank1.jpg') ;
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