net = load('/media/minhnd/LinuxExtendedPar/Projects/TRINHVANDUY_CODE/pretrained_models/imagenet-vgg-verydeep-19.mat');
net = vl_simplenn_tidy(net);

% Obtain and preprocess an image.
im = imread('/media/minhnd/LinuxExtendedPar/Projects/TRINHVANDUY_CODE/tank1.jpg') ;
im_ = single(im); % note: 255 range
resized_im = imresize(im_, net.meta.normalization.imageSize(1:2), 'AntiAliasing', false);
im_ = resized_im - net.meta.normalization.averageImage;

% Run the CNN.
res = vl_simplenn(net, im_) ;

% Show the classification result.
scores = squeeze(gather(res(end).x));
[bestScore, best] = max(scores);
figure(1); clf ; imagesc(im);
title(sprintf('%s (%d), score %.3f',...
   net.meta.classes.description{best}, best, bestScore));