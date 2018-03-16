from scipy.io import loadmat
from numpy import squeeze

weights = squeeze(loadmat('pretrained_models/imagenet-vgg-verydeep-19.mat')['layers'])

print(weights[40][0][0][0][0][0].shape)