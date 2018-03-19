## ResNet-101 in Keras

This is an [Keras](https://keras.io/) implementation of ResNet-101 with [ImageNet](http://www.image-net.org/) pre-trained weights. I converted the weights from [Caffe](http://caffe.berkeleyvision.org/) provided by the authors of the paper. The implementation supports both [Theano](http://deeplearning.net/software/theano/) and [TensorFlow](https://www.tensorflow.org/) backends. Just in case you are curious about how the conversion is done, you can visit my [blog post](https://flyyufelix.github.io/2017/03/23/caffe-to-keras.html) for more details.

ResNet Paper:

```
Deep Residual Learning for Image Recognition.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
arXiv:1512.03385
```

## Fine-tuning

Check [this](https://github.com/flyyufelix/cnn_finetune/blob/master/resnet_101.py) out to see example of fine-tuning ResNet-101 with your own dataset.

## Contents

model and usage demo: resnet-101_keras.py

Weights (Theano): [resnet101_weights_th.h5](https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing)

Weights (TensorFlow): [resnet101_weights_tf.h5](https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing)