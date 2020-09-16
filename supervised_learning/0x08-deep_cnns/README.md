# 0x08. Deep Convolutional Architectures

## General
* What is a skip connection?
* What is a bottleneck layer?
* What is the Inception Network?
* What is ResNet? ResNeXt? DenseNet?
* How to replicate a network architecture by reading a journal article

## 0. Inception Block
Write a function `def inception_block(A_prev, filters):` that builds an inception block as described in [Going Deeper with Convolutions (2014)](https://arxiv.org/pdf/1409.4842.pdf):

* `A_prev` is the output from the previous layer
* `filters` is a tuple or list containing `F1, F3R, F3,F5R, F5, FPP`, respectively:
    * `F1` is the number of filters in the 1x1 convolution
    * `F3R` is the number of filters in the 1x1 convolution before the 3x3 convolution
    * `F3` is the number of filters in the 3x3 convolution
    * `F5R` is the number of filters in the 1x1 convolution before the 5x5 convolution
    * `F5` is the number of filters in the 5x5 convolution
    * `FPP` is the number of filters in the 1x1 convolution after the max pooling
* All convolutions inside the inception block should use a rectified linear activation (ReLU)
* Returns: the concatenated output of the inception block
```
ubuntu@alexa-ml:~/supervised_learning/0x08-deep_cnns$ cat 0-main.py 
#!/usr/bin/env python3

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block

if __name__ == '__main__':
    X = K.Input(shape=(224, 224, 3))
    Y = inception_block(X, [64, 96, 128, 16, 32, 32])
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
ubuntu@alexa-ml:~/supervised_learning/0x08-deep_cnns$ ./0-main.py 
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 224, 224, 96) 384         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 224, 224, 16) 64          input_1[0][0]                    
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 224, 224, 3)  0           input_1[0][0]                    
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 224, 224, 64) 256         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 224, 224, 128 110720      conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 224, 224, 32) 12832       conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 224, 224, 32) 128         max_pooling2d[0][0]              
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 224, 224, 256 0           conv2d[0][0]                     
                                                                 conv2d_2[0][0]                   
                                                                 conv2d_4[0][0]                   
                                                                 conv2d_5[0][0]                   
==================================================================================================
Total params: 124,384
Trainable params: 124,384
Non-trainable params: 0
__________________________________________________________________________________________________
ubuntu@alexa-ml:~/supervised_learning/0x08-deep_cnns$
```