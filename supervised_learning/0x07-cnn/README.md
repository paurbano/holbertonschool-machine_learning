# 0x07. Convolutional Neural Networks

## General
* **What is a convolutional layer?**
    
* **What is a pooling layer?**
  It's a layer that reduce computing in the network and to prevent or control overfitting. Use same concept that pool to obtein pooling units using max, average funtions
* **Forward propagation over convolutional and pooling layers**
* **Back propagation over convolutional and pooling layers**
* **How to build a CNN using Tensorflow and Keras**

# Tasks

## 0. Convolutional Forward Prop
Write a function `def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):` that performs forward propagation over a convolutional layer of a neural network:

* `A_prev` is a `numpy.ndarray` of shape `(m, h_prev, w_prev, c_prev)` containing the output of the previous layer
  * m is the number of examples
  * h_prev is the height of the previous layer
  * w_prev is the width of the previous layer
  * c_prev is the number of channels in the previous layer
* `W` is a `numpy.ndarray` of shape `(kh, kw, c_prev, c_new)` containing the kernels for the convolution
    * kh is the filter height
    * kw is the filter width
    * c_prev is the number of channels in the previous layer
    * c_new is the number of channels in the output
* `b` is a `numpy.ndarray` of shape `(1, 1, 1, c_new)` containing the biases applied to the convolution
* `activation` is an activation function applied to the convolution
* `padding` is a string that is either `same` or `valid`, indicating the type of padding used
* `stride` is a tuple of `(sh, sw)` containing the strides for the convolution
    * `sh` is the stride for the height
    * `sw` is the stride for the width
* you may `import numpy as np`
* Returns: the output of the convolutional layer
```
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ cat 0-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
conv_forward = __import__('0-conv_forward').conv_forward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))

    W = np.random.randn(3, 3, 1, 2)
    b = np.random.randn(1, 1, 1, 2)

    def relu(Z):
        return np.maximum(Z, 0)

    plt.imshow(X_train[0])
    plt.show()
    A = conv_forward(X_train_c, W, b, relu, padding='valid')
    print(A.shape)
    plt.imshow(A[0, :, :, 0])
    plt.show()
    plt.imshow(A[0, :, :, 1])
    plt.show()
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ ./0-main.py
```

## 1. Pooling Forward Prop
Write a function `def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):` that performs forward propagation over a pooling layer of a neural network:

* `A_prev` is a `numpy.ndarray` of shape `(m, h_prev, w_prev, c_prev)` containing the output of the previous layer
    * `m` is the number of examples
    * `h_prev` is the height of the previous layer
    * `w_prev` is the width of the previous layer
    * `c_prev` is the number of channels in the previous layer
* `kernel_shape` is a tuple of `(kh, kw)` containing the size of the kernel for the pooling
    * `kh` is the kernel height
    * `kw` is the kernel width
stride is a tuple of (sh, sw) containing the strides for the pooling
sh is the stride for the height
sw is the stride for the width
mode is a string containing either max or avg, indicating whether to perform maximum or average pooling, respectively
you may import numpy as np
Returns: the output of the pooling layer