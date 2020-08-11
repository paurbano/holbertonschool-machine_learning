# 0x01. Multiclass Classification

# General
* **What is multiclass classification?**
    multiclass or multinomial classification is the problem of classifying instances into one of three or more classes (classifying instances into one of two classes is called binary classification).
* **What is a one-hot vector?**
    
* **How to encode/decode one-hot vectors**
* **What is the softmax function and when do you use it?**
* **What is cross-entropy loss?**
* **What is pickling in Python?**

# 0. One-Hot Encode
Write a function `def one_hot_encode(Y, classes):` that converts a numeric label vector into a one-hot matrix:

* Y is a numpy.ndarray with shape (m,) containing numeric class labels
    * m is the number of examples
* classes is the maximum number of classes found in Y
* Returns: a one-hot encoding of Y with shape (classes, m), or None on failure
```
alexa@ubuntu-xenial:0x01-multiclass_classification$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('0-one_hot_encode').one_hot_encode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)
alexa@ubuntu-xenial:0x01-multiclass_classification$ ./0-main.py
[5 0 4 1 9 2 1 3 1 4]
[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 1. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
alexa@ubuntu-xenial:0x01-multiclass_classification$
```