# 0x05. Regularization

## General
* **What is regularization? What is its purpose?**
     Regularization is a technique which makes slight modifications to the learning algorithm such that the model generalizes better. Is the process of adding information in order to solve an ill-posed problem or to prevent overfitting
     its purpose is for help in reducing overfitting.
* What is are L1 and L2 regularization? What is the difference between the two methods?
* What is dropout?
* What is early stopping?
* What is data augmentation?
* How do you implement the above regularization methods in Numpy? Tensorflow?
* What are the pros and cons of the above regularization methods?

## 0. L2 Regularization Cost
Write a function `def l2_reg_cost(cost, lambtha, weights, L, m):` that calculates the cost of a neural network with L2 regularization:

* `cost` is the cost of the network without L2 regularization
* `labtha` is the regularization parameter
* `weights` is a dictionary of the weights and biases (numpy.ndarrays) of the neural network
* `L` is the number of layers in the neural network
* `m` is the number of data points used
* Returns: the cost of the network accounting for L2 regularization

```
ubuntu@alexa-ml:~/0x05-regularization$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
l2_reg_cost = __import__('0-l2_reg_cost').l2_reg_cost

if __name__ == '__main__':
    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['W2'] = np.random.randn(128, 256)
    weights['W3'] = np.random.randn(10, 128)

    cost = np.abs(np.random.randn(1))

    print(cost)
    cost = l2_reg_cost(cost, 0.1, weights, 3, 1000)
    print(cost)
ubuntu@alexa-ml:~/0x05-regularization$ ./0-main.py 
[0.41842822]
[0.45158952]
ubuntu@alexa-ml:~/0x05-regularization$ 
```
