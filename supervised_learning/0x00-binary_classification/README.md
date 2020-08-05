# 0x00. Binary Classification

## General
* **What is a model?**
     is a representation of a person or a system which provides some information about it. a thing used as an example to follow or imitate. A machine-learning model is the output generated when you train your machine-learning algorithm with data.

* **What is supervised learning?**
    Supervised learning is so named because the data scientist acts as a guide to teach the algorithm what conclusions it should come up with. It’s similar to the way a child might learn arithmetic from a teacher

* **What is a prediction?**
    A prediction is a forecast, but not only about the weather. Pre means “before” and diction has to do with talking. So a prediction is a statement about the future. It's a guess, sometimes based on facts or evidence, but not always.

* **What is a node?**
    A node is a basic unit of a data structure, such as a linked list or tree data structure. Nodes contain data and also may link to other nodes.

* **What is a weight?**
    Weight is the parameter within a neural network that transforms input data within the network's hidden layers. A neural network is a series of nodes, or neurons. Within each node is a set of inputs, weight, and a bias value

* **What is a bias?**
    Bias is disproportionate weight in favor of or against an idea or thing,In science and engineering, a bias is a systematic error

* **What are activation functions?**
    Activation functions are mathematical equations that determine the output of a neural network. The function is attached to each neuron in the network, and determines whether it should be activated (“fired”) or not, based on whether each neuron’s input is relevant for the model’s prediction.
    * **Sigmoid?**
        it is nonlinear in nature having a characteristic "S"-shaped curve or sigmoid curve. A common example of a sigmoid function is the logistic function, any small changes in the values of X in that region will cause values of Y to change significantly

        ![](https://github.com/paurbano/holbertonschool-machine_learning/tree/master/images/sigmoid.png)
    * **Tanh?**
        Hyperbolic tangent: f(x) = 2 sigmoid(2x) - 1. Deciding between the sigmoid or tanh will depend on your requirement of gradient strength.
    * **Relu?**
        Its a linear function A(x) = max(0,x) It gives an output x if x is positive and 0 otherwise. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.
    * **Softmax?**
        Or **normalized exponential function** is a function that takes as input a vector `z` of `K` real numbers, and normalizes it into a probability distribution consisting of `K` probabilities proportional to the exponentials of the input numbers.That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval **(0,1)**, and the components will add up to 1, so that they can be interpreted as probabilities.

* **What is a layer?**
    Layer is a general term that applies to a collection of 'nodes' operating together at a specific depth within a neural network,is the highest-level building block in deep learning. A layer is a container that usually receives weighted input, transforms it with a set of mostly non-linear functions and then passes these values as output to the next layer.

* **What is a hidden layer?**
    The interior layers are sometimes called “hidden layers” because they are not directly observable from the systems inputs and outputs. analyze and process the input features

* **What is Logistic Regression?**
    es un tipo de análisis de regresión utilizado para predecir el resultado de una variable categórica (una variable que puede adoptar un número limitado de categorías) en función de las variables independientes o predictoras.
    La idea es que la regresión logística aproxime la probabilidad de obtener "0" (no ocurre cierto suceso) o "1" (ocurre el suceso) con el valor de la variable explicativa x 

* **What is a loss function?**
    Loss functions are used to determine the error (aka “the loss”) between the output of our algorithms and the given target value.  In layman’s terms, the loss function expresses how far off the mark our computed output is. `Machines` learn by means of a loss function. It's a method of evaluating how well specific algorithm models the given data. If predictions deviates too much from actual results, loss function would cough up a very large number.

* **What is a cost function?**
    cost function is a measure of how wrong the model is in terms of its ability to estimate the relationship between X and y. This is typically expressed as a difference or distance between the predicted value and the actual value.

* **What is forward propagation?**
    How neural networks make predictions 

* **What is Gradient Descent?**
    Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model.

* **What is back propagation?**
    Backpropagation (backward propagation) is an important mathematical tool for improving the accuracy of predictions in data mining and machine learning. Essentially, backpropagation it’s a technique used to calculate derivatives quickly.
    The backpropagation algorithm works by computing the gradient of the loss function with respect to each weight by the chain rule, computing the gradient one layer at a time, iterating backward from the last layer to avoid redundant calculations of intermediate terms in the chain rule.

* **What is a Computation Graph?**
    A computational graph is defined as a directed graph where the nodes correspond to mathematical operations. Computational graphs are a way of expressing and evaluating a mathematical expression

* **How to initialize weights/biases**
    Techniques generally practised to initialize parameters:
        * Zero initialization
        * Random initialization
    
    In general practice biases are initialized with 0 and weights are initialized with random numbers
    
    Weigths and biases are both matrices:
    * W<sup>[l]</sup> - weigth matrix of dimension (*size of layer l*,  *size of layer l-1*)
    * b<sup>[l]</sup> - bias matrix of dimension (*size of layer l*, *1*)
    
    For more details [here](https://www.deeplearning.ai/ai-notes/initialization/),or [here](https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78)

* The importance of vectorization
    Optimice processing of training reducing the use of loops.
* How to split up your data

# Tasks
## 0. Neuron
Write a class Neuron that defines a single neuron performing binary classification:

* class constructor: def __init__(self, nx):
    * nx is the number of input features to the neuron
        * If nx is not an integer, raise a TypeError with the exception: nx must be an integer
        * If nx is less than 1, raise a ValueError with the exception: nx must be a positive integer
    * All exceptions should be raised in the order listed above
* Public instance attributes:
    * W: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.
    * b: The bias for the neuron. Upon instantiation, it should be initialized to 0.
    * A: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.

```
alexa@ubuntu-xenial:0x00-binary_classification$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('0-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.W.shape)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)
alexa@ubuntu-xenial:0x00-binary_classification$ ./0-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
(1, 784)
0
0
10
alexa@ubuntu-xenial:0x00-binary_classification$
```
File: [`0-neuron.py`]