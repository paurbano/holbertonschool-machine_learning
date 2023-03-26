# 0x00. Binary Classification
<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/clasificacionBinaria.jpg" alt="" loading="lazy" style=""></p>

<h2>Background Context</h2>
<p>At the end of this project, you should be able to build your own binary image classifier from scratch using <code>numpy</code>. </p>
<h2>Resources <em>(same as previous project)</em></h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="https://blogs.oracle.com/ai-and-datascience/post/supervised-vs-unsupervised-machine-learning" title="Supervised vs. Unsupervised Machine Learning" target="_blank">Supervised vs. Unsupervised Machine Learning</a> </li>
<li><a href="https://www.quora.com/How-would-you-explain-neural-networks-to-someone-who-knows-very-little-about-AI-or-neurology/answer/Yohan-John" title="How would you explain neural networks to someone who knows very little about AI or neurology?" target="_blank">How would you explain neural networks to someone who knows very little about AI or neurology?</a> </li>
<li><a href="https://neuralnetworksanddeeplearning.com/chap1.html" title="Using Neural Nets to Recognize Handwritten Digits" target="_blank">Using Neural Nets to Recognize Handwritten Digits</a> (<em>until “A simple network to classify handwritten digits” (excluded)</em>)</li>
<li><a href="https://www.youtube.com/watch?v=wL17g67vU88" title="Forward propagation" target="_blank">Forward propagation</a> </li>
<li><a href="https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0" title="Understanding Activation Functions in Neural Networks" target="_blank">Understanding Activation Functions in Neural Networks</a></li>
<li><a href="https://en.wikipedia.org/wiki/Loss_function" title="Loss function" target="_blank">Loss function</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Gradient_descent" title="Gradient descent" target="_blank">Gradient descent</a> </li>
<li><a href="http://colah.github.io/posts/2015-08-Backprop/" title="Calculus on Computational Graphs: Backpropagation" target="_blank">Calculus on Computational Graphs: Backpropagation</a> </li>
<li><a href="/https://www.youtube.com/watch?v=tIeHLnjs5U8" title="Backpropagation calculus" target="_blank">Backpropagation calculus</a> </li>
<li><a href="https://www.youtube.com/watch?v=n1l-9lIMW7E&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=3" title="What is a Neural Network?" target="_blank">What is a Neural Network?</a> </li>
<li><a href="https://www.youtube.com/watch?v=BYGpKPY9pO0&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=5" title="Supervised Learning with a Neural Network" target="_blank">Supervised Learning with a Neural Network</a> </li>
<li><a href="https://www.youtube.com/watch?v=eqEc66RFY0I&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=8" title="Binary Classification" target="_blank">Binary Classification</a> </li>
<li><a href="https://www.youtube.com/watch?v=hjrYrynGWGA&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=9" title="Logistic Regression" target="_blank">Logistic Regression</a> </li>
<li><a href="https://www.youtube.com/watch?v=SHEPb1JHw5o&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=10" title="Logistic Regression Cost Function" target="_blank">Logistic Regression Cost Function</a></li>
<li><a href="https://www.youtube.com/watch?v=uJryes5Vk1o&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=11" title="Gradient Descent" target="_blank">Gradient Descent</a></li>
<li><a href="https://www.youtube.com/watch?v=hCP1vGoCdYU&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=14" title="Computation Graph" target="_blank">Computation Graph</a> </li>
<li><a href="https://www.youtube.com/watch?v=z_xiwjEdAC4&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=16" title="Logistic Regression Gradient Descent" target="_blank">Logistic Regression Gradient Descent</a> </li>
<li><a href="https://www.youtube.com/watch?v=qsIrQi0fzbY&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=18" title="Vectorization" target="_blank">Vectorization</a></li>
<li><a href="https://www.youtube.com/watch?v=okpqeEUdEkY&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=20" title="Vectorizing Logistic Regression" target="_blank">Vectorizing Logistic Regression</a></li>
<li><a href="https://www.youtube.com/watch?v=2BkqApHKwn0&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=21" title="Vectorizing Logistic Regression's Gradient Computation" target="_blank">Vectorizing Logistic Regression’s Gradient Computation</a> </li>
<li><a href="https://www.youtube.com/watch?v=V2QlTmh6P2Y&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=23" title="A Note on Python/Numpy Vectors" target="_blank">A Note on Python/Numpy Vectors</a> </li>
<li><a href="https://www.youtube.com/watch?v=CcRkHl75Z-Y&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=27" title="Neural Network Representations" target="_blank">Neural Network Representations</a> </li>
<li><a href="https://www.youtube.com/watch?v=rMOdrD61IoU&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=28" title="Computing Neural Network Output" target="_blank">Computing Neural Network Output</a> </li>
<li><a href="https://www.youtube.com/watch?v=xy5MOQpx3aQ&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=29" title="Vectorizing Across Multiple Examples" target="_blank">Vectorizing Across Multiple Examples</a> </li>
<li><a href="https://www.youtube.com/watch?v=7bLEWDZng_M&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=34" title="Gradient Descent For Neural Networks" target="_blank">Gradient Descent For Neural Networks</a> </li>
<li><a href="https://www.youtube.com/watch?v=6by6Xas_Kho&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=36" title="Random Initialization" target="_blank">Random Initialization</a> </li>
<li><a href="https://www.youtube.com/watch?v=2gw5tE2ziqA&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=37" title="Deep L-Layer Neural Network" target="_blank">Deep L-Layer Neural Network</a> </li>
<li><a href="https://www.youtube.com/watch?v=1waHlpKiNyY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc" title="Train/Dev/Test Sets" target="_blank">Train/Dev/Test Sets</a> </li>
<li><a href="https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e" title="Random Initialization For Neural Networks : A Thing Of The Past" target="_blank">Random Initialization For Neural Networks : A Thing Of The Past</a> </li>
<li><a href="https://ww1.deepdish.io/?sub1=4b797490-7fb3-11ed-b724-921214c455a1" title="Initialization of deep networks" target="_blank">Initialization of deep networks</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Multiclass_classification" title="Multiclass classification" target="_blank">Multiclass classification</a> </li>
<li><a href="https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/" title="Derivation: Derivatives for Common Neural Network Activation Functions" target="_blank">Derivation: Derivatives for Common Neural Network Activation Functions</a> </li>
<li><a href="https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f?gi=a4f47cf027f7" title="What is One Hot Encoding? Why And When do you have to use it?" target="_blank">What is One Hot Encoding? Why And When do you have to use it?</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Softmax_function" title="Softmax function" target="_blank">Softmax function</a> </li>
<li><a href="https://www.quora.com/What-is-the-intuition-behind-SoftMax-function" title="What is the intuition behind SoftMax function?" target="_blank">What is the intuition behind SoftMax function?</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Cross_entropy" title="Cross entropy" target="_blank">Cross entropy</a> </li>
<li><a href="https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy" title="Loss Functions: Cross-Entropy" target="_blank">Loss Functions: Cross-Entropy</a> </li>
<li><a href="https://www.youtube.com/watch?v=LLux1SW--oM&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=32" title="Softmax Regression" target="_blank">Softmax Regression</a> (<em>Note: I suggest watching this video at 1.5x - 2x speed</em>)</li>
<li><a href="https://www.youtube.com/watch?v=ueO_Ph0Pyqk&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=33" title="Training Softmax Classifier" target="_blank">Training Softmax Classifier</a> (<em>Note: I suggest watching this video at 1.5x - 2x speed</em>)</li>
<li><a href="https://numpy.org/doc/1.18/reference/generated/numpy.zeros.html" title="numpy.zeros" target="_blank">numpy.zeros</a> </li>
<li><a href="https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randn.html" title="numpy.random.randn" target="_blank">numpy.random.randn</a> </li>
<li><a href="https://numpy.org/doc/1.18/reference/generated/numpy.exp.html" title="numpy.exp" target="_blank">numpy.exp</a> </li>
<li><a href="https://numpy.org/doc/1.18/reference/generated/numpy.log.html" title="numpy.log" target="_blank">numpy.log</a> </li>
<li><a href="https://numpy.org/doc/1.18/reference/generated/numpy.sqrt.html" title="numpy.sqrt" target="_blank">numpy.sqrt</a> </li>
<li><a href="https://numpy.org/doc/1.18/reference/generated/numpy.where.html" title="numpy.where" target="_blank">numpy.where</a> </li>
<li><a href="https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.amax.html" title="numpy.max" target="_blank">numpy.max</a> </li>
<li><a href="https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.sum.html" title="numpy.sum" target="_blank">numpy.sum</a> </li>
<li><a href="https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.argmax.html" title="numpy.argmax" target="_blank">numpy.argmax</a> </li>
<li><a href="https://yasoob.me/2013/08/02/what-is-pickle-in-python/" title="What is Pickle in python?" target="_blank">What is Pickle in python?</a> </li>
<li><a href="https://docs.python.org/3/library/pickle.html" title="pickle" target="_blank">pickle</a> </li>
<li><a href="https://docs.python.org/3/library/pickle.html#pickle.dump" title="pickle.dump" target="_blank">pickle.dump</a> </li>
<li><a href="https://docs.python.org/3/library/pickle.html#pickle.load" title="pickle.load" target="_blank">pickle.load</a> </li>
</ul>
<p><strong>Optional</strong>:</p>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Predictive_analytics" title="Predictive analytics" target="_blank">Predictive analytics</a> </li>
<li><a href="https://towardsdatascience.com/maximum-likelihood-estimation-984af2dcfcac" title="Maximum Likelihood Estimation" target="_blank">Maximum Likelihood Estimation</a></li>
</ul>

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
* What is multiclass classification?
* What is a one-hot vector?
* How to encode/decode one-hot vectors
* What is the softmax function and when do you use it?
* What is cross-entropy loss?
* What is pickling in Python?

<h2>More Info</h2>
<h3>Matrix Multiplications</h3>
<p>For all matrix multiplications in the following tasks, please use <a href="/rltoken/Ox8bY8ogmUftzjR96IrMDw" title="numpy.matmul" target="_blank">numpy.matmul</a></p>
<h3>Testing your code</h3>
<p>In order to test your code, you’ll need DATA! Please download these datasets (<a href="https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-ml/Binary_Train.npz" title="Binary_Train.npz" target="_blank">Binary_Train.npz</a>, <a href="https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-ml/Binary_Dev.npz" title="Binary_Dev.npz" target="_blank">Binary_Dev.npz</a>, <a href="https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-ml/MNIST.npz" title="MNIST.npz" target="_blank">MNIST.npz</a>) to go along with all of the following main files. You <strong>do not</strong> need to upload these files to GitHub. Your code will not necessarily be tested with these datasets. All of the following code assumes that you have stored all of your datasets in a separate <code>data</code> directory.</p>
<pre><code>alexa@ubuntu-xenial:$ cat show_data.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(Y[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./show_data.py
alexa@ubuntu-xenial:$
alexa@ubuntu-xenial:$ cat show_multi_data.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

lib = np.load('../data/MNIST.npz')
print(lib.files)
X_train_3D = lib['X_train']
Y_train = lib['Y_train']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_train_3D[i])
    plt.title(str(Y_train[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./show_multi_data.py
['Y_test', 'X_test', 'X_train', 'Y_train', 'X_valid', 'Y_valid']
</code></pre>
<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/testBinaryTrain.png" alt="" loading="lazy" style=""></p>

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

<div class="panel-heading panel-heading-actions">
    <h3 class="panel-title">
      1. Privatize Neuron
    </h3>

</div>
<p>Write a class <code>Neuron</code> that defines a single neuron performing binary classification (Based on <code>0-neuron.py</code>):</p>
<ul>
<li>class constructor: <code>def __init__(self, nx):</code>

<ul>
<li><code>nx</code> is the number of input features to the neuron

<ul>
<li>If <code>nx</code> is not an integer, raise a <code>TypeError</code> with the exception: <code>nx must be a integer</code></li>
<li>If <code>nx</code> is less than 1, raise a <code>ValueError</code> with the exception: <code>nx must be positive</code></li>
</ul></li>
<li>All exceptions should be raised in the order listed above</li>
</ul></li>
<li><strong>Private</strong> instance attributes:

<ul>
<li><code>__W</code>: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.</li>
<li><code>__b</code>: The bias for the neuron. Upon instantiation, it should be initialized to 0.</li>
<li><code>__A</code>: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.</li>
<li>Each private attribute should have a corresponding getter function (no setter function).</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('1-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)
alexa@ubuntu-xenial:$ ./1-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0
0
Traceback (most recent call last):
  File "./1-main.py", line 16, in &lt;module&gt;
    neuron.A = 10
AttributeError: can't set attribute
alexa@ubuntu-xenial:$
</code></pre>
File: <code>1-neuron.py</code>

<div class="panel-heading panel-heading-actions">
    <h3 class="panel-title">
      2. Neuron Forward Propagation
    </h3>
</div>
<p>Write a class <code>Neuron</code> that defines a single neuron performing binary classification (Based on <code>1-neuron.py</code>):</p>
<ul>
<li>Add the public method <code>def forward_prop(self, X):</code>

<ul>
<li>Calculates the forward propagation of the neuron</li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li>Updates the private attribute <code>__A</code></li>
<li>The neuron should use a sigmoid activation function</li>
<li>Returns the private attribute <code>__A</code></li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('2-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
neuron._Neuron__b = 1
A = neuron.forward_prop(X)
if (A is neuron.A):
        print(A)
vagrant@ubuntu-xe
alexa@ubuntu-xenial:$ ./2-main.py
[[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
  1.13141966e-06 6.55799932e-01]]
alexa@ubuntu-xenial:$
</code></pre>
File: <code>2-neuron.py</code>

<div class="panel-heading panel-heading-actions">
    <h3 class="panel-title">
      3. Neuron Cost
    </h3>
</div>
<p>Write a class <code>Neuron</code> that defines a single neuron performing binary classification (Based on <code>2-neuron.py</code>):</p>
<ul>
<li>Add the public method <code>def cost(self, Y, A):</code>

<ul>
<li>Calculates the cost of the model using logistic regression</li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li><code>A</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) containing the activated output of the neuron for each example</li>
<li>To avoid division by zero errors, please use <code>1.0000001 - A</code> instead of <code>1 - A</code></li>
<li>Returns the cost</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('3-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
cost = neuron.cost(Y, A)
print(cost)
alexa@ubuntu-xenial:$ ./3-main.py
4.365104944262272
alexa@ubuntu-xenial:$
</code></pre>
File: <code>3-neuron.py</code>

<h3 class="panel-title">4. Evaluate Neuron</h3>
<p>Write a class <code>Neuron</code> that defines a single neuron performing binary classification (Based on <code>3-neuron.py</code>):</p>
<ul>
<li>Add the public method <code>def evaluate(self, X, Y):</code>

<ul>
<li>Evaluates the neuron’s predictions</li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li>Returns the neuron’s prediction and the cost of the network, respectively

<ul>
<li>The prediction should be a <code>numpy.ndarray</code> with shape (1, <code>m</code>) containing the predicted labels for each example</li>
<li>The label values should be 1 if the output of the network is &gt;= 0.5 and 0 otherwise</li>
</ul></li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('4-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A, cost = neuron.evaluate(X, Y)
print(A)
print(cost)
alexa@ubuntu-xenial:$ ./4-main.py
[[0 0 0 ... 0 0 0]]
4.365104944262272
alexa@ubuntu-xenial:$
</code></pre>
File: <code>4-neuron.py</code>

<h3 class="panel-title">
      5. Neuron Gradient Descent
</h3>
<p>Write a class <code>Neuron</code> that defines a single neuron performing binary classification (Based on <code>4-neuron.py</code>):</p>
<ul>
<li>Add the public method <code>def gradient_descent(self, X, Y, A, alpha=0.05):</code>

<ul>
<li>Calculates one pass of gradient descent on the neuron</li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li><code>A</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) containing the activated output of the neuron for each example</li>
<li><code>alpha</code> is the learning rate</li>
<li>Updates the private attributes <code>__W</code> and <code>__b</code></li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('5-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
neuron.gradient_descent(X, Y, A, 0.5)
print(neuron.W)
print(neuron.b)
alexa@ubuntu-xenial:$ ./5-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0.2579495783615682
alexa@ubuntu-xenial:$
</code></pre>
File: <code>5-neuron.py</code>

<h3 class="panel-title">
      6. Train Neuron
    </h3>
<p>Write a class <code>Neuron</code> that defines a single neuron performing binary classification (Based on <code>5-neuron.py</code>):</p>
<ul>
<li>Add the public method <code>def train(self, X, Y, iterations=5000, alpha=0.05):</code>

<ul>
<li>Trains the neuron</li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li><code>iterations</code> is the number of iterations to train over

<ul>
<li>if <code>iterations</code> is not an integer, raise a <code>TypeError</code> with the exception <code>iterations must be an integer</code></li>
<li>if <code>iterations</code> is not positive, raise a <code>ValueError</code> with the exception <code>iterations must be a positive integer</code></li>
</ul></li>
<li><code>alpha</code> is the learning rate

<ul>
<li>if <code>alpha</code> is not a float, raise a <code>TypeError</code> with the exception <code>alpha must be a float</code></li>
<li>if <code>alpha</code> is not positive, raise a <code>ValueError</code> with the exception <code>alpha must be positive</code></li>
</ul></li>
<li>All exceptions should be raised in the order listed above</li>
<li>Updates the private attributes <code>__W</code>, <code>__b</code>, and <code>__A</code></li>
<li>You are allowed to use one loop</li>
<li>Returns the evaluation of the training data after <code>iterations</code> of training have occurred</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 6-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('6-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=10)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", np.round(cost, decimals=10))
print("Train accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Train data:", np.round(A, decimals=10))
print("Train Neuron A:", np.round(neuron.A, decimals=10))

A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", np.round(cost, decimals=10))
print("Dev accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Dev data:", np.round(A, decimals=10))
print("Dev Neuron A:", np.round(neuron.A, decimals=10))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()

alexa@ubuntu-xenial:$ ./6-main.py
Train cost: 1.3805076999
Train accuracy: 64.737465456%
Train data: [[0 0 0 ... 0 0 1]]
Train Neuron A: [[2.70000000e-08 2.18229559e-01 1.63492900e-04 ... 4.66530830e-03
  6.06518000e-05 9.73817942e-01]]
Dev cost: 1.4096194345
Dev accuracy: 64.4917257683%
Dev data: [[1 0 0 ... 0 0 1]]
Dev Neuron A: [[0.85021134 0.         0.3526692  ... 0.10140937 0.         0.99555018]]
</code></pre>
<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/10/7251476d2c799bd66b1b.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20230325T230517Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=053b72086266d40879867d3e7c9522a124e35497a619ace0da7e52f2b0160792" alt="" loading="lazy" style=""></p>
<p><em>Not that great… Let’s get more data!</em></p>
File: <code>6-neuron.py</code>

<h3 class="panel-title">
      7. Upgrade Train Neuron
    </h3>
<p>Write a class <code>Neuron</code> that defines a single neuron performing binary classification (Based on <code>6-neuron.py</code>):</p>
<ul>
<li>Update the public method <code>train</code> to <code>def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):</code>

<ul>
<li>Trains the neuron by updating the private attributes <code>__W</code>, <code>__b</code>, and <code>__A</code></li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li><code>iterations</code> is the number of iterations to train over

<ul>
<li>if <code>iterations</code> is not an integer, raise a <code>TypeError</code> with the exception <code>iterations must be an integer</code></li>
<li>if <code>iterations</code> is not positive, raise a <code>ValueError</code> with the exception <code>iterations must be a positive integer</code></li>
</ul></li>
<li><code>alpha</code> is the learning rate

<ul>
<li>if <code>alpha</code> is not a float, raise a <code>TypeError</code> with the exception <code>alpha must be a float</code></li>
<li>if <code>alpha</code> is not positive, raise a <code>ValueError</code> with the exception <code>alpha must be positive</code></li>
</ul></li>
<li><code>verbose</code> is a boolean that defines whether or not to print information about the training. If <code>True</code>, print <code>Cost after {iteration} iterations: {cost}</code> every <code>step</code> iterations:

<ul>
<li> Include data from the 0th and last iteration</li>
</ul></li>
<li> <code>graph</code> is a boolean that defines whether or not to graph information about the training once the training has completed. If <code>True</code>:

<ul>
<li> Plot the training data every <code>step</code> iterations as a blue line</li>
<li> Label the x-axis as <code>iteration</code></li>
<li> Label the y-axis as <code>cost</code></li>
<li> Title the plot <code>Training Cost</code></li>
<li> Include data from the 0th and last iteration</li>
</ul></li>
<li> Only if either <code>verbose</code> or <code>graph</code> are <code>True</code>:

<ul>
<li>if <code>step</code> is not an integer, raise a <code>TypeError</code> with the exception <code>step must be an integer</code></li>
<li>if <code>step</code> is not positive or is greater than <code>iterations</code>, raise a <code>ValueError</code> with the exception <code>step must be positive and &lt;= iterations</code></li>
</ul></li>
<li>All exceptions should be raised in the order listed above</li>
<li> The 0th iteration should represent the state of the neuron before any training has occurred</li>
<li> You are allowed to use one loop</li>
<li> You can use <code>import matplotlib.pyplot as plt</code></li>
<li>Returns: the evaluation of the training data after <code>iterations</code> of training have occurred</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 7-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('7-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=3000)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./7-main.py
Cost after 0 iterations: 4.365104944262272
Cost after 100 iterations: 0.11955134491351888

...

Cost after 3000 iterations: 0.013386353289868338
</code></pre>
<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/10/23a1fc5f90d79c63bd78.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20230325T230517Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=f381099e429b769c3ed1eb4facca8e27da30dd5e99dd5e81d7ce6e5c8f1e0caa" alt="" loading="lazy" style=""></p>
<pre><code>Train cost: 0.013386353289868338
Train accuracy: 99.66837741808132%
Dev cost: 0.010803484515167197
Dev accuracy: 99.81087470449172%
</code></pre>
<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/10/6ffa61a3b13fb602f400.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20230325T230517Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=4e644b8cf749856056f82767766c9135d268be6e8122cb2093e6f8aa21b9164f" alt="" loading="lazy" style=""></p>
File: <code>7-neuron.py</code>

<h3 class="panel-title">
    8. NeuralNetwork
</h3>
<p>Write a class <code>NeuralNetwork</code> that defines a neural network with one hidden layer performing binary classification:</p>
<ul>
<li>class constructor: <code>def __init__(self, nx, nodes):</code>

<ul>
<li><code>nx</code> is the number of input features

<ul>
<li>If <code>nx</code> is not an integer, raise a <code>TypeError</code> with the exception: <code>nx must be an integer</code></li>
<li>If <code>nx</code> is less than 1, raise a <code>ValueError</code> with the exception: <code>nx must be a positive integer</code></li>
</ul></li>
<li><code>nodes</code> is the number of nodes found in the hidden layer

<ul>
<li>If <code>nodes</code> is not an integer, raise a <code>TypeError</code> with the exception: <code>nodes must be an integer</code></li>
<li>If <code>nodes</code> is less than 1, raise a <code>ValueError</code> with the exception: <code>nodes must be a positive integer</code></li>
</ul></li>
<li>All exceptions should be raised in the order listed above</li>
</ul></li>
<li>Public instance attributes:

<ul>
<li><code>W1</code>: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.</li>
<li><code>b1</code>: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.</li>
<li><code>A1</code>: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.</li>
<li><code>W2</code>: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.</li>
<li><code>b2</code>: The bias for the output neuron. Upon instantiation, it should be initialized to 0.</li>
<li><code>A2</code>: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 8-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('8-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
print(nn.W1)
print(nn.W1.shape)
print(nn.b1)
print(nn.W2)
print(nn.W2.shape)
print(nn.b2)
print(nn.A1)
print(nn.A2)
nn.A1 = 10
print(nn.A1)
alexa@ubuntu-xenial:$ ./8-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
(3, 784)
[[0.]
 [0.]
 [0.]]
[[ 1.06160017 -1.18488744 -1.80525169]]
(1, 3)
0
0
0
10
alexa@ubuntu-xenial:$
</code></pre>
File:<code>8-neural_network.py</code>

<h3 class="panel-title">
      9. Privatize NeuralNetwork
    </h3>
<p>Write a class <code>NeuralNetwork</code> that defines a neural network with one hidden layer performing binary classification (based on <code>8-neural_network.py</code>):</p>
<ul>
<li>class constructor: <code>def __init__(self, nx, nodes):</code>

<ul>
<li><code>nx</code> is the number of input features

<ul>
<li>If <code>nx</code> is not an integer, raise a <code>TypeError</code> with the exception: <code>nx must be an integer</code></li>
<li>If <code>nx</code> is less than 1, raise a <code>ValueError</code> with the exception: <code>nx must be a positive integer</code></li>
</ul></li>
<li><code>nodes</code> is the number of nodes found in the hidden layer

<ul>
<li>If <code>nodes</code> is not an integer, raise a <code>TypeError</code> with the exception: <code>nodes must be an integer</code></li>
<li>If <code>nodes</code> is less than 1, raise a <code>ValueError</code> with the exception: <code>nodes must be a positive integer</code></li>
</ul></li>
<li>All exceptions should be raised in the order listed above</li>
</ul></li>
<li><strong>Private</strong> instance attributes:

<ul>
<li><code>W1</code>: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.</li>
<li><code>b1</code>: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.</li>
<li><code>A1</code>: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.</li>
<li><code>W2</code>: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.</li>
<li><code>b2</code>: The bias for the output neuron. Upon instantiation, it should be initialized to 0.</li>
<li><code>A2</code>: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.</li>
<li>Each private attribute should have a corresponding getter function (no setter function).</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 9-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('9-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
print(nn.A1)
print(nn.A2)
nn.A1 = 10
print(nn.A1)
alexa@ubuntu-xenial:$ ./9-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
[[0.]
 [0.]
 [0.]]
[[ 1.06160017 -1.18488744 -1.80525169]]
0
0
0
Traceback (most recent call last):
  File "./9-main.py", line 19, in &lt;module&gt;
    nn.A1 = 10
AttributeError: can't set attribute
alexa@ubuntu-xenial:$
</code></pre>
File:<code>9-neural_network.py</code>

<h3 class="panel-title">
      10. NeuralNetwork Forward Propagation
    </h3>
<p>Write a class <code>NeuralNetwork</code> that defines a neural network with one hidden layer performing binary classification (based on <code>9-neural_network.py</code>):</p>
<ul>
<li>Add the public method <code>def forward_prop(self, X):</code>

<ul>
<li>Calculates the forward propagation of the neural network</li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li>Updates the private attributes <code>__A1</code> and <code>__A2</code></li>
<li>The neurons should use a sigmoid activation function</li>
<li>Returns the private attributes <code>__A1</code> and <code>__A2</code>, respectively</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 10-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('10-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
nn._NeuralNetwork__b1 = np.ones((3, 1))
nn._NeuralNetwork__b2 = 1
A1, A2 = nn.forward_prop(X)
if A1 is nn.A1:
        print(A1)
if A2 is nn.A2:
        print(A2)
alexa@ubuntu-xenial:$ ./10-main.py
[[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
  1.13141966e-06 6.55799932e-01]
 [9.99652394e-01 9.99999995e-01 6.77919152e-01 ... 1.00000000e+00
  9.99662771e-01 9.99990554e-01]
 [5.57969669e-01 2.51645047e-02 4.04250047e-04 ... 1.57024117e-01
  9.97325173e-01 7.41310459e-02]]
[[0.23294587 0.44286405 0.54884691 ... 0.38502756 0.12079644 0.593269  ]]
alexa@ubuntu-xenial:$
</code></pre>
File:<code>10-neural_network.py</code>

<div class="panel-heading panel-heading-actions">
    <h3 class="panel-title">
      11. NeuralNetwork Cost
    </h3>
</div>
<p>Write a class <code>NeuralNetwork</code> that defines a neural network with one hidden layer performing binary classification (based on <code>10-neural_network.py</code>):</p>
<ul>
<li>Add the public method <code>def cost(self, Y, A):</code>

<ul>
<li>Calculates the cost of the model using logistic regression</li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li><code>A</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) containing the activated output of the neuron for each example</li>
<li>To avoid division by zero errors, please use <code>1.0000001 - A</code> instead of <code>1 - A</code></li>
<li>Returns the cost</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 11-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('11-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
_, A = nn.forward_prop(X)
cost = nn.cost(Y, A)
print(cost)
alexa@ubuntu-xenial:$ ./11-main.py
0.7917984405648548
alexa@ubuntu-xenial:$
</code></pre>
File: <code>11-neural_network.py</code>

<h3 class="panel-title">
      12. Evaluate NeuralNetwork
    </h3>
<p>Write a class <code>NeuralNetwork</code> that defines a neural network with one hidden layer performing binary classification (based on <code>11-neural_network.py</code>):</p>
<ul>
<li>Add the public method <code>def evaluate(self, X, Y):</code>

<ul>
<li>Evaluates the neural network’s predictions</li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li>Returns the neuron’s prediction and the cost of the network, respectively

<ul>
<li>The prediction should be a <code>numpy.ndarray</code> with shape (1, <code>m</code>) containing the predicted labels for each example</li>
<li>The label values should be 1 if the output of the network is &gt;= 0.5 and 0 otherwise</li>
</ul></li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 12-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('12-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
A, cost = nn.evaluate(X, Y)
print(A)
print(cost)
alexa@ubuntu-xenial:$ ./12-main.py
[[0 0 0 ... 0 0 0]]
0.7917984405648548
alexa@ubuntu-xenial:$
</code></pre>
File:<code>12-neural_network.py</code>

<h3 class="panel-title">
      13. NeuralNetwork Gradient Descent
    </h3>
<p>Write a class <code>NeuralNetwork</code> that defines a neural network with one hidden layer performing binary classification (based on <code>12-neural_network.py</code>):</p>
<ul>
<li>Add the public method <code>def gradient_descent(self, X, Y, A1, A2, alpha=0.05):</code>

<ul>
<li>Calculates one pass of gradient descent on the neural network</li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li><code>A1</code> is the output of the hidden layer</li>
<li><code>A2</code> is the predicted output</li>
<li><code>alpha</code> is the learning rate</li>
<li>Updates the private attributes <code>__W1</code>, <code>__b1</code>, <code>__W2</code>, and <code>__b2</code></li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 13-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('13-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
A1, A2 = nn.forward_prop(X)
nn.gradient_descent(X, Y, A1, A2, 0.5)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
alexa@ubuntu-xenial:$ ./13-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
[[ 0.003193  ]
 [-0.01080922]
 [-0.01045412]]
[[ 1.06583858 -1.06149724 -1.79864091]]
[[0.15552509]]
alexa@ubuntu-xenial:$
</code></pre>
File:<code>13-neural_network.py</code>

<h3 class="panel-title">
      14. Train NeuralNetwork
    </h3>
<p>Write a class <code>NeuralNetwork</code> that defines a neural network with one hidden layer performing binary classification (based on <code>13-neural_network.py</code>):</p>
<ul>
<li>Add the public method <code>def train(self, X, Y, iterations=5000, alpha=0.05):</code>

<ul>
<li>Trains the neural network</li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li><code>iterations</code> is the number of iterations to train over

<ul>
<li>if <code>iterations</code> is not an integer, raise a <code>TypeError</code> with the exception <code>iterations must be an integer</code></li>
<li>if <code>iterations</code> is not positive, raise a <code>ValueError</code> with the exception <code>iterations must be a positive integer</code></li>
</ul></li>
<li><code>alpha</code> is the learning rate

<ul>
<li>if <code>alpha</code> is not a float, raise a <code>TypeError</code> with the exception <code>alpha must be a float</code></li>
<li>if <code>alpha</code> is not positive, raise a <code>ValueError</code> with the exception <code>alpha must be positive</code></li>
</ul></li>
<li>All exceptions should be raised in the order listed above</li>
<li>Updates the private attributes <code>__W1</code>, <code>__b1</code>,  <code>__A1</code>, <code>__W2</code>, <code>__b2</code>, and <code>__A2</code></li>
<li>You are allowed to use one loop</li>
<li>Returns the evaluation of the training data after <code>iterations</code> of training have occurred</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 14-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

NN = __import__('14-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train, iterations=100)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./14-main.py
Train cost: 0.4680930945144984
Train accuracy: 84.69009080142123%
Dev cost: 0.45985938789496067
Dev accuracy: 86.52482269503547%
alexa@ubuntu-xenial:$
</code></pre>
<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/10/c744ae1dae04204a56ab.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20230325T230517Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=db249025f052ef0fc607f5544ebc7f8188e5a7918e4fe1aad31f6e22cff2bac0" alt="" loading="lazy" style=""></p>
<p><em>Pretty good… but there are still some incorrect labels. We need more data to see why…</em></p>
File:<code>14-neural_network.py</code>

<h3 class="panel-title">
      15. Upgrade Train NeuralNetwork
    </h3>
<p>Write a class <code>NeuralNetwork</code> that defines a neural network with one hidden layer performing binary classification (based on <code>14-neural_network.py</code>):</p>
<ul>
<li>Update the public method <code>train</code> to <code>def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):</code>

<ul>
<li>Trains the neural network</li>
<li><code>X</code> is a <code>numpy.ndarray</code> with shape (<code>nx</code>, <code>m</code>) that contains the input data

<ul>
<li><code>nx</code> is the number of input features to the neuron</li>
<li><code>m</code> is the number of examples</li>
</ul></li>
<li><code>Y</code> is a <code>numpy.ndarray</code> with shape (1, <code>m</code>) that contains the correct labels for the input data</li>
<li><code>iterations</code> is the number of iterations to train over

<ul>
<li>if <code>iterations</code> is not an integer, raise a <code>TypeError</code> with the exception <code>iterations must be an integer</code></li>
<li>if <code>iterations</code> is not positive, raise a <code>ValueError</code> with the exception <code>iterations must be a positive integer</code></li>
</ul></li>
<li><code>alpha</code> is the learning rate

<ul>
<li>if <code>alpha</code> is not a float, raise a <code>TypeError</code> with the exception <code>alpha must be a float</code></li>
<li>if <code>alpha</code> is not positive, raise a <code>ValueError</code> with the exception <code>alpha must be positive</code></li>
</ul></li>
<li>Updates the private attributes <code>__W1</code>, <code>__b1</code>, <code>__A1</code>, <code>__W2</code>, <code>__b2</code>, and <code>__A2</code></li>
<li><code>verbose</code> is a boolean that defines whether or not to print information about the training. If <code>True</code>, print <code>Cost after {iteration} iterations: {cost}</code> every <code>step</code> iterations:

<ul>
<li> Include data from the 0th and last iteration</li>
</ul></li>
<li> <code>graph</code> is a boolean that defines whether or not to graph information about the training once the training has completed. If <code>True</code>:

<ul>
<li> Plot the training data every <code>step</code> iterations as a blue line</li>
<li> Label the x-axis as <code>iteration</code></li>
<li> Label the y-axis as <code>cost</code></li>
<li> Title the plot <code>Training Cost</code></li>
<li> Include data from the 0th and last iteration</li>
</ul></li>
<li> Only if either <code>verbose</code> or <code>graph</code> are <code>True</code>:

<ul>
<li>if <code>step</code> is not an integer, raise a <code>TypeError</code> with the exception <code>step must be an integer</code></li>
<li>if <code>step</code> is not positive and less than or equal to <code>iterations</code>, raise a <code>ValueError</code> with the exception <code>step must be positive and &lt;= iterations</code></li>
</ul></li>
<li>All exceptions should be raised in the order listed above</li>
<li> The 0th iteration should represent the state of the neuron before any training has occurred</li>
<li> You are allowed to use one loop</li>
<li>Returns the evaluation of the training data after <code>iterations</code> of training have occurred</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 15-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

NN = __import__('15-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./15-main.py
Cost after 0 iterations: 0.7917984405648547
Cost after 100 iterations: 0.4680930945144984

...

Cost after 5000 iterations: 0.024369225667283875
</code></pre>
<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/10/eab7b0b266e6b2b671f9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20230325T230517Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=b3dc0d4475d6727d8d4d98b1147d1a7d125c6222014c10b14d62547220a07100" alt="" loading="lazy" style=""></p>
<pre><code>Train cost: 0.024369225667283875
Train accuracy: 99.3999210422424%
Dev cost: 0.020330639788072768
Dev accuracy: 99.57446808510639%
</code></pre>
<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/10/af979f502eec55142860.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20230325T230517Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=a9e058979fd39acd67dadb7f2689e3f32e4fe9e424819c9fa58cad9434386a19" alt="" loading="lazy" style=""></p>
File:<code>15-neural_network.py</code>

<h3 class="panel-title">
      16. DeepNeuralNetwork
    </h3>
<p>Write a class <code>DeepNeuralNetwork</code> that defines a deep neural network performing binary classification:</p>
<ul>
<li>class constructor: <code>def __init__(self, nx, layers):</code>

<ul>
<li><code>nx</code> is the number of input features

<ul>
<li>If <code>nx</code> is not an integer, raise a <code>TypeError</code> with the exception: <code>nx must be an integer</code></li>
<li>If <code>nx</code> is less than 1, raise a <code>ValueError</code> with the exception: <code>nx must be a positive integer</code></li>
</ul></li>
<li><code>layers</code> is a list representing the number of nodes in each layer of the network

<ul>
<li>If <code>layers</code> is not a list or an empty list, raise a <code>TypeError</code> with the exception: <code>layers must be a list of positive integers</code></li>
<li>The first value in <code>layers</code> represents the number of nodes in the first layer, …</li>
<li>If the elements in <code>layers</code> are not all positive integers, raise a <code>TypeError</code> with the exception <code>layers must be a list of positive integers</code></li>
</ul></li>
<li>All exceptions should be raised in the order listed above</li>
<li>Sets the public instance attributes:

<ul>
<li><code>L</code>: The number of layers in the neural network.</li>
<li><code>cache</code>: A dictionary to hold all intermediary values of the network. Upon instantiation, it should be set to an empty dictionary.</li>
<li><code>weights</code>: A dictionary to hold all weights and biased of the network. Upon instantiation:

<ul>
<li>The weights of the network should be initialized using the <code>He et al.</code> method and saved in the <code>weights</code> dictionary using the key <code>W{l}</code> where <code>{l}</code> is the hidden layer the weight belongs to</li>
<li>The biases of the network should be initialized to 0’s and saved in the <code>weights</code> dictionary using the key <code>b{l}</code> where <code>{l}</code> is the hidden layer the bias belongs to</li>
</ul></li>
</ul></li>
<li>You are allowed to use one loop</li>
</ul></li>
</ul>
<pre><code>alexa@ubuntu-xenial:$ cat 16-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print(deep.L)
alexa@ubuntu-xenial:$ ./16-main.py
{}
{'b3': array([[0.]]), 'W2': array([[ 0.4609219 ,  0.56004008, -1.2250799 , -0.09454199,  0.57799141],
       [-0.16310703,  0.06882082, -0.94578088, -0.30359994,  1.15661914],
       [-0.49841799, -0.9111359 ,  0.09453424,  0.49877298,  0.75503205]]), 'W3': array([[-0.42271877,  0.18165055,  0.4444639 ]]), 'b2': array([[0.],
       [0.],
       [0.]]), 'W1': array([[ 0.0890981 ,  0.02021099,  0.04943373, ...,  0.02632982,
         0.03090699, -0.06775582],
       [ 0.02408701,  0.00749784,  0.02672082, ...,  0.00484894,
        -0.00227857,  0.00399625],
       [ 0.04295829, -0.04238217, -0.05110231, ..., -0.00364861,
         0.01571416, -0.05446546],
       [ 0.05361891, -0.05984585, -0.09117898, ..., -0.03094292,
        -0.01925805, -0.06308145],
       [-0.01667953, -0.04216413,  0.06239623, ..., -0.02024521,
        -0.05159656, -0.02373981]]), 'b1': array([[0.],
       [0.],
       [0.],
       [0.],
       [0.]])}
3
10
alexa@ubuntu-xenial:$
</code></pre>
File:<code>16-deep_neural_network.py</code>

