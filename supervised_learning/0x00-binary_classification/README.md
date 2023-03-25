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

    <div>
        <span class="label label-info">
          mandatory
        </span>
    </div>
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

