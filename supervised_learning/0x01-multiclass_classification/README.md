# 0x01. Multiclass Classification

<h2 tabindex="-1" dir="auto"><a id="user-content-readme" class="anchor" aria-hidden="true" href="#readme"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>Resources</h2>
<p dir="auto"><strong>Read or watch</strong>:</p>
<ul dir="auto">
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg" title="Multiclass classification">Multiclass classification</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/krZzggd-4r5fsm7J9HNYPQ" title="Derivation: Derivatives for Common Neural Network Activation Functions">Derivation: Derivatives for Common Neural Network Activation Functions</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/2d2caYmx9ulpY1F5BQjFCw" title="What is One Hot Encoding? Why And When do you have to use it?">What is One Hot Encoding? Why And When do you have to use it?</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/qo1iqiNRmbJ6TT735yDi2w" title="Softmax function">Softmax function</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/R6SOD-SEQ5CEVwZt8BE7RQ" title="What is the intuition behind SoftMax function?">What is the intuition behind SoftMax function?</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/aAydHAsto3SH9fVuoxoPyg" title="Cross entropy">Cross entropy</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/eFtqFGQb9i87VYuVXTcSaw" title="Loss Functions: Cross-Entropy">Loss Functions: Cross-Entropy</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/Tb1OUtLpFJbpRwjkcg_3mQ" title="Softmax Regression">Softmax Regression</a> (<em>Note: I suggest watching this video at 1.5x - 2x speed</em>)</li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/elYQKuvvcOQD1m0uRzLe3w" title="Training Softmax Classifier">Training Softmax Classifier</a> (<em>Note: I suggest watching this video at 1.5x - 2x speed</em>)</li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/V0n0v0Bf3JWXL0HvNOfDYQ" title="What is Pickle in python?">What is Pickle in python?</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/dIIiln-0zMvyNU8_X2489w" title="numpy.max">numpy.max</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/TxReKsm8qpXy0bUkCsNcrA" title="numpy.sum">numpy.sum</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/khzCPahSnOFmLz9ZKmqspQ" title="numpy.argmax">numpy.argmax</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/mBhamUwokBCNKc8Do7WGUQ" title="pickle">pickle</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/BXVRHw3G2bWdh9XTxreVIA" title="pickle.dump">pickle.dump</a> </li>
<li><a href="https://intranet.hbtn.io/rltoken/ZggoiEvv6Yi28ddpDdRf5A/rltoken/Nc2jgG8os13kpadHOq4utg/rltoken/Ifm1r_Chh1s68guu-EI8ww" title="pickle.load">pickle.load</a> </li>
</ul>

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

# 1. One-Hot Encode
Write a function `def one_hot_decode(one_hot):` that converts a one-hot matrix into a vector of labels:

* `one_hot` is a one-hot encoded `numpy.ndarray` with shape `(classes, m`)
    * `classes` is the maximum number of classes
    * `m` is the number of examples
* Returns: a `numpy.ndarray` with shape (`m`, ) containing the numeric labels for each example, or `None` on failure
```
alexa@ubuntu-xenial:0x01-multiclass_classification$ ./1-main.py
    [5 0 4 1 9 2 1 3 1 4]
    [5 0 4 1 9 2 1 3 1 4]
alexa@ubuntu-xenial:0x01-multiclass_classification$
```

# 2. Persistence is Key
Update the class `DeepNeuralNetwork` (based on 23-deep_neural_network.py):

* Create the instance method `def save(self, filename):`

    * Saves the instance object to a file in `pickle` format
    * `filename` is the file to which the object should be saved
    * If `filename` does not have the extension `.pkl`, add it

* Create the static method `def load(filename)`:

    * Loads a pickled `DeepNeuralNetwork` object
    * `filename` is the file from which the object should be loaded
    * Returns: the loaded object, or `None` if `filename` doesn’t exist
```
    alexa@ubuntu-xenial:0x01-multiclass_classification$ ls 2-output*
    ls: cannot access '2-output*': No such file or directory
    alexa@ubuntu-xenial:0x01-multiclass_classification$ ./2-main.py
    Cost after 0 iterations: 0.7773240521521816
    Cost after 100 iterations: 0.18751378071323066
    Cost after 200 iterations: 0.12117095705345622
    Cost after 300 iterations: 0.09031067302785326
    Cost after 400 iterations: 0.07222364349190777
    Cost after 500 iterations: 0.060335256947006956
    True
    alexa@ubuntu-xenial:0x01-multiclass_classification$ ls 2-output*
    2-output.pkl
    alexa@ubuntu-xenial:0x01-multiclass_classification$
```
# 3. Update DeepNeuralNetwork
Update the class `DeepNeuralNetwork` to perform multiclass classification (based on `2-deep_neural_network.py`):

* You will need to update the instance methods `forward_prop`, `cost`, and `evaluate`
* `Y` is now a one-hot `numpy.ndarray` of shape (`classes, m`)

Ideally, you should not have to change the `__init__`, `gradient_descent`, or `train` instance methods

Because the training process takes such a long time, I have pretrained a model for you to load and finish training [`3-saved.pkl`](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/3-saved.pkl). This model has already been trained for 2000 iterations.

The training process may take up to 5 minutes
```
    ubuntu@alexa-ml:~$ ./3-main.py
    Cost after 0 iterations: 0.4388904112857044
    Cost after 10 iterations: 0.4377828804163359
    Cost after 20 iterations: 0.43668839872612714
    Cost after 30 iterations: 0.43560674736059446
    Cost after 40 iterations: 0.43453771176806555
    Cost after 50 iterations: 0.4334810815993252
    Cost after 60 iterations: 0.43243665061046205
    Cost after 70 iterations: 0.4314042165687683
    Cost after 80 iterations: 0.4303835811615513
    Cost after 90 iterations: 0.4293745499077264
    Cost after 100 iterations: 0.42837693207206473
    Train cost: 0.42837693207206473
    Train accuracy: 88.442%
    Validation cost: 0.39517557351173044
    Validation accuracy: 89.64%
```
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://github.com/PierreBeaujuge/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-multiclass_classification/0x01-images/img_2.png"><img src="https://github.com/PierreBeaujuge/holbertonschool-machine_learning/raw/master/supervised_learning/0x01-multiclass_classification/0x01-images/img_2.png" alt="" style="max-width: 100%;"></a>
<em>As you can see, our training has become very slow and is beginning to plateau. Let’s alter the model a little and see if we get a better result</em></p>

# 4. All the Activations
Update the class `DeepNeuralNetwork` to perform multiclass classification (based on `3-deep_neural_network.py`):

* Update the `__init__` method to `def __init__(self, nx, layers, activation='sig')`:
    * activation represents the type of activation function used in the hidden layers
        * `sig` represents a sigmoid activation
        * `tanh` represents a tanh activation
        * if `activation` is not `sig` or `tanh`, raise a `ValueError` with the exception: `activation must be   'sig' or 'tanh'`
    * Create the private attribute `__activation` and set it to the value of `activation`
    * Create a getter for the private attribute `__activation`
* Update the `forward_prop` and `gradient_descent` instance methods to use the `__activation` function in the hidden layers

Because the training process takes such a long time, I have pre-trained a model for you to load and finish training [`4-saved.pkl`](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/4-saved.pkl). This model has already been trained for 2000 iterations.

The training process may take up to 5 minutes
```
    alexa@ubuntu-xenial:0x01-multiclass_classification$ ./4-main.py
    Sigmoid activation:
    Train cost: 0.42837693207206473
    Train accuracy: 88.442%
    Validation cost: 0.39517557351173044
    Validation accuracy: 89.64%
    Test cost: 0.4074169894615401
    Test accuracy: 89.0%
```
<img src="https://github.com/PierreBeaujuge/holbertonschool-machine_learning/raw/master/supervised_learning/0x01-multiclass_classification/0x01-images/img_3.png" alt="" style="max-width: 100%;">

```
    Tanh activaiton:
    Cost after 0 iterations: 0.18061815622291985
    Cost after 10 iterations: 0.18012009542718577
    Cost after 20 iterations: 0.1796242897834926
    Cost after 30 iterations: 0.17913072860418564
    Cost after 40 iterations: 0.1786394012066576
    Cost after 50 iterations: 0.17815029691267442
    Cost after 60 iterations: 0.1776634050478437
    Cost after 70 iterations: 0.1771787149412177
    Cost after 80 iterations: 0.1766962159250237
    Cost after 90 iterations: 0.1762158973345138
    Cost after 100 iterations: 0.1757377485079266
    Train cost: 0.1757377485079266
    Train accuracy: 95.006%
    Validation cost: 0.17689309600397934
    Validation accuracy: 95.13000000000001%
    Test cost: 0.1809489808838737
    Test accuracy: 94.77%
```
<img src="https://github.com/PierreBeaujuge/holbertonschool-machine_learning/raw/master/supervised_learning/0x01-multiclass_classification/0x01-images/img_4.png" alt="" style="max-width: 100%;">

The training of this model is also getting slow and plateauing after about 2000 iterations. However, just by changing the activation function, we have nearly halved the model’s cost and increased its accuracy by about 6%
