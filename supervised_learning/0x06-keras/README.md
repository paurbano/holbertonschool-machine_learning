# 0x06. Keras

## General
* **What is Keras?**
    is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML.It offers a higher-level, more intuitive set of abstractions that make it easy to develop deep learning models regardless of the computational backend used
  
* **What is a model?**
    model can be a mathematical representation of a real-world process.
* **How to instantiate a model (2 ways)**
    You can create a Sequential model by passing a list of layers to the Sequential constructor:
    ```
    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu"),
            layers.Dense(3, activation="relu"),
            layers.Dense(4),
        ]
    )
    ```
    You can also create a Sequential model incrementally via the add() method:
    ```
    model = keras.Sequential()
    model.add(layers.Dense(2, activation="relu"))
    model.add(layers.Dense(3, activation="relu"))
    model.add(layers.Dense(4))
    ```
* **How to build a layer**
    ```
    # Create 3 layers
    layer1 = layers.Dense(2, activation="relu", name="layer1")
    layer2 = layers.Dense(3, activation="relu", name="layer2")
    ```

* How to add regularization to a layer
* How to add dropout to a layer
* How to add batch normalization
* How to compile a model
* How to optimize a model
* How to fit a model
* How to use validation data
* How to perform early stopping
* How to measure accuracy
* How to evaluate a model
* How to make a prediction with a model
* How to access the weights/outputs of a model
* What is HDF5?
* How to save and load a model’s weights, a model’s configuration, and the entire model

## 0. Sequential
Write a function `def build_model(nx, layers, activations, lambtha, keep_prob):` that builds a neural network with the Keras library:

* `nx` is the number of input features to the network
* `layers` is a list containing the number of nodes in each layer of the network
* `activations` is a list containing the activation functions used for each layer of the network
* `lambtha` is the L2 regularization parameter
* `keep_prob` is the probability that a node will be kept for dropout
* You are not allowed to use the `Input` class
* Returns: the keras model

```
ubuntu@alexa-ml:~/0x06-keras$ cat 0-main.py 
#!/usr/bin/env python3

build_model = __import__('0-sequential').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
ubuntu@alexa-ml:~/0x06-keras$ ./0-main.py 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[<tf.Tensor 'dense/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_1/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_2/kernel/Regularizer/add:0' shape=() dtype=float32>]
ubuntu@alexa-ml:~/0x06-keras$
```
File: `0-sequential.py`

## 1. Input
Write a function `def build_model(nx, layers, activations, lambtha, keep_prob):` that builds a neural network with the Keras library:

* `nx` is the number of input features to the network
* `layers` is a list containing the number of nodes in each layer of the network
* `activations` is a list containing the activation functions used for each layer of the network
* `lambtha` is the L2 regularization parameter
* `keep_prob` is the probability that a node will be kept for dropout
* You are not allowed to use the `Sequential` class
* Returns: the keras model