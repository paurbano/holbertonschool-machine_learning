# 0x09. Transfer Learning

## General
* **What is a transfer learning?**
  Transfer learning generally refers to a process where a model trained on one problem is used in some way on a second related problem.
  In deep learning, transfer learning is a technique whereby a neural network model is first trained on a problem similar to the problem that is being solved.

* **What is fine-tuning?**
 consists of unfreezing the entire model and re-training it on the new data with a very low learning rate. This can potentially achieve meaningful improvements, by incrementally adapting the pretrained features to the new data. For more info refer to [Keras](https://keras.io/guides/transfer_learning/#finetuning)

* **What is a frozen layer? How and why do you freeze a layer?**
  When the weights of layer is no longer updated its value during training, these do to take advantange of its previous learning. Setting `layer.trainable` to `False` freeze the layer:
  ```
  layer = keras.layers.Dense(3)
  layer.build((None, 4))  # Create the weights
  layer.trainable = False  # Freeze the layer
  ```

* **How to use transfer learning with Keras applications**
  The most common incarnation of transfer learning in the context of deep learning is the following worfklow:

  1. Take layers from a previously trained model.
  2. Freeze them, so as to avoid destroying any of the information they contain during future training rounds.
  3. Add some new, trainable layers on top of the frozen layers. They will learn to turn the old features into predictions on a new dataset.
  4. Train the new layers on your dataset.

## 0. Transfer Knowledge
Python script that trains a convolutional neural network to classify the CIFAR 10 dataset:

* You must use one of the applications listed in Keras Applications
* Your script must save your trained model in the current working directory as cifar10.h5
* Your saved model should be compiled
* Your saved model should have a validation accuracy of 87% or higher
* Your script should not run when the file is imported

In the same file, write a function `def preprocess_data(X, Y):` that pre-processes the data for your model:

*  `X` is a `numpy.ndarray` of shape `(m, 32, 32, 3)` containing the CIFAR 10 data, where m is the number of data points
*  `Y` is a `numpy.ndarray` of shape `(m,)` containing the CIFAR 10 labels for X
* Returns: `X_p, Y_p`
  * `X_p` is a `numpy.ndarray` containing the preprocessed X
  * `Y_p` is a `numpy.ndarray` containing the preprocessed Y
```
alexa@ubuntu-xenial:0x09-transfer_learning$ cat 0-main.py
#!/usr/bin/env python3

import tensorflow.keras as K
preprocess_data = __import__('0-transfer').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
alexa@ubuntu-xenial:0x09-transfer_learning$ ./0-main.py
10000/10000 [==============================] - 159s 16ms/sample - loss: 0.3329 - acc: 0.8864
```
