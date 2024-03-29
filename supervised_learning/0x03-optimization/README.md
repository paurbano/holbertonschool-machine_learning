# 0x03. Optimization

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/11/2bc924532bc4a901e74d.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20231127%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20231127T215040Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=62d6a8e0d251271f0d16ddb46fce55a6768418260883ed82ffb829fc69fbca16" alt="" loading="lazy" style=""></p>
<h2>Resources</h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)" title="Hyperparameter (machine learning)" target="_blank">Hyperparameter (machine learning)</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Feature_scaling" title="Feature scaling" target="_blank">Feature scaling</a> </li>
<li><a href="https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e" title="Why, How and When to Scale your Features" target="_blank">Why, How and When to Scale your Features</a></li>
<li><a href="https://www.jeremyjordan.me/batch-normalization/" title="Normalizing your data" target="_blank">Normalizing your data</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Moving_average" title="Moving average" target="_blank">Moving average</a> </li>
<li><a href="https://www.ruder.io/optimizing-gradient-descent/" title="An overview of gradient descent optimization algorithms" target="_blank">An overview of gradient descent optimization algorithms</a> </li>
<li><a href="https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/" title="A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size" target="_blank">A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size</a> </li>
<li><a href="https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d" title="Stochastic Gradient Descent with momentum" target="_blank">Stochastic Gradient Descent with momentum</a> </li>
<li><a href="https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a" title="Understanding RMSprop" target="_blank">Understanding RMSprop</a> </li>
<li><a href="https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c" title="Adam" target="_blank">Adam</a> </li>
<li><a href="https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1" title="Learning Rate Schedules" target="_blank">Learning Rate Schedules</a></li>
<li><a href="https://www.deeplearning.ai/" title="deeplearning.ai" target="_blank">deeplearning.ai</a> videos (<em>Note: I suggest watching these video at 1.5x - 2x speed</em>):

<ul>
<li><a href="https://www.youtube.com/watch?v=FDCfw-YqWTE&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=10" title="Normalizing Inputs" target="_blank">Normalizing Inputs</a> </li>
<li><a href="https://www.youtube.com/watch?v=4qJaSmvhxi8&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=16" title="Mini Batch Gradient Descent" target="_blank">Mini Batch Gradient Descent</a></li>
<li><a href="https://www.youtube.com/watch?v=-_4Zi8fCZO4&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=17" title="Understanding Mini-Batch Gradient Descent" target="_blank">Understanding Mini-Batch Gradient Descent</a></li>
<li><a href="https://www.youtube.com/watch?v=lAq96T8FkTw&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=18" title="Exponentially Weighted Averages" target="_blank">Exponentially Weighted Averages</a></li>
<li><a href="https://www.youtube.com/watch?v=NxTFlzBjS-4&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=19" title="Understanding Exponentially Weighted Averages" target="_blank">Understanding Exponentially Weighted Averages</a></li>
<li><a href="https://www.youtube.com/watch?v=lWzo8CajF5s&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=20" title="Bias Correction of Exponentially Weighted Averages" target="_blank">Bias Correction of Exponentially Weighted Averages</a></li>
<li><a href="https://www.youtube.com/watch?v=k8fTYJPd3_I&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=21" title="Gradient Descent With Momentum" target="_blank">Gradient Descent With Momentum</a></li>
<li><a href="https://www.youtube.com/watch?v=_e-LFe_igno&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=22" title="RMSProp" target="_blank">RMSProp</a></li>
<li><a href="https://www.youtube.com/watch?v=JXQT_vxqwIs&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=23" title="Adam Optimization Algorithm" target="_blank">Adam Optimization Algorithm</a></li>
<li><a href="https://www.youtube.com/watch?v=QzulmoOg2JE&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=24" title="Learning Rate Decay" target="_blank">Learning Rate Decay</a></li>
<li><a href="https://www.youtube.com/watch?v=tNIpEZLv_eg&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=28" title="Normalizing Activations in a Network" target="_blank">Normalizing Activations in a Network</a></li>
<li><a href="https://www.youtube.com/watch?v=em6dfRxYkYU&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=29" title="Fitting Batch Norm Into Neural Networks" target="_blank">Fitting Batch Norm Into Neural Networks</a></li>
<li><a href="https://www.youtube.com/watch?v=nUUqwaxLnWs&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=30" title="Why Does Batch Norm Work?" target="_blank">Why Does Batch Norm Work?</a></li>
<li><a href="https://www.youtube.com/watch?v=5qefnAek8OA&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=31" title="Batch Norm At Test Time" target="_blank">Batch Norm At Test Time</a></li>
<li><a href="https://www.youtube.com/watch?v=fODpu1-lNTw&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=34" title="The Problem of Local Optima" target="_blank">The Problem of Local Optima</a></li>
</ul></li>
</ul>
<p><strong>References</strong>:</p>
<ul>
<li><a href="https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html" title="numpy.random.permutation" target="_blank">numpy.random.permutation</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/nn/moments.md" title="tf.nn.moments" target="_blank">tf.nn.moments</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/MomentumOptimizer.md" title="tf.train.MomentumOptimizer" target="_blank">tf.train.MomentumOptimizer</a> </li>
<li><a href="https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/RMSPropOptimizer.md" title="tf.train.RMSPropOptimizer" target="_blank">tf.train.RMSPropOptimizer</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/RMSPropOptimizer.md" title="tf.train.AdamOptimizer" target="_blank">tf.train.AdamOptimizer</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/nn/batch_normalization" title="tf.nn.batch_normalization" target="_blank">tf.nn.batch_normalization</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/inverse_time_decay.md" title="tf.train.inverse_time_decay" target="_blank">tf.train.inverse_time_decay</a> </li>
</ul>
<h2>Learning Objectives</h2>
<p>At the end of this project, you are expected to be able to <a href="https://fs.blog/feynman-learning-technique/" title="explain to anyone" target="_blank">explain to anyone</a>, <strong>without the help of Google</strong>:</p>

# General
* **What is a hyperparameter?**
    
* How and why do you normalize your input data?
* What is a saddle point?
* What is stochastic gradient descent?
* What is mini-batch gradient descent?
* What is a moving average? How do you implement it?
* What is gradient descent with momentum? How do you implement it?
* What is RMSProp? How do you implement it?
* What is Adam optimization? How do you implement it?
* What is learning rate decay? How do you implement it?
* What is batch normalization? How do you implement it?

<h2>Requirements</h2>
<h3>General</h3>
<ul>
<li>Allowed editors: <code>vi</code>, <code>vim</code>, <code>emacs</code></li>
<li>All your files will be interpreted/compiled on Ubuntu 20.04 LTS using <code>python3</code> (version 3.8)</li>
<li>Your files will be executed with <code>numpy</code> (version 1.19.2) and tensorflow (version 2.6)</li>
<li>All your files should end with a new line</li>
<li>The first line of all your files should be exactly <code>#!/usr/bin/env python3</code></li>
<li>A <code>README.md</code> file, at the root of the folder of the project, is mandatory</li>
<li>Your code should use the <code>pycodestyle</code> style (version 2.6)</li>
<li>All your modules should have documentation (<code>python3 -c 'print(__import__("my_module").__doc__)'</code>)</li>
<li>All your classes should have documentation (<code>python3 -c 'print(__import__("my_module").MyClass.__doc__)'</code>)</li>
<li>All your functions (inside and outside a class) should have documentation (<code>python3 -c 'print(__import__("my_module").my_function.__doc__)'</code> and <code>python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'</code>)</li>
<li>Unless otherwise noted, you are not allowed to import any module except <code>import numpy as np</code> and/or <code>import tensorflow.compat.v1. as tf</code></li>
<li>You should not import any module unless it is being used</li>
<li>All your files must be executable</li>
<li>The length of your files will be tested using <code>wc</code></li>
</ul>
<h2>More Info</h2>
<h3>Eager execution</h3>
<p>In projects that have tensorflow 1, you’ll find in the in main files this line <code>tf.disable_eager_execution()</code> after importing tensorflow. <br>
<em>Take a look at the <a href="/rltoken/UYWwctzCb1VhD1sjmGwt0g" title="purpose of tf.compat.v1" target="_blank">purpose of tf.compat.v1</a></em></p>
<h3>Testing</h3>
<p>Please use the following checkpoints for to accompany the following <code>tensorflow</code> main files. You do not need to push these files to GitHub. Your code will not be tested with these files.</p>
<ul>
<li><a href="https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-ml/graph.ckpt.data-00000-of-00001" title="graph.ckpt.data-00000-of-00001" target="_blank">graph.ckpt.data-00000-of-00001</a></li>
<li><a href="https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-ml/graph.ckpt.index" title="graph.ckpt.index" target="_blank">graph.ckpt.index</a></li>
<li><a href="https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-ml/graph.ckpt.meta" title="graph.ckpt.meta" target="_blank">graph.ckpt.meta</a></li>
</ul>


# Tasks

## 0. Normalization Constants
Write the function `def normalization_constants(X):` that calculates the normalization (standardization) constants of a matrix:

    * X is the numpy.ndarray of shape (m, nx) to normalize
        * m is the number of data points
        * nx is the number of features
    * Returns: the mean and standard deviation of each feature, respectively

```
ubuntu@alexa-ml:~/0x03-optimization$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(m)
    print(s)
ubuntu@alexa-ml:~/0x03-optimization$ ./0-main.py 
[ 0.11961603  2.08201297 -3.59232261]
[2.01576449 1.034667   9.52002619]
ubuntu@alexa-ml:~/0x03-optimization
```
File : `0-norm_constants.py`

## 1. Normalize
Write the function `def normalize(X, m, s):` that normalizes (standardizes) a matrix:

* X is the numpy.ndarray of shape (d, nx) to normalize
    * `d` is the number of data points
    * `nx` is the number of features
* `m` is a `numpy.ndarray` of shape `(nx,)` that contains the mean of all features of `X`
* `s` is a `numpy.ndarray` of shape `(nx,)` that contains the standard deviation of all features of `X`
* Returns: The normalized `X` matrix

```
ubuntu@alexa-ml:~/0x03-optimization$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants
normalize = __import__('1-normalize').normalize

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(X[:10])
    X = normalize(X, m, s)
    print(X[:10])
    m, s = normalization_constants(X)
    print(m)
    print(s)
ubuntu@alexa-ml:~/0x03-optimization$ ./1-main.py 
[[  3.52810469   3.8831507   -6.69181838]
 [  0.80031442   0.65224094  -5.39379178]
 [  1.95747597   0.729515     7.99659596]
 [  4.4817864    2.96939671   3.55263731]
 [  3.73511598   0.82687659   3.40131526]
 [ -1.95455576   3.94362119 -19.16956044]
 [  1.90017684   1.58638102  -3.24326124]
 [ -0.30271442   1.25254519 -10.38030909]
 [ -0.2064377    3.92294203  -0.20075401]
 [  0.821197     3.48051479  -3.9815039 ]]
[[ 1.69091612  1.74078977 -0.32557639]
 [ 0.33768746 -1.38186686 -0.18922943]
 [ 0.91174338 -1.3071819   1.21732003]
 [ 2.16402779  0.85765153  0.75051893]
 [ 1.79361228 -1.21308245  0.73462381]
 [-1.02897526  1.79923417 -1.63625998]
 [ 0.88331787 -0.47902557  0.03666601]
 [-0.20951378 -0.80167608 -0.71302183]
 [-0.1617519   1.77924787  0.35625623]
 [ 0.34804709  1.35164437 -0.04088028]]
[ 2.44249065e-17 -4.99600361e-16  1.46549439e-16]
[1. 1. 1.]
ubuntu@alexa-ml:~/0x03-optimization$
```
File: `1-normalize.py`

<h3 class="panel-title">
      2. Shuffle Data
    </h3>
<p>Write the function <code>def shuffle_data(X, Y):</code> that shuffles the data points in two matrices the same way:</p>
<ul>
<li><code>X</code> is the first <code>numpy.ndarray</code> of shape <code>(m, nx)</code> to shuffle

<ul>
<li><code>m</code> is the number of data points</li>
<li><code>nx</code> is the number of features in <code>X</code></li>
</ul></li>
<li><code>Y</code> is the second <code>numpy.ndarray</code> of shape <code>(m, ny)</code> to shuffle

<ul>
<li><code>m</code> is the same number of data points as in <code>X</code></li>
<li><code>ny</code> is the number of features in <code>Y</code></li>
</ul></li>
<li>Returns: the shuffled <code>X</code> and <code>Y</code> matrices</li>
</ul>
<p><em>Hint: you should use <a href="https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.permutation.html" title="numpy.random.permutation" target="_blank">numpy.random.permutation</a></em></p>

<pre><code>
ubuntu@alexa-ml:~/0x03-optimization$ cat 2-main.py 
#!/usr/bin/env python3

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data

if __name__ == '__main__':
    X = np.array([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8], 
                [9, 10]])
    Y = np.array([[11, 12],
                [13, 14],
                [15, 16],
                [17, 18],
                [19, 20]])

    np.random.seed(0)
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    print(X_shuffled)
    print(Y_shuffled)
ubuntu@alexa-ml:~/0x03-optimization$ ./2-main.py 
[[ 5  6]
 [ 1  2]
 [ 3  4]
 [ 7  8]
 [ 9 10]]
[[15 16]
 [11 12]
 [13 14]
 [17 18]
 [19 20]]
ubuntu@alexa-ml:~/0x03-optimization$
</code></pre>
<li>File: <code>2-shuffle_data.py</code></li>

<h3 class="panel-title">
      3. Mini-Batch
    </h3>
<p>Write the function <code>def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):</code> that trains a loaded neural network model using mini-batch gradient descent:</p>
<ul>
<li>
<code>X_train</code> is a <code>numpy.ndarray</code> of shape <code>(m, 784)</code> containing the training data

<ul>
<li><code>m</code> is the number of data points</li>
<li><code>784</code> is the number of input features</li>
</ul></li>
<li><code>Y_train</code> is a one-hot <code>numpy.ndarray</code> of shape <code>(m, 10)</code> containing the training labels

<ul>
<li><code>10</code> is the number of classes the model should classify</li>
</ul></li>
<li><code>X_valid</code> is a <code>numpy.ndarray</code> of shape <code>(m, 784)</code> containing the validation data</li>
<li><code>Y_valid</code> is a one-hot <code>numpy.ndarray</code> of shape <code>(m, 10)</code> containing the validation labels</li>
<li><code>batch_size</code> is the number of data points in a batch</li>
<li><code>epochs</code> is the number of times the training should pass through the whole dataset</li>
<li><code>load_path</code> is the path from which to load the model</li>
<li><code>save_path</code> is the path to where the model should be saved after training</li>
<li>Returns: the path where the model was saved</li>
<li>Your training function should allow for a smaller final batch (a.k.a. use the <em>entire</em>  training set)</li>
<li>1) import meta graph and restore session</li>
<li>2) Get the following tensors and ops from the collection restored

<ul>
<li><code>x</code> is a placeholder for the input data</li>
<li><code>y</code> is a placeholder for the labels</li>
<li><code>accuracy</code> is an op to calculate the accuracy of the model</li>
<li><code>loss</code> is an op to calculate the cost of the model</li>
<li><code>train_op</code> is an op to perform one pass of gradient descent on the model</li>
</ul></li>
<li>3) loop over epochs:

<ul>
<li>shuffle data</li>
<li>loop over the batches:

<ul>
<li>get <code>X_batch</code> and <code>Y_batch</code> from data</li>
<li>train your model</li>
</ul></li>
</ul></li>
<li>4) Save session</li>
<li>You should use <code>shuffle_data = __import__('2-shuffle_data').shuffle_data</code></li>
<li>Before the first epoch and after every subsequent epoch, the following should be printed:

<ul>
<li><code>After {epoch} epochs:</code> where <code>{epoch}</code> is the current epoch</li>
<li><code>\tTraining Cost: {train_cost}</code> where <code>{train_cost}</code> is the cost of the model on the entire training set</li>
<li><code>\tTraining Accuracy: {train_accuracy}</code> where <code>{train_accuracy}</code> is the accuracy of the model on the entire training set</li>
<li><code>\tValidation Cost: {valid_cost}</code> where <code>{valid_cost}</code> is the cost of the model on the entire validation set</li>
<li><code>\tValidation Accuracy: {valid_accuracy}</code> where <code>{valid_accuracy}</code> is the accuracy of the model on the entire validation set</li>
</ul></li>
<li>After every 100 steps gradient descent within an epoch, the following should be printed:

<ul>
<li><code>\tStep {step_number}:</code> where <code>{step_number}</code> is the number of times gradient descent has been run in the current epoch</li>
<li><code>\t\tCost: {step_cost}</code> where <code>{step_cost}</code> is the cost of the model on the current mini-batch</li>
<li><code>\t\tAccuracy: {step_accuracy}</code> where <code>{step_accuracy}</code> is the accuracy of the model on the current mini-batch</li>
<li>Advice: the function <a href="https://docs.python.org/3/library/functions.html#func-range" title="range" target="_blank">range</a> can help you to handle this loop inside your dataset by using <code>batch_size</code> as step value</li>
</ul></li>
</ul>

<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
train_mini_batch = __import__('3-mini_batch').train_mini_batch

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    alpha = 0.01
    iterations = 5000

    np.random.seed(0)
    save_path = train_mini_batch(X_train, Y_train_oh, X_valid, Y_valid_oh,
                                 epochs=10, load_path='./graph.ckpt',
                                 save_path='./model.ckpt')
    print('Model saved in path: {}'.format(save_path))
ubuntu@alexa-ml:~/0x03-optimization$ ./3-main.py 
2018-11-10 02:10:48.277854: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
After 0 epochs:
    Training Cost: 2.8232288360595703
    Training Accuracy: 0.08726000040769577
    Validation Cost: 2.810532331466675
    Validation Accuracy: 0.08640000224113464
    Step 100:
        Cost: 0.9012309908866882
        Accuracy: 0.6875
    Step 200:
        Cost: 0.6328266263008118
        Accuracy: 0.8125

    ...

    Step 1500:
        Cost: 0.27602481842041016
        Accuracy: 0.9375
After 1 epochs:
    Training Cost: 0.3164157569408417
    Training Accuracy: 0.9101600050926208
    Validation Cost: 0.291348934173584
    Validation Accuracy: 0.9168999791145325

...

After 9 epochs:
    Training Cost: 0.12963168323040009
    Training Accuracy: 0.9642800092697144
    Validation Cost: 0.13914340734481812
    Validation Accuracy: 0.961899995803833
    Step 100:
        Cost: 0.10656605660915375
        Accuracy: 1.0
    Step 200:
        Cost: 0.09849657118320465
        Accuracy: 1.0

    ...

    Step 1500:
        Cost: 0.0914708822965622
        Accuracy: 0.96875
After 10 epochs:
    Training Cost: 0.12012937664985657
    Training Accuracy: 0.9669600129127502
    Validation Cost: 0.13320672512054443
    Validation Accuracy: 0.9635999798774719
Model saved in path: ./model.ckpt
ubuntu@alexa-ml:~/0x03-optimization$
</code></pre>
<li>File: <code>3-mini_batch.py</code></li>

<h3 class="panel-title">
      4. Moving Average
    </h3>
<p>Write the function <code>def moving_average(data, beta):</code> that calculates the weighted moving average of a data set:</p>
<ul>
<li><code>data</code> is the list of data to calculate the moving average of</li>
<li><code>beta</code> is the weight used for the moving average</li>
<li>Your moving average calculation should use bias correction</li>
<li>Returns: a list containing the moving averages of <code>data</code></li>
</ul>

<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 4-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
moving_average = __import__('4-moving_average').moving_average

if __name__ == '__main__':
        data = [72, 78, 71, 68, 66, 69, 79, 79, 65, 64, 66, 78, 64, 64, 81, 71, 69,
                65, 72, 64, 60, 61, 62, 66, 72, 72, 67, 67, 67, 68, 75]
        days = list(range(1, len(data) + 1))
        m_avg = moving_average(data, 0.9)
        print(m_avg)
        plt.plot(days, data, 'r', days, m_avg, 'b')
        plt.xlabel('Day of Month')
        plt.ylabel('Temperature (Fahrenheit)')
        plt.title('SF Maximum Temperatures in October 2018')
        plt.legend(['actual', 'moving_average'])
        plt.show()
ubuntu@alexa-ml:~/0x03-optimization$ ./4-main.py 
[72.0, 75.15789473684211, 73.62361623616238, 71.98836871183484, 70.52604332006544, 70.20035470453027, 71.88706986789997, 73.13597603396988, 71.80782582850702, 70.60905915023126, 69.93737009120935, 71.0609712312634, 70.11422355031073, 69.32143707981284, 70.79208718739721, 70.81760741911772, 70.59946700377961, 69.9406328280786, 70.17873340222755, 69.47534437750306, 68.41139351151023, 67.58929643210207, 66.97601174673004, 66.86995043877324, 67.42263231561797, 67.91198666959514, 67.8151574064495, 67.72913996327617, 67.65262186609462, 67.68889744321645, 68.44900744806469]
</code></pre>

<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/task4Optimizacion.png" alt="" loading="lazy" style=""></p>

<li>File: <code>4-moving_average.py</code></li>

<h3 class="panel-title">
      5. Momentum
    </h3>
<p>Write the function <code>def update_variables_momentum(alpha, beta1, var, grad, v):</code> that updates a variable using the gradient descent with momentum optimization algorithm:</p>
<ul>
<li><code>alpha</code> is the learning rate</li>
<li><code>beta1</code> is the momentum weight</li>
<li><code>var</code> is a <code>numpy.ndarray</code> containing the variable to be updated</li>
<li><code>grad</code> is a <code>numpy.ndarray</code> containing the gradient of <code>var</code></li>
<li><code>v</code> is the previous first moment of <code>var</code></li>
<li>Returns: the updated variable and the new moment, respectively</li>
</ul>

<pre>
<code>ubuntu@alexa-ml:~/0x03-optimization$ cat 5-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
update_variables_momentum = __import__('5-momentum').update_variables_momentum

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev = np.zeros((nx, 1))
    db_prev = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev = update_variables_momentum(0.01, 0.9, W, dW, dW_prev)
        b, db_prev = update_variables_momentum(0.01, 0.9, b, db, db_prev)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.where(A >= 0.5, 1, 0)
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i, 0]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@alexa-ml:~/0x03-optimization$ ./5-main.py 
Cost after 0 iterations: 4.365105010037203
Cost after 100 iterations: 0.5729736703124042
Cost after 200 iterations: 0.2449357405113111
Cost after 300 iterations: 0.17711325087582164
Cost after 400 iterations: 0.14286111618067307
Cost after 500 iterations: 0.12051674907075896
Cost after 600 iterations: 0.10450664363662196
Cost after 700 iterations: 0.09245615061035156
Cost after 800 iterations: 0.08308760082979068
Cost after 900 iterations: 0.07562924162824029
Cost after 1000 iterations: 0.0695782354732263
</code></pre>

<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/task5Optimizacion.png" alt="" loading="lazy" style=""></p>
<li>File: <code>5-momentum.py</code></li>

<h3 class="panel-title">
      6. Momentum Upgraded
    </h3>
<p>Write the function <code>def create_momentum_op(loss, alpha, beta1):</code> that creates the training operation for a neural network in <code>tensorflow</code> using the gradient descent with momentum optimization algorithm:</p>
<ul>
<li><code>loss</code> is the loss of the network</li>
<li><code>alpha</code> is the learning rate</li>
<li><code>beta1</code> is the momentum weight</li>
<li>Returns: the momentum optimization operation</li>
</ul>

<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 6-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
create_momentum_op = __import__('6-momentum').create_momentum_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        train_op = create_momentum_op(loss, 0.01, 0.9)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            if not (i % 100):
                cost = sess.run(loss, feed_dict={x:X, y:Y_oh})
                print('Cost after {} iterations: {}'.format(i, cost))
            sess.run(train_op, feed_dict={x:X, y:Y_oh})
        cost, Y_pred_oh = sess.run((loss, y_pred), feed_dict={x:X, y:Y_oh})
        print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@alexa-ml:~/0x03-optimization$ ./6-main.py 
2018-11-10 00:15:42.968586: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Cost after 0 iterations: 2.8232274055480957
Cost after 100 iterations: 0.356641948223114
Cost after 200 iterations: 0.29699304699897766
Cost after 300 iterations: 0.26470813155174255
Cost after 400 iterations: 0.24141179025173187
Cost after 500 iterations: 0.22264979779720306
Cost after 600 iterations: 0.20677044987678528
Cost after 700 iterations: 0.19298051297664642
Cost after 800 iterations: 0.18082040548324585
Cost after 900 iterations: 0.16998952627182007
Cost after 1000 iterations: 0.1602744460105896
</code></pre>
<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/task6Optimizacion.png" alt="" loading="lazy" style=""></p>
<li>File: <code>6-momentum.py</code></li>

<h3 class="panel-title">
      7. RMSProp
    </h3>
<p>Write the function <code>def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):</code> that updates a variable using the RMSProp optimization algorithm:</p>
<ul>
<li><code>alpha</code> is the learning rate</li>
<li><code>beta2</code> is the RMSProp weight</li>
<li><code>epsilon</code> is a small number to avoid division by zero</li>
<li><code>var</code> is a <code>numpy.ndarray</code> containing the variable to be updated</li>
<li><code>grad</code> is a <code>numpy.ndarray</code> containing the gradient of <code>var</code></li>
<li><code>s</code> is the previous second moment of <code>var</code></li>
<li>Returns: the updated variable and the new moment, respectively</li>
</ul>
<ul>
<li><code>alpha</code> is the learning rate</li>
<li><code>beta2</code> is the RMSProp weight</li>
<li><code>epsilon</code> is a small number to avoid division by zero</li>
<li><code>var</code> is a <code>numpy.ndarray</code> containing the variable to be updated</li>
<li><code>grad</code> is a <code>numpy.ndarray</code> containing the gradient of <code>var</code></li>
<li><code>s</code> is the previous second moment of <code>var</code></li>
<li>Returns: the updated variable and the new moment, respectively</li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 7-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
update_variables_RMSProp = __import__('7-RMSProp').update_variables_RMSProp

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev = np.zeros((nx, 1))
    db_prev = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev = update_variables_RMSProp(0.001, 0.9, 1e-8, W, dW, dW_prev)
        b, db_prev = update_variables_RMSProp(0.001, 0.9, 1e-8, b, db, db_prev)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.where(A &gt;= 0.5, 1, 0)
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i, 0]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@alexa-ml:~/0x03-optimization$ ./7-main.py 
Cost after 0 iterations: 4.365105010037203
Cost after 100 iterations: 1.3708321848806053
Cost after 200 iterations: 0.22693392990308764
Cost after 300 iterations: 0.05133394800221906
Cost after 400 iterations: 0.01836557116372359
Cost after 500 iterations: 0.008176390663315372
Cost after 600 iterations: 0.004091348850058557
Cost after 700 iterations: 0.002195647208708407
Cost after 800 iterations: 0.001148167933229118
Cost after 900 iterations: 0.0005599361043400206
Cost after 1000 iterations: 0.0002655839831275339
</code></pre>
<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/task7Optimizacion.png" alt="" loading="lazy" style=""></p>
<li>File: <code>7-RMSProp.py</code></li>
<h3 class="panel-title">
      8. RMSProp Upgraded
    </h3>
<p>Write the function <code>def create_RMSProp_op(loss, alpha, beta2, epsilon):</code> that creates the training operation for a neural network in <code>tensorflow</code> using the RMSProp optimization algorithm:</p>
<ul>
<li><code>loss</code> is the loss of the network</li>
<li><code>alpha</code> is the learning rate</li>
<li><code>beta2</code> is the RMSProp weight</li>
<li><code>epsilon</code> is a small number to avoid division by zero</li>
<li>Returns: the RMSProp optimization operation</li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 8-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
create_RMSProp_op = __import__('8-RMSProp').create_RMSProp_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        train_op = create_RMSProp_op(loss, 0.001, 0.9, 1e-8)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            if not (i % 100):
                cost = sess.run(loss, feed_dict={x:X, y:Y_oh})
                print('Cost after {} iterations: {}'.format(i, cost))
            sess.run(train_op, feed_dict={x:X, y:Y_oh})
        cost, Y_pred_oh = sess.run((loss, y_pred), feed_dict={x:X, y:Y_oh})
        print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@alexa-ml:~/0x03-optimization$ ./8-main.py 
2018-11-10 00:28:48.894342: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Cost after 0 iterations: 2.8232274055480957
Cost after 100 iterations: 0.48531609773635864
Cost after 200 iterations: 0.21557031571865082
Cost after 300 iterations: 0.13388566672801971
Cost after 400 iterations: 0.07422538101673126
Cost after 500 iterations: 0.05024252086877823
Cost after 600 iterations: 0.02709660679101944
Cost after 700 iterations: 0.015626247972249985
Cost after 800 iterations: 0.008653616532683372
Cost after 900 iterations: 0.005407326854765415
Cost after 1000 iterations: 0.003452717326581478
</code></pre>
<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/task8Optimizacion.png" alt="" loading="lazy" style=""></p>
<li>File: <code>8-RMSProp.py</code></li>
<h3 class="panel-title">
      9. Adam
    </h3>
<p>Write the function <code>def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):</code> that updates a variable in place using the Adam optimization algorithm:</p>
<ul>
<li><code>alpha</code> is the learning rate</li>
<li><code>beta1</code> is the weight used for the first moment</li>
<li><code>beta2</code> is the weight used for the second moment</li>
<li><code>epsilon</code> is a small number to avoid division by zero</li>
<li><code>var</code> is a <code>numpy.ndarray</code> containing the variable to be updated</li>
<li><code>grad</code> is a <code>numpy.ndarray</code> containing the gradient of <code>var</code></li>
<li><code>v</code> is the previous first moment of <code>var</code></li>
<li><code>s</code> is the previous second moment of <code>var</code></li>
<li><code>t</code> is the time step used for bias correction</li>
<li>Returns: the updated variable, the new first moment, and the new second moment, respectively</li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 9-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
update_variables_Adam = __import__('9-Adam').update_variables_Adam

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev1 = np.zeros((nx, 1))
    db_prev1 = 0
    dW_prev2 = np.zeros((nx, 1))
    db_prev2 = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev1, dW_prev2 = update_variables_Adam(0.001, 0.9, 0.99, 1e-8, W, dW, dW_prev1, dW_prev2, i + 1)
        b, db_prev1, db_prev2 = update_variables_Adam(0.001, 0.9, 0.99, 1e-8, b, db, db_prev1, db_prev2, i + 1)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.where(A &gt;= 0.5, 1, 0)
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i, 0]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@alexa-ml:~/0x03-optimization$ ./9-main.py
Cost after 0 iterations: 4.365105010037203
Cost after 100 iterations: 1.5950468370180395
Cost after 200 iterations: 0.390276184856453
Cost after 300 iterations: 0.13737908627614337
Cost after 400 iterations: 0.06963385247882238
Cost after 500 iterations: 0.043186805401891
Cost after 600 iterations: 0.029615890163981955
Cost after 700 iterations: 0.02135952185721115
Cost after 800 iterations: 0.01576513402620876
Cost after 900 iterations: 0.011813533123333355
Cost after 1000 iterations: 0.008996494409788116
</code></pre>
<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/task9Optimizacion.png" alt="" loading="lazy" style=""></p>
<li>File: <code>9-Adam.py</code></li>
<h3 class="panel-title">
      10. Adam Upgraded
    </h3>
<p>Write the function <code>def create_Adam_op(loss, alpha, beta1, beta2, epsilon):</code> that creates the training operation for a neural network in <code>tensorflow</code> using the Adam optimization algorithm:</p>
<ul>
<li><code>loss</code> is the loss of the network</li>
<li><code>alpha</code> is the learning rate</li>
<li><code>beta1</code> is the weight used for the first moment</li>
<li><code>beta2</code> is the weight used for the second moment</li>
<li><code>epsilon</code> is a small number to avoid division by zero</li>
<li>Returns: the Adam optimization operation</li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 10-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
create_Adam_op = __import__('10-Adam').create_Adam_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        train_op = create_Adam_op(loss, 0.001, 0.9, 0.99, 1e-8)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            if not (i % 100):
                cost = sess.run(loss, feed_dict={x:X, y:Y_oh})
                print('Cost after {} iterations: {}'.format(i, cost))
            sess.run(train_op, feed_dict={x:X, y:Y_oh})
        cost, Y_pred_oh = sess.run((loss, y_pred), feed_dict={x:X, y:Y_oh})
        print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@alexa-ml:~/0x03-optimization$ ./10-main.py 
2018-11-09 23:37:09.188702: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Cost after 0 iterations: 2.8232274055480957
Cost after 100 iterations: 0.17724855244159698
Cost after 200 iterations: 0.0870152935385704
Cost after 300 iterations: 0.03907731547951698
Cost after 400 iterations: 0.014239841140806675
Cost after 500 iterations: 0.0048021236434578896
Cost after 600 iterations: 0.0018489329377189279
Cost after 700 iterations: 0.000814757077023387
Cost after 800 iterations: 0.00038969298475421965
Cost after 900 iterations: 0.00019614089978858829
Cost after 1000 iterations: 0.00010206626757280901
</code></pre>
<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/task10Optimizacion.png" alt="" loading="lazy" style=""></p>
<li>File: <code>10-Adam.py</code></li>
<h3 class="panel-title">
      11. Learning Rate Decay
    </h3>
<p>Write the function <code>def learning_rate_decay(alpha, decay_rate, global_step, decay_step):</code> that updates the learning rate using inverse time decay in <code>numpy</code>:</p>
<ul>
<li><code>alpha</code> is the original learning rate</li>
<li><code>decay_rate</code> is the weight used to determine the rate at which <code>alpha</code> will decay</li>
<li><code>global_step</code> is the number of passes of gradient descent that have elapsed</li>
<li><code>decay_step</code> is the number of passes of gradient descent that should occur before alpha is decayed further</li>
<li>the learning rate decay should occur in a stepwise fashion</li>
<li>Returns: the updated value for <code>alpha</code></li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 11-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
learning_rate_decay = __import__('11-learning_rate_decay').learning_rate_decay

if __name__ == '__main__':
    alpha_init = 0.1
    for i in range(100):
        alpha = learning_rate_decay(alpha_init, 1, i, 10)
        print(alpha)
ubuntu@alexa-ml:~/0x03-optimization$ ./11-main.py
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.03333333333333333
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.016666666666666666
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.014285714285714287
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.011111111111111112
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
ubuntu@alexa-ml:~/0x03-optimization$
</code></pre>
<li>File: <code>11-learning_rate_decay.py</code></li>
<h3 class="panel-title">
      12. Learning Rate Decay Upgraded
    </h3>
<p>Write the function <code>def learning_rate_decay(alpha, decay_rate, global_step, decay_step):</code> that creates a learning rate decay operation in <code>tensorflow</code> using inverse time decay:</p>
<ul>
<li><code>alpha</code> is the original learning rate</li>
<li><code>decay_rate</code> is the weight used to determine the rate at which <code>alpha</code> will decay</li>
<li><code>global_step</code> is the number of passes of gradient descent that have elapsed</li>
<li><code>decay_step</code> is the number of passes of gradient descent that should occur before alpha is decayed further</li>
<li>the learning rate decay should occur in a stepwise fashion</li>
<li>Returns: the learning rate decay operation</li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 12-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        global_step = tf.Variable(0, trainable=False)
        alpha = 0.1
        alpha = learning_rate_decay(alpha, 1, global_step, 10)
        train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss, global_step=global_step)
        init = tf.global_variables_initializer()
        sess.run(init)       
        for i in range(100):
            a = sess.run(alpha)
            print(a)
            sess.run(train_op, feed_dict={x:X, y:Y_oh})
ubuntu@alexa-ml:~/0x03-optimization$ ./12-main.py
2018-11-10 00:54:20.318892: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.05
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.033333335
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.025
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.02
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.016666668
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.014285714
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.0125
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.011111111
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
ubuntu@alexa-ml:~/0x03-optimization$
</code></pre>
<li>File: <code>12-learning_rate_decay.py</code></li>
<h3 class="panel-title">
      13. Batch Normalization
    </h3>
<p>Write the function <code>def batch_norm(Z, gamma, beta, epsilon):</code> that normalizes an unactivated output of a neural network using batch normalization:</p>
<ul>
<li><code>Z</code> is a <code>numpy.ndarray</code> of shape <code>(m, n)</code> that should be normalized

<ul>
<li><code>m</code> is the number of data points</li>
<li><code>n</code> is the number of features in <code>Z</code></li>
</ul></li>
<li><code>gamma</code> is a <code>numpy.ndarray</code> of shape <code>(1, n)</code> containing the scales used for batch normalization</li>
<li><code>beta</code> is a <code>numpy.ndarray</code> of shape <code>(1, n)</code> containing the offsets used for batch normalization</li>
<li><code>epsilon</code> is a small number used to avoid division by zero</li>
<li>Returns: the normalized <code>Z</code> matrix</li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 13-main.py 
#!/usr/bin/env python3

import numpy as np
batch_norm = __import__('13-batch_norm').batch_norm

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    Z = np.concatenate((a, b, c), axis=1)
    gamma = np.random.rand(1, 3)
    beta = np.random.rand(1, 3)
    print(Z[:10])
    Z_norm = batch_norm(Z, gamma, beta, 1e-8)
    print(Z_norm[:10])
ubuntu@alexa-ml:~/0x03-optimization$ ./13-main.py 
[[  3.52810469   3.8831507   -6.69181838]
 [  0.80031442   0.65224094  -5.39379178]
 [  1.95747597   0.729515     7.99659596]
 [  4.4817864    2.96939671   3.55263731]
 [  3.73511598   0.82687659   3.40131526]
 [ -1.95455576   3.94362119 -19.16956044]
 [  1.90017684   1.58638102  -3.24326124]
 [ -0.30271442   1.25254519 -10.38030909]
 [ -0.2064377    3.92294203  -0.20075401]
 [  0.821197     3.48051479  -3.9815039 ]]
[[ 1.48744676  0.95227435  0.82862045]
 [ 0.63640337 -0.29189903  0.83717117]
 [ 0.99742624 -0.26214198  0.92538004]
 [ 1.78498595  0.60040182  0.89610557]
 [ 1.55203222 -0.22464954  0.89510874]
 [-0.22308868  0.9755606   0.74642361]
 [ 0.97954948  0.06782387  0.85133774]
 [ 0.29226936 -0.06073115  0.8043226 ]
 [ 0.32230674  0.96759737  0.87138019]
 [ 0.64291853  0.79722549  0.84647459]]
ubuntu@alexa-ml:~/0x03-optimization$
</code></pre>
<li>File: <code>13-batch_norm.py</code></li>
<h3 class="panel-title">
      14. Batch Normalization Upgraded
    </h3>
<p>Write the function <code>def create_batch_norm_layer(prev, n, activation):</code> that creates a batch normalization layer for a neural network in <code>tensorflow</code>:</p>
<ul>
<li><code>prev</code> is the activated output of the previous layer</li>
<li><code>n</code> is the number of nodes in the layer to be created</li>
<li><code>activation</code> is the activation function that should be used on the output of the layer</li>
<li>you should use the <code>tf.keras.layers.Dense</code> layer as the base layer with kernal initializer <code>tf.keras.initializers.VarianceScaling(mode='fan_avg')</code></li>
<li>your layer should incorporate two trainable parameters, <code>gamma</code> and <code>beta</code>, initialized as vectors of <code>1</code> and <code>0</code> respectively</li>
<li>you should use an <code>epsilon</code> of <code>1e-8</code></li>
<li>Returns: a tensor of the activated output for the layer</li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 14-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    X = X_3D.reshape((X_3D.shape[0], -1))

    tf.set_random_seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    a = create_batch_norm_layer(x, 256, tf.nn.tanh)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(a, feed_dict={x:X[:5]}))
ubuntu@alexa-ml:~/0x03-optimization$ ./14-main.py 
[[-0.6847082  -0.8220385  -0.35229233 ...  0.464784   -0.8326035
  -0.96122414]
 [-0.77318543 -0.66306996  0.7523017  ...  0.811305    0.79587764
   0.47134086]
 [-0.21438502 -0.11646973 -0.59783506 ... -0.95093447 -0.67656237
   0.26563355]
 [ 0.3159215   0.93362606  0.8738444  ...  0.26363495 -0.320637
   0.683548  ]
 [ 0.9421419   0.37344548 -0.8536682  ... -0.06270568  0.85227346
   0.3293217 ]]
ubuntu@alexa-ml:~/0x03-optimization$
</code></pre>
<li>File: <code>14-batch_norm.py</code></li>
<h3 class="panel-title">
      15. Put it all together and what do you get?
    </h3>
<p>Complete the script <code>15-model.py</code> to write the function <code>def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):</code> that builds, trains, and saves a neural network model in <code>tensorflow</code> using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization:</p>
<ul>
<li><code>Data_train</code> is a tuple containing the training inputs and training labels, respectively</li>
<li><code>Data_valid</code> is a tuple containing the validation inputs and validation labels, respectively</li>
<li><code>layers</code> is a list containing the number of nodes in each layer of the network</li>
<li><code>activation</code> is a list containing the activation functions used for each layer of the network</li>
<li><code>alpha</code> is the learning rate</li>
<li><code>beta1</code> is the weight for the first moment of Adam Optimization</li>
<li><code>beta2</code> is the weight for the second moment of Adam Optimization</li>
<li><code>epsilon</code> is a small number used to avoid division by zero</li>
<li><code>decay_rate</code> is the decay rate for inverse time decay of the learning rate <em>(the corresponding decay step should be <code>1</code>)</em></li>
<li><code>batch_size</code> is the number of data points that should be in a mini-batch</li>
<li><code>epochs</code> is the number of times the training should pass through the whole dataset</li>
<li><code>save_path</code> is the path where the model should be saved to</li>
<li>Returns: the path where the model was saved</li>
<li>Your training function should allow for a smaller final batch (a.k.a. use the <em>entire</em>  training set)</li>
<li>the learning rate should remain the same within the an epoch (a.k.a. all mini-batches within an epoch should use the same learning rate)</li>
<li>Before each epoch, you should shuffle your training data</li>
<li>Before the first epoch and after every subsequent epoch, the following should be printed:

<ul>
<li><code>After {epoch} epochs:</code> where <code>{epoch}</code> is the current epoch</li>
<li><code>\tTraining Cost: {train_cost}</code> where <code>{train_cost}</code> is the cost of the model on the entire training set</li>
<li><code>\tTraining Accuracy: {train_accuracy}</code> where <code>{train_accuracy}</code> is the accuracy of the model on the entire training set</li>
<li><code>\tValidation Cost: {valid_cost}</code> where <code>{valid_cost}</code> is the cost of the model on the entire validation set</li>
<li><code>\tValidation Accuracy: {valid_accuracy}</code> where <code>{valid_accuracy}</code> is the accuracy of the model on the entire validation set</li>
</ul></li>
<li>After every 100 steps of gradient descent within an epoch, the following should be printed:

<ul>
<li><code>\tStep {step_number}:</code> where <code>{step_number}</code> is the number of times gradient descent has been run in the current epoch</li>
<li><code>\t\tCost: {step_cost}</code> where <code>{step_cost}</code> is the cost of the model on the current mini-batch</li>
<li><code>\t\tAccuracy: {step_accuracy}</code> where <code>{step_accuracy}</code> is the accuracy of the model on the current mini-batch</li>
</ul></li>
</ul>
<p><em>Note: the input data does not need to be normalized as it has already been scaled to a range of [0, 1]</em></p>
<pre><code>ubuntu@alexa-ml:~/0x03-optimization$ cat 15-model.py
def forward_prop(prev, layers, activations, epsilon):
    #all layers get batch_normalization but the last one, that stays without any activation or normalization


def shuffle_data(X, Y):
    # fill the function


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid

    # initialize x, y and add them to collection 

    # initialize y_pred and add it to collection

    # intialize loss and add it to collection

    # intialize accuracy and add it to collection

    # intialize global_step variable
    # hint: not trainable

    # compute decay_steps

    # create "alpha" the learning rate decay operation in tensorflow

    # initizalize train_op and add it to collection 
    # hint: don't forget to add global_step parameter in optimizer().minimize()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            # print training and validation cost and accuracy

            # shuffle data

            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled

                # run training operation

                                # print batch cost and accuracy

        # print training and validation cost and accuracy again

        # save and return the path to where the model was saved

ubuntu@alexa-ml:~/0x03-optimization$ cat 15-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
model = __import__('15-model').model

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]

    np.random.seed(0)
    tf.set_random_seed(0)
    save_path = model((X_train, Y_train_oh), (X_valid, Y_valid_oh), layer_sizes,
                                 activations, save_path='./model.ckpt')
    print('Model saved in path: {}'.format(save_path))
ubuntu@alexa-ml:~/0x03-optimization$ ./15-main.py 
After 0 epochs:
    Training Cost: 2.5810317993164062
    Training Accuracy: 0.16808000206947327
    Validation Cost: 2.5596187114715576
    Validation Accuracy: 0.16859999299049377
    Step 100:
        Cost: 0.297500342130661
        Accuracy 0.90625
    Step 200:
        Cost: 0.27544915676116943
        Accuracy 0.875

    ...

    Step 1500:
        Cost: 0.09414251148700714
        Accuracy 1.0
After 1 epochs:
    Training Cost: 0.13064345717430115
    Training Accuracy: 0.9625800251960754
    Validation Cost: 0.14304184913635254
    Validation Accuracy: 0.9595000147819519

...

After 4 epochs:
    Training Cost: 0.03584253042936325
    Training Accuracy: 0.9912999868392944
    Validation Cost: 0.0853486955165863
    Validation Accuracy: 0.9750999808311462
    Step 100:
        Cost: 0.03150765225291252
        Accuracy 1.0
    Step 200:
        Cost: 0.020879564806818962
        Accuracy 1.0

    ...

    Step 1500:
        Cost: 0.015160675160586834
        Accuracy 1.0
After 5 epochs:
    Training Cost: 0.025094907730817795
    Training Accuracy: 0.9940199851989746
    Validation Cost: 0.08191727101802826
    Validation Accuracy: 0.9750999808311462
Model saved in path: ./model.ckpt
ubuntu@alexa-ml:~/0x03-optimization$
</code></pre>
<p><em>Look at that! 99.4% accuracy on training set and 97.5% accuracy on the validation set!</em></p>
<li>File: <code>15-model.py</code></li>
<h3 class="panel-title">
      16. If you can't explain it simply, you don't understand it well enough
    </h3>
<p>Write a blog post explaining the mechanics, pros, and cons of the following optimization techniques:</p>
<ul>
<li>Feature Scaling</li>
<li>Batch normalization</li>
<li>Mini-batch gradient descent</li>
<li>Gradient descent with momentum</li>
<li>RMSProp optimization</li>
<li>Adam optimization</li>
<li>Learning rate decay</li>
</ul>
<p>Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.</p>
<p>When done, please add all URLs below (blog post, tweet, etc.)</p>
<p>Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.</p>
