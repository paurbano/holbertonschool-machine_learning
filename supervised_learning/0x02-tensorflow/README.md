# 0x02. Tensorflow
<h2>Resources</h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/low_level_intro.md" title="Low Level Intro" target="_blank">Low Level Intro</a>  <strong>(Excluding <code>Datasets</code> and <code>Feature columns</code>)</strong></li>
<li><a href="https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/graphs.md" title="Graphs" target="_blank">Graphs</a> </li>
<li><a href="https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/tensors.md" title="Tensors" target="_blank">Tensors</a> </li>
<li><a href="https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/variables.md" title="Variables" target="_blank">Variables</a> </li>
<li><a href="https://www.databricks.com/tensorflow/placeholders" title="Placeholders" target="_blank">Placeholders</a> </li>
<li><a href="https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md" title="Save and Restore" target="_blank">Save and Restore</a> <strong>(Up to <code>Save and restore models</code>, excluded)</strong></li>
<li><a href="https://stackoverflow.com/questions/41265035/tensorflow-why-there-are-3-files-after-saving-the-model" title="TensorFlow, why there are 3 files after saving the model?" target="_blank">TensorFlow, why there are 3 files after saving the model?</a> </li>
<li><a href="https://docs.w3cub.com/tensorflow~python/meta_graph" title="Exporting and Importing a MetaGraph" target="_blank">Exporting and Importing a MetaGraph</a> </li>
<li><a href="https://stackoverflow.com/questions/42072234/tensorflow-import-meta-graph-and-use-variables-from-it" title="TensorFlow - import meta graph and use variables from it" target="_blank">TensorFlow - import meta graph and use variables from it</a> </li>
</ul>
<p><strong>References</strong>: </p>
<ul>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/Graph" title="tf.Graph" target="_blank">tf.Graph</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/Session" title="tf.Session" target="_blank">tf.Session</a> 

<ul>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/Session#run" title="tf.Session.run" target="_blank">tf.Session.run</a> </li>
</ul></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/Tensor" title="tf.Tensor" target="_blank">tf.Tensor</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/Variable" title="tf.Variable" target="_blank">tf.Variable</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/constant" title="tf.constant" target="_blank">tf.constant</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/placeholder" title="tf.placeholder" target="_blank">tf.placeholder</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/Operation" title="tf.Operation" target="_blank">tf.Operation</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/keras" title="tf.keras.layers" target="_blank">tf.keras.layers</a> 

<ul>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/layers/Dense" title="tf.keras.layers.Dense" target="_blank">tf.keras.layers.Dense</a> </li>
</ul></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/keras/initializers/VarianceScaling" title="tf.keras.initializers.VarianceScaling" target="_blank">tf.keras.initializers.VarianceScaling</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/nn" title="tf.nn" target="_blank">tf.nn</a> 

<ul>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/nn/relu" title="tf.nn.relu" target="_blank">tf.nn.relu</a></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/math/sigmoid" title="tf.nn.sigmoid" target="_blank">tf.nn.sigmoid</a></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/math/tanh" title="tf.nn.tanh" target="_blank">tf.nn.tanh</a></li>
</ul></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/losses" title="tf.losses" target="_blank">tf.losses</a> 

<ul>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/losses/softmax_cross_entropy" title="tf.losses.softmax_cross_entropy" target="_blank">tf.losses.softmax_cross_entropy</a> </li>
</ul></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train" title="tf.train" target="_blank">tf.train</a>

<ul>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/import_meta_graph" title="tf.train.import_meta_graph" target="_blank">tf.train.import_meta_graph</a></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/GradientDescentOptimizer" title="tf.train.GradientDescentOptimizer" target="_blank">tf.train.GradientDescentOptimizer</a> 

<ul>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/GradientDescentOptimizer#minimize" title="tf.train.GradientDescentOptimizer.minimize" target="_blank">tf.train.GradientDescentOptimizer.minimize</a> </li>
</ul></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/Saver" title="tf.train.Saver" target="_blank">tf.train.Saver</a> 

<ul>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/Saver#save" title="tf.train.Saver.save" target="_blank">tf.train.Saver.save</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/Saver#restore" title="tf.train.Saver.restore" target="_blank">tf.train.Saver.restore</a></li>
</ul></li>
</ul></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/add_to_collection" title="tf.add_to_collection" target="_blank">tf.add_to_collection</a></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/get_collection" title="tf.get_collection" target="_blank">tf.get_collection</a></li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/global_variables_initializer" title="tf.global_variables_initializer" target="_blank">tf.global_variables_initializer</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/argmax" title="tf.argmax" target="_blank">tf.argmax</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/math/equal" title="tf.math.equal" target="_blank">tf.math.equal</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/set_random_seed" title="tf.set_random_seed" target="_blank">tf.set_random_seed</a> </li>
<li><a href="https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/keras/backend/name_scope" title="tf.keras.backend.name_scope" target="_blank">tf.keras.backend.name_scope</a> </li>
</ul>
<h2>Learning Objectives</h2>
<p>At the end of this project, you are expected to be able to <a href="https://fs.blog/feynman-learning-technique/" title="explain to anyone" target="_blank">explain to anyone</a>, <strong>without the help of Google</strong>:</p>

# General
* **What is tensorflow?**
    Is a set of tools (API´s    ) that let you build models of ML, training and optimize them in a easy way.
* **What is a session? graph?**
    * **Graph:** A computational graph is a series of TensorFlow operations arranged into a graph. The graph is composed of two types of objects.
        * `tf.Operation` (or "ops"): The nodes of the graph. Operations describe calculations that consume and produce tensors.
        * `tf.Tensor`: The edges in the graph. These represent the values that will flow through the graph. Most TensorFlow functions return `tf.Tensors`.

    Important: `tf.Tensors` do not have values, they are just handles to elements in the computation graph.
    * **session** Is the way that TensorFlow evaluates the tensors. througth tf.Session object, When you request the output of a node with Session.run TensorFlow backtracks through the graph and runs all the nodes that provide input to the requested output node. 

* **What are tensors?**
    The central unit of data in TensorFlow. Tensors are arrays of any number of dimensions with a datatype base (dtype) and classified by its rank. A tensor's rank is its number of dimensions, while its shape is a tuple of integers specifying the array's length along each dimension.
    ```
    3. # a rank 0 tensor; a scalar with shape [],
    [1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
    [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
    [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
    ```

    A `tf.Tensor` has the following properties:

    * a data type (float32, int32, or string, for example)
    * a shape
    
    These are some types of tensors:

    * tf.Variable
    * tf.constant
    * tf.placeholder
    * tf.SparseTensor
    With the exception of tf.Variable, the value of a tensor is immutable

* **What are variables? constants? placeholders? How do you use them?**

* What are operations? How do you use them?
* What are namespaces? How do you use them?
* How to train a neural network in tensorflow
* What is a checkpoint?
* How to save/load a model with tensorflow
* What is the graph collection?
* How to add and get variables from the collection

<h2>Requirements</h2>
<h3>General</h3>
<ul>
<li>Allowed editors: <code>vi</code>, <code>vim</code>, <code>emacs</code></li>
<li>All your files will be interpreted/compiled on Ubuntu 20.04 LTS using <code>python3</code> (version 3.8)</li>
<li>Your files will be executed with <code>numpy</code> (version 1.19.2) and <code>tensorflow</code> (version 2.6)</li>
<li>All your files should end with a new line</li>
<li>The first line of all your files should be exactly <code>#!/usr/bin/env python3</code></li>
<li>A <code>README.md</code> file, at the root of the folder of the project, is mandatory</li>
<li>Your code should use the <code>pycodestyle</code> style (version 2.6)</li>
<li>All your modules should have documentation (<code>python3 -c 'print(__import__("my_module").__doc__)'</code>)</li>
<li>All your classes should have documentation (<code>python3 -c 'print(__import__("my_module").MyClass.__doc__)'</code>)</li>
<li>All your functions (inside and outside a class) should have documentation (<code>python3 -c 'print(__import__("my_module").my_function.__doc__)'</code> and <code>python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'</code>)</li>
<li>Unless otherwise noted, you are not allowed to import any module except <code>import tensorflow.compat.v1 as tf</code></li>
<li>All your files must be executable</li>
<li>The length of your files will be tested using <code>wc</code></li>
</ul>
<h2>More Info</h2>
<h3>Installing Tensorflow 2.6</h3>
<pre><code>$ pip install --user tensorflow==2.6
</code></pre>
<h3>Optimize Tensorflow (Optional)</h3>
<p>to make use of your GPU, follow the steps in the <a href=https://www.tensorflow.org/install/pip?hl=es-419" title="tensorflow official website" target="_blank">tensorflow official website</a>.  <br>
This will make training MUCH faster!</p>

<h2 class="gap">Tasks</h2>
<h3 class="panel-title">
      0. Placeholders
    </h3>
<p>Write the function <code>def create_placeholders(nx, classes):</code> that returns two placeholders, <code>x</code> and <code>y</code>, for the neural network:</p>
<ul>
<li><code>nx</code>: the number of feature columns in our data</li>
<li><code>classes</code>: the number of classes in our classifier</li>
<li>Returns: placeholders named <code>x</code> and <code>y</code>, respectively

<ul>
<li><code>x</code> is the placeholder for the input data to the neural network</li>
<li><code>y</code> is the placeholder for the one-hot labels for the input data </li>
</ul></li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x02-tensorflow$ cat 0-main.py 
#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_placeholders = __import__('0-create_placeholders').create_placeholders

x, y = create_placeholders(784, 10)
print(x)
print(y)
ubuntu@alexa-ml:\~/0x02-tensorflow$ ./0-main.py
Tensor("x:0", shape=(?, 784), dtype=float32)
Tensor("y:0", shape=(?, 10), dtype=float32)
ubuntu@alexa-ml:\~/0x02-tensorflow$
</code></pre>

<div class="list-group">
    <!-- Task URLs -->

<!-- Github information -->
<div class="list-group-item">
<p><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x02-tensorflow</code></li>
    <li>File: <code>0-create_placeholders.py</code></li>
</ul>
</div>

<!-- Self-paced manual review -->
</div>

<h3 class="panel-title">
      1. Layers
    </h3>
<p>Write the function <code>def create_layer(prev, n, activation):</code></p>
<ul>
<li><code>prev</code> is the tensor output of the previous layer</li>
<li><code>n</code> is the number of nodes in the layer to create</li>
<li><code>activation</code> is the activation function that the layer should use</li>
<li>use <code>tf.keras.initializers.VarianceScaling(mode='fan_avg')</code> to implement<code>He et. al</code> initialization for the layer weights</li>
<li>each layer should be given the name <code>layer</code></li>
<li>Returns: the tensor output of the layer</li>
</ul>
<pre><code>ubuntu@alexa-ml:~/0x02-tensorflow$ cat 1-main.py 
#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_placeholders = __import__('0-create_placeholders').create_placeholders
create_layer = __import__('1-create_layer').create_layer

x, y = create_placeholders(784, 10)
l = create_layer(x, 256, tf.nn.tanh)
print(l)
ubuntu@alexa-ml:\~/0x02-tensorflow$ ./1-main.py 
Tensor("layer/Tanh:0", shape=(?, 256), dtype=float32)
ubuntu@alexa-ml:\~/0x02-tensorflow$ 
</code></pre>

<div data-role="task3677" data-position="3" id="task-num-2">
      <div class="panel panel-default task-card " id="task-3677">
  <span id="user_id" data-id="1283"></span>

  <div class="panel-heading panel-heading-actions">
    <h3 class="panel-title">
      2. Forward Propagation
    </h3>
  </div>

  <div class="panel-body">
    <span id="user_id" data-id="1283"></span>



<!-- Task Body -->
<p>Write the function <code>def forward_prop(x, layer_sizes=[], activations=[]):</code> that creates the forward propagation graph for the neural network:</p>

<ul>
<li><code>x</code> is the placeholder for the input data</li>
<li><code>layer_sizes</code> is a list containing the number of nodes in each layer of the network</li>
<li><code>activations</code> is a list containing the activation functions for each layer of the network</li>
<li>Returns: the prediction of the network in tensor form</li>
<li>For this function, you should import your <code>create_layer</code> function with <code>create_layer = __import__('1-create_layer').create_layer</code></li>
</ul>

<pre><code>ubuntu@alexa-ml:~0x02-tensorflow$ cat 2-main.py 
#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
print(y_pred)
ubuntu@alexa-ml:~0x02-tensorflow$ ./2-main.py 
Tensor("layer_2/BiasAdd:0", shape=(?, 10), dtype=float32)
ubuntu@alexa-ml: \~/0x02-tensorflow$ 
</code></pre>

  </div>

  <div class="list-group">
    <!-- Task URLs -->

<!-- Technical information -->
<div class="list-group-item">
<p><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x02-tensorflow</code></li>
    <li>File: <code>2-forward_prop.py</code></li>
</ul>
</div>

<!-- Self-paced manual review -->
 </div>

## 2. Forward Propagation
Write the function `def forward_prop(x, layer_sizes=[], activations=[]):` that creates the forward propagation graph for the neural network:

* `x` is the placeholder for the input data
* `layer_sizes` is a list containing the number of nodes in each layer of the network
* `activations` is a list containing the activation functions for each layer of the network
* Returns: the prediction of the network in tensor form
* For this function, you should import your create_layer function with `create_layer = __import__('1-create_layer').create_layer`

.

    ubuntu@alexa-ml:~/0x02-tensorflow$ cat 2-main.py 
    #!/usr/bin/env python3

    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    create_placeholders = __import__('0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop

    x, y = create_placeholders(784, 10)
    y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
    print(y_pred)
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./2-main.py 
    Tensor("layer_2/BiasAdd:0", shape=(?, 10), dtype=float32)
    ubuntu@alexa-ml:~/0x02-tensorflow$ 

## 3. Accuracy
Write the function `def calculate_accuracy(y, y_pred):` that calculates the accuracy of a prediction:

* `y` is a placeholder for the labels of the input data
* `y_pred` is a tensor containing the network’s predictions
* Returns: a tensor containing the decimal accuracy of the prediction
*hint:* accuracy = correct_predictions / all_predictions

.

    ubuntu@alexa-ml:~/0x02-tensorflow$ cat 3-main.py 
    #!/usr/bin/env python3

    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    create_placeholders = __import__('0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy

    x, y = create_placeholders(784, 10)
    y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
    accuracy = calculate_accuracy(y, y_pred)
    print(accuracy)
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./3-main.py 
    Tensor("Mean:0", shape=(), dtype=float32)
    ubuntu@alexa-ml:~/0x02-tensorflow$

# 4. Loss
Write the function `def calculate_loss(y, y_pred):` that calculates the softmax cross-entropy loss of a prediction:

* `y` is a placeholder for the labels of the input data
* `y_pred` is a tensor containing the network’s predictions
* Returns: a tensor containing the loss of the prediction

.

    ubuntu@alexa-ml:~/0x02-tensorflow$ cat 4-main.py 
    #!/usr/bin/env python3

    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    create_placeholders = __import__('0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_loss = __import__('4-calculate_loss').calculate_loss

    x, y = create_placeholders(784, 10)
    y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
    loss = calculate_loss(y, y_pred)
    print(loss)
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./4-main.py 
    Tensor("softmax_cross_entropy_loss/value:0", shape=(), dtype=float32)
    ubuntu@alexa-ml:~/0x02-tensorflow$ 

# 5. Train_Op
Write the function `def create_train_op(loss, alpha):` that creates the training operation for the network:

* `loss` is the loss of the network’s prediction
* `alpha` is the learning rate
* Returns: an operation that trains the network using gradient descent

.

    ubuntu@alexa-ml:~/0x02-tensorflow$ cat 5-main.py 
    #!/usr/bin/env python3

    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    create_placeholders = __import__('0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_train_op = __import__('5-create_train_op').create_train_op


    x, y = create_placeholders(784, 10)
    y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, 0.01)
    print(train_op)
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./5-main.py 
    name: "GradientDescent"
    op: "NoOp"
    input: "^GradientDescent/update_layer/kernel/ApplyGradientDescent"
    input: "^GradientDescent/update_layer/bias/ApplyGradientDescent"
    input: "^GradientDescent/update_layer_1/kernel/ApplyGradientDescent"
    input: "^GradientDescent/update_layer_1/bias/ApplyGradientDescent"
    input: "^GradientDescent/update_layer_2/kernel/ApplyGradientDescent"
    input: "^GradientDescent/update_layer_2/bias/ApplyGradientDescent"

    ubuntu@alexa-ml:~/0x02-tensorflow$

# 6. Train
<div class="panel-body">
    <span id="user_id" data-id="1283"></span>

<!-- Task Body -->
<p>Write the function <code>def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):</code> that builds, trains, and saves a neural network classifier:</p>

<ul>
<li><code>X_train</code> is a <code>numpy.ndarray</code> containing the training input data</li>
<li><code>Y_train</code> is a <code>numpy.ndarray</code> containing the training labels</li>
<li><code>X_valid</code> is a <code>numpy.ndarray</code> containing the validation input data</li>
<li><code>Y_valid</code> is a <code>numpy.ndarray</code> containing the validation labels</li>
<li><code>layer_sizes</code> is a list containing the number of nodes in each layer of the network</li>
<li><code>activations</code> is a list containing the activation functions for each layer of the network</li>
<li><code>alpha</code> is the learning rate</li>
<li><code>iterations</code> is the number of iterations to train over</li>
<li><code>save_path</code> designates where to save the model</li>
<li>Add the following to the graph’s collection

<ul>
<li>placeholders <code>x</code> and <code>y</code></li>
<li>tensors <code>y_pred</code>, <code>loss</code>, and <code>accuracy</code></li>
<li>operation <code>train_op</code></li>
</ul></li>
<li>After every 100 iterations, the 0th iteration, and <code>iterations</code> iterations, print the following:

<ul>
<li><code>After {i} iterations:</code> where i is the iteration</li>
<li><code>\tTraining Cost: {cost}</code> where <code>{cost}</code> is the training cost</li>
<li><code>\tTraining Accuracy: {accuracy}</code> where <code>{accuracy}</code> is the training accuracy</li>
<li><code>\tValidation Cost: {cost}</code> where <code>{cost}</code> is the validation cost</li>
<li><code>\tValidation Accuracy: {accuracy}</code> where <code>{accuracy}</code> is the validation accuracy</li>
</ul></li>
<li><em>Reminder: the 0th iteration represents the model before any training has occurred</em></li>
<li>After training has completed, save the model to <code>save_path</code></li>
<li>You may use the following imports:

<ul>
<li><code>calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy</code></li>
<li><code>calculate_loss = __import__('4-calculate_loss').calculate_loss</code></li>
<li><code>create_placeholders = __import__('0-create_placeholders').create_placeholders</code></li>
<li><code>create_train_op = __import__('5-create_train_op').create_train_op</code></li>
<li><code>forward_prop = __import__('2-forward_prop').forward_prop</code></li>
</ul></li>
<li>You are not allowed to use <code>tf.saved_model</code></li>
<li>Returns: the path where the model was saved</li>
</ul>

<pre><code>ubuntu@alexa-ml:~/0x02-tensorflow$ cat 6-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
train = __import__('6-train').train

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

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
    iterations = 1000

    tf.set_random_seed(0)
    save_path = train(X_train, Y_train_oh, X_valid, Y_valid_oh, layer_sizes,
                      activations, alpha, iterations, save_path="./model.ckpt")
    print("Model saved in path: {}".format(save_path))
ubuntu@alexa-ml:~/0x02-tensorflow$ ./6-main.py 
2018-11-03 01:04:55.281078: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
After 0 iterations:
    Training Cost: 2.8232274055480957
    Training Accuracy: 0.08726000040769577
    Validation Cost: 2.810533285140991
    Validation Accuracy: 0.08640000224113464
After 100 iterations:
    Training Cost: 0.8393384218215942
    Training Accuracy: 0.7824000120162964
    Validation Cost: 0.7826032042503357
    Validation Accuracy: 0.8061000108718872
After 200 iterations:
    Training Cost: 0.6094841361045837
    Training Accuracy: 0.8396000266075134
    Validation Cost: 0.5562412142753601
    Validation Accuracy: 0.8597999811172485

...

After 1000 iterations:
    Training Cost: 0.352960467338562
    Training Accuracy: 0.9004999995231628
    Validation Cost: 0.32148978114128113
    Validation Accuracy: 0.909600019454956
Model saved in path: ./model.ckpt
ubuntu@alexa-ml:~/0x02-tensorflow$ ls model.ckpt*
model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta
ubuntu@alexa-ml:~/0x02-tensorflow$
</code></pre>

</div>

# 7. Evaluate
Write the function ``def evaluate(X, Y, save_path):`` that evaluates the output of a neural network:

* `X` is a `numpy.ndarray` containing the input data to evaluate
* `Y` is a `numpy.ndarray` containing the one-hot labels for X
* `save_path` is the location to load the model from
* You are not allowed to use `tf.saved_model`
* Returns: the network’s prediction, accuracy, and loss, respectively

<pre><code>ubuntu@alexa-ml:~/0x02-tensorflow$ cat 7-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
evaluate = __import__('7-evaluate').evaluate

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_test_3D = lib['X_test']
    Y_test = lib['Y_test']
    X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))
    Y_test_oh = one_hot(Y_test, 10)

    Y_pred_oh, accuracy, cost = evaluate(X_test, Y_test_oh, './model.ckpt')
    print("Test Accuracy:", accuracy)
    print("Test Cost:", cost)

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_test_3D[i])
        plt.title(str(Y_test[i]) + ' : ' + str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@alexa-ml:~/0x02-tensorflow$ ./7-main.py
2018-11-03 02:08:30.767168: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Test Accuracy: 0.9096
Test Cost: 0.32148978
</code></pre>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/11/1a553e937dc9500036f8.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20231122%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20231122T221749Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=16cf49655d92023b1ffca9aa12eda9ff00d8fa71ad8e996c3633930ba16e246b" alt="" loading="lazy" style=""></p>

