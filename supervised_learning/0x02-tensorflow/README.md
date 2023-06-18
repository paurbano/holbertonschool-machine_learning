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
