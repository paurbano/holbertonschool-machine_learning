# 0x02. Tensorflow

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
