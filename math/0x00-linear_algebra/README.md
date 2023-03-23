<h2>Resources</h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="https://www.youtube.com/watch?v=fNk_zzaMoSs" title="Introduction to vectors" target="_blank">Introduction to vectors</a> </li>
<li><a href="https://math.stackexchange.com/questions/2782717/what-exactly-is-a-matrix" title="What is a matrix?" target="_blank">What is a matrix?</a> (<em>not <a href="https://www.imdb.com/title/tt0133093/" title="the matrix" target="_blank">the matrix</a></em>)</li>
<li><a href="https://en.wikipedia.org/wiki/Transpose" title="Transpose" target="_blank">Transpose</a> </li>
<li><a href="https://www.youtube.com/watch?v=BzWahqwaS8k" title="Understanding the dot product" target="_blank">Understanding the dot product</a> </li>
<li><a href="https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/" title="Matrix Multiplication" target="_blank">Matrix Multiplication</a> </li>
<li><a href="https://www.quora.com/What-is-the-relationship-between-matrix-multiplication-and-the-dot-product" title="What is the relationship between matrix multiplication and the dot product?" target="_blank">What is the relationship between matrix multiplication and the dot product?</a> </li>
<li><a href="https://www.youtube.com/watch?v=rW2ypKLLxGk" title="The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices" target="_blank">The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices</a> (<em>advanced</em>)</li>
<li><a href="https://numpy.org/doc/stable/user/quickstart.html" title="numpy tutorial" target="_blank">numpy tutorial</a> (<em>until Shape Manipulation (excluded)</em>)</li>
<li><a href="https://www.oreilly.com/library/view/python-for-data/9781449323592/ch04.html" title="numpy basics" target="_blank">numpy basics</a> (<em>until Universal Functions (included)</em>)</li>
<li><a href="https://numpy.org/doc/stable/reference/arrays.indexing.html#basic-slicing-and-indexing" title="array indexing" target="_blank">array indexing</a> </li>
<li><a href="http://scipy-lectures.org/intro/numpy/operations.html" title="numerical operations on arrays" target="_blank">numerical operations on arrays</a> </li>
<li><a href="https://numpy.org/doc/stable/user/basics.broadcasting.html" title="Broadcasting" target="_blank">Broadcasting</a> </li>
<li><a href="https://towardsdatascience.com/two-cool-features-of-python-numpy-mutating-by-slicing-and-broadcasting-3b0b86e8b4c7" title="numpy mutations and broadcasting" target="_blank">numpy mutations and broadcasting</a> </li>
</ul>
<p><strong>References</strong>:</p>
<ul>
<li><a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html" title="numpy.ndarray" target="_blank">numpy.ndarray</a> </li>
<li><a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html" title="numpy.ndarray.shape" target="_blank">numpy.ndarray.shape</a> </li>
<li><a href="https://numpy.org/doc/stable/reference/generated/numpy.transpose.html" title="numpy.transpose" target="_blank">numpy.transpose</a> </li>
<li><a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html" title="numpy.ndarray.transpose" target="_blank">numpy.ndarray.transpose</a> </li>
<li><a href="https://numpy.org/doc/stable/reference/generated/numpy.matmul.html" title="numpy.matmul" target="_blank">numpy.matmul</a> </li>
</ul>

# 0x00. Linear Algebra
General
* What is a vector?
* What is a matrix?
* What is a transpose?
* What is the shape of a matrix?
* What is an axis?
* What is a slice?
* How do you slice a vector/matrix?
* What are element-wise operations?
* How do you concatenate vectors/matrices?
* What is the dot product?
* What is matrix multiplication?
* What is Numpy?
* What is parallelization and why is it important?
* What is broadcasting?

## Installing pip 19.1

    wget https://bootstrap.pypa.io/get-pip.py
    sudo python3 get-pip.py
    rm get-pip.py

## Installing numpy 1.15, scipy 1.3, and pycodestyle 2.5

    $ pip install --user numpy==1.15
    $ pip install --user scipy==1.3
    $ pip install --user pycodestyle==2.5

    To check that all have been successfully downloaded, use pip list

# Tasks

## 0. Slice Me Up
Complete the following source code (found below):

* arr1 should be the first two numbers of arr
* arr2 should be the last five numbers of arr
* arr3 should be the 2nd through 6th numbers of arr
* You are not allowed to use any loops or conditional statements
* Your program should be exactly 8 lines

File: [0-slice_me_up.py](https://github.com/paurbano/holbertonschool-machine_learning/blob/master/math/0x00-linear_algebra/0-slice_me_up.py)

## 1. Trim Me Down
Complete the following source code (found below):

* the_middle should be a 2D matrix containing the 3rd and 4th columns of matrix
* You are not allowed to use any conditional statements
* You are only allowed to use one for loop
* Your program should be exactly 6 lines

example:

    alexa@ubuntu-xenial:0x00-linear_algebra$ cat 1-trim_me_down.py 
    #!/usr/bin/env python3
    matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
    the_middle = []
    # your code here
    print("The middle columns of the matrix are: {}".format(the_middle))
    alexa@ubuntu-xenial:0x00-linear_algebra$ ./1-trim_me_down.py 
    The middle columns of the matrix are: [[9, 4], [7, 3], [4, 6]]
    alexa@ubuntu-xenial:0x00-linear_algebra$ wc -l 1-trim_me_down.py 
    6 1-trim_me_down.py
    alexa@ubuntu-xenial:0x00-linear_algebra$

File: [1-trim_me_down.py]

## 2. Size Me Please
Write a function `def matrix_shape(matrix):` that calculates the shape of a matrix:

* You can assume all elements in the same dimension are of the same type/shape
* The shape should be returned as a list of integers

File: [2-size_me_please.py]

## 3. Flip Me Over
Write a function `def matrix_transpose(matrix):` that returns the transpose of a 2D matrix, matrix:

* You must return a new matrix
* You can assume that matrix is never empty
* You can assume all elements in the same dimension are of the same type/shape

## 4. Line Up
Write a function `def add_arrays(arr1, arr2):` that adds two arrays element-wise:

You can assume that arr1 and arr2 are lists of ints/floats
You must return a new list
If arr1 and arr2 are not the same shape, return None
