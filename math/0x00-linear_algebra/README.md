<h2>Resources</h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="/rltoken/C05mTOfKzZgz_AVSosNKIw" title="Introduction to vectors" target="_blank">Introduction to vectors</a> </li>
<li><a href="/rltoken/vLe4BBPfmLXy2s_Idqo87w" title="What is a matrix?" target="_blank">What is a matrix?</a> (<em>not <a href="/rltoken/Zad2ReJ9SU4IuQ3ZX986qw" title="the matrix" target="_blank">the matrix</a></em>)</li>
<li><a href="/rltoken/xHWwQjqH9tgEcskvFQaV7A" title="Transpose" target="_blank">Transpose</a> </li>
<li><a href="/rltoken/2tYcOFY35stXjd0nhTpgFA" title="Understanding the dot product" target="_blank">Understanding the dot product</a> </li>
<li><a href="/rltoken/pV4znghCxaXAAny4Ou-cNw" title="Matrix Multiplication" target="_blank">Matrix Multiplication</a> </li>
<li><a href="/rltoken/ih50DhE4FvilyItYPo1x5A" title="What is the relationship between matrix multiplication and the dot product?" target="_blank">What is the relationship between matrix multiplication and the dot product?</a> </li>
<li><a href="/rltoken/DnAvjbmojZutluWV9OJVOg" title="The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices" target="_blank">The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices</a> (<em>advanced</em>)</li>
<li><a href="/rltoken/MBHHb0eiN0OummbEdI9g_Q" title="numpy tutorial" target="_blank">numpy tutorial</a> (<em>until Shape Manipulation (excluded)</em>)</li>
<li><a href="/rltoken/L8RdIDGi3GGO-_erGcMORg" title="numpy basics" target="_blank">numpy basics</a> (<em>until Universal Functions (included)</em>)</li>
<li><a href="/rltoken/1LPk4EosRetS_C7eX-mQNA" title="array indexing" target="_blank">array indexing</a> </li>
<li><a href="/rltoken/slRzAgt6aom5-Nj5XSdUcQ" title="numerical operations on arrays" target="_blank">numerical operations on arrays</a> </li>
<li><a href="/rltoken/xgq6QIOHufhg8lHCZn0jwA" title="Broadcasting" target="_blank">Broadcasting</a> </li>
<li><a href="/rltoken/Q5FEVV4BArJtnJnbReng7Q" title="numpy mutations and broadcasting" target="_blank">numpy mutations and broadcasting</a> </li>
</ul>
<p><strong>References</strong>:</p>
<ul>
<li><a href="/rltoken/Ah-QtZhAhFSYnloj837a8Q" title="numpy.ndarray" target="_blank">numpy.ndarray</a> </li>
<li><a href="/rltoken/mvx-STJbJ4Nn1N_BFfpnaQ" title="numpy.ndarray.shape" target="_blank">numpy.ndarray.shape</a> </li>
<li><a href="/rltoken/I1V8iDWar7Hnoh_VwQzZ_Q" title="numpy.transpose" target="_blank">numpy.transpose</a> </li>
<li><a href="/rltoken/iv73fN04gTbpeV_XcIIaPQ" title="numpy.ndarray.transpose" target="_blank">numpy.ndarray.transpose</a> </li>
<li><a href="/rltoken/MbHJEqjwavimnL8HRtaYCA" title="numpy.matmul" target="_blank">numpy.matmul</a> </li>
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
