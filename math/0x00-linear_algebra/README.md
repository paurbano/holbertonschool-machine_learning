# 0x00. Linear Algebra
General
What is a vector?
What is a matrix?
What is a transpose?
What is the shape of a matrix?
What is an axis?
What is a slice?
How do you slice a vector/matrix?
What are element-wise operations?
How do you concatenate vectors/matrices?
What is the dot product?
What is matrix multiplication?
What is Numpy?
What is parallelization and why is it important?
What is broadcasting?

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
