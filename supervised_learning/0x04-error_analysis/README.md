# 0x04. Error Analysis

<h2>Resources</h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Confusion_matrix" title="Confusion matrix" target="_blank">Confusion matrix</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Type_I_and_type_II_errors" title="Type I and type II errors" target="_blank">Type I and type II errors</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Sensitivity_and_specificity" title="Sensitivity and specificity" target="_blank">Sensitivity and specificity</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Precision_and_recall" title="Precision and recall" target="_blank">Precision and recall</a> </li>
<li><a href="https://en.wikipedia.org/wiki/F-score" title="F1 score" target="_blank">F1 score</a> </li>
<li><a href="https://machinelearningmastery.com/confusion-matrix-machine-learning/" title="What is a Confusion Matrix in Machine Learning?" target="_blank">What is a Confusion Matrix in Machine Learning?</a> </li>
<li><a href="https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/" title="Simple guide to confusion matrix terminology" target="_blank">Simple guide to confusion matrix terminology</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff" title="Bias-variance tradeoff" target="_blank">Bias-variance tradeoff</a> </li>
<li><a href="https://www.quora.com/What-is-bias-and-variance" title="What is bias and variance" target="_blank">What is bias and variance</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Bayes_error_rate" title="Bayes error rate" target="_blank">Bayes error rate</a> </li>
<li><a href="https://stats.stackexchange.com/questions/302900/what-is-bayes-error-in-machine-learning" title="What is Bayes Error in machine learning?" target="_blank">What is Bayes Error in machine learning?</a> </li>
<li><a href="https://www.youtube.com/watch?v=SjQyLhQIXSM&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=3" title="Bias/Variance" target="_blank">Bias/Variance</a> (<em>Note: I suggest watching this video at 1.5x - 2x speed</em>)</li>
<li><a href="https://www.youtube.com/watch?v=C1N_PDHuJ6Q&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=4" title="Basic Recipe for Machine Learning" target="_blank">Basic Recipe for Machine Learning</a> (<em>Note: I suggest watching this video at 1.5x - 2x speed</em>)</li>
<li><a href="https://www.youtube.com/watch?v=J3HHOwcrkK8&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=9" title="Why Human Level Performance" target="_blank">Why Human Level Performance</a> (<em>Note: I suggest watching this video at 1.5x - 2x speed</em>)</li>
<li><a href="https://www.youtube.com/watch?v=CZf3oo0fuh0&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=10" title="Avoidable Bias" target="_blank">Avoidable Bias</a> (<em>Note: I suggest watching this video at 1.5x - 2x speed</em>)</li>
<li><a href="https://www.youtube.com/watch?v=NUmbgp1h64E&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=11" title="Understanding Human-Level Performance" target="_blank">Understanding Human-Level Performance</a> (<em>Note: I suggest watching this video at 1.5x - 2x speed</em>)</li>
</ul>

## General
* What is the confusion matrix?
* What is type I error? type II?
* What is sensitivity? specificity? precision? recall?
* What is an F1 score?
* What is bias? variance?
* What is irreducible error?
* What is Bayes error?
* How can you approximate Bayes error?
* How to calculate bias and variance
* How to create a confusion matrix

## 0. Create Confusion
Write the function `def create_confusion_matrix(labels, logits):` that creates a confusion matrix:

* labels is a one-hot numpy.ndarray of shape (m, classes) containing the correct labels for each data point
    * `m` is the number of data points
    * `classes` is the number of classes
* `logits` is a one-hot `numpy.ndarray` of shape `(m, classes)` containing the predicted labels
* Returns: a confusion `numpy.ndarray` of shape `(classes, classes)` with row indices representing the correct labels and column indices representing the predicted labels

```
alexa@ubuntu-xenial:0x04-error_analysis$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix

if __name__ == '__main__':
    lib = np.load('labels_logits.npz')
    labels = lib['labels']
    logits = lib['logits']

    np.set_printoptions(suppress=True)
    confusion = create_confusion_matrix(labels, logits)
    print(confusion)
    np.savez_compressed('confusion.npz', confusion=confusion)
alexa@ubuntu-xenial:0x04-error_analysis$ ./0-main.py 
[[4701.    0.   36.   17.   12.   81.   38.   11.   35.    1.]
 [   0. 5494.   36.   21.    3.   38.    7.   13.   59.    7.]
 [  64.   93. 4188.  103.  108.   17.  162.   80.  132.   21.]
 [  30.   48.  171. 4310.    2.  252.   22.   86.  128.   52.]
 [  17.   27.   35.    0. 4338.   11.   84.    9.   27.  311.]
 [  89.   57.   45.  235.   70. 3631.  123.   33.  163.   60.]
 [  47.   32.   87.    1.   64.   83. 4607.    0.   29.    1.]
 [  26.   95.   75.    7.   58.   18.    1. 4682.   13.  200.]
 [  31.  153.   82.  174.   27.  179.   64.    7. 4003.  122.]
 [  48.   37.   39.   71.  220.   49.    8.  244.   46. 4226.]]
alexa@ubuntu-xenial:0x04-error_analysis$

```

File: `0-create_confusion.py`

## 1. Sensitivity
Write the function `def sensitivity(confusion):` that calculates the sensitivity for each class in a confusion matrix:

* confusion is a confusion `numpy.ndarray of shape (classes, classes)` where row indices represent the correct labels and column indices represent the predicted labels
    * classes is the number of classes
* Returns: a `numpy.ndarray` of shape `(classes,)` containing the sensitivity of each class

```
alexa@ubuntu-xenial:0x04-error_analysis$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(sensitivity(confusion))
alexa@ubuntu-xenial:0x04-error_analysis$ ./1-main.py 
[0.95316302 0.96759422 0.84299517 0.84493237 0.89277629 0.80581447
 0.93051909 0.9047343  0.82672449 0.84723336]
alexa@ubuntu-xenial:0x04-error_analysis$ 
```
### Repo:

* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/0x04-error_analysis`
* File: `1-sensitivity.py`

## 2. Precision
Write the function `def precision(confusion):` that calculates the precision for each class in a confusion matrix:

* confusion is a confusion `numpy.ndarray` of shape `(classes, classes)` where row indices represent the correct labels and column indices represent the predicted labels
    * classes is the number of classes
* Returns: a `numpy.ndarray` of shape `(classes,)` containing the precision of each class
```
alexa@ubuntu-xenial:0x04-error_analysis$ cat 2-main.py 
#!/usr/bin/env python3

import numpy as np
precision = __import__('2-precision').precision

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(precision(confusion))
alexa@ubuntu-xenial:0x04-error_analysis$ ./2-main.py 
[0.93033841 0.91020543 0.87359199 0.87264628 0.88494492 0.83298922
 0.90050821 0.90648596 0.86364617 0.84503099]
alexa@ubuntu-xenial:0x04-error_analysis$
```

File: `2-precision.py`

## 3. Specificity
Write the function `def specificity(confusion):` that calculates the specificity for each class in a confusion matrix:

* confusion is a confusion `numpy.ndarray` of shape `(classes, classes)` where row indices represent the correct labels and column indices represent the predicted labels
    * `classes` is the number of classes
* Returns: a `numpy.ndarray` of shape `(classes,)` containing the specificity of each class
```
alexa@ubuntu-xenial:0x04-error_analysis$ cat 3-main.py 
#!/usr/bin/env python3

import numpy as np
specificity = __import__('3-specificity').specificity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(specificity(confusion))
alexa@ubuntu-xenial:0x04-error_analysis$ ./3-main.py 
[0.99218958 0.98777131 0.9865429  0.98599078 0.98750582 0.98399789
 0.98870119 0.98922476 0.98600469 0.98278237]
alexa@ubuntu-xenial:0x04-error_analysis$
```
When there are more than two classes in a confusion matrix, sensitivity is not a useful metric as there are inherently more actual negatives than actual positives. It is much better to use sensitivity (recall) and precision.

File: `3-specificity.py`

## 4. F1 score mandatory
Write the function def f1_score(confusion): that calculates the F1 score of a confusion matrix:

* confusion is a confusion `numpy.ndarray` of shape `(classes, classes)` where row indices represent the correct labels and column indices represent the predicted labels
    * `classes` is the number of classes
* Returns: a `numpy.ndarray` of shape (classes,) containing the F1 score of each class
* You may use `sensitivity = __import__('1-sensitivity').sensitivity` and `precision = __import__('2-precision').precision`
```
alexa@ubuntu-xenial:0x04-error_analysis$ cat 4-main.py 
#!/usr/bin/env python3

import numpy as np
f1_score = __import__('4-f1_score').f1_score

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(f1_score(confusion))
alexa@ubuntu-xenial:0x04-error_analysis$ ./4-main.py 
[0.94161242 0.93802288 0.8580209  0.85856574 0.88884336 0.81917654
 0.91526771 0.90560928 0.8447821  0.84613074]
alexa@ubuntu-xenial:0x04-error_analysis$
```

* File: `4-f1_score.py`

## 5. Dealing with Error
In the text file `5-error_handling`, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex. A,B,C):

Scenarios:
```
1. High Bias, High Variance
2. High Bias, Low Variance
3. Low Bias, High Variance
4. Low Bias, Low Variance
```
Approaches:
```
A. Train more
B. Try a different architecture
C. Get more data
D. Build a deeper network
E. Use regularization
F. Nothing
```

File: `5-error_handling`

## 6. Compare and Contrast
Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file `6-compare_and_contrast`

<p><img src="https://github.com/paurbano/
holbertonschool-machine_learning/blob/master/images/task6aErrorAnalisis.png" alt="" loading="lazy" style=""></p>
<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/task6bErrorAnalisis.png" alt="" loading="lazy" style=""></p>


Most important issue:
```
A. High Bias
B. High Variance
C. Nothing
```
