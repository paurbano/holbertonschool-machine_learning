# 0x03. Probability

## General
* **What is probability?**

Probability is the branch of mathematics concerning numerical descriptions of how likely an event is to occur or how likely it is that a proposition is true. The probability of an event is a number between 0 and 1, where, roughly speaking, 0 indicates impossibility of the event and 1 indicates certainty

* **Basic probability notation**
    * or = U
    * and = Ո
    * Given = |

* **What is independence? What is disjoint?**
    * independence = Two events are independent, when the probability of occurence of one event is not influence by the occurence of other happens, it means both are not related.
        * P(A) = P(A|B) or
        * P(A).P(B) = P(A Ո B)
    * disjoint = Means that two events are mutually exclusive they can´t happen at the same time, hence they can´t be independent.

* **What is a union? intersection?**
    * union = The probability that Events A or B occur is the probability of the `union` of A and B. The probability of the union of Events A and B is denoted by P(A ∪ B) .
    * intersection = The probability that Events A and B both occur is the probability of the `intersection` of A and B. The probability of the intersection of Events A and B is denoted by P(A ∩ B). If Events A and B are mutually exclusive, P(A ∩ B) = 0.

* **What are the general addition and multiplication rules?**
    *  Mulitplication Rule  = The rule of multiplication applies to the situation when we want to know the probability of the intersection of two events; that is, we want to know the probability that two events (Event A and Event B) both occur.
    * The rule of addition applies to the following situation. We have two events, and we want to know the probability that either event occurs.

* **What is a probability distribution?**
is the mathematical function that gives the probabilities of occurrence of different possible outcomes for an experiment.

* What is a probability distribution function? probability mass function?
    * probability distribution function =  some function that may be used to define a particular **probability distribution**
    * probability mass function = In probability and statistics, a probability mass function **(PMF)** is a function that gives the probability that a discrete random variable is exactly equal to some value. Sometimes it is also known as the discrete density function. The probability mass function is often the primary means of defining a discrete probability distribution, and such functions exist for either scalar or multivariate random variables whose domain is discrete.

* **What is a cumulative distribution function?**
**cumulative distribution function (CDF)** of a real-valued random variable **X**, or just distribution function of **X**, evaluated at *x*, is the probability that `X` will take a value less than or equal to `x`

* **What is a percentile?**
A **percentile** (or a **centile**) is a measure used in statistics indicating the value below which a given percentage of observations in a group of observations falls. For example, the 20th percentile is the value (or score) below which 20% of the observations may be found. Equivalently, 80% of the observations are found above the 20th percentile.

* What is mean, standard deviation, and variance?
    * mean = The mean and the median are summary measures used to describe the most "typical" value in a set of values. Statisticians sometimes refer to the mean and median as measures of central tendency.
    The **mean** of a sample or a population is computed by adding all of the observations and dividing by the number of observations.

    * standard deviation = is a measure of the amount of variation or dispersion of a set of values. A low standard deviation indicates that the values tend to be close to the mean of the set, while a high standard deviation indicates that the values are spread out over a wider range.

    * variance = is the expectation of the squared deviation of a random variable from its mean. Informally, it measures how far a set of numbers are spread out from their average value. Variance has a central role in statistics, where some ideas that use it include descriptive statistics, statistical inference, hypothesis testing, goodness of fit, and Monte Carlo sampling. Variance is an important tool in the sciences, where statistical analysis of data is common. The variance is the square of the standard deviation, the second central moment of a distribution, and the covariance of the random variable with itself.

* Common probability distributions
    * Bernoulli and Uniform
    * Binomial and Hypergeometric
    * Poisson
    * Geometric and Negative Binomial
    * Exponential and Weibull
    * Normal, Log-Normal, Student’s t, and Chi-squared
    * Gamma and Beta

# Tasks

## 0. Initialize Poisson
Create a class Poisson that represents a poisson distribution:

* Class contructor `def __init__(self, data=None, lambtha=1.):`
    * `data` is a list of the data to be used to estimate the distribution
    * `lambtha` is the expected number of occurences in a given time frame
    * Sets the instance attribute `lambtha`
        * Saves `lambtha` as a float
    * If `data`  is not given, (i.e. `None`):
        * Use the given `lambtha`
        * If `lambtha` is not a positive value, raise a `ValueError` with the message `lambtha must be a positive value`
    * If `data` is given:
        * Calculate the `lambtha` of `data`
        * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`
```
alexa@ubuntu-xenial:0x03-probability$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('Lambtha:', p1.lambtha)

p2 = Poisson(lambtha=5)
print('Lambtha:', p2.lambtha)
alexa@ubuntu-xenial:0x03-probability$ ./0-main.py 
Lambtha: 4.84
Lambtha: 5.0
alexa@ubuntu-xenial:0x03-probability$
```
File: `poisson.py`

## Poisson PMF
Update the class `Poisson`:

* Instance method `def pmf(self, k):`
    * Calculates the value of the PMF for a given number of “successes”
    * k is the number of “successes”
        * If k is not an integer, convert it to an integer
        * If k is out of range, return 0
    * Returns the PMF value for k

```
alexa@ubuntu-xenial:0x03-probability$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('P(9):', p1.pmf(9))

p2 = Poisson(lambtha=5)
print('P(9):', p2.pmf(9))
alexa@ubuntu-xenial:0x03-probability$ ./1-main.py 
P(9): 0.03175849616802446
P(9): 0.036265577412911795
alexa@ubuntu-xenial:0x03-probability$
```

## 2. Poisson CDF
Update the class `Poisson:`

* Instance method `def cdf(self, k):`
    * Calculates the value of the CDF for a given number of “successes”
    * `k` is the number of “successes”
        * If `k` is not an integer, convert it to an integer
        * If `k` is out of range, return `0`
    * Returns the CDF value for `k`

```
alexa@ubuntu-xenial:0x03-probability$ cat 2-main.py 
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('F(9):', p1.cdf(9))

p2 = Poisson(lambtha=5)
print('F(9):', p2.cdf(9))
alexa@ubuntu-xenial:0x03-probability$ ./2-main.py 
F(9): 0.9736102067423525
F(9): 0.9681719426208609
alexa@ubuntu-xenial:0x03-probability$ 
```
File: `poisson.py`

## 3. Initialize Exponential
Create a class `Exponential` that represents an exponential distribution:

* Class contructor `def __init__(self, data=None, lambtha=1.):`
    * `data` is a list of the data to be used to estimate the distribution
    * `lambtha` is the expected number of occurences in a given time frame
    * Sets the instance attribute `lambtha`
        * Saves `lambtha` as a float
    * If `data` is not given (i.e. `None`):
        * Use the given `lambtha`
        * If `lambtha` is not a positive value, raise a `ValueError` with the message `lambtha must be a positive value`
    * If `data` is given:
        * Calculate the `lambtha` of `data`
        * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

```
alexa@ubuntu-xenial:0x03-probability$ cat 3-main.py 
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('Lambtha:', e1.lambtha)

e2 = Exponential(lambtha=2)
print('Lambtha:', e2.lambtha)
alexa@ubuntu-xenial:0x03-probability$ ./3-main.py 
Lambtha: 2.1771114730906937
Lambtha: 2.0
alexa@ubuntu-xenial:0x03-probability$
```

## 4. Exponential PDF
Update the class `Exponential:`

* Instance method `def pdf(self, x):`
    * Calculates the value of the PDF for a given time period
    * `x` is the time period
    * Returns the PDF value for `x`
    * If `x` is out of range, return `0`

```
alexa@ubuntu-xenial:0x03-probability$ cat 4-main.py 
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('f(1):', e1.pdf(1))

e2 = Exponential(lambtha=2)
print('f(1):', e2.pdf(1))
alexa@ubuntu-xenial:0x03-probability$ ./4-main.py 
f(1): 0.24681591903431568
f(1): 0.2706705664650693
alexa@ubuntu-xenial:0x03-probability$
```

## 5. Exponential CDF
Update the class `Exponential:`

* Instance method `def cdf(self, x):`
    * Calculates the value of the CDF for a given time period
    * `x` is the time period
    * Returns the CDF value for `x`
    * If `x` is out of range, return `0`

```
alexa@ubuntu-xenial:0x03-probability$ cat 5-main.py 
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('F(1):', e1.cdf(1))

e2 = Exponential(lambtha=2)
print('F(1):', e2.cdf(1))
alexa@ubuntu-xenial:0x03-probability$ ./5-main.py 
F(1): 0.886631473819791
F(1): 0.8646647167674654
alexa@ubuntu-xenial:0x03-probability$
```
File: `exponential.py`