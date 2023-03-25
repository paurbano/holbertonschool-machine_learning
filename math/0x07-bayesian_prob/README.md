# 0x07. Bayesian Probability
<p><img src="https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/bayesianProbability.png" alt="" loading="lazy" style=""></p>
<h2>Resources</h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Bayesian_probability" title="Bayesian probability" target="_blank">Bayesian probability</a></li>
<li><a href="https://en.wikipedia.org/wiki/Bayesian_statistics" title="Bayesian statistics" target="_blank">Bayesian statistics</a></li>
<li><a href="https://www.youtube.com/watch?v=XQoLVl31ZfQ" title="Bayes' Theorem - The Simplest Case" target="_blank">Bayes’ Theorem - The Simplest Case</a></li>
<li><a href="https://www.youtube.com/watch?v=BrK7X_XlGB8" title="A visual guide to Bayesian thinking" target="_blank">A visual guide to Bayesian thinking</a></li>
<li><a href="https://onlinestatbook.com/2/probability/base_rates.html" title="Base Rates" target="_blank">Base Rates</a></li>
<li><a href="https://www.youtube.com/playlist?list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm" title="Bayesian statistics: a comprehensive course" target="_blank">Bayesian statistics: a comprehensive course</a>

<ul>
<li><a href="https://www.youtube.com/watch?v=EbyUsf_jUjk&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=15" title="Bayes' rule - an intuitive explanation" target="_blank">Bayes’ rule - an intuitive explanation</a></li>
<li><a href="https://www.youtube.com/watch?v=i567qvWejJA&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=16" title="Bayes' rule in statistics" target="_blank">Bayes’ rule in statistics</a></li>
<li><a href="https://www.youtube.com/watch?v=c69a_viMRQU&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=17" title="Bayes' rule in inference - likelihood" target="_blank">Bayes’ rule in inference - likelihood</a></li>
<li><a href="https://www.youtube.com/watch?v=a5QDDZLGSXY&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=18" title="Bayes' rule in inference - the prior and denominator" target="_blank">Bayes’ rule in inference - the prior and denominator</a></li>
<li><a href="https://www.youtube.com/watch?v=QEzeLh6L9Tg&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=25" title="Bayes' rule denominator: discrete and continuous" target="_blank">Bayes’ rule denominator: discrete and continuous</a></li>
<li><a href="https://www.youtube.com/watch?v=sm60vapz2jQ&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=26" title="Bayes' rule: why likelihood is not a probability" target="_blank">Bayes’ rule: why likelihood is not a probability</a></li>
</ul></li>
</ul>

## Learning Objectives
* What is Bayesian Probability?
* What is Bayes’ rule and how do you use it?
* What is a base rate?
* What is a prior?
* What is a posterior?
* What is a likelihood?

## 0. Likelihood
You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, `n` patients take the drug and x patients develop severe side effects. You can assume that `x` follows a binomial distribution.

Write a function `def likelihood(x, n, P):` that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects:

* `x` is the number of patients that develop severe side effects
* `n` is the total number of patients observed
* `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
* If n is not a positive integer, raise a `ValueError` with the message n must be a positive integer
* If x is not an integer that is greater than or equal to 0, raise a `ValueError` with the message x must be an integer that is greater than or equal to 0`` 
* If `x` is greater than n, raise a `ValueError` with the message `x cannot be greater than n`
* If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
* If any value in P is not in the range [0, 1], raise a `ValueError` with the message `All values in P must be in the range [0, 1]`
* Returns: a 1D `numpy.ndarray` containing the likelihood of obtaining the data, x and n, for each probability in P, respectively
```
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    likelihood = __import__('0-likelihood').likelihood

    P = np.linspace(0, 1, 11) # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(likelihood(26, 130, P))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./0-main.py 
[0.00000000e+00 2.71330957e-04 8.71800070e-02 3.07345706e-03
 5.93701546e-07 1.14387595e-12 1.09257177e-20 6.10151799e-32
 9.54415702e-49 1.00596671e-78 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$
```
File: [0-likelihood.py]

## 1. Intersection
Based on `0-likelihood.py`, write a function `def intersection(x, n, P, Pr):` that calculates the intersection of obtaining this data with the various hypothetical probabilities:

* `x` is the number of patients that develop severe side effects
* `n` is the total number of patients observed
* `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
* `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of P
* If `n` is not a positive integer, raise a ValueError with the message n must be a positive integer
* If `x` is not an integer that is greater than or equal to 0, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
* If `x` is greater than n, raise a `ValueError` with the message `x cannot be greater than n`
* If `P` is not a 1D `numpy.ndarray`, raise a TypeError with the message P must be a 1D numpy.ndarray
* If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
* If any value in P or Pr is not in the range [0, 1], raise a `ValueError` with the message `All values in {P} must be in the range [0, 1] where {P} is the incorrect variable`
* If Pr does not sum to 1, raise a `ValueError` with the message `Pr must sum to 1` Hint: use numpy.isclose
* All exceptions should be raised in the above order
* Returns: a 1D `numpy.ndarray` containing the intersection of obtaining x and n with each probability in P, respectively
```
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    intersection = __import__('1-intersection').intersection

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11 # this prior assumes that everything is equally as likely
    print(intersection(26, 130, P, Pr))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./1-main.py 
[0.00000000e+00 2.46664506e-05 7.92545518e-03 2.79405187e-04
 5.39728678e-08 1.03988723e-13 9.93247059e-22 5.54683454e-33
 8.67650639e-50 9.14515194e-80 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$
```
## 2. Marginal Probability
Based on `1-intersection.py`, write a function `def marginal(x, n, P, Pr):` that calculates the marginal probability of obtaining the data:

* x is the number of patients that develop severe side effects
* n is the total number of patients observed
* P is a 1D numpy.ndarray containing the various hypothetical probabilities of patients developing severe side effects
* Pr is a 1D numpy.ndarray containing the prior beliefs about P
* If n is not a positive integer, raise a ValueError with the message n must be a positive integer
* If x is not an integer that is greater than or equal to 0, raise a ValueError with the message x must be an integer that is greater than or equal to 0
* If x is greater than n, raise a ValueError with the message x cannot be greater than n
* If P is not a 1D numpy.ndarray, raise a TypeError with the message P must be a 1D numpy.ndarray
* If Pr is not a numpy.ndarray with the same shape as P, raise a TypeError with the message Pr must be a numpy.ndarray with the same shape as P
* If any value in P or Pr is not in the range [0, 1], raise a ValueError with the message All values in {P} must be in the range [0, 1] where {P} is the incorrect variable
* If Pr does not sum to 1, raise a ValueError with the message Pr must sum to 1
* All exceptions should be raised in the above order
* Returns: the marginal probability of obtaining x and n
```
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    marginal = __import__('2-marginal').marginal

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(marginal(26, 130, P, Pr))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./2-main.py 
0.008229580791426582
alexa@ubuntu-xenial:0x07-bayesian_prob$
```
File: 2-marginal.py

## 3. Posterior
Based on `2-marginal.py`, write a function `def posterior(x, n, P, Pr):` that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data:

* x is the number of patients that develop severe side effects
* n is the total number of patients observed
* P is a 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects
* Pr is a 1D numpy.ndarray containing the prior beliefs of P
* If n is not a positive integer, raise a ValueError with the message n must be a positive integer
* If x is not an integer that is greater than or equal to 0, raise a ValueError with the message x must be an integer that is greater than or equal to 0
* If x is greater than n, raise a ValueError with the message x cannot be greater than n
* If P is not a 1D numpy.ndarray, raise a TypeError with the message P must be a 1D numpy.ndarray
* If Pr is not a numpy.ndarray with the same shape as P, raise a TypeError with the message Pr must be a numpy.ndarray with the same shape as P
* If any value in P or Pr is not in the range [0, 1], raise a ValueError with the message All values in {P} must be in the range [0, 1] where {P} is the incorrect variable
* If Pr does not sum to 1, raise a ValueError with the message Pr must sum to 1
* All exceptions should be raised in the above order
* Returns: the posterior probability of each probability in P given x and n, respectively
```
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    posterior = __import__('3-posterior').posterior

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(posterior(26, 130, P, Pr))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./3-main.py 
[0.00000000e+00 2.99729127e-03 9.63044824e-01 3.39513268e-02
 6.55839819e-06 1.26359684e-11 1.20692303e-19 6.74011797e-31
 1.05430721e-47 1.11125368e-77 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$
```
File: [3-posterior.py]