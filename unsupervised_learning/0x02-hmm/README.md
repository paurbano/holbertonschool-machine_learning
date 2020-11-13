# 0x02. Hidden Markov Models

## Learning Objectives
* What is the Markov property?
* What is a Markov chain?
* What is a state?
* What is a transition probability/matrix?
* What is a stationary state?
* What is a regular Markov chain?
* How to determine if a transition matrix is regular
* What is an absorbing state?
* What is a transient state?
* What is a recurrent state?
* What is an absorbing Markov chain?
* What is a Hidden Markov Model?
* What is a hidden state?
* What is an observation?
* What is an emission probability/matrix?
* What is a Trellis diagram?
* What is the Forward algorithm and how do you implement it?
* What is decoding?
* What is the Viterbi algorithm and how do you implement it?
* What is the Forward-Backward algorithm and how do you implement it?
* What is the Baum-Welch algorithm and how do you implement it?

## 0. Markov Chain
Write the function `def markov_chain(P, s, t=1):` that determines the probability of a markov chain being in a particular state after a specified number of iterations:

* `P` is a square 2D `numpy.ndarray` of shape `(n, n)` representing the transition matrix
    * `P[i, j]` is the probability of transitioning from state `i` to state `j`
    * `n` is the number of states in the markov chain
* `s` is a `numpy.ndarray` of shape `(1, n)` representing the probability of starting in each state
* `t` is the number of iterations that the markov chain has been through
* Returns: a `numpy.ndarray` of shape `(1, n)` representing the probability of being in a specific state after `t` iterations, or `None` on failure
```
alexa@ubuntu-xenial:0x02-hmm$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
markov_chain = __import__('0-markov_chain').markov_chain

if __name__ == "__main__":
    P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3], [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
    s = np.array([[1, 0, 0, 0]])
    print(markov_chain(P, s, 300))
alexa@ubuntu-xenial:0x02-hmm$ ./0-main.py
[[0.2494929  0.26335362 0.23394185 0.25321163]]
alexa@ubuntu-xenial:0x02-hmm$
```
File: [0-markov_chain.py]

## 1. Regular Chains
Write the function `def regular(P):` that determines the steady state probabilities of a regular markov chain:

* P is a is a square 2D `numpy.ndarray` of shape `(n, n)` representing the transition matrix
    * `P[i, j]` is the probability of transitioning from state i to state j
    * `n` is the number of states in the markov chain
* Returns: a `numpy.ndarray` of shape `(1, n)` containing the steady state probabilities, or `None` on failure
```
alexa@ubuntu-xenial:0x02-hmm$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
regular = __import__('1-regular').regular

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[0.8, 0.2, 0, 0, 0],
                [0.25, 0.75, 0, 0, 0],
                [0, 0, 0.5, 0.2, 0.3],
                [0, 0, 0.3, 0.5, .2],
                [0, 0, 0.2, 0.3, 0.5]])
    e = np.array([[1, 0.25, 0, 0, 0],
                [0.25, 0.75, 0, 0, 0],
                [0, 0.1, 0.5, 0.2, 0.2],
                [0, 0.1, 0.2, 0.5, .2],
                [0, 0.1, 0.2, 0.2, 0.5]])
    print(regular(a))
    print(regular(b))
    print(regular(c))
    print(regular(d))
    print(regular(e))
alexa@ubuntu-xenial:0x02-hmm$ ./1-main.py
None
[[0.42857143 0.57142857]]
[[0.2494929  0.26335362 0.23394185 0.25321163]]
None
None
alexa@ubuntu-xenial:0x02-hmm$
```
File: [1-regular.py]

## 2. Absorbing Chains
Write the function `def absorbing(P):` that determines if a markov chain is absorbing:

* `P` is a is a square 2D `numpy.ndarray` of shape `(n, n)` representing the standard transition matrix
    * `P[i, j]` is the probability of transitioning from state `i` to state `j`
    * `n` is the number of states in the markov chain
* Returns: `True` if it is absorbing, or `False` on failure
```
alexa@ubuntu-xenial:0x02-hmm$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
absorbing = __import__('2-absorbing').absorbing

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[1, 0, 0, 0, 0],
                  [0.25, 0.75, 0, 0, 0],
                  [0, 0, 0.5, 0.2, 0.3],
                  [0, 0, 0.3, 0.5, .2],
                  [0, 0, 0.2, 0.3, 0.5]])
    e = np.array([[1, 0, 0, 0, 0],
                  [0.25, 0.75, 0, 0, 0],
                  [0, 0.1, 0.5, 0.2, 0.2],
                  [0, 0.1, 0.2, 0.5, .2],
                  [0, 0.1, 0.2, 0.2, 0.5]])
    print(absorbing(a))
    print(absorbing(b))
    print(absorbing(c))
    print(absorbing(d))
    print(absorbing(e))
alexa@ubuntu-xenial:0x02-hmm$ ./2-main.py
True
False
False
False
True
alexa@ubuntu-xenial:0x02-hmm$
```

File: [2-absorbing.py]

## 3. The Forward Algorithm
Write the function `def forward(Observation, Emission, Transition, Initial):` that performs the forward algorithm for a hidden markov model:

* `Observation` is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation
    * T is the number of observations
* `Emission` is a `numpy.ndarray` of shape `(N, M)` containing the emission probability of a specific observation given a hidden state
    * `Emission[i, j]` is the probability of observing j given the hidden state i
    * `N` is the number of hidden states
    * `M` is the number of all possible observations
* `Transition` is a 2D `numpy.ndarray` of shape `(N, N)` containing the transition probabilities
    * `Transition[i, j]` is the probability of transitioning from the hidden state `i` to `j`
* `Initial` a `numpy.ndarray` of shape `(N, 1)` containing the probability of starting in a particular hidden state
* Returns: `P`, `F`, or `None, None` on failure
    * `P` is the likelihood of the observations given the model
    * `F` is a `numpy.ndarray` of shape `(N, T)` containing the forward path probabilities
    * `F[i, j]` is the probability of being in hidden state `i` at time `j` given the previous observations

```
alexa@ubuntu-xenial:0x02-hmm$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np
forward = __import__('3-forward').forward

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, F = forward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(F)
alexa@ubuntu-xenial:0x02-hmm$ ./3-main.py
1.7080966131859584e-214
[[0.00000000e+000 0.00000000e+000 2.98125000e-004 ... 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [2.00000000e-002 0.00000000e+000 3.18000000e-003 ... 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [2.50000000e-001 3.31250000e-002 0.00000000e+000 ... 2.13885975e-214
  1.17844112e-214 0.00000000e+000]
 [1.00000000e-002 4.69000000e-002 0.00000000e+000 ... 2.41642482e-213
  1.27375484e-213 9.57568349e-215]
 [0.00000000e+000 8.00000000e-004 0.00000000e+000 ... 1.96973759e-214
  9.65573676e-215 7.50528264e-215]]
alexa@ubuntu-xenial:0x02-hmm$
```
File: [3-forward.py]

## 4. The Viretbi Algorithm
Write the function `def viterbi(Observation, Emission, Transition, Initial):` that calculates the most likely sequence of hidden states for a hidden markov model:
* `Observation` is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation
    * T is the number of observations
* `Emission` is a `numpy.ndarray` of shape `(N, M)` containing the emission probability of a specific observation given a hidden state
    * `Emission[i, j]` is the probability of observing j given the hidden state i
    * `N` is the number of hidden states
    * `M` is the number of all possible observations
* `Transition` is a 2D `numpy.ndarray` of shape `(N, N)` containing the transition probabilities
    * `Transition[i, j]` is the probability of transitioning from the hidden state `i` to `j`
* `Initial` a `numpy.ndarray` of shape `(N, 1)` containing the probability of starting in a particular hidden state
* Returns: `path`, `P`, or `None, None` on failure
    * `path` is the a list of length `T` containing the most likely sequence of hidden states
    * `P` is the probability of obtaining the `path` sequence
```
alexa@ubuntu-xenial:0x02-hmm$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
viterbi = __import__('4-viterbi').viterbi

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    path, P = viterbi(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(path)
alexa@ubuntu-xenial:0x02-hmm$ ./4-main.py
4.701733355108224e-252
[2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2, 3, 3, 3, 2, 1, 2, 1, 1, 2, 2, 2, 3, 3, 2, 2, 3, 4, 4, 3, 3, 2, 2, 3, 3, 3, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3, 3, 2, 1, 2, 1, 1, 1, 2, 2, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 2, 3, 4, 4, 4, 3, 2, 1, 0, 0, 0, 1, 2, 2, 1, 1, 2, 3, 3, 2, 1, 1, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 2, 1, 1, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3]
alexa@ubuntu-xenial:0x02-hmm$
```
File: [4-viterbi.py]

## 5. The Backward Algorithm
Write the function `def backward(Observation, Emission, Transition, Initial):` that performs the backward algorithm for a hidden markov model:
* `Observation` is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation
    * T is the number of observations
* `Emission` is a `numpy.ndarray` of shape `(N, M)` containing the emission probability of a specific observation given a hidden state
    * `Emission[i, j]` is the probability of observing j given the hidden state i
    * `N` is the number of hidden states
    * `M` is the number of all possible observations
* `Transition` is a 2D `numpy.ndarray` of shape `(N, N)` containing the transition probabilities
    * `Transition[i, j]` is the probability of transitioning from the hidden state `i` to `j`
* `Initial` a `numpy.ndarray` of shape `(N, 1)` containing the probability of starting in a particular hidden state
* Returns: `P`, `B`, or `None, None` on failure
    * `P` is the likelihood of the observations given the model
    * `B` is a `numpy.ndarray` of shape `(N, T)` containing the backward path probabilities
    * `B[i, j]` is the probability of generating the future observations from hidden state `i` at time `j`
```
alexa@ubuntu-xenial:0x02-hmm$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np
backward = __import__('5-backward').backward

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, B = backward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(B)
alexa@ubuntu-xenial:0x02-hmm$ ./5-main.py
1.7080966131859631e-214
[[1.28912952e-215 6.12087935e-212 1.00555701e-211 ... 6.75000000e-005
  0.00000000e+000 1.00000000e+000]
 [3.86738856e-214 2.69573528e-212 4.42866330e-212 ... 2.02500000e-003
  0.00000000e+000 1.00000000e+000]
 [6.44564760e-214 5.15651808e-213 8.47145100e-213 ... 2.31330000e-002
  2.70000000e-002 1.00000000e+000]
 [1.93369428e-214 0.00000000e+000 0.00000000e+000 ... 6.39325000e-002
  1.15000000e-001 1.00000000e+000]
 [1.28912952e-215 0.00000000e+000 0.00000000e+000 ... 5.77425000e-002
  2.19000000e-001 1.00000000e+000]]
alexa@ubuntu-xenial:0x02-hmm$
```

## 6. The Baum-Welch Algorithm
Write the function `def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):` that performs the Baum-Welch algorithm for a hidden markov model:
* `Observation` is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation
    * T is the number of observations
* `Transition` is a 2D `numpy.ndarray` of shape `(M, M)` containing the transition probabilities
    * * `M` is the number of hidden states
* `Emission` is a `numpy.ndarray` of shape `(M, N)` that contains the initialized emission probabilities
    * `N` is the number of output states
* `Initial` a `numpy.ndarray` of shape `(M, 1)` containing the probability of starting in a particular hidden state
* `iterations` is the number of times expectation-maximization should be performed
* Returns: the converged `Transition, Emission`, or `None, None` on failure
```
alexa@ubuntu-xenial:0x02-hmm$ cat 6-main.py
#!/usr/bin/env python3

import numpy as np
baum_welch = __import__('6-baum_welch').baum_welch

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00],
                         [0.40, 0.50, 0.10]])
    Transition = np.array([[0.60, 0.4],
                           [0.30, 0.70]])
    Initial = np.array([0.5, 0.5])
    Hidden = [np.random.choice(2, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(2, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(3, p=Emission[s]))
    Observations = np.array(Observations)
    T_test = np.ones((2, 2)) / 2
    E_test = np.abs(np.random.randn(2, 3))
    E_test = E_test / np.sum(E_test, axis=1).reshape((-1, 1))
    T, E = baum_welch(Observations, T_test, E_test, Initial.reshape((-1, 1)))
    print(np.round(T, 2))
    print(np.round(E, 2))
alexa@ubuntu-xenial:0x02-hmm$ ./6-main.py
[[0.81 0.19]
 [0.28 0.72]]
[[0.82 0.18 0.  ]
 [0.26 0.58 0.16]]
alexa@ubuntu-xenial:0x02-hmm$
```

File: [6-baum_welch.py]