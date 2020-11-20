<html>
<h1 class="gap">0x03. Hyperparameter Tuning</h1>

<h2>Resources</h2>
<ul>
<li><a href="/rltoken/LVmEm_zt83iEKEQ8D2_oaw" title="Hyperparameter Tuning in Practice" target="_blank">Hyperparameter Tuning in Practice</a></li>
<li><a href="/rltoken/1suHjfI2RmB7HGUkPe0qSA" title="Orthogonalization" target="_blank">Orthogonalization</a> </li>
<li><a href="/rltoken/mHDoDo0R2RgDsns-ho_B-Q" title="Single Number Evaluation Metric" target="_blank">Single Number Evaluation Metric</a> </li>
<li><a href="/rltoken/S-CqOTV5KvSz-6zgqVYSGQ" title="Satisficing and Optimizing Metrics" target="_blank">Satisficing and Optimizing Metrics</a> </li>
<li><a href="/rltoken/qJMsx3m-MecQGuHIiWmPVg" title="Gaussian process" target="_blank">Gaussian process</a></li>
<li><a href="/rltoken/jGdxmGdHXEATzO1ie3M2Zg" title="Kriging" target="_blank">Kriging</a></li>
<li><a href="/rltoken/k6HZ2Sg5pRuXPE05wmarfA" title="Machine learning - Introduction to Gaussian processes" target="_blank">Machine learning - Introduction to Gaussian processes</a> </li>
<li><a href="/rltoken/t3m8B0_XJTUYblxKW2qY-Q" title="Machine learning - Gaussian processes" target="_blank">Machine learning - Gaussian processes</a> </li>
<li><a href="/rltoken/cfPPx0YkYByAo1d-Yxb5oQ" title="Quick Start to Gaussian Process Regression" target="_blank">Quick Start to Gaussian Process Regression</a></li>
<li><a href="/rltoken/3Xgwc7ddcXcBoRaurvfKXA" title="Gaussian processes" target="_blank">Gaussian processes</a></li>
<li><a href="/rltoken/r_AicUZLRVINuXAEiR0ugA" title="Machine learning - Bayesian optimization and multi-armed bandits" target="_blank">Machine learning - Bayesian optimization and multi-armed bandits</a> </li>
<li><a href="/rltoken/3nJ8PFZXkdbX33nMrDuuTA" title="Bayesian Optimization" target="_blank">Bayesian Optimization</a></li>
<li><a href="/rltoken/ICAbvZAnezCius35JPtixg" title="Bayesian Optimization" target="_blank">Bayesian Optimization</a></li>
<li><a href="/rltoken/Zg6LyIfrtOr-RWTMrGXhXw" title="A Tutorial on Bayesian Optimization" target="_blank">A Tutorial on Bayesian Optimization</a></li>
<li><a href="/rltoken/dArkynGcqzwjsxahKI4arg" title="GPy documentation" target="_blank">GPy documentation</a>

<ul>
<li><a href="/rltoken/CVHw9tEMXCKL_GQx7YcpcA" title="GPy.kern.src" target="_blank">GPy.kern.src</a></li>
<li><a href="/rltoken/a4u5-JZkKxHIIlbMafkixw" title="GPy.plotting.gpy_plot" target="_blank">GPy.plotting.gpy_plot</a></li>
</ul></li>
<li><a href="/rltoken/MxJBvbwWsx4Mo833htW5DQ" title="GPyOpt documentation" target="_blank">GPyOpt documentation</a>

<ul>
<li><a href="/rltoken/Dw6-zZ3eol5cHYh4kxHWgA" title="GPyOpt.methods.bayesian_optimization" target="_blank">GPyOpt.methods.bayesian_optimization</a></li>
<li><a href="/rltoken/v-zTr1M0n1ZI5jG4to2uPg" title="GPyOpt.core.task.space" target="_blank">GPyOpt.core.task.space</a></li>
</ul></li>
</ul>
<h2>Learning Objectives</h2>
<ul>
<li>What is Hyperparameter Tuning?</li>
<li>What is random search? grid search?</li>
<li>What is a Gaussian Process?</li>
<li>What is a mean function?</li>
<li>What is a Kernel function?</li>
<li>What is Gaussian Process Regression/Kriging?</li>
<li>What is Bayesian Optimization?</li>
<li>What is an Acquisition function?</li>
<li>What is Expected Improvement?</li>
<li>What is Knowledge Gradient?</li>
<li>What is Entropy Search/Predictive Entropy Search?</li>
<li>What is GPy?</li>
<li>What is GPyOpt?</li>
</ul>
<h2>Install GPy and GPyOpt</h2>
<pre><code>pip install --user GPy
pip install --user gpyopt
</code></pre>
<h2 class="gap">Tasks</h2>
<section class="formatted-content">
            <div data-role="task4715" data-position="1">
              <div class=" clearfix gap" id="task-4715">
<span id="user_id" data-id="1283"></span>
</div>


    </div>

  <h4 class="task">
    0. Initialize Gaussian Process
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create the class <code>GaussianProcess</code> that represents a noiseless 1D Gaussian process:</p>

<ul>
<li><p>Class constructor: <code>def __init__(self, X_init, Y_init, l=1, sigma_f=1)</code>:</p>

<ul>
<li><code>X_init</code> is a <code>numpy.ndarray</code> of shape <code>(t, 1)</code> representing the inputs already sampled with the black-box function</li>
<li><code>Y_init</code> is a <code>numpy.ndarray</code> of shape <code>(t, 1)</code> representing the outputs of the black-box function for each input in <code>X_init</code></li>
<li><code>t</code> is the number of initial samples</li>
<li><code>l</code> is the length parameter for the kernel</li>
<li><code>sigma_f</code> is the standard deviation given to the output of the black-box function</li>
<li>Sets the public instance attributes <code>X</code>, <code>Y</code>, <code>l</code>, and <code>sigma_f</code> corresponding to the respective constructor inputs</li>
<li>Sets the public instance attribute <code>K</code>, representing the current covariance kernel matrix for the Gaussian process</li>
</ul></li>
<li><p>Public instance method <code>def kernel(self, X1, X2):</code> that calculates the covariance kernel matrix between two matrices:</p>

<ul>
<li><code>X1</code> is a <code>numpy.ndarray</code> of shape <code>(m, 1)</code></li>
<li><code>X2</code> is a <code>numpy.ndarray</code> of shape <code>(n, 1)</code></li>
<li>the kernel should use the Radial Basis Function (RBF)</li>
<li>Returns: the covariance kernel matrix as a <code>numpy.ndarray</code> of shape <code>(m, n)</code></li>
</ul></li>
</ul>

<pre><code>root@alexa-ml2-1:~/0x03-hyperparameter_opt# cat 0-main.py 
#!/usr/bin/env python3

GP = __import__('0-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    print(gp.X is X_init)
    print(gp.Y is Y_init)
    print(gp.l)
    print(gp.sigma_f)
    print(gp.K.shape, gp.K)
    print(np.allclose(gp.kernel(X_init, X_init), gp.K))
root@alexa-ml2-1:~/0x03-hyperparameter_opt# ./0-main.py 
True
True
0.6
2
(2, 2) [[4.         0.13150595]
 [0.13150595 4.        ]]
True
root@alexa-ml2-1:~/0x03-hyperparameter_opt# 
</code></pre>


</div>
<span id="user_id" data-id="1283"></span>
</div>


    </div>

  <h4 class="task">
    1. Gaussian Process Prediction
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Based on <code>0-gp.py</code>, update the class <code>GaussianProcess</code>:</p>

<ul>
<li>Public instance method <code>def predict(self, X_s):</code> that predicts the mean and standard deviation of points in a Gaussian process:

<ul>
<li><code>X_s</code> is a <code>numpy.ndarray</code> of shape <code>(s, 1)</code> containing all of the points whose mean and standard deviation should be calculated

<ul>
<li><code>s</code> is the number of sample points</li>
</ul></li>
<li>Returns: <code>mu, sigma</code>

<ul>
<li><code>mu</code> is a <code>numpy.ndarray</code> of shape <code>(s,)</code> containing the mean for each point in <code>X_s</code>, respectively</li>
<li><code>sigma</code> is a <code>numpy.ndarray</code> of shape <code>(s,)</code> containing the variance for each point in <code>X_s</code>, respectively</li>
</ul></li>
</ul></li>
</ul>

<pre><code>root@alexa-ml2-1:~/0x03-hyperparameter_opt# cat 1-main.py
#!/usr/bin/env python3

GP = __import__('1-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_s = np.random.uniform(-np.pi, 2*np.pi, (10, 1))
    mu, sig = gp.predict(X_s)
    print(mu.shape, mu)
    print(sig.shape, sig)
root@alexa-ml2-1:~/0x03-hyperparameter_opt# ./1-main.py
(10,) [ 0.20148983  0.93469135  0.14512328 -0.99831012  0.21779183 -0.05063668
 -0.00116747  0.03434981 -1.15092063  0.9221554 ]
(10,) [1.90890408 0.01512125 3.91606789 2.42958747 3.81083574 3.99817545
 3.99999903 3.9953012  3.05639472 0.37179608]
root@alexa-ml2-1:~/0x03-hyperparameter_opt# 
</code></pre>



</div>
<span id="user_id" data-id="1283"></span>
</div>


    </div>

  <h4 class="task">
    2. Update Gaussian Process
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Based on <code>1-gp.py</code>, update the class <code>GaussianProcess</code>:</p>

<ul>
<li>Public instance method <code>def update(self, X_new, Y_new):</code> that updates a Gaussian Process:

<ul>
<li><code>X_new</code> is a <code>numpy.ndarray</code> of shape <code>(1,)</code> that represents the new sample point</li>
<li><code>Y_new</code> is a <code>numpy.ndarray</code> of shape <code>(1,)</code> that represents the new sample function value</li>
<li>Updates the public instance attributes <code>X</code>, <code>Y</code>, and <code>K</code></li>
</ul></li>
</ul>

<pre><code>root@alexa-ml2-1:~/0x03-hyperparameter_opt# cat 2-main.py
#!/usr/bin/env python3

GP = __import__('2-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_new = np.random.uniform(-np.pi, 2*np.pi, 1)
    print('X_new:', X_new)
    Y_new = f(X_new)
    print('Y_new:', Y_new)
    gp.update(X_new, Y_new)
    print(gp.X.shape, gp.X)
    print(gp.Y.shape, gp.Y)
    print(gp.K.shape, gp.K)
root@alexa-ml2-1:~/0x03-hyperparameter_opt# ./2-main.py
X_new: [2.53931833]
Y_new: [1.99720866]
(3, 1) [[2.03085276]
 [3.59890832]
 [2.53931833]]
(3, 1) [[ 0.92485357]
 [-2.33925576]
 [ 1.99720866]]
(3, 3) [[4.         0.13150595 2.79327536]
 [0.13150595 4.         0.84109203]
 [2.79327536 0.84109203 4.        ]]
root@alexa-ml2-1:~/0x03-hyperparameter_opt# 
</code></pre>

</div>
<span id="user_id" data-id="1283"></span>

</div>


    </div>

  <h4 class="task">
    3. Initialize Bayesian Optimization
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create the class <code>BayesianOptimization</code> that performs Bayesian optimization on a noiseless 1D Gaussian process: </p>

<ul>
<li>Class constructor <code>def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):</code>

<ul>
<li><code>f</code> is the black-box function to be optimized</li>
<li><code>X_init</code> is a <code>numpy.ndarray</code> of shape <code>(t, 1)</code> representing the inputs already sampled with the black-box function</li>
<li><code>Y_init</code> is a <code>numpy.ndarray</code> of shape <code>(t, 1)</code> representing the outputs of the black-box function for each input in <code>X_init</code></li>
<li><code>t</code> is the number of initial samples</li>
<li><code>bounds</code> is a tuple of <code>(min, max)</code> representing the bounds of the space in which to look for the optimal point</li>
<li><code>ac_samples</code> is the number of samples that should be analyzed during acquisition</li>
<li><code>l</code> is the length parameter for the kernel</li>
<li><code>sigma_f</code> is the standard deviation given to the output of the black-box function</li>
<li><code>xsi</code> is the exploration-exploitation factor for acquisition</li>
<li><code>minimize</code> is a <code>bool</code> determining whether optimization should be performed for minimization (<code>True</code>) or maximization (<code>False</code>)</li>
<li>Sets the following public instance attributes:

<ul>
<li><code>f</code>: the black-box function</li>
<li><code>gp</code>: an instance of the class <code>GaussianProcess</code></li>
<li><code>X_s</code>: a <code>numpy.ndarray</code> of shape <code>(ac_samples, 1)</code> containing all acquisition sample points, evenly spaced between <code>min</code> and <code>max</code></li>
<li><code>xsi</code>: the exploration-exploitation factor</li>
<li><code>minimize</code>: a <code>bool</code> for minimization versus maximization</li>
</ul></li>
</ul></li>
<li>You may use <code>GP = __import__('2-gp').GaussianProcess</code></li>
</ul>

<pre><code>root@alexa-ml2-1:~/0x03-hyperparameter_opt# cat 3-main.py 
#!/usr/bin/env python3

GP = __import__('2-gp').GaussianProcess
BO = __import__('4-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=2, sigma_f=3, xsi=0.05)
    print(bo.f is f)
    print(type(bo.gp) is GP)
    print(bo.gp.X is X_init)
    print(bo.gp.Y is Y_init)
    print(bo.gp.l)
    print(bo.gp.sigma_f)
    print(bo.X_s.shape, bo.X_s)
    print(bo.xsi)
    print(bo.minimize)
root@alexa-ml2-1:~/0x03-hyperparameter_opt# ./3-main.py 
True
True
True
True
2
3
(50, 1) [[-3.14159265]
 [-2.94925025]
 [-2.75690784]
 [-2.56456543]
 [-2.37222302]
 [-2.17988062]
 [-1.98753821]
 [-1.7951958 ]
 [-1.60285339]
 [-1.41051099]
 [-1.21816858]
 [-1.02582617]
 [-0.83348377]
 [-0.64114136]
 [-0.44879895]
 [-0.25645654]
 [-0.06411414]
 [ 0.12822827]
 [ 0.32057068]
 [ 0.51291309]
 [ 0.70525549]
 [ 0.8975979 ]
 [ 1.08994031]
 [ 1.28228272]
 [ 1.47462512]
 [ 1.66696753]
 [ 1.85930994]
 [ 2.05165235]
 [ 2.24399475]
 [ 2.43633716]
 [ 2.62867957]
 [ 2.82102197]
 [ 3.01336438]
 [ 3.20570679]
 [ 3.3980492 ]
 [ 3.5903916 ]
 [ 3.78273401]
 [ 3.97507642]
 [ 4.16741883]
 [ 4.35976123]
 [ 4.55210364]
 [ 4.74444605]
 [ 4.93678846]
 [ 5.12913086]
 [ 5.32147327]
 [ 5.51381568]
 [ 5.70615809]
 [ 5.89850049]
 [ 6.0908429 ]
 [ 6.28318531]]
0.05
True
root@alexa-ml2-1:~/0x03-hyperparameter_opt# 
</code></pre>

</div>
<span id="user_id" data-id="1283"></span>
</div>


    </div>

  <h4 class="task">
    4. Bayesian Optimization - Acquisition
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Based on <code>3-bayes_opt.py</code>, update the class <code>BayesianOptimization</code>:</p>

<ul>
<li>Public instance method <code>def acquisition(self):</code> that calculates the next best sample location:

<ul>
<li>Uses the Expected Improvement acquisition function</li>
<li>Returns: <code>X_next, EI</code>

<ul>
<li><code>X_next</code> is a <code>numpy.ndarray</code> of shape <code>(1,)</code> representing the next best sample point</li>
<li><code>EI</code> is a <code>numpy.ndarray</code> of shape <code>(ac_samples,)</code> containing the expected improvement of each potential sample</li>
</ul></li>
</ul></li>
<li>You may use <code>from scipy.stats import norm</code></li>
</ul>

<pre><code>root@alexa-ml2-1:~/0x03-hyperparameter_opt# cat 4-main.py
#!/usr/bin/env python3

BO = __import__('4-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2, xsi=0.05)
    X_next, EI = bo.acquisition()

    print(EI)
    print(X_next)

    plt.scatter(X_init.reshape(-1), Y_init.reshape(-1), color='g')
    plt.plot(bo.X_s.reshape(-1), EI.reshape(-1), color='r')
    plt.axvline(x=X_next)
    plt.show()
root@alexa-ml2-1:~/0x03-hyperparameter_opt# ./4-main.py 
[6.77642382e-01 6.77642382e-01 6.77642382e-01 6.77642382e-01
 6.77642382e-01 6.77642382e-01 6.77642382e-01 6.77642382e-01
 6.77642379e-01 6.77642362e-01 6.77642264e-01 6.77641744e-01
 6.77639277e-01 6.77628755e-01 6.77588381e-01 6.77448973e-01
 6.77014261e-01 6.75778547e-01 6.72513223e-01 6.64262238e-01
 6.43934968e-01 5.95940851e-01 4.93763541e-01 3.15415142e-01
 1.01026267e-01 1.73225936e-03 4.29042673e-28 0.00000000e+00
 4.54945116e-13 1.14549081e-02 1.74765619e-01 3.78063126e-01
 4.19729153e-01 2.79303426e-01 7.84942221e-02 0.00000000e+00
 8.33323492e-02 3.25320033e-01 5.70580150e-01 7.20239593e-01
 7.65975535e-01 7.52693111e-01 7.24099594e-01 7.01220863e-01
 6.87941196e-01 6.81608621e-01 6.79006118e-01 6.78063616e-01
 6.77759591e-01 6.77671794e-01]
[4.55210364]
</code></pre>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/3/5effc2d0e2e92ea16833.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201117%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20201117T155502Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=4a4c33c0e93c16ff544ca35206f2458c9612bf3aa6881604b124ea40ded9a0b0" alt="" style=""></p>

</div>
<span id="user_id" data-id="1283"></span>
</div>


    </div>

  <h4 class="task">
    5. Bayesian Optimization
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Based on <code>4-bayes_opt.py</code>, update the class <code>BayesianOptimization</code>:</p>

<ul>
<li>Public instance method <code>def optimize(self, iterations=100):</code> that optimizes the black-box function:

<ul>
<li><code>iterations</code> is the maximum number of iterations to perform</li>
<li>If the next proposed point is one that has already been sampled, optimization should be stopped early</li>
<li>Returns: <code>X_opt, Y_opt</code>

<ul>
<li><code>X_opt</code> is a <code>numpy.ndarray</code> of shape <code>(1,)</code> representing the optimal point</li>
<li><code>Y_opt</code> is a <code>numpy.ndarray</code> of shape <code>(1,)</code> representing the optimal function value</li>
</ul></li>
</ul></li>
</ul>

<pre><code>root@alexa-ml2-1:~/0x03-hyperparameter_opt# cat 5-main.py
#!/usr/bin/env python3

BO = __import__('5-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2)
    X_opt, Y_opt = bo.optimize(50)
    print('Optimal X:', X_opt)
    print('Optimal Y:', Y_opt)
    print('All sample inputs:', bo.gp.X)
root@alexa-ml2-1:~/0x03-hyperparameter_opt# ./5-main.py
Optimal X: [0.8975979]
Optimal Y: [-2.92478374]
All sample inputs: [[ 2.03085276]
 [ 3.59890832]
 [ 4.55210364]
 [ 5.89850049]
 [-3.14159265]
 [-0.83348377]
 [ 0.70525549]
 [-2.17988062]
 [ 3.01336438]
 [ 3.97507642]
 [ 1.28228272]
 [ 5.12913086]
 [ 0.12822827]
 [ 6.28318531]
 [-1.60285339]
 [-2.75690784]
 [-2.56456543]
 [ 0.8975979 ]
 [ 2.43633716]
 [-0.44879895]]
root@alexa-ml2-1:~/0x03-hyperparameter_opt# 
</code></pre>
</div>
<span id="user_id" data-id="1283"></span>

</div>


    </div>

  <h4 class="task">
    6. Bayesian Optimization with GPyOpt
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a python script that optimizes a machine learning model of your choice using <code>GPyOpt</code>:</p>

<ul>
<li>Your script should optimize at least 5 different hyperparameters. E.g. learning rate, number of units in a layer, dropout rate, L2 regularization weight, batch size</li>
<li>Your model should be optimized on a single satisficing metric</li>
<li>Your model should save a checkpoint of its best iteration during each training session

<ul>
<li>The filename of the checkpoint should specify the values of the hyperparameters being tuned</li>
</ul></li>
<li>Your model should perform early stopping</li>
<li>Bayesian optimization should run for a maximum of 30 iterations</li>
<li>Once optimization has been performed, your script should plot the convergence</li>
<li>Your script should save a report of the optimization to the file <code>'bayes_opt.txt'</code></li>
<li>There are no restrictions on imports</li>
</ul>

<p>Once you have finished your script, write a blog post describing your approach to this task. Your blog post should include:</p>

<ul>
<li>A description of what a Gaussian Process is</li>
<li>A description of Bayesian Optimization</li>
<li>The particular model that you chose to optimize</li>
<li>The reasons you chose to focus on your specific hyperparameters</li>
<li>The reason you chose your satisficing matric</li>
<li>Your reasoning behind any other approach choices</li>
<li>Any conclusions you made from performing this optimization</li>
<li>Final thoughts</li>
</ul>

<p>Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.</p>

<p>When done, please add all URLs below (blog post, tweet, etc.)</p>

<p>Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.</p>

</div>
 
      </section>
</html>