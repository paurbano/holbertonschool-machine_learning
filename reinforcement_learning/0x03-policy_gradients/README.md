<h1 class="gap">0x03. Policy Gradients</h1>
<p>In this project, you will implement your own Policy Gradient in your loop of reinforcement learning (by using the Monte-Carlo policy gradient algorithm - also called <code>REINFORCE</code>).</p>

<h2>Resources</h2>
<ul>
<li><a href="/rltoken/hfQFTtnxkkdO7AjwJvJb6Q" title="How Policy Gradient Reinforcement Learning Works" target="_blank">How Policy Gradient Reinforcement Learning Works</a></li>
<li><a href="/rltoken/bM4ElarNtJMNI71i2bkPNw" title="Policy Gradients in a Nutshell" target="_blank">Policy Gradients in a Nutshell</a></li>
<li><a href="/rltoken/Ehf_ISuQx-hUUB21P4uMvQ" title="RL Course by David Silver - Lecture 7: Policy Gradient Methods" target="_blank">RL Course by David Silver - Lecture 7: Policy Gradient Methods</a></li>
<li><a href="/rltoken/wxP1EioedlosWi-op63zLA" title="Reinforcement Learning 6: Policy Gradients and Actor Critics" target="_blank">Reinforcement Learning 6: Policy Gradients and Actor Critics</a></li>
<li><a href="/rltoken/EiARIynXiIJXqw9P8o0jtg" title="Policy Gradient Algorithms" target="_blank">Policy Gradient Algorithms</a></li>
</ul>

<h2 class="gap">Tasks</h2>

<div data-role="task7539" data-position="1" id="task-num-0">
        <div class="panel panel-default task-card " id="task-7539">
  <span id="user_id" data-id="1283"></span>

  <div class="panel-heading">
    <h3 class="panel-title">
      0. Simple Policy function
    </h3>


  </div>

  <div class="panel-body">
    <span id="user_id" data-id="1283"></span>

    

<!-- Progress vs Score -->

<!-- Task Body -->
<p>Write a function that computes to policy with a weight of a matrix.</p>

<ul>
<li>Prototype: <code>def policy(matrix, weight):</code></li>
</ul>

<pre><code>$ cat 0-main.py
#!/usr/bin/env python3
"""
Main file
"""
import numpy as np
from policy_gradient import policy


weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01], 
    [1.14374817e-04, 3.02332573e-01], 
    [1.46755891e-01, 9.23385948e-02], 
    [1.86260211e-01, 3.45560727e-01]
    ]))
state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214,  0.01636746,  0.01196594, -0.03095031]
    ]))

res = policy(state, weight)
print(res)

$
$ ./0-main.py
[[0.50351642 0.49648358]]
$
</code></pre>

  </div>

  <div class="list-group">
    <!-- Task URLs -->

<!-- Github information -->
<div class="list-group-item">
<p><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x03-policy_gradients</code></li>
    <li>File: <code>policy_gradient.py</code></li>
</ul>
</div>
  </div>

  <div class="panel-footer">
      
<div>
</div>
</div>
</div>
</div>

</div>

<!--1 -->
<div data-role="task7540" data-position="2" id="task-num-1">
        <div class="panel panel-default task-card " id="task-7540">
  <span id="user_id" data-id="1283"></span>

  <div class="panel-heading">
    <h3 class="panel-title">
      1. Compute the Monte-Carlo policy gradient
    </h3>
  </div>

  <div class="panel-body">
    <span id="user_id" data-id="1283"></span>
<!-- Progress vs Score -->

<!-- Task Body -->
<p>By using the previous function created <code>policy</code>, write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix.</p>

<ul>
<li>Prototype: <code>def policy_gradient(state, weight):</code>

<ul>
<li><code>state</code>: matrix representing the current observation of the environment</li>
<li><code>weight</code>: matrix of random weight</li>
</ul></li>
<li>Return: the action and the gradient (in this order)</li>
</ul>

<pre><code>$ cat 1-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import numpy as np
from policy_gradient import policy_gradient

env = gym.make('CartPole-v1')
np.random.seed(1)

weight = np.random.rand(4, 2)
state = env.reset()[None,:]
print(weight)
print(state)

action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()

$ 
$ ./1-main.py
[[4.17022005e-01 7.20324493e-01]
 [1.14374817e-04 3.02332573e-01]
 [1.46755891e-01 9.23385948e-02]
 [1.86260211e-01 3.45560727e-01]]
[[ 0.04228739 -0.04522399  0.01190918 -0.03496226]]
0
[[ 0.02106907 -0.02106907]
 [-0.02253219  0.02253219]
 [ 0.00593357 -0.00593357]
 [-0.01741943  0.01741943]]
$ 
</code></pre>

<p>*Results can be different since <code>weight</code> is randomized *</p>

  </div>

  <div class="list-group">
    <!-- Task URLs -->

<!-- Github information -->
<div class="list-group-item">
<p><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x03-policy_gradients</code></li>
    <li>File: <code>policy_gradient.py</code></li>
</ul>
</div>
  </div>

  <div class="panel-footer">
      
<div>
</div>
</div>

</div>
</div>

</div>

<!--2 -->
<div data-role="task7541" data-position="3" id="task-num-2">
        <div class="panel panel-default task-card " id="task-7541">
  <span id="user_id" data-id="1283"></span>

  <div class="panel-heading">
    <h3 class="panel-title">
      2. Implement the training
    </h3>
  </div>

  <div class="panel-body">
    <span id="user_id" data-id="1283"></span>

    

<!-- Progress vs Score -->

<!-- Task Body -->
<p>By using the previous function created <code>policy_gradient</code>, write a function that implements a full training.</p>

<ul>
<li>Prototype: <code>def train(env, nb_episodes, alpha=0.000045, gamma=0.98):</code>

<ul>
<li><code>env</code>: initial environment</li>
<li><code>nb_episodes</code>: number of episodes used for training</li>
<li><code>alpha</code>: the learning rate</li>
<li><code>gamma</code>: the discount factor</li>
</ul></li>
<li>Return: all values of the score (sum of all rewards during one episode loop)</li>
</ul>

<p>Since the training is quite long, please print the current episode number and the score after each loop. To display these information on the same line, you can use <code>end="\r", flush=False</code> of the print function.</p>

<p>With the following main file, you should have this result plotted:</p>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/12/e2fff0551f5173b824a8ee1b2e67aff72d7309e2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210311%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210311T172848Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=9f5faffd4ad7ed3060ce69f91ff4a622b9798fc7d442b919281350285607adb4" alt="" style=""></p>

<pre><code>$ cat 2-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym
import matplotlib.pyplot as plt
import numpy as np

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000)

plt.plot(np.arange(len(scores)), scores)
plt.show()
env.close()

$ 
$ ./2-main.py
</code></pre>

<p><em>Results can be different we have multiple randomization</em></p>

<p>Also, we highly encourage you to play with <code>alpha</code> and <code>gamma</code> to change the trend of the plot</p>

  </div>

  <div class="list-group">
    <!-- Task URLs -->

<!-- Github information -->
<div class="list-group-item">
<p><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x03-policy_gradients</code></li>
    <li>File: <code>train.py</code></li>
</ul>
</div>
  </div>

  <div class="panel-footer">
      
<div>
</div>


</div>

</div>
</div>

</div>

<!--3 -->
<div data-role="task7542" data-position="4" id="task-num-3">
        <div class="panel panel-default task-card " id="task-7542">
  <span id="user_id" data-id="1283"></span>

  <div class="panel-heading">
    <h3 class="panel-title">
      3. Animate iteration
    </h3>
  </div>

  <div class="panel-body">
    <span id="user_id" data-id="1283"></span>

    

<!-- Progress vs Score -->

<!-- Task Body -->
<p>Update the prototype of the <code>train</code> function by adding a last optional parameter <code>show_result</code> (default: <code>False</code>).</p>

<p>When this parameter is <code>True</code>, render the environment every 1000 episodes computed.</p>

<pre><code>$ cat 3-main.py
#!/usr/bin/env python3
"""
Main file
"""
import gym

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000, 0.000045, 0.98, True)

env.close()

$ 
$ ./3-main.py
</code></pre>

<p><em>Results can be different we have multiple randomization</em></p>

<p><strong>Result after few episodes:</strong></p>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/12/51a3d986d9c96960ddd0c009f7eaac5a2ce9f549.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210311%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210311T172848Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=e629585d9e7ad3da8d8f546112f02daecf1be6ed9ad640cfe09a82448395b91e" alt="" style=""></p>

<p><strong>Result after more episodes:</strong></p>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/12/8dadd3f7918aa188cde1b5c6ac2aafddac8a081f.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210311%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210311T172848Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=e3fffaa012e1bee3fc453f8da226bbfe9ad45883dcc91d528cef0133dab07673" alt="" style=""></p>

<p><strong>Result after 10000 episodes:</strong></p>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/12/da9d7deed16c5c9aec05e26bf14cf8b76e70dcce.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210311%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210311T172849Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=8da7eab1c488ad5e9597011eef4c7b478355a9a4fcc4cce5829049c2e1bda3e0" alt="" style=""></p>

  </div>

  <div class="list-group">
    <!-- Task URLs -->

<!-- Github information -->
<div class="list-group-item">
<p><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x03-policy_gradients</code></li>
    <li>File: <code>train.py</code></li>
</ul>
</div>
  </div>

  <div class="panel-footer">
      
<div>
</div>
</div>



  </div>
</div>

</div>