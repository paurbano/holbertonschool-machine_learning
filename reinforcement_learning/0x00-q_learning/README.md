<h1 class="gap">0x00. Q-learning</h1>

<h2>Resources</h2>

<ul>
<li><a href="/rltoken/uSJcrn4-wamVCfbQQtI9EA" title="An introduction to Reinforcement Learning" target="_blank">An introduction to Reinforcement Learning</a></li>
<li><a href="/rltoken/u3DP9_6-G97oU8eDjDBvIg" title="Simple Reinforcement Learning: Q-learning" target="_blank">Simple Reinforcement Learning: Q-learning</a></li>
<li><a href="/rltoken/km2Nyp6zyAast1k5v9P_wQ" title="Markov Decision Processes (MDPs) - Structuring a Reinforcement Learning Problem" target="_blank">Markov Decision Processes (MDPs) - Structuring a Reinforcement Learning Problem</a></li>
<li><a href="/rltoken/mM6iGVu8uSr7siZJCM-D-Q" title="Expected Return - What Drives a Reinforcement Learning Agent in an MDP" target="_blank">Expected Return - What Drives a Reinforcement Learning Agent in an MDP</a></li>
<li><a href="/rltoken/HgOMxHB7SipUwDk6s3ZhUA" title="Policies and Value Functions - Good Actions for a Reinforcement Learning Agent" target="_blank">Policies and Value Functions - Good Actions for a Reinforcement Learning Agent</a></li>
<li><a href="/rltoken/Pd4kGKXr9Pd0qQ4RO93Xww" title="What do Reinforcement Learning Algorithms Learn - Optimal Policies" target="_blank">What do Reinforcement Learning Algorithms Learn - Optimal Policies</a></li>
<li><a href="/rltoken/vj2E0Jizi5qUKn6hLUnVSQ" title="Q-Learning Explained - A Reinforcement Learning Technique" target="_blank">Q-Learning Explained - A Reinforcement Learning Technique</a></li>
<li><a href="/rltoken/zQNxN36--R7hzP0ktiKOsg" title="Exploration vs. Exploitation - Learning the Optimal Reinforcement Learning Policy" target="_blank">Exploration vs. Exploitation - Learning the Optimal Reinforcement Learning Policy</a></li>
<li><a href="/rltoken/GMcf0lCJ-SlaF6FSUKaozA" title="OpenAI Gym and Python for Q-learning - Reinforcement Learning Code Project" target="_blank">OpenAI Gym and Python for Q-learning - Reinforcement Learning Code Project</a></li>
<li><a href="/rltoken/GE2nKBHgehHdd_XN7lK0Gw" title="Train Q-learning Agent with Python - Reinforcement Learning Code Project" target="_blank">Train Q-learning Agent with Python - Reinforcement Learning Code Project</a></li>
<li><a href="/rltoken/Dz37ih49PpmrJicq_IP3aA" title="Markov Decision Processes" target="_blank">Markov Decision Processes</a></li>
</ul>

<h2>Learning Objectives</h2>
<ul>
<li>What is a Markov Decision Process?</li>
<li>What is an environment?</li>
<li>What is an agent?</li>
<li>What is a state?</li>
<li>What is a policy function?</li>
<li>What is a value function? a state-value function? an action-value function?</li>
<li>What is a discount factor?</li>
<li>What is the Bellman equation?</li>
<li>What is epsilon greedy?</li>
<li>What is Q-learning?</li>
</ul>
<h2>Installing OpenAI’s Gym</h2>
<pre><code>pip install --user gym</code></pre>

<h2 class="gap">Tasks</h2>

<section class="formatted-content">
            <div data-role="task6249" data-position="1">
              <div class=" clearfix gap" id="task-6249">
<span id="user_id" data-id="1283"></span>

</div>


</div>

  <h4 class="task">
    0. Load the Environment
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def load_frozen_lake(desc=None, map_name=None, is_slippery=False):</code> that loads the pre-made <code>FrozenLakeEnv</code> evnironment from OpenAI’s <code>gym</code>:</p>

<ul>
<li><code>desc</code> is either <code>None</code> or a list of lists containing a custom description of the map to load for the environment</li>
<li><code>map_name</code> is either <code>None</code> or a string containing the pre-made map to load</li>
<li><em>Note: If both <code>desc</code> and <code>map_name</code> are <code>None</code>, the environment will load a randomly generated 8x8 map</em></li>
<li><code>is_slippery</code> is a boolean to determine if the ice is slippery</li>
<li>Returns: the environment</li>
</ul>

<pre><code>$ cat 0-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
import numpy as np

np.random.seed(0)
env = load_frozen_lake()
print(env.desc)
print(env.P[0][0])
env = load_frozen_lake(is_slippery=True)
print(env.desc)
print(env.P[0][0])
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.desc)
env = load_frozen_lake(map_name='4x4')
print(env.desc)
$ ./0-main.py
[[b'S' b'F' b'F' b'F' b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'F' b'F' b'H' b'F' b'F']
 [b'F' b'H' b'F' b'H' b'H' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'H' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'H' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'H' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'G']]
[(1.0, 0, 0.0, False)]
[[b'S' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
 [b'H' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'H']
 [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'H']
 [b'F' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'G']]
[(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 8, 0.0, True)]
[[b'S' b'F' b'F']
 [b'F' b'H' b'H']
 [b'F' b'F' b'G']]
[[b'S' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'H']
 [b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'G']]
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x00-q_learning</code></li>
    <li>File: <code>0-load_env.py</code></li>
</ul>

</div>

</div>
<div data-role="task6250" data-position="2">
    <div class=" clearfix gap" id="task-6250">
<span id="user_id" data-id="1283"></span>

</div>


</div>

  <h4 class="task">
    1. Initialize Q-table
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def q_init(env):</code> that initializes the Q-table:</p>

<ul>
<li><code>env</code> is the <code>FrozenLakeEnv</code> instance</li>
<li>Returns: the Q-table as a <code>numpy.ndarray</code> of zeros</li>
</ul>

<pre><code>$ cat 1-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init

env = load_frozen_lake()
Q = q_init(env)
print(Q.shape)
env = load_frozen_lake(is_slippery=True)
Q = q_init(env)
print(Q.shape)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)
print(Q.shape)
env = load_frozen_lake(map_name='4x4')
Q = q_init(env)
print(Q.shape)
$ ./1-main.py
(64, 4)
(64, 4)
(9, 4)
(16, 4)
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x00-q_learning</code></li>
    <li>File: <code>1-q_init.py</code></li>
</ul>

</div>

</div>
<div data-role="task6251" data-position="3">
    <div class=" clearfix gap" id="task-6251">
<span id="user_id" data-id="1283"></span>


</div>


</div>

  <h4 class="task">
    2. Epsilon Greedy
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def epsilon_greedy(Q, state, epsilon):</code> that uses epsilon-greedy to determine the next action:</p>

<ul>
<li><code>Q</code> is a <code>numpy.ndarray</code> containing the q-table</li>
<li><code>state</code> is the current state</li>
<li><code>epsilon</code> is the epsilon to use for the calculation</li>
<li>You should sample <code>p</code> with <code>numpy.random.uniformn</code> to determine if your algorithm should explore or exploit</li>
<li>If exploring, you should pick the next action with <code>numpy.random.randint</code> from all possible actions</li>
<li>Returns: the next action index</li>
</ul>

<pre><code>$ cat 2-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
import numpy as np

desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)
Q[7] = np.array([0.5, 0.7, 1, -1])
np.random.seed(0)
print(epsilon_greedy(Q, 7, 0.5))
np.random.seed(1)
print(epsilon_greedy(Q, 7, 0.5))
$ ./2-main.py
2
0
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x00-q_learning</code></li>
    <li>File: <code>2-epsilon_greedy.py</code></li>
</ul>

</div>

</div>
<div data-role="task6252" data-position="4">
    <div class=" clearfix gap" id="task-6252">
<span id="user_id" data-id="1283"></span>

</div>


</div>

  <h4 class="task">
    3. Q-learning
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write the function <code>def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):</code> that performs Q-learning:</p>

<ul>
<li><code>env</code> is the <code>FrozenLakeEnv</code> instance</li>
<li><code>Q</code> is a <code>numpy.ndarray</code> containing the Q-table</li>
<li><code>episodes</code> is the total number of episodes to train over</li>
<li><code>max_steps</code> is the maximum number of steps per episode</li>
<li><code>alpha</code> is the learning rate</li>
<li><code>gamma</code> is the discount rate</li>
<li><code>epsilon</code> is the initial threshold for epsilon greedy</li>
<li><code>min_epsilon</code> is the minimum value that <code>epsilon</code> should decay to</li>
<li><code>epsilon_decay</code> is the decay rate for updating <code>epsilon</code> between episodes</li>
<li>When the agent falls in a hole, the reward should be updated to be <code>-1</code></li>
<li>Returns: <code>Q, total_rewards</code>

<ul>
<li><code>Q</code> is the updated Q-table</li>
<li><code>total_rewards</code> is a list containing the rewards per episode</li>
</ul></li>
</ul>

<pre><code>$ cat 3-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, total_rewards  = train(env, Q)
print(Q)
split_rewards = np.split(np.array(total_rewards), 10)
for i, rewards in enumerate(split_rewards):
    print((i+1) * 500, ':', np.mean(rewards))
$ ./3-main.py
[[ 0.96059593  0.970299    0.95098488  0.96059396]
 [ 0.96059557 -0.77123208  0.0094072   0.37627228]
 [ 0.18061285 -0.1         0.          0.        ]
 [ 0.97029877  0.9801     -0.99999988  0.96059583]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.98009763  0.98009933  0.99        0.9702983 ]
 [ 0.98009922  0.98999782  1.         -0.99999952]
 [ 0.          0.          0.          0.        ]]
500 : 0.812
1000 : 0.88
1500 : 0.9
2000 : 0.9
2500 : 0.88
3000 : 0.844
3500 : 0.892
4000 : 0.896
4500 : 0.852
5000 : 0.928
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x00-q_learning</code></li>
    <li>File: <code>3-q_learning.py</code></li>
</ul>

</div>

</div>
<div data-role="task6253" data-position="5">
    <div class=" clearfix gap" id="task-6253">
<span id="user_id" data-id="1283"></span>
</div>


</div>

  <h4 class="task">
    4. Play
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def play(env, Q, max_steps=100):</code> that has the trained agent play an episode:</p>

<ul>
<li><code>env</code> is the <code>FrozenLakeEnv</code> instance</li>
<li><code>Q</code> is a <code>numpy.ndarray</code> containing the Q-table</li>
<li><code>max_steps</code> is the maximum number of steps in the episode</li>
<li>Each state of the board should be displayed via the console</li>
<li>You should always exploit the Q-table</li>
<li>Returns: the total rewards for the episode</li>
</ul>

<pre><code>$ cat 4-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
play = __import__('4-play').play

import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, total_rewards  = train(env, Q)
print(play(env, Q))
$ ./4-main.py

`S`FF
FHH
FFG
  (Down)
SFF
`F`HH
FFG
  (Down)
SFF
FHH
`F`FG
  (Right)
SFF
FHH
F`F`G
  (Right)
SFF
FHH
FF`G`
1.0
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x00-q_learning</code></li>
    <li>File: <code>4-play.py</code></li>
</ul>

</div>

</div>
</section>