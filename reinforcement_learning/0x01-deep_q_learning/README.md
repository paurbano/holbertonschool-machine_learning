<h1 class="gap">0x01. Deep Q-learning</h1>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/8/9239a27ccd609cb9092aba0e6bb55ba7b5cf0b6b.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210220%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210220T220532Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=eb26937088f0f5500eec225ae53724538b2036cdb72a30c7ccaad842e3859a45" alt="" style=""></p>

<h2>Resources</h2>
<ul>
<li><a href="/rltoken/vf8M2yFL9vWcFftBWFG2KQ" title="Deep Q-Learning - Combining Neural Networks and Reinforcement Learning" target="_blank">Deep Q-Learning - Combining Neural Networks and Reinforcement Learning</a></li>
<li><a href="/rltoken/LciKBr548xY_iD4QkUatNw" title="Replay Memory Explained - Experience for Deep Q-Network Training" target="_blank">Replay Memory Explained - Experience for Deep Q-Network Training</a></li>
<li><a href="/rltoken/ZwReaNdr4Ei4GxWr-56oFg" title="Training a Deep Q-Network - Reinforcement Learning" target="_blank">Training a Deep Q-Network - Reinforcement Learning</a></li>
<li><a href="/rltoken/xAP3VzSnw0HLwjrBRn46Xw" title="Training a Deep Q-Network with Fixed Q-targets - Reinforcement Learning" target="_blank">Training a Deep Q-Network with Fixed Q-targets - Reinforcement Learning</a></li>
</ul>

<p><strong>References</strong>:</p>
<ul>
<li><a href="/rltoken/mSQhyiu7FEaFi_qTft1G2w" title="keras-rl" target="_blank">keras-rl</a>

<ul>
<li><a href="https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py" title="rl.policy" target="_blank">rl.policy</a></li>
<li><a href="https://github.com/keras-rl/keras-rl/blob/master/rl/memory.py" title="rl.memory" target="_blank">rl.memory</a></li>
<li><a href="https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py" title="rl.agents.dqn" target="_blank">rl.agents.dqn</a></li>
</ul></li>
<li><a href="https://arxiv.org/pdf/1312.5602.pdf" title="Playing Atari with Deep Reinforcement Learning" target="_blank">Playing Atari with Deep Reinforcement Learning</a></li>
</ul>

<h2>Learning Objectives</h2>
<ul>
<li>What is Deep Q-learning?</li>
<li>What is the policy network?</li>
<li>What is replay memory?</li>
<li>What is the target network?</li>
<li>Why must we utilize two separate networks during training?</li>
<li>What is keras-rl? How do you use it?</li>
</ul>
<h2>Installing Keras-RL</h2>
<pre><code>pip install --user keras-rl
</code></pre>
<h3>Dependencies (that should already be installed)</h3>
<pre><code>pip install --user keras==2.2.4
pip install --user Pillow
pip install --user h5py
</code></pre>
<h2 class="gap">Tasks</h2>

<section class="formatted-content">
            <div data-role="task6254" data-position="1">
              <div class=" clearfix gap" id="task-6254">
<span id="user_id" data-id="1283"></span>


</div>

<h4 class="task">
0. Breakout
    <span class="alert alert-warning mandatory-optional">
    mandatory
    </span>
</h4>

  
<!-- Progress vs Score -->

<!-- Task Body -->
<p>Write a python script <code>train.py</code> that utilizes <code>keras</code>, <code>keras-rl</code>, and <code>gym</code> to train an agent that can play Atari’s Breakout:</p>

<ul>
<li>Your script should utilize <code>keras-rl</code>‘s <code>DQNAgent</code>, <code>SequentialMemory</code>, and <code>EpsGreedyQPolicy</code></li>
<li>Your script should save the final policy network as <code>policy.h5</code></li>
</ul>

<p>Write a python script <code>play.py</code> that can display a game played by the agent trained by <code>train.py</code>:</p>

<ul>
<li>Your script should load the policy network saved in <code>policy.h5</code></li>
<li>Your agent should use the <code>GreedyQPolicy</code></li>
</ul>


<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>reinforcement_learning/0x01-deep_q_learning</code></li>
    <li>File: <code>train.py, play.py</code></li>
</ul>

</div>

</div>
</section>
