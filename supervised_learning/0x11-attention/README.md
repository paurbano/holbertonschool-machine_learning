<h1 class="gap">0x11. Attention</h1>
<h2>Learning Objectives</h2>
<h3>General</h3>
<ul>
<li>What is the attention mechanism?</li>
<li>How to apply attention to RNNs</li>
<li>What is a transformer?</li>
<li>How to create an encoder-decoder transformer model</li>
<li>What is GPT? </li>
<li>What is BERT?</li>
<li>What is self-supervised learning?</li>
<li>How to use BERT for specific NLP tasks</li>
<li>What is SQuAD? GLUE?</li>
</ul>

<h2>Update Tensorflow to 1.15</h2>
<p>In order to complete the following tasks, you will need to update <code>tensorflow</code> to version 1.15, which will also update <code>numpy</code> to version 1.16</p>
<pre><code>pip install --user tensorflow==1.15</code></pre>

<h2 class="gap">Tasks</h2>

<section class="formatted-content">
            <div data-role="task5420" data-position="1">
              <div class=" clearfix gap" id="task-5420">
<span id="user_id" data-id="1283"></span>

    <div class="student_task_controls">

      <!-- button Done -->
        <button class="student_task_done btn btn-default no" data-task-id="5420">
          <span class="no"><i class="fa fa-square-o"></i></span>
          <span class="yes"><i class="fa fa-check-square-o"></i></span>
          <span class="pending"><i class="fa fa-spinner fa-pulse"></i></span>
          Done<span class="no pending">?</span><span class="yes">!</span>
        </button>
        <br>

      <!-- button Help! -->
      <button class="users_done_for_task btn btn-default btn-default" data-task-id="5420" data-project-id="570" data-toggle="modal" data-target="#task-5420-users-done-modal">
        Help
      </button>
      <div class="modal fade users-done-modal" id="task-5420-users-done-modal" data-task-id="5420" data-project-id="570">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
            <h4 class="modal-title">Students who are done with "0. RNN Encoder"</h4>
        </div>
        <div class="modal-body">
            <div class="list-group">
            </div>
            <div class="spinner">
                <div class="bounce1"></div>
                <div class="bounce2"></div>
                <div class="bounce3"></div>
            </div>
            <div class="error"></div>
        </div>
        </div>
    </div>
</div>


    </div>

  <h4 class="task">
    0. RNN Encoder
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create a class <code>RNNEncoder</code> that inherits from <code>tensorflow.keras.layers.Layer</code> to encode for machine translation:</p>

<ul>
<li>Class constructor <code>def __init__(self, vocab, embedding, units, batch):</code>

<ul>
<li><code>vocab</code> is an integer representing the size of the input vocabulary</li>
<li><code>embedding</code> is an integer representing the dimensionality of the embedding vector</li>
<li><code>units</code> is an integer representing the number of hidden units in the RNN cell</li>
<li><code>batch</code> is an integer representing the batch size</li>
<li>Sets the following public instance attributes:

<ul>
<li><code>batch</code> - the batch size</li>
<li><code>units</code> - the number of hidden units in the RNN cell</li>
<li><code>embedding</code> - a <code>keras</code> Embedding layer that converts words from the vocabulary into an embedding vector</li>
<li><code>gru</code> - a <code>keras</code> GRU layer with <code>units</code> units

<ul>
<li>Should return both the full sequence of outputs as well as the last hidden state</li>
<li>Recurrent weights should be initialized with <code>glorot_uniform</code></li>
</ul></li>
</ul></li>
</ul></li>
<li>Public instance method <code>def initialize_hidden_state(self):</code>

<ul>
<li>Initializes the hidden states for the RNN cell to a tensor of zeros</li>
<li>Returns: a tensor of shape <code>(batch, units)</code>containing the initialized hidden states</li>
</ul></li>
<li> Public instance method <code>def call(self, x, initial):</code>

<ul>
<li> <code>x</code> is a tensor of shape <code>(batch, input_seq_len)</code> containing the input to the encoder layer as word indices within the vocabulary</li>
<li> <code>initial</code> is a tensor of shape <code>(batch, units)</code> containing the initial hidden state</li>
<li> Returns: <code>outputs, hidden</code>

<ul>
<li> <code>outputs</code> is a tensor of shape <code>(batch, input_seq_len, units)</code>containing the outputs of the encoder</li>
<li> <code>hidden</code> is a tensor of shape <code>(batch, units)</code> containing the last hidden state of the encoder</li>
</ul></li>
</ul></li>
</ul>

<pre><code>$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNEncoder = __import__('0-rnn_encoder').RNNEncoder

encoder = RNNEncoder(1024, 128, 256, 32)
print(encoder.batch)
print(encoder.units)
print(type(encoder.embedding))
print(type(encoder.gru))

initial = encoder.initialize_hidden_state()
print(initial)
x = tf.convert_to_tensor(np.random.choice(1024, 320).reshape((32, 10)))
outputs, hidden = encoder(x, initial)
print(outputs)
print(hidden)
$ ./0-main.py
32
256
&lt;class 'tensorflow.python.keras.layers.embeddings.Embedding'&gt;
&lt;class 'tensorflow.python.keras.layers.recurrent.GRU'&gt;
Tensor("zeros:0", shape=(32, 256), dtype=float32)
Tensor("rnn_encoder/gru/transpose_1:0", shape=(32, 10, 256), dtype=float32)
Tensor("rnn_encoder/gru/while/Exit_2:0", shape=(32, 256), dtype=float32)
$
</code></pre>

<p><em>Ignore the Warning messages in the output</em></p>


  <!-- Task URLs -->

  <!-- Github information -->
    <p class="sm-gap"><strong>Repo:</strong></p>
    <ul>
      <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
        <li>Directory: <code>supervised_learning/0x11-attention</code></li>
        <li>File: <code>0-rnn_encoder.py</code></li>
    </ul>





</div>
 
            </div>
            <div data-role="task5421" data-position="2">
              <div class=" clearfix gap" id="task-5421">
<span id="user_id" data-id="1283"></span>

    <div class="student_task_controls">

      <!-- button Done -->
        <button class="student_task_done btn btn-default no" data-task-id="5421">
          <span class="no"><i class="fa fa-square-o"></i></span>
          <span class="yes"><i class="fa fa-check-square-o"></i></span>
          <span class="pending"><i class="fa fa-spinner fa-pulse"></i></span>
          Done<span class="no pending">?</span><span class="yes">!</span>
        </button>
        <br>

      <!-- button Help! -->
      <button class="users_done_for_task btn btn-default btn-default" data-task-id="5421" data-project-id="570" data-toggle="modal" data-target="#task-5421-users-done-modal">
        Help
      </button>
      <div class="modal fade users-done-modal" id="task-5421-users-done-modal" data-task-id="5421" data-project-id="570">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
            <h4 class="modal-title">Students who are done with "1. Self Attention"</h4>
        </div>
        <div class="modal-body">
            <div class="list-group">
            </div>
            <div class="spinner">
                <div class="bounce1"></div>
                <div class="bounce2"></div>
                <div class="bounce3"></div>
            </div>
            <div class="error"></div>
        </div>
        </div>
    </div>
</div>


    </div>

  <h4 class="task">
    1. Self Attention
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create a class <code>SelfAttention</code> that inherits from <code>tensorflow.keras.layers.Layer</code> to calculate the attention for machine translation based on <a href="/rltoken/YlDIODUFbkYQbRL3a5CwEQ" title="this paper" target="_blank">this paper</a>:</p>

<ul>
<li>Class constructor <code>def __init__(self, units):</code>

<ul>
<li><code>units</code> is an integer representing the number of hidden units in the alignment model</li>
<li> Sets the following public instance attributes:

<ul>
<li> <code>W</code> - a Dense layer with <code>units</code> units, to be applied to the previous decoder hidden state</li>
<li> <code>U</code> - a Dense layer with <code>units</code> units, to be applied to the encoder hidden states</li>
<li> <code>V</code> - a Dense layer with <code>1</code> units, to be applied to the tanh of the sum of the outputs of <code>W</code> and <code>U</code></li>
</ul></li>
</ul></li>
<li>Public instance method <code>def call(self, s_prev, hidden_states):</code>

<ul>
<li><code>s_prev</code> is a tensor of shape <code>(batch, units)</code> containing the previous decoder hidden state</li>
<li><code>hidden_states</code> is a tensor of shape <code>(batch, input_seq_len, units)</code>containing the outputs of the encoder</li>
<li>Returns: <code>context, weights</code>

<ul>
<li><code>context</code> is a tensor of shape <code>(batch, units)</code> that contains the context vector for the decoder</li>
<li><code>weights</code> is a tensor of shape <code>(batch, input_seq_len, 1)</code> that contains the attention weights</li>
</ul></li>
</ul></li>
</ul>

<pre><code>$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

attention = SelfAttention(256)
print(attention.W)
print(attention.U)
print(attention.V)
s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)), preferred_dtype='float32')
hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)), preferred_dtype='float32')
context, weights = attention(s_prev, hidden_states)
print(context)
print(weights)
$ ./1-main.py
&lt;tensorflow.python.keras.layers.core.Dense object at 0x12309d3c8&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb28536b38&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb28536e48&gt;
Tensor("self_attention/Sum:0", shape=(32, 256), dtype=float64)
Tensor("self_attention/transpose_1:0", shape=(32, 10, 1), dtype=float64)
$
</code></pre>

<p><em>Ignore the Warning messages in the output</em></p>


  <!-- Task URLs -->

  <!-- Github information -->
    <p class="sm-gap"><strong>Repo:</strong></p>
    <ul>
      <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
        <li>Directory: <code>supervised_learning/0x11-attention</code></li>
        <li>File: <code>1-self_attention.py</code></li>
    </ul>





</div>
 
            </div>
            <div data-role="task5422" data-position="3">
              <div class=" clearfix gap" id="task-5422">
<span id="user_id" data-id="1283"></span>

    <div class="student_task_controls">

      <!-- button Done -->
        <button class="student_task_done btn btn-default no" data-task-id="5422">
          <span class="no"><i class="fa fa-square-o"></i></span>
          <span class="yes"><i class="fa fa-check-square-o"></i></span>
          <span class="pending"><i class="fa fa-spinner fa-pulse"></i></span>
          Done<span class="no pending">?</span><span class="yes">!</span>
        </button>
        <br>

      <!-- button Help! -->
      <button class="users_done_for_task btn btn-default btn-default" data-task-id="5422" data-project-id="570" data-toggle="modal" data-target="#task-5422-users-done-modal">
        Help
      </button>
      <div class="modal fade users-done-modal" id="task-5422-users-done-modal" data-task-id="5422" data-project-id="570">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
            <h4 class="modal-title">Students who are done with "2. RNN Decoder"</h4>
        </div>
        <div class="modal-body">
            <div class="list-group">
            </div>
            <div class="spinner">
                <div class="bounce1"></div>
                <div class="bounce2"></div>
                <div class="bounce3"></div>
            </div>
            <div class="error"></div>
        </div>
        </div>
    </div>
</div>


    </div>

  <h4 class="task">
    2. RNN Decoder
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create a class <code>RNNDecoder</code> that inherits from <code>tensorflow.keras.layers.Layer</code> to decode for machine translation:</p>

<ul>
<li>Class constructor <code>def __init__(self, vocab, embedding, units, batch):</code>

<ul>
<li><code>vocab</code> is an integer representing the size of the output vocabulary</li>
<li><code>embedding</code> is an integer representing the dimensionality of the embedding vector</li>
<li><code>units</code> is an integer representing the number of hidden units in the RNN cell</li>
<li><code>batch</code> is an integer representing the batch size</li>
<li>Sets the following public instance attributes:

<ul>
<li><code>embedding</code> -  a <code>keras</code> Embedding layer that converts words from the vocabulary into an embedding vector</li>
<li><code>gru</code> - a <code>keras</code> GRU layer with <code>units</code> units

<ul>
<li>Should return both the full sequence of outputs as well as the last hidden state</li>
<li>Recurrent weights should be initialized with <code>glorot_uniform</code></li>
</ul></li>
<li><code>F</code> - a Dense layer with <code>vocab</code> units</li>
</ul></li>
</ul></li>
<li>Public instance method <code>def call(self, x, s_prev, hidden_states):</code>

<ul>
<li><code>x</code> is a tensor of shape <code>(batch, 1)</code> containing the previous word in the target sequence as an index of the target vocabulary</li>
<li><code>s_prev</code> is a tensor of shape <code>(batch, units)</code> containing the previous decoder hidden state</li>
<li><code>hidden_states</code> is a tensor of shape <code>(batch, input_seq_len, units)</code>containing the outputs of the encoder</li>
<li>You should use <code>SelfAttention = __import__('1-self_attention').SelfAttention</code></li>
<li>You should concatenate the context vector with x in that order</li>
<li>Returns: <code>y, s</code>

<ul>
<li><code>y</code> is a tensor of shape <code>(batch, vocab)</code> containing the output word as a one hot vector in the target vocabulary</li>
<li><code>s</code> is a tensor of shape <code>(batch, units)</code> containing the new decoder hidden state</li>
</ul></li>
</ul></li>
</ul>

<pre><code>$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNDecoder = __import__('2-rnn_decoder').RNNDecoder

decoder = RNNDecoder(2048, 128, 256, 32)
print(decoder.embedding)
print(decoder.gru)
print(decoder.F)
x = tf.convert_to_tensor(np.random.choice(2048, 32).reshape((32, 1)))
s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)).astype('float32'))
hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)).astype('float32'))
y, s = decoder(x, s_prev, hidden_states)
print(y)
print(s)
$ ./2-main.py
&lt;tensorflow.python.keras.layers.embeddings.Embedding object at 0x1321113c8&gt;
&lt;tensorflow.python.keras.layers.recurrent.GRU object at 0xb375aab00&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb375d5128&gt;
Tensor("rnn_decoder/dense/BiasAdd:0", shape=(32, 2048), dtype=float32)
Tensor("rnn_decoder/gru/while/Exit_2:0", shape=(32, 256), dtype=float32)
$
</code></pre>

<p><em>Ignore the Warning messages in the output</em></p>


  <!-- Task URLs -->

  <!-- Github information -->
    <p class="sm-gap"><strong>Repo:</strong></p>
    <ul>
      <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
        <li>Directory: <code>supervised_learning/0x11-attention</code></li>
        <li>File: <code>2-rnn_decoder.py</code></li>
    </ul>





</div>
 
            </div>
            <div data-role="task5424" data-position="5">
              <div class=" clearfix gap" id="task-5424">
<span id="user_id" data-id="1283"></span>

    <div class="student_task_controls">

      <!-- button Done -->
        <button class="student_task_done btn btn-default no" data-task-id="5424">
          <span class="no"><i class="fa fa-square-o"></i></span>
          <span class="yes"><i class="fa fa-check-square-o"></i></span>
          <span class="pending"><i class="fa fa-spinner fa-pulse"></i></span>
          Done<span class="no pending">?</span><span class="yes">!</span>
        </button>
        <br>

      <!-- button Help! -->
      <button class="users_done_for_task btn btn-default btn-default" data-task-id="5424" data-project-id="570" data-toggle="modal" data-target="#task-5424-users-done-modal">
        Help
      </button>
      <div class="modal fade users-done-modal" id="task-5424-users-done-modal" data-task-id="5424" data-project-id="570">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
            <h4 class="modal-title">Students who are done with "3. Positional Encoding"</h4>
        </div>
        <div class="modal-body">
            <div class="list-group">
            </div>
            <div class="spinner">
                <div class="bounce1"></div>
                <div class="bounce2"></div>
                <div class="bounce3"></div>
            </div>
            <div class="error"></div>
        </div>
        </div>
    </div>
</div>


    </div>

  <h4 class="task">
    3. Positional Encoding
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write the function <code>def positional_encoding(max_seq_len, dm):</code> that calculates the positional encoding for a transformer:</p>

<ul>
<li><code>max_seq_len</code> is an integer representing the maximum sequence length</li>
<li><code>dm</code> is the model depth</li>
<li>Returns: a <code>numpy.ndarray</code> of shape <code>(max_seq_len, dm)</code> containing the positional encoding vectors</li>
<li>You should use <code>import numpy as np</code></li>
</ul>

<pre><code>$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
positional_encoding = __import__('4-positional_encoding').positional_encoding

PE = positional_encoding(30, 512)
print(PE.shape)
print(PE)
$ ./4-main.py
(30, 512)
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [ 9.56375928e-01 -2.92138809e-01  7.91416314e-01 ...  9.99995791e-01
   2.79890525e-03  9.99996083e-01]
 [ 2.70905788e-01 -9.62605866e-01  9.53248145e-01 ...  9.99995473e-01
   2.90256812e-03  9.99995788e-01]
 [-6.63633884e-01 -7.48057530e-01  2.94705106e-01 ...  9.99995144e-01
   3.00623096e-03  9.99995481e-01]]
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
    <p class="sm-gap"><strong>Repo:</strong></p>
    <ul>
      <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
        <li>Directory: <code>supervised_learning/0x11-attention</code></li>
        <li>File: <code>4-positional_encoding.py</code></li>
    </ul>





</div>
 
            </div>
            <div data-role="task5425" data-position="6">
              <div class=" clearfix gap" id="task-5425">
<span id="user_id" data-id="1283"></span>

    <div class="student_task_controls">

      <!-- button Done -->
        <button class="student_task_done btn btn-default no" data-task-id="5425">
          <span class="no"><i class="fa fa-square-o"></i></span>
          <span class="yes"><i class="fa fa-check-square-o"></i></span>
          <span class="pending"><i class="fa fa-spinner fa-pulse"></i></span>
          Done<span class="no pending">?</span><span class="yes">!</span>
        </button>
        <br>

      <!-- button Help! -->
      <button class="users_done_for_task btn btn-default btn-default" data-task-id="5425" data-project-id="570" data-toggle="modal" data-target="#task-5425-users-done-modal">
        Help
      </button>
      <div class="modal fade users-done-modal" id="task-5425-users-done-modal" data-task-id="5425" data-project-id="570">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
            <h4 class="modal-title">Students who are done with "4. Scaled Dot Product Attention"</h4>
        </div>
        <div class="modal-body">
            <div class="list-group">
            </div>
            <div class="spinner">
                <div class="bounce1"></div>
                <div class="bounce2"></div>
                <div class="bounce3"></div>
            </div>
            <div class="error"></div>
        </div>
        </div>
    </div>
</div>


    </div>

  <h4 class="task">
    4. Scaled Dot Product Attention
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/7/8f5aadef511d9f646f5009756035b472073fe896.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210105%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210105T161816Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=c5a03d1c2aeeffe6a9c333ac7e8b18482e6fcd8e5cb8fb778268b6c9b7795267" alt="" style=""></p>

<p>Write the function <code>def sdp_attention(Q, K, V, mask=None)</code> that calculates the scaled dot product attention:</p>

<ul>
<li><code>Q</code> is a tensor with its last two dimensions as <code>(..., seq_len_q, dk)</code>  containing the query matrix</li>
<li><code>K</code> is a tensor with its last two dimensions as <code>(..., seq_len_v, dk)</code>  containing the key matrix</li>
<li><code>V</code> is a tensor with its last two dimensions as <code>(..., seq_len_v, dv)</code>  containing the value matrix</li>
<li><code>mask</code> is a tensor that can be broadcast into <code>(..., seq_len_q, seq_len_v)</code> containing the optional mask, or defaulted to <code>None</code>

<ul>
<li>if <code>mask</code> is not <code>None</code>, multiply <code>-1e9</code> to the mask and add it to the scaled matrix multiplication </li>
</ul></li>
<li>The preceding dimensions of <code>Q</code>, <code>K</code>, and <code>V</code> are the same</li>
<li>Returns: <code>output, weights</code>

<ul>
<li><code>output</code>a tensor with its last two dimensions as <code>(..., seq_len_q, dv)</code> containing the scaled dot product attention</li>
<li><code>weights</code> a tensor with its last two dimensions as <code>(..., seq_len_q, seq_len_v)</code> containing the attention weights</li>
</ul></li>
</ul>

<pre><code>$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention

np.random.seed(0)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 10, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 512)).astype('float32'))
output, weights = sdp_attention(Q, K, V)
print(output)
print(weights)
$ ./5-main.py
Tensor("MatMul_1:0", shape=(50, 10, 512), dtype=float32)
Tensor("Softmax:0", shape=(50, 10, 15), dtype=float32)
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
    <p class="sm-gap"><strong>Repo:</strong></p>
    <ul>
      <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
        <li>Directory: <code>supervised_learning/0x11-attention</code></li>
        <li>File: <code>5-sdp_attention.py</code></li>
    </ul>





</div>
 
            </div>
            <div data-role="task5426" data-position="7">
              <div class=" clearfix gap" id="task-5426">
<span id="user_id" data-id="1283"></span>

    <div class="student_task_controls">

      <!-- button Done -->
        <button class="student_task_done btn btn-default no" data-task-id="5426">
          <span class="no"><i class="fa fa-square-o"></i></span>
          <span class="yes"><i class="fa fa-check-square-o"></i></span>
          <span class="pending"><i class="fa fa-spinner fa-pulse"></i></span>
          Done<span class="no pending">?</span><span class="yes">!</span>
        </button>
        <br>

      <!-- button Help! -->
      <button class="users_done_for_task btn btn-default btn-default" data-task-id="5426" data-project-id="570" data-toggle="modal" data-target="#task-5426-users-done-modal">
        Help
      </button>
      <div class="modal fade users-done-modal" id="task-5426-users-done-modal" data-task-id="5426" data-project-id="570">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
            <h4 class="modal-title">Students who are done with "5. Multi Head Attention"</h4>
        </div>
        <div class="modal-body">
            <div class="list-group">
            </div>
            <div class="spinner">
                <div class="bounce1"></div>
                <div class="bounce2"></div>
                <div class="bounce3"></div>
            </div>
            <div class="error"></div>
        </div>
        </div>
    </div>
</div>


    </div>

  <h4 class="task">
    5. Multi Head Attention
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/7/4a5aaa54ebdc32529b4f09a5f22789dc267e0796.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210105%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210105T161816Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=a5378897479661111ba7fde0a99de63c8d3cc8e3cc74bddcddfd870debc58a9c" alt="" style=""></p>

<p>Create a class <code>MultiHeadAttention</code> that inherits from <code>tensorflow.keras.layers.Layer</code> to perform multi head attention:</p>

<ul>
<li>Class constructor <code>def __init__(self, dm, h):</code>

<ul>
<li><code>dm</code> is an integer representing the dimensionality of the model</li>
<li><code>h</code> is an integer representing the number of heads</li>
<li><code>dm</code> is divisible by <code>h</code></li>
<li>Sets the following public instance attributes:

<ul>
<li><code>h</code> - the number of heads</li>
<li><code>dm</code> - the dimensionality of the model</li>
<li><code>depth</code> - the depth of each attention head</li>
<li><code>Wq</code> - a Dense layer with <code>dm</code> units, used to generate the query matrix</li>
<li><code>Wk</code> - a Dense layer with <code>dm</code> units, used to generate the key matrix</li>
<li><code>Wv</code> - a Dense layer with <code>dm</code> units, used to generate the value matrix</li>
<li><code>linear</code> - a Dense layer with <code>dm</code> units, used to generate the attention output</li>
</ul></li>
</ul></li>
<li>Public instance method <code>def call(self, Q, K, V, mask):</code>

<ul>
<li><code>Q</code> is a tensor of shape <code>(batch, seq_len_q, dk)</code> containing the input to generate the query matrix</li>
<li><code>K</code> is a tensor of shape <code>(batch, seq_len_v, dk)</code> containing the input to generate the key matrix</li>
<li><code>V</code> is a tensor of shape <code>(batch, seq_len_v, dv)</code> containing the input to generate the value matrix</li>
<li><code>mask</code> is always <code>None</code></li>
<li>Returns: <code>output, weights</code>

<ul>
<li><code>output</code>a tensor with its last two dimensions as <code>(..., seq_len_q, dm)</code> containing the scaled dot product attention</li>
<li><code>weights</code> a tensor with its last three dimensions as <code>(..., h, seq_len_q, seq_len_v)</code> containing the attention weights</li>
</ul></li>
</ul></li>
<li>You should use <code>sdp_attention = __import__('5-sdp_attention').sdp_attention</code></li>
</ul>

<pre><code>$ cat 6-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

mha = MultiHeadAttention(512, 8)
print(mha.dm)
print(mha.h)
print(mha.depth)
print(mha.Wq)
print(mha.Wk)
print(mha.Wv)
print(mha.linear)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
output, weights = mha(Q, K, V, None)
print(output)
print(weights)
$ ./6-main.py
512
8
64
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb2c585b38&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb2c585e48&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb2c5b1198&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb2c5b14a8&gt;
Tensor("multi_head_attention/dense_3/BiasAdd:0", shape=(50, 15, 512), dtype=float32)
Tensor("multi_head_attention/Softmax:0", shape=(50, 8, 15, 15), dtype=float32)
$
</code></pre>

<p><em>Ignore the Warning messages in the output</em></p>


  <!-- Task URLs -->

  <!-- Github information -->
    <p class="sm-gap"><strong>Repo:</strong></p>
    <ul>
      <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
        <li>Directory: <code>supervised_learning/0x11-attention</code></li>
        <li>File: <code>6-multihead_attention.py</code></li>
    </ul>





</div>
 
            </div>
            <div data-role="task5427" data-position="8">
              <div class=" clearfix gap" id="task-5427">
<span id="user_id" data-id="1283"></span>

    <div class="student_task_controls">

      <!-- button Done -->
        <button class="student_task_done btn btn-default no" data-task-id="5427">
          <span class="no"><i class="fa fa-square-o"></i></span>
          <span class="yes"><i class="fa fa-check-square-o"></i></span>
          <span class="pending"><i class="fa fa-spinner fa-pulse"></i></span>
          Done<span class="no pending">?</span><span class="yes">!</span>
        </button>
        <br>

      <!-- button Help! -->
      <button class="users_done_for_task btn btn-default btn-default" data-task-id="5427" data-project-id="570" data-toggle="modal" data-target="#task-5427-users-done-modal">
        Help
      </button>
      <div class="modal fade users-done-modal" id="task-5427-users-done-modal" data-task-id="5427" data-project-id="570">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
            <h4 class="modal-title">Students who are done with "6. Transformer Encoder Block"</h4>
        </div>
        <div class="modal-body">
            <div class="list-group">
            </div>
            <div class="spinner">
                <div class="bounce1"></div>
                <div class="bounce2"></div>
                <div class="bounce3"></div>
            </div>
            <div class="error"></div>
        </div>
        </div>
    </div>
</div>


    </div>

  <h4 class="task">
    6. Transformer Encoder Block
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/7/50a5309eae279760a5d6fc6031aa045eafd0e605.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210105%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210105T161816Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=a70e1ae14c2ae468054465e2d8fc804517b6c7d9869337a7ee46cff76d333427" alt="" style=""></p>

<p>Create a class <code>EncoderBlock</code> that inherits from <code>tensorflow.keras.layers.Layer</code> to create an encoder block for a transformer:</p>

<ul>
<li>Class constructor <code>def __init__(self, dm, h, hidden, drop_rate=0.1):</code>

<ul>
<li><code>dm</code> - the dimensionality of the model</li>
<li><code>h</code> - the number of heads</li>
<li><code>hidden</code> - the number of hidden units in the fully connected layer</li>
<li><code>drop_rate</code> - the dropout rate</li>
<li>Sets the following public instance attributes:

<ul>
<li><code>mha</code> - a <code>MultiHeadAttention</code> layer</li>
<li><code>dense_hidden</code> - the hidden dense layer with <code>hidden</code> units and <code>relu</code> activation</li>
<li><code>dense_output</code> - the output dense layer with <code>dm</code> units</li>
<li><code>layernorm1</code> - the first layer norm layer, with <code>epsilon=1e-6</code></li>
<li><code>layernorm2</code> - the second layer norm layer, with <code>epsilon=1e-6</code></li>
<li><code>dropout1</code> - the first dropout layer</li>
<li><code>dropout2</code> - the second dropout layer</li>
</ul></li>
</ul></li>
<li>Public instance method <code>call(self, x, training, mask=None):</code>

<ul>
<li><code>x</code> - a tensor of shape <code>(batch, input_seq_len, dm)</code>containing the input to the encoder block</li>
<li><code>training</code> - a boolean to determine if the model is training</li>
<li><code>mask</code> - the mask to be applied for multi head attention</li>
<li>Returns: a tensor of shape <code>(batch, input_seq_len, dm)</code> containing the block’s output</li>
</ul></li>
<li>You should use <code>MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention</code></li>
</ul>

<pre><code>$ cat 7-main
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock

eblock = EncoderBlock(512, 8, 2048)
print(eblock.mha)
print(eblock.dense_hidden)
print(eblock.dense_output)
print(eblock.layernorm1)
print(eblock.layernorm2)
print(eblock.dropout1)
print(eblock.dropout2)
x = tf.random.uniform((32, 10, 512))
output = eblock(x, True, None)
print(output)
$ ./7-main.py
&lt;6-multihead_attention.MultiHeadAttention object at 0x12c61b390&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb31ae1860&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb31ae1b70&gt;
&lt;tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb31ae1e80&gt;
&lt;tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb31aea128&gt;
&lt;tensorflow.python.keras.layers.core.Dropout object at 0xb31aea390&gt;
&lt;tensorflow.python.keras.layers.core.Dropout object at 0xb31aea518&gt;
Tensor("encoder_block/layer_normalization_1/batchnorm/add_1:0", shape=(32, 10, 512), dtype=float32)
$
</code></pre>

<p><em>Ignore the Warning messages in the output</em></p>


  <!-- Task URLs -->

  <!-- Github information -->
    <p class="sm-gap"><strong>Repo:</strong></p>
    <ul>
      <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
        <li>Directory: <code>supervised_learning/0x11-attention</code></li>
        <li>File: <code>7-transformer_encoder_block.py</code></li>
    </ul>





</div>
 
            </div>
            <div data-role="task5428" data-position="9">
              <div class=" clearfix gap" id="task-5428">
<span id="user_id" data-id="1283"></span>

    <div class="student_task_controls">

      <!-- button Done -->
        <button class="student_task_done btn btn-default no" data-task-id="5428">
          <span class="no"><i class="fa fa-square-o"></i></span>
          <span class="yes"><i class="fa fa-check-square-o"></i></span>
          <span class="pending"><i class="fa fa-spinner fa-pulse"></i></span>
          Done<span class="no pending">?</span><span class="yes">!</span>
        </button>
        <br>

      <!-- button Help! -->
      <button class="users_done_for_task btn btn-default btn-default" data-task-id="5428" data-project-id="570" data-toggle="modal" data-target="#task-5428-users-done-modal">
        Help
      </button>
      <div class="modal fade users-done-modal" id="task-5428-users-done-modal" data-task-id="5428" data-project-id="570">
    <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
            <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">×</span></button>
            <h4 class="modal-title">Students who are done with "7. Transformer Decoder Block"</h4>
        </div>
        <div class="modal-body">
            <div class="list-group">
            </div>
            <div class="spinner">
                <div class="bounce1"></div>
                <div class="bounce2"></div>
                <div class="bounce3"></div>
            </div>
            <div class="error"></div>
        </div>
        </div>
    </div>
</div>


    </div>

  <h4 class="task">
    7. Transformer Decoder Block
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create a class <code>DecoderBlock</code> that inherits from <code>tensorflow.keras.layers.Layer</code> to create an encoder block for a transformer:</p>

<ul>
<li>Class constructor <code>def __init__(self, dm, h, hidden, drop_rate=0.1):</code>

<ul>
<li><code>dm</code> - the dimensionality of the model</li>
<li><code>h</code> - the number of heads</li>
<li><code>hidden</code> - the number of hidden units in the fully connected layer</li>
<li><code>drop_rate</code> - the dropout rate</li>
<li>Sets the following public instance attributes:

<ul>
<li><code>mha1</code> - the first <code>MultiHeadAttention</code> layer</li>
<li><code>mha2</code> - the second <code>MultiHeadAttention</code> layer</li>
<li><code>dense_hidden</code> - the hidden dense layer with <code>hidden</code> units and <code>relu</code> activation</li>
<li><code>dense_output</code> - the output dense layer with <code>dm</code> units</li>
<li><code>layernorm1</code> - the first layer norm layer, with <code>epsilon=1e-6</code></li>
<li><code>layernorm2</code> - the second layer norm layer, with <code>epsilon=1e-6</code></li>
<li><code>layernorm3</code> - the third layer norm layer, with <code>epsilon=1e-6</code></li>
<li><code>dropout1</code> - the first dropout layer</li>
<li><code>dropout2</code> - the second dropout layer</li>
<li><code>dropout3</code> - the third dropout layer</li>
</ul></li>
</ul></li>
<li>Public instance method <code>def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):</code>

<ul>
<li><code>x</code> - a tensor of shape <code>(batch, target_seq_len, dm)</code>containing the input to the decoder block</li>
<li><code>encoder_output</code> - a tensor of shape <code>(batch, input_seq_len, dm)</code>containing the output of the encoder</li>
<li><code>training</code> - a boolean to determine if the model is training</li>
<li><code>look_ahead_mask</code> - the mask to be applied to the first multi head attention layer</li>
<li><code>padding_mask</code> - the mask to be applied to the second multi head attention layer</li>
<li>Returns: a tensor of shape <code>(batch, target_seq_len, dm)</code> containing the block’s output</li>
</ul></li>
<li>You should use <code>MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention</code></li>
</ul>

<pre><code>$ cat 8-main.py
#!/usr/bin/env python3

import tensorflow as tf
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock

dblock = DecoderBlock(512, 8, 2048)
print(dblock.mha1)
print(dblock.mha2)
print(dblock.dense_hidden)
print(dblock.dense_output)
print(dblock.layernorm1)
print(dblock.layernorm2)
print(dblock.layernorm3)
print(dblock.dropout1)
print(dblock.dropout2)
print(dblock.dropout3)
x = tf.random.uniform((32, 15, 512))
hidden_states = tf.random.uniform((32, 10, 512))
output = dblock(x, hidden_states, False, None, None)
print(output)
$ ./8-main.py
&lt;6-multihead_attention.MultiHeadAttention object at 0x1313f4400&gt;
&lt;6-multihead_attention.MultiHeadAttention object at 0xb368bc9b0&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb368c37b8&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb368c3ac8&gt;
&lt;tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb368c3dd8&gt;
&lt;tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb368cb080&gt;
&lt;tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb368cb2e8&gt;
&lt;tensorflow.python.keras.layers.core.Dropout object at 0xb368cb550&gt;
&lt;tensorflow.python.keras.layers.core.Dropout object at 0xb368cb6d8&gt;
&lt;tensorflow.python.keras.layers.core.Dropout object at 0xb368cb828&gt;
Tensor("decoder_block/layer_normalization_2/batchnorm/add_1:0", shape=(32, 15, 512), dtype=float32)
$
</code></pre>

<p><em>Ignore the Warning messages in the output</em></p>


<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x11-attention</code></li>
    <li>File: <code>8-transformer_decoder_block.py</code></li>
</ul>

</div>
 
</div>
<div data-role="task5429" data-position="10">
    <div class=" clearfix gap" id="task-5429">
<span id="user_id" data-id="1283"></span>

</div>

  <h4 class="task">
    8. Transformer Encoder
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create a class <code>Encoder</code> that inherits from <code>tensorflow.keras.layers.Layer</code> to create the encoder for a transformer:</p>

<ul>
<li>Class constructor <code>def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):</code>

<ul>
<li><code>N</code> - the number of blocks in the encoder</li>
<li><code>dm</code> - the dimensionality of the model</li>
<li><code>h</code> - the number of heads</li>
<li><code>hidden</code> - the number of hidden units in the fully connected layer</li>
<li><code>input_vocab</code> - the size of the input vocabulary</li>
<li><code>max_seq_len</code> - the maximum sequence length possible</li>
<li><code>drop_rate</code> - the dropout rate</li>
<li>Sets the following public instance attributes:

<ul>
<li><code>N</code> - the number of blocks in the encoder</li>
<li><code>dm</code> - the dimensionality of the model</li>
<li><code>embedding</code> - the embedding layer for the inputs</li>
<li><code>positional_encoding</code> - a <code>numpy.ndarray</code> of shape <code>(max_seq_len, dm)</code> containing the positional encodings</li>
<li><code>blocks</code> - a list of length <code>N</code> containing all of the <code>EncoderBlock</code>‘s</li>
<li><code>dropout</code> - the dropout layer, to be applied to the positional encodings</li>
</ul></li>
</ul></li>
<li>Public instance method <code>call(self, x, training, mask):</code>

<ul>
<li><code>x</code> - a tensor of shape <code>(batch, input_seq_len, dm)</code>containing the input to the encoder</li>
<li><code>training</code> - a boolean to determine if the model is training</li>
<li><code>mask</code> - the mask to be applied for multi head attention</li>
<li>Returns: a tensor of shape <code>(batch, input_seq_len, dm)</code> containing the encoder output</li>
</ul></li>
<li>You should use <code>positional_encoding = __import__('4-positional_encoding').positional_encoding</code> and <code>EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock</code></li>
</ul>

<pre><code>$ cat 9-main.py
#!/usr/bin/env python3

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder

encoder = Encoder(6, 512, 8, 2048, 10000, 1000)
print(encoder.dm)
print(encoder.N)
print(encoder.embedding)
print(encoder.positional_encoding)
print(encoder.blocks)
print(encoder.dropout)
x = tf.random.uniform((32, 10))
output = encoder(x, True, None)
print(output)
$ ./9-main.py
512
6
&lt;tensorflow.python.keras.layers.embeddings.Embedding object at 0xb2981acc0&gt;
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [-8.97967480e-01 -4.40061818e-01  4.26195541e-01 ...  9.94266169e-01
   1.03168405e-01  9.94663903e-01]
 [-8.55473152e-01  5.17847165e-01  9.86278111e-01 ...  9.94254673e-01
   1.03271514e-01  9.94653203e-01]
 [-2.64607527e-02  9.99649853e-01  6.97559894e-01 ...  9.94243164e-01
   1.03374623e-01  9.94642492e-01]]
ListWrapper([&lt;7-transformer_encoder_block.EncoderBlock object at 0xb2981aef0&gt;, &lt;7-transformer_encoder_block.EncoderBlock object at 0xb29850ba8&gt;, &lt;7-transformer_encoder_block.EncoderBlock object at 0xb298647b8&gt;, &lt;7-transformer_encoder_block.EncoderBlock object at 0xb29e502e8&gt;, &lt;7-transformer_encoder_block.EncoderBlock object at 0xb29e5add8&gt;, &lt;7-transformer_encoder_block.EncoderBlock object at 0xb29e6c908&gt;])
&lt;tensorflow.python.keras.layers.core.Dropout object at 0xb29e7c470&gt;
Tensor("encoder/encoder_block_5/layer_normalization_11/batchnorm/add_1:0", shape=(32, 10, 512), dtype=float32)
$
</code></pre>

<p><em>Ignore the Warning messages in the output</em></p>
<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x11-attention</code></li>
    <li>File: <code>9-transformer_encoder.py</code></li>
</ul>

</div>
</div>
<div data-role="task5430" data-position="11">
    <div class=" clearfix gap" id="task-5430">
<span id="user_id" data-id="1283"></span>
</div>


    </div>

  <h4 class="task">
    9. Transformer Decoder
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

 <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create a class <code>Decoder</code> that inherits from <code>tensorflow.keras.layers.Layer</code> to create the decoder for a transformer:</p>

<ul>
<li>Class constructor <code>def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):</code></li>
<li><code>N</code> - the number of blocks in the encoder

<ul>
<li><code>dm</code> - the dimensionality of the model</li>
<li><code>h</code> - the number of heads</li>
<li><code>hidden</code> - the number of hidden units in the fully connected layer</li>
<li><code>target_vocab</code> - the size of the target vocabulary</li>
<li><code>max_seq_len</code> - the maximum sequence length possible</li>
<li><code>drop_rate</code> - the dropout rate</li>
<li>Sets the following public instance attributes:

<ul>
<li><code>N</code> - the number of blocks in the encoder</li>
<li><code>dm</code> - the dimensionality of the model</li>
<li><code>embedding</code> - the embedding layer for the targets</li>
<li><code>positional_encoding</code> - a <code>numpy.ndarray</code> of shape <code>(max_seq_len, dm)</code> containing the positional encodings</li>
<li><code>blocks</code> - a list of length <code>N</code> containing all of the <code>DecoderBlock</code>‘s</li>
<li><code>dropout</code> - the dropout layer, to be applied to the positional encodings</li>
</ul></li>
</ul></li>
<li>Public instance method <code>def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):</code>

<ul>
<li><code>x</code> - a tensor of shape <code>(batch, target_seq_len, dm)</code>containing the input to the decoder</li>
<li><code>encoder_output</code> - a tensor of shape <code>(batch, input_seq_len, dm)</code>containing the output of the encoder</li>
<li><code>training</code> - a boolean to determine if the model is training</li>
<li><code>look_ahead_mask</code> - the mask to be applied to the first multi head attention layer</li>
<li><code>padding_mask</code> - the mask to be applied to the second multi head attention layer</li>
<li>Returns: a tensor of shape <code>(batch, target_seq_len, dm)</code> containing the decoder output</li>
</ul></li>
<li>You should use <code>positional_encoding = __import__('4-positional_encoding').positional_encoding</code> and <code>DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock</code></li>
</ul>

<pre><code>$ cat 10-main.py
#!/usr/bin/env python3

import tensorflow as tf
Decoder = __import__('10-transformer_decoder').Decoder

decoder = Decoder(6, 512, 8, 2048, 12000, 1500)
print(decoder.dm)
print(decoder.N)
print(decoder.embedding)
print(decoder.positional_encoding)
print(decoder.blocks)
print(decoder.dropout)
x = tf.random.uniform((32, 15))
hidden_states = tf.random.uniform((32, 10, 512))
output = decoder(x, hidden_states, True, None, None)
print(output)
$ ./10-main.py
512
6
&lt;tensorflow.python.keras.layers.embeddings.Embedding object at 0xb2cdede48&gt;
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [ 9.99516416e-01 -3.10955511e-02 -8.59441209e-01 ...  9.87088496e-01
   1.54561841e-01  9.87983116e-01]
 [ 5.13875021e-01 -8.57865061e-01 -6.94580536e-02 ...  9.87071278e-01
   1.54664258e-01  9.87967088e-01]
 [-4.44220699e-01 -8.95917390e-01  7.80301396e-01 ...  9.87054048e-01
   1.54766673e-01  9.87951050e-01]]
ListWrapper([&lt;8-transformer_decoder_block.DecoderBlock object at 0xb2ce0f0b8&gt;, &lt;8-transformer_decoder_block.DecoderBlock object at 0xb2ce29ef0&gt;, &lt;8-transformer_decoder_block.DecoderBlock object at 0xb2d711b00&gt;, &lt;8-transformer_decoder_block.DecoderBlock object at 0xb2d72c710&gt;, &lt;8-transformer_decoder_block.DecoderBlock object at 0xb2d744320&gt;, &lt;8-transformer_decoder_block.DecoderBlock object at 0xb2d755ef0&gt;])
&lt;tensorflow.python.keras.layers.core.Dropout object at 0xb2d76db38&gt;
Tensor("decoder/decoder_block_5/layer_normalization_17/batchnorm/add_1:0", shape=(32, 15, 512), dtype=float32)
$
</code></pre>

<p><em>Ignore the Warning messages in the output</em></p>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x11-attention</code></li>
    <li>File: <code>10-transformer_decoder.py</code></li>
</ul>

</div>
 
</div>
<div data-role="task5431" data-position="12">
    <div class=" clearfix gap" id="task-5431">
<span id="user_id" data-id="1283"></span>

</div>


</div>

<h4 class="task">
10. Transformer Network
    <span class="alert alert-warning mandatory-optional">
    mandatory
    </span>
</h4>

<!-- Progress vs Score -->

<!-- Task Body -->
<p>Create a class <code>Transformer</code> that inherits from <code>tensorflow.keras.Model</code> to create a transformer network:</p>

<ul>
<li>Class constructor <code>def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):</code>

<ul>
<li><code>N</code> - the number of blocks in the encoder and decoder</li>
<li><code>dm</code> - the dimensionality of the model</li>
<li><code>h</code> - the number of heads</li>
<li><code>hidden</code> - the number of hidden units in the fully connected layers</li>
<li><code>input_vocab</code> - the size of the input vocabulary</li>
<li><code>target_vocab</code> - the size of the target vocabulary</li>
<li><code>max_seq_input</code> - the maximum sequence length possible for the input</li>
<li><code>max_seq_target</code> - the maximum sequence length possible for the target</li>
<li><code>drop_rate</code> - the dropout rate</li>
<li>Sets the following public instance attributes:

<ul>
<li><code>encoder</code> - the encoder layer</li>
<li><code>decoder</code> - the decoder layer</li>
<li><code>linear</code> - a final Dense layer with <code>target_vocab</code> units</li>
</ul></li>
</ul></li>
<li>Public instance method <code>def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):</code>

<ul>
<li><code>inputs</code> - a tensor of shape <code>(batch, input_seq_len)</code>containing the inputs</li>
<li><code>target</code> - a tensor of shape <code>(batch, target_seq_len)</code>containing the target</li>
<li><code>training</code> - a boolean to determine if the model is training</li>
<li><code>encoder_mask</code> - the padding mask to be applied to the encoder</li>
<li><code>look_ahead_mask</code> - the look ahead mask to be applied to the decoder</li>
<li><code>decoder_mask</code> - the padding mask to be applied to the decoder</li>
<li>Returns: a tensor of shape <code>(batch, target_seq_len, target_vocab)</code> containing the transformer output</li>
</ul></li>
<li>You should use <code>Encoder = __import__('9-transformer_encoder').Encoder</code> and <code>Decoder = __import__('10-transformer_decoder').Decoder</code></li>
</ul>

<pre><code>$ cat 11-main.py
#!/usr/bin/env python3

import tensorflow as tf
Transformer = __import__('11-transformer').Transformer

transformer = Transformer(6, 512, 8, 2048, 10000, 12000, 1000, 1500)
print(transformer.encoder)
print(transformer.decoder)
print(transformer.linear)
x = tf.random.uniform((32, 10))
y = tf.random.uniform((32, 15))
output = transformer(x, y, True, None, None, None)
print(output)
$ ./11-main.py
&lt;9-transformer_encoder.Encoder object at 0xb2edc5128&gt;
&lt;10-transformer_decoder.Decoder object at 0xb2f412b38&gt;
&lt;tensorflow.python.keras.layers.core.Dense object at 0xb2fd68898&gt;
Tensor("transformer/dense_96/BiasAdd:0", shape=(32, 15, 12000), dtype=float32)
$
</code></pre>

<p><em>Ignore the Warning messages in the output</em></p>


<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x11-attention</code></li>
    <li>File: <code>11-transformer.py</code></li>
</ul>

</div>

</div>
</section>