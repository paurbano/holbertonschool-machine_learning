<h1 class="gap">0x12. Transformer Applications</h1>
<h2>Resources</h2>
<p><strong>Read or watch:</strong></p>
<ul>
<li><a href="/rltoken/jxxAqYmZVG_896LjsHA0SA" title="TFDS Overview" target="_blank">TFDS Overview</a></li>
<li><a href="/rltoken/3jhsMi8_VIZd2uzlyN-SaQ" title="TFDS Keras Example" target="_blank">TFDS Keras Example</a></li>
<li><a href="/rltoken/PBFAFa4q7sbMhLyrBg84Xg" title="Customizing what happens in fit" target="_blank">Customizing what happens in fit</a></li>
</ul>
<p><strong>References:</strong></p>
<ul>
<li><a href="/rltoken/_Sot-yIEG4acO7oABwji-Q" title="tfds" target="_blank">tfds</a>

<ul>
<li><a href="/rltoken/zlfIaVsEPgK3M-PFqYx8kw" title="tfds.load" target="_blank">tfds.load</a></li>
<li><a href="/rltoken/pVFn4RX89X8AK9l1CICrZw" title="tfds.features.text.SubwordTextEncoder" target="_blank">tfds.features.text.SubwordTextEncoder</a></li>
</ul></li>
<li><a href="/rltoken/C1R6GSnrg7By7F1ZozYALQ" title="tf.py_function" target="_blank">tf.py_function</a></li>
<li><a href="/rltoken/4EiwSWc51djgL5YL8CPyWw" title="tf.linalg.band_part" target="_blank">tf.linalg.band_part</a></li>
</ul>

<h3>General</h3>
<ul>
<li>How to use Transformers for Machine Translation</li>
<li>How to write a custom train/test loop in Keras</li>
<li>How to use Tensorflow Datasets</li>
</ul>
<h2>TF Datasets</h2>
<p>For machine translation, we will be using the prepared <a href="/rltoken/JpNiruFnMoCN2ElftkLWUw" title="Tensorflow Datasets" target="_blank">Tensorflow Datasets</a> <a href="/rltoken/w3kBudIiwPqWRxfTEld95g" title="ted_hrlr_translate/pt_to_en" target="_blank">ted_hrlr_translate/pt_to_en</a>  for English to Portuguese translation</p>
<p>To download Tensorflow Datasets, please use:</p>
<pre><code>pip install --user tensorflow-datasets
</code></pre>
<p>To use this dataset, we will have to use the Tensorflow 2.0 compat within Tensorflow 1.15 and download the content:</p>
<pre><code>$ cat load_dataset.py
#!/usr/bin/env python3
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()
pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
for pt, en in pt2en_train.take(1):
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))
$ ./load_dataset.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
$
</code></pre>
<h2 class="gap">Tasks</h2>
<div data-role="task7113" data-position="1">
              <div class=" clearfix gap" id="task-7113">
<span id="user_id" data-id="1283"></span>

</div>

</div>

  <h4 class="task">
    0. Dataset
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

<p>Create the class <code>Dataset</code> that loads and preps a dataset for machine translation:</p>

<ul>
<li>Class constructor <code>def __init__(self):</code>

<ul>
<li>creates the instance attributes:

<ul>
<li> <code>data_train</code>, which contains the <code>ted_hrlr_translate/pt_to_en</code> <code>tf.data.Dataset</code> <code>train</code> split, loaded <code>as_supervided</code></li>
<li> <code>data_valid</code>, which contains the <code>ted_hrlr_translate/pt_to_en</code> <code>tf.data.Dataset</code> <code>validate</code> split, loaded <code>as_supervided</code></li>
<li> <code>tokenizer_pt</code> is the Portuguese tokenizer created from the training set</li>
<li><code>tokenizer_en</code> is the English tokenizer created from the training set</li>
</ul></li>
</ul></li>
<li>Create the instance method <code>def tokenize_dataset(self, data):</code> that creates sub-word tokenizers for our dataset:

<ul>
<li><code>data</code> is a <code>tf.data.Dataset</code> whose examples are formatted as a tuple <code>(pt, en)</code>

<ul>
<li><code>pt</code> is the <code>tf.Tensor</code> containing the Portuguese sentence</li>
<li><code>en</code> is the <code>tf.Tensor</code> containing the corresponding English sentence</li>
</ul></li>
<li>The maximum vocab size should be set to <code>2**15</code></li>
<li>Returns: <code>tokenizer_pt, tokenizer_en</code>

<ul>
<li><code>tokenizer_pt</code> is the Portuguese tokenizer</li>
<li><code>tokenizer_en</code> is the English tokenizer</li>
</ul></li>
</ul></li>
</ul>

<pre><code>$ cat 0-main.py
#!/usr/bin/env python3

Dataset = __import__('0-dataset').Dataset
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
data = Dataset()
for pt, en in data.data_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
for pt, en in data.data_valid.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))
$ ./0-main.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
tinham comido peixe com batatas fritas ?
did they eat fish and chips ?
&lt;class 'tensorflow_datasets.core.features.text.subword_text_encoder.SubwordTextEncoder'&gt;
&lt;class 'tensorflow_datasets.core.features.text.subword_text_encoder.SubwordTextEncoder'&gt;
$
</code></pre>
  <!-- Task URLs -->
  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x12-transformer_apps</code></li>
    <li>File: <code>0-dataset.py</code></li>
</ul>

</div>
 
</div>

<div data-role="task7114" data-position="2">
              <div class=" clearfix gap" id="task-7114">
<span id="user_id" data-id="1283"></span>

</div>

<div data-role="task7115" data-position="3">
              <div class=" clearfix gap" id="task-7115">
<span id="user_id" data-id="1283"></span>

</div>

</div>

  <h4 class="task">
    2. TF Encode
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Update the class <code>Dataset</code>:</p>

<ul>
<li>Create the instance method <code>def tf_encode(self, pt, en):</code> that acts as a <code>tensorflow</code> wrapper for the <code>encode</code> instance method

<ul>
<li>Make sure to set the shape of the <code>pt</code> and <code>en</code> return tensors</li>
</ul></li>
<li>Update the class constructor <code>def __init__(self):</code>

<ul>
<li>update the <code>data_train</code> and <code>data_validate</code> attributes by tokenizing the examples</li>
</ul></li>
</ul>

<pre><code>$ cat 2-main.py
#!/usr/bin/env python3

Dataset = __import__('2-dataset').Dataset
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
data = Dataset()
print('got here')
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
$ ./2-main.py
tf.Tensor(
[30138     6    36 17925    13     3  3037     1  4880     3   387  2832
    18 18444     1     5     8     3 16679 19460   739     2 30139], shape=(23,), dtype=int64) tf.Tensor(
[28543     4    56    15  1266 20397 10721     1    15   100   125   352
     3    45  3066     6  8004     1    88    13 14859     2 28544], shape=(23,), dtype=int64)
tf.Tensor([30138   289 15409  2591    19 20318 26024 29997    28 30139], shape=(10,), dtype=int64) tf.Tensor([28543    93    25   907  1366     4  5742    33 28544], shape=(9,), dtype=int64)
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x12-transformer_apps</code></li>
    <li>File: <code>2-dataset.py</code></li>
</ul>

</div>
 
</div>

</div>

  <h4 class="task">
    1. Encode Tokens
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Update the class <code>Dataset</code>:</p>

<ul>
<li>Create the instance method <code>def encode(self, pt, en):</code> that encodes a translation into tokens:

<ul>
<li><code>pt</code> is the <code>tf.Tensor</code> containing the Portuguese sentence</li>
<li><code>en</code> is the <code>tf.Tensor</code> containing the corresponding English sentence</li>
<li>The tokenized sentences should include the start and end of sentence tokens</li>
<li>The start token should be indexed as <code>vocab_size</code></li>
<li>The end token should be indexed as <code>vocab_size + 1</code></li>
<li>Returns: <code>pt_tokens, en_tokens</code>

<ul>
<li><code>pt_tokens</code> is a <code>np.ndarray</code> containing the Portuguese tokens</li>
<li><code>en_tokens</code> is a <code>np.ndarray.</code> containing the English tokens</li>
</ul></li>
</ul></li>
</ul>

<pre><code>$ cat 1-main.py
#!/usr/bin/env python3

Dataset = __import__('1-dataset').Dataset
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
data = Dataset()
for pt, en in data.data_train.take(1):
    print(data.encode(pt, en))
for pt, en in data.data_valid.take(1):
    print(data.encode(pt, en))
$ ./1-main.py
([30138, 6, 36, 17925, 13, 3, 3037, 1, 4880, 3, 387, 2832, 18, 18444, 1, 5, 8, 3, 16679, 19460, 739, 2, 30139], [28543, 4, 56, 15, 1266, 20397, 10721, 1, 15, 100, 125, 352, 3, 45, 3066, 6, 8004, 1, 88, 13, 14859, 2, 28544])
([30138, 289, 15409, 2591, 19, 20318, 26024, 29997, 28, 30139], [28543, 93, 25, 907, 1366, 4, 5742, 33, 28544])
$
</code></pre>

  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x12-transformer_apps</code></li>
    <li>File: <code>1-dataset.py</code></li>
</ul>

</div>
 
</div>

<div data-role="task7116" data-position="4">
              <div class=" clearfix gap" id="task-7116">
<span id="user_id" data-id="1283"></span>

</div>
</div>

  <h4 class="task">
    3. Pipeline
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Update the class <code>Dataset</code> to set up the data pipeline:</p>

<ul>
<li>Update the class constructor <code>def __init__(self, batch_size, max_len):</code>

<ul>
<li><code>batch_size</code> is the batch size for training/validation</li>
<li><code>max_len</code> is the maximum number of tokens allowed per example sentence</li>
<li>update the <code>data_train</code> attribute by performing the following actions:

<ul>
<li>filter out all examples that have either sentence with more than <code>max_len</code> tokens</li>
<li>cache the dataset to increase performance</li>
<li>shuffle the entire dataset</li>
<li>split the dataset into padded batches of size <code>batch_size</code></li>
<li>prefetch the dataset using <code>tf.data.experimental.AUTOTUNE</code> to increase performance</li>
</ul></li>
<li>update the <code>data_validate</code> attribute by performing the following actions:

<ul>
<li>filter out all examples that have either sentence with more than <code>max_len</code> tokens</li>
<li>split the dataset into padded batches of size <code>batch_size</code></li>
</ul></li>
</ul></li>
</ul>

<pre><code>$ cat 3-main.py
#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)
data = Dataset(32, 40)
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
$ ./3-main.py
tf.Tensor(
[[30138  1029   104 ...     0     0     0]
 [30138    40     8 ...     0     0     0]
 [30138    12    14 ...     0     0     0]
 ...
 [30138    72 23483 ...     0     0     0]
 [30138  2381   420 ...     0     0     0]
 [30138     7 14093 ...     0     0     0]], shape=(32, 39), dtype=int64) tf.Tensor(
[[28543   831   142 ...     0     0     0]
 [28543    16    13 ...     0     0     0]
 [28543    19     8 ...     0     0     0]
 ...
 [28543    18    27 ...     0     0     0]
 [28543  2648   114 ... 28544     0     0]
 [28543  9100 19214 ...     0     0     0]], shape=(32, 37), dtype=int64)
tf.Tensor(
[[30138   289 15409 ...     0     0     0]
 [30138    86   168 ...     0     0     0]
 [30138  5036     9 ...     0     0     0]
 ...
 [30138  1157 29927 ...     0     0     0]
 [30138    33   837 ...     0     0     0]
 [30138   126  3308 ...     0     0     0]], shape=(32, 32), dtype=int64) tf.Tensor(
[[28543    93    25 ...     0     0     0]
 [28543    11    20 ...     0     0     0]
 [28543    11  2850 ...     0     0     0]
 ...
 [28543    11   406 ...     0     0     0]
 [28543     9   152 ...     0     0     0]
 [28543     4   272 ...     0     0     0]], shape=(32, 35), dtype=int64)
$
</code></pre>


  <!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x12-transformer_apps</code></li>
    <li>File: <code>3-dataset.py</code></li>
</ul>

</div>
 
</div>
<div data-role="task7117" data-position="5">
              <div class=" clearfix gap" id="task-7117">
<span id="user_id" data-id="1283"></span>

</div>

</div>

  <h4 class="task">
    4. Create Masks
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create the function <code>def create_masks(inputs, target):</code> that creates all masks for training/validation:</p>

<ul>
<li><code>inputs</code> is a tf.Tensor of shape <code>(batch_size, seq_len_in)</code> that contains the input sentence</li>
<li><code>target</code> is a tf.Tensor of shape <code>(batch_size, seq_len_out)</code> that contains the target sentence</li>
<li>This function should only use <code>tensorflow</code> operations in order to properly function in the training step</li>
<li>Returns: <code>encoder_mask</code>, <code>look_ahead_mask</code>, <code>decoder_mask</code>

<ul>
<li><code>encoder_mask</code> is the <code>tf.Tensor</code> padding mask of shape <code>(batch_size, 1, 1, seq_len_in)</code> to be applied in the encoder</li>
<li><code>look_ahead_mask</code> is the <code>tf.Tensor</code> look ahead mask of shape <code>(batch_size, 1, seq_len_out, seq_len_out)</code> to be applied in the decoder </li>
<li> <code>decoder_mask</code> is the <code>tf.Tensor</code> padding mask of shape <code>(batch_size, 1, 1, seq_len_in)</code> to be applied in the decoder</li>
</ul></li>
</ul>

<pre><code>$ cat 4-main.py
#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)
data = Dataset(32, 40)
for inputs, target in data.data_train.take(1):
    print(create_masks(inputs, target))
$ ./4-main.py
(&lt;tf.Tensor: id=414557, shape=(32, 1, 1, 39), dtype=float32, numpy=
array([[[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)&gt;, &lt;tf.Tensor: id=414589, shape=(32, 1, 37, 37), dtype=float32, numpy=
array([[[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 0., 1., 1.],
         [0., 0., 0., ..., 0., 1., 1.],
         [0., 0., 0., ..., 0., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)&gt;), &lt;tf.Tensor: id=414573, shape=(32, 1, 1, 39), dtype=float32, numpy=
array([[[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)&gt;
$
</code></pre>


<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x12-transformer_apps</code></li>
    <li>File: <code>4-create_masks.py</code></li>
    </ul>

</div>
 
</div>

<div data-role="task7118" data-position="6">
              <div class=" clearfix gap" id="task-7118">
<span id="user_id" data-id="1283"></span>

</div>
</div>

  <h4 class="task">
    5. Train
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Take your implementation of a transformer from our <a href="/rltoken/xFGAKD-jaUWnsvOXMTPcvw" title="previous project" target="_blank">previous project</a> and save it to the file <code>5-transformer.py</code>. Note, you may need to make slight adjustments to this model to get it to functionally train.</p>

<p>Write a the function <code>def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):</code> that creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset:</p>

<ul>
<li><code>N</code> the number of blocks in the encoder and decoder</li>
<li><code>dm</code> the dimensionality of the model</li>
<li><code>h</code> the number of heads</li>
<li><code>hidden</code> the number of hidden units in the fully connected layers</li>
<li><code>max_len</code> the maximum number of tokens per sequence</li>
<li><code>batch_size</code> the batch size for training</li>
<li><code>epochs</code> the number of epochs to train for</li>
<li>You should use the following imports:

<ul>
<li><code>Dataset = __import__('3-dataset').Dataset</code></li>
<li><code>create_masks = __import__('4-create_masks').create_masks</code></li>
<li><code>Transformer = __import__('5-transformer').Transformer</code></li>
</ul></li>
<li>Your model should be trained with Adam optimization with <code>beta_1=0.9</code>, <code>beta_2=0.98</code>, <code>epsilon=1e-9</code>

<ul>
<li>The learning rate should be scheduled using the following equation with <code>warmup_steps=4000</code>:</li>
<li><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/9/39ceb6fefc25283cd8ee7a3f302ae799b6051bcd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210112%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210112T211417Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=e728bc2933f23640e995d8934e3a23f7de278197aece94ec08969da069011f75" alt="" style=""></li>
</ul></li>
<li>Your model should use sparse categorical crossentropy loss, ignoring padded tokens</li>
<li>Your model should print the following information about the training:

<ul>
<li> Every 50 batches,  you should print <code>Epoch {Epoch number}, batch {batch_number}: loss {training_loss} accuracy {training_accuracy}</code></li>
<li>Every epoch, you should print <code>Epoch {Epoch number}: loss {training_loss} accuracy {training_accuracy}</code></li>
</ul></li>
<li>Returns the trained model</li>
</ul>

<pre><code>$ cat 5-main.py
#!/usr/bin/env python3
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)
transformer = train_transformer(4, 128, 8, 512, 32, 40, 2)
print(type(transformer))
$ ./5-main.py
Epoch 1, batch 0: loss 10.26855754852295 accuracy 0.0
Epoch 1, batch 50: loss 10.23129940032959 accuracy 0.0009087905054911971

...

Epoch 1, batch 1000: loss 7.164522647857666 accuracy 0.06743457913398743
Epoch 1, batch 1050: loss 7.076988220214844 accuracy 0.07054812461137772
Epoch 1: loss 7.038494110107422 accuracy 0.07192815840244293
Epoch 2, batch 0: loss 5.177524089813232 accuracy 0.1298387050628662
Epoch 2, batch 50: loss 5.189461708068848 accuracy 0.14023463428020477

...

Epoch 2, batch 1000: loss 4.870367527008057 accuracy 0.15659810602664948
Epoch 2, batch 1050: loss 4.858142375946045 accuracy 0.15731287002563477
Epoch 2: loss 4.852652549743652 accuracy 0.15768977999687195
&lt;class '5-transformer.Transformer'&gt;
$
</code></pre>

<p><em>Note: In this example, we only train for 2 epochs since the full training takes quite a long time. If you’d like to properly train your model, you’ll have to train for 20+ epochs</em></p>


  <!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x12-transformer_apps</code></li>
    <li>File: <code>5-transformer.py, 5-train.py</code></li>
</ul>

</div>
 
</div>
