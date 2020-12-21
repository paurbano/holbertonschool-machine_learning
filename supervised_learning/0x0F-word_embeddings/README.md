<h1 class="gap">0x0F. Natural Language Processing - Word Embeddings</h1>

<h3>General</h3>
<ul>
<li>What is natural language processing?</li>
<li>What is a word embedding?</li>
<li>What is bag of words?</li>
<li>What is TF-IDF?</li>
<li>What is CBOW?</li>
<li>What is a skip-gram?</li>
<li>What is an n-gram?</li>
<li>What is negative sampling?</li>
<li>What is word2vec, GloVe, fastText, ELMo?</li>
</ul>

<h2>Download Gensim 3.8.x</h2>
<pre><code>pip install --user gensim==3.8</code></pre>

<h2>Download Keras 2.2.5</h2>
<pre><code>pip install --user keras==2.2.5</code></pre>

<h2 class="gap">Tasks</h2>
<section class="formatted-content">
            <div data-role="task5397" data-position="1">
              <div class=" clearfix gap" id="task-5397">
<span id="user_id" data-id="1283"></span>

  <h4 class="task">
    0. Bag Of Words
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>


  <!-- Task Body -->
  <p>Write a function <code>def bag_of_words(sentences, vocab=None):</code> that creates a bag of words embedding matrix:</p>

<ul>
<li><code>sentences</code> is a list of sentences to analyze</li>
<li><code>vocab</code> is a list of the vocabulary words to use for the analysis

<ul>
<li>If <code>None</code>, all words within <code>sentences</code> should be used</li>
</ul></li>
<li>Returns: <code>embeddings, features</code>

<ul>
<li><code>embeddings</code> is a <code>numpy.ndarray</code> of shape <code>(s, f)</code> containing the embeddings

<ul>
<li><code>s</code> is the number of sentences in <code>sentences</code></li>
<li><code>f</code> is the number of features analyzed</li>
</ul></li>
<li><code>features</code> is a list of the features used for <code>embeddings</code></li>
</ul></li>
</ul>

<pre><code>$ cat 0-main.py
#!/usr/bin/env python3

bag_of_words = __import__('0-bag_of_words').bag_of_words

sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]
E, F = bag_of_words(sentences)
print(E)
print(F)
$ ./0-main.py
[[0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0]
 [1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
 [1 0 0 0 2 0 0 1 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0]
 [0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1]
 [0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1 1 1]
 [0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0]]
['are', 'awesome', 'beautiful', 'cake', 'children', 'future', 'good', 'grandchildren', 'holberton', 'is', 'learning', 'life', 'machine', 'nlp', 'no', 'not', 'one', 'our', 'said', 'school', 'that', 'the', 'very', 'was']
$
</code></pre>
<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x0F-word_embeddings</code></li>
    <li>File: <code>0-bag_of_words.py</code></li>
</ul>

</div>
</div>
</div>
<div data-role="task5398" data-position="2">
    <div class=" clearfix gap" id="task-5398">
<span id="user_id" data-id="1283"></span>
  <h4 class="task">
    1. TF-IDF
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

<!-- Task Body -->
<p>Write a function <code>def tf_idf(sentences, vocab=None):</code> that creates a TF-IDF embedding:</p>

<ul>
<li><code>sentences</code> is a list of sentences to analyze</li>
<li><code>vocab</code> is a list of the vocabulary words to use for the analysis

<ul>
<li>If <code>None</code>, all words within <code>sentences</code> should be used</li>
</ul></li>
<li>Returns: <code>embeddings, features</code>

<ul>
<li><code>embeddings</code> is a <code>numpy.ndarray</code> of shape <code>(s, f)</code> containing the embeddings

<ul>
<li><code>s</code> is the number of sentences in <code>sentences</code></li>
<li><code>f</code> is the number of features analyzed</li>
</ul></li>
<li><code>features</code> is a list of the features used for <code>embeddings</code></li>
</ul></li>
</ul>

<pre><code>$ cat 1-main.py
#!/usr/bin/env python3

tf_idf = __import__('1-tf_idf').tf_idf

sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]
vocab = ["awesome", "learning", "children", "cake", "good", "none", "machine"]
E, F = tf_idf(sentences, vocab)
print(E)
print(F)
$ ./1-main.py
[[1.         0.         0.         0.         0.         0.
  0.        ]
 [0.5098139  0.60831315 0.         0.         0.         0.
  0.60831315]
 [0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         1.         0.         0.         0.
  0.        ]
 [0.         0.         1.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.70710678 0.70710678 0.
  0.        ]
 [0.         0.         0.         0.70710678 0.70710678 0.
  0.        ]
 [0.         0.         0.         0.         0.         0.
  0.        ]]
['awesome', 'learning', 'children', 'cake', 'good', 'none', 'machine']
$</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x0F-word_embeddings</code></li>
    <li>File: <code>1-tf_idf.py</code></li>
</ul>

</div>
</div>
 
</div>
<div data-role="task5399" data-position="3">
    <div class=" clearfix gap" id="task-5399">
<span id="user_id" data-id="1283"></span>

  <h4 class="task">
    2. Train Word2Vec
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):</code> that creates and trains a <code>gensim</code> <code>word2vec</code> model:</p>

<ul>
<li><code>sentences</code> is a list of sentences to be trained on</li>
<li><code>size</code> is the dimensionality of the embedding layer</li>
<li><code>min_count</code> is the minimum number of occurrences of a word for use in training</li>
<li><code>window</code> is the maximum distance between the current and predicted word within a sentence</li>
<li><code>negative</code> is the size of negative sampling</li>
<li><code>cbow</code> is a boolean to determine the training type; <code>True</code> is for CBOW; <code>False</code> is for Skip-gram</li>
<li><code>iterations</code> is the number of iterations to train over</li>
<li><code>seed</code> is the seed for the random number generator</li>
<li><code>workers</code> is the number of worker threads to train the model</li>
<li>Returns: the trained model</li>
</ul>

<pre><code>$ cat 2-main.py
#!/usr/bin/env python3

from gensim.test.utils import common_texts
word2vec_model = __import__('2-word2vec').word2vec_model

print(common_texts[:2])
w2v = word2vec_model(common_texts, min_count=1)
print(w2v.wv["computer"])
$ ./2-main.py
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
[-3.0043968e-03  1.5343886e-03  4.0832465e-03  3.7239199e-03
  4.9583608e-04  4.8461729e-03 -1.0620747e-03  8.2803884e-04
  9.7367732e-04 -6.7797926e-05 -1.5526683e-03  1.8058836e-03
 -4.3851901e-03  4.7258494e-04  2.8616134e-03 -2.2246949e-03
  2.7494587e-03 -3.5267104e-03  3.0259083e-03  2.7240592e-03
  2.6110576e-03 -4.5409841e-03  4.9135066e-03  8.2884904e-04
  2.7018311e-03  1.5654180e-03 -1.5859824e-03  9.3057036e-04
  3.7275942e-03 -3.6502020e-03  2.8285771e-03 -4.2384453e-03
  3.2712172e-03 -1.9101484e-03 -1.8624340e-03 -5.6956144e-04
 -1.5617535e-03 -2.3851227e-03 -1.4313431e-05 -4.3398165e-03
  3.9115595e-03 -3.0616210e-03  1.7589398e-03 -3.4103722e-03
  4.7280011e-03  1.9380470e-03 -3.3873315e-03  8.4065803e-04
  2.6089977e-03  1.7012059e-03 -2.7421617e-03 -2.2240754e-03
 -5.3690566e-04  2.9577864e-03  2.3726511e-03  3.2704175e-03
  2.0853498e-03 -1.1927494e-03 -2.1565862e-03 -9.0970926e-04
 -2.8641665e-04 -3.4961947e-03  1.1104723e-03  1.2320089e-03
 -5.9017556e-04 -3.0594901e-03  3.6974431e-03 -1.8557351e-03
 -3.8218759e-03  9.2711346e-04 -4.3113795e-03 -4.4118706e-03
  4.7748778e-03 -4.5557776e-03 -2.2665847e-03 -8.2379003e-04
 -7.9581753e-04 -1.3048936e-03  1.9261248e-03  3.1299898e-03
 -1.9034051e-03 -2.0335305e-03 -2.6451424e-03  1.7377195e-03
  6.7217485e-04 -2.4134698e-03  4.3735080e-03 -3.2599240e-03
 -2.2431149e-03  4.4288361e-03  1.4923669e-04 -2.2144278e-03
 -8.9370424e-04 -2.7281314e-04 -1.7176758e-03  1.2485087e-03
  1.3230384e-03  1.7001784e-04  3.5425189e-03 -1.7469387e-04]
$</code></pre>

<p><em>Note: gensim is not inherently deterministic and therefore your outputs may vary</em></p>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x0F-word_embeddings</code></li>
    <li>File: <code>2-word2vec.py</code></li>
</ul>

</div>

</div>


</div>

</div>
<div data-role="task5400" data-position="4">
    <div class=" clearfix gap" id="task-5400">
<span id="user_id" data-id="1283"></span>
  <h4 class="task">
    3. Extract Word2Vec
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def gensim_to_keras(model):</code> that converts a <code>gensim</code> <code>word2vec</code> model to a <code>keras</code> Embedding layer:</p>

<ul>
<li><code>model</code> is a trained <code>gensim</code> <code>word2vec</code> models</li>
<li>Returns: the trainable <code>keras</code> Embedding</li>
</ul>

<pre><code>$ cat 3-main.py
#!/usr/bin/env python3

from gensim.test.utils import common_texts
word2vec_model = __import__('2-word2vec').word2vec_model
gensim_to_keras = __import__('3-gensim_to_keras').gensim_to_keras

print(common_texts[:2])
w2v = word2vec_model(common_texts, min_count=1)
print(gensim_to_keras(w2v))
$ ./3-main.py
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
Using TensorFlow backend.
&lt;keras.layers.embeddings.Embedding object at 0x7f72e2c1bd30&gt;
$</code></pre>


  <!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x0F-word_embeddings</code></li>
    <li>File: <code>3-gensim_to_keras.py</code></li>
</ul>

</div>

</div>


</div>
 
</div>
<div data-role="task5401" data-position="5">
    <div class=" clearfix gap" id="task-5401">
<span id="user_id" data-id="1283"></span>

<h4 class="task">
    4. FastText
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1):</code> that creates and trains a <code>genism</code> <code>fastText</code> model:</p>

<ul>
<li><code>sentences</code> is a list of sentences to be trained on</li>
<li><code>size</code> is the dimensionality of the embedding layer</li>
<li><code>min_count</code> is the minimum number of occurrences of a word for use in training</li>
<li><code>window</code> is the maximum distance between the current and predicted word within a sentence</li>
<li><code>negative</code> is the size of negative sampling</li>
<li><code>cbow</code> is a boolean to determine the training type; <code>True</code> is for CBOW; <code>False</code> is for Skip-gram</li>
<li><code>iterations</code> is the number of iterations to train over</li>
<li><code>seed</code> is the seed for the random number generator</li>
<li><code>workers</code> is the number of worker threads to train the model</li>
<li>Returns: the trained model</li>
</ul>

<pre><code>$ cat 4-main.py
#!/usr/bin/env python3

from gensim.test.utils import common_texts
fasttext_model = __import__('4-fasttext').fasttext_model

print(common_texts[:2])
ft = fasttext_model(common_texts, min_count=1)
print(ft.wv["computer"])
$ ./4-main.py
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
[-2.3464665e-03 -1.4542247e-04 -3.9549544e-05 -1.5817649e-03
 -2.1579072e-03  4.5148263e-04  9.9494774e-04  3.2517681e-05
  1.7035202e-04  6.8571279e-04 -2.0803163e-04  5.3083687e-04
  1.2990861e-03  3.5418154e-04  2.1087916e-03  1.1022155e-03
  6.2364555e-04  1.8612258e-05  1.8982493e-05  1.3051173e-03
 -6.0260214e-04  1.6334689e-03 -1.0172457e-06  1.4247939e-04
  1.1081318e-04  1.8327738e-03 -3.3656979e-04 -3.7365756e-04
  8.0635358e-04 -1.2945861e-04 -1.1031038e-04  3.4695750e-04
 -2.1932719e-04  1.4800908e-03  7.7851227e-04  8.6328381e-04
 -9.7545242e-04  6.0775197e-05  7.1560958e-04  3.6474539e-04
  3.3428212e-05 -1.0499550e-03 -1.2412234e-03 -1.8492664e-04
 -4.8664736e-04  1.9178988e-04 -6.3863385e-04  3.3325219e-04
 -1.5724128e-03  1.0003068e-03  1.7905374e-04  7.8452297e-04
  1.2625050e-04  8.1183662e-04 -4.9907330e-04  1.0475471e-04
  1.4351985e-03  4.9145994e-05 -1.4620423e-03  3.1466845e-03
  2.0059240e-05  1.6659468e-03 -4.3319576e-04  1.3077060e-03
 -2.0228853e-03  5.7626975e-04 -1.4056480e-03 -4.2292831e-04
  6.4076332e-04 -8.5614284e-04  1.9028617e-04  6.0735084e-04
  2.6121829e-04 -1.0566596e-03  1.0602509e-03  1.2843860e-03
  7.9715136e-04  2.8305652e-04  1.9187009e-04 -1.0519206e-03
 -8.2213630e-04 -2.1762338e-04 -1.7580058e-04  1.2764390e-04
 -1.5695200e-03  1.3364316e-03 -1.5765150e-03  1.4802803e-03
  1.5476452e-03  2.1928034e-04 -9.3281898e-04  3.2964293e-04
 -1.0146293e-03 -1.3567278e-03  1.8070930e-03 -4.2649341e-04
 -1.9074128e-03  7.1639987e-04 -1.3686880e-03  3.7073060e-03]
$
</code></pre>

<p><em>Note: gensim is not inherently deterministic and therefore your outputs may vary</em></p>
<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x0F-word_embeddings</code></li>
    <li>File: <code>4-fasttext.py</code></li>
</ul>
</div>
</div>

</div>

</div>
<div data-role="task5402" data-position="6">
    <div class=" clearfix gap" id="task-5402">
<span id="user_id" data-id="1283"></span>

  <h4 class="task">
    5. ELMo
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

   <!-- Task Body -->
  <p>When training an ELMo embedding model, you are training:</p>

<ol>
<li>The internal weights of the BiLSTM</li>
<li>The character embedding layer</li>
<li>The weights applied to the hidden states</li>
</ol>

<p>In the text file <code>5-elmo</code>, write the letter answer, followed by a newline, that lists the correct statements:</p>

<ul>
<li>A. 1, 2, 3</li>
<li>B. 1, 2</li>
<li>C. 2, 3</li>
<li>D. 1, 3</li>
<li>E. 1</li>
<li>F. 2</li>
<li>G. 3</li>
<li>H. None of the above</li>
</ul>


<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x0F-word_embeddings</code></li>
    <li>File: <code>5-elmo</code></li>
</ul>
</div>

</div>


</div>
 
</div>
</section>
