<h1 class="gap">0x10. Natural Language Processing - Evaluation Metrics</h1>
<h2>Learning Objectives</h2>
<h3>General</h3>
<ul>
<li>What are the applications of natural language processing?</li>
<li>What is a BLEU score?</li>
<li>What is a ROUGE score?</li>
<li>What is perplexity?</li>
<li>When should you use one evaluation metric over another?</li>
</ul>
<h2 class="gap">Tasks</h2>
<section class="formatted-content">
            <div data-role="task5417" data-position="1">
              <div class=" clearfix gap" id="task-5417">
<span id="user_id" data-id="1283"></span>


  <h4 class="task">
    0. Unigram BLEU score
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write the function <code>def uni_bleu(references, sentence):</code> that calculates the unigram BLEU score for a sentence:</p>

<ul>
<li><code>references</code> is a list of reference translations

<ul>
<li>each reference translation is a list of the words in the translation</li>
</ul></li>
<li><code>sentence</code> is a list containing the model proposed sentence</li>
<li>Returns: the unigram BLEU score</li>
</ul>

<pre><code>$ cat 0-main.py
#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))
$ ./0-main.py
0.6549846024623855
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x10-nlp_metrics</code></li>
    <li>File: <code>0-uni_bleu.py</code></li>
</ul>

</div>
</div>
<span id="user_id" data-id="1283"></span>

  <h4 class="task">
    1. N-gram BLEU score
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write the function <code>def ngram_bleu(references, sentence, n):</code> that calculates the n-gram BLEU score for a sentence:</p>

<ul>
<li><code>references</code> is a list of reference translations

<ul>
<li>each reference translation is a list of the words in the translation</li>
</ul></li>
<li><code>sentence</code> is a list containing the model proposed sentence</li>
<li><code>n</code> is the size of the n-gram to use for evaluation</li>
<li>Returns: the n-gram BLEU score</li>
</ul>

<pre><code>$ cat 1-main.py
#!/usr/bin/env python3

ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(ngram_bleu(references, sentence, 2))
$ ./1-main.py
0.6140480648084865
$
</code></pre>


  <!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x10-nlp_metrics</code></li>
    <li>File: <code>1-ngram_bleu.py</code></li>
</ul>

</div>
 
</div>
<div data-role="task5419" data-position="3">
    <div class=" clearfix gap" id="task-5419">
<span id="user_id" data-id="1283"></span>

</div>

  <h4 class="task">
    2. Cumulative N-gram BLEU score
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write the function <code>def cumulative_bleu(references, sentence, n):</code> that calculates the cumulative n-gram BLEU score for a sentence:</p>

<ul>
<li><code>references</code> is a list of reference translations

<ul>
<li>each reference translation is a list of the words in the translation</li>
</ul></li>
<li><code>sentence</code> is a list containing the model proposed sentence</li>
<li><code>n</code> is the size of the largest n-gram to use for evaluation</li>
<li>All n-gram scores should be weighted evenly</li>
<li>Returns: the cumulative n-gram BLEU score</li>
</ul>

<pre><code>$ cat 2-main.py
#!/usr/bin/env python3

cumulative_bleu = __import__('1-cumulative_bleu').cumulative_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(cumulative_bleu(references, sentence, 4))
$ ./2-main.py
0.5475182535069453
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x10-nlp_metrics</code></li>
    <li>File: <code>2-cumulative_bleu.py</code></li>
</ul>


</div>

</div>
</section>