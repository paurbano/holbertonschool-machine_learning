<h1 class="gap">0x13. QA Bot</h1>

<h3>General</h3>
<ul>
<li>What is Question-Answering?</li>
<li>What is Semantic Search?</li>
<li>What is BERT?</li>
<li>How to develop a QA chatbot</li>
<li>How to use the <code>transformers</code> library</li>
<li>How to use the <code>tensorflow-hub</code> library</li>
</ul>

<h2>Upgrade to Tensorflow 2.3</h2>
<p><code>pip install --user tensorflow==2.3</code></p>

<h2>Install Tensorflow Hub</h2>
<p><code>pip install --user tensorflow-hub</code></p>

<h2>Install Transformers</h2>
<p><code>pip install --user transformers</code></p>

<h2>Zendesk Articles</h2>
<p>For this project, we will be using a collection of Holberton USA Zendesk Articles, <a href="/rltoken/ujQQ_o24BufzUPyWrR7IGA" title="ZendeskArticles.zip" target="_blank">ZendeskArticles.zip</a>.</p>

<h2 class="gap">Tasks</h2>
<section class="formatted-content">
            <div data-role="task7158" data-position="1">
              <div class=" clearfix gap" id="task-7158">
<span id="user_id" data-id="1283"></span>

</div>


</div>

  <h4 class="task">
    0. Question Answering
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def question_answer(question, reference):</code> that finds a snippet of text within a reference document to answer a question:</p>

<ul>
<li><code>question</code> is a string containing the question to answer</li>
<li><code>reference</code> is a string containing the reference document from which to find the answer</li>
<li>Returns: a string containing the answer</li>
<li>If no answer is found, return <code>None</code></li>
<li>Your function should use the  <code>bert-uncased-tf2-qa</code> model from the <code>tensorflow-hub</code> library</li>
<li>Your function should use the pre-trained <code>BertTokenizer</code>, <code>bert-large-uncased-whole-word-masking-finetuned-squad</code>, from the <code>transformers</code> library</li>
</ul>

<pre><code>$ cat 0-main.py
#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
$ ./0-main.py
from 9 : 00 am to 3 : 00 pm
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x13-qa_bot</code></li>
    <li>File: <code>0-qa.py</code></li>
</ul>
</div>
 
</div>
<div data-role="task7159" data-position="2">
    <div class=" clearfix gap" id="task-7159">
<span id="user_id" data-id="1283"></span>

</div>


</div>

  <h4 class="task">
    1. Create the loop
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Create a script that takes in input from the user with the prompt <code>Q:</code> and prints <code>A:</code> as a response. If the user inputs <code>exit</code>, <code>quit</code>, <code>goodbye</code>, or <code>bye</code>, case insensitive, print <code>A: Goodbye</code> and exit.</p>

<pre><code>$ ./1-loop.py
Q: Hello
A:
Q: How are you?
A:
Q: BYE
A: Goodbye
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>0x13-qa_bot</code></li>
    <li>File: <code>1-loop.py</code></li>
</ul>

</div>
 
</div>
<div data-role="task7160" data-position="3">
    <div class=" clearfix gap" id="task-7160">
<span id="user_id" data-id="1283"></span>

</div>


 </div>

  <h4 class="task">
    2. Answer Questions
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Based on the previous tasks, write a function <code>def answer_loop(reference):</code> that answers questions from a reference text:</p>

<ul>
<li><code>reference</code> is the reference text</li>
<li>If the answer cannot be found in the reference text, respond with <code>Sorry, I do not understand your question.</code></li>
</ul>

<pre><code>$ cat 2-main.py
#!/usr/bin/env python3

answer_loop = __import__('2-qa').answer_loop

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

answer_loop(reference)
$ ./2-main.py
Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: Sorry, I do not understand your question.
Q: What does PLD stand for?
A: peer learning days
Q: EXIT
A: Goodbye
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x13-qa_bot</code></li>
    <li>File: <code>2-qa.py</code></li>
</ul>
</div>
 
</div>
<div data-role="task7161" data-position="4">
    <div class=" clearfix gap" id="task-7161">
<span id="user_id" data-id="1283"></span>

</div>


   </div>

  <h4 class="task">
    3. Semantic Search
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def semantic_search(corpus_path, sentence):</code> that performs semantic search on a corpus of documents:</p>

<ul>
<li><code>corpus_path</code> is the path to the corpus of reference documents on which to perform semantic search</li>
<li><code>sentence</code> is the sentence from which to perform semantic search</li>
<li>Returns: the reference text of the document most similar to <code>sentence</code></li>
</ul>

<pre><code>$ cat 3-main.py
#!/usr/bin/env python3

semantic_search = __import__('3-semantic_search').semantic_search

print(semantic_search('ZendeskArticles', 'When are PLDs?'))
$ ./ 3-main.py
PLD Overview
Peer Learning Days (PLDs) are a time for you and your peers to ensure that each of you understands the concepts you've encountered in your projects, as well as a time for everyone to collectively grow in technical, professional, and soft skills. During PLD, you will collaboratively review prior projects with a group of cohort peers.
PLD Basics
PLDs are mandatory on-site days from 9:00 AM to 3:00 PM. If you cannot be present or on time, you must use a PTO. 
No laptops, tablets, or screens are allowed until all tasks have been whiteboarded and understood by the entirety of your group. This time is for whiteboarding, dialogue, and active peer collaboration. After this, you may return to computers with each other to pair or group program. 
Peer Learning Days are not about sharing solutions. This doesn't empower peers with the ability to solve problems themselves! Peer learning is when you share your thought process, whether through conversation, whiteboarding, debugging, or live coding. 
When a peer has a question, rather than offering the solution, ask the following:
"How did you come to that conclusion?"
"What have you tried?"
"Did the man page give you a lead?"
"Did you think about this concept?"
Modeling this form of thinking for one another is invaluable and will strengthen your entire cohort.
Your ability to articulate your knowledge is a crucial skill and will be required to succeed during technical interviews and through your career. 
$
</code></pre>


  <!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x13-qa_bot</code></li>
    <li>File: <code>3-semantic_search.py</code></li>
</ul>
</div>
 
</div>
<span id="user_id" data-id="1283"></span>

</div>

</div>

  <h4 class="task">
    4. Multi-reference Question Answering
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Based on the previous tasks, write a function <code>def question_answer(coprus_path):</code> that answers questions from multiple reference texts:</p>

<ul>
<li><code>corpus_path</code> is the path to the corpus of reference documents</li>
</ul>

<pre><code>$ cat 4-main.py
#!/usr/bin/env python3

qa_bot = __import__('4-qa').qa_bot

qa_bot('ZendeskArticles')
$ ./4-main.py
Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: help you train for technical interviews
Q: What does PLD stand for?
A: peer learning days
Q: goodbye
A: Goodbye
$
</code></pre>


  <!-- Task URLs -->

  <!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
<li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
<li>Directory: <code>supervised_learning/0x13-qa_bot</code></li>
<li>File: <code>4-qa.py</code></li>
</ul>
</div>
 
</div>
</section>