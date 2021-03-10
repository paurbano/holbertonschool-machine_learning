<h1 class="gap">0x00. Pandas</h1>

<h2>Learning Objectives</h2>
<p>At the end of this project, you are expected to be able to <a href="/rltoken/65LBg59o8_y9yFiZryOidQ" title="explain to anyone" target="_blank">explain to anyone</a>, <strong>without the help of Google</strong>:</p>

<h3>General</h3>
<ul>
<li>What is <code>pandas</code>?</li>
<li>What is a <code>pd.DataFrame</code>? How do you create one?</li>
<li>What is a <code>pd.Series</code>? How do you create one?</li>
<li>How to load data from a file</li>
<li>How to perform indexing on a <code>pd.DataFrame</code></li>
<li>How to use hierarchical indexing with a <code>pd.DataFrame</code></li>
<li>How to slice a <code>pd.DataFrame</code></li>
<li>How to reassign columns</li>
<li>How to sort a <code>pd.DataFrame</code></li>
<li>How to use boolean logic with a <code>pd.DataFrame</code></li>
<li>How to merge/concatenate/join <code>pd.DataFrame</code>s</li>
<li>How to get statistical information from a <code>pd.DataFrame</code></li>
<li>How to visualize a <code>pd.DataFrame</code></li>
</ul>

<h2>Download Pandas 0.24.x</h2>
<pre><code>pip install --user pandas
</code></pre>
<h2>Datasets</h2>
<p>For this project, we will be using the <a href="/rltoken/sHhO6vV0SMvlZgp9ol9EZw" title="coinbase" target="_blank">coinbase</a> and <a href="/rltoken/Lp3j65_o9UW6OoEQTScNzA" title="bitstamp" target="_blank">bitstamp</a> datasets, as seen previously in <a href="/rltoken/F3p6aIUM2SM9khIMs6rpXQ" title="0x0E. Time Series Forecasting" target="_blank">0x0E. Time Series Forecasting</a></p>

<h2 class="gap">Tasks</h2>
<div data-role="task7044" data-position="1" id="task-num-0">
        <div class="panel panel-default task-card " id="task-7044">
  <span id="user_id" data-id="1283"></span>

  <div class="panel-heading">
    <h3 class="panel-title">
      0. From Numpy
    </h3>
  </div>

  <div class="panel-body">
    <span id="user_id" data-id="1283"></span>

    

<!-- Progress vs Score -->

<!-- Task Body -->
<p>Write a function <code>def from_numpy(array):</code> that creates a <code>pd.DataFrame</code> from a <code>np.ndarray</code>:</p>

<ul>
<li><code>array</code> is the <code>np.ndarray</code> from which you should create the <code>pd.DataFrame</code></li>
<li>The columns of the <code>pd.DataFrame</code> should be labeled in alphabetical order and capitalized.  There will not be more than 26 columns.</li>
<li>Returns: the newly created <code>pd.DataFrame</code></li>
</ul>

<pre><code>$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
from_numpy = __import__('0-from_numpy').from_numpy

np.random.seed(0)
A = np.random.randn(5, 8)
print(from_numpy(A))
B = np.random.randn(9, 3)
print(from_numpy(B))
$ ./0-main.py
          A         B         C         D         E         F         G         H
0  1.764052  0.400157  0.978738  2.240893  1.867558 -0.977278  0.950088 -0.151357
1 -0.103219  0.410599  0.144044  1.454274  0.761038  0.121675  0.443863  0.333674
2  1.494079 -0.205158  0.313068 -0.854096 -2.552990  0.653619  0.864436 -0.742165
3  2.269755 -1.454366  0.045759 -0.187184  1.532779  1.469359  0.154947  0.378163
4 -0.887786 -1.980796 -0.347912  0.156349  1.230291  1.202380 -0.387327 -0.302303
          A         B         C
0 -1.048553 -1.420018 -1.706270
1  1.950775 -0.509652 -0.438074
2 -1.252795  0.777490 -1.613898
3 -0.212740 -0.895467  0.386902
4 -0.510805 -1.180632 -0.028182
5  0.428332  0.066517  0.302472
6 -0.634322 -0.362741 -0.672460
7 -0.359553 -0.813146 -1.726283
8  0.177426 -0.401781 -1.630198
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
    <li>Directory: <code>pipeline/0x00-pandas</code></li>
    <li>File: <code>0-from_numpy.py</code></li>
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

<div data-role="task7045" data-position="2" id="task-num-1">
        <div class="panel panel-default task-card " id="task-7045">
  <span id="user_id" data-id="1283"></span>

  <div class="panel-heading">
    <h3 class="panel-title">
      1. From Dictionary
    </h3>
  </div>

  <div class="panel-body">
    <span id="user_id" data-id="1283"></span>

    

<!-- Progress vs Score -->

<!-- Task Body -->
<p>Write a python script that created a <code>pd.DataFrame</code> from a dictionary:</p>

<ul>
<li>The first column should be labeled <code>First</code> and have the values <code>0.0</code>, <code>0.5</code>, <code>1.0</code>, and <code>1.5</code></li>
<li>The second column should be labeled <code>Second</code> and have the values <code>one</code>, <code>two</code>, <code>three</code>, <code>four</code></li>
<li>The rows should be labeled <code>A</code>, <code>B</code>, <code>C</code>, and <code>D</code>, respectively</li>
<li>The <code>pd.DataFrame</code> should be saved into the variable <code>df</code></li>
</ul>

<pre><code>$ cat 1-main.py
#!/usr/bin/env python3

df = __import__('1-from_dictionary').df

print(df)
$ ./1-main.py
   First Second
A    0.0    one
B    0.5    two
C    1.0  three
D    1.5   four
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
    <li>Directory: <code>pipeline/0x00-pandas</code></li>
    <li>File: <code>1-from_dictionary.py</code></li>
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

<!-- tASK2-->

<div data-role="task7046" data-position="3" id="task-num-2">
        <div class="panel panel-default task-card " id="task-7046">
  <span id="user_id" data-id="1283"></span>

  <div class="panel-heading">
    <h3 class="panel-title">
      2. From File
    </h3>

  </div>

  <div class="panel-body">
    <span id="user_id" data-id="1283"></span>

    

<!-- Progress vs Score -->

<!-- Task Body -->
<p>Write a function <code>def from_file(filename, delimiter):</code> that loads data from a file as a <code>pd.DataFrame</code>:</p>

<ul>
<li><code>filename</code> is the file to load from</li>
<li><code>delimiter</code> is the column separator</li>
<li>Returns: the loaded <code>pd.DataFrame</code></li>
</ul>

<pre><code>$ cat 2-main.py
#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
print(df1.head())
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')
print(df2.tail())
$ ./2-main.py
    Timestamp   Open   High    Low  Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
0  1417411980  300.0  300.0  300.0  300.0          0.01                3.0           300.0
1  1417412040    NaN    NaN    NaN    NaN           NaN                NaN             NaN
2  1417412100    NaN    NaN    NaN    NaN           NaN                NaN             NaN
3  1417412160    NaN    NaN    NaN    NaN           NaN                NaN             NaN
4  1417412220    NaN    NaN    NaN    NaN           NaN                NaN             NaN
          Timestamp     Open     High      Low    Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
4363452  1587513360  6847.97  6856.35  6847.97  6856.35      0.125174         858.128697     6855.498790
4363453  1587513420  6850.23  6856.13  6850.23  6850.89      1.224777        8396.781459     6855.763449
4363454  1587513480  6846.50  6857.45  6846.02  6857.45      7.089168       48533.089069     6846.090966
4363455  1587513540  6854.18  6854.98  6854.18  6854.98      0.012231          83.831604     6854.195090
4363456  1587513600  6850.60  6850.60  6850.60  6850.60      0.014436          98.896906     6850.600000
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
    <li>Directory: <code>pipeline/0x00-pandas</code></li>
    <li>File: <code>2-from_file.py</code></li>
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
<div data-role="task7047" data-position="4" id="task-num-3">
        <div class="panel panel-default task-card " id="task-7047">
  <span id="user_id" data-id="1283"></span>

  <div class="panel-heading">
    <h3 class="panel-title">
      3. Rename
    </h3>
  </div>

  <div class="panel-body">
    <span id="user_id" data-id="1283"></span>

<!-- Progress vs Score -->

<!-- Task Body -->
<p>Complete the script below to perform the following:</p>

<ul>
<li>Rename the column <code>Timestamp</code> to <code>Datetime</code></li>
<li>Convert the timestamp values to datatime values</li>
<li>Display only the  <code>Datetime</code> and <code>Close</code> columns</li>
</ul>

<pre><code>$ cat 3-rename.py
#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

print(df.tail())
$ ./3-rename.py
                   Datetime    Close
2099755 2019-01-07 22:02:00  4006.01
2099756 2019-01-07 22:03:00  4006.01
2099757 2019-01-07 22:04:00  4006.01
2099758 2019-01-07 22:05:00  4005.50
2099759 2019-01-07 22:06:00  4005.99
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
    <li>Directory: <code>pipeline/0x00-pandas</code></li>
    <li>File: <code>3-rename.py</code></li>
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