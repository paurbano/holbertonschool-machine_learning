<h1 class="gap">0x0E. Time Series Forecasting</h1>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/7/3b16b59e54876f2cc4fe9dcf887ac40585057e2c.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201213%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20201213T152857Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=77446af6906bbcfeafbb2cdde79a7c551e7fbec26bfa5af47145ea20e2db4ed8" alt="" style=""></p>

<h2>Resources</h2>
<p><strong>Read or watch:</strong></p>
<ul>
<li><a href="/rltoken/HmkmzkQ7_A-h5KKzFQ_tJg" title="Time Series Prediction" target="_blank">Time Series Prediction</a></li>
<li><a href="/rltoken/_QoRZ53rwY7yYVV2SM3frw" title="Time Series Forecasting" target="_blank">Time Series Forecasting</a></li>
<li><a href="/rltoken/jLo-utlk8pzUzIMRbOJAPA" title="Time Series Talk : Stationarity" target="_blank">Time Series Talk : Stationarity</a></li>
<li><a href="/rltoken/ulRRdAVAZr2KYM2ghlBRNQ" title="tf.data: Build TensorFlow input pipelines" target="_blank">tf.data: Build TensorFlow input pipelines</a></li>
<li><a href="/rltoken/7H-EjwlfVHGCoWHDCjIU-g" title="Tensorflow Datasets" target="_blank">Tensorflow Datasets</a></li>
</ul>

<p><strong>Definitions to skim</strong></p>
<ul>
<li><a href="/rltoken/eDzuZndaRfiXvecn4KvoHQ" title="Time Series" target="_blank">Time Series</a></li>
<li><a href="/rltoken/JN26Hp5uM1OgIPUkF1gsYA" title="Stationary Process" target="_blank">Stationary Process</a></li>
</ul>

<p><strong>References:</strong></p>
<ul>
<li><a href="/rltoken/1aM6PvPAN3kdBtvLB_hnrw" title="tf.keras.layers.SimpleRNN" target="_blank">tf.keras.layers.SimpleRNN</a></li>
<li><a href="/rltoken/PUtluakWAs8wcw3rsmYJ2A" title="tf.keras.layers.GRU" target="_blank">tf.keras.layers.GRU</a></li>
<li><a href="/rltoken/0Cocg6XxDqjxeAUKYQLhGg" title="tf.keras.layers.LSTM" target="_blank">tf.keras.layers.LSTM</a></li>
<li><a href="/rltoken/Wzagcu07guZFjx88UTmIBA" title="tf.data.Dataset" target="_blank">tf.data.Dataset</a></li>
</ul>

<h2>Learning Objectives</h2>
<h3>General</h3>
<ul>
<li>What is time series forecasting?</li>
<li>What is a stationary process?</li>
<li>What is a sliding window?</li>
<li>How to preprocess time series data</li>
<li>How to create a data pipeline in tensorflow for time series data</li>
<li>How to perform time series forecasting with RNNs in tensorflow</li>
</ul>

<h2 class="gap">Tasks</h2>

<section class="formatted-content">
            <div data-role="task5395" data-position="1">
              <div class=" clearfix gap" id="task-5395">
<span id="user_id" data-id="1283"></span>

<h4 class="task">
0. When to Invest
</h4>

<!-- Progress vs Score -->

<!-- Task Body -->
<p>Bitcoin (BTC) became a trending topic after its <a href="/rltoken/vjTWl4bomgHoPdlYDGJM0w" title="price" target="_blank">price</a> peaked in 2018. Many have sought to predict its value in order to accrue wealth. Let’s attempt to use our knowledge of RNNs to attempt just that.</p>

<p>Given the <a href="/rltoken/_-9LQxYpc6qTM7K_AI58-g" title="coinbase" target="_blank">coinbase</a> and <a href="/rltoken/0zZKYc5-xlxGFbxTfCVrBA" title="bitstamp" target="_blank">bitstamp</a> datasets, write a script, <code>forecast_btc.py</code>, that creates, trains, and validates a keras model for the forecasting of BTC:</p>

<ul>
<li>Your model should use the past 24 hours of BTC data to predict the value of BTC at the close of the following hour (approximately how long the average transaction takes):</li>
<li>The datasets are formatted such that every row represents a 60 second time window containing:

<ul>
<li>The start time of the time window in Unix time</li>
<li>The open price in USD at the start of the time window</li>
<li>The high price in USD within the time window</li>
<li>The low price in USD within the time window</li>
<li>The close price in USD at end of the time window</li>
<li>The amount of BTC transacted in the time window</li>
<li>The amount of Currency (USD) transacted in the time window</li>
<li>The <a href="/rltoken/79YPxEkzc7Q1rc92f1MOOQ" title="volume-weighted average price" target="_blank">volume-weighted average price</a> in USD for the time window</li>
</ul></li>
<li>Your model should use an RNN architecture of your choosing</li>
<li>Your model should use mean-squared error (MSE) as its cost function</li>
<li>You should use a <code>tf.data.Dataset</code> to feed data to your model</li>
</ul>

<p>Because the dataset is <a href="/rltoken/Keixv8XzPLglpNSCkUiOpQ" title="raw" target="_blank">raw</a>, you will need to create a script, <code>preprocess_data.py</code> to preprocess this data. Here are some things to consider:</p>

<ul>
<li>Are all of the data points useful?</li>
<li>Are all of the data features useful?</li>
<li>Should you rescale the data?</li>
<li>Is the current time window relevant?</li>
<li>How should you save this preprocessed data?</li>
</ul>

<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>supervised_learning/0x0E-time_series</code></li>
    <li>File: <code>README.md, forecast_btc.py, preprocess_data.py</code></li>
</ul>

</div>

</div>

</div>

<span id="user_id" data-id="1283"></span>

</div>



<h4 class="task">1. Everyone wants to know</h4>

<!-- Progress vs Score -->

<!-- Task Body -->
<p>Everyone wants to know how to make money with BTC! Write a blog post explaining your process in completing the task above:</p>

<ul>
<li>An introduction to Time Series Forecasting</li>
<li>An explanation of your preprocessing method and why you chose it</li>
<li>An explanation of how you set up your <code>tf.data.Dataset</code> for your model inputs</li>
<li>An explanation of the model architecture that you used</li>
<li>A results section containing the model performance and corresponding graphs</li>
<li>A conclusion of your experience, your thoughts on forecasting BTC, and a link to your github with the relevant code</li>
</ul>

<p>Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.</p>

<p>When done, please add all URLs below (blog post, shared link, etc.)</p>

<p>Please, remember that these blogs <strong>must be written in English</strong> to further your technical ability in a variety of settings.</p>

</div>

</div>
</section>
