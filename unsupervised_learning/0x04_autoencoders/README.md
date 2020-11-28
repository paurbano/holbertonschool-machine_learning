<h1 class="gap">0x04. Autoencoders</h1>
<article id="description" class="gap formatted-content">
    <p><strong>Read or watch</strong>:</p>

<ul>
<li><a href="/rltoken/WMbe5eWUHhyFet68Kf9pwQ" title="Autoencoder - definition" target="_blank">Autoencoder - definition</a></li>
<li><a href="/rltoken/9c-itgW_s8KR3osdRmmnKw" title="Autoencoder - loss function" target="_blank">Autoencoder - loss function</a></li>
<li><a href="/rltoken/3hm5-oOrajXejrCH4P1-Mg" title="Deep learning - deep autoencoder" target="_blank">Deep learning - deep autoencoder</a></li>
<li><a href="/rltoken/qoMN_ZeSetmF92dXhinEAA" title="Introduction to autoencoders" target="_blank">Introduction to autoencoders</a></li>
<li><a href="/rltoken/6TUoTdWuAyB4SGEw5Jf-Cg" title="Variational Autoencoders - EXPLAINED!" target="_blank">Variational Autoencoders - EXPLAINED!</a> <em>up to</em> <strong>12:55</strong></li>
<li><a href="/rltoken/VV_w6WaFyY8e0jF5qXmy4g" title="Variational Autoencoders" target="_blank">Variational Autoencoders</a></li>
<li><a href="/rltoken/VaCGL8qmByv1wmHx9RrjrQ" title="Intuitively Understanding Variational Autoencoders" target="_blank">Intuitively Understanding Variational Autoencoders</a></li>
<li><a href="/rltoken/c6_rGYC2Ei_mATkAY_vSBA" title="Deep Generative Models" target="_blank">Deep Generative Models</a> <em>up to</em> <strong>Generative Adversarial Networks</strong></li>
</ul>

<p><strong>Definitions to skim</strong>:</p>

<ul>
<li><a href="/rltoken/M0-vm4JIoP-Msl5SemHDDA" title="Kullback–Leibler divergence" target="_blank">Kullback–Leibler divergence</a> <em>recall its use in t-SNE</em></li>
<li><a href="/rltoken/kgDcseTs0_TQ5X7nPs7EeQ" title="Autoencoder" target="_blank">Autoencoder</a></li>
<li><a href="/rltoken/oICoCfZORJNQFwcpR-Kq1w" title="Generative model" target="_blank">Generative model</a></li>
</ul>

<p><strong>References</strong>:</p>

<ul>
<li><a href="/rltoken/lAQEyGkZx9q4vnXTD8noxQ" title="The Deep Learning textbook - Chapter 14: Autoencoders" target="_blank">The Deep Learning textbook - Chapter 14: Autoencoders</a></li>
<li><a href="/rltoken/EsJJMSqKlVcbBvjLEMhutQ" title="Reducing the Dimensionality of Data with Neural Networks 2006" target="_blank">Reducing the Dimensionality of Data with Neural Networks 2006</a></li>
</ul>

<h2>Learning Objectives</h2>

<ul>
<li>What is an autoencoder?</li>
<li>What is latent space?</li>
<li>What is a bottleneck?</li>
<li>What is a sparse autoencoder?</li>
<li>What is a convolutional autoencoder?</li>
<li>What is a generative model?</li>
<li>What is a variational autoencoder?</li>
<li>What is the Kullback-Leibler divergence?</li>
</ul>

<h2>Requirements</h2>

<h3>General</h3>

<ul>
<li>Allowed editors: <code>vi</code>, <code>vim</code>, <code>emacs</code></li>
<li>All your files will be interpreted/compiled on Ubuntu 16.04 LTS using <code>python3</code> (version 3.5)</li>
<li>Your files will be executed with <code>numpy</code> (version 1.15) and <code>tensorflow</code> (version 1.12)</li>
<li>All your files should end with a new line</li>
<li>The first line of all your files should be exactly <code>#!/usr/bin/env python3</code></li>
<li>A <code>README.md</code> file, at the root of the folder of the project, is mandatory</li>
<li>Your code should use the <code>pycodestyle</code> style (version 2.4)</li>
<li>All your modules should have documentation (<code>python3 -c 'print(__import__("my_module").__doc__)'</code>)</li>
<li>All your classes should have documentation (<code>python3 -c 'print(__import__("my_module").MyClass.__doc__)'</code>)</li>
<li>All your functions (inside and outside a class) should have documentation (<code>python3 -c 'print(__import__("my_module").my_function.__doc__)'</code> and <code>python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'</code>)</li>
<li>Unless otherwise noted, you are not allowed to import any module except <code>import tensorflow.keras as keras</code></li>
<li>All your files must be executable</li>
</ul>

</article>

<h2 class="gap">Tasks</h2>

<section class="formatted-content">
            <div data-role="task4926" data-position="1">
              <div class=" clearfix gap" id="task-4926">
<span id="user_id" data-id="1283"></span>

</div>
  <h4 class="task">
    0. "Vanilla" Autoencoder
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>
  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def autoencoder(input_dims, hidden_layers, latent_dims):</code> that creates an autoencoder:</p>

<ul>
<li><code>input_dims</code> is an integer containing the dimensions of the model input</li>
<li><code>hidden_layers</code> is a list containing the number of nodes for each hidden layer in the encoder, respectively

<ul>
<li>the hidden layers should be reversed for the decoder</li>
</ul></li>
<li><code>latent_dims</code> is an integer containing the dimensions of the latent space representation</li>
<li>Returns: <code>encoder, decoder, auto</code>

<ul>
<li><code>encoder</code> is the encoder model</li>
<li><code>decoder</code> is the decoder model</li>
<li><code>auto</code> is the full autoencoder model</li>
</ul></li>
<li>The autoencoder model should be compiled using adam optimization and binary cross-entropy loss</li>
<li>All layers should use a <code>relu</code> activation except for the last layer in the decoder, which should use <code>sigmoid</code></li>
</ul>

<pre><code>$ cat 0-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('0-vanilla').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [128, 64], 32)
auto.fit(x_train, x_train, epochs=50,batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i].reshape((28, 28)))
plt.show()
$ ./0-main.py
Epoch 1/50
60000/60000 [==============================] - 5s 85us/step - loss: 0.2504 - val_loss: 0.1667
Epoch 2/50
60000/60000 [==============================] - 5s 84us/step - loss: 0.1498 - val_loss: 0.1361
Epoch 3/50
60000/60000 [==============================] - 5s 83us/step - loss: 0.1312 - val_loss: 0.1242
Epoch 4/50
60000/60000 [==============================] - 5s 79us/step - loss: 0.1220 - val_loss: 0.1173
Epoch 5/50
60000/60000 [==============================] - 5s 79us/step - loss: 0.1170 - val_loss: 0.1132

...

Epoch 46/50
60000/60000 [==============================] - 5s 80us/step - loss: 0.0852 - val_loss: 0.0850
Epoch 47/50
60000/60000 [==============================] - 5s 81us/step - loss: 0.0851 - val_loss: 0.0846
Epoch 48/50
60000/60000 [==============================] - 5s 84us/step - loss: 0.0850 - val_loss: 0.0848
Epoch 49/50
60000/60000 [==============================] - 5s 80us/step - loss: 0.0849 - val_loss: 0.0845
Epoch 50/50
60000/60000 [==============================] - 5s 85us/step - loss: 0.0848 - val_loss: 0.0844
6.5280433
</code></pre>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/6/9c11026a690360f31ce76fd7e5e3515e7cda6925.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201123%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20201123T171245Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=33929ca7ec24db28f6e700fd615f8ca7ba1b375619f3ba679155413052ff9fe3" alt="" style=""></p>

</div>
<span id="user_id" data-id="1283"></span>
</div>
  <h4 class="task">
    1. Sparse Autoencoder
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):</code> that creates a sparse autoencoder:</p>

<ul>
<li><code>input_dims</code> is an integer containing the dimensions of the model input</li>
<li><code>hidden_layers</code> is a list containing the number of nodes for each hidden layer in the encoder, respectively

<ul>
<li>the hidden layers should be reversed for the decoder</li>
</ul></li>
<li><code>latent_dims</code> is an integer containing the dimensions of the latent space representation</li>
<li><code>lambtha</code> is the regularization parameter used for L1 regularization on the encoded output</li>
<li>Returns: <code>encoder, decoder, auto</code>

<ul>
<li><code>encoder</code> is the encoder model</li>
<li><code>decoder</code> is the decoder model</li>
<li><code>auto</code> is the sparse autoencoder model</li>
</ul></li>
<li>The sparse autoencoder model should be compiled using adam optimization and binary cross-entropy loss</li>
<li>All layers should use a <code>relu</code> activation except for the last layer in the decoder, which should use <code>sigmoid</code></li>
</ul>

<pre><code>$ cat 1-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('1-sparse').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [128, 64], 32, 10e-6)
auto.fit(x_train, x_train, epochs=100,batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i].reshape((28, 28)))
plt.show()
$ ./1-main.py
Epoch 1/50
60000/60000 [==============================] - 6s 102us/step - loss: 0.3123 - val_loss: 0.2538
Epoch 2/100
60000/60000 [==============================] - 6s 96us/step - loss: 0.2463 - val_loss: 0.2410
Epoch 3/100
60000/60000 [==============================] - 5s 90us/step - loss: 0.2400 - val_loss: 0.2381
Epoch 4/100
60000/60000 [==============================] - 5s 80us/step - loss: 0.2379 - val_loss: 0.2360
Epoch 5/100
60000/60000 [==============================] - 5s 82us/step - loss: 0.2360 - val_loss: 0.2339

...

Epoch 96/100
60000/60000 [==============================] - 5s 80us/step - loss: 0.1602 - val_loss: 0.1609
Epoch 97/100
60000/60000 [==============================] - 5s 84us/step - loss: 0.1601 - val_loss: 0.1608
Epoch 98/100
60000/60000 [==============================] - 5s 87us/step - loss: 0.1601 - val_loss: 0.1601
Epoch 99/100
60000/60000 [==============================] - 5s 89us/step - loss: 0.1601 - val_loss: 0.1604
Epoch 100/100
60000/60000 [==============================] - 5s 82us/step - loss: 0.1597 - val_loss: 0.1601
0.016292876
</code></pre>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/6/4259218c39ff1a590ed499023c84b5a5242b91c2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201123%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20201123T171245Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=0478a9b1a0619ea6e84282d6144ff0a8e02af36db0f6006bece58495292f4049" alt="" style=""></p>

</div>
<span id="user_id" data-id="1283"></span>
</div>
  <h4 class="task">
    2. Convolutional Autoencoder
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def autoencoder(input_dims, filters, latent_dims):</code> that creates a convolutional autoencoder:</p>

<ul>
<li><code>input_dims</code> is a tuple of integers containing the dimensions of the model input</li>
<li><code>filters</code> is a list containing the number of filters for each convolutional layer in the encoder, respectively

<ul>
<li>the filters should be reversed for the decoder</li>
</ul></li>
<li><code>latent_dims</code> is a tuple of integers containing the dimensions of the latent space representation</li>
<li>Each convolution in the encoder should use a kernel size of <code>(3, 3)</code> with same padding and <code>relu</code> activation, followed by max pooling of size <code>(2, 2)</code></li>
<li>Each convolution in the decoder, except for the last two, should use a filter size of <code>(3, 3)</code> with same padding and <code>relu</code> activation, followed by upsampling of size <code>(2, 2)</code>

<ul>
<li>The second to last convolution should instead use valid padding</li>
<li>The last convolution should have the same number of filters as the number of channels in <code>input_dims</code> with <code>sigmoid</code> activation and no upsampling</li>
</ul></li>
<li>Returns: <code>encoder, decoder, auto</code>

<ul>
<li><code>encoder</code> is the encoder model</li>
<li><code>decoder</code> is the decoder model</li>
<li><code>auto</code> is the full autoencoder model</li>
</ul></li>
<li>The autoencoder model should be compiled using adam optimization and binary cross-entropy loss</li>
</ul>

<pre><code>$ cat 2-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('2-convolutional').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print(x_train.shape)
print(x_test.shape)
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder((28, 28, 1), [16, 8, 8], (4, 4, 8))
auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)[:,:,:,0]

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i,:,:,0])
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i])
plt.show()
$ ./2-main.py
Epoch 1/50
60000/60000 [==============================] - 49s 810us/step - loss: 63.9743 - val_loss: 43.5109
Epoch 2/50
60000/60000 [==============================] - 48s 804us/step - loss: 39.9287 - val_loss: 37.1333
Epoch 3/50
60000/60000 [==============================] - 48s 803us/step - loss: 35.7883 - val_loss: 34.1952
Epoch 4/50
60000/60000 [==============================] - 48s 792us/step - loss: 33.4408 - val_loss: 32.2462
Epoch 5/50
60000/60000 [==============================] - 47s 791us/step - loss: 31.8871 - val_loss: 30.9729

...

Epoch 46/50
60000/60000 [==============================] - 45s 752us/step - loss: 23.9016 - val_loss: 23.6926
Epoch 47/50
60000/60000 [==============================] - 45s 754us/step - loss: 23.9029 - val_loss: 23.7102
Epoch 48/50
60000/60000 [==============================] - 45s 750us/step - loss: 23.8331 - val_loss: 23.5239
Epoch 49/50
60000/60000 [==============================] - 46s 771us/step - loss: 23.8047 - val_loss: 23.5510
Epoch 50/50
60000/60000 [==============================] - 46s 772us/step - loss: 23.7744 - val_loss: 23.4939
2.4494107
</code></pre>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/6/9ae05866da7bb93a051bd59028bc0885d14ea71c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201123%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20201123T171245Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=050c7b16ec33a1b9b43a2559ddd743a48bc6bc76695dc83b067a654280592711" alt="" style=""></p>

</div>
<span id="user_id" data-id="1283"></span>

</div>


  <h4 class="task">
    3. Variational Autoencoder
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

  

  <!-- Progress vs Score -->

  <!-- Task Body -->
  <p>Write a function <code>def autoencoder(input_dims, hidden_layers, latent_dims):</code> that creates a variational autoencoder:</p>

<ul>
<li><code>input_dims</code> is an integer containing the dimensions of the model input</li>
<li><code>hidden_layers</code> is a list containing the number of nodes for each hidden layer in the encoder, respectively

<ul>
<li>the hidden layers should be reversed for the decoder</li>
</ul></li>
<li><code>latent_dims</code> is an integer containing the dimensions of the latent space representation</li>
<li>Returns: <code>encoder, decoder, auto</code>

<ul>
<li><code>encoder</code> is the encoder model, which should output the latent representation, the mean, and the log variance, respectively</li>
<li><code>decoder</code> is the decoder model</li>
<li><code>auto</code> is the full autoencoder model</li>
</ul></li>
<li>The autoencoder model should be compiled using adam optimization and binary cross-entropy loss</li>
<li>All layers should use a <code>relu</code> activation except for the mean and log variance layers in the encoder, which should use <code>None</code>,  and the last layer in the decoder, which should use <code>sigmoid</code></li>
</ul>

<pre><code>$ cat 3-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('3-variational').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [512], 2)
auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded, mu, log_sig = encoder.predict(x_test[:10])
print(mu)
print(np.exp(log_sig / 2))
reconstructed = decoder.predict(encoded).reshape((-1, 28, 28))
x_test = x_test.reshape((-1, 28, 28))

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i])
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i])
plt.show()


l1 = np.linspace(-3, 3, 25)
l2 = np.linspace(-3, 3, 25)
L = np.stack(np.meshgrid(l1, l2, sparse=False, indexing='ij'), axis=2)
G = decoder.predict(L.reshape((-1, 2)), batch_size=125)

for i in range(25*25):
    ax = plt.subplot(25, 25, i + 1)
    ax.axis('off')
    plt.imshow(G[i].reshape((28, 28)))
plt.show()
$ ./3-main.py
Epoch 1/50
60000/60000 [==============================] - 15s 242us/step - loss: 214.4525 - val_loss: 177.2306
Epoch 2/50
60000/60000 [==============================] - 11s 175us/step - loss: 171.7558 - val_loss: 168.7191
Epoch 3/50
60000/60000 [==============================] - 11s 182us/step - loss: 167.4977 - val_loss: 166.5061
Epoch 4/50
60000/60000 [==============================] - 11s 179us/step - loss: 165.6473 - val_loss: 165.1279
Epoch 5/50
60000/60000 [==============================] - 11s 181us/step - loss: 164.0918 - val_loss: 163.7083

...

Epoch 46/50
60000/60000 [==============================] - 15s 249us/step - loss: 148.1491 - val_loss: 151.3205
Epoch 47/50
60000/60000 [==============================] - 12s 204us/step - loss: 148.0358 - val_loss: 151.2141
Epoch 48/50
60000/60000 [==============================] - 11s 179us/step - loss: 147.9396 - val_loss: 151.3823
Epoch 49/50
60000/60000 [==============================] - 13s 223us/step - loss: 147.8144 - val_loss: 151.4026
Epoch 50/50
60000/60000 [==============================] - 11s 189us/step - loss: 147.6572 - val_loss: 151.1969
[[-0.33454233 -3.0770888 ]
 [-0.68772286  0.52945304]
 [ 3.1372023  -1.5037178 ]
 [-0.46997875  2.4711971 ]
 [-2.239822   -0.91364074]
 [ 2.7829633  -1.2185467 ]
 [-0.8319831  -0.97430193]
 [-1.3994675  -0.16924876]
 [-0.2642493  -0.45080736]
 [-0.3476941  -1.5133704 ]]
[[0.07307572 0.18656202]
 [0.04450396 0.03617072]
 [0.15917557 0.09816898]
 [0.07885559 0.056187  ]
 [0.11542598 0.07378525]
 [0.14280568 0.0857826 ]
 [0.0790622  0.07540198]
 [0.08175724 0.05216441]
 [0.05364255 0.05444151]
 [0.04280119 0.07214296]]
</code></pre>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/6/746ae5925f1f5dd60ebd3e5e2df45bfeb955f939.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201123%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20201123T171245Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=69cbc81d42378225f692ed4a83afc66562d4f007060a92221ea7fd07cdfdc2c8" alt="" style=""></p>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/6/aa86806adfdc4c2ea7394df066867c47d0beaf93.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201123%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20201123T171245Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=68e6e9a8a75a05f1eaf462570e626435ce9c7d5434799688ccb261ece49b22fa" alt="" style=""></p>


</div>
