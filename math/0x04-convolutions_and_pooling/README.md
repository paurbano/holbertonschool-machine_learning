# 0x04. Convolutions and Pooling

## General
* **What is a convolution?**
    Mathematical operation on two functions that produces a third function representing how the shape of one is modified by the other.
* **What is max pooling? average pooling?**
* What is a kernel/filter?
* What is padding?
* What is “same” padding? “valid” padding?
* What is a stride?
* What are channels?
* How to perform a convolution over an image
* How to perform max/average pooling over an image

<h2 class="gap">Tasks</h2>
<div data-role="task3757" data-position="1">
              <div class=" clearfix gap" id="task-3757">
<span id="user_id" data-id="1283"></span>
</div>


</div>

<h4 class="task">
0. Valid Convolution
    <span class="alert alert-warning mandatory-optional">
    mandatory
    </span>
</h4>

<!-- Task Body -->
<p>Write a function <code>def convolve_grayscale_valid(images, kernel):</code> that performs a valid convolution on grayscale images:</p>

<ul>
<li><code>images</code> is a <code>numpy.ndarray</code> with shape <code>(m, h, w)</code> containing multiple grayscale images

<ul>
<li><code>m</code> is the number of images</li>
<li><code>h</code> is the height in pixels of the images</li>
<li><code>w</code> is the width in pixels of the images</li>
</ul></li>
<li><code>kernel</code> is a <code>numpy.ndarray</code> with shape <code>(kh, kw)</code> containing the kernel for the convolution

<ul>
<li><code>kh</code> is the height of the kernel</li>
<li><code>kw</code> is the width of the kernel</li>
</ul></li>
<li>You are only allowed to use two <code>for</code> loops; any other loops of any kind are not allowed</li>
<li>Returns: a <code>numpy.ndarray</code> containing the convolved images</li>
</ul>

<pre><code>ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 0-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./0-main.py 
(50000, 28, 28)
(50000, 26, 26)
</code></pre>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210221%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210221T001528Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=0decf54c3ad66bcb2ffcc52158f015dee0cb2dd34d11d52faf11bd602055336a" alt="" style=""></p>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6e1b02cc87497f12f17e.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210221%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210221T001528Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=d382b75861bdcb5202e6e5edef2e645f6c7c4387230788e7a87ee70824d37f9b" alt="" style=""></p>


<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>math/0x04-convolutions_and_pooling</code></li>
    <li>File: <code>0-convolve_grayscale_valid.py</code></li>
</ul>

</div>

</div>

</div>

</div>

</div>

</div>

<h4 class="task">1. Same Convolution</h4>
 
<!-- Task Body -->
<p>Write a function <code>def convolve_grayscale_same(images, kernel):</code> that performs a same convolution on grayscale images:</p>

<ul>
<li><code>images</code> is a <code>numpy.ndarray</code> with shape <code>(m, h, w)</code> containing multiple grayscale images

<ul>
<li><code>m</code> is the number of images</li>
<li><code>h</code> is the height in pixels of the images</li>
<li><code>w</code> is the width in pixels of the images</li>
</ul></li>
<li><code>kernel</code> is a <code>numpy.ndarray</code> with shape <code>(kh, kw)</code> containing the kernel for the convolution

<ul>
<li><code>kh</code> is the height of the kernel</li>
<li><code>kw</code> is the width of the kernel</li>
</ul></li>
<li>if necessary, the image should be padded with 0’s</li>
<li>You are only allowed to use two <code>for</code> loops; any other loops of any kind are not allowed</li>
<li>Returns: a <code>numpy.ndarray</code> containing the convolved images</li>
</ul>

<pre><code>ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 1-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_same(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./1-main.py 
(50000, 28, 28)
(50000, 28, 28)
</code></pre>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210221%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210221T001528Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=0decf54c3ad66bcb2ffcc52158f015dee0cb2dd34d11d52faf11bd602055336a" alt="" style=""></p>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/b32bba8fea86011c3372.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210221%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210221T001528Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=801326ddf4f25aa5f607617f7af80eed527720e4911d75c299bb457c2c77425f" alt="" style=""></p>


<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>math/0x04-convolutions_and_pooling</code></li>
    <li>File: <code>1-convolve_grayscale_same.py</code></li>
</ul>

</div>

</div>

</div>

</div>

</div>

</div>

<h4 class="task">2. Convolution with Padding</h4>

<!-- Task Body -->
<p>Write a function <code>def convolve_grayscale_padding(images, kernel, padding):</code> that performs a convolution on grayscale images with custom padding:</p>

<ul>
<li><code>images</code> is a <code>numpy.ndarray</code> with shape <code>(m, h, w)</code> containing multiple grayscale images

<ul>
<li><code>m</code> is the number of images</li>
<li><code>h</code> is the height in pixels of the images</li>
<li><code>w</code> is the width in pixels of the images</li>
</ul></li>
<li><code>kernel</code> is a <code>numpy.ndarray</code> with shape <code>(kh, kw)</code> containing the kernel for the convolution

<ul>
<li><code>kh</code> is the height of the kernel</li>
<li><code>kw</code> is the width of the kernel</li>
</ul></li>
<li><code>padding</code> is a tuple of <code>(ph, pw)</code>

<ul>
<li><code>ph</code> is the padding for the height of the image</li>
<li><code>pw</code> is the padding for the width of the image</li>
<li>the image should be padded with 0’s</li>
</ul></li>
<li>You are only allowed to use two <code>for</code> loops; any other loops of any kind are not allowed</li>
<li>Returns: a <code>numpy.ndarray</code> containing the convolved images</li>
</ul>

<pre><code>ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 2-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_padding(images, kernel, (2, 4))
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./2-main.py 
(50000, 28, 28)
(50000, 30, 34)
</code></pre>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210221%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210221T001528Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=0decf54c3ad66bcb2ffcc52158f015dee0cb2dd34d11d52faf11bd602055336a" alt="" style=""></p>

<p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/3f178b675c1e2fdc86bd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210221%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20210221T001528Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=2a0bc5374adef57eee968f6e64c6de4c37949763888cf5b65f0966d15d5e02c1" alt="" style=""></p>


<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>math/0x04-convolutions_and_pooling</code></li>
    <li>File: <code>2-convolve_grayscale_padding.py</code></li>
</ul>

</div>

</div>

</div>

</div>

</div>

</div>