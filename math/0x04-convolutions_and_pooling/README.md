# 0x04. Convolutions and Pooling
<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/11/ed9ca14839ad0201f19e.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20230325T164628Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=17f2bd9b49d133713769d1d1fafb10ac5fa7e542dc88debe28ea0662b5e3d2c5" alt="" loading="lazy" style=""></p>
<h2>Resources</h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="https://setosa.io/ev/image-kernels/" title="Image Kernels" target="_blank">Image Kernels</a> </li>
<li><a href="https://github.com/Machinelearninguru/Image-Processing-Computer-Vision/blob/master/Convolutional%20Neural%20Network/Convolutional%20Layers/README.md" title="Undrestanding Convolutional Layers" target="_blank">Undrestanding Convolutional Layers</a> </li>
<li><a href="https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53" title="A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way" target="_blank">A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way</a></li>
<li><a href="https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks" title="What is max pooling in convolutional neural networks?" target="_blank">What is max pooling in convolutional neural networks?</a> </li>
<li><a href="https://www.youtube.com/watch?v=XuD4C8vJzEQ&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=3" title="Edge Detection Examples" target="_blank">Edge Detection Examples</a></li>
<li><a href="https://www.youtube.com/watch?v=smHa2442Ah4&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=5" title="Padding" target="_blank">Padding</a></li>
<li><a href="https://www.youtube.com/watch?v=tQYZaDn_kSg&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=6" title="Strided Convolutions" target="_blank">Strided Convolutions</a> </li>
<li><a href="https://www.youtube.com/watch?v=KTB_OFoAQcc&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=7" title="Convolutions over Volumes" target="_blank">Convolutions over Volumes</a></li>
<li><a href="https://www.youtube.com/watch?v=8oOgPUO-TBY&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=10" title="Pooling Layers" target="_blank">Pooling Layers</a></li>
<li><a href="https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python" title="Implementing 'SAME' and 'VALID' padding of Tensorflow in Python" target="_blank">Implementing ‘SAME’ and ‘VALID’ padding of Tensorflow in Python</a>

<ul>
<li><strong>NOTE: In this document, there is a mistake regarding valid padding. Floor rounding should be used for valid padding instead of ceiling</strong></li>
</ul></li>
</ul>
<p><strong>Definitions to skim:</strong></p>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Convolution" title="Convolution" target="_blank">Convolution</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Kernel_(image_processing)" title="Kernel (image processing)" target="_blank">Kernel (image processing)</a> </li>
</ul>
<p><strong>References:</strong></p>
<ul>
<li><a href="https://numpy.org/doc/stable/reference/generated/numpy.pad.html" title="numpy.pad" target="_blank">numpy.pad</a> </li>
<li><a href="https://arxiv.org/pdf/1603.07285.pdf" title="A guide to convolution arithmetic for deep learning" target="_blank">A guide to convolution arithmetic for deep learning</a> </li>
</ul>
<h3>Testing</h3>
<p>Please download <a href="https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-ml/animals_1.npz" title="this dataset" target="_blank">this dataset</a> for use in some of the following main files.</p>

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

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20230325T164628Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=1ba08d0ace99590a8af82df48fe88855facced4bd3c03b51541d0b49ef10b861" alt="" loading="lazy" style=""></p>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/6e1b02cc87497f12f17e.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&amp;X-Amz-Date=20230325T164628Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=a592283336fe8c1bcbaf73f2b30ed0016ac7c68e130c66e4ace42fa516817509" alt="" loading="lazy" style=""></p>
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

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=1ba08d0ace99590a8af82df48fe88855facced4bd3c03b51541d0b49ef10b861" alt="" loading="lazy" style=""></p>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/b32bba8fea86011c3372.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6cc5710aeba4b628a5fe5037685bc60188fbb2f94214466b1df66f26f4c334db" alt="" style=""></p>


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

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=1ba08d0ace99590a8af82df48fe88855facced4bd3c03b51541d0b49ef10b861" alt="" style=""></p>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/3f178b675c1e2fdc86bd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=9bb0020a0ee7b95e51e473699943d01eb00172fb21373181ec2a71e0fe5a7c24" alt="" style=""></p>


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
<!--task3 -->
<div data-role="task3760" data-position="4">
              <div class=" clearfix gap" id="task-3760">
<span id="user_id" data-id="1283"></span>

</div>

<h4 class="task">
3. Strided Convolution
    <span class="alert alert-warning mandatory-optional">
    mandatory
    </span>
</h4>


<!-- Task Body -->
<p>Write a function <code>def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):</code> that performs a convolution on grayscale images:</p>

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
<li><code>padding</code> is either a tuple of <code>(ph, pw)</code>, ‘same’, or ‘valid’

<ul>
<li>if ‘same’, performs a same convolution</li>
<li>if ‘valid’, performs a valid convolution</li>
<li>if a tuple:

<ul>
<li><code>ph</code> is the padding for the height of the image</li>
<li><code>pw</code> is the padding for the width of the image</li>
</ul></li>
<li>the image should be padded with 0’s</li>
</ul></li>
<li><code>stride</code> is a tuple of <code>(sh, sw)</code>

<ul>
<li><code>sh</code> is the stride for the height of the image</li>
<li><code>sw</code> is the stride for the width of the image</li>
</ul></li>
<li>You are only allowed to use two <code>for</code> loops; any other loops of any kind are not allowed <em>Hint: loop over <code>i</code> and <code>j</code></em></li>
<li>Returns: a <code>numpy.ndarray</code> containing the convolved images</li>
</ul>

<pre><code>ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 3-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./3-main.py 
(50000, 28, 28)
(50000, 13, 13)
</code></pre>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=1ba08d0ace99590a8af82df48fe88855facced4bd3c03b51541d0b49ef10b861" alt="" style=""></p>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/036ccba7dccf211dab76.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=cc538768e7b774b7db8d22f7fcf368de0704857148c864476e66c2bf12154de2" alt="" style=""></p>


<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>math/0x04-convolutions_and_pooling</code></li>
    <li>File: <code>3-convolve_grayscale.py</code></li>
</ul>

</div>

</div>

</div>

</div>

</div>

</div>
<!--4 -->
<div data-role="task3761" data-position="5">
<div class=" clearfix gap" id="task-3761">
<span id="user_id" data-id="1283"></span>
</div>
</div>

  <h4 class="task">
    4. Convolution with Channels
      <span class="alert alert-warning mandatory-optional">
        mandatory
      </span>
  </h4>

<!-- Task Body -->
<p>Write a function <code>def convolve_channels(images, kernel, padding='same', stride=(1, 1)):</code> that performs a convolution on images with channels:</p>

<ul>
<li><code>images</code> is a <code>numpy.ndarray</code> with shape <code>(m, h, w, c)</code> containing multiple images

<ul>
<li><code>m</code> is the number of images</li>
<li><code>h</code> is the height in pixels of the images</li>
<li><code>w</code> is the width in pixels of the images</li>
<li><code>c</code> is the number of channels in the image</li>
</ul></li>
<li><code>kernel</code> is a <code>numpy.ndarray</code> with shape <code>(kh, kw, c)</code> containing the kernel for the convolution

<ul>
<li><code>kh</code> is the height of the kernel</li>
<li><code>kw</code> is the width of the kernel</li>
</ul></li>
<li><code>padding</code> is either a tuple of <code>(ph, pw)</code>, ‘same’, or ‘valid’

<ul>
<li>if ‘same’, performs a same convolution</li>
<li>if ‘valid’, performs a valid convolution</li>
<li>if a tuple:

<ul>
<li><code>ph</code> is the padding for the height of the image</li>
<li><code>pw</code> is the padding for the width of the image</li>
</ul></li>
<li>the image should be padded with 0’s</li>
</ul></li>
<li><code>stride</code> is a tuple of <code>(sh, sw)</code>

<ul>
<li><code>sh</code> is the stride for the height of the image</li>
<li><code>sw</code> is the stride for the width of the image</li>
</ul></li>
<li>You are only allowed to use two <code>for</code> loops; any other loops of any kind are not allowed</li>
<li>Returns: a <code>numpy.ndarray</code> containing the convolved images</li>
</ul>

<pre><code>ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 4-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_channels = __import__('4-convolve_channels').convolve_channels


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    print(images.shape)
    kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]], [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]], [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
    images_conv = convolve_channels(images, kernel, padding='valid')
    print(images_conv.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_conv[0])
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./4-main.py 
(10000, 32, 32, 3)
(10000, 30, 30)
</code></pre>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e16104855e016c3afe585e0cefcdf33482ce3542651de629aa460e18d79eb48e" alt="" style=""></p>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/8bc039fb38d60601b01a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4206f2266ff191136e1744464b10f6bb5711c53fd9f9254418beb4db6c45e318" alt="" style=""></p>

<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>math/0x04-convolutions_and_pooling</code></li>
    <li>File: <code>4-convolve_channels.py</code></li>
</ul>

</div>

</div>

</div>

</div>

</div>

</div>


<!--task 5 -->
<div data-role="task3762" data-position="6">
    <div class=" clearfix gap" id="task-3762">
    <span id="user_id" data-id="1283"></span>

    </div>
</div>

<h4 class="task">
5. Multiple Kernels
    <span class="alert alert-warning mandatory-optional">
    mandatory
    </span>
</h4>

<!-- Task Body -->
<p>Write a function <code>def convolve(images, kernels, padding='same', stride=(1, 1)):</code> that performs a convolution on images using multiple kernels:</p>

<ul>
<li><code>images</code> is a <code>numpy.ndarray</code> with shape <code>(m, h, w, c)</code> containing multiple images

<ul>
<li><code>m</code> is the number of images</li>
<li><code>h</code> is the height in pixels of the images</li>
<li><code>w</code> is the width in pixels of the images</li>
<li><code>c</code> is the number of channels in the image</li>
</ul></li>
<li><code>kernels</code> is a <code>numpy.ndarray</code> with shape <code>(kh, kw, c, nc)</code> containing the kernels for the convolution

<ul>
<li><code>kh</code> is the height of a kernel</li>
<li><code>kw</code> is the width of a kernel</li>
<li><code>nc</code> is the number of kernels</li>
</ul></li>
<li><code>padding</code> is either a tuple of <code>(ph, pw)</code>, ‘same’, or ‘valid’

<ul>
<li>if ‘same’, performs a same convolution</li>
<li>if ‘valid’, performs a valid convolution</li>
<li>if a tuple:

<ul>
<li><code>ph</code> is the padding for the height of the image</li>
<li><code>pw</code> is the padding for the width of the image</li>
</ul></li>
<li>the image should be padded with 0’s</li>
</ul></li>
<li><code>stride</code> is a tuple of <code>(sh, sw)</code>

<ul>
<li><code>sh</code> is the stride for the height of the image</li>
<li><code>sw</code> is the stride for the width of the image</li>
</ul></li>
<li>You are only allowed to use three <code>for</code> loops; any other loops of any kind are not allowed</li>
<li>Returns: a <code>numpy.ndarray</code> containing the convolved images</li>
</ul>

<pre><code>ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 5-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve = __import__('5-convolve').convolve


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    print(images.shape)
    kernels = np.array([[[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[0, -1, 1], [0, -1, 1], [0, -1, 1]]],
                       [[[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]], [[5, 0, 0], [5, 0, 0], [5, 0, 0]], [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]],
                       [[[0, 1, -1], [0, 1, -1], [0, 1, -1]], [[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]], [[0, -1, -1], [0, -1, -1], [0, -1, -1]]]])

    images_conv = convolve(images, kernels, padding='valid')
    print(images_conv.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_conv[0, :, :, 0])
    plt.show()
    plt.imshow(images_conv[0, :, :, 1])
    plt.show()
    plt.imshow(images_conv[0, :, :, 2])
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./5-main.py 
(10000, 32, 32, 3)
(10000, 30, 30, 3)
</code></pre>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e16104855e016c3afe585e0cefcdf33482ce3542651de629aa460e18d79eb48e" alt="" style=""></p>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/6d6319bb470e3566e885.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=9447a25d03cd6bc1485b680306cbe212aa7250322fe3fd3215047a2d954825f6" alt="" style=""></p>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/1370dd6200e942eee8f9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ace8fcd47683666ffa93730e1004d72685e5e93d4b4ca6cebb3342501409b3cb" alt="" style=""></p>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/a24b7d741b3c378f9f89.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=1593251498e5b964b97387f2f4b85d5297bb09edd8802b408ab736145c58f876" alt="" style=""></p>

<!-- Task URLs -->

<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>math/0x04-convolutions_and_pooling</code></li>
    <li>File: <code>5-convolve.py</code></li>
</ul>
</div>

</div>

</div>

</div>

</div>

<!-- -->

<div data-role="task3763" data-position="7">
<div class=" clearfix gap" id="task-3763">
<span id="user_id" data-id="1283"></span>
</div>
</div>

<h4 class="task">
6. Pooling
    <span class="alert alert-warning mandatory-optional">
    mandatory
    </span>
</h4>
<!-- Task Body -->
<p>Write a function <code>def pool(images, kernel_shape, stride, mode='max'):</code> that performs pooling on images:</p>

<ul>
<li><code>images</code> is a <code>numpy.ndarray</code> with shape <code>(m, h, w, c)</code> containing multiple images

<ul>
<li><code>m</code> is the number of images</li>
<li><code>h</code> is the height in pixels of the images</li>
<li><code>w</code> is the width in pixels of the images</li>
<li><code>c</code> is the number of channels in the image</li>
</ul></li>
<li><code>kernel_shape</code> is a tuple of <code>(kh, kw)</code> containing the kernel shape for the pooling

<ul>
<li><code>kh</code> is the height of the kernel</li>
<li><code>kw</code> is the width of the kernel</li>
</ul></li>
<li><code>stride</code> is a tuple of <code>(sh, sw)</code>

<ul>
<li><code>sh</code> is the stride for the height of the image</li>
<li><code>sw</code> is the stride for the width of the image</li>
</ul></li>
<li><code>mode</code> indicates the type of pooling

<ul>
<li><code>max</code> indicates max pooling</li>
<li><code>avg</code> indicates average pooling</li>
</ul></li>
<li>You are only allowed to use two <code>for</code> loops; any other loops of any kind are not allowed</li>
<li>Returns: a <code>numpy.ndarray</code> containing the pooled images</li>
</ul>

<pre><code>ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 6-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
pool = __import__('6-pool').pool


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    print(images.shape)
    images_pool = pool(images, (2, 2), (2, 2), mode='avg')
    print(images_pool.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_pool[0] / 255)
    plt.show()
ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./6-main.py 
(10000, 32, 32, 3)
(10000, 16, 16, 3)
</code></pre>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e16104855e016c3afe585e0cefcdf33482ce3542651de629aa460e18d79eb48e" alt="" style=""></p>

<p><img src="https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2018/12/ab4705f939c3a8e487bb.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230325%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230325T164628Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4d4e30cd44dbdf1bb67f21c91a1e2e55a69d3576249d79fe882fa72aaa9112ff" alt="" style=""></p>

<!-- Task URLs -->
<!-- Github information -->
<p class="sm-gap"><strong>Repo:</strong></p>
<ul>
    <li>GitHub repository: <code>holbertonschool-machine_learning</code></li>
    <li>Directory: <code>math/0x04-convolutions_and_pooling</code></li>
    <li>File: <code>6-pool.py</code></li>
</ul>
</div>
</div>

</div>

</div>

</div>

</div>