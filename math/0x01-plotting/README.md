# 0x01. Plotting

<h2>Resources</h2>
<p><strong>Read or watch</strong>:</p>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Plot_%28graphics%29" title="Plot (graphics)" target="_blank">Plot (graphics)</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Scatter_plot" title="Scatter plot" target="_blank">Scatter plot</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Line_chart" title="Line chart" target="_blank">Line chart</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Bar_chart" title="Bar chart" target="_blank">Bar chart</a> </li>
<li><a href="https://en.wikipedia.org/wiki/Histogram" title="Histogram" target="_blank">Histogram</a> </li>
</ul>
<p><strong>References</strong>:</p>
<ul>
<li><a href="https://matplotlib.org/stable/tutorials/introductory/pyplot.html" title="Pyplot tutorial" target="_blank">Pyplot tutorial</a> </li>
<li><a href="https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html" title="matplotlib.pyplot" target="_blank">matplotlib.pyplot</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html" title="matplotlib.pyplot.plot" target="_blank">matplotlib.pyplot.plot</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html" title="matplotlib.pyplot.scatter" target="_blank">matplotlib.pyplot.scatter</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html" title="matplotlib.pyplot.bar" target="_blank">matplotlib.pyplot.bar</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html" title="matplotlib.pyplot.hist" target="_blank">matplotlib.pyplot.hist</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlabel.html" title="matplotlib.pyplot.xlabel" target="_blank">matplotlib.pyplot.xlabel</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylabel.html" title="matplotlib.pyplot.ylabel" target="_blank">matplotlib.pyplot.ylabel</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html" title="matplotlib.pyplot.title" target="_blank">matplotlib.pyplot.title</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html" title="matplotlib.pyplot.subplot" target="_blank">matplotlib.pyplot.subplot</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html" title="matplotlib.pyplot.subplots" target="_blank">matplotlib.pyplot.subplots</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot2grid.html" title="matplotlib.pyplot.subplot2grid" target="_blank">matplotlib.pyplot.subplot2grid</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html" title="matplotlib.pyplot.suptitle" target="_blank">matplotlib.pyplot.suptitle</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xscale.html" title="matplotlib.pyplot.xscale" target="_blank">matplotlib.pyplot.xscale</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html" title="matplotlib.pyplot.yscale" target="_blank">matplotlib.pyplot.yscale</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlim.html" title="matplotlib.pyplot.xlim" target="_blank">matplotlib.pyplot.xlim</a> </li>
<li><a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylim.html" title="matplotlib.pyplot.ylim" target="_blank">matplotlib.pyplot.ylim</a> </li>
<li><a href="https://matplotlib.org/2.0.2/mpl_toolkits/mplot3d/tutorial.html" title="mplot3d tutorial" target="_blank">mplot3d tutorial</a> </li>
<li><a href="https://matplotlib.org/stable/tutorials/index.html" title="additional tutorials" target="_blank">additional tutorials</a> </li>
</ul>

## General
* What is a plot?
* What is a scatter plot? line graph? bar graph? histogram?
* What is matplotlib?
* How to plot data with matplotlib
* How to label a plot
* How to scale an axis
* How to plot multiple sets of data at the same time

## Installing Matplotlib 3.0

    pip install --user matplotlib==3.0
    pip install --user Pillow
    sudo apt-get install python3-tk

To check that it has been successfully downloaded, use `pip list`.

## Configure X11 Forwarding
Update your `Vagrantfile` to include the following:

    Vagrant.configure(2) do |config|
    ...
    config.ssh.forward_x11 = true
    end

If you are running vagrant on a Mac, you will have to install XQuartz and restart your computer.

If you are running vagrant on a Windows computer, you may have to follow [these instructions](https://jcook0017.medium.com/how-to-enable-x11-forwarding-in-windows-10-on-a-vagrant-virtual-box-running-ubuntu-d5a7b34363f).

Once complete, you should simply be able to vagrant ssh to log into your VM and then any GUI application should forward to your local machine.

Hint for emacs users: you will have to use emacs -nw to prevent it from launching its GUI.

# Tasks

## 0. Line Graph
Complete the following source code to plot y as a line graph:

* y should be plotted as a solid red line
* The x-axis should range from 0 to 10
```
    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt

    y = np.arange(0, 11) ** 3

    # your code here
```
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/664b2543b48ef4918687.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200719%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200719T213618Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=06c27f290e0c1763ae31538e755c3f7decb2180b2e53fb5116a6c540194199ef)

File: [`0-line.py`](https://github.com/paurbano/holbertonschool-machine_learning/blob/master/math/0x01-plotting/0-line.py)

## 1. Scatter
Complete the following source code to plot x ↦ y as a scatter plot:

* The x-axis should be labeled Height (in)
* The y-axis should be labeled Weight (lbs)
* The title should be Men's Height vs Weight
* The data should be plotted as magenta points

File: [`1-scatter.py`](https://github.com/paurbano/holbertonschool-machine_learning/blob/master/math/0x01-plotting/1-scatter.py)

## 2. Change of scale
Complete the following source code to plot x ↦ y as a line graph:

* The x-axis should be labeled Time (years)
* The y-axis should be labeled Fraction Remaining
* The title should be Exponential Decay of C-14
* The y-axis should be logarithmically scaled
* The x-axis should range from 0 to 28650

```
    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    # your code here
```

File: [`2-change_scale.py`](https://github.com/paurbano/holbertonschool-machine_learning/blob/master/math/0x01-plotting/2-change_scale.py)

## 3. Two is better than one
Complete the following source code to plot x ↦ y1 and x ↦ y2 as line graphs:

* The x-axis should be labeled Time (years)
* The y-axis should be labeled Fraction Remaining
* The title should be Exponential Decay of Radioactive Elements
* The x-axis should range from 0 to 20,000
* The y-axis should range from 0 to 1
* x ↦ y1 should be plotted with a dashed red line
* x ↦ y2 should be plotted with a solid green line
* A legend labeling x ↦ y1 as C-14 and x ↦ y2 as Ra-226 should be placed in the upper right hand corner of the plot

```
    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    # your code here
```

File: [`3-two.py`](https://github.com/paurbano/holbertonschool-machine_learning/blob/master/math/0x01-plotting/3-two.py)

## 4. Frequency
Complete the following source code to plot a histogram of student scores for a project:

* The x-axis should be labeled Grades
* The y-axis should be labeled Number of Students
* The x-axis should have bins every 10 units
* The title should be Project A
* The bars should be outlined in black
```
    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # your code here
```

File: [`4-frequency.py`](https://github.com/paurbano/holbertonschool-machine_learning/blob/master/math/0x01-plotting/4-frequency.py)

## 5. All in One
Complete the following source code to plot all 5 previous graphs in one figure:

* All axis labels and plot titles should have a font size of `x-small` (to fit nicely in one figure)
* The plots should make a 3 x 2 grid
* The last plot should take up two column widths (see below)
* The title of the figure should be `All in One`
```
    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # your code here
```
File: [`5-all_in_one.py`](https://github.com/paurbano/holbertonschool-machine_learning/blob/master/math/0x01-plotting/5-all_in_one.py)

## 6. Stacking Bars
Complete the following source code to plot a stacked bar graph:

* fruit is a matrix representing the number of fruit various people possess
    * The columns of fruit represent the number of fruit Farrah, Fred, and Felicia have, respectively
    * The rows of fruit represent the number of apples, bananas, oranges, and peaches, respectively
* The bars should represent the number of fruit each person possesses:
    * The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
    * Each fruit should be represented by a specific color:
        * `apples` = red
        * `bananas` = yellow
        * `oranges` = orange (`#ff8000`)
        * `peaches` = peach (`#ffe5b4`)
        * A legend should be used to indicate which fruit is represented by each color
    * The bars should be stacked in the same order as the rows of fruit, from bottom to top
    * The bars should have a width of 0.5
* The y-axis should be labeled `Quantity of Fruit`
* The y-axis should range from 0 to 80 with ticks every 10 units
* The title should be `Number of Fruit per Person`
```
    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))

    # your code here
```

File: [`6-bars.py`](https://github.com/paurbano/holbertonschool-machine_learning/blob/master/math/0x01-plotting/6-bars.py)