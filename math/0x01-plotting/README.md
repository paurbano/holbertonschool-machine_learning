# 0x01. Plotting

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

If you are running vagrant on a Windows computer, you may have to follow these instructions.

Once complete, you should simply be able to vagrant ssh to log into your VM and then any GUI application should forward to your local machine.

Hint for emacs users: you will have to use emacs -nw to prevent it from launching its GUI.

# Tasks

## 0. Line Graph
Complete the following source code to plot y as a line graph:

* y should be plotted as a solid red line
* The x-axis should range from 0 to 10

    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt

    y = np.arange(0, 11) ** 3

    # your code here

[](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/664b2543b48ef4918687.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200719%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200719T213618Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=06c27f290e0c1763ae31538e755c3f7decb2180b2e53fb5116a6c540194199ef)

File: `0-line.py`

## 1. Scatter
Complete the following source code to plot x ↦ y as a scatter plot:

* The x-axis should be labeled Height (in)
* The y-axis should be labeled Weight (lbs)
* The title should be Men's Height vs Weight
* The data should be plotted as magenta points

## 2. Change of scale 