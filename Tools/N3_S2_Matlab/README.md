This directory contains code for simulating a particular Caffe CNN in Matlab.  The motivation is to have an alternative to the somewhat computationally expensive procedure of extracting a tile centered on every pixel and then running each tile through the CNN individually.  Since adjacent tiles have many pixels in common it is more efficient to convolve the Caffe weights with the entire image.

As of this writing, this code is specific to a particular CNN/experiment and is not meant for general purpose use.  The toplevel script is **run_n3_s2.m**, which emulates a particular CNN that has 3 convolutional layers and 2 inner product layers followed by a softmax.


