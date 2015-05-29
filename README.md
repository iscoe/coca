# COCA 
 This repository is for the Caffe-based Object Classification and
 Annotation (COCA) project, which uses convolutional neural networks to find
 biological structures in image data.

The general approach is to apply a sliding window detector to classify individual pixels/voxels in a 3D volume of image data.  The effort initially began as an exercise in reproducing the results from [Ciresan et. al. 2012](http://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images) using Theano.
As of May 2015 this package is primarily a wrapper that performs data augmentation for Caffe.


In order to use this package, one must first have Caffe (and the
associated Python API, termed "pycaffe") installed.  Note that pycaffe evolves over time so depending upon your particular version of Caffe some minor code surgery may be required.  As of May 2015 this code has been verified to work with Caffe versions:

 * Development branch October 3, 2014
 * Main branch May 28, 2015 (after applying this Caffe [patch](https://github.com/BVLC/caffe/issues/2334) )

You must also supply data for training and test.  For one example of
how to use COCA with real data, please see the
[vesicle](https://github.com/openconnectome/vesicle) project.

