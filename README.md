# coca
 This repository is for the Caffe-based Object Classification and
 Annotation project, which uses convolutional neural networks to find
 biological structures in image data.

As of May 2015 this package is primarily a wrapper that takes care of
pre-processing image data to facilitate using Caffe in a
sliding-window fashion to classify individual pixels/voxels in a 3D
volume of image data.  Examples include detecting membranes and
[synapses](http://arxiv.org/abs/1403.3724)  at the pixel level.

In order to use this package, one must first have Caffe (and the
associate Python API) installed.   Note that this was developed using
a somewhat old version of Caffe (circa December 2012) so it may be
incompatible with newer versions due to API changes.

You must also supply data for training and test.  For one example of
how to use COCA with real data, please see the
[vesicle](https://github.com/openconnectome/vesicle) project.

