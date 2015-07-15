# COCA 
 This repository is for the Caffe-based Object Classification and
 Annotation (COCA) project, which uses convolutional neural networks to find
 structures in image data.

The general approach is to apply a sliding window detector to classify individual pixels/voxels in a 3D volume of image data.  The effort initially began as an exercise in reproducing the results from [Ciresan et. al. 2012](http://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images) and evolved into a wrapper that performs data augmentation for [Caffe](http://caffe.berkeleyvision.org/).


In order to use this package, one must first have Caffe (and the
associated Python API, termed "pycaffe") installed.  Note that pycaffe evolves over time so depending upon your particular version some minor code surgery may be required.  As of May 2015 this code has been verified to work with Caffe versions:

 * Development branch October 3, 2014
 * Main branch May 28, 2015 (after applying this Caffe [patch](https://github.com/BVLC/caffe/issues/2334) )

You must also supply data for training and test.

### Files

There are three main files that comprise COCA:

- **train.py**  Used to train new Caffe models.
- **deploy.py**   Evaluates new data using previously trained Caffe models.
- **emlib.py**   Helper functions used by the train and deploy scripts.

You should only need to interact with train.py and deploy.py (unless you want to change how things work under the hood).  Both train.py and deploy.py are meant to be run from the command line; use the "-h" flag to see available command line options, e.g.


    pekalmj1-ml1:coca pekalmj1$ python train.py -h
    usage: train.py [-h] [-s SOLVER] [-gpu GPU] -X TRAINFILENAME -Y LABELSFILENAME
                    [-M MASKFILENAME] [--train-slices TRAINSLICESEXPR]
                    [--valid-slices VALIDSLICESEXPR]
                    [--snapshot-prefix SNAPPREFIX] [--omit-labels OMITLABELS]
                    [--rotate-data ROTATEDATA]

    optional arguments:
      -h, --help            show this help message and exit
      -s SOLVER             Caffe solver prototxt file to use for training
      -gpu GPU              Specifies GPU device ID to use (implies use Caffe in
                            GPU mode)
      -X TRAINFILENAME      Filename of the training data volume
      -Y LABELSFILENAME     Filename of the training labels volume
      -M MASKFILENAME       Filename of the voxel mask volume (optional)
      --train-slices TRAINSLICESEXPR
                            A python-evaluatable string indicating which slices should be used for training
      --valid-slices VALIDSLICESEXPR
                            A python-evaluatable string indicating which slices should be used for validation
      --snapshot-prefix SNAPPREFIX
                            (optional) override the "snapshot_prefix" in the solver file
      --omit-labels OMITLABELS
                            (optional) list of labels to omit
      --rotate-data ROTATEDATA
                            (optional) set to 1 to apply arbitrary rotations to tiles
    
For one example of how to use COCA with real data, please see the
[vesicle](https://github.com/openconnectome/vesicle) project.  In particular, see the "vesicle-cnn" directory.
