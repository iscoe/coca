"""Creates lmdb database files from EM data for use with Caffe.

  This script provides an alternative to the 'train.py' and
  'deploy.py' scripts in the parent directory.  
  Instead of dynamically generating subtiles for
  training, this script creates a single batch of
  of training tiles and writes them to an LMDB database.  
  This allows one to use the command-line version of Caffe (or 
  Caffe con Troll) to train a classifier.  Note this will not
  produce exactly the same result since the subsampling of the
  majority class and the synthetic data augmentation are 
  done slightly differently in these two approaches.

  This script requires emlib (from the parent directory) as
  well as python's lmdb interface (pip install lmdb) and
  Caffe's python interface.

  EXAMPLE USAGE:
  PYTHONPATH=/home/pekalmj1/Apps/caffe/python python make_lmdb.py

"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"



import argparse
import numpy as np
import lmdb

import caffe



def get_args():
    """Command line parameters for the training procedure.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-X', dest='trainFileName', 
            type=str, required=True, 
            help='Filename of the EM data volume')

    parser.add_argument('-Y', dest='labelsFileName', 
            type=str, default='',
            help='Filename of the training labels volume (if applicable)')
    
    parser.add_argument('--use-slices', dest='slicesExpr', 
            type=str, default='range(0,20)', 
            help='A python-evaluatable string indicating which slices should be used from the volumes')
    

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    print args

    

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
