""" Preprocess the ISBI data set.
"""


__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import argparse, os.path
import numpy as np
from scipy.stats.mstats import mquantiles
import scipy.io

import emlib



def get_args():
    """Command line parameters for the 'deploy' procedure.

    You will probably want to override the train/valid/test split
    to better suit your problem of interest...
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-X', dest='dataFileName', type=str, required=True,
		    help='EM data file')

    parser.add_argument('-Y', dest='labelsFileName', type=str, required=True,
		    help='Ground truth labels for X')

    parser.add_argument('--train-slices', dest='trainSlices', 
		    type=str, default='range(10)', 
		    help='which slices to use for training')

    parser.add_argument('--valid-valid', dest='validSlices', 
		    type=str, default='range(10,20)', 
		    help='which slices to use for validation')

    parser.add_argument('--test-valid', dest='testSlices', 
		    type=str, default='range(20,30)', 
		    help='which slices to use for test')

    parser.add_argument('--brightness-quantile', dest='brightQuant', 
		    type=float, default=0.97,
		    help='top quantile for non-membrane pixels.')

    parser.add_argument('--out-dir', dest='outDir', 
		    type=str, default='./', 
		    help='output directory')

    args = parser.parse_args()
    assert(args.brightQuant <= 1.0)
    assert(args.brightQuant > 0)

    # map strings to python objects (XXX: a cleaner way than eval)
    args.trainSlices = eval(args.trainSlices)
    args.validSlices = eval(args.validSlices)
    args.testSlices = eval(args.testSlices)

    return args



if __name__ == "__main__":
    args = get_args();

    #outDir = os.path.split(args.dataFileName)[0]
    if not os.path.isdir(args.outDir):
        os.mkdir(args.outDir)


    X = emlib.load_cube(args.dataFileName, np.uint8)
    Y = emlib.load_cube(args.labelsFileName, np.uint8)

    # remap Y labels from ISBI convention to membrane-vs-non-membrane
    Y[Y==0] = 1;      #  membrane 
    Y[Y==255] = 0;    #  non-membrane

    # change type of Y so can use -1 as a value.
    Y = Y.astype(np.int8)

    Xtrain = X[args.trainSlices,:,:]; Ytrain = Y[args.trainSlices,:,:]
    Xvalid = X[args.validSlices,:,:]; Yvalid = Y[args.validSlices,:,:]
    Xtest = X[args.testSlices,:,:];   Ytest = Y[args.testSlices,:,:]

    # brightness thresholding 
    thresh = mquantiles(np.concatenate((Xtrain[Ytrain==1], Xvalid[Yvalid==1])), args.brightQuant)
    pctOmitted = 100.0*np.sum(X > thresh) / np.prod(np.size(X))
    print('[preprocess]: percent of pixels omitted by brightness filter: %0.2f' % pctOmitted)

    Ytrain[Xtrain > thresh] = -1
    Yvalid[Xvalid > thresh] = -1
    Ytest[Xtest > thresh] = -1

    # EM data preprocessing
    # For now, do nothing.

    # save results
    np.save(os.path.join(outDir, 'Xtrain.npy'), Xtrain)
    np.save(os.path.join(outDir, 'Ytrain.npy'), Ytrain)
    np.save(os.path.join(outDir, 'Xvalid.npy'), Xvalid)
    np.save(os.path.join(outDir, 'Yvalid.npy'), Yvalid)
    np.save(os.path.join(outDir, 'Xtest.npy'), Xtest)
    np.save(os.path.join(outDir, 'Ytest.npy'), Ytest)
    
    print('[preprocess]: done!')
