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



# For these experiments, split the ISBI2012 train cube as follows:
train = lambda X: X[0:10,:,:]
valid = lambda X: X[10:20,:,:]
test = lambda X: X[20:30,:,:]



def get_args():
    """Command line parameters for the 'deploy' procedure.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-X', dest='dataFileName', type=str, required=True,
		    help='EM data file')

    parser.add_argument('-Y', dest='labelsFileName', type=str, required=True,
		    help='Ground truth labels for X')

    parser.add_argument('--brightness-quantile', 
		    dest='brightQuant', type=float, 
		    default=0.97,
		    help='top quantile for non-membrane pixels.')

    args = parser.parse_args()
    assert(args.brightQuant <= 1.0)
    assert(args.brightQuant > 0)
    return args



if __name__ == "__main__":
    args = get_args();
 
    X = emlib.load_cube(args.dataFileName, np.uint8)
    Y = emlib.load_cube(args.labelsFileName, np.uint8)

    # remap Y labels from ISBI convention to membrane-vs-non-membrane
    Y[Y==0] = 1;      #  membrane 
    Y[Y==255] = 0;    #  non-membrane

    # change type of Y so can use -1 as a value.
    Y = Y.astype(np.int8)

    Xtrain = train(X);   Ytrain = train(Y)
    Xvalid = valid(X);   Yvalid = valid(Y)
    Xtest  = test(X);    Ytest = test(Y)

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
    outDir = os.path.split(args.dataFileName)[0]
    np.save(os.path.join(outDir, 'Xtrain.npy'), Xtrain)
    np.save(os.path.join(outDir, 'Ytrain.npy'), Ytrain)
    np.save(os.path.join(outDir, 'Xvalid.npy'), Xvalid)
    np.save(os.path.join(outDir, 'Yvalid.npy'), Yvalid)
    np.save(os.path.join(outDir, 'Xtest.npy'), Xtest)
    np.save(os.path.join(outDir, 'Ytest.npy'), Ytest)
    
    print('[preprocess]: done!')
