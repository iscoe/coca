"""  
  Uses PyCaffe to generate pixel-level predictions for an EM volume.

  This is a modified version of (the slightly more complicated) 
  ../../deploy.py script

  See the Makefile for example usage.
"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import sys, os, argparse, time
import pdb

import numpy as np
import scipy.io

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import emlib


def numel(X): return np.prod(X.shape)



def get_args():
    """Command line parameters for the 'deploy' procedure.
    """
 
    parser = argparse.ArgumentParser()

    #----------------------------------------
    # Parameters for defining the neural network
    #----------------------------------------
    parser.add_argument('-n', dest='netFile', type=str, required=True,
                        help='Caffe network prototxt file')
    parser.add_argument('-m', dest='model', type=str, required=True,
                        help='Model weights file generated during training')

    parser.add_argument('--gpu', dest='gpu', type=int,
		    default=1, 
		    help='GPU id to use (or -1 for CPU)')


    #----------------------------------------
    # Data set parameters.
    #----------------------------------------
    parser.add_argument('-X', dest='dataFileName', type=str, required=True,
		    help='EM data file')

    parser.add_argument('--eval-slices', dest='evalSliceExpr', type=str,
		    default='', 
		    help='Which slices to process (default is all)')
    parser.add_argument('--yhat', dest='outFileNameY', type=str, 
		    default='', 
		    help='Probability output file (full path)')
    parser.add_argument('--max-brightness', 
		    dest='maxBrightness', type=int, 
		    default=sys.maxint,
		    help='Only evaluates pixels with this brightness')

    return parser.parse_args()




def _eval_cube(net, X, M, batchDim):
    """Uses Caffe to make predictions for the EM cube X."""
    
    assert(batchDim[2] == batchDim[3])  # currently we assume tiles are square

    #--------------------------------------------------
    # initialize variables and storage needed in the processing loop below 
    #--------------------------------------------------
    tileRadius = int(batchDim[2]/2)
    Xi = np.zeros(batchDim, dtype=np.float32)
    yDummy = np.zeros((batchDim[0],), dtype=np.float32) 
    cnnTime = 0.0                                 # time spent doing core CNN operations
    nClasses = net.blobs['prob'].data.shape[1]    # *** Assumes a layer called "prob"

    # allocate memory for return values
    Yhat = np.zeros((nClasses, X.shape[0], X.shape[1], X.shape[2]))
        
    print "[deploy]: Yhat shape: %s" % str(Yhat.shape)
    sys.stdout.flush()

    #--------------------------------------------------
    # process the cube
    #--------------------------------------------------
    tic = time.time()
    lastChatter = None
 
    for Idx, pct in emlib.interior_pixel_generator(X, tileRadius, batchDim[0], mask=M):
        # populate the mini-batch buffer
        for jj in range(Idx.shape[0]):
            a = Idx[jj,1] - tileRadius
            b = Idx[jj,1] + tileRadius + 1
            c = Idx[jj,2] - tileRadius
            d = Idx[jj,2] + tileRadius + 1
            Xi[jj, 0, :, :] = X[ Idx[jj,0], a:b, c:d ]

        # CNN forward pass
        _tmp = time.time()
        net.set_input_arrays(Xi, yDummy)
        out = net.forward()
        yiHat = out['prob']
        cnnTime += time.time() - _tmp

        # On some version of Caffe, yiHat is (batchSize, nClasses, 1, 1)
        # On newer versions, it is natively (batchSize, nClasses)
        # The squeeze here is to accommodate older versions
        yiHat = np.squeeze(yiHat) 

        # store the per-class probability estimates.
        #
        # * Note that on the final iteration, the size of yiHat may not match
        #   the remaining space in Yhat (unless we get lucky and the data cube
        #   size is a multiple of the mini-batch size).  This is why we slice
        #   yijHat before assigning to Yhat.
        for jj in range(nClasses):
            yijHat = yiHat[:,jj]                   # get slice containing probabilities for class j
            assert(len(yijHat.shape)==1)           # should be a vector (vs tensor)
            Yhat[jj, Idx[:,0], Idx[:,1], Idx[:,2]] = yijHat[:Idx.shape[0]]   # (*)

        # provide feedback on progress so far    
        elapsed = (time.time() - tic) / 60.
        if (lastChatter is None) or ((elapsed - lastChatter) > 2):
            print('[deploy]:  %0.2f min elapsed (%0.2f CNN min, %0.2f%% complete)' % (elapsed, cnnTime/60., 100.*pct))
            sys.stdout.flush()
            lastChatter = elapsed


    # all done!
    print('[deploy]: Finished processing cube.')
    print('[deploy]: Net time was: %0.2f min (%0.2f CNN min)' % (elapsed, cnnTime/60.))
    return Yhat


    

if __name__ == "__main__":
    args = get_args()

    #----------------------------------------
    # parse information from the prototxt files
    #----------------------------------------
    netFn = str(args.netFile)  # unicode->str to avoid caffe API problems
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(), netParam)
 
    batchDim = emlib.infer_data_dimensions(netFn)

    if len(args.outFileNameY):
        outFileNameY = args.outFileNameY
    else:
        outFileNameY = os.path.join(os.path.split(args.netFile)[0], 'Yhat_' + os.path.split(args.dataFileName)[-1])



    #----------------------------------------
    # Load and preprocess data set
    #----------------------------------------
    X = emlib.load_cube(args.dataFileName, np.float32)

    # mirror edges so that every pixel in the original data set can act
    # as the center pixel of some tile    
    borderSize = int(batchDim[2]/2)
    X = emlib.mirror_edges(X, borderSize)

    if len(args.evalSliceExpr):  # optional: pare down to a subset of slices
        idx = eval(args.evalSliceExpr)
        X = X[idx,:,:]

    # pixels that are sufficiently bright are trivial to classify
    # and can be omitted.
    Mask = np.ones(X.shape, dtype=np.bool)
    if args.maxBrightness < sys.maxint:
	    Mask[X > args.maxBrightness] = False

    # rescale (assumes this was done during training!)
    X = X / 255.0


    #----------------------------------------
    # Create the Caffe network
    #
    # Note: This assumes a recent (>= May 2015) version of Pycaffe.
    #       Older versions had different APIs
    #----------------------------------------
    phaseTest = 1  # 1 := test mode
    net = caffe.Net(netFn, args.model, phaseTest)

    for name, blobs in net.params.iteritems():
        print("%s : %s" % (name, blobs[0].data.shape))
        
    # specify training mode and CPU or GPU
    if args.gpu >= 0:
	caffe.set_mode_gpu()
	caffe.set_device(args.gpu)
    else:
	caffe.set_mode_cpu()

    #----------------------------------------
    # Do it
    #----------------------------------------
    print('==========================================================')
    print(args)
    print('')
    print('[deploy]: data shape: %s' % str(X.shape))
    print('[deploy]: batch shape: %s' % str(batchDim))
    print('[deploy]: probability output file: %s' % outFileNameY)
    print('[deploy]: mask is omitting %0.2f%% of the raw data' % (100 * np.sum(Mask==0) / numel(Mask)))
    print('==========================================================')
    sys.stdout.flush()

    Yhat = _eval_cube(net, X, Mask, batchDim)
 
    # discard border/mirroring
    Yhat = Yhat[:, :, borderSize:(-borderSize), borderSize:(-borderSize)]
     
    print('[deploy]: Finished.  Saving estimates...')
    np.save(outFileNameY, Yhat)
    scipy.io.savemat(outFileNameY+".mat", {'Yhat' : Yhat})

    print('[deploy]: exiting...')


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
