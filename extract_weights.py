# Extracts weights from a Caffe model file (saves weights from each
# layer to separate files).
#
#  Example:
#    PYTHONPATH=~/Apps/caffe/python python coca/extract_weights.py -m iter_180000.caffemodel  -n n3-net.prototxt
#
# May 2015, Mike Pekala


import sys, os, argparse, time
import pdb

import numpy as np
import scipy.io



def get_args():
 
    parser = argparse.ArgumentParser('Extract CNN weights')

    #----------------------------------------
    # Parameters for defining the neural network
    #----------------------------------------
    parser.add_argument('-m', dest='modelFile', type=str, required=True,
                        help='Caffe model weights file to parse')
    parser.add_argument('-n', dest='netFile', type=str, required=True,
                        help='Network prototxt file associated with model')

    return parser.parse_args()




if __name__ == "__main__":
    import caffe

    args = get_args()
    net = caffe.Net(args.netFile, args.modelFile)
    for name, blobs in net.params.iteritems():
        for ii in range(len(blobs)):
            # assume here index 0 are the weights and 1 is the bias
            if ii == 0:
                name2 = name + "Weight"
            elif ii == 1:
                name2 = name + "Bias"
            else:
                pass # This is not expected
                
            print("%s : %s" % (name2, blobs[ii].data.shape))
            scipy.io.savemat(name2+".mat", {name2 : blobs[ii].data})
