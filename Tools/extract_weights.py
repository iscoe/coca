# Extracts weights from a Caffe model file.
#
#  This script saves the weights from each Caffe DNN layer to a
#  separate file.  These weight files can be analyzed individually or
#  used to replicate the action of the CNN in another environment
#  (e.g. Matlab).
#
#  This script requires the PyCaffe interface, a model file (created
#  during a Caffe training run) and the corresponding Caffe network
#  prototxt file.
#
#  Example:
#    PYTHONPATH=~/Apps/caffe/python python coca/Tools/extract_weights.py -m iter_180000.caffemodel  -n n3-net.prototxt
#
# May 2015, Mike Pekala


import sys, os, argparse, time
import pdb

import numpy as np
import scipy.io



def get_args():
    parser = argparse.ArgumentParser('Extract CNN weights')

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
            # Assume here index 0 are the weights and 1 is the bias.
            # This seems to be the case in Caffe.
            if ii == 0:
                name2 = name + "Weight"
            elif ii == 1:
                name2 = name + "Bias"
            else:
                pass # This is not expected
                
            print("%s : %s" % (name2, blobs[ii].data.shape))
            scipy.io.savemat(name2+".mat", {name2 : blobs[ii].data})
