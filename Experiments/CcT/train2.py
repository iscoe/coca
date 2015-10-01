""" A refactored version of ../../train.py

** NOTE **
This code assumes your network has the following layer types/tops:

    top name |  Layer type
    ---------+-----------------
      loss   |  SoftmaxWithLoss
      acc    |  Accuracy
      prob   |  Softmax
"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import sys, os, argparse, time
import pdb

import numpy as np
import scipy

import emlib




numel = lambda(X): np.prod(X.shape)



def _get_args():
    """Command line parameters.
    """
    
    parser = argparse.ArgumentParser()

    #----------------------------------------
    # Parameters for defining and training the neural network
    #----------------------------------------
    parser.add_argument('--solver', dest='solver', 
		    type=str, required=True, 
		    help='Caffe prototxt file')
 
    parser.add_argument('--gpu', dest='gpu', 
		    type=int, default=-1, 
		    help='GPU ID to use')


    #----------------------------------------
    # Data set parameters.  
    #----------------------------------------
    parser.add_argument('--x-train', dest='emTrainFile', 
		    type=str, required=True,
		    help='Filename of the training data volume')
    parser.add_argument('--y-train', dest='labelsTrainFile', 
		    type=str, required=True, 
		    help='Filename of the training labels volume')

    parser.add_argument('--x-valid', dest='emValidFile', 
		    type=str, required=True,
		    help='Filename of the validation data volume')
    parser.add_argument('--y-valid', dest='labelsValidFile', 
		    type=str, required=True, 
		    help='Filename of the validation labels volume')

    
    parser.add_argument('--omit-labels', dest='omitLabels', 
		    type=str, default='(-2, -1,)', 
		    help='(optional) list of labels to omit')
    parser.add_argument('--rotate-data', dest='rotateData', 
		    type=int, default=0, 
		    help='(optional) 1 := apply arbitrary rotations')
    parser.add_argument('--only-slices', dest='onlySlices', 
		    type=str, default='', 
		    help='(optional) limit experiment to a subset of slices')

    return parser.parse_args()



#-------------------------------------------------------------------------------
# Functions for training a CNN
#-------------------------------------------------------------------------------
def _xform_minibatch(X, rotate=False):
    """Synthetic data augmentation for one mini-batch.
    
    Parameters: 
       X := Mini-batch data (# slices, # channels, rows, colums) 
       
       rotate := a boolean; when true, will rotate the mini-batch X
                 by some angle in [0, 2*pi)

    Note: for some reason, the implementation of row and column reversals, e.g.
               X[:,:,::-1,:]
          break PyCaffe.  Numpy must be doing something under the hood 
          (e.g. changing from C order to Fortran order) to implement this 
          efficiently which is incompatible w/ PyCaffe.  
          Hence the explicit construction of X2 with order 'C'.

    """
    X2 = np.zeros(X.shape, dtype=np.float32, order='C')

    toss = np.random.rand()
    if toss < .2:
        X2[:,:,:,:] = X[:,:,::-1,:]    # fliplr
    elif toss < .4:
        X2[:,:,:,:] = X[:,:,:,::-1]    # flipud
    else:
        X2[...] = X[...]               # no transformation

    if rotate:
        angle = np.random.rand() * 360.0
        fillColor = np.max(X)
        X2 = scipy.ndimage.rotate(X2, angle, axes=(2,3), reshape=False, cval=fillColor)

    return X2



class TrainInfo:
    def __init__(self, solverParam):
        self.param = solverParam

        self.isModeStep = (solverParam.lr_policy == u'step')

        # This code only supports some learning strategies
        if not self.isModeStep:
            raise ValueError('Sorry - I only support step policy at this time')

        if (solverParam.solver_type != solverParam.SolverType.Value('SGD')):
            raise ValueError('Sorry - I only support SGD at this time')

        self.iter = 0
        self.cnnTime = 0
        self.netTime = 0

        #--------------------------------------------------
        # SGD parameters.  SGD with momentum is of the form:
        #
        #    V_{t+1} = \mu V_t - \alpha \nablaL(W_t)
        #    W_{t+1} = W_t + V_{t+1}
        #
        # where W are the weights and V the previous update.
        # Ref: http://caffe.berkeleyvision.org/tutorial/solver.html
        #
        #--------------------------------------------------
        self.alpha = solverParam.base_lr  # := learning rate
        self.mu = solverParam.momentum    # := momentum
        self.gamma = solverParam.gamma    # := step factor
        self.V = {}                       # := previous values (for momentum)

        # XXX: weight decay
        # XXX: layer-specific weights




def train_one_epoch(solver, X, Y, 
        trainInfo, 
        batchDim,
        outDir='./', 
        omitLabels=[], 
        data_augment=None):
    """ Trains a CNN for a single epoch.
      batchDim := (nExamples, nFilters, width, height)
    """

    # Pre-allocate some variables & storate.
    #
    tileRadius = int(batchDim[2]/2)
    Xi = np.zeros(batchDim, dtype=np.float32)
    yi = np.zeros((batchDim[0],), dtype=np.float32)
    yMax = np.max(Y).astype(np.int32)
    
    tic = time.time()
    it = emlib.stratified_interior_pixel_generator(Y, tileRadius, batchDim[0], omitLabels=omitLabels) 

    for Idx, epochPct in it: 
        # Map the indices Idx -> tiles Xi and labels yi 
        # 
        # Note: if Idx.shape[0] < batchDim[0] (last iteration of an epoch) 
        # a few examples from the previous minibatch will be "recycled" here. 
        # This is intentional (to keep batch sizes consistent even if data 
        # set size is not a multiple of the minibatch size). 
        # 
        for jj in range(Idx.shape[0]): 
            a = Idx[jj,1] - tileRadius 
            b = Idx[jj,1] + tileRadius + 1 
            c = Idx[jj,2] - tileRadius 
            d = Idx[jj,2] + tileRadius + 1 
            Xi[jj, 0, :, :] = X[ Idx[jj,0], a:b, c:d ]
            yi[jj] = Y[ Idx[jj,0], Idx[jj,1], Idx[jj,2] ] 

        # label-preserving data transformation (synthetic data generation)
        if data_augment is not None:
            Xi = data_augment(Xi)

        #----------------------------------------
        # one forward/backward pass and update weights
        #----------------------------------------
        _tmp = time.time()
        solver.net.set_input_arrays(Xi, yi)
        out = solver.net.forward()
        solver.net.backward()

        # SGD with momentum
        for lIdx, layer in enumerate(solver.net.layers):
            for bIdx, blob in enumerate(layer.blobs):
                key = (lIdx, bIdx)
                V = trainInfo.V.get(key, 0.0)
                Vnext = (trainInfo.mu * V) - (trainInfo.alpha * blob.diff)
                blob.data[...] += Vnext
                trainInfo.V[key] = Vnext

        # (try to) extract some useful info from the net
        loss = out.get('loss', None)
        acc = out.get('acc', None)  # accuracy is not required...

        # update run statistics
        trainInfo.cnnTime += time.time() - _tmp
        trainInfo.iter += 1
        trainInfo.netTime += (time.time() - tic)
        tic = time.time()


        #----------------------------------------
        # Some events occur on regular intervals.
        # Address these here.
        #----------------------------------------
        if (trainInfo.iter % trainInfo.param.snapshot) == 0:
            fn = os.path.join(outDir, 'iter_%06d.caffemodel' % trainInfo.iter)
            solver.net.save(str(fn))

        if trainInfo.isModeStep and ((trainInfo.iter % trainInfo.param.stepsize) ==0):
            trainInfo.alpha *= trainInfo.gamma

        if (trainInfo.iter % trainInfo.param.display) == 1: 
            print "[train]: completed iteration %d of %d (epoch complete=%0.2f%%;" % (trainInfo.iter, trainInfo.param.max_iter, 100.*epochPct)
            print "[train]:     %0.2f min elapsed (%0.2f CNN min)" % (trainInfo.netTime/60., trainInfo.cnnTime/60.)
            if loss: 
                print "[train]:     loss=%0.2f" % loss
            if acc: 
                print "[train]:     accuracy (train volume)=%0.2f" % acc
            sys.stdout.flush()
 
        if trainInfo.iter >= trainInfo.param.max_iter:
            break  # in case we hit max_iter on a non-epoch boundary
                
    # all finished with this epoch
    print "[train]:    epoch complete."
    return losses, acc



#-------------------------------------------------------------------------------
# Functions for "deploying" a CNN (i.e. forward pass only)
#-------------------------------------------------------------------------------


def predict(net, X, Y, batchDim, outDir, omitLabels=[]):
    """Generates a prediction for a volume.
    """

    # Pre-allocate some variables & storate.
    #
    tileRadius = int(batchDim[2]/2)
    Xi = np.zeros(batchDim, dtype=np.float32)
    yi = np.zeros((batchDim[0],), dtype=np.float32)
    yMax = np.max(Y).astype(np.int32)

    Mask = np.ones(Y.shape, dtype=bool)
    for yIgnore in omitLabels:
        Mask[Yvalid==yIgnore] = False

    print "[train]:  Evaluating %0.2f%% of cube" % (100.0*np.sum(Mask)/numel(Mask)) 
    it = emlib.interior_pixel_generator(Y, tileRadius, batchDim[0], mask=Mask)
    for Idx, epochPct in it: 
        # Extract subtiles from validation data set 
        for jj in range(Idx.shape[0]): 
            a = Idx[jj,1] - tileRadius 
            b = Idx[jj,1] + tileRadius + 1 
            c = Idx[jj,2] - tileRadius 
            d = Idx[jj,2] + tileRadius + 1 
            Xi[jj, 0, :, :] = Xvalid[ Idx[jj,0], a:b, c:d ]
            yi[jj] = Yvalid[ Idx[jj,0], Idx[jj,1], Idx[jj,2] ] 

        #---------------------------------------- 
        # one forward pass; no backward pass 
        #----------------------------------------
        net.set_input_arrays(Xi, yi)
        out = net.forward()

   


def _load_data(args):
    """Loads data sets and does basic preprocessing.
    """
    Xtrain = emlib.load_cube(args.emTrainFile, np.float32)
    Ytrain = emlib.load_cube(args.labelsTrainFile, np.float32)
    Xvalid = emlib.load_cube(args.emValidFile, np.float32)
    Yvalid = emlib.load_cube(args.labelsValidFile, np.float32)

    # usually we expect fewer slices in Z than pixels in X or Y.
    # Make sure the dimensions look ok before proceeding.
    assert(Xtrain.shape[0] < Xtrain.shape[1])
    assert(Xtrain.shape[0] < Xtrain.shape[2])

    # take a subset of slices (optional)
    if args.onlySlices:
        useSlices = eval(args.onlySlices)
        Xtrain = Xtrain[useSlices,:,:]
        Ytrain = Ytrain[useSlices,:,:]
        Xvalid = Xvalid[useSlices,:,:]
        Yvalid = Yvalid[useSlices,:,:]

    print('[train]: training data shape: %s' % str(Xtrain.shape))
    print('[train]: validation data shape: %s' % str(Xvalid.shape))

    # Class labels must be natural numbers (contiguous integers starting at 0)
    # because they are mapped to indices at the output of the network.
    # This next bit of code remaps the native y values to these indices.
    Ytrain = emlib.fix_class_labels(Ytrain, eval(args.omitLabels))
    Yvalid = emlib.fix_class_labels(Yvalid, eval(args.omitLabels))

    print('[train]: yAll is %s' % str(np.unique(Ytrain)))
    print('[train]: will use %0.2f%% of volume for training' % (100.*np.sum(Ytrain>=0)/numel(Ytrain)))

    # mirror edges of images so that every pixel in the original data set 
    # can act as a center pixel of some tile    
    borderSize = int(batchDim[2]/2)
    Xtrain = emlib.mirror_edges(Xtrain, borderSize)
    Ytrain = emlib.mirror_edges(Ytrain, borderSize)
    Xvalid = emlib.mirror_edges(Xvalid, borderSize)
    Yvalid = emlib.mirror_edges(Yvalid, borderSize)

    # Scale data to live in [0 1]
    offset = np.min(Xtrain)
    sf = 1.0*(np.max(Xtrain) - np.min(Xtrain))
    Xtrain = (Xtrain+offset) / sf
    Xvalid = (Xvalid+offset) / sf


    return Xtrain, Ytrain, Xvalid, Yvalid


#-------------------------------------------------------------------------------
if __name__ == "__main__":
    args = _get_args()
    
    import caffe
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format

    #----------------------------------------
    # parse information from the prototxt files
    #----------------------------------------
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(args.solver).read(), solverParam)

    netFn = solverParam.net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(), netParam)
    
    batchDim = emlib.infer_data_dimensions(netFn)
    assert(batchDim[2] == batchDim[3])  # tiles must be square
    print('[train]: batch shape: %s' % str(batchDim))

    outDir = str(solverParam.snapshot_prefix)   # unicode -> str
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
   
    # choose a synthetic data generating function
    if args.rotateData:
        syn_func = lambda V: _xform_minibatch(V, True)
        print('[train]:   WARNING: applying arbitrary rotations to data.  This may degrade performance in some cases...\n')
    else:
        syn_func = lambda V: _xform_minibatch(V, False)

    # specify CPU or GPU
    if args.gpu >= 0:
	caffe.set_mode_gpu()
	caffe.set_device(args.gpu)
    else:
	caffe.set_mode_cpu()

    #----------------------------------------
    # Create the Caffe solver
    # Note this assumes a relatively recent PyCaffe
    #----------------------------------------
    solver = caffe.SGDSolver(args.solver)
    for name, blobs in solver.net.params.iteritems():
        for bIdx, b in enumerate(blobs):
            print("%s[%d] : %s" % (name, bIdx, b.data.shape))
    for ii,layer in enumerate(solver.net.layers):
        print("layer %d: %s" % (ii, layer.type))
        for jj,blob in enumerate(layer.blobs):
            print("  blob %d has size %s" % (jj, str(blob.data.shape)))


    #----------------------------------------
    # Do training; save results
    #----------------------------------------
    Xtrain, Ytrain, Xvalid, Yvalid = _load_data(args)
    trainInfo = TrainInfo(solverParam)
    sys.stdout.flush()

    epoch = 1
    while trainInfo.iter < solverParam.max_iter: 
        print "[train]: Starting epoch %d" % epoch
        train_one_epoch(solver, Xtrain, Ytrain, 
            trainInfo, batchDim, outDir, 
            omitLabels=[-1], 
            data_augment=syn_func)

        # TODO: evaluate on validation data
        epoch += 1
 
    solver.net.save(str(os.path.join(outDir, 'final.caffemodel')))
    np.save(os.path.join(outDir, '%s_losses' % outDir), losses)
    np.save(os.path.join(outDir, '%s_acc' % outDir), acc)
    
    print('[train]: all done!')


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
