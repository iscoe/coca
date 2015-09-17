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

  # Create a training volume using the first 20 slices of ISBI 2012
  PYTHONPATH=/home/pekalmj1/Apps/caffe/python:.. python make_lmdb.py \
          --use-slices "range(0,20)" \
          -X /home/pekalmj1/Data/EM_2012/train-volume.tif \
          -Y /home/pekalmj1/Data/EM_2012/train-labels.tif \
          -o train.lmdb

  # Create a valiation volume using the last 10 slices of ISBI 2012
  PYTHONPATH=/home/pekalmj1/Apps/caffe/python:.. python make_lmdb.py \
          --use-slices "range(20,30)" \
          -X /home/pekalmj1/Data/EM_2012/train-volume.tif \
          -Y /home/pekalmj1/Data/EM_2012/train-labels.tif \
          -o valid.lmdb

  # Create a test volume using (a subset of) ISBI 2012 test
  PYTHONPATH=/home/pekalmj1/Apps/caffe/python:.. python make_lmdb.py \
          --use-slices "range(0,5)" \
          -X /home/pekalmj1/Data/EM_2012/test-volume.tif \
          -o test.lmdb


  Note: I make pretty liberal use of eval() below, so this is
        not very "secure"...

"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"



import argparse, os.path, time
import numpy as np
import lmdb

import caffe

import emlib



def get_args():
    """Command line parameters for the training procedure.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-X', dest='emFileName', 
            type=str, required=True, 
            help='Filename of the EM data volume')

    parser.add_argument('-Y', dest='labelsFileName', 
            type=str, default='',
            help='Filename of the training labels volume (if applicable)')

    parser.add_argument('-o', dest='outDir', 
            type=str, default='./out.lmdb',
            help='output directory name')


    parser.add_argument('--omit-labels', dest='omitLabels', 
            type=str, default='[]', 
            help='list of labels to omit')

    parser.add_argument('--tile-size', dest='tileSize', 
            type=int, default=65,
            help='size of each data tile (should be an odd, positive integer)')
    
    parser.add_argument('--use-slices', dest='slicesExpr', 
            type=str, default='range(0,20)', 
            help='A python-evaluatable string indicating which slices should be used from the volumes')
    
    return parser.parse_args()



def fix_class_labels(Yin, omitLabels):
    """Class labels must be contiguous natural numbers starting at 0.
    This is because they are mapped to indices at the output of the CNN.
    This function remaps the input y values if needed.

    Any pixels that should be ignored will have class label of -1.
    """
    if Yin is None: return None

    yAll = np.sort(np.unique(Yin))
    yAll = [y for y in yAll if y not in omitLabels]

    Yout = -1*np.ones(Y.shape, dtype=Y.dtype)
    for yIdx, y in enumerate(yAll):
        Yout[Y==y] = yIdx

    return Yout




if __name__ == "__main__":
    args = get_args()
    tileRadius = np.floor(args.tileSize/2)
    nMiniBatch = 1000 # here, a "mini-batch" is really an LMDB transaction size

    # make sure we don't clobber an existing output
    if os.path.exists(args.outDir):
        raise RuntimeError('Output path "%s" already exists; please move out of the way and try again' % args.outDir)


    print('[make_lmdb]: loading EM data file: %s' % args.emFileName)
    X = emlib.load_cube(args.emFileName, np.float32)

    if args.labelsFileName: 
        print('[make_lmdb]: loading labels file: %s' % args.labelsFileName) 
        Y = emlib.load_cube(args.labelsFileName, np.float32)
        Y = fix_class_labels(Y, eval(args.omitLabels))
        assert(Y.shape == X.shape)
    else:
        print('[make_lmdb]: no labels file; assuming this is a test volume')
        Y = np.zeros(X.shape)
   
    # usually we expect fewer slices in Z than pixels in X or Y.
    # Make sure the dimensions look ok before proceeding.
    assert(X.shape[0] < X.shape[1])
    assert(X.shape[0] < X.shape[2])

    # Identify the subset of the data to use for training.
    sliceIdx = eval(args.slicesExpr)
    X = X[sliceIdx, :, :]  # python puts the z dimension first...
    Y = Y[sliceIdx, :, :]

    print('[make_lmdb]: EM volume shape: %s' % str(X.shape))
    print('[make_lmdb]: yAll is %s' % np.unique(Y))
    print('[make_lmdb]: %d pixels will be omitted\n' % np.sum(Y==-1))

    # Create the output database.
    # Multiply the actual size by a fudge factor to get a safe upper bound
    dbSize = (X.nbytes * args.tileSize * args.tileSize + Y.nbytes) * 10
    env = lmdb.open(args.outDir, map_size=dbSize)

    # Extract all possible tiles.
    # This corresponds to extracting one "epoch" worth of tiles.
    tileId = 0
    lastChatter = -1
    tic = time.time()

    it = emlib.stratified_interior_pixel_generator(Y, tileRadius, nMiniBatch, omitLabels=[-1])
    for Idx, epochPct in it: 
        # Each mini-batch will be added to the database as a single transaction.
        with env.begin(write=True) as txn:
            # Translate indices Idx -> tiles Xi and labels yi.
            for jj in range(Idx.shape[0]):
                yi = int(Y[ Idx[jj,0], Idx[jj,1], Idx[jj,2] ])
                a = Idx[jj,1] - tileRadius
                b = Idx[jj,1] + tileRadius + 1
                c = Idx[jj,2] - tileRadius
                d = Idx[jj,2] + tileRadius + 1
                Xi = X[ Idx[jj,0], a:b, c:d ]
                assert(Xi.shape == (args.tileSize, args.tileSize))

                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = 1
                datum.height = Xi.shape[0]
                datum.width = Xi.shape[1]
                try: 
                    datum.data = Xi.tobytes()
                except AttributeError:
                    datum.data = Xi.tostring() # for numpy < 1.9
                datum.label = yi
                strId = '{:08}'.format(tileId)

                txn.put(strId.encode('ascii'), datum.SerializeToString())
                tileId += 1

        #if np.floor(epochPct) > lastChatter: 
        print('[make_lmdb] %% %0.2f done (%0.2f min)' % ((100*epochPct), (time.time() - tic)/60))
        lastChatter = epochPct



# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
