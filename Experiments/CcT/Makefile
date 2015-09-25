# This is a quick-and-dirty approach to setting up the ISBI 2012 
# membrane classification problem so we can use the command-line 
# interface to Caffe (and Caffe con Troll).  
# 
# 1. To create the LMDB database (from raw ISBI data):
#      make lmdb-train
#      make lmdb-valid
#
# 2. To train a model using Caffe:
#      make caffe-train
# 
# 3. To generate timing estimates for Caffe:
#      make caffe-time-gpu
#      make caffe-time-cpu
# 
# 4. To generate timing estimates for Caffe con Troll (CcT):
#      TODO 
#
# NOTES:
# For expediency, we create LMDB databases that contain many 
# pre-computed tiles; in the past we created these tiles "lazily" to
# avoid creating large data sets with lots of redundancy.
#
# A future possible direction is some combination of CcT and 
# dense classification techniques (e.g. the semantic segmentation
# approach of Long et. al.)
#
# This directory contains the following:
#
#   o ISBI2012                  : Contains the original ISBI 2012 data set.
#   o n3-{net,solver}.prototxt  : Caffe configuration files.

BASE_DIR=../..


# we need PyCaffe and emlib.py in the PYTHONPATH
PY=PYTHONPATH=/home/pekalmj1/Apps/caffe/python:$(BASE_DIR) python

# for profiling (optional)
#PY=PYTHONPATH=/home/pekalmj1/Apps/caffe/python:.. python -m cProfile -s cumtime

# Define the subset of ISBI2012 to use for train and validation.
# For now, we'll use 20 slices for train and the last 10 for validation.
S_TRAIN="range(0,20)"
S_VALID="range(20,30)"

# The maximum number of tiles to extract
N_TILES=100000

# Specify which model to use in timing experiments.
CAFFE_MODEL=_iter_10000.caffemodel
CCT_MODEL=trained_model.bin.25-09-2015-04-46-54

# Number of iterations to use in timing experiments
NITERS=100


TAR=CcT-experiment.tar


#-------------------------------------------------------------------------------
default:
	echo "please explicitly choose a target"

tar :
	\rm -f $(BASE_DIR)/$(TAR)
	pushd $(BASE_DIR)/.. && tar cvf $(TAR) `find ./coca -name \*.py -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./coca -name \*.m -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./coca -name \*.md -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./coca -name \*.txt -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./coca -name \*.tif -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./coca -name Makefile -print`
	pushd $(BASE_DIR)/.. && tar rvf $(TAR) `find ./coca -name n3-\* -print`


#-------------------------------------------------------------------------------
# Targets for creating LMDB databases
#-------------------------------------------------------------------------------

# Creates an LMDB data set for training.
# This is just a ("tile-ified") subset of the ISBI 2012 training data set.
lmdb-train:
	$(PY) $(BASE_DIR)/Tools/make_lmdb.py --use-slices $(S_TRAIN) \
		-X ./ISBI2012/train-volume.tif \
		-Y ./ISBI2012/train-labels.tif \
		--num-examples $(N_TILES) \
		-o ./train.lmdb


# Creates an LMDB data set for validation.
# This is also a subset of the ISBI 2012 training data set
lmdb-valid:
	$(PY) $(BASE_DIR)/Tools/make_lmdb.py --use-slices $(S_VALID) \
		-X ./ISBI2012/train-volume.tif \
		-Y ./ISBI2012/train-labels.tif \
		--num-examples $(N_TILES) \
		-o ./valid.lmdb


#-------------------------------------------------------------------------------
# Targets for working with Caffe
#-------------------------------------------------------------------------------
caffe-train:
	nohup caffe train -solver lenet-solver.prototxt -gpu 1 > caffe.train.out &


caffe-time-gpu:
	caffe time -model lenet-net.prototxt -weights $(CAFFE_MODEL) -iterations $(NITERS) -gpu 2


caffe-time-cpu:
	nohup caffe time -model lenet-net.prototxt -weights $(CAFFE_MODEL) -iterations $(NITERS) > caffe.time.cpu.out &


#-------------------------------------------------------------------------------
# Targets for working with CcT
#-------------------------------------------------------------------------------
cct-train:
	nohup caffe-ct train lenet-solver.prototxt > cct.train.out &


cct-time-cpu:
	nohup caffe-ct test lenet-solver.prototxt -i $(CCT_MODEL) > cct.time.cpu.out &



clean :
	\rm -f train_preprocessed.bin val_preprocessed.bin
