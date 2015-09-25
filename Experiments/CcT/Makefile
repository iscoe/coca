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

# Specify which Caffe model to use in all timing experiments.
CMODEL=_iter_6000.caffemodel

# Number of iterations to use in timing experiments
NITERS=30


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
# Train a classifier
#-------------------------------------------------------------------------------
caffe-train:
	caffe train -solver lenet-solver.prototxt -gpu 1


cct-train:
	caffe-ct train n3-solver.prototxt


#-------------------------------------------------------------------------------
# Timing estimate for caffe
#-------------------------------------------------------------------------------
caffe-time-gpu:
	caffe time -model lenet-net-caffe.prototxt -weights $(CMODEL) -iterations $(NITERS) -gpu 1


caffe-time-cpu:
	caffe time -model n3-net-caffe.prototxt -weights $(CMODEL) -iterations $(NITERS)


#-------------------------------------------------------------------------------
# Timing estimate for CcT
#-------------------------------------------------------------------------------
time-cct-cpu:
	caffe-ct train n3-solver-cct.prototxt


time-cct-gpu:
	echo "TODO"


clean :
	\rm -f train_preprocessed.bin val_preprocessed.bin
