#-------------------------------------------------------------------------------
# This makefile sets up a few classification problems related to the 
# ISBI 2012 challenge data set.  The goal is to make it relatively easy
# to run timing experiments with Caffe and Caffe con Troll (CcT) on this
# data.  Also, we provide ways of extracting probability maps.
# 
# 1. To create the LMDB database from raw ISBI data:
#      make lmdb-train
#      make lmdb-valid
#
# 2. Train a lenet model and (on a different GPU) a n3 model:
#      make CNN=lenet GPU=1 caffe-train
#      make CNN=n3 GPU=2 caffe-train
# 
# 3. To generate timing estimates for Caffe:
#      make caffe-time-gpu
#      make caffe-time-cpu
# 
# 4. To generate timing estimates for Caffe con Troll (CcT):
#      make cct-time-cpu
#
# 5. Extract predictions from the Caffe model:
#      make CNN=lenet GPU=3 caffe-predict
#
# 6. Extract predictions for CcT:
#      TODO: this is going to require some new code.
#
# NOTES:
# o For CcT compatability, we create LMDB databases that contain many 
#   pre-computed tiles; in the past we created these tiles "lazily" to
#   avoid creating large data sets with lots of redundancy.
#
# o A future possible direction is some combination of CcT and 
#   dense classification techniques (e.g. the semantic segmentation
#   approach of Long et. al.)
#
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# PARAMETERS
#
# The parameters in this section you may want to change in order to alter
# the experimental setup or match the setup of your local system.
#-------------------------------------------------------------------------------

# Specify where pycaffe is located on your system
PYCAFFE=/home/pekalmj1/Apps/caffe/python


# Specify which CNN model to use.
# For now, the main options are 'lenet' or 'n3'
CNN=lenet
SOLVER=$(CNN)/$(CNN)-solver.prototxt
NET=$(CNN)/$(CNN)-net.prototxt
DEPLOY=$(CNN)/$(CNN)-deploy.prototxt


# Any pixels whose intensity is > this value will be omitted
# from the analysis (these are trivially non-membrane).  Set
# to 256 if you want to omit nothing.
MAX_BRIGHT=142


# Define the subset of ISBI2012 to use for train and validation.
# For now, we'll use 20 slices for train and the last 10 for validation.
S_TRAIN="range(0,10)"
S_VALID="range(10,20)"


# Specify which model to use in timing experiments.
CAFFE_MODEL=$(CNN)/_iter_4000.caffemodel  # TODO: increase this value!!
CCT_MODEL=trained_model.bin.25-09-2015-04-46-54



#-------------------------------------------------------------------------------
# OTHER MACROS
# These you can probably ignore...
#-------------------------------------------------------------------------------

# This just points to the part
BASE_DIR=../..

# we need PyCaffe and emlib.py in the PYTHONPATH
PY=PYTHONPATH=$(PYCAFFE):$(BASE_DIR) python
# for profiling (optional)
#PY=PYTHONPATH=$(PYCAFFE):.. python -m cProfile -s cumtime

# The maximum number of tiles to extract
N_TILES=100000


# Number of iterations to use in timing experiments
NITERS=100

TAR=CcT-experiment.tar

GPU=1


#-------------------------------------------------------------------------------
# Administrative targets
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
# Creating LMDB databases
#-------------------------------------------------------------------------------

# Creates an LMDB data set for training.
# This is just a ("tile-ified") subset of the ISBI 2012 training data set.
lmdb-train:
	$(PY) $(BASE_DIR)/Tools/make_lmdb.py --use-slices $(S_TRAIN) \
		-X ./ISBI2012/train-volume.tif \
		-Y ./ISBI2012/train-labels.tif \
		--num-examples $(N_TILES) \
		--max-brightness $(MAX_BRIGHT) \
		-o ./train.lmdb


# Creates an LMDB data set for validation.
# This is also a subset of the ISBI 2012 training data set
lmdb-valid:
	$(PY) $(BASE_DIR)/Tools/make_lmdb.py --use-slices $(S_VALID) \
		-X ./ISBI2012/train-volume.tif \
		-Y ./ISBI2012/train-labels.tif \
		--num-examples $(N_TILES) \
		--max-brightness $(MAX_BRIGHT) \
		-o ./valid.lmdb


# Deletes LMDB directories
lmdb-clean:
		\rm -rf ./train.lmdb ./valid.lmdb


#-------------------------------------------------------------------------------
# Working with Caffe
#-------------------------------------------------------------------------------

# Train a model
caffe-train:
	nohup caffe train -solver $(SOLVER) -gpu $(GPU) > caffe.$(CNN).train.out &


# Produce timing estimates; for GPU mode
caffe-time-gpu:
	caffe time -model $(NET) -weights $(CAFFE_MODEL) -iterations $(NITERS) -gpu $(GPU)


# Produce timing estimates; for CPU mode
caffe-time-cpu:
	nohup caffe time -model $(NET) -weights $(CAFFE_MODEL) -iterations $(NITERS) > caffe.time.cpu.out &


# Use this to extract probability maps.
# For now, we use PyCaffe for this (already in hand, and time is short)
caffe-predict:
	$(PY) deploy2.py -n $(DEPLOY) -m $(CAFFE_MODEL) -X ISBI2012/train-volume.tif --eval-slices $(S_VALID) --gpu $(GPU) --max-brightness $(MAX_BRIGHT) --eval-pct .45



#-------------------------------------------------------------------------------
# Working with CcT
#-------------------------------------------------------------------------------
cct-train:
	nohup caffe-ct train $(SOLVER) > cct.train.out &


cct-time-cpu:
	nohup caffe-ct test $(SOLVER) -i $(CCT_MODEL) > cct.time.cpu.out &


cct-clean :
	\rm -f train_preprocessed.bin val_preprocessed.bin
