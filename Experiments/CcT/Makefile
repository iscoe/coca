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
# 2. Train models using Caffe:
#      make CNN=lenet GPU=1 caffe-train
#      make CNN=lenet GPU=2 pycaffe-train
#      make CNN=n3 GPU=3 caffe-train
#      make CNN=n3 GPU=4 pycaffe-train
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
# the experimental setup and match the setup of your local system.
#-------------------------------------------------------------------------------

# Specify where pycaffe is located on your system
PYCAFFE=/home/pekalmj1/Apps/caffe/python

# Experiment parameters related to the CNN.
# (primarily which network to use)
CNN=lenet
SOLVER=$(CNN)/$(CNN)-solver.prototxt
NET=$(CNN)/$(CNN)-net.prototxt
DEPLOY=$(CNN)/$(CNN)-deploy.prototxt

# Experiment parameters related to the data set.
# Specify the train/valid/test breakdown.
# Also, when subsampling, how many tiles to use.
EXPERIMENT=ISBI_Train20
S_TRAIN="range(0,20)"
S_VALID="range(20,30)"
S_TEST="[]"
N_TILES=200000

# Specify which model to use in timing experiments.
CAFFE_MODEL=$(CNN)/_iter_15000.caffemodel
CCT_MODEL=trained_model.bin.25-09-2015-04-46-54



#-------------------------------------------------------------------------------
# OTHER MACROS
# These you can probably ignore...
#-------------------------------------------------------------------------------

# This just points to the part
BASE_DIR=../..

# Different ways to run python.
# (we always need PyCaffe and emlib.py in the PYTHONPATH)
PY=PYTHONPATH=$(PYCAFFE):$(BASE_DIR) python
PYNOHUP=PYTHONPATH=$(PYCAFFE):$(BASE_DIR) nohup python
PYPROF=PYTHONPATH=$(PYCAFFE):.. python -m cProfile -s cumtime


# Number of iterations to use in timing experiments
NITERS=100

TAR=CcT-experiment.tar

# You probably want to override this from the command line.
# Note that on our cluster, it is probably a good idea to
# avoid using gpu 0
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
# Data preprocessing
# (includes creating LMDB databases)
# You have to do this (just once) before running any other targets.
#-------------------------------------------------------------------------------

data:
	$(PY) preprocess.py -X ./ISBI2012/train-volume.tif -Y \
		./ISBI2012/train-labels.tif \
		--train-slices $(S_TRAIN) \
		--valid-slices $(S_VALID) \
		--test-slices $(S_TEST) \
		--out-dir $(EXPERIMENT)

	$(PY) $(BASE_DIR)/Tools/make_lmdb.py \
		-X $(EXPERIMENT)/ISBI2012/Xtrain.npy \
		-Y $(EXPERIMENT)/ISBI2012/Ytrain.npy \
		--num-examples $(N_TILES) \
		-o $(EXPERIMENT)/train.lmdb

	$(PY) $(BASE_DIR)/Tools/make_lmdb.py \
		-X $(EXPERIMENT)/ISBI2012/Xvalid.npy \
		-Y $(EXPERIMENT)/ISBI2012/Yvalid.npy \
		--num-examples $(N_TILES) \
		-o $(EXPERIMENT)/valid.lmdb


# Deletes data preprocessing
data-clean:
	\rm -rf ./train.lmdb ./valid.lmdb ./ISBI2012/{X,Y}*npy


#-------------------------------------------------------------------------------
# Working with Caffe
#-------------------------------------------------------------------------------

#--------------------------------------------------
# Train a model using either:
# 
#   a) Command-line caffe and pre-computed tiles
#      stored in an LMDB database
#      
#   b) PyCaffe and dynamically "lazily" created tiles
#      (this is a much larger data set)
#      
#--------------------------------------------------
caffe-train:
	nohup caffe train \
		-solver $(CNN)/$(CNN)-solver.prototxt \
	       	-gpu $(GPU) \
		> $(CNN)/caffe.$(CNN).train.out &


pycaffe-train:
	$(PYNOHUP) emcnn.py \
		--x-train ISBI2012/Xtrain.npy \
		--y-train ISBI2012/Ytrain.npy \
		--x-valid ISBI2012/Xvalid.npy \
		--y-valid ISBI2012/Yvalid.npy \
		--solver $(CNN)/$(CNN)-solver-py.prototxt \
		--gpu $(GPU) \
		> $(CNN)/pycaffe.$(CNN).train.out &


#--------------------------------------------------
# Produce timing estimates (using model created above)
# using caffe command line for either:
# 
#   a) GPU cards
#   b) CPU only (very slow!)
#  
#--------------------------------------------------
caffe-time-gpu:
	caffe time -model $(NET) -weights $(CAFFE_MODEL) -iterations $(NITERS) -gpu $(GPU)


caffe-time-cpu:
	nohup caffe time -model $(NET) -weights $(CAFFE_MODEL) -iterations $(NITERS) > caffe.time.cpu.out &


#--------------------------------------------------
# Generate pixel-level predictions using Caffe.
#
# Note this works with either model developed during training
# (just point the --model argument at the correct file)
#--------------------------------------------------
caffe-predict:
	$(PY) deploy2.py -n $(DEPLOY) -m $(CAFFE_MODEL) -X ISBI2012/train-volume.tif  --gpu $(GPU) --max-brightness $(MAX_BRIGHT) --eval-pct .45



#-------------------------------------------------------------------------------
# Working with CcT
#-------------------------------------------------------------------------------
cct-train:
	nohup caffe-ct train $(SOLVER) > cct.train.out &


cct-time-cpu:
	nohup caffe-ct test $(SOLVER) -i $(CCT_MODEL) > cct.time.cpu.out &


cct-clean :
	\rm -f train_preprocessed.bin val_preprocessed.bin
