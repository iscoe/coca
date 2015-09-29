# ipython -i -- view_cube.py

import numpy as np
import pylab as plt


inFile='./lenet/Yhat_train-volume.tif.npy'

Yhat = np.load(inFile)

sliceId = 1;

plt.imshow(Yhat[1,sliceId,:,:], cmap='bone')
plt.colorbar()
plt.show()
