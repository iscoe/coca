# ipython -i -- view_cube.py

import numpy as np
import pylab as plt


inFile='./lenet/Yhat_train-volume.tif.npy'

Yhat = np.load(inFile)

plt.imshow(Yhat[0,0,:,:], cmap='bone')
plt.colorbar()
plt.show()
