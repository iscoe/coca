"""
Unit tests for code in 'emutils.py'.

April 2015, mjp
"""

import unittest
import numpy
import pdb

# make sure parent directory is in pythonpath before running...
from emlib import *



class TestUtils(unittest.TestCase):
    
    def test_mirror_edges(self):
        X = numpy.random.rand(10,3,3);
        b = 2  # b := border size
        Xm = mirror_edges(X,b)

        # make sure the result has the proper size
        assert(Xm.shape[0] == X.shape[0]);
        assert(Xm.shape[1] == X.shape[1]+2*b);
        assert(Xm.shape[2] == X.shape[2]+2*b);

        # make sure the data looks reasonable
        self.assertTrue(numpy.all(Xm[:,:,b-1] == Xm[:,:,b]))
        self.assertTrue(numpy.all(Xm[:, b:-b, b:-b] == X))

        
    def test_interior_pixel_generator(self):
        b = 10  # b := border size
        Z = numpy.zeros((2,100,100), dtype=np.int32)
        for idx, pct  in interior_pixel_generator(Z,b,30):
            Z[idx[:,0],idx[:,1],idx[:,2]] += 1

        self.assertTrue(numpy.all(Z[:,b:-b,b:-b]==1))
        Z[:,b:-b,b:-b] = 0
        self.assertTrue(numpy.all(Z==0))

        
    def test_stratified_interior_pixel_generator(self):
        b = 10  # b := border size

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For a 50/50 split of pixels in the interior, the generator
        # should reproduce the entire interior.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Y = numpy.zeros((2,100,100))
        Y[:,0:50,:] = 1

        Z = numpy.zeros(Y.shape)
        for idx,pct in stratified_interior_pixel_generator(Y,b,30):
            Z[idx[:,0],idx[:,1],idx[:,2]] += 1

        self.assertTrue(numpy.all(Z[:,b:-b,b:-b]==1))
        Z[:,b:-b,b:-b] = 0
        self.assertTrue(numpy.all(Z==0))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For a random input, should see a 50/50 split of class
        # labels, but not necessarily hit the entire interior.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Y = numpy.random.rand(2,100,100) > 0.5
        nOne=0; nZero=0;
        for idx,pct in stratified_interior_pixel_generator(Y,b,30):
            slices = idx[:,0];  rows = idx[:,1];  cols = idx[:,2]
            nOne += numpy.sum(Y[slices,rows,cols] == 1)  
            nZero += numpy.sum(Y[slices,rows,cols] == 0)
        self.assertTrue(nOne == nZero) 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For an input tensor with "no-ops", the sampler should only
        # return pixels with a positive or negative label.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Y = numpy.zeros((2,100,100))
        Y[:,0:20,0:20] = 1      
        Y[:,50:70,50:70] = -1
        Z = numpy.zeros(Y.shape)
        nPos=0; nNeg=0; nTotal=0;
        for idx,pct in stratified_interior_pixel_generator(Y,0,10,omitLabels=[0]):
            slices = idx[:,0];  rows = idx[:,1];  cols = idx[:,2]
            Z[slices,rows,cols] = Y[slices,rows,cols]
            nPos += numpy.sum(Y[slices,rows,cols] == 1)
            nNeg += numpy.sum(Y[slices,rows,cols] == -1)
            nTotal += len(slices)
            
        self.assertTrue(nPos == 20*20*2);
        self.assertTrue(nNeg == 20*20*2);
        self.assertTrue(nTotal == 20*20*2*2);
        self.assertTrue(np.all(Y == Z))


if __name__ == "__main__":
    unittest.main()
