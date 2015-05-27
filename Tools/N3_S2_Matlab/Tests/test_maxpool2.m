% TEST_MAXPOOL2  Unit tests for maxpool2()
%
% May 2015, mjp

addpath('..');


X = [ 1 2 3 ;
      4 5 6 ;
      7 8 9 ];

Xmp = maxpool2(X,1);
Rv =  [ 5 6 ;
        8 9 ];

assert(all(all(Xmp == Rv)));
