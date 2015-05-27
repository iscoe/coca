% TEST_CONV_LAYER  Unit tests for conv_layer()
%
% May 2015, mjp

addpath('..');


X = [ 1 2 3 ;
      4 5 6 ;
      7 8 9 ];

F = zeros(2, 2, 1, 2);
F(:,:,1,1) = [1 1 ; 
              1 1 ];
F(:,:,1,2) = [2 2 ; 
              2 2 ];

bias = [1 1];

Xout = conv_layer(X,F,bias);
Rv1 =  bias(1) + [ 12 16 ;
                   24 28 ];
Rv2 =  bias(2) + 2* [ 12 16 ;
                      24 28 ];

assert(all(all(Xout(:,:,1) == Rv1)));
assert(all(all(Xout(:,:,2) == Rv2)));
