% May 2015, mjp

%-------------------------------------------------------------------------------
% Load data
%-------------------------------------------------------------------------------

fprintf('[%s]: loading data and filters...\n', mfilename);
load ~/Data/SynapseData3/X_test.mat

% Due to memory limitations, only process a single slice.
whichSlice = 1; 
X = single(X_test(:,:,whichSlice));
clear X_test;

% mirror edges
% note: hardcoded tile size...
X = mirror_edges(X, 32);


%-------------------------------------------------------------------------------
% Load filters
%-------------------------------------------------------------------------------

% Helper functions for dealing with convolutional filter weights.
%
%  (#output filt, #input filt, height, width) -> (height, width, #input filters, #output filters)
reshape_conv_w = @(W) permute(W, [3 4 2 1]); 

% 4d tensor -> 1d vector (one scalar per filter)
reshape_b = @(B) squeeze(permute(B, [4 1 2 3]));


%--------------------------------------------------
% Layer 1 - CNN
%--------------------------------------------------
load conv1Weight
load conv1Bias

conv1Weight = reshape_conv_w(conv1Weight);
conv1Bias = reshape_b(conv1Bias);

%--------------------------------------------------
% Layer 2 - CNN
%--------------------------------------------------
load conv2Weight
load conv2Bias

conv2Weight = reshape_conv_w(conv2Weight);
conv2Bias = reshape_b(conv2Bias);

%--------------------------------------------------
% Layer 3 - CNN
%--------------------------------------------------
load conv3Weight
load conv3Bias

conv3Weight = reshape_conv_w(conv3Weight);
conv3Bias = reshape_b(conv3Bias);

%--------------------------------------------------
% Layer 4 - IP
%--------------------------------------------------
load ip1Weight
load ip1Bias
load ip2Weight
load ip2Bias

% -> (width, height, nInChannels, nOutChannels)
%    (5, 5, 48, 200)
%  Note I'm hardcoding the assumption that 
%  the tile size is 5x5 at the outlet of
%  pool3.
%
ip1Weight = reshape(ip1Weight, [200 5 5 48]);
ip1Weight = permute(ip1Weight, [3 2 4 1]);
ip1Bias = squeeze(ip1Bias);


%--------------------------------------------------
% Layer 5 - IP
%--------------------------------------------------
ip2Weight = reshape(ip2Weight, [2 200]);
ip2Weight = ip2Weight(2,:);  % 2 := probability of target class
ip2Bias = squeeze(ip2Bias);
ip2Bias = ip2Bias(2);        % 2 := probability of target class



%-------------------------------------------------------------------------------
% Apply filters
%-------------------------------------------------------------------------------

tic

%--------------------------------------------------
fprintf('[%s]: Applying CNN layer 1\n', mfilename);
%--------------------------------------------------
X1 = conv_layer(X, conv1Weight, conv1Bias);
X1 = max(X1, 0);
X1 = maxpool2(X1);

%--------------------------------------------------
fprintf('[%s]: Applying CNN layer 2\n', mfilename);
%--------------------------------------------------
X2 = conv_layer(X1, conv2Weight, conv2Bias);
X2 = max(X2, 0);
X2 = maxpool2(X2);

%--------------------------------------------------
fprintf('[%s]: Applying CNN layer 3\n', mfilename);
%--------------------------------------------------
X3 = conv_layer(X2, conv3Weight, conv3Bias);
X3 = max(X3, 0);
X3 = maxpool2(X3);


%--------------------------------------------------
fprintf('[%s]: Applying IP layer 1\n', mfilename);
%--------------------------------------------------
% MJP: Think of the IP layer as a special-case convolutional layer where
%      the kernel size is the same as the image size.
%
% The first inner product layer is a bit tricky.  In Caffe, this would
% correspond to vectorizing a single tile (which is 5 x 5 x 48) and
% then multiplying by a 1200x1 vector N times, where N is the other
% dimension of ip1.
X4 = conv_layer(X3, ip1Weight, ip1Bias);

X4 = max(X4,0);  % evidently I left a RELU after IP1



%--------------------------------------------------
fprintf('[%s]: Applying IP layer 2\n', mfilename);
%--------------------------------------------------
assert(isvector(ip2Weight));
X5 = X4(:,:,1) * ip2Weight(1);
for ii = 2:size(X4,3)
    X5 = X5 + X4(:,:,ii) * ip2Weight(ii);
end
X5 = X5 + ip2Bias;

% softmax
% MJP: this isn't quite right, unfortunately
%X5 = softmax(X5);

toc


