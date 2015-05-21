% APPLY  Applies a CNN to a data volume.
%
%   As of this writing, this script is limited to processing a single
%   slice from the data volume (due to memory limitations).  
%     TODO: add a for loop to process all slices.
%     TODO: map subsampled points to appropriate place in overall volume.
%
%   Note: there are a lot of hardcoded assumptions in this script; as
%   such, it is only applicable to reproducing the results of a
%   specific CNN.
%
% May 2015, mjp

Param.dataFile = '~/Data/SynapseData3/X_test.mat';   % the data volume to process
Param.whichSlice = 1;                                % the slice to process
Param.weightDir = 'Example';                         % directory containing CNN weights
Param.tileSize = 32;

Param

%-------------------------------------------------------------------------------
%% Load data
%-------------------------------------------------------------------------------

fprintf('[%s]: loading data...\n', mfilename);
load(Param.dataFile);

X = single(X_test(:,:,Param.whichSlice));
clear X_test;

% mirror edges
% note: hardcoded tile size...
X = mirror_edges(X, Param.tileSize);


%-------------------------------------------------------------------------------
%% Load CNN weights
% Note this is hard-coding a specific network architecture.
%-------------------------------------------------------------------------------

fprintf('[%s]: loading weights...\n', mfilename);

% Helper functions for dealing with convolutional filter weights.
%
%  (#output filt, #input filt, height, width) -> (height, width, #input filters, #output filters)
reshape_conv_w = @(W) permute(W, [3 4 2 1]); 

% 4d tensor -> 1d vector (one scalar per filter)
reshape_b = @(B) squeeze(permute(B, [4 1 2 3]));


%--------------------------------------------------
% Layer 1 - convolution
%--------------------------------------------------
load(fullfile(Param.weightDir, 'conv1Weight'));
load(fullfile(Param.weightDir, 'conv1Bias'));

conv1Weight = reshape_conv_w(conv1Weight);
conv1Bias = reshape_b(conv1Bias);

%--------------------------------------------------
% Layer 2 - convolution
%--------------------------------------------------
load(fullfile(Param.weightDir, 'conv2Weight'));
load(fullfile(Param.weightDir, 'conv2Bias'));

conv2Weight = reshape_conv_w(conv2Weight);
conv2Bias = reshape_b(conv2Bias);

%--------------------------------------------------
% Layer 3 - convolution
%--------------------------------------------------
load(fullfile(Param.weightDir, 'conv3Weight'));
load(fullfile(Param.weightDir, 'conv3Bias'));

conv3Weight = reshape_conv_w(conv3Weight);
conv3Bias = reshape_b(conv3Bias);

%--------------------------------------------------
% Layer 4 - inner product
%--------------------------------------------------
load(fullfile(Param.weightDir, 'ip1Weight'));
load(fullfile(Param.weightDir, 'ip1Bias'));

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
% Layer 5 - inner product
%--------------------------------------------------
load(fullfile(Param.weightDir, 'ip2Weight'));
load(fullfile(Param.weightDir, 'ip2Bias'));

ip2Weight = reshape(ip2Weight, [2 200]);
ip2Weight = permute(ip2Weight, [2 1]);
assert(size(ip2Weight,2) == 2);

ip2Bias = squeeze(ip2Bias);



%===============================================================================
% Apply CNN
%===============================================================================

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

X4 = max(X4,0);  % evidently I had a RELU after IP1


%--------------------------------------------------
fprintf('[%s]: Applying IP layer 2\n', mfilename);
%--------------------------------------------------
% TODO: generalize this?
X5neg = X4(:,:,1) * ip2Weight(1,1);
X5pos = X4(:,:,1) * ip2Weight(1,2);
for ii = 2:size(X4,3)
    X5neg = X5neg + X4(:,:,ii) * ip2Weight(ii,1);
    X5pos = X5pos + X4(:,:,ii) * ip2Weight(ii,2);
end
X5neg = X5neg + ip2Bias(1);
X5pos = X5pos + ip2Bias(2);

% softmax
% TODO: put this in it's own function
mv = max([ X5neg(:) ; X5pos(:) ]);  % subtract max val for numerical stability.
                                    % (ref: Caffe's softmax_layer.cpp )
Ypos = exp(X5pos - mv) ./ ( exp(X5pos - mv) + exp(X5neg - mv) );

toc


fprintf('[%s]: done processing slice %d\n', mfilename, Param.whichSlice);
figure; imagesc(Ypos); colorbar; 
title('Probability of Synapse (1/8 resolution)');
