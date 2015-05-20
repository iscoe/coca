function Xout = conv_layer(X, F, bias)
% CONV_LAYER  Simulates a single CNN convolutional layer
%
%   X := either a (width x height) data matrix OR
%        a (width x height x nInputChannels) data tensor
%
%   F := a (width x height x nInputChannels x nOutputChannels) tensor
%        extracted from a Caffe model file.
%
%   bias := a (nOutChannels x 1) vector
%
% May 2015, mjp

convType = 'valid';
op = @(A, B) conv2(A, flipud(fliplr(B)), convType);  % really we want 2d correlation...


% make sure the filter and bias are the right size
assert(length(size(F)) == 4);
[w h nInChan nOutChan] = size(F);
assert(length(bias) == nOutChan);
assert(isvector(bias));

assert(w == h);  % assume square filters for now


% X and F must agree on the # of input channels
if length(size(X)) == 2
    assert(nInChan == size(X,3));
end


%----------------------------------------
% Run the calculation
%----------------------------------------
if strcmp(convType, 'valid')
    Xout = zeros(size(X,1) - 2*floor(w/2), size(X,2) - 2*floor(w/2), nOutChan);
elseif strcmp(convType, 'same')
    Xout = zeros(size(X,1), size(X,2), nOutChan);
else
    error('unexpected conv type');
end


for ii = 1:nOutChan
    if length(size(X)) == 2
        % Special case where input has a single filter channel.
        %
        Fi = F(:,:,1,ii); 
        Xout(:,:,ii) = op(X, Fi);
    else
        % More general case where X has > 1 input channel.
        %
        for jj = 1:nInChan
            Fj = F(:,:,jj,ii);
            Xout(:,:,ii) = Xout(:,:,ii) + op(X(:,:,jj), Fj);
        end
    end
 
    Xout(:,:,ii) = Xout(:,:,ii) + bias(ii);
end
