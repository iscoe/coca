function Xout = conv_layer(X, F, bias)
% CONV_LAYER  Simulates a single CNN convolutional layer
%
%  PARAMETERS:
%   X := either a (width x height) data matrix 
%                       -OR-
%        a (width x height x nInputChannels) data tensor
%
%   F := Convolution kernel.  A 4d tensor with dimensions
%        (width x height x nInputChannels x nOutputChannels)
%        Presumably extracted from a Caffe model file.
%
%   bias := a (nOutChannels x 1) vector
%
%  RETURNS:
%   A tensor with dimensions (width x height x nOutputChannels)
%
% May 2015, mjp


% The difference between convolution and correlation is a rotation
% by 180 degrees.  Hence the flipud(fliplr(.))
%
% Update: using anonymous functions was causing a performance hit, so
% inline this instead.
%
% conv_op = @(A, B) conv2(A, flipud(fliplr(B)), 'valid'); 


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
wOut = size(X,1) - w + 1;
hOut = size(X,2) - h + 1;
Xout = zeros(wOut, hOut, nOutChan);

for ii = 1:nOutChan
    if length(size(X)) == 2
        % Special case where input has a single filter channel.
        %
        Fi = F(:,:,1,ii); 
        Xout(:,:,ii) = conv2(X, flipud(fliplr(Fi)), 'valid');
    else
        % More general case where X has > 1 input channel.
        %
        for jj = 1:nInChan
            Fj = F(:,:,jj,ii);
            Xout(:,:,ii) = Xout(:,:,ii) + conv2(X(:,:,jj), flipud(fliplr(Fj)), 'valid');
        end
    end
 
    Xout(:,:,ii) = Xout(:,:,ii) + bias(ii);
end
