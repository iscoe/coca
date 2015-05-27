function Xout = maxpool2(X, stride)
% MAXPOOL2  Max pool operator with a kernel size of 2.
%
%   Xout = maxpool(X, stride=2)
%
%   Note: It should be possible to implement this using blocproc, e.g., 
%
%        Xout(:,:,ii) = blockproc(X(:,:,ii), [2 2], @(bs) max(bs.data(:)));
%
%   however, in the special case of kernel=2, the explict slicing
%   operations below seem to be faster.
%
%  May 2015, mjp

if nargin < 2, stride=2; end

assert(length(size(X)) <= 3);
assert(stride >= 1);

[m,n,p] = size(X);    % if X is a matrix, p will be 1 which is fine

mm = length(1:stride:m-1);
nn = length(1:stride:n-1);
Xout = zeros(mm, nn, p);

Z = zeros(m-1, n-1, 4);

for ii = 1:p
    % The max in a 2x2 cell can be calculated by stacking the four pixels
    % in question up along the z axis and then taking the max along z.
    Z(:,:,:) = 0;
    Z(:,:,1) = X(1:end-1, 1:end-1, ii);    % upper left
    Z(:,:,2) = X(1:end-1, 2:end, ii);      % upper right
    Z(:,:,3) = X(2:end, 1:end-1, ii);      % lower left
    Z(:,:,4) = X(2:end, 2:end, ii);        % lower right
    Tmp = max(Z, [], 3); 
   
    % implement stride 
    if stride > 1
        Xout(:,:,ii) = Tmp(1:stride:end, 1:stride:end);
    else
        Xout(:,:,ii) = Tmp;
    end
end
