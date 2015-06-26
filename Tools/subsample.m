function M = subsample(X, ds, method)
% SUBSAMPLE Creates a masking matrix that corresponds to
% downsampling the tensor X by some fixed factor in the XY plane.
%
%   subsample(X, ds, method)
%
%   where:
%   X : A a tensor with shape (x, y, z/slices)
%   ds : a scalar downsampling factor
%   method : Use 'xy' to downsample only in the xy plane (i.e. by slice); use
%            'sobol' to do randomized sampling across all dimensions.
%
%  Presumably the values that are not processed by the CNN can be filled in
%  by interpolation; e.g. see inpaint_nans() on the Matlab file
%  exchange.
%
%  Example:
%    M = subsample(Yhat, 5, 'sobol');
%    YhatSub = M .* Yhat;
%    YhatSub(~M) = NaN;
%
%    Z1 = inpaint_nans(YhatSub(:,:,1));
%    Z1 = min(1, Z1);
%    Z1 = max(0, Z1);
%
%  May 2015, mjp

if nargin < 3, method='xy'; end
assert(isscalar(ds) && ds > 1);
ds = floor(ds);

assert(length(size(X)) == 3);  % assuming a tensor

[x,y,z] = size(X);
M = logical(zeros(x,y,z));


switch lower(method)
  case 'xy'
    M(1:ds:x, 1:ds:y, :) = 1;
      
  case 'sobol'
    nSamp = floor(x*y*z/ds);
    p = sobolset(3);
    sub = net(p, nSamp);
    sub = 1 + floor(bsxfun(@times, sub, [x y z]));  % map indices from [0 1] to integer indices
    idx = sub2ind(size(M), sub(:,1), sub(:,2), sub(:,3));
    M(idx) = 1;
    
  otherwise
    error('unrecognized method')
end
