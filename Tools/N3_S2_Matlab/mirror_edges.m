function Xm = mirror_edges(X, nPixels)
% MIRROR_EDGES - Reflects border of nPixels around X.
%
%    The original Python Caffe code expands the original image by
%    reflecting the border (as per Ciresan's original paper).
%    Artifically expanding the image in this fashion makes the 'valid'
%    portion of a convolution the same size as the original image.
%
%    PARAMETERS:
%      X := a data volume (3d tensor)
%      nPixels := the size of the border to add via mirroring
%
%    RETURNS:
%      Xm := a copy of X with mirrored edges
%
% May 2015, mjp

[m,n,p] = size(X);
Xm = NaN*ones(m+2*nPixels, n+2*nPixels, p);

% image interior
Xm(nPixels+1:m+nPixels, nPixels:n+nPixels-1, :) = X;

% Note: don't need to handle corners explicitly (e.g. as was done
% in Python code); careful mirroring of edges will take care of
% this natrually.

for ii = 1:p
    % do left and right edges
    rows = (nPixels+1):size(Xm,1)-nPixels;
    Xm(rows, 1:nPixels, ii) = fliplr(X(:,1:nPixels, ii));
    Xm(rows, end-nPixels:end, ii) = fliplr(X(:,end-nPixels:end, ii));

    % Now, top and bottom edges.
    % Grab these from Xm (vs X) so we get properly reflected corners.
    Xm(1:nPixels, :, ii) = flipud(Xm(nPixels+1:2*nPixels, :, ii));    
    Xm(end-nPixels:end, :, ii) = flipud(Xm(end-2*nPixels:end-nPixels, :, ii)); 
end

% make sure we didn't miss anything in the operations above
assert(all(all(isfinite(Xm))));
