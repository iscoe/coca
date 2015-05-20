function Xm = mirror_edges(X, nPixels)
%  MIRROR_EDGES

flip_corner = @(M) fliplr(flipud(M));
    
[m,n,p] = size(X);
Xm = zeros(m+2*nPixels, n+2*nPixels, p);

Xm(nPixels:m+nPixels-1, nPixels:n+nPixels-1, :) = X;

% TODO: don't need to do corners explicitly - will fall out properly if mirror borders.

for ii = 1:p
    Xi = X(:,:,ii);
    
    % top left corner
    Xm(1:nPixels, 1:nPixels, ii) = flip_corner(Xi(1:nPixels, 1:nPixels));

    % top right corner
    Xm(1:nPixels, n+nPixels+1:end, ii) = flip_corner(Xi(1:nPixels, n-nPixels+1:end));

    % bottom left corner
    Xm(m+nPixels+1:end, 1:nPixels) = flip_corner(Xi(m-nPixels+1:end, 1:nPixels));
    
    % bottom right corner
    Xm(m+nPixels+1:end, n+nPixels+1:end, ii) = flip_corner(Xi(m-nPixels+1:end, n-nPixels+1:end));
    
    % top border
    Xm(1:nPixels, nPixels+1:n+nPixels, ii) = flipud(Xi(1:nPixels, :));

    % bottom border
    Xm(m+nPixels+1:end, nPixels+1:n+nPixels, ii) = flipud(Xi(m-nPixels+1:end, :));

    % left border
    Xm(nPixels+1:m+nPixels, 1:nPixels, ii) = fliplr(Xi(:, 1:nPixels));

    % right border
    Xm(nPixels+1:m+nPixels, n+nPixels+1:end, ii) = fliplr(Xi(:, n-nPixels+1:end));
end


