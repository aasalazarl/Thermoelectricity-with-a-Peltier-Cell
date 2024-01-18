function [c,norm_r] = linefiterr(x,y,s)
% linefiterr    Least-squares fit of data to y = c(1)*x + c(2)
%               Here we also take into account data errors, 
%                (assumed to be uncorrelated)
%
% Synopsis:   c     = linefiterr(x,y,s)
%            [c,norm_r] = linefiterr(x,y,s)
%
% Input:   x,y = vectors of independent and dependent variables
%          s = vector of errors (on y)
%
% Output:  c  = vector of slope, c(1), and intercept, c(2) of least sq. line fit
%          norm_r = (optional) norm of the residual vector

if length(y)~= length(x),  error('x and y are not compatible');  end
if length(s)~= length(y),  error('s and y are not compatible');  end

x = x(:);  y = y(:);    %  Make sure that x and y are column vectors
A = [x ones(size(x))];  %  m-by-n matrix of overdetermined system
W = diag(1./s);         % Weighting matrix
WTW = W'*W;             % 
c = (A'*WTW*A)\(A'*WTW*y);      %  Solve equations
%c=(W*A)\(W*y);
if nargout>1
  r = W*y - W*A*c;
  norm_r = norm(r);
end
