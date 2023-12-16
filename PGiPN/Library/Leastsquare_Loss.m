%% This file is to compute the objective value, gradient and Hessian matrix
%% of the loss function: f(x) = ||Ax-b||^2
%% ************************************************************************


function fun = Leastsquare_Loss(A, b)

fun.full = @(x) Call_Leastsquare_Loss(A, b, x);
fun.reduce = @(u, S) Call_Reduced_Leastsquare_Loss(A, b, u, S);
end


function [fval, grad, hess] = Call_Leastsquare_Loss(A, b, x)

[~, n] = size(A);
nnz = sum(abs(x)>0);
if  n > 5000 && nnz/n < 0.01
    J = find(abs(x)>0);
    B = A(:, J);
    u = x(J);
    loss = B * u - b;
    fval = 0.5*norm(loss,2)^2;
    if nargout >= 2
        grad = A' * loss;
        if nargout >= 3
            hess = A'*A;
        end
    end
else
    loss = A*x-b;
    fval = 0.5*norm(loss,2)^2;
    if nargout >= 2
        grad = A' * loss;
        if nargout >= 3
            hess = A'*A;
        end
    end
end
end

function [fval, grad, hess] = Call_Reduced_Leastsquare_Loss(A, b, u, S)

B = A(:, S);
loss = B * u - b;
fval = 0.5*norm(loss,2)^2;
if nargout >= 2
    grad = B' * loss;
    if nargout >= 3
        hess = B'*B;
    end
end
end