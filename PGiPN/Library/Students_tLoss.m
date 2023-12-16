%% This file is to compute the objective value, gradient and Hessian matrix
%% of the loss function: f(x) = Sum log[(1+ (Ax-b)_i)/nu]
%% ************************************************************************

function fun = Students_tLoss(A, b, nu)
fun.full = @(x) Call_Students_tLoss(A, b, nu, x);
fun.reduce = @(u, S) Call_Reduced_Students_tLoss(A, b, nu, u, S);
end

function [fval, grad, hess] = Call_Students_tLoss(A, b, nu, x)

[~, n] = size(A);
nnz = sum(abs(x)>0);
if  n > 5000 && nnz/n < 0.01
    J = find(abs(x)>0);
    B = A(:, J);
    u = x(J);
    loss = B * u - b;
    fval = sum(log(1+(loss.^2)/nu));
    if nargout >= 2
        loss_sq = loss.^2;
        t = loss./(nu+loss_sq);
        grad = 2* A' * t;
        if nargout >= 3
            dd = 2*(nu-loss_sq)./(nu+loss_sq).^2;
            hess = A'*diag(max(dd, 0))*A;
        end
    end
else
    loss = A*x-b;
    fval = sum(log(1+(loss.^2)/nu));
    if nargout >= 2
        loss_sq = loss.^2;
        t = loss./(nu+loss_sq);
        grad = 2* A' * t;
        if nargout >= 3
            dd = 2*(nu-loss_sq)./(nu+loss_sq).^2;
            hess = A'*diag(max(dd, 0))*A;
        end
    end
end
end

function [fval, grad, hess] = Call_Reduced_Students_tLoss(A, b, nu, u, S)

B = A(:, S);
loss = B * u - b;
fval = sum(log(1+(loss.^2)/nu));
    if nargout >= 2
        loss_sq = loss.^2;
        t = loss./(nu+loss_sq);
        grad = 2* B' * t;
        if nargout >= 3
            dd = 2*(nu-loss_sq)./(nu+loss_sq).^2;
            hess = B'*diag(max(dd, 0))*B;
        end
    end
end