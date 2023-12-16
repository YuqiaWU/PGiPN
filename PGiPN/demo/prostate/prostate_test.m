prostate_data = load('Prostate_cancer.txt');

full_A = prostate_data(:, 2:9);
full_b = prostate_data(:, 10);

train_num = 50;
index_train = randperm(size(full_A, 1), train_num);
index_test = setdiff(1:size(full_A, 1), index_train);
A = full_A(index_train, :);
b = full_b(index_train);
[m, n] = size(A);

AT = A';
Amap  = @(x) A*x;
ATmap = @(x) AT*x;
AATmap = @(x) Amap(ATmap(x));
eigsopt.issym = 1;
eigsopt.tol = 0.001;
Lip = eigs(AATmap,length(b),1,'LM',eigsopt);
B= sparse(n-1, n);
for j = 1:n-1
    B(j, j:j+1) = [1, -1];
end
lambdamax = max(abs(A'*b));
lambda = 1.0e-4 * lambdamax;
%% Algorithm
lb = -1000; ub = 1000;
prob.A = A;
prob.B = B;
prob.b = b;
prob.reg = FusedL0_box(lambda, 0.1*lambda, lb, ub);
prob.m = m;
prob.n = n;
prob.lambda1 = lambda;
prob.lambda2 = 0.3*lambda;
prob.lb = lb;
prob.ub = ub;
prob.lfun = Leastsquare_Loss(A, b);

options.x0 = zeros(n, 1);
options.tol = 1.0e-4;
options.iter_print =1;
options.result_print = 1;
options.lambdamax = lambdamax;
options.maxiter = 5000;
options.Ini_step = 1/0.95*Lip;
options.BBstep = 1;

options.r = 4;

[out1] =  PGiPN_main(prob,options);
[out2] =  PGiPN_relaxed(prob,options);



