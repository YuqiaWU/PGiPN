This Matlab package solves the fused L0-norms regularization problems:

            min_{x\in\mathbb{R}^n} f(x) + lam1 * ||Bx||_0 + lam2 * ||x||_0,

where f is twice continuously differentiable, lam1>=0, lam2>=0, || ||_0 is the cardinality function,
      
      B\in R^{(n-1)*n} with B_{i,i} = 1, B{i,i+1} = -1 for all i = 1,...,n-1, and B_{i,j} =0 otherwise.


To use this package, you need first run startup.m to install the software, and you may install Gurobi.

Example on how to use 'PGiPN' can be found in [demo folder].