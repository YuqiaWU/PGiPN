%% This file is to minimize
%% min f(x) + lambda_1 * ||Bx||_0 + lambda2 * ||x||_0, s.t. lb<=x<=ub
%% where f is a twice continuously differentiable function
%% B is a matrix with dimension (n-1)*n with B_{ii} = 1, B_{i,i+1} = -1, and B_{ij} = 0 otherwise.
%% by using PGiPN, a hybrid of Proximal Gradient and inexact Project regularized Newton method
%% ***************************************************************************
function [out] = PGiPN_main(prob,options)

main_tstart = clock; 
if isfield(prob,'B');     B     = prob.B;                                           end
if isfield(prob,'reg');   reg   = prob.reg;                                         end
if isfield(prob,'m');     m     = prob.m;                                           end
if isfield(prob,'n');     n     = prob.n;                                           end
if isfield(prob,'lb');    lb    = prob.lb;                                          end
if isfield(prob,'ub');    ub    = prob.ub;                                          end
if isfield(prob,'lfun');  lfun  = prob.lfun;                                        end

if isfield(options,'maxiter');         maxiter        = options.maxiter;            end
if isfield(options,'iter_print');      iter_print     = options.iter_print;         end
if isfield(options,'result_print');    result_print   = options.result_print;       end
if isfield(options,'tol');             tol            = options.tol;                end
if isfield(options,'r');               r              = options.r;                  end
if isfield(options,'Ini_step'); Ini_step= options.Ini_step; else; Ini_step =1;      end
if isfield(options,'BBstep'); BBstep = options.BBstep ; else; BBstep = 0;           end
if isfield(options,'x0'); x0 = options.x0;     else; x0 = zeros(size(prob.A,2), 1); end


if (iter_print)
    fprintf('\n *****************************************************');
    fprintf('******************************************');
    fprintf('\n ************* PGiPN **************************************');
    fprintf('*******************************************');
end
%% *********************************************
%% **************** Preperation ****************

tau = 2;

sec_time = 0;  % Total time used for second-order step
sec_iter = 0;  % Number of iterations entering second-order part
x = x0;
times_for_same_supp = 0;
[fobjold,grad_fxnew] = lfun.full(x);
gobjold = reg.val(x);
obj_old = fobjold + gobjold;
old_search = 1;
%% ************************ Main loop *******************************
for iter = 1:maxiter

    %% *************** first order step ***********************************
    grad_fx = grad_fxnew;
    search_num1 = 0;
    if BBstep
        if iter == 1
            mu = Ini_step;
        else
            tk = (sk'*rk)/(sk'*sk);
            mu = min(max(tk,1.0e-20),1.0e20);
        end
    else
        mu = Ini_step;
    end
    while 1
        search_num1 = search_num1+1;
        gk = x -(1/mu)*grad_fx;
        xbar = reg.prox(gk, mu);
        lossobj = lfun.full(xbar);
        obj_new = lossobj + reg.val(xbar);
        normdiff = norm(x-xbar);
        if obj_new <= obj_old - 1.0e-8 * normdiff^2
            xnew = xbar;
            break
        else
            mu = tau * mu;
        end
    end
    if search_num1 > 2 || old_search > 2
        tau = 10;
    else
        tau = 2;
    end
    old_search = search_num1;
    
    [T_new, Tsignabs] = reg.Bxsupp(xnew); T_nnz = sum(abs(T_new)>0); 
    [S_new, Ssignabs] = reg.xsupp(xnew); S_nnz = sum(abs(S_new)>0); 
    
    if iter > 2 && sum(abs(Tsignabs - Tsignabso)) ==0 && sum(abs(Ssignabs - Ssignabso))==0
        times_for_same_supp = times_for_same_supp + 1;
    else
        times_for_same_supp = 0;
    end
    
    if times_for_same_supp >= r && mu*norm(xbar - x)>tol
        cond = 1;
    else
        cond = 0;
    end
    
    if (times_for_same_supp>=r+3 && search_num2 >5) || times_for_same_supp>20
        cond = 0;
        times_for_same_supp = 0;
    end
    
    %% ******************* second order step ********************
    search_num2 = 0;
    N_step = 1;
    if cond
        sec_start = clock;
        red_nnz = sum(abs(S)>0);
        Tc = setdiff(1:n-1, T);
        BTcS = B(Tc, S);
        
        u = x(S);
        [robjf, rgradf, hess] = lfun.reduce(u, S);
        muk = 1.0e-3*(mu*normdiff)^(1/2);
        G = hess + muk*speye(red_nnz);
        
        model.Q = sparse(G/2);
        model.A = sparse(BTcS);
        model.obj = rgradf - G * u;
        model.sense = '=';
        model.lb = lb*ones(1, red_nnz);
        model.ub = ub*ones(1, red_nnz);
        model.sense = '=';
        params.outputflag = 0;
        params.Cutoff = 0.5*u'*G*u + (rgradf - G * u)'*u;
        params.OptimalityTol = max(1.0e-9, min(0.01, 0.5*min(1/mu, 1)*min((mu*normdiff), (mu*normdiff)^(5/3))));
        params.ScaleFlag = 3;
        results = gurobi(model, params);
        % To avoid infeasibility, make a projection of v onto Pi_k.
        v = results.x;
        vfull = zeros(n, 1); vfull(S) = v;
        vfull = proj_Bxbox(vfull, BTcS, S, lb, ub);
        ds = vfull(S) - u;
        
        ut =  u + N_step * ds;
        rgradf_ds = 1.0e-4 * rgradf' * ds;
        if rgradf_ds < 0
            robjf1 = robjf + rgradf_ds;
            robjf2 = lfun.reduce(ut, S);
            while robjf2 > robjf1
                N_step = N_step * 1/2;
                ut = u + N_step*ds;
                robjf2 = lfun.reduce(ut, S); 
                robjf1 = robjf + N_step * rgradf_ds;
                search_num2 = search_num2+1; % count the line search time
            end
            xnew = zeros(n, 1);
            xnew(S) = ut;
            sec_iter = sec_iter + 1;
        else
            xnew = xbar;
            cond = 0;
        end
        [T_new, ~] = reg.Bxsupp(xnew);
        [S_new, ~] = reg.xsupp(xnew);
        sec_time = sec_time + etime(clock, sec_start); % Cumulative time for second order step
    end   
    
    %% ************** Judgement of optimality condition *******************
    
    [fobj, grad_fxnew] = lfun.full(xnew);
    obj_new = fobj+reg.val(xnew);
    obj_diff = obj_new - obj_old;
    obj_old = obj_new;
    
    opt_measure = mu*max(abs(x-xbar));
    iter_time = etime(clock, main_tstart);
     
    if (iter_print)&&(mod(iter,1)==0)
        fprintf('\n %3d     %3.2e   %3.2e  %.3f  %i   %.2f  %.3f    %i %3.2e %i',iter,opt_measure,obj_diff,iter_time, search_num1, N_step , sum(abs(T_new)>1.0e-4), sum(abs(S_new)>0), norm(x-xbar, 'inf'), cond);
    end
    %%
    %% ************* check stopping criterion ******************
    %%
    
    if (opt_measure<tol)
        out.xopt = xnew;
        out.cput = etime(clock, main_tstart);
        out.iter = iter;
        out.obj = obj_new;
        out.sec_iter = sec_iter;
        out.sec_time = sec_time;
        out.solve_ok = 1;
        out.Tnnz = T_nnz;
        out.Snnz = S_nnz;
        if result_print
            fprintf('\n*************************** Result printed by PGiPN *******************************');
            fprintf('\n | iter | sec_iter | total time | sec time  ');
            fprintf('\n    %i      %i       %g        %g       ', iter, sec_iter, out.cput, sec_time);
            fprintf('\n**********************************************************************************');
        end
        return;
    end
    
    sk = xnew - x; rk = grad_fxnew - grad_fx;
    x = xnew; T = T_new; S = S_new;
    Tsignabso = Tsignabs; Ssignabso = Ssignabs;
end
if (iter==maxiter)
    out.xopt = xnew;
    out.cput = etime(clock, main_tstart);
    out.iter = iter;
    out.obj = obj_new;
    out.sec_iter = sec_iter;
    out.sec_time = sec_time;
    out.solve_ok = 0;
    out.nnz = S_nnz;
    if result_print
        fprintf('\n The algorithm cannot achieve the required precision in the maximal iteration!');
    end
    return;
end
end
