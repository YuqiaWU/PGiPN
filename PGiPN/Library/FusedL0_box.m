%% This file considers the fused L0 norms plus a box constraint.
%% B is a matrix with dimension (n-1)*n with B_{ii} = 1, B_{i,i+1} = -1, and B_{ij} = 0 otherwise.
%% call_Bxsupp: return the support of Bx
%% call_xsupp: return the support of x
%% call_objval: return the value of lam1*||Bx||_0 + lam2||x||_0
%% call_prox: return one optimal solution of 
%%            min_x 0.5*||x-z||^2 + lam1*||Bx||_0 + lam2||x||_0, s.t. lb<=x<=ub 
%% ******************************************************************************

function obj = FusedL0_box(lam1, lam2, lb, ub)
obj.val = @(x) call_objval(x, lam1, lam2, lb, ub);
obj.prox = @(z, mu) call_prox(z, lam1, lam2, mu, lb, ub);
obj.Bxsupp = @(x) call_Bxsupp(x);
obj.xsupp = @(x) call_xsupp(x);
end

function [T, Tsignabs] = call_Bxsupp(x)
leng = length(x);
Bx = x(1:leng-1) - x(2:leng);
T = find(abs(Bx) > 0);
Tsignabs = zeros(leng-1, 1);
Tsignabs(T) = 1;
end

function [S, Ssignabs] = call_xsupp(x)
S = find(abs(x) > 0);
Ssignabs = zeros(length(x), 1);
Ssignabs(S) = 1;
end

function [val] = call_objval(x, lam1, lam2, lb, ub)
leng = length(x);
Bx = x(1:leng-1) - x(2:leng);
val = lam1 * sum(abs(Bx) > 0) + lam2 * sum(abs(x)>0);
end


function [prox] = call_prox(z, lam1, lam2, mu, lb, ub)
lambda1 = lam1/mu;
lambda2 = lam2/mu;
n = length(z);
if lb<0 && 0<ub
    alpha_mat = [1, lb, 0, 0; 1, 0, 0, 1; 0, 0, ub, 1];
    cost_mat = [0.5, -z(1), 0.5*z(1)^2; 0.5, -z(1), 0.5*z(1)^2; 0.5, -z(1), 0.5*z(1)^2];
elseif lb == 0
    alpha_mat = [1, 0, 0, 1; 0, 0, ub, 1];
    cost_mat = [0.5, -z(1), 0.5*z(1)^2; 0.5, -z(1), 0.5*z(1)^2];
elseif ub == 0
    alpha_mat = [lb, 0, 0, 0; 1, 0, 0, 1];
    cost_mat = [0.5, -z(1), 0.5*z(1)^2; 0.5, -z(1), 0.5*z(1)^2];
else
    alpha_mat = [1, lb, ub, 1];
    cost_mat = [0.5, -z(1), 0.5*z(1)^2];
end

R_list = [1, lb, ub, 1, 0];
opt_mat = [];
opt_mat(1) = 0;
for i = 2:n
    [cost_matnew, alpha_matnew, const_list, iszeroRSsm1] = cost_generator_L0(cost_mat, alpha_mat, z(i), lambda1, lambda2);
    if isempty(const_list)
        R_listnew = R_list;
    else
        RSSm1 = alpha_matnew(const_list, :);
        R_listnew = compute_Rlist_L0(RSSm1, R_list, lb, ub, iszeroRSsm1);
        R_listnew = [R_listnew; RSSm1, (i-1)*ones(size(RSSm1, 1), 1)];
    end
    [leng, ~] = size(R_listnew);
    [~, opt] = costminL0_mex(cost_matnew, alpha_matnew(:, 2:3), lambda2);
    for k = 1:leng
        if (R_listnew(k, 2) < opt && opt< R_listnew(k, 3)) || (R_listnew(k, 2) == opt && R_listnew(k, 1) == 1) || ...
                (R_listnew(k, 3) == opt && R_listnew(k, 4) == 1)
            opt_mat(i) = R_listnew(k,5);
            break;
        end
    end
    R_list = R_listnew;
    cost_mat = cost_matnew;
    alpha_mat = alpha_matnew;
end

jump_point = [];
t = 1;
while 1
    if t == 1
        jump_point(t) = opt_mat(n);
        t = t + 1;
    else
        jump_point(t) = opt_mat(jump_point(t-1));
        t = t + 1;
    end
    if jump_point(t-1) <=1
        break;
    end
end

xopt = [];
for i = 1:length(jump_point)
    if i == 1
        ztmp = z(jump_point(i)+1:n);
        lengztmp = n - jump_point(i);
        gtmp = sum(ztmp)/lengztmp;
        if gtmp >= lb && gtmp <= ub
            gopt = gtmp.*sign(max(abs(gtmp) - sqrt(2 * lambda2), 0));
            xopt(jump_point(i)+1:n) = gopt;
        else
            tmpfun = @(t) 0.5*norm(t*ones(lengztmp, 1) - ztmp) + lengztmp * lambda2 * abs(sign(t));
            objzero = tmpfun(0); objlb = tmpfun(lb); objub = tmpfun(ub);
            pointlist = [0; lb; ub];
            objlist = [objzero; objlb ; objub]; 
            [~, index] = sort(objlist, 'ascend');
            gopt = pointlist(index(1));
            xopt(jump_point(i)+1:n) = gopt;
        end
    else
        ztmp = z(jump_point(i)+1:jump_point(i-1));
        lengztmp = jump_point(i-1)-jump_point(i);
        gtmp = sum(ztmp)/lengztmp;
        if gtmp >= lb && gtmp <= ub
            gopt = gtmp.*sign(max(abs(gtmp) - sqrt(2 * lambda2), 0));
            xopt(jump_point(i)+1:jump_point(i-1)) = gopt;
        else
            tmpfun = @(t) 0.5*norm(t*ones(lengztmp, 1) - ztmp) + lengztmp * lambda2 * abs(sign(t));
            objzero = tmpfun(0); objlb = tmpfun(lb); objub = tmpfun(ub);
            pointlist = [0; lb; ub];
            objlist = [objzero; objlb ; objub]; 
            [~, index] = sort(objlist, 'ascend');
            gopt = pointlist(index(1));
            xopt(jump_point(i)+1:jump_point(i-1)) = gopt;
        end
    end
    if i == length(jump_point)
        ztmp = z(1:jump_point(i));
        lengztmp = jump_point(i);
        gtmp = sum(ztmp)/lengztmp;
        if gtmp >= lb && gtmp <= ub
            gopt = gtmp.*sign(max(abs(gtmp) - sqrt(2 * lambda2), 0));
            xopt(1:jump_point(i)) = gopt;
        else
            tmpfun = @(t) 0.5*norm(t*ones(lengztmp, 1) - ztmp) + lengztmp * lambda2 * abs(sign(t));
            objzero = tmpfun(0); objlb = tmpfun(lb); objub = tmpfun(ub);
            pointlist = [0; lb; ub];
            objlist = [objzero; objlb ; objub]; 
            [~, index] = sort(objlist, 'ascend');
            gopt = pointlist(index(1));
            xopt(1:jump_point(i)) = gopt;
        end
    end
end
prox = xopt';
end


function [cost_matnew, alpha_matnew, const_list, iszeroRSsm1] = cost_generator_L0(cost_mat, alpha_mat, y, lambda1, lambda2)
[p, ~] = size(cost_mat);
optval = costminL0_mex(cost_mat, alpha_mat(:, 2:3), lambda2) + lambda1;
t = 1;
cost_mattmp = [];
alpha_mattmp = [];
const_list = [];
s = 1;
iscontain0 = 0;
for i = 1:p
    ct1 = cost_mat(i,1);ct2 = cost_mat(i,2);ct3 = cost_mat(i,3);
    ap1 = alpha_mat(i,2);ap2 = alpha_mat(i,3);
    bp1 = alpha_mat(i,1:2); bp2 = alpha_mat(i, 3:4);
    if ap1 == 0 && ap2 == 0
        iscontain0 = 1;
        isnowcontain0 = 1;
    else
        isnowcontain0 = 0;
    end
    if ~isnowcontain0
        %% ************************************************************************
        ct3minus = ct3+ct1*2*lambda2-optval; % not consider the case where x = 0
        delta = ct2^2-4*ct1*ct3minus;
        if delta <= 0
            cost_mattmp(t,:) = [0, 0, optval];
            alpha_mattmp(t,:) = alpha_mat(i,:);
            t = t+1;
        else
            if ct1 > 0
                sol1 = (-ct2-sqrt(delta))/(2*ct1);
                sol2 = (-ct2+sqrt(delta))/(2*ct1);
                if sol2 <=ap1 || sol1>=ap2
                    cost_mattmp(t,:) = [0, 0, optval];
                    alpha_mattmp(t,:) = alpha_mat(i,:);
                    t = t+1;
                elseif sol1 <=ap1 && ap2<=sol2
                    cost_mattmp(t,:) = cost_mat(i,:);
                    alpha_mattmp(t,:) = alpha_mat(i,:);
                    t = t+1;
                elseif ap1 < sol1 && sol2<ap2
                    cost_mattmp(t,:) = [0, 0, optval];
                    alpha_mattmp(t,:) = [bp1, sol1, 0];
                    t = t+1;
                    cost_mattmp(t,:) = cost_mat(i,:);
                    alpha_mattmp(t,:) = [1, sol1, sol2, 1];
                    t = t+1;
                    cost_mattmp(t,:) = [0, 0, optval];
                    alpha_mattmp(t,:) = [0, sol2, bp2];
                    t = t+1;
                elseif sol1<=ap1 && ap1<sol2 && sol2<ap2
                    cost_mattmp(t,:) = cost_mat(i,:);
                    alpha_mattmp(t,:) = [bp1, sol2, 1];
                    t = t+1;
                    cost_mattmp(t,:) = [0, 0, optval];
                    alpha_mattmp(t,:) = [0, sol2, bp2];
                    t = t+1;
                else
                    cost_mattmp(t,:) = [0, 0, optval];
                    alpha_mattmp(t,:) = [bp1, sol1, 1];
                    t = t+1;
                    cost_mattmp(t,:) = cost_mat(i,:);
                    alpha_mattmp(t,:) = [0, sol1, bp2];
                    t = t+1;
                end
            else
                if ct2 == 0
                    if ct3_new > 0
                        cost_mattmp(t,:) = [0, 0, optval];
                        alpha_mattmp(t,:) = [bp1, bp2];
                        t = t+1;
                    else
                        cost_mattmp(t,:) = cost_mat(i,:);
                        alpha_mattmp(t,:) = [bp1, bp2];
                        t = t+1;
                    end
                elseif ct2>0
                    if ct2*ap1 + ct3 >= optval
                        cost_mattmp(t,:) = [0, 0, optval];
                        alpha_mattmp(t,:) = [bp1, bp2];
                        t = t+1;
                    elseif ct2*ap2 + ct3<=optval
                        cost_mattmp(t,:) = cost_mat(i,:);
                        alpha_mattmp(t,:) = [bp1, bp2];
                        t = t+1;
                    else
                        cost_mattmp(t,:) = cost_mat(i,:);
                        alpha_mattmp(t,:) = [bp1, optval/ct2, 1];
                        t = t+1;
                        cost_mattmp(t,:) = [0, 0, optval];
                        alpha_mattmp(t,:) = [0, optval/ct2, bp2];
                        t = t+1;
                    end
                else
                    if ct2*ap2 + ct3>= optval
                        cost_mattmp(t,:) = [0, 0, optval];
                        alpha_mattmp(t,:) = [bp1, bp2];
                        t = t+1;
                    elseif ct2*ap1+ ct3<=optval
                        cost_mattmp(t,:) = cost_mat(i,:);
                        alpha_mattmp(t,:) = [bp1, bp2];
                        t = t+1;
                    else
                        cost_mattmp(t,:) = [0, 0, optval];
                        alpha_mattmp(t,:) = [bp1, optval/ct2, 1];
                        t = t+1;
                        cost_mattmp(t,:) = cost_mat(i,:);
                        alpha_mattmp(t,:) = [0, optval/ct2, bp2];
                        t = t+1;
                    end
                end
            end
        end
    else
        if ct3 < optval
            cost_mattmp(t,:) = [0, 0, ct3];
            alpha_mattmp(t,:) = [1, 0, 0, 1];
            iszeroRSsm1 = 0;
            t =  t + 1;
        else
            cost_mattmp(t,:) = [0, 0, optval];
            alpha_mattmp(t,:) = [1, 0, 0, 1];
            iszeroRSsm1 = 1;
            t =  t + 1;
        end
    end

end

% To avoid sol1 or sol2 equals to zero, which leads to (a, 0] or [0, b) in
% alpha mat.
for j = 1:t-1
    if alpha_mattmp(j, 2)<0 && alpha_mattmp(j,3)==0 
        alpha_mattmp(j, 4) = 0;
    end
    if alpha_mattmp(j, 2)==0 && alpha_mattmp(j,3)>0 
        alpha_mattmp(j, 1) = 0;
    end
end


leng = size(cost_mattmp, 1);
clear_list= [];
i = 1; k = 0;
if size(cost_mattmp, 1)>=2
    while 1
        while all(cost_mattmp(i,:) == cost_mattmp(i+1,:)) && alpha_mattmp(i,3) == alpha_mattmp(i+1, 2) && alpha_mattmp(i,4)+ alpha_mattmp(i+1,1) == 1 ...
                && ~(abs(alpha_mattmp(i,2))+ abs(alpha_mattmp(i,3))== 0) && ~(abs(alpha_mattmp(i+1,2))+ abs(alpha_mattmp(i+1,3))== 0)
            k = k+1; i = i+1;
            if i >= leng
                break;
            end
        end
        
        if k == 0
            i = i + 1;
        end
        if k > 0
            alpha_mattmp(i-k, 3:4) = alpha_mattmp(i,3:4);
            clear_list = [clear_list, i-k+1:i];
            k = 0;
        end
        if i >=leng
            break;
        end
    end
end
if ~isempty(clear_list)
    alpha_mattmp(clear_list,:) = [];
    cost_mattmp(clear_list,:) = [];
end

leng = size(cost_mattmp, 1);
const_list = [];
for i = 1:leng
    if all(cost_mattmp(i, 1:2) == [0, 0])  %     && (~all(alpha_mattmp(i, 2:3) == [0,0])  )
        if all(alpha_mattmp(i, 2:3) == [0,0]) 
            if iszeroRSsm1
                const_list = [const_list, i];
            end
        else
            const_list = [const_list, i];
        end
    end
end

% leng = size(cost_mattmp, 1);
add_term = ones(leng, 1) * [0.5; -y; 0.5*y^2]';
alpha_matnew = alpha_mattmp;
cost_matnew = cost_mattmp + add_term;
end

function [R_listnew] = compute_Rlist_L0(RSsm1, R_list, lb, ub, iszeroRSsm1)

[m, ~] = size(R_list);
[n, ~] = size(RSsm1);
R_listnew = [];
t = 1;
s = 1;
diff_RS = [];

% Compute (R_s^{s-1})^c
if n == 1
    if RSsm1(1,2) == lb % lb<= 0 < ub
        if lb == 0
            if RSsm1(1,3) == ub
                diff_RS = [1, 0, 0, 1];
            elseif  RSsm1(1,3) > 0 && RSsm1(1,3) < ub
                diff_RS = [1, 0, 0, 1; 1- RSsm1(1,4), RSsm1(1,3), ub, 1];
            elseif RSsm1(1,3) == 0 && RSsm1(1,3) < ub
                diff_RS = [0, 0, ub, 1];
            end
        else % lb < 0
            if RSsm1(1,3) < 0
                diff_RS = [1-RSsm1(1,4), RSsm1(1,3), 0, 0;1, 0, 0, 1; 0, 0, ub, 1];
            elseif RSsm1(1,3) == 0
                diff_RS = [1, 0, 0, 1; 0, 0, ub, 1];
            else
                diff_RS = [1-RSsm1(1,4), RSsm1(1,3), ub, 1];
            end
        end
    else % RSsm1(1,2) > lb
        if lb == 0
            if RSsm1(1,3) == ub
                diff_RS = [1, 0, 0, 1; 0, 0, RSsm1(1,2), 1- RSsm1(1,1)];
            elseif  RSsm1(1,3) > 0 && RSsm1(1,3) < ub
                diff_RS = [1, 0, 0, 1; 0, 0, RSsm1(1,2), 1- RSsm1(1,1); 1- RSsm1(1,4), RSsm1(1,3), ub, 1];
            end
        else % lb<0
            if RSsm1(1,3) < 0
                diff_RS = [1, lb, RSsm1(1,2), 1-RSsm1(1); 1-RSsm1(1,4), RSsm1(1,3), 0, 0;1, 0, 0, 1; 0, 0, ub, 1];
            elseif  RSsm1(1,3) == 0
                diff_RS = [1, lb, RSsm1(1,2), 1-RSsm1(1);1, 0, 0, 1; 0, 0, ub, 1];
            else
                diff_RS = [1, lb, RSsm1(1,2), 1-RSsm1(1);1-RSsm1(1,4), RSsm1(1,3), ub, 1];
            end
        end
    end
else
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if RSsm1(1,2) == lb
        if lb == 0
            if RSsm1(1,3) > 0
                diff_RS(s, 1:4) = [1, 0, 0, 1];
                s = s + 1;
            end
            if n == 2
                if RSsm1(1,3) == 0 && RSsm1(2,2) == 0 && RSsm1(2,3) == ub
                    R_listnew = [];
                    return;
                end
            end
            
            for i = 1:n-1          
                if ~((RSsm1(i,3) == RSsm1(i+1,2)) && (1 - RSsm1(i,4) + 1 - RSsm1(i+1,1)<2))
                    diff_RS(s,1:4) = [1 - RSsm1(i,4), RSsm1(i,3), RSsm1(i+1,2), 1 - RSsm1(i+1,1)];
                    s = s + 1;
                end
            end

        else % lb<0 and RSsm1(1,2) == lb
            for i = 1:n-1
                % if ~(RSsm1(i, 2) == 0 && RSsm1(i, 3) == 0)
                if RSsm1(i, 2) < 0 && RSsm1(i, 3) == 0 && RSsm1(i+1, 2) > 0
                    diff_RS(s,1:4) = [1, 0, 0, 1];
                    s = s+1;
                    diff_RS(s,1:4) = [0, 0, RSsm1(i+1, 2), 1-RSsm1(i+1, 1)];
                    s = s+1;
                elseif RSsm1(i, 3) < 0 && RSsm1(i+1, 2) > 0
                    diff_RS(s,1:4) = [1-RSsm1(i, 4), RSsm1(i, 3), 0, 0];
                    s = s+1;
                    diff_RS(s,1:4) = [1, 0, 0, 1];
                    s = s+1;
                    diff_RS(s,1:4) = [0, 0, RSsm1(i+1, 2), 1-RSsm1(i+1, 1)];
                    s = s+1;
                elseif RSsm1(i, 3) < 0 && RSsm1(i+1, 2) == 0 && RSsm1(i+1, 3) > 0
                    diff_RS(s,1:4) = [1-RSsm1(i,4), RSsm1(i, 3), 0, 0];
                    s = s+1;
                    diff_RS(s,1:4) = [1, 0, 0, 1];
                    s = s+1;
                else
                    if ~((RSsm1(i,3) == RSsm1(i+1,2)) && (1 - RSsm1(i,4) + 1 - RSsm1(i+1,1)<2))
                        diff_RS(s,1:4) = [1 - RSsm1(i,4), RSsm1(i,3), RSsm1(i+1,2), 1 - RSsm1(i+1,1)];
                        s = s + 1;
                    end
                end
                % end
            end
        end
    else  % RSsm1(1,2) > lb
        if lb == 0
            diff_RS(s,1:4) = [1, 0, 0, 1];
            s = s + 1;
            diff_RS(s,1:4) = [0, 0, RSsm1(1,2), 1 - RSsm1(1,1)];
            s = s + 1;
            for i = 1:n-1                
                if ~((RSsm1(i,3) == RSsm1(i+1,2)) && (1 - RSsm1(i,4) + 1 - RSsm1(i+1,1) <2))
                    diff_RS(s,1:4) = [1 - RSsm1(i,4), RSsm1(i,3), RSsm1(i+1,2), 1 - RSsm1(i+1,1)];
                    s = s + 1;
                end
            end
        else % RSsm1(1,2) > lb and lb < 0 % Here RSsm1(1,2) must <= 0 since 0 is separated.
            diff_RS(s,1:4) = [1, lb, RSsm1(1,2), 1 - RSsm1(1,1)];
            s = s + 1;
            for i = 1:n-1
                if RSsm1(i, 2) < 0 && RSsm1(i, 3) == 0 && RSsm1(i+1, 2) > 0
                    diff_RS(s,1:4) = [1, 0, 0, 1];
                    s = s+1;
                    diff_RS(s,1:4) = [0, 0, RSsm1(i+1, 2), 1-RSsm1(i+1, 1)];
                    s = s+1;
                elseif RSsm1(i, 3) < 0 && RSsm1(i+1, 2) > 0
                    diff_RS(s,1:4) = [1-RSsm1(i, 4), RSsm1(i, 3), 0, 0];
                    s = s+1;
                    diff_RS(s,1:4) = [1, 0, 0, 1];
                    s = s+1;
                    diff_RS(s,1:4) = [0, 0, RSsm1(i+1, 2), 1-RSsm1(i+1, 1)];
                    s = s+1;
                elseif RSsm1(i, 3) < 0 && RSsm1(i+1, 2) == 0 && RSsm1(i+1, 3) > 0
                    diff_RS(s,1:4) = [1-RSsm1(i,4), RSsm1(i, 3), 0, 0];
                    s = s+1;
                    diff_RS(s,1:4) = [1, 0, 0, 1];
                    s = s+1;
                else
                    if ~((RSsm1(i,3) == RSsm1(i+1,2)) && (1 - RSsm1(i,4) + 1 - RSsm1(i+1,1)<2))
                        diff_RS(s,1:4) = [1 - RSsm1(i,4), RSsm1(i,3), RSsm1(i+1,2), 1 - RSsm1(i+1,1)];
                        s = s + 1;
                    end
                end

            end
        end
    end
    
    if RSsm1(n, 3) < ub
        diff_RS(s, :) = [1 - RSsm1(n,4), RSsm1(n,3), ub, 1];
    end
end

if ~isempty(diff_RS)
    for i = 1:m
        num1 = R_list(i, 5);
        for j = 1:size(diff_RS, 1)
            Intersection = Interval_Intersection(R_list(i,1:4), diff_RS(j, :));
            if ~isempty(Intersection)
                R_listnew(t, :) = [Intersection, num1];
                t = t + 1;
            end
        end
    end
else
    R_listnew = [];
    return;
end
    
    
R_len = size(R_listnew, 1);
clear_list = [];
if R_len > 1
    for i = 1:R_len-1
        if R_listnew(R_len-i, 5) == R_listnew(R_len-i+1, 5) && R_listnew(R_len-i+1, 2) == R_listnew(R_len-i, 3) && ...
                ~(abs(R_listnew(R_len-i+1, 2)) + abs(R_listnew(R_len-i, 3)) == 0)
            R_listnew(R_len-i, 3) = R_listnew(R_len-i+1, 3);
            clear_list = [clear_list, R_len-i+1];
        end
    end
end
R_listnew(clear_list, :) = [];
end

function [optval, opt] = cost_min_L0(cost_mat, alpha_mat, lambda2)
[p, ~] = size(cost_mat);
val_list = zeros(p,1);
opt_list = [];
for i = 1:p
    ct1 = cost_mat(i,1);ct2 = cost_mat(i,2);ct3 = cost_mat(i,3);
    ap1= alpha_mat(i,2);ap2 = alpha_mat(i,3);
    if ~(ap1== 0 && ap2 == 0)
        g = -ct2/ct1*0.5;
        minp = g*sign(max(abs(g) - sqrt(lambda2/ct1), 0));
        operator = @(z) ct1*z^2 + ct2*z + ct3 + lambda2 * ct1 * 2;
        valminp = operator(minp);
        valap1 = operator(ap1);
        valap2 = operator(ap2);
        if valminp <= min(valap1, valap2) && minp <= ap2 && minp >= ap1
            val_list(i) = valminp;
            opt_list(i) = minp;
        elseif valap1 <= valap2
            val_list(i) = valap1;
            opt_list(i) = ap1;
        else
            val_list(i) = valap2;
            opt_list(i) = ap2;
        end
    else
        val_list(i) = ct3;
        opt_list(i) = 0;
    end
end
[~, index] = sort(val_list, 'ascend');
optval = val_list(index(1));
opt = opt_list(index(1));
end

function intersections = Interval_Intersection(arr1, arr2)
    
    intersections = [];
    
    for i = 1:size(arr1, 1)
        for j = 1:size(arr2, 1)
            is_intersecting = (arr1(i, 3) > arr2(j, 2) && arr1(i, 2) < arr2(j, 3)) || ...
                              (arr2(j, 3) > arr1(i, 2) && arr2(j, 2) < arr1(i, 3)) || ...
                              (arr1(i, 3) == arr2(j, 2) && arr1(i, 4)+ arr2(j, 1) == 2) || ...
                              (arr2(j, 3) == arr1(i, 2) && arr2(j, 4)+ arr1(i, 1) == 2);
                              
            
            if is_intersecting
                left_endpoint = max(arr1(i, 2), arr2(j, 2));
                right_endpoint = min(arr1(i, 3), arr2(j, 3));
                
                if arr1(i, 2) > arr2(j, 2)
                    left_type = arr1(i, 1);
                elseif arr1(i, 2) < arr2(j, 2)
                    left_type = arr2(j, 1);
                else
                    left_type = min(arr1(i, 1), arr2(j, 1));
                end

                if arr1(i, 3) < arr2(j, 3)
                    right_type = arr1(i, 4);
                elseif arr1(i, 3) > arr2(j, 3)
                    right_type = arr2(j, 4);
                else
                    right_type = min(arr1(i, 4), arr2(j, 4));
                end
                
                intersections = [intersections; left_type, left_endpoint, right_endpoint, right_type];
            end
        end
    end
end