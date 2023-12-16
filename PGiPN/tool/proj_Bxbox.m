%% ************************************************************************
% This package is to compute the projection of z onto Pi, which is of the form
% Pi = {z : B_{T_k^c}z = 0, z_{S_k^c} = 0, l<= z <= u}, where B is the
% matrix defining the fused Lasso.
%% ************************************************************************
function [x] = proj_Bxbox(z, BTcS, S, l, u)

Sum2BTkcSk = sum(abs(BTcS), 2);
BTcS((Sum2BTkcSk == 0),:) = [];

n = length(z);
w = z(S);
% create a new vector to verify the new Tc
s = length(S);
y = [];
if mod(s, 2)==1
    y(1:2:s) = 1:-1:(2-(s+1)/2);
    y(2:2:s-1) = 2:1:((s-1)/2+1);
else
    y(1:2:s-1) = 1:-1:(2-s/2);
    y(2:2:s) = 2:1:(s/2+1);
end
Tkc = abs(BTcS*y');

length_Tkc = length(Tkc);
length_S = length(S);
interval_list = interval_list_generator(Tkc, length_Tkc);
diff_interval_list = diff_interval_list_generator(interval_list, length_S);

row_interlist = size(interval_list, 1);
for i = 1:row_interlist
    v1 = interval_list(i, 1);
    v2 = interval_list(i, 2);
    v(v1:v2) = ones(v2+1-v1, 1) * max(min(sum(w(v1:v2)/(v2+1 - v1)),u), l);
end
v(diff_interval_list) = max(min(w(diff_interval_list),u), l);
v = v';
x = zeros(n, 1);
x(S) =v;
end
