function [diff_interval_list] = diff_interval_list_generator(interval_list, length)
% This function returns an vector but not list of intervals.
n = size(interval_list, 1);
j = 1;
diff_interval_list = [];
if ~isempty(interval_list)
    if  interval_list(1, 1) > 1
        diff_interval_list = 1:interval_list(1,1)-1;
        row = 2;
    end
    while j < n
        if interval_list(j, 2) < interval_list(j+1, 1) - 1
            diff_interval_list = [diff_interval_list, interval_list(j, 2)+1:interval_list(j+1,1)- 1];
        end
        j = j+1;
    end
    if interval_list(end, 2) < length
        diff_interval_list = [diff_interval_list, interval_list(j, 2)+1:length];
    end

else
    diff_interval_list = 1:length;
end