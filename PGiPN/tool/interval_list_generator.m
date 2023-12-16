function [interval_list] = interval_list_generator(Tkc, length_Tkc)
interval_list = [];
j = 1;
row = 1;
while j<length_Tkc
    interval_list(row, 1) = Tkc(j);
    while j+1<= length_Tkc && Tkc(j)+1== Tkc(j+1)
        j = j+1;
    end
    interval_list(row, 2)= Tkc(j)+1;
    j = j + 1;
    row = row + 1;
end
if ~isempty(interval_list) && interval_list(end, 2) <  Tkc(end)
    interval_list(row, :) = [Tkc(end), Tkc(end)+1];
end
end