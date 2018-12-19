function [interactionfeatures] = interactiongen(data_in)
%% this function generates a matrix with interaction features

k = 1;
interactionfeatures=data_in;
for i = 1 : size(data_in, 2)-1
   for j= i+1 : size(data_in, 2)
       interactionfeatures(:,size(data_in, 2)+k)=data_in(:,i).*data_in(:,j);
       k = k + 1;
   end
end
end

