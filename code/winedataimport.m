function [data_total_test, data_total_train, data_white_test, data_white_train, data_red_test, data_red_train] = winedataimport()

% This function imports the wine samples from the csv files, creates a test
% and a training set and stores them in separate tables

filename = 'winequality-red.csv';
delimiter = ';';
startRow = 2;
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';
fileID = fopen('data/winequality-red.csv','r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
redwines = table(dataArray{1:end-1}, 'VariableNames', {'fixedacidity','volatileacidity','citricacid','residualsugar','chlorides','freesulfurdioxide','totalsulfurdioxide','density','pH','sulphates','alcohol','quality'});

filename = 'winequality-white.csv';
delimiter = ';';
startRow = 2;
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';
fileID = fopen('data/winequality-white.csv','r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
whitewines = table(dataArray{1:end-1}, 'VariableNames', {'fixedacidity','volatileacidity','citricacid','residualsugar','chlorides','freesulfurdioxide','totalsulfurdioxide','density','pH','sulphates','alcohol','quality'});

%% add type of wines to datasets (red: 0, white: 1)
whitewines = horzcat(table(ones(size(whitewines,1), 1), 'VariableNames', {'WineType'}),whitewines);
redwines = horzcat(table(zeros(size(redwines,1),1), 'VariableNames', {'WineType'}),redwines);

%% combine the two datasets
totalwinedata = vertcat(whitewines, redwines); 

%% randomly split data into training and test set
numberofwines = size(totalwinedata, 1);
n_white_total = size(whitewines,1);
n_red_total = size (redwines,1);
n_test_samples = round(numberofwines*0.2);
n_train_samples = numberofwines - n_test_samples;

random_index = randperm(numberofwines);
test_indices = random_index(1:n_test_samples);
train_indices = random_index(n_test_samples+1:end);

data_total_test = totalwinedata(test_indices, :);
data_total_train = totalwinedata(train_indices, :);

%% based on the separation of the combined dataset, create test and training sets for red and white wines only
test_indices_white = test_indices((test_indices <= n_white_total));
test_indices_red = test_indices((test_indices > n_white_total)) - n_white_total;
train_indices_white =train_indices((train_indices <= n_white_total));
train_indices_red =train_indices((train_indices > n_white_total)) - n_white_total;

data_white_test = whitewines(test_indices_white,:);
data_white_train = whitewines(train_indices_white,:);
data_red_test = redwines(test_indices_red,:);
data_red_train = redwines(train_indices_red,:);

%% create bar chart showing the wine qualities distribution 

n_quality = zeros(10,1);
for i = 0:10
   n_quality(i+1) = sum(totalwinedata.quality == i);
end
figure;
bar((0:10),n_quality);
xlabel("Wine Quality");
ylabel("Number of Samples");
title("Distribution of Samples with given Wine Quality");




%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;

end