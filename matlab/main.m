% data = readmatrix('../data/b1382.csv');
% b1382 = data(:,[5 2]);

sampling_rate = 10;
data = readmatrix('../data/b1383.txt');
b1383 = data(:,[1 3 7]);
b1383(:,1) = b1383(:,1) / 5;
b1383(:,1) = smoothdata(b1383(:,1),'movmean',1/sampling_rate,'SamplePoints',b1383(:,3));
lvdt = resample(b1383(:,1),b1383(:,3),sampling_rate,'linear');
b1383(:,2) = b1383(:,2) / 2.5;
b1383(:,2) = smoothdata(b1383(:,2),'movmean',1/sampling_rate,'SamplePoints',b1383(:,3));
shear = resample(b1383(:,2),b1383(:,3),sampling_rate,'linear');
b1383 = [lvdt shear];
b1383 = array2timetable(b1383, 'SampleRate',sampling_rate);
clear data y ty sampling_rate lvdt shear;