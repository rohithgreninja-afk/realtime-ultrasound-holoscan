data=load('C:\Users\rohit\Downloads\Real Time Image Processing Project\OASBUD.mat');
disp(fieldnames(data));

%size and type of data in the file
filePath = 'C:\Users\rohit\Downloads\Real Time Image Processing Project\OASBUD.mat';
load(filePath);
disp(class(data));
disp(size(data));

%Shows the fields of the struct array
disp(fieldnames(data));

%displaying rf data of one patient
disp(data(1).id); % patient id
disp(data(1).class); %mm is malignant
disp(size(data(1).rf1)); %rf signal array dimensions
disp(class(data(1).rf1)); % rf signal array data

%visualising the raw RF data
figure;
imagesc(data(1).rf1);
colormap gray;
colorbar;
title('Raw RF Data - Patient 1 (Malignant)');
xlabel('Transducer Elements');
ylabel('Time Samples (Depth)');

%Check acquistion of paramters stored
disp(data(1).rf1(1:5, 1));
fs_check = (size(data(1).rf1, 1));
fprintf('Number of time samples: %d\n', fs_check);
fprintf('Number of elements: %d\n', size(data(1).rf1, 2));

%check if dataset has seperate field for sampling freq.

load('C:\Users\rohit\Downloads\Real Time Image Processing Project\OASBUD.mat');
whos

%Defining acquistion parameters
fs = 25e6;          % Sampling frequency: 25 MHz
c  = 1540;          % Speed of sound in soft tissue (m/s)
fc = 5e6;           % Centre frequency of the ultrasound pulse: 5 MHz
pitch = 0.245e-3;   % Distance between adjacent transducer elements: 0.245 mm

% Derived values
lambda     = c / fc;               % Wavelength in metres
dt         = 1 / fs;               % Time between each sample in seconds
num_samples  = size(data(1).rf1, 1); % 1824
num_elements = size(data(1).rf1, 2); % 510

fprintf('Wavelength:         %.4f mm\n', lambda * 1000);
fprintf('Time per sample:    %.4f microseconds\n', dt * 1e6);
fprintf('Total depth range:  %.2f mm\n', (num_samples * dt * c / 2) * 1000);
fprintf('Transducer width:   %.2f mm\n', num_elements * pitch * 1000);