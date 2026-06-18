% Phase4_MegaTrain.m
% Combined training: BUSI + BUS-UCLM + BUS-BRA + BrEaST + OASBUD-PNG
% Save as: C:\Users\rohit\Documents\MATLAB Code\Phase4_MegaTrain.m
% Run with R2024b only

clearvars; clc;
addpath(genpath('C:\ProgramData\MATLAB\SupportPackages\R2024b\toolbox\nnet\supportpackages\mobilenetv2'));

%% -------------------------------------------------------------------------
% SECTION 1: BUSI
% -------------------------------------------------------------------------
busiPath = fullfile('C:\Users\rohit\OneDrive\Documents\MATLAB\Examples\R2026a', ...
    'supportfiles\image\data\Dataset_BUSI\Dataset_BUSI_with_GT');

busiDS = imageDatastore(busiPath, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'FileExtensions', '.png');
isMask     = contains(busiDS.Files, '_mask');
cleanFiles = busiDS.Files(~isMask);
busiDS     = imageDatastore(cleanFiles, 'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true);
busiDS.Labels = categorical(lower(string(busiDS.Labels)));
fprintf('BUSI: %d images\n', numel(busiDS.Files));
disp(countcats(busiDS.Labels));

%% -------------------------------------------------------------------------
% SECTION 2: BUS-UCLM
% -------------------------------------------------------------------------
uclmRoot = fullfile('C:\Users\rohit\Downloads\Real Time Image Processing Project', ...
    'BUS-UCLM Breast ultrasound lesion segmentation dataset', ...
    'BUS-UCLM Breast ultrasound lesion segmentation dataset', 'BUS-UCLM');

info     = readtable(fullfile(uclmRoot, 'INFO.csv'));
cleanIdx = strcmp(info.Doppler, 'No');
info     = info(cleanIdx, :);
fullPaths = fullfile(fullfile(uclmRoot, 'images'), info.Image);
labelStrs = lower(info.Label);
missing   = ~isfile(fullPaths);
fullPaths(missing) = []; labelStrs(missing) = [];
uclmDS = imageDatastore(fullPaths, 'Labels', categorical(labelStrs));
fprintf('BUS-UCLM: %d images\n', numel(uclmDS.Files));
disp(countcats(uclmDS.Labels));

%% -------------------------------------------------------------------------
% SECTION 3: BUS-BRA
% -------------------------------------------------------------------------
busbraRoot   = 'C:\Users\rohit\Downloads\Real Time Image Processing Project\BUSBRA\BUSBRA';
busbraImgDir = fullfile(busbraRoot, 'Images');
busbraCSV    = fullfile(busbraRoot, 'bus_data.csv');

busInfo   = readtable(busbraCSV);
filenames = busInfo.ID;     % e.g. 'bus_0001-l'
labels    = lower(busInfo.Pathology);  % 'benign' or 'malignant'

% Build full paths -- images are PNG files named by ID
fullPathsBRA = fullfile(busbraImgDir, strcat(filenames, '.png'));
missing = ~isfile(fullPathsBRA);
if any(missing)
    fprintf('BUS-BRA: %d files not found\n', sum(missing));
    fullPathsBRA(missing) = [];
    labels(missing) = [];
end
busbraDS = imageDatastore(fullPathsBRA, 'Labels', categorical(labels));
fprintf('BUS-BRA: %d images\n', numel(busbraDS.Files));
disp(countcats(busbraDS.Labels));

%% -------------------------------------------------------------------------
% SECTION 4: BrEaST
% -------------------------------------------------------------------------
breastImgDir = fullfile('C:\Users\rohit\Downloads\Real Time Image Processing Project', ...
    'BrEaST-Lesions_USG-images_and_masks-Dec-15-2023', ...
    'BrEaST-Lesions_USG-images_and_masks');
breastXLSX = fullfile('C:\Users\rohit\Downloads\Real Time Image Processing Project', ...
    'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx');

breastInfo  = readtable(breastXLSX);
imgFiles    = breastInfo.Image_filename;
classLabels = lower(breastInfo.Classification);  % benign/malignant/normal

fullPathsBrEaST = fullfile(breastImgDir, imgFiles);
missing = ~isfile(fullPathsBrEaST);
if any(missing)
    fprintf('BrEaST: %d files not found\n', sum(missing));
    fullPathsBrEaST(missing) = [];
    classLabels(missing)     = [];
end
breastDS = imageDatastore(fullPathsBrEaST, 'Labels', categorical(classLabels));
fprintf('BrEaST: %d images\n', numel(breastDS.Files));
disp(countcats(breastDS.Labels));

%% -------------------------------------------------------------------------
% SECTION 5: OASBUD PNG (augmented reconstructions)
% -------------------------------------------------------------------------
oasbRoot = 'C:\Users\rohit\Downloads\OASBUD_PNG';

if exist(oasbRoot, 'dir')
    oasbDS = imageDatastore(oasbRoot, 'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', 'FileExtensions', '.png');
    oasbDS.Labels = categorical(lower(string(oasbDS.Labels)));
    fprintf('OASBUD PNG: %d images\n', numel(oasbDS.Files));
    disp(countcats(oasbDS.Labels));
    useOASBUD = true;
else
    fprintf('OASBUD PNG not found -- skipping (run Phase4_OASBUD_Export.m first)\n');
    useOASBUD = false;
end

%% -------------------------------------------------------------------------
% SECTION 6: Combine all datasets
% -------------------------------------------------------------------------
allFiles  = [busiDS.Files;   uclmDS.Files;   busbraDS.Files;  breastDS.Files];
allLabels = [busiDS.Labels;  uclmDS.Labels;  busbraDS.Labels; breastDS.Labels];

if useOASBUD
    allFiles  = [allFiles;  oasbDS.Files];
    allLabels = [allLabels; oasbDS.Labels];
end

combinedDS = imageDatastore(allFiles, 'Labels', allLabels);

fprintf('\n=== Combined Dataset ===\n');
fprintf('Total: %d images\n', numel(combinedDS.Files));
disp(countcats(combinedDS.Labels));

%% -------------------------------------------------------------------------
% SECTION 7: Train / Val / Test split 70/15/15
% -------------------------------------------------------------------------
rng(42);
[trainDS, tempDS] = splitEachLabel(combinedDS, 0.70, 'randomized');
[valDS,   testDS] = splitEachLabel(tempDS,     0.50, 'randomized');

fprintf('Train: %d   Val: %d   Test: %d\n', ...
    numel(trainDS.Files), numel(valDS.Files), numel(testDS.Files));

%% -------------------------------------------------------------------------
% SECTION 8: Augmentation and resizing
% -------------------------------------------------------------------------
inputSize = [224 224 3];

augmenter = imageDataAugmenter( ...
    'RandXReflection',  true, ...
    'RandYReflection',  false, ...
    'RandRotation',     [-20 20], ...
    'RandXTranslation', [-15 15], ...
    'RandYTranslation', [-15 15], ...
    'RandXShear',       [-5 5], ...
    'RandYShear',       [-5 5]);
augTrain = augmentedImageDatastore(inputSize, trainDS, ...
    'DataAugmentation',   augmenter, ...
    'ColorPreprocessing', 'gray2rgb');
augVal  = augmentedImageDatastore(inputSize, valDS,  'ColorPreprocessing', 'gray2rgb');
augTest = augmentedImageDatastore(inputSize, testDS, 'ColorPreprocessing', 'gray2rgb');

%% -------------------------------------------------------------------------
% SECTION 9: MobileNetV2 -- modify for output classes
% -------------------------------------------------------------------------
net    = mobilenetv2;
lgraph = layerGraph(net);

classNames = categories(combinedDS.Labels);
numClasses = numel(classNames);
fprintf('\nClasses (%d): ', numClasses);
disp(classNames');

newFC      = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
                 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newSoftmax = softmaxLayer('Name', 'new_softmax');
newOutput  = classificationLayer('Name', 'new_classoutput', 'Classes', classNames);

lgraph = replaceLayer(lgraph, 'Logits',                     newFC);
lgraph = replaceLayer(lgraph, 'Logits_softmax',             newSoftmax);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newOutput);

%% -------------------------------------------------------------------------
% SECTION 10: Class weights -- boost malignant, cap at 2.5
% -------------------------------------------------------------------------
trainCounts  = countcats(trainDS.Labels);
fprintf('\nTraining counts:\n');
disp(array2table(trainCounts', 'VariableNames', classNames'));

rawWeights   = max(trainCounts) ./ trainCounts;
classWeights = sqrt(rawWeights);

malIdx = strcmp(classNames, 'malignant');
if classWeights(malIdx) > 2.5
    classWeights(malIdx) = 2.5;
end

fprintf('Class weights:\n');
for i = 1:numClasses
    fprintf('  %s: %.3f\n', classNames{i}, classWeights(i));
end

newOutput = classificationLayer('Name', 'new_classoutput', ...
    'Classes', classNames, 'ClassWeights', classWeights);
lgraph = replaceLayer(lgraph, 'new_classoutput', newOutput);

%% -------------------------------------------------------------------------
% SECTION 11: Training options
% -------------------------------------------------------------------------
options = trainingOptions('adam', ...
    'InitialLearnRate',    1e-4, ...
    'MaxEpochs',           30, ...
    'MiniBatchSize',       16, ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      augVal, ...
    'ValidationFrequency', 20, ...
    'Verbose',             true, ...
    'Plots',               'training-progress', ...
    'ExecutionEnvironment','gpu');

%% -------------------------------------------------------------------------
% SECTION 12: Train
% -------------------------------------------------------------------------
fprintf('\nStarting mega training...\n');
tic;
[trainedNet, trainInfo] = trainNetwork(augTrain, lgraph, options);
elapsed = toc;
fprintf('Training complete: %.1f seconds (%.1f minutes)\n', elapsed, elapsed/60);

%% -------------------------------------------------------------------------
% SECTION 13: Test evaluation
% -------------------------------------------------------------------------
testPreds = classify(trainedNet, augTest);
testAcc   = mean(testPreds == testDS.Labels);
fprintf('\nTest accuracy: %.2f%%\n', testAcc * 100);

fprintf('\n%-12s  Precision  Recall    F1\n', 'Class');
fprintf('%s\n', repmat('-',1,45));
for i = 1:numClasses
    tp = sum(testPreds == classNames{i} & testDS.Labels == classNames{i});
    fp = sum(testPreds == classNames{i} & testDS.Labels ~= classNames{i});
    fn = sum(testPreds ~= classNames{i} & testDS.Labels == classNames{i});
    precision = tp / (tp + fp + eps);
    recall    = tp / (tp + fn + eps);
    f1        = 2 * precision * recall / (precision + recall + eps);
    fprintf('%-12s  %.1f%%      %.1f%%     %.4f\n', ...
        classNames{i}, precision*100, recall*100, f1);
end

%% -------------------------------------------------------------------------
% SECTION 14: Confusion matrix
% -------------------------------------------------------------------------
figure('Name', 'Mega Model - Test Set');
confusionchart(testDS.Labels, testPreds, ...
    'Title',         'Mega Combined Model Test Set', ...
    'RowSummary',    'row-normalized', ...
    'ColumnSummary', 'column-normalized');

saveas(gcf, 'C:\Users\rohit\Documents\MATLAB Code\Project_Figures\Phase4\confusion_matrix_mega.png');
fprintf('Confusion matrix saved\n');

%% -------------------------------------------------------------------------
% SECTION 15: Save and export ONNX
% -------------------------------------------------------------------------
save('C:\Users\rohit\Documents\MATLAB Code\trainedMobileNetV2_mega.mat', ...
     'trainedNet', 'trainInfo', 'classNames');
fprintf('Model saved: trainedMobileNetV2_mega.mat\n');

setenv('PATH', [getenv('PATH') ';C:\ProgramData\MATLAB\SupportPackages\R2024b\bin\win64']);
addpath(genpath('C:\ProgramData\MATLAB\SupportPackages\R2024b\toolbox\nnet\supportpackages\onnx'));
exportONNXNetwork(trainedNet, 'C:\Users\rohit\Documents\MATLAB Code\trainedMobileNetV2_mega.onnx');
fprintf('ONNX exported: trainedMobileNetV2_mega.onnx\n');
