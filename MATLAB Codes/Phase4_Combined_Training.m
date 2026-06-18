% Phase4_Combined_Training.m
% Retrain MobileNetV2 on BUSI + BUS-UCLM (combined dataset)
% Malignant weight pushed to 3.0
% Run with R2024b only

clearvars; clc;

addpath(genpath('C:\ProgramData\MATLAB\SupportPackages\R2024b\toolbox\nnet\supportpackages\mobilenetv2'));

%% -------------------------------------------------------------------------
% SECTION 1: Load BUSI dataset
% -------------------------------------------------------------------------
busiPath = fullfile('C:\Users\rohit\OneDrive\Documents\MATLAB\Examples\R2026a', ...
    'supportfiles\image\data\Dataset_BUSI\Dataset_BUSI_with_GT');

busiDS = imageDatastore(busiPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource',       'foldernames', ...
    'FileExtensions',    '.png');

isMask     = contains(busiDS.Files, '_mask');
cleanFiles = busiDS.Files(~isMask);

busiDS = imageDatastore(cleanFiles, ...
    'LabelSource',       'foldernames', ...
    'IncludeSubfolders', true);

busiDS.Labels = categorical(lower(string(busiDS.Labels)));

fprintf('BUSI images loaded: %d\n', numel(busiDS.Files));
disp(countcats(busiDS.Labels));

%% -------------------------------------------------------------------------
% SECTION 2: Load BUS-UCLM dataset
% -------------------------------------------------------------------------
uclmRoot = fullfile('C:\Users\rohit\Downloads\Real Time Image Processing Project', ...
    'BUS-UCLM Breast ultrasound lesion segmentation dataset', ...
    'BUS-UCLM Breast ultrasound lesion segmentation dataset', ...
    'BUS-UCLM');

uclmImagesPath = fullfile(uclmRoot, 'images');
infoFile       = fullfile(uclmRoot, 'INFO.csv');

info     = readtable(infoFile);
cleanIdx = strcmp(info.Doppler, 'No');
info     = info(cleanIdx, :);

fprintf('BUS-UCLM images after Doppler filter: %d\n', height(info));

filenames = info.Image;
labelStrs = lower(info.Label);
fullPaths = fullfile(uclmImagesPath, filenames);

missing = ~isfile(fullPaths);
if any(missing)
    fprintf('WARNING: %d files not found, skipping\n', sum(missing));
    fullPaths(missing) = [];
    labelStrs(missing) = [];
end

uclmLabels = categorical(labelStrs);
uclmDS     = imageDatastore(fullPaths, 'Labels', uclmLabels);

fprintf('BUS-UCLM images loaded: %d\n', numel(uclmDS.Files));
disp(countcats(uclmDS.Labels));

%% -------------------------------------------------------------------------
% SECTION 3: Combine
% -------------------------------------------------------------------------
allFiles  = [busiDS.Files;  uclmDS.Files];
allLabels = [busiDS.Labels; uclmDS.Labels];

combinedDS = imageDatastore(allFiles, 'Labels', allLabels);

fprintf('\nCombined dataset total: %d images\n', numel(combinedDS.Files));
disp(countcats(combinedDS.Labels));

%% -------------------------------------------------------------------------
% SECTION 4: Train / Val / Test split  70 / 15 / 15
% -------------------------------------------------------------------------
rng(42);

[trainDS, tempDS] = splitEachLabel(combinedDS, 0.70, 'randomized');
[valDS,   testDS] = splitEachLabel(tempDS,     0.50, 'randomized');

fprintf('Train: %d   Val: %d   Test: %d\n', ...
    numel(trainDS.Files), numel(valDS.Files), numel(testDS.Files));

%% -------------------------------------------------------------------------
% SECTION 5: Augmentation and resizing
% -------------------------------------------------------------------------
inputSize = [224 224 3];

augTrain = augmentedImageDatastore(inputSize, trainDS, ...
    'DataAugmentation', imageDataAugmenter( ...
        'RandXReflection',  true, ...
        'RandYReflection',  false, ...
        'RandRotation',     [-15 15], ...
        'RandXTranslation', [-10 10], ...
        'RandYTranslation', [-10 10]));

augVal  = augmentedImageDatastore(inputSize, valDS);
augTest = augmentedImageDatastore(inputSize, testDS);

%% -------------------------------------------------------------------------
% SECTION 6: Load MobileNetV2 and modify for 3-class output
% -------------------------------------------------------------------------
net    = mobilenetv2;
lgraph = layerGraph(net);

classNames = categories(combinedDS.Labels);
numClasses = numel(classNames);

newFC      = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
                 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newSoftmax = softmaxLayer('Name', 'new_softmax');
newOutput  = classificationLayer('Name', 'new_classoutput', 'Classes', classNames);

lgraph = replaceLayer(lgraph, 'Logits',                     newFC);
lgraph = replaceLayer(lgraph, 'Logits_softmax',             newSoftmax);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newOutput);

%% -------------------------------------------------------------------------
% SECTION 7: Class weights -- malignant capped at 3.0
% -------------------------------------------------------------------------
trainCounts  = countcats(trainDS.Labels);
fprintf('\nTraining class counts:\n');
disp(array2table(trainCounts', 'VariableNames', classNames'));

rawWeights   = max(trainCounts) ./ trainCounts;
classWeights = sqrt(rawWeights);

malIdx = strcmp(classNames, 'malignant');
if classWeights(malIdx) > 3.0
    classWeights(malIdx) = 3.0;
end

fprintf('Class weights: benign=%.3f  malignant=%.3f  normal=%.3f\n', ...
    classWeights(1), classWeights(2), classWeights(3));

newOutput = classificationLayer('Name', 'new_classoutput', ...
    'Classes', classNames, 'ClassWeights', classWeights);
lgraph = replaceLayer(lgraph, 'new_classoutput', newOutput);

%% -------------------------------------------------------------------------
% SECTION 8: Training options
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
% SECTION 9: Train
% -------------------------------------------------------------------------
fprintf('\nStarting training...\n');
tic;
[trainedNet, trainInfo] = trainNetwork(augTrain, lgraph, options);
elapsed = toc;
fprintf('Training complete: %.1f seconds (%.1f minutes)\n', elapsed, elapsed/60);

%% -------------------------------------------------------------------------
% SECTION 10: Test set evaluation
% -------------------------------------------------------------------------
testPreds = classify(trainedNet, augTest);
testAcc   = mean(testPreds == testDS.Labels);
fprintf('\nTest accuracy: %.2f%%\n', testAcc * 100);

fprintf('\n%-12s  Precision  Recall    F1\n', 'Class');
fprintf('%s\n', repmat('-',1,45));
for i = 1:numel(classNames)
    tp = sum(testPreds == classNames{i} & testDS.Labels == classNames{i});
    fp = sum(testPreds == classNames{i} & testDS.Labels ~= classNames{i});
    fn = sum(testPreds ~= classNames{i} & testDS.Labels == classNames{i});
    precision = tp / (tp + fp);
    recall    = tp / (tp + fn);
    f1        = 2 * precision * recall / (precision + recall);
    fprintf('%-12s  %.1f%%      %.1f%%     %.4f\n', ...
        classNames{i}, precision*100, recall*100, f1);
end

%% -------------------------------------------------------------------------
% SECTION 11: Confusion matrix
% -------------------------------------------------------------------------
figure('Name', 'Combined Model - Test Set');
confusionchart(testDS.Labels, testPreds, ...
    'Title',         'Combined Model Test Set -- Malignant Weight 3.0', ...
    'RowSummary',    'row-normalized', ...
    'ColumnSummary', 'column-normalized');

saveas(gcf, 'C:\Users\rohit\Documents\MATLAB Code\Project_Figures\Phase4\confusion_matrix_w3.png');
fprintf('Confusion matrix saved\n');

%% -------------------------------------------------------------------------
% SECTION 12: Save and export ONNX
% -------------------------------------------------------------------------
save('C:\Users\rohit\Documents\MATLAB Code\trainedMobileNetV2_combined.mat', ...
     'trainedNet', 'trainInfo', 'classNames');
fprintf('Model saved: trainedMobileNetV2_combined.mat\n');

setenv('PATH', [getenv('PATH') ';C:\ProgramData\MATLAB\SupportPackages\R2024b\bin\win64']);
addpath(genpath('C:\ProgramData\MATLAB\SupportPackages\R2024b\toolbox\nnet\supportpackages\onnx'));
exportONNXNetwork(trainedNet, 'C:\Users\rohit\Documents\MATLAB Code\trainedMobileNetV2_combined.onnx');
fprintf('ONNX exported: trainedMobileNetV2_combined.onnx\n');