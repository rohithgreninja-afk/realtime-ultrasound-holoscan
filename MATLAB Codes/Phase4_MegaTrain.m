% Phase4_MegaTrain.m
% Combined training: BUSI + BUS-UCLM + BUS-BRA + BrEaST + OASBUD-PNG
% Run with R2024b only.
%
% Prerequisite: Phase4_OASBUD_Export.m (writes OASBUD_PNG_PATH and its manifest).
% Data: see buildCombinedDatastore.m for the dataset environment variables.
% Output (RESULTS_DIR):
%   trainedMobileNetV2_mega.mat / .onnx   trained dlnetwork and ONNX export
%   cnn_split.mat                         train/val/test file keys, seed, OASBUD training patients
%   training_log.txt                      seed, split sizes per class, epochs, class weights
% Copy the three model/split files into MATLAB Codes/ to make them the committed model.
%
% The test partition is not touched here. Evaluate it once with Phase6_CNN_Evaluation.m.

clearvars; clc;

resultsDir = getenv('RESULTS_DIR');
if isempty(resultsDir)
    error('RESULTS_DIR environment variable not set. Set it with setenv(''RESULTS_DIR'', ''/path/to/results'').');
end
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% Optional: support package root, if the MobileNet-v2 / ONNX support packages
% are not already on the MATLAB path.
spRoot = getenv('MATLAB_SUPPORT_PACKAGES_ROOT');
if ~isempty(spRoot)
    addpath(genpath(fullfile(spRoot, 'toolbox', 'nnet', 'supportpackages')));
    setenv('PATH', [getenv('PATH') ';' fullfile(spRoot, 'bin', 'win64')]);
end

seed            = 42;
splitFractions  = [0.70 0.15 0.15];   % train / val / test, by patient group within each class
malignantWeight = 1.5;                % loss weight for the malignant class, others 1.0
maxEpochs       = 30;
miniBatchSize   = 16;

%% SECTION 1: Combined dataset
[combinedDS, meta] = buildCombinedDatastore();
classNames = categories(combinedDS.Labels);
numClasses = numel(classNames);

%% SECTION 2: One stratified, patient-grouped split 70/15/15
rng(seed);
[groups, firstIdx, gi] = unique(meta.Group, 'stable');
groupLabel = meta.Label(firstIdx);
part = zeros(numel(groups), 1);   % 1 train, 2 val, 3 test
for c = 1:numClasses
    g = find(groupLabel == classNames{c});
    g = g(randperm(numel(g)));
    nTr = round(splitFractions(1) * numel(g));
    nVa = round(splitFractions(2) * numel(g));
    part(g(1:nTr)) = 1;
    part(g(nTr+1:nTr+nVa)) = 2;
    part(g(nTr+nVa+1:end)) = 3;
end
imgPart = part(gi);

trainFiles = cellstr(meta.Key(imgPart == 1));
valFiles   = cellstr(meta.Key(imgPart == 2));
testFiles  = cellstr(meta.Key(imgPart == 3));
trainDS = subset(combinedDS, find(imgPart == 1));
valDS   = subset(combinedDS, find(imgPart == 2));

% OASBUD patients used for training images (everything else in OASBUD.mat is hold-out)
man = load(fullfile(getenv('OASBUD_PNG_PATH'), 'oasbud_export_manifest.mat'));
oasbudTrainIds = cellstr(man.exportedIds);
oasbudTrainIdx = man.exportedIdx;
oasbudSourceFile = man.sourceFile;

save(fullfile(resultsDir, 'cnn_split.mat'), 'trainFiles', 'valFiles', 'testFiles', ...
    'seed', 'splitFractions', 'classNames', 'oasbudTrainIds', 'oasbudTrainIdx', ...
    'oasbudSourceFile', '-v7');

splitCounts = zeros(3, numClasses);
for p = 1:3
    splitCounts(p, :) = countcats(meta.Label(imgPart == p))';
end
disp(array2table(splitCounts, 'VariableNames', classNames, 'RowNames', {'train','val','test'}));

%% SECTION 3: Augmentation and resizing
inputSize = [224 224 3];
augmenter = imageDataAugmenter( ...
    'RandXReflection',  true, ...
    'RandRotation',     [-20 20], ...
    'RandXTranslation', [-15 15], ...
    'RandYTranslation', [-15 15], ...
    'RandXShear',       [-5 5], ...
    'RandYShear',       [-5 5]);
augTrain = augmentedImageDatastore(inputSize, trainDS, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore(inputSize, valDS, 'ColorPreprocessing', 'gray2rgb');

%% SECTION 4: MobileNetV2 with a new classification head
net = imagePretrainedNetwork("mobilenetv2", NumClasses=numClasses);
net = setLearnRateFactor(net, "Logits", "Weights", 10);
net = setLearnRateFactor(net, "Logits", "Bias", 10);

%% SECTION 5: Class weights
classWeights = ones(1, numClasses);
classWeights(strcmp(classNames, 'malignant')) = malignantWeight;
lossFcn = @(Y, T) crossentropy(Y, T, classWeights, WeightsFormat="C");

%% SECTION 6: Train
options = trainingOptions('adam', ...
    'InitialLearnRate',    1e-4, ...
    'MaxEpochs',           maxEpochs, ...
    'MiniBatchSize',       miniBatchSize, ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      augVal, ...
    'ValidationFrequency', 20, ...
    'Metrics',             'accuracy', ...
    'OutputNetwork',       'last-iteration', ...
    'Verbose',             true, ...
    'Plots',               'none', ...
    'ExecutionEnvironment','gpu');

fprintf('\nStarting mega training...\n');
tic;
[trainedNet, trainInfo] = trainnet(augTrain, net, lossFcn, options);
elapsed = toc;
fprintf('Training complete: %.1f minutes\n', elapsed/60);

%% SECTION 7: Save, export ONNX, write log
save(fullfile(resultsDir, 'trainedMobileNetV2_mega.mat'), 'trainedNet', 'trainInfo', 'classNames');
exportONNXNetwork(trainedNet, fullfile(resultsDir, 'trainedMobileNetV2_mega.onnx'));

valHist = trainInfo.ValidationHistory;
fid = fopen(fullfile(resultsDir, 'training_log.txt'), 'w');
fprintf(fid, 'Phase4_MegaTrain.m training log (%s, MATLAB %s)\n\n', ...
    string(datetime('now'), 'yyyy-MM-dd HH:mm'), version('-release'));
fprintf(fid, 'Seed: %d\nSplit: %.2f / %.2f / %.2f (train/val/test), stratified by class, grouped by patient\n', ...
    seed, splitFractions);
fprintf(fid, 'Epochs: %d\nMini-batch size: %d\nOptimizer: adam, learn rate 1e-4\n', maxEpochs, miniBatchSize);
fprintf(fid, 'Class weights: %s\n', strjoin(string(classNames') + "=" + string(classWeights), ', '));
fprintf(fid, 'Output network: last iteration\nTraining time: %.1f minutes\n\n', elapsed/60);
fprintf(fid, '%-6s %8s %8s %8s\n', 'Split', classNames{:});
rows = {'train','val','test'};
for p = 1:3
    fprintf(fid, '%-6s %8d %8d %8d\n', rows{p}, splitCounts(p, :));
end
fprintf(fid, 'Total  %8d %8d %8d\n\n', sum(splitCounts, 1));
fprintf(fid, 'Images per source: %s\n', strjoin(compose('%s=%d', ...
    string(unique(meta.Source, 'stable')), groupcounts(categorical(meta.Source, unique(meta.Source, 'stable')))), ', '));
fprintf(fid, 'OASBUD training patients (%s): %d\n', oasbudSourceFile, numel(oasbudTrainIds));
fprintf(fid, 'Final validation accuracy: %.2f%%\n', valHist.Accuracy(end));
fclose(fid);
type(fullfile(resultsDir, 'training_log.txt'));
