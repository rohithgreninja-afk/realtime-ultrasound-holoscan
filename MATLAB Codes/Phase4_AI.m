% =========================================================
% Phase4_AI.m
% AI-Based Classification using BUSI Dataset
% =========================================================

innerPath = 'C:\Users\rohit\OneDrive\Documents\MATLAB\Examples\R2026a\supportfiles\image\data\Dataset_BUSI\Dataset_BUSI_with_GT';

% Check dataset is accessible
if ~exist(innerPath, 'dir')
    error('BUSI dataset not found at expected path. Check the path.');
end

% Count images per class
classes = {'benign', 'malignant', 'normal'};
fprintf('BUSI Dataset Summary:\n');
fprintf('%-12s  %s\n', 'Class', 'Images');
fprintf('%s\n', repmat('-', 1, 25));
total = 0;
for i = 1:3
    files = dir(fullfile(innerPath, classes{i}, '*.png'));
    imgs  = files(~contains({files.name}, '_mask'));
    fprintf('%-12s  %d\n', classes{i}, length(imgs));
    total = total + length(imgs);
end
fprintf('%-12s  %d\n', 'TOTAL', total);


% ── Part 1: Load Dataset ──────────────────────────────────
% Build file list manually excluding mask files
allFiles = {};
allLabels = {};

classes = {'benign', 'malignant', 'normal'};
for i = 1:3
    files = dir(fullfile(innerPath, classes{i}, '*.png'));
    for j = 1:length(files)
        if ~contains(files(j).name, '_mask')
            allFiles{end+1} = fullfile(innerPath, classes{i}, files(j).name);
            allLabels{end+1} = classes{i};
        end
    end
end

% Create imageDatastore from filtered file list
imds = imageDatastore(allFiles, 'Labels', categorical(allLabels));

fprintf('\nDataset loaded:\n');
fprintf('  Total images: %d\n', numel(imds.Files));
fprintf('  Classes: %s\n', strjoin(string(unique(imds.Labels)), ', '));

labelCount = countEachLabel(imds);
disp(labelCount);

% ── Split into train / validation / test ──────────────────
[imdsTrain, imdsValTest] = splitEachLabel(imds, 0.70, 'randomized');
[imdsVal,   imdsTest]    = splitEachLabel(imdsValTest, 0.50, 'randomized');

fprintf('\nDataset split:\n');
fprintf('  Training:   %d images\n', numel(imdsTrain.Files));
fprintf('  Validation: %d images\n', numel(imdsVal.Files));
fprintf('  Test:       %d images\n', numel(imdsTest.Files));

fprintf('\nTraining set class distribution:\n');
disp(countEachLabel(imdsTrain));

% ── Part 2: Preprocessing and Augmentation ───────────────
% All images must be resized to 224x224 for MobileNetV2
% Images are RGB uint8 — no conversion needed

inputSize = [224 224 3];

% Augmentation for training set only
% Helps prevent overfitting on small dataset
augmenter = imageDataAugmenter( ...
    'RandXReflection',  true, ...
    'RandRotation',     [-10 10], ...
    'RandXScale',       [0.9 1.1], ...
    'RandYScale',       [0.9 1.1], ...
    'RandXShear',       [-5 5], ...
    'RandYShear',       [-5 5]);

% Augmented datastore for training
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

% Validation and test — resize only, no augmentation
augimdsVal  = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

fprintf('Preprocessing configured:\n');
fprintf('  Input size: %dx%dx%d\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('  Training augmentation: flips, rotation +/-10, scale 0.9-1.1\n');
fprintf('  Training samples: %d\n', augimdsTrain.NumObservations);
fprintf('  Validation samples: %d\n', augimdsVal.NumObservations);
fprintf('  Test samples: %d\n', augimdsTest.NumObservations);

% ── Compute class weights for imbalance ───────────────────
numClasses   = 3;
classCounts  = [306, 147, 93];   % benign, malignant, normal
totalSamples = sum(classCounts);
classWeights = totalSamples ./ (numClasses .* classCounts);
classWeights = classWeights / min(classWeights);  % normalise so min = 1

fprintf('\nClass weights (to handle imbalance):\n');
fprintf('  benign:    %.3f\n', classWeights(1));
fprintf('  malignant: %.3f\n', classWeights(2));
fprintf('  normal:    %.3f\n', classWeights(3));


% ── Part 3: Transfer Learning with MobileNetV2 ────────────
% Load pretrained MobileNetV2
net = mobilenetv2;

fprintf('MobileNetV2 loaded:\n');
fprintf('  Input size: %dx%dx%d\n', net.Layers(1).InputSize);
fprintf('  Total layers: %d\n', numel(net.Layers));
fprintf('  Original output classes: %d\n', net.Layers(end).OutputSize);

% Extract the layer graph and modify for 3-class output
lgraph = layerGraph(net);

% Find the final classification layers to replace
% MobileNetV2 ends with: Logits -> Logits_softmax -> ClassificationOutput
% We replace the last 3 layers with our own

newFCLayer = fullyConnectedLayer(3, ...
    'Name',            'new_fc', ...
    'WeightLearnRateFactor',  10, ...
    'BiasLearnRateFactor',    10);

newSoftmax = softmaxLayer('Name', 'new_softmax');

newOutput  = classificationLayer( ...
    'Name',         'new_output', ...
    'Classes',      categories(imdsTrain.Labels), ...
    'ClassWeights', classWeights);

% Replace layers
lgraph = replaceLayer(lgraph, 'Logits',               newFCLayer);
lgraph = replaceLayer(lgraph, 'Logits_softmax',        newSoftmax);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newOutput);

fprintf('\nNetwork modified for transfer learning:\n');
fprintf('  New FC layer: 3 outputs, LR factor x10\n');
fprintf('  Class weights applied: [%.2f  %.2f  %.2f]\n', classWeights);
fprintf('  Frozen layers: all except new_fc, new_softmax, new_output\n');

% ── Part 4: Training Options ──────────────────────────────
opts = trainingOptions('adam', ...
    'InitialLearnRate',    1e-4, ...
    'MaxEpochs',           30, ...
    'MiniBatchSize',       16, ...
    'ValidationData',      augimdsVal, ...
    'ValidationFrequency', 10, ...
    'Shuffle',             'every-epoch', ...
    'ExecutionEnvironment','gpu', ...
    'Plots',               'training-progress', ...
    'Verbose',             true, ...
    'OutputNetwork',       'best-validation-loss');

fprintf('Training options configured:\n');
fprintf('  Optimizer:       Adam\n');
fprintf('  Learning rate:   1e-4\n');
fprintf('  Max epochs:      30\n');
fprintf('  Mini batch size: 16\n');
fprintf('  Execution:       GPU\n');
fprintf('  Output:          Best validation loss model\n');

% ── Train the network ─────────────────────────────────────
fprintf('\nStarting training...\n');
fprintf('Watch the Training Progress window for loss and accuracy curves.\n\n');

[trainedNet, trainInfo] = trainNetwork(augimdsTrain, lgraph, opts);

fprintf('\nTraining complete.\n');
fprintf('Best validation accuracy: %.2f%%\n', max(trainInfo.ValidationAccuracy));
fprintf('Final training accuracy:  %.2f%%\n', trainInfo.TrainingAccuracy(end));

% ── Part 5: Evaluate on Test Set ─────────────────────────
fprintf('Evaluating on test set...\n');

% Run predictions on test set
[predLabels, scores] = classify(trainedNet, augimdsTest);
trueLabels = imdsTest.Labels;

% Overall accuracy
accuracy = mean(predLabels == trueLabels) * 100;
fprintf('\nTest Set Results:\n');
fprintf('  Overall Accuracy: %.2f%%\n', accuracy);

% Confusion matrix
figure;
cm = confusionchart(trueLabels, predLabels, ...
    'Title', 'Phase 4 — Test Set Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
cm.FontSize = 12;

% Per-class metrics
classes = {'benign', 'malignant', 'normal'};
fprintf('\nPer-Class Metrics:\n');
fprintf('%-12s  %8s  %8s  %8s\n', 'Class', 'Precision', 'Recall', 'F1');
fprintf('%s\n', repmat('-', 1, 44));

for i = 1:3
    classLabel = categorical(classes(i));
    TP = sum(predLabels == classLabel & trueLabels == classLabel);
    FP = sum(predLabels == classLabel & trueLabels ~= classLabel);
    FN = sum(predLabels ~= classLabel & trueLabels == classLabel);

    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1        = 2 * precision * recall / (precision + recall + eps);

    fprintf('%-12s  %8.4f  %8.4f  %8.4f\n', classes{i}, precision, recall, f1);
end

% Save the trained network
save('trainedMobileNetV2.mat', 'trainedNet', 'trainInfo');
fprintf('\nTrained network saved to trainedMobileNetV2.mat\n');

% ── Save all Phase 4 figures ──────────────────────────────
outputFolder = 'C:\Users\rohit\Documents\MATLAB Code\Project_Figures\Phase4';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

% Save confusion matrix
saveas(gcf, fullfile(outputFolder, 'confusion_matrix.png'));
fprintf('Confusion matrix saved.\n');