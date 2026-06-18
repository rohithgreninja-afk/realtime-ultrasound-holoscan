% Phase6_CNN_Evaluation.m
% =========================================================
% Phase 6 - Comprehensive CNN Evaluation (Mega Model)
% =========================================================
% Generates: confusion matrix, per-class metrics, ROC curve,
%            precision-recall curve, latency table,
%            OASBUD authentic test results.
%
% DOES NOT RETRAIN. Loads trainedMobileNetV2_mega.mat only.
% Estimated run time: 10-15 min (classify 665 images, GPU).
% =========================================================
clearvars; close all; clc;

%% ===== PATHS =====
matlabCodeDir = 'C:\Users\rohit\Documents\MATLAB Code\';
matFilePath   = fullfile(matlabCodeDir, 'trainedMobileNetV2_mega.mat');
figDir        = fullfile(matlabCodeDir, 'Project_Figures', 'Phase6');
if ~exist(figDir,'dir'), mkdir(figDir); end

busiRoot   = ['C:\Users\rohit\OneDrive\Documents\MATLAB\Examples\R2026a\supportfiles\' ...
              'image\data\Dataset_BUSI\Dataset_BUSI_with_GT\'];
uclmRoot   = ['C:\Users\rohit\Downloads\Real Time Image Processing Project\' ...
              'BUS-UCLM Breast ultrasound lesion segmentation dataset\' ...
              'BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM\'];
busbraRoot = ['C:\Users\rohit\Downloads\Real Time Image Processing Project\' ...
              'BUSBRA\BUSBRA\'];
breastRoot = ['C:\Users\rohit\Downloads\Real Time Image Processing Project\' ...
              'BrEaST-Lesions_USG-images_and_masks-Dec-15-2023\' ...
              'BrEaST-Lesions_USG-images_and_masks\'];
breastXLSX = ['C:\Users\rohit\Downloads\Real Time Image Processing Project\' ...
              'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx'];
oasbud_png = 'C:\Users\rohit\Downloads\OASBUD_PNG\';
oasbud_mat = ['C:\Users\rohit\Downloads\Real Time Image Processing Project\' ...
              'OASBUD.mat'];

fprintf('=== Phase 6 CNN Evaluation ===\n');
fprintf('Figures -> %s\n\n', figDir);

%% ===== SECTION 1: LOAD MODEL =====
fprintf('[1/9] Loading mega model...\n');
S = load(matFilePath);
trainedNet = S.trainedNet;          % DAGNetwork
classNames = categorical(S.classNames);  % [benign, malignant, normal]
trainInfo  = S.trainInfo;           % training history
fprintf('  Network class:  %s\n', class(trainedNet));
fprintf('  Output classes: %s\n', strjoin(string(classNames), ', '));
fprintf('  Final val accuracy (from training): %.2f%%\n\n', trainInfo.FinalValidationAccuracy);

%% ===== SECTION 2: RECONSTRUCT COMBINED DATASTORE =====
fprintf('[2/9] Reconstructing combined imageDatastore...\n');

% --- BUSI (folder-based labels) ---
dsBUSI = imageDatastore(busiRoot, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'FileExtensions', '.png');
% Keep only non-mask images and relabel in one rebuild
keepIdx = ~contains(dsBUSI.Files, '_mask');
dsBUSI  = imageDatastore(dsBUSI.Files(keepIdx), ....
    'Labels', categorical(lower(string(dsBUSI.Labels(keepIdx)))));
fprintf('  BUSI:         %4d images\n', numel(dsBUSI.Files));

% --- BUS-UCLM (CSV labels, Doppler filtered) ---
info_uclm = readtable(fullfile(uclmRoot, 'INFO.csv'));
keepIdx   = strcmp(info_uclm.Doppler, 'No');
info_uclm = info_uclm(keepIdx, :);
uclmFiles = string(fullfile(uclmRoot, 'images', string(info_uclm.Image)));
existMask   = isfile(uclmFiles);
uclmLabels  = categorical(lower(string(info_uclm.Label(existMask))));
dsUCLM = imageDatastore(uclmFiles(existMask), 'Labels', uclmLabels);
fprintf('  BUS-UCLM:     %4d images\n', numel(dsUCLM.Files));

% --- BUS-BRA (CSV labels, dual-view -l/-r images) ---
busbra_csv  = readtable(fullfile(busbraRoot, 'bus_data.csv'));
imgDir_bra  = fullfile(busbraRoot, 'Images');
% Each CSV row is one image: ID = filename without .png
busbraFiles  = {};
busbraLabels = {};
for i = 1:height(busbra_csv)
    fname = fullfile(imgDir_bra, [char(busbra_csv.ID{i}) '.png']);
    lbl   = char(lower(busbra_csv.Pathology{i}));
    if isfile(fname) && ismember(lbl, {'benign','malignant'})
        busbraFiles{end+1}  = fname;
        busbraLabels{end+1} = lbl;
    end
end
dsBUSBRA = imageDatastore(busbraFiles', 'Labels', categorical(busbraLabels'));
fprintf('  BUS-BRA:      %4d images\n', numel(dsBUSBRA.Files));

% --- BrEaST (XLSX labels matched to image files) ---
tbl_breast = readtable(breastXLSX);
% Column names check -- print first time to verify
% disp(tbl_breast.Properties.VariableNames)
breastFiles  = {};
breastLabels = {};
allPNGs = dir(fullfile(breastRoot, '*.png'));
if isempty(allPNGs)
    allPNGs = dir(fullfile(breastRoot, '**', '*.png'));
end
for i = 1:height(tbl_breast)
    lbl = lower(strtrim(string(tbl_breast.Classification(i))));
    if ~ismember(lbl, {'benign','malignant','normal'}), continue; end
    % Try to find matching PNG by case identifier
    % Adjust 'Case' to the actual ID column name in your XLSX if needed
    % Image_filename column holds exact filename e.g. 'case001.png'
    imgFname = strtrim(string(tbl_breast.Image_filename(i)));
    % Skip mask files (contain '_tumor' or '_other')
    if contains(imgFname, '_tumor') || contains(imgFname, '_other'), continue; end
    matchPNG = fullfile(breastRoot, char(imgFname));
    if ~isfile(matchPNG), continue; end
    breastFiles{end+1}  = matchPNG;
    breastLabels{end+1} = char(lbl);
end
dsBrEaST = imageDatastore(breastFiles', 'Labels', categorical(breastLabels'));
fprintf('  BrEaST:       %4d images\n', numel(dsBrEaST.Files));

% --- OASBUD PNG (folder-based labels) ---
dsOASBUD_png = imageDatastore(oasbud_png, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'FileExtensions', '.png');
dsOASBUD_png.Labels = categorical(lower(string(dsOASBUD_png.Labels)));
fprintf('  OASBUD-PNG:   %4d images\n', numel(dsOASBUD_png.Files));

% --- Combine all five sources ---
allFiles  = [dsBUSI.Files; dsUCLM.Files; dsBUSBRA.Files; ...
             dsBrEaST.Files; dsOASBUD_png.Files];
allLabels = categorical([string(dsBUSI.Labels);  string(dsUCLM.Labels); ...
                         string(dsBUSBRA.Labels); string(dsBrEaST.Labels); ...
                         string(dsOASBUD_png.Labels)]);
dsCombined = imageDatastore(allFiles, 'Labels', allLabels);
fprintf('\n  Combined total: %d images\n', numel(dsCombined.Files));
tabulate(string(dsCombined.Labels));

%% ===== SECTION 3: SPLIT -- SAME SEED AS TRAINING =====
fprintf('\n[3/9] Splitting with rng(42) to recover test partition...\n');
rng(42);
[~, ~, dsTest] = splitEachLabel(dsCombined, 0.70, 0.15, 0.15, 'randomized');
testLabels_true = dsTest.Labels;
fprintf('  Test set: %d images\n', numel(dsTest.Files));
tabulate(string(testLabels_true));

%% ===== SECTION 4: CLASSIFY TEST SET (GPU) =====
fprintf('\n[4/9] Classifying test set on GPU...\n');
augTest = augmentedImageDatastore([224 224 3], dsTest, ...
    'ColorPreprocessing', 'gray2rgb');
tic;
[predLabels, scores] = classify(trainedNet, augTest, ...
    'MiniBatchSize', 32, 'ExecutionEnvironment', 'gpu');
t_classify_total = toc;
accuracy = mean(predLabels == testLabels_true);
fprintf('  Overall accuracy:  %.2f%%\n', 100*accuracy);
fprintf('  Total time:        %.1f s for %d images\n', ...
    t_classify_total, numel(dsTest.Files));
fprintf('  Avg per image:     %.2f ms\n\n', ...
    1000*t_classify_total/numel(dsTest.Files));

%% ===== SECTION 5: CONFUSION MATRIX =====
fprintf('[5/9] Generating confusion matrix...\n');
fig_cm = figure('Name','Confusion Matrix','Position',[50 50 720 620]);
cm = confusionchart(testLabels_true, predLabels, ...
    'Title', sprintf('Mega Model Test Set  (n=%d)   Accuracy: %.2f%%', ...
                     numel(dsTest.Files), 100*accuracy), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
cm.FontSize = 12;
cm.DiagonalColor  = [0.18 0.53 0.18];
cm.OffDiagonalColor = [0.85 0.18 0.18];
saveas(fig_cm, fullfile(figDir, 'confusion_matrix_test.png'));
fprintf('  Saved: confusion_matrix_test.png\n');

%% ===== SECTION 6: PER-CLASS METRICS =====
fprintf('\n[6/9] Per-class precision / recall / F1\n');
fprintf('%-12s  %9s  %9s  %9s\n', 'Class', 'Precision', 'Recall', 'F1-score');
fprintf('%s\n', repmat('-',1,46));
classOrder = {'benign','malignant','normal'};
metrics = struct();
for c = 1:numel(classOrder)
    cl = classOrder{c};
    if ~any(classNames == cl), continue; end
    TP = sum(predLabels == cl & testLabels_true == cl);
    FP = sum(predLabels == cl & testLabels_true ~= cl);
    FN = sum(predLabels ~= cl & testLabels_true == cl);
    prec   = TP ./ (TP + FP + eps);
    recall = TP ./ (TP + FN + eps);
    f1     = 2 .* prec .* recall ./ (prec + recall + eps);
    metrics.(cl) = struct('prec',prec,'recall',recall,'f1',f1, ...
                          'TP',TP,'FP',FP,'FN',FN);
    fprintf('%-12s  %8.2f%%  %8.2f%%  %9.4f\n', cl, 100*prec, 100*recall, f1);
end

%% ===== SECTION 7: ROC CURVE -- MALIGNANT VS REST =====
fprintf('\n[7/9] ROC and Precision-Recall curves...\n');
malIdx      = find(classNames == 'malignant');
malScores   = scores(:, malIdx);
trueIsMal   = (testLabels_true == 'malignant');
[rocX, rocY, ~, AUC_ROC] = perfcurve(trueIsMal, malScores, true);
[prX,  prY,  ~, AUC_PR]  = perfcurve(trueIsMal, malScores, true, ...
    'XCrit', 'reca', 'YCrit', 'prec');
fprintf('  Malignant detection AUC-ROC: %.4f\n', AUC_ROC);
fprintf('  Malignant detection AUC-PR:  %.4f\n', AUC_PR);

fig_roc = figure('Name','ROC Curve','Position',[100 100 560 520]);
plot(rocX, rocY, 'b-', 'LineWidth', 2.5); hold on;
plot([0 1],[0 1],'k--','LineWidth',1.2);
xlabel('False Positive Rate','FontSize',13);
ylabel('True Positive Rate','FontSize',13);
title(sprintf('ROC Curve -- Malignant Detection  (AUC = %.4f)', AUC_ROC), ...
    'FontSize',14);
legend(sprintf('Mega model  AUC=%.3f', AUC_ROC), ...
    'Random baseline','Location','SouthEast','FontSize',12);
grid on; set(gca,'FontSize',12);
saveas(fig_roc, fullfile(figDir, 'roc_malignant.png'));
fprintf('  Saved: roc_malignant.png\n');

fig_pr = figure('Name','PR Curve','Position',[700 100 560 520]);
baserate = sum(trueIsMal) / numel(trueIsMal);
plot(prX, prY, 'r-', 'LineWidth', 2.5); hold on;
yline(baserate, 'k--', 'LineWidth',1.2);
xlabel('Recall','FontSize',13);
ylabel('Precision','FontSize',13);
title(sprintf('Precision-Recall -- Malignant Detection  (AUC-PR = %.4f)', AUC_PR), ...
    'FontSize',14);
legend(sprintf('Mega model  AUC-PR=%.3f', AUC_PR), ...
    sprintf('Baseline (prevalence=%.2f)', baserate), ...
    'Location','SouthWest','FontSize',12);
grid on; set(gca,'FontSize',12);
saveas(fig_pr, fullfile(figDir, 'pr_malignant.png'));
fprintf('  Saved: pr_malignant.png\n');

%% ===== SECTION 8: LATENCY MEASUREMENT =====
fprintf('\n[8/9] Measuring single-image inference latency (GPU)...\n');
% Read one test image and resize to 224x224 RGB
img_raw = imread(dsTest.Files{1});
img_r   = imresize(img_raw, [224 224]);
if size(img_r,3) == 1
    img_r = repmat(img_r, [1 1 3]);
end
% Warm-up pass (first GPU call has JIT overhead)
for w = 1:3
    classify(trainedNet, img_r, 'ExecutionEnvironment', 'gpu');
end
% Timed runs
nRuns   = 50;
t_runs  = zeros(1, nRuns);
for r = 1:nRuns
    t0 = tic;
    classify(trainedNet, img_r, 'ExecutionEnvironment', 'gpu');
    t_runs(r) = toc(t0) * 1000;
end
fprintf('  Single-image MATLAB GPU inference (n=%d):\n', nRuns);
fprintf('    Mean: %.2f ms  |  Std: %.2f ms  |  Min: %.2f ms  |  Max: %.2f ms\n', ...
    mean(t_runs), std(t_runs), min(t_runs), max(t_runs));

fig_lat = figure('Name','Latency Distribution','Position',[100 680 560 380]);
histogram(t_runs, 20, 'FaceColor',[0.12 0.47 0.71]);
xlabel('Inference time (ms)','FontSize',12);
ylabel('Count','FontSize',12);
title(sprintf('MATLAB GPU Inference Latency  (n=%d runs)\nMean=%.2f ms  Std=%.2f ms', ...
    nRuns, mean(t_runs), std(t_runs)),'FontSize',13);
grid on; set(gca,'FontSize',11);
saveas(fig_lat, fullfile(figDir, 'latency_distribution.png'));
fprintf('  Saved: latency_distribution.png\n');

%% ===== SECTION 9: OASBUD AUTHENTIC TEST (60 held-out patients) =====
fprintf('\n[9/9] OASBUD authentic test -- held-out patients...\n');
data    = load(oasbud_mat);
nTotal  = numel(data.data);
fprintf('  Total OASBUD patients: %d\n', nTotal);

% Identify all malignant (label=0) and benign (label=1) patients
allLabels_oasbud = arrayfun(@(x) x.class, data.data);
malPat = find(allLabels_oasbud == 0);
benPat = find(allLabels_oasbud == 1);
fprintf('  Malignant patients: %d  |  Benign patients: %d\n', ...
    numel(malPat), numel(benPat));

% Training used first 20 of each -- held-out is the rest
testMalPat = malPat(21:end);
testBenPat = benPat(21:end);
testPatients = [testMalPat(:); testBenPat(:)];
fprintf('  Held-out: %d malignant + %d benign = %d patients\n', ...
    numel(testMalPat), numel(testBenPat), numel(testPatients));

predOASBUD = strings(numel(testPatients), 1);
trueOASBUD = strings(numel(testPatients), 1);
lat_oasbud = zeros(numel(testPatients), 1);

for i = 1:numel(testPatients)
    p   = testPatients(i);
    rf  = double(data.data(p).rf1);   % [depth x 510]
    lbl = data.data(p).class;         % 0=malignant 1=benign
    trueOASBUD(i) = ternary(lbl == 0, 'malignant', 'benign');
    % A-line beamforming: per-column Hilbert envelope detection
    env   = abs(hilbert(rf));
    % Power law compression gamma=0.3
    env   = env ./ max(env(:));
    bmode = env .^ 0.3;
    % Resize to 224x224 and convert to uint8 RGB
    img_bmode = imresize(bmode, [224 224]);
    img_rgb   = repmat(uint8(img_bmode * 255), [1 1 3]);
    % Classify
    t0 = tic;
    pred = classify(trainedNet, img_rgb, 'ExecutionEnvironment', 'gpu');
    lat_oasbud(i)  = toc(t0) * 1000;
    predOASBUD(i)  = string(pred);
    if mod(i, 10) == 0
        fprintf('  %d/%d\n', i, numel(testPatients));
    end
end

% Convert to categorical for confusionchart
trueOASBUD_cat = categorical(trueOASBUD);
predOASBUD_cat = categorical(predOASBUD);

% Metrics
oasbud_acc  = mean(predOASBUD_cat == trueOASBUD_cat);
oasbud_mal_tp = sum(predOASBUD == 'malignant' & trueOASBUD == 'malignant');
oasbud_mal_fn = sum(predOASBUD ~= 'malignant' & trueOASBUD == 'malignant');
oasbud_mal_recall = oasbud_mal_tp / (oasbud_mal_tp + oasbud_mal_fn + eps);
fprintf('\n  OASBUD authentic test accuracy:    %.1f%% (%d/%d)\n', ...
    100*oasbud_acc, sum(predOASBUD_cat==trueOASBUD_cat), numel(testPatients));
fprintf('  Malignant recall (OASBUD authentic): %.1f%%\n', 100*oasbud_mal_recall);
fprintf('  Avg latency per patient:              %.2f ms\n', mean(lat_oasbud));

fig_oasbud = figure('Name','OASBUD Authentic','Position',[50 50 620 540]);
cm2 = confusionchart(trueOASBUD_cat, predOASBUD_cat, ...
    'Title', sprintf('OASBUD Authentic Test (n=%d held-out patients)   Acc: %.1f%%', ...
                      numel(testPatients), 100*oasbud_acc), ...
    'RowSummary', 'row-normalized');
cm2.FontSize = 13;
cm2.DiagonalColor   = [0.18 0.53 0.18];
cm2.OffDiagonalColor = [0.85 0.18 0.18];
saveas(fig_oasbud, fullfile(figDir, 'confusion_oasbud_authentic.png'));
fprintf('  Saved: confusion_oasbud_authentic.png\n');

%% ===== SUMMARY =====
fprintf('\n=======================================\n');
fprintf('PHASE 6 MATLAB EVALUATION SUMMARY\n');
fprintf('=======================================\n');
fprintf('CNN Test Set (n=%d images)\n', numel(dsTest.Files));
fprintf('  Overall accuracy:  %.2f%%\n', 100*accuracy);
fprintf('  Malignant AUC-ROC: %.4f\n', AUC_ROC);
fprintf('  Malignant AUC-PR:  %.4f\n', AUC_PR);
fprintf('  MATLAB GPU latency (mean): %.2f ms/image\n', mean(t_runs));
fprintf('\nOASBUD Authentic Test (n=%d patients)\n', numel(testPatients));
fprintf('  Accuracy:          %.1f%%\n', 100*oasbud_acc);
fprintf('  Malignant recall:  %.1f%%\n', 100*oasbud_mal_recall);
fprintf('\nFigures saved to:\n  %s\n', figDir);
fprintf('=======================================\n');

%% helper
function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end