% Phase6_CNN_Evaluation.m
% =========================================================
% Phase 6 - CNN Evaluation (Mega Model) on the saved hold-out split
% =========================================================
% Loads the committed model (trainedMobileNetV2_mega.mat) and the split
% file written by Phase4_MegaTrain.m (cnn_split.mat). Never re-splits.
%
% Sections 2-8 (combined test set) need the dataset environment variables
% listed in buildCombinedDatastore.m; if any is missing they are skipped
% with a warning. Section 9 (OASBUD hold-out patients) runs on the bundled
% sample with only the committed model and split file.
%
% Output (RESULTS_DIR): cnn_eval.mat, cnn_eval.md, oasbud_holdout_eval.mat,
%                       figures/Phase6/*.png
% =========================================================
clearvars; close all; clc;

%% ===== PATHS =====
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);

resultsDir = getenv('RESULTS_DIR');
if isempty(resultsDir)
    error('RESULTS_DIR environment variable not set. Set it with setenv(''RESULTS_DIR'', ''/path/to/results'').');
end
figDir = fullfile(resultsDir, 'figures', 'Phase6');
if ~exist(figDir, 'dir'), mkdir(figDir); end

oasbud_mat = getenv('OASBUD_PATH');
if isempty(oasbud_mat)
    oasbud_mat = fullfile(repoRoot, 'data', 'sample', 'OASBUD_sample.mat');
    fprintf('OASBUD_PATH not set, using bundled sample: %s\n', oasbud_mat);
end

%% ===== SECTION 1: LOAD MODEL AND SPLIT =====
fprintf('[1/9] Loading mega model and split file...\n');
S = load(fullfile(scriptDir, 'trainedMobileNetV2_mega.mat'));
trainedNet = S.trainedNet;
classNames = string(S.classNames(:))';   % [benign malignant normal]
split = load(fullfile(scriptDir, 'cnn_split.mat'));
fprintf('  Network class:  %s\n', class(trainedNet));
fprintf('  Output classes: %s\n', strjoin(classNames, ', '));
fprintf('  Split seed: %d | train %d | val %d | test %d images\n', split.seed, ...
    numel(split.trainFiles), numel(split.valFiles), numel(split.testFiles));

%% ===== SECTIONS 2-8: COMBINED TEST SET =====
dsVars = {'BUSI_PATH','BUS_UCLM_PATH','BUSBRA_PATH','BREAST_IMG_PATH','BREAST_XLSX_PATH','OASBUD_PNG_PATH'};
missingVars = dsVars(cellfun(@(v) isempty(getenv(v)), dsVars));
runTestSet = isempty(missingVars);
if ~runTestSet
    warning('Skipping Sections 2-8 (combined test set): %s not set.', strjoin(missingVars, ', '));
end

if runTestSet
    % --- SECTION 2: rebuild datastore and select the saved test files ---
    fprintf('\n[2/9] Building combined datastore and selecting saved test files...\n');
    [dsCombined, meta] = buildCombinedDatastore();
    [found, loc] = ismember(split.testFiles, cellstr(meta.Key));
    if ~all(found)
        error('%d test files from cnn_split.mat are missing from the datasets, e.g. %s', ...
            sum(~found), split.testFiles{find(~found, 1)});
    end
    overlap = intersect(split.testFiles, [split.trainFiles; split.valFiles]);
    assert(isempty(overlap), 'Split file has %d files in both test and train/val.', numel(overlap));
    dsTest = subset(dsCombined, loc);
    testLabels_true = dsTest.Labels;
    fprintf('  Test set: %d images\n', numel(dsTest.Files));
    tabulate(string(testLabels_true));

    % --- SECTION 3/4: classify test set (GPU) ---
    fprintf('\n[3-4/9] Classifying test set...\n');
    augTest = augmentedImageDatastore([224 224 3], dsTest, 'ColorPreprocessing', 'gray2rgb');
    tic;
    scores = minibatchpredict(trainedNet, augTest, 'MiniBatchSize', 32);
    t_classify_total = toc;
    predLabels = scores2label(scores, classNames);
    testLabels_true = categorical(testLabels_true, classNames);
    accuracy = mean(predLabels == testLabels_true);
    fprintf('  Overall accuracy:  %.2f%%\n', 100*accuracy);
    fprintf('  Avg per image:     %.2f ms (batched)\n', 1000*t_classify_total/numel(dsTest.Files));

    % --- SECTION 5: confusion matrix ---
    fprintf('[5/9] Confusion matrix...\n');
    confMat = confusionmat(testLabels_true, predLabels, 'Order', categorical(classNames));
    fig_cm = figure('Name','Confusion Matrix','Position',[50 50 720 620]);
    cm = confusionchart(testLabels_true, predLabels, ...
        'Title', sprintf('Mega Model Test Set  (n=%d)   Accuracy: %.2f%%', ...
                         numel(dsTest.Files), 100*accuracy), ...
        'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
    cm.FontSize = 12;
    saveas(fig_cm, fullfile(figDir, 'confusion_matrix_test.png'));

    % --- SECTION 6: per-class metrics ---
    fprintf('\n[6/9] Per-class precision / recall / F1\n');
    fprintf('%-12s  %9s  %9s  %9s  %6s\n', 'Class', 'Precision', 'Recall', 'F1-score', 'n');
    perClass = table('Size', [numel(classNames) 5], ...
        'VariableTypes', {'string','double','double','double','double'}, ...
        'VariableNames', {'Class','Precision','Recall','F1','Support'});
    for c = 1:numel(classNames)
        TP = confMat(c,c); FP = sum(confMat(:,c)) - TP; FN = sum(confMat(c,:)) - TP;
        prec = TP / (TP + FP); rec = TP / (TP + FN); f1 = 2*prec*rec / (prec + rec);
        perClass(c,:) = {classNames(c), prec, rec, f1, TP + FN};
        fprintf('%-12s  %8.2f%%  %8.2f%%  %9.4f  %6d\n', classNames(c), 100*prec, 100*rec, f1, TP + FN);
    end

    % --- SECTION 7: ROC and PR, malignant vs rest ---
    fprintf('\n[7/9] ROC and Precision-Recall curves...\n');
    malScores = scores(:, classNames == "malignant");
    trueIsMal = (testLabels_true == 'malignant');
    [rocX, rocY, ~, AUC_ROC] = perfcurve(trueIsMal, malScores, true);
    [prX,  prY,  ~, AUC_PR]  = perfcurve(trueIsMal, malScores, true, 'XCrit', 'reca', 'YCrit', 'prec');
    fprintf('  Malignant AUC-ROC: %.4f\n  Malignant AUC-PR:  %.4f\n', AUC_ROC, AUC_PR);

    fig_roc = figure('Name','ROC Curve','Position',[100 100 560 520]);
    plot(rocX, rocY, 'b-', 'LineWidth', 2.5); hold on; plot([0 1],[0 1],'k--');
    xlabel('False Positive Rate'); ylabel('True Positive Rate'); grid on;
    title(sprintf('ROC, Malignant vs Rest (AUC = %.4f)', AUC_ROC));
    saveas(fig_roc, fullfile(figDir, 'roc_malignant.png'));

    fig_pr = figure('Name','PR Curve','Position',[700 100 560 520]);
    plot(prX, prY, 'r-', 'LineWidth', 2.5); hold on; yline(mean(trueIsMal), 'k--');
    xlabel('Recall'); ylabel('Precision'); grid on;
    title(sprintf('Precision-Recall, Malignant vs Rest (AUC-PR = %.4f)', AUC_PR));
    saveas(fig_pr, fullfile(figDir, 'pr_malignant.png'));

    % --- SECTION 8: single-image latency ---
    fprintf('\n[8/9] Single-image inference latency...\n');
    img_r = imresize(imread(dsTest.Files{1}), [224 224]);
    if size(img_r,3) == 1, img_r = repmat(img_r, [1 1 3]); end
    % predict on a GPU dlarray: the per-call overhead of minibatchpredict would dominate
    X = gpuArray(dlarray(single(img_r), 'SSCB'));
    gpu = gpuDevice;
    for w = 1:5, predict(trainedNet, X); end
    wait(gpu);
    nRuns = 50; t_runs = zeros(1, nRuns);
    for r = 1:nRuns
        t0 = tic; Y = predict(trainedNet, X); wait(gpu); t_runs(r) = toc(t0) * 1000; %#ok<NASGU>
    end
    fprintf('  Mean: %.2f ms | Std: %.2f ms (n=%d)\n', mean(t_runs), std(t_runs), nRuns);

    % --- Save ---
    testFiles = split.testFiles; %#ok<NASGU>
    save(fullfile(resultsDir, 'cnn_eval.mat'), 'accuracy', 'perClass', 'AUC_ROC', 'AUC_PR', ...
        'confMat', 'classNames', 'scores', 'predLabels', 'testLabels_true', 'testFiles', 't_runs');

    fid = fopen(fullfile(resultsDir, 'cnn_eval.md'), 'w');
    fprintf(fid, '# CNN hold-out evaluation\n\nProduced by `MATLAB Codes/Phase6_CNN_Evaluation.m` on the test partition stored in `MATLAB Codes/cnn_split.mat` (seed %d, %d images). The test images were never used for training or validation.\n\n', ...
        split.seed, numel(testLabels_true));
    fprintf(fid, '| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n');
    for c = 1:height(perClass)
        fprintf(fid, '| %s | %.2f%% | %.2f%% | %.4f | %d |\n', perClass.Class(c), ...
            100*perClass.Precision(c), 100*perClass.Recall(c), perClass.F1(c), perClass.Support(c));
    end
    fprintf(fid, '\nOverall accuracy: %.2f%% (%d/%d)\n\n', 100*accuracy, sum(predLabels == testLabels_true), numel(testLabels_true));
    fprintf(fid, 'Malignant vs rest: AUC-ROC %.4f, AUC-PR %.4f\n\n', AUC_ROC, AUC_PR);
    fprintf(fid, 'Confusion matrix (rows true, columns predicted, order %s):\n\n', strjoin(classNames, ', '));
    fprintf(fid, '| true \\ pred | %s |\n|---|---|---|---|\n', strjoin(classNames, ' | '));
    for c = 1:numel(classNames)
        fprintf(fid, '| %s | %d | %d | %d |\n', classNames(c), confMat(c,:));
    end
    fprintf(fid, '\nSingle-image MATLAB inference latency: mean %.2f ms, std %.2f ms (n=%d)\n', mean(t_runs), std(t_runs), nRuns);
    fclose(fid);
    fprintf('  Saved cnn_eval.mat and cnn_eval.md to %s\n', resultsDir);
end

%% ===== SECTION 9: OASBUD HOLD-OUT PATIENTS =====
fprintf('\n[9/9] OASBUD hold-out patients...\n');
data = load(oasbud_mat);
ids  = arrayfun(@(s) string(s.id), data.data);
cls  = arrayfun(@(s) double(s.class), data.data);   % 0 = benign, 1 = malignant
isHeldOut = ~ismember(ids, string(split.oasbudTrainIds));
testPatients = find(isHeldOut(:));
fprintf('  Patients in file: %d | used for training: %d | held out: %d (%d benign, %d malignant)\n', ...
    numel(ids), sum(~isHeldOut), numel(testPatients), sum(cls(testPatients) == 0), sum(cls(testPatients) == 1));

if numel(testPatients) < 2 || numel(unique(cls(testPatients))) < 2
    warning('Section 9 skipped: fewer than two held-out patients or only one class in %s.', oasbud_mat);
else
    oasbudNames = ["benign" "malignant"];
    trueOASBUD = categorical(oasbudNames(cls(testPatients) + 1)', oasbudNames);
    predOASBUD = strings(numel(testPatients), 1);
    probOASBUD = zeros(numel(testPatients), numel(classNames));
    for i = 1:numel(testPatients)
        p = testPatients(i);
        % Same steps as the Holoscan pipeline: A-line envelope, power law,
        % uint8, bilateral filter (Phase 3 selection), bilinear resize to 224, gray to RGB.
        env   = abs(hilbert(double(data.data(p).rf1)));
        bmode = (env ./ (max(env(:)) + 1e-12)) .^ 0.3;
        img   = im2uint8(imbilatfilt(im2double(uint8(bmode * 255)), 'DegreeOfSmoothing', 0.1, 'SpatialSigma', 2));
        img   = repmat(imresize(img, [224 224], 'bilinear'), [1 1 3]);
        probOASBUD(i,:) = minibatchpredict(trainedNet, img);
        % Binary decision as in InferenceOp: normal counts as not malignant
        pMal = probOASBUD(i, classNames == "malignant");
        predOASBUD(i) = oasbudNames(1 + (pMal > 1 - pMal));
    end
    predOASBUD = categorical(predOASBUD, oasbudNames);
    oasbud_acc = mean(predOASBUD == trueOASBUD);
    malRecall  = sum(predOASBUD == 'malignant' & trueOASBUD == 'malignant') / sum(trueOASBUD == 'malignant');
    benRecall  = sum(predOASBUD == 'benign' & trueOASBUD == 'benign') / sum(trueOASBUD == 'benign');
    fprintf('  OASBUD hold-out accuracy: %.1f%% (%d/%d)\n', 100*oasbud_acc, sum(predOASBUD == trueOASBUD), numel(testPatients));
    fprintf('  Malignant recall: %.1f%% | Benign recall: %.1f%%\n', 100*malRecall, 100*benRecall);

    fig_oasbud = figure('Name','OASBUD hold-out','Position',[50 50 620 540]);
    confusionchart(trueOASBUD, predOASBUD, 'RowSummary', 'row-normalized', ...
        'Title', sprintf('OASBUD hold-out (n=%d patients)  Acc: %.1f%%', numel(testPatients), 100*oasbud_acc));
    saveas(fig_oasbud, fullfile(figDir, 'confusion_oasbud_holdout.png'));
    heldOutIds = ids(testPatients); %#ok<NASGU>
    [~, oasbudFile] = fileparts(oasbud_mat); %#ok<ASGLU>
    save(fullfile(resultsDir, 'oasbud_holdout_eval.mat'), 'oasbud_acc', 'malRecall', 'benRecall', ...
        'heldOutIds', 'trueOASBUD', 'predOASBUD', 'probOASBUD', 'oasbudFile');
end
fprintf('\nFigures saved to %s\n', figDir);
