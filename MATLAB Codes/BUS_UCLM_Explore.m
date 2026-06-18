% =========================================================
% BUS_UCLM_Explore.m
% Explore and audit the BUS-UCLM dataset structure
% =========================================================

busUCLMPath = 'C:\Users\rohit\Downloads\Real Time Image Processing Project\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM';

% ── Read INFO.csv ─────────────────────────────────────────
info = readtable(fullfile(busUCLMPath, 'INFO.csv'));

fprintf('Column names:\n');
disp(info.Properties.VariableNames);

fprintf('\nFirst 10 rows:\n');
disp(info(1:10, :));

fprintf('\nTotal images: %d\n', height(info));

% ── Read Label column specifically ────────────────────────
labels = info.Label;
classes = unique(labels);

fprintf('\nUnique labels in Label column:\n');
disp(classes);

fprintf('\nClass distribution:\n');
for i = 1:length(classes)
    count = sum(strcmp(labels, classes{i}));
    fprintf('  %-12s: %d\n', classes{i}, count);
end

% ── Check images folder ───────────────────────────────────
imgFiles = dir(fullfile(busUCLMPath, 'images', '*.png'));
fprintf('\nTotal PNG files in images folder: %d\n', length(imgFiles));

% ── Visual check — show one image per class ───────────────
figure('Position', [100 100 1200 400]);
plotIdx = 1;
for i = 1:length(classes)
    classRows = info(strcmp(info.Label, classes{i}), :);
    imgName   = classRows.Image{1};
    imgPath   = fullfile(busUCLMPath, 'images', imgName);
    img       = imread(imgPath);

    subplot(1, length(classes), plotIdx);
    imshow(img);
    title(sprintf('%s\n%s', classes{i}, imgName), ...
          'Interpreter', 'none', 'FontSize', 9);
    plotIdx = plotIdx + 1;
end
sgtitle('BUS-UCLM — One Sample Per Class');

% ── Filter out images with black borders (Marks = Yes means annotations) ──
% Also filter Doppler = Yes since those have colour overlay
cleanRows = info(strcmp(info.Doppler, 'No'), :);
fprintf('\nAfter removing Doppler images: %d remaining\n', height(cleanRows));

% Count clean distribution
fprintf('Clean class distribution:\n');
classes = {'Benign', 'Malignant', 'Normal'};
for i = 1:3
    count = sum(strcmp(cleanRows.Label, classes{i}));
    fprintf('  %-12s: %d\n', classes{i}, count);
end

% ── Show combined dataset totals ──────────────────────────
fprintf('\n========= Combined Dataset Plan =========\n');
fprintf('BUSI:      benign=437, malignant=210, normal=133  (total=780)\n');
fprintf('BUS-UCLM:  benign=174, malignant=90,  normal=419  (total=683)\n');
fprintf('Combined:  benign=%d,  malignant=%d, normal=%d   (total=%d)\n', ...
    437+174, 210+90, 133+419, 780+683);