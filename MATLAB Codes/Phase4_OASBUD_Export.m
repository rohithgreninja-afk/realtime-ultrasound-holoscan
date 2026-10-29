% Phase4_OASBUD_Export.m
% Export the first N malignant + first N benign OASBUD patients as PNG
% training images, plus augmented copies (imageDataAugmenter).
% Uses A-line reconstruction (hilbert + power law), same as the pipeline.
% Output: OASBUD_PNG_PATH/malignant/ and /benign/, plus
%         OASBUD_PNG_PATH/oasbud_export_manifest.mat listing the exported
%         patients. Every patient not listed there is an OASBUD hold-out
%         patient for Phase6_CNN_Evaluation.m Section 9 and the pipeline benchmark.
%
% OASBUD class field: 0 = benign, 1 = malignant. This matches the dataset's
% 48/52 benign/malignant counts and its BI-RADS field (class 0 is mostly
% BI-RADS 3, class 1 mostly 4c/5).

clearvars; clc;

%% Settings
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);

dataPath = getenv('OASBUD_PATH');
if isempty(dataPath)
    dataPath = fullfile(repoRoot, 'data', 'sample', 'OASBUD_sample.mat');
    fprintf('OASBUD_PATH not set, using bundled sample: %s\n', dataPath);
end

outRoot = getenv('OASBUD_PNG_PATH');
if isempty(outRoot)
    error('OASBUD_PNG_PATH environment variable not set. Set it with setenv(''OASBUD_PNG_PATH'', ''/path/to/OASBUD_PNG'').');
end

nPerClass = 20;    % first N patients of each class are exported for training
augPerPat = 20;    % augmented versions per patient
gamma     = 0.3;
imgSize   = [224 224];
classNames = ["benign" "malignant"];   % index = class field + 1
rng(42);

augmenter = imageDataAugmenter( ...
    'RandXReflection',  true, ...
    'RandRotation',     [-20 20], ...
    'RandScale',        [1 1.15], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

%% Load dataset and select patients from the loaded class field
data = load(dataPath);
cls  = arrayfun(@(s) double(s.class), data.data);
ids  = arrayfun(@(s) string(s.id), data.data);
fprintf('OASBUD loaded: %d patients (%d benign, %d malignant)\n', ...
    numel(cls), sum(cls == 0), sum(cls == 1));

selIdx = [];
for k = [1 0]
    idx = find(cls(:) == k);
    if numel(idx) < nPerClass
        warning('Only %d %s patients available, exporting all of them.', numel(idx), classNames(k+1));
    end
    selIdx = [selIdx; idx(1:min(nPerClass, numel(idx)))]; %#ok<AGROW>
end

%% Refuse to mix with an earlier export
for lbl = classNames
    outDir = fullfile(outRoot, lbl);
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    if ~isempty(dir(fullfile(outDir, '*.png')))
        error('%s already contains PNG files. Point OASBUD_PNG_PATH at an empty folder.', outDir);
    end
end

%% Export
for p = selIdx'
    label  = classNames(cls(p) + 1);
    outDir = fullfile(outRoot, label);
    stem   = sprintf('oasbud_%s_r%03d_%s', char(label), p, char(regexprep(lower(ids(p)), '[^a-z0-9]', '')));

    base = imresize(reconstruct(double(data.data(p).rf1), gamma), imgSize);
    writeChecked(base, outDir, [stem '_base.png'], data.data(p).class, classNames);
    view2 = imresize(reconstruct(double(data.data(p).rf2), gamma), imgSize);
    writeChecked(view2, outDir, [stem '_view2.png'], data.data(p).class, classNames);

    for a = 1:augPerPat
        img = augment(augmenter, base);
        writeChecked(img, outDir, sprintf('%s_aug%03d.png', stem, a), data.data(p).class, classNames);
    end
    fprintf('Patient %3d %-14s (%s): %d images\n', p, ids(p), label, augPerPat + 2);
end

exportedIdx = selIdx;
exportedIds = ids(selIdx);
exportedClass = cls(selIdx);
[~, sourceFile] = fileparts(dataPath);
save(fullfile(outRoot, 'oasbud_export_manifest.mat'), ...
    'exportedIdx', 'exportedIds', 'exportedClass', 'sourceFile', 'nPerClass', 'augPerPat');

fprintf('\nExport complete.\n');
fprintf('Malignant training images: %d\n', numel(dir(fullfile(outRoot,'malignant','*.png'))));
fprintf('Benign training images:    %d\n', numel(dir(fullfile(outRoot,'benign','*.png'))));

function bmode = reconstruct(rf, gamma)
    env   = abs(hilbert(rf));
    bmode = (env / (max(env(:)) + 1e-12)) .^ gamma;
end

function writeChecked(img, outDir, fname, classField, classNames)
    % Guard: the folder name must match the label derived from the class field
    [~, folderLabel] = fileparts(outDir);
    if ~strcmp(folderLabel, classNames(double(classField) + 1))
        error('Label guard: class %d would be written to %s', classField, outDir);
    end
    imwrite(im2uint8(img), fullfile(outDir, fname));
end
