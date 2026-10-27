function [imds, meta] = buildCombinedDatastore()
% buildCombinedDatastore  Single source of truth for the combined CNN dataset.
% Loads BUSI, BUS-UCLM, BUS-BRA, BrEaST and OASBUD-PNG and returns one
% labelled imageDatastore. Called by both Phase4_MegaTrain.m and
% Phase6_CNN_Evaluation.m so training and evaluation see identical data.
%
% Environment variables (all required, error names the missing one):
%   BUSI_PATH         Dataset_BUSI_with_GT folder
%   BUS_UCLM_PATH     BUS-UCLM folder (contains INFO.csv and images/)
%   BUSBRA_PATH       BUSBRA folder (contains bus_data.csv and Images/)
%   BREAST_IMG_PATH   BrEaST-Lesions_USG-images_and_masks folder
%   BREAST_XLSX_PATH  BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx
%   OASBUD_PNG_PATH   output folder of Phase4_OASBUD_Export.m
%
% meta is a table with one row per image:
%   Key    'SOURCE/relative/path', portable across machines (used in split files)
%   Source dataset name
%   Group  patient-level grouping id; images sharing a Group are always
%          kept in the same split partition

busiRoot   = requireEnv('BUSI_PATH');
uclmRoot   = requireEnv('BUS_UCLM_PATH');
busbraRoot = requireEnv('BUSBRA_PATH');
breastRoot = requireEnv('BREAST_IMG_PATH');
breastXLSX = requireEnv('BREAST_XLSX_PATH');
oasbRoot   = requireEnv('OASBUD_PNG_PATH');

files = strings(0,1); labels = strings(0,1);
src = strings(0,1); rel = strings(0,1); grp = strings(0,1);

% BUSI: folder labels, drop *_mask*.png. No patient ids, one group per image.
d = dir(fullfile(busiRoot, '*', '*.png'));
d = d(~contains({d.name}, '_mask'));
for i = 1:numel(d)
    [~, cls] = fileparts(d(i).folder);
    r = string(cls) + "/" + string(d(i).name);
    add(fullfile(d(i).folder, d(i).name), lower(cls), "BUSI", r, "BUSI_" + r);
end

% BUS-UCLM: CSV labels, Doppler images excluded. Group by patient prefix (ALWI_000 -> ALWI).
info = readtable(fullfile(uclmRoot, 'INFO.csv'), 'Delimiter', ';');
info = info(strcmp(info.Doppler, 'No'), :);
for i = 1:height(info)
    f = fullfile(uclmRoot, 'images', info.Image{i});
    if ~isfile(f), continue; end
    pid = extractBefore(string(info.Image{i}), "_");
    add(f, lower(info.Label{i}), "BUS_UCLM", "images/" + string(info.Image{i}), "UCLM_" + pid);
end

% BUS-BRA: CSV labels, two views per case share a group.
bra = readtable(fullfile(busbraRoot, 'bus_data.csv'));
for i = 1:height(bra)
    lbl = lower(string(bra.Pathology{i}));
    f = fullfile(busbraRoot, 'Images', [bra.ID{i} '.png']);
    if ~isfile(f) || ~ismember(lbl, ["benign" "malignant"]), continue; end
    add(f, lbl, "BUSBRA", "Images/" + string(bra.ID{i}) + ".png", "BUSBRA_" + string(bra.Case(i)));
end

% BrEaST: XLSX labels, mask files and unknown labels excluded.
tb = readtable(breastXLSX);
for i = 1:height(tb)
    lbl = lower(strtrim(string(tb.Classification(i))));
    if ~ismember(lbl, ["benign" "malignant" "normal"]), continue; end
    fn = strtrim(string(tb.Image_filename(i)));
    if contains(fn, "_tumor") || contains(fn, "_other"), continue; end
    f = fullfile(breastRoot, fn);
    if ~isfile(f), continue; end
    add(f, lbl, "BREAST", fn, "BREAST_" + fn);
end

% OASBUD-PNG: folder labels, group by patient id embedded in the file name.
d = dir(fullfile(oasbRoot, '*', 'oasbud_*.png'));
for i = 1:numel(d)
    [~, cls] = fileparts(d(i).folder);
    tok = regexp(d(i).name, '^oasbud_[a-z]+_r\d+_(.+)_(base|view2|aug\d+)\.png$', 'tokens', 'once');
    assert(~isempty(tok), 'Unexpected OASBUD-PNG file name: %s', d(i).name);
    r = string(cls) + "/" + string(d(i).name);
    add(fullfile(d(i).folder, d(i).name), lower(cls), "OASBUD_PNG", r, "OASBUD_" + string(tok{1}));
end

labels = categorical(labels);
imds = imageDatastore(cellstr(files), 'Labels', labels);
meta = table(src + "/" + rel, src, grp, labels, 'VariableNames', {'Key','Source','Group','Label'});

fprintf('Combined dataset: %d images\n', numel(files));
srcs = unique(src, 'stable');
for s = srcs'
    m = src == s;
    c = countcats(labels(m));
    fprintf('  %-11s %5d  (%s)\n', s, sum(m), strjoin(string(categories(labels))' + "=" + string(c'), ', '));
end

    function add(f, lbl, s, r, g)
        files(end+1,1)  = string(f);
        labels(end+1,1) = string(lbl);
        src(end+1,1)    = s;
        rel(end+1,1)    = r;
        grp(end+1,1)    = g;
    end
end

function v = requireEnv(name)
v = getenv(name);
if isempty(v)
    error('buildCombinedDatastore:envMissing', ...
        'Environment variable %s is not set. Set it with setenv(''%s'', ''/path/to/data'').', name, name);
end
end
