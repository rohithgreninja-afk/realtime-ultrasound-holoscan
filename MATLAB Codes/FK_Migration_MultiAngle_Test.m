% FK_Migration_MultiAngle_Test.m
% Verification test for fk_migration_planewave_multiangle.m against a
% known wire-target frame, comparing 1 angle vs the FULL 75 angles USING
% THE SAME FUNCTION both times, so internal FFT padding/scaling stays
% identical between the two runs and the raw amplitude comparison is
% actually valid. Display is cropped before the known FFT boundary
% artefact near the maximum depth, since nothing meaningful sits past
% the wires anyway.

clearvars; clc; close all;

%% ---- Settings ----
dataRoot  = getenv('PLANEWAVE_DATA_PATH');
if isempty(dataRoot)
    dataRoot = 'C:\Users\rohit\Downloads\CIRS040GSE\CIRS040GSE';
    fprintf('PLANEWAVE_DATA_PATH not set, using default: %s\n', dataRoot);
end
condition = 'low_attenuation';
frameIdx  = 6;   % readme.txt: frames 6-10 are wire targets
N_ANGLES_TO_USE = 75;   % full angle range, not a subsample this time
MAX_DISPLAY_DEPTH_MM = 45;  % crop before the FFT boundary artefact near max depth

conditionDir = fullfile(dataRoot, condition);

%% ---- Load header ----
headerFiles = dir(fullfile(conditionDir, 'USHEADER_*.mat'));
if isempty(headerFiles)
    headerFiles = dir(fullfile(dataRoot, 'USHEADER_*.mat'));
end
assert(~isempty(headerFiles), 'No USHEADER_*.mat found under %s', conditionDir);
hdr = load(fullfile(headerFiles(1).folder, headerFiles(1).name));
USHEADER = hdr.USHEADER;

xmitAngles = double(squeeze(USHEADER.xmitAngles));
n_ang = numel(xmitAngles);
n_ele = size(USHEADER.xmitDelay, 2);
fs    = double(USHEADER.fs);
pitch = double(USHEADER.pitch);
c     = double(USHEADER.c);
lensDelay = 96;

fprintf('Header: %d angles, %d elements, fs=%.2f MHz, pitch=%.3f mm, c=%.0f m/s\n', ...
    n_ang, n_ele, fs/1e6, pitch*1000, c);

%% ---- Load one data frame ----
dataFiles = dir(fullfile(conditionDir, 'USDATA_*.mat'));
assert(~isempty(dataFiles), 'No USDATA_*.mat found under %s', conditionDir);
fname = fullfile(dataFiles(frameIdx).folder, dataFiles(frameIdx).name);
fprintf('Loading frame %d: %s\n', frameIdx, dataFiles(frameIdx).name);
d = load(fname);
USDATA = double(squeeze(d.USDATA));

%% ---- Preprocessing: lens delay, angle delay correction, channel 125 ----
USDATA = USDATA(lensDelay+1:end, :, :);
nT = size(USDATA, 1);

anglesRad = deg2rad(xmitAngles);
delaySec  = abs((n_ele - 1)/2 * pitch * sin(anglesRad) / c);
delaySamples = floor(delaySec * fs);

for iAng = 1:n_ang
    dly = delaySamples(iAng);
    if dly > 0
        USDATA(1:nT-dly, :, iAng) = USDATA(dly+1:end, :, iAng);
        USDATA(nT-dly+1:end, :, iAng) = 0;
    end
end

if n_ele > 126
    USDATA(:, 126, :) = 2 * USDATA(:, 126, :);
end

%% ---- Select angle indices ----
angleIdx = round(linspace(1, n_ang, min(N_ANGLES_TO_USE, n_ang)));
angleIdx = unique(angleIdx);
fprintf('Using %d angles: %.1f to %.1f degrees\n', ...
    numel(angleIdx), xmitAngles(angleIdx(1)), xmitAngles(angleIdx(end)));

[~, broadsideIdx] = min(abs(xmitAngles));

rfMultiAngle  = USDATA(:, :, angleIdx);
anglesSelected = anglesRad(angleIdx);

rfOneAngle    = USDATA(:, :, broadsideIdx);
angleOne      = anglesRad(broadsideIdx);

%% ---- Common axes ----
x_elements = ((0:n_ele-1) - (n_ele-1)/2) * pitch;

%% ---- Same function, 1 angle vs full angle set: internal scaling now identical ----
fprintf('Running fk_migration_planewave_multiangle with 1 angle...\n');
tic;
fk_1angle_raw = fk_migration_planewave_multiangle(rfOneAngle, fs, pitch, c, angleOne);
fprintf('Done in %.2f s\n', toc);

fprintf('Running fk_migration_planewave_multiangle with %d angles (this will take longer)...\n', numel(angleIdx));
tic;
fk_multi_raw = fk_migration_planewave_multiangle(rfMultiAngle, fs, pitch, c, anglesSelected);
fprintf('Done in %.2f s\n', toc);

%% ---- Quantitative comparison, on the raw envelope, before any display normalisation ----
function [wire_peak, bg_std, cnr] = measure_contrast(raw, z_axis_mm, wire_depths_mm, bg_depth_range_mm)
    envelope = abs(hilbert(raw));
    wire_peak = zeros(size(wire_depths_mm));
    for i = 1:numel(wire_depths_mm)
        [~, row] = min(abs(z_axis_mm - wire_depths_mm(i)));
        window = max(1,row-2):min(size(envelope,1),row+2);
        wire_peak(i) = max(max(envelope(window, :)));
    end
    bgRows = find(z_axis_mm >= bg_depth_range_mm(1) & z_axis_mm <= bg_depth_range_mm(2));
    bg_region = envelope(bgRows, :);
    bg_std = std(bg_region(:));
    cnr = mean(wire_peak) / bg_std;
end

z_axis_mm = ((0:size(fk_1angle_raw,1)-1) * c / (2*fs)) * 1000;
wire_depths_mm = [5, 12, 20, 32];
bg_depth_range_mm = [35, MAX_DISPLAY_DEPTH_MM];

[peak_1, bg_1, cnr_1] = measure_contrast(fk_1angle_raw, z_axis_mm, wire_depths_mm, bg_depth_range_mm);
[peak_n, bg_n, cnr_n] = measure_contrast(fk_multi_raw,  z_axis_mm, wire_depths_mm, bg_depth_range_mm);

fprintf('\n=== Quantitative contrast comparison (same function, valid comparison) ===\n');
fprintf('%-25s %12s %12s\n', 'Metric', '1 angle', sprintf('%d angles', numel(angleIdx)));
fprintf('%-25s %12.4g %12.4g\n', 'Mean wire peak', mean(peak_1), mean(peak_n));
fprintf('%-25s %12.4g %12.4g\n', 'Background std dev', bg_1, bg_n);
fprintf('%-25s %12.4g %12.4g\n', 'Contrast-to-noise ratio', cnr_1, cnr_n);
fprintf('CNR improvement: %.1f%%\n', 100*(cnr_n/cnr_1 - 1));

%% ---- Post-processing for display, cropped before the boundary artefact ----
function img = postprocess(raw, z_axis_mm, max_depth_mm)
    cropRows = z_axis_mm <= max_depth_mm;
    envelope = abs(hilbert(raw(cropRows, :)));
    envNorm  = envelope / (max(envelope(:)) + eps);
    db = 20*log10(envNorm + 1e-6);
    db = max(db, -60);
    imgNorm  = (db + 60) / 60;
    imgUint8 = uint8(imgNorm * 255);
    img = medfilt2(imgUint8, [3 3]);
end

fk_1angle_img = postprocess(fk_1angle_raw, z_axis_mm, MAX_DISPLAY_DEPTH_MM);
fk_multi_img  = postprocess(fk_multi_raw,  z_axis_mm, MAX_DISPLAY_DEPTH_MM);

z_image_cropped = z_axis_mm(z_axis_mm <= MAX_DISPLAY_DEPTH_MM);

%% ---- Display side by side ----
figure('Position', [50 50 1100 700]);

subplot(1,2,1);
imagesc(x_elements*1000, z_image_cropped, fk_1angle_img);
colormap gray; colorbar;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title('F-K Migration -- 1 Angle');

subplot(1,2,2);
imagesc(x_elements*1000, z_image_cropped, fk_multi_img);
colormap gray; colorbar;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title(sprintf('F-K Migration -- %d Angles Compounded', numel(angleIdx)));

sgtitle(sprintf('F-K Migration: 1 Angle vs %d Angles, Full Range -- %s', ...
    numel(angleIdx), dataFiles(frameIdx).name), 'Interpreter', 'none');

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
figOutputFolder = fullfile(repoRoot, 'Project Figures', 'Phase7');
if ~exist(figOutputFolder, 'dir'), mkdir(figOutputFolder); end
saveas(gcf, fullfile(figOutputFolder, 'fk_single_vs_multiangle.png'));
fprintf('Figure saved to %s\n', figOutputFolder);
