% FK_Migration_MultiAngle_Test.m
% Verification test for fk_migration_planewave_multiangle.m against the
% same known wire-target frame used to verify the broadside-only version,
% comparing single-angle f-k, multi-angle f-k, and DAS side by side.

clearvars; clc; close all;

%% ---- Settings ----
dataRoot  = getenv('PLANEWAVE_DATA_PATH');
if isempty(dataRoot)
    dataRoot = 'C:\Users\rohit\Downloads\CIRS040GSE\CIRS040GSE';
    fprintf('PLANEWAVE_DATA_PATH not set, using default: %s\n', dataRoot);
end
condition = 'low_attenuation';
frameIdx  = 6;   % readme.txt: frames 6-10 are wire targets
N_ANGLES_TO_USE = 15;  % subsample from the 75 available for reasonable runtime

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

%% ---- Select subsampled angle indices, spread across the full range ----
angleIdx = round(linspace(1, n_ang, N_ANGLES_TO_USE));
angleIdx = unique(angleIdx);
fprintf('Using %d angles: %.1f to %.1f degrees\n', ...
    numel(angleIdx), xmitAngles(angleIdx(1)), xmitAngles(angleIdx(end)));

rfMultiAngle = USDATA(:, :, angleIdx);
anglesSelected = anglesRad(angleIdx);

[~, broadsideIdx] = min(abs(xmitAngles));
rfSingleAngle = USDATA(:, :, broadsideIdx);

%% ---- Common axes ----
x_elements = ((0:n_ele-1) - (n_ele-1)/2) * pitch;

%% ---- Single-angle f-k (already verified) ----
fprintf('Running single-angle f-k migration (verified)...\n');
tic;
fk_single_raw = fk_migration_planewave(rfSingleAngle, fs, pitch, c);
fprintf('Done in %.2f s\n', toc);

%% ---- Multi-angle f-k (this is the test) ----
fprintf('Running multi-angle f-k migration, %d angles...\n', numel(angleIdx));
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

z_axis_mm = ((0:size(fk_single_raw,1)-1) * c / (2*fs)) * 1000;
wire_depths_mm = [5, 12, 20, 32];   % approximate depths of 4 visible wires, read off the earlier figure
bg_depth_range_mm = [45, 60];       % empty region below the wires, above the edge artefact

[peak_single, bg_single, cnr_single] = measure_contrast(fk_single_raw, z_axis_mm, wire_depths_mm, bg_depth_range_mm);
[peak_multi,  bg_multi,  cnr_multi ] = measure_contrast(fk_multi_raw,  z_axis_mm, wire_depths_mm, bg_depth_range_mm);

fprintf('\n=== Quantitative contrast comparison (raw envelope, pre-normalisation) ===\n');
fprintf('%-25s %12s %12s\n', 'Metric', 'Single-angle', 'Multi-angle');
fprintf('%-25s %12.1f %12.1f\n', 'Mean wire peak', mean(peak_single), mean(peak_multi));
fprintf('%-25s %12.2f %12.2f\n', 'Background std dev', bg_single, bg_multi);
fprintf('%-25s %12.2f %12.2f\n', 'Contrast-to-noise ratio', cnr_single, cnr_multi);
fprintf('CNR improvement: %.1f%%\n', 100*(cnr_multi/cnr_single - 1));

%% ---- Post-processing for display ----
function img = postprocess(raw)
    envelope = abs(hilbert(raw));
    envNorm  = envelope / (max(envelope(:)) + eps);
    db = 20*log10(envNorm + 1e-6);
    db = max(db, -60);
    imgNorm  = (db + 60) / 60;
    imgUint8 = uint8(imgNorm * 255);
    img = medfilt2(imgUint8, [3 3]);
end

fk_single_img = postprocess(fk_single_raw);
fk_multi_img  = postprocess(fk_multi_raw);

nt0 = size(fk_single_raw, 1);
z_image = (0:nt0-1) * c / (2*fs);

%% ---- Display side by side ----
figure('Position', [50 50 1100 700]);

subplot(1,2,1);
imagesc(x_elements*1000, z_image*1000, fk_single_img);
colormap gray; colorbar;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title('F-K Migration -- Single Angle (broadside)');

subplot(1,2,2);
imagesc(x_elements*1000, z_image*1000, fk_multi_img);
colormap gray; colorbar;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title(sprintf('F-K Migration -- %d Angles Compounded', numel(angleIdx)));

sgtitle(sprintf('F-K Migration: Single vs Multi-Angle -- %s', ...
    dataFiles(frameIdx).name), 'Interpreter', 'none');

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
figOutputFolder = fullfile(repoRoot, 'Project Figures', 'Phase7');
if ~exist(figOutputFolder, 'dir'), mkdir(figOutputFolder); end
saveas(gcf, fullfile(figOutputFolder, 'fk_single_vs_multiangle.png'));
fprintf('Figure saved to %s\n', figOutputFolder);
