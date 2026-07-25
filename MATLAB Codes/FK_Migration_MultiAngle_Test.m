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
fprintf('Running multi-angle f-k migration, %d angles (UNVERIFIED)...\n', numel(angleIdx));
tic;
fk_multi_raw = fk_migration_planewave_multiangle(rfMultiAngle, fs, pitch, c, anglesSelected);
fprintf('Done in %.2f s\n', toc);

%% ---- Post-processing ----
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
title(sprintf('F-K Migration -- %d Angles Compounded (UNVERIFIED)', numel(angleIdx)));

sgtitle(sprintf('F-K Migration: Single vs Multi-Angle -- %s', ...
    dataFiles(frameIdx).name), 'Interpreter', 'none');

fprintf('\nCheck the figure: wires should be sharper and background cleaner in the\n');
fprintf('multi-angle panel than the single-angle panel. If the multi-angle panel is\n');
fprintf('garbled, shows doubled/ghost wires, or wires in wrong positions, do not\n');
fprintf('trust it -- report back what you see.\n');

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
figOutputFolder = fullfile(repoRoot, 'Project Figures', 'Phase7');
if ~exist(figOutputFolder, 'dir'), mkdir(figOutputFolder); end
saveas(gcf, fullfile(figOutputFolder, 'fk_single_vs_multiangle.png'));
fprintf('Figure saved to %s\n', figOutputFolder);
