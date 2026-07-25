% PlaneWave_Reconstruct_Test.m
% First end-to-end reconstruction test for the plane wave dataset
% (CIRS040GSE, Zenodo 7986407), in MATLAB, before touching Python/Holoscan.
%
% Loads one USDATA frame, applies the three corrections the dataset's own
% reference implementation requires (lens delay trim, angle-dependent
% delay correction, channel 125 doubling), reconstructs a single-angle
% (broadside) image using das_beamform_planewave, then applies the same
% envelope detection + log compression + median filter used elsewhere in
% this project, and displays it.

clearvars; clc; close all;

%% ---- Settings ----
dataRoot  = getenv('PLANEWAVE_DATA_PATH');
if isempty(dataRoot)
    dataRoot = 'C:\Users\rohit\Downloads\CIRS040GSE\CIRS040GSE';
    fprintf('PLANEWAVE_DATA_PATH not set, using default: %s\n', dataRoot);
end
condition = 'low_attenuation';
frameIdx  = 6;   % readme.txt: frames 6-10 are wire targets, a clean visual test

conditionDir = fullfile(dataRoot, condition);

%% ---- Load header ----
headerFiles = dir(fullfile(conditionDir, 'USHEADER_*.mat'));
if isempty(headerFiles)
    headerFiles = dir(fullfile(dataRoot, 'USHEADER_*.mat'));
end
assert(~isempty(headerFiles), 'No USHEADER_*.mat found under %s', conditionDir);
hdr = load(fullfile(headerFiles(1).folder, headerFiles(1).name));
USHEADER = hdr.USHEADER;

xmitAngles = double(squeeze(USHEADER.xmitAngles));   % degrees
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
assert(frameIdx <= numel(dataFiles), 'frameIdx exceeds available files');

fname = fullfile(dataFiles(frameIdx).folder, dataFiles(frameIdx).name);
fprintf('Loading frame %d: %s\n', frameIdx, dataFiles(frameIdx).name);
d = load(fname);
USDATA = double(squeeze(d.USDATA));   % cast from int16, drop singleton dim
fprintf('USDATA shape after squeeze: %s\n', mat2str(size(USDATA)));
% Expect [time_samples, n_ele, n_ang] in MATLAB's own native order

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

if n_ele > 126  % 0-indexed channel 125 -> MATLAB index 126
    USDATA(:, 126, :) = 2 * USDATA(:, 126, :);
end

fprintf('Preprocessing applied: lens delay trim, angle delay correction, channel 125 fix\n');

%% ---- Single-angle (broadside) reconstruction ----
[~, broadsideIdx] = min(abs(xmitAngles));
txAngle = anglesRad(broadsideIdx);
rfSingleAngle = USDATA(:, :, broadsideIdx);

x_elements = ((0:n_ele-1) - (n_ele-1)/2) * pitch;
zMax = (size(rfSingleAngle,1) / fs) * c / 2;
z_image = linspace(1e-3, zMax, 400);
x_image = linspace(x_elements(1), x_elements(end), 300);

fprintf('Reconstructing single angle (%.1f degrees, index %d)...\n', ...
    xmitAngles(broadsideIdx), broadsideIdx);
tic;
bmode_raw = das_beamform_planewave(rfSingleAngle, x_elements, z_image, x_image, fs, c, txAngle);
fprintf('Done in %.2f s\n', toc);

%% ---- Envelope detection, log compression, median filter ----
envelope = abs(hilbert(bmode_raw));
envNorm  = envelope / (max(envelope(:)) + eps);
db = 20*log10(envNorm + 1e-6);
db = max(db, -60);

imgNorm  = (db + 60) / 60;
imgUint8 = uint8(imgNorm * 255);
imgMedian = medfilt2(imgUint8, [3 3]);

%% ---- Display ----
figure('Position', [100 100 700 700]);
imagesc(x_image*1000, z_image*1000, imgMedian);
colormap gray; colorbar;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title(sprintf('Plane Wave Reconstruction -- Single Angle (%.1f deg)\n%s', ...
    xmitAngles(broadsideIdx), dataFiles(frameIdx).name), 'Interpreter', 'none');

fprintf('\nReconstruction complete. Check the figure window for the image.\n');
