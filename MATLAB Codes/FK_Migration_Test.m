% FK_Migration_Test.m
% Verification test for fk_migration_planewave.m against a known
% wire-target frame, compared side by side with the already-verified DAS
% result. This is the same validation discipline already applied to the
% DAS delay law and the data loader: prove it against real data with a
% known-good expected outcome (sharp point targets from wires) before
% trusting it anywhere in the pipeline.

clearvars; clc; close all;

%% ---- Settings ----
dataRoot  = getenv('PLANEWAVE_DATA_PATH');
if isempty(dataRoot)
    dataRoot = 'C:\Users\rohit\Downloads\CIRS040GSE\CIRS040GSE';
    fprintf('PLANEWAVE_DATA_PATH not set, using default: %s\n', dataRoot);
end
condition = 'low_attenuation';
frameIdx  = 6;   % readme.txt: frames 6-10 are wire targets

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

%% ---- Get broadside angle data ----
[~, broadsideIdx] = min(abs(xmitAngles));
rfSingleAngle = USDATA(:, :, broadsideIdx);
fprintf('Using broadside angle: %.1f degrees (index %d)\n', ...
    xmitAngles(broadsideIdx), broadsideIdx);

%% ---- DAS reconstruction (already verified) ----
x_elements = ((0:n_ele-1) - (n_ele-1)/2) * pitch;
zMax = (size(rfSingleAngle,1) / fs) * c / 2;
z_image_das = linspace(1e-3, zMax, 400);
x_image_das = linspace(x_elements(1), x_elements(end), 300);

fprintf('Running DAS (already verified)...\n');
tic;
das_raw = das_beamform_planewave(rfSingleAngle, x_elements, z_image_das, x_image_das, ...
    fs, c, anglesRad(broadsideIdx));
fprintf('DAS done in %.2f s\n', toc);

%% ---- F-K migration reconstruction (unverified, this is the test) ----
fprintf('Running f-k migration (unverified)...\n');
tic;
fk_raw = fk_migration_planewave(rfSingleAngle, fs, pitch, c);
fprintf('F-K migration done in %.2f s\n', toc);

% f-k migration's own native depth/lateral axes (see fk_migration_planewave.m header)
nt0 = size(fk_raw, 1);
z_image_fk = (0:nt0-1) * c / (2*fs);
x_image_fk = x_elements;   % same physical positions as DAS's x_elements

%% ---- Common post-processing: envelope, log compression, median filter ----
function img = postprocess(raw)
    envelope = abs(hilbert(raw));
    envNorm  = envelope / (max(envelope(:)) + eps);
    db = 20*log10(envNorm + 1e-6);
    db = max(db, -60);
    imgNorm  = (db + 60) / 60;
    imgUint8 = uint8(imgNorm * 255);
    img = medfilt2(imgUint8, [3 3]);
end

das_img = postprocess(das_raw);
fk_img  = postprocess(fk_raw);

%% ---- Display side by side ----
figure('Position', [50 50 1100 700]);

subplot(1,2,1);
imagesc(x_image_das*1000, z_image_das*1000, das_img);
colormap gray; colorbar;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title('DAS (verified)');

subplot(1,2,2);
imagesc(x_image_fk*1000, z_image_fk*1000, fk_img);
colormap gray; colorbar;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title('F-K Migration (UNVERIFIED -- check for sharp wire points)');

sgtitle(sprintf('DAS vs F-K Migration -- %s, broadside %.1f deg', ...
    dataFiles(frameIdx).name, xmitAngles(broadsideIdx)), 'Interpreter', 'none');

fprintf('\nCheck the figure: wires should appear as sharp discrete points in BOTH panels.\n');
fprintf('If f-k migration shows no structure, smearing, or garbled output, do not\n');
fprintf('trust it -- report back what you see before this goes anywhere further.\n');
