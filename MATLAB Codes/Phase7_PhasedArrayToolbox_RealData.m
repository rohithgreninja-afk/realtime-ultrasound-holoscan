% Phase7_PhasedArrayToolbox_RealData.m
% =========================================================
% Phased Array System Toolbox demonstration on REAL per-element data
% =========================================================
% Replaces Phase7_PhasedArrayToolbox_Demo.m's synthetic point-target
% data with genuine recorded channel data from the CIRS040GSE dataset
% (Zenodo 7986407), a wire-target frame, broadside acquisition. This is
% the correct kind of data for the toolbox: real per-element channel
% signals with real array geometry, unlike OASBUD, whose columns are
% pre-formed scan lines (see README Dataset Strategy section).
%
% Uses phased.ULA with the array's actual geometry (128 elements,
% pitch read directly from the dataset header, matching the real
% Verasonics L12-5 acquisition) and phased.PhaseShiftBeamformer steered
% toward broadside on the analytic (Hilbert) signal of the real recorded
% data. PhaseShiftBeamformer expects narrowband complex data; ultrasound
% RF is wideband, so this is a reasonable simplification for
% demonstrating the toolbox on real recorded channel data, not a claim
% of full wideband image reconstruction (that is what das_beamform_planewave.m
% and fk_migration_planewave.m are for).

clearvars; clc; close all;

%% ---- Load real data ----
dataRoot = getenv('PLANEWAVE_DATA_PATH');
if isempty(dataRoot)
    error(['PLANEWAVE_DATA_PATH environment variable not set. Point it at the folder containing ' ...
           'USHEADER_*.mat and low_attenuation/, e.g. setenv(''PLANEWAVE_DATA_PATH'', ''/path/to/CIRS040GSE/CIRS040GSE'')']);
end
condition = 'low_attenuation';
frameIdx = 6; % wire target, per readme.txt

conditionDir = fullfile(dataRoot, condition);
headerFiles = dir(fullfile(conditionDir, 'USHEADER_*.mat'));
if isempty(headerFiles)
    headerFiles = dir(fullfile(dataRoot, 'USHEADER_*.mat'));
end
assert(~isempty(headerFiles), 'No USHEADER_*.mat found under %s', conditionDir);
hdr = load(fullfile(headerFiles(1).folder, headerFiles(1).name));
USHEADER = hdr.USHEADER;

xmitAngles = double(squeeze(USHEADER.xmitAngles));
n_ele = size(USHEADER.xmitDelay, 2);
fs    = double(USHEADER.fs);
pitch = double(USHEADER.pitch);
c     = double(USHEADER.c);
fc    = 7.8e6; % L12-5 transducer centre frequency, per dataset documentation
lensDelay = 96;

dataFiles = dir(fullfile(conditionDir, 'USDATA_*.mat'));
assert(~isempty(dataFiles), 'No USDATA_*.mat found under %s', conditionDir);
fname = fullfile(dataFiles(frameIdx).folder, dataFiles(frameIdx).name);
fprintf('Loading real data: %s\n', dataFiles(frameIdx).name);
d = load(fname);
USDATA = double(squeeze(d.USDATA));
USDATA = USDATA(lensDelay+1:end, :, :);
if n_ele > 126
    USDATA(:, 126, :) = 2 * USDATA(:, 126, :);
end

[~, broadsideIdx] = min(abs(xmitAngles));
rf = USDATA(:, :, broadsideIdx);   % real per-element receive data, broadside

fprintf('Real array data loaded: %d elements, %d samples, %.3f mm pitch, fc=%.1f MHz\n', ...
    n_ele, size(rf,1), pitch*1000, fc/1e6);

%% ---- Build the array with the REAL geometry from the header ----
array = phased.ULA('NumElements', n_ele, 'ElementSpacing', pitch);
array.Element.FrequencyRange = [0.5*fc, 1.5*fc];

%% ---- Convert real RF to complex baseband (Hilbert analytic signal) ----
rf_analytic = hilbert(rf);

%% ---- Phased Array System Toolbox beamformer, steered toward broadside ----
beamformer = phased.PhaseShiftBeamformer('SensorArray', array, ...
    'OperatingFrequency', fc, 'PropagationSpeed', c, ...
    'Direction', [0; 0], 'WeightsOutputPort', false);

yBeamformed = beamformer(rf_analytic);

fprintf('Beamformed output: %d samples (from %d real channels)\n', ...
    numel(yBeamformed), n_ele);

%% ---- Compare single element vs beamformed ----
figure('Position', [100 100 900 500]);

subplot(2,1,1);
plot(real(rf_analytic(:, round(n_ele/2))));
title('Single Element (centre), Real Recorded Data');
xlabel('Sample'); ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(real(yBeamformed));
title('phased.PhaseShiftBeamformer Output, Real Recorded Data');
xlabel('Sample'); ylabel('Amplitude');
grid on;

sgtitle(sprintf('Phased Array System Toolbox on Real CIRS040GSE Data (%s, wire target)', ...
    dataFiles(frameIdx).name), 'Interpreter', 'none');

snr_improvement = 10*log10(var(real(yBeamformed)) / var(real(rf_analytic(:,round(n_ele/2)))));
fprintf('\nSNR improvement (beamformed vs single element): %.2f dB\n', snr_improvement);
fprintf('This uses the real Verasonics L12-5 array geometry (%d elements, %.3f mm\n', ...
    n_ele, pitch*1000);
fprintf('pitch) and real recorded channel data from the CIRS040GSE phantom,\n');
fprintf('replacing the earlier synthetic point-target demonstration.\n');

resultsDir = getenv('RESULTS_DIR');
if isempty(resultsDir)
    error('RESULTS_DIR environment variable not set. Set it with setenv(''RESULTS_DIR'', ''/path/to/results'').');
end
figOutputFolder = fullfile(resultsDir, 'figures', 'Phase7');
if ~exist(figOutputFolder, 'dir'), mkdir(figOutputFolder); end
saveas(gcf, fullfile(figOutputFolder, 'phased_array_toolbox_real_data.png'));
fprintf('Figure saved to %s\n', figOutputFolder);

%% ---- 2-D image: phased.TimeDelayBeamformer scan-angle sweep vs hand-written DAS ----
% Wideband time-delay beamforming of the same broadside frame, steered over
% a sweep of receive angles (Direction supplied per call). Each steered
% output is one image line; sample time t maps to range r = c*t/2 and the
% line is drawn along its steering angle. The toolbox beamformer assumes
% far-field plane-wave arrivals, so near-field targets are less focused than
% in the per-pixel DAS image; the custom DAS kernel stays the GPU Coder target.
scanAngles = -25:0.25:25;   % degrees, azimuth from broadside
tdb = phased.TimeDelayBeamformer('SensorArray', array, 'SampleRate', fs, ...
    'PropagationSpeed', c, 'DirectionSource', 'Input port');
tdbLines = zeros(size(rf, 1), numel(scanAngles));
for k = 1:numel(scanAngles)
    tdbLines(:, k) = tdb(rf, [scanAngles(k); 0]);
end
range = (0:size(rf, 1)-1)' * c / (2 * fs);
keep  = range >= 1e-3 & range <= 55e-3;   % both images cropped to 1-55 mm (edge artefacts beyond)
range = range(keep);
[TH, R] = meshgrid(deg2rad(scanAngles), range);
tdbEnv = abs(hilbert(tdbLines));
tdbEnv = tdbEnv(keep, :);
tdbDb  = max(20*log10(tdbEnv / max(tdbEnv(:)) + eps), -60);

elemPos = getElementPosition(array);
x_elements = elemPos(2, :);
z_das = linspace(1e-3, 55e-3, 400);
x_das = linspace(x_elements(1), x_elements(end), 300);
dasRaw = das_beamform_planewave(rf, x_elements, z_das, x_das, fs, c, 0);
dasEnv = abs(hilbert(dasRaw));
dasDb  = max(20*log10(dasEnv / max(dasEnv(:)) + eps), -60);

fig2 = figure('Position', [100 100 1100 600]);
subplot(1, 2, 1);
surf(1000 * R .* sin(TH), 1000 * R .* cos(TH), zeros(size(R)), tdbDb, 'EdgeColor', 'none');
view(2); axis ij image; colormap gray; clim([-60 0]); colorbar;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title(sprintf('phased.TimeDelayBeamformer, %d steering angles', numel(scanAngles)));
subplot(1, 2, 2);
imagesc(x_das * 1000, z_das * 1000, dasDb, [-60 0]); axis image; colormap gray; colorbar;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title('Hand-written plane-wave DAS (das\_beamform\_planewave.m)');
sgtitle(sprintf('Real CIRS040GSE channel data, %s, broadside transmit', dataFiles(frameIdx).name), 'Interpreter', 'none');
fig5Folder = fullfile(resultsDir, 'figures', 'Phase5');
if ~exist(fig5Folder, 'dir'), mkdir(fig5Folder); end
saveas(fig2, fullfile(fig5Folder, 'toolbox_tdb_vs_das.png'));
fprintf('Figure saved to %s\n', fullfile(fig5Folder, 'toolbox_tdb_vs_das.png'));
