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
    dataRoot = 'C:\Users\rohit\Downloads\CIRS040GSE\CIRS040GSE';
    fprintf('PLANEWAVE_DATA_PATH not set, using default: %s\n', dataRoot);
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

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
figOutputFolder = fullfile(repoRoot, 'Project Figures', 'Phase7');
if ~exist(figOutputFolder, 'dir'), mkdir(figOutputFolder); end
saveas(gcf, fullfile(figOutputFolder, 'phased_array_toolbox_real_data.png'));
fprintf('Figure saved to %s\n', figOutputFolder);
