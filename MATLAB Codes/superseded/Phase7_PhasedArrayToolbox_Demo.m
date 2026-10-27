% Phase7_PhasedArrayToolbox_Demo.m
% =========================================================
% Phased Array System Toolbox compliance demonstration
% =========================================================
% This script is a standalone demonstration of the MATLAB Phased Array
% System Toolbox, using phased.ULA and phased.PhaseShiftBeamformer on
% synthetic point-target data.
%
% It is NOT part of the OASBUD processing pipeline. The OASBUD dataset's
% RF columns are pre-formed scan lines (single-focus beamforming already
% applied by the acquisition scanner), not raw per-element channel data,
% so toolbox beamformers that operate on individual array-element signals
% do not apply to this dataset. See README "Beamforming" notes and the
% custom das_beamform.m / DAS-family scripts for the beamforming approach
% actually used on OASBUD, and the BeamformingOp Hilbert-envelope method
% used in the deployed Holoscan pipeline.
%
% This script exists to demonstrate the toolbox itself, using data where
% its beamformers are the physically appropriate tool: a synthetic
% uniform linear array (matching the OASBUD source transducer's nominal
% specs) receiving a plane wave from a point target.
% =========================================================

clearvars; clc;

%% ── Array definition ──────────────────────────────────────
% Matches the nominal OASBUD source transducer (Ultrasonix L14-5/38):
% 128 elements, 0.30 mm pitch, 10 MHz centre frequency.
fc          = 10e6;         % Centre frequency (Hz)
c           = 1540;         % Speed of sound in soft tissue (m/s)
lambda      = c / fc;       % Wavelength (m)
pitch       = 0.30e-3;      % Element spacing (m)
numElements = 128;

array = phased.ULA('NumElements', numElements, 'ElementSpacing', pitch);
array.Element.FrequencyRange = [0.5*fc, 1.5*fc];

fprintf('Array: %d elements, %.3f mm pitch, fc = %.1f MHz\n', ...
    numElements, pitch*1000, fc/1e6);

%% ── Synthetic point-target signal ─────────────────────────
% A single point reflector at broadside, some depth in front of the
% array, illuminated by a short pulse. Signal is captured per-element
% with the correct inter-element delays for a wave arriving from that
% target, plus additive noise.
fs        = 40e6;                 % Sampling frequency (Hz), matches OASBUD
targetAng = [0; 0];                % Broadside (azimuth 0, elevation 0)
snr_dB    = 10;

collector = phased.WidebandCollector('Sensor', array, ...
    'SampleRate', fs, 'PropagationSpeed', c, 'ModulatedInput', false);

pulseLen = 64;
pulse    = sin(2*pi*fc*(0:pulseLen-1)'/fs) .* hamming(pulseLen);
sigLen   = 512;
txSignal = zeros(sigLen, 1);
txSignal(50:50+pulseLen-1) = pulse;

rxSignal = collector(txSignal, targetAng);
rxSignal = rxSignal + 0.1 * randn(size(rxSignal)) * norm(rxSignal(:)) / sqrt(numel(rxSignal)) / (10^(snr_dB/20));

fprintf('Synthetic received data: %d samples x %d elements\n', size(rxSignal,1), size(rxSignal,2));

%% ── Phased Array System Toolbox beamformer ────────────────
beamformer = phased.PhaseShiftBeamformer('SensorArray', array, ...
    'OperatingFrequency', fc, 'PropagationSpeed', c, ...
    'Direction', targetAng, 'WeightsOutputPort', false);

yBeamformed = beamformer(rxSignal);

fprintf('Beamformed output: %d samples\n', numel(yBeamformed));

%% ── Compare: single element vs beamformed ─────────────────
figure('Name', 'Phased Array Toolbox Demo', 'Position', [100 100 900 500]);

subplot(2,1,1);
plot(real(rxSignal(:,round(numElements/2))));
title('Single Element (centre), Real Part');
xlabel('Sample'); ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(real(yBeamformed));
title('phased.PhaseShiftBeamformer Output, Real Part');
xlabel('Sample'); ylabel('Amplitude');
grid on;

sgtitle('Phased Array System Toolbox — ULA + PhaseShiftBeamformer Demo');

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);
figOutputFolder = fullfile(repoRoot, 'Project Figures', 'Phase7');
if ~exist(figOutputFolder, 'dir'), mkdir(figOutputFolder); end
saveas(gcf, fullfile(figOutputFolder, 'phased_array_toolbox_demo.png'));

fprintf('\nSNR improvement (beamformed vs single element): %.2f dB\n', ...
    10*log10(var(real(yBeamformed)) / var(real(rxSignal(:,round(numElements/2))))));
fprintf('Figure saved to %s\n', figOutputFolder);
fprintf('\nNote: this script demonstrates the toolbox on synthetic array data.\n');
fprintf('It is not applied to OASBUD, whose RF columns are pre-formed scan\n');
fprintf('lines rather than raw per-element channel data. See README for details.\n');
