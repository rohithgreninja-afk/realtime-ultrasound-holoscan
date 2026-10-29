% run_enhancement_subsystem.m
% Driver for UltrasoundEnhancementSubsystem.slx. Loads one OASBUD patient,
% builds rf_ts for the From Workspace block, simulates the model and shows
% the enhanced B-mode written by the To Workspace block (enhanced_bmode).
%
% Data: OASBUD_PATH, or the bundled data/sample/OASBUD_sample.mat if unset.
% Output: RESULTS_DIR/figures/Simulink/Simulink_Functional_Output.png
% Check: the script errors unless the output is finite, the size of the
% input RF matrix, and not all zero.

clearvars; close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);
resultsDir = getenv('RESULTS_DIR');
if isempty(resultsDir)
    error('RESULTS_DIR environment variable not set. Set it with setenv(''RESULTS_DIR'', ''/path/to/results'').');
end
oasbudPath = getenv('OASBUD_PATH');
if isempty(oasbudPath)
    oasbudPath = fullfile(repoRoot, 'data', 'sample', 'OASBUD_sample.mat');
    fprintf('OASBUD_PATH not set, using bundled sample: %s\n', oasbudPath);
end

%% One patient, timeseries shaped [nSamples nElements nTime]
d  = load(oasbudPath);
patIdx = 1;
rf = double(d.data(patIdx).rf1);
rf_ts = timeseries(reshape(rf, [size(rf) 1]), 0);
fprintf('Patient %s: RF %d x %d\n', string(d.data(patIdx).id), size(rf, 1), size(rf, 2));

%% Simulate
model = 'UltrasoundEnhancementSubsystem';
% Keep Simulink build artifacts (slprj, .slxc) inside RESULTS_DIR
cacheDir = fullfile(resultsDir, 'simulink_cache');
Simulink.fileGenControl('set', 'CacheFolder', cacheDir, 'CodeGenFolder', cacheDir, 'createDir', true);
load_system(fullfile(scriptDir, [model '.slx']));
simIn = Simulink.SimulationInput(model);
simIn = setVariable(simIn, 'rf_ts', rf_ts);
out = sim(simIn);
enhanced = out.enhanced_bmode;
if isa(enhanced, 'timeseries'), enhanced = enhanced.Data; end
enhanced = enhanced(:, :, end);
close_system(model, 0);

%% Check
assert(isequal(size(enhanced), size(rf)), 'Output size %s does not match RF size %s', ...
    mat2str(size(enhanced)), mat2str(size(rf)));
assert(all(isfinite(enhanced(:))), 'Output contains non-finite values');
assert(nnz(enhanced) > 0, 'Output is all zero');
fprintf('Simulink output: %d x %d, range %.2f to %.2f, %.1f%% non-zero pixels\n', ...
    size(enhanced, 1), size(enhanced, 2), min(enhanced(:)), max(enhanced(:)), 100*nnz(enhanced)/numel(enhanced));

%% Display and save
fig = figure('Position', [100 100 500 800]);
imagesc(enhanced, [0 255]); colormap gray; colorbar;
xlabel('Scan line'); ylabel('Sample (depth)');
title(sprintf('UltrasoundEnhancementSubsystem output, patient %s', string(d.data(patIdx).id)));
figDir = fullfile(resultsDir, 'figures', 'Simulink');
if ~exist(figDir, 'dir'), mkdir(figDir); end
saveas(fig, fullfile(figDir, 'Simulink_Functional_Output.png'));
fprintf('Saved %s\n', fullfile(figDir, 'Simulink_Functional_Output.png'));
