% =========================================================
% Phase5_ReconstructionComparison.m
% Compare 6 reconstruction methods on OASBUD Patient 1
% Methods:
%   1. Original DAS (Phase 2 baseline)
%   2. DAS + TGC
%   3. DAS + TGC + Coherence Factor
%   4. DMAS (Delay-Multiply-and-Sum)
%   5. f-k Migration
%   6. MVDR (phased.MVDRBeamformer, Phased Array System Toolbox), runs last
% =========================================================

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);
oasbudPath = getenv('OASBUD_PATH');
if isempty(oasbudPath)
    oasbudPath = fullfile(repoRoot, 'data', 'sample', 'OASBUD_sample.mat');
    fprintf('OASBUD_PATH not set, using bundled sample: %s\n', oasbudPath);
end
load(oasbudPath);

% ── Acquisition parameters ───────────────────────────────
fs    = 40e6;
c     = 1540;
pitch = 0.30e-3;

% ── Patient 1 (Malignant) ────────────────────────────────
rf = data(1).rf1;
[num_samples, num_elements] = size(rf);
fprintf('RF size: %d x %d\n', num_samples, num_elements);

% ── Build grids ──────────────────────────────────────────
% Element positions from the array model (phased.ULA is centred on zero)
ula = phased.ULA('NumElements', num_elements, 'ElementSpacing', pitch);
elemPos = getElementPosition(ula);
x_elements = elemPos(2, :);
depth_axis  = (0:num_samples-1)*(c/(2*fs));
x_image     = linspace(x_elements(1), x_elements(end), 200);
z_image     = linspace(depth_axis(1),  depth_axis(end), 300);

% ── TGC ──────────────────────────────────────────────────
tgc_gain = linspace(1, 3, num_samples)';
rf_tgc   = rf .* tgc_gain;

% ═══════════════════════════════════════════════════════
% METHOD 1: Original DAS
% ═══════════════════════════════════════════════════════
fprintf('Method 1: Original DAS... ');
tic;
bmode_das = das_beamform(rf, x_elements, z_image, x_image, fs, c);
t1 = toc;
fprintf('done in %.2fs\n', t1);

% ═══════════════════════════════════════════════════════
% METHOD 2: DAS + TGC
% ═══════════════════════════════════════════════════════
fprintf('Method 2: DAS + TGC... ');
tic;
bmode_tgc = das_beamform(rf_tgc, x_elements, z_image, x_image, fs, c);
t2 = toc;
fprintf('done in %.2fs\n', t2);

% ═══════════════════════════════════════════════════════
% METHOD 3: DAS + TGC + Coherence Factor
% ═══════════════════════════════════════════════════════
fprintf('Method 3: DAS + TGC + CF (may take 1-2 min)... ');
tic;
bmode_cf = das_beamform_cf(rf_tgc, x_elements, z_image, x_image, fs, c);
t3 = toc;
fprintf('done in %.2fs\n', t3);

% ═══════════════════════════════════════════════════════
% METHOD 4: DMAS
% ═══════════════════════════════════════════════════════
fprintf('Method 4: DMAS (may take 2-3 min)... ');
tic;
bmode_dmas = beamform_dmas(rf_tgc, x_elements, z_image, x_image, fs, c);
t4 = toc;
fprintf('done in %.2fs\n', t4);

% ═══════════════════════════════════════════════════════
% METHOD 5: f-k Migration
% ═══════════════════════════════════════════════════════
fprintf('Method 5: f-k Migration... ');
tic;
bmode_fk = beamform_fk(rf_tgc, x_elements, z_image, x_image, fs, c);
t5 = toc;
fprintf('done in %.2fs\n', t5);

% ═══════════════════════════════════════════════════════
% METHOD 6: MVDR (Phased Array System Toolbox phased.MVDRBeamformer)
% ═══════════════════════════════════════════════════════
% Each image column is delay-aligned to its focal points, so the array is
% steered to broadside; the column's depth samples are the snapshots used
% for the covariance estimate.
fprintf('Method 6: MVDR (phased.MVDRBeamformer, may take a few minutes)... ');
tic;
rfc_tgc = hilbert(rf_tgc);
rfc_tgc = rfc_tgc / sqrt(mean(abs(rfc_tgc(:)).^2));   % unit power so the loading factor is relative
mvdr = phased.MVDRBeamformer('SensorArray', ula, 'OperatingFrequency', 10e6, ...
    'PropagationSpeed', c, 'Direction', [0; 0], 'DiagonalLoadingFactor', 0.01);
mvdr_env = zeros(numel(z_image), numel(x_image));
for ix = 1:numel(x_image)
    dist = sqrt((x_image(ix) - x_elements).^2 + z_image(:).^2);
    idx  = min(max(round((2 * dist / c) * fs) + 1, 1), num_samples);
    aligned = rfc_tgc(idx + (0:num_elements-1) * num_samples);
    mvdr_env(:, ix) = abs(mvdr(aligned));
end
bmode_mvdr = 20*log10(mvdr_env + eps);
bmode_mvdr = max(bmode_mvdr - max(bmode_mvdr(:)), -60);
t6 = toc;
fprintf('done in %.2fs\n', t6);

% ── Plot all 6 methods ───────────────────────────────────
methods = {bmode_das, bmode_tgc, bmode_cf, bmode_dmas, bmode_fk, bmode_mvdr};
titles  = {'1. Original DAS', '2. DAS + TGC', '3. DAS+TGC+CF', ...
           '4. DMAS', '5. f-k Migration', '6. MVDR (phased.MVDRBeamformer)'};
times   = {t1, t2, t3, t4, t5, t6};

figure('Position', [50 50 1600 800]);
for i = 1:6
    subplot(2, 3, i);
    imagesc(x_image*1000, z_image*1000, methods{i});
    colormap gray; colorbar;
    title(sprintf('%s\n(%.1fs)', titles{i}, times{i}), 'FontSize', 9);
    xlabel('Lateral (mm)'); ylabel('Depth (mm)');
    caxis([-60 0]);
end
sgtitle('Phase 5 -- Reconstruction Method Comparison (Patient 1 Malignant)', ...
        'FontSize', 12, 'FontWeight', 'bold');

% ── Compute quality metrics ──────────────────────────────
fprintf('\n========= Quality Metrics =========\n');
fprintf('%-25s  %8s\n', 'Method', 'Dynamic Range (dB)');
fprintf('%s\n', repmat('-', 1, 40));
for i = 1:6
    dr = max(methods{i}(:)) - min(methods{i}(:));
    fprintf('%-25s  %8.2f\n', titles{i}, dr);
end
fprintf('\nHigher dynamic range = more visible tissue contrast.\n');
fprintf('Figures generated. Compare arc artefact suppression.\n');
