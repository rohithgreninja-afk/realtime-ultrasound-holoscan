% =========================================================
% Phase5_ReconstructionComparison.m
% Compare 6 reconstruction methods on OASBUD Patient 1
% Methods:
%   1. Original DAS (Phase 2 baseline)
%   2. DAS + TGC
%   3. DAS + TGC + Coherence Factor
%   4. DMAS (Delay-Multiply-and-Sum)
%   5. f-k Migration
%   6. MVBF (Minimum Variance Beamformer) — slow, runs last
% =========================================================

load('C:\Users\rohit\Downloads\Real Time Image Processing Project\OASBUD.mat');

% ── Acquisition parameters ───────────────────────────────
fs    = 25e6;
c     = 1540;
pitch = 0.245e-3;

% ── Patient 1 (Malignant) ────────────────────────────────
rf = data(1).rf1;
[num_samples, num_elements] = size(rf);
fprintf('RF size: %d x %d\n', num_samples, num_elements);

% ── Build grids ──────────────────────────────────────────
x_elements = (0:num_elements-1)*pitch;
x_elements = x_elements - mean(x_elements);
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
% METHOD 6: MVBF — WARNING: very slow (5-10 mins)
% ═══════════════════════════════════════════════════════
fprintf('Method 6: MVBF (slow — 5-10 mins)... ');
tic;
bmode_mvbf = beamform_mvbf(rf_tgc, x_elements, z_image, x_image, fs, c);
t6 = toc;
fprintf('done in %.2fs\n', t6);

% ── Plot all 6 methods ───────────────────────────────────
methods = {bmode_das, bmode_tgc, bmode_cf, bmode_dmas, bmode_fk, bmode_mvbf};
titles  = {'1. Original DAS', '2. DAS + TGC', '3. DAS+TGC+CF', ...
           '4. DMAS', '5. f-k Migration', '6. MVBF (Adaptive)'};
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
sgtitle('Phase 5 — Reconstruction Method Comparison (Patient 1 Malignant)', ...
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
