function results = Phase3_Enhancement(bmode, roi)
% Phase3_Enhancement  Speckle filter comparison used to select the deployed filter.
%
%   results = Phase3_Enhancement()            runs on the default patient (below)
%   results = Phase3_Enhancement(bmode, roi)  runs on a caller-supplied image
%
% Prerequisite: none when called without arguments. The input is built here
% from OASBUD record '43mw' (in the bundled sample, an OASBUD hold-out
% patient) with the same A-line reconstruction the pipeline uses
% (Hilbert envelope, power law gamma 0.3, uint8). OASBUD_PATH overrides the
% data file. bmode must be a uint8 image or a double image in [0, 1].
% roi is a struct with fields lesion, background, homogeneous ([r1 r2 c1 c2])
% and boundary (logical mask of the lesion boundary band).
%
% Metrics (filters are never scored against the noisy input):
%   Real data: CNR and gCNR between lesion and background ROI, speckle SNR
%              (mean/std) in a homogeneous ROI, edge-to-speckle gradient
%              ratio (mean gradient on the annotated lesion boundary divided by
%              mean gradient in the homogeneous ROI).
%   Phantom:   clean phantom with known regions, imnoise speckle at variance
%              0.01, 0.05 and 0.10, PSNR and SSIM against the clean phantom,
%              gradient magnitude correlation with the clean phantom.
% Selection rule: lowest mean rank over real CNR, gCNR, speckle SNR, edge
% ratio and phantom PSNR, SSIM, gradient correlation at variance 0.05.
%
% Output (RESULTS_DIR): enhancement_eval.mat, enhancement_eval.md,
%                       figures/Phase3/*.png

resultsDir = getenv('RESULTS_DIR');
if isempty(resultsDir)
    error('RESULTS_DIR environment variable not set. Set it with setenv(''RESULTS_DIR'', ''/path/to/results'').');
end
figDir = fullfile(resultsDir, 'figures', 'Phase3');
if ~exist(figDir, 'dir'), mkdir(figDir); end

if nargin == 0
    patientId = "43mw";
    oasbudPath = getenv('OASBUD_PATH');
    if isempty(oasbudPath)
        oasbudPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'sample', 'OASBUD_sample.mat');
    end
    d   = load(oasbudPath);
    rec = d.data(arrayfun(@(s) string(s.id), d.data) == patientId);
    assert(isscalar(rec), 'OASBUD record %s not found in %s', patientId, oasbudPath);
    env   = abs(hilbert(double(rec.rf1)));
    bmode = uint8((env / (max(env(:)) + 1e-12)) .^ 0.3 * 255);
    % ROIs in pixel coordinates [row1 row2 col1 col2] of the 1824 x 510 image
    roi.lesion      = [780 940 250 320];   % inside the annotated hypoechoic lesion
    roi.background  = [780 940 380 450];   % parenchyma at the same depth, right of the lesion
    roi.homogeneous = [780 940 100 170];   % parenchyma at the same depth, left of the lesion
    % Boundary band: annotated lesion outline (OASBUD roi1) dilated by 4 pixels
    roi.boundary    = imdilate(bwperim(rec.roi1), strel('disk', 4));
end
img = im2double(bmode);

%% Candidate filters (all Image Processing Toolbox)
names = ["None", "Median 3x3", "Wiener 5x5", "Bilateral"];
filters = { ...
    @(I) I, ...
    @(I) im2double(medfilt2(im2uint8(I), [3 3])), ...
    @(I) im2double(wiener2(im2uint8(I), [5 5])), ...
    @(I) imbilatfilt(I, 'DegreeOfSmoothing', 0.1, 'SpatialSigma', 2)};
nF = numel(names);

%% Real-data metrics
realOut = cell(1, nF);
cnrR = zeros(1, nF); gcnrR = zeros(1, nF); ssnrR = zeros(1, nF); egrR = zeros(1, nF);
for f = 1:nF
    J = filters{f}(img);
    realOut{f} = J;
    L = crop(J, roi.lesion); B = crop(J, roi.background); H = crop(J, roi.homogeneous);
    cnrR(f)  = abs(mean(L) - mean(B)) / sqrt(var(L) + var(B));
    gcnrR(f) = gcnr(L, B);
    ssnrR(f) = mean(H) / std(H);
    G = imgradient(J);
    egrR(f)  = mean(G(roi.boundary)) / mean(crop(G, roi.homogeneous));
end

%% Synthetic phantom with known ground truth
[cc, rr] = meshgrid(1:400, 1:400);
clean = 0.5 * ones(400);
clean((rr-200).^2 + (cc-130).^2 < 60^2) = 0.15;   % hypoechoic lesion
clean((rr-200).^2 + (cc-290).^2 < 40^2) = 0.85;   % hyperechoic lesion
phLesion = [170 230 100 160]; phBack = [40 100 170 230];
variances = [0.01 0.05 0.10];
psnrP = zeros(numel(variances), nF); ssimP = psnrP; gcorrP = psnrP; cnrP = psnrP;
Gclean = imgradient(clean);
rng(0);
for v = 1:numel(variances)
    noisy = imnoise(clean, 'speckle', variances(v));
    for f = 1:nF
        J = filters{f}(noisy);
        if v == 2, phOut{f} = J; noisy05 = noisy; end %#ok<AGROW>
        psnrP(v,f)  = psnr(J, clean);
        ssimP(v,f)  = ssim(J, clean);
        G = imgradient(J);
        gcorrP(v,f) = corr(G(:), Gclean(:));
        L = crop(J, phLesion); B = crop(J, phBack);
        cnrP(v,f)   = abs(mean(L) - mean(B)) / sqrt(var(L) + var(B));
    end
end

%% Selection among the real filters (column 1 is the unfiltered baseline)
crit = [cnrR; gcnrR; ssnrR; egrR; psnrP(2,:); ssimP(2,:); gcorrP(2,:)];
critNames = ["Real CNR","Real gCNR","Real speckle SNR","Real edge ratio", ...
             "Phantom PSNR (v=0.05)","Phantom SSIM (v=0.05)","Phantom grad corr (v=0.05)"];
ranks = zeros(size(crit, 1), nF - 1);
for k = 1:size(crit, 1)
    [~, order] = sort(crit(k, 2:end), 'descend');
    ranks(k, order) = 1:nF-1;
end
meanRank = mean(ranks, 1);
[~, best] = min(meanRank);
selected = names(best + 1);

%% Print and save
fprintf('\n%-28s', 'Metric'); fprintf('%14s', names); fprintf('\n');
for k = 1:size(crit, 1)
    fprintf('%-28s', critNames(k)); fprintf('%14.4f', crit(k, :)); fprintf('\n');
end
fprintf('%-28s%14s', 'Mean rank (filters only)', '-'); fprintf('%14.2f', meanRank); fprintf('\n');
fprintf('Selected filter: %s\n', selected);

results = struct('names', names, 'cnrReal', cnrR, 'gcnrReal', gcnrR, 'speckleSNRReal', ssnrR, ...
    'edgeRatioReal', egrR, 'variances', variances, 'psnrPhantom', psnrP, 'ssimPhantom', ssimP, ...
    'gradCorrPhantom', gcorrP, 'cnrPhantom', cnrP, 'meanRank', meanRank, 'selected', selected, 'roi', roi);
save(fullfile(resultsDir, 'enhancement_eval.mat'), '-struct', 'results');

fid = fopen(fullfile(resultsDir, 'enhancement_eval.md'), 'w');
fprintf(fid, '# Enhancement filter comparison\n\nProduced by `MATLAB Codes/Phase3_Enhancement.m`.\n\n');
fprintf(fid, '## Real data (OASBUD hold-out patient, A-line B-mode)\n\nROIs [row1 row2 col1 col2]: lesion %s, background %s, homogeneous %s; boundary band from the OASBUD lesion outline.\n\n', ...
    mat2str(roi.lesion), mat2str(roi.background), mat2str(roi.homogeneous));
fprintf(fid, '| Filter | CNR | gCNR | Speckle SNR | Edge ratio |\n|---|---|---|---|---|\n');
for f = 1:nF
    fprintf(fid, '| %s | %.4f | %.4f | %.4f | %.4f |\n', names(f), cnrR(f), gcnrR(f), ssnrR(f), egrR(f));
end
fprintf(fid, '\n## Synthetic phantom (scored against the clean phantom)\n\n| Filter | Variance | PSNR (dB) | SSIM | Gradient corr | CNR |\n|---|---|---|---|---|---|\n');
for v = 1:numel(variances)
    for f = 1:nF
        fprintf(fid, '| %s | %.2f | %.2f | %.4f | %.4f | %.4f |\n', names(f), variances(v), ...
            psnrP(v,f), ssimP(v,f), gcorrP(v,f), cnrP(v,f));
    end
end
fprintf(fid, '\nMean rank over the seven criteria (1 is best): %s\n\nSelected filter: **%s**\n', ...
    strjoin(names(2:end) + " " + compose('%.2f', meanRank), ', '), selected);
fclose(fid);

%% Figures (captured handles)
fig1 = figure('Position', [50 50 1400 520]);
tiledlayout(fig1, 1, nF, 'TileSpacing', 'compact');
for f = 1:nF
    nexttile; imagesc(realOut{f}, [0 1]); colormap gray; axis image off; hold on;
    drawRect(roi.lesion, 'r'); drawRect(roi.background, 'g'); drawRect(roi.homogeneous, 'c');
    title(sprintf('%s\ngCNR %.3f  sSNR %.2f', names(f), gcnrR(f), ssnrR(f)));
end
sgtitle(fig1, 'Real data: lesion (red), background (green), homogeneous (cyan) ROIs');
saveas(fig1, fullfile(figDir, 'Phase3_RealData_Comparison.png'));

fig2 = figure('Position', [50 50 1400 520]);
tiledlayout(fig2, 1, nF + 1, 'TileSpacing', 'compact');
nexttile; imshow(clean); title('Clean phantom');
for f = 1:nF
    nexttile; imshow(phOut{f}); title(sprintf('%s\nPSNR %.2f  SSIM %.3f', names(f), psnrP(2,f), ssimP(2,f)));
end
sgtitle(fig2, 'Phantom, speckle variance 0.05 (None = noisy input)');
saveas(fig2, fullfile(figDir, 'Phase3_Phantom_Comparison.png'));

fig3 = figure('Position', [50 50 1000 520]);
tiledlayout(fig3, 1, 2, 'TileSpacing', 'compact');
nexttile; imagesc(img, [0 1]); colormap gray; axis image off; title('Unfiltered B-mode');
nexttile; imagesc(realOut{best + 1}, [0 1]); colormap gray; axis image off;
title(sprintf('Selected: %s (mean rank %.2f)', selected, meanRank(best)));
sgtitle(fig3, 'Phase 3 selected enhancement');
saveas(fig3, fullfile(figDir, 'Phase3_Final_Summary.png'));
fprintf('Figures saved to %s\n', figDir);
end

function v = crop(I, r)
v = reshape(I(r(1):r(2), r(3):r(4)), [], 1);
end

function g = gcnr(a, b)
% Generalised CNR: 1 minus the overlap of the two intensity histograms
edges = linspace(0, 1, 257);
pa = histcounts(a, edges, 'Normalization', 'probability');
pb = histcounts(b, edges, 'Normalization', 'probability');
g = 1 - sum(min(pa, pb));
end

function drawRect(r, c)
rectangle('Position', [r(3) r(1) r(4)-r(3) r(2)-r(1)], 'EdgeColor', c, 'LineWidth', 1.5);
end
