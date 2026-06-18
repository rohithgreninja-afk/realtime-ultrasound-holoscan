% =========================================================
% Phase3_Enhancement.m
% Image Enhancement for Ultrasound B-Mode Images
% =========================================================

% Convert bmode_log to uint8 for image processing toolbox functions
% The toolbox functions expect values in 0-255 range
bmode_norm = mat2gray(bmode_log);          % normalise to 0-1
bmode_uint8 = im2uint8(bmode_norm);        % convert to 0-255 uint8

fprintf('Input image:\n');
fprintf('  Size:  %d x %d\n', size(bmode_uint8,1), size(bmode_uint8,2));
fprintf('  Class: %s\n', class(bmode_uint8));
fprintf('  Range: %d to %d\n', min(bmode_uint8(:)), max(bmode_uint8(:)));

% ── Step 1: Median Filter (Speckle Reduction) ─────────────
bmode_median = medfilt2(bmode_uint8, [3 3]);

fprintf('\nAfter median filter:\n');
fprintf('  Range: %d to %d\n', min(bmode_median(:)), max(bmode_median(:)));

% ── Visualise side by side ────────────────────────────────
figure;
subplot(1,2,1);
imshow(bmode_uint8);
title('Original B-Mode');

subplot(1,2,2);
imshow(bmode_median);
title('After Median Filter (3x3)');

sgtitle('Phase 3 Step 1 — Speckle Reduction: Median Filter');

% ── Step 2: Wiener Filter ─────────────────────────────────
bmode_wiener = wiener2(bmode_uint8, [5 5]);

fprintf('After Wiener filter:\n');
fprintf('  Range: %d to %d\n', min(bmode_wiener(:)), max(bmode_wiener(:)));

% ── Compare median vs wiener ──────────────────────────────
figure;
subplot(1,3,1);
imshow(bmode_uint8);
title('Original');

subplot(1,3,2);
imshow(bmode_median);
title('Median Filter (3x3)');

subplot(1,3,3);
imshow(bmode_wiener);
title('Wiener Filter (5x5)');

sgtitle('Phase 3 Step 2 — Median vs Wiener Comparison');

%Axis
figure;
subplot(1,3,1);
imagesc(x_image*1000, z_image*1000, bmode_uint8);
colormap gray; colorbar;
title('Original');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

subplot(1,3,2);
imagesc(x_image*1000, z_image*1000, bmode_median);
colormap gray; colorbar;
title('Median Filter (3x3)');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

subplot(1,3,3);
imagesc(x_image*1000, z_image*1000, bmode_wiener);
colormap gray; colorbar;
title('Wiener Filter (5x5)');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

sgtitle('Phase 3 Step 2 — Median vs Wiener Comparison');

% ── Step 3: Adaptive Histogram Equalization ───────────────
% Apply on top of the median filtered image
bmode_median_norm = im2double(bmode_median);  % convert back to 0-1
bmode_ahe = adapthisteq(bmode_median_norm, 'ClipLimit', 0.01, 'NumTiles', [8 8]);

fprintf('After adaptive histogram equalization:\n');
fprintf('  Range: %.4f to %.4f\n', min(bmode_ahe(:)), max(bmode_ahe(:)));

figure;
subplot(1,2,1);
imagesc(x_image*1000, z_image*1000, bmode_median);
colormap gray; colorbar;
title('Median Filter Only');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

subplot(1,2,2);
imagesc(x_image*1000, z_image*1000, bmode_ahe);
colormap gray; colorbar;
title('Median + Adaptive Histogram EQ');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

sgtitle('Phase 3 Step 3 — Contrast Enhancement');

% Try three different clip limits to find the best balance
bmode_ahe_low  = adapthisteq(bmode_median_norm, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
bmode_ahe_mid  = adapthisteq(bmode_median_norm, 'ClipLimit', 0.05, 'NumTiles', [8 8]);
bmode_ahe_high = adapthisteq(bmode_median_norm, 'ClipLimit', 0.10, 'NumTiles', [8 8]);

figure;
subplot(1,4,1);
imagesc(x_image*1000, z_image*1000, bmode_median);
colormap gray; colorbar;
title('Median Only');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

subplot(1,4,2);
imagesc(x_image*1000, z_image*1000, bmode_ahe_low);
colormap gray; colorbar;
title('ClipLimit 0.02');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

subplot(1,4,3);
imagesc(x_image*1000, z_image*1000, bmode_ahe_mid);
colormap gray; colorbar;
title('ClipLimit 0.05');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

subplot(1,4,4);
imagesc(x_image*1000, z_image*1000, bmode_ahe_high);
colormap gray; colorbar;
title('ClipLimit 0.10');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

sgtitle('Phase 3 Step 3 — Adaptive Histogram EQ: ClipLimit Comparison');

% Use median filtered image as input for bilateral filter
% Reset to double for imbilatfilt
bmode_for_bilateral = im2double(bmode_median);

fprintf('Ready for Step 4: Bilateral Filter\n');
fprintf('Input range: %.4f to %.4f\n', min(bmode_for_bilateral(:)), max(bmode_for_bilateral(:)));

% ── Step 4: Bilateral Filter (Edge-Preserving Smoothing) ──
% DegreeOfSmoothing controls intensity similarity threshold
% SpatialSigma controls the spatial neighbourhood size
bmode_bilateral = imbilatfilt(bmode_for_bilateral, ...
    'DegreeOfSmoothing', 0.1, ...
    'SpatialSigma', 2);

fprintf('After bilateral filter:\n');
fprintf('  Range: %.4f to %.4f\n', min(bmode_bilateral(:)), max(bmode_bilateral(:)));

% ── Compare all stages ────────────────────────────────────
figure;
subplot(1,3,1);
imagesc(x_image*1000, z_image*1000, bmode_uint8);
colormap gray; colorbar;
title('Original B-Mode');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

subplot(1,3,2);
imagesc(x_image*1000, z_image*1000, bmode_median);
colormap gray; colorbar;
title('After Median Filter');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

subplot(1,3,3);
imagesc(x_image*1000, z_image*1000, bmode_bilateral);
colormap gray; colorbar;
title('After Bilateral Filter');
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

sgtitle('Phase 3 Step 4 — Bilateral Filter Comparison');

ref   = im2double(bmode_uint8);
med   = im2double(bmode_median);
wien  = im2double(bmode_wiener);
bilat = bmode_bilateral;

psnr_median    = psnr(med,   ref);
psnr_wiener    = psnr(wien,  ref);
psnr_bilateral = psnr(bilat, ref);

ssim_median    = ssim(med,   ref);
ssim_wiener    = ssim(wien,  ref);
ssim_bilateral = ssim(bilat, ref);

fprintf('\n========= Image Quality Metrics =========\n');
fprintf('%-20s  %8s  %8s\n', 'Filter', 'PSNR(dB)', 'SSIM');
fprintf('%s\n', repmat('-',1,40));
fprintf('%-20s  %8.2f  %8.4f\n', 'Median (3x3)',  psnr_median,    ssim_median);
fprintf('%-20s  %8.2f  %8.4f\n', 'Wiener (5x5)',  psnr_wiener,    ssim_wiener);
fprintf('%-20s  %8.2f  %8.4f\n', 'Bilateral',     psnr_bilateral, ssim_bilateral);
fprintf('=========================================\n');


% ── Final: Save enhanced image and produce summary figure ─
bmode_enhanced = bmode_median;

% Save for use in Phase 4
save('bmode_enhanced.mat', 'bmode_enhanced', 'x_image', 'z_image');
fprintf('Enhanced image saved to bmode_enhanced.mat\n');

% Final summary figure
figure;
subplot(1,2,1);
imagesc(x_image*1000, z_image*1000, bmode_uint8);
colormap gray; colorbar;
title(sprintf('Original B-Mode\nRange: 0-255'));
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

subplot(1,2,2);
imagesc(x_image*1000, z_image*1000, bmode_enhanced);
colormap gray; colorbar;
title(sprintf('Enhanced (Median 3x3)\nPSNR: %.2f dB  |  SSIM: %.4f', ...
    psnr_median, ssim_median));
xlabel('Lateral (mm)'); ylabel('Depth (mm)');

sgtitle('Phase 3 Final — Selected Enhancement Pipeline', ...
    'FontSize', 13, 'FontWeight', 'bold');

fprintf('\nPhase 3 complete. Selected filter: Median 3x3\n');
fprintf('PSNR: %.2f dB  |  SSIM: %.4f\n', psnr_median, ssim_median);

% Save all Phase 3 figures to disk
outputFolder = 'C:\Users\rohit\Documents\MATLAB Code\Project_Figures\Phase3';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

figure(2); saveas(gcf, fullfile(outputFolder, 'Step1_Median_Filter.png'));
figure(3); saveas(gcf, fullfile(outputFolder, 'Step2_Median_vs_Wiener.png'));
figure(5); saveas(gcf, fullfile(outputFolder, 'Step3_ClipLimit_Comparison.png'));
figure(6); saveas(gcf, fullfile(outputFolder, 'Step4_Bilateral_Filter.png'));
figure(24); saveas(gcf, fullfile(outputFolder, 'Phase3_Final_Summary.png'));

fprintf('All Phase 3 figures saved to %s\n', outputFolder);