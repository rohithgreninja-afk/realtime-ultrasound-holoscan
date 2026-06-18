% =========================================================
% Beamforming + Comparison Script
% Real-Time Medical Image Processing Project
% =========================================================

% Load data
load('C:\Users\rohit\Downloads\Real Time Image Processing Project\OASBUD.mat');

% Define acquisition parameters
fs    = 25e6;       % Sampling frequency: 25 MHz
c     = 1540;       % Speed of sound in soft tissue (m/s)
pitch = 0.245e-3;   % Element spacing: 0.245 mm

% ── Element and image grid ────────────────────────────────
rf = data(1).rf1;
[num_samples, num_elements] = size(rf);

x_elements = (0:num_elements-1) * pitch;
x_elements = x_elements - mean(x_elements);

depth_axis = (0:num_samples-1) * (c / (2 * fs));

x_image = linspace(x_elements(1), x_elements(end), 200);
z_image = linspace(depth_axis(1),  depth_axis(end), 300);

% ── Beamform Patient 1 (Malignant) ───────────────────────
if ~exist('bmode', 'var')
    fprintf('Beamforming Patient 1 (Malignant)...\n');
    bmode = zeros(length(z_image), length(x_image));

    for iz = 1:length(z_image)
        for ix = 1:length(x_image)
            pixel_sum = 0;
            for ie = 1:num_elements
                dx = x_image(ix) - x_elements(ie);
                dz = z_image(iz);
                dist = sqrt(dx^2 + dz^2);
                sample_idx = round((dist / c) * fs) + 1;
                if sample_idx >= 1 && sample_idx <= num_samples
                    pixel_sum = pixel_sum + rf(sample_idx, ie);
                end
            end
            bmode(iz, ix) = pixel_sum;
        end
        if mod(iz, 50) == 0
            fprintf('  Row %d of 300 done\n', iz);
        end
    end
    fprintf('Beamforming complete.\n');
else
    disp('bmode already exists, skipping Patient 1 beamform.');
end

% ── Process Patient 1 ────────────────────────────────────
envelope   = abs(hilbert(bmode));
bmode_log  = 20 * log10(envelope + 1);
bmode_log  = bmode_log - max(bmode_log(:));
bmode_log(bmode_log < -60) = -60;

fprintf('Patient 1 stats:\n');
fprintf('  bmode size:   %d x %d\n',  size(bmode,1), size(bmode,2));
fprintf('  Min value:    %.2f\n',      min(bmode(:)));
fprintf('  Max value:    %.2f\n',      max(bmode(:)));
fprintf('  Envelope max: %.2f\n',      max(envelope(:)));
fprintf('  Log range:    %.2f to %.2f dB\n', min(bmode_log(:)), max(bmode_log(:)));

% ── Find first benign patient ─────────────────────────────
benign_idx = find([data.class] == 1, 1);
fprintf('\nUsing Patient %d as benign case\n', benign_idx);
fprintf('Patient 4 rf1 size: %d x %d\n', size(data(benign_idx).rf1));
fprintf('Patient 4 rf1 max:  %.2f\n', max(max(abs(data(benign_idx).rf1))));

% ── Beamform Benign Patient ───────────────────────────────
if ~exist('bmode_b', 'var')
    fprintf('Beamforming Patient %d (Benign)...\n', benign_idx);
    rf_b   = data(benign_idx).rf1;
    bmode_b = zeros(length(z_image), length(x_image));

    for iz = 1:length(z_image)
        for ix = 1:length(x_image)
            pixel_sum = 0;
            for ie = 1:num_elements
                dx = x_image(ix) - x_elements(ie);
                dz = z_image(iz);
                dist = sqrt(dx^2 + dz^2);
                sample_idx = round((dist / c) * fs) + 1;
                if sample_idx >= 1 && sample_idx <= num_samples
                    pixel_sum = pixel_sum + rf_b(sample_idx, ie);
                end
            end
            bmode_b(iz, ix) = pixel_sum;
        end
        if mod(iz, 50) == 0
            fprintf('  Row %d of 300 done\n', iz);
        end
    end
    fprintf('Beamforming complete.\n');
else
    disp('bmode_b already exists, skipping benign beamform.');
end

% ── Process Benign Patient ────────────────────────────────
envelope_b  = abs(hilbert(bmode_b));
bmode_log_b = 20 * log10(envelope_b + 1);
bmode_log_b = bmode_log_b - max(bmode_log_b(:));
bmode_log_b(bmode_log_b < -60) = -60;

% ── Plot Comparison ───────────────────────────────────────
figure;
subplot(1,2,1);
imagesc(x_image*1000, z_image*1000, bmode_log, [-60 0]);
colormap gray; colorbar;
title('Patient 1 — Malignant');
xlabel('Lateral (mm)');
ylabel('Depth (mm)');

subplot(1,2,2);
imagesc(x_image*1000, z_image*1000, bmode_log_b, [-60 0]);
colormap gray; colorbar;
title(sprintf('Patient %d — Benign', benign_idx));
xlabel('Lateral (mm)');
ylabel('Depth (mm)');

sgtitle('B-Mode Comparison: Malignant vs Benign', ...
        'FontSize', 13, 'FontWeight', 'bold');