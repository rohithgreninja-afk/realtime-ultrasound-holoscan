% Phase5_BeamformingImproved.m
% Reconstruction comparison with fixed normalization and compression options
% Save as: C:\Users\rohit\Documents\MATLAB Code\Phase5_BeamformingImproved.m

clearvars; clc;

%% -------------------------------------------------------------------------
% SECTION 1: Load OASBUD data
% -------------------------------------------------------------------------
data   = load('C:\Users\rohit\Downloads\Real Time Image Processing Project\OASBUD.mat');

patIdx = 1;
rf     = double(data.data(patIdx).rf1);
fs     = 40e6;
c      = 1540;

N          = size(rf, 2);
pitch      = 0.3e-3;
x_elements = ((0:N-1) - (N-1)/2) * pitch;

num_samples = size(rf, 1);
depth_max   = (num_samples / fs) * c / 2;
z_image     = linspace(1e-3, depth_max, 300);
x_image     = linspace(-60e-3, 60e-3, 300);
nz          = length(z_image);
nx          = length(x_image);

fprintf('Patient %d | RF size: %dx%d | Depth: %.1f mm | Class: %d\n', ...
    patIdx, size(rf,1), size(rf,2), depth_max*1000, data.data(patIdx).class);

%% -------------------------------------------------------------------------
% SECTION 2: Delay computation (shared across all methods)
% -------------------------------------------------------------------------
dx2  = reshape(x_image, [1 nx 1]) - reshape(x_elements, [1 1 N]);
dz2  = reshape(z_image, [nz 1 1]);
dist = sqrt(dx2.^2 + dz2.^2);

sample_idx = round((dist / c) * fs) + 1;
sample_idx = max(1, min(size(rf,1), sample_idx));

%% -------------------------------------------------------------------------
% SECTION 3: DAS beamforming
% -------------------------------------------------------------------------
fprintf('Running DAS...\n'); tic;

rf_complex = hilbert(rf);
das_complex = zeros(nz, nx);
for ie = 1:N
    idx = sample_idx(:,:,ie);
    das_complex = das_complex + reshape(rf_complex(idx, ie), nz, nx);
end
fprintf('DAS done: %.2fs\n', toc);

%% -------------------------------------------------------------------------
% SECTION 4: DAS+CF beamforming
% -------------------------------------------------------------------------
fprintf('Running DAS+CF...\n'); tic;

delayed_complex = zeros(nz, nx, N);
for ie = 1:N
    idx = sample_idx(:,:,ie);
    delayed_complex(:,:,ie) = reshape(rf_complex(idx, ie), nz, nx);
end

coh_sum   = abs(sum(delayed_complex, 3)).^2;
incoh_sum = N * sum(abs(delayed_complex).^2, 3) + eps;
cf        = max(0, min(1, coh_sum ./ incoh_sum));

dascf_complex = sum(delayed_complex, 3) .* cf;
fprintf('DAS+CF done: %.2fs\n', toc);

%% -------------------------------------------------------------------------
% SECTION 5: Envelopes
% -------------------------------------------------------------------------
env_das   = abs(das_complex);
env_dascf = abs(dascf_complex);

%% -------------------------------------------------------------------------
% SECTION 6: Compression functions
% -------------------------------------------------------------------------
function bmode = compress_log(env)
    env_norm = env / (max(env(:)) + eps);
    bmode    = 20 * log10(env_norm + eps);
    bmode    = max(bmode, -60);
    bmode    = bmode / 60 + 1;   % scale to [0 1] for display
end

function bmode = compress_power(env, gamma)
    bmode = (env / (max(env(:)) + eps)) .^ gamma;
end

function bmode = compress_sigmoid(env, k)
    env_norm = env / (max(env(:)) + eps);
    bmode    = 1 ./ (1 + exp(-k * (env_norm - 0.5)));
end

%% -------------------------------------------------------------------------
% SECTION 7: Apply all compressions
% -------------------------------------------------------------------------
bmode_das_log    = compress_log(env_das);
bmode_dascf_log  = compress_log(env_dascf);

bmode_das_pow    = compress_power(env_das,   0.3);
bmode_dascf_pow  = compress_power(env_dascf, 0.3);

bmode_das_sig    = compress_sigmoid(env_das,   10);
bmode_dascf_sig  = compress_sigmoid(env_dascf, 10);

%% -------------------------------------------------------------------------
% SECTION 8: Display 2x3 grid
% -------------------------------------------------------------------------
x_mm = x_image * 1000;
z_mm = z_image * 1000;

figure('Name', 'Beamforming + Compression Comparison', ...
       'Position', [50 50 1400 800]);

titles = {'DAS -- Log',          'DAS -- Power (γ=0.3)',    'DAS -- Sigmoid', ...
          'DAS+CF -- Log',       'DAS+CF -- Power (γ=0.3)', 'DAS+CF -- Sigmoid'};

images = {bmode_das_log,   bmode_das_pow,   bmode_das_sig, ...
          bmode_dascf_log, bmode_dascf_pow, bmode_dascf_sig};

for i = 1:6
    subplot(2, 3, i);
    imagesc(x_mm, z_mm, images{i});
    colormap(gca, 'gray');
    colorbar;
    axis image;
    xlabel('Lateral (mm)');
    ylabel('Depth (mm)');
    title(titles{i});
end

sgtitle(sprintf('Patient %d (Class %d) -- Compression Comparison', ...
    patIdx, data.data(patIdx).class));

saveas(gcf, 'C:\Users\rohit\Documents\MATLAB Code\Project_Figures\Phase5\beamforming_compression_comparison.png');
fprintf('Figure saved\n');

%% -------------------------------------------------------------------------
% SECTION 9: Multi-patient power law check (patients 1-3)
% -------------------------------------------------------------------------
fprintf('\nMulti-patient check...\n');

figure('Name', 'Multi-Patient Power Law', 'Position', [50 50 1200 400]);

for p = 1:3
    rf_p  = double(data.data(p).rf1);
    ns_p  = size(rf_p, 1);
    dep_p = (ns_p / fs) * c / 2;
    z_p   = linspace(1e-3, dep_p, 300);

    dx2_p  = reshape(x_image, [1 nx 1]) - reshape(x_elements, [1 1 N]);
    dz2_p  = reshape(z_p, [nz 1 1]);
    dist_p = sqrt(dx2_p.^2 + dz2_p.^2);
    idx_p  = max(1, min(ns_p, round((dist_p / c) * fs) + 1));

    rfc_p = hilbert(rf_p);
    das_p = zeros(nz, nx);
    for ie = 1:N
        das_p = das_p + reshape(rfc_p(idx_p(:,:,ie), ie), nz, nx);
    end

    env_p   = abs(das_p);
    bmode_p = compress_power(env_p, 0.3);

    if data.data(p).class == 0
        lstr = 'Malignant';
    else
        lstr = 'Benign';
    end

    subplot(1, 3, p);
    imagesc(x_mm, z_p*1000, bmode_p);
    colormap(gca, 'gray');
    colorbar;
    axis image;
    xlabel('Lateral (mm)');
    ylabel('Depth (mm)');
    title(sprintf('Patient %d -- %s', p, lstr));
end

sgtitle('DAS + Power Law (γ=0.3) -- Patients 1-3');
saveas(gcf, 'C:\Users\rohit\Documents\MATLAB Code\Project_Figures\Phase5\multipatient_powerlaw.png');
fprintf('Done\n');