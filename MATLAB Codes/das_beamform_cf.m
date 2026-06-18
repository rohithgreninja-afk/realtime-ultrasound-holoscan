function bmode = das_beamform_cf(rf, x_elements, z_image, x_image, fs, c)
% DAS with Coherence Factor weighting
% Inputs: rf [samples x elements], x_elements, z_image, x_image, fs, c
% Output: bmode [nz x nx] log-compressed image

N  = size(rf, 2);
nz = length(z_image);
nx = length(x_image);

% Pre-compute analytic signal for all elements
rf_complex = hilbert(rf);  % [num_samples x N]

% Build distance matrix [nz x nx x N]
dx2  = reshape(x_image, [1 nx 1]) - reshape(x_elements, [1 1 N]);
dz2  = reshape(z_image, [nz 1  1]);
dist = sqrt(dx2.^2 + dz2.^2);

% Sample indices
sample_idx = round((dist / c) * fs) + 1;
sample_idx = max(1, min(size(rf,1), sample_idx));

% Gather delayed real and complex samples [nz x nx x N]
delayed_real    = zeros(nz, nx, N);
delayed_complex = zeros(nz, nx, N);
for ie = 1:N
    idx = sample_idx(:,:,ie);
    delayed_real(:,:,ie)    = reshape(rf(idx, ie),         nz, nx);
    delayed_complex(:,:,ie) = reshape(rf_complex(idx, ie), nz, nx);
end

% Coherence factor [nz x nx]
coh_sum   = abs(sum(delayed_complex, 3)).^2;
incoh_sum = N * sum(abs(delayed_complex).^2, 3) + eps;
cf = max(0, min(1, coh_sum ./ incoh_sum));

% Apply CF weighting and sum
das_sum  = sum(delayed_real, 3);
weighted = cf .* das_sum;
% Envelope detection -- abs of the already-complex weighted sum
env = abs(hilbert(sum(delayed_complex, 3)) .* cf);

% Normalize to peak then log compress
env   = env / (max(env(:)) + eps);
bmode = 20 * log10(env + eps);
bmode = max(bmode, -60);