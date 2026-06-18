function bmode = beamform_dmas(rf, x_elements, z_image, x_image, fs, c)
% Delay-Multiply-and-Sum (DMAS) beamformer
% Multiplies pairs of delayed signals — suppresses incoherent noise naturally
% Reference: Matrone et al., IEEE TUFFC 2015

N  = size(rf, 2);
nz = length(z_image);
nx = length(x_image);

% Build sample index matrix [nz x nx x N]
dx2  = reshape(x_image, [1 nx 1]) - reshape(x_elements, [1 1 N]);
dz2  = reshape(z_image, [nz 1  1]);
dist = sqrt(dx2.^2 + dz2.^2);
sample_idx = round((dist/c)*fs) + 1;
sample_idx = max(1, min(size(rf,1), sample_idx));

% Gather delayed signals [nz x nx x N]
delayed = zeros(nz, nx, N);
for ie = 1:N
    delayed(:,:,ie) = reshape(rf(sample_idx(:,:,ie), ie), nz, nx);
end

% DMAS: sum all unique pairwise products
dmas_sum = zeros(nz, nx);
for i = 1:N-1
    for j = i+1:N
        prod = delayed(:,:,i) .* delayed(:,:,j);
        % Signed square root to preserve sign
        dmas_sum = dmas_sum + sign(prod) .* sqrt(abs(prod));
    end
end

% Envelope and log compression
env   = abs(hilbert(dmas_sum));
bmode = 20*log10(env + 1);
bmode = bmode - max(bmode(:));
bmode = max(bmode, -60);
end
