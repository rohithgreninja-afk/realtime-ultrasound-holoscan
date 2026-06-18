function bmode = beamform_mvbf(rf, x_elements, z_image, x_image, fs, c)
% Minimum Variance Beamformer (MVBF) — Capon beamformer
% Computes per-pixel adaptive weights that minimise output variance
% Much stronger artefact suppression than DAS
% Note: slow — uses a subarray approach for tractability

N  = size(rf, 2);
nz = length(z_image);
nx = length(x_image);
L  = min(32, floor(N/2));  % subarray length for efficiency

% Build sample index matrix [nz x nx x N]
dx2  = reshape(x_image, [1 nx 1]) - reshape(x_elements, [1 1 N]);
dz2  = reshape(z_image, [nz 1  1]);
dist = sqrt(dx2.^2 + dz2.^2);
sample_idx = round((dist/c)*fs) + 1;
sample_idx = max(1, min(size(rf,1), sample_idx));

bmode_raw = zeros(nz, nx);
steering  = ones(L, 1) / L;  % DAS steering vector

for iz = 1:nz
    for ix = 1:nx
        % Gather delayed signals for this pixel
        d = zeros(N, 1);
        for ie = 1:N
            d(ie) = rf(sample_idx(iz,ix,ie), ie);
        end
        % Subarray averaging for covariance estimation
        nsub = N - L + 1;
        R = zeros(L, L);
        for k = 1:nsub
            sub = d(k:k+L-1);
            R = R + sub * sub';
        end
        R = R / nsub;
        % Diagonal loading for stability
        R = R + 0.01*trace(R)/L * eye(L);
        % Capon weights
        try
            Rinv = inv(R);
            w = Rinv * steering / (steering' * Rinv * steering);
            bmode_raw(iz,ix) = real(w' * d(1:L));
        catch
            bmode_raw(iz,ix) = mean(d);
        end
    end
end

% Envelope and log compression
env   = abs(hilbert(bmode_raw));
bmode = 20*log10(env + 1);
bmode = bmode - max(bmode(:));
bmode = max(bmode, -60);
end
