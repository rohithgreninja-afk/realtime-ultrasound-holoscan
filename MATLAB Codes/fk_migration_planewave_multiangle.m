function bmode_raw = fk_migration_planewave_multiangle(rf, fs, pitch, c, tx_angles)
%FK_MIGRATION_PLANEWAVE_MULTIANGLE  Stolt's f-k migration, arbitrary steering angles, compounded
%   Generalisation of fk_migration_planewave.m (which only handles the
%   broadside/0-degree case) to arbitrary plane wave steering angles,
%   with running-average compounding across angles, matching the
%   compounding approach already used for multi-angle DAS in
%   BeamformingOp.py.
%
%   Reimplementation, with this project's own variable names and
%   structure, of the general algorithm described in:
%
%     Garcia D et al., "Stolt's f-k migration for plane wave ultrasound
%     imaging," IEEE Trans Ultrason Ferroelectr Freq Control,
%     2013;60:1853-1867.
%
%   with reference to the published implementation at:
%     https://github.com/rehmanali1994/Plane_Wave_Ultrasound_Stolt_F-K_Migration.github.io
%   This function follows the same overall sequence of operations as
%   that reference (temporal FFT, steering-angle time compensation,
%   spatial FFT, Stolt remap with angle-dependent ERM correction,
%   evanescent removal, obliquity correction, inverse FFTs, running
%   compounding), reimplemented independently.
%
%   UNVERIFIED. This is meaningfully more complex than the broadside-only
%   version already verified in fk_migration_planewave.m (steering-angle
%   compensation is applied in two separate places, at different stages
%   of the transform). Run FK_Migration_MultiAngle_Test.m against the
%   wire-target frame and confirm the wires resolve to sharp points
%   before trusting this anywhere further.
%
%   Inputs:
%     rf         - RF data [num_samples x num_elements x num_angles]
%     fs         - Sampling frequency, Hz
%     pitch      - Element pitch, m
%     c          - Speed of sound, m/s
%     tx_angles  - Steering angles for each frame in rf's 3rd dimension, radians
%
%   Output:
%     bmode_raw - Compounded migrated RF-domain signal [num_samples x num_elements].
%                 Depth axis: z = (0:num_samples-1)*c/(2*fs).
%                 Lateral axis: x = ((0:num_elements-1)-(num_elements-1)/2)*pitch.
%                 Still needs envelope detection and log compression afterward.

[nt0, nx0, n_frames] = size(rf);
assert(numel(tx_angles) == n_frames, ...
    'tx_angles must have one entry per angle in rf''s 3rd dimension');

% Zero-padding
nt = 4 * nt0;
if mod(nt, 2) == 1, nt = nt + 1; end
nx = round(1.5 * nx0);
if mod(nx, 2) == 1, nx = nx + 1; end

f0 = (0:nt/2)' * fs / nt;
kx_vec = [0:nx/2, -nx/2+1:-1] / pitch / nx;
[kx, f] = meshgrid(kx_vec, f0);

% Temporal FFT of all frames at once, keep only positive frequencies
% (the signal is real, so negative frequencies are redundant)
RF = fft(double(rf), nt, 1);
RF(nt/2+2:nt, :, :) = [];

migSIG = zeros(nt, nx);

for k = 1:n_frames
    frameK = RF(:, :, k);
    sinA = sin(tx_angles(k));
    cosA = cos(tx_angles(k));

    % ERM velocity for this steering angle
    v = c / sqrt(1 + cosA + sinA^2);

    % Steering-angle time compensation, applied before the spatial FFT.
    % The (nx0-1)*(angle<0) offset matters: for a negative steering angle
    % the LAST element fires first instead of the first, so the delay
    % reference anchor flips to the other end of the array. Omitting this
    % term silently misaligns every negative-angle frame before
    % compounding, which cancels signal instead of reinforcing it.
    if sinA ~= 0
        offset = (nx0 - 1) * (tx_angles(k) < 0);
        dt = sinA * (offset - (0:nx0-1)) * pitch / c;
        frameK = frameK .* exp(-2i*pi * f0 .* dt);
    end

    % Spatial FFT
    frameK = fft(frameK, nx, 2);

    % Stolt mapping with angle-dependent ERM correction
    C = (1 + cosA + sinA^2) / (1 + cosA)^1.5;
    fkz = v * sqrt(kx.^2 + 4*f.^2/c^2 * C^2);

    % Remove evanescent components (non-physical, would otherwise blow up)
    isevanescent = abs(f) ./ abs(kx) < c;
    frameK(isevanescent) = 0;

    % Interpolate from temporal frequency f onto depth wavenumber fkz
    frameK = interp_freq_remap(fs/nt, frameK, fkz);

    % Obliquity factor
    frameK = frameK .* f ./ fkz;
    frameK(1) = 0;

    % Inverse temporal FFT: rebuild negative frequencies via conjugate
    % symmetry, then IFFT
    frameK = [frameK; conj([frameK(nt/2:-1:2, 1), frameK(nt/2:-1:2, end:-1:2)])]; %#ok<AGROW>
    frameK = ifft(frameK);

    % Post-compensation for steering angle, applied after the temporal
    % IFFT but before the spatial IFFT
    if sinA ~= 0
        Cc = sinA / (2 - cosA);
        dx = -Cc * (0:nt-1) / fs * c / 2;
        frameK = frameK .* exp(-2i*pi * kx(1,:) .* dx');
    end

    % Running-average compounding across angles
    migSIG = ((k-1)*migSIG + frameK) / k;
end

% Final spatial inverse FFT
migSIG = ifft(migSIG, [], 2, 'symmetric');
bmode_raw = migSIG(1:nt0, 1:nx0);

end


function yi = interp_freq_remap(dx, y, xi)
% Linear interpolation of y (columns) from its native sample spacing dx
% onto the target sample positions xi, per column.
siz = size(y);
yi = zeros(siz);
idx = xi / dx + 1;
outOfRange = idx > (siz(1) - 1);
idx(outOfRange) = 1;
idxFloor = floor(idx);
for col = 1:siz(2)
    ii = idxFloor(:, col);
    frac = ii - idx(:, col);
    yi(:, col) = y(ii, col) .* (frac + 1) - y(ii+1, col) .* frac;
end
yi(outOfRange) = 0;
end
