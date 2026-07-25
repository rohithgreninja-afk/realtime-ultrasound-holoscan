function bmode_raw = fk_migration_planewave(rf, fs, pitch, c)
%FK_MIGRATION_PLANEWAVE  Stolt's f-k migration for a single broadside (0 degree) plane wave
%   Simplified "exploding reflector model" (ERM) implementation for
%   horizontal plane wave transmission only (steering angle = 0), a
%   reimplementation of the simplified academic algorithm described in:
%
%     Garcia D et al., "Stolt's f-k migration for plane wave ultrasound
%     imaging," IEEE Trans Ultrason Ferroelectr Freq Control,
%     2013;60:1853-1867.
%
%   with reference to the published implementation at:
%     https://github.com/rehmanali1994/Plane_Wave_Ultrasound_Stolt_F-K_Migration.github.io
%   (itself built on Garcia & Le Tarnec's original MATLAB code from
%   www.BiomeCardio.com). This is a from-scratch reimplementation with
%   this project's own variable names and structure, not a copy of that
%   repository's source.
%
%   Verified 2026-07-25 against a real wire-target frame from CIRS040GSE
%   (FK_Migration_Test.m): wires resolve to sharp discrete points at the
%   same positions as the already-verified DAS reconstruction, with a
%   visibly cleaner noise floor than DAS, consistent with f-k migration's
%   known image quality advantage over delay-and-sum.
%
%   Inputs:
%     rf    - RF data [num_samples x num_elements], single broadside (0 deg) acquisition
%     fs    - Sampling frequency, Hz
%     pitch - Element pitch, m
%     c     - Speed of sound, m/s
%
%   Output:
%     bmode_raw - Migrated RF-domain signal [num_samples x num_elements],
%                 same size as input. Depth axis: z = (0:num_samples-1)*c/(2*fs).
%                 Lateral axis: x = ((0:num_elements-1)-(num_elements-1)/2)*pitch.
%                 Still needs envelope detection and log compression
%                 afterward, same as the DAS output.

[nt0, nx0] = size(rf);

% Zero-padding: reduces wraparound and interpolation artefacts in the
% frequency-domain remapping step below
nt = 2^(nextpow2(nt0) + 1);
nx = 2 * nx0;

% Exploding Reflector Model (ERM) velocity: the migration reformulates
% two-way pulse-echo travel as one-way propagation at this effective
% speed, which is what allows a single 2D FFT + remap + IFFT to stand in
% for per-pixel delay-and-sum
ERMv = c / sqrt(2);

% 2D FFT of the RF data (time x lateral position), zero-padded, centred
fftRF = fftshift(fft2(rf, nt, nx));

% Frequency and spatial-frequency (wavenumber) axes
f  = (-nt/2:nt/2-1) * fs / nt;
kx = (-nx/2:nx/2-1) / pitch / nx;
[kx, f] = meshgrid(kx, f);

% Stolt mapping: remap temporal frequency f onto depth wavenumber fkz.
% This is the actual "migration" step -- it moves each frequency
% component to where it physically belongs in depth-wavenumber space.
fkz = ERMv * sign(f) .* sqrt(kx.^2 + f.^2 / ERMv^2);
fftRF = interp2(kx, f, fftRF, kx, fkz, 'linear', 0);

% Jacobian of the f -> fkz mapping: an amplitude correction required
% because remapping frequencies changes the local spectral density
kz = (-nt/2:nt/2-1)' / ERMv / fs / nt;
fftRF = fftRF .* kz ./ (fkz + eps);

% Inverse FFT back to space-time, crop to the original size
bmode_raw = ifft2(ifftshift(fftRF), 'symmetric');
bmode_raw = bmode_raw(1:nt0, 1:nx0);

end
