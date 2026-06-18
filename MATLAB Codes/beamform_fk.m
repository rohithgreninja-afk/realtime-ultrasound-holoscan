function bmode = beamform_fk(rf, x_elements, z_image, x_image, fs, c)
% f-k Migration (Stolt Migration) for plane wave ultrasound
% Works in frequency domain — fast and effective for single plane wave
% Reference: Garcia et al., IEEE TUFFC 2013

[num_samples, N] = size(rf);
nz = length(z_image);
nx = length(x_image);
pitch = x_elements(2) - x_elements(1);

% Step 1: 2D FFT of RF data (time x space)
RF2D = fft2(rf);

% Step 2: Build frequency and wavenumber axes
dt = 1/fs;
dx = pitch;
freq   = (0:num_samples-1)/(num_samples*dt);   % temporal frequency [Hz]
kx     = (0:N-1)/(N*dx);                       % spatial frequency [1/m]

% Centre the spatial frequency axis
kx = kx - kx(ceil(N/2));
RF2D = fftshift(RF2D, 2);

% Step 3: Stolt interpolation
% For plane wave: kz = sqrt((f/c)^2 - kx^2)
[KX, F] = meshgrid(kx, freq);
KZ2 = (F/c).^2 - KX.^2;
valid = KZ2 > 0;
KZ    = zeros(size(KZ2));
KZ(valid) = sqrt(KZ2(valid));

% Step 4: Build output image by back-projecting frequency components
% Use direct summation approach (simplified Stolt)
img = zeros(nz, nx);
z_vec = z_image(:);     % column
x_vec = x_image(:)';    % row

% Limit to reasonable frequency range
f_max  = fs/2;
f_step = f_max / 20;   % use 20 frequency bands for speed
f_bands = f_step:f_step:f_max;

for fi = 1:length(f_bands)
    f0   = f_bands(fi);
    [~,fidx] = min(abs(freq - f0));
    for ki = 1:N
        kz_val = (f0/c)^2 - kx(ki)^2;
        if kz_val <= 0, continue; end
        kz = sqrt(kz_val);
        phase = exp(1i*2*pi*(kx(ki)*x_vec + kz*z_vec));
        img = img + RF2D(fidx,ki) * phase;
    end
end

% Envelope and log compression
env   = abs(img);
bmode = 20*log10(env + 1);
bmode = bmode - max(bmode(:));
bmode = max(bmode, -60);
end
