% Load data and define parameters
load('C:\Users\rohit\Downloads\Real Time Image Processing Project\OASBUD.mat');

fs    = 25e6;
c     = 1540;
pitch = 0.245e-3;

% Extract RF data for patient 1
rf = data(1).rf1;
[num_samples, num_elements] = size(rf);

% Define the output image grid
% Lateral axis: position of each element along the probe
x_elements = (0:num_elements-1) * pitch;
x_elements = x_elements - mean(x_elements); % centre around zero

% Depth axis: convert time samples to metres
depth_axis = (0:num_samples-1) * (c / (2 * fs));

% Define image pixel grid (downsample for speed)
x_image = linspace(x_elements(1), x_elements(end), 200);  % 200 lateral pixels
z_image = linspace(depth_axis(1),  depth_axis(end),  300); % 300 depth pixels

% Preallocate output image
bmode = zeros(length(z_image), length(x_image));

fprintf('Beamforming %d x %d pixels...\n', length(z_image), length(x_image));

% Delay and Sum loop
for iz = 1:length(z_image)
    for ix = 1:length(x_image)

        pixel_sum = 0;

        for ie = 1:num_elements
            % Distance from this pixel to this element
            dx = x_image(ix) - x_elements(ie);
            dz = z_image(iz);
            dist = sqrt(dx^2 + dz^2);

            % Convert distance to a sample index
            sample_idx = round((dist / c) * fs) + 1;

            % Add the signal value if within bounds
            if sample_idx >= 1 && sample_idx <= num_samples
                pixel_sum = pixel_sum + rf(sample_idx, ie);
            end
        end

        bmode(iz, ix) = pixel_sum;
    end

    % Progress update every 50 rows
    if mod(iz, 50) == 0
        fprintf('  Row %d of %d done\n', iz, length(z_image));
    end
end

fprintf('Beamforming complete.\n');


%BMODE 
fprintf('bmode size:  %d x %d\n', size(bmode,1), size(bmode,2));
fprintf('Min value:   %.2f\n', min(bmode(:)));
fprintf('Max value:   %.2f\n', max(bmode(:)));
fprintf('Mean value:  %.2f\n', mean(bmode(:)));

figure;
imagesc(bmode);
colormap gray;
colorbar;
title('Raw Beamformed Output (before envelope + log compression)');
xlabel('Lateral pixels');
ylabel('Depth pixels');

%Hilbert transform
% Apply Hilbert transform column by column
% Each column is one lateral position, we extract the envelope along depth
envelope = abs(hilbert(bmode));

fprintf('Envelope min:  %.2f\n', min(envelope(:)));
fprintf('Envelope max:  %.2f\n', max(envelope(:)));
fprintf('Envelope mean: %.2f\n', mean(envelope(:)));

figure;
imagesc(envelope);
colormap gray;
colorbar;
title('After Hilbert Transform (Envelope)');
xlabel('Lateral pixels');
ylabel('Depth pixels');

%converting to a 60db range
% Log compression
bmode_log = 20 * log10(envelope + 1);  % +1 avoids log(0)

% Normalize to 0-60 dB dynamic range
bmode_log = bmode_log - max(bmode_log(:));  % shift so max = 0 dB
dynamic_range = 60;                          % show 60 dB range
bmode_log(bmode_log < -dynamic_range) = -dynamic_range;  % clip below -60 dB

fprintf('After log compression:\n');
fprintf('Min: %.2f dB\n', min(bmode_log(:)));
fprintf('Max: %.2f dB\n', max(bmode_log(:)));

figure;
imagesc(bmode_log, [-dynamic_range 0]);
colormap gray;
colorbar;
title('B-Mode Image - Patient 1 (Malignant)');
xlabel('Lateral pixels');
ylabel('Depth pixels');