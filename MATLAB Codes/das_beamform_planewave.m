function bmode = das_beamform_planewave(rf, x_elements, z_image, x_image, fs, c, tx_angle)
%DAS_BEAMFORM_PLANEWAVE  Delay-and-sum beamformer for plane wave imaging
%   GPU Coder entry point. Distinct from das_beamform.m: that function
%   assumes classical per-element pulse-echo (each element fires and
%   receives independently, so delay = 2*dist/c). Plane wave imaging is
%   physically different -- all elements fire together to form a flat
%   wavefront, so the transmit delay follows the wavefront's angle and
%   arrival time, not distance from a single element. Only the receive
%   half still uses the familiar per-element spherical delay.
%
%  Inputs:
%    rf         - RF data matrix [num_samples x num_elements], double, single steering angle
%    x_elements - Lateral positions of transducer elements [1 x num_elements], double
%    z_image    - Depth positions of output pixels [1 x num_depth], double
%    x_image    - Lateral positions of output pixels [1 x num_lateral], double
%    fs         - Sampling frequency in Hz, double scalar
%    c          - Speed of sound in m/s, double scalar
%    tx_angle   - Plane wave steering angle in radians, double scalar
%
%  Output:
%    bmode      - Beamformed output image [num_depth x num_lateral], double

num_samples  = size(rf, 1);
num_depth    = length(z_image);
num_lateral  = length(x_image);
num_elements = length(x_elements);

bmode = zeros(num_depth, num_lateral);

sin_a = sin(tx_angle);
cos_a = cos(tx_angle);

for iz = 1:num_depth
    z = z_image(iz);
    for ix = 1:num_lateral
        x = x_image(ix);

        % Transmit delay: planar wavefront arrival time at (x, z),
        % NOT distance from any single element.
        t_tx = (x * sin_a + z * cos_a) / c;

        pixel_sum = 0.0;
        for ie = 1:num_elements
            % Receive delay: still the familiar per-element spherical
            % distance, since each element independently records the echo.
            dx = x - x_elements(ie);
            dist_rx = sqrt(dx*dx + z*z);
            t_rx = dist_rx / c;

            t_total = t_tx + t_rx;
            sample_idx = round(t_total * fs) + 1;

            if sample_idx >= 1 && sample_idx <= num_samples
                pixel_sum = pixel_sum + rf(sample_idx, ie);
            end
        end
        bmode(iz, ix) = pixel_sum;
    end
end

end