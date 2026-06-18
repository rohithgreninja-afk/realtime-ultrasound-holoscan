function bmode = das_beamform(rf, x_elements, z_image, x_image, fs, c)
%DAS_BEAMFORM  Delay-and-Sum beamformer entry point for GPU Coder
%
%  Inputs:
%    rf         - Raw RF data matrix [num_samples x num_elements], double
%    x_elements - Lateral positions of transducer elements [1 x num_elements], double
%    z_image    - Depth positions of output pixels [1 x num_depth], double
%    x_image    - Lateral positions of output pixels [1 x num_lateral], double
%    fs         - Sampling frequency in Hz, double scalar
%    c          - Speed of sound in m/s, double scalar
%
%  Output:
%    bmode      - Beamformed output image [num_depth x num_lateral], double

num_samples  = size(rf, 1);
num_depth    = length(z_image);
num_lateral  = length(x_image);
num_elements = length(x_elements);

bmode = zeros(num_depth, num_lateral);

for iz = 1:num_depth
    for ix = 1:num_lateral
        pixel_sum = 0.0;
        for ie = 1:num_elements
            dx = x_image(ix) - x_elements(ie);
            dz = z_image(iz);
            dist       = sqrt(dx*dx + dz*dz);
            sample_idx = round((dist / c) * fs) + 1;
            if sample_idx >= 1 && sample_idx <= num_samples
                pixel_sum = pixel_sum + rf(sample_idx, ie);
            end
        end
        bmode(iz, ix) = pixel_sum;
    end
end

end