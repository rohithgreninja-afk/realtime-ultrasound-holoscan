function bmode = aline_reconstruct(rf, gamma)
%ALINE_RECONSTRUCT  Per-column Hilbert envelope detection + power-law compression
%   GPU Coder entry point. This is the correct reconstruction method for
%   OASBUD, matching BeamformingOp._aline_reconstruct in
%   Holoscan/beamforming_op.py exactly: no delay calculation, no array
%   geometry, a Hilbert transform down each column followed by power-law
%   compression against the global peak.
%
%  Inputs:
%    rf    - RF data matrix [num_samples x num_columns], double
%    gamma - power-law compression exponent, double scalar (0.3 in the deployed pipeline)
%
%  Output:
%    bmode - compressed envelope image [num_samples x num_columns], double

analytic = hilbert(rf);            % MATLAB's hilbert() operates per-column on a matrix by default
envelope = abs(analytic);
env_max  = max(envelope(:));
env_norm = envelope / (env_max + 1e-12);
bmode    = env_norm .^ gamma;

end
