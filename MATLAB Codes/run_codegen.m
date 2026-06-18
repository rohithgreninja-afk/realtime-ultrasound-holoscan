% =========================================================
% run_codegen.m
% GPU Coder code generation for das_beamform
% Real-Time Medical Image Processing Project
% =========================================================

% ── Environment Setup (runs every time) ──────────────────
setenv('ProgramFiles(x86)', 'C:\Program Files (x86)');
setenv('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6');
setenv('PATH', [getenv('PATH') ';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin']);
mex -setup:'C:\Program Files\MATLAB\R2024b\bin\win64\mexopts\msvcpp2022.xml' C++
fprintf('Environment configured: VS2022 + CUDA 12.6\n');
% ─────────────────────────────────────────────────────────

% ── Load data ─────────────────────────────────────────────
load('C:\Users\rohit\Downloads\Real Time Image Processing Project\OASBUD.mat');

% ── Define acquisition parameters ────────────────────────
fs    = 25e6;       % Sampling frequency: 25 MHz
c     = 1540;       % Speed of sound in soft tissue (m/s)
pitch = 0.245e-3;   % Element spacing: 0.245 mm

% ── Build image grids ─────────────────────────────────────
rf = data(1).rf1;
[num_samples, num_elements] = size(rf);

x_elements = (0:num_elements-1) * pitch;
x_elements = x_elements - mean(x_elements);

depth_axis = (0:num_samples-1) * (c / (2*fs));

x_image = linspace(x_elements(1), x_elements(end), 200);
z_image = linspace(depth_axis(1),  depth_axis(end), 300);

% ── Define argument types for codegen ────────────────────
% These tell GPU Coder the exact size and type of each input
% [0 0] means fixed size — not variable
arg_rf         = coder.typeof(rf,         [num_samples, num_elements], [0 0]);
arg_x_elements = coder.typeof(x_elements, [1, num_elements],           [0 0]);
arg_z_image    = coder.typeof(z_image,    [1, 300],                    [0 0]);
arg_x_image    = coder.typeof(x_image,    [1, 200],                    [0 0]);
arg_fs         = coder.typeof(fs);
arg_c          = coder.typeof(c);

% ── Set working directory to MATLAB Code folder ───────────
cd('C:\Users\rohit\Documents\MATLAB Code');

% ── Configure GPU Coder ───────────────────────────────────
cfg = coder.gpuConfig('lib');
cfg.GpuConfig.SelectCudaDevice = 0;   % RTX 4070 is device 0
cfg.GpuConfig.MallocMode       = 'discrete';
cfg.GenerateReport             = true;

fprintf('Toolchain: %s\n', cfg.Toolchain);
fprintf('GPU device: %d\n', cfg.GpuConfig.SelectCudaDevice);

% ── Run codegen ───────────────────────────────────────────
fprintf('\nRunning GPU Coder...\n');
codegen('das_beamform', '-config', cfg, '-args', ...
    {arg_rf, arg_x_elements, arg_z_image, arg_x_image, arg_fs, arg_c}, ...
    '-report');
fprintf('CUDA code generation complete.\n');

% ── Show generated files ──────────────────────────────────
genDir = 'C:\Users\rohit\Documents\MATLAB Code\codegen\lib\das_beamform';
fprintf('\nGenerated files:\n');
files = dir(genDir);
for i = 1:length(files)
    if ~files(i).isdir
        fprintf('  %-45s  %.1f KB\n', files(i).name, files(i).bytes/1024);
    end
end
fprintf('\nKey file: das_beamform.cu — this is your CUDA kernel.\n');