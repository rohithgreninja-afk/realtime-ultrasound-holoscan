% aline_run_codegen.m
% GPU Coder code generation for aline_reconstruct
% Generates the CUDA kernel for the per-column Hilbert envelope detection
% and power-law compression method actually used for OASBUD reconstruction.

% ── Environment Setup (runs every time) ──────────────────
setenv('ProgramFiles(x86)', 'C:\Program Files (x86)');
setenv('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6');
setenv('PATH', [getenv('PATH') ';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin']);
mex -setup:'C:\Program Files\MATLAB\R2024b\bin\win64\mexopts\msvcpp2022.xml' C++
fprintf('Environment configured: VS2022 + CUDA 12.6\n');
% ─────────────────────────────────────────────────────────

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);
cd(scriptDir);

% OASBUD depth varies per patient (1040-2864 rows observed), so the row
% dimension is declared variable-size with an upper bound. Column count
% is fixed at 510 (constant across all OASBUD patients).
MAX_SAMPLES = 3000;
NUM_COLS    = 510;

arg_rf    = coder.typeof(double(0), [MAX_SAMPLES, NUM_COLS], [1 0]);  % rows variable, cols fixed
arg_gamma = coder.typeof(0.3);

cfg = coder.gpuConfig('lib');
cfg.GpuConfig.SelectCudaDevice = 0;   % RTX 4070 is device 0
cfg.GpuConfig.MallocMode       = 'discrete';
cfg.GenerateReport             = true;

fprintf('Toolchain: %s\n', cfg.Toolchain);
fprintf('GPU device: %d\n', cfg.GpuConfig.SelectCudaDevice);

fprintf('\nRunning GPU Coder on aline_reconstruct...\n');
codegen('aline_reconstruct', '-config', cfg, '-args', {arg_rf, arg_gamma}, '-report');
fprintf('CUDA code generation complete.\n');

genDir = fullfile(scriptDir, 'codegen', 'lib', 'aline_reconstruct');
fprintf('\nGenerated files:\n');
files = dir(genDir);
for i = 1:length(files)
    if ~files(i).isdir
        fprintf('  %-45s  %.1f KB\n', files(i).name, files(i).bytes/1024);
    end
end
fprintf('\nKey file: aline_reconstruct.cu -- this is your CUDA kernel.\n');
fprintf('Also check aline_reconstruct.h for the generated C function signature,\n');
fprintf('needed to correct the ctypes binding in Holoscan/aline_cuda.py.\n');
