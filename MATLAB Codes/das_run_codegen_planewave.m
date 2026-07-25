% das_run_codegen_planewave.m
% GPU Coder code generation for das_beamform, targeting the plane wave
% dataset (Zenodo 7986407). Unlike the original run_codegen.m, this uses
% variable-size bounds for BOTH the sample count and element count, since
% this dataset's array geometry differs from OASBUD's and the exact
% element count depends on which probe configuration was used for
% acquisition. Run this once real data is available to confirm the
% bounds below are generous enough.

setenv('ProgramFiles(x86)', 'C:\Program Files (x86)');
setenv('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6');
setenv('PATH', [getenv('PATH') ';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin']);
mex -setup:'C:\Program Files\MATLAB\R2024b\bin\win64\mexopts\msvcpp2022.xml' C++
fprintf('Environment configured: VS2022 + CUDA 12.6\n');

scriptDir = fileparts(mfilename('fullpath'));
cd(scriptDir);

% Generous upper bounds. Verasonics L12-5 arrays are typically 128
% elements; the acquisition system supports up to 256 channels. Sample
% count depends on imaging depth and fs=~31.25MHz typical for this probe
% at 7.8MHz centre frequency -- 4000 samples covers a substantial depth
% range with margin. Adjust MAX_ELEMENTS/MAX_SAMPLES if a real data file
% reports different dimensions once downloaded.
MAX_SAMPLES  = 4000;
MAX_ELEMENTS = 256;
IMG_DEPTH_PX = 400;
IMG_LATERAL_PX = 300;

arg_rf         = coder.typeof(double(0), [MAX_SAMPLES, MAX_ELEMENTS], [1 1]);   % both dims variable
arg_x_elements = coder.typeof(double(0), [1, MAX_ELEMENTS], [0 1]);              % variable length
arg_z_image    = coder.typeof(double(0), [1, IMG_DEPTH_PX], [0 0]);              % fixed
arg_x_image    = coder.typeof(double(0), [1, IMG_LATERAL_PX], [0 0]);            % fixed
arg_fs         = coder.typeof(0);
arg_c          = coder.typeof(0);

cfg = coder.gpuConfig('lib');
cfg.GpuConfig.SelectCudaDevice = 0;
cfg.GpuConfig.MallocMode       = 'discrete';
cfg.GenerateReport             = true;

fprintf('\nRunning GPU Coder on das_beamform (plane wave, variable-size)...\n');
codegen('das_beamform', '-config', cfg, '-args', ...
    {arg_rf, arg_x_elements, arg_z_image, arg_x_image, arg_fs, arg_c}, ...
    '-report', '-o', 'das_beamform_planewave');
fprintf('Done. Check codegen/lib/das_beamform_planewave/ for the .cu/.h files.\n');
fprintf('Also check das_beamform_planewave.h for the generated C++ signature,\n');
fprintf('needed for the ctypes binding.\n');
