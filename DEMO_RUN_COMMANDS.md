# Demo Run Commands

Step-by-step commands to run both pipelines from a clean WSL2 environment, used for
the project demo video. Both pipelines already have their GPU Coder CUDA libraries
built and verified (see README GPU Coder Acceleration sections); these commands run
the pipelines using those libraries, they do not rebuild them. If you're setting up
CUDA acceleration from scratch, see the README's Setup section instead.

## 1. Clone the repository

```bash
cd ~
git clone https://github.com/rohithgreninja-afk/realtime-ultrasound-holoscan.git
cd realtime-ultrasound-holoscan
```

## 2. Activate the Holoscan environment

```bash
source ~/holoscan-env-310/bin/activate && ulimit -s 32768
```

Your prompt should now start with `(holoscan-env-310)`.

## 3. Install h5py (Pipeline 2 dependency, not needed for Pipeline 1)

```bash
pip install h5py
```

## 4. Point both pipelines at their already-built CUDA libraries

```bash
export ALINE_CUDA_LIB_PATH=~/aline_cuda_src/aline_reconstruct.so
export DAS_PLANEWAVE_CUDA_LIB_PATH=~/das_planewave_src/das_beamform_planewave.so
```

Adjust these paths if your `.so` files live somewhere else. If either file doesn't
exist yet, see the README's WSL2 build steps for that pipeline before continuing --
both pipelines fall back to a CPU implementation automatically if the library isn't
found, so they'll still run, just without acceleration.

## 5. Run Pipeline 1 (OASBUD, live classification)

```bash
cd ~/realtime-ultrasound-holoscan/Holoscan
export OASBUD_PATH="/mnt/c/Users/rohit/Downloads/Real Time Image Processing Project/OASBUD.mat"
python3 medical_imaging_pipeline.py 100
python3 Phase6_Pipeline_Benchmark.py
```

Confirm `BeamformingOp ready -- ... CUDA (GPU Coder)` appears near the top of the
output, not `scipy (fallback)`, before recording, since that confirms acceleration
is genuinely active.

## 6. Run Pipeline 2 (plane wave, real-time acceleration demonstration)

```bash
cd ~/realtime-ultrasound-holoscan/HoloscanPlaneWave
export PLANEWAVE_DATA_PATH="/mnt/c/Users/rohit/Downloads/CIRS040GSE/CIRS040GSE"
python3 medical_imaging_pipeline.py 20
python3 PlaneWave_Pipeline_Benchmark.py
```

Same check: confirm `CUDA (GPU Coder)` appears in the startup message, not `numpy
(fallback)`.

## Expected headline numbers

These are what should appear in the benchmark output if everything is working
correctly, for reference while recording:

**Pipeline 1:** beamforming 12.94 ms (74% of frame), total 17.46 ms, 57.3 fps mean
throughput, 87.37% classification accuracy on the held-out test set.

**Pipeline 2:** beamforming 8.44 ms (2.0% of frame, essentially solved by CUDA),
data load 387.74 ms (93.2% of frame, the real remaining bottleneck), total 416.11 ms,
2.4 fps mean throughput. No classification output, this pipeline is a reconstruction
and acceleration demonstration only; see README Dataset Strategy for why.

## Reconstructed output locations

- Pipeline 1 results: `Holoscan/pipeline_results.npy`
- Pipeline 2 reconstructed frames: `HoloscanPlaneWave/planewave_results/*.png`
- Pipeline 2 benchmark frames and chart: `HoloscanPlaneWave/planewave_benchmark_results/`

## Optional: F-k migration comparison (MATLAB, not part of either live pipeline)

F-k migration is a documented comparison method for Pipeline 2, not wired into the
live real-time pipeline (see README Pipeline 2 section for why). To reproduce the
comparison figures in MATLAB:

```matlab
cd 'C:\path\to\your\clone\MATLAB Codes'
FK_Migration_Test              % broadside only, vs DAS
FK_Migration_MultiAngle_Test   % 1 angle vs 75 angles, with CNR measurement
```
