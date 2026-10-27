# Demo Run Commands

Step-by-step commands to run both pipelines from a clean WSL2 environment. Both pipelines use already-built GPU Coder CUDA libraries (see the README Setup section to build them); these commands run the pipelines, they do not rebuild the libraries.

## 1. Clone the repository

```bash
cd ~
git clone https://github.com/rohithgreninja-afk/realtime-ultrasound-holoscan.git
cd realtime-ultrasound-holoscan
```

## 2. Activate the Holoscan environment

```bash
source ~/holoscan-env-310/bin/activate && ulimit -s 32768
pip install h5py   # Pipeline 2 only
```

Your prompt should now start with `(holoscan-env-310)`.

## 3. Point both pipelines at their CUDA libraries and an output folder

```bash
export ALINE_CUDA_LIB_PATH=/path/to/aline_reconstruct.so
export DAS_PLANEWAVE_CUDA_LIB_PATH=/path/to/das_beamform_planewave.so
export RESULTS_DIR=~/realtime-ultrasound-holoscan/results
```

If a library is missing, that pipeline falls back to a CPU implementation and still runs, without acceleration.

## 4. Run Pipeline 1 (OASBUD, live classification)

```bash
cd ~/realtime-ultrasound-holoscan/Holoscan
export OASBUD_PATH=/path/to/OASBUD.mat
python3 medical_imaging_pipeline.py 60
python3 Phase6_Pipeline_Benchmark.py
```

Both commands use only the 60 OASBUD hold-out patients. Confirm `CUDA (GPU Coder)` appears in the startup output, not `scipy (fallback)`, before recording.

## 5. Run Pipeline 2 (plane wave, real-time acceleration demonstration)

```bash
cd ~/realtime-ultrasound-holoscan/HoloscanPlaneWave
export PLANEWAVE_DATA_PATH=/path/to/CIRS040GSE/CIRS040GSE
python3 medical_imaging_pipeline.py 20
python3 PlaneWave_Pipeline_Benchmark.py
```

Without the full download, `PLANEWAVE_DATA_PATH=../data/sample/planewave` runs one bundled frame. Same check: confirm `CUDA (GPU Coder)` in the startup message, not `numpy (fallback)`.

## Expected headline numbers

These are the values in the committed logs under `results/logs/`. Timings vary with the machine and its load; accuracy does not.

**Pipeline 1 benchmark** (`Phase6_Pipeline_Benchmark.py`, 60 hold-out patients, `results/logs/pipeline1_benchmark_holdout.txt`): beamforming 16.51 ms, enhancement 8.16 ms, ONNX inference 3.34 ms, total 28.07 ms, 35.6 fps mean throughput, pipeline accuracy (hold-out patients only) 55.0% (33/60).

**MATLAB CNN evaluation** (`Phase6_CNN_Evaluation.m`, separate from the pipeline, `results/cnn_eval.md`): 81.76% accuracy on the 636-image hold-out test set, malignant AUC-ROC 0.9105.

**Pipeline 2 benchmark** (`PlaneWave_Pipeline_Benchmark.py`, 20 frames, `results/logs/pipeline2_benchmark.txt`): beamforming 7.68 ms (1.7% of the frame), data load 419.17 ms, total 461.27 ms, 2.2 fps mean throughput. No classification output; see README Dataset Strategy.

## Output locations

- Pipeline 1 results: `$RESULTS_DIR/pipeline_results.npy`
- Pipeline 1 benchmark: `$RESULTS_DIR/pipeline1_benchmark/`
- Pipeline 2 reconstructed frames: `$RESULTS_DIR/pipeline2_frames/*.png`
- Pipeline 2 benchmark frames and chart: `$RESULTS_DIR/pipeline2_benchmark/`

## Optional: F-k migration comparison (MATLAB, not part of either live pipeline)

```matlab
setenv('PLANEWAVE_DATA_PATH', '/path/to/CIRS040GSE/CIRS040GSE');
setenv('RESULTS_DIR', '/path/to/results');
cd '/path/to/realtime-ultrasound-holoscan/MATLAB Codes'
FK_Migration_Test              % broadside only, vs DAS
FK_Migration_MultiAngle_Test   % 1 angle vs 75 angles, prints CNR improvement: 63.4%
```
