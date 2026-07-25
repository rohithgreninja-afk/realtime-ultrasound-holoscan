# Real-Time Acceleration for Medical Image Processing

**MathWorks MATLAB-Simulink Challenge Project**
Rohith Ram V | 24BCE0543 | B.Tech CSE | VIT Vellore

A complete end-to-end pipeline for real-time breast ultrasound classification: raw RF data goes in, a malignant/benign/normal prediction comes out at 57 fps on an NVIDIA RTX 4070, deployed via NVIDIA Holoscan SDK, with beamforming accelerated through GPU Coder-generated CUDA.

---

## Pipeline Architecture

```
OASBUD RF data [1]
     |
     v
DataSourceOp        -- loads .mat file, emits frames one at a time
     |
     v
BeamformingOp       -- per-column Hilbert envelope + power law compression (gamma=0.3)
                        GPU Coder CUDA kernel if available, scipy fallback otherwise
     |
     v
EnhancementOp       -- median filter 3x3 -> normalise -> resize 224x224 bilinear -> RGB
     |
     v
InferenceOp         -- ONNX MobileNetV2 mega model via CUDAExecutionProvider
     |
     v
OutputOp            -- record prediction, print summary, save results
```

---

## Dataset Strategy: Why OASBUD and the Plane-Wave Phantom Dataset Serve Different Roles

This project deliberately uses two different raw ultrasound sources for two different
purposes, rather than one dataset for everything. Both limitations below were confirmed
experimentally, not assumed; see `Project_Documentation.docx`, Section 13.3, for the
full test methodology and result figures.

### Why OASBUD is not used for GPU-accelerated delay-and-sum beamforming or the Phased Array System Toolbox

OASBUD's radio frequency data looks like raw sensor output, but it is not raw per-element
transducer channel data. Each of its 510 columns is already a pre-formed scan line: the
original clinical scanner had already run its own beamforming before the data was saved.
There is no raw array-element geometry left in the data for delay-and-sum, the GPU
Coder-generated CUDA kernel (`das_beamform.cu`), or any Phased Array System Toolbox
beamformer (`phased.PhaseShiftBeamformer` and similar) to operate on correctly. Running
any of these against OASBUD, correct physics or not, produces an incoherent image with
no diagnostic value, confirmed directly by reconstructing real OASBUD data both ways and
comparing the result. The correct and only method used for OASBUD anywhere in this
project is per-column Hilbert envelope detection with power-law compression
(`BeamformingOp`, GPU Coder-accelerated via `aline_reconstruct.m`, see below), which
involves no delay calculation and no array geometry assumption.

### Why the plane-wave phantom dataset cannot replace OASBUD for classification accuracy

The Ultrasound Plane Wave Raw Data, 75 Angles dataset [6] is genuine raw per-element
channel data, which is exactly what OASBUD is not, making it the correct data source for
demonstrating GPU Coder CUDA acceleration and the Phased Array System Toolbox on
delay-and-sum-family methods. But it is acquired from physical test phantoms (a
multi-purpose calibration phantom and a breast-mimicking phantom), not patient tissue,
and carries no malignant or benign labels of any kind. Its built-in categories
(hyperechoic, hypoechoic, no lesion) describe physical reflectivity properties used to
check image resolution and contrast, not a diagnosis. This dataset therefore cannot
contribute to, substitute for, or validate any part of this project's
malignant/benign/normal classification accuracy.

### The resulting split

| | OASBUD | Plane-wave phantom dataset [6] |
|---|---|---|
| Data type | Pre-formed scan lines | Raw per-element channel data |
| Used for | AI classification (malignant/benign), via GPU Coder-accelerated per-column Hilbert | Delay-and-sum GPU Coder CUDA + Phased Array System Toolbox demonstration, real-time acceleration only (see Pipeline 2 below) |
| Not used for | Delay-and-sum beamforming, Phased Array System Toolbox | Any classification accuracy claim (no disease labels exist) |

The two datasets are not interchangeable in either direction, and this project does not
claim otherwise anywhere in the pipeline, results, or documentation.

---

## GPU Coder CUDA Acceleration

The reconstruction method used for OASBUD, per-column Hilbert envelope detection and
power-law compression, was accelerated by generating CUDA directly from MATLAB
(`MATLAB Codes/aline_reconstruct.m`, GPU Coder) and binding the resulting library into
the live `BeamformingOp` through ctypes. This is distinct from `das_beamform.m` /
`das_beamform.cu`, an earlier GPU Coder target implementing delay-and-sum beamforming:
that method is not used for OASBUD, since OASBUD's RF columns are pre-formed scan lines
rather than raw per-element channel data (see Project_Documentation.docx, Section 5.2),
so accelerating it would not have helped the deployed pipeline. `aline_reconstruct.m`
instead implements the exact Hilbert/power-law method OASBUD actually needs.

The CUDA path's output was verified against the existing scipy implementation across all
100 OASBUD patients before deployment (max absolute difference 7.64e-13, floating-point
noise). `BeamformingOp` falls back to scipy automatically if the compiled library isn't
present, so the pipeline runs either way; see [Setup](#setup) to build the accelerated
library.

**Before / after (100 OASBUD frames, same benchmark script and hardware):**

| Metric | Before (scipy) | After (CUDA) |
|---|---|---|
| Beamforming latency (mean) | 23.50 ms | 12.94 ms |
| Total pipeline latency (mean) | 29.37 ms | 17.46 ms |
| Throughput (mean) | 33.5 fps | 57.3 fps |
| Throughput (95th percentile) | 15.6 fps | 39.9 fps |
| Classification accuracy | 73.0% | 73.0% |

Beamforming latency fell 45%, total pipeline latency fell 41%, and classification
accuracy is unchanged, as expected given the sub-1e-12 numerical agreement between the
two implementations: this accelerated the pipeline without altering a single
classification outcome.

---

## Pipeline 2: Plane Wave Beamforming (Real-Time Acceleration Demonstration)

A second, completely separate Holoscan application, `HoloscanPlaneWave/`, runs
genuine raw per-element channel data instead of OASBUD. It shares no operators with
the pipeline above beyond the choice of median filter for speckle reduction, and it
has no classification stage: the dataset behind it (CIRS040GSE, a physical calibration
phantom, see [Dataset Strategy](#dataset-strategy-why-oasbud-and-the-plane-wave-phantom-dataset-serve-different-roles)
above) has no disease labels to classify.

### Why the physics is different here

OASBUD's DAS-family comparison methods assume classical per-element pulse-echo: each
element fires and receives independently, so delay is simply twice the element-to-pixel
distance over the speed of sound. Plane wave imaging is physically different, all
elements fire together to form a flat wavefront, so the transmit delay follows the
wavefront's arrival angle and time rather than distance from any single element; only
the receive half still uses the familiar per-element spherical delay. This required a
new function, `das_beamform_planewave.m`, distinct from `das_beamform.m`.

### GPU Coder CUDA acceleration

Same build process already proven for the OASBUD pipeline's `aline_reconstruct.m`
target: variable-size code generation (element count and sample count both vary by
acquisition), a Linux shared library build in WSL2 reusing the same toolchain fixes
already resolved for the OASBUD pipeline (GCC version pin, the glibc/CUDA header patch,
cuFFT linking), an `extern "C"` wrapper for the ctypes binding, and numerical
verification before deployment: the CUDA output matched a numpy reference
implementation with **zero difference** (exact bit-for-bit match) across 5 real frames,
tested at the broadside steering angle, and the pipeline falls back to numpy
automatically if the compiled library isn't present.

### Two reconstruction methods, one live, one for comparison

**Delay-and-sum (`das_beamform_planewave.m`)** is the live, GPU Coder-accelerated
method used in the deployed pipeline.

**F-k migration (`fk_migration_planewave.m`, `fk_migration_planewave_multiangle.m`)**
is documented as a comparison method, the same role DMAS and MVBF play for OASBUD in
Section 6, not wired into the live real-time pipeline. This is the reconstruction
method the dataset's own original authors use as their reference implementation
(Garcia D et al., "Stolt's f-k migration for plane wave ultrasound imaging," IEEE Trans
Ultrason Ferroelectr Freq Control, 2013;60:1853-1867), reimplemented independently with
attribution rather than copied from the reference source. Both a broadside-only version
and a full multi-angle version (arbitrary steering angles, running-average compounding)
were built and verified against a real wire-target frame: wires resolve to sharp,
correctly-positioned points, and compounding across all 75 available angles improved
measured contrast-to-noise ratio by 56% versus a single angle (background noise
standard deviation dropped ~40%), measured directly on the raw envelope rather than by
visual inspection of independently-normalised display images.

### Phased Array System Toolbox, on real data

`Phase7_PhasedArrayToolbox_RealData.m` replaces the earlier synthetic point-target
demonstration with genuine recorded channel data from a CIRS040GSE wire-target frame,
using `phased.ULA` with the array's real geometry (128 elements, pitch read directly
from the dataset header) and `phased.PhaseShiftBeamformer` steered toward broadside on
the real recorded data's analytic (Hilbert) signal.

### Pipeline benchmark

Per-stage latency (data load, beamforming, enhancement, save), isolated the same way as
the OASBUD pipeline's benchmark so the CUDA speedup specific to beamforming can be seen
rather than blended into a single per-frame number, measured by
`PlaneWave_Pipeline_Benchmark.py` across all 20 real CIRS040GSE frames, single broadside
angle, GPU Coder CUDA beamforming backend:

| Stage | Mean (ms) | Share |
|---|---|---|
| Data load | 387.74 | 93.2% |
| Beamforming | 8.44 | 2.0% |
| Enhancement | 11.79 | 2.8% |
| Save | 8.05 | 1.9% |
| **Total** | **416.11** | **100%** |

Throughput: **2.4 fps** (mean)

This result is worth reading carefully rather than at face value. Beamforming itself,
the part CUDA actually accelerates, is 8.44 ms, essentially negligible, 2% of the frame.
That specific part of this work succeeded completely: the GPU Coder CUDA kernel took
beamforming from being the dominant cost to being nearly free. But data loading, reading
the HDF5 file and applying the lens delay, angle delay, and channel 125 corrections, is
387.74 ms, 93% of the total frame time, and CUDA was never going to touch that, since it
only accelerates the beamforming math, not disk I/O or the preprocessing steps ahead of
it. The overall 2.4 fps throughput reflects this bottleneck, not a limitation of the
acceleration work itself. Optimising the data loading path (a faster HDF5 read pattern,
or pre-applying the fixed corrections once per dataset rather than per frame) is a real,
identified next step, not yet attempted, and would very likely improve overall
throughput far more than any further beamforming optimisation could at this point.

---

## Image Enhancement

Four candidate speckle-reduction filters from the Image Processing Toolbox were compared on reconstructed B-mode images, scored by PSNR and structural similarity (SSIM):

| Filter | PSNR | SSIM | Notes |
|---|---|---|---|
| **Median 3x3 (selected)** | **24.77 dB** | **0.4011** | Best SSIM, used throughout the pipeline |
| Wiener 5x5 | -- | -- | Evaluated, did not outperform median |
| Adaptive histogram equalisation | -- | -- | Excluded: amplifies arc-shaped artefacts in log-compressed images |
| Bilateral filter | -- | -- | Over-smoothed; SSIM 0.3329 |

The 3x3 median filter was selected and is applied in `EnhancementOp` before resize, matching the Phase 3 analysis exactly.

![Enhancement Filter Comparison](Project%20Figures/Phase3/Phase3_Final_Summary.png)

---

## Simulink

Two Simulink models are included in the `Simulink/` folder.

**UltrasoundPipelineDiagram.slx** is an architecture-level block diagram showing the complete five-stage pipeline (DataSource, Beamforming, Enhancement, Inference, Output) in sequence, mirroring the deployed Holoscan structure. It serves as a visual reference for the pipeline architecture.

**UltrasoundEnhancementSubsystem.slx** is a fully functional Simulink model that simulates the core image processing subsystem using executable MATLAB Function blocks. A From Workspace source feeds a real OASBUD patient RF matrix into two sequential blocks: EnvelopeDetection (Hilbert transform, 20 log10 compression, dynamic range normalisation) and Enhancement (3x3 median filter, PSNR 24.77 dB, SSIM 0.4011, output scaled 0-255). A To Workspace sink captures the result. The model was simulated in MATLAB R2024b with MSVC 2022, producing a 1824x510 enhanced B-mode image (values 0-253). Simulink was used for architectural representation and functional subsystem validation; all CNN training, GPU Coder acceleration, and real-time deployment were performed in MATLAB and Holoscan.

![Simulink Pipeline Diagram](Project%20Figures/Simulink/Simulink_Pipeline_Diagram_Simple.png)

![Simulink Functional Output](Project%20Figures/Simulink_Functional_Output.png)

---

## Results

This section reports what was actually measured: classification performance on the
held-out test set, and real-time throughput on the deployed pipeline.

### CNN Evaluation (665 test images)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| benign | 88.5% | 89.7% | 0.891 |
| malignant | 86.9% | 80.8% | 0.838 |
| normal | 83.9% | 94.0% | 0.886 |

**Overall Test Accuracy: 87.37%**

Malignant detection AUC-ROC: **0.9509** | AUC-PR: **0.8918**

### Pipeline Benchmark (100 OASBUD frames, GPU Coder CUDA beamforming)

| Stage | Mean (ms) | Share |
|---|---|---|
| Beamforming | 12.94 | 74% |
| Enhancement | 2.39 | 14% |
| ONNX Inference | 2.12 | 12% |
| **Total** | **17.46** | **100%** |

Throughput: **57.3 fps** (mean) | **39.9 fps** (p95) | Accuracy: **73.0%**

See [GPU Coder CUDA Acceleration](#gpu-coder-cuda-acceleration) above for the before/after comparison against the pre-acceleration scipy baseline.

---

## Training Dataset (Mega Model)

| Dataset | Images | Source |
|---|---|---|
| BUSI [2] | 780 | Standard benchmark, clinical scanner PNG |
| BUS-UCLM [3] | 646 | Spanish clinical scanner, Doppler filtered |
| BUS-BRA [4] | 1875 | Zenodo, largest available breast US dataset |
| BrEaST [5] | 256 | Cancer Imaging Archive, CC-BY 4.0 |
| OASBUD-PNG [1] | 880 | Reconstructed from OASBUD RF via A-line pipeline |
| **Total** | **4437** | Train 3106 / Val 666 / Test 665 |

Bracketed numbers reference the full dataset citations in [References](#references).

---

## Repository Structure

```
realtime-ultrasound-holoscan/
├── MATLAB Codes/
│   ├── das_beamform.m              DAS beamformer (GPU Coder entry point, comparison method only -- not used for OASBUD)
│   ├── run_codegen.m               GPU Coder script -> generates das_beamform CUDA lib
│   ├── aline_reconstruct.m         Per-column Hilbert + power-law GPU Coder entry point (the method actually used for OASBUD)
│   ├── aline_run_codegen.m         GPU Coder script -> generates aline_reconstruct CUDA lib
│   ├── das_beamform_planewave.m    Plane-wave DAS entry point (Pipeline 2, live/accelerated method)
│   ├── das_run_codegen_planewave.m GPU Coder script -> generates das_beamform_planewave CUDA lib
│   ├── fk_migration_planewave.m           F-k migration, broadside only (Pipeline 2, comparison method)
│   ├── fk_migration_planewave_multiangle.m F-k migration, arbitrary angles + compounding (Pipeline 2, comparison method)
│   ├── PlaneWave_Reconstruct_Test.m       First-reconstruction DAS test against real data
│   ├── FK_Migration_Test.m                F-k migration verification (broadside) against real wire target
│   ├── FK_Migration_MultiAngle_Test.m     F-k migration verification (multi-angle) with quantitative CNR measurement
│   ├── Phase7_PhasedArrayToolbox_Demo.m       Phased Array System Toolbox, synthetic data (superseded, kept for reference)
│   ├── Phase7_PhasedArrayToolbox_RealData.m   Phased Array System Toolbox, real CIRS040GSE channel data
│   ├── Phase3_Enhancement.m        Enhancement filter comparison
│   ├── Phase4_MegaTrain.m          Mega model training (all 5 datasets)
│   ├── Phase6_CNN_Evaluation.m     Comprehensive Phase 6 evaluation
│   ├── codegen/                    Generated CUDA source (das_beamform.cu, aline_reconstruct.cu, das_beamform_planewave.cu and supporting files)
│   ├── trainedMobileNetV2_mega.mat Trained network weights
│   └── trainedMobileNetV2_mega.onnx ONNX export for Python inference
│
├── Holoscan/
│   ├── medical_imaging_pipeline.py Main Holoscan application -- entry point
│   ├── data_source_op.py           RF data loader (defaults to bundled sample, override with OASBUD_PATH)
│   ├── beamforming_op.py           A-line reconstruction, CUDA path via aline_cuda.py with scipy fallback
│   ├── aline_cuda.py               ctypes binding to the GPU Coder-generated aline_reconstruct CUDA library
│   ├── aline_reconstruct_c_wrapper.cpp  extern "C" wrapper (GPU Coder's C++ output name-mangles otherwise)
│   ├── aline_cuda_verify.py        Correctness check: CUDA output vs scipy reference across all 100 OASBUD patients
│   ├── enhancement_op.py           Median filter + normalisation + resize
│   ├── inference_op.py             ONNX inference operator (model path override: ONNX_MODEL_PATH)
│   ├── output_op.py                Result recording (output dir override: RESULTS_DIR)
│   ├── Phase6_Pipeline_Benchmark.py Standalone timing benchmark (uses the CUDA path automatically when available)
│   └── requirements.txt            Python dependencies for the Holoscan pipeline
│
├── HoloscanPlaneWave/               Pipeline 2 -- completely separate app, real per-element data, no classification stage
│   ├── medical_imaging_pipeline.py Main Holoscan application -- entry point
│   ├── data_source_op.py           Loads CIRS040GSE/CIRS073_RUMC, applies lens delay/angle delay/channel 125 corrections
│   ├── beamforming_op.py           Plane-wave DAS, CUDA path via das_cuda_planewave.py with numpy fallback
│   ├── das_cuda_planewave.py       ctypes binding to the GPU Coder-generated das_beamform_planewave CUDA library
│   ├── das_beamform_planewave_c_wrapper.cpp  extern "C" wrapper
│   ├── das_cuda_planewave_verify.py Correctness check: CUDA output vs numpy reference on real data
│   ├── enhancement_op.py           Envelope detection + log compression + median filter (viewable B-mode, not a classifier tile)
│   ├── output_op.py                Saves reconstructed frames, no classification output
│   └── PlaneWave_Pipeline_Benchmark.py Per-stage timing benchmark
│
├── Simulink/
│   ├── UltrasoundPipelineDiagram.slx        Five-stage architecture diagram (visual reference)
│   └── UltrasoundEnhancementSubsystem.slx   Functional model -- envelope detection + median filter simulated in MATLAB R2024b
│
├── data/
│   └── sample/
│       └── OASBUD_sample.mat       10-patient bundled sample (5 malignant, 5 benign) for one-click testing
│                                   (kept small -- ~31 MB, ~3.1 MB/patient -- to stay well under GitHub's
│                                   50 MB soft limit; full reported results use all 100 OASBUD patients,
│                                   reproducible via the OASBUD_PATH override, see Setup below)
│
├── Project Figures/                Figures from Phases 3-6 and the Simulink diagram
│
├── Project_Documentation.docx      Complete project report (background, methodology, results, references)
│
├── .gitignore
└── README.md
```

Note: `Holoscan/aline_reconstruct.so`, the compiled CUDA library itself, is not committed
(machine- and OS-specific build output). Build it locally following the steps below;
`BeamformingOp` falls back to scipy automatically if it isn't present.

---

## Setup

### MATLAB (Windows)

- MATLAB R2024b with Deep Learning Toolbox, GPU Coder, Image Processing Toolbox
- CUDA 12.6, VS2022 Build Tools (MSVC v17), cuDNN v9.22
- RTX 4070 or equivalent NVIDIA GPU

Every session, run at the top of any codegen script:

```matlab
setenv('ProgramFiles(x86)', 'C:\Program Files (x86)');
setenv('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6');
```

Run `aline_run_codegen.m` (or `run_codegen.m` for the das_beamform comparison method) to generate CUDA source under `codegen/lib/`.

### Building the CUDA-accelerated beamforming library (WSL2)

The generated `.lib` from GPU Coder is Windows-only and cannot be loaded by Python in
WSL2. Build a Linux `.so` from the generated CUDA source instead:

```bash
# Copy the generated source (and the extern "C" wrapper) into one folder
mkdir -p ~/aline_cuda_src
cp "/mnt/c/Users/<you>/Documents/MATLAB Code/codegen/lib/aline_reconstruct/"* ~/aline_cuda_src/
cp Holoscan/aline_reconstruct_c_wrapper.cpp ~/aline_cuda_src/
cp "/mnt/c/Program Files/MATLAB/R2024b/extern/include/tmwtypes.h" ~/aline_cuda_src/
cd ~/aline_cuda_src

# CUDA 12.6 does not support the GCC version WSL2 Ubuntu 24.10 ships by default
sudo apt install -y gcc-13 g++-13

# If nvcc errors with "exception specification is incompatible" for cospi/sinpi/rsqrt,
# this is a known CUDA 12.6 / newer-glibc conflict (NVIDIA forums, confirmed fix):
# patch the six affected declarations to add the matching noexcept(true).
CUDA_MATH_H="/usr/local/cuda-12.6/targets/x86_64-linux/include/crt/math_functions.h"
sudo cp "$CUDA_MATH_H" "$CUDA_MATH_H.bak"
sudo sed -i \
  -e 's/\(double[[:space:]]*sinpi(double x)\);/\1 noexcept (true);/' \
  -e 's/\(float[[:space:]]*sinpif(float x)\);/\1 noexcept (true);/' \
  -e 's/\(double[[:space:]]*cospi(double x)\);/\1 noexcept (true);/' \
  -e 's/\(float[[:space:]]*cospif(float x)\);/\1 noexcept (true);/' \
  -e 's/\(double[[:space:]]*rsqrt(double x)\);/\1 noexcept (true);/' \
  -e 's/\(float[[:space:]]*rsqrtf(float x)\);/\1 noexcept (true);/' \
  "$CUDA_MATH_H"

nvcc -ccbin g++-13 -Xcompiler -fPIC -shared -o aline_reconstruct.so *.cu *.cpp -I. -lcufft

cp aline_reconstruct.so ~/project/Holoscan/   # or wherever your Holoscan/ folder lives

# Verify correctness against the scipy reference before trusting it in the live pipeline
cd ~/project/Holoscan   # or wherever aline_cuda_verify.py lives
export ALINE_CUDA_LIB_PATH=~/aline_cuda_src/aline_reconstruct.so
export OASBUD_PATH="/path/to/full/OASBUD.mat"
python3 aline_cuda_verify.py
```

A clean `PASS`, differences on the order of `1e-13` across all 100 patients, confirms the
library is safe to use. Full technical detail (why each of these steps is needed, and the
complete before/after benchmark) is in `Project_Documentation.docx`, Section 9 and Appendix 13.4.

### Holoscan (WSL2)

```bash
# Activate environment
source ~/holoscan-env-310/bin/activate && ulimit -s 32768

# Run from inside the cloned repo's Holoscan/ folder.
# No path editing needed -- both scripts default to the bundled
# 10-patient sample in data/sample/OASBUD_sample.mat and the
# ONNX model already committed under MATLAB Codes/.
cd realtime-ultrasound-holoscan/Holoscan
pip install -r requirements.txt   # holoscan, onnxruntime-gpu, scipy, pillow, matplotlib

# Run the full pipeline (10 sample frames)
python3 medical_imaging_pipeline.py 10

# Run the Phase 6 timing benchmark (10 sample frames)
python3 Phase6_Pipeline_Benchmark.py
```

To reproduce the full reported results (100 OASBUD patients, full BUSI/BUS-UCLM/etc.
training set), point the scripts at the full datasets instead of the bundled sample
via environment variables -- no code edits required:

```bash
export OASBUD_PATH="/path/to/full/OASBUD.mat"
export ONNX_MODEL_PATH="/path/to/trainedMobileNetV2_mega.onnx"   # optional, defaults to the committed model
export RESULTS_DIR="/path/to/output/folder"                      # optional, defaults next to the script
python3 medical_imaging_pipeline.py 100
```

The full OASBUD dataset [1], BUSI [2], BUS-UCLM [3], BUS-BRA [4], and BrEaST [5]
datasets used for full evaluation and training are not bundled in this repository due
to size; see the dataset references below for download links.

Requirements: WSL2 Ubuntu 24.10, Python 3.10, Holoscan 4.2.0, ONNX Runtime GPU.

---

## Quick Test / Verification

After installing requirements, run the pipeline on the bundled 10-frame sample to verify
the setup end-to-end. The sample is deliberately kept at 10 patients (5 malignant, 5 benign)
rather than the full 100: each OASBUD patient record is large (~3.1 MB), so 10 patients
already lands at ~31 MB, comfortably under GitHub's 50 MB soft-warning threshold while
still giving a class-balanced smoke test. This is meant to verify correctness, not
reproduce the full benchmark numbers reported above -- for that, run against the full
dataset via `OASBUD_PATH` (see Setup).

```bash
cd realtime-ultrasound-holoscan/Holoscan
python3 medical_imaging_pipeline.py 10
```

Expected output: a per-frame table (true label, prediction, confidence) for all 10 sample
patients, followed by a summary block reporting overall accuracy, malignant/benign recall,
average frame latency, and throughput in fps. Console output ending in `Pipeline Summary`
with no exceptions confirms the model, data loader, and ONNX runtime are all working
correctly. Results are also saved to `Holoscan/pipeline_results.npy`.

---

## Critical Notes

**ONNX inference in Python:**
- Input: raw `float32` values in range 0-255. No manual normalisation.
- Output node `new_softmax` already contains probabilities. No manual softmax.
- Class order: index 0 = benign, index 1 = malignant, index 2 = normal.

**OASBUD dataset [1]:**
- RF depth varies per patient (1040-2864 rows). Never hardcode 1824.
- `class` field: 0 = malignant, 1 = benign (counterintuitive).
- In Python: `getattr(patient, 'class')` -- `class` is a reserved keyword.
- Sampling frequency 40 MHz, element pitch 0.30 mm (Ultrasonix L14-5/38 transducer). Do not use 25 MHz / 0.245 mm.

**GPU Coder:**
- Always use R2024b. R2026a rejects VS2026 (MSVC v18).
- MEX build fails (bundled CUDA 12.2 missing cicc.exe). Library build succeeds and is what Holoscan uses.
- The generated `.lib` is Windows-only. Build a Linux `.so` from the generated `.cu`/`.cpp` source in WSL2 instead; see Setup.
- GPU Coder's C++ output name-mangles the exported symbol; ctypes needs the `extern "C"` wrapper in `aline_reconstruct_c_wrapper.cpp` to call it by name.

**DAS beamforming (das_beamform.m and related files):**
- Uses two-way (pulse-echo) travel time: `sample_idx = round((2*dist/c)*fs) + 1`.
- Not used for OASBUD reconstruction under any circumstances, correct physics or not, since OASBUD's RF columns are pre-formed scan lines rather than raw per-element data. See Project_Documentation.docx Section 5.2.

---

## References

**Datasets**

[1] Piotrzkowska-Wroblewska, H., Dobruch-Sobczak, K., Byra, M., and Nowicki, A. Open access database of raw ultrasonic signals acquired from malignant and benign breast lesions (OASBUD). Zenodo. https://doi.org/10.5281/zenodo.545928

[2] Al-Dhabyani, W., Gomaa, M., Khaled, H., and Fahmy, A. (2020). Dataset of Breast Ultrasound Images (BUSI). Data in Brief, 28, 104863.

[3] BUS-UCLM: Breast Ultrasound Lesion Segmentation Dataset, University of Castilla-La Mancha.

[4] BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-Aided Diagnosis Systems. Zenodo.

[5] BrEaST: Breast Lesions Ultrasound Dataset. The Cancer Imaging Archive, CC-BY 4.0 licence.

[6] Ultrasound Plane Wave Raw Data, 75 Angles, Breast Phantom and Calibration Phantom Dataset (CIRS040GSE). Zenodo, record 7986407.

**Tools and models**

[7] Howard, A., Zhmoginov, A., Chen, L.-C., Sandler, M., and Zhu, M. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.

[8] MathWorks. MATLAB R2024b, including the Deep Learning Toolbox, Image Processing Toolbox, GPU Coder, and Phased Array System Toolbox.

[9] NVIDIA Corporation. NVIDIA Holoscan SDK, version 4.2.0.

---

## Generative AI Usage

Generative AI (Claude, Anthropic) was used during this project to assist with debugging
(ONNX inference normalisation/softmax issues, GPU Coder environment configuration, WSL2
CUDA toolchain issues), pipeline integration, and documentation drafting. All technical
decisions, beamforming corrections, model architecture choices, and reported results were
reviewed, tested, and verified by the author. The author can explain and reproduce all
code in this repository.

---

## Contact

Rohith Ram V -- 24BCE0543, B.Tech CSE, VIT Vellore
GitHub: [@rohithgreninja-afk](https://github.com/rohithgreninja-afk)
