# Real-Time Acceleration for Medical Image Processing

**MathWorks MATLAB-Simulink Challenge Project**
Rohith Ram V | 24BCE0543 | B.Tech CSE | VIT Vellore

A complete end-to-end pipeline for real-time breast ultrasound classification: raw RF data goes in, a malignant/benign/normal prediction comes out at 34 fps on an NVIDIA RTX 4070, deployed via NVIDIA Holoscan SDK.

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
     |
     v
EnhancementOp       -- normalise -> resize 224x224 bilinear -> convert to RGB
     |
     v
InferenceOp         -- ONNX MobileNetV2 mega model via CUDAExecutionProvider
     |
     v
OutputOp            -- record prediction, print summary, save results
```

---

## Image Enhancement

Four candidate speckle-reduction filters from the Image Processing Toolbox were compared on reconstructed B-mode images, scored by PSNR and structural similarity (SSIM):

| Filter | PSNR | SSIM | Notes |
|---|---|---|---|
| **Median 3x3 (selected)** | **24.77 dB** | **0.4011** | Best SSIM, used throughout the pipeline |
| Wiener 5x5 | -- | -- | Evaluated, did not outperform median |
| Adaptive histogram equalisation | -- | -- | Excluded: amplifies arc-shaped artefacts in log-compressed images |
| Bilateral filter | -- | -- | Over-smoothed; SSIM 0.3329 |

The 3x3 median filter was selected and is the `EnhancementOp` stage in the deployed pipeline.

![Enhancement Filter Comparison](Project%20Figures/Phase3/Phase3_Final_Summary.png)

---

## Simulink Representation

The same five-stage pipeline (DataSource -> Beamforming -> Enhancement -> Inference -> Output) is also expressed as a Simulink block diagram, `Simulink/UltrasoundPipelineDiagram.slx`. This is an illustrative architecture diagram only: it contains plain labelled blocks with no underlying `matlab.System` classes, no executable logic, and no simulation behaviour of any kind, it exists purely to visualise the pipeline structure. The actual real-time execution path remains the Holoscan pipeline described above.

![Simulink Pipeline Diagram](Project%20Figures/Simulink/Simulink_Pipeline_Diagram_Simple.png)

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

### Pipeline Benchmark (100 OASBUD frames)

| Stage | Mean (ms) | Share |
|---|---|---|
| Beamforming | 23.50 | 80% |
| Enhancement | 2.33 | 8% |
| ONNX Inference | 3.49 | 12% |
| **Total** | **29.37** | **100%** |

Throughput: **34.0 fps** (mean) | **16.0 fps** (p95) | Accuracy: **73.0%**

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
│   ├── das_beamform.m              DAS beamformer (GPU Coder entry point)
│   ├── run_codegen.m               GPU Coder script -> generates CUDA lib
│   ├── Phase3_Enhancement.m        Enhancement filter comparison
│   ├── Phase4_MegaTrain.m          Mega model training (all 5 datasets)
│   ├── Phase6_CNN_Evaluation.m     Comprehensive Phase 6 evaluation
│   ├── codegen/                    Generated CUDA source (das_beamform.cu and supporting files)
│   ├── trainedMobileNetV2_mega.mat Trained network weights
│   └── trainedMobileNetV2_mega.onnx ONNX export for Python inference
│
├── Holoscan/
│   ├── medical_imaging_pipeline.py Main Holoscan application
│   ├── data_source_op.py           RF data loader
│   ├── beamforming_op.py           A-line reconstruction
│   ├── enhancement_op.py           Normalisation and resize
│   ├── inference_op.py             ONNX inference operator
│   ├── output_op.py                Result recording
│   └── Phase6_Pipeline_Benchmark.py Standalone timing benchmark
│
├── Simulink/
│   └── UltrasoundPipelineDiagram.slx  Five-stage architecture diagram (illustrative only, no executable logic)
│
├── Project Figures/                Figures from Phases 3-6 and the Simulink diagram
│
├── Project_Documentation.docx      Complete project report (background, methodology, results, references)
│
└── README.md
```

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

### Holoscan (WSL2)

```bash
# Activate environment
source ~/holoscan-env-310/bin/activate && ulimit -s 32768

# Run full pipeline
python3 ~/project/medical_imaging_pipeline.py

# Run Phase 6 benchmark
python3 ~/project/Phase6_Pipeline_Benchmark.py
```

Requirements: WSL2 Ubuntu 24.10, Python 3.10, Holoscan 4.2.0, ONNX Runtime GPU.

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

**GPU Coder:**
- Always use R2024b. R2026a rejects VS2026 (MSVC v18).
- MEX build fails (bundled CUDA 12.2 missing cicc.exe). Library build succeeds and is what Holoscan uses.

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