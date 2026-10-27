# AUDIT (temporary, deleted before the final commit)

Audit of branch `review-response` at commit 2d0b4b6 against the MathWorks review, performed before any change. Status values: FIXED (already addressed), PRESENT (the problem exists as described), PARTIAL (partly addressed or partly different from the review text).

## Discrepancies between the review text and the real repository

The review refers to files without underscores. The real names are `Phase4_MegaTrain.m`, `Phase6_CNN_Evaluation.m`, `Phase4_OASBUD_Export.m`, `Phase3_Enhancement.m`, `Holoscan/enhancement_op.py`, `DEMO_RUN_COMMANDS.md` and `FK_Migration_MultiAngle_Test.m`. The plane-wave Holoscan app lives at `HoloscanPlaneWave/` in the repo root, not under `Holoscan/`. The figures folder is `Project Figures/` (with a space), not `Project_Figures/`. No git tags exist locally or on the remote, so there is no V1.2 tag; V1.3 will be the first tag. All line numbers quoted in the review match the real files.

## New finding not in the review: OASBUD class labels are inverted

Every OASBUD consumer in the repo treats `class == 0` as malignant (`Phase6_CNN_Evaluation.m:286-288,308`, `Holoscan/data_source_op.py:49,84`, `Holoscan/output_op.py:20,52`, `Holoscan/Phase6_Pipeline_Benchmark.py:121,145`, `Phase5_BeamformingImproved.m:296`, README.md:521). The full OASBUD.mat contradicts this. It has 48 records with class 0 and 52 with class 1, and the OASBUD paper reports 48 benign and 52 malignant lesions. The BI-RADS field agrees: 31 of the 48 class-0 records are BI-RADS 3 (probably benign) and only one is BI-RADS 5, while 50 of the 52 class-1 records are BI-RADS 4b, 4c or 5. So class 0 is benign and class 1 is malignant. Consequences: the 40 OASBUD patients exported by `Phase4_OASBUD_Export.m:33-34` were written to the opposite class folders (880 training PNGs with swapped labels), and every OASBUD pipeline accuracy so far (73.0%, Section 9 of the evaluation) was scored against inverted ground truth.

## Phase 1: Dataset integrity and CNN retraining

1a, PRESENT. Datastore construction is duplicated in `Phase4_MegaTrain.m:12-136` and `Phase6_CNN_Evaluation.m:68-148`. The BrEaST block in evaluation filters labels and skips mask names (`Phase6_CNN_Evaluation.m:117-130`); training does not (`Phase4_MegaTrain.m:89-100`). The two scripts also build BUS-BRA differently (label filter only in evaluation, lines 96-103). `Phase4_AI.m` and `Phase4_Combined_Training.m` carry older copies of the same logic.

1b, PRESENT. `Phase4_OASBUD_Export.m:33-34` hardcodes full-dataset indices (up to 66), so it indexes past the 10-record bundled sample. It writes before it fails because patient loops run in order. Augmentation is hand-rolled (lines 86-133). There is no label check on the output folder. With the label inversion above, the folder assignment is also wrong on the full dataset.

1c, PRESENT. Training splits with two successive `splitEachLabel` calls after `rng(42)` (`Phase4_MegaTrain.m:145-147`). Evaluation re-splits with a single three-way call (`Phase6_CNN_Evaluation.m:152-153`), which produces a different partition. No file list is saved. OASBUD-PNG images from one patient (base, view2 and 20 augmentations) can land in train and test at the same time, because the split is per image, not per patient.

1d, PRESENT. Uses `mobilenetv2` (`Phase4_MegaTrain.m:174`), `layerGraph` (175), `classificationLayer` (185, 211), `trainNetwork` (234), `classify` (241, and `Phase6_CNN_Evaluation.m:163,256,263,319`), and a DAGNetwork. Class weights are computed as sqrt of inverse frequency with a 2.5 cap (lines 198-204), not a single configurable value.

1e, 1f, 1g, PRESENT. No training log, no saved evaluation file, ONNX exported with `exportONNXNetwork` from a DAGNetwork. README.md:270-280 and DEMO_RUN_COMMANDS.md:74 report 87.37%, AUC-ROC 0.9509 and AUC-PR 0.8918, all from the unrecoverable split.

## Phase 2: Pipeline accuracy and preprocessing

2a, PRESENT. `Holoscan/Phase6_Pipeline_Benchmark.py:119-145` scores all patients in the file, including the 40 used for training. The benchmark also takes a 3-class argmax (line 139) where `Holoscan/inference_op.py` merges normal into benign, so the benchmark and the live pipeline make different decisions.

2b, PRESENT. `Holoscan/enhancement_op.py:22-23` min-max renormalises before the median filter; `Phase6_CNN_Evaluation.m:315-316` does not. In addition, the benchmark's `enhance()` (`Phase6_Pipeline_Benchmark.py:58-64`) has no median filter at all, so it is not an exact copy of `EnhancementOp` as its docstring claims.

2c, PRESENT. `DEMO_RUN_COMMANDS.md:73-74` attaches the MATLAB CNN figure (87.37%) to the Pipeline 1 benchmark line, which actually prints 73.0%.

## Phase 3: Enhancement filter selection

3a, 3b, PRESENT. `Phase3_Enhancement.m:172-183` scores PSNR and SSIM against the unfiltered input. No CNR, gCNR, speckle SNR, edge measure or phantom test exists.

3c, PRESENT. Median 3x3 is deployed in `Holoscan/enhancement_op.py:24`, `HoloscanPlaneWave/enhancement_op.py:210`, and the Simulink Enhancement block, whose comment cites "PSNR 24.77 dB, SSIM 0.4011". README.md:236-245 repeats those numbers.

3d, PRESENT. `Phase3_Enhancement.m` is a script that reads `bmode_log`, `x_image`, `z_image` from the workspace with no stated prerequisite (they come from `Beamforming.m`).

3e, PRESENT. `Phase3_Enhancement.m:228-232` saves by figure number, so the saved files depend on what else is open; figure 24 does not exist in a fresh session.

## Phase 4: Reproducibility and run instructions

4a, PRESENT. `Phase6_CNN_Evaluation.m:22-43` errors on four dataset variables. Section 9 uses `malPat(21:end)` (line 294), which is empty on the 10-patient sample.

4b, PARTIAL. `.gitignore` excludes `MATLAB Codes/codegen/`. `das_beamform.cu` and its siblings are tracked anyway (force-added), but `aline_reconstruct.cu` and `das_beamform_planewave.cu` are not in the repo. README.md:333 claims all three are under `codegen/`.

4c, PRESENT. Fallbacks to `C:\Users\rohit\Downloads\CIRS040GSE\CIRS040GSE` in `FK_Migration_MultiAngle_Test.m:15`, `FK_Migration_Test.m:18`, `Phase7_PhasedArrayToolbox_RealData.m:27`, `PlaneWave_Reconstruct_Test.m:17`.

4d, PRESENT. No plane-wave sample is bundled; `HoloscanPlaneWave/data_source_op.py:43-50` requires the external download.

4e, PRESENT. No single Reproduce section. Several README numbers have no named source output file.

## Phase 5: Toolbox usage and Simulink

5a, PRESENT. `Phase7_PhasedArrayToolbox_RealData.m` only beamforms toward broadside and plots a 1-D trace. No 2-D toolbox image exists.

5b, PRESENT. `beamform_mvbf.m` is a hand-written Capon beamformer used in `Phase5_ReconstructionComparison.m`. Element positions use the manual idiom in `Beamforming.m:21-22`, `Beamforming_Comparing_Patients.m:25`, `Phase5_ReconstructionComparison.m:33-34`, `run_codegen.m:39`, and the centred variant in `Phase5_BeamformingImproved.m:26`, `PlaneWave_Reconstruct_Test.m:84`, `FK_Migration_Test.m:81`, `FK_Migration_MultiAngle_Test.m:88`.

5c, PARTIAL. README.md:257 already calls `UltrasoundEnhancementSubsystem.slx` functional; the "illustrative" wording is in the report, not the README. No driver script exists. The model reads `rf_ts` (From Workspace) and writes `enhanced_bmode`; StopTime is 0. The EnvelopeDetection block uses 60 dB log compression, not the gamma 0.3 power law of the deployed pipeline, and the Enhancement block comment cites the unreproducible PSNR/SSIM.

5d, PRESENT. `Phase5_BeamformingImproved.m:66` allocates `zeros(nz, nx, N)` complex, 300 x 300 x 510, about 734 MB.

5e, PRESENT. `fk_migration_planewave_multiangle.m:106` is `frameK(1) = 0`.

## Phase 6: Cleanup

6a, PRESENT. Superseded scripts sit beside live ones: `Phase4_AI.m`, `Phase4_Combined_Training.m`, `Phase7_PhasedArrayToolbox_Demo.m`, `Beamforming_Comparing_Patients.m`, `BUS_UCLM_Explore.m`.

6b, PRESENT. `Phase4_MegaTrain.m:10,279-280` hardcode `C:\ProgramData\MATLAB\SupportPackages\R2024b\...`.

6c, PRESENT. `Demo Video.mov` (48 MB) is committed.

6d, PRESENT. Personal paths: the four files in 4c. Em dashes: `Holoscan/data_source_op.py`, `Beamforming_Comparing_Patients.m`, `beamform_dmas.m`, `beamform_fk.m`, `beamform_mvbf.m`, `BUS_UCLM_Explore.m`, `Phase3_Enhancement.m`, `Phase4_AI.m`, `Phase5_ReconstructionComparison.m`, `Phase7_PhasedArrayToolbox_Demo.m`, `run_codegen.m`. No hits for "Veer" or any reviewer name in text files.
