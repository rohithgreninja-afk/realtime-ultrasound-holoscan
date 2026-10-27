#!/usr/bin/env python3
"""
Phase6_Pipeline_Benchmark.py
Measures per-stage latency and accuracy on the OASBUD hold-out patients
(patients not used for CNN training, per MATLAB Codes/cnn_split.mat),
calling the same functions the Holoscan operators use.
Set OASBUD_INCLUDE_TRAINING=1 to score every record in the file instead;
that number includes training patients and is labelled as such.

Run in WSL2:
  source ~/holoscan-env-310/bin/activate && ulimit -s 32768
  export RESULTS_DIR=/path/to/results
  python3 Phase6_Pipeline_Benchmark.py
"""

import time, os, sys
import numpy as np
from scipy.signal import hilbert
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aline_cuda import aline_reconstruct_cuda, cuda_available
from data_source_op import load_oasbud_records
from enhancement_op import prepare_tile
from inference_op import binary_prediction

# ---- Paths ----
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
OASBUD_PATH = os.environ.get(
    'OASBUD_PATH', os.path.join(_REPO_ROOT, 'data', 'sample', 'OASBUD_sample.mat'))
ONNX_PATH   = os.environ.get(
    'ONNX_MODEL_PATH', os.path.join(_REPO_ROOT, 'MATLAB Codes', 'trainedMobileNetV2_mega.onnx'))
if not os.environ.get('RESULTS_DIR'):
    sys.exit('RESULTS_DIR environment variable not set. export RESULTS_DIR=/path/to/results')
OUTPUT_DIR  = os.path.join(os.environ['RESULTS_DIR'], 'pipeline1_benchmark')
N_WARMUP    = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=== Phase 6 Pipeline Benchmark ===')
print(f'OASBUD : {OASBUD_PATH}')
print(f'Model  : {ONNX_PATH}')
print(f'Output : {OUTPUT_DIR}')
print(f'Beamforming backend : {"CUDA (GPU Coder)" if cuda_available() else "scipy (fallback)"}\n')

# ============================================================
# Stage functions -- same code paths as the Holoscan operators
# ============================================================

def beamform(rf):
    """BeamformingOp._aline_reconstruct -- CUDA path if available, scipy fallback otherwise"""
    if cuda_available():
        return aline_reconstruct_cuda(rf.astype(np.float64), 0.3).astype(np.float32)
    analytic = hilbert(rf.astype(np.float64), axis=0)
    envelope = np.abs(analytic)
    env_norm = envelope / (envelope.max() + 1e-12)
    return np.power(env_norm, 0.3).astype(np.float32)

def enhance(bmode):
    """EnhancementOp: prepare_tile (uint8, 3x3 filter, resize 224x224, RGB)"""
    return prepare_tile(bmode)

def preprocess(image):
    """InferenceOp._infer -- layout conversion only"""
    img_chw = np.transpose(image.astype(np.float32), (2, 0, 1))
    return np.expand_dims(img_chw, axis=0)   # [1 x 3 x 224 x 224]

# ============================================================
# Load data
# ============================================================
print('Loading OASBUD ...')
t0 = time.perf_counter()
patients, n_in_file = load_oasbud_records(OASBUD_PATH)
n_patients = len(patients)
INCLUDE_TRAINING = os.environ.get('OASBUD_INCLUDE_TRAINING', '0') == '1'
SUBSET = 'all records, INCLUDES TRAINING PATIENTS' if INCLUDE_TRAINING else 'hold-out patients only'
print(f'  {n_patients} of {n_in_file} patients ({SUBSET}) in {(time.perf_counter()-t0)*1000:.1f} ms\n')
if n_patients == 0:
    sys.exit('No hold-out patients in this file. Set OASBUD_PATH to the full OASBUD.mat.')

# ============================================================
# Init ONNX session
# ============================================================
print('Initialising ONNX session ...')
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session    = ort.InferenceSession(
    ONNX_PATH, sess_options=opts,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
input_name = session.get_inputs()[0].name
active_ep  = session.get_providers()[0]
print(f'  Input : {input_name}  {session.get_inputs()[0].shape}')
print(f'  Output: {session.get_outputs()[0].name}')
print(f'  EP    : {active_ep}\n')

# ============================================================
# Warm-up
# ============================================================
print(f'Warm-up ({N_WARMUP} frames) ...')
for i in range(N_WARMUP):
    rf = np.asarray(patients[i % n_patients]['rf1'], dtype=np.float64)
    session.run(None, {input_name: preprocess(enhance(beamform(rf)))})
print('  Done.\n')

# ============================================================
# Timed run
# ============================================================
t_beamform  = np.zeros(n_patients)
t_enhance   = np.zeros(n_patients)
t_preproc   = np.zeros(n_patients)
t_inference = np.zeros(n_patients)
t_total     = np.zeros(n_patients)
preds, gt   = [], []

print(f'Timing {n_patients} frames ...')
for i, p in enumerate(patients):
    rf  = np.asarray(p['rf1'], dtype=np.float64)
    lbl = int(p['class'])  # 0=benign  1=malignant

    frame_t0 = time.perf_counter()

    t0 = time.perf_counter()
    bmode = beamform(rf)
    t_beamform[i] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    enhanced = enhance(bmode)
    t_enhance[i] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    img_batch = preprocess(enhanced)
    t_preproc[i] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    probs = session.run(None, {input_name: img_batch})[0][0]
    pred  = binary_prediction(probs)   # same decision as InferenceOp
    t_inference[i] = (time.perf_counter() - t0) * 1000

    t_total[i] = (time.perf_counter() - frame_t0) * 1000

    preds.append(pred)
    gt.append('malignant' if lbl == 1 else 'benign')

    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{n_patients}  last_frame={t_total[i]:.1f} ms')

# ============================================================
# Accuracy
# ============================================================
preds    = np.array(preds)
gt       = np.array(gt)
accuracy = float(np.mean(preds == gt))

# ============================================================
# Print results table
# ============================================================
print()
print('=' * 62)
print('PHASE 6 PIPELINE BENCHMARK')
print('=' * 62)
print(f'  Execution provider  : {active_ep}')
print(f'  Beamforming backend : {"CUDA (GPU Coder)" if cuda_available() else "scipy (fallback)"}')
print(f'  Frames timed        : {n_patients} ({SUBSET})')
print()
print(f'  {"Stage":<22} {"Mean":>7} {"Std":>7} {"Min":>7} {"Max":>7}   ms')
print(f'  {"-"*22} {"-"*7} {"-"*7} {"-"*7} {"-"*7}')

stages = [
    ('Beamforming',     t_beamform),
    ('Enhancement',     t_enhance),
    ('Preprocess',      t_preproc),
    ('ONNX Inference',  t_inference),
    ('--- Total ---',   t_total),
]
for name, arr in stages:
    print(f'  {name:<22} {arr.mean():>7.2f} {arr.std():>7.2f} {arr.min():>7.2f} {arr.max():>7.2f}')

fps_mean = 1000.0 / t_total.mean()
fps_p95  = 1000.0 / np.percentile(t_total, 95)
print()
print(f'  Throughput (mean frame) : {fps_mean:.1f} fps')
print(f'  Throughput (p95  frame) : {fps_p95:.1f} fps')
print()
print(f'  Pipeline accuracy ({SUBSET}) : {100*accuracy:.1f}%  ({int(np.sum(preds==gt))}/{n_patients})')
for cls in ['benign', 'malignant']:
    tp = int(np.sum((preds == cls) & (gt == cls)))
    fn = int(np.sum((preds != cls) & (gt == cls)))
    fp = int(np.sum((preds == cls) & (gt != cls)))
    n  = int(np.sum(gt == cls))
    recall = tp / (tp + fn + 1e-9)
    prec   = tp / (tp + fp + 1e-9)
    print(f'  {cls:<12} recall={100*recall:.1f}%  precision={100*prec:.1f}%  ({tp}/{n})')
unique, counts = np.unique(preds, return_counts=True)
print(f'  Prediction distribution : {dict(zip(unique, counts.tolist()))}')
print('=' * 62)

# ============================================================
# Figure 1: Stage latency bar chart with error bars
# ============================================================
stage_names  = ['Beamforming', 'Enhancement', 'Preprocess', 'ONNX Inference']
stage_arrays = [t_beamform,    t_enhance,     t_preproc,    t_inference]
means = [a.mean() for a in stage_arrays]
stds  = [a.std()  for a in stage_arrays]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(stage_names, means, yerr=stds, capsize=6,
              color=colors, edgecolor='black', linewidth=0.8)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.15,
            f'{mean:.2f} ms', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
ax.set_ylabel('Latency (ms)', fontsize=13)
backend_label = "CUDA" if cuda_available() else "scipy"
ax.set_title(
    f'Pipeline Stage Latency  (n={n_patients} frames, beamforming: {backend_label})\n'
    f'Total mean: {t_total.mean():.2f} ms  |  {fps_mean:.1f} fps',
    fontsize=13
)
ax.grid(axis='y', alpha=0.4)
ax.set_ylim(0, max(means) * 1.4)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/stage_latency_bar.png', dpi=150)
print(f'  Saved: stage_latency_bar.png')

# ============================================================
# Figure 2: Per-frame total latency over time
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(range(1, n_patients + 1), t_total, 'b-o', markersize=3, linewidth=1.2)
ax2.axhline(t_total.mean(), color='r', linestyle='--', linewidth=1.5,
            label=f'Mean {t_total.mean():.2f} ms')
ax2.axhline(33.3, color='orange', linestyle=':', linewidth=1.8,
            label='30 fps target (33.3 ms)')
ax2.set_xlabel('Frame index', fontsize=12)
ax2.set_ylabel('Total latency (ms)', fontsize=12)
ax2.set_title(f'Per-Frame Total Latency -- {n_patients} OASBUD Patients ({SUBSET})', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
plt.tight_layout()
fig2.savefig(f'{OUTPUT_DIR}/per_frame_latency.png', dpi=150)
print(f'  Saved: per_frame_latency.png')

# ============================================================
# Figure 3: Stacked bar -- stage composition of mean frame
# ============================================================
fig3, ax3 = plt.subplots(figsize=(5, 5))
bottom = 0
for name, mean, color in zip(stage_names, means, colors):
    ax3.bar('Mean frame', mean, bottom=bottom, color=color,
            label=f'{name} ({mean:.2f} ms)', edgecolor='white', linewidth=0.5)
    ax3.text(0, bottom + mean / 2,
             f'{100 * mean / t_total.mean():.1f}%',
             ha='center', va='center',
             color='white', fontweight='bold', fontsize=11)
    bottom += mean
ax3.set_ylabel('Latency (ms)', fontsize=13)
ax3.set_title(f'Stage Composition\n(total mean = {t_total.mean():.2f} ms)', fontsize=13)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig3.savefig(f'{OUTPUT_DIR}/stage_composition.png', dpi=150)
print(f'  Saved: stage_composition.png')

# ============================================================
# Save raw timing arrays
# ============================================================
np.save(f'{OUTPUT_DIR}/benchmark_timing.npy', {
    't_beamform':  t_beamform,
    't_enhance':   t_enhance,
    't_preproc':   t_preproc,
    't_inference': t_inference,
    't_total':     t_total,
    'preds':       preds,
    'gt':          gt,
    'accuracy':    accuracy,
    'subset':      SUBSET,
    'patient_ids': [str(p['id']) for p in patients],
    'fps':         fps_mean,
    'beamforming_backend': 'cuda' if cuda_available() else 'scipy',
})
print(f'  Saved: benchmark_timing.npy')
print()
print('Benchmark complete.')
