"""
PlaneWave_Pipeline_Benchmark.py

Per-stage timing benchmark for the plane wave pipeline, same discipline
as Holoscan/Phase6_Pipeline_Benchmark.py: measures data loading,
beamforming, enhancement, and save separately across all real frames,
rather than one blended per-frame number, so the actual CUDA speedup on
beamforming specifically can be seen instead of being diluted by disk
I/O and preprocessing time.

Usage:
  export PLANEWAVE_DATA_PATH=/path/to/CIRS040GSE
  export DAS_PLANEWAVE_CUDA_LIB_PATH=/path/to/das_beamform_planewave.so
  python3 PlaneWave_Pipeline_Benchmark.py
"""
import os
import sys
import time
import numpy as np
from PIL import Image
from scipy.signal import hilbert
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_source_op import DataSourceOp
from das_cuda_planewave import das_beamform_planewave, cuda_available

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get('PLANEWAVE_RESULTS_DIR',
                             os.path.join(_THIS_DIR, 'planewave_benchmark_results'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=== Plane Wave Pipeline Benchmark ===')
print(f'Beamforming backend : {"CUDA (GPU Coder)" if cuda_available() else "numpy (fallback)"}')
print(f'Output              : {OUTPUT_DIR}\n')


def enhance(bmode):
    envelope = abs(hilbert(bmode, axis=0))
    env_norm = envelope / (envelope.max() + 1e-12)
    db = 20 * np.log10(env_norm + 1e-6)
    db = np.clip(db, -60, 0)
    img_norm = (db + 60) / 60
    img_uint8 = (img_norm * 255).astype(np.uint8)
    return median_filter(img_uint8, size=3)


op = DataSourceOp.__new__(DataSourceOp)
op.start()
n_frames = len(op.data_files)
x_elements = (np.arange(op.n_ele) - (op.n_ele - 1) / 2) * op.pitch
broadside_idx = int(np.argmin(np.abs(op.xmit_angles)))
tx_angle = np.deg2rad(op.xmit_angles[broadside_idx])

print(f'Frames to benchmark : {n_frames}')
print(f'Backend confirmed   : {"CUDA" if cuda_available() else "numpy"}\n')

t_load = np.zeros(n_frames)
t_beamform = np.zeros(n_frames)
t_enhance = np.zeros(n_frames)
t_save = np.zeros(n_frames)
t_total = np.zeros(n_frames)

print('Warm-up frame (JIT/first-call overhead)...')
rf3d_w = op._load_frame(op.data_files[0])
das_beamform_planewave(rf3d_w[:, :, broadside_idx], x_elements,
                        np.linspace(1e-3, 1e-2, 400), np.linspace(-1e-2, 1e-2, 300),
                        op.fs, op.c, tx_angle)
print('Done.\n')

print(f'Timing {n_frames} frames...')
for i, path in enumerate(op.data_files):
    frame_t0 = time.perf_counter()

    t0 = time.perf_counter()
    rf3d = op._load_frame(path)
    rf_single = rf3d[:, :, broadside_idx]
    t_load[i] = (time.perf_counter() - t0) * 1000

    num_samples = rf_single.shape[0]
    z_max = (num_samples / op.fs) * op.c / 2
    z_image = np.linspace(1e-3, z_max, 400)
    x_image = np.linspace(x_elements[0], x_elements[-1], 300)

    t0 = time.perf_counter()
    bmode = das_beamform_planewave(rf_single, x_elements, z_image, x_image,
                                    op.fs, op.c, tx_angle)
    t_beamform[i] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    enhanced = enhance(bmode)
    t_enhance[i] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    fname = os.path.join(OUTPUT_DIR, f'benchmark_frame_{i:04d}.png')
    Image.fromarray(enhanced, mode='L').save(fname)
    t_save[i] = (time.perf_counter() - t0) * 1000

    t_total[i] = (time.perf_counter() - frame_t0) * 1000

    if (i + 1) % 5 == 0:
        print(f'  {i+1}/{n_frames}  last_frame={t_total[i]:.1f} ms')

print()
print('=' * 62)
print('PLANE WAVE PIPELINE BENCHMARK (per stage, single broadside angle)')
print('=' * 62)
print(f'  Beamforming backend : {"CUDA (GPU Coder)" if cuda_available() else "numpy (fallback)"}')
print(f'  Frames timed        : {n_frames}')
print()
print(f'  {"Stage":<14} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}   ms')
print(f'  {"-"*14} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')

stages = [
    ('Data load',    t_load),
    ('Beamforming',  t_beamform),
    ('Enhancement',  t_enhance),
    ('Save',         t_save),
    ('--- Total ---', t_total),
]
for name, arr in stages:
    print(f'  {name:<14} {arr.mean():>8.2f} {arr.std():>8.2f} {arr.min():>8.2f} {arr.max():>8.2f}')

fps_mean = 1000.0 / t_total.mean()
print()
print(f'  Throughput (mean frame) : {fps_mean:.1f} fps')
print(f'  Beamforming share of total: {100*t_beamform.mean()/t_total.mean():.1f}%')
print('=' * 62)

stage_names = ['Data load', 'Beamforming', 'Enhancement', 'Save']
stage_arrays = [t_load, t_beamform, t_enhance, t_save]
means = [a.mean() for a in stage_arrays]
stds = [a.std() for a in stage_arrays]
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(stage_names, means, yerr=stds, capsize=6, color=colors,
              edgecolor='black', linewidth=0.8)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.15,
            f'{mean:.1f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Latency (ms)', fontsize=13)
backend_label = 'CUDA' if cuda_available() else 'numpy'
ax.set_title(f'Plane Wave Pipeline Stage Latency (n={n_frames} frames, beamforming: {backend_label})\n'
             f'Total mean: {t_total.mean():.2f} ms  |  {fps_mean:.1f} fps', fontsize=13)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'planewave_stage_latency_bar.png'), dpi=150)
print(f'\nSaved: planewave_stage_latency_bar.png')

np.save(os.path.join(OUTPUT_DIR, 'planewave_benchmark_timing.npy'), {
    't_load': t_load, 't_beamform': t_beamform, 't_enhance': t_enhance,
    't_save': t_save, 't_total': t_total, 'fps': fps_mean,
    'backend': 'cuda' if cuda_available() else 'numpy',
})
print('Saved: planewave_benchmark_timing.npy')
print('\nBenchmark complete.')
