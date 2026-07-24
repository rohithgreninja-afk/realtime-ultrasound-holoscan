"""
aline_cuda_verify.py

Standalone correctness check: compares aline_reconstruct_cuda() output
against the scipy reference implementation on real OASBUD sample data,
patient by patient. Run this before trusting the CUDA path in the live
Holoscan pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert
from aline_cuda import aline_reconstruct_cuda, cuda_available

SAMPLE_PATH = os.environ.get(
    "OASBUD_PATH",
    os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', 'OASBUD_sample.mat')
)


def scipy_reference(rf, gamma=0.3):
    analytic = hilbert(rf, axis=0)
    envelope = np.abs(analytic)
    env_norm = envelope / (envelope.max() + 1e-12)
    return np.power(env_norm, gamma).astype(np.float64)


def main():
    if not cuda_available():
        print("CUDA library not loaded. Set ALINE_CUDA_LIB_PATH, or place "
              "aline_reconstruct.so in this folder, then try again.")
        return

    print(f"Loading OASBUD data from {SAMPLE_PATH}")
    mat = sio.loadmat(SAMPLE_PATH, squeeze_me=True, struct_as_record=False)
    data = mat['data']

    max_diffs = []
    for i, patient in enumerate(data):
        rf = patient.rf1.astype(np.float64)
        ref = scipy_reference(rf)
        cuda_out = aline_reconstruct_cuda(rf, gamma=0.3)

        if cuda_out.shape != ref.shape:
            print(f"Patient {i+1} ({patient.id}): SHAPE MISMATCH, "
                  f"scipy {ref.shape} vs cuda {cuda_out.shape}")
            max_diffs.append(float('inf'))
            continue

        diff = np.abs(ref - cuda_out)
        max_diffs.append(diff.max())
        print(f"Patient {i+1} ({patient.id}): shape {rf.shape}, "
              f"max abs diff = {diff.max():.2e}, mean abs diff = {diff.mean():.2e}")

    overall_max = max(max_diffs)
    print(f"\nOverall max abs diff across all patients: {overall_max:.2e}")
    if overall_max < 1e-6:
        print("PASS: CUDA output matches scipy reference within floating point tolerance.")
        print("Safe to proceed with wiring aline_cuda into BeamformingOp.")
    else:
        print("FAIL: CUDA output diverges from scipy reference. Do not wire this "
              "into the live pipeline yet, something in the binding is wrong.")


if __name__ == "__main__":
    main()
