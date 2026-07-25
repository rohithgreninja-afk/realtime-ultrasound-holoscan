"""
das_cuda_planewave_verify.py

Standalone correctness check: compares das_beamform_planewave_cuda()
output against the numpy reference implementation on real CIRS040GSE
data, single broadside angle, across several real frames. Run this
before trusting the CUDA path in the live Holoscan pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from data_source_op import DataSourceOp
from das_cuda_planewave import (
    das_beamform_planewave_cuda,
    das_beamform_planewave_numpy,
    cuda_available,
    IMG_DEPTH_PX,
    IMG_LATERAL_PX,
)


def main():
    if not cuda_available():
        print("CUDA library not loaded. Set DAS_PLANEWAVE_CUDA_LIB_PATH, or "
              "place das_beamform_planewave.so in this folder, then try again.")
        return

    data_root = os.environ.get("PLANEWAVE_DATA_PATH")
    if not data_root:
        print("PLANEWAVE_DATA_PATH not set.")
        return

    op = DataSourceOp.__new__(DataSourceOp)
    op.start()

    n_frames_to_check = int(os.environ.get("PLANEWAVE_VERIFY_N_FRAMES", 5))
    frame_indices = list(range(min(n_frames_to_check, len(op.data_files))))

    x_elements = (np.arange(op.n_ele) - (op.n_ele - 1) / 2) * op.pitch
    broadside_idx = int(np.argmin(np.abs(op.xmit_angles)))
    tx_angle = np.deg2rad(op.xmit_angles[broadside_idx])

    max_diffs = []
    for i in frame_indices:
        rf3d = op._load_frame(op.data_files[i])   # [time, elements, angles]
        rf = rf3d[:, :, broadside_idx]

        num_samples = rf.shape[0]
        z_max = (num_samples / op.fs) * op.c / 2
        z_image = np.linspace(1e-3, z_max, IMG_DEPTH_PX)
        x_image = np.linspace(x_elements[0], x_elements[-1], IMG_LATERAL_PX)

        ref = das_beamform_planewave_numpy(rf, x_elements, z_image, x_image,
                                            op.fs, op.c, tx_angle)
        cuda_out = das_beamform_planewave_cuda(rf, x_elements, z_image, x_image,
                                                op.fs, op.c, tx_angle)

        if cuda_out.shape != ref.shape:
            print(f"Frame {i} ({os.path.basename(op.data_files[i])}): SHAPE MISMATCH, "
                  f"numpy {ref.shape} vs cuda {cuda_out.shape}")
            max_diffs.append(float('inf'))
            continue

        diff = np.abs(ref - cuda_out)
        max_diffs.append(diff.max())
        print(f"Frame {i} ({os.path.basename(op.data_files[i])}): "
              f"max abs diff = {diff.max():.2e}, mean abs diff = {diff.mean():.2e}")

    overall_max = max(max_diffs)
    print(f"\nOverall max abs diff across {len(frame_indices)} frame(s): {overall_max:.2e}")
    if overall_max < 1e-6:
        print("PASS: CUDA output matches numpy reference within floating point tolerance.")
        print("Safe to proceed with wiring das_cuda_planewave into BeamformingOp.")
    else:
        print("FAIL: CUDA output diverges from numpy reference. Do not wire this "
              "into the live pipeline yet, something in the binding is wrong.")


if __name__ == "__main__":
    main()
