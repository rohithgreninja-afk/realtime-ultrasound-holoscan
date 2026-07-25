"""
data_source_op.py (Plane Wave pipeline)

Loads raw plane wave RF data from the CIRS040GSE / CIRS073_RUMC dataset
(Zenodo record 7986407, Schoop et al.) and emits one frame at a time.

This is genuine raw per-element channel data, unlike OASBUD, so it is the
correct input for delay-and-sum beamforming and the GPU Coder CUDA kernel.
It carries no malignancy labels -- see README Dataset Strategy section --
so this pipeline never emits a classification label, only reconstructed
images.

Expected folder layout (matches the dataset as published):
  <PLANEWAVE_DATA_PATH>/
      USHEADER_<timestamp>.mat       -- transducer + acquisition metadata
      **/USDATA_*.mat                -- one file per recorded frame (HDF5/v7.3 .mat)
"""
import os
import glob
import numpy as np
import holoscan
from scipy.io import loadmat
import h5py


class DataSourceOp(holoscan.core.Operator):

    def setup(self, spec):
        spec.output("rf_frame")
        spec.output("meta")

    def start(self):
        data_root = os.environ.get("PLANEWAVE_DATA_PATH")
        if not data_root:
            raise RuntimeError(
                "PLANEWAVE_DATA_PATH not set. Point it at the unzipped "
                "CIRS040GSE (or CIRS073_RUMC) folder, e.g.:\n"
                "  export PLANEWAVE_DATA_PATH=/path/to/CIRS040GSE"
            )

        header_files = glob.glob(os.path.join(data_root, "USHEADER_*.mat"))
        if not header_files:
            raise RuntimeError(f"No USHEADER_*.mat file found under {data_root}")
        header_file = header_files[0]

        usheader = loadmat(header_file, struct_as_record=False)["USHEADER"][0][0]
        self.xmit_angles = np.squeeze(usheader.xmitAngles).astype(np.float64)  # degrees
        self.n_ang = len(np.atleast_1d(self.xmit_angles))
        self.n_ele = usheader.xmitDelay.shape[1]
        self.fs = float(np.squeeze(usheader.fs))
        self.pitch = float(np.squeeze(usheader.pitch))
        self.c = float(np.squeeze(usheader.c))
        self.lens_delay = 96  # fixed acoustic lens correction, per dataset documentation

        print(f"Plane wave header loaded: {self.n_ang} angles, {self.n_ele} elements, "
              f"fs={self.fs/1e6:.2f} MHz, pitch={self.pitch*1000:.3f} mm, c={self.c:.0f} m/s")

        self.data_files = sorted(glob.glob(
            os.path.join(data_root, "**", "USDATA_*.mat"), recursive=True))
        if not self.data_files:
            raise RuntimeError(f"No USDATA_*.mat files found under {data_root}")
        print(f"Found {len(self.data_files)} frame(s)")

        self.frame_idx = 0
        self.max_frames = int(os.environ.get("PLANEWAVE_MAX_FRAMES", len(self.data_files)))

        # Precompute the angle-dependent delay correction (same for every frame,
        # since it only depends on array geometry and steering angle, not the data)
        angles_rad = np.deg2rad(self.xmit_angles)
        delay_sec = np.abs((self.n_ele - 1) / 2 * self.pitch * np.sin(angles_rad) / self.c)
        self.delay_samples = np.floor(delay_sec * self.fs).astype(int)

    def _load_frame(self, path):
        with h5py.File(path, "r") as f:
            usdata = np.swapaxes(np.squeeze(np.array(f.get("USDATA"))), 0, 2).astype(np.float64)
        # usdata shape: [time_samples, elements, angles]

        # Lens delay correction
        usdata = usdata[self.lens_delay:, :, :]

        # Angle-dependent delay correction
        n_t = usdata.shape[0]
        for i_ang in range(usdata.shape[2]):
            d = self.delay_samples[i_ang] if self.n_ang > 1 else self.delay_samples
            if d > 0:
                usdata[: n_t - d, :, i_ang] = usdata[d:, :, i_ang]
                usdata[n_t - d :, :, i_ang] = 0

        # Known hardware quirk: channel 125 is split/shared, needs doubling
        if usdata.shape[1] > 125:
            usdata[:, 125, :] = 2 * usdata[:, 125, :]

        return usdata

    def compute(self, op_input, op_output, context):
        if self.frame_idx >= self.max_frames or self.frame_idx >= len(self.data_files):
            return

        path = self.data_files[self.frame_idx]
        rf = self._load_frame(path)

        meta = {
            "frame_idx": self.frame_idx,
            "file": os.path.basename(path),
            "fs": self.fs,
            "pitch": self.pitch,
            "c": self.c,
            "n_ele": self.n_ele,
            "xmit_angles": self.xmit_angles,
        }

        op_output.emit(rf, "rf_frame")
        op_output.emit(meta, "meta")
        self.frame_idx += 1
