"""
beamforming_op.py (Plane Wave pipeline)

Delay-and-sum beamforming for genuine raw per-element plane wave data
(CIRS040GSE / CIRS073_RUMC, Zenodo 7986407). Uses the corrected plane-wave
delay law (das_beamform_planewave.m): planar transmit wavefront + per-
element spherical receive, NOT the classical per-element pulse-echo model
used elsewhere in this project for the OASBUD comparison methods.

For real-time throughput, beamforms a single steering angle (closest to
broadside/0 degrees) by default. Set PLANEWAVE_N_ANGLES to compound more
angles for a quality comparison at the cost of latency.
"""
import os
import numpy as np
import holoscan
from das_cuda_planewave import das_beamform_planewave, cuda_available


class BeamformingOp(holoscan.core.Operator):

    def setup(self, spec):
        spec.input("rf_frame")
        spec.input("meta")
        spec.output("bmode")
        spec.output("meta")

    def start(self):
        self.n_angles = int(os.environ.get("PLANEWAVE_N_ANGLES", 1))
        mode = "CUDA (GPU Coder)" if cuda_available() else "numpy (fallback)"
        print(f"BeamformingOp ready -- plane wave DAS, {mode}, "
              f"{self.n_angles} angle(s) per frame")

    def compute(self, op_input, op_output, context):
        rf = op_input.receive("rf_frame")      # [samples, elements, angles]
        meta = op_input.receive("meta")

        fs = meta["fs"]
        pitch = meta["pitch"]
        c = meta["c"]
        n_ele = meta["n_ele"]
        xmit_angles_deg = np.atleast_1d(meta["xmit_angles"])

        x_elements = (np.arange(n_ele) - (n_ele - 1) / 2) * pitch

        num_samples = rf.shape[0]
        z_max = (num_samples / fs) * c / 2
        z_image = np.linspace(1e-3, z_max, 400)
        x_image = np.linspace(x_elements[0], x_elements[-1], 300)

        # Pick angle indices: broadside first, then spread outward if
        # compounding more than one
        broadside_idx = int(np.argmin(np.abs(xmit_angles_deg)))
        if self.n_angles == 1:
            angle_indices = [broadside_idx]
        else:
            angle_indices = np.linspace(0, len(xmit_angles_deg) - 1,
                                         self.n_angles).astype(int).tolist()

        bmode_sum = np.zeros((len(z_image), len(x_image)))
        for k, ai in enumerate(angle_indices):
            tx_angle = np.deg2rad(xmit_angles_deg[ai])
            rf_angle = rf[:, :, ai] if rf.ndim == 3 else rf
            img = das_beamform_planewave(rf_angle, x_elements, z_image, x_image,
                                          fs, c, tx_angle)
            bmode_sum = (k * bmode_sum + img) / (k + 1)  # running average, matches
                                                          # the dataset's own reference
                                                          # compounding approach

        op_output.emit(bmode_sum.astype(np.float32), "bmode")
        op_output.emit(meta, "meta")
