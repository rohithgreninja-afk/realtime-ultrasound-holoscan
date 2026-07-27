import holoscan
import numpy as np
from scipy.signal import hilbert
from aline_cuda import aline_reconstruct_cuda, cuda_available

class BeamformingOp(holoscan.core.Operator):

    FS          = 40e6
    C           = 1540.0
    PROBE_WIDTH = 38e-3
    GAMMA       = 0.3

    def setup(self, spec):
        spec.input("rf_frame")
        spec.input("label")
        spec.output("bmode")
        spec.output("label")

    def start(self):
        mode = "CUDA (GPU Coder)" if cuda_available() else "scipy (fallback)"
        print(f"BeamformingOp ready -- A-line mode, {mode}")

    def compute(self, op_input, op_output, context):
        rf    = op_input.receive("rf_frame")
        label = op_input.receive("label")
        bmode = self._aline_reconstruct(rf)
        op_output.emit(bmode.astype(np.float32), "bmode")
        op_output.emit(label, "label")

    def _aline_reconstruct(self, rf):
        if cuda_available():
            return aline_reconstruct_cuda(rf, self.GAMMA).astype(np.float32)
        analytic = hilbert(rf, axis=0)
        envelope = np.abs(analytic)
        env_norm = envelope / (envelope.max() + 1e-12)
        bmode    = np.power(env_norm, self.GAMMA)
        return bmode.astype(np.float32)
