import holoscan
import numpy as np
from scipy.signal import hilbert

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
        print("BeamformingOp ready -- A-line mode")

    def compute(self, op_input, op_output, context):
        rf    = op_input.receive("rf_frame")
        label = op_input.receive("label")
        bmode = self._aline_reconstruct(rf)
        op_output.emit(bmode.astype(np.float32), "bmode")
        op_output.emit(label, "label")

    def _aline_reconstruct(self, rf):
        analytic = hilbert(rf, axis=0)
        envelope = np.abs(analytic)
        env_norm = envelope / (envelope.max() + 1e-12)
        bmode    = np.power(env_norm, self.GAMMA)
        return bmode.astype(np.float32)
