"""
enhancement_op.py (Plane Wave pipeline)

Unlike the OASBUD pipeline's EnhancementOp, this one is not preparing a
224x224 RGB tile for a classifier -- there is no classifier here, since
this dataset carries no diagnostic labels (see README Dataset Strategy).
This operator's job is turning the raw beamformed sum into a viewable
B-mode image: envelope detection, log compression, median filter for
speckle reduction, matching the same 3x3 median filter selection used
in Phase 3 for OASBUD.
"""
import numpy as np
import holoscan
from scipy.signal import hilbert
from scipy.ndimage import median_filter


class EnhancementOp(holoscan.core.Operator):

    DYNAMIC_RANGE_DB = 60

    def setup(self, spec):
        spec.input("bmode")
        spec.input("meta")
        spec.output("enhanced")
        spec.output("meta")

    def compute(self, op_input, op_output, context):
        bmode = op_input.receive("bmode")
        meta = op_input.receive("meta")
        enhanced = self._prepare(bmode)
        op_output.emit(enhanced, "enhanced")
        op_output.emit(meta, "meta")

    def _prepare(self, bmode):
        # Envelope detection (per lateral column, along depth)
        analytic = hilbert(bmode, axis=0)
        envelope = np.abs(analytic)

        # Log compression to a fixed dynamic range
        env_norm = envelope / (envelope.max() + 1e-12)
        db = 20 * np.log10(env_norm + 1e-6)
        db = np.clip(db, -self.DYNAMIC_RANGE_DB, 0)

        # Normalise to 0-255 for display/saving
        img_norm = (db + self.DYNAMIC_RANGE_DB) / self.DYNAMIC_RANGE_DB
        img_uint8 = (img_norm * 255).astype(np.uint8)

        # Same 3x3 median filter selected in Phase 3 for OASBUD
        img_median = median_filter(img_uint8, size=3)

        return img_median
