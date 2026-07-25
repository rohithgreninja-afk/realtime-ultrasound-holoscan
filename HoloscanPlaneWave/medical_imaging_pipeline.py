"""
medical_imaging_pipeline.py (Plane Wave pipeline)

Real-time GPU-accelerated ultrasound beamforming demonstration on genuine
raw per-element plane wave data (CIRS040GSE / CIRS073_RUMC, Zenodo
7986407). This is a completely separate application from the OASBUD
pipeline in Holoscan/ -- no operators are shared beyond the enhancement
approach (median filter selection), and there is deliberately no
classification stage, since this dataset carries no malignant/benign
labels. See README Dataset Strategy section for why these two pipelines
exist separately.

Usage:
  export PLANEWAVE_DATA_PATH=/path/to/CIRS040GSE
  python3 medical_imaging_pipeline.py [max_frames]

Optional:
  export PLANEWAVE_N_ANGLES=1        # angles compounded per frame (default 1, real-time)
  export PLANEWAVE_RESULTS_DIR=...   # where PNGs get saved
  export DAS_PLANEWAVE_CUDA_LIB_PATH=... # compiled CUDA library, falls back to numpy if absent
"""
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import holoscan
from holoscan.core import Application
from holoscan.conditions import CountCondition

from data_source_op import DataSourceOp
from beamforming_op import BeamformingOp
from enhancement_op import EnhancementOp
from output_op import OutputOp


class PlaneWavePipeline(Application):

    def compose(self):
        # Determine how many times DataSourceOp should run. Holoscan
        # operators loop compute() indefinitely by default; without an
        # explicit CountCondition the scheduler never learns the source
        # is finite, and the app hangs after the last real frame instead
        # of stopping.
        max_frames_env = os.environ.get("PLANEWAVE_MAX_FRAMES")
        if max_frames_env:
            frame_count = int(max_frames_env)
        else:
            data_root = os.environ.get("PLANEWAVE_DATA_PATH", "")
            condition = os.environ.get("PLANEWAVE_CONDITION", "low_attenuation")
            condition_dir = os.path.join(data_root, condition)
            n_available = len(glob.glob(os.path.join(condition_dir, "USDATA_*.mat")))
            frame_count = n_available if n_available > 0 else 20  # sane fallback

        print(f"DataSourceOp will run for {frame_count} frame(s)")

        data_source = DataSourceOp(self, CountCondition(self, count=frame_count),
                                    name="data_source")
        beamforming = BeamformingOp(self, name="beamforming")
        enhancement = EnhancementOp(self, name="enhancement")
        output = OutputOp(self, name="output")

        self.add_flow(data_source, beamforming, {("rf_frame", "rf_frame"), ("meta", "meta")})
        self.add_flow(beamforming, enhancement, {("bmode", "bmode"), ("meta", "meta")})
        self.add_flow(enhancement, output, {("enhanced", "enhanced"), ("meta", "meta")})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.environ["PLANEWAVE_MAX_FRAMES"] = sys.argv[1]

    print("=== Plane Wave Beamforming Pipeline ===")
    print("Real-time acceleration demonstration -- CIRS040GSE / CIRS073_RUMC")
    print("No classification stage: this dataset has no diagnostic labels.\n")

    app = PlaneWavePipeline()
    app.run()
