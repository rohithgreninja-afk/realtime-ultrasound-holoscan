"""
output_op.py (Plane Wave pipeline)

Saves each reconstructed frame as a PNG and prints per-frame timing.
No classification output -- this dataset has no diagnostic labels, so
this pipeline never claims one (see README Dataset Strategy section).
"""
import os
import time
import numpy as np
from PIL import Image
import holoscan


class OutputOp(holoscan.core.Operator):

    def setup(self, spec):
        spec.input("enhanced")
        spec.input("meta")

    def start(self):
        self.output_dir = os.environ.get(
            "PLANEWAVE_RESULTS_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "planewave_results")
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_times = []
        self.last_time = time.perf_counter()
        print(f"OutputOp ready -- saving frames to {self.output_dir}")

    def compute(self, op_input, op_output, context):
        enhanced = op_input.receive("enhanced")
        meta = op_input.receive("meta")

        now = time.perf_counter()
        frame_ms = (now - self.last_time) * 1000
        self.last_time = now
        self.frame_times.append(frame_ms)

        idx = meta["frame_idx"]
        fname = os.path.join(self.output_dir, f"frame_{idx:04d}.png")
        Image.fromarray(enhanced, mode="L").save(fname)

        print(f"Frame {idx}: {meta['file']}  {frame_ms:.1f} ms  -> {os.path.basename(fname)}")

    def stop(self):
        if not self.frame_times:
            return
        arr = np.array(self.frame_times[1:])  # drop first (includes warmup/JIT)
        if len(arr) == 0:
            return
        print("\n=== Plane Wave Pipeline Summary ===")
        print(f"Frames processed: {len(self.frame_times)}")
        print(f"Mean frame time:  {arr.mean():.1f} ms")
        print(f"Throughput:       {1000/arr.mean():.1f} fps (mean)")
        print(f"Results saved to: {self.output_dir}")
