# =========================================================
# data_source_op.py
# DataSourceOp — loads RF frames from OASBUD .mat file
# and emits them one at a time into the Holoscan pipeline
# =========================================================

import holoscan
import numpy as np
import scipy.io

class DataSourceOp(holoscan.core.Operator):
    """
    Loads the OASBUD dataset and emits one RF frame per pipeline cycle.
    Emits both the RF data and the ground truth label for evaluation.
    """

    def setup(self, spec):
        # Output ports — what this operator sends downstream
        spec.output("rf_frame")   # raw RF matrix [1824 x 510] float64
        spec.output("label")      # ground truth: 0=malignant, 1=benign

    def start(self):
        # Runs once before first frame — load the dataset here
        mat_path = "/mnt/c/Users/rohit/Downloads/Real Time Image Processing Project/OASBUD.mat"
        print(f"Loading OASBUD dataset from {mat_path}...")
        mat = scipy.io.loadmat(mat_path, simplify_cells=True)
        self.data    = mat['data']          # list of 100 patient dicts
        self.index   = 0                    # current frame index
        self.n_total = len(self.data)
        print(f"Dataset loaded: {self.n_total} patient records")

    def compute(self, op_input, op_output, context):
        if self.index >= self.n_total:
            return  # all frames processed

        # Get current patient record
        record = self.data[self.index]
        rf     = np.array(record['rf1'], dtype=np.float64)  # [1824 x 510]
        label  = int(record['class'])                        # 0 or 1

        # Emit to pipeline
        op_output.emit(rf,    "rf_frame")
        op_output.emit(label, "label")

        print(f"Frame {self.index+1}/{self.n_total} emitted | "
              f"Label: {'benign' if label==1 else 'malignant'} | "
              f"RF shape: {rf.shape}")

        self.index += 1