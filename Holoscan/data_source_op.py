# =========================================================
# data_source_op.py
# DataSourceOp -- loads RF frames from OASBUD .mat file
# and emits them one at a time into the Holoscan pipeline.
#
# OASBUD class field: 0 = benign, 1 = malignant (48 / 52 records,
# consistent with the BI-RADS field).
#
# By default only the OASBUD hold-out patients are emitted: patients
# whose images were exported for CNN training (listed in
# MATLAB Codes/cnn_split.mat, oasbudTrainIds) are skipped. Set
# OASBUD_INCLUDE_TRAINING=1 to emit every record in the file.
# =========================================================

import os
import holoscan
import numpy as np
import scipy.io

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
SPLIT_PATH = os.path.join(_REPO_ROOT, "MATLAB Codes", "cnn_split.mat")


def load_oasbud_records(mat_path=None, include_training=None):
    """Return (records, n_in_file). records is a list of dicts with
    'id', 'rf1' and 'class', restricted to hold-out patients unless
    include_training is true."""
    if mat_path is None:
        mat_path = os.environ.get(
            "OASBUD_PATH", os.path.join(_REPO_ROOT, "data", "sample", "OASBUD_sample.mat"))
    if include_training is None:
        include_training = os.environ.get("OASBUD_INCLUDE_TRAINING", "0") == "1"
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"OASBUD data not found at {mat_path}. Either ensure the bundled "
            f"data/sample/OASBUD_sample.mat exists, or set the OASBUD_PATH "
            f"environment variable to point at the full OASBUD.mat file."
        )
    data = scipy.io.loadmat(mat_path, simplify_cells=True)["data"]
    if isinstance(data, dict):
        data = [data]
    if include_training:
        return list(data), len(data)
    split = scipy.io.loadmat(SPLIT_PATH, simplify_cells=True)
    train_ids = set(np.atleast_1d(split["oasbudTrainIds"]).tolist())
    return [r for r in data if str(r["id"]) not in train_ids], len(data)


class DataSourceOp(holoscan.core.Operator):
    """
    Loads the OASBUD dataset and emits one RF frame per pipeline cycle.
    Emits both the RF data and the ground truth label for evaluation.
    """

    def setup(self, spec):
        # Output ports -- what this operator sends downstream
        spec.output("rf_frame")   # raw RF matrix [depth x 510] float64
        spec.output("label")      # ground truth: 0=benign, 1=malignant

    def start(self):
        # Runs once before first frame -- load the dataset here
        self.data, n_file = load_oasbud_records()
        self.index   = 0
        self.n_total = len(self.data)
        include = os.environ.get("OASBUD_INCLUDE_TRAINING", "0") == "1"
        print(f"Dataset loaded: {self.n_total} of {n_file} patient records "
              f"({'all records, includes training patients' if include else 'hold-out patients only'})")

    def compute(self, op_input, op_output, context):
        if self.index >= self.n_total:
            return  # all frames processed

        # Get current patient record
        record = self.data[self.index]
        rf     = np.array(record['rf1'], dtype=np.float64)  # [depth x 510]
        label  = int(record['class'])                        # 0 or 1

        # Emit to pipeline
        op_output.emit(rf,    "rf_frame")
        op_output.emit(label, "label")

        print(f"Frame {self.index+1}/{self.n_total} emitted | "
              f"Label: {'malignant' if label == 1 else 'benign'} | "
              f"RF shape: {rf.shape}")

        self.index += 1
