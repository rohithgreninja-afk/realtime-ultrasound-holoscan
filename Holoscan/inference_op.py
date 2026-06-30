# =========================================================
# inference_op.py  --  v1.7
# InferenceOp -- ONNX inference with MobileNetV2 mega model
# Input:  raw float32 [1 x 3 x 224 x 224], values 0-255
# No manual softmax -- already applied in ONNX graph
# No manual normalisation -- baked into model by exportONNXNetwork
#
# Normal class handling (v1.7):
#   Both benign and normal mean not cancer.  Keeping normal as a
#   separate exit route reduces malignant recall without clinical
#   benefit.  Fix: p_clinical_benign = p_benign + p_normal,
#   then decide malignant vs benign only.
# =========================================================

import os
import holoscan
import numpy as np
import onnxruntime as ort


class InferenceOp(holoscan.core.Operator):

    CLASSES   = ['benign', 'malignant', 'normal']
    _THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_THIS_DIR)
    ONNX_PATH  = os.environ.get(
        'ONNX_MODEL_PATH',
        os.path.join(_REPO_ROOT, 'MATLAB Codes', 'trainedMobileNetV2_mega.onnx')
    )

    def setup(self, spec):
        spec.input('enhanced')
        spec.input('label')
        spec.output('scores')
        spec.output('prediction')
        spec.output('label')

    def start(self):
        print(f'Loading ONNX model from {self.ONNX_PATH}')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session    = ort.InferenceSession(self.ONNX_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f'Model loaded. Input: {self.input_name} | EP: {self.session.get_providers()[0]}')
        print('Normal handling: p_normal absorbed into p_benign before decision.')

    def compute(self, op_input, op_output, context):
        image = op_input.receive('enhanced')
        label = op_input.receive('label')
        scores, prediction = self._infer(image)
        op_output.emit(scores,     'scores')
        op_output.emit(prediction, 'prediction')
        op_output.emit(label,      'label')

    def _infer(self, image):
        img_float = image.astype(np.float32)
        img_chw   = np.transpose(img_float, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)

        outputs = self.session.run(None, {self.input_name: img_batch})
        probs   = outputs[0][0].copy()

        # Merge normal into benign -- both mean not cancer
        p_benign    = float(probs[0]) + float(probs[2])
        p_malignant = float(probs[1])

        prediction = 'malignant' if p_malignant > p_benign else 'benign'

        return probs.astype(np.float32), prediction
