import holoscan
import numpy as np
from PIL import Image

class EnhancementOp(holoscan.core.Operator):

    def setup(self, spec):
        spec.input("bmode")
        spec.input("label")
        spec.output("enhanced")
        spec.output("label")

    def compute(self, op_input, op_output, context):
        bmode = op_input.receive("bmode")
        label = op_input.receive("label")
        enhanced = self._prepare(bmode)
        op_output.emit(enhanced, "enhanced")
        op_output.emit(label, "label")

    def _prepare(self, bmode):
        img_norm   = (bmode - bmode.min()) / (bmode.max() - bmode.min() + 1e-8)
        img_uint8  = (img_norm * 255).astype(np.uint8)
        pil_img    = Image.fromarray(img_uint8, mode='L')
        pil_resized = pil_img.resize((224, 224), Image.BILINEAR)
        pil_rgb    = pil_resized.convert('RGB')
        return np.array(pil_rgb, dtype=np.uint8)
