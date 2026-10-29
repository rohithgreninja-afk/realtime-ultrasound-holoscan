"""
enhancement_op.py

Turns the beamformed B-mode (power law output, already in [0, 1]) into the
224x224 RGB tile the CNN expects.

Preprocessing choice: no min-max renormalisation. The B-mode is scaled to
uint8 directly (bmode * 255, rounded), exactly as Phase4_OASBUD_Export.m and
Phase6_CNN_Evaluation.m do in MATLAB, so the pipeline, the MATLAB hold-out
evaluation and the training images share one intensity mapping.

Speckle filter: bilateral, DegreeOfSmoothing 0.1, SpatialSigma 2, selected in
Phase3_Enhancement.m (results/enhancement_eval.md). bilateral_filter() follows
MATLAB imbilatfilt: 9x9 neighbourhood, replicate padding, range kernel
exp(-d^2 / (2 * DegreeOfSmoothing)) on intensities in [0, 1]. Runs on the GPU
with cupy when available, numpy otherwise.
"""
import holoscan
import numpy as np
from PIL import Image

try:
    import cupy as cp
    _bilateral_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void bilateral(const float* a, float* out, int h, int w, int r,
                   float inv2ss, float inv2dos) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) return;
        float c = a[y * w + x], num = 0.0f, den = 0.0f;
        for (int dy = -r; dy <= r; dy++) {
            int yy = min(max(y + dy, 0), h - 1);
            for (int dx = -r; dx <= r; dx++) {
                int xx = min(max(x + dx, 0), w - 1);
                float q = a[yy * w + xx], d = q - c;
                float wgt = __expf(-(dy * dy + dx * dx) * inv2ss - d * d * inv2dos);
                num += wgt * q;
                den += wgt;
            }
        }
        out[y * w + x] = num / den;
    }''', 'bilateral')
except ImportError:
    cp = None


def bilateral_filter(img, degree_of_smoothing=0.1, spatial_sigma=2.0):
    """imbilatfilt equivalent for a 2-D float image in [0, 1]."""
    r = int(np.ceil(2 * spatial_sigma))
    inv2ss, inv2dos = 1.0 / (2 * spatial_sigma ** 2), 1.0 / (2 * degree_of_smoothing)
    if cp is not None:
        a = cp.ascontiguousarray(cp.asarray(img, dtype=cp.float32))
        out = cp.empty_like(a)
        h, w = a.shape
        _bilateral_kernel(((w + 15) // 16, (h + 15) // 16), (16, 16),
                          (a, out, np.int32(h), np.int32(w), np.int32(r),
                           np.float32(inv2ss), np.float32(inv2dos)))
        return out.get().astype(np.float64)
    a = np.asarray(img, dtype=np.float64)
    p = np.pad(a, r, mode='edge')
    num = np.zeros_like(a)
    den = np.zeros_like(a)
    h, w = a.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            q = p[r + dy:r + dy + h, r + dx:r + dx + w]
            wgt = np.exp(-(dy * dy + dx * dx) * inv2ss - (q - a) ** 2 * inv2dos)
            num += wgt * q
            den += wgt
    return num / den


def prepare_tile(bmode):
    """B-mode in [0, 1] -> filtered, resized 224x224x3 uint8 tile."""
    img_uint8   = np.clip(np.rint(np.asarray(bmode, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    img_filt    = bilateral_filter(img_uint8 / 255.0)
    img_filt    = np.clip(np.rint(img_filt * 255.0), 0, 255).astype(np.uint8)
    pil_resized = Image.fromarray(img_filt, mode='L').resize((224, 224), Image.BILINEAR)
    return np.array(pil_resized.convert('RGB'), dtype=np.uint8)


class EnhancementOp(holoscan.core.Operator):

    def setup(self, spec):
        spec.input("bmode")
        spec.input("label")
        spec.output("enhanced")
        spec.output("label")

    def compute(self, op_input, op_output, context):
        bmode = op_input.receive("bmode")
        label = op_input.receive("label")
        op_output.emit(prepare_tile(bmode), "enhanced")
        op_output.emit(label, "label")
