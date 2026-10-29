"""
enhancement_op.py (Plane Wave pipeline)

Unlike the OASBUD pipeline's EnhancementOp, this one is not preparing a
224x224 RGB tile for a classifier -- there is no classifier here, since
this dataset carries no diagnostic labels (see README Dataset Strategy).
This operator's job is turning the raw beamformed sum into a viewable
B-mode image: envelope detection, log compression, and the bilateral
speckle filter selected in Phase3_Enhancement.m (DegreeOfSmoothing 0.1,
SpatialSigma 2, same implementation as Holoscan/enhancement_op.py).
"""
import numpy as np
import holoscan
from scipy.signal import hilbert

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


DYNAMIC_RANGE_DB = 60


def prepare_bmode(bmode):
    """Beamformed sum -> envelope, 60 dB log compression, uint8, bilateral filter."""
    envelope = np.abs(hilbert(bmode, axis=0))
    env_norm = envelope / (envelope.max() + 1e-12)
    db = np.clip(20 * np.log10(env_norm + 1e-6), -DYNAMIC_RANGE_DB, 0)
    img_uint8 = ((db + DYNAMIC_RANGE_DB) / DYNAMIC_RANGE_DB * 255).astype(np.uint8)
    img_filt = bilateral_filter(img_uint8 / 255.0)
    return np.clip(np.rint(img_filt * 255.0), 0, 255).astype(np.uint8)


class EnhancementOp(holoscan.core.Operator):

    def setup(self, spec):
        spec.input("bmode")
        spec.input("meta")
        spec.output("enhanced")
        spec.output("meta")

    def compute(self, op_input, op_output, context):
        bmode = op_input.receive("bmode")
        meta = op_input.receive("meta")
        enhanced = prepare_bmode(bmode)
        op_output.emit(enhanced, "enhanced")
        op_output.emit(meta, "meta")

