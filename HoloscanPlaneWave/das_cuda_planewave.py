"""
das_cuda_planewave.py

ctypes binding to the GPU Coder-generated CUDA library for
das_beamform_planewave (correct plane-wave delay law: planar transmit
wavefront + per-element spherical receive).

Matches the real generated signature from das_beamform_planewave.h:

    void das_beamform_planewave(
        const double cpu_rf_data[], const int rf_size[2],
        const double cpu_x_elements_data[], const int x_elements_size[2],
        const double cpu_z_image[400], const double cpu_x_image[300],
        double fs, double c, double tx_angle, double cpu_bmode[120000]);

z_image/x_image were declared fixed-size in codegen (400 and 300 points),
so the output image is always exactly 400x300 -- unlike aline_reconstruct,
there is no output-size array to read back. Callers MUST pass z_image and
x_image with exactly 400 and 300 points respectively, matching what
beamforming_op.py already builds.

MATLAB/GPU Coder use column-major (Fortran) array order internally, unlike
numpy's default row-major (C) order, so the RF array is explicitly
converted before the call. x_elements, z_image, x_image are all 1D so
column-major vs row-major makes no difference for them.

Falls back to a numpy reference implementation (numerically identical
physics, just not GPU-accelerated) if the compiled library isn't present.
"""
import ctypes
import os
import numpy as np

_LIB_PATH = os.environ.get(
    "DAS_PLANEWAVE_CUDA_LIB_PATH",
    os.path.join(os.path.dirname(__file__), "das_beamform_planewave.so")
)

IMG_DEPTH_PX = 400
IMG_LATERAL_PX = 300

_lib = None
if os.path.isfile(_LIB_PATH):
    _lib = ctypes.CDLL(_LIB_PATH)
    _lib.das_beamform_planewave_c.argtypes = [
        ctypes.POINTER(ctypes.c_double),   # cpu_rf_data (flat, column-major)
        ctypes.POINTER(ctypes.c_int),      # rf_size[2] = {rows, cols}
        ctypes.POINTER(ctypes.c_double),   # cpu_x_elements_data
        ctypes.POINTER(ctypes.c_int),      # x_elements_size[2] = {1, n_ele}
        ctypes.POINTER(ctypes.c_double),   # cpu_z_image[400], fixed
        ctypes.POINTER(ctypes.c_double),   # cpu_x_image[300], fixed
        ctypes.c_double,                   # fs
        ctypes.c_double,                   # c
        ctypes.c_double,                   # tx_angle
        ctypes.POINTER(ctypes.c_double),   # cpu_bmode[120000], output, preallocated
    ]
    _lib.das_beamform_planewave_c.restype = None
    print(f"das_beamform_planewave CUDA library loaded from {_LIB_PATH}")
else:
    print(f"das_beamform_planewave CUDA library not found at {_LIB_PATH}, "
          f"falling back to numpy reference implementation")


def cuda_available():
    return _lib is not None


def das_beamform_planewave_numpy(rf, x_elements, z_image, x_image, fs, c, tx_angle):
    """Reference implementation, matches das_beamform_planewave.m exactly.
    Vectorized over elements per depth row for reasonable speed."""
    num_samples = rf.shape[0]
    nz, nx = len(z_image), len(x_image)
    bmode = np.zeros((nz, nx))

    sin_a, cos_a = np.sin(tx_angle), np.cos(tx_angle)

    for iz in range(nz):
        z = z_image[iz]
        t_tx = (x_image * sin_a + z * cos_a) / c          # shape (nx,)
        dx = x_image[:, None] - x_elements[None, :]        # shape (nx, n_ele)
        dist_rx = np.sqrt(dx**2 + z**2)
        t_rx = dist_rx / c                                  # shape (nx, n_ele)
        t_total = t_tx[:, None] + t_rx                      # shape (nx, n_ele)

        sample_idx = np.round(t_total * fs).astype(np.int64)
        valid = (sample_idx >= 0) & (sample_idx < num_samples)
        sample_idx_c = np.clip(sample_idx, 0, num_samples - 1)

        n_ele = rf.shape[1]
        gathered = rf[sample_idx_c, np.arange(n_ele)[None, :]]
        gathered = np.where(valid, gathered, 0.0)
        bmode[iz, :] = gathered.sum(axis=1)

    return bmode


def das_beamform_planewave_cuda(rf, x_elements, z_image, x_image, fs, c, tx_angle):
    """Calls the GPU Coder-generated CUDA library."""
    if _lib is None:
        raise RuntimeError("das_beamform_planewave CUDA library not loaded")

    if len(z_image) != IMG_DEPTH_PX or len(x_image) != IMG_LATERAL_PX:
        raise ValueError(
            f"z_image/x_image must have exactly {IMG_DEPTH_PX}/{IMG_LATERAL_PX} "
            f"points (fixed at codegen time), got {len(z_image)}/{len(x_image)}"
        )

    rows, cols = rf.shape
    rf_fortran = np.asfortranarray(rf, dtype=np.float64)
    rf_size = (ctypes.c_int * 2)(rows, cols)

    n_ele = len(x_elements)
    x_elements_c = np.ascontiguousarray(x_elements, dtype=np.float64)
    x_elements_size = (ctypes.c_int * 2)(1, n_ele)

    z_image_c = np.ascontiguousarray(z_image, dtype=np.float64)
    x_image_c = np.ascontiguousarray(x_image, dtype=np.float64)

    bmode_out = np.empty(IMG_DEPTH_PX * IMG_LATERAL_PX, dtype=np.float64)

    _lib.das_beamform_planewave_c(
        rf_fortran.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rf_size,
        x_elements_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x_elements_size,
        z_image_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x_image_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(fs),
        ctypes.c_double(c),
        ctypes.c_double(tx_angle),
        bmode_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    # Output is column-major (MATLAB convention): reshape accordingly,
    # then convert to normal row-major for the rest of the Python pipeline
    bmode = bmode_out.reshape((IMG_DEPTH_PX, IMG_LATERAL_PX), order='F')
    return np.ascontiguousarray(bmode)


def das_beamform_planewave(rf, x_elements, z_image, x_image, fs, c, tx_angle):
    """Public entry point: CUDA if available, numpy reference otherwise."""
    if cuda_available():
        return das_beamform_planewave_cuda(rf, x_elements, z_image, x_image, fs, c, tx_angle)
    return das_beamform_planewave_numpy(rf, x_elements, z_image, x_image, fs, c, tx_angle)
