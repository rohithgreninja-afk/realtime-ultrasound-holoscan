"""
das_cuda_planewave.py

ctypes binding to the GPU Coder-generated CUDA library for
das_beamform_planewave (correct plane-wave delay law: planar transmit
wavefront + per-element spherical receive).

The exact call signature below is a PLACEHOLDER pending the real generated
header from das_run_codegen_planewave.m -- update _lib.das_beamform_planewave_c
argtypes and the call itself once that header is available, same as the
aline_reconstruct binding needed correcting after its first real build.

Falls back to a numpy reference implementation (numerically identical
physics, just not GPU-accelerated) if the compiled library isn't present,
so this pipeline is runnable end-to-end before the CUDA build exists.
"""
import ctypes
import os
import numpy as np

_LIB_PATH = os.environ.get(
    "DAS_PLANEWAVE_CUDA_LIB_PATH",
    os.path.join(os.path.dirname(__file__), "das_beamform_planewave.so")
)

_lib = None
if os.path.isfile(_LIB_PATH):
    _lib = ctypes.CDLL(_LIB_PATH)
    # PLACEHOLDER -- confirm against the real das_beamform_planewave.h
    # once das_run_codegen_planewave.m has actually been run.
    # _lib.das_beamform_planewave_c.argtypes = [...]
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
    """Calls the GPU Coder-generated CUDA library. Raises if not loaded."""
    if _lib is None:
        raise RuntimeError("das_beamform_planewave CUDA library not loaded")
    raise NotImplementedError(
        "CUDA call not yet wired up -- run das_run_codegen_planewave.m, "
        "inspect the generated .h, then fill in this function the same "
        "way aline_cuda.py's binding was corrected after its real build."
    )


def das_beamform_planewave(rf, x_elements, z_image, x_image, fs, c, tx_angle):
    """Public entry point: CUDA if available, numpy reference otherwise."""
    if cuda_available():
        try:
            return das_beamform_planewave_cuda(rf, x_elements, z_image, x_image, fs, c, tx_angle)
        except NotImplementedError:
            pass  # binding not finished yet, fall through to reference
    return das_beamform_planewave_numpy(rf, x_elements, z_image, x_image, fs, c, tx_angle)
