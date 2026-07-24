"""
aline_cuda.py

ctypes binding to the GPU Coder-generated CUDA library for aline_reconstruct
(per-column Hilbert envelope detection + power-law compression, the correct
beamforming method for OASBUD).

PLACEHOLDER: the exact call signature below is not yet confirmed against the
real generated header. Run MATLAB Codes/aline_run_codegen.m, then update the
_lib.aline_reconstruct(...) call and argtypes to match the actual
aline_reconstruct.h produced by GPU Coder before relying on this in production.
"""
import ctypes
import numpy as np
import os

_LIB_PATH = os.environ.get(
    "ALINE_CUDA_LIB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "MATLAB Codes", "codegen", "lib",
                 "aline_reconstruct", "aline_reconstruct.so")
)

_lib = None
if os.path.isfile(_LIB_PATH):
    _lib = ctypes.CDLL(_LIB_PATH)
    # TODO: set argtypes/restype once aline_reconstruct.h is available, e.g.:
    # _lib.aline_reconstruct.argtypes = [
    #     ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    #     ctypes.c_double, ctypes.POINTER(ctypes.c_double)
    # ]
    print(f"aline CUDA library loaded from {_LIB_PATH}")
else:
    print(f"aline CUDA library not found at {_LIB_PATH}, falling back to scipy")


def cuda_available():
    return _lib is not None


def aline_reconstruct_cuda(rf: np.ndarray, gamma: float = 0.3) -> np.ndarray:
    """Calls the GPU Coder-generated CUDA library. Raises if not loaded --
    callers should check cuda_available() first or catch this.

    PLACEHOLDER call signature -- update after inspecting the generated
    aline_reconstruct.h from aline_run_codegen.m (MATLAB Codes/).
    """
    if _lib is None:
        raise RuntimeError("aline CUDA library not loaded")
    rows, cols = rf.shape
    rf_c = np.ascontiguousarray(rf, dtype=np.float64)
    out = np.empty_like(rf_c)
    _lib.aline_reconstruct(
        rf_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_double(gamma),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    return out
