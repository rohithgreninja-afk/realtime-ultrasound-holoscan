"""
aline_cuda.py

ctypes binding to the GPU Coder-generated CUDA library for aline_reconstruct
(per-column Hilbert envelope detection + power-law compression, the correct
beamforming method for OASBUD).

Calls aline_reconstruct_c, a plain-C wrapper (see aline_reconstruct_c_wrapper.cpp)
around the real generated aline_reconstruct function. GPU Coder generates
aline_reconstruct as C++, which gets its symbol name mangled by the compiler,
so ctypes cannot call it directly by name -- the wrapper re-exports it under
extern "C" linkage instead. The real signature, matching aline_reconstruct.h:

    void aline_reconstruct(const double cpu_rf_data[], const int rf_size[2],
                            double b_gamma, double cpu_bmode_data[],
                            int bmode_size[2]);

MATLAB/GPU Coder use column-major (Fortran) array order internally, unlike
numpy's default row-major (C) order, so arrays are explicitly converted
before and after the call.

The library at ALINE_CUDA_LIB_PATH (or the default path below) must be a
Linux shared object (.so) built from the generated .cu/.cpp source files
plus aline_reconstruct_c_wrapper.cpp, linked against cufft (-lcufft). The
.lib produced directly by GPU Coder on Windows cannot be loaded here.
"""
import ctypes
import numpy as np
import os

_LIB_PATH = os.environ.get(
    "ALINE_CUDA_LIB_PATH",
    os.path.join(os.path.dirname(__file__), "aline_reconstruct.so")
)

_lib = None
if os.path.isfile(_LIB_PATH):
    _lib = ctypes.CDLL(_LIB_PATH)
    _lib.aline_reconstruct_c.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # cpu_rf_data (flat, column-major)
        ctypes.POINTER(ctypes.c_int),     # rf_size[2] = {rows, cols}
        ctypes.c_double,                  # b_gamma
        ctypes.POINTER(ctypes.c_double),  # cpu_bmode_data (output, preallocated by caller)
        ctypes.POINTER(ctypes.c_int),     # bmode_size[2] (output, filled by the function)
    ]
    _lib.aline_reconstruct_c.restype = None
    print(f"aline CUDA library loaded from {_LIB_PATH}")
else:
    print(f"aline CUDA library not found at {_LIB_PATH}, falling back to scipy")


def cuda_available():
    return _lib is not None


def aline_reconstruct_cuda(rf: np.ndarray, gamma: float = 0.3) -> np.ndarray:
    """Calls the GPU Coder-generated CUDA library. Raises if not loaded --
    callers should check cuda_available() first or catch this."""
    if _lib is None:
        raise RuntimeError("aline CUDA library not loaded")

    rows, cols = rf.shape

    # MATLAB/GPU Coder expect column-major order
    rf_fortran = np.asfortranarray(rf, dtype=np.float64)
    rf_size    = (ctypes.c_int * 2)(rows, cols)
    bmode_size = (ctypes.c_int * 2)(0, 0)
    bmode_out  = np.empty((rows, cols), dtype=np.float64, order='F')

    _lib.aline_reconstruct_c(
        rf_fortran.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rf_size,
        ctypes.c_double(gamma),
        bmode_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        bmode_size
    )

    out_rows, out_cols = bmode_size[0], bmode_size[1]
    # Convert back to normal row-major (C order) for the rest of the Python pipeline
    result = np.ascontiguousarray(bmode_out[:out_rows, :out_cols])
    return result
