// aline_reconstruct_c_wrapper.cpp
//
// GPU Coder generates aline_reconstruct as a C++ function, which means its
// exported symbol name gets mangled by the compiler and ctypes cannot find
// it by its plain name. This wrapper re-exposes it under extern "C" linkage
// so Python's ctypes can call it directly.
//
// Build: place this file alongside the generated .cu/.h/.cpp files and
// include it in the nvcc compile command that produces aline_reconstruct.so

#include "aline_reconstruct.h"

extern "C" {
void aline_reconstruct_c(const double cpu_rf_data[], const int rf_size[2],
                          double b_gamma, double cpu_bmode_data[],
                          int bmode_size[2]) {
    aline_reconstruct(cpu_rf_data, rf_size, b_gamma, cpu_bmode_data, bmode_size);
}
}
