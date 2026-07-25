// das_beamform_planewave_c_wrapper.cpp
//
// GPU Coder generates das_beamform_planewave as a C++ function, which
// means its exported symbol name gets mangled by the compiler and ctypes
// cannot find it by its plain name. This wrapper re-exposes it under
// extern "C" linkage so Python's ctypes can call it directly.
//
// Build: place this file alongside the generated .cu/.h/.cpp files and
// include it in the nvcc compile command that produces
// das_beamform_planewave.so

#include "das_beamform_planewave.h"

extern "C" {
void das_beamform_planewave_c(
    const double cpu_rf_data[], const int rf_size[2],
    const double cpu_x_elements_data[], const int x_elements_size[2],
    const double cpu_z_image[400], const double cpu_x_image[300], double fs,
    double c, double tx_angle, double cpu_bmode[120000]) {
    das_beamform_planewave(cpu_rf_data, rf_size, cpu_x_elements_data,
                            x_elements_size, cpu_z_image, cpu_x_image, fs, c,
                            tx_angle, cpu_bmode);
}
}
