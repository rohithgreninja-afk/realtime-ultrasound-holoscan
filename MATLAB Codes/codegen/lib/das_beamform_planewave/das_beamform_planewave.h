//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: das_beamform_planewave.h
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 27-Oct-2026 19:17:32
//

#ifndef DAS_BEAMFORM_PLANEWAVE_H
#define DAS_BEAMFORM_PLANEWAVE_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
extern void das_beamform_planewave(
    const double cpu_rf_data[], const int rf_size[2],
    const double cpu_x_elements_data[], const int x_elements_size[2],
    const double cpu_z_image[400], const double cpu_x_image[300], double fs,
    double c, double tx_angle, double cpu_bmode[120000]);

#endif
//
// File trailer for das_beamform_planewave.h
//
// [EOF]
//
