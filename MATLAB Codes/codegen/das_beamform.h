//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: das_beamform.h
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 20-May-2026 10:56:28
//

#ifndef DAS_BEAMFORM_H
#define DAS_BEAMFORM_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
extern void das_beamform(const double cpu_rf[930240],
                         const double cpu_x_elements[510],
                         const double cpu_z_image[300],
                         const double cpu_x_image[200], double fs, double c,
                         double cpu_bmode[60000]);

#endif
//
// File trailer for das_beamform.h
//
// [EOF]
//
