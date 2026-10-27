//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: aline_reconstruct.h
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 27-Oct-2026 19:17:05
//

#ifndef ALINE_RECONSTRUCT_H
#define ALINE_RECONSTRUCT_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
extern void aline_reconstruct(const double cpu_rf_data[], const int rf_size[2],
                              double b_gamma, double cpu_bmode_data[],
                              int bmode_size[2]);

#endif
//
// File trailer for aline_reconstruct.h
//
// [EOF]
//
