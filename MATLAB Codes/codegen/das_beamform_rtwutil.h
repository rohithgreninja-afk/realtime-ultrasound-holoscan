//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: das_beamform_rtwutil.h
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 20-May-2026 10:56:28
//

#ifndef DAS_BEAMFORM_RTWUTIL_H
#define DAS_BEAMFORM_RTWUTIL_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
extern void b_checkCudaError(cudaError_t errorCode, const char *file,
                             int b_line);

extern void b_gpuThrowError(unsigned int errorCode, const char *errorName,
                            const char *errorString, const char *file,
                            int b_line);

#endif
//
// File trailer for das_beamform_rtwutil.h
//
// [EOF]
//
