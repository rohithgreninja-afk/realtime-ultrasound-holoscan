//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: das_beamform_rtwutil.cu
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 20-May-2026 10:56:28
//

// Include Files
#include "das_beamform_rtwutil.h"
#include <stdio.h>
#include <stdlib.h>

// Function Definitions
//
// Arguments    : cudaError_t errorCode
//                const char *file
//                int b_line
// Return Type  : void
//
void b_checkCudaError(cudaError_t errorCode, const char *file, int b_line)
{
  if (errorCode != cudaSuccess) {
    b_gpuThrowError(errorCode, cudaGetErrorName(errorCode),
                    cudaGetErrorString(errorCode), file, b_line);
  }
}

//
// Arguments    : unsigned int errorCode
//                const char *errorName
//                const char *errorString
//                const char *file
//                int b_line
// Return Type  : void
//
void b_gpuThrowError(unsigned int errorCode, const char *errorName,
                     const char *errorString, const char *file, int b_line)
{
  fprintf(stderr,
          "CUDA error [%d,%s] : %s\nFile: \"%s\"\nLine: %d\nTerminating "
          "execution...",
          errorCode, errorName, errorString, file, b_line);
  exit(1);
}

//
// File trailer for das_beamform_rtwutil.cu
//
// [EOF]
//
