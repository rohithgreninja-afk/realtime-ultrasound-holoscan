//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: das_beamform_terminate.cu
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 20-May-2026 10:56:28
//

// Include Files
#include "das_beamform_terminate.h"
#include "das_beamform_data.h"
#include "das_beamform_rtwutil.h"
#include "MWMemoryManager.hpp"
#include <stdio.h>
#include <stdlib.h>

// Function Declarations
static void checkCudaError(cudaError_t errorCode, const char *file, int b_line);

static void gpuThrowError(unsigned int errorCode, const char *errorName,
                          const char *errorString, const char *file,
                          int b_line);

// Function Definitions
//
// Arguments    : cudaError_t errorCode
//                const char *file
//                int b_line
// Return Type  : void
//
static void checkCudaError(cudaError_t errorCode, const char *file, int b_line)
{
  if (errorCode != cudaSuccess) {
    gpuThrowError(errorCode, cudaGetErrorName(errorCode),
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
static void gpuThrowError(unsigned int errorCode, const char *errorName,
                          const char *errorString, const char *file, int b_line)
{
  fprintf(stderr,
          "CUDA error [%d,%s] : %s\nFile: \"%s\"\nLine: %d\nTerminating "
          "execution...",
          errorCode, errorName, errorString, file, b_line);
  exit(1);
}

//
// Arguments    : void
// Return Type  : void
//
void das_beamform_terminate()
{
  checkCudaError(cudaGetLastError(), __FILE__, __LINE__);
  b_checkCudaError(mwMemoryManagerTerminate(), __FILE__, __LINE__);
  isInitialized_das_beamform = false;
}

//
// File trailer for das_beamform_terminate.cu
//
// [EOF]
//
