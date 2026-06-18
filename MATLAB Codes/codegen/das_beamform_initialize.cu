//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: das_beamform_initialize.cu
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 20-May-2026 10:56:28
//

// Include Files
#include "das_beamform_initialize.h"
#include "das_beamform_data.h"
#include "MWMemoryManager.hpp"

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void das_beamform_initialize()
{
  cudaGetLastError();
  mwMemoryManagerInit(256U, 0U, 8U, 2048U);
  cudaSetDevice(0);
  isInitialized_das_beamform = true;
}

//
// File trailer for das_beamform_initialize.cu
//
// [EOF]
//
