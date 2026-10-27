//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: aline_reconstruct_initialize.cu
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 27-Oct-2026 19:17:05
//

// Include Files
#include "aline_reconstruct_initialize.h"
#include "aline_reconstruct_data.h"
#include "MWMemoryManager.hpp"

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void aline_reconstruct_initialize()
{
  cudaGetLastError();
  mwMemoryManagerInit(256U, 0U, 8U, 2048U);
  cudaSetDevice(0);
  isInitialized_aline_reconstruct = true;
}

//
// File trailer for aline_reconstruct_initialize.cu
//
// [EOF]
//
