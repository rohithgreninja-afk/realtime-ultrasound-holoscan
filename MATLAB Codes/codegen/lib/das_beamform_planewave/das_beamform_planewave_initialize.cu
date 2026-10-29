//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: das_beamform_planewave_initialize.cu
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 27-Oct-2026 19:17:32
//

// Include Files
#include "das_beamform_planewave_initialize.h"
#include "das_beamform_planewave_data.h"
#include "MWMemoryManager.hpp"

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void das_beamform_planewave_initialize()
{
  cudaGetLastError();
  mwMemoryManagerInit(256U, 0U, 8U, 2048U);
  cudaSetDevice(0);
  isInitialized_das_beamform_planewave = true;
}

//
// File trailer for das_beamform_planewave_initialize.cu
//
// [EOF]
//
