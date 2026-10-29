//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: aline_reconstruct.cu
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 27-Oct-2026 19:17:05
//

// Include Files
#include "aline_reconstruct.h"
#include "aline_reconstruct_data.h"
#include "aline_reconstruct_initialize.h"
#include "aline_reconstruct_rtwutil.h"
#include "MWCUFFTPlanManager.hpp"
#include "MWCudaDimUtility.hpp"
#include "MWCudaMemoryFunctions.hpp"
#include "MWLaunchParametersUtilities.hpp"
#include "cufft.h"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

// Function Declarations
static __global__ void aline_reconstruct_kernel1(creal_T x_data[1530000]);

static __global__ void aline_reconstruct_kernel10(const int N, const int b,
                                                  creal_T x_data[1530000]);

static __global__ void aline_reconstruct_kernel11(const short x_size_dim0,
                                                  const int N, const int b,
                                                  creal_T x_data[1530000]);

static __global__ void aline_reconstruct_kernel12(creal_T x_data[510]);

static __global__ void aline_reconstruct_kernel13(creal_T x_data[510]);

static __global__ void aline_reconstruct_kernel14(creal_T x_data[510]);

static __global__ void
aline_reconstruct_kernel15(const creal_T x_data[510], const int analytic,
                           creal_T analytic_data[1530000]);

static __global__ void aline_reconstruct_kernel2(creal_T x_data[1530000]);

static __global__ void aline_reconstruct_kernel3(creal_T x_data[1530000]);

static __global__ void
aline_reconstruct_kernel4(const creal_T x_data[1530000], const int analytic,
                          creal_T analytic_data[1530000]);

static __global__ void
aline_reconstruct_kernel5(const creal_T analytic_data[1530000], const int b,
                          double envelope_data[1530000]);

static __global__ void
aline_reconstruct_kernel6(const double env_max,
                          const double envelope_data[1530000], const int b,
                          double env_norm_data[1530000]);

static __global__ void
aline_reconstruct_kernel7(const double b_gamma,
                          const double env_norm_data[1530000], const int b,
                          double bmode_data[]);

static __global__ void aline_reconstruct_kernel8(const short x_size_dim0,
                                                 const int halfn, const int N,
                                                 const int b,
                                                 creal_T x_data[1530000]);

static __global__ void
aline_reconstruct_kernel9(const double x_re, const creal_T x_data[1530000],
                          const int x_size, creal_T analytic_data[1530000]);

static unsigned long long computeNumIters(int ub, int b_ub);

static unsigned long long computeNumIters(int ub);

static void gpuThrowError(const char *file, int b_line);

// Function Definitions
//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                creal_T x_data[1530000]
// Return Type  : void
//
static __global__
    __launch_bounds__(128,
                      1) void aline_reconstruct_kernel1(creal_T x_data[1530000])
{
  int i;
  i = static_cast<int>(mwGetGlobalThreadIndex());
  if (i < 254) {
    x_data[509 - i].re = x_data[i + 1].re;
    x_data[509 - i].im = -x_data[i + 1].im;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int N
//                const int b
//                creal_T x_data[1530000]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void aline_reconstruct_kernel10(
    const int N, const int b, creal_T x_data[1530000])
{
  unsigned long long gStride;
  unsigned long long gThreadId;
  unsigned long long loopEnd;
  gThreadId = mwGetGlobalThreadIndex();
  gStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long long>(b);
  for (unsigned long long idx{gThreadId}; idx <= loopEnd; idx += gStride) {
    int i;
    i = static_cast<int>(idx);
    x_data[(N - i) - 1].re = x_data[i + 1].re;
    x_data[(N - i) - 1].im = -x_data[i + 1].im;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const short x_size_dim0
//                const int N
//                const int b
//                creal_T x_data[1530000]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void aline_reconstruct_kernel11(
    const short x_size_dim0, const int N, const int b, creal_T x_data[1530000])
{
  unsigned long long gStride;
  unsigned long long gThreadId;
  unsigned long long loopEnd;
  gThreadId = mwGetGlobalThreadIndex();
  gStride = mwGetTotalThreadsLaunched();
  loopEnd = 510ULL * (static_cast<unsigned long long>(b) + 1ULL) - 1ULL;
  for (unsigned long long idx{gThreadId}; idx <= loopEnd; idx += gStride) {
    int i;
    int j;
    j = static_cast<int>(idx % (static_cast<unsigned long long>(b) + 1ULL));
    i = static_cast<int>((idx - static_cast<unsigned long long>(j)) /
                         (static_cast<unsigned long long>(b) + 1ULL));
    x_data[((N - j) + static_cast<int>(x_size_dim0) * i) - 1].re =
        x_data[(j + static_cast<int>(x_size_dim0) * i) + 1].re;
    x_data[((N - j) + static_cast<int>(x_size_dim0) * i) - 1].im =
        -x_data[(j + static_cast<int>(x_size_dim0) * i) + 1].im;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                creal_T x_data[510]
// Return Type  : void
//
static __global__
    __launch_bounds__(128,
                      1) void aline_reconstruct_kernel12(creal_T x_data[510])
{
  int i;
  i = static_cast<int>(mwGetGlobalThreadIndex());
  if (i < 254) {
    x_data[509 - i].re = x_data[i + 1].re;
    x_data[509 - i].im = -x_data[i + 1].im;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                creal_T x_data[510]
// Return Type  : void
//
static __global__
    __launch_bounds__(128,
                      1) void aline_reconstruct_kernel13(creal_T x_data[510])
{
  int i;
  i = static_cast<int>(mwGetGlobalThreadIndex());
  if (i < 509) {
    if (i + 2 <= 255) {
      x_data[i + 1].re *= 2.0;
      x_data[i + 1].im *= 2.0;
    } else if (i + 2 >= 257) {
      x_data[i + 1].re = 0.0;
      x_data[i + 1].im = 0.0;
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                creal_T x_data[510]
// Return Type  : void
//
static __global__
    __launch_bounds__(128,
                      1) void aline_reconstruct_kernel14(creal_T x_data[510])
{
  int j;
  j = static_cast<int>(mwGetGlobalThreadIndex());
  if (j < 510) {
    double ai;
    double ar;
    double re;
    ar = x_data[j].re;
    ai = x_data[j].im;
    if (ai == 0.0) {
      re = ar / 510.0;
      ar = 0.0;
    } else if (ar == 0.0) {
      re = 0.0;
      ar = ai / 510.0;
    } else {
      re = ar / 510.0;
      ar = ai / 510.0;
    }
    x_data[j].re = re;
    x_data[j].im = ar;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const creal_T x_data[510]
//                const int analytic
//                creal_T analytic_data[1530000]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void aline_reconstruct_kernel15(
    const creal_T x_data[510], const int analytic,
    creal_T analytic_data[1530000])
{
  unsigned long long gStride;
  unsigned long long gThreadId;
  unsigned long long loopEnd;
  gThreadId = mwGetGlobalThreadIndex();
  gStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long long>(analytic);
  for (unsigned long long idx{gThreadId}; idx <= loopEnd; idx += gStride) {
    int j;
    j = static_cast<int>(idx);
    analytic_data[j] = x_data[j];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                creal_T x_data[1530000]
// Return Type  : void
//
static __global__
    __launch_bounds__(128,
                      1) void aline_reconstruct_kernel2(creal_T x_data[1530000])
{
  int i;
  i = static_cast<int>(mwGetGlobalThreadIndex());
  if (i < 509) {
    if (i + 2 <= 255) {
      x_data[i + 1].re *= 2.0;
      x_data[i + 1].im *= 2.0;
    } else if (i + 2 >= 257) {
      x_data[i + 1].re = 0.0;
      x_data[i + 1].im = 0.0;
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                creal_T x_data[1530000]
// Return Type  : void
//
static __global__
    __launch_bounds__(128,
                      1) void aline_reconstruct_kernel3(creal_T x_data[1530000])
{
  int j;
  j = static_cast<int>(mwGetGlobalThreadIndex());
  if (j < 510) {
    double ai;
    double ar;
    double re;
    ar = x_data[j].re;
    ai = x_data[j].im;
    if (ai == 0.0) {
      re = ar / 510.0;
      ar = 0.0;
    } else if (ar == 0.0) {
      re = 0.0;
      ar = ai / 510.0;
    } else {
      re = ar / 510.0;
      ar = ai / 510.0;
    }
    x_data[j].re = re;
    x_data[j].im = ar;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const creal_T x_data[1530000]
//                const int analytic
//                creal_T analytic_data[1530000]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void aline_reconstruct_kernel4(
    const creal_T x_data[1530000], const int analytic,
    creal_T analytic_data[1530000])
{
  unsigned long long gStride;
  unsigned long long gThreadId;
  unsigned long long loopEnd;
  gThreadId = mwGetGlobalThreadIndex();
  gStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long long>(analytic);
  for (unsigned long long idx{gThreadId}; idx <= loopEnd; idx += gStride) {
    int j;
    j = static_cast<int>(idx);
    analytic_data[j] = x_data[j];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const creal_T analytic_data[1530000]
//                const int b
//                double envelope_data[1530000]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void aline_reconstruct_kernel5(
    const creal_T analytic_data[1530000], const int b,
    double envelope_data[1530000])
{
  unsigned long long gStride;
  unsigned long long gThreadId;
  unsigned long long loopEnd;
  gThreadId = mwGetGlobalThreadIndex();
  gStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long long>(b);
  for (unsigned long long idx{gThreadId}; idx <= loopEnd; idx += gStride) {
    int j;
    j = static_cast<int>(idx);
    envelope_data[j] = hypot(analytic_data[j].re, analytic_data[j].im);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double env_max
//                const double envelope_data[1530000]
//                const int b
//                double env_norm_data[1530000]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void aline_reconstruct_kernel6(
    const double env_max, const double envelope_data[1530000], const int b,
    double env_norm_data[1530000])
{
  unsigned long long gStride;
  unsigned long long gThreadId;
  unsigned long long loopEnd;
  gThreadId = mwGetGlobalThreadIndex();
  gStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long long>(b);
  for (unsigned long long idx{gThreadId}; idx <= loopEnd; idx += gStride) {
    int j;
    j = static_cast<int>(idx);
    env_norm_data[j] = envelope_data[j] / env_max;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double b_gamma
//                const double env_norm_data[1530000]
//                const int b
//                double bmode_data[]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void aline_reconstruct_kernel7(
    const double b_gamma, const double env_norm_data[1530000], const int b,
    double bmode_data[])
{
  unsigned long long gStride;
  unsigned long long gThreadId;
  unsigned long long loopEnd;
  gThreadId = mwGetGlobalThreadIndex();
  gStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long long>(b);
  for (unsigned long long idx{gThreadId}; idx <= loopEnd; idx += gStride) {
    int j;
    j = static_cast<int>(idx);
    bmode_data[j] = pow(env_norm_data[j], b_gamma);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const short x_size_dim0
//                const int halfn
//                const int N
//                const int b
//                creal_T x_data[1530000]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void aline_reconstruct_kernel8(
    const short x_size_dim0, const int halfn, const int N, const int b,
    creal_T x_data[1530000])
{
  unsigned long long gStride;
  unsigned long long gThreadId;
  unsigned long long loopEnd;
  gThreadId = mwGetGlobalThreadIndex();
  gStride = mwGetTotalThreadsLaunched();
  loopEnd = 510ULL * (static_cast<unsigned long long>(b) + 1ULL) - 1ULL;
  for (unsigned long long idx{gThreadId}; idx <= loopEnd; idx += gStride) {
    int i;
    int j;
    i = static_cast<int>(idx % (static_cast<unsigned long long>(b) + 1ULL));
    j = static_cast<int>((idx - static_cast<unsigned long long>(i)) /
                         (static_cast<unsigned long long>(b) + 1ULL));
    if (i + 2 <= N) {
      x_data[(i + static_cast<int>(x_size_dim0) * j) + 1].re *= 2.0;
      x_data[(i + static_cast<int>(x_size_dim0) * j) + 1].im *= 2.0;
    } else if (i >= halfn) {
      x_data[(i + static_cast<int>(x_size_dim0) * j) + 1].re = 0.0;
      x_data[(i + static_cast<int>(x_size_dim0) * j) + 1].im = 0.0;
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double x_re
//                const creal_T x_data[1530000]
//                const int x_size
//                creal_T analytic_data[1530000]
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void aline_reconstruct_kernel9(
    const double x_re, const creal_T x_data[1530000], const int x_size,
    creal_T analytic_data[1530000])
{
  unsigned long long gStride;
  unsigned long long gThreadId;
  unsigned long long loopEnd;
  gThreadId = mwGetGlobalThreadIndex();
  gStride = mwGetTotalThreadsLaunched();
  loopEnd = static_cast<unsigned long long>(x_size);
  for (unsigned long long idx{gThreadId}; idx <= loopEnd; idx += gStride) {
    double ai;
    double ar;
    int j;
    j = static_cast<int>(idx);
    ar = x_data[j].re;
    ai = x_data[j].im;
    if (ai == 0.0) {
      analytic_data[j].re = ar / x_re;
      analytic_data[j].im = 0.0;
    } else if (ar == 0.0) {
      analytic_data[j].re = 0.0;
      analytic_data[j].im = ai / x_re;
    } else {
      analytic_data[j].re = ar / x_re;
      analytic_data[j].im = ai / x_re;
    }
  }
}

//
// Arguments    : int ub
//                int b_ub
// Return Type  : unsigned long long
//
static unsigned long long computeNumIters(int ub, int b_ub)
{
  unsigned long long n;
  unsigned long long numIters;
  boolean_T overflow;
  overflow = false;
  n = 0ULL;
  if (ub >= 0) {
    n = static_cast<unsigned long long>(ub + 1);
  }
  numIters = n;
  n = 0ULL;
  if (b_ub >= 0) {
    n = static_cast<unsigned long long>(b_ub + 1);
    overflow = (numIters > MAX_uint64_T / n);
  }
  numIters *= n;
  if (overflow) {
    gpuThrowError(__FILE__, __LINE__);
  }
  return numIters;
}

//
// Arguments    : int ub
// Return Type  : unsigned long long
//
static unsigned long long computeNumIters(int ub)
{
  unsigned long long numIters;
  numIters = 0ULL;
  if (ub >= 0) {
    numIters = static_cast<unsigned long long>(ub + 1);
  }
  return numIters;
}

//
// Arguments    : const char *file
//                int b_line
// Return Type  : void
//
static void gpuThrowError(const char *file, int b_line)
{
  fprintf(stderr, "%s\nFile: \"%s\"\nLine: %d\nTerminating execution...",
          "Unable to launch kernel. Loop nest contains too many iterations.",
          file, b_line);
  exit(1);
}

//
// ALINE_RECONSTRUCT  Per-column Hilbert envelope detection + power-law
// compression
//    GPU Coder entry point. This is the correct reconstruction method for
//    OASBUD, matching BeamformingOp._aline_reconstruct in
//    Holoscan/beamforming_op.py exactly: no delay calculation, no array
//    geometry, a Hilbert transform down each column followed by power-law
//    compression against the global peak.
//
//   Inputs:
//     rf    - RF data matrix [num_samples x num_columns], double
//     gamma - power-law compression exponent, double scalar (0.3 in the
//     deployed pipeline)
//
//   Output:
//     bmode - compressed envelope image [num_samples x num_columns], double
//
// Arguments    : const double cpu_rf_data[]
//                const int rf_size[2]
//                double b_gamma
//                double cpu_bmode_data[]
//                int bmode_size[2]
// Return Type  : void
//
void aline_reconstruct(const double cpu_rf_data[], const int rf_size[2],
                       double b_gamma, double cpu_bmode_data[],
                       int bmode_size[2])
{
  static double cpu_envelope_data[1530000];
  dim3 block;
  dim3 grid;
  creal_T(*gpu_analytic_data)[1530000];
  creal_T(*gpu_x_data)[1530000];
  creal_T(*b_gpu_x_data)[510];
  double(*gpu_env_norm_data)[1530000];
  double(*gpu_envelope_data)[1530000];
  double env_max;
  double *gpu_bmode_data;
  double *gpu_rf_data;
  int analytic_size[2];
  int N;
  int dim;
  int halfn;
  short x_size[2];
  boolean_T exitg1;
  boolean_T validLaunchParams;
  if (!isInitialized_aline_reconstruct) {
    aline_reconstruct_initialize();
  }
  b_checkCudaError(mwCudaMalloc(&gpu_env_norm_data, 12240000ULL), __FILE__,
                   __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_envelope_data, 12240000ULL), __FILE__,
                   __LINE__);
  b_checkCudaError(mwCudaMalloc(&b_gpu_x_data, 8160ULL), __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_x_data, 24480000ULL), __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_analytic_data, 24480000ULL), __FILE__,
                   __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_bmode_data, 1530000U * sizeof(double)),
                   __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_rf_data, 1530000U * sizeof(double)),
                   __FILE__, __LINE__);
  dim = 1;
  halfn = 1;
  exitg1 = false;
  while ((!exitg1) && (halfn < 3)) {
    if (rf_size[halfn - 1] != 1) {
      dim = halfn;
      exitg1 = true;
    } else {
      halfn++;
    }
  }
  if (rf_size[0] == 1) {
    cufftHandle f_fftPlanHandle;
    cufftHandle fftPlanHandle;
    dim = 510;
    fftPlanHandle = acquireCUFFTPlan(1, &dim, &dim, 1, 510, CUFFT_D2Z, 1);
    b_checkCudaError(cudaMemcpy(gpu_rf_data, cpu_rf_data, 510U * sizeof(double),
                                cudaMemcpyHostToDevice),
                     __FILE__, __LINE__);
    cufftExecD2Z(fftPlanHandle, (cufftDoubleReal *)&gpu_rf_data[0],
                 (cufftDoubleComplex *)&(*gpu_x_data)[0]);
    aline_reconstruct_kernel1<<<dim3(2U, 1U, 1U), dim3(128U, 1U, 1U)>>>(
        *gpu_x_data);
    aline_reconstruct_kernel2<<<dim3(4U, 1U, 1U), dim3(128U, 1U, 1U)>>>(
        *gpu_x_data);
    dim = 510;
    f_fftPlanHandle = acquireCUFFTPlan(1, &dim, &dim, 1, 510, CUFFT_Z2Z, 1);
    mw::cufftExecZ2ZInPlace(f_fftPlanHandle,
                            (cufftDoubleComplex *)&(*gpu_x_data)[0],
                            CUFFT_INVERSE);
    aline_reconstruct_kernel3<<<dim3(4U, 1U, 1U), dim3(128U, 1U, 1U)>>>(
        *gpu_x_data);
    analytic_size[0] = 1;
    mwGetLaunchParameters1D(computeNumIters(509), &grid, &block, 2147483647U);
    validLaunchParams = mwValidateLaunchParameters(grid, block);
    if (validLaunchParams) {
      aline_reconstruct_kernel4<<<grid, block>>>(*gpu_x_data, 509,
                                                 *gpu_analytic_data);
    }
  } else if (dim == 1) {
    unsigned long long numIters;
    if (rf_size[0] == 0) {
      x_size[0] = 0;
    } else {
      cufftHandle c_fftPlanHandle;
      dim = rf_size[0];
      x_size[0] = static_cast<short>(rf_size[0]);
      c_fftPlanHandle =
          acquireCUFFTPlan(1, &dim, &dim, 1, rf_size[0], CUFFT_D2Z, 510);
      b_checkCudaError(cudaMemcpy(gpu_rf_data, cpu_rf_data,
                                  static_cast<unsigned int>(rf_size[0] * 510) *
                                      sizeof(double),
                                  cudaMemcpyHostToDevice),
                       __FILE__, __LINE__);
      cufftExecD2Z(c_fftPlanHandle, (cufftDoubleReal *)&gpu_rf_data[0],
                   (cufftDoubleComplex *)&(*gpu_x_data)[0]);
      dim = static_cast<int>(
          (static_cast<double>(static_cast<short>(rf_size[0])) + 1.0) / 2.0);
      if (static_cast<short>(rf_size[0]) == 1) {
        mwGetLaunchParameters1D(computeNumIters(dim - 2), &grid, &block,
                                2147483647U);
        validLaunchParams = mwValidateLaunchParameters(grid, block);
        if (validLaunchParams) {
          aline_reconstruct_kernel10<<<grid, block>>>(
              static_cast<int>(static_cast<short>(rf_size[0])), dim - 2,
              *gpu_x_data);
        }
      } else {
        numIters = computeNumIters(509, dim - 2);
        mwGetLaunchParameters1D(numIters, &grid, &block, 2147483647U);
        validLaunchParams = mwValidateLaunchParameters(grid, block);
        if (validLaunchParams) {
          aline_reconstruct_kernel11<<<grid, block>>>(
              static_cast<short>(rf_size[0]),
              static_cast<int>(static_cast<short>(rf_size[0])), dim - 2,
              *gpu_x_data);
        }
      }
    }
    halfn = rf_size[0] >> 1;
    numIters = computeNumIters(509, rf_size[0] - 2);
    mwGetLaunchParameters1D(numIters, &grid, &block, 2147483647U);
    if ((static_cast<unsigned int>(rf_size[0]) & 1U) == 0U) {
      N = halfn;
    } else {
      N = halfn + 1;
    }
    validLaunchParams = mwValidateLaunchParameters(grid, block);
    if (validLaunchParams) {
      aline_reconstruct_kernel8<<<grid, block>>>(x_size[0], halfn, N,
                                                 rf_size[0] - 2, *gpu_x_data);
    }
    if (x_size[0] == 0) {
      analytic_size[0] = 0;
    } else {
      cufftHandle e_fftPlanHandle;
      dim = x_size[0];
      e_fftPlanHandle = acquireCUFFTPlan(
          1, &dim, &dim, 1, static_cast<int>(x_size[0]), CUFFT_Z2Z, 510);
      mw::cufftExecZ2ZInPlace(e_fftPlanHandle,
                              (cufftDoubleComplex *)&(*gpu_x_data)[0],
                              CUFFT_INVERSE);
      analytic_size[0] = x_size[0];
      N = x_size[0] * 510 - 1;
      mwGetLaunchParameters1D(computeNumIters(N), &grid, &block, 2147483647U);
      validLaunchParams = mwValidateLaunchParameters(grid, block);
      if (validLaunchParams) {
        aline_reconstruct_kernel9<<<grid, block>>>(
            static_cast<double>(x_size[0]), *gpu_x_data, N, *gpu_analytic_data);
      }
    }
  } else {
    cufftHandle b_fftPlanHandle;
    cufftHandle d_fftPlanHandle;
    dim = 510;
    b_fftPlanHandle = acquireCUFFTPlan(1, &dim, &dim, 1, 510, CUFFT_D2Z, 1);
    b_checkCudaError(
        cudaMemcpy(gpu_rf_data, cpu_rf_data,
                   static_cast<unsigned int>(rf_size[0] * 510) * sizeof(double),
                   cudaMemcpyHostToDevice),
        __FILE__, __LINE__);
    cufftExecD2Z(b_fftPlanHandle, (cufftDoubleReal *)&gpu_rf_data[0],
                 (cufftDoubleComplex *)&(*b_gpu_x_data)[0]);
    aline_reconstruct_kernel12<<<dim3(2U, 1U, 1U), dim3(128U, 1U, 1U)>>>(
        *b_gpu_x_data);
    aline_reconstruct_kernel13<<<dim3(4U, 1U, 1U), dim3(128U, 1U, 1U)>>>(
        *b_gpu_x_data);
    dim = 510;
    d_fftPlanHandle = acquireCUFFTPlan(1, &dim, &dim, 1, 510, CUFFT_Z2Z, 1);
    mw::cufftExecZ2ZInPlace(d_fftPlanHandle,
                            (cufftDoubleComplex *)&(*b_gpu_x_data)[0],
                            CUFFT_INVERSE);
    aline_reconstruct_kernel14<<<dim3(4U, 1U, 1U), dim3(128U, 1U, 1U)>>>(
        *b_gpu_x_data);
    analytic_size[0] = rf_size[0];
    N = rf_size[0] * 510 - 1;
    mwGetLaunchParameters1D(computeNumIters(N), &grid, &block, 2147483647U);
    validLaunchParams = mwValidateLaunchParameters(grid, block);
    if (validLaunchParams) {
      aline_reconstruct_kernel15<<<grid, block>>>(*b_gpu_x_data, N,
                                                  *gpu_analytic_data);
    }
  }
  //  MATLAB's hilbert() operates per-column on a matrix by default
  mwGetLaunchParameters1D(computeNumIters(analytic_size[0] * 510 - 1), &grid,
                          &block, 2147483647U);
  validLaunchParams = mwValidateLaunchParameters(grid, block);
  if (validLaunchParams) {
    aline_reconstruct_kernel5<<<grid, block>>>(
        *gpu_analytic_data, analytic_size[0] * 510 - 1, *gpu_envelope_data);
  }
  N = analytic_size[0] * 510;
  b_checkCudaError(cudaMemcpy(cpu_envelope_data, *gpu_envelope_data,
                              12240000ULL, cudaMemcpyDeviceToHost),
                   __FILE__, __LINE__);
  env_max = cpu_envelope_data[0];
  for (dim = 0; dim <= N - 2; dim++) {
    boolean_T p;
    if (std::isnan(cpu_envelope_data[dim + 1])) {
      p = false;
    } else if (std::isnan(env_max)) {
      p = true;
    } else {
      p = (env_max < cpu_envelope_data[dim + 1]);
    }
    if (p) {
      env_max = cpu_envelope_data[dim + 1];
    }
  }
  env_max += 1.0E-12;
  N = analytic_size[0] * 510 - 1;
  mwGetLaunchParameters1D(computeNumIters(N), &grid, &block, 2147483647U);
  validLaunchParams = mwValidateLaunchParameters(grid, block);
  if (validLaunchParams) {
    aline_reconstruct_kernel6<<<grid, block>>>(env_max, *gpu_envelope_data, N,
                                               *gpu_env_norm_data);
  }
  bmode_size[0] = analytic_size[0];
  bmode_size[1] = 510;
  mwGetLaunchParameters1D(computeNumIters(analytic_size[0] * 510 - 1), &grid,
                          &block, 2147483647U);
  validLaunchParams = mwValidateLaunchParameters(grid, block);
  if (validLaunchParams) {
    aline_reconstruct_kernel7<<<grid, block>>>(b_gamma, *gpu_env_norm_data,
                                               analytic_size[0] * 510 - 1,
                                               gpu_bmode_data);
  }
  b_checkCudaError(
      cudaMemcpy(cpu_bmode_data, gpu_bmode_data,
                 static_cast<unsigned int>(analytic_size[0] * 510) *
                     sizeof(double),
                 cudaMemcpyDeviceToHost),
      __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(gpu_rf_data), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(gpu_bmode_data), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_analytic_data), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_x_data), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*b_gpu_x_data), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_envelope_data), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_env_norm_data), __FILE__, __LINE__);
}

//
// File trailer for aline_reconstruct.cu
//
// [EOF]
//
