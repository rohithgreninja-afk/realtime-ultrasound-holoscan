//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: das_beamform.cu
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 20-May-2026 10:56:28
//

// Include Files
#include "das_beamform.h"
#include "das_beamform_data.h"
#include "das_beamform_initialize.h"
#include "das_beamform_rtwutil.h"
#include "MWCudaDimUtility.hpp"
#include "MWCudaMemoryFunctions.hpp"

// Function Declarations
static __global__ void
das_beamform_kernel1(const double rf[930240], const double fs, const double c,
                     const double x_elements[510], const double x_image[200],
                     const double z_image[300], double bmode[60000]);

// Function Definitions
//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double rf[930240]
//                const double fs
//                const double c
//                const double x_elements[510]
//                const double x_image[200]
//                const double z_image[300]
//                double bmode[60000]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void das_beamform_kernel1(
    const double rf[930240], const double fs, const double c,
    const double x_elements[510], const double x_image[200],
    const double z_image[300], double bmode[60000])
{
  unsigned long long gThreadId;
  int ix;
  int iz;
  gThreadId = mwGetGlobalThreadIndex();
  ix = static_cast<int>(gThreadId % 200ULL);
  iz = static_cast<int>((gThreadId - static_cast<unsigned long long>(ix)) /
                        200ULL);
  if (iz < 300) {
    double d;
    double pixel_sum;
    pixel_sum = 0.0;
    d = z_image[iz];
    for (int ie{0}; ie < 510; ie++) {
      double dx;
      dx = x_image[ix] - x_elements[ie];
      dx = round(sqrt(dx * dx + d * d) / c * fs);
      if ((dx + 1.0 >= 1.0) && (dx + 1.0 <= 1824.0)) {
        pixel_sum += rf[(static_cast<int>(dx + 1.0) + 1824 * ie) - 1];
      }
    }
    bmode[iz + 300 * ix] = pixel_sum;
  }
}

//
// DAS_BEAMFORM  Delay-and-Sum beamformer entry point for GPU Coder
//
//   Inputs:
//     rf         - Raw RF data matrix [num_samples x num_elements], double
//     x_elements - Lateral positions of transducer elements [1 x num_elements],
//     double z_image    - Depth positions of output pixels [1 x num_depth],
//     double x_image    - Lateral positions of output pixels [1 x num_lateral],
//     double fs         - Sampling frequency in Hz, double scalar c          -
//     Speed of sound in m/s, double scalar
//
//   Output:
//     bmode      - Beamformed output image [num_depth x num_lateral], double
//
// Arguments    : const double cpu_rf[930240]
//                const double cpu_x_elements[510]
//                const double cpu_z_image[300]
//                const double cpu_x_image[200]
//                double fs
//                double c
//                double cpu_bmode[60000]
// Return Type  : void
//
void das_beamform(const double cpu_rf[930240], const double cpu_x_elements[510],
                  const double cpu_z_image[300], const double cpu_x_image[200],
                  double fs, double c, double cpu_bmode[60000])
{
  double(*gpu_rf)[930240];
  double(*gpu_bmode)[60000];
  double(*gpu_x_elements)[510];
  double(*gpu_z_image)[300];
  double(*gpu_x_image)[200];
  if (!isInitialized_das_beamform) {
    das_beamform_initialize();
  }
  b_checkCudaError(mwCudaMalloc(&gpu_bmode, 480000ULL), __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_x_image, 1600ULL), __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_z_image, 2400ULL), __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_x_elements, 4080ULL), __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_rf, 7441920ULL), __FILE__, __LINE__);
  b_checkCudaError(
      cudaMemcpy(*gpu_rf, cpu_rf, 7441920ULL, cudaMemcpyHostToDevice), __FILE__,
      __LINE__);
  b_checkCudaError(cudaMemcpy(*gpu_x_elements, cpu_x_elements, 4080ULL,
                              cudaMemcpyHostToDevice),
                   __FILE__, __LINE__);
  b_checkCudaError(
      cudaMemcpy(*gpu_x_image, cpu_x_image, 1600ULL, cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  b_checkCudaError(
      cudaMemcpy(*gpu_z_image, cpu_z_image, 2400ULL, cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  das_beamform_kernel1<<<dim3(118U, 1U, 1U), dim3(512U, 1U, 1U)>>>(
      *gpu_rf, fs, c, *gpu_x_elements, *gpu_x_image, *gpu_z_image, *gpu_bmode);
  b_checkCudaError(
      cudaMemcpy(cpu_bmode, *gpu_bmode, 480000ULL, cudaMemcpyDeviceToHost),
      __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_rf), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_x_elements), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_z_image), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_x_image), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_bmode), __FILE__, __LINE__);
}

//
// File trailer for das_beamform.cu
//
// [EOF]
//
