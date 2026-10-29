//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: das_beamform_planewave.cu
//
// GPU Coder version                    : 24.2
// CUDA/C/C++ source code generated on  : 27-Oct-2026 19:17:32
//

// Include Files
#include "das_beamform_planewave.h"
#include "das_beamform_planewave_data.h"
#include "das_beamform_planewave_initialize.h"
#include "das_beamform_planewave_rtwutil.h"
#include "MWCudaDimUtility.hpp"
#include "MWCudaMemoryFunctions.hpp"
#include <cmath>

// Function Declarations
static __global__ void das_beamform_planewave_kernel1(
    const double rf_data[], const int rf_size_dim0, const double fs,
    const double x_elements_data[], const int x_elements_size_dim1,
    const double c, const double cos_a, const double sin_a,
    const double x_image[300], const double z_image[400], double bmode[120000]);

// Function Definitions
//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double rf_data[]
//                const int rf_size_dim0
//                const double fs
//                const double x_elements_data[]
//                const int x_elements_size_dim1
//                const double c
//                const double cos_a
//                const double sin_a
//                const double x_image[300]
//                const double z_image[400]
//                double bmode[120000]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void das_beamform_planewave_kernel1(
    const double rf_data[], const int rf_size_dim0, const double fs,
    const double x_elements_data[], const int x_elements_size_dim1,
    const double c, const double cos_a, const double sin_a,
    const double x_image[300], const double z_image[400], double bmode[120000])
{
  unsigned long long gThreadId;
  int ix;
  int iz;
  gThreadId = mwGetGlobalThreadIndex();
  ix = static_cast<int>(gThreadId % 300ULL);
  iz = static_cast<int>((gThreadId - static_cast<unsigned long long>(ix)) /
                        300ULL);
  if (iz < 400) {
    double d;
    double pixel_sum;
    double t_tx;
    d = z_image[iz];
    //  Transmit delay: planar wavefront arrival time at (x, z),
    //  NOT distance from any single element.
    t_tx = (x_image[ix] * sin_a + d * cos_a) / c;
    pixel_sum = 0.0;
    for (int ie{0}; ie < x_elements_size_dim1; ie++) {
      double dx;
      //  Receive delay: still the familiar per-element spherical
      //  distance, since each element independently records the echo.
      dx = x_image[ix] - x_elements_data[ie];
      dx = round((t_tx + sqrt(dx * dx + d * d) / c) * fs);
      if ((dx + 1.0 >= 1.0) &&
          (dx + 1.0 <= static_cast<double>(rf_size_dim0))) {
        pixel_sum +=
            rf_data[(static_cast<int>(dx + 1.0) + rf_size_dim0 * ie) - 1];
      }
    }
    bmode[iz + 400 * ix] = pixel_sum;
  }
}

//
// DAS_BEAMFORM_PLANEWAVE  Delay-and-sum beamformer for plane wave imaging
//    GPU Coder entry point. Distinct from das_beamform.m: that function
//    assumes classical per-element pulse-echo (each element fires and
//    receives independently, so delay = 2*dist/c). Plane wave imaging is
//    physically different -- all elements fire together to form a flat
//    wavefront, so the transmit delay follows the wavefront's angle and
//    arrival time, not distance from a single element. Only the receive
//    half still uses the familiar per-element spherical delay.
//
//   Inputs:
//     rf         - RF data matrix [num_samples x num_elements], double, single
//     steering angle x_elements - Lateral positions of transducer elements [1 x
//     num_elements], double z_image    - Depth positions of output pixels [1 x
//     num_depth], double x_image    - Lateral positions of output pixels [1 x
//     num_lateral], double fs         - Sampling frequency in Hz, double scalar
//     c          - Speed of sound in m/s, double scalar
//     tx_angle   - Plane wave steering angle in radians, double scalar
//
//   Output:
//     bmode      - Beamformed output image [num_depth x num_lateral], double
//
// Arguments    : const double cpu_rf_data[]
//                const int rf_size[2]
//                const double cpu_x_elements_data[]
//                const int x_elements_size[2]
//                const double cpu_z_image[400]
//                const double cpu_x_image[300]
//                double fs
//                double c
//                double tx_angle
//                double cpu_bmode[120000]
// Return Type  : void
//
void das_beamform_planewave(const double cpu_rf_data[], const int rf_size[2],
                            const double cpu_x_elements_data[],
                            const int x_elements_size[2],
                            const double cpu_z_image[400],
                            const double cpu_x_image[300], double fs, double c,
                            double tx_angle, double cpu_bmode[120000])
{
  double(*gpu_bmode)[120000];
  double(*gpu_z_image)[400];
  double(*gpu_x_image)[300];
  double *gpu_rf_data;
  double *gpu_x_elements_data;
  if (!isInitialized_das_beamform_planewave) {
    das_beamform_planewave_initialize();
  }
  b_checkCudaError(mwCudaMalloc(&gpu_bmode, 960000ULL), __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_x_image, 2400ULL), __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_z_image, 3200ULL), __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_x_elements_data, 256U * sizeof(double)),
                   __FILE__, __LINE__);
  b_checkCudaError(mwCudaMalloc(&gpu_rf_data, 1024000U * sizeof(double)),
                   __FILE__, __LINE__);
  b_checkCudaError(
      cudaMemcpy(gpu_rf_data, cpu_rf_data,
                 static_cast<unsigned int>(rf_size[0] * rf_size[1]) *
                     sizeof(double),
                 cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  b_checkCudaError(
      cudaMemcpy(gpu_x_elements_data, cpu_x_elements_data,
                 static_cast<unsigned int>(x_elements_size[1]) * sizeof(double),
                 cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  b_checkCudaError(
      cudaMemcpy(*gpu_x_image, cpu_x_image, 2400ULL, cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  b_checkCudaError(
      cudaMemcpy(*gpu_z_image, cpu_z_image, 3200ULL, cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  das_beamform_planewave_kernel1<<<dim3(235U, 1U, 1U), dim3(512U, 1U, 1U)>>>(
      gpu_rf_data, rf_size[0], fs, gpu_x_elements_data, x_elements_size[1], c,
      std::cos(tx_angle), std::sin(tx_angle), *gpu_x_image, *gpu_z_image,
      *gpu_bmode);
  b_checkCudaError(
      cudaMemcpy(cpu_bmode, *gpu_bmode, 960000ULL, cudaMemcpyDeviceToHost),
      __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(gpu_rf_data), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(gpu_x_elements_data), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_z_image), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_x_image), __FILE__, __LINE__);
  b_checkCudaError(mwCudaFree(*gpu_bmode), __FILE__, __LINE__);
}

//
// File trailer for das_beamform_planewave.cu
//
// [EOF]
//
