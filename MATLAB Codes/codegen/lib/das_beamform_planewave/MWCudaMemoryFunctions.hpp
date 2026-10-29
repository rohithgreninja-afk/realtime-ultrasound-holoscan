// Copyright 2021-2024 The MathWorks, Inc.

#ifndef MW_CUDA_MEMORY_FUNCTIONS_HPP
#define MW_CUDA_MEMORY_FUNCTIONS_HPP

#if defined(MW_GPU_MEMORY_MANAGER) && !defined(__CUDA_ARCH__)

#if defined(_WIN32) || defined(__LCC__)
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

namespace gcmemory {

DLL_EXPORT cudaError_t mwCudaMallocImpl(void** devPtr, size_t size);
DLL_EXPORT cudaError_t mwCudaMallocManagedImpl(void** devPtr, size_t size);
DLL_EXPORT cudaError_t mwCudaFreeImpl(void* devPtr);

} // namespace gcmemory

template<typename MWType_>
cudaError_t mwCudaMalloc(MWType_** devPtr, size_t size) {
    return gcmemory::mwCudaMallocImpl(static_cast<void**>(static_cast<void*>(devPtr)), size);
}

template<typename MWType_>
cudaError_t mwCudaMallocManaged(MWType_** devPtr, size_t size) {
    return gcmemory::mwCudaMallocManagedImpl(static_cast<void**>(static_cast<void*>(devPtr)), size);
}

template<typename MWType_>
cudaError_t mwCudaFree(MWType_* devPtr) {
    return gcmemory::mwCudaFreeImpl(static_cast<void*>(devPtr));
}

#else
#include "cuda_runtime.h"

template<typename MWType_>
__host__ __device__
cudaError_t mwCudaMalloc(MWType_** devPtr, size_t size) {
    return cudaMalloc(devPtr, size);
}

template<typename MWType_>
__host__
cudaError_t mwCudaMallocManaged(MWType_** devPtr, size_t size) {
    return cudaMallocManaged(devPtr, size);
}

template<typename MWType_>
__host__ __device__
cudaError_t mwCudaFree(MWType_* devPtr) {
    return cudaFree(devPtr);
}

#endif // #ifdef MW_GPU_MEMORY_MANAGER
#endif // #ifndef MW_CUDA_MEMORY_FUNCTIONS_HPP
