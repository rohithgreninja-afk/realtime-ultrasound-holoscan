/* Copyright 2019-2020 The MathWorks, Inc. */
#ifndef __MWCUFFTPLANMANGER_HPP__
#define __MWCUFFTPLANMANGER_HPP__

#include <cufft.h>

cufftHandle acquireCUFFTPlan(int rank, int* n, int* inembed, int istride, int idist, cufftType type, int batch);

namespace mw {

inline void cufftExecC2CInPlace(cufftHandle plan, cufftComplex* data, int direction) {
    ::cufftExecC2C(plan, data, data, direction);
}

inline void cufftExecZ2ZInPlace(cufftHandle plan, cufftDoubleComplex* data, int direction) {
    ::cufftExecZ2Z(plan, data, data, direction);
}
    
}

#endif
