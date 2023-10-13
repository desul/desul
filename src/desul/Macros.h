#ifndef DESUL_MACROS_H
#define DESUL_MACROS_H

// SYCL needs no specifiers on functions
// TODO: Determine how to handle OpenACC and OpenMP target offload
#if defined(__CUDACC__) || \
    defined(__HIPCC__)
#define DESUL_GLOBAL __global__
#define DESUL_HOST __host__
#define DESUL_DEVICE __device__
#define DESUL_HOST_DEVICE __host__ __device__
#else
#define DESUL_GLOBAL
#define DESUL_HOST
#define DESUL_DEVICE
#define DESUL_HOST_DEVICE
#endif

#if defined(__CUDA_ARCH__) || \
    defined(__HIP_DEVICE_COMPILE__) || \
    defined(__SYCL_DEVICE_ONLY__)
#define DESUL_DEVICE_COMPILE
#endif

#endif // DESUL_MACROS_H

