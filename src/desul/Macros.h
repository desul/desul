#ifndef DESUL_MACROS_H
#define DESUL_MACROS_H

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DESUL_GPUCC
#define DESUL_HOST_DEVICE __host__ __device__
#else
#define DESUL_HOST_DEVICE
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
#define DESUL_DEVICE_COMPILE
#endif

#endif // DESUL_MACROS_H

