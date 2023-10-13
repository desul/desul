#ifndef DESUL_MACROS_H
#define DESUL_MACROS_H

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DESUL_GPUCC
#define DESUL_HOST_DEVICE __host__ __device__
#else // defined(CARE_GPUCC)
#define DESUL_HOST_DEVICE
#endif // defined(CARE_GPUCC)

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define DESUL_DEVICE_COMPILE
#endif

#endif // DESUL_MACROS_H

