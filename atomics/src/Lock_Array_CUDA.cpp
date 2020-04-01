/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#include <atomics/Lock_Array.hpp>
#include <cinttypes>

#ifdef DESUL_HAVE_CUDA_ATOMICS
#ifdef __CUDACC_RDC__
namespace desul {
namespace Impl {
__device__ __constant__ int32_t* CUDA_SPACE_ATOMIC_LOCKS_DEVICE = nullptr;
}
}  // namespace desul
#endif

namespace desul {

namespace {

__global__ void init_lock_arrays_cuda_kernel() {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < CUDA_SPACE_ATOMIC_MASK + 1) {
    Impl::CUDA_SPACE_ATOMIC_LOCKS_DEVICE[i] = 0;
  }
}

}  // namespace

namespace Impl {

int32_t* CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;

void init_lock_arrays_cuda() {
  if (CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;
  auto error_malloc = cudaMalloc(&CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h,
                                 sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  auto error_sync1 = cudaDeviceSynchronize();
  DESUL_IMPL_COPY_CUDA_LOCK_ARRAYS_TO_DEVICE();
  init_lock_arrays_cuda_kernel<<<(CUDA_SPACE_ATOMIC_MASK + 1 + 255) / 256, 256>>>();
  auto error_sync2 = cudaDeviceSynchronize();
}

void finalize_lock_arrays_cuda() {
  if (CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;
  cudaFree(CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h);
  CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
#ifdef __CUDACC_RDC__
  DESUL_IMPL_COPY_CUDA_LOCK_ARRAYS_TO_DEVICE();
#endif
}

}  // namespace Impl

}  // namespace desul
#endif
