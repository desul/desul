/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <DESUL_Atomics_Lock_Array.hpp>
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
