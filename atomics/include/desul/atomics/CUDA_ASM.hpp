
// ************************************************************************
//
//                        Kokkos v. 3.0
//              Copyright (2019) Sandia Corporation
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

#if defined(__CUDA_ARCH__) || (defined(__clang__) && !defined(__NVCC__))

#include <cassert>


namespace desul {
namespace Impl {

inline __device__ int __stronger_order_simt_(int a, int b) {
  if (b == __ATOMIC_SEQ_CST) return __ATOMIC_SEQ_CST;
  if (b == __ATOMIC_RELAXED) return a;
  switch (a) {
    case __ATOMIC_SEQ_CST:
    case __ATOMIC_ACQ_REL: return a;
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE:
      if (b != __ATOMIC_ACQUIRE)
        return __ATOMIC_ACQ_REL;
      else
        return __ATOMIC_ACQUIRE;
    case __ATOMIC_RELEASE:
      if (b != __ATOMIC_RELEASE)
        return __ATOMIC_ACQ_REL;
      else
        return __ATOMIC_RELEASE;
    case __ATOMIC_RELAXED: return b;
    default: assert(0);
  }
  return __ATOMIC_SEQ_CST;
}

inline constexpr __device__ bool __atomic_always_lock_free_simt_(size_t size,
                                                                 void *) {
  return size <= 8;
}
inline __device__ bool __atomic_is_lock_free_simt_(size_t size, void *ptr) {
  return __atomic_always_lock_free_simt_(size, ptr);
}

inline bool __device__ __atomic_always_lock_free_simt(size_t size, void *ptr) {
  return __atomic_always_lock_free_simt_(size, const_cast<void *>(ptr));
}
inline bool __device__ __atomic_is_lock_free_simt(size_t size, void *ptr) {
  return __atomic_is_lock_free_simt_(size, const_cast<void *>(ptr));
}


// Include CUDA ptx asm based atomics for scope system
#define  DESUL_IMPL_CUDA_MEMORY_SCOPE MemoryScopeNode
#define  DESUL_IMPL_CUDA_PTX_SCOPE "sys"
#include "desul/atomics/CUDA_ASM.inc"
#include "desul/atomics/CUDA_ASM_undefs.inc"
#undef   DESUL_IMPL_CUDA_PTX_SCOPE
#undef   DESUL_IMPL_CUDA_MEMORY_SCOPE

// Include CUDA ptx asm based atomics for scope gpu
#define  DESUL_IMPL_CUDA_MEMORY_SCOPE MemoryScopeDevice
#define  DESUL_IMPL_CUDA_PTX_SCOPE "gpu"
#include "desul/atomics/CUDA_ASM.inc"
#include "desul/atomics/CUDA_ASM_undefs.inc"
#undef   DESUL_IMPL_CUDA_PTX_SCOPE
#undef   DESUL_IMPL_CUDA_MEMORY_SCOPE

// Include CUDA ptx asm based atomics for scope system
#define  DESUL_IMPL_CUDA_MEMORY_SCOPE MemoryScopeCore
#define  DESUL_IMPL_CUDA_PTX_SCOPE "cta"
#include "desul/atomics/CUDA_ASM.inc"
#include "desul/atomics/CUDA_ASM_undefs.inc"
#undef   DESUL_IMPL_CUDA_PTX_SCOPE
#undef   DESUL_IMPL_CUDA_MEMORY_SCOPE
} // namespace Impl
} // namespace desul

#endif //CUDA_ARCH

