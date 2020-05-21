/*
@HEADER
================================================================================

ORIGINAL LICENSE
----------------

Copyright (c) 2018, NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

================================================================================

LICENSE ASSOCIATED WITH SUBSEQUENT MODIFICATIONS
------------------------------------------------

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
@HEADER
*/

#define __simt_fence_signal_() asm volatile("" ::: "memory")
#define __simt_fence_sc_() \
  asm volatile("fence.sc." DESUL_IMPL_CUDA_PTX_SCOPE ";" ::: "memory")
#define __simt_fence_() asm volatile("fence." DESUL_IMPL_CUDA_PTX_SCOPE ";" ::: "memory")

#define __simt_load_acquire_8_as_32(ptr, ret)             \
  asm volatile("ld.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b8 %0, [%1];" \
               : "=r"(ret)                                \
               : "l"(ptr)                                 \
               : "memory")
#define __simt_load_relaxed_8_as_32(ptr, ret)             \
  asm volatile("ld.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b8 %0, [%1];" \
               : "=r"(ret)                                \
               : "l"(ptr)                                 \
               : "memory")
#define __simt_store_release_8_as_32(ptr, desired)                    \
  asm volatile("st.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b8 [%0], %1;" ::"l"(ptr), \
               "r"(desired)                                           \
               : "memory")
#define __simt_store_relaxed_8_as_32(ptr, desired)                    \
  asm volatile("st.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b8 [%0], %1;" ::"l"(ptr), \
               "r"(desired)                                           \
               : "memory")

#define __simt_load_acquire_16(ptr, ret)                   \
  asm volatile("ld.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b16 %0, [%1];" \
               : "=h"(ret)                                 \
               : "l"(ptr)                                  \
               : "memory")
#define __simt_load_relaxed_16(ptr, ret)                   \
  asm volatile("ld.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b16 %0, [%1];" \
               : "=h"(ret)                                 \
               : "l"(ptr)                                  \
               : "memory")
#define __simt_store_release_16(ptr, desired)                          \
  asm volatile("st.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b16 [%0], %1;" ::"l"(ptr), \
               "h"(desired)                                            \
               : "memory")
#define __simt_store_relaxed_16(ptr, desired)                          \
  asm volatile("st.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b16 [%0], %1;" ::"l"(ptr), \
               "h"(desired)                                            \
               : "memory")

#define __simt_load_acquire_32(ptr, ret)                   \
  asm volatile("ld.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1];" \
               : "=r"(ret)                                 \
               : "l"(ptr)                                  \
               : "memory")
#define __simt_load_relaxed_32(ptr, ret)                   \
  asm volatile("ld.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1];" \
               : "=r"(ret)                                 \
               : "l"(ptr)                                  \
               : "memory")
#define __simt_store_release_32(ptr, desired)                          \
  asm volatile("st.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 [%0], %1;" ::"l"(ptr), \
               "r"(desired)                                            \
               : "memory")
#define __simt_store_relaxed_32(ptr, desired)                          \
  asm volatile("st.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 [%0], %1;" ::"l"(ptr), \
               "r"(desired)                                            \
               : "memory")
#define __simt_exch_release_32(ptr, old, desired)                     \
  asm volatile("atom.exch.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                            \
               : "l"(ptr), "r"(desired)                               \
               : "memory")
#define __simt_exch_acquire_32(ptr, old, desired)                     \
  asm volatile("atom.exch.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                            \
               : "l"(ptr), "r"(desired)                               \
               : "memory")
#define __simt_exch_acq_rel_32(ptr, old, desired)                     \
  asm volatile("atom.exch.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                            \
               : "l"(ptr), "r"(desired)                               \
               : "memory")
#define __simt_exch_relaxed_32(ptr, old, desired)                     \
  asm volatile("atom.exch.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                            \
               : "l"(ptr), "r"(desired)                               \
               : "memory")
#define __simt_cas_release_32(ptr, old, expected, desired)               \
  asm volatile("atom.cas.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2, %3;" \
               : "=r"(old)                                               \
               : "l"(ptr), "r"(expected), "r"(desired)                   \
               : "memory")
#define __simt_cas_acquire_32(ptr, old, expected, desired)               \
  asm volatile("atom.cas.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2, %3;" \
               : "=r"(old)                                               \
               : "l"(ptr), "r"(expected), "r"(desired)                   \
               : "memory")
#define __simt_cas_acq_rel_32(ptr, old, expected, desired)               \
  asm volatile("atom.cas.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2, %3;" \
               : "=r"(old)                                               \
               : "l"(ptr), "r"(expected), "r"(desired)                   \
               : "memory")
#define __simt_cas_relaxed_32(ptr, old, expected, desired)               \
  asm volatile("atom.cas.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2, %3;" \
               : "=r"(old)                                               \
               : "l"(ptr), "r"(expected), "r"(desired)                   \
               : "memory")
#define __simt_add_release_32(ptr, old, addend)                      \
  asm volatile("atom.add.release." DESUL_IMPL_CUDA_PTX_SCOPE ".u32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(addend)                               \
               : "memory")
#define __simt_add_acquire_32(ptr, old, addend)                      \
  asm volatile("atom.add.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".u32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(addend)                               \
               : "memory")
#define __simt_add_acq_rel_32(ptr, old, addend)                      \
  asm volatile("atom.add.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".u32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(addend)                               \
               : "memory")
#define __simt_add_relaxed_32(ptr, old, addend)                      \
  asm volatile("atom.add.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".u32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(addend)                               \
               : "memory")
#define __simt_and_release_32(ptr, old, andend)                      \
  asm volatile("atom.and.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(andend)                               \
               : "memory")
#define __simt_and_acquire_32(ptr, old, andend)                      \
  asm volatile("atom.and.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(andend)                               \
               : "memory")
#define __simt_and_acq_rel_32(ptr, old, andend)                      \
  asm volatile("atom.and.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(andend)                               \
               : "memory")
#define __simt_and_relaxed_32(ptr, old, andend)                      \
  asm volatile("atom.and.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(andend)                               \
               : "memory")
#define __simt_or_release_32(ptr, old, orend)                       \
  asm volatile("atom.or.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                          \
               : "l"(ptr), "r"(orend)                               \
               : "memory")
#define __simt_or_acquire_32(ptr, old, orend)                       \
  asm volatile("atom.or.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                          \
               : "l"(ptr), "r"(orend)                               \
               : "memory")
#define __simt_or_acq_rel_32(ptr, old, orend)                       \
  asm volatile("atom.or.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                          \
               : "l"(ptr), "r"(orend)                               \
               : "memory")
#define __simt_or_relaxed_32(ptr, old, orend)                       \
  asm volatile("atom.or.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                          \
               : "l"(ptr), "r"(orend)                               \
               : "memory")
#define __simt_xor_release_32(ptr, old, xorend)                      \
  asm volatile("atom.xor.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(xorend)                               \
               : "memory")
#define __simt_xor_acquire_32(ptr, old, xorend)                      \
  asm volatile("atom.xor.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(xorend)                               \
               : "memory")
#define __simt_xor_acq_rel_32(ptr, old, xorend)                      \
  asm volatile("atom.xor.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(xorend)                               \
               : "memory")
#define __simt_xor_relaxed_32(ptr, old, xorend)                      \
  asm volatile("atom.xor.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b32 %0, [%1], %2;" \
               : "=r"(old)                                           \
               : "l"(ptr), "r"(xorend)                               \
               : "memory")

#define __simt_load_acquire_64(ptr, ret)                   \
  asm volatile("ld.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1];" \
               : "=l"(ret)                                 \
               : "l"(ptr)                                  \
               : "memory")
#define __simt_load_relaxed_64(ptr, ret)                   \
  asm volatile("ld.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1];" \
               : "=l"(ret)                                 \
               : "l"(ptr)                                  \
               : "memory")
#define __simt_store_release_64(ptr, desired)                          \
  asm volatile("st.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 [%0], %1;" ::"l"(ptr), \
               "l"(desired)                                            \
               : "memory")
#define __simt_store_relaxed_64(ptr, desired)                          \
  asm volatile("st.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 [%0], %1;" ::"l"(ptr), \
               "l"(desired)                                            \
               : "memory")
#define __simt_exch_release_64(ptr, old, desired)                     \
  asm volatile("atom.exch.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                            \
               : "l"(ptr), "l"(desired)                               \
               : "memory")
#define __simt_exch_acquire_64(ptr, old, desired)                     \
  asm volatile("atom.exch.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                            \
               : "l"(ptr), "l"(desired)                               \
               : "memory")
#define __simt_exch_acq_rel_64(ptr, old, desired)                     \
  asm volatile("atom.exch.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                            \
               : "l"(ptr), "l"(desired)                               \
               : "memory")
#define __simt_exch_relaxed_64(ptr, old, desired)                     \
  asm volatile("atom.exch.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                            \
               : "l"(ptr), "l"(desired)                               \
               : "memory")
#define __simt_cas_release_64(ptr, old, expected, desired)               \
  asm volatile("atom.cas.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2, %3;" \
               : "=l"(old)                                               \
               : "l"(ptr), "l"(expected), "l"(desired)                   \
               : "memory")
#define __simt_cas_acquire_64(ptr, old, expected, desired)               \
  asm volatile("atom.cas.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2, %3;" \
               : "=l"(old)                                               \
               : "l"(ptr), "l"(expected), "l"(desired)                   \
               : "memory")
#define __simt_cas_acq_rel_64(ptr, old, expected, desired)               \
  asm volatile("atom.cas.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2, %3;" \
               : "=l"(old)                                               \
               : "l"(ptr), "l"(expected), "l"(desired)                   \
               : "memory")
#define __simt_cas_relaxed_64(ptr, old, expected, desired)               \
  asm volatile("atom.cas.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2, %3;" \
               : "=l"(old)                                               \
               : "l"(ptr), "l"(expected), "l"(desired)                   \
               : "memory")
#define __simt_add_release_64(ptr, old, addend)                      \
  asm volatile("atom.add.release." DESUL_IMPL_CUDA_PTX_SCOPE ".u64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(addend)                               \
               : "memory")
#define __simt_add_acquire_64(ptr, old, addend)                      \
  asm volatile("atom.add.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".u64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(addend)                               \
               : "memory")
#define __simt_add_acq_rel_64(ptr, old, addend)                      \
  asm volatile("atom.add.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".u64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(addend)                               \
               : "memory")
#define __simt_add_relaxed_64(ptr, old, addend)                      \
  asm volatile("atom.add.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".u64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(addend)                               \
               : "memory")
#define __simt_and_release_64(ptr, old, andend)                      \
  asm volatile("atom.and.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(andend)                               \
               : "memory")
#define __simt_and_acquire_64(ptr, old, andend)                      \
  asm volatile("atom.and.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(andend)                               \
               : "memory")
#define __simt_and_acq_rel_64(ptr, old, andend)                      \
  asm volatile("atom.and.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(andend)                               \
               : "memory")
#define __simt_and_relaxed_64(ptr, old, andend)                      \
  asm volatile("atom.and.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(andend)                               \
               : "memory")
#define __simt_or_release_64(ptr, old, orend)                       \
  asm volatile("atom.or.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                          \
               : "l"(ptr), "l"(orend)                               \
               : "memory")
#define __simt_or_acquire_64(ptr, old, orend)                       \
  asm volatile("atom.or.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                          \
               : "l"(ptr), "l"(orend)                               \
               : "memory")
#define __simt_or_acq_rel_64(ptr, old, orend)                       \
  asm volatile("atom.or.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                          \
               : "l"(ptr), "l"(orend)                               \
               : "memory")
#define __simt_or_relaxed_64(ptr, old, orend)                       \
  asm volatile("atom.or.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                          \
               : "l"(ptr), "l"(orend)                               \
               : "memory")
#define __simt_xor_release_64(ptr, old, xorend)                      \
  asm volatile("atom.xor.release." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(xorend)                               \
               : "memory")
#define __simt_xor_acquire_64(ptr, old, xorend)                      \
  asm volatile("atom.xor.acquire." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(xorend)                               \
               : "memory")
#define __simt_xor_acq_rel_64(ptr, old, xorend)                      \
  asm volatile("atom.xor.acq_rel." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(xorend)                               \
               : "memory")
#define __simt_xor_relaxed_64(ptr, old, xorend)                      \
  asm volatile("atom.xor.relaxed." DESUL_IMPL_CUDA_PTX_SCOPE ".b64 %0, [%1], %2;" \
               : "=l"(old)                                           \
               : "l"(ptr), "l"(xorend)                               \
               : "memory")

#define __simt_nanosleep(timeout) \
  asm volatile("nanosleep.u32 %0;" ::"r"(unsigned(timeout)) :)

/*
    base
*/

#define DO__atomic_load_simt_(bytes, bits)                                 \
  template <class type,                                                    \
            typename std::enable_if<sizeof(type) == bytes, int>::type = 0> \
  void __device__ __atomic_load_simt_(const type *ptr, type *ret,          \
                                      int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {                      \
    int##bits##_t tmp = 0;                                                 \
    switch (memorder) {                                                    \
      case __ATOMIC_SEQ_CST: __simt_fence_sc_();                           \
      case __ATOMIC_CONSUME:                                               \
      case __ATOMIC_ACQUIRE: __simt_load_acquire_##bits(ptr, tmp); break;  \
      case __ATOMIC_RELAXED: __simt_load_relaxed_##bits(ptr, tmp); break;  \
      default: assert(0);                                                  \
    }                                                                      \
    memcpy(ret, &tmp, bytes);                                              \
  }
DO__atomic_load_simt_(1, 32) 

DO__atomic_load_simt_(2, 16)

DO__atomic_load_simt_(4, 32) 

DO__atomic_load_simt_(8, 64)

template <class type>
type __device__ __atomic_load_n_simt_(const type *ptr, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  type ret;
  __atomic_load_simt_(ptr, &ret, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
  return ret;
}

#define DO__atomic_store_simt_(bytes, bits)                                  \
  template <class type,                                                      \
            typename std::enable_if<sizeof(type) == bytes, int>::type = 0>   \
  void __device__ __atomic_store_simt_(type *ptr, type *val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) { \
    int##bits##_t tmp = 0;                                                   \
    memcpy(&tmp, val, bytes);                                                \
    switch (memorder) {                                                      \
      case __ATOMIC_RELEASE: __simt_store_release_##bits(ptr, tmp); break;   \
      case __ATOMIC_SEQ_CST: __simt_fence_sc_();                             \
      case __ATOMIC_RELAXED: __simt_store_relaxed_##bits(ptr, tmp); break;   \
      default: assert(0);                                                    \
    }                                                                        \
  }
DO__atomic_store_simt_(1, 32) 
DO__atomic_store_simt_(2, 16)
DO__atomic_store_simt_(4, 32) 
DO__atomic_store_simt_(8, 64)

template <class type>
void __device__
  __atomic_store_n_simt_(type *ptr, type val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_store_simt_(ptr, &val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}

#define DO__atomic_compare_exchange_simt_(bytes, bits)                     \
  template <class type,                                                    \
            typename std::enable_if<sizeof(type) == bytes, int>::type = 0> \
  bool __device__ __atomic_compare_exchange_simt_(                         \
      type *ptr, type *expected, const type *desired, bool,                \
      int success_memorder, int failure_memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {                        \
    int##bits##_t tmp = 0, old = 0, old_tmp;                               \
    memcpy(&tmp, desired, bytes);                                          \
    memcpy(&old, expected, bytes);                                         \
    old_tmp = old;                                                         \
    switch (__stronger_order_simt_(success_memorder, failure_memorder)) {  \
      case __ATOMIC_SEQ_CST: __simt_fence_sc_();                           \
      case __ATOMIC_CONSUME:                                               \
      case __ATOMIC_ACQUIRE:                                               \
        __simt_cas_acquire_##bits(ptr, old, old_tmp, tmp);                 \
        break;                                                             \
      case __ATOMIC_ACQ_REL:                                               \
        __simt_cas_acq_rel_##bits(ptr, old, old_tmp, tmp);                 \
        break;                                                             \
      case __ATOMIC_RELEASE:                                               \
        __simt_cas_release_##bits(ptr, old, old_tmp, tmp);                 \
        break;                                                             \
      case __ATOMIC_RELAXED:                                               \
        __simt_cas_relaxed_##bits(ptr, old, old_tmp, tmp);                 \
        break;                                                             \
      default: assert(0);                                                  \
    }                                                                      \
    bool const ret = old == old_tmp;                                       \
    memcpy(expected, &old, bytes);                                         \
    return ret;                                                            \
  }

DO__atomic_compare_exchange_simt_(4, 32)
DO__atomic_compare_exchange_simt_(8, 64)

template <class type,
typename std::enable_if<sizeof(type) <= 2, int>::type = 0>
bool __device__
__atomic_compare_exchange_simt_(type *ptr, type *expected,
                                const type *desired, bool,
                                int success_memorder,
                                int failure_memorder,
                                DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  using R            = typename std::conditional<std::is_volatile<type>::value,
                                      volatile uint32_t, uint32_t>::type;
  auto const aligned = (R *)((intptr_t)ptr & ~(sizeof(uint32_t) - 1));
  auto const offset  = uint32_t((intptr_t)ptr & (sizeof(uint32_t) - 1)) * 8;
  auto const mask    = ((1 << sizeof(type) * 8) - 1) << offset;

  uint32_t old = *expected << offset, old_value;
  while (1) {
    old_value = (old & mask) >> offset;
    if (old_value != *expected) break;
    uint32_t const attempt = (old & ~mask) | (*desired << offset);
    if (__atomic_compare_exchange_simt_(aligned, &old, &attempt, true,
                                        success_memorder, failure_memorder, 
                                        DESUL_IMPL_CUDA_MEMORY_SCOPE()))
      return true;
  }
  *expected = old_value;
  return false;
}

template <class type>
bool __device__ __atomic_compare_exchange_n_simt_(type *ptr, type *expected,
                                                  type desired, bool weak,
                                                  int success_memorder,
                                                  int failure_memorder,
                                                  DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_compare_exchange_simt_(ptr, expected, &desired, weak,
                                         success_memorder, failure_memorder,
                                         DESUL_IMPL_CUDA_MEMORY_SCOPE());
}

#define DO__atomic_exchange_simt_(bytes, bits)                                 \
  template <class type,                                                        \
            typename std::enable_if<sizeof(type) == bytes, int>::type = 0>     \
  void __device__ __atomic_exchange_simt_(type *ptr, type *val, type *ret,     \
                                          int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {                      \
    int##bits##_t tmp = 0;                                                     \
    memcpy(&tmp, val, bytes);                                                  \
    switch (memorder) {                                                        \
      case __ATOMIC_SEQ_CST: __simt_fence_sc_();                               \
      case __ATOMIC_CONSUME:                                                   \
      case __ATOMIC_ACQUIRE: __simt_exch_acquire_##bits(ptr, tmp, tmp); break; \
      case __ATOMIC_ACQ_REL: __simt_exch_acq_rel_##bits(ptr, tmp, tmp); break; \
      case __ATOMIC_RELEASE: __simt_exch_release_##bits(ptr, tmp, tmp); break; \
      case __ATOMIC_RELAXED: __simt_exch_relaxed_##bits(ptr, tmp, tmp); break; \
      default: assert(0);                                                      \
    }                                                                          \
    memcpy(ret, &tmp, bytes);                                                  \
  }
DO__atomic_exchange_simt_(4, 32) DO__atomic_exchange_simt_(8, 64)

    template <class type,
              typename std::enable_if<sizeof(type) <= 2, int>::type = 0>
    void __device__
    __atomic_exchange_simt_(type *ptr, type *val, type *ret, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  type expected = __atomic_load_n_simt_(ptr, __ATOMIC_RELAXED);
  while (!__atomic_compare_exchange_simt_(ptr, &expected, val, true, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE(),
                                          memorder))
    ;
  *ret = expected;
}

template <class type>
type __device__ __atomic_exchange_n_simt_(type *ptr, type val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  type ret;
  __atomic_exchange_simt_(ptr, &val, &ret, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
  return ret;
}

#define DO__atomic_fetch_add_simt_(bytes, bits)                               \
  template <class type, class delta,                                          \
            typename std::enable_if<sizeof(type) == bytes, int>::type = 0>    \
  type __device__ __atomic_fetch_add_simt_(type *ptr, delta val,              \
                                           int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {                    \
    type ret;                                                                 \
    switch (memorder) {                                                       \
      case __ATOMIC_SEQ_CST: __simt_fence_sc_();                              \
      case __ATOMIC_CONSUME:                                                  \
      case __ATOMIC_ACQUIRE: __simt_add_acquire_##bits(ptr, ret, val); break; \
      case __ATOMIC_ACQ_REL: __simt_add_acq_rel_##bits(ptr, ret, val); break; \
      case __ATOMIC_RELEASE: __simt_add_release_##bits(ptr, ret, val); break; \
      case __ATOMIC_RELAXED: __simt_add_relaxed_##bits(ptr, ret, val); break; \
      default: assert(0);                                                     \
    }                                                                         \
    return ret;                                                               \
  }
DO__atomic_fetch_add_simt_(4, 32) 
DO__atomic_fetch_add_simt_(8, 64)

template <class type, class delta,
              typename std::enable_if<sizeof(type) <= 2, int>::type = 0>
    type __device__
    __atomic_fetch_add_simt_(type *ptr, delta val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  type expected      = __atomic_load_n_simt_(ptr, __ATOMIC_RELAXED);
  type const desired = expected + val;
  while (!__atomic_compare_exchange_simt_(ptr, &expected, &desired, true,
                                          memorder, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE()))
    ;
  return expected;
}

#define DO__atomic_fetch_sub_simt_(bytes, bits)                                \
  template <class type, class delta,                                           \
            typename std::enable_if<sizeof(type) == bytes, int>::type = 0>     \
  type __device__ __atomic_fetch_sub_simt_(type *ptr, delta val,               \
                                           int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {                     \
    type ret;                                                                  \
    switch (memorder) {                                                        \
      case __ATOMIC_SEQ_CST: __simt_fence_sc_();                               \
      case __ATOMIC_CONSUME:                                                   \
      case __ATOMIC_ACQUIRE: __simt_add_acquire_##bits(ptr, ret, -val); break; \
      case __ATOMIC_ACQ_REL: __simt_add_acq_rel_##bits(ptr, ret, -val); break; \
      case __ATOMIC_RELEASE: __simt_add_release_##bits(ptr, ret, -val); break; \
      case __ATOMIC_RELAXED: __simt_add_relaxed_##bits(ptr, ret, -val); break; \
      default: assert(0);                                                      \
    }                                                                          \
    return ret;                                                                \
  }
DO__atomic_fetch_sub_simt_(4, 32) 
DO__atomic_fetch_sub_simt_(8, 64)

    template <class type, class delta,
              typename std::enable_if<sizeof(type) <= 2, int>::type = 0>
    type __device__
    __atomic_fetch_sub_simt_(type *ptr, delta val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  type expected      = __atomic_load_n_simt_(ptr, __ATOMIC_RELAXED);
  type const desired = expected - val;
  while (!__atomic_compare_exchange_simt_(ptr, &expected, &desired, true,
                                          memorder, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE()))
    ;
  return expected;
}

#define DO__atomic_fetch_and_simt_(bytes, bits)                               \
  template <class type,                                                       \
            typename std::enable_if<sizeof(type) == bytes, int>::type = 0>    \
  type __device__ __atomic_fetch_and_simt_(type *ptr, type val,               \
                                           int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {                    \
    type ret;                                                                 \
    switch (memorder) {                                                       \
      case __ATOMIC_SEQ_CST: __simt_fence_sc_();                              \
      case __ATOMIC_CONSUME:                                                  \
      case __ATOMIC_ACQUIRE: __simt_and_acquire_##bits(ptr, ret, val); break; \
      case __ATOMIC_ACQ_REL: __simt_and_acq_rel_##bits(ptr, ret, val); break; \
      case __ATOMIC_RELEASE: __simt_and_release_##bits(ptr, ret, val); break; \
      case __ATOMIC_RELAXED: __simt_and_relaxed_##bits(ptr, ret, val); break; \
      default: assert(0);                                                     \
    }                                                                         \
    return ret;                                                               \
  }
DO__atomic_fetch_and_simt_(4, 32) DO__atomic_fetch_and_simt_(8, 64)

    template <class type, class delta,
              typename std::enable_if<sizeof(type) <= 2, int>::type = 0>
    type __device__
    __atomic_fetch_and_simt_(type *ptr, delta val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  type expected      = __atomic_load_n_simt_(ptr, __ATOMIC_RELAXED);
  type const desired = expected & val;
  while (!__atomic_compare_exchange_simt_(ptr, &expected, &desired, true,
                                          memorder, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE()))
    ;
  return expected;
}

#define DO__atomic_fetch_xor_simt_(bytes, bits)                               \
  template <class type,                                                       \
            typename std::enable_if<sizeof(type) == bytes, int>::type = 0>    \
  type __device__ __atomic_fetch_xor_simt_(type *ptr, type val,               \
                                           int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {                    \
    type ret;                                                                 \
    switch (memorder) {                                                       \
      case __ATOMIC_SEQ_CST: __simt_fence_sc_();                              \
      case __ATOMIC_CONSUME:                                                  \
      case __ATOMIC_ACQUIRE: __simt_xor_acquire_##bits(ptr, ret, val); break; \
      case __ATOMIC_ACQ_REL: __simt_xor_acq_rel_##bits(ptr, ret, val); break; \
      case __ATOMIC_RELEASE: __simt_xor_release_##bits(ptr, ret, val); break; \
      case __ATOMIC_RELAXED: __simt_xor_relaxed_##bits(ptr, ret, val); break; \
      default: assert(0);                                                     \
    }                                                                         \
    return ret;                                                               \
  }
DO__atomic_fetch_xor_simt_(4, 32) 

DO__atomic_fetch_xor_simt_(8, 64)

    template <class type, class delta,
              typename std::enable_if<sizeof(type) <= 2, int>::type = 0>
    type __device__
    __atomic_fetch_xor_simt_(type *ptr, delta val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  type expected      = __atomic_load_n_simt_(ptr, __ATOMIC_RELAXED);
  type const desired = expected ^ val;
  while (!__atomic_compare_exchange_simt_(ptr, &expected, &desired, true,
                                          memorder, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE()))
    ;
  return expected;
}

#define DO__atomic_fetch_or_simt_(bytes, bits)                                 \
  template <class type,                                                        \
            typename std::enable_if<sizeof(type) == bytes, int>::type = 0>     \
  type __device__ __atomic_fetch_or_simt_(type *ptr, type val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) { \
    type ret;                                                                  \
    switch (memorder) {                                                        \
      case __ATOMIC_SEQ_CST: __simt_fence_sc_();                               \
      case __ATOMIC_CONSUME:                                                   \
      case __ATOMIC_ACQUIRE: __simt_or_acquire_##bits(ptr, ret, val); break;   \
      case __ATOMIC_ACQ_REL: __simt_or_acq_rel_##bits(ptr, ret, val); break;   \
      case __ATOMIC_RELEASE: __simt_or_release_##bits(ptr, ret, val); break;   \
      case __ATOMIC_RELAXED: __simt_or_relaxed_##bits(ptr, ret, val); break;   \
      default: assert(0);                                                      \
    }                                                                          \
    return ret;                                                                \
  }
DO__atomic_fetch_or_simt_(4, 32) 

DO__atomic_fetch_or_simt_(8, 64)

    template <class type, class delta,
              typename std::enable_if<sizeof(type) <= 2, int>::type = 0>
    type __device__
    __atomic_fetch_or_simt_(type *ptr, delta val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  type expected      = __atomic_load_n_simt_(ptr, __ATOMIC_RELAXED);
  type const desired = expected | val;
  while (!__atomic_compare_exchange_simt_(ptr, &expected, &desired, true,
                                          memorder, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE()))
    ;
  return expected;
}

template <class type>
inline bool __device__ __atomic_test_and_set_simt_(type *ptr, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_exchange_n_simt_((char *)ptr, (char)1, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE()) == 1;
}
template <class type>
inline void __device__ __atomic_clear_simt_(type *ptr, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_store_n_simt_((char *)ptr, (char)0, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}

/*
    fences
*/

inline void __device__ __atomic_thread_fence_simt(int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  switch (memorder) {
    case __ATOMIC_SEQ_CST: __simt_fence_sc_(); break;
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE:
    case __ATOMIC_ACQ_REL:
    case __ATOMIC_RELEASE: __simt_fence_(); break;
    case __ATOMIC_RELAXED: break;
    default: assert(0);
  }
}
inline void __device__ __atomic_signal_fence_simt(int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_thread_fence_simt(memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}

/*
    non-volatile
*/

template <class type>
type __device__ __atomic_load_n_simt(const type *ptr, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_load_n_simt_(const_cast<const type *>(ptr), memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_load_simt(const type *ptr, type *ret, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_load_simt_(const_cast<const type *>(ptr), ret, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_store_n_simt(type *ptr, type val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_store_n_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_store_simt(type *ptr, type *val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_store_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
type __device__ __atomic_exchange_n_simt(type *ptr, type val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_exchange_n_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_exchange_simt(type *ptr, type *val, type *ret,
                                       int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_exchange_simt_(const_cast<type *>(ptr), val, ret, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
bool __device__ __atomic_compare_exchange_n_simt(type *ptr, type *expected,
                                                 type desired, bool weak,
                                                 int success_memorder,
                                                 int failure_memorder,
                                                 DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_compare_exchange_n_simt_(const_cast<type *>(ptr), expected,
                                           desired, weak, success_memorder,
                                           failure_memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
bool __device__ __atomic_compare_exchange_simt(type *ptr, type *expected,
                                               type *desired, bool weak,
                                               int success_memorder,
                                               int failure_memorder,
                                               DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_compare_exchange_simt_(const_cast<type *>(ptr), expected,
                                         desired, weak, success_memorder,
                                         failure_memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type, class delta>
type __device__ __atomic_fetch_add_simt(type *ptr, delta val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_add_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type, class delta>
type __device__ __atomic_fetch_sub_simt(type *ptr, delta val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_sub_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
type __device__ __atomic_fetch_and_simt(type *ptr, type val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_and_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
type __device__ __atomic_fetch_xor_simt(type *ptr, type val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_xor_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
type __device__ __atomic_fetch_or_simt(type *ptr, type val, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_or_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
bool __device__ __atomic_test_and_set_simt(void *ptr, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_test_and_set_simt_(const_cast<void *>(ptr), memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_clear_simt(void *ptr, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_clear_simt_(const_cast<void *>(ptr), memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
/*
    volatile
*/

template <class type>
type __device__ __atomic_load_n_simt(const volatile type *ptr, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_load_n_simt_(const_cast<const type *>(ptr), memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_load_simt(const volatile type *ptr, type *ret,
                                   int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_load_simt_(const_cast<const type *>(ptr), ret, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_store_n_simt(volatile type *ptr, type val,
                                      int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_store_n_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_store_simt(volatile type *ptr, type *val,
                                    int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_store_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
type __device__ __atomic_exchange_n_simt(volatile type *ptr, type val,
                                         int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_exchange_n_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_exchange_simt(volatile type *ptr, type *val, type *ret,
                                       int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  __atomic_exchange_simt_(const_cast<type *>(ptr), val, ret, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
bool __device__ __atomic_compare_exchange_n_simt(volatile type *ptr,
                                                 type *expected, type desired,
                                                 bool weak,
                                                 int success_memorder,
                                                 int failure_memorder,
                                                 DESUL_IMPL_CUDA_MEMORY_SCOPE()) {
  return __atomic_compare_exchange_n_simt_(const_cast<type *>(ptr), expected,
                                           desired, weak, success_memorder,
                                           failure_memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE);
}
template <class type>
bool __device__ __atomic_compare_exchange_simt(volatile type *ptr,
                                               type *expected, type *desired,
                                               bool weak, int success_memorder,
                                               int failure_memorder,
                                               DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_compare_exchange_simt_(const_cast<type *>(ptr), expected,
                                         desired, weak, success_memorder,
                                         failure_memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type, class delta>
type __device__ __atomic_fetch_add_simt(volatile type *ptr, delta val,
                                        int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_add_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type, class delta>
type __device__ __atomic_fetch_sub_simt(volatile type *ptr, delta val,
                                        int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_sub_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
type __device__ __atomic_fetch_and_simt(volatile type *ptr, type val,
                                        int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_and_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
type __device__ __atomic_fetch_xor_simt(volatile type *ptr, type val,
                                        int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_xor_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
type __device__ __atomic_fetch_or_simt(volatile type *ptr, type val,
                                       int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_fetch_or_simt_(const_cast<type *>(ptr), val, memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
bool __device__ __atomic_test_and_set_simt(volatile void *ptr, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_test_and_set_simt_(const_cast<void *>(ptr), memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}
template <class type>
void __device__ __atomic_clear_simt(volatile void *ptr, int memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE) {
  return __atomic_clear_simt_(const_cast<void *>(ptr), memorder, DESUL_IMPL_CUDA_MEMORY_SCOPE());
}

