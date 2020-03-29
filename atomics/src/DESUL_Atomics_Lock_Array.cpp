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

namespace desul {
namespace Impl {
const uint32_t HOST_SPACE_ATOMIC_MASK = 0xFFFF;
const uint32_t HOST_SPACE_ATOMIC_XOR_MASK = 0x5A39;
static int32_t HOST_SPACE_ATOMIC_LOCKS_DEVICE[HOST_SPACE_ATOMIC_MASK + 1];
}  // namespace Impl

namespace Impl {
void init_lock_arrays() {
  static int is_initialized = 0;
  if (!is_initialized)
    for (int i = 0; i < static_cast<int>(HOST_SPACE_ATOMIC_MASK + 1); i++)
      HOST_SPACE_ATOMIC_LOCKS_DEVICE[i] = 0;
#ifdef DESUL_HAVE_CUDA_ATOMICS
  init_lock_arrays_cuda();
#endif
}

void finalize_lock_arrays() {
#ifdef DESUL_HAVE_CUDA_ATOMICS
  finalize_lock_arrays_cuda();
#endif
}

bool lock_address(void* ptr, MemoryScopeNode) {
  return 0 == atomic_compare_exchange(
                  &HOST_SPACE_ATOMIC_LOCKS_DEVICE[((uint64_t(ptr) >> 2) &
                                                   HOST_SPACE_ATOMIC_MASK) ^
                                                  HOST_SPACE_ATOMIC_XOR_MASK],
                  0,
                  1,
                  MemoryOrderSeqCst(),
                  MemoryScopeDevice());
}
void unlock_address(void* ptr, MemoryScopeNode) {
  (void)atomic_compare_exchange(
      &HOST_SPACE_ATOMIC_LOCKS_DEVICE[((uint64_t(ptr) >> 2) & HOST_SPACE_ATOMIC_MASK) ^
                                      HOST_SPACE_ATOMIC_XOR_MASK],
      int32_t(1),
      int32_t(0),
      MemoryOrderSeqCst(),
      MemoryScopeDevice());
}
bool lock_address(void* ptr, MemoryScopeDevice) {
  return 0 == atomic_compare_exchange(
                  &HOST_SPACE_ATOMIC_LOCKS_DEVICE[((uint64_t(ptr) >> 2) &
                                                   HOST_SPACE_ATOMIC_MASK) ^
                                                  HOST_SPACE_ATOMIC_XOR_MASK],
                  0,
                  1,
                  MemoryOrderSeqCst(),
                  MemoryScopeDevice());
}
void unlock_address(void* ptr, MemoryScopeDevice) {
  (void)atomic_compare_exchange(
      &HOST_SPACE_ATOMIC_LOCKS_DEVICE[((uint64_t(ptr) >> 2) & HOST_SPACE_ATOMIC_MASK) ^
                                      HOST_SPACE_ATOMIC_XOR_MASK],
      1,
      0,
      MemoryOrderSeqCst(),
      MemoryScopeDevice());
}
bool lock_address(void* ptr, MemoryScopeCore) {
  return 0 == atomic_compare_exchange(
                  &HOST_SPACE_ATOMIC_LOCKS_DEVICE[((uint64_t(ptr) >> 2) &
                                                   HOST_SPACE_ATOMIC_MASK) ^
                                                  HOST_SPACE_ATOMIC_XOR_MASK],
                  0,
                  1,
                  MemoryOrderSeqCst(),
                  MemoryScopeDevice());
}
void unlock_address(void* ptr, MemoryScopeCore) {
  (void)atomic_compare_exchange(
      &HOST_SPACE_ATOMIC_LOCKS_DEVICE[((uint64_t(ptr) >> 2) & HOST_SPACE_ATOMIC_MASK) ^
                                      HOST_SPACE_ATOMIC_XOR_MASK],
      1,
      0,
      MemoryOrderSeqCst(),
      MemoryScopeDevice());
}
}  // namespace Impl
}  // namespace desul
