/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#include <atomics/Lock_Array.hpp>
#include <cinttypes>

namespace desul {
namespace Impl {
const uint32_t HOST_SPACE_ATOMIC_MASK = 0xFFFF;
const uint32_t HOST_SPACE_ATOMIC_XOR_MASK = 0x5A39;
static int32_t HOST_SPACE_ATOMIC_LOCKS_DEVICE[HOST_SPACE_ATOMIC_MASK + 1];
inline static int32_t* get_host_lock_(void* ptr) {
  return &HOST_SPACE_ATOMIC_LOCKS_DEVICE[((uint64_t(ptr) >> 2) &
                                          HOST_SPACE_ATOMIC_MASK) ^
                                         HOST_SPACE_ATOMIC_XOR_MASK];
}
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

template <typename MS>
bool lock_address(void* ptr, MS) {
  return 0 == atomic_compare_exchange(
                  Impl::get_host_lock_(ptr), 0, 1, MemoryOrderSeqCst(), MS());
}
template <typename MS>
void unlock_address(void* ptr, MS) {
  (void)atomic_compare_exchange(
      Impl::get_host_lock_(ptr), int32_t(1), int32_t(0), MemoryOrderSeqCst(), MS());
}
}  // namespace Impl
}  // namespace desul
