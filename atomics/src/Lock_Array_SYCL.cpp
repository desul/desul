/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

// FIXME_SYCL Use SYCL_EXT_ONEAPI_DEVICE_GLOBAL when available instead
#ifdef DESUL_SYCL_DEVICE_GLOBAL_SUPPORTED

#include <cinttypes>
#include <desul/atomics/Lock_Array_SYCL.hpp>
#include <sstream>
#include <string>

namespace desul {
namespace Impl {
SYCL_EXTERNAL
sycl::ext::oneapi::experimental::device_global<
    int32_t*,
    decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::oneapi::experimental::device_image_scope))>
    SYCL_SPACE_ATOMIC_LOCKS_DEVICE;
SYCL_EXTERNAL
sycl::ext::oneapi::experimental::device_global<
    int32_t*,
    decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::oneapi::experimental::device_image_scope))>
    SYCL_SPACE_ATOMIC_LOCKS_NODE;
}  // namespace Impl
}  // namespace desul

namespace desul {
namespace Impl {

int32_t* SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
int32_t* SYCL_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;

template <typename T>
void init_lock_arrays_sycl() {
  if (SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;

  sycl::queue q;

  SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h =
      sycl::malloc_device<int32_t>(SYCL_SPACE_ATOMIC_MASK + 1, q);
  SYCL_SPACE_ATOMIC_LOCKS_NODE_h =
      sycl::malloc_host<int32_t>(SYCL_SPACE_ATOMIC_MASK + 1, q);

  q.memcpy(SYCL_SPACE_ATOMIC_LOCKS_DEVICE,
           &SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h,
           sizeof(int32_t*));
  q.memcpy(
      SYCL_SPACE_ATOMIC_LOCKS_NODE, &SYCL_SPACE_ATOMIC_LOCKS_NODE_h, sizeof(int32_t*));

  q.memset(SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h,
           0,
           sizeof(int32_t) * (SYCL_SPACE_ATOMIC_MASK + 1));
  q.memset(SYCL_SPACE_ATOMIC_LOCKS_NODE_h,
           0,
           sizeof(int32_t) * (SYCL_SPACE_ATOMIC_MASK + 1));

  q.wait_and_throw();
}

template <typename T>
void finalize_lock_arrays_sycl() {
  if (SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;

  sycl::queue q;
  sycl::free(SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h, q);
  sycl::free(SYCL_SPACE_ATOMIC_LOCKS_NODE_h, q);
  SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
  SYCL_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;
}

template void init_lock_arrays_sycl<int>();
template void finalize_lock_arrays_sycl<int>();

}  // namespace Impl
}  // namespace desul
#endif
