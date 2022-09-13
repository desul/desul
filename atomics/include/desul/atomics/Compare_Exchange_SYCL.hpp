/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_SYCL_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_SYCL_HPP_

#include <desul/atomics/Adapt_SYCL.hpp>
#include <desul/atomics/Common.hpp>
#include <desul/atomics/Thread_Fence_SYCL.hpp>

// FIXME_SYCL SYCL2020 dictates that <sycl/sycl.hpp> is the header to include
// but icpx 2022.1.0 and earlier versions only provide <CL/sycl.hpp>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace desul {
namespace Impl {

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 4, T> host_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  sycl_atomic_ref<unsigned int, MemoryOrder, MemoryScope> dest_ref(
      *reinterpret_cast<unsigned int*>(dest));
  dest_ref.compare_exchange_strong(*reinterpret_cast<unsigned int*>(&compare),
                                   *reinterpret_cast<unsigned int*>(&value));
  return compare;
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 8, T> host_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long is 64-bit");
  sycl_atomic_ref<unsigned long long int, MemoryOrder, MemoryScope> dest_ref(
      *reinterpret_cast<unsigned long long int*>(dest));
  dest_ref.compare_exchange_strong(*reinterpret_cast<unsigned long long int*>(&compare),
                                   *reinterpret_cast<unsigned long long int*>(&value));
  return compare;
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 4, T> host_atomic_exchange(T* const dest,
                                                         T value,
                                                         MemoryOrder,
                                                         MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  sycl_atomic_ref<unsigned int, MemoryOrder, MemoryScope> dest_ref(
      *reinterpret_cast<unsigned int*>(dest));
  unsigned int return_val = dest_ref.exchange(*reinterpret_cast<unsigned int*>(&value));
  return reinterpret_cast<T&>(return_val);
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 8, T> host_atomic_exchange(T* const dest,
                                                         T value,
                                                         MemoryOrder,
                                                         MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long is 64-bit");
  sycl_atomic_ref<unsigned long long int, MemoryOrder, MemoryScope> dest_ref(
      *reinterpret_cast<unsigned long long int*>(dest));
  unsigned long long int return_val =
      dest_ref.exchange(reinterpret_cast<unsigned long long int&>(value));
  return reinterpret_cast<T&>(return_val);
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<(sizeof(T) != 8) && (sizeof(T) != 4), T> host_atomic_compare_exchange(
    T* const /*dest*/, T compare, T /*value*/, MemoryOrder, MemoryScope) {
  assert(false);  // FIXME_SYCL not implemented
  return compare;
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<(sizeof(T) != 8) && (sizeof(T) != 4), T> host_atomic_exchange(
    T* const /*dest*/, T value, MemoryOrder, MemoryScope) {
  assert(false);  // FIXME_SYCL not implemented
  return value;
}

}  // namespace Impl
}  // namespace desul

#endif
