/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_SYCL_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_SYCL_HPP_
#include <CL/sycl.hpp>
#include "desul/atomics/Common.hpp"
#include "desul/atomics/SYCLConversions.hpp"

#ifdef DESUL_HAVE_SYCL_ATOMICS

namespace desul {

template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if_t<sizeof(T) == 4, T> atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  sycl_atomic_ref<unsigned int, MemoryScope> dest_ref(
      *reinterpret_cast<unsigned int*>(dest));
  dest_ref.compare_exchange_strong(*reinterpret_cast<unsigned int*>(&compare),
                                   *reinterpret_cast<unsigned int*>(&value),
                                   DesulToSYCLMemoryOrder<MemoryOrder>::value);
  return compare;
}
template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if_t<sizeof(T) == 8, T> atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long  is 64-bit");
  sycl_atomic_ref<unsigned long long int, MemoryScope> dest_ref(
      *reinterpret_cast<unsigned long long int*>(dest));
  dest_ref.compare_exchange_strong(*reinterpret_cast<unsigned long long int*>(&compare),
                                   *reinterpret_cast<unsigned long long int*>(&value),
                                   DesulToSYCLMemoryOrder<MemoryOrder>::value);
  return compare;
}

template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if_t<sizeof(T) == 4, T> atomic_exchange(T* const dest,
                                                             T value,
                                                             MemoryOrder,
                                                             MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  sycl_atomic_ref<unsigned int, MemoryScope> dest_ref(
      *reinterpret_cast<unsigned int*>(dest));
  unsigned int return_val =
      dest_ref.exchange(*reinterpret_cast<unsigned int*>(&value),
                        DesulToSYCLMemoryOrder<MemoryOrder>::value);
  return reinterpret_cast<T&>(return_val);
}
template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if_t<sizeof(T) == 8, T> atomic_exchange(T* const dest,
                                                             T value,
                                                             MemoryOrder,
                                                             MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long  is 64-bit");
  sycl_atomic_ref<unsigned long long int, MemoryScope> dest_ref(
      *reinterpret_cast<unsigned long long int*>(dest));
  unsigned long long int return_val =
      dest_ref.exchange(reinterpret_cast<unsigned long long int&>(value),
                        DesulToSYCLMemoryOrder<MemoryOrder>::value);
  return reinterpret_cast<T&>(return_val);
}

template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if_t<(sizeof(T) != 8) && (sizeof(T) != 4), T>
atomic_compare_exchange(
    T* const /*dest*/, T compare, T /*value*/, MemoryOrder, MemoryScope) {
  // FIXME_SYCL not implemented
  static_assert((sizeof(T) == 8) || (sizeof(T) == 4), "Not implemented!");
  return compare;
}

template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if_t<(sizeof(T) != 8) && (sizeof(T) != 4), T> atomic_exchange(
    T* const /*dest*/, T value, MemoryOrder, MemoryScope) {
  // FIXME_SYCL not implemented
  static_assert((sizeof(T) == 8) || (sizeof(T) == 4), "Not implemented!");
  return value;
}

}  // namespace desul

#endif
#endif
