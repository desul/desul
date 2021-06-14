/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_SYCL_HPP_
#define DESUL_ATOMICS_SYCL_HPP_

#ifdef DESUL_HAVE_SYCL_ATOMICS
#include "desul/atomics/Common.hpp"
#include "desul/atomics/SYCLConversions.hpp"

namespace desul {

namespace Impl {
template <class T>
struct is_sycl_atomic_type {
  static constexpr bool value =
      std::is_same<T, int>::value || std::is_same<T, unsigned int>::value ||
      std::is_same<T, long>::value || std::is_same<T, unsigned long>::value ||
      std::is_same<T, long long>::value ||
      std::is_same<T, unsigned long long int>::value || std::is_same<T, float>::value ||
      std::is_same<T, double>::value;
};
}  // namespace Impl

// Atomic Add
template <class T, class MemoryOrder /*, class MemoryScope*/>
inline typename std::enable_if_t<Impl::is_sycl_atomic_type<T>::value, T>
atomic_fetch_add(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  sycl_atomic_ref<T, MemoryScopeDevice> dest_ref(*dest);
  return dest_ref.fetch_add(val, DesulToSYCLMemoryOrder<MemoryOrder>::value);
}

// Atomic Sub
template <class T, class MemoryOrder /*, class MemoryScope*/>
inline typename std::enable_if_t<Impl::is_sycl_atomic_type<T>::value, T>
atomic_fetch_sub(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  sycl_atomic_ref<T, MemoryScopeDevice> dest_ref(*dest);
  return dest_ref.fetch_sub(val, DesulToSYCLMemoryOrder<MemoryOrder>::value);
}

// Atomic Max
template <class T, class MemoryOrder /*, class MemoryScope*/>
inline typename std::enable_if_t<Impl::is_sycl_atomic_type<T>::value, T>
atomic_fetch_max(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  sycl_atomic_ref<T, MemoryScopeDevice> dest_ref(*dest);
  return dest_ref.fetch_max(val, DesulToSYCLMemoryOrder<MemoryOrder>::value);
}

// Atomic Min
template <class T, class MemoryOrder /*, class MemoryScope*/>
inline typename std::enable_if_t<Impl::is_sycl_atomic_type<T>::value, T>
atomic_fetch_min(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  sycl_atomic_ref<T, MemoryScopeDevice> dest_ref(*dest);
  return dest_ref.fetch_min(val, DesulToSYCLMemoryOrder<MemoryOrder>::value);
}

// Atomic And
template <class T, class MemoryOrder /*, class MemoryScope*/>
inline typename std::enable_if_t<Impl::is_sycl_atomic_type<T>::value, T>
atomic_fetch_and(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  sycl_atomic_ref<T, MemoryScopeDevice> dest_ref(*dest);
  return dest_ref.fetch_and(val, DesulToSYCLMemoryOrder<MemoryOrder>::value);
}

// Atomic XOR
template <class T, class MemoryOrder /*, class MemoryScope*/>
inline typename std::enable_if_t<Impl::is_sycl_atomic_type<T>::value, T>
atomic_fetch_xor(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  sycl_atomic_ref<T, MemoryScopeDevice> dest_ref(*dest);
  return dest_ref.fetch_xor(val, DesulToSYCLMemoryOrder<MemoryOrder>::value);
}

// Atomic OR
template <class T, class MemoryOrder /*, class MemoryScope*/>
inline typename std::enable_if_t<Impl::is_sycl_atomic_type<T>::value, T>
atomic_fetch_or(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  sycl_atomic_ref<T, MemoryScopeDevice> dest_ref(*dest);
  return dest_ref.fetch_or(val, DesulToSYCLMemoryOrder<MemoryOrder>::value);
}

}  // namespace desul
#endif  // DESUL_HAVE_SYCL_ATOMICS
#endif  // DESUL_ATOMICS_SYCL_HPP_
