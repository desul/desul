/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_CUDA_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_CUDA_HPP_
#include "desul/atomics/Common.hpp"

#ifdef DESUL_HAVE_CUDA_ATOMICS
namespace desul {

// Only include if compiling device code, or the CUDA compiler is not NVCC (i.e. Clang)
#if defined(__CUDA_ARCH__) || !defined(__NVCC__)
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  uint32_t return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      *(reinterpret_cast<unsigned int*>(&compare)),
                                      *(reinterpret_cast<unsigned int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  uint64_t return_val =
      atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                *(reinterpret_cast<unsigned long long int*>(&compare)),
                *(reinterpret_cast<unsigned long long int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  uint32_t return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      *(reinterpret_cast<unsigned int*>(&compare)),
                                      *(reinterpret_cast<unsigned int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  uint64_t return_val =
      atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                *(reinterpret_cast<unsigned long long int*>(&compare)),
                *(reinterpret_cast<unsigned long long int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
#endif
}  // namespace desul
#endif
#endif
