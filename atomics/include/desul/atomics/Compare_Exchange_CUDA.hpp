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
__device__ void atomic_thread_fence(MemoryOrderRelease, MemoryScopeDevice) {
  __threadfence();
}
__device__ void atomic_thread_fence(MemoryOrderRelease, MemoryScopeCore) {
  __threadfence_block();
}
__device__ void atomic_thread_fence(MemoryOrderRelease, MemoryScopeNode) {
  __threadfence_system();
}
__device__ void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeDevice) {
  __threadfence();
}
__device__ void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeCore) {
  __threadfence_block();
}
__device__ void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeNode) {
  __threadfence_system();
}

/*
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(int) == 4, "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      *(reinterpret_cast<unsigned int*>(&compare)),
                                      *(reinterpret_cast<unsigned int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(unsigned long long) == 8, "this function assumes an unsigned long long  is 64-bit");
  unsigned long long int return_val =
      atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                *(reinterpret_cast<unsigned long long int*>(&compare)),
                *(reinterpret_cast<unsigned long long int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  static_assert(sizeof(int) == 4, "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      *(reinterpret_cast<unsigned int*>(&compare)),
                                      *(reinterpret_cast<unsigned int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  static_assert(sizeof(unsigned long long) == 8, "this function assumes an unsigned long long  is 64-bit");
  unsigned long long int return_val =
      atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                *(reinterpret_cast<unsigned long long int*>(&compare)),
                *(reinterpret_cast<unsigned long long int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelease, MemoryScope) {
  static_assert(sizeof(int) == 4, "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      *(reinterpret_cast<unsigned int*>(&compare)),
                                      *(reinterpret_cast<unsigned int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelease, MemoryScope) {
  static_assert(sizeof(unsigned long long) == 8, "this function assumes an unsigned long long  is 64-bit");
  unsigned long long int return_val =
      atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                *(reinterpret_cast<unsigned long long int*>(&compare)),
                *(reinterpret_cast<unsigned long long int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderAcquire, MemoryScope) {
  static_assert(sizeof(int) == 4, "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      *(reinterpret_cast<unsigned int*>(&compare)),
                                      *(reinterpret_cast<unsigned int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderAcquire, MemoryScope) {
  static_assert(sizeof(unsigned long long) == 8, "this function assumes an unsigned long long  is 64-bit");
  unsigned long long int return_val =
      atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                *(reinterpret_cast<unsigned long long int*>(&compare)),
                *(reinterpret_cast<unsigned long long int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderAcqRel, MemoryScope) {
  static_assert(sizeof(int) == 4, "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      *(reinterpret_cast<unsigned int*>(&compare)),
                                      *(reinterpret_cast<unsigned int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderAcqRel, MemoryScope) {
  static_assert(sizeof(unsigned long long) == 8, "this function assumes an unsigned long long  is 64-bit");
  unsigned long long int return_val =
      atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                *(reinterpret_cast<unsigned long long int*>(&compare)),
                *(reinterpret_cast<unsigned long long int*>(&value)));
  return *(reinterpret_cast<T*>(&return_val));
}*/
#endif

}  // namespace desul
#endif

#include <desul/atomics/cuda/CUDA_asm_exchange.hpp>
namespace desul {

#if defined(__CUDA_ARCH__) || !defined(__NVCC__)
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  atomic_thread_fence(MemoryOrderAcquire(),MemoryScope());
  T return_val = atomic_compare_exchange(dest,compare,value,MemoryOrderRelaxed(),MemoryScope());
  atomic_thread_fence(MemoryOrderRelease(),MemoryScope());
  return return_val;
}
template <typename T, class MemoryScope>
__device__ typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  atomic_thread_fence(MemoryOrderAcquire(),MemoryScope());
  T return_val = atomic_compare_exchange(dest,compare,value,MemoryOrderRelaxed(),MemoryScope());
  atomic_thread_fence(MemoryOrderRelease(),MemoryScope());
  return return_val;
}
#endif
}
#endif
