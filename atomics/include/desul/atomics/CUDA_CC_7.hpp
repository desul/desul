/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_CUDA_CC_7_HPP_
#define DESUL_ATOMICS_CUDA_CC_7_HPP_

#include<type_traits>
#include<desul/atomics/CUDA_ASM.hpp>

#define DESUL_CUDA_INTEGRAL_OP_ATOMICS(MEMORY_ORDER, MEMORY_SCOPE)                 \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_add(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_fetch_add_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());  \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_sub(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_fetch_sub_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());  \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_and(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_fetch_and_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());  \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_or(   \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_fetch_or_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());   \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_xor(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_fetch_xor_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());  \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_nand( \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_fetch_nand_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE()); \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_add_fetch(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_add_fetch_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());  \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_sub_fetch(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_sub_fetch_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());  \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_and_fetch(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_and_fetch_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());  \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_or_fetch(   \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_or_fetch_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());   \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_xor_fetch(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_xor_fetch_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE());  \
  }                                                                               \
  template <typename T>                                                           \
  __device__									  \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_nand_fetch( \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return Impl::__atomic_nand_fetch_simt(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value, MEMORY_SCOPE()); \
  }

#define DESUL_CUDA_FLOAT_OP_ATOMICS(SCALAR, MEMORY_ORDER, MEMORY_SCOPE) \
__device__ \
SCALAR atomic_add_fetch(SCALAR* const dest, SCALAR value, MEMORY_ORDER, MEMORY_SCOPE) { \
  return atomicAdd(dest,value)+value; \
} \
__device__ \
SCALAR atomic_fetch_add(SCALAR* const dest, SCALAR value, MEMORY_ORDER, MEMORY_SCOPE) { \
  return atomicAdd(dest,value); \
}\
__device__ \
SCALAR atomic_sub_fetch(SCALAR* const dest, SCALAR value, MEMORY_ORDER, MEMORY_SCOPE) { \
  return atomicAdd(dest,-value)-value; \
} \
__device__ \
SCALAR atomic_fetch_sub(SCALAR* const dest, SCALAR value, MEMORY_ORDER, MEMORY_SCOPE) { \
  return atomicAdd(dest,-value); \
} 

namespace desul {
DESUL_CUDA_INTEGRAL_OP_ATOMICS(MemoryOrderRelaxed, MemoryScopeNode)
DESUL_CUDA_INTEGRAL_OP_ATOMICS(MemoryOrderRelaxed, MemoryScopeDevice)
DESUL_CUDA_INTEGRAL_OP_ATOMICS(MemoryOrderRelaxed, MemoryScopeCore)
DESUL_CUDA_INTEGRAL_OP_ATOMICS(MemoryOrderSeqCst, MemoryScopeNode)
DESUL_CUDA_INTEGRAL_OP_ATOMICS(MemoryOrderSeqCst, MemoryScopeDevice)
DESUL_CUDA_INTEGRAL_OP_ATOMICS(MemoryOrderSeqCst, MemoryScopeCore)


DESUL_CUDA_FLOAT_OP_ATOMICS(float, MemoryOrderRelaxed, MemoryScopeDevice)
DESUL_CUDA_FLOAT_OP_ATOMICS(double, MemoryOrderRelaxed, MemoryScopeDevice)

/* Should this work instead?
template<class MemoryOrder, class MemoryScope>
__device__
double atomic_add_fetch(double* const dest, double value, MemoryOrder, MemoryScope) {
  return Impl::__atomic_add_fetch_simt(dest, value, GCCMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE());
}
template<class MemoryOrder, class MemoryScope>
__device__
double atomic_fetch_add(double* const dest, double value, MemoryOrder, MemoryScope) {
  return Impl::__atomic_fetch_add_simt(dest, value, GCCMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE());
}
template<class MemoryOrder, class MemoryScope>
__device__
double atomic_sub_fetch(double* const dest, double value, MemoryOrder, MemoryScope) {
  return Impl::__atomic_sub_fetch_simt(dest, value, GCCMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE());
}
template<class MemoryOrder, class MemoryScope>
__device__
double atomic_fetch_sub(double* const dest, double value, MemoryOrder, MemoryScope) {
  return Impl::__atomic_fetch_sub_simt(dest, value, GCCMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE());
}
template<class MemoryOrder, class MemoryScope>
__device__
double atomic_add_fetch(float* const dest, double value, MemoryOrder, MemoryScope) {
  return Impl::__atomic_add_fetch_simt(dest, value, GCCMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE());
}
template<class MemoryOrder, class MemoryScope>
__device__
double atomic_fetch_add(float* const dest, double value, MemoryOrder, MemoryScope) {
  return Impl::__atomic_fetch_add_simt(dest, value, GCCMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE());
}
template<class MemoryOrder, class MemoryScope>
__device__
double atomic_sub_fetch(float* const dest, double value, MemoryOrder, MemoryScope) {
  return Impl::__atomic_sub_fetch_simt(dest, value, GCCMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE());
}
template<class MemoryOrder, class MemoryScope>
__device__
double atomic_fetch_sub(float* const dest, double value, MemoryOrder, MemoryScope) {
  return Impl::__atomic_fetch_sub_simt(dest, value, GCCMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE());
}
*/
}  // namespace desul
#endif
