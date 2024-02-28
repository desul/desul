/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_FECH_OP_HIP_HPP_
#define DESUL_ATOMICS_FECH_OP_HIP_HPP_

#include "desul/atomics/Adapt_GCC.hpp"

namespace desul {
namespace Impl {
#if __has_builtin(__hip_atomic_fetch_add) && __has_builtin(__hip_atomic_fetch_min) && \
    __has_builtin(__hip_atomic_fetch_max)
// The above intrinsics are the minimum set needed to implement the math
// operators:
//  fetch_add, fetch_sub, fetch_min, fetch_max, fetch_inc, fetch_dec
#define DESUL_HAVE_HIP_ATOMIC_MATH_DETAIL
#endif
#if __has_builtin(__hip_atomic_fetch_and) && __has_builtin(__hip_atomic_fetch_or) && \
    __has_builtin(__hip_atomic_fetch_xor)
// The above intrinsics are needed for bitwise operations on integral types.
#define DESUL_HAVE_HIP_ATOMIC_BIT_DETAIL
#endif

template <typename MemoryTag>
struct DesulToBuiltin;

template <>
struct DesulToBuiltin<MemoryScopeCaller> {
  constexpr static auto value = __HIP_MEMORY_SCOPE_SINGLETHREAD;
};
template <>
struct DesulToBuiltin<MemoryScopeCore> {
  constexpr static auto value = __HIP_MEMORY_SCOPE_WORKGROUP;
};
template <>
struct DesulToBuiltin<MemoryScopeDevice> {
  constexpr static auto value = __HIP_MEMORY_SCOPE_AGENT;
};
template <>
struct DesulToBuiltin<MemoryScopeNode> {
  constexpr static auto value = __HIP_MEMORY_SCOPE_SYSTEM;
};

#ifdef DESUL_HAVE_HIP_ATOMIC_MATH_DETAIL
template <typename T, typename MemoryOrder, typename MemoryScope>
inline __device__ std::enable_if_t<std::is_arithmetic<T>::value, T>
device_atomic_fetch_add(T* ptr, T val, MemoryOrder, MemoryScope) {
  return __hip_atomic_fetch_add(
      ptr, val, GCCMemoryOrder<MemoryOrder>::value, DesulToBuiltin<MemoryScope>::value);
}

template <typename T, typename MemoryOrder, typename MemoryScope>
inline __device__ std::enable_if_t<std::is_arithmetic<T>::value, T>
device_atomic_fetch_sub(T* ptr, T val, MemoryOrder, MemoryScope) {
#if __has_builtin(__hip_atomic_fetch_sub)
  return __hip_atomic_fetch_sub(
      ptr, val, GCCMemoryOrder<MemoryOrder>::value, DesulToBuiltin<MemoryScope>::value);
#else
  return __hip_atomic_fetch_add(ptr,
                                -val,
                                GCCMemoryOrder<MemoryOrder>::value,
                                DesulToBuiltin<MemoryScope>::value);
#endif
}

template <typename T, typename MemoryOrder, typename MemoryScope>
inline __device__ std::enable_if_t<std::is_arithmetic<T>::value, T>
device_atomic_fetch_min(T* ptr, T val, MemoryOrder, MemoryScope) {
  return __hip_atomic_fetch_min(
      ptr, val, GCCMemoryOrder<MemoryOrder>::value, DesulToBuiltin<MemoryScope>::value);
}

template <typename T, typename MemoryOrder, typename MemoryScope>
inline __device__ std::enable_if_t<std::is_arithmetic<T>::value, T>
device_atomic_fetch_max(T* ptr, T val, MemoryOrder, MemoryScope) {
  return __hip_atomic_fetch_max(
      ptr, val, GCCMemoryOrder<MemoryOrder>::value, DesulToBuiltin<MemoryScope>::value);
}

template <typename T, typename MemoryOrder, typename MemoryScope>
inline __device__ std::enable_if_t<std::is_arithmetic<T>::value, T>
device_atomic_fetch_inc(T* ptr, MemoryOrder, MemoryScope) {
  return __hip_atomic_fetch_add(ptr,
                                T{1},
                                GCCMemoryOrder<MemoryOrder>::value,
                                DesulToBuiltin<MemoryScope>::value);
}

template <typename T, typename MemoryOrder, typename MemoryScope>
inline __device__ std::enable_if_t<std::is_arithmetic<T>::value, T>
device_atomic_fetch_dec(T* ptr, MemoryOrder, MemoryScope) {
#if __has_builtin(__hip_atomic_fetch_sub)
  return __hip_atomic_fetch_sub(ptr,
                                T{1},
                                GCCMemoryOrder<MemoryOrder>::value,
                                DesulToBuiltin<MemoryScope>::value);
#else
  return __hip_atomic_fetch_add(ptr,
                                -(T{1}),
                                GCCMemoryOrder<MemoryOrder>::value,
                                DesulToBuiltin<MemoryScope>::value);
#endif
}
#else
// clang-format off
inline __device__                int device_atomic_fetch_add(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_add(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_add(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__              float device_atomic_fetch_add(             float* ptr,              float val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__             double device_atomic_fetch_add(            double* ptr,             double val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }

inline __device__                int device_atomic_fetch_sub(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_sub(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_sub(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -val); }
inline __device__              float device_atomic_fetch_sub(             float* ptr,              float val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -val); }
inline __device__             double device_atomic_fetch_sub(            double* ptr,             double val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -val); }

inline __device__                int device_atomic_fetch_min(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMin(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_min(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMin(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_min(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMin(ptr,  val); }

inline __device__                int device_atomic_fetch_max(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMax(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_max(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMax(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_max(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMax(ptr,  val); }

inline __device__                int device_atomic_fetch_inc(               int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1   ); }
inline __device__       unsigned int device_atomic_fetch_inc(      unsigned int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1u  ); }
inline __device__ unsigned long long device_atomic_fetch_inc(unsigned long long* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1ull); }

inline __device__                int device_atomic_fetch_dec(               int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  1  ); }
inline __device__       unsigned int device_atomic_fetch_dec(      unsigned int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  1u ); }
inline __device__ unsigned long long device_atomic_fetch_dec(unsigned long long* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -1  ); }
// clang-format on
#endif

#ifdef DESUL_HAVE_HIP_ATOMIC_BIT_DETAIL
template <typename T, typename MemoryOrder, typename MemoryScope>
inline __device__ std::enable_if_t<std::is_integral<T>::value, T>
device_atomic_fetch_and(T* ptr, T val, MemoryOrder, MemoryScope) {
  return __hip_atomic_fetch_and(
      ptr, val, GCCMemoryOrder<MemoryOrder>::value, DesulToBuiltin<MemoryScope>::value);
}

template <typename T, typename MemoryOrder, typename MemoryScope>
inline __device__ std::enable_if_t<std::is_integral<T>::value, T>
device_atomic_fetch_or(T* ptr, T val, MemoryOrder, MemoryScope) {
  return __hip_atomic_fetch_or(
      ptr, val, GCCMemoryOrder<MemoryOrder>::value, DesulToBuiltin<MemoryScope>::value);
}

template <typename T, typename MemoryOrder, typename MemoryScope>
inline __device__ std::enable_if_t<std::is_integral<T>::value, T>
device_atomic_fetch_xor(T* ptr, T val, MemoryOrder, MemoryScope) {
  return __hip_atomic_fetch_xor(
      ptr, val, GCCMemoryOrder<MemoryOrder>::value, DesulToBuiltin<MemoryScope>::value);
}
#else
// clang-format off
inline __device__                int device_atomic_fetch_and(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAnd(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_and(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAnd(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_and(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAnd(ptr,  val); }

inline __device__                int device_atomic_fetch_or (               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicOr (ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_or (      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicOr (ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_or (unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicOr (ptr,  val); }

inline __device__                int device_atomic_fetch_xor(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicXor(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_xor(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicXor(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_xor(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicXor(ptr,  val); }
// clang-format on
#endif

// clang-format off
inline __device__       unsigned int device_atomic_fetch_inc_mod(  unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicInc(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_dec_mod(  unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicDec(ptr,  val); }
// clang-format on

#define DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, TYPE)                                \
  template <class MemoryOrder>                                                         \
  __device__ TYPE device_atomic_fetch_##OP(                                            \
      TYPE* ptr, TYPE val, MemoryOrder, MemoryScopeDevice) {                           \
    __threadfence();                                                                   \
    TYPE return_val =                                                                  \
        device_atomic_fetch_##OP(ptr, val, MemoryOrderRelaxed(), MemoryScopeDevice()); \
    __threadfence();                                                                   \
    return return_val;                                                                 \
  }                                                                                    \
  template <class MemoryOrder>                                                         \
  __device__ TYPE device_atomic_fetch_##OP(                                            \
      TYPE* ptr, TYPE val, MemoryOrder, MemoryScopeCore) {                             \
    return device_atomic_fetch_##OP(ptr, val, MemoryOrder(), MemoryScopeDevice());     \
  }

#define DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(OP) \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, int)           \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, unsigned int)  \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, unsigned long long)

#define DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(OP) \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, float)               \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, double)

#ifndef DESUL_HAVE_HIP_ATOMIC_MATH_DETAIL
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(min)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(max)
#endif

#ifndef DESUL_HAVE_HIP_ATOMIC_BIT_DETAIL
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(and)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(or)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(xor)
#endif

#ifndef DESUL_HAVE_HIP_ATOMIC_MATH_DETAIL
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(add)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(add)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(sub)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(sub)

DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(inc)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(dec)
#endif

DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(inc_mod, unsigned int)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(dec_mod, unsigned int)

#undef DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT
#undef DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL
#undef DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP

#undef DESUL_HAVE_HIP_ATOMIC_MATH_DETAIL
#undef DESUL_HAVE_HIP_ATOMIC_BIT_DETAIL

}  // namespace Impl
}  // namespace desul

#endif
