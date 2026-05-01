/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_FECH_OP_HIP_HPP_
#define DESUL_ATOMICS_FECH_OP_HIP_HPP_

#include <desul/atomics/Adapt_HIP.hpp>
#include <type_traits>

namespace desul {
namespace Impl {

template <class To, class From>
DESUL_FORCEINLINE_FUNCTION To desul_bit_cast(const From& x) {
  static_assert(sizeof(To) == sizeof(From));
  static_assert(std::is_trivially_copyable_v<To>);
  static_assert(std::is_trivially_copyable_v<From>);
#if defined(__clang__)
  return __builtin_bit_cast(To, x);
#else
  To out;
  __builtin_memcpy(&out, &x, sizeof(To));
  return out;
#endif
}

#define DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, T)                           \
  template <class MemoryOrder, class MemoryScope>                       \
  __device__ inline T device_atomic_fetch_##OP(                         \
      T* ptr, T val, MemoryOrder, MemoryScope) {                        \
    return __hip_atomic_fetch_##OP(ptr,                                 \
                                   val,                                 \
                                   HIPMemoryOrder<MemoryOrder>::value,  \
                                   HIPMemoryScope<MemoryScope>::value); \
  }

template <typename T, class MemoryOrder, class MemoryScope>
  __device__ inline T device_atomic_load_intrinsic(
      const T* const ptr, MemoryOrder, MemoryScope) {
    return __hip_atomic_load(ptr, HIPMemoryOrder<MemoryOrder>::value,
                                  HIPMemoryScope<MemoryScope>::value);
  }

template <typename T, class MemoryOrder, class MemoryScope>
  __device__ inline void device_atomic_store_intrinsic(
       T* const dest, const T val, MemoryOrder, MemoryScope) {
    __hip_atomic_store(dest, val, HIPMemoryOrder<MemoryOrder>::value,
                                  HIPMemoryScope<MemoryScope>::value);
  }

#define DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(OP) \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, int)           \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, long)          \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, long long)     \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, unsigned int)  \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, unsigned long) \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, unsigned long long)

#define DESUL_IMPL_HIP_ATOMIC_FETCH_OP_FLOATING_POINT(OP) \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, float)               \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, double)

DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(add)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(min)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(max)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(and)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(or)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(xor)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_FLOATING_POINT(add)

template <typename T, class MemoryOrder, class MemoryScope>
__device__ inline T device_atomic_min_intrinsic(T* ptr,
                                            T val,
                                            MemoryOrder,
                                            MemoryScope) {
  static constexpr auto hip_mem_order = HIPMemoryOrder<MemoryOrder>::value;
  static constexpr auto hip_mem_scope = HIPMemoryScope<MemoryScope>::value;
#if defined(__has_builtin) && __has_builtin(__hip_atomic_load)
  // When the memory ordering is relaxed, we want to early exit if no
  // update is necessary.
  if constexpr (hip_mem_order == __ATOMIC_RELAXED) {
    bool val_is_neg_zero = false;
    const T old = __hip_atomic_load(ptr, hip_mem_order, hip_mem_scope);
    if constexpr (std::is_floating_point_v<T>) {
      using unsigned_int_t = std::conditional_t<std::is_same_v<double, T>, uint64_t, uint32_t>;
      constexpr unsigned_int_t bitwise_negative_zero = std::is_same_v<double, T> ?
                                        0x8000000000000000ULL : 0x80000000U;

      // we want to avoid dispatching the intrinsic in the case where *ptr <= val
      // If the old value is 0.0f, and ptr is -0.0f, we must update +0.0 -> -0.0,
      // so check manually for this case.
      val_is_neg_zero = (bitwise_negative_zero == desul_bit_cast<unsigned_int_t>(val)) && (old == T(0));
    }
    if (!(old > val) && !(val_is_neg_zero)) {
        return old;
    }
  }
#endif
  return __hip_atomic_fetch_min(ptr, val, hip_mem_order, hip_mem_scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
__device__ inline T device_atomic_max_intrinsic(T* ptr,
                                            T val,
                                            MemoryOrder,
                                            MemoryScope) {
  static constexpr auto hip_mem_order = HIPMemoryOrder<MemoryOrder>::value;
  static constexpr auto hip_mem_scope = HIPMemoryScope<MemoryScope>::value;
#if defined(__has_builtin) && __has_builtin(__hip_atomic_load)
  // When the memory ordering is relaxed, we want to early exit if no
  // update is necessary.
  if constexpr (hip_mem_order == __ATOMIC_RELAXED) {
    const T old = __hip_atomic_load(ptr, hip_mem_order, hip_mem_scope);
    bool old_is_neg_zero = false;
    if constexpr (std::is_floating_point_v<T>) {
      using unsigned_int_t = std::conditional_t<std::is_same_v<double, T>, uint64_t, uint32_t>;
      constexpr unsigned_int_t bitwise_negative_zero = std::is_same_v<double, T> ?
                                        0x8000000000000000ULL : 0x80000000U;

      old_is_neg_zero = bitwise_negative_zero == desul_bit_cast<unsigned_int_t>(val) && val == T(0);
    }
    if (!(old < val) && !(old_is_neg_zero)) {
      return old;
    }
  }
#endif
  return __hip_atomic_fetch_max(ptr, val, hip_mem_order, hip_mem_scope);
}

#undef DESUL_IMPL_HIP_ATOMIC_FETCH_OP_FLOATING_POINT
#undef DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL
#undef DESUL_IMPL_HIP_ATOMIC_FETCH_OP

#define DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(T)                             \
  template <class MemoryOrder, class MemoryScope>                      \
  __device__ inline T device_atomic_fetch_sub(                         \
      T* ptr, T val, MemoryOrder, MemoryScope) {                       \
    return __hip_atomic_fetch_add(ptr,                                 \
                                  -val,                                \
                                  HIPMemoryOrder<MemoryOrder>::value,  \
                                  HIPMemoryScope<MemoryScope>::value); \
  }

DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(int)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(long)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(long long)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(unsigned int)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(unsigned long)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(unsigned long long)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(float)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(double)

#undef DESUL_IMPL_HIP_ATOMIC_FETCH_SUB

#define DESUL_IMPL_HIP_ATOMIC_FETCH_INC(T)                                        \
  template <class MemoryOrder, class MemoryScope>                                 \
  __device__ inline T device_atomic_fetch_inc(T* ptr, MemoryOrder, MemoryScope) { \
    return __hip_atomic_fetch_add(ptr,                                            \
                                  1,                                              \
                                  HIPMemoryOrder<MemoryOrder>::value,             \
                                  HIPMemoryScope<MemoryScope>::value);            \
  }                                                                               \
  template <class MemoryOrder, class MemoryScope>                                 \
  __device__ inline T device_atomic_fetch_dec(T* ptr, MemoryOrder, MemoryScope) { \
    return __hip_atomic_fetch_add(ptr,                                            \
                                  -1,                                             \
                                  HIPMemoryOrder<MemoryOrder>::value,             \
                                  HIPMemoryScope<MemoryScope>::value);            \
  }

DESUL_IMPL_HIP_ATOMIC_FETCH_INC(int)
DESUL_IMPL_HIP_ATOMIC_FETCH_INC(long)
DESUL_IMPL_HIP_ATOMIC_FETCH_INC(long long)
DESUL_IMPL_HIP_ATOMIC_FETCH_INC(unsigned int)
DESUL_IMPL_HIP_ATOMIC_FETCH_INC(unsigned long)
DESUL_IMPL_HIP_ATOMIC_FETCH_INC(unsigned long long)

#undef DESUL_IMPL_HIP_ATOMIC_FETCH_INC

#define DESUL_IMPL_HIP_ATOMIC_FETCH_INC_MOD(MEMORY_SCOPE, MEMORY_SCOPE_STRING_LITERAL) \
  template <class MemoryOrder>                                                         \
  __device__ inline unsigned int device_atomic_fetch_inc_mod(                          \
      unsigned int* ptr, unsigned int val, MemoryOrder, MEMORY_SCOPE) {                \
    return __builtin_amdgcn_atomic_inc32(                                              \
        ptr, val, HIPMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE_STRING_LITERAL);    \
  }                                                                                    \
  template <class MemoryOrder>                                                         \
  __device__ inline unsigned int device_atomic_fetch_dec_mod(                          \
      unsigned int* ptr, unsigned int val, MemoryOrder, MEMORY_SCOPE) {                \
    return __builtin_amdgcn_atomic_dec32(                                              \
        ptr, val, HIPMemoryOrder<MemoryOrder>::value, MEMORY_SCOPE_STRING_LITERAL);    \
  }

DESUL_IMPL_HIP_ATOMIC_FETCH_INC_MOD(MemoryScopeCore, "workgroup")
DESUL_IMPL_HIP_ATOMIC_FETCH_INC_MOD(MemoryScopeDevice, "agent")
DESUL_IMPL_HIP_ATOMIC_FETCH_INC_MOD(MemoryScopeNode, "")
DESUL_IMPL_HIP_ATOMIC_FETCH_INC_MOD(MemoryScopeSystem, "")

#undef DESUL_IMPL_HIP_ATOMIC_FETCH_INC_MOD

#define DESUL_IMPL_HIP_ATOMIC_LOAD_STORE(T)                                           \
  template <class MemoryOrder, class MemoryScope>                                     \
  __device__ inline T device_atomic_load(T* ptr, MemoryOrder, MemoryScope) {          \
    return __hip_atomic_load(                                                         \
        ptr, HIPMemoryOrder<MemoryOrder>::value, HIPMemoryScope<MemoryScope>::value); \
  }                                                                                   \
  template <class MemoryOrder, class MemoryScope>                                     \
  __device__ inline void device_atomic_store(                                         \
      T* ptr, T val, MemoryOrder, MemoryScope) {                                      \
    return __hip_atomic_store(ptr,                                                    \
                              val,                                                    \
                              HIPMemoryOrder<MemoryOrder>::value,                     \
                              HIPMemoryScope<MemoryScope>::value);                    \
  }

DESUL_IMPL_HIP_ATOMIC_LOAD_STORE(int)
DESUL_IMPL_HIP_ATOMIC_LOAD_STORE(long)
DESUL_IMPL_HIP_ATOMIC_LOAD_STORE(long long)
DESUL_IMPL_HIP_ATOMIC_LOAD_STORE(unsigned int)
DESUL_IMPL_HIP_ATOMIC_LOAD_STORE(unsigned long)
DESUL_IMPL_HIP_ATOMIC_LOAD_STORE(unsigned long long)
DESUL_IMPL_HIP_ATOMIC_LOAD_STORE(float)
DESUL_IMPL_HIP_ATOMIC_LOAD_STORE(double)

#undef DESUL_IMPL_HIP_ATOMIC_LOAD_STORE

}  // namespace Impl
}  // namespace desul

#endif
