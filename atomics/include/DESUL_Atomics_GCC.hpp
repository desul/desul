/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#ifndef DESUL_ATOMICS_GCC_HPP_
#define DESUL_ATOMICS_GCC_HPP_

#ifdef DESUL_HAVE_GCC_ATOMICS

#include<type_traits>
/*
Built - in Function : type __atomic_add_fetch(type * ptr, type val, int memorder)
Built - in Function : type __atomic_sub_fetch(type * ptr, type val, int memorder)
Built - in Function : type __atomic_and_fetch(type * ptr, type val, int memorder)
Built - in Function : type __atomic_xor_fetch(type * ptr, type val, int memorder)
Built - in Function : type __atomic_or_fetch(type * ptr, type val, int memorder)
Built - in Function : type __atomic_nand_fetch(type * ptr, type val, int memorder)
*/

#define DESUL_GCC_INTEGRAL_OP_ATOMICS(MEMORY_ORDER, MEMORY_SCOPE)                 \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_add(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_fetch_add(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);  \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_sub(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_fetch_sub(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);  \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_and(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_fetch_and(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);  \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_or(   \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_fetch_or(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);   \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_xor(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_fetch_xor(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);  \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_fetch_nand( \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_fetch_nand(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value); \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_add_fetch(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_add_fetch(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);  \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_sub_fetch(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_sub_fetch(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);  \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_and_fetch(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_and_fetch(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);  \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_or_fetch(   \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_or_fetch(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);   \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_xor_fetch(  \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_xor_fetch(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);  \
  }                                                                               \
  template <typename T>                                                           \
  typename std::enable_if<std::is_integral<T>::value, T>::type atomic_nand_fetch( \
      T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) {                       \
    return __atomic_nand_fetch(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value); \
  }

namespace desul {
DESUL_GCC_INTEGRAL_OP_ATOMICS(MemoryOrderRelaxed, MemoryScopeNode)
DESUL_GCC_INTEGRAL_OP_ATOMICS(MemoryOrderRelaxed, MemoryScopeDevice)
DESUL_GCC_INTEGRAL_OP_ATOMICS(MemoryOrderRelaxed, MemoryScopeCore)
DESUL_GCC_INTEGRAL_OP_ATOMICS(MemoryOrderSeqCst, MemoryScopeNode)
DESUL_GCC_INTEGRAL_OP_ATOMICS(MemoryOrderSeqCst, MemoryScopeDevice)
DESUL_GCC_INTEGRAL_OP_ATOMICS(MemoryOrderSeqCst, MemoryScopeCore)
}  // namespace desul
#endif
#endif
