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
#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_MSVC_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_MSVC_HPP_
#include <DESUL_Atomics_Common.hpp>
#include <type_traits>
#ifdef DESUL_HAVE_MSVC_ATOMICS

#ifndef DESUL_HAVE_16BYTE_COMPARE_AND_SWAP
#define DESUL_HAVE_16BYTE_COMPARE_AND_SWAP
#endif
#include <windows.h>

namespace desul {
template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 1, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  CHAR return_val =
      _InterlockedCompareExchange8((CHAR*)dest, *((CHAR*)&val), *((CHAR*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 2, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  SHORT return_val =
      _InterlockedCompareExchange16((SHORT*)dest, *((SHORT*)&val), *((SHORT*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  LONG return_val =
      _InterlockedCompareExchange((LONG*)dest, *((LONG*)&val), *((LONG*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  LONG64 return_val = _InterlockedCompareExchange64(
      (LONG64*)dest, *((LONG64*)&val), *((LONG64*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 16, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  Dummy16ByteValue* val16 = reinterpret_cast<Dummy16ByteValue*>(&val);
  (void)_InterlockedCompareExchange128(reinterpret_cast<LONG64*>(dest),
                                       val16->value2,
                                       val16->value1,
                                       (reinterpret_cast<LONG64*>(&compare)));
  return compare;
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 1, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  CHAR return_val =
      _InterlockedCompareExchange8((CHAR*)dest, *((CHAR*)&val), *((CHAR*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 2, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  SHORT return_val =
      _InterlockedCompareExchange16((SHORT*)dest, *((SHORT*)&val), *((SHORT*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  LONG return_val =
      _InterlockedCompareExchange((LONG*)dest, *((LONG*)&val), *((LONG*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  LONG64 return_val = _InterlockedCompareExchange64(
      (LONG64*)dest, *((LONG64*)&val), *((LONG64*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 16, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  Dummy16ByteValue* val16 = reinterpret_cast<Dummy16ByteValue*>(&val);
  (void)_InterlockedCompareExchange128(reinterpret_cast<LONG64*>(dest),
                                       val16->value2,
                                       val16->value1,
                                       (reinterpret_cast<LONG64*>(&compare)));
  return compare;
}

}  // namespace desul
#endif
#endif