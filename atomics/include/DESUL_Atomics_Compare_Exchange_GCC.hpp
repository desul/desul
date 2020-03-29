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
#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_GCC_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_GCC_HPP_
#include <DESUL_Atomics_Common.hpp>

#ifdef DESUL_HAVE_GCC_ATOMICS
#if !defined(DESUL_HAVE_16BYTE_COMPARE_AND_SWAP) && !defined(__CUDACC__)
// This doesn't work in WSL??
//#define DESUL_HAVE_16BYTE_COMPARE_AND_SWAP
#endif
namespace desul {
template <class MemoryOrderDesul>
struct GCCMemoryOrder;

template <>
struct GCCMemoryOrder<MemoryOrderRelaxed> {
  static constexpr int value = __ATOMIC_RELAXED;
};

template <>
struct GCCMemoryOrder<MemoryOrderSeqCst> {
  static constexpr int value = __ATOMIC_SEQ_CST;
};

template <typename T, class MemoryScope>
T atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  (void)__atomic_compare_exchange_n(
      dest, &compare, value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return compare;
}
template <typename T, class MemoryScope>
T atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  (void)__atomic_compare_exchange_n(
      dest, &compare, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return compare;
}
}  // namespace desul
#endif
#endif
