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

#ifndef DESUL_ATOMICS_COMMON_HPP_
#define DESUL_ATOMICS_COMMON_HPP_
#include <DESUL_Atomics_Macros.hpp>
#include <cinttypes>
namespace desul {
struct alignas(16) Dummy16ByteValue {
  int64_t value1;
  int64_t value2;
  bool operator!=(Dummy16ByteValue v) const {
    return (value1 != v.value1) || (value2 != v.value2);
  }
  bool operator==(Dummy16ByteValue v) const {
    return (value1 == v.value1) && (value2 == v.value2);
  }
};
}  // namespace desul

// MemoryOrder Tags

namespace desul {
// Memory order relaxed
struct MemoryOrderRelaxed {};
// Memory order sequential consistent
struct MemoryOrderSeqCst {};
// Memory order acquire release
struct MemoryOrderAcqRel {};
}  // namespace desul

// Memory Scope Tags

namespace desul {
// Entire machine scope (e.g. for global arrays)
struct MemoryScopeSystem {};
// Node level
struct MemoryScopeNode {};
// Device or socket scope (i.e. a CPU socket, a single GPU)
struct MemoryScopeDevice {};
// Core scoped (i.e. a shared Level 1 cache)
struct MemoryScopeCore {};
}  // namespace desul
#endif
