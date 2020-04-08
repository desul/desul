/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMMON_HPP_
#define DESUL_ATOMICS_COMMON_HPP_
#include "desul/atomics/Macros.hpp"
#include <cstdint>
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
