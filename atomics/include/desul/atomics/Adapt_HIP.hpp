/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_ADAPT_HIP_HPP_
#define DESUL_ATOMICS_ADAPT_HIP_HPP_

#include <desul/atomics/Common.hpp>

namespace desul {
namespace Impl {

template <typename MemoryScope>
struct HIPMemoryScope;

template <>
struct HIPMemoryScope<MemoryScopeCaller> {
  constexpr static auto value = __HIP_MEMORY_SCOPE_SINGLETHREAD;
};

template <>
struct HIPMemoryScope<MemoryScopeCore> {
  constexpr static auto value = __HIP_MEMORY_SCOPE_WORKGROUP;
};

template <>
struct HIPMemoryScope<MemoryScopeDevice> {
  constexpr static auto value = __HIP_MEMORY_SCOPE_AGENT;
};

template <>
struct HIPMemoryScope<MemoryScopeSystem> {
  constexpr static auto value = __HIP_MEMORY_SCOPE_SYSTEM;
};

}  // namespace Impl
}  // namespace desul

#endif
