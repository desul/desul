/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_SYCL_CONVERSIONS_HPP_
#define DESUL_ATOMICS_SYCL_CONVERSIONS_HPP_
#ifdef DESUL_HAVE_SYCL_ATOMICS
#include <CL/sycl.hpp>
#include "desul/atomics/Common.hpp"

namespace desul {

#ifdef __INTEL_LLVM_COMPILER
namespace sycl_atomic = ::sycl::ONEAPI;
#else
namespace sycl_atomic = ::sycl;
#endif

template <class MemoryOrder>
struct DesulToSYCLMemoryOrder;
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderSeqCst> {
  static constexpr sycl_atomic::memory_order value = sycl_atomic::memory_order::seq_cst;
};
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderAcquire> {
  static constexpr sycl_atomic::memory_order value = sycl_atomic::memory_order::acquire;
};
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderRelease> {
  static constexpr sycl_atomic::memory_order value = sycl_atomic::memory_order::release;
};
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderAcqRel> {
  static constexpr sycl_atomic::memory_order value = sycl_atomic::memory_order::acq_rel;
};
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderRelaxed> {
  static constexpr sycl_atomic::memory_order value = sycl_atomic::memory_order::relaxed;
};

template <class MemoryScope>
struct DesulToSYCLMemoryScope;
template <>
struct DesulToSYCLMemoryScope<MemoryScopeCore> {
  static constexpr sycl_atomic::memory_scope value =
      sycl_atomic::memory_scope::work_group;
};
template <>
struct DesulToSYCLMemoryScope<MemoryScopeDevice> {
  static constexpr sycl_atomic::memory_scope value = sycl_atomic::memory_scope::device;
};
template <>
struct DesulToSYCLMemoryScope<MemoryScopeSystem> {
  static constexpr sycl_atomic::memory_scope value = sycl_atomic::memory_scope::system;
};

}  // namespace desul

#endif
#endif
