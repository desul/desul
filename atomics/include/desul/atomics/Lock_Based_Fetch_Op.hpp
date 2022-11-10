/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_LOCK_BASED_FETCH_OP_HPP_
#define DESUL_ATOMICS_LOCK_BASED_FETCH_OP_HPP_

#include <desul/atomics/Macros.hpp>

#ifdef DESUL_HAVE_CUDA_ATOMICS
#include <desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp>
#endif
#ifdef DESUL_HAVE_HIP_ATOMICS
#include <desul/atomics/Lock_Based_Fetch_Op_HIP.hpp>
#endif
// FIXME_SYCL Use SYCL_EXT_ONEAPI_DEVICE_GLOBAL when available instead
#ifdef DESUL_SYCL_DEVICE_GLOBAL_SUPPORTED
#include <desul/atomics/Lock_Based_Fetch_Op_SYCL.hpp>
#elif defined(DESUL_HAVE_SYCL_ATOMICS)
#include <desul/atomics/Lock_Based_Fetch_Op_Unimplemented.hpp>
#endif

#include <desul/atomics/Lock_Based_Fetch_Op_Host.hpp>

#endif
