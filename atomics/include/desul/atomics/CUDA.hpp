/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_CUDA_HPP_
#define DESUL_ATOMICS_CUDA_HPP_

#ifdef DESUL_HAVE_CUDA_ATOMICS
#if defined(__CUDA_ARCH__) || (defined(__clang__) && !defined(__NVCC__))
#include "desul/atomics/CUDA_CC_7.hpp"
#endif
#endif
#endif
