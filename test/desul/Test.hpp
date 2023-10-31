#ifndef DESUL_TEST_HPP
#define DESUL_TEST_HPP

#include "desul/Macros.h"
#include "gtest/gtest.h"

// TODO: Add set up and tear down macros
//       For CUDA set up, create a stream

///
/// Host test set up macro
///
#define DESUL_HOST_TEST(DESUL_SUITE_NAME, DESUL_TEST_NAME) \
TEST(host_ ## DESUL_SUITE_NAME, DESUL_TEST_NAME) { \
   const bool passed = desul::test::DESUL_SUITE_NAME::DESUL_TEST_NAME(); \
   EXPECT_TRUE(passed); \
}

///
/// CUDA test set up macro
///
#if defined(__CUDACC__)
#define DESUL_CUDA_TEST(DESUL_SUITE_NAME, DESUL_TEST_NAME) \
namespace desul { \
   namespace test { \
      namespace DESUL_SUITE_NAME { \
         __global__ void DESUL_TEST_NAME ## _cuda_kernel(bool* passed) { \
            *passed = DESUL_TEST_NAME(); \
         } \
      } \
   } \
} \
\
TEST(cuda_ ## DESUL_SUITE_NAME, DESUL_TEST_NAME) { \
   bool* pinnedBuffer; \
   cudaMallocHost((void**) &pinnedBuffer, sizeof(bool)); \
   desul::test::DESUL_SUITE_NAME::DESUL_TEST_NAME ## _cuda_kernel<<<1, 1>>>(pinnedBuffer); \
   cudaDeviceSynchronize(); \
   const bool passed = *pinnedBuffer; \
   cudaFreeHost(&pinnedBuffer); \
   EXPECT_TRUE(passed); \
}
#else
#define DESUL_CUDA_TEST(DESUL_SUITE_NAME, DESUL_TEST_NAME)
#endif

///
/// Macros to test all enabled programming models
///
#define DESUL_TEST_BEGIN(DESUL_SUITE_NAME, DESUL_TEST_NAME) \
namespace desul { \
   namespace test { \
      namespace DESUL_SUITE_NAME { \
         DESUL_HOST_DEVICE bool DESUL_TEST_NAME () {

#define DESUL_TEST_END(DESUL_SUITE_NAME, DESUL_TEST_NAME) \
         } \
      } \
   } \
} \
DESUL_HOST_TEST(DESUL_SUITE_NAME, DESUL_TEST_NAME) \
DESUL_CUDA_TEST(DESUL_SUITE_NAME, DESUL_TEST_NAME)

#endif // DESUL_TEST_HPP
