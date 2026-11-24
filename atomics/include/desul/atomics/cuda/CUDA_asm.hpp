#include <limits>
namespace desul {
namespace Impl {
// Choose the variant of atomics we are using later
// The __isGlobal intrinsic was only introduced in CUDA 11.2
// It also stopped working in NVC++ 23.1 - it works in 22.11
// this is a bug in NVHPC, not treating CUDA intrinsics correctly
// FIXME_NVHPC
#if !defined(DESUL_IMPL_ATOMIC_CUDA_PTX_PREDICATE) && \
    !defined(DESUL_IMPL_ATOMIC_CUDA_PTX_ISGLOBAL)
#if ((__CUDACC_VER_MAJOR__ > 11) ||                                   \
     ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ > 1))) && \
    (!defined(__NVCOMPILER) || __NVCOMPILER_MAJOR__ < 23)
#define DESUL_IMPL_ATOMIC_CUDA_PTX_ISGLOBAL
#else
#define DESUL_IMPL_ATOMIC_CUDA_PTX_PREDICATE
#endif
#endif

// The arithmetic operations are implemented for the fixed-width integer types.
// The other integer types (e.g. long long where int64_t is long) are forwarded to
// the fixed-width integer type of same size and signedness.
template <class T>
using cuda_asm_fixed_width_integer_t =
    std::conditional_t<std::is_signed_v<T>,
                       std::conditional_t<sizeof(T) == 4, int32_t, int64_t>,
                       std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>;

template <class T>
inline constexpr bool cuda_asm_is_forwarded_integer_v =
    !std::is_same_v<T, cuda_asm_fixed_width_integer_t<T>> &&
    (std::is_same_v<T, long> || std::is_same_v<T, unsigned long> ||
     std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>);

#include <desul/atomics/cuda/cuda_cc7_asm.inc>

}  // namespace Impl
}  // namespace desul
