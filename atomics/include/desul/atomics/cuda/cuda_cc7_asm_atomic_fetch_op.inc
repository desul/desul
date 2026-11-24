#ifdef DESUL_IMPL_ATOMIC_CUDA_PTX_ISGLOBAL
#include "cuda_cc7_asm_atomic_fetch_op.inc_isglobal"
#endif

#ifdef DESUL_IMPL_ATOMIC_CUDA_PTX_PREDICATE
#include "cuda_cc7_asm_atomic_fetch_op.inc_predicate"
#endif

// Integer types that are not one of the fixed-width types handled above
#define __DESUL_IMPL_CUDA_ASM_ATOMIC_FETCH_OP_FORWARD(op) \
template<class ctype> \
inline __device__ typename std::enable_if<cuda_asm_is_forwarded_integer_v<ctype>, ctype>::type device_atomic_fetch_##op(ctype* dest, ctype value, __DESUL_IMPL_CUDA_ASM_MEMORY_ORDER order, __DESUL_IMPL_CUDA_ASM_MEMORY_SCOPE scope) { \
  return device_atomic_fetch_##op(reinterpret_cast<cuda_asm_fixed_width_integer_t<ctype>*>(dest), value, order, scope); \
}

__DESUL_IMPL_CUDA_ASM_ATOMIC_FETCH_OP_FORWARD(add)
__DESUL_IMPL_CUDA_ASM_ATOMIC_FETCH_OP_FORWARD(sub)
__DESUL_IMPL_CUDA_ASM_ATOMIC_FETCH_OP_FORWARD(min)
__DESUL_IMPL_CUDA_ASM_ATOMIC_FETCH_OP_FORWARD(max)

#undef __DESUL_IMPL_CUDA_ASM_ATOMIC_FETCH_OP_FORWARD
