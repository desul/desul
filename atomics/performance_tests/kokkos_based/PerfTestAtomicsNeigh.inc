
#define COMBINE(A, B, C, D, E, F) A##_##B##_##C##_##D##_##E##_##F
#define INDIRECTION(A, B, C, D, E, F) COMBINE(A, B, C, D, E, F)
#define TESTNAME                   \
  INDIRECTION(kokkos_random_neigh, \
              MEMORY_OP,           \
              SCALAR_NAME,         \
              EXECUTION_SPACE,     \
              MEMORY_ORDER,        \
              MEMORY_SCOPE)

#define ADD_NS(A, B) A ::B
TEST(atomic, TESTNAME) {
  desul::Impl::ensure_lock_arrays_on_device();
  test_atomic_perf_random_neighs<
      SCALAR,
      MEMORY_OP<ADD_NS(desul, MEMORY_ORDER), ADD_NS(desul, MEMORY_SCOPE)>,
      ADD_NS(Kokkos, EXECUTION_SPACE),
      ADD_NS(Kokkos, MEMORY_SPACE)>(1000000);
}
#undef ADD_NS
#undef TESTNAME
#undef INDIRECTION
#undef COMBINE
