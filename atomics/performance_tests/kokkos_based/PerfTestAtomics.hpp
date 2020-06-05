#include<gtest/gtest.h>
#include<desul/atomics.hpp>
#include<Kokkos_Core.hpp>
#include<cstdlib>

template<class T, class U>
inline void MY_ASSERT_EQ(T val1, U val2) {
  ASSERT_EQ(val1,val2);
}
template<class T, int N>
struct compound_type {
  T v[N];

  KOKKOS_FUNCTION
  compound_type(T val) {
    for(int i=0; i<N; i++) v[i] = val;
  }

  KOKKOS_FUNCTION
  compound_type(int val) {
    for(int i=0; i<N; i++) v[i] = static_cast<T>(val);
  }
  
  KOKKOS_FUNCTION
  compound_type() {
    for(int i=0; i<N; i++) v[i] = static_cast<T>(0);
  }

  KOKKOS_FUNCTION 
  compound_type(const compound_type& a) {
    for(int i=0; i<N; i++)
      v[i] = a[i];
  }
  KOKKOS_FUNCTION 
  compound_type(const volatile compound_type& a) {
    for(int i=0; i<N; i++)
      v[i] = a[i];
  }

  KOKKOS_FUNCTION 
  compound_type& operator = (const compound_type& a) {
    for(int i=0; i<N; i++) {
      v[i] = a[i];
    }
    return *this;
  }

  KOKKOS_FUNCTION 
  volatile compound_type& operator = (const volatile compound_type& a) volatile {
    for(int i=0; i<N; i++) {
      v[i] = a[i];
    }
    return *this;
  }
  KOKKOS_FUNCTION 
  volatile compound_type& operator = (const compound_type& a) volatile {
    for(int i=0; i<N; i++) {
      v[i] = a[i];
    }
    return *this;
  }

  KOKKOS_FUNCTION
  T& operator [] (const int i) { return v[i]; }

  KOKKOS_FUNCTION
  const T& operator [] (const int i) const { return v[i]; }
  
  KOKKOS_FUNCTION
  const volatile T& operator [] (const int i) const volatile { return v[i]; }

  KOKKOS_FUNCTION
  bool operator < (compound_type a) const {
    int count = 0;
    for(int i=0; i<N; i++)
      if(v[i] < a[i]) count++;
    return count > N/2;
  }

  KOKKOS_FUNCTION
  bool operator > (compound_type a) const {
    int count = 0;
    for(int i=0; i<N; i++)
      if(v[i] > a[i]) count++;
    return count > N/2;
  }

  KOKKOS_FUNCTION
  compound_type& operator += (compound_type a) {
    for(int i=0; i<N; i++)
      v[i] += a[i];
    return *this;
  }
  KOKKOS_FUNCTION
  compound_type operator + (compound_type a) const {
    compound_type<T,N> b(T(0));
    for(int i=0; i<N; i++)
      b[i] = v[i] + a[i];
    return b;
  }
  KOKKOS_FUNCTION
  compound_type& operator *= (compound_type a) const {
    for(int i=0; i<N; i++)
      v[i] *= a[i];
    return *this;
  }
  KOKKOS_FUNCTION
  compound_type operator * (compound_type a) const {
    compound_type<T,N> b(T(0));
    for(int i=0; i<N; i++)
      b[i] = v[i] * a[i];
    return b;
  }
  KOKKOS_FUNCTION
  compound_type operator - (compound_type a) const {
    compound_type<T,N> b(T(0));
    for(int i=0; i<N; i++)
      b[i] = v[i] - a[i];
    return b;
  }

};

template<class T>
struct tolerance {
  static constexpr T value = 0;
};

template<class T, int N>
struct tolerance<compound_type<T,N>> {
  static constexpr T value = 0;
};

template<>
struct tolerance<double> {
  static constexpr double value = 1e-14;
};

template<>
struct tolerance<float> {
  static constexpr double value = 1e-6;
};

// On some compilers abs(unsigned) doesn't work (ambiguous) need wrapper for that 
template<class T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<std::is_unsigned<T>::value,T>::type my_abs(T val) {
  return val;
}

template<class T>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<!std::is_unsigned<T>::value,T>::type my_abs(T val) {
  using std::abs;
  return abs(val);
}

template<class T, int N>
KOKKOS_INLINE_FUNCTION
T my_abs(compound_type<T,N> val) {
  T sum = 0;
  for(int i=0; i<N; i++) sum += my_abs(val[i]);
  return sum;
}



// This test will fill an index array with N equi-distributed random numbers in the interval 0-K-1
// and then combine values from a larger array of length N into an smaller array of length K
// DeviceType is a Kokkos concept that combines both execution space and memory space.
template<class Scalar, class Combiner, class DeviceType>
struct TestAtomicPerformance_RandomLocation {
  using execution_space_t = typename DeviceType::execution_space;
  using exec_mem_space_t = typename execution_space_t::memory_space;
  using memory_space_t = typename DeviceType::memory_space;
  using scalar_t = Scalar;
  using combiner_t = Combiner;

  using indicies_t = Kokkos::View<int*, exec_mem_space_t>;
  using src_values_t = Kokkos::View<scalar_t*, exec_mem_space_t>; 
  using dst_values_t = Kokkos::View<scalar_t*, memory_space_t>;

  indicies_t indicies;
  src_values_t src_values;
  dst_values_t dst_values;
  
  combiner_t combiner;

  KOKKOS_FUNCTION
  void operator() (const int i) const {
    combiner(&dst_values(indicies(i)),src_values(i));
  }
  
  TestAtomicPerformance_RandomLocation(int N, int K, combiner_t combiner_) {
    indicies  = indicies_t("desul::Tests::PerfRandLoc::indicies",N);
    src_values = src_values_t("desul::Tests::PerfRandLoc::indicies",N);
    dst_values = dst_values_t("desul::Tests::PerfRandLoc::indicies",K);

    auto h_indicies = Kokkos::create_mirror_view(indicies);
    auto h_src_values = Kokkos::create_mirror_view(src_values);
    auto h_dst_values = Kokkos::create_mirror_view(dst_values);

    srand(318391);
    for(int i=0; i<N; i++) {
      h_indicies(i) = rand()%K;
      h_src_values(i) = Scalar(rand());
      if(i<K) h_dst_values(i) = Scalar(rand());
    }

    Kokkos::deep_copy(indicies,h_indicies);
    Kokkos::deep_copy(src_values,h_src_values);
    Kokkos::deep_copy(dst_values,h_dst_values);
  }
  
};

template<class Scalar, class Combiner, class ExecutionSpace, class MemorySpace>
double test_atomic_perf_random_location(int N, int K, Scalar, Combiner combiner, ExecutionSpace exec_space, MemorySpace) {
  TestAtomicPerformance_RandomLocation<Scalar,Combiner,Kokkos::Device<ExecutionSpace,MemorySpace>>
    test(N,K,combiner);

  auto org_indicies = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),test.indicies);
  auto org_src_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),test.src_values);
  auto org_dst_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),test.dst_values);

  Kokkos::Timer timer;
  Kokkos::parallel_for("desul::Tests::PerfRandLoc",N,test);
  Kokkos::fence();
  double time = timer.seconds();

  // Add correctnes check

  auto result_device = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),test.dst_values);

  for(int i=0; i<N; i++) {
    combiner(&org_dst_values(org_indicies(i)),org_src_values(i));
  }

  int errors = 0;
  Kokkos::parallel_reduce("desul::Tests::PerfRandLoc::Check",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,result_device.extent(0)),
    KOKKOS_LAMBDA(const int i, int& count) {
      auto diff = result_device(i)-org_dst_values(i);
      auto sum = result_device(i)+org_dst_values(i);
      using std::abs;
      if(my_abs(sum)>0) {
        if(my_abs(diff)/my_abs(sum)>tolerance<Scalar>::value) {
           count++;
           if(i%10000==0) printf("%i %i %lf %lf %e %e\n",i,K,double(my_abs(diff)),double(my_abs(sum)),double(my_abs(diff)/my_abs(sum)),double(tolerance<Scalar>::value));
        }
      } else {
        if(my_abs(diff)!=0) {
          count++;
          if(i%10000==0) printf("%i %i %lf %lf %e %e\n",i,K,double(my_abs(diff)),double(my_abs(sum)),double(my_abs(diff)/my_abs(sum)),double(tolerance<Scalar>::value));
        }
      }
  },errors);
  if(0!=errors) printf("PerfRandLoc correctness check failed: %i\n",errors);
  MY_ASSERT_EQ(0,errors);
  return time;
}

template<class Scalar, class Combiner, class ExecutionSpace, class MemorySpace>
void test_atomic_perf_random_loc(int N) {
  {
    int M = N/10;
    int K = 1;
    double time_random_loc = test_atomic_perf_random_location(M, K, Scalar(), Combiner() , ExecutionSpace(), MemorySpace());
    printf("RandomLocTest Time: %e s Throughput: %lf GOPs Config: %s %i %i\n",time_random_loc,1.0e-9*M/time_random_loc,typeid(Scalar).name(),M,K);
  }
  {
    int M = N;
    int K = 200;
    if(K<N) {
      double time_random_loc = test_atomic_perf_random_location(M, K, Scalar(), Combiner() , ExecutionSpace(), MemorySpace());
      printf("RandomLocTest Time: %e s Throughput: %lf GOPs Config: %s %i %i\n",time_random_loc,1.0e-9*M/time_random_loc,typeid(Scalar).name(),M,K);
    }
  }
  {
    int M = N;
    int K = 20000;
    if(K<N) {
      double time_random_loc = test_atomic_perf_random_location(M, K, Scalar(), Combiner() , ExecutionSpace(), MemorySpace());
      printf("RandomLocTest Time: %e s Throughput: %lf GOPs Config: %s %i %i\n",time_random_loc,1.0e-9*M/time_random_loc,typeid(Scalar).name(),M,K);
    }
  }
  {
    int M = N;
    int K = N;
    double time_random_loc = test_atomic_perf_random_location(M, K, Scalar(), Combiner() , ExecutionSpace(), MemorySpace());
    printf("RandomLocTest Time: %e s Throughput: %lf GOPs Config: %s %i %i\n",time_random_loc,1.0e-9*M/time_random_loc,typeid(Scalar).name(),M,K);
  }
}

// This test will fill an index array with N indicies offset by equi-distributed random numbers 
// in the interval 0-K-1 from its position in the array. 
// The test then combines values from a larger array of length N into that position. 
// This tests simulates scatter add behavior into neighbors (particles, cells etc.)
// DeviceType is a Kokkos concept that combines both execution space and memory space.
template<class Scalar, class Combiner, class DeviceType>
struct TestAtomicPerformance_RandomNeighs {
  using execution_space_t = typename DeviceType::execution_space;
  using exec_mem_space_t = typename execution_space_t::memory_space;
  using memory_space_t = typename DeviceType::memory_space;
  using scalar_t = Scalar;
  using combiner_t = Combiner;

  using indicies_t = Kokkos::View<int**, exec_mem_space_t>;
  using src_values_t = Kokkos::View<scalar_t*, exec_mem_space_t>; 
  using dst_values_t = Kokkos::View<scalar_t*, memory_space_t>;

  indicies_t indicies;
  src_values_t src_values;
  dst_values_t dst_values;
  
  combiner_t combiner;

  KOKKOS_FUNCTION
  void operator() (const int i) const {
    for(int j=0;j<indicies.extent(1);j++)
      combiner(&dst_values(indicies(i,j)),src_values(i));
  }
  
  TestAtomicPerformance_RandomNeighs(int N, int K, int D, combiner_t combiner_) {
    indicies  = indicies_t("desul::Tests::PerfRandLoc::indicies",N,K);
    src_values = src_values_t("desul::Tests::PerfRandLoc::indicies",N);
    dst_values = dst_values_t("desul::Tests::PerfRandLoc::indicies",N);

    auto h_indicies = Kokkos::create_mirror_view(indicies);
    auto h_src_values = Kokkos::create_mirror_view(src_values);
    auto h_dst_values = Kokkos::create_mirror_view(dst_values);

    srand(318391);
    for(int i=0; i<N; i++) {
      for(int j=0; j<K;j++)
        h_indicies(i,j) = (i+rand()%D)%N;
      h_src_values(i) = Scalar(rand());
      h_dst_values(i) = Scalar(rand());
    }

    Kokkos::deep_copy(indicies,h_indicies);
    Kokkos::deep_copy(src_values,h_src_values);
    Kokkos::deep_copy(dst_values,h_dst_values);
  }
  
};

template<class Scalar, class Combiner, class ExecutionSpace, class MemorySpace>
double test_atomic_perf_random_neighborhood(int N, int K, int D, Scalar, Combiner combiner, ExecutionSpace exec_space, MemorySpace) {
  TestAtomicPerformance_RandomNeighs<Scalar,Combiner,Kokkos::Device<ExecutionSpace,MemorySpace>>
    test(N,K,D,combiner);

  auto org_indicies = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),test.indicies);
  auto org_src_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),test.src_values);
  auto org_dst_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),test.dst_values);

  Kokkos::Timer timer;
  Kokkos::parallel_for("desul::Tests::PerfRandNeigh",N,test);
  Kokkos::fence();
  double time = timer.seconds();

  // Add correctnes check
  auto result_device = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),test.dst_values);

  for(int i=0; i<N; i++)  {
    for(int j=0;j<org_indicies.extent(1);j++)
      combiner(&org_dst_values(org_indicies(i,j)),org_src_values(i));
  }
  using std::abs;
  int errors = 0;
  Kokkos::parallel_reduce("desul::Tests::PerfRandNeigh::Check",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,result_device.extent(0)),
    KOKKOS_LAMBDA(const int i, int& count) {
      auto diff = result_device(i)-org_dst_values(i);
      auto sum = result_device(i)+org_dst_values(i);
      if(my_abs(sum)>0) {
        if(my_abs(diff)/my_abs(sum)>tolerance<Scalar>::value) { count++;
	if(i%10000==0) printf("%i %i %lf %lf %e %e\n",i,K,double(my_abs(diff)),double(my_abs(sum)),double(my_abs(diff))/double(my_abs(sum)),double(tolerance<Scalar>::value)); }
      } else {
        if(my_abs(diff)!=0) { count++;
           if(i%10000==0) printf("%i %i %lf %lf %e %e\n",i,K,double(my_abs(diff)),double(my_abs(sum)),double(my_abs(diff)/my_abs(sum)),double(tolerance<Scalar>::value)); }
      }
  },errors);
  if(0!=errors) printf("PerfRandNeigh correctness check failed: %i\n",errors);
  MY_ASSERT_EQ(0,errors);
  return time;
}

template<class Scalar, class Combiner, class ExecutionSpace, class MemorySpace>
void test_atomic_perf_random_neighs(int N) {
  {
    int M = N;
    int K = 20;
    int D = 20;
    double time_random_loc = test_atomic_perf_random_neighborhood(M, K, D, Scalar(), Combiner() , ExecutionSpace(), MemorySpace());
    printf("RandomNeighTest Time: %e s Throughput: %lf GOPs Config: %s %i %i %i\n",time_random_loc,1.0e-9*M*K/time_random_loc,typeid(Scalar).name(),M,D,K);
  }
  {
    int M = N;
    int K = 20;
    int D = 2000;
    double time_random_loc = test_atomic_perf_random_neighborhood(M, K, D, Scalar(), Combiner() , ExecutionSpace(), MemorySpace());
    printf("RandomNeighTest Time: %e s Throughput: %lf GOPs Config: %s %i %i %i\n",time_random_loc,1.0e-9*M*K/time_random_loc,typeid(Scalar).name(),M,D,K);
  }
  {
    int M = N/10;
    int K = 200;
    int D = 200;
    double time_random_loc = test_atomic_perf_random_neighborhood(M, K, D, Scalar(), Combiner() , ExecutionSpace(), MemorySpace());
    printf("RandomNeighTest Time: %e s Throughput: %lf GOPs Config: %s %i %i %i\n",time_random_loc,1.0e-9*M*K/time_random_loc,typeid(Scalar).name(),M,D,K);
  }
  {
    int M = N/10;
    int K = 200;
    int D = 2000;
    double time_random_loc = test_atomic_perf_random_neighborhood(M, K, D, Scalar(), Combiner() , ExecutionSpace(), MemorySpace());
    printf("RandomNeighTest Time: %e s Throughput: %lf GOPs Config: %s %i %i %i\n",time_random_loc,1.0e-9*M*K/time_random_loc,typeid(Scalar).name(),M,D,K);
  }
}

template<class MemoryOrder,class MemoryScope>
struct atomic_add_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar upd) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    desul::atomic_add(dest,upd,MemoryOrder(),MemoryScope());
    #else
    Kokkos::atomic_add(dest,upd); 
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_fetch_add_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar upd) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    (void) desul::atomic_fetch_add(dest,upd,MemoryOrder(),MemoryScope());
    #else
    (void) Kokkos::atomic_fetch_add(dest,upd); 
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_sub_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar upd) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    desul::atomic_sub(dest,upd,MemoryOrder(),MemoryScope());
    #else
    Kokkos::atomic_sub(dest,upd); 
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_fetch_sub_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar upd) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    (void) desul::atomic_fetch_sub(dest,upd,MemoryOrder(),MemoryScope());
    #else
    (void) Kokkos::atomic_fetch_sub(dest,upd); 
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_inc_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    desul::atomic_inc(dest,MemoryOrder(),MemoryScope());
    #else
    Kokkos::atomic_increment(dest);
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_fetch_inc_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    (void) desul::atomic_fetch_inc(dest,MemoryOrder(),MemoryScope());
    #else
    Kokkos::atomic_increment(dest);
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_dec_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    desul::atomic_dec(dest,MemoryOrder(),MemoryScope());
    #else
    Kokkos::atomic_decrement(dest);
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_fetch_dec_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    (void) desul::atomic_fetch_dec(dest,MemoryOrder(),MemoryScope());
    #else
    Kokkos::atomic_decrement(dest);
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_min_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar upd) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    desul::atomic_min(dest,upd,MemoryOrder(),MemoryScope());
    #else
    (void) Kokkos::atomic_fetch_min(dest,upd);
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_fetch_min_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar upd) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    (void) desul::atomic_fetch_min(dest,upd,MemoryOrder(),MemoryScope());
    #else
    (void) Kokkos::atomic_fetch_min(dest,upd);
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_max_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar upd) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    desul::atomic_max(dest,upd,MemoryOrder(),MemoryScope());
    #else
    (void) Kokkos::atomic_fetch_max(dest,upd);
    #endif
  }
};

template<class MemoryOrder,class MemoryScope>
struct atomic_fetch_max_op {
  template<class Scalar>
  KOKKOS_INLINE_FUNCTION
  void operator() (Scalar* dest, Scalar upd) const {
    #ifndef DESUL_IMPL_TESTS_USE_KOKKOS_ATOMICS 
    (void) desul::atomic_fetch_max(dest,upd,MemoryOrder(),MemoryScope());
    #else
    (void) Kokkos::atomic_fetch_max(dest,upd);
    #endif
  }
};
