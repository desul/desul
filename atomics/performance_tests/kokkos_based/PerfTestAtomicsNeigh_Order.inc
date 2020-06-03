#define MEMORY_ORDER MemoryOrderRelaxed
#include "PerfTestAtomicsNeigh.inc"
#undef MEMORY_ORDER
#define MEMORY_ORDER MemoryOrderRelease
#include "PerfTestAtomicsNeigh.inc"
#undef MEMORY_ORDER
#define MEMORY_ORDER MemoryOrderAcquire
#include "PerfTestAtomicsNeigh.inc"
#undef MEMORY_ORDER
#define MEMORY_ORDER MemoryOrderAcqRel
#include "PerfTestAtomicsNeigh.inc"
#undef MEMORY_ORDER
#define MEMORY_ORDER MemoryOrderSeqCst
#include "PerfTestAtomicsNeigh.inc"
#undef MEMORY_ORDER
