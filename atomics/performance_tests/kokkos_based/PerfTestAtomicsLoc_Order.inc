#define MEMORY_ORDER MemoryOrderRelaxed
#include "PerfTestAtomicsLoc.inc"
#undef MEMORY_ORDER
#define MEMORY_ORDER MemoryOrderRelease
#include "PerfTestAtomicsLoc.inc"
#undef MEMORY_ORDER
#define MEMORY_ORDER MemoryOrderAcquire
#include "PerfTestAtomicsLoc.inc"
#undef MEMORY_ORDER
#define MEMORY_ORDER MemoryOrderAcqRel
#include "PerfTestAtomicsLoc.inc"
#undef MEMORY_ORDER
#define MEMORY_ORDER MemoryOrderSeqCst
#include "PerfTestAtomicsLoc.inc"
#undef MEMORY_ORDER
