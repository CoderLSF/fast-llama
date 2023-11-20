#pragma once

#if defined(__APPLE__) || defined(_WIN32) || defined(__CYGWIN__)
#define DISABLE_NUMA
#endif

#ifdef DISABLE_NUMA
namespace cpuft {
inline bool is_numa_available() noexcept {
    return false; 
}

inline char* numa_malloc(size_t size) noexcept {
    return nullptr;
}

inline void numa_free(void* ptr, size_t size) noexcept {
    return;
}
}
#else
/*************************************< For Linux >*************************************/
#if defined(__linux__)

#include <numa.h>
#include <sched.h>
#include <stddef.h>

namespace cpuft {

inline bool is_numa_available() {
    return numa_available() >= 0;
}

inline char* numa_malloc(size_t size) {
    return reinterpret_cast<char*>(numa_alloc_onnode(size, numa_node_of_cpu(sched_getcpu())));
}

inline void numa_free(void* ptr, size_t size) {
    ::numa_free(ptr, size);
}

} // namespace cpuft
#endif
#endif
