#pragma once

/*************************************< For Linux >*************************************/
#if defined(__linux__)

#include <numa.h>
#include <sched.h>
#include <stddef.h>

namespace cpuft {

inline char* aligned_malloc(size_t size, size_t alignment=64) {
    return reinterpret_cast<char*>(::aligned_alloc(alignment, size));
}
inline void aligned_free(void* ptr) noexcept {
    return free(ptr);
}

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

/*************************************< For Mac >*************************************/
#elif defined(__APPLE__)

#include <stdio.h>
#include <stdlib.h>

namespace cpuft {

inline char* aligned_malloc(size_t size, size_t alignment=64) {
    printf("aligned_malloc() size:%lu\talignment:%lu\n", size, alignment);
    return reinterpret_cast<char*>(::aligned_alloc(alignment, size));
}
inline void aligned_free(void* ptr) noexcept {
    return free(ptr);
}

inline bool is_numa_available() {
    return false;
}

inline char* numa_malloc(size_t size) {
    printf("numa_alloc() size:%lu\n", size);
    return nullptr;
}

inline void numa_free(void* ptr, size_t size) {
}

}

/*************************************< For Windows >***********************************/
#elif defined(_WIN32) || defined(__CYGWIN__)  // Windows系统

#include <Windows.h>
#include <cstddef>
#include <cstdlib>

namespace cpuft {

inline char* aligned_malloc(size_t size, size_t alignment=64) noexcept {
    return reinterpret_cast<char*>(_aligned_malloc(size, alignment));
}
inline void aligned_free(void* ptr) noexcept {
    return _aligned_free(ptr);
}

inline bool is_numa_available() noexcept {
    return false; 
}

inline char* numa_malloc(size_t size) noexcept {
    return nullptr;
}

inline void numa_free(void* ptr, size_t size) noexcept {
    return;
}

} // namespace cpuft
#endif
