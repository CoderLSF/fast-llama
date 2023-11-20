#pragma once

#include "numa_utility.h"

/*************************************< For Linux >*************************************/
#if defined(__linux__)

#include <sched.h>
#include <stddef.h>

namespace cpuft {

inline char* aligned_malloc(size_t size, size_t alignment=64) {
    return reinterpret_cast<char*>(::aligned_alloc(alignment, size));
}
inline void aligned_free(void* ptr) noexcept {
    return free(ptr);
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

} // namespace cpuft
#endif
