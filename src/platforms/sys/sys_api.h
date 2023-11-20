// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Shaofeng Liu (liushaofeng01@baidu.com)
//
// 

#pragma once

#if defined(__linux__) // Linux系统
#include <pthread.h>
#include <sched.h>

namespace cpuft {

inline int get_num_processors() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

inline void bind_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset); // 将线程绑定到第一个CPU核
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Failed to set thread affinity" << std::endl;
    }
}

}
#elif defined(__APPLE__) // Windows系统
#include <sys/sysctl.h>
#include <pthread.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>

namespace cpuft {

inline int get_num_processors() {
    int num_cores = 0;
    size_t len = sizeof(num_cores);
    int name[] = {CTL_HW, HW_AVAILCPU};  // 或使用 HW_NCPU
    if (sysctl(name, 2, &num_cores, &len, NULL, 0) == 0) {
        return num_cores;
    }
    return -1;
}

inline void bind_cpu(int cpu_id) {
    thread_extended_policy_data_t extended_policy;
    thread_affinity_policy_data_t affinity_policy;

    extended_policy.timeshare = 0; // 设置为非时间片共享
    affinity_policy.affinity_tag = cpu_id + 1; // 设置亲和性标签

    // 获取当前线程的mach端口
    thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());

    // 设置线程策略
    if (thread_policy_set(mach_thread, THREAD_EXTENDED_POLICY,
                    (thread_policy_t)&extended_policy, THREAD_EXTENDED_POLICY_COUNT) != KERN_SUCCESS) {
        std::cerr << "Failed to set thread extended policy" << std::endl;
    }

    if (thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                    (thread_policy_t)&affinity_policy, THREAD_AFFINITY_POLICY_COUNT) != KERN_SUCCESS) {
        std::cerr << "Failed to set thread affinity policy" << std::endl;
    }
}

}

#elif defined(_WIN32) || defined(__CYGWIN__)  // Windows系统
#include <Windows.h>

namespace cpuft {

inline int get_num_processors() {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return sysInfo.dwNumberOfProcessors;
}

inline void bind_cpu(int cpu_id) {
    HANDLE hThread = GetCurrentThread();
    DWORD_PTR mask = cpu_id; // 将线程绑定到第一个CPU核
    if (!SetThreadAffinityMask(hThread, mask)) {
        std::cerr << "Failed to set thread affinity" << std::endl;
    }
}

}
#endif
