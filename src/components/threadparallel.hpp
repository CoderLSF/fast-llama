/***************************************************************************************
 @Author: Liu Shaofeng
 @Date: 2023/10/16
 ***************************************************************************************/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#ifdef __linux__ // Linux系统
    #include <pthread.h>
    #include <sched.h>
#elif _WIN32 // Windows系统
    #include <Windows.h>
#endif

#include <atomic>
#include <algorithm>
#include <functional>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <map>

namespace cpuft {

enum class NotifyTreeType {
    PLAIN = 0,
    BTREE,
    SQUARE,
    AUTO,
};

struct ThreadGroupInfo {
    short group_id=0;
    short thread_id=0;
    int   num_threads=0;
};

class ThreadGroupBuilder {
public:
    int add_group(int num_threads) noexcept {
        auto it = _id_map.find(num_threads);
        if (it != _id_map.end()) {
            return it->second;
        }

        int gid = int(_groups.size());
        _id_map[num_threads] = gid;
        _groups.push_back(num_threads);
        return gid;
    }

    int size() const noexcept {
        return int(_groups.size());
    }

    int thread_num(int group_id) const noexcept {
        return _groups[group_id];
    }

    const auto& groups() const noexcept {
        return _groups;
    }
    const std::map<int, int>& id_map() const noexcept {
        return _id_map;
    }

private:
    std::map<int, int>  _id_map;
    std::vector<int>    _groups;
};

template <typename ProcessorClass, typename GlobalData, typename ThreadData>
class ThreadParallel {
public:
    enum class Error {
        OK = 0,
        THREAD_INIT_ERROR,
        INVALID_PARAMS,
    };
    struct NotifyGroup {
        int                               group_id;
        int                               num_threads;
        std::vector<sem_t*>               root;
        std::vector<std::vector<sem_t*>>  tree;
    };


    ~ThreadParallel() {
        _stop = true;
        notify_task(0);
        for (auto& t : _threads) {
            t.join();
        }
    }

    bool init(ProcessorClass* processor, const GlobalData& gd, int num_threads, bool use_numa,
            NotifyTreeType ntt=NotifyTreeType::AUTO, int notify_btsize=4, int notify_max_depth=3) {
        _err = Error::OK;
        if (processor == nullptr) {
            std::cerr << "No processor is specified." << std::endl;
            _err = Error::INVALID_PARAMS;
            return false;
        }
        _processor = processor;

        int num_processors = get_num_processors();
        if (num_processors < 1) {
            std::cerr << "Failed to get number of processors." << std::endl;
            return false;
        }
        if (num_threads <= 0) {
            num_threads = num_processors > 1 ? (num_processors / 2) : 1; 
        }

        std::vector<int> bind_cpus;
        if (use_numa && num_processors > 1) {
            bind_cpus.resize(num_threads, 0);
            if (num_threads == 1) {
                //bind_cpu(0);
                bind_cpus[0] = 1;
            } else if (num_threads >= num_processors) {
                //bind_cpu(0);
                int npc = num_threads / num_processors;
                int rem = num_threads % num_processors;
                for (int i = 0; i < (num_threads - rem) * npc; ++i) {
                    bind_cpus[i] = i / npc;
                }
                for (int i = (num_threads - rem) * npc; i < num_threads; ++i) {
                    bind_cpus[i] = num_processors - rem + i / (npc + 1);
                }
            } else if constexpr (false) {
                //bind_cpu(0);
                int cmid = num_processors / 2;
                int tmid = num_threads / 2;
                int cpu_id = 0;
                int intv = cmid / tmid;
                for (int i = 0; i < tmid; ++i) {
                    bind_cpus[i] = cpu_id;
                    cpu_id += intv;
                }
                cpu_id = cmid;
                intv = (num_processors - cmid) / (num_threads - tmid);
                for (int i = tmid; i < num_threads; ++i) {
                    bind_cpus[i] = cpu_id;
                    cpu_id += intv;
                }
            } else {
                int intv = 1;
                for (; num_threads * 2 * intv <= num_processors; intv *= 2);
                //bind_cpu(intv-1);
                int cpu_id = (num_processors - num_threads * intv) / 2;
                for (int i = 0; i < num_threads; i += 1, cpu_id += intv) {
                    if (i > num_threads / 2 && cpu_id < num_processors/2) {
                        cpu_id = num_processors / 2;
                    }
                    if (cpu_id == 0 && intv > 1) {
                        bind_cpus[i] = intv/2;
                    } else {
                        bind_cpus[i] = cpu_id;
                    }
                }
            }
        }

        _tnis.resize(num_threads);

        {
            ThreadGroupBuilder tgb;
            tgb.add_group(num_threads);
            if (!_processor->parallel_global_init(gd, num_threads, use_numa, tgb)) {
                fprintf(stderr, "Initialization failed\n");
                return false;
            }

            _num_notify_groups = tgb.size();
            _notify_groups.reset(new NotifyGroup[_num_notify_groups]);
            for (int gid = 0; gid < _num_notify_groups; ++gid) {
                auto& ng = _notify_groups[gid];
                ng.num_threads = tgb.thread_num(gid);
                ng.group_id    = gid;
                build_notify_tree(num_threads, ntt, notify_btsize, notify_max_depth, ng);
            }
        }

        _num_processing.store(num_threads);
        _num_threads = num_threads;
        _threads.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            int cpu_id = bind_cpus.empty() ? -1 : bind_cpus[i];
            _threads.emplace_back([this, i, cpu_id, &gd](){ thread_func(gd, i, cpu_id); });
        }

        wait();
        return _err == Error::OK;
    }

    void notify_task(int group_id=0) {
        auto& ng = _notify_groups[group_id];
        _cur_group_id = group_id;
        _num_processing.store((int64_t)ng.num_threads);
        for (auto sem : ng.root) {
            sem_post(sem);
        }
    }
    void wait() {
        std::unique_lock<std::mutex> lock(_done_mtx);
        _done_cv.wait(lock, [this] { return _num_processing.load() == 0; });
    }

protected:
    inline int task_done() noexcept {
        int v = int(_num_processing.fetch_add(-1) - 1);
        if (v == 0) {
            _done_cv.notify_one();
        }
        return v;
    }

    void thread_func(const GlobalData& gd, int thread_id, int cpu_id) {
        if (cpu_id >= 0) {
            //fprintf(stderr, "thread:%3d\t\tcpu:%3d\n", thread_id, cpu_id);
            bind_cpu(cpu_id);
        }
#ifdef SEQUENTIAL_INIT
        if (thread_id > 0) {
            sem_wait(&_tnis[thread_id-1].sem);
        }
#endif

        ThreadData td;
        bool init_ok = _processor->parallel_thread_init(_tnis[thread_id].tgi, gd, td);
#ifdef SEQUENTIAL_INIT
        if (thread_id < _num_threads - 1) {
            sem_post(&_tnis[thread_id].sem);
        }
#endif

        task_done();
        if (!init_ok) {
            _err = Error::THREAD_INIT_ERROR;
            return;
        }

        auto* ptgi = &_tnis[thread_id].tgi;

        struct GroupIdMapNode {
            int                 tid = -1;
            ThreadGroupInfo*    tgi = nullptr;
            sem_t**             sems = nullptr;
            int                 num_children = 0;
        };
        std::unique_ptr<GroupIdMapNode[]> group_info_map(new GroupIdMapNode[_num_notify_groups]);
        for (auto& tgi : *ptgi) {
            auto& gi = group_info_map[tgi.group_id];
            gi.tid = tgi.thread_id;
            gi.tgi = &tgi;

            auto& children = _notify_groups[tgi.group_id].tree[tgi.thread_id];
            gi.sems = children.data();
            gi.num_children = int(children.size());
        }

        sem_t* task_sem = &_tnis[thread_id].sem;
        while (!_stop) {
            sem_wait(task_sem);
            auto& gi = group_info_map[_cur_group_id];
            if (gi.tid >= 0) {
                for (int i = 0; i < gi.num_children; ++i) {
                    sem_post(gi.sems[i]);
                }
                if (!_stop) {
                    _processor->parallel_process_task(td, *gi.tgi);
                }
            }

            task_done();
        }
    }

    void build_notify_tree(int total_thread_num, NotifyTreeType ntt, int btsize, int max_depth, NotifyGroup& ng) {
        if (ntt == NotifyTreeType::AUTO) {
            ntt = NotifyTreeType::BTREE;
            if (ng.num_threads <= 4) {
                btsize = 2;
                max_depth = 2;
            } else if (ng.num_threads < 8) {
                btsize = 3;
                max_depth = 3;
            } else if (ng.num_threads <= 228) {
                btsize = 4;
                max_depth = 2;
            } else {
                btsize = 4;
                max_depth = 3;
            }
        }

        struct Candidate {
            sem_t*      sem = nullptr;
            short       thread_id = 0;
            short       global_id = 0;
        };
        std::vector<Candidate> cands(ng.num_threads);
        int intv = ng.num_threads > 1 ? (total_thread_num - 1) / (ng.num_threads - 1) : 1;
        intv = intv < 1 ? 1 : intv;
        for (int i = 0, tid=0; i < ng.num_threads; ++i, tid += intv) {
            cands[i] = Candidate{&_tnis[tid].sem, short(i), short(tid)};
            _tnis[tid].tgi.emplace_back(short(ng.group_id), short(i), ng.num_threads);
        }

        ng.tree.resize(ng.num_threads);

        if (ng.num_threads < 4) {
            for (int i = 0; i < ng.num_threads; i++) {
                ng.root.push_back(cands[i].sem);
            }
            return;
        }

        if (ntt == NotifyTreeType::BTREE) {
            std::function<void(int, int, int, int)> build_btree = [&](int s, int e, int p, int depth) {
                if (s >= e) {
                    return;
                }
                if (depth >= max_depth) {
                    for (int i = s; i < e; ++i) {
                        if (p < 0) {
                            ng.root.push_back(cands[i].sem);
                        } else {
                            ng.tree[p].push_back(cands[i].sem);
                        }
                    }
                    return;
                }

                const int step = (e - s + btsize - 1) / btsize;
                for (int i = s; i < e; i += step) {
                    int ie = i + step < e ? i + step : e;
                    if (p < 0) {
                        ng.root.push_back(cands[i].sem);
                    } else {
                        ng.tree[p].push_back(cands[i].sem);
                    }
                    build_btree(i + 1, ie, i, depth+1);
                }
            };
            build_btree(0, ng.num_threads, -1, 1);
        } else if (ntt == NotifyTreeType::SQUARE) {
            int num_groups = 1;
            for (; num_groups*num_groups < ng.num_threads; ++num_groups);
            for (int i = 0; i < ng.num_threads; i += num_groups) {
                int cn = i + num_groups < ng.num_threads ? num_groups - 1: ng.num_threads - i - 1;
                ng.root.push_back(cands[i].sem);
                for (int j = i + 1; j < i + 1 + cn; ++j) {
                    ng.tree[i].push_back(cands[j].sem);
                }
            }
        } else {
            for (int i = 0; i < ng.num_threads; ++i) {
                ng.root.push_back(cands[i].sem);
            }
        }
    }

    void bind_cpu(int cpu_id) {
        #ifdef __linux__ // Linux系统
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); // 将线程绑定到第一个CPU核
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Failed to set thread affinity" << std::endl;
        }
        #elif _WIN32 // Windows系统
        HANDLE hThread = GetCurrentThread();
        DWORD_PTR mask = cpu_id; // 将线程绑定到第一个CPU核
        if (!SetThreadAffinityMask(hThread, mask)) {
            std::cerr << "Failed to set thread affinity" << std::endl;
        }
        #endif
    }

    static int get_num_processors() {
        #ifdef __linux__ // Linux系统
        return sysconf(_SC_NPROCESSORS_ONLN);
        #elif _WIN32 // Windows系统
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        return sysInfo.dwNumberOfProcessors;
        #endif
    }

private:
    struct ThreadNotifyInfo {
        sem_t                           sem;
        std::vector<ThreadGroupInfo>    tgi;

        ThreadNotifyInfo() {
            sem_init(&sem, 0, 0);
        }
        ~ThreadNotifyInfo() {
            sem_destroy(&sem);
        }
    };

    std::atomic<int64_t>            _num_processing = 0;
    int                             _cur_group_id;
    int                             _num_notify_groups;
    std::unique_ptr<NotifyGroup[]>  _notify_groups;

    std::mutex                      _done_mtx;
    std::condition_variable         _done_cv;
    bool                            _stop = false;

    std::vector<ThreadNotifyInfo>   _tnis;
    ProcessorClass*                 _processor;
    std::vector<std::thread>        _threads;
    int                             _num_threads = 0;
    Error                           _err = Error::OK;
};

static std::string serialize_vector(const auto& vec) {
    std::string s;
    for (auto v : vec) {
        if (s.size() > 0) {
            s += ", " + std::to_string(v);
        } else {
            s = std::to_string(v);
        }
    }
    return s;
};

} // namespace cpuft

