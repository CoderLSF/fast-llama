/***************************************************************************************
 @Author: Liu Shaofeng
 @Date: 2023/10/16
 ***************************************************************************************/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "malloc_utility.h"

namespace cpuft {

template <typename T, size_t ALIGN_SIZE=1>
struct AlignedMemory {
public:
    AlignedMemory() {}
    AlignedMemory(size_t num_elements, bool use_numa_=false) {
        alloc(num_elements, use_numa_);
    }
    AlignedMemory(AlignedMemory&& other) {
        ptr_          = other.ptr_;
        num_elements_ = other.num_elements_;
        offset_       = other.offset_;
        use_numa_     = other.use_numa_;
        other.ptr_          = nullptr;
        other.num_elements_ = 0;
    }
    AlignedMemory(const AlignedMemory& other) noexcept {
        alloc(other.num_elements_, other.use_numa_);;
        memcpy(ptr_, other.ptr_, sizeof(T) * num_elements_);
    }
    AlignedMemory& operator=(AlignedMemory&& other) noexcept {
        ptr_          = other.ptr_;
        num_elements_ = other.num_elements_;
        offset_       = other.offset_;
        use_numa_     = other.use_numa_;
        other.ptr_          = nullptr;
        other.num_elements_ = 0;
        return *this;
    };
    AlignedMemory& operator=(const AlignedMemory& other) noexcept {
        alloc(other.num_elements_, other.use_numa_);;
        memcpy(ptr_, other.ptr_, sizeof(T) * num_elements_);
        return *this;
    }

    ~AlignedMemory() {
        free();
    }

    T* get() const {
        return ptr_;
    }
    T* data() const {
        return ptr_;
    }
    operator T*() const {
        return ptr_;
    }
    T& operator[](size_t i) const {
        return ptr_[i];
    }
    T* operator->() const {
        return ptr_;
    }
    bool empty() const {
        return ptr_ == nullptr;
    }
    size_t size() const {
        return num_elements_;
    }

    bool alloc(size_t num_elements, bool use_numa=true) {
        free();
        use_numa_ = use_numa;
        if (num_elements <= 0 || ((unsigned)ALIGN_SIZE & ((unsigned)ALIGN_SIZE - 1)) != 0) {
            //fprintf(stderr, "[AlignedMemory::alloc()] invalid parameters\n");
            return false;
        }

        size_t mem_size = sizeof(T) * num_elements + ALIGN_SIZE - 1;
        //fprintf(stderr, "[AlignedMemory::alloc()] mem_size:%lu\n", mem_size);
        if (use_numa_) {
            ptr_ = numa_malloc(mem_size);
        } else {
            ptr_ = reinterpret_cast<T*>(aligned_malloc(mem_size, ALIGN_SIZE));
        }
        if (ptr_ == nullptr) {
            //fprintf(stderr, "[AlignedMemory::alloc()] Insufficient memory for size:%lu\n", mem_size);
            return false;
        }
        if (ALIGN_SIZE > 1 && use_numa) {
            uint64_t addr = reinterpret_cast<uint64_t>(ptr_);
            uint64_t mask = (uint64_t)ALIGN_SIZE - 1;
            if ((addr & mask) != 0) {
                addr = (addr + mask) & ~mask;
                offset_ = static_cast<uint16_t>(addr - reinterpret_cast<uint64_t>(ptr_));
                ptr_    = reinterpret_cast<T*>(addr);
            }
        }

        num_elements_ = num_elements;
        return true;
    }

    void free() {
        if (ptr_ == nullptr) {
            return;
        }
        if (use_numa_) {
            numa_free(reinterpret_cast<void*>(reinterpret_cast<uint64_t>(ptr_) - offset_), sizeof(T)*num_elements_ + ALIGN_SIZE - 1);
        } else {
            aligned_free(ptr_);
        }
        ptr_ = nullptr;
        offset_ = 0;
    }

private:
    T*              ptr_          = nullptr;
    uint32_t        num_elements_ = 0;
    uint16_t        offset_       = 0;
    bool            use_numa_     = false;
};

class SequentialAllocator {
public:
    class AutoRestart {
    public:
        AutoRestart(SequentialAllocator& alloc) : _alloc(alloc) {}
        ~AutoRestart() { _alloc.restart(); }
    private:
        SequentialAllocator& _alloc;
    };

    SequentialAllocator(void* mem) : _mem(mem) {}
    void reset(void* mem) noexcept {
        _mem = mem;
        _pos = 0;
    }

    void* alloc(size_t size) noexcept {
        auto res = reinterpret_cast<char*>(_mem) + _pos;
        _pos += (size + 63) & ~63ul;
        return res;
    }
    void* offset(size_t distance) const noexcept {
        return reinterpret_cast<char*>(_mem) + _pos + distance;
    }
    void advance(size_t distance) noexcept {
        _pos += distance;
    }
    void backward(size_t distance) noexcept {
        _pos -= distance;
    }
    void set_pos(size_t pos) noexcept {
        _pos = pos;
    }
    size_t get_pos() const noexcept {
        return _pos;
    }

    void* get() const noexcept {
        return reinterpret_cast<char*>(_mem) + _pos;
    }

    SequentialAllocator fork() noexcept {
        return SequentialAllocator(reinterpret_cast<char*>(_mem) + _pos);
    }

    void restart() noexcept {
        _pos = 0;
    }
    AutoRestart auto_restart() noexcept {
        return *this;
    }

private:
    size_t  _pos = 0;
    void*   _mem = nullptr;
};

} //namespace cpuft
