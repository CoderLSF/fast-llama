#pragma once

/*************************************< For Linux >**************************************/
#if defined(__linux__)
#include <semaphore.h>

namespace cpuft {

class Semaphore {
public:
    Semaphore(int value=0) {
        sem_init(&_sem, 0, value);
    }
    ~Semaphore() {
        sem_destroy(&_sem);
    }

    void acquire() noexcept {
        sem_wait(&_sem);
    }
    void release() noexcept {
        sem_post(&_sem);
    }

private:
    sem_t _sem;
};

} // namespace cpuft

/*************************************< For Linux >**************************************/
#elif defined(__APPLE__)
#include <mach/semaphore.h>
#include <mach/mach.h>

namespace cpuft {

class Semaphore {
public:
    Semaphore(int value=0) {
        semaphore_create(mach_task_self(), &_sem, SYNC_POLICY_FIFO, value);
    }
    ~Semaphore() {
        semaphore_destroy(mach_task_self(), _sem);
    }

    void acquire() noexcept {
        semaphore_wait(_sem);
    }
    void release() noexcept {
        semaphore_signal(_sem);
    }

private:
    semaphore_t _sem;
};

} // namespace cpuft

/*************************************< For Windows >************************************/
#elif defined(_WIN32) || defined(__CYGWIN__)

#include <Windows.h>

namespace cpuft {
  
class Semaphore {  
public:  
    Semaphore(int value=0) {  
        _sem = CreateSemaphore(NULL, value, LONG_MAX, NULL);  
        if (_sem == NULL) {  
            throw std::runtime_error("Unable to create semaphore");  
        }  
    }  
    ~Semaphore() {  
        CloseHandle(_sem);  
    }  
  
    void acquire() noexcept {  
        DWORD result = WaitForSingleObject(_sem, INFINITE);  
        if (result != WAIT_OBJECT_0) {  
            throw std::runtime_error("Unable to acquire semaphore");  
        }  
    }  
    void release() noexcept {  
        if (!ReleaseSemaphore(_sem, 1, NULL)) {  
            throw std::runtime_error("Unable to release semaphore");  
        }  
    }  
  
private:  
    HANDLE _sem;  
};

} // namespace cpuft
#endif

