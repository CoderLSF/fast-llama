/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 @author: Liu Shaofeng
 @date: 2023/10/16
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <sys/time.h>
#include <stdint.h>

#include <functional>
#include <string_view>
#include <span>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace cpuft {

template <typename... Args>
std::string format(const char* fmt, Args... args) {
    size_t size = snprintf(nullptr, 0, fmt, args...) + 1;

    std::unique_ptr<char[]> buf(new char[size]);

    std::string res;
    res.resize(size, '\0');
    snprintf(res.data(), size, fmt, args...);
    return res;
}

std::vector<std::string_view> ssplit(const std::string& s, std::string_view delim);
int stoi(std::string_view s);
float float16_to_float32(uint16_t f16);

template <typename T>
void print_summary(std::string_view prefix, const T& arr) {
    std::cout << prefix << "(size:" << arr.size() << ") [";
    size_t n = arr.size() <= 10 ? arr.size() : 8;
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::setw(5) << arr[i];
    }
    if (arr.size() > n) {
        std::cout << ", ..., " << arr[arr.size() - 1];
    }
    std::cout << "]" << std::endl;
}

class Timer {
public:
    Timer() : _time_us(get_timestamp_us()) {}

    void reset() noexcept {
        _time_us = get_timestamp_us();
    }

    int64_t elapsed_us() const noexcept {
        return get_timestamp_us() - _time_us;
    }

    static int64_t get_timestamp_us() noexcept {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return (int64_t)tv.tv_sec * 1000000 + (int64_t)tv.tv_usec;
    }

private:
    int64_t _time_us;
};

class Defer {
public:
    Defer(std::function<void()> cb) : _cb(cb) {}
    ~Defer() { _cb(); }

private:
    std::function<void()> _cb;
};

} // namespace cpuft
