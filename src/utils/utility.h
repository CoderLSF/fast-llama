/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 @author: Liu Shaofeng
 @date: 2023/10/16
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <stdint.h>

#include <chrono>  
#include <functional>
#include <string_view>
#include <span>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace cpuft {

inline auto min(auto a, auto b) {
    return a <= b ? a : b;
}
inline auto min(auto a, auto b, auto c) {
    return (a <= b && a <= c) ? a : (b <= c ? b : c);
}
inline auto max(auto a, auto b) {
    return a >= b ? a : b;
}
inline auto max(auto a, auto b, auto c) {
    return (a >= b && a >= c) ? a : (b >= c ? b : c);
}

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
    Timer() : _start_time(get_timestamp_us()) {}

    void reset() noexcept {
        _start_time = get_timestamp_us();
    }

    int64_t elapsed_us() const noexcept {
        return get_timestamp_us() - _start_time;
    }

    static int64_t get_timestamp_us() noexcept {
        auto now = std::chrono::high_resolution_clock::now();
        auto now_microseconds = std::chrono::time_point_cast<std::chrono::microseconds>(now);
        auto epoch = now_microseconds.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
    }

private:
    int64_t _start_time = 0;
};

class Defer {
public:
    Defer(std::function<void()> cb) : _cb(cb) {}
    ~Defer() { _cb(); }

private:
    std::function<void()> _cb;
};

} // namespace cpuft
