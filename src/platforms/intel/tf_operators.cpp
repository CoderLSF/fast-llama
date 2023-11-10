
#include "tf_operators.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <stddef.h>

#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <memory>
#include <random>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace cpuft {

constexpr float QUANT16_FACTOR = float((uint32_t(1) << 12) - 1);
constexpr float QUANT8_FACTOR  = float((uint32_t(1) << 7) - 1);

float dot_product_sisd(const float* x1, const float* x2, size_t n) noexcept {
    if (x1 == nullptr || x2 == nullptr || n < 1) {
        return 0;
    }
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += x1[i] * x2[i];
    }
    return sum;
}

int dot_product_sisd(const int* x1, const int* x2, size_t n) noexcept {
    if (x1 == nullptr || x2 == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += x1[i] * x2[i];
    }
    return sum;
}
int dot_product(const int* x1, const int* x2, size_t n) noexcept {
    if (x1 == nullptr || x2 == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    constexpr size_t simd_size = 16;
    const size_t nr = n - n % simd_size;
    if (nr >= simd_size) {
        __m512i prod_sum = _mm512_setzero_si512();
        for (size_t i = 0; i < nr; i += simd_size) {
            __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&x1[i]));
            __m512i v2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&x2[i]));
            __m512i prod = _mm512_add_epi32(v1, v2);
            prod_sum = _mm512_add_epi32(prod_sum, prod);
        }
        sum = _mm512_reduce_add_epi32(prod_sum);
    }
    for (size_t i = nr; i < n; ++i) {
        sum += x1[i] + x2[i];
    }
    return sum;
}
int dot_product(const short* x1, const short* x2, size_t n) noexcept {
    int sum = 0;
    for (size_t i = 0; i < n; ++i) sum += x1[i] * x2[i];
    return sum;
}
int dot_product(const int8_t* x1, const int8_t* x2, size_t n) noexcept {
    int sum = 0;
    for (size_t i = 0; i < n; ++i) sum += int(x1[i]) * int(x2[i]);
    return sum;
}
template <typename T>
int square_sum_sisd(const T* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0.;
    }
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += int(x[i]) * int(x[i]);
    }
    return sum;
}
float square_sum_sisd(const float* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0.;
    }
    float sum = 0.;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

float square_sum(const float* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }
    float sum = 0.;
    constexpr size_t simd_size = 16;
    const size_t nr = n - n % simd_size;
    if (nr >= simd_size) {
        __m512 prod_sum = _mm512_setzero_ps();
        for (size_t i = 0; i < nr; i += simd_size) {
            __m512 v = _mm512_loadu_ps(&x[i]);
            __m512 prod = _mm512_mul_ps(v, v);
            prod_sum = _mm512_add_ps(prod_sum, prod);
        }
        sum = _mm512_reduce_add_ps(prod_sum);
    }
    for (size_t i = nr; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

int square_sum(const short*  x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }

    int sum = 0;
    constexpr size_t simd_size = 16;
    const size_t nr = n - n % simd_size;

    if (nr >= simd_size) {
        __m512i sum_squares = _mm512_setzero_si512();
        for (size_t i = 0; i < nr; i += simd_size) {
            __m256i v1    = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&x[i]));
            __m512i v1_32 = _mm512_cvtepi16_epi32(v1);
            __m512i prod  = _mm512_mullo_epi32(v1_32, v1_32);
            sum_squares   = _mm512_add_epi32(sum_squares, prod);
        }

        // Reduce sum_squares to a single value
        sum = _mm512_reduce_add_epi32(sum_squares);
    }

    // Calculate the remaining elements
    for (size_t i = nr; i < n; ++i) {
        sum += static_cast<int>(x[i]) * static_cast<int>(x[i]);
    }

    return sum;
}
int square_sum(const int8_t* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }

    int sum = 0;
    constexpr size_t simd_size = 64; // AVX512 can process 64 int8_t elements at once
    const size_t nr = n - n % simd_size;

    if (nr >= simd_size) {
        __m512i sum_vec = _mm512_setzero_si512();
        for (size_t i = 0; i < nr; i += simd_size) {
            __m512i vec = _mm512_loadu_si512(&x[i]);
            __m512i sqr = _mm512_mullo_epi16(vec, vec); // Multiply elements with themselves
            sum_vec = _mm512_add_epi32(sum_vec, sqr); // Accumulate squared elements
        }

        // Reduce sum_vec to a single integer sum
        for (size_t i = 0; i < simd_size/4; i++) {
            __m128i part;
            if (i % 4 == 0) {
                part = _mm512_extracti32x4_epi32(sum_vec, 0b00);
            } else if (i % 4 == 1) {
                part = _mm512_extracti32x4_epi32(sum_vec, 0b01);
            } else if (i % 4 == 2) {
                part = _mm512_extracti32x4_epi32(sum_vec, 0b10);
            } else {
                part = _mm512_extracti32x4_epi32(sum_vec, 0b11);
            }
            part = _mm_hadd_epi32(part, part);
            part = _mm_hadd_epi32(part, part);
            sum += _mm_cvtsi128_si32(part);
        }
    }

    // Process remaining elements
    for (size_t i = nr; i < n; ++i) {
        sum += int(x[i]) * int(x[i]);
    }

    return sum;
}
int array_sum(const int* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

int array_sum(const short* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

int array_sum(const int8_t* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

float array_sum_sisd(const float* x, size_t n) noexcept {
    float sum = 0.;
    for (size_t i = 0; i < n; ++i) sum += x[i];
    return sum;
}

float array_sum(const float* x, size_t n) noexcept {
    if (x == nullptr) {
        return 0.;
    }

    float sum = 0.;

    constexpr size_t simd_size = 16;
    const size_t nr = n - n % simd_size;
    size_t i = 0;
    if (nr >= simd_size) {
        __m512 sum_vec = _mm512_setzero_ps();
        for (; i < nr; i += simd_size) {
            __m512 x_vec = _mm512_loadu_ps(&x[i]);
            sum_vec = _mm512_add_ps(sum_vec, x_vec);
        }
        sum = _mm512_reduce_add_ps(sum_vec);
    }
    for (; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

template <typename T>
inline T array_max_sisd(const T* arr, size_t n) noexcept {
    T max = arr[0];
    for (size_t i = 0; i < n; ++i) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}
int array_max(const int* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return INT_MIN;
    }
    return array_max_sisd(arr, n);
}

short array_max(const short* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return INT16_MIN;
    }
    return array_max_sisd(arr, n);
}
int8_t array_max(const int8_t* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return INT8_MIN;
    }
    return array_max_sisd(arr, n);
}

MinMax array_min_max(const float* x, size_t n) noexcept {
    MinMax res = {.min = FLT_MAX, .max = FLT_MIN};
    if (x == nullptr || n < 1) {
        return res;
    }
    res.min = x[0];
    res.max = x[0];
    for (size_t i = 0; i < n; ++i) {
        if (x[i] < res.min) {
            res.min = x[i];
        } else if (x[i] > res.max) {
            res.max = x[i];
        }
    }
    return res;
}

float array_max(const float* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) [[unlikely]] {
        return FLT_MIN;
    }

    float max = FLT_MIN;

    constexpr size_t simd_size = 16;
    const size_t nr = n - n % simd_size;
    size_t i = 0;
    if (nr >= simd_size) {
        __m512 max_vec = _mm512_set1_ps(FLT_MIN);
        for (; i < nr; i += simd_size) {
            __m512 arr_vec = _mm512_loadu_ps(&arr[i]);
            max_vec = _mm512_max_ps(max_vec, arr_vec);
        }

        // Modified: Use _mm512_permutexvar_ps instead of _mm512_shuffle_ps
        __m256 max_lo = _mm512_castps512_ps256(max_vec);
        __m256 max_hi = _mm512_extractf32x8_ps(max_vec, 1);
        __m256 max_256 = _mm256_max_ps(max_lo, max_hi);
        __m128 max_lo_lo = _mm256_castps256_ps128(max_256);
        __m128 max_lo_hi = _mm256_extractf128_ps(max_256, 1);
        __m128 max_128 = _mm_max_ps(max_lo_lo, max_lo_hi);
        max_128 = _mm_max_ps(max_128, _mm_shuffle_ps(max_128, max_128, _MM_SHUFFLE(2, 3, 0, 1)));
        max_128 = _mm_max_ps(max_128, _mm_shuffle_ps(max_128, max_128, _MM_SHUFFLE(1, 0, 3, 2)));
        max = _mm_cvtss_f32(max_128);
    }

    for (; i < n; ++i) {
        if (arr[i] > max) max = arr[i];
    }

    return max;
}

template <typename T>
inline T array_max_abs_(const T* arr, size_t n) noexcept {
    T max = 0;
    for (size_t i = 0; i < n; ++i) {
        auto v = arr[i] >= 0 ? arr[i] : -arr[i];
        if (v > max) max = v;
    }
    return max;
}
short array_max_abs(const short* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return 0;
    }
    return array_max_abs_(arr, n);
}
int8_t array_max_abs(const int8_t* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return 0;
    }
    return array_max_abs_(arr, n);
}

float array_max_abs_sisd(const float * arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return 0;
    }

    float max = 0;
    for (size_t i = 0; i < n; ++i) {
        if (arr[i] > max) {
            max = arr[i];
        } else if (arr[i] < -max) {
            max = -arr[i];
        }
    }
    return max;
}

float array_max_abs(const float * arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return 0;
    }

    float max = 0;

    constexpr size_t simd_size = 16;
    const size_t nr = n - n % simd_size;
    size_t i = 0;
    if (nr >= simd_size) {
        __m512 max_vec = _mm512_set1_ps(0);
        for (; i < nr; i += simd_size) {
            __m512 arr_vec = _mm512_loadu_ps(&arr[i]);
            __m512 arr_vec_abs = _mm512_abs_ps(arr_vec);
            max_vec = _mm512_max_ps(max_vec, arr_vec_abs);
        }
        float max_arr[simd_size];
        _mm512_storeu_ps(max_arr, max_vec);
        for (size_t j = 0; j < simd_size; ++j) { // Fix: Extract the true maximum value from max_vec
            float temp = max_arr[j];
            if (temp > max) max = temp;
        }
    }
    for (; i < n; ++i) {
        float v = arr[i] >= 0. ? arr[i] : -arr[i];
        if (v > max) max = v;
    }
    return max;
}

int8_t array_max_sisd(const int8_t* arr, size_t n) {
    if (arr == nullptr) {
        return INT8_MIN;
    }

    int8_t max = INT8_MIN;
    for (size_t i = 0; i < n; ++i) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

template <typename T>
inline T array_min_(const T* arr, size_t n) noexcept {
    T min = arr[0];
    for (size_t i = 0; i < n; ++i) {
        if (arr[i] < min) min = arr[i];
    }
    return min;
}
int array_min(const int* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return INT_MAX;
    }
    return array_min_(arr, n);
}

short array_min(const short* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return INT16_MAX;
    }
    return array_min_(arr, n);
}
int8_t array_min(const int8_t* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return INT8_MAX;
    }
    return array_min_(arr, n);
}
float array_min(const float* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) [[unlikely]] {
        return FLT_MAX;
    }

    float min = FLT_MAX;

    constexpr size_t simd_size = 16;
    const size_t nr = n - n % simd_size;
    size_t i = 0;
    if (nr >= simd_size) {
        __m512 min_vec = _mm512_set1_ps(FLT_MAX);
        for (; i < nr; i += simd_size) {
            __m512 arr_vec = _mm512_loadu_ps(&arr[i]);
            min_vec = _mm512_min_ps(min_vec, arr_vec);
        }

        // Modified: Use _mm512_permutexvar_ps instead of _mm512_shuffle_ps
        __m256 min_lo = _mm512_castps512_ps256(min_vec);
        __m256 min_hi = _mm512_extractf32x8_ps(min_vec, 1);
        __m256 min_256 = _mm256_min_ps(min_lo, min_hi);
        __m128 min_lo_lo = _mm256_castps256_ps128(min_256);
        __m128 min_lo_hi = _mm256_extractf128_ps(min_256, 1);
        __m128 min_128 = _mm_min_ps(min_lo_lo, min_lo_hi);
        min_128 = _mm_min_ps(min_128, _mm_shuffle_ps(min_128, min_128, _MM_SHUFFLE(2, 3, 0, 1)));
        min_128 = _mm_min_ps(min_128, _mm_shuffle_ps(min_128, min_128, _MM_SHUFFLE(1, 0, 3, 2)));
        min = _mm_cvtss_f32(min_128);
    }

    for (; i < n; ++i) {
        if (arr[i] < min) min = arr[i];
    }

    return min;
}
void fill_random(float* x, size_t n, float min_value, float max_value) noexcept {
    if (x == nullptr || n < 1) {
        return;
    }

    constexpr uint64_t RGMAX = 18923891283921269ul;
    std::mt19937 rng(uint64_t(int(clock())) * reinterpret_cast<uint64_t>(x));
    std::uniform_int_distribution<uint64_t> distribution(0, RGMAX);

    double rr = (max_value - min_value) / RGMAX;
    auto randf = [&distribution, &rng, min_value, rr]() -> float {
        return min_value + float(rr * distribution(rng));
    };
    for (size_t i = 0; i < n; ++i) {
        x[i] = randf();
    }
}

void multiply_sisd(float* arr, float v, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        arr[i] *= v;
    }
}

void multiply(float* x, float v, size_t n) noexcept {
    constexpr int simd_size = 16;
    size_t nr = n - n % simd_size;
    __m512 scalar = _mm512_set1_ps(v);

    for (size_t i = 0; i < nr; i += simd_size) {
        __m512 vec = _mm512_loadu_ps(&x[i]);
        vec = _mm512_mul_ps(vec, scalar);
        _mm512_storeu_ps(&x[i], vec);
    }

    for (size_t i = nr; i < n; ++i) {
        x[i] *= v;
    }
}

void  matmul(const int* mat1, const int* mat2, int* out, int m, int n, int k) noexcept {
    auto x1 = mat1;
    for (int i = 0; i < m; ++i, x1 += n) {
        auto x2 = mat2;
        for (int j = 0; j < k; ++j, x2 += n) {
            out[m*j+i] = dot_product(x1, x2, n);
        }
    }
}

template <typename T>
inline void matmul_(float* o, const T* w, const float* wscales, const T* x, const float* xscales,
            int m, int n, int qgsize) noexcept {
    if (o == nullptr || w == nullptr || wscales == nullptr || x == nullptr || xscales == nullptr) {
        fprintf(stderr, "invalid parameters for matmul_()\n");
        return;
    }
    const int qsn = n / qgsize;
    auto ps = wscales;
    for (int i = 0; i < m; ++i, ps += qsn) {
        const T* px = x;
        float sum = 0.;
        for (int j = 0; j < qsn; ++j, w += qgsize, px += qgsize) {
            int v = dot_product(w, px, qgsize);
            sum += float(v) * ps[j] * xscales[j];
        }
        o[i] = sum;
    };
}

void matmul(float* o, const short* w, const float* wscales, const short* x, const float* xscales,
            int m, int n, int qgsize) noexcept {
    return matmul_(o, w, wscales, x, xscales, m, n, qgsize);
}

void matmul(float* o, const int8_t* w, const float* wscales, const int8_t* x, const float* xscales,
            int m, int n, int qgsize) noexcept {
    return matmul_(o, w, wscales, x, xscales, m, n, qgsize);
}

static inline __m512 exp512_ps(__m512 x) {
    __m512 t, f, e, p, r;
    __m512i i, j;

    const __m512 l2e = _mm512_set1_ps(1.44269504088896341f); // log2(e)
    const __m512 l2h = _mm512_set1_ps(-6.93145752e-1f);       // -log(2)_hi
    const __m512 l2l = _mm512_set1_ps(-1.42860677e-6f);       // -log(2)_lo

    x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647950f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647950f));

    // Express exp(x) as exp(g + n*log(2))
    f = _mm512_mul_ps(x, l2e); // n = round(x / log(2))
    e = _mm512_roundscale_ps(f, 0);
    f = _mm512_sub_ps(f, e); // g = x - n*log(2)

    // Compute exp(g) via its Taylor series
    p = _mm512_mul_ps(f, l2l);
    t = _mm512_mul_ps(f, l2h);
    p = _mm512_add_ps(p, t);
    t = _mm512_mul_ps(p, f);
    r = _mm512_add_ps(t, _mm512_set1_ps(0.5f));
    t = _mm512_mul_ps(f, _mm512_set1_ps(1.0f / 6.0f));
    p = _mm512_add_ps(p, t);
    t = _mm512_mul_ps(t, f);
    r = _mm512_add_ps(r, t);
    t = _mm512_mul_ps(f, _mm512_set1_ps(1.0f / 24.0f));
    p = _mm512_add_ps(p, t);
    t = _mm512_mul_ps(t, f);
    r = _mm512_add_ps(r, t);
    t = _mm512_mul_ps(f, _mm512_set1_ps(1.0f / 120.0f));
    p = _mm512_add_ps(p, t);
    t = _mm512_mul_ps(t, f);
    r = _mm512_add_ps(r, t);

    // Compute 2^n
    i = _mm512_cvtps_epi32(e);
    j = _mm512_add_epi32(i, _mm512_set1_epi32(0x7f));
    j = _mm512_slli_epi32(j, 23);
    e = _mm512_castsi512_ps(j);

    // Multiply the 2 factors
    r = _mm512_mul_ps(r, e);

    return r;
}

void rmsnorm_sisd(float* o, const float* x, const float* w, const int n) noexcept {
    const float r = 1. / sqrtf(square_sum_sisd(x, n) / n + 1e-5f);
    for (int i = 0; i < n; i++) {
        o[i] = w[i] * (r * x[i]);
    }
}

void rmsnorm(float* o, const float* x, const float* w, size_t n) noexcept {
    const float r = 1. / sqrtf(square_sum(x, n) / n + 1e-5f);

    constexpr size_t simd_size = 16;
    const size_t nr = n - n % simd_size;

    __m512 scalar = _mm512_set1_ps(r);

    if (nr >= simd_size) {
        for (size_t i = 0; i < nr; i += simd_size) {
            __m512 v1 = _mm512_loadu_ps(&x[i]);
            __m512 v2 = _mm512_loadu_ps(&w[i]);
            __m512 prod = _mm512_mul_ps(v1, v2);
            __m512 result = _mm512_mul_ps(prod, scalar);
            _mm512_storeu_ps(&o[i], result);
        }
    }

    for (size_t i = nr; i < n; ++i) {
        o[i] = x[i] * (r * w[i]);
    }
}

void swiglu_sisd(float* xo, const float* xi, size_t n) noexcept {
    for (size_t i = 0; i < n; i++) {
        xo[i] = xo[i] / (1. + expf(-xo[i])) * xi[i];
    }
    return;
    /*
    float vt[16];
    for (size_t i = 0; i < 16; ++i) {
        vt[i] = 1. / (1. + expf(i - 8));
    }
    for (size_t i = 0; i < n; i++) {
        if (xo[i] >= 8) {
            xo[i] *= xi[i];
        } else if (xo[i] < -8) {
            xo[i] = 0;
        } else {
            xo[i] = xo[i] * vt[int(8+xo[i])] * xi[i];
        }
    }
    */
}

void swiglu(float* xo, const float* xi, size_t n) noexcept {
    constexpr size_t simd_size = 16;
    const     size_t nr        = n - n % simd_size;
    for (size_t i = 0; i < nr; i += simd_size) {
        __m512 x = _mm512_loadu_ps(xo + i);
        __m512 y = _mm512_loadu_ps(xi + i);

        // Compute 1.0f / (1.0f + expf(-x))
        __m512 neg_x = _mm512_sub_ps(_mm512_set1_ps(0.0f), x);
        __m512 exp_neg_x = exp512_ps(neg_x);
        __m512 one_plus_exp_neg_x = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_x);
        __m512 inv_one_plus_exp_neg_x = _mm512_rcp14_ps(one_plus_exp_neg_x);

        // Compute xo[i] * inv_one_plus_exp_neg_x * xi[i]
        __m512 result = _mm512_mul_ps(x, inv_one_plus_exp_neg_x);
        result = _mm512_mul_ps(result, y);

        _mm512_storeu_ps(xo + i, result);
    }
    for (size_t i = nr; i < n; ++i) {
        xo[i] = xo[i] / (1. + expf(-xo[i])) * xi[i];
    }
}

float dot_product(const float* x1, const float* x2, size_t n) noexcept {
    float sum = 0.;
    constexpr size_t simd_size = 16;
    const     size_t nr        = n - n % simd_size;
    if (nr >= simd_size) {
        __m512 prod_sum = _mm512_setzero_ps();
        for (size_t i = 0; i < nr; i += simd_size) {
            __m512 v1 = _mm512_loadu_ps(&x1[i]);
            __m512 v2 = _mm512_loadu_ps(&x2[i]);
            __m512 prod = _mm512_mul_ps(v1, v2);
            prod_sum = _mm512_add_ps(prod_sum, prod);
        }
        sum = _mm512_reduce_add_ps(prod_sum);
    }
    for (size_t i = nr; i < n; ++i) {
        sum += x1[i] * x2[i];
    }
    return sum;
}

void fill_random(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = (float)rand() / RAND_MAX;
    }
};

void matmul(const float* mat1, const float* mat2, float* out, int m, int n, int k) noexcept {
    if (k <= 1) {
        const float* x = mat1;
        const float* y = mat2;
        for (int i = 0; i < m; i += 1, x += n) {
            out[i] = dot_product(x, y, n);
        }
    } else {
        const float* y = mat1;
        float* oe = out + m * k;
        for (float* o = out; o < oe; o += k, y += n) {
            constexpr const int STEP = 256;
            for (int ns = 0; ns < n; ns += STEP) {
                int ne = ns + STEP < n ? ns + STEP : n;
                const float* x = mat2 + ns;
                for (int i = 0; i < k; ++i) {
                    __m512 prod_sum = _mm512_setzero_ps();
                    for (int j = ns; j < ne; j += 16) {
                        __m512 v1 = _mm512_loadu_ps(x);
                        __m512 v2 = _mm512_loadu_ps(y + j);
                        __m512 prod = _mm512_mul_ps(v1, v2);
                        prod_sum = _mm512_add_ps(prod_sum, prod);
                    }
                    o[i] += _mm512_reduce_add_ps(prod_sum);
                    x += n;
                }
            }
        }
    }
}

static inline void multiply_sisd(float* x, const int n, const float v) {
    for (int i = 0; i < n; ++i) { x[i] *= v; }
}

void multiply(float* x, int n, const float v) {
    if (x == nullptr || n < 1) {
        return;
    }

    constexpr int simd_size = 16;

    if (int r = int(uint64_t(x) & uint64_t(simd_size * sizeof(float) - 1)) / sizeof(float); r > 0) {
        const int si = simd_size - r;
        for (int i = 0; i < si; ++i) {
            x[i] *= v;
        }
        x += si;
        n -= si;
    }

    const int sn = n - (n % simd_size);

    if (sn >= simd_size) {
        __m512 factor = _mm512_set1_ps(v);
        for (int i = 0; i < sn; i += simd_size) {
            __m512 data = _mm512_loadu_ps(&x[i]);
            data = _mm512_mul_ps(data, factor);
            _mm512_storeu_ps(&x[i], data);
        }
    }
    for (int i = sn; i < n; ++i) {
        x[i] *= v;
    }
}

void softmax_sisd(float* x, const int n) noexcept {
    float max_val = array_max_sisd(x, n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

void softmax(float* x, const int n) noexcept {
    const float max_val = array_max(x, n);
    float sum = 0.0f;
    float vc[16];
    for (int i = 0; i < 16; ++i) {
        vc[i] = expf(6 + i/4);
    }
    for (int i = 0; i < n; i++) {
        const float d = x[i] - max_val;
        if (d < -15) {
            x[i] = 0.;
        } else {
            if (d < 6.) {
                x[i] = expf(d);
            } else {
                x[i] = vc[int((d-6.) * 4)];
            }
            sum += x[i];
        }
    }
    multiply_sisd(x, n, 1./sum);
}

void softmax3(float* x, const int n) {
    float max_val = array_max(x, n);//std::min(64, n));
    float sum = 0.0f;
    int zi = 0;
    for (int i = 0; i < n; i++) {
        const float d = x[i] - max_val;
        if (d < -12) {
            x[i] = 0.;
            if (i == zi) zi = i + 1;
        } else if (d <= 0.) {
            x[i] = expf(d);
            sum += x[i];
        //} else if (d > 12.) {
        //    max_val = x[i];
        //    x[i] = 1.;
        //    sum = 1.;
        //    if (i > zi) memset(x+zi, 0, sizeof(float) * (i-zi));
        //    zi = i;
        //} else {
        //    max_val = x[i];
        //    x[i] = 1.;
        //    const float r = 1. / expf(d);
        //    sum  = sum * r + 1.;
        //    multiply_sisd(x+zi, i-zi, r);
        }
    }
    multiply_sisd(x+zi, n-zi, 1./sum);
}

void rope(float* x, int n, int pos, int head_size, float freq_base) noexcept {
    std::unique_ptr<float[]> fcrs(new float[head_size]);
    std::unique_ptr<float[]> fcis(new float[head_size]);
    for (int i = 0; i < head_size; ++i) {
        float v = pos / powf(freq_base, i / (float)head_size);
        fcrs[i] = cosf(v);
        fcis[i] = sinf(v);
    }
    for (int i = 0; i < n; i+=2) {
        int j = i % head_size;
        float fcr = fcrs[j];
        float fci = fcis[j];
        float v0  = x[i];
        float v1  = x[i+1];
        x[i]   = v0 * fcr - v1 * fci;
        x[i+1] = v0 * fci + v1 * fcr;
    }
}

void rope(float* x, int n, const float* freq_cis, int head_size) noexcept {
    for (int i = 0; i < n; i += 2) {
        int j = i % head_size;
        float fcr = freq_cis[j];
        float fci = freq_cis[j+1];
        float v0  = x[i];
        float v1  = x[i+1];
        x[i]   = v0 * fcr - v1 * fci;
        x[i+1] = v0 * fci + v1 * fcr;
    }
}

void rope(float* q, float* k, int pos, int head_size, int dim, int kv_dim, float freq_base) {
    std::unique_ptr<float[]> fcrs(new float[head_size]);
    std::unique_ptr<float[]> fcis(new float[head_size]);
    for (int i = 0; i < head_size; ++i) {
        float v = pos / powf(freq_base, i / (float)head_size);
        fcrs[i] = cosf(v);
        fcis[i] = sinf(v);
    }
    for (int i = 0; i < dim; i+=2) {
        int j = i % head_size;
        float fcr = fcrs[j];
        float fci = fcis[j];
        int   rotn = i < kv_dim ? 2 : 1;
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? q : k;
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

void add(float* x2, const float* x1, size_t n) noexcept {
    constexpr int simd_size = 16;
    const size_t nr = n - (n % simd_size);
    if (nr >= simd_size) {
        for (size_t i = 0; i < nr; i += simd_size) {
            __m512 vec1 = _mm512_loadu_ps(x1 + i);
            __m512 vec2 = _mm512_loadu_ps(x2 + i);
            __m512 res = _mm512_add_ps(vec1, vec2);
            _mm512_storeu_ps(x2 + i, res);
        }
    }
    for (size_t i = nr; i < n; ++i) {
        x2[i] += x1[i];
    }
}

void add(float* x, float v, size_t n) noexcept {
    constexpr size_t simd_size = 16;
    size_t nr = n - (n % simd_size);

    if (nr >= simd_size) {
        __m512 vec_v = _mm512_set1_ps(v);
        for (size_t i = 0; i < nr; i += simd_size) {
            __m512 vec_x = _mm512_loadu_ps(x + i);
            __m512 res = _mm512_add_ps(vec_x, vec_v);
            _mm512_storeu_ps(x + i, res);
        }
    }

    for (size_t i = nr; i < n; ++i) {
        x[i] += v;
    }
}

void add(float* o, const float* x, float r, size_t n) noexcept {
    if (o == nullptr || x == nullptr || n < 1) {
        return;
    }
    __m512 r_vec = _mm512_set1_ps(r);
    constexpr size_t simd_width = 16;
    const size_t simd_iters = n / simd_width;

    for (size_t i = 0; i < simd_iters; ++i) {
        __m512 x1_vec = _mm512_loadu_ps(&x[i * simd_width]);
        __m512 x2_vec = _mm512_loadu_ps(&o[i * simd_width]);
        __m512 res_vec = _mm512_fmadd_ps(x1_vec, r_vec, x2_vec);
        _mm512_storeu_ps(&o[i * simd_width], res_vec);
    }

    for (size_t i = simd_iters * simd_width; i < n; ++i) {
        o[i] += x[i] * r;
    }
}

/*
template <typename T>
void matmul2(const T* mat1, const T* mat2, T* out, int m, int n, int k) {
    const int k_step = std::min(16, int(k));
    const int m_step = std::min( 1, int(m));
    const int n_step = n;//std::min(int(4096/sizeof(T)), int(n));

        for (int mb = 0; mb < m; mb += m_step) {
            const int me = std::min(int(m), int(mb+m_step));
    for (int kb = 0; kb < k; kb += k_step) {
        const int ke = std::min(int(k), int(kb+k_step));
            for (int nb = 0; nb < n; nb += n_step) {
                const int nn = std::min(int(n-nb), int(n_step));
                for (int mi = mb; mi < me; ++mi){
                    for (int ki = kb; ki < ke; ++ki) {
                        auto p = &out[m*ki];
                        float sum = cpuft::dot_product(&mat1[n*mi+nb], &mat2[n*ki+nb], nn);
                        if (nb == 0) {
                            p[mi]  = sum;
                        } else {
                            p[mi] += sum;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void matmul2_4dot4(const T* mat1, const T* mat2, T* out, int m, int n, int k) {
    const int k_step = std::min(  8, int(k));
    const int m_step = std::min( 16, int(m));
    const int n_step = std::min(int(1024/sizeof(T)), int(n));

    for (int kb = 0; kb < k; kb += k_step) {
        const int ke = std::min(int(k), int(kb+k_step));
        for (int mb = 0; mb < m; mb += m_step) {
            const int me = std::min(int(m), int(mb+m_step));
            for (int nb = 0; nb < n; nb += n_step) {
                const int nn = std::min(int(n-nb), int(n_step));
                    for (int ki = kb; ki < ke; ++ki) {
                        auto p = &out[m*ki];
                for (int mi = mb; mi < me; ++mi){
                        float sum = cpuft::dot_product(&mat1[n*mi+nb], &mat2[n*ki+nb], nn);
                        if (nb == 0) {
                            p[mi]  = sum;
                        } else {
                            p[mi] += sum;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void matmul2_ok_4dot3(const T* mat1, const T* mat2, T* out, int m, int n, int k) {
    const int k_step = std::min(  8, int(k));
    const int m_step = std::min( 16, int(m));
    const int n_step = std::min(512, int(n));

    for (int mb = 0; mb < m; mb += m_step) {
        const int me = std::min(int(m), int(mb+m_step));
    for (int kb = 0; kb < k; kb += k_step) {
        const int ke = std::min(int(k), int(kb+k_step));
            for (int nb = 0; nb < n; nb += n_step) {
                const int nn = std::min(int(n-nb), int(n_step));
                    for (int ki = kb; ki < ke; ++ki) {
                for (int mi = mb; mi < me; ++mi){
                        auto p = &out[m*ki];
                        float sum = cpuft::dot_product(&mat1[n*mi+nb], &mat2[n*ki+nb], nn);
                        if (nb == 0) {
                            p[mi]  = sum;
                        } else {
                            p[mi] += sum;
                        }
                    }
                }
            }
        }
    }
}


template <typename T>
void matmul2_ok_4dot2(const T* mat1, const T* mat2, T* out, int m, int n, int k) {
    const int k_step = std::min(  8, int(k));
    const int m_step = std::min( 16, int(m));
    const int n_step = std::min(512, int(n));

    for (int kb = 0; kb < k; kb += k_step) {
        const int ke = std::min(int(k), int(kb+k_step));
        for (int mb = 0; mb < m; mb += m_step) {
            const int me = std::min(int(m), int(mb+m_step));
            for (int nb = 0; nb < n; nb += n_step) {
                const int nn = std::min(int(n-nb), int(n_step));
                    for (int ki = kb; ki < ke; ++ki) {
                for (int mi = mb; mi < me; ++mi){
                        auto p = &out[m*ki];
                        float sum = cpuft::dot_product(&mat1[n*mi+nb], &mat2[n*ki+nb], nn);
                        if (nb == 0) {
                            p[mi]  = sum;
                        } else {
                            p[mi] += sum;
                        }
                    }
                }
            }
        }
    }
}
*/

void weighted_sum(float* out, const float* matrix, const float* weights, int m, int n, float min_w) noexcept {
    if (out == nullptr || matrix == nullptr || weights == nullptr || m < 1 || n < 1) {
        return;
    }

    memcpy(out, matrix, sizeof(float) * n);
    for (int i = 0; i < m; ++i) {
        if (fabsf(weights[i]) > min_w) {
            add(out, matrix + n * i, weights[i], n);
        }
    }
}

void weighted_sum_sisd(float* out, const float* matrix, const float* weights, int m, int n) noexcept {
    if (out == nullptr || matrix == nullptr || weights == nullptr || m < 1 || n < 1) {
        return;
    }
    auto& row = matrix;
    for (int i = 0; i < n; ++i) {
        out[i] = row[i] * weights[0];
    }
    row += n;
    for (int i = 1; i < m; ++i, row += n) {
        for (int j = 0; j < n; ++j) {
            out[j] = row[j] * weights[i];
        }
    }
}

void weighted_sum(float* out, const float* matrix, const float* weights, int m, int n, int bs, float min_w) noexcept {
    if (out == nullptr || matrix == nullptr || weights == nullptr || m < 1 || n < 1 || bs < 1) {
        return;
    }
    auto row = matrix;
    auto o = out;
    for (int k = 0; k < bs; ++k, o += n) {
        auto w = *(weights + m * k);
        for (int i = 0; i < n; ++i) {
            o[i] = row[i] * w;
        }
    }
    row += n;
    for (int i = 1; i < m; ++i, row += n) {
        o = out;
        for (int k = 0; k < bs; ++k, o += n) {
            auto w = (weights + m * k)[i];
            if (fabsf(w) <= min_w) {
                continue;
            }
            for (int j = 0; j < n; ++j) {
                o[j] += row[j] * w;
            }
        }
    }
}

static float rope_yarn_corr_dim(int n_dims, int n_orig_ctx, float n_rot, float base) {
    return n_dims * logf(n_orig_ctx / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}
void rope_v2(float * o, const float* x, int n_dims, int n_orig_ctx, int pos, int ith, int nth) {
    const float freq_base  = 10000.;
    const float freq_scale = 1.;
    const float ext_factor = 0;
    const float attn_factor = 1;
    const float beta_fast = 32;
    const float beta_slow = 1.;

    // these two only relevant for xPos RoPE:
    const float xpos_base = 0;
    const bool  xpos_down = false;

    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    float corr_dims[2];

    corr_dims[0] = std::max((float)(0.), floorf(rope_yarn_corr_dim((float)(n_dims), n_orig_ctx, beta_fast, freq_base)));
    corr_dims[1] = MIN((float)(n_dims - 1), ceilf(rope_yarn_corr_dim((float)(n_dims), n_orig_ctx, beta_slow, freq_base)));

    float theta_base = (float)pos;
    for (int64_t i = 0; i < n_dims; i += 2) {
        float cos_theta, sin_theta;
        {
            float mscale = attn_factor;
            // Get n-d rotational scaling corrected for extrapolation
            float theta_interp = freq_scale * theta_base;
            float theta = theta_interp;
            if (ext_factor != 0.0f) {
                float ramp_mix = 1 - MIN((float)(1), std::max((float)(0), (i / 2 - corr_dims[0]) / std::max((float)(0.001f), corr_dims[1] - corr_dims[0])));
                theta = theta_interp * (1 - ramp_mix) + theta_base * ramp_mix;
                // Get n-d magnitude scaling corrected for interpolation
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            cos_theta = cosf(theta) * mscale;
            sin_theta = sinf(theta) * mscale;
        }

        // zeta scaling for xPos only:
        float zeta = xpos_base != 0.0f ? powf((i + 0.4f * n_dims) / (1.4f * n_dims), pos / xpos_base) : 1.0f;
        if (xpos_down) zeta = 1.0f / zeta;

        theta_base *= theta_scale;

        const float x0 = x[i];
        const float x1 = x[i+1];
        o[i]   = x0*cos_theta*zeta - x1*sin_theta*zeta;
        o[i+1] = x0*sin_theta*zeta + x1*cos_theta*zeta;
    }
}

} // namespace cpuft
