#ifndef __arm64__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <iostream>
#include <memory>
#include <algorithm>
#include <limits>
#include <type_traits>

#if defined(__AVX_512F__) || defined(__AVX2__) || defined(__SSE__)
    #include <immintrin.h>
    #ifdef __SSE__
        #include <emmintrin.h>
    #endif
    #ifdef _WIN32
        #include <intrin.h>
    #endif
#endif

#include "base_constants.h"
#include "cpuinfo.h"
#include "simd.h"

namespace cpuft::simd {

constexpr size_t AVX512_SIMD_SIZE = 64;
constexpr size_t AVX256_SIMD_SIZE = 32;
constexpr size_t AVX128_SIMD_SIZE = 16;
constexpr size_t MIN_SIMD_NUM = 1;

size_t get_simd_size_in_bytes() {
   if (cpu_support_avx512()) {
      return AVX512_SIMD_SIZE;
   } else if (cpu_support_avx256()) {
      return AVX256_SIMD_SIZE;
   } else if (cpu_support_avx128()) {
      return AVX128_SIMD_SIZE;
   } else {
      return 0;
   }
}
size_t g_simd_size = get_simd_size_in_bytes();
size_t get_simd_size() noexcept {
    return g_simd_size;
}
size_t set_simd_size(size_t simd_size) noexcept {
    if (auto v = get_simd_size_in_bytes(); simd_size > v) {
        simd_size = v;
    }
    for (; (simd_size & (simd_size-1)) != 0; simd_size &= simd_size - 1);
    g_simd_size = simd_size;
    return g_simd_size;
}

template <typename T>
T array_max_sisd(const T* x, size_t n) {
   T res = x[0];
   for (size_t i = 0; i < n; ++i) {
      if (x[i] > res) res = x[i];
   }
   return res;
}
template <typename T>
T array_min_sisd(const T* x, size_t n) {
   T res = x[0];
   for (size_t i = 0; i < n; ++i) {
      if (x[i] < res) res = x[i];
   }
   return res;
}
template <typename T>
T array_max_abs_sisd(const T* x, size_t n) {
   T res = 0;
   for (size_t i = 0; i < n; ++i) {
      if (x[i] > res) res = x[i];
      else if (-x[i] > res) res = -x[i];
   }
   return res;
}

#ifdef __SSE__
float array_max_avx128(const float* x, size_t n) {
   __m128 max_val = _mm_loadu_ps(x);
   size_t i;
   for (i = 4; i + 3 < n; i += 4) {
      __m128 data = _mm_loadu_ps(x + i);
      max_val = _mm_max_ps(max_val, data);
   }

   alignas(16) float partial_max[4];
   _mm_storeu_ps(partial_max, max_val);
   float total_max = partial_max[0];
   for (int j = 1; j < 4; ++j) {
      if (partial_max[j] > total_max) total_max = partial_max[j];
   }

   // Handle remaining elements
   for (; i < n; ++i) {
      if (x[i] > total_max) total_max = x[i];
   }

   return total_max;
}
#endif
#ifdef __AVX2__
float array_max_avx256(const float* x, size_t n) {
   __m256 max_val = _mm256_loadu_ps(x);
   size_t i;
   for (i = 8; i + 7 < n; i += 8) {
      __m256 data = _mm256_loadu_ps(x + i);
      max_val = _mm256_max_ps(max_val, data);
   }
   alignas(32) float partial_max[8];
   _mm256_storeu_ps(partial_max, max_val);
   float total_max = partial_max[0];
   for (int j = 1; j < 8; ++j) {
      if (partial_max[j] > total_max) total_max = partial_max[j];
   }
   for (; i < n; ++i) {
      if (x[i] > total_max) total_max = x[i];
   }
   return total_max;
}
#endif
#ifdef __AVX_512F__
float array_max_avx512(const float* x, size_t n) {
    __m512 max_val = _mm512_loadu_ps(x);
    size_t i;
    for (i = 16; i + 15 < n; i += 16) {
        __m512 data = _mm512_loadu_ps(x + i);
        max_val = _mm512_max_ps(max_val, data);
    }
    alignas(64) float partial_max[16];
    _mm512_storeu_ps(partial_max, max_val);
    float total_max = partial_max[0];
    for (int j = 1; j < 16; ++j) {
        if (partial_max[j] > total_max) total_max = partial_max[j];
    }
    for (; i < n; ++i) {
        if (x[i] > total_max) total_max = x[i];
    }
    return total_max;
}
#endif
#ifdef __SSE__
short array_max_avx128(const short* x, size_t n) {
   __m128i max_val = _mm_loadu_si128((__m128i*)(x));
   size_t i;
   for (i = 8; i + 7 < n; i += 8) {
      __m128i data = _mm_loadu_si128((__m128i*)(x + i));
      max_val = _mm_max_epi16(max_val, data);
   }
   alignas(16) short max_array[8];
   _mm_storeu_si128((__m128i*)max_array, max_val);
   short max_result = max_array[0];
   for (int j = 1; j < 8; ++j) {
      if (max_array[j] > max_result) max_result = max_array[j];
   }

   for (; i < n; ++i) {
      if (x[i] > max_result) max_result = x[i];
   }
   return max_result;
}
#endif
#ifdef __AVX2__
short array_max_avx256(const short* x, size_t n) {
   __m256i max_val = _mm256_loadu_si256((__m256i*)(x));
   size_t i;
   for (i = 16; i + 15 < n; i += 16) {
      __m256i data = _mm256_loadu_si256((__m256i*)(x + i));
      max_val = _mm256_max_epi16(max_val, data);
   }

   // Find max in max_val vector
   alignas(32) short max_array[16];
   _mm256_storeu_si256((__m256i*)max_array, max_val);
   short max_result = max_array[0];
   for (int j = 1; j < 16; ++j) {
      if (max_array[j] > max_result) max_result = max_array[j];
   }
   for (; i < n; ++i) {
      if (x[i] > max_result) max_result = x[i];
   }
   return max_result;
}
#endif
#ifdef __SSE__
int8_t array_max_avx128(const int8_t* x, size_t n) {
   __m128i max_values = _mm_loadu_si128((__m128i*)x);
   size_t i;
   for (i = 16; i + 15 < n; i += 16) {
      __m128i data = _mm_loadu_si128((__m128i*)(x + i));
      max_values = _mm_max_epi8(max_values, data);
   }
   alignas(16) int8_t partial_max[16];
   _mm_storeu_si128((__m128i*)partial_max, max_values);
   int8_t total_max = partial_max[0];
   for (int j = 1; j < 16; ++j) {
      if (partial_max[j] > total_max) total_max = partial_max[j];
   }
   for (; i < n; ++i) {
      if (x[i] > total_max) total_max = x[i];
   }
   return total_max;
}
#endif
#ifdef __AVX2__
int8_t array_max_avx256(const int8_t* x, size_t n) {
   __m256i max_values = _mm256_loadu_si256((__m256i*)x);
   size_t i;
   for (i = 32; i + 31 < n; i += 32) {
      __m256i data = _mm256_loadu_si256((__m256i*)(x + i));
      max_values = _mm256_max_epi8(max_values, data);
   }
   alignas(32) int8_t partial_max[32];
   _mm256_storeu_si256((__m256i*)partial_max, max_values);
   int8_t total_max = partial_max[0];
   for (int j = 1; j < 32; ++j) {
      if (partial_max[j] > total_max) total_max = partial_max[j];
   }
   for (; i < n; ++i) {
      if (x[i] > total_max) total_max = x[i];
   }
   return total_max;
}
#endif

float array_max(const float* x, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return FLT_MIN;
   }
#ifdef __AVX_512F__
   if (g_simd_size >= AVX512_SIMD_SIZE && n >= AVX512_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_avx512(x, n);
   } else
#endif
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_avx256(x, n);
   } else
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_avx128(x, n);
   } else 
#endif
   {
      return array_max_sisd(x, n);
   }
}
short array_max(const short* x, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return INT16_MIN;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_avx256(x, n);

   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_avx128(x, n);
   } else 
#endif
   {
      return array_max_sisd(x, n);
   }
}
int8_t array_max(const int8_t* x, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return INT8_MIN;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_avx256(x, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_avx128(x, n);
   } else 
#endif
   {
      return array_max_sisd(x, n);
   }
}

#ifdef __SSE__
float array_max_abs_avx128(const float* x, size_t n) {
   __m128 abs_mask = _mm_set1_ps(-0.0f); // Set all sign bits to get an absolute value mask
   __m128 max_abs = _mm_setzero_ps();
   size_t i = 0;
   for (; i + 3 < n; i += 4) {
      __m128 data = _mm_loadu_ps(x + i);
      __m128 abs_data = _mm_andnot_ps(abs_mask, data); // Compute absolute values
      max_abs = _mm_max_ps(max_abs, abs_data); // Update max_abs
   }
   alignas(16) float partial_max[4];
   _mm_storeu_ps(partial_max, max_abs);
   float total_max = 0;
   for (int j = 0; j < 4; ++j) {
      total_max = std::max(total_max, partial_max[j]);
   }
   for (; i < n; ++i) {
      if (x[i] > total_max) total_max = x[i];
      else if (-x[i] > total_max) total_max = -x[i];
   }
   return total_max;
}
#endif

#ifdef __AVX2__
float array_max_abs_avx256(const float* x, size_t n) {
   __m256 abs_mask = _mm256_set1_ps(-0.0f); // Set all sign bits to get an absolute value mask
   __m256 max_abs = _mm256_setzero_ps();
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m256 data = _mm256_loadu_ps(x + i);
      __m256 abs_data = _mm256_andnot_ps(abs_mask, data); // Compute absolute values
      max_abs = _mm256_max_ps(max_abs, abs_data); // Update max_abs
   }

   // Horizontal max
   alignas(32) float partial_max[8];
   _mm256_storeu_ps(partial_max, max_abs);
   float total_max = 0;
   for (int j = 0; j < 8; ++j) {
      total_max = std::max(total_max, partial_max[j]);
   }
   for (; i < n; ++i) {
      if (x[i] > total_max) total_max = x[i];
      else if (-x[i] > total_max) total_max = -x[i];
   }
   return total_max;
}
#endif
#ifdef __SSE__
short array_max_abs_avx128(const short* x, size_t n) {
   __m128i max_abs = _mm_setzero_si128();
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m128i data = _mm_loadu_si128((__m128i*)(x + i));
      __m128i abs_data = _mm_abs_epi16(data);
      max_abs = _mm_max_epi16(max_abs, abs_data);
   }

   alignas(16) short partial_max[8];
   _mm_storeu_si128((__m128i*)partial_max, max_abs);
   short total_max = 0;
   for (int j = 0; j < 8; ++j) {
      if (partial_max[j] > total_max) {
         total_max = partial_max[j];
      }
   }

   for (; i < n; ++i) {
      if (x[i] > total_max) {
         total_max = x[i];
      } else if (-x[i] > total_max) {
         total_max = -x[i];
      }
   }
   return total_max;
}
#endif
#ifdef __AVX2__
short array_max_abs_avx256(const short* x, size_t n) {
   __m256i max_abs = _mm256_setzero_si256();
   size_t i = 0;
   for (; i + 15 < n; i += 16) {
      __m256i data = _mm256_loadu_si256((__m256i*)(x + i));
      __m256i abs_data = _mm256_abs_epi16(data);
      max_abs = _mm256_max_epi16(max_abs, abs_data);
   }

   alignas(32) short partial_max[16];
   _mm256_storeu_si256((__m256i*)partial_max, max_abs);
   short total_max = 0;
   for (int j = 0; j < 16; ++j) {
      if (partial_max[j] > total_max) {
         total_max = partial_max[j];
      }
   }

   for (; i < n; ++i) {
      if (x[i] > total_max) {
         total_max = x[i];
      } else if (-x[i] > total_max) {
         total_max = -x[i];
      }
   }
   return total_max;
}
#endif
#ifdef __SSE__
int8_t array_max_abs_avx128(const int8_t* x, size_t n) {
    __m128i max_values = _mm_setzero_si128();
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m128i data = _mm_loadu_si128((__m128i*)(x + i));
        __m128i abs_data = _mm_abs_epi8(data);
        max_values = _mm_max_epu8(max_values, abs_data);
    }

    alignas(16) uint8_t partial_max[16];
    _mm_storeu_si128((__m128i*)partial_max, max_values);
    uint8_t total_max = 0;
    for (int j = 0; j < 16; ++j) {
        if (partial_max[j] > total_max) {
            total_max = partial_max[j];
        }
    }

    for (; i < n; ++i) {
        if (x[i] > total_max) {
            total_max = x[i];
        } else if (-x[i] > total_max) {
            total_max = -x[i];
        }
    }
    return total_max;
}
#endif

#ifdef __AVX2__
int8_t array_max_abs_avx256(const int8_t* x, size_t n) {
   __m256i max_values = _mm256_setzero_si256();
   size_t i = 0;
   for (; i + 31 < n; i += 32) {
      __m256i data = _mm256_loadu_si256((__m256i*)(x + i));
      __m256i abs_data = _mm256_abs_epi8(data);
      max_values = _mm256_max_epu8(max_values, abs_data);
   }
   alignas(32) uint8_t partial_max[32];
   _mm256_storeu_si256((__m256i*)partial_max, max_values);
   uint8_t total_max = 0;
   for (int j = 0; j < 32; ++j) {
      if (partial_max[j] > total_max) {
         total_max = partial_max[j];
      }
   }
   for (; i < n; ++i) {
      int8_t abs_value = x[i] >= 0 ? x[i] : -x[i];
      if (abs_value > total_max) {
         total_max = abs_value;
      }
   }
   return total_max;
}
#endif

float array_max_abs(const float* x, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return 0;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_abs_avx256(x, n);
   } else
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_abs_avx128(x, n);
   } else 
#endif
   {
      return array_max_abs_sisd(x, n);
   }
}
short array_max_abs(const short* x, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return 0;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_abs_avx256(x, n);
   } else 
#endif
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_abs_avx128(x, n);
   } else {
      return array_max_abs_sisd(x, n);
   }
}
int8_t array_max_abs(const int8_t* x, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return 0;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_abs_avx256(x, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_max_abs_avx128(x, n);
   } else 
#endif
   {
      return array_max_abs_sisd(x, n);
   }
}

MinMax array_min_max(const float* x, size_t n) noexcept {
    return MinMax{ array_min(x, n), array_max(x, n) };
}

#ifdef __SSE__
float array_min_avx128(const float* x, size_t n) {
   __m128 min_val = _mm_loadu_ps(x);
   size_t i;
   for (i = 4; i + 3 < n; i += 4) {
      __m128 data = _mm_loadu_ps(x + i);
      min_val = _mm_min_ps(min_val, data);
   }
   alignas(16) float partial_min[4];
   _mm_storeu_ps(partial_min, min_val);
   float res = partial_min[0];
   for (int j = 1; j < 4; ++j) {
      if (partial_min[j] < res) res = partial_min[j];
   }

   for (; i < n; ++i) {
      if (x[i] < res) res = x[i];
   }
   return res;
}
#endif

#ifdef __AVX2__
float array_min_avx256(const float* x, size_t n) {
   __m256 min_val = _mm256_loadu_ps(x);
   size_t i;
   for (i = 8; i + 7 < n; i += 8) {
      __m256 data = _mm256_loadu_ps(x + i);
      min_val = _mm256_min_ps(min_val, data);
   }

   alignas(32) float partial_min[8];
   _mm256_storeu_ps(partial_min, min_val);
   float res = partial_min[0];
   for (int j = 1; j < 8; ++j) {
      if (partial_min[j] < res) res = partial_min[j];
   }
   for (; i < n; ++i) {
      if (x[i] < res) res = x[i];
   }
   return res;
}
#endif

#ifdef __SSE__
short array_min_avx128(const short* x, size_t n) {
   __m128i min_vec = _mm_loadu_si128((__m128i*)(x));
   size_t i;
   for (i = 8; i + 7 < n; i += 8) {
      __m128i data = _mm_loadu_si128((__m128i*)(x + i));
      min_vec = _mm_min_epi16(min_vec, data);
   }
   alignas(16) short partial_min[8];
   _mm_storeu_si128((__m128i*)partial_min, min_vec);
   short min_val = partial_min[0];
   for (int j = 1; j < 8; ++j) {
      if (partial_min[j] < min_val) {
         min_val = partial_min[j];
      }
   }
   for (; i < n; ++i) {
      if (x[i] < min_val) min_val = x[i];
   }
   return min_val;
}
#endif

#ifdef __AVX2__
short array_min_avx256(const short* x, size_t n) {
   __m256i min_vec = _mm256_loadu_si256((__m256i*)(x));
   size_t i;
   for (i = 16; i + 15 < n; i += 16) {
      __m256i data = _mm256_loadu_si256((__m256i*)(x + i));
      min_vec = _mm256_min_epi16(min_vec, data);
   }

   alignas(32) short partial_min[16];
   _mm256_storeu_si256((__m256i*)partial_min, min_vec);
   short min_val = partial_min[0];
   for (int j = 1; j < 16; ++j) {
      if (partial_min[j] < min_val) {
         min_val = partial_min[j];
      }
   }

   for (; i < n; ++i) {
      if (x[i] < min_val) min_val = x[i];
   }
   return min_val;
}
#endif

#ifdef __SSE__
int8_t array_min_avx128(const int8_t* x, size_t n) {
   __m128i min_vals = _mm_loadu_si128((__m128i*)(x));
   size_t i;
   for (i = 16; i + 15 < n; i += 16) {
      __m128i data = _mm_loadu_si128((__m128i*)(x + i));
      min_vals = _mm_min_epi8(min_vals, data);
   }

   alignas(16) int8_t partial_min[16];
   _mm_storeu_si128((__m128i*)partial_min, min_vals);
   int8_t min_val = partial_min[0];
   for (int j = 1; j < 16; ++j) {
      if (partial_min[j] < min_val) min_val = partial_min[j];
   }

   // Handle remaining elements
   for (; i < n; ++i) {
      if (x[i] < min_val) min_val = x[i];
   }

   return min_val;
}
#endif

#ifdef __AVX2__
int8_t array_min_avx256(const int8_t* x, size_t n) {
   __m256i min_vals = _mm256_loadu_si256((__m256i*)(x));
   size_t i;
   for (i = 32; i + 31 < n; i += 32) {
      __m256i data = _mm256_loadu_si256((__m256i*)(x + i));
      min_vals = _mm256_min_epi8(min_vals, data);
   }

   alignas(32) int8_t partial_min[32];
   _mm256_storeu_si256((__m256i*)partial_min, min_vals);
   int8_t min_val = partial_min[0];
   for (int j = 1; j < 32; ++j) {
      if (partial_min[j] < min_val) min_val = partial_min[j];
   }

   for (; i < n; ++i) {
      if (x[i] < min_val) min_val = x[i];
   }
   return min_val;
}
#endif

float array_min(const float* x, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return FLT_MIN;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_min_avx256(x, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_min_avx128(x, n);
   } else 
#endif
   {
      return array_min_sisd(x, n);
   }
}
short array_min(const short* x, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return INT16_MIN;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_min_avx256(x, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_min_avx128(x, n);
   } else 
#endif
   {
      return array_min_sisd(x, n);
   }
}
int8_t array_min(const int8_t* x, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return INT8_MIN;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_min_avx256(x, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return array_min_avx128(x, n);
   } else 
#endif
   {
      return array_min_sisd(x, n);
   }
}

float array_sum_sisd(const float *x, size_t n) {
    float sum = 0.;
    for (size_t i = 0; i < n; ++i) sum += x[i];
    return sum;
}
template <typename T>
int array_sum_sisd(const T *x, size_t n) {
    int sum = 0.;
    for (size_t i = 0; i < n; ++i) sum += x[i];
    return sum;
}
#ifdef __SSE__
float array_sum_avx128(const float *x, size_t n) {
    float sum = 0;
    size_t i = 0;
    if (n >= 4) {
    __m128 sum_vector = _mm_setzero_ps();
        for (; i + 3 < n; i += 4) {
            __m128 data = _mm_loadu_ps(x + i);
            sum_vector = _mm_add_ps(sum_vector, data);
        }
        __m128 temp = _mm_movehl_ps(sum_vector, sum_vector);
        sum_vector = _mm_add_ps(sum_vector, temp);
        temp = _mm_shuffle_ps(sum_vector, sum_vector, 1);
        sum = _mm_cvtss_f32(_mm_add_ss(sum_vector, temp));
    }
    for (; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}
#endif

#ifdef __AVX2__
float array_sum_avx256(const float *x, size_t n) {
    float sum = 0;
    size_t i = 0;
    if (n >= 8) {
        __m256 sum_vector = _mm256_setzero_ps();
        for (; i + 7 < n; i += 8) {
            __m256 data = _mm256_loadu_ps(x + i);
            sum_vector = _mm256_add_ps(sum_vector, data);
        }
        __m128 low = _mm256_castps256_ps128(sum_vector);
        __m128 high = _mm256_extractf128_ps(sum_vector, 1);
        __m128 reduced = _mm_add_ps(low, high);
        __m128 temp = _mm_movehl_ps(reduced, reduced);
        reduced = _mm_add_ps(reduced, temp);
        temp = _mm_shuffle_ps(reduced, reduced, 1);
        sum = _mm_cvtss_f32(_mm_add_ss(reduced, temp));
    }
    for (; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}
#endif

#ifdef __SSE__
int array_sum_avx128(const short* x, size_t n) {
   __m128i sum = _mm_setzero_si128();
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
       __m128i data = _mm_loadu_si128((__m128i*)(x + i));
       __m128i data_lo = _mm_cvtepi16_epi32(_mm_unpacklo_epi64(data, data));
       __m128i data_hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(data, data));
       sum = _mm_add_epi32(sum, _mm_add_epi32(data_lo, data_hi));
   }

   alignas(16) int partial_sum[4];
   _mm_storeu_si128((__m128i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 4; ++j) {
       total_sum += partial_sum[j];
   }

   for (; i < n; ++i) {
       total_sum += x[i];
   }
   return total_sum;
}
#endif

#ifdef __AVX2__
int array_sum_avx256(const short* x, size_t n) {
   __m256i sum = _mm256_setzero_si256();
   size_t i = 0;
   for (; i + 15 < n; i += 16) {
       __m256i data = _mm256_loadu_si256((__m256i*)(x + i));
       __m256i data_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data, 0));
       __m256i data_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data, 1));
       sum = _mm256_add_epi32(sum, _mm256_add_epi32(data_lo, data_hi));
   }
   alignas(32) int partial_sum[8];
   _mm256_storeu_si256((__m256i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 8; ++j) {
       total_sum += partial_sum[j];
   }
   for (; i < n; ++i) {
       total_sum += x[i];
   }
   return total_sum;
}
#endif

#ifdef __SSE__
int array_sum_avx128(const int8_t *x, size_t n) {
    __m128i sum = _mm_setzero_si128();
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m128i data = _mm_loadu_si128((__m128i *)(x + i));
        __m128i data16_low = _mm_cvtepi8_epi16(data);
        __m128i data16_high = _mm_cvtepi8_epi16(_mm_srli_si128(data, 8));
        sum = _mm_add_epi16(sum, data16_low);
        sum = _mm_add_epi16(sum, data16_high);
    }
    alignas(16) int16_t partial_sum[8];
    _mm_storeu_si128((__m128i *)partial_sum, sum);
    int sum_int = 0;
    for (int j = 0; j < 8; ++j) {
        sum_int += partial_sum[j];
    }
    for (; i < n; ++i) {
        sum_int += x[i];
    }
    return sum_int;
}
#endif
#ifdef __AVX2__
int array_sum_avx256(const int8_t *x, size_t n) {
    __m256i sum = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 31 < n; i += 32) {
        __m256i data = _mm256_loadu_si256((__m256i *)(x + i));
        __m256i data16_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data, 0));
        __m256i data16_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data, 1));
        sum = _mm256_add_epi16(sum, data16_low);
        sum = _mm256_add_epi16(sum, data16_high);
    }
    alignas(32) int16_t partial_sum[16];
    _mm256_storeu_si256((__m256i *)partial_sum, sum);
    int sum_int = 0;
    for (int j = 0; j < 16; ++j) {
        sum_int += partial_sum[j];
    }
    for (; i < n; ++i) {
        sum_int += x[i];
    }
    return sum_int;
}
#endif

float array_sum(const float* x, size_t n) noexcept {
    if (x == nullptr || n < 1) [[unlikely]] {
        return 0.;
    }
#ifdef __AVX2__
    if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
        return array_sum_avx256(x, n);
    } else 
#endif
#ifdef __SSE__
    if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
        return array_sum_avx128(x, n);
    } else 
#endif
    {
        return array_sum_sisd(x, n);
    }
}
int array_sum(const short* x, size_t n) noexcept {
    if (x == nullptr || n < 1) [[unlikely]] {
        return 0.;
    }
#ifdef __AVX2__
    if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
        return array_sum_avx256(x, n);
    } else 
#endif
#ifdef __SSE__
    if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
        return array_sum_avx128(x, n);
    } else 
#endif
    {
        return array_sum_sisd(x, n);
    }
}
int array_sum(const int8_t* x, size_t n) noexcept {
    if (x == nullptr || n < 1) [[unlikely]] {
        return 0.;
    }
#ifdef __AVX2__
    if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
        return array_sum_avx256(x, n);
    } else 
#endif
#ifdef __SSE__
    if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
        return array_sum_avx128(x, n);
    } else 
#endif
    {
        return array_sum_sisd(x, n);
    }
}

float square_sum_sisd(const float* x, size_t n) noexcept {
   float res = 0.;
   for (size_t i = 0; i < n; ++i) {
      res += x[i] * x[i];
   }
   return res;
}
int square_sum_sisd(const short* x, size_t n) noexcept {
   int res = 0;
   for (size_t i = 0; i < n; ++i) {
      res += int(x[i]) * int(x[i]);
   }
   return res;
}
int square_sum_sisd(const int8_t* x, size_t n) noexcept {
   int res = 0;
   for (size_t i = 0; i < n; ++i) {
      res += int(x[i]) * int(x[i]);
   }
   return res;
}

#ifdef __SSE__
float square_sum_avx128(const float* x, size_t n) {
   __m128 sum = _mm_setzero_ps();
   size_t i = 0;
   for (; i + 3 < n; i += 4) {
      __m128 data = _mm_loadu_ps(x + i);
      __m128 square = _mm_mul_ps(data, data);
      sum = _mm_add_ps(sum, square);
   }
   alignas(16) float partial_sum[4];
   _mm_storeu_ps(partial_sum, sum);
   float res = 0;
   for (int j = 0; j < 4; ++j) {
      res += partial_sum[j];
   }

   for (; i < n; ++i) {
      res += x[i] * x[i];
   }
   return res;
}
#endif
#ifdef __AVX2__
float square_sum_avx256(const float* x, size_t n) {
   __m256 sum = _mm256_setzero_ps();
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m256 data = _mm256_loadu_ps(x + i);
      __m256 square = _mm256_mul_ps(data, data);
      sum = _mm256_add_ps(sum, square);
   }
   alignas(32) float partial_sum[8];
   _mm256_storeu_ps(partial_sum, sum);
   float res = 0;
   for (int j = 0; j < 8; ++j) {
      res += partial_sum[j];
   }

   for (; i < n; ++i) {
      res += x[i] * x[i];
   }
   return res;
}
#endif

#ifdef __SSE__
int square_sum_avx128(const short* x, size_t n) {
   __m128i sum = _mm_setzero_si128();
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m128i data = _mm_loadu_si128((__m128i*)(x + i));
      __m128i data_lo = _mm_cvtepi16_epi32(_mm_unpacklo_epi64(data, data));
      __m128i data_hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(data, data));
      __m128i square_lo = _mm_mullo_epi32(data_lo, data_lo);
      __m128i square_hi = _mm_mullo_epi32(data_hi, data_hi);
      sum = _mm_add_epi32(sum, _mm_add_epi32(square_lo, square_hi));
   }
   alignas(16) int partial_sum[4];
   _mm_storeu_si128((__m128i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 4; ++j) {
      total_sum += partial_sum[j];
   }
   for (; i < n; ++i) {
      total_sum += int(x[i]) * int(x[i]);
   }
   return total_sum;
}
#endif

#ifdef __AVX2__
int square_sum_avx256(const short* x, size_t n) {
   __m256i sum = _mm256_setzero_si256();
   size_t i = 0;
   for (; i + 15 < n; i += 16) {
      __m256i data = _mm256_loadu_si256((__m256i*)(x + i));
      __m256i data_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data, 0));
      __m256i data_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data, 1));
      __m256i square_lo = _mm256_mullo_epi32(data_lo, data_lo);
      __m256i square_hi = _mm256_mullo_epi32(data_hi, data_hi);
      sum = _mm256_add_epi32(sum, _mm256_add_epi32(square_lo, square_hi));
   }
   alignas(32) int partial_sum[8];
   _mm256_storeu_si256((__m256i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 8; ++j) {
      total_sum += partial_sum[j];
   }
   for (; i < n; ++i) {
      total_sum += int(x[i]) * int(x[i]);
   }
   return total_sum;
}
#endif

#ifdef __SSE__
int square_sum_avx128(const int8_t* x, size_t n) {
   __m128i sum = _mm_setzero_si128();
   size_t i = 0;
   for (; i + 15 < n; i += 16) {
      __m128i data = _mm_loadu_si128((__m128i*)(x + i));
      __m128i data_lo = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpacklo_epi64(data, data)),
                                       _mm_cvtepi8_epi16(_mm_unpacklo_epi64(data, data)));
      __m128i data_hi = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(data, data)),
                                       _mm_cvtepi8_epi16(_mm_unpackhi_epi64(data, data)));
      sum = _mm_add_epi32(sum, _mm_add_epi32(data_lo, data_hi));
   }
   alignas(16) int partial_sum[4];
   _mm_storeu_si128((__m128i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 4; ++j) {
      total_sum += partial_sum[j];
   }
   for (; i < n; ++i) {
      total_sum += int(x[i]) * int(x[i]);
   }
   return total_sum;
}
#endif

#ifdef __AVX2__
int square_sum_avx256(const int8_t* x, size_t n) {
   __m256i sum = _mm256_setzero_si256();
   size_t i = 0;
   for (; i + 31 < n; i += 32) {
      __m256i data = _mm256_loadu_si256((__m256i*)(x + i));
      __m256i data_lo = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(data, 0)),
                                          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data, 0)));
      __m256i data_hi = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(data, 1)),
                                          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data, 1)));
      sum = _mm256_add_epi32(sum, _mm256_add_epi32(data_lo, data_hi));
   }
   alignas(32) int partial_sum[8];
   _mm256_storeu_si256((__m256i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 8; ++j) {
      total_sum += partial_sum[j];
   }

   // Handle remaining elements
   for (; i < n; ++i) {
      total_sum += int(x[i]) * int(x[i]);
   }

   return total_sum;
}
#endif

float square_sum(const float* x, size_t n) noexcept {
    if (x == nullptr || n < 1) [[unlikely]] {
        return 0.;
    }
#ifdef __AVX2
    if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
        return square_sum_avx256(x, n);
    } else 
#endif
#ifdef __SSE__
    if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
        return square_sum_avx128(x, n);
    } else 
#endif
    {
        return square_sum_sisd(x, n);
    }
}
int square_sum(const short* x, size_t n) noexcept {
    if (x == nullptr || n < 1) [[unlikely]] {
        return 0.;
    }
#ifdef __AVX2__
    if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
        return square_sum_avx256(x, n);
    } else 
#endif
#ifdef __SSE__
    if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
        return square_sum_avx128(x, n);
    } else 
#endif
    {
        return square_sum_sisd(x, n);
    }
}
int square_sum(const int8_t* x, size_t n) noexcept {
    if (x == nullptr || n < 1) [[unlikely]] {
        return 0.;
    }
#ifdef __AVX2__
    if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
        return square_sum_avx256(x, n);
    } else 
#endif
#ifdef __SSE__
    if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
        return square_sum_avx128(x, n);
    } else 
#endif
    {
        return square_sum_sisd(x, n);
    }
}

void add_sisd(float* x, float v, size_t n) noexcept {
   for (size_t i = 0; i < n; ++i) {
      x[i] += v;
   }
}
void add_sisd(float* x1, const float* x2, size_t n) noexcept {
   for (size_t i = 0; i < n; ++i) {
      x1[i] += x2[i];
   }
}
void add_sisd(float* x1, const float* x2, float r, size_t n) noexcept {
   for (size_t i = 0; i < n; ++i) {
      x1[i] += x2[i] * r;
   }
}

#ifdef __SSE__
void add_avx128(float* x, float v, size_t n) {
   __m128 vec_v = _mm_set1_ps(v);
   size_t i = 0;
   for (; i + 3 < n; i += 4) {
      __m128 data = _mm_loadu_ps(x + i);
      data = _mm_add_ps(data, vec_v);
      _mm_storeu_ps(x + i, data);
   }

   // Handle remaining elements
   for (; i < n; ++i) {
      x[i] += v;
   }
}
#endif
#ifdef __AVX2__
void add_avx256(float* x, float v, size_t n) {
   __m256 vec_v = _mm256_set1_ps(v);
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m256 data = _mm256_loadu_ps(x + i);
      data = _mm256_add_ps(data, vec_v);
      _mm256_storeu_ps(x + i, data);
   }
   for (; i < n; ++i) {
      x[i] += v;
   }
}
#endif
#ifdef __SSE__
void add_avx128(float* x1, const float* x2, size_t n) {
   size_t i = 0;
   for (; i + 3 < n; i += 4) {
      __m128 data1 = _mm_loadu_ps(x1 + i);
      __m128 data2 = _mm_loadu_ps(x2 + i);
      __m128 result = _mm_add_ps(data1, data2);
      _mm_storeu_ps(x1 + i, result);
   }
   for (; i < n; ++i) {
      x1[i] += x2[i];
   }
}
#endif
#ifdef __AVX2__
void add_avx256(float* x1, const float* x2, size_t n) {
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m256 data1 = _mm256_loadu_ps(x1 + i);
      __m256 data2 = _mm256_loadu_ps(x2 + i);
      __m256 result = _mm256_add_ps(data1, data2);
      _mm256_storeu_ps(x1 + i, result);
   }
   for (; i < n; ++i) {
      x1[i] += x2[i];
   }
}
#endif
void add_avx128(float* x1, const float* x2, float r, size_t n) {
   __m128 multiplier = _mm_set1_ps(r);
   size_t i = 0;
   for (; i + 3 < n; i += 4) {
      __m128 data1 = _mm_loadu_ps(x1 + i);
      __m128 data2 = _mm_loadu_ps(x2 + i);
      data2 = _mm_mul_ps(data2, multiplier);
      data1 = _mm_add_ps(data1, data2);
      _mm_storeu_ps(x1 + i, data1);
   }

   // Handle remaining elements
   for (; i < n; ++i) {
      x1[i] += x2[i] * r;
   }
}
#ifdef __AVX2__
void add_avx256(float* x1, const float* x2, float r, size_t n) {
   __m256 multiplier = _mm256_set1_ps(r);
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m256 data1 = _mm256_loadu_ps(x1 + i);
      __m256 data2 = _mm256_loadu_ps(x2 + i);
      data2 = _mm256_mul_ps(data2, multiplier);
      data1 = _mm256_add_ps(data1, data2);
      _mm256_storeu_ps(x1 + i, data1);
   }
   for (; i < n; ++i) {
      x1[i] += x2[i] * r;
   }
}
#endif

void add(float* x, float v, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return add_avx256(x, v, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return add_avx128(x, v, n);
   } else 
#endif
   {
      return add_sisd(x, v, n);
   }
}
void add(float* x1, const float* x2, size_t n) noexcept {
   if (x1 == nullptr || x2 == nullptr || n < 1) {
      return;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return add_avx256(x1, x2, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return add_avx128(x1, x2, n);
   } else 
#endif
   {
      return add_sisd(x1, x2, n);
   }
}
void add(float* x1, const float*x2, float r, size_t n) noexcept {
   if (x1 == nullptr || x2 == nullptr || n < 1) {
      return;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return add_avx256(x1, x2, r, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return add_avx128(x1, x2, r, n);
   } else 
#endif
   {
      return add_sisd(x1, x2, r, n);
   }
}

inline void multiply_sisd(float* x, float r, size_t n) noexcept {
   for (size_t i = 0; i < n; ++i) x[i] *= r;
}
inline void multiply_sisd(float* x, const float *w, float r, size_t n) noexcept {
   for (size_t i = 0; i < n; ++i) {
      x[i] *= w[i] * r;
   }
}
#ifdef __SSE__
void multiply_avx128(float* x, float r, size_t n) noexcept {
   __m128 multiplier = _mm_set1_ps(r);
   size_t i = 0;
   for (; i + 3 < n; i += 4) {
      __m128 data = _mm_loadu_ps(x + i);
      data = _mm_mul_ps(data, multiplier);
      _mm_storeu_ps(x + i, data);
   }
   for (; i < n; ++i) {
      x[i] *= r;
   }
}
#endif
#ifdef __AVX2__
void multiply_avx256(float* x, float r, size_t n) {
   __m256 multiplier = _mm256_set1_ps(r);
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m256 data = _mm256_loadu_ps(x + i);
      data = _mm256_mul_ps(data, multiplier);
      _mm256_storeu_ps(x + i, data);
   }
   for (; i < n; ++i) {
      x[i] *= r;
   }
}
#endif
#ifdef __SSE__
void multiply_avx128(float* x, const float* w, float r, size_t n) {
   __m128 factor = _mm_set1_ps(r);
   size_t i = 0;
   for (; i + 3 < n; i += 4) {
      __m128 x_data = _mm_loadu_ps(x + i);
      __m128 w_data = _mm_loadu_ps(w + i);
      __m128 result = _mm_mul_ps(_mm_mul_ps(x_data, w_data), factor);
      _mm_storeu_ps(x + i, result);
   }

   // Handle remaining elements
   for (; i < n; ++i) {
      x[i] *= w[i] * r;
   }
}
#endif
#ifdef __AVX2__
void multiply_avx256(float* x, const float* w, float r, size_t n) {
   __m256 factor = _mm256_set1_ps(r);
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m256 x_data = _mm256_loadu_ps(x + i);
      __m256 w_data = _mm256_loadu_ps(w + i);
      __m256 result = _mm256_mul_ps(_mm256_mul_ps(x_data, w_data), factor);
      _mm256_storeu_ps(x + i, result);
   }
   for (; i < n; ++i) {
      x[i] *= w[i] * r;
   }
}
#endif
void multiply(float* x, float r, size_t n) noexcept {
   if (x == nullptr || n < 1) {
      return;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return multiply_avx256(x, r, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return multiply_avx128(x, r, n);
   } else 
#endif
   {
      return multiply_sisd(x, r, n);
   }
}
void multiply(float* x, const float* w, float r, size_t n) noexcept {
   if (x == nullptr || w == nullptr || n < 1) {
      return;
   }
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return multiply_avx256(x, w, r, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return multiply_avx128(x, w, r, n);
   } else 
#endif
   {
      return multiply_sisd  (x, w, r, n);
   }
}

inline float dot_product_sisd(const float* x1, const float* x2, size_t n) noexcept {
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {  sum += x1[i] * x2[i];  }
    return sum;
}
template <typename T>
inline int dot_product_sisd(const T* x1, const T* x2, size_t n) noexcept {
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {  sum += int(x1[i]) * int(x2[i]);  }
    return sum;
}

#ifdef __SSE__
float dot_product_avx128(const float* x1, const float* x2, size_t n) {
   __m128 sum = _mm_setzero_ps();
   size_t i = 0;
   for (; i + 3 < n; i += 4) {
      __m128 data1 = _mm_loadu_ps(x1 + i);
      __m128 data2 = _mm_loadu_ps(x2 + i);
      __m128 prod = _mm_mul_ps(data1, data2);
      sum = _mm_add_ps(sum, prod);
   }

   alignas(16) float partial_sum[4];
   _mm_storeu_ps(partial_sum, sum);
   float total_sum = 0;
   for (int j = 0; j < 4; ++j) {
      total_sum += partial_sum[j];
   }
   for (; i < n; ++i) {
      total_sum += x1[i] * x2[i];
   }
   return total_sum;
}
#endif
#ifdef __AVX2__
float dot_product_avx256(const float* x1, const float* x2, size_t n) {
   __m256 sum = _mm256_setzero_ps();
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m256 data1 = _mm256_loadu_ps(x1 + i);
      __m256 data2 = _mm256_loadu_ps(x2 + i);
      __m256 prod = _mm256_mul_ps(data1, data2);
      sum = _mm256_add_ps(sum, prod);
   }
   alignas(32) float partial_sum[8];
   _mm256_storeu_ps(partial_sum, sum);
   float total_sum = 0;
   for (int j = 0; j < 8; ++j) {
      total_sum += partial_sum[j];
   }

   for (; i < n; ++i) {
      total_sum += x1[i] * x2[i];
   }
   return total_sum;
}
#endif
#ifdef __AVX_512F__
float dot_product_avx512(const float* x1, const float* x2, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    size_t i;
    for (i = 0; i + 15 < n; i += 16) {
        __m512 data1 = _mm512_loadu_ps(x1 + i);
        __m512 data2 = _mm512_loadu_ps(x2 + i);
        __m512 prod = _mm512_mul_ps(data1, data2);
        sum = _mm512_add_ps(sum, prod);
    }
    alignas(64) float partial_sum[16];
    _mm512_storeu_ps(partial_sum, sum);
    float total_sum = 0;
    for (int j = 0; j < 16; ++j) {
        total_sum += partial_sum[j];
    }

    for (; i < n; ++i) {
        total_sum += x1[i] * x2[i];
    }
    return total_sum;
}
#endif
#ifdef __SSE__
int dot_product_avx128(const short* x1, const short* x2, size_t n) {
   __m128i sum = _mm_setzero_si128();
   size_t i = 0;
   for (; i + 7 < n; i += 8) {
      __m128i data1 = _mm_loadu_si128((__m128i*)(x1 + i));
      __m128i data2 = _mm_loadu_si128((__m128i*)(x2 + i));

      __m128i data1_lo = _mm_cvtepi16_epi32(_mm_unpacklo_epi64(data1, data1));
      __m128i data1_hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(data1, data1));
      __m128i data2_lo = _mm_cvtepi16_epi32(_mm_unpacklo_epi64(data2, data2));
      __m128i data2_hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(data2, data2));

      __m128i prod_lo = _mm_mullo_epi32(data1_lo, data2_lo);
      __m128i prod_hi = _mm_mullo_epi32(data1_hi, data2_hi);

      sum = _mm_add_epi32(sum, _mm_add_epi32(prod_lo, prod_hi));
   }

   alignas(16) int partial_sum[4];
   _mm_storeu_si128((__m128i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 4; ++j) {
      total_sum += partial_sum[j];
   }
   for (; i < n; ++i) {
      total_sum += int(x1[i]) * int(x2[i]);
   }
   return total_sum;
}
#endif
#ifdef __AVX2__
int dot_product_avx256(const short* x1, const short* x2, size_t n) {
   __m256i sum = _mm256_setzero_si256();
   size_t i = 0;
   for (; i + 15 < n; i += 16) {
      __m256i data1 = _mm256_loadu_si256((__m256i*)(x1 + i));
      __m256i data2 = _mm256_loadu_si256((__m256i*)(x2 + i));

      __m256i data1_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0));
      __m256i data1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 1));
      __m256i data2_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data2, 0));
      __m256i data2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data2, 1));

      __m256i prod_lo = _mm256_mullo_epi32(data1_lo, data2_lo);
      __m256i prod_hi = _mm256_mullo_epi32(data1_hi, data2_hi);

      sum = _mm256_add_epi32(sum, _mm256_add_epi32(prod_lo, prod_hi));
   }
   alignas(32) int partial_sum[8];
   _mm256_storeu_si256((__m256i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 8; ++j) {
      total_sum += partial_sum[j];
   }

   for (; i < n; ++i) {
      total_sum += int(x1[i]) * int(x2[i]);
   }
   return total_sum;
}
#endif
#ifdef __AVX_512F__
int dot_product_avx512(const short* x1, const short* x2, size_t n) {
    __m512i sum = _mm512_setzero_si512();
    size_t i = 0;
    for (i; i + 31 < n; i += 32) {
        __m512i data1 = _mm512_loadu_si512((__m512i*)(x1 + i));
        __m512i data2 = _mm512_loadu_si512((__m512i*)(x2 + i));

        __m256i data1_lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(data1, 0));
        __m256i data1_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(data1, 1));
        __m256i data2_lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(data2, 0));
        __m256i data2_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(data2, 1));

        __m256i prod_lo = _mm256_mullo_epi32(data1_lo, data2_lo);
        __m256i prod_hi = _mm256_mullo_epi32(data1_hi, data2_hi);

        sum = _mm512_add_epi32(sum, _mm512_inserti64x4(sum, _mm256_add_epi32(prod_lo, prod_hi), 0));
        sum = _mm512_add_epi32(sum, _mm512_inserti64x4(sum, _mm256_add_epi32(prod_lo, prod_hi), 1));
    }
    alignas(64) int partial_sum[16];
    _mm512_storeu_si512((__m512i*)partial_sum, sum);
    int total_sum = 0;
    for (int j = 0; j < 16; ++j) {
        total_sum += partial_sum[j];
    }
    for (; i < n; ++i) {
        total_sum += int(x1[i]) * int(x2[i]);
    }
    return total_sum;
}
#endif
#ifdef __SSE__
int dot_product_avx128(const int8_t* x1, const int8_t* x2, size_t n) {
   __m128i sum = _mm_setzero_si128();
   size_t i = 0;
   for (; i + 15 < n; i += 16) {
      __m128i data1 = _mm_loadu_si128((__m128i*)(x1 + i));
      __m128i data2 = _mm_loadu_si128((__m128i*)(x2 + i));

      __m128i product_low = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpacklo_epi64(data1, data1)),
      _mm_cvtepi8_epi16(_mm_unpacklo_epi64(data2, data2)));

      __m128i product_high = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(data1, data1)),
      _mm_cvtepi8_epi16(_mm_unpackhi_epi64(data2, data2)));

      sum = _mm_add_epi32(sum, _mm_add_epi32(product_low, product_high));
   }

   alignas(16) int partial_sum[4];
   _mm_storeu_si128((__m128i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 4; ++j) {
      total_sum += partial_sum[j];
   }
   for (; i < n; ++i) {
      total_sum += int(x1[i]) * int(x2[i]);
   }
   return total_sum;
}
#endif

#ifdef __AVX2__
int dot_product_avx256(const int8_t* x1, const int8_t* x2, size_t n) {
   __m256i sum = _mm256_setzero_si256();
   size_t i = 0;
   for (; i + 31 < n; i += 32) {
      __m256i data1 = _mm256_loadu_si256((__m256i*)(x1 + i));
      __m256i data2 = _mm256_loadu_si256((__m256i*)(x2 + i));

      __m256i product_low = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(data1, 0)),
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data2, 0)));

      __m256i product_high = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(data1, 1)),
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data2, 1)));

      sum = _mm256_add_epi32(sum, _mm256_add_epi32(product_low, product_high));
   }

   alignas(32) int partial_sum[8];
   _mm256_storeu_si256((__m256i*)partial_sum, sum);
   int total_sum = 0;
   for (int j = 0; j < 8; ++j) {
      total_sum += partial_sum[j];
   }
   for (; i < n; ++i) {
      total_sum += int(x1[i]) * int(x2[i]);
   }

   return total_sum;
}
#endif

#ifdef __AVX_512F__
int dot_product_avx512(const int8_t* x1, const int8_t* x2, size_t n) {
    __m512i sum = _mm512_setzero_si512();
    size_t i;
    for (i = 0; i + 63 < n; i += 64) {
        __m512i data1 = _mm512_loadu_si512((__m512i*)(x1 + i));
        __m512i data2 = _mm512_loadu_si512((__m512i*)(x2 + i));

        __m512i product_low = _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm512_extracti64x2_epi64(data1, 0)),
                                                _mm512_cvtepi8_epi16(_mm512_extracti64x2_epi64(data2, 0)));

        __m512i product_high = _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm512_extracti64x2_epi64(data1, 1)),
                                                 _mm512_cvtepi8_epi16(_mm512_extracti64x2_epi64(data2, 1)));

        sum = _mm512_add_epi32(sum, _mm512_add_epi32(product_low, product_high));
    }

    alignas(64) int partial_sum[16];
    _mm512_storeu_si512((__m512i*)partial_sum, sum);
    int total_sum = 0;
    for (int j = 0; j < 16; ++j) {
        total_sum += partial_sum[j];
    }
    for (; i < n; ++i) {
        total_sum += int(x1[i]) * int(x2[i]);
    }

    return total_sum;
}
#endif

float dot_product(const float* x1, const float* x2, size_t n) noexcept {
   if (x1 == nullptr || x2 == nullptr || n < 1) {
      return 0;
   }
#ifdef __AVX_512F__
   if (g_simd_size >= AVX512_SIMD_SIZE && n >= AVX512_SIMD_SIZE * MIN_SIMD_NUM) {
      return dot_product_avx512(x1, x2, n);
   } else
#endif
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return dot_product_avx256(x1, x2, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return dot_product_avx128(x1, x2, n);
   } else 
#endif
   {
      return dot_product_sisd(x1, x2, n);
   }
}
int dot_product(const short* x1, const short* x2, size_t n) noexcept {
   if (x1 == nullptr || x2 == nullptr || n < 1) {
      return 0;
   }
#ifdef __AVX_512F__
   if (g_simd_size >= AVX512_SIMD_SIZE && n >= AVX512_SIMD_SIZE * MIN_SIMD_NUM) {
      return dot_product_avx512(x1, x2, n);
   } else
#endif
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return dot_product_avx256(x1, x2, n);
   } else 
#endif
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return dot_product_avx128(x1, x2, n);
   } else {
      return dot_product_sisd(x1, x2, n);
   }
}
int dot_product(const int8_t* x1, const int8_t* x2, size_t n) noexcept {
   if (x1 == nullptr || x2 == nullptr || n < 1) {
      return 0;
   }
#ifdef __AVX_512F__
   if (g_simd_size >= AVX512_SIMD_SIZE && n >= AVX512_SIMD_SIZE * MIN_SIMD_NUM) {
      return dot_product_avx512(x1, x2, n);
   } else
#endif
#ifdef __AVX2__
   if (g_simd_size >= AVX256_SIMD_SIZE && n >= AVX256_SIMD_SIZE * MIN_SIMD_NUM) {
      return dot_product_avx256(x1, x2, n);
   } else 
#endif
#ifdef __SSE__
   if (g_simd_size >= AVX128_SIMD_SIZE && n >= AVX128_SIMD_SIZE * MIN_SIMD_NUM) {
      return dot_product_avx128(x1, x2, n);
   } else 
#endif
   {
      return dot_product_sisd(x1, x2, n);
   }
}

void sigmoid_sisd(float* x, size_t n) noexcept {
    for (size_t i = 0; i < n; i++) {
        x[i] = 1. / (1. + expf(-x[i]));
    }
}

void rmsnorm(float* x, const float* w, size_t n) noexcept {
    const float r = 1. / sqrtf(square_sum_sisd(x, n) / n + 1e-5f);
    multiply(x, w, r, n);
}
void rmsnorm(float* o, const float* x, const float* w, size_t n) noexcept {
    const float r = 1. / sqrtf(square_sum(x, n) / n + 1e-5f);
    if constexpr(true) {
      memcpy(o, x, sizeof(*x) * n);
      multiply(o, w, r, n);
    } else {
       for (int i = 0; i < n; i++) {
           o[i] = w[i] * (r * x[i]);
       }
    }
}

void swiglu(float* xo, const float* xr, size_t n) noexcept {
    for (size_t i = 0; i < n; i++) {
        xo[i] = xo[i] / (1. + expf(-xo[i])) * xr[i];
    }
}
void swiglu(float* x, size_t n) noexcept {
    for (size_t i = 0; i < n; i++) {
        x[i] /= 1. + expf(-x[i]);
    }
}

} // namespace cpuft::simd

#ifdef TEST_X86_SIMD
using namespace cpuft::simd;

template <typename T>
void test_array_sum(T min_val, T max_val, const size_t n) {
   std::unique_ptr<T[]> arr(new T[n]);
   for (size_t i = 0; i < n; ++i) {
      if constexpr (std::is_same<T, float>::value) {
         arr[i] = ((rand() % 10000) / 10000.0) * (max_val - min_val) + min_val;
      } else {
         auto r = rand();
         arr[i] = (r % int(max_val - min_val)) + min_val;
      }
   }
   auto x = arr.get();

   if constexpr (std::is_same<T, float>::value) {
      float sisd_res     = array_sum_sisd  (x, n);
      float simd_res_128 = array_sum_avx128(x, n);
      float simd_res_256 = array_sum_avx256(x, n);
      printf("test_array_sum() item size:%lu|\tsisd:%g\tavx128:%g\tavx256:%g\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   } else {
      int sisd_res     = array_sum_sisd  (x, n);
      int simd_res_128 = array_sum_avx128(x, n);
      int simd_res_256 = array_sum_avx256(x, n);
      printf("test_array_sum() item size:%lu|\tsisd:%d\tavx128:%d\tavx256:%d\n",
            sizeof(T), sisd_res, simd_res_128, simd_res_256);
   }
}

template <typename T>
void test_array_max(T min_val, T max_val, const size_t n) {
   std::unique_ptr<T[]> arr(new T[n]);
   for (size_t i = 0; i < n; ++i) {
      if constexpr (std::is_same<T, float>::value) {
         arr[i] = ((rand() % 10000) / 10000.0) * (max_val - min_val) + min_val;
      } else {
         auto r = rand();
         arr[i] = (r % int(max_val - min_val)) + min_val;
      }
   }
   auto x = arr.get();

   if constexpr (std::is_same<T, float>::value) {
      float sisd_res     = array_max_sisd  (x, n);
      float simd_res_128 = array_max_avx128(x, n);
      float simd_res_256 = array_max_avx256(x, n);
      printf("test_array_max() item size:%lu|\tsisd:%g\tavx128:%g\tavx256:%g\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   } else {
      int sisd_res     = array_max_sisd  (x, n);
      int simd_res_128 = array_max_avx128(x, n);
      int simd_res_256 = array_max_avx256(x, n);
      printf("test_array_max() item size:%lu|\tsisd:%d\tavx128:%d\tavx256:%d\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   }
}

template <typename T>
void test_array_max_abs(T min_val, T max_val, const size_t n) {
   std::unique_ptr<T[]> arr(new T[n]);
   for (size_t i = 0; i < n; ++i) {
      if constexpr (std::is_same<T, float>::value) {
         arr[i] = ((rand() % 10000) / 10000.0) * (max_val - min_val) + min_val;
      } else {
         auto r = rand();
         arr[i] = (r % int(max_val - min_val)) + min_val;
      }
   }
   auto x = arr.get();

   if constexpr (std::is_same<T, float>::value) {
      float sisd_res     = array_max_abs_sisd  (x, n);
      float simd_res_128 = array_max_abs_avx128(x, n);
      float simd_res_256 = array_max_abs_avx256(x, n);
      printf("test_array_abs_max() item size:%lu|\tsisd:%g\tavx128:%g\tavx256:%g\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   } else {
      int sisd_res     = array_max_abs_sisd  (x, n);
      int simd_res_128 = array_max_abs_avx128(x, n);
      int simd_res_256 = array_max_abs_avx256(x, n);
      printf("test_array_abs_max() item size:%lu|\tsisd:%d\tavx128:%d\tavx256:%d\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   }
}

template <typename T>
void test_array_min(T min_val, T max_val, const size_t n) {
   std::unique_ptr<T[]> arr(new T[n]);
   for (size_t i = 0; i < n; ++i) {
      if constexpr (std::is_same<T, float>::value) {
         arr[i] = ((rand() % 10000) / 10000.0) * (max_val - min_val) + min_val;
      } else {
         auto r = rand();
         arr[i] = (r % int(max_val - min_val)) + min_val;
      }
   }
   auto x = arr.get();

   if constexpr (std::is_same<T, float>::value) {
      float sisd_res     = array_min_sisd  (x, n);
      float simd_res_128 = array_min_avx128(x, n);
      float simd_res_256 = array_min_avx256(x, n);
      printf("test_array_min() item size:%lu|\tsisd:%g\tavx128:%g\tavx256:%g\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   } else {
      int sisd_res     = array_min_sisd  (x, n);
      int simd_res_128 = array_min_avx128(x, n);
      int simd_res_256 = array_min_avx256(x, n);
      printf("test_array_min() item size:%lu|\tsisd:%d\tavx128:%d\tavx256:%d\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   }
}

template <typename T>
void test_square_sum(T min_val, T max_val, const size_t n) {
   std::unique_ptr<T[]> arr(new T[n]);
   for (size_t i = 0; i < n; ++i) {
      if constexpr (std::is_same<T, float>::value) {
         arr[i] = ((rand() % 10000) / 10000.0) * (max_val - min_val) + min_val;
      } else {
         auto r = rand();
         arr[i] = (r % int(max_val - min_val)) + min_val;
      }
   }
   auto x = arr.get();

   if constexpr (std::is_same<T, float>::value) {
      float sisd_res     = square_sum_sisd  (x, n);
      float simd_res_128 = square_sum_avx128(x, n);
      float simd_res_256 = square_sum_avx256(x, n);
      printf("test_square_sum() item size:%lu|\tsisd:%g\tavx128:%g\tavx256:%g\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   } else {
      int sisd_res     = square_sum_sisd  (x, n);
      int simd_res_128 = square_sum_avx128(x, n);
      int simd_res_256 = square_sum_avx256(x, n);
      printf("test_square_sum() item size:%lu|\tsisd:%d\tavx128:%d\tavx256:%d\n",
            sizeof(T), sisd_res, simd_res_128, simd_res_256);
   }
}

template <typename T>
void test_dot_product(T min_val, T max_val, const size_t n) {
   std::unique_ptr<T[]> arr1(new T[n]);
   std::unique_ptr<T[]> arr2(new T[n]);
   for (size_t i = 0; i < n; ++i) {
      if constexpr (std::is_same<T, float>::value) {
         arr1[i] = ((rand() % 10000) / 10000.0) * (max_val - min_val) + min_val;
         arr2[i] = ((rand() % 10000) / 10000.0) * (max_val - min_val) + min_val;
      } else {
         arr1[i] = (rand() % int(max_val - min_val)) + min_val;
         arr2[i] = (rand() % int(max_val - min_val)) + min_val;
      }
   }
   auto x1 = arr1.get();
   auto x2 = arr2.get();

   if constexpr (std::is_same<T, float>::value) {
      float sisd_res     = dot_product_sisd  (x1, x2, n);
      float simd_res_128 = dot_product_avx128(x1, x2, n);
      float simd_res_256 = dot_product_avx256(x1, x2, n);
      printf("test_dot_product() item size:%lu|\tsisd:%g\tavx128:%g\tavx256:%g\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   } else {
      int sisd_res     = dot_product_sisd  (x1, x2, n);
      int simd_res_128 = dot_product_avx128(x1, x2, n);
      int simd_res_256 = dot_product_avx256(x1, x2, n);
      printf("test_dot_product() item size:%lu|\tsisd:%d\tavx128:%d\tavx256:%d\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   }
}

template <typename T>
void test_addv(T min_val, T max_val, const size_t n) {
   std::unique_ptr<T[]> arr0(new T[n]);
   std::unique_ptr<T[]> arr1(new T[n]);
   std::unique_ptr<T[]> arr2(new T[n]);
   for (size_t i = 0; i < n; ++i) {
      if constexpr (std::is_same<T, float>::value) {
         arr1[i] = ((rand() % 10000) / 10000.0) * (max_val - min_val) + min_val;
         arr2[i] = ((rand() % 10000) / 10000.0) * (max_val - min_val) + min_val;
      } else {
         arr1[i] = (rand() % int(max_val - min_val)) + min_val;
         arr2[i] = (rand() % int(max_val - min_val)) + min_val;
      }
   }
   auto x0 = arr0.get();
   auto x1 = arr1.get();
   auto x2 = arr2.get();

   if constexpr (std::is_same<T, float>::value) {
      memcpy(x0, x1, sizeof(T)*n);
      add_sisd  (x0, x2, 0.123, n);
      float sisd_res = square_sum_sisd  (x0, n);

      memcpy(x0, x1, sizeof(T)*n);
      add_avx128(x0, x2, 0.123, n);
      float simd_res_128 = square_sum_avx128(x0, n);

      memcpy(x0, x1, sizeof(T)*n);
      add_avx256(x0, x2, 0.123, n);
      float simd_res_256 = square_sum_avx256(x0, n);
      printf("test_addv() item size:%lu|\tsisd:%g\tavx128:%g\tavx256:%g\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   } else {
      memcpy(x0, x1, sizeof(T)*n);
      add_sisd  (x0, x2, 0.123, n);
      int sisd_res     = square_sum_sisd  (x1, n);

      memcpy(x0, x1, sizeof(T)*n);
      add_avx128(x0, x2, 0.123, n);
      int simd_res_128 = square_sum_avx128(x1, n);

      memcpy(x0, x1, sizeof(T)*n);
      add_avx256(x0, x2, 0.123, n);
      int simd_res_256 = square_sum_avx256(x0, n);
      printf("test_addv() item size:%lu|\tsisd:%d\tavx128:%d\tavx256:%d\n",
         sizeof(T), sisd_res, simd_res_128, simd_res_256);
   }
}

int main(int argc, char** argv) {

   srand(time(nullptr));

   size_t n;
   n = 512;
   test_array_sum<float>(float(-1.0), float(1.), n);
   test_array_sum<short>(short(-5700), short(5700), n);
   test_array_sum<int8_t>(int8_t(-127), int8_t(127), n);
   std::cout << std::endl;

   n = 256;
   test_array_max<float>(float(-1.0), float(1.), n);
   test_array_max<short>(short(-15700), short(15700), n);
   test_array_max<int8_t>(int8_t(-127), int8_t(127), n);
   std::cout << std::endl;

   n = 256;
   test_array_min<float>(float(-1.0), float(1.), n);
   test_array_min<short>(short(-15700), short(15700), n);
   test_array_min<int8_t>(int8_t(-127), int8_t(127), n);
   std::cout << std::endl;

   n = 128;
   test_array_max_abs<float>(float(-1.0), float(1.), n);
   test_array_max_abs<short>(short(-15700), short(15700), n);
   test_array_max_abs<int8_t>(int8_t(-127), int8_t(127), n);
   std::cout << std::endl;

   n = 512;
   test_array_sum<float>(float(-1.0), float(1.), n);
   test_array_sum<short>(short(-5700), short(5700), n);
   test_array_sum<int8_t>(int8_t(-127), int8_t(127), n);
   std::cout << std::endl;

   n = 512;
   test_square_sum<float>(float(-1.0), float(1.), n);
   test_square_sum<short>(short(-5700), short(5700), n);
   test_square_sum<int8_t>(int8_t(-127), int8_t(127), n);
   std::cout << std::endl;

   n = 512;
   test_dot_product<float>(float(-1.0), float(1.), n);
   test_dot_product<short>(short(-15700), short(15700), n);
   test_dot_product<int8_t>(int8_t(-127), int8_t(127), n);
   std::cout << std::endl;

   n = 512;
   test_addv<float>(float(-1.0), float(1.), n);
   std::cout << std::endl;


   return 0;
}
#endif
#endif
