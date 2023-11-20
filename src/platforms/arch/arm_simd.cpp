// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Shaofeng Liu (liushaofeng01@baidu.com)
//
// 


#if defined(__arm64__)
//#if defined(__ARM_ARCH_8_32__)

#include <arm_neon.h>

#include <cstdlib>
#include <cstdint>
#include <cfloat>
#include <cmath>

#include <algorithm>

#include "simd.h"

namespace cpuft::simd {

static constexpr unsigned SIMD_SIZE = 16; // in bytes

auto get_max(auto a, auto b) {
    return a >= b ? a : b;
}
auto get_max(auto a, auto b, auto c) {
    return (a >= b && a >= c) ? a : (b >= c ? b : c);
}

float dot_product(const float* x1, const float* x2, size_t n) noexcept {
    float sum = 0.;
    constexpr size_t simd_size = SIMD_SIZE / sizeof(float);
    const size_t nr = n - n % simd_size;
    if (nr >= simd_size) {
        float32x4_t prod_sum = vdupq_n_f32(0);
        for (size_t i = 0; i < nr; i += simd_size) {
            float32x4_t v1 = vld1q_f32(&x1[i]);
            float32x4_t v2 = vld1q_f32(&x2[i]);
            float32x4_t prod = vmulq_f32(v1, v2);
            prod_sum = vaddq_f32(prod_sum, prod);
        }
        float32x2_t sum_high_low = vpadd_f32(vget_low_f32(prod_sum), vget_high_f32(prod_sum));
        sum = vget_lane_f32(vpadd_f32(sum_high_low, sum_high_low), 0);
    }
    for (size_t i = nr; i < n; ++i) {
        sum += x1[i] * x2[i];
    }
    return sum;
}

int dot_product(const int* x1, const int* x2, size_t n) noexcept {
    if (x1 == nullptr || x2 == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    constexpr size_t simd_size = SIMD_SIZE / sizeof(int);
    const size_t nr = n - n % simd_size;
    int32x4_t prod_sum = vdupq_n_s32(0);
    for (size_t i = 0; i < nr; i += simd_size) {
        int32x4_t v1 = vld1q_s32(&x1[i]);
        int32x4_t v2 = vld1q_s32(&x2[i]);
        int32x4_t prod = vaddq_s32(v1, v2);
        prod_sum = vaddq_s32(prod_sum, prod);
    }
    int32_t result[4];
    vst1q_s32(result, prod_sum);
    sum = result[0] + result[1] + result[2] + result[3];
    for (size_t i = nr; i < n; ++i) {
        sum += x1[i] + x2[i];
    }
    return sum;
}

int dot_product(const short* x1, const short* x2, size_t n) noexcept {
    if (x1 == nullptr || x2 == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    constexpr size_t simd_size = SIMD_SIZE / sizeof(short);
    const size_t nr = n - n % simd_size;
    int32x4_t prod_sum = vdupq_n_s32(0);
    for (size_t i = 0; i < nr; i += simd_size) {
        int16x8_t v1 = vld1q_s16(&x1[i]);
        int16x8_t v2 = vld1q_s16(&x2[i]);
        int32x4_t prod_low = vmull_s16(vget_low_s16(v1), vget_low_s16(v2));
        int32x4_t prod_high = vmull_s16(vget_high_s16(v1), vget_high_s16(v2));
        prod_sum = vaddq_s32(prod_sum, vaddq_s32(prod_low, prod_high));
    }
    int32_t result[4];
    vst1q_s32(result, prod_sum);
    sum = result[0] + result[1] + result[2] + result[3];
    for (size_t i = nr; i < n; ++i) {
        sum += x1[i] * x2[i];
    }
    return sum;
}

int dot_product(const int8_t* x1, const int8_t* x2, size_t n) noexcept {
    int sum = 0;
    constexpr size_t simd_size = SIMD_SIZE;
    const size_t nr = n - n % simd_size;

    int32x4_t prod_sum = vdupq_n_s32(0);
    for (size_t i = 0; i < nr; i += simd_size) {
        int8x16_t v1 = vld1q_s8(&x1[i]);
        int8x16_t v2 = vld1q_s8(&x2[i]);
        int16x8_t temp_low = vmull_s8(vget_low_s8(v1), vget_low_s8(v2));
        int16x8_t temp_high = vmull_s8(vget_high_s8(v1), vget_high_s8(v2));
        int32x4_t low_sum = vpaddlq_s16(temp_low);
        int32x4_t high_sum = vpaddlq_s16(temp_high);
        prod_sum = vaddq_s32(prod_sum, low_sum);
        prod_sum = vaddq_s32(prod_sum, high_sum);
    }

    int32_t result[4];
    vst1q_s32(result, prod_sum);
    sum = result[0] + result[1] + result[2] + result[3];

    for (size_t i = nr; i < n; ++i) {
        sum += x1[i] * x2[i];
    }

    return sum;
}

void swiglu(float* xo, const float* xi, size_t n) noexcept {
    for (size_t i = 0; i < n; i++) {
        xo[i] = xo[i] / (1. + expf(-xo[i])) * xi[i];
    }
}
/* void swiglu(float *xo, const float *xi, size_t n) noexcept {
    const float32x4_t one = vdupq_n_f32(1.0f);

    for (size_t i = 0; i < n; i += 4) {
        // Load 4 float values from xo and xi
        float32x4_t xo_vec = vld1q_f32(xo + i);
        float32x4_t xi_vec = vld1q_f32(xi + i);

        // Calculate exp(-xo) for 4 float values in parallel
        float32x4_t neg_xo_vec = vnegq_f32(xo_vec);
        float32x4_t exp_vec = vexpq_f32(neg_xo_vec);

        // Calculate (1 + exp(-xo)) for 4 float values in parallel
        float32x4_t one_plus_exp_vec = vaddq_f32(one, exp_vec);

        // Calculate (xo / (1 + exp(-xo))) for 4 float values in parallel
        float32x4_t div_vec = vdivq_f32(xo_vec, one_plus_exp_vec);

        // Calculate (xo / (1 + exp(-xo))) * xi for 4 float values in parallel
        float32x4_t result_vec = vmulq_f32(div_vec, xi_vec);

        // Store the result back to xo
        vst1q_f32(xo + i, result_vec);
    }
}*/

float square_sum(const float* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }
    float sum = 0.;
    constexpr size_t simd_size = SIMD_SIZE / sizeof(float);
    const size_t nr = n - n % simd_size;
    float32x4_t prod_sum = vdupq_n_f32(0);
    for (size_t i = 0; i < nr; i += simd_size) {
        float32x4_t v = vld1q_f32(&x[i]);
        float32x4_t prod = vmulq_f32(v, v);
        prod_sum = vaddq_f32(prod_sum, prod);
    }
    float result[4];
    vst1q_f32(result, prod_sum);
    sum = result[0] + result[1] + result[2] + result[3];
    for (size_t i = nr; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

int square_sum(const short* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }

    int sum = 0;
    constexpr size_t simd_size = SIMD_SIZE / sizeof(short);
    const size_t nr = n - n % simd_size;

    if (nr >= simd_size) {
        int32x4_t sum_squares = vdupq_n_s32(0);
        for (size_t i = 0; i < nr; i += simd_size) {
            int16x8_t v1    = vld1q_s16(&x[i]);
            int32x4_t v1_32_low  = vmovl_s16(vget_low_s16(v1));
            int32x4_t v1_32_high = vmovl_s16(vget_high_s16(v1));
            int32x4_t prod_low   = vmulq_s32(v1_32_low, v1_32_low);
            int32x4_t prod_high  = vmulq_s32(v1_32_high, v1_32_high);
            sum_squares = vaddq_s32(sum_squares, prod_low);
            sum_squares = vaddq_s32(sum_squares, prod_high);
        }

        // Reduce sum_squares to a single value
        int32_t result[4];
        vst1q_s32(result, sum_squares);
        sum = result[0] + result[1] + result[2] + result[3];
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
    constexpr size_t simd_size = SIMD_SIZE;
    const size_t nr = n - n % simd_size;

    if (nr >= simd_size) {
        int32x4_t sum_vec = vdupq_n_s32(0);
        for (size_t i = 0; i < nr; i += simd_size) {
            int8x16_t vec = vld1q_s8(&x[i]);
            int16x8_t vec_low = vmovl_s8(vget_low_s8(vec));
            int16x8_t vec_high = vmovl_s8(vget_high_s8(vec));
            int32x4_t sqr_low = vmull_s16(vget_low_s16(vec_low), vget_low_s16(vec_low));
            int32x4_t sqr_high = vmull_s16(vget_high_s16(vec_low), vget_high_s16(vec_low));
            sqr_low = vmlal_s16(sqr_low, vget_low_s16(vec_high), vget_low_s16(vec_high));
            sqr_high = vmlal_s16(sqr_high, vget_high_s16(vec_high), vget_high_s16(vec_high));
            sum_vec = vaddq_s32(sum_vec, sqr_low);
            sum_vec = vaddq_s32(sum_vec, sqr_high);
        }

        // Reduce sum_vec to a single integer sum
        int32_t result[4];
        vst1q_s32(result, sum_vec);
        sum = result[0] + result[1] + result[2] + result[3];
    }

    // Process remaining elements
    for (size_t i = nr; i < n; ++i) {
        sum += int(x[i]) * int(x[i]);
    }

    return sum;
}

float array_sum(const float* x, size_t n) noexcept {
    if (x == nullptr) {
        return 0.;
    }

    float sum = 0.;

    constexpr size_t simd_size = SIMD_SIZE / sizeof(float);
    const size_t nr = n - n % simd_size;
    size_t i = 0;
    if (nr >= simd_size) {
        float32x4_t sum_vec = vdupq_n_f32(0);
        for (; i < nr; i += simd_size) {
            float32x4_t x_vec = vld1q_f32(&x[i]);
            sum_vec = vaddq_f32(sum_vec, x_vec);
        }
        float result[4];
        vst1q_f32(result, sum_vec);
        sum = result[0] + result[1] + result[2] + result[3];
    }
    for (; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

int array_sum(const int* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    constexpr size_t simd_size = SIMD_SIZE / sizeof(int);
    const size_t nr = n - n % simd_size;
    int32x4_t sum_vec = vdupq_n_s32(0);

    for (size_t i = 0; i < nr; i += simd_size) {
        int32x4_t x_vec = vld1q_s32(&x[i]);
        sum_vec = vaddq_s32(sum_vec, x_vec);
    }

    int32_t result[4];
    vst1q_s32(result, sum_vec);
    sum = result[0] + result[1] + result[2] + result[3];

    for (size_t i = nr; i < n; ++i) {
        sum += x[i];
    }

    return sum;
}

int array_sum(const short* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    constexpr size_t simd_size = SIMD_SIZE / sizeof(short);
    const size_t nr = n - n % simd_size;
    int16x8_t sum_simd = vdupq_n_s16(0);
    for (size_t i = 0; i < nr; i += simd_size) {
        int16x8_t x_simd = vld1q_s16(&x[i]);
        sum_simd = vaddq_s16(sum_simd, x_simd);
    }
    int16_t result[simd_size];
    vst1q_s16(result, sum_simd);
    for (size_t i = 0; i < simd_size; ++i) {
        sum += result[i];
    }
    for (size_t i = nr; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

int array_sum(const int8_t* x, size_t n) noexcept {
    if (x == nullptr || n < 1) {
        return 0;
    }
    int sum = 0;
    constexpr size_t simd_size = SIMD_SIZE;
    const size_t nr = n - n % simd_size;
    int16x8_t prod_sum1 = vdupq_n_s16(0);
    int16x8_t prod_sum2 = vdupq_n_s16(0);
    for (size_t i = 0; i < nr; i += simd_size) {
        int8x16_t v = vld1q_s8(&x[i]);
        int16x8_t v_low = vmovl_s8(vget_low_s8(v));
        int16x8_t v_high = vmovl_s8(vget_high_s8(v));
        prod_sum1 = vaddq_s16(prod_sum1, v_low);
        prod_sum2 = vaddq_s16(prod_sum2, v_high);
    }
    int16_t result1[8];
    int16_t result2[8];
    vst1q_s16(result1, prod_sum1);
    vst1q_s16(result2, prod_sum2);
    sum = result1[0] + result1[1] + result1[2] + result1[3] + result1[4] + result1[5] + result1[6] + result1[7]
        + result2[0] + result2[1] + result2[2] + result2[3] + result2[4] + result2[5] + result2[6] + result2[7];
    for (size_t i = nr; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

float array_max(const float* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return FLT_MIN;
    }

    float max = FLT_MIN;

    constexpr size_t simd_size = SIMD_SIZE / sizeof(float);
    const size_t nr = n - n % simd_size;
    size_t i = 0;
    if (nr >= simd_size) {
        float32x4_t max_vec = vdupq_n_f32(FLT_MIN);
        for (; i < nr; i += simd_size) {
            float32x4_t arr_vec = vld1q_f32(&arr[i]);
            max_vec = vmaxq_f32(max_vec, arr_vec);
        }

        // Reduce the max_vec to a single maximum value
        float32x2_t max_vec_high = vget_high_f32(max_vec);
        float32x2_t max_vec_low = vget_low_f32(max_vec);
        float32x2_t max_vec_2 = vpmax_f32(max_vec_high, max_vec_low);
        float32x2_t max_vec_1 = vpmax_f32(max_vec_2, max_vec_2);
        max = vget_lane_f32(max_vec_1, 0);
    }

    for (; i < n; ++i) {
        if (arr[i] > max) max = arr[i];
    }

    return max;
}

int8_t array_max(const int8_t* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return INT8_MIN;
    }
    int8_t max = arr[0];
    size_t i = 0;

    constexpr size_t simd_size = SIMD_SIZE;

    // 使用SIMD处理16个元素
    if (n >= simd_size) {
        int8x16_t max_abs = vabsq_s8(vld1q_s8(arr));
        for (i = simd_size; i < n; i += simd_size) {
            int8x16_t current = vabsq_s8(vld1q_s8(arr + i));
            max_abs = vmaxq_s8(max_abs, current);
        }

        // 从SIMD向量中提取最大值
        int8x8_t max_abs_low = vmax_s8(vget_low_s8(max_abs), vget_high_s8(max_abs));
        int8x8_t max_abs_high = vext_s8(max_abs_low, max_abs_low, 4);
        max_abs_low = vmax_s8(max_abs_low, max_abs_high);
        max_abs_high = vext_s8(max_abs_low, max_abs_low, 2);
        max_abs_low = vmax_s8(max_abs_low, max_abs_high);
        max_abs_high = vext_s8(max_abs_low, max_abs_low, 1);
        max = vmax_s8(max_abs_low, max_abs_high)[0];
    }

    // 处理剩余的元素
    for (; i < n; ++i) {
        if (arr[i] > max) max = arr[i];
    }

    return max;
}

short array_max(const short* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return INT16_MIN;
    }

    constexpr size_t simd_size = SIMD_SIZE / sizeof(short);
    size_t simd_iterations = n / simd_size;

    int16x8_t max_vector = vld1q_s16(arr);

    for (size_t i = 1; i < simd_iterations; ++i) {
        int16x8_t current_vector = vld1q_s16(arr + i * simd_size);
        max_vector = vmaxq_s16(max_vector, current_vector);
    }

    int16_t max_values[simd_size];
    vst1q_s16(max_values, max_vector);

    short simd_max = max_values[0];
    for (int i = 1; i < 8; ++i) {
        if (max_values[i] > simd_max) {
            simd_max = max_values[i];
        }
    }

    for (size_t i = simd_iterations * simd_size; i < n; ++i) {
        if (arr[i] > simd_max) {
            simd_max = arr[i];
        }
    }

    return simd_max;
}

float array_max_abs(const float * arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return 0;
    }

    float max = 0;

    constexpr size_t simd_size = SIMD_SIZE / sizeof(float);
    const size_t nr = n - n % simd_size;
    size_t i = 0;
    if (nr >= simd_size) {
        float32x4_t max_vec = vdupq_n_f32(0);
        for (; i < nr; i += simd_size) {
            float32x4_t arr_vec = vld1q_f32(&arr[i]);
            float32x4_t arr_vec_abs = vabsq_f32(arr_vec);
            max_vec = vmaxq_f32(max_vec, arr_vec_abs);
        }
        float max_arr[simd_size];
        vst1q_f32(max_arr, max_vec);
        for (size_t j = 0; j < simd_size; ++j) {
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

short array_max_abs(const short* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return 0;
    }

    constexpr size_t simd_size = SIMD_SIZE / sizeof(short);
    const size_t nr = n - n % simd_size;
    int16x8_t max_abs_vec = vdupq_n_s16(0);

    for (size_t i = 0; i < nr; i += simd_size) {
        int16x8_t vec = vld1q_s16(&arr[i]);
        int16x8_t abs_vec = vabsq_s16(vec);
        max_abs_vec = vmaxq_s16(max_abs_vec, abs_vec);
    }

    int16_t max_values[simd_size];
    vst1q_s16(max_values, max_abs_vec);

    short max_abs = 0;
    for (int i = 0; i < simd_size; ++i) {
        if (max_values[i] > max_abs) {
            max_abs = max_values[i];
        }
    }

    for (size_t i = nr; i < n; ++i) {
        short v = arr[i] >= 0 ? arr[i] : -arr[i];
        if (v > max_abs) {
            max_abs = v;
        }
    }

    return max_abs;
}

int8_t array_max_abs(const int8_t* arr, size_t n) noexcept {
    int8_t vmax = 0;
    int8x16_t max_abs_vector = vdupq_n_s8(0);

    constexpr size_t simd_size = SIMD_SIZE;
    const size_t nr = n & ~(simd_size - 1);

    size_t i = 0;
    if (nr > 0) {
        for (; i < nr; i += simd_size) {
            // Load 16 bytes from arr into a NEON vector
            int8x16_t data_vector = vld1q_s8(arr + i);

            // Take the absolute value of each element in the vector
            int8x16_t abs_vector = vabsq_s8(data_vector);

            // Update the running maximum
            max_abs_vector = vmaxq_s8(max_abs_vector, abs_vector);
        }

        // Reduce the maximum value in the vector to a single value
        vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 0));
        vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 1));
        vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 2));
        vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 3));
        vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 4));
        vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 5));
        vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 6));
        vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 7));
        if (SIMD_SIZE > 8) {
            vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector,  8));
            vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector,  9));
            vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 10));
            vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 11));
            vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 12));
            vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 13));
            vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 14));
            vmax = get_max(vmax, vgetq_lane_s8(max_abs_vector, 15));
        }
    }
    for (; i < n; ++i) {
        if (arr[i] > vmax) {
            vmax = arr[i];
        } else if (-arr[i] > vmax) {
            vmax = -arr[i];
        }
    }

    return vmax;
}

float array_min(const float* arr, size_t n) noexcept {
    if (n == 0) return 0;
    constexpr size_t simd_size = SIMD_SIZE / sizeof(float);
    const size_t nr = n - n % simd_size;
    float32x4_t min_vec = vdupq_n_f32(arr[0]);
    for (size_t i = 0; i < nr; i += simd_size) {
        float32x4_t v = vld1q_f32(&arr[i]);
        min_vec = vminq_f32(min_vec, v);
    }
    float min = std::min(vgetq_lane_f32(min_vec, 0), std::min(vgetq_lane_f32(min_vec, 1), std::min(vgetq_lane_f32(min_vec, 2), vgetq_lane_f32(min_vec, 3))));
    for (size_t i = nr; i < n; ++i) {
        if (arr[i] < min) min = arr[i];
    }
    return min;
}

MinMax array_min_max(const float* x, size_t n) noexcept {
  if (x == nullptr || n < 1) {
    return {0.0f, 0.0f};
  }

  constexpr size_t simd_size = SIMD_SIZE / sizeof(float);
  const size_t nr = n - n % simd_size;

  float32x4_t min_vec = vld1q_f32(x);
  float32x4_t max_vec = vld1q_f32(x);

  for (size_t i = simd_size; i < nr; i += simd_size) {
    float32x4_t cur_vec = vld1q_f32(x + i);
    min_vec = vminq_f32(min_vec, cur_vec);
    max_vec = vmaxq_f32(max_vec, cur_vec);
  }

  float min_array[simd_size];
  float max_array[simd_size];
  vst1q_f32(min_array, min_vec);
  vst1q_f32(max_array, max_vec);

  float min_val = std::min({min_array[0], min_array[1], min_array[2], min_array[3]});
  float max_val = std::max({max_array[0], max_array[1], max_array[2], max_array[3]});

  for (size_t i = nr; i < n; ++i) {
    if (x[i] < min_val) {
      min_val = x[i];
    } else if (x[i] > max_val) {
      max_val = x[i];
    }
  }

  return {min_val, max_val};
}

short array_min(const short* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return 0;
    }
    constexpr size_t simd_size = SIMD_SIZE / sizeof(short);
    const size_t nr = n - n % simd_size;
    int16x8_t min_vec = vdupq_n_s16(SHRT_MAX);
    for (size_t i = 0; i < nr; i += simd_size) {
        int16x8_t vec = vld1q_s16(&arr[i]);
        min_vec = vminq_s16(min_vec, vec);
    }
    int16_t min_values[simd_size];
    vst1q_s16(min_values, min_vec);
    short min = min_values[0];
    for (size_t i = 1; i < simd_size; ++i) {
        if (min_values[i] < min) min = min_values[i];
    }
    for (size_t i = nr; i < n; ++i) {
        if (arr[i] < min) min = arr[i];
    }
    return min;
}

int8_t array_min(const int8_t* arr, size_t n) noexcept {
    if (arr == nullptr || n < 1) {
        return 0;
    }

    constexpr size_t simd_size = SIMD_SIZE;
    const size_t nr = n - n % simd_size;
    int8x16_t min_vector = vdupq_n_s8(INT8_MAX);

    for (size_t i = 0; i < nr; i += simd_size) {
        int8x16_t v1 = vld1q_s8(&arr[i]);
        min_vector = vminq_s8(min_vector, v1);
    }

    int8_t min_array[16];
    vst1q_s8(min_array, min_vector);

    int8_t min = min_array[0];
    for (int i = 1; i < 16; ++i) {
        if (min_array[i] < min) {
            min = min_array[i];
        }
    }

    for (size_t i = nr; i < n; ++i) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }

    return min;
}

void rmsnorm(float* o, const float* x, const float* w, size_t n) noexcept {
    const float r = 1. / sqrtf(square_sum(x, n) / n + 1e-5f);

    constexpr size_t simd_size = SIMD_SIZE / sizeof(float);
    const size_t nr = n - n % simd_size;

    float32x4_t scalar = vdupq_n_f32(r);

    for (size_t i = 0; i < nr; i += simd_size) {
        float32x4_t v1 = vld1q_f32(&x[i]);
        float32x4_t v2 = vld1q_f32(&w[i]);
        float32x4_t prod = vmulq_f32(v1, v2);
        float32x4_t result = vmulq_f32(prod, scalar);
        vst1q_f32(&o[i], result);
    }

    for (size_t i = nr; i < n; ++i) {
        o[i] = x[i] * (r * w[i]);
    }
}

void multiply(float* x, float v, size_t n) noexcept {
    constexpr int simd_size = SIMD_SIZE / sizeof(float);
    size_t nr = n - n % simd_size;
    float32x4_t scalar = vdupq_n_f32(v);

    for (size_t i = 0; i < nr; i += simd_size) {
        float32x4_t vec = vld1q_f32(&x[i]);
        vec = vmulq_f32(vec, scalar);
        vst1q_f32(&x[i], vec);
    }

    for (size_t i = nr; i < n; ++i) {
        x[i] *= v;
    }
}

void add(float* x2, const float* x1, size_t n) noexcept {
    constexpr int simd_size = SIMD_SIZE / sizeof(float);
    const size_t nr = n - (n % simd_size);
    for (size_t i = 0; i < nr; i += simd_size) {
        float32x4_t vec1 = vld1q_f32(x1 + i);
        float32x4_t vec2 = vld1q_f32(x2 + i);
        float32x4_t res = vaddq_f32(vec1, vec2);
        vst1q_f32(x2 + i, res);
    }
    for (size_t i = nr; i < n; ++i) {
        x2[i] += x1[i];
    }
}

void add(float* x, float v, size_t n) noexcept {
    constexpr size_t simd_size = SIMD_SIZE / sizeof(float);
    size_t nr = n - (n % simd_size);

    if (nr >= simd_size) {
        float32x4_t vec_v = vdupq_n_f32(v);
        for (size_t i = 0; i < nr; i += simd_size) {
            float32x4_t vec_x = vld1q_f32(x + i);
            float32x4_t res = vaddq_f32(vec_x, vec_v);
            vst1q_f32(x + i, res);
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
    float32x4_t r_vec = vdupq_n_f32(r);
    constexpr size_t simd_width = 4;
    const size_t simd_iters = n / simd_width;

    for (size_t i = 0; i < simd_iters; ++i) {
        float32x4_t x1_vec = vld1q_f32(&x[i * simd_width]);
        float32x4_t x2_vec = vld1q_f32(&o[i * simd_width]);
        float32x4_t res_vec = vfmaq_f32(x2_vec, x1_vec, r_vec);
        vst1q_f32(&o[i * simd_width], res_vec);
    }

    for (size_t i = simd_iters * simd_width; i < n; ++i) {
        o[i] += x[i] * r;
    }
}

} // namespace cpuft::simd
//#endif
#endif
