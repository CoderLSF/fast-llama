/************************************************************************

 @Author: Liu Shaofeng
 @Date:   2023/10/21 08:38
 @Brief:  General operators for quantization
 ************************************************************************/
#pragma once

#include <stdint.h>
#include <stdlib.h>

#include <vector>
#include <utility>

namespace cpuft::quant {

enum class QuantType {
    NONE = 0, // no quantization
    INT16,    // 16-bits quantization
    INT8,     //  8-bits quantization
    INT4,     //  4-bits quantization

    _MAX_,
};
union Bit4Pair {
    struct {
        int8_t a:4;
        int8_t b:4;
    };
    uint8_t v;
};
constexpr int QUANT16_FACTOR = 5792;
constexpr int QUANT8_FACTOR  = 127;
constexpr int QUANT4_FACTOR  = 7;

// n 是64和group_size的整数倍
void   quantize(QuantType qt, void*  qx,  float* qs,      const float* x,  size_t n, int group_size) noexcept;
void dequantize(QuantType qt, float* out, const void* qx, const float* qs, size_t n, int group_size) noexcept;

inline float get_value(QuantType qt, const void* qx, const float* qs, size_t i, int group_size) noexcept {
    switch(qt) {
        case QuantType::NONE:
            return reinterpret_cast<const float*>(qx)[i];
        case QuantType::INT16:
            return reinterpret_cast<const short*>(qx)[i] * qs[i/group_size];
        case QuantType::INT8:
            return reinterpret_cast<const int8_t*>(qx)[i] * qs[i/group_size];
        case QuantType::INT4:
            if ((i&1) == 0) {
                return reinterpret_cast<const Bit4Pair*>(qx)[i/2].a * qs[i/group_size];
            } else {
                return reinterpret_cast<const Bit4Pair*>(qx)[i/2].b * qs[i/group_size];
            }
        default:
            break;
    }
    return 0.;
}

void fill_random(QuantType qt, void* qx, float* qs, size_t n, int group_size, float min_value, float max_value) noexcept;

float sum(QuantType qt, const void* qx, const float* qs, size_t n, int group_size) noexcept;
float min(QuantType qt, const void* qx, const float* qs, size_t n, int group_size) noexcept;
float max(QuantType qt, const void* qx, const float* qs, size_t n, int group_size) noexcept;

void add(QuantType qt, void* qx1, float* scales1, const void* qx2, const float* scales2, size_t n, int group_size) noexcept;
void add(QuantType qt, void* qx, float* scales, float v, size_t n, int group_size) noexcept;
void mul(QuantType qt, void* qx, float factor, float* qs, size_t n, int group_size) noexcept;

float dot_product(QuantType qt, const void* qx1, const float* qs1, const void* qx2, const float* qs2, size_t n, int gs) noexcept;
float square_sum(QuantType qt, const void* qx, const float* qs, size_t n, int gs) noexcept;

void weighted_sum(QuantType qt, float* out, const void* matrix, const float* scales, int group_size,
                const float* weights, int m, int n, float min_w=0.) noexcept;
void weighted_sum(QuantType qt, float* out, const void* matrix, const float* scales, int group_size,
                const float* weights, int m, int n, int bs, float min_w=0.) noexcept;

// n 是64和group_size的整数倍
void matmul(QuantType qt, float* qout,
            const void* qmat1, const float* scales1,
            const void* qmat2, const float* scales2,
            int m, int n, int w, int group_size);

} // namespace cpuft::quant
