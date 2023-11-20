
#include "quant_operators.h"

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <random>
#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <memory>
#include <utility>
#include <type_traits>

#include "base_constants.h"
#include "tf_operators.h"
#include "utility.h"
#include "simd.h"

namespace cpuft::quant {

template <typename T>
inline void quantize(T* qx, float* qs, const float* x, size_t n, int gs) noexcept {
    constexpr float QF = (std::is_same<T, Bit4Pair>::value) ? QUANT4_FACTOR : (sizeof(T) == 1 ? QUANT8_FACTOR : QUANT16_FACTOR);
    //constexpr float  F = QF - 0.5/QF;
    constexpr float  F = QF;
    for (size_t i = 0, e = (n+gs-1)/gs; i < e; ++i, x += gs) {
        auto gn = std::min(size_t(n - gs*i), size_t(gs));
        float r = array_max_abs(x, gn) / F;
        qs[i] = r;
        if constexpr (std::is_same<T, Bit4Pair>::value) {
            for (size_t j = 0; j < gn; j += 2) {
                qx[j/2].v = (uint8_t(x[j] / r) & 15) | (uint8_t(x[j+1] / r) << 4);
            }
            qx += gs/2;
        } else {
            for (size_t j = 0; j < gn; ++j) {
                qx[j] = x[j] / r;
            }
            qx += gs;
        }
    }
}

template <typename T>
inline void dequantize_(float* out, const T* qx, const float* qs, size_t n, int gs) noexcept {
    for (size_t i = 0; i < n; i += gs) {
        float r = qs[i/gs];
        auto gn = std::min(size_t(n), size_t(i+gs));
        if constexpr (std::is_same<T, Bit4Pair>::value) {
            for (size_t j = i; j < gn; j += 2) {
                out[j]   = qx[j/2].a * r;
                out[j+1] = qx[j/2].b * r;
            }
        } else {
            for (size_t j = i; j < gn; ++j) {
                out[j]   = qx[j] * r;
            }
        }
    }
}

inline void dequantize_(float* out, const Bit4Pair* qx, const float* qs, size_t n, int gs) noexcept {
    for (size_t i = 0; i < n; i += gs) {
        float s = qs[i/gs];
        auto gn = std::min(size_t(n), size_t(i+gs));
        for (size_t j = i; j < gn; j+=2) {
            out[j]   = int(qx[j/2].a) * s;
            out[j+1] = int(qx[j/2].b) * s;
        }
    }
}

void quantize(QuantType qt, void* qx, float* qs, const float* x, size_t n, int gs) noexcept {
    if (qx == nullptr || x == nullptr || n < 1) {
        return;
    }
    if (qt == QuantType::NONE) {
        memcpy(qx, x, sizeof(float) * n);
        return;
    }
    if (qs == nullptr || gs < 1) {
        return;
    }

    if (qt == QuantType::INT16) {
        quantize(reinterpret_cast<short*>(qx), qs, x, n, gs);
    } else if (qt == QuantType::INT8) {
        quantize(reinterpret_cast<int8_t*>(qx), qs, x, n, gs);
    } else if (qt == QuantType::INT4) {
        quantize(reinterpret_cast<Bit4Pair*>(qx), qs, x, n, gs);
    }
}

void dequantize(QuantType qt, float* out, const void* qx, const float* qs,  size_t n, int gs) noexcept {
    if (qx == nullptr || out == nullptr || n < 1) {
        return;
    }
    if (qt == QuantType::NONE) {
        memcpy(out, qx, sizeof(float) * n);
        return;
    }
    if (qs == nullptr || gs < 1) {
        return;
    }
    if (qt == QuantType::INT16) {
        return dequantize_(out, (const short*)(qx),   qs, n, gs);
    } else if (qt == QuantType::INT8) {
        return dequantize_(out, (const int8_t*)(qx),   qs, n, gs);
    } else if (qt == QuantType::INT4) {
        return dequantize_(out, (const Bit4Pair*)(qx), qs, n, gs);
    }
}

inline float sum(const float* qx, size_t n) noexcept {
    return cpuft::array_sum(qx, n);
}
template <typename T>
inline float sum(const T* qx, const float* qs, size_t n, int gs) noexcept {
    float sum = 0.;
    for (size_t i = 0; i < n; i += gs) {
        auto v = cpuft::array_sum(qx+i, std::min(size_t(gs), size_t(n-i)));
        sum += qs[i/gs] * v;//cpuft::array_sum(qx+i, std::min(size_t(gs), size_t(n-i)));
    }
    return sum;
}

inline float sum(const Bit4Pair* qx, const float* qs, size_t n, int gs) noexcept {
    n /= 2;
    gs /= 2;
    float sum = 0.;
    for (size_t i = 0; i < n; i += gs) {
        auto gn = std::min(int(n), int(i+gs));
        int v = 0;
        for (int j = i; j < gn; ++j) {
            v += qx[j].a + qx[j].b;
        }
        sum += qs[i/gs] * v;
    }
    return sum;
}

float sum(QuantType qt, const void* qx, const float* qs, size_t n, int group_size) noexcept {
    if (qt == QuantType::NONE) {
        return cpuft::array_sum(reinterpret_cast<const float*>(qx), n);
    } else if (qt == QuantType::INT16) {
        return sum(reinterpret_cast<const short*>(qx), qs, n, group_size);
    } else if (qt == QuantType::INT8) {
        return sum(reinterpret_cast<const int8_t*>(qx), qs, n, group_size);
    } else if (qt == QuantType::INT4) {
        return sum(reinterpret_cast<const Bit4Pair*>(qx), qs, n, group_size);
    }
    return 0.;
}

template <typename T>
inline float max(const T* qx, const float* qs, size_t n, int gs) noexcept {
    float res = FLT_MIN;
    for (size_t i = 0; i < n; i += gs) {
        auto v = qs[i/gs] * cpuft::array_max(qx+i, std::min(size_t(n-i), size_t(gs)));
        if (v > res) {
            res = v;
        }
    }
    return res;
}

inline float max(const Bit4Pair* qx, const float* qs, size_t n, int gs) noexcept {
    float res = FLT_MIN;
    n /= 2;
    gs /= 2;
    for (size_t i = 0; i < n; i += gs) {
        int v = INT32_MAX;
        for (size_t j = i, e = std::min(size_t(n), size_t(i+gs)); j < e; ++j) {
            if (qx[j].a < v) {
                v = qx[j].a;
            } else if (qx[j].b < v) {
                v = qx[j].b;
            }
        }
        auto x = qs[i/gs] * v;
        if (x > res) {
            res = x;
        }
    }
    return res;
}

template <typename T>
inline float min(const T* qx, const float* qs, size_t n, int gs) noexcept {
    float res = FLT_MAX;
    for (size_t i = 0; i < n; i += gs) {
        auto v = qs[i/gs] * cpuft::array_max(qx+i, std::min(size_t(n-i), size_t(gs)));
        if (v < res) {
            res = v;
        }
    }
    return res;
}

inline float min(const Bit4Pair* qx, const float* qs, size_t n, int gs) noexcept {
    float res = FLT_MAX;
    n /= 2;
    gs /= 2;
    for (size_t i = 0; i < n; i += gs) {
        int v = -16;
        for (size_t j = i, e = std::min(size_t(n), size_t(i+gs)); j < e; ++j) {
            if (qx[j].a > v) {
                v = qx[j].a;
            } else if (qx[j].b > v) {
                v = qx[j].b;
            }
        }
        auto x = qs[i/gs] * v;
        if (x > res) {
            res = x;
        }
    }
    return res;
}

float max(QuantType qt, const void* qx, const float* qs, size_t n, int gs) noexcept {
    if (qt == QuantType::NONE) {
        return cpuft::array_max(reinterpret_cast<const float*>(qx), n);
    } else if (qt == QuantType::INT16) {
        return max(reinterpret_cast<const short*>(qx), qs, n, gs);
    } else if (qt == QuantType::INT8) {
        return max(reinterpret_cast<const int8_t*>(qx), qs, n, gs);
    } else if (qt == QuantType::INT4) {
        return max(reinterpret_cast<const Bit4Pair*>(qx), qs, n, gs);
    }
    return FLT_MIN;
}

float min(QuantType qt, const void* qx, const float* qs, size_t n, int gs) noexcept {
    if (qt == QuantType::NONE) {
        return cpuft::array_min(reinterpret_cast<const float*>(qx), n);
    } else if (qt == QuantType::INT16) {
        return min(reinterpret_cast<const short*>(qx), qs, n, gs);
    } else if (qt == QuantType::INT8) {
        return min(reinterpret_cast<const int8_t*>(qx), qs, n, gs);
    } else if (qt == QuantType::INT4) {
        return min(reinterpret_cast<const Bit4Pair*>(qx), qs, n, gs);
    }
    return FLT_MIN;
}

template <typename T>
void matmul(float* out,
            const T* mat1, const float* scales1,
            const T* mat2, const float* scales2,
            int m, int n, int w, int gs) {
    constexpr int BS = 16;
    const int sn = (n+gs-1)/gs;

    for (int wi = 0; wi < w; wi += BS) {
        const int wn = BS <= w - wi ? BS : w-wi;
        for (int mi = 0; mi < m; mi += gs) {
            const int mn = std::min(int(mi+gs), int(m));
            for (int i = 0; i < wn; ++i) {
                memset(out + m * i + mi, 0, sizeof(float) * (mn-mi));
            }

            for (int ni = 0; ni < n; ni += gs) {
                auto x2 = mat2 + ni;
                auto s2 = scales2 + ni / gs;
                auto o = out;
                for (int i = 0; i < wn; ++i, o += m, x2 += n, s2 += sn) {
                    for (int j = mi; j < mn; ++j) {
                        auto s = scales1[(n*j+ni)/gs] * *s2;
                        o[j] += s * simd::dot_product(x2, &mat1[n * j + ni], gs <= n - ni ? gs : n - ni);
                    }
                }
            }
        }
        out  += m * BS;
        mat2 += n * BS;
        scales2 += sn * BS;
    }
}

template <typename T>
void matmul1(float* out,
            const T* mat1, const float* scales1,
            const T* mat2, const float* scales2,
            int m, int n, int w, int gs) {
    const int sn = (n+gs-1)/gs;
    auto& x1 = mat1;
    auto& s1 = scales1;
    for (int i = 0; i < m; ++i, x1 += n, s1 += sn) {
        auto x2 = mat2;
        auto s2 = scales2;
        auto po = out + i;
        for (int j = 0; j < w; ++j, x2 += n, s2 += sn) {
            float sum = 0;
            for (int k = 0; k < n; k += gs) {
                int v = simd::dot_product(x1+k, x2+k, std::min(int(n-k), int(gs)));
                sum += v * s1[k/gs] * s2[k/gs];
            }
            po[m*j] = sum;
        }
    }
}

void matmul(float* out,
            const Bit4Pair* mat1, const float* scales1,
            const Bit4Pair* mat2, const float* scales2,
            int m, int n, int w, int gs) {
    int8_t tmp1[256];
    int8_t tmp2[256];

    n  /= 2;
    gs /= 2;
    auto& x1 = mat1;
    for (int t = 0; t < m; ++t, x1 += n) {
        auto x2 = mat2;
        for (int k = 0; k < w; ++k, x2 += n) {
            for (int i = 0; i < gs; ++i) {
                tmp2[i]   = x2[i].a;
                tmp2[i+1] = x2[i].b;
            }
            float sum = 0;
            for (int i = 0; i < n; i += gs) {
                for (int j = 0; j < gs; ++j) {
                    tmp1[j*2]   = x1[j].a;
                    tmp1[j*2+1] = x1[j].b;
                }
                //int v = dot_product(tmp1, tmp2, std::min(int(n-i), int(gs)));
                sum += scales1[i/gs] * scales2[i/gs] * simd::dot_product(&tmp1[0], &tmp2[0], gs*2);
            }
            out[m*k+t] = sum;
        }
    }
}

void matmul(float* out, const float* mat1, const float* mat2, int m, int n, int k) noexcept {
    const float* x1 = mat1;
    for (int i = 0; i < m; i += 1, x1 += n) {
        const float* x2 = mat2;
        for (int j = 0; j < k; ++j, x2 += n) {
            out[m*j+i] = simd::dot_product(x1, x2, n);
        }
    }
}

template <typename T>
inline void add(T* qx1, float* qs1, const T* qx2, const float* qs2, size_t n, int gs) noexcept {
    float tmp[256];
    for (size_t i = 0; i < n; i += gs) {
        auto& s1 = qs1[i/gs];
        auto& s2 = qs2[i/gs];
        auto gn = std::min(size_t(n-i), size_t(gs));
        for (size_t j = 0; j < gn; ++j) {
            tmp[j] = s1 * qx1[i+j] + s2 * qx2[i+j];
        }
        quantize(qx1+i, &qs1[i/gs], tmp, gn, gs);
    }
}
inline void add(Bit4Pair* qx1, float* qs1, const Bit4Pair* qx2, const float* qs2, size_t n, int gs) noexcept {
    float tmp[256];
    for (size_t i = 0; i < n; i += gs) {
        auto& s1 = qs1[i/gs];
        auto& s2 = qs2[i/gs];
        auto gn = std::min(size_t(n-i), size_t(gs));
        for (size_t j = 0; j < gn; j += 2) {
            auto k = (i + j) / 2;
            tmp[j]   = s1 * qx1[k].a + s2 * qx2[k].a;
            tmp[j+1] = s1 * qx1[k].b + s2 * qx2[k].b;
        }
        quantize(qx1+i, &qs1[i/gs], tmp, gn, gs);
    }
}

void add(QuantType qt, void* qx1, float* qs1, const void* qx2, const float* qs2, size_t n, int gs) noexcept {
    if (qx1 == nullptr || qx2 == nullptr || n < 1) {
        return;
    }
    if (qt == QuantType::NONE) {
        cpuft::add(reinterpret_cast<float*>(qx1), reinterpret_cast<const float*>(qx2), n);
        return;
    }
    if (qs1 == nullptr || qs2 == nullptr || gs < 1) {
        return;
    }
    if (qt == QuantType::INT16) {
        add(reinterpret_cast<short*>(qx1), qs1, reinterpret_cast<const short*>(qx2), qs2, n, gs);
    } else if (qt == QuantType::INT8) {
        add(reinterpret_cast<int8_t*>(qx1), qs1, reinterpret_cast<const int8_t*>(qx2), qs2, n, gs);
    } else if (qt == QuantType::INT4) {
        add(reinterpret_cast<Bit4Pair*>(qx1), qs1, reinterpret_cast<const Bit4Pair*>(qx2), qs2, n, gs);
    }
}

template <typename T>
inline void add(T* qx, float* qs, float v, size_t n, int group_size) noexcept {
    //constexpr float QF = (std::is_same<T, Bit4Pair>::value) ? QUANT4_FACTOR : (sizeof(T) == 1 ? QUANT8_FACTOR : QUANT16_FACTOR);
    //constexpr float  F = QF - 0.5/QF;
    // TODO: to be implemented
    return;
}

void add(QuantType qt, void* qx, float* qs, float v, size_t n, int gs) noexcept {
    if (qx == nullptr || n < 1) {
        return;
    }
    if (qt == QuantType::NONE) {
        return cpuft::add(reinterpret_cast<float*>(qx), v, n);
    }
    if (qs == nullptr || gs < 1) {
        return;
    }
    if (qt == QuantType::INT16) {
        add(reinterpret_cast<short*>(qx), qs, v, n, gs);
    } else if (qt == QuantType::INT8) {
        add(reinterpret_cast<int8_t*>(qx), qs, v, n, gs);
    } else if (qt == QuantType::INT4) {
        add(reinterpret_cast<Bit4Pair*>(qx), qs, v, n, gs);
    }
}

inline void mul(float* x, size_t n, float v) noexcept {
    for (size_t i = 0; i < n; ++i) { x[i] *= v; }
}

void mul(QuantType qt, void* qx, float factor, float* qs, size_t n, int group_size) noexcept {
    if (qx == nullptr || n < 1) {
        return;
    }
    if (qt == QuantType::NONE) {
        return mul(reinterpret_cast<float*>(qx), n, factor);
    }
    if (qs == nullptr || group_size < 1) {
        return;
    }
    for (size_t i = 0, sn = (n + group_size - 1) / group_size; i < sn; ++i) {
        qs[i] *= factor;
    }
}

template <typename T>
void weighted_sum(float* out, const T* matrix, const float* scales, int gs,
                  const float* weights, int m, int n, float min_w) noexcept {
    auto& row = matrix;
    for (int i = 0; i < n; i += gs) {
        auto s = scales[i/gs] * weights[0];
        if (s == 0.) {
            continue;
        }
        if constexpr (std::is_same<T, Bit4Pair>::value) {
            for (int j = i, e = std::min(int(n), int(i+gs)); j < e; j += 2) {
                out[j]   = row[j/2].a * s;
                out[j+1] = row[j/2].b * s;
            }
        } else {
            for (int j = i, e = std::min(int(n), int(i+gs)); j < e; ++j) {
                out[j] = row[j] * s;
            }
        }
    }
    row += n;
    for (int i = 1; i < m; ++i, row += n, scales += n) {
        if (weights[i] <= min_w) {
            continue;
        }
        for (int j = 0; j < n; j += gs) {
            auto s = scales[j/gs] * weights[i];
            if (s == 0.) {
                continue;
            }
            if constexpr (std::is_same<T, Bit4Pair>::value) {
                for (int k = j, e = std::min(int(n), int(j+gs)); k < e; k += 2) {
                    out[k]   += row[k/2].a * s;
                    out[k+1] += (QUANT4_FACTOR - row[k/2].b) * s;
                }
            } else {
                for (int k = j, e = std::min(int(n), int(j+gs)); k < e; ++k) {
                    out[k] += row[k] * s;
                }
            }
        }
    }
}

void weighted_sum(QuantType qt, float* out, const void* matrix, const float* scales, int gs, const float* weights,
                int m, int n, float min_w) noexcept {
    if (out == nullptr || matrix == nullptr || weights == nullptr || n < 1) {
        return;
    }
    if (qt == QuantType::NONE) {
        return cpuft::weighted_sum(out, reinterpret_cast<const float*>(matrix), weights, m, n, min_w);
    }
    if (scales == nullptr || gs < 1) {
        return;
    }
    if (qt == QuantType::INT16) {
        weighted_sum(out, reinterpret_cast<const short*>(matrix),    scales, gs, weights, m, n, min_w);
    } else if (qt == QuantType::INT8) {
        weighted_sum(out, reinterpret_cast<const int8_t*>(matrix),   scales, gs, weights, m, n, min_w);
    } else if (qt == QuantType::INT4) {
        weighted_sum(out, reinterpret_cast<const Bit4Pair*>(matrix), scales, gs, weights, m, n, min_w);
    }
}

template <typename T>
void weighted_sum(float* out, const T* matrix, const float* scales, int gs,
                  const float* weights, int m, int n, int bs, float min_w) noexcept {
    auto& row = matrix;
    auto o = out;
    for (int k = 0; k < bs; ++k, o += n) {
        float w = *(weights + m * k);
        for (int i = 0; i < n; i += gs) {
            auto s = scales[i/gs] * w;
            if constexpr (std::is_same<T, Bit4Pair>::value) {
                for (int j = i, e = std::min(int(n), int(i+gs)); j < e; j += 2) {
                    o[j]   = row[j/2].a * s;
                    o[j+1] = row[j/2].b * s;
                }
            } else {
                for (int j = i, e = std::min(int(n), int(i+gs)); j < e; ++j) {
                    o[j] = row[j] * s;
                }
            }
        }
    }
    row += n;
    for (int i = 1; i < m; ++i, row += n, scales += n/gs) {
        auto o = out;
        for (int k = 0; k < bs; ++k, o += n) {
            float w = (weights + m * k)[i];
            if (w <= min_w) {
                continue;
            }
            for (int j = 0; j < n; j += gs) {
                auto s = scales[j/gs] * w;
                if constexpr (std::is_same<T, Bit4Pair>::value) {
                    for (int t = j, e = std::min(int(n), int(j+gs)); t < e; t += 2) {
                        o[t]   += row[t/2].a * s;
                        o[t+1] += row[t/2].b * s;
                    }
                } else {
                    for (int t = j, e = std::min(int(n), int(j+gs)); t < e; ++t) {
                        o[t] += row[t] * s;
                    }
                }
            }
        }
    }
}

void weighted_sum(QuantType qt, float* out, const void* matrix, const float* scales, int gs,
                const float*weights, int m, int n, int bs, float min_w) noexcept {
    if (qt == QuantType::NONE) {
        return cpuft::weighted_sum(out, reinterpret_cast<const float*>(matrix), weights, m, n, bs, min_w);
    }
    if (out == nullptr || matrix == nullptr || scales == nullptr || weights == nullptr || m < 1 || n < 1 || bs < 1 || gs < 1) {
        return;
    }
    if (qt == QuantType::INT16) {
        weighted_sum(out, reinterpret_cast<const short*>(matrix),    scales, gs, weights, m, n, bs, min_w);
    } else if (qt == QuantType::INT8) {
        weighted_sum(out, reinterpret_cast<const int8_t*>(matrix),   scales, gs, weights, m, n, bs, min_w);
    } else if (qt == QuantType::INT4) {
        weighted_sum(out, reinterpret_cast<const Bit4Pair*>(matrix), scales, gs, weights, m, n, bs, min_w);
    }
}

void matmul(QuantType qt, float* out,
            const void* mat1, const float* scales1,
            const void* mat2, const float* scales2,
            int m, int n, int w, int gs) {
    if (mat1 == nullptr || mat2 == nullptr || out == nullptr) {
        return;
    }
    if (qt == QuantType::NONE) {
        return matmul(out, (const float*)(mat1), (const float*)(mat2), m, n, w);
    }
    if (scales1 == nullptr || scales2 == nullptr || gs < 1) {
        return;
    }
    if (qt == QuantType::INT16) {
        return matmul(out, (const short*)(mat1),    scales1, (const short*)(mat2), scales2, m, n, w, gs);
    } else if (qt == QuantType::INT8) {
        return matmul(out, (const int8_t*)(mat1),   scales1, (const int8_t*)(mat2), scales2, m, n, w, gs);
    } else if (qt == QuantType::INT4) {
        return matmul(out, (const Bit4Pair*)(mat1), scales1, (const Bit4Pair*)(mat2), scales2, m, n, w, gs);
    }
}

template <typename T>
inline void fill_random_(T* qx, float* qs, size_t n, int gs, float min_value, float max_value) noexcept {
    if (qx == nullptr || qs == nullptr || n < 1 || gs < 1 || max_value <= min_value) {
        return;
    }

    const float absmax = std::max(fabsf(max_value), fabsf(min_value));

    constexpr float QF = sizeof(T) == 1 ? QUANT8_FACTOR : QUANT16_FACTOR;

    constexpr uint64_t RGMAX = 18923891283921269ul;
    std::mt19937 rng(uint64_t(std::time(nullptr)) * reinterpret_cast<uint64_t>(qx));
    std::uniform_int_distribution<uint64_t> distribution(0, RGMAX);

    double rr = (max_value - min_value) / RGMAX;
    auto randf = [&distribution, &rng, min_value, rr]() -> float {
        return min_value + float(rr * distribution(rng));
    };

    for (size_t i = 0; i < n/gs; ++i, qx += gs) {
        float maxv = 0.;
        for (int j = 0; j < gs; ++j) {
            float rv = randf();
            if (fabsf(rv) > maxv) {
                maxv = fabsf(rv);
            }
            qx[j] = T((rv / absmax) * QF);
        }
        qs[i] = maxv / QF;
        float r = absmax / maxv;
        for (int j = 0; j < gs; ++j) {
            qx[j] = std::min(T(qx[j] * r), T(QF));
        }
    }
}

void fill_random(int* qx, float* qs, size_t n, int gs, float min_value, float max_value) noexcept {
    return fill_random_(qx, qs, n, gs, min_value, max_value);
}
void fill_random(short* qx, float* qs, size_t n, int gs, float min_value, float max_value) noexcept {
    return fill_random_(qx, qs, n, gs, min_value, max_value);
}
void fill_random(int8_t* qx, float* qs, size_t n, int gs, float min_value, float max_value) noexcept {
    return fill_random_(qx, qs, n, gs, min_value, max_value);
}

void fill_random(Bit4Pair* qx, float* qs, size_t n, int gs, float min_value, float max_value) noexcept {
    std::unique_ptr<float[]> tmp(new float[gs]);
    for (size_t i = 0; i < n; i += gs) {
        auto bn = std::min((n-i), size_t(gs));
        cpuft::fill_random(tmp.get(), bn, min_value, max_value);
        quantize(&qx[i/2], &qs[i/gs], tmp.get(), bn, gs);
    }
}

void fill_random(QuantType qt,
        void* qx, float* qs, size_t n, int gs,
        float min_value, float max_value) noexcept {
    if (qt == QuantType::NONE) {
        cpuft::fill_random(reinterpret_cast<float*>(qx), n, min_value, max_value);
    } else if (qt == QuantType::INT16) {
        fill_random(reinterpret_cast<short*>(qx), qs, n, gs, min_value, max_value);
    } else if (qt == QuantType::INT8) {
        fill_random(reinterpret_cast<int8_t*>(qx), qs, n, gs, min_value, max_value);
    } else if (qt == QuantType::INT4) {
        fill_random(reinterpret_cast<Bit4Pair*>(qx), qs, n, gs, min_value, max_value);
    }

}

template <typename T>
inline float dot_product(const T* qx1, const float* qs1, const T* qx2, const float* qs2, size_t n, int gs) noexcept {
    float sum = 0.;
    for (size_t i = 0; i < n; i += gs) {
        auto v = simd::dot_product(qx1+i, qx2+i, std::min(size_t(n), size_t(i+gs)));
        sum += qs1[i/gs] * qs2[i/gs] * v;
    }
    return sum;
}

inline float dot_product(const Bit4Pair* qx1, const float* qs1, const Bit4Pair* qx2, const float* qs2, size_t n, int gs) noexcept {
    int8_t tmp1[128];
    int8_t tmp2[128];
    float sum = 0.;
    for (size_t i = 0; i < n; i += gs) {
        auto gn = cpuft::min(n-i, size_t(gs));
        for (size_t j = 0; j < gn; j+=2) {
            tmp1[j]   = qx1[(i+j)/2].a;
            tmp1[j+1] = qx1[(i+j)/2].b;
            tmp2[j]   = qx2[(i+j)/2].a;
            tmp2[j+1] = qx2[(i+j)/2].b;
        }
        auto v = simd::dot_product(tmp1, tmp2, gn-i);
        sum += qs1[i/gs] * qs2[i/gs] * v;
    }
    return sum;
}

float dot_product(QuantType qt, const void* qx1, const float* qs1, const void* qx2, const float* qs2, size_t n, int gs) noexcept {
    if (qx1 == nullptr || qx2 == nullptr || n < 1) {
        return 0.;
    }
    if (qt == QuantType::NONE) {
        return cpuft::simd::dot_product(reinterpret_cast<const float*>(qx1), reinterpret_cast<const float*>(qx2), n);
    }
    if (qs1 == nullptr || qs2 == nullptr || gs < 1) {
        return 0.;
    }
    if (qt == QuantType::INT16) {
        return dot_product( reinterpret_cast<const short*>(qx1), qs1,
                reinterpret_cast<const short*>(qx2), qs2, n, gs);
    } else if (qt == QuantType::INT8) {
        return dot_product( reinterpret_cast<const int8_t*>(qx1), qs1,
                reinterpret_cast<const int8_t*>(qx2), qs2, n, gs);
    } else if (qt == QuantType::INT4) {
        return dot_product( reinterpret_cast<const Bit4Pair*>(qx1), qs1,
                reinterpret_cast<const Bit4Pair*>(qx2), qs2, n, gs);
    }
    return 0.;
}

template <typename T>
inline float square_sum(const T* qx, const float* qs, size_t n, int gs) noexcept {
    float sum = 0.;
    for (size_t i = 0; i < n; i += gs) {
        auto s = qs[i/gs];
        if constexpr (std::is_same<T, Bit4Pair>::value) {
            auto gn = std::min(size_t(n), size_t(i+gs));
            int isum = 0;
            for (size_t j = i; j < gn; ++j) {
                auto& v = qx[j/2];
                isum += int(v.a * v.a + v.b * v.b);
            }
            sum += s * s * isum;
        } else {
            auto v = cpuft::square_sum(qx+i, std::min(size_t(n), size_t(i+gs)));
            sum += s * s * v;
        }
    }
    return sum;
}
float square_sum(QuantType qt, const void* qx, const float* qs, size_t n, int gs) noexcept {
    if (qx == nullptr || n < 1) {
        return 0.;
    }
    if (qt == QuantType::NONE) {
        return cpuft::square_sum(reinterpret_cast<const float*>(qx), n);
    }
    if (qs == nullptr || gs < 1) {
        return 0.;
    }
    if (qt == QuantType::INT16) {
        return square_sum(reinterpret_cast<const short*>(qx), qs, n, gs);
    } else if (qt == QuantType::INT8) {
        return square_sum(reinterpret_cast<const int8_t*>(qx), qs, n, gs);
    } else if (qt == QuantType::INT4) {
        return square_sum(reinterpret_cast<const Bit4Pair*>(qx), qs, n, gs);
    }
    return 0.;
}

} // namespace cpuft::quant
