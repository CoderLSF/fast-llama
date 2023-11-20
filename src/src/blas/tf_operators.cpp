
#include "tf_operators.h"

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

#include "simd.h"
#include "utility.h"
#include "base_constants.h"

namespace cpuft {

float dot_product_sisd(const float* x1, const float* x2, size_t n) noexcept {
    if (x1 == nullptr || x2 == nullptr || n < 1) [[unlikely]] {
        return 0;
    }
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += x1[i] * x2[i];
    }
    return sum;
}

int dot_product_sisd(const int* x1, const int* x2, size_t n) noexcept {
    if (x1 == nullptr || x2 == nullptr || n < 1) [[unlikely]] {
        return 0;
    }
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += x1[i] * x2[i];
    }
    return sum;
}

template <typename T>
int square_sum_sisd(const T* x, size_t n) noexcept {
    if (x == nullptr || n < 1) [[unlikely]] {
        return 0.;
    }
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += int(x[i]) * int(x[i]);
    }
    return sum;
}
float square_sum_sisd(const float* x, size_t n) noexcept {
    if (x == nullptr || n < 1) [[unlikely]] {
        return 0.;
    }
    float sum = 0.;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
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

void rmsnorm_sisd(float* o, const float* x, const float* w, const int n) noexcept {
    const float r = 1. / sqrtf(square_sum_sisd(x, n) / n + 1e-5f);
    for (int i = 0; i < n; i++) {
        o[i] = w[i] * (r * x[i]);
    }
}

void silu(float* x, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        x[i] = x[i] / (1 + expf(-x[i]));
    }
}

void swiglu_sisd(float* xo, const float* xi, size_t n) noexcept {
    for (size_t i = 0; i < n; i++) {
        xo[i] = xo[i] / (1. + expf(-xo[i])) * xi[i];
    }
    return;
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
                    o[i] += dot_product(x, y + ns, ne - ns);
                    x += n;
                }
            }
        }
    }
}

static inline void multiply_sisd(float* x, const int n, const float v) {
    for (int i = 0; i < n; ++i) { x[i] *= v; }
}

void softmax_sisd(float* x, const int n) noexcept {
    float max_val = array_max(x, n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

void softmax(float* x, size_t n) noexcept {
    const float max_val = array_max(x, n);
    float sum = 0.0f;
    float vc[16];
    for (int i = 0; i < 16; ++i) {
        vc[i] = expf(6 + i/4);
    }
    for (size_t i = 0; i < n; i++) {
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
    multiply(x, 1./sum, n);
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

void weighted_sum(float* out, const float* matrix, const float* weights, int m, int n, float min_w) noexcept {
    if (out == nullptr || matrix == nullptr || weights == nullptr || m < 1 || n < 1) {
        return;
    }

    //memcpy(out, matrix, sizeof(float) * n);
    memset(out, 0, sizeof(float) * n);
    printf("memset(out, 0, sizeof(float) * n)\n");
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
    corr_dims[1] = cpuft::min((float)(n_dims - 1), ceilf(rope_yarn_corr_dim((float)(n_dims), n_orig_ctx, beta_slow, freq_base)));

    float theta_base = (float)pos;
    for (int64_t i = 0; i < n_dims; i += 2) {
        float cos_theta, sin_theta;
        {
            float mscale = attn_factor;
            // Get n-d rotational scaling corrected for extrapolation
            float theta_interp = freq_scale * theta_base;
            float theta = theta_interp;
            if (ext_factor != 0.0f) {
                float ramp_mix = 1 - cpuft::min((float)(1), std::max((float)(0), (i / 2 - corr_dims[0]) / std::max((float)(0.001f), corr_dims[1] - corr_dims[0])));
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
