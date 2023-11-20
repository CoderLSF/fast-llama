/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 @Author: Liu Shaofeng
 @Date: 2023/10/16
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <map>
#include <memory>
#include <span>
#include <vector>

#include "simd.h"

namespace cpuft {
using namespace ::cpuft::simd;

void softmax(float* x, size_t size) noexcept;
void softmax_sisd(float* x, int size) noexcept;

void rmsnorm_sisd(float* o, const float* x, const float* weight, const int size) noexcept;

void silu(float* x, size_t n) noexcept;

/**
 * @brief 根据输入矩阵和权重向量，计算加权求和结果，并将结果存储在输出向量中。
 * @param output_vector 输出结果向量，长度为 n
 * @param matrix 输入矩阵，大小为 rows x columns
 * @param weights 权重向量，长度为 rows，元素总和为 1
 * @param rows 矩阵的行数
 * @param columns 矩阵的列数
 */
void weighted_sum(float* out, const float* matrix, const float* weights, int m, int n, float min_w=0) noexcept;
void weighted_sum(float* out, const float* matrix, const float* weights, int m, int n, int bs, float min_w=0) noexcept;

void fill_random(float* x, size_t n, float min_value, float max_value) noexcept;

float dot_product_sisd(const float*  x1, const float*  x2, size_t n) noexcept;
int   dot_product_sisd(const   int*  x1, const   int*  x2, size_t n) noexcept;

void  matmul(const float* mat1, const float* mat2, float* out, int m, int n, int k) noexcept;
void  matmul(const   int* mat1, const   int* mat2,   int* out, int m, int n, int k) noexcept;

void fill_random(float* x, int n);

void rope(float* x, int n, int pos, int head_size, float freq_base=10000.) noexcept;
void rope(float* x, int n, const float* freq_cis, int head_size) noexcept;
void rope(float* q, float* k, int pos, int head_size, int dim, int kv_dim, float freq_base=10000.);

void rope_v2(float * o, const float* x, int head_size, int n_orig_ctx, int pos, int ith, int nth);

} // namespace cpuft
